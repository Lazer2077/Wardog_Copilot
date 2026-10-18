"""Asset-free unit tests: geometry, planner, estimator, actuator, lock detection, config."""
import math

import cv2
import numpy as np
import pytest

from wardog_ad.actuator import Axis, PWMActuator
from wardog_ad.config import Config, Region
from wardog_ad.controllers import AircraftPilot
from wardog_ad.estimator import StateEstimator, VehicleState
from wardog_ad.geometry import angle_delta, bearing, curvature, resample
from wardog_ad.input_win import FakeKeyboard
from wardog_ad.localization import Fix
from wardog_ad.perception import FlareController, MissileLockDetector
from wardog_ad.planner import Avoid, Path, RoadGraph, RouteError


def test_bearing_and_delta():
    assert bearing((0, 0), (0, -1)) == pytest.approx(0)
    assert bearing((0, 0), (1, 0)) == pytest.approx(90)
    assert bearing((0, 0), (0, 1)) == pytest.approx(180)
    assert angle_delta(350, 10) == pytest.approx(20)
    assert angle_delta(10, 350) == pytest.approx(-20)


def test_resample_and_curvature():
    pts = resample([(0, 0), (100, 0)], 1.0)
    assert len(pts) == 101 and np.allclose(np.diff(pts[:, 0]), 1.0)
    circle = [(50 * math.cos(a), 50 * math.sin(a)) for a in np.linspace(0, math.pi, 400)]
    k = curvature(resample(circle, 1.0), step=3)
    assert np.median(k) == pytest.approx(1 / 50, rel=.05)


ROADS = [
    {'id': 'a', 'kind': 'major', 'points': [[0, 0], [100, 0]]},
    {'id': 'b', 'kind': 'major', 'points': [[100, 0], [100, 100]]},
    {'id': 'c', 'kind': 'minor', 'points': [[0, 0], [0, 100], [100, 100]]},
    # crosses road 'c' without a shared vertex -> must NOT be a junction
    {'id': 'd', 'kind': 'major', 'points': [[-20, 50], [20, 50]]},
]


def test_route_prefers_major_and_respects_avoid():
    g = RoadGraph(ROADS)
    route = g.route([(0, 2), (100, 98)])
    assert any(abs(x - 100) < 1e-6 and abs(y) < 1e-6 for x, y in route)  # via the major road corner
    detour = g.route([(0, 2), (100, 98)], avoid=[Avoid(90, -10, 110, 10)])
    assert all(not (90 <= x <= 110 and -10 <= y <= 10) for x, y in detour)
    with pytest.raises(RouteError):
        g.route([(-20, 50), (100, 98)])  # road 'd' only crosses, it is not connected


def test_path_progress_and_offset():
    p = Path([(0, 0), (0, -100)], smoothing=0)
    gap, s = p.nearest((3, -40))
    assert gap == pytest.approx(3) and s == pytest.approx(40)
    assert p.signed_offset((3, -40), s) > 0      # heading north, +x is to the right
    assert p.signed_offset((-3, -40), s) < 0


def test_estimator_speed_heading_and_reversing():
    est = StateEstimator(meters_per_pixel=5.0)
    rng = np.random.default_rng(0)
    state = None
    for i in range(30):  # 2 px/s east = 10 m/s
        t = i * 0.125
        state = est.update(Fix(100 + 2 * t + rng.normal(0, .2), 100 + rng.normal(0, .2), 90.0, valid=True), t) or state
    assert state.speed_ms == pytest.approx(10, abs=1.5)
    assert state.heading == pytest.approx(90, abs=1)
    est.reset()
    for i in range(30):  # facing east, moving west
        t = i * 0.125
        state = est.update(Fix(100 - 1 * t, 100, 90.0, valid=True), t) or state
    assert state.speed_ms < -3


def test_estimator_rejects_teleport():
    est = StateEstimator(meters_per_pixel=5.0)
    for i in range(10):
        est.update(Fix(100, 100, 0.0, valid=True), i * .1)
    assert est.update(Fix(400, 400, 0.0, valid=True), 1.0) is None
    assert est.pos[0] == pytest.approx(100)


def test_pwm_duty_cycle():
    kb = FakeKeyboard()
    act = PWMActuator(kb, {'steer': Axis('A', 'D', period_s=0.2)})
    act.enabled = True
    act.command(steer=0.5)
    on = 0
    for i in range(1000):
        act.update(i * 0.001)
        on += 'D' in kb.down
        assert 'A' not in kb.down
    assert on / 1000 == pytest.approx(0.5, abs=.03)
    act.release()
    assert not kb.down


def _warning(on: bool):
    roi = np.full((60, 200, 3), 30, np.uint8)
    if on:
        cv2.rectangle(roi, (20, 15), (180, 45), (0, 0, 230), -1)
        cv2.putText(roi, 'LOCK', (70, 38), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    return roi


def test_lock_detector_colour_blink_and_template(tmp_path):
    t = Config().aircraft
    det = MissileLockDetector(t)
    assert not det.update(_warning(False), 0.0)
    assert not det.update(_warning(True), 0.1)          # needs 2 confirmations
    assert det.update(_warning(True), 0.2)
    assert det.update(_warning(False), 0.4)             # blink gap bridged by hold_s
    assert not det.update(_warning(False), 1.0)
    path = tmp_path / 'tpl.png'
    cv2.imwrite(str(path), _warning(True)[10:50, 15:185])
    det = MissileLockDetector(t, path)
    assert det.measure(_warning(True)).positive
    assert not det.measure(_warning(False)).positive


def test_flare_bursts_and_cooldown():
    t = Config().aircraft
    shots = []
    fc = FlareController(t, lambda: shots.append(now))
    now = 0.0
    while now < 6.0:
        fc.update(True, now)
        now = round(now + 0.05, 3)
    # first burst after flare_delay_s, then a burst every (burst interval + cooldown) while locked
    period = (t.flare_burst - 1) * t.flare_interval_s + t.flare_cooldown_s
    bursts = 1 + int((6.0 - t.flare_delay_s - 0.05) // period)
    assert len(shots) == bursts * t.flare_burst
    assert shots[0] == pytest.approx(t.flare_delay_s, abs=.051)
    assert shots[1] - shots[0] == pytest.approx(t.flare_interval_s, abs=.051)
    fc2 = FlareController(t, lambda: shots.append(-1))
    fc2.update(False, 0.0)
    assert shots.count(-1) == 0


def test_aircraft_pilot_turns_toward_waypoint():
    t = Config().aircraft
    pilot = AircraftPilot([(0, 0), (1000, 0)], t, 5.0)
    st = VehicleState(0, 0, 0.0, 80, 0.0, None, 0.0, 'arrow')  # heading north, target east
    out = pilot.step(st, 0.0)
    assert out.axes['roll'] > 0.5
    st = VehicleState(0, 0, 90.0, 80, 0.0, None, 0.0, 'arrow')
    out = pilot.step(st, 0.0)
    assert out.axes['roll'] == 0 and abs(out.axes['yaw']) < 1e-9


def test_config_roundtrip(tmp_path):
    cfg = Config(profile='aircraft')
    cfg.minimap = Region(10, 20, 300, 300)
    cfg.aircraft.flare_burst = 5
    path = tmp_path / 'c.json'
    cfg.save(path)
    loaded = Config.load(path)
    assert loaded.profile == 'aircraft' and loaded.minimap == Region(10, 20, 300, 300)
    assert loaded.aircraft.flare_burst == 5 and isinstance(loaded.truck.road_kinds, tuple)
