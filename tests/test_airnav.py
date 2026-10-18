"""Helicopter dead-reckoning navigation: minimap fixes are optional corrections."""
import logging
import math

import numpy as np
import pytest

from wardog_ad.airnav import AirNavigator
from wardog_ad.assets import DEFAULT_ASSETS
from wardog_ad.config import Config
from wardog_ad.localization import Fix


def good_fix(x, y, t):
    return Fix(x, y, None, .8, 40, .5, .5, 0.0, True, 'ok', t)


def fly(nav, t0, seconds, speed_kmh, heading, dt=.1):
    t = t0
    for _ in range(int(seconds / dt)):
        t += dt
        nav.predict(t, speed_kmh, heading)
    return t


def test_dead_reckoning_moves_along_compass_heading():
    nav = AirNavigator(5.0)
    nav.start((100, 100), 0.0)
    fly(nav, 0.0, 10, 180, 90)               # 50 m/s east for 10 s = 500 m = 100 px
    st = nav.state()
    assert st.x == pytest.approx(200, abs=1) and st.y == pytest.approx(100, abs=1)
    assert nav.sigma_m > 10                   # uncertainty grows with distance flown


def test_bad_matches_are_gated_and_consistent_ones_reset():
    nav = AirNavigator(5.0)
    nav.start((100, 100), 0.0)
    t = fly(nav, 0.0, 2, 72, 0)               # 40 m north
    assert not nav.correct(good_fix(400, 400, t))          # 2 km away: a wrong match
    assert nav.state().x == pytest.approx(100, abs=1)
    assert nav.correct(good_fix(101, 92.5, t))             # plausible: blended in
    weak = Fix(150, 150, None, .2, 9, .5, .5, 0, True, 'ok', t)
    assert not nav.correct(weak)                           # low confidence: ignored
    for i in range(3):                                     # three agreeing far fixes: we were wrong
        nav.correct(good_fix(600 + i, 600, t))
    assert nav.state().x == pytest.approx(602, abs=2)


def test_learns_speed_scale_and_compass_bias():
    nav = AirNavigator(5.0)
    true = np.array([100.0, 500.0])
    t = 0.0
    nav.correct(good_fix(*true, t))
    for _ in range(12):                       # HUD reads 10 % fast and 3 deg left of the true course
        for _ in range(200):                  # 20 s legs (~220 m) between fixes
            t += .1
            r = math.radians(30.0)
            true += np.array([math.sin(r), -math.cos(r)]) * (40 / 3.6) * .1 / 5.0
            nav.predict(t, 40 * 1.1, 27.0)
        nav.correct(good_fix(*true, t))
    assert nav.speed_scale == pytest.approx(1 / 1.1, abs=.03)
    assert nav.heading_bias == pytest.approx(3.0, abs=1.0)


@pytest.mark.skipif(not (DEFAULT_ASSETS / 'maps.json').exists(), reason='map assets not cloned')
def test_flies_to_destination_without_the_minimap():
    """Minimap only for the first two minutes, biased instruments, then dead reckoning only."""
    from wardog_ad.assets import load_map
    from wardog_ad.autopilot import Autopilot
    from wardog_ad.input_win import FakeKeyboard
    from wardog_ad.sim import Body, SimSensors, SimWorld, TruthLocator
    logging.disable(logging.WARNING)
    assets = load_map('ozeti')
    goal = (1300, 300)
    cfg = Config(profile='aircraft')
    cfg.aircraft.lock_source = 'none'
    cfg.destinations = {'ozeti': list(goal)}
    kb = FakeKeyboard()
    w = SimWorld(cfg, assets.load_image(), assets.meters_per_pixel, Body(300, 1200, 200.0), kb, hills=60.0)
    truth = TruthLocator(w)

    class EarlyOnly:
        def locate(self, image, anchor, north_up):
            return truth.locate(image, anchor, north_up) if w.time < 120 else Fix(reason='no match', timestamp=w.time)

    class Biased(SimSensors):
        def hud(self):
            h = w.hud_reading()
            h.heading = (h.heading + 2.0) % 360
            h.speed_kmh = round(h.speed_kmh * .95)
            return h

    pilot = Autopilot(cfg, assets, kb, Biased(w), locator=EarlyOnly())
    assert pilot.go()
    next_slow = 0.0
    while w.time < 600 and not (pilot.arrived and not pilot.engaged):
        if w.time >= next_slow:
            pilot.tick(w.time)
            next_slow += 1 / cfg.loop_hz
        pilot.fast_control(w.time)
        pilot.actuate(w.time)
        w.step(1 / 30)
    logging.disable(logging.NOTSET)
    assert pilot.arrived and w.crashes == 0
    assert math.hypot(w.body.x - goal[0], w.body.y - goal[1]) * assets.meters_per_pixel < 80


def test_direction_comes_from_the_fixes_not_the_pitch():
    """Braking nose-up at a crawl used to flip the estimate to 'backwards' and run away."""
    from wardog_ad.airnav import AirNavigator
    from wardog_ad.localization import Fix
    nav = AirNavigator(1.0)
    nav.start((0.0, 0.0), 0.0)
    t, x = 0.0, 0.0
    for _ in range(20):                                  # creeping forwards (north) at 1.5 m/s
        t += .15
        x -= 1.5 * .15
        nav.predict(t, 5.0, 0.0, pitch=-.6)             # nose up, braking: the old guess said 'backwards'
        nav.correct(Fix(0.0, x, 0.0, 1.0, 50, .5, .5, 0.0, True, 'test', t))
    assert nav.direction == 1.0 and nav.speed_ms > 0
