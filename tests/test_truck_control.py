"""Truck control regressions independent of map assets and Windows input."""
import math

import pytest

from wardog_ad.config import TruckTuning
from wardog_ad.controllers import TruckDriver
from wardog_ad.estimator import VehicleState
from wardog_ad.planner import Path


def state(x=0, y=0, speed=0, heading=90):
    return VehicleState(x, y, heading, speed, 0, None, 0, 'arrow')


def driver(tuning=None):
    return TruckDriver(Path([(0, 0), (200, 0)], smoothing=0), tuning or TruckTuning(), 1)


def test_arrival_stays_in_braking_after_overshoot():
    d = driver()
    d.progress = 180  # preceding approach updates already tracked this segment
    out = d.step(state(x=190, speed=16), 0)
    assert not out.done and out.axes['throttle'] == -1
    out = d.step(state(x=240, speed=10), 1)
    assert out.mode == 'arrived' and not out.done
    assert out.axes['throttle'] == -1
    assert d.step(state(x=245, speed=.5), 2).done


def test_arrival_brakes_reverse_motion():
    d = driver()
    d.progress = 180
    out = d.step(state(x=190, speed=-4), 0)
    assert not out.done and out.axes['throttle'] == 1


def test_rejoining_straight_road_obeys_lateral_limit_below_cruise_floor():
    t = TruckTuning(min_speed_ms=12, lateral_accel=.5)
    out = driver(t).step(state(y=10, speed=5), 0)
    assert out.info['kappa_max'] > 0
    assert out.info['v_target'] <= math.sqrt(t.lateral_accel / out.info['kappa_max']) + 1e-9
    assert out.info['v_target'] < t.min_speed_ms


def test_speed_integrator_does_not_wind_up_at_full_throttle():
    d = driver()
    for i in range(20):
        d.step(state(x=i * .1, speed=2), i * .1, motion_level=1)
    assert d.integral == pytest.approx(0)


def test_new_path_clears_old_manoeuvre_state():
    d = driver()
    d.steer_i = .6
    d.off_since = d.slow_since = d.last_time = 1
    d.history.append((1, 0, 0))
    d.stuck_points.append((0, 0))
    d.mode = 'arrived'
    d.set_path(Path([(0, 0), (400, 0)]))
    assert d.steer_i == d.integral == d.progress == 0
    assert d.off_since is d.slow_since is d.last_time is None
    assert not d.history and not d.stuck_points
    assert d.step(state(), 2).mode == 'drive'


def test_off_route_watchdog_accepts_zero_timestamp():
    d = driver()
    assert not d.step(state(y=60, speed=4), 0).replan
    assert not d.step(state(y=60, speed=4), 2).replan
    assert d.step(state(y=60, speed=4), 3.1).replan
