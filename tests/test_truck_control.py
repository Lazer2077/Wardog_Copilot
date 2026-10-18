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


@pytest.mark.parametrize('motion',[None,float('nan'),float('inf'),-.1,1.])
def test_no_stuck_reverse_without_valid_static_visual_evidence(motion):
    d = driver()
    for i in range(350):
        out = d.step(state(),i*.1,motion_level=motion)
        assert out.mode == 'drive' and out.axes['throttle'] >= 0
    assert d.recovery.attempts == 0


def test_startup_grace_and_sustained_low_speed_are_both_required():
    import json
    t = TruckTuning(startup_grace_s=3,stuck_s=1)
    d = driver(t)
    for i in range(40):
        out = d.step(state(),i*.1,motion_level=0.)
        assert out.mode == 'drive'
        json.dumps(out.info,allow_nan=False)
    assert d.step(state(),4.1,motion_level=0.).mode == 'reverse'
    moving = driver(t)
    for i in range(150):
        assert moving.step(state(x=i*.2,speed=2),i*.1,motion_level=0.).mode == 'drive'


def test_cooldown_and_shared_attempt_budget_survive_route_and_controller_reset():
    from wardog_ad.controllers import TruckRecoveryState
    t = TruckTuning(startup_grace_s=1,stuck_s=1,reverse_s=.4,bypass_s=.4,
                    recovery_cooldown_s=3,max_reverse_attempts=2)
    guard = TruckRecoveryState()
    d = TruckDriver(Path([(0,0),(200,0)],smoothing=0),t,1,recovery=guard)
    starts=[];last='drive';blocked=None
    for i in range(200):
        now=i*.1
        out=d.step(state(),now,motion_level=0.)
        if out.mode=='reverse' and last!='reverse':starts.append(now)
        last=out.mode
        if out.takeover:
            blocked=out
            break
    assert len(starts)==2
    assert starts[1]-starts[0] >= t.reverse_s+t.bypass_s+t.recovery_cooldown_s+t.stuck_s
    assert blocked.mode=='blocked' and blocked.axes['throttle']==0 and not blocked.replan
    d.set_path(Path([(0,0),(300,0)],smoothing=0))
    assert guard.attempts==2
    recreated=TruckDriver(d.path,t,1,recovery=guard)
    for i in range(41):out=recreated.step(state(),30+i*.1,motion_level=0.)
    assert out.takeover and guard.attempts==2
    # Only physically leaving the recovery area restores the budget.
    escaped=TruckDriver(d.path,t,1,recovery=guard)
    escaped.step(state(x=40,speed=3),40,motion_level=1.)
    assert guard.attempts==0


def test_kturns_share_grace_cooldown_and_reverse_limit():
    t = TruckTuning(startup_grace_s=1,reverse_s=.5,recovery_cooldown_s=2,max_reverse_attempts=1)
    d=driver(t)
    assert d.step(state(heading=270),0,motion_level=0.).mode=='recovery wait'
    assert d.step(state(heading=270),1,motion_level=0.).mode=='kturn'
    assert d.step(state(heading=270),1.5,motion_level=0.).mode=='recovery wait'
    assert d.step(state(heading=270),3,motion_level=0.).mode=='recovery wait'
    out=d.step(state(heading=270),3.6,motion_level=0.)
    assert out.takeover and out.axes['throttle']==0
