from dataclasses import replace

import numpy as np
import pytest

from wardog_ad.config import TruckTuning
from wardog_ad.estimator import VehicleState
from wardog_ad.local_planner import ReferenceLocalPlanner, PerceptionContext, LocalPlanRuleController, ego_point, shadow_rule_control
from wardog_ad.planner import Path


def state(**kw):
    defaults = dict(x=50.,y=100.,heading=0.,speed_ms=8.,yaw_rate=0.,course=0.,timestamp=1.,heading_source='arrow')
    return VehicleState(**dict(defaults,**kw))


def test_ego_coordinates_and_reference_path_are_geometrically_consistent():
    assert np.allclose(ego_point([55,90],state(),2),[10,20])
    assert np.allclose(ego_point([60,105],state(heading=90),2),[10,20])
    path = Path([[50,100],[50,0]])
    plan = ReferenceLocalPlanner(TruckTuning(),2).plan(path,state(),route_id=7,now=1)
    assert plan.valid and plan.route_id == 7
    assert plan.command == 'straight'
    assert plan.target_ego_m[0] == pytest.approx(0)
    assert plan.target_ego_m[1] > 0
    assert len(plan.reference_map_px) >= 12
    assert not plan.obstacle_checked and not plan.execution_approved
    assert plan.trajectory_ego_m == plan.reference_ego_m


def test_offroute_missing_and_nonfinite_states_do_not_generate_advice():
    tuning = TruckTuning()
    planner = ReferenceLocalPlanner(tuning,2)
    path = Path([[50,100],[50,0]])
    plan = planner.plan(path,state(x=100),now=1)
    assert not plan.valid and plan.requires_replan
    assert not shadow_rule_control(path,state(x=100),plan,tuning,2,1)['valid']
    assert not planner.plan(path,None).valid
    assert not planner.plan(path,state(yaw_rate=float('nan'))).valid


def test_fresh_blocked_perception_requests_stop_before_reverse_manoeuvre():
    tuning = TruckTuning()
    path = Path([[50,100],[50,0]])
    st = state(heading=180,speed_ms=1)
    plan = ReferenceLocalPlanner(tuning,2).plan(path,st,now=1,
                      perception=PerceptionContext(.9,obstacle_checked=True,blocked=True))
    output = LocalPlanRuleController(tuning).step(st,plan,1)
    assert plan.target_speed_ms == 0 and plan.requires_replan
    assert output.mode == 'local_plan_stop'
    assert output.axes['steer'] == 0 and output.axes['throttle'] < 0
    stale = ReferenceLocalPlanner(tuning,2).plan(path,st,now=1,
                      perception=PerceptionContext(.1,obstacle_checked=True,blocked=True))
    assert not stale.obstacle_checked


def test_shadow_advice_and_live_controller_share_local_plan_contract(monkeypatch):
    import wardog_ad.input_win as inputs
    monkeypatch.setattr(inputs,'WindowsKeyboard',lambda:pytest.fail('shadow attempted keyboard output'))
    tuning = TruckTuning()
    path = Path([[50,100],[50,0]])
    st = state()
    plan = ReferenceLocalPlanner(tuning,2).plan(path,st,now=1)
    advice = shadow_rule_control(path,st,plan,tuning,2,1)
    assert advice['valid'] and advice['applied'] is False
    assert set(advice['axes']) == {'steer','throttle'}
    output = LocalPlanRuleController(tuning).step(st,replace(plan,target_speed_ms=2.),1)
    assert output.info['v_target'] <= 2
    assert LocalPlanRuleController(tuning).step(st,plan,3).mode == 'invalid_local_plan'
