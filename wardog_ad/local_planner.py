"""Truck local-planning contract shared by shadow collection and control.

Map coordinates are pixels. All ego coordinates are [right, forward] in metres,
not TCP's legacy [right, -forward]. This baseline has NO visual obstacle model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Protocol

from .geometry import angle_delta, bearing, distance


def ego_point(point, state, mpp):
    dx, dy = (point[0]-state.x)*mpp, (point[1]-state.y)*mpp
    h = math.radians(state.heading)
    return [dx*math.cos(h)+dy*math.sin(h), dx*math.sin(h)-dy*math.cos(h)]


@dataclass(frozen=True)
class PerceptionContext:
    t: float
    obstacle_checked: bool = False
    blocked: bool = False


@dataclass
class LocalPlan:
    t: float
    route_id: int
    valid: bool = False
    reason: str = 'missing_state_or_route'
    coordinate_frame: str = 'vehicle_right_forward_m'
    reference_map_px: list = field(default_factory=list)
    reference_ego_m: list = field(default_factory=list)
    trajectory_map_px: list = field(default_factory=list)
    trajectory_ego_m: list = field(default_factory=list)
    target_map_px: list | None = None
    target_ego_m: list | None = None
    target_speed_ms: float | None = None
    progress_m: float | None = None
    remaining_m: float | None = None
    cross_track_m: float | None = None
    command: str | None = None
    obstacle_checked: bool = False
    requires_replan: bool = False
    execution_approved: bool = False
    planner: str = 'reference_corridor_v1'


class LocalPlanner(Protocol):
    def plan(self, path, state, *, progress_px: float, route_id: int, now: float,
             perception: PerceptionContext | None = None) -> LocalPlan: ...


class ReferenceLocalPlanner:
    """Sample the map reference and speed profile; future vision can replace it."""
    def __init__(self, tuning, meters_per_pixel, *, horizon_m=60., spacing_m=5.):
        if not (meters_per_pixel > 0 and horizon_m > 0 and spacing_m > 0):
            raise ValueError('local-planner scales must be positive')
        self.t, self.mpp = tuning, meters_per_pixel
        self.horizon_m, self.spacing_m = horizon_m, spacing_m

    def plan(self, path, state, *, progress_px=0., route_id=0, now=0., perception=None):
        out = LocalPlan(t=now, route_id=route_id)
        if path is None or state is None:
            return out
        if not all(math.isfinite(v) for v in (state.x,state.y,state.heading,state.speed_ms,state.yaw_rate,now)):
            out.reason = 'nonfinite_state'
            return out
        cross, s = path.nearest(state.position, max(0,progress_px-20/self.mpp), progress_px+120/self.mpp)
        progress = max(progress_px, s)
        out.progress_m, out.remaining_m, out.cross_track_m = progress*self.mpp, max(0,path.length-progress)*self.mpp, cross*self.mpp
        if out.cross_track_m > self.t.off_route_m:
            out.reason, out.requires_replan = 'off_route', True
            return out
        lookahead = self.t.lookahead_base_m + self.t.lookahead_gain_s*max(0,state.speed_ms)
        horizon = max(self.horizon_m, lookahead + self.spacing_m)
        end = min(path.length,progress+horizon/self.mpp)
        positions, p = [], progress
        while p < end:
            positions.append(p)
            p += self.spacing_m/self.mpp
        positions.append(end)
        out.reference_map_px = [list(map(float,path.at(p))) for p in positions]
        out.reference_ego_m = [ego_point(p,state,self.mpp) for p in out.reference_map_px]
        out.trajectory_map_px = [p[:] for p in out.reference_map_px]
        out.trajectory_ego_m = [p[:] for p in out.reference_ego_m]
        out.target_map_px = list(map(float,path.at(progress+lookahead/self.mpp)))
        out.target_ego_m = ego_point(out.target_map_px,state,self.mpp)
        alpha = angle_delta(state.heading,bearing(state.position,out.target_map_px))
        out.command = 'left' if alpha < -25 else 'right' if alpha > 25 else 'straight'
        k = abs(2*math.sin(math.radians(alpha))/max(1,distance(state.position,out.target_map_px)*self.mpp))
        k = max(k,path.max_curvature(progress,end)/self.mpp)
        curve = math.sqrt(self.t.lateral_accel/k) if k > 1e-6 else self.t.max_speed_ms
        goal = math.sqrt(3*max(0,out.remaining_m-self.t.arrive_radius_m*.5))
        out.target_speed_ms = min(self.t.max_speed_ms,curve,max(2,goal),
                                  self.t.max_speed_ms*max(.25,math.cos(math.radians(min(abs(alpha),75)))))
        if out.remaining_m < self.t.arrive_radius_m and distance(state.position,path.points[-1])*self.mpp < self.t.arrive_radius_m*1.5:
            out.target_speed_ms = 0.
        out.valid, out.reason = True, 'reference_only_no_visual_obstacle_check'
        if perception is not None and 0 <= now-perception.t <= .5:
            out.obstacle_checked = perception.obstacle_checked
            if perception.blocked:
                out.target_speed_ms, out.reason, out.requires_replan = 0., 'blocked_stop_required', True
        return out


class LocalPlanRuleController:
    """Pure pursuit + yaw feedback + P speed control on the supplied local plan.

    Stateless by design: manual driving does not supply the actuation history
    needed to interpret an autonomous PI integral or stuck-recovery state.
    """
    def __init__(self,tuning):
        self.t = tuning

    def step(self,state,plan,now):
        from .controllers import Output
        if not plan.valid or not 0 <= now-plan.t <= 1.:
            return Output(dict(steer=0.,throttle=0.),'invalid_local_plan',replan=True)
        speed, target_speed = state.speed_ms, plan.target_speed_ms
        if target_speed == 0:
            brake = 0. if abs(speed)<=.5 else -math.copysign(min(1.,abs(speed)/4),speed)
            return Output(dict(steer=0.,throttle=brake),'local_plan_stop',dict(v_target=0.,reason=plan.reason))
        # Targets are in the original ego frame of this sample, so the caller
        # must supply a newly planned sample rather than repeatedly reuse it.
        right,forward = plan.target_ego_m
        dist2 = max(1.,right*right+forward*forward)
        alpha = math.degrees(math.atan2(right,forward))
        if abs(alpha)>110 and abs(speed)<3:
            return Output(dict(steer=0.,throttle=0.),'local_target_behind',dict(v_target=0.),replan=True)
        kappa = 2*right/dist2
        rate_cmd = math.degrees(max(0.,speed)*kappa)
        steer = self.t.steer_gain*kappa/self.t.max_curvature+self.t.yaw_kp*(rate_cmd-state.yaw_rate)
        throttle = self.t.speed_kp*(target_speed-speed)
        if speed>target_speed+self.t.brake_margin_ms:
            throttle = -min(1.,(speed-target_speed)/4)
        elif speed<1:
            throttle = max(0.,throttle)
        return Output(dict(steer=max(-1.,min(1.,steer)),throttle=max(-1.,min(1.,throttle))),
                      'local_track',dict(v_target=target_speed,alpha=alpha,rate_cmd=rate_cmd,kappa=kappa))


def shadow_rule_control(path, state, plan, tuning, mpp, now):
    """Stateless counterfactual advice, NOT expert targets and NOT sent keys."""
    from .recorder import _plain
    if not plan.valid:
        return dict(t=now, route_id=plan.route_id, valid=False, reason=plan.reason, applied=False, axes=None)
    output = LocalPlanRuleController(tuning).step(state,plan,now)
    return dict(t=now, route_id=plan.route_id, valid=output.mode != 'invalid_local_plan', applied=False, controller='LocalPlanRuleController_v1',
                state_policy='reset_each_observation; no accumulated control/recovery history',
                axes=_plain(output.axes), mode=output.mode, info=_plain(output.info),
                done=output.done, replan=output.replan, coordinate_frame='steer_right_positive; throttle_forward_positive',
                target_speed_ms=plan.target_speed_ms)
