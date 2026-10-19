"""Vehicle controllers.  Both return axis commands in [-1, 1] for the PWM actuator.

Truck:    pure pursuit on the road path + curvature/goal speed profile + PI speed
          loop, with stuck recovery (reverse out) and K-turns for targets behind.
Aircraft: waypoint sequencing + heading -> turn-rate -> roll cascade (or rudder).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

from .estimator import VehicleState
from .geometry import angle_delta, bearing, distance, heading_vector
from .planner import Path


@dataclass
class Output:
    axes: dict = field(default_factory=dict)
    mode: str = 'drive'
    info: dict = field(default_factory=dict)
    done: bool = False
    replan: bool = False
    takeover: bool = False


@dataclass
class TruckRecoveryState:
    """Shared across route replans and pauses; losing localization grants no new attempts."""
    origin: tuple | None = None
    attempts: int = 0
    cooldown_until: float = 0.

    def observe(self, point, mpp, reset_m):
        if self.origin is not None and distance(point, self.origin)*mpp > reset_m:
            self.origin,self.attempts,self.cooldown_until = None,0,0.


class TruckDriver:
    def __init__(self, path: Path, tuning, meters_per_pixel: float, *, recovery=None):
        self.path = path
        self.t = tuning
        self.mpp = meters_per_pixel
        self.progress = 0.0
        self.integral = 0.0
        self.steer_i = 0.0
        self.mode = 'drive'
        self.mode_until = 0.0
        self.mode_steer = 0.0
        self.slow_since: float | None = None
        self.history: deque = deque()
        self.stuck_points: list = []
        self.off_since: float | None = None
        self.last_time: float | None = None
        self.started_at: float | None = None
        self.recovery = recovery if recovery is not None else TruckRecoveryState()

    def set_path(self, path: Path):
        self.path, self.progress, self.integral, self.mode = path, 0.0, 0.0, 'drive'
        self.steer_i = 0.0
        self.last_time = self.slow_since = self.off_since = None
        self.history.clear()
        self.stuck_points.clear()
        self.mode_until = self.mode_steer = 0.0
        self.started_at = None

    def _begin_reverse(self, st, now, mode, steer, info):
        t,r = self.t,self.recovery
        if r.attempts >= max(0,t.max_reverse_attempts):
            self.mode = 'blocked'
            info.update(recovery_reason='reverse_limit',reverse_attempts=r.attempts)
            return Output({'steer':0.,'throttle':0.},'blocked',info,takeover=True)
        r.origin = st.position if r.origin is None else r.origin
        r.attempts += 1
        duration = t.reverse_s + (t.bypass_s if mode == 'reverse' else 0.)
        r.cooldown_until = now + duration + t.recovery_cooldown_s
        self.mode,self.mode_until,self.mode_steer = mode,now+t.reverse_s,steer
        self.history.clear()
        self.slow_since = None
        info.update(recovery_reason='stuck' if mode == 'reverse' else 'target_behind',reverse_attempts=r.attempts)
        return Output({'steer':steer,'throttle':-1.},mode,info)

    def _px(self, meters: float) -> float:
        return meters / self.mpp

    def step(self, st: VehicleState, now: float, motion_level: float | None = None, hud=None) -> Output:
        dt = 0.1 if self.last_time is None else max(1e-3, min(0.5, now - self.last_time))
        self.last_time = now
        path, t = self.path, self.t
        speed = st.speed_ms
        if self.started_at is None:
            self.started_at = now
        self.recovery.observe(st.position,self.mpp,t.recovery_reset_m)

        # progress along the path: search a window around the previous progress so
        # loops and parallel roads cannot make it jump.
        gap, s = path.nearest(st.position, self.progress - self._px(20), self.progress + self._px(120))
        self.progress = max(self.progress, s)
        cross_m = gap * self.mpp
        remaining_m = (path.length - self.progress) * self.mpp
        goal_m = distance(st.position, path.points[-1]) * self.mpp
        info = dict(progress_m=self.progress * self.mpp, remaining_m=remaining_m, cross_m=cross_m, speed=speed,
                    reverse_attempts=self.recovery.attempts, motion_level=motion_level)
        if self.mode == 'blocked':
            info['recovery_reason'] = 'reverse_limit'
            return Output({'steer':0.,'throttle':0.},'blocked',info,takeover=True)

        if self.mode == 'arrived' or (remaining_m < t.arrive_radius_m and goal_m < t.arrive_radius_m * 1.5):
            self.mode = 'arrived'
            # Latch stopping even if momentum carries us outside the arrival radius.
            stopped = abs(speed) <= 1.0
            brake = 0.0 if stopped else -math.copysign(1.0, speed)
            return Output({'steer': 0.0, 'throttle': brake}, 'arrived', info, done=stopped)

        # off-route watchdog -> ask the supervisor to re-plan from here
        if cross_m > t.off_route_m:
            if self.off_since is None:
                self.off_since = now
            if now - self.off_since > t.off_route_s:
                self.off_since = None
                return Output({'steer': 0.0, 'throttle': 0.0}, self.mode, info, replan=True)
        else:
            self.off_since = None

        # timed manoeuvres: reverse out, then bypass with the opposite lock
        if self.mode in ('reverse', 'kturn') and now < self.mode_until:
            return Output({'steer': self.mode_steer, 'throttle': -1.0}, self.mode, info)
        if self.mode == 'reverse':
            self.mode, self.mode_until = 'bypass', now + t.bypass_s
            self.recovery.cooldown_until = now + t.bypass_s + t.recovery_cooldown_s
            self.mode_steer = -self.mode_steer
        if self.mode == 'bypass' and now < self.mode_until:
            return Output({'steer': self.mode_steer, 'throttle': 0.7}, 'bypass', info)
        if self.mode in ('bypass', 'kturn'):
            self.recovery.cooldown_until = max(self.recovery.cooldown_until,now+t.recovery_cooldown_s)
            self.mode = 'drive'
            self.slow_since = None
            self.history.clear()
            self.integral = self.steer_i = 0.0

        lookahead_m = t.lookahead_base_m + t.lookahead_gain_s * max(0.0, speed)
        target = path.at(self.progress + self._px(lookahead_m))
        alpha = angle_delta(st.heading, bearing(st.position, target))
        dist_m = max(1.0, distance(st.position, target) * self.mpp)
        info.update(alpha=alpha, lookahead_m=lookahead_m)

        # target behind: K-turn (reverse with opposite lock swings the nose round)
        if abs(alpha) > 110 and abs(speed) < 3.0:
            if now-self.started_at < t.startup_grace_s or now < self.recovery.cooldown_until:
                info['recovery_reason'] = 'startup_grace' if now-self.started_at < t.startup_grace_s else 'cooldown'
                return Output({'steer':0.,'throttle':0.},'recovery wait',info)
            return self._begin_reverse(st,now,'kturn',-math.copysign(1.,alpha),info)

        kappa = 2.0 * math.sin(math.radians(alpha)) / dist_m
        # Feed-forward from pure pursuit plus a yaw-rate loop: the key-PWM -> wheel angle
        # mapping of a game is unknown and non-linear (steering ramps), so the integral
        # adapts it until the vehicle actually turns at v * kappa.
        rate_cmd = math.degrees(max(0.0, speed) * kappa)
        rate_err = rate_cmd - st.yaw_rate
        if speed > 1.0:
            self.steer_i = max(-t.yaw_i_limit, min(t.yaw_i_limit, self.steer_i + t.yaw_ki * rate_err * dt))
        else:
            self.steer_i *= 0.9
        steer = t.steer_gain * kappa / t.max_curvature + t.yaw_kp * rate_err + self.steer_i
        steer = max(-1.0, min(1.0, steer))
        info.update(rate_cmd=rate_cmd, yaw_rate=st.yaw_rate)

        # speed profile
        preview_m = max(30.0, speed * speed / 4.0 + 10.0)
        k_max = max(abs(kappa), path.max_curvature(self.progress, self.progress + self._px(preview_m)) / self.mpp)
        v_curve = math.sqrt(t.lateral_accel / k_max) if k_max > 1e-6 else t.max_speed_ms
        v_goal = math.sqrt(2 * 1.5 * max(0.0, remaining_m - t.arrive_radius_m * .5)) if remaining_m < t.goal_slowdown_m * 3 else t.max_speed_ms
        v_heading = t.max_speed_ms * max(0.25, math.cos(math.radians(min(abs(alpha), 75))))
        # A cruise floor must never override the lateral acceleration constraint.
        v_target = min(t.max_speed_ms, v_curve, v_heading, max(2.0, v_goal))
        error = v_target - speed
        candidate = max(-20.0, min(20.0, self.integral + error * dt))
        demand = t.speed_kp * error + t.speed_ki * candidate
        # Integrate while unsaturated, or when error unwinds saturation.
        if abs(demand) <= 1.0 or demand * error < 0.0:
            self.integral = candidate
        throttle = t.speed_kp * error + t.speed_ki * self.integral
        if speed > v_target + t.brake_margin_ms:
            throttle = -min(1.0, (speed - v_target) / 4.0)
            self.integral = min(self.integral, 0.0)
        elif speed < 1.0:
            throttle = max(0.0, throttle)       # never reverse by accident from standstill
        throttle = max(-1.0, min(1.0, throttle))
        info.update(v_target=v_target, kappa_max=k_max)

        # stuck: pushing throttle for stuck_s but the position has not moved (displacement
        # is robust to fix noise, unlike the speed estimate) and the 3-D view is frozen.
        self.history.append((now, st.x, st.y))
        while self.history and now - self.history[0][0] > t.stuck_s:
            self.history.popleft()
        span = now - self.history[0][0]
        moved_m = distance(self.history[0][1:], (st.x, st.y)) * self.mpp
        samples = np.asarray(self.history,dtype=float)
        times = samples[:,0]-samples[:,0].mean()
        variance = float(times @ times)
        window_speed = (float(np.linalg.norm(times @ samples[:,1:] / variance))*self.mpp
                        if variance > 1e-6 else math.inf)
        frozen = motion_level is not None and math.isfinite(motion_level) and 0 <= motion_level < t.stuck_motion_threshold
        grace = now-self.started_at < t.startup_grace_s
        cooling = now < self.recovery.cooldown_until
        info.update(stuck_span_s=span,stuck_moved_m=moved_m,
                    stuck_window_speed_ms=window_speed if math.isfinite(window_speed) else None,
                    recovery_reason='startup_grace' if grace else 'cooldown' if cooling
                    else 'no_visual_motion' if motion_level is None else 'monitoring')
        if (not grace and not cooling and throttle > 0.4 and frozen
                and window_speed <= t.stuck_max_speed_ms and moved_m < t.stuck_move_m):
            if self.slow_since is None:
                self.slow_since = now
            if now - self.slow_since >= t.stuck_s and span >= t.stuck_s * .8:
                if self.recovery.attempts >= max(0,t.max_reverse_attempts):
                    return self._begin_reverse(st,now,'reverse',0.,info)
                here = (st.x, st.y)
                self.stuck_points.append(here)
                nearby = [p for p in self.stuck_points if distance(p, here) * self.mpp < 30]
                if len(nearby) >= t.stuck_attempts:
                    # the road is blocked here: avoid the spot and route around it
                    self.stuck_points = [p for p in self.stuck_points if p not in nearby]
                    self.slow_since = None
                    self.history.clear()
                    # the obstacle is in front: centre the no-go zone ahead of the truck so
                    # the re-plan starts from the road behind it
                    ahead = heading_vector(st.heading) * t.blocked_radius_m * .9 / self.mpp
                    info['blocked_at'] = (st.x + ahead[0], st.y + ahead[1])
                    return Output({'steer': 0.0, 'throttle': 0.0}, 'blocked', info, replan=True)
                turn = -steer if abs(steer) > .2 else (1.0 if alpha < 0 else -1.0)
                return self._begin_reverse(st,now,'reverse',turn,info)
        else:
            self.slow_since = None

        return Output({'steer': steer, 'throttle': throttle}, 'drive', info)


class AircraftPilot:
    def __init__(self, waypoints: list, tuning, meters_per_pixel: float):
        self.waypoints = [tuple(p) for p in waypoints]
        self.t = tuning
        self.mpp = meters_per_pixel
        self.index = 1  # waypoints[0] is the start

    def step(self, st: VehicleState, now: float, motion_level=None, hud=None) -> Output:
        t = self.t
        target = self.waypoints[self.index]
        dist_m = distance(st.position, target) * self.mpp
        last = self.index == len(self.waypoints) - 1
        radius = t.arrive_radius_m if last else t.waypoint_radius_m
        # also advance when we have flown past an intermediate waypoint
        passed = False
        if not last:
            leg = bearing(self.waypoints[self.index - 1], target)
            passed = abs(angle_delta(leg, bearing(target, st.position))) < 80 and dist_m < radius * 4
        if dist_m < radius or passed:
            if last:
                return Output({'roll': 0.0, 'yaw': 0.0, 'pitch': 0.0}, 'arrived',
                              dict(distance_m=dist_m, waypoint=self.index), done=True)
            self.index += 1
            target = self.waypoints[self.index]
            dist_m = distance(st.position, target) * self.mpp

        error = angle_delta(st.heading, bearing(st.position, target))
        info = dict(distance_m=dist_m, waypoint=self.index, heading_error=error, turn_rate=st.yaw_rate)
        roll = yaw = 0.0
        if t.steering == 'yaw' or abs(error) < t.rudder_threshold_deg:
            yaw = max(-1.0, min(1.0, t.rudder_gain * error))
        if t.steering == 'bank':
            rate_cmd = max(-t.max_turn_rate_dps, min(t.max_turn_rate_dps, t.heading_to_rate * error))
            roll = max(-1.0, min(1.0, t.rate_gain * (rate_cmd - st.yaw_rate)))
            if abs(error) < t.rudder_threshold_deg and abs(st.yaw_rate) < 2.0:
                roll = 0.0  # wings-level band; rudder trims the rest
            info['rate_cmd'] = rate_cmd
        pitch = t.turn_pitch_gain * abs(roll)
        return Output({'roll': roll, 'yaw': yaw, 'pitch': pitch}, 'fly', info)


class HeliPilot:
    """Helicopter control from the HUD, identified from real flight logs (2026-10-17).

    Measured in the game:
      * W/S keep tilting the nose (~3.7 ticks/s): pitch is closed on the HUD triangles in
        the FAST loop (``inner``, ~30 Hz).
      * Pitch sets ACCELERATION (~0.76 m/s^2 per tick); speed is held almost level.  Nose-up
        braking is weak (~0.5-0.75 m/s^2), so slowing down must start early.
      * Shift/Ctrl command climb/descent with ~1.2 s delay and ~2.5 s lag (auto-hover holds
        height when released).  The vertical loop therefore runs on a Smith predictor:
        the climb rate expected once the commands already sent have taken effect.

    Phases: ground -> takeoff/climb -> cruise -> approach -> hover -> descend -> flare ->
    settle -> landed (done).  Axes: collective (+ Shift), cyclic (+ W), yaw (+ E).
    """

    def __init__(self, waypoints: list, tuning, meters_per_pixel: float):
        self.t = tuning
        self.mpp = meters_per_pixel
        self.airborne = False
        self.spd_i = 0.0
        self.alt_hist: deque = deque()
        self.ground_hist: deque = deque()
        self.terrain_rate = 0.0
        self.last_time: float | None = None
        self.last_hud = -1e9
        self.vz = 0.0
        self.pitch_target = 0.0
        self.cyclic = 0.0
        self.inner_time: float | None = None
        self.last_pitch: tuple | None = None    # (time, pitch)
        self.pitch_rate = 0.0
        # vertical Smith predictor
        self.cmd_hist: deque = deque()
        self.collective = 0.0
        self.vz_i = 0.0
        self.vz_fast = self.vz_slow = 0.0
        self.terrain = None              # TerrainMemory (set by the autopilot)
        self.open_ground = None          # OpenGround from the base map (set by the autopilot)
        self.velocity = None             # callable(now) -> measured ground velocity m/s (map x, y)
        self._vel_now = None
        self._vg_on = False
        self._turning = False
        self.compass_fresh = None        # callable(now) -> compass read recently (set by the autopilot)
        self.takeoff_ground = None       # ground height (m ASL) where we lifted off
        self.set_waypoints(waypoints)

    def set_waypoints(self, waypoints: list):
        """New route (also mid-air: keeps the flight state, only the targets change)."""
        self.waypoints = [tuple(p) for p in waypoints]
        self.index = 1
        self.landing = None          # None | 'hover' | 'descend' | 'flare' | 'settle' | 'landed'
        self.phase_since = None
        self.touch_since = None
        self.land_here = False       # land on the spot: ignore the route and the position
        self.land_point = None       # where the land key was pressed
        self.spot_checks = 0         # times the touchdown was moved to flatter ground
        self.spot_moved = False      # then fly onto that cell, not just anywhere in the radius
        self.spot_origin = None      # the touchdown point before any terrain move
        self.open_checked = False    # the touchdown was checked for open ground
        self.jhook = None            # {'since', 'side'} while reversing at speed
        self._hook_votes = 0         # consecutive ticks that asked for the J-hook landing
        self.hover_since = None      # when the hover over the landing point began
        self.closest_m = math.inf    # closest pass on the final leg (overshoot detection)

    def land_now(self, point=None):
        """Stop, hover and land where we are.  With ``point`` (the position when the key was
        pressed) it comes back to within ``land_here_radius_m`` of it after braking;
        without a position it lands wherever it stops (HUD only)."""
        self.land_here = True
        self.land_point = tuple(point) if point is not None else None
        if self.land_point is not None:
            spot = self._open_spot(self.land_point, .8 * self.t.land_here_radius_m)
            if spot is not None:
                self.land_point = spot         # forest / water / roofs here: the nearest clearing
        if self.landing not in ('descend', 'flare', 'settle', 'landed'):
            self.landing, self.phase_since, self.hover_since = 'hover', None, None

    # ------------------------------------------------------------------ inner
    def inner(self, pitch: float | None, now: float) -> float:
        """Pitch hold: W/S from the HUD pitch (ticks, + = nose down).  Call at ~30 Hz."""
        t = self.t
        self.inner_time = now
        if pitch is None:
            self.last_pitch = None
            self.cyclic = 0.0          # attitude unknown: hands off, never push blind
            return self.cyclic
        if self.last_pitch is not None and now - self.last_pitch[0] > 1e-3:
            rate = (pitch - self.last_pitch[1]) / (now - self.last_pitch[0])
            self.pitch_rate += (rate - self.pitch_rate) * .5
        self.last_pitch = (now, pitch)
        cyclic = t.pitch_kp * (self.pitch_target - pitch) - t.pitch_kd * self.pitch_rate
        self.cyclic = max(-1.0, min(1.0, cyclic))
        return self.cyclic

    # ------------------------------------------------------------------ estimation
    def _rates(self, now, hud):
        """Climb rate from ASL only (AGL jumps with the terrain), plus the terrain rate
        under us; single-frame OCR glitches are rejected against the running line fit."""
        def fit(hist):
            if len(hist) >= 3:
                ts, vs = np.array(hist).T
                if np.ptp(ts) > .4:
                    return np.polyfit(ts - now, vs, 1)
            return None

        def add(hist, value, tol):
            if value is None:
                return
            line = fit(hist)
            if line is not None and abs(np.polyval(line, 0.0) - value) > tol:
                return
            hist.append((now, value))

        for hist, window in ((self.alt_hist, 1.0), (self.ground_hist, 2.0)):
            while hist and now - hist[0][0] > window:
                hist.popleft()
        add(self.alt_hist, hud.asl_m, 8.0)
        if hud.asl_m is not None and hud.agl_m is not None:
            add(self.ground_hist, hud.asl_m - hud.agl_m, 10.0)
        line = fit(self.alt_hist)
        if line is not None:
            self.vz = float(np.clip(line[0], -20, 20))
        line = fit(self.ground_hist)
        self.terrain_rate = float(np.clip(line[0], -15, 15)) if line is not None else 0.0
        return self.vz

    def _predict_vz(self, now, dt):
        """Smith predictor: measured climb rate + what the commands still in the pipe add."""
        t = self.t
        self.cmd_hist.append((now, self.collective))
        while self.cmd_hist and now - self.cmd_hist[0][0] > t.vert_delay_s + 1.0:
            self.cmd_hist.popleft()
        delayed = self.cmd_hist[0][1]
        for ts, u in self.cmd_hist:
            if ts <= now - t.vert_delay_s:
                delayed = u
        k = min(1.0, dt / t.vert_tau_s)
        self.vz_fast += (t.vert_gain * self.collective - self.vz_fast) * k
        self.vz_slow += (t.vert_gain * delayed - self.vz_slow) * k
        return self.vz + (self.vz_fast - self.vz_slow)

    # ------------------------------------------------------------------ outer
    def hold(self, now: float, hud) -> Output:
        """No position: keep altitude, level out and hover in place."""
        return self._fly(now, hud, None)

    def step(self, st: VehicleState, now: float, motion_level=None, hud=None) -> Output:
        if self.land_here and self.land_point is None:
            return self._fly(now, hud, None)
        if self.land_here:
            self.waypoints, self.index = [self.land_point, self.land_point], 1
        target = self.waypoints[self.index]
        dist_m = distance(st.position, target) * self.mpp
        last = self.index == len(self.waypoints) - 1
        if not last and dist_m < self.t.waypoint_radius_m:
            self.index += 1
            target = self.waypoints[self.index]
            dist_m = distance(st.position, target) * self.mpp
            last = self.index == len(self.waypoints) - 1
        to_target = bearing(st.position, target)
        course = st.course if st.course is not None else st.heading
        nose_error = angle_delta(st.heading, to_target)
        # moving forwards at speed: steer where we GO (the same as the nose unless we slide)
        steer = angle_delta(course, to_target) if st.speed_ms > 8.0 else nose_error
        t = self.t
        set_up = last and t.auto_land and t.jhook_landing and not self.land_here and self.landing is None
        if set_up and t.hook_offset_m > 0 and dist_m > 2 * t.hook_offset_m:
            # J-hook set-up: pass beside the spot, not over it (the tail needs room to swing)
            steer += math.degrees(math.asin(t.hook_offset_m / dist_m))
        nav = dict(error=steer, nose_error=nose_error, dist_m=dist_m, last=last,
                   yaw_rate=st.yaw_rate, speed_ms=st.speed_ms, pos=st.position, target=target,
                   heading=st.heading, bearing=to_target,
                   course=course)
        return self._fly(now, hud, nav)

    def _look_ahead(self, nav, hud, agl, spd, cruise_agl):
        """Highest ground already known along the way -> (height deficit m, speed limit km/h)."""
        t = self.t
        if self.terrain is None or nav is None or hud.asl_m is None:
            return None
        self.terrain.record(nav['pos'], hud.asl_m - agl)
        look_px = max(t.lookahead_min_m, abs(spd) / 3.6 * t.lookahead_s) / self.mpp
        found = self.terrain.highest_ahead(nav['pos'], nav['target'], look_px)
        if found is None:
            return None
        height, dist_px = found
        if self.takeoff_ground is not None and height < self.takeoff_ground + t.rising_ground_m:
            return None      # nothing known ahead above the take-off spot's ground: the height
                             # rules cover it (it used to hold every take-off at 40 km/h for the
                             # ground right next to it); any real rise ahead still counts
        deficit = height + t.lookahead_clear_frac * cruise_agl - hud.asl_m
        if deficit <= 0:
            return None
        climb_time = deficit / t.climb_capacity
        return deficit, 3.6 * dist_px * self.mpp / max(climb_time, 1.0)

    def _jhook(self, now, nav, spd, agl):
        """Reverse direction at speed.  Returns (pitch target, yaw) while hooking, else None.

        Whether the motion follows the nose at once (arcade) or slides on (thrust/drag
        physics) is not known yet, so the cyclic does not assume either: it pushes the
        measured velocity towards 'cruise speed at the target' with the thrust it has
        along the nose - nose up while the nose still points away, nose down (braking
        the old motion and accelerating home at once) when it has come round."""
        t = self.t
        v = abs(spd) / 3.6
        course = nav['course'] if nav['speed_ms'] >= -1.0 else (nav['heading'] + 180) % 360
        err_v, err_n = angle_delta(course, nav['bearing']), nav['nose_error']
        if self._vg_on:
            return None                   # the vector guidance turns round by itself (faster: 30 s vs 37 s)
        if self.jhook is None:
            if not (t.jhook and abs(err_v) > t.jhook_trigger_deg and abs(spd) > t.jhook_min_kmh
                    and agl > t.turn_agl_m):
                return None
            self.jhook = dict(since=now, side=1.0 if err_n >= 0 else -1.0, best=nav['dist_m'], best_t=now)
        elif nav['dist_m'] > self.jhook['best'] + 1000 and now - self.jhook['best_t'] > 20:
            self.jhook = None             # not coming round (bad direction estimate?): normal flight
            return None
        elif (abs(err_v) < t.jhook_exit_deg and abs(err_n) < t.jhook_exit_deg) \
                or now - self.jhook['since'] > t.jhook_max_s or (v < 8 and abs(err_n) < t.jhook_exit_deg):
            self.jhook = None
            return None
        if nav['dist_m'] < self.jhook['best']:
            self.jhook['best'], self.jhook['best_t'] = nav['dist_m'], now
        if abs(err_n) < 150:
            self.jhook['side'] = 1.0 if err_n >= 0 else -1.0     # keep one side through 180 deg
        if abs(err_n) > 30:
            yaw = self.jhook['side'] * t.yaw_max
        else:
            yaw = max(-t.yaw_max, min(t.yaw_max, t.yaw_kp * err_n - t.yaw_kd * nav['yaw_rate']))
        nose = heading_vector(nav['heading'])
        vel = heading_vector(course) * v
        want = heading_vector(nav['bearing']) * t.cruise_kmh / 3.6
        g = 9.81
        drag = g * math.tan(math.radians(10 * t.ref_pitch_ticks)) / (t.ref_speed_kmh / 3.6) ** 2
        a_nose = t.jhook_gain * float(nose @ (want - vel)) + drag * v * float(nose @ vel)
        pitch = math.degrees(math.atan(a_nose / g)) / 10
        return max(-t.pitch_back_max, min(t.pitch_fwd_max, pitch)), yaw

    def _vector(self, now, nav, v_target_kmh, spd, stop_m):
        """Velocity-vector guidance -> (pitch ticks, heading error to fly, info) or None.

        The game's helicopter keeps its velocity; only rotor thrust along the nose (and drag)
        changes it - measured from the real flights (``fit-slip``: best fit 0.02/s, i.e. the
        motion hardly follows the nose by itself).  Pointing the nose at the target therefore
        circles it.  Instead: want a velocity (towards the target; on the last leg capped by a
        braking curve that stops on the point - the predicted stop point), take the
        acceleration that gets there (plus drag), point the nose along it (or against it,
        nose up - whichever needs less turning) and tilt by its size."""
        t = self.t
        vel = self._vel_now
        if vel is None:
            # only on a MEASURED velocity (minimap fixes): guessed from the unsigned HUD speed
            # it thrust the wrong way near the hover and ran away (sim, minimap lost: 91 km/h
            # in the flare).  Without fixes the previous steering flies.
            return None
        vel = np.asarray(vel, dtype=float)
        dist = nav['dist_m']
        to = heading_vector(nav['bearing'])
        v_want = max(0.0, v_target_kmh) / 3.6
        if stop_m is not None:
            # predictive: never faster than what can still stop on the point, allowing for the
            # ~1 s the helicopter takes to answer
            v = float(np.linalg.norm(vel))
            room = max(0.0, dist - stop_m - v * t.hook_lag_s)
            v_want = min(v_want, math.sqrt(2 * t.vg_decel_ms2 * room))
        want = to * v_want
        drag = 9.81 * math.tan(math.radians(10 * t.ref_pitch_ticks)) / (t.ref_speed_kmh / 3.6) ** 2
        speed = float(np.linalg.norm(vel))
        acc = t.vg_gain * (want - vel) + drag * speed * vel
        a = float(np.linalg.norm(acc))
        v_stop = speed * speed / (2 * t.vg_decel_ms2)
        info = dict(v_want=round(v_want * 3.6), acc=round(a, 2),
                    stop_ahead_m=round(v_stop + speed * t.hook_lag_s))
        if a < t.vg_dead_ms2:
            return 0.0, 0.0, info
        heading = nav['heading']
        h_acc = math.degrees(math.atan2(acc[0], -acc[1])) % 360
        e_fwd = angle_delta(heading, h_acc)
        e_back = angle_delta(heading, (h_acc + 180) % 360)
        braking = speed > 3.0 and float(acc @ vel) < 0
        # braking: whichever needs less turning (nose up, or nose down if we slide backwards);
        # speeding up towards where we want to go: the nose faces the thrust (turn round,
        # never fly backwards - it did, at 40 km/h, the whole way in the first try)
        forward = abs(e_fwd) <= abs(e_back) if braking or v_want < 3 else True
        reversing = v_want > 12 and abs(angle_delta(math.degrees(math.atan2(vel[0], -vel[1])) % 360,
                                                    nav['bearing'])) > 120
        if braking and reversing and speed < t.vg_pivot_kmh / 3.6:
            # turning round: start the 180 deg pivot while the last of the speed is braked off
            # (a pivot takes ~4 s; stopping first and then pivoting wasted that time)
            forward = True
        info['reversing'] = reversing
        err = e_fwd if forward else e_back
        # thrust only once the nose is (nearly) where it should be: full thrust 38 deg off the
        # wanted direction is exactly the centripetal force of an orbit round the point
        along = max(0.0, math.cos(math.radians(err))) ** 4
        tilt = math.degrees(math.atan(a * along / 9.81)) / 10
        info.update(nose_for='accel' if forward else 'brake', acc_heading=round(h_acc))
        return (tilt if forward else -tilt), err, info

    def _hook_distance(self, spd) -> float:
        """Distance to start the J-hook from this speed: braking room + a margin."""
        v = abs(spd) / 3.6
        return v * v / (2 * self.t.hook_decel_ms2) * 1.15 + self.t.hook_end_m

    def _enter(self, landing, now):
        self.landing, self.phase_since = landing, now
        self.spd_i = 0.0      # a speed integral wound up in cruise would hold the nose down while braking

    def _open_spot(self, target, radius_m):
        """The nearest clearly open ground (base map) if ``target`` is forest, water or town."""
        t = self.t
        og = self.open_ground() if callable(self.open_ground) else self.open_ground
        if og is None or not t.open_ground:
            return None
        here = og.cost(*target)
        if here <= t.open_ok_cost:
            return None
        spot, cost = og.best_near(target, radius_m)
        if spot is None or cost > t.open_max_cost or cost > here - t.open_gain:
            return None
        return spot

    def _flatter_spot(self, nav, info) -> bool:
        """Before going down: if the terrain memory says the touchdown cell is a slope, a
        ledge or a forest edge, move the touchdown to the flattest known cell nearby."""
        t = self.t
        if self.terrain is None or nav is None or self.spot_checks >= 2:
            return False
        target = nav['target']
        rough = self.terrain.roughness(*target)
        if rough is None:
            return False                  # not seen well enough yet: ask again next tick
        self.spot_checks += 1
        if rough <= t.flat_max_m:
            self.spot_checks = 2          # flat: settled
            return False
        # both moves stay within reach of the ORIGINAL point (where F10 was pressed / the goal)
        self.spot_origin = getattr(self, 'spot_origin', None) or tuple(target)
        shift = t.land_here_radius_m if self.land_here else t.spot_shift_m
        spot = self.terrain.flattest_near(self.spot_origin, shift / self.mpp, t.flat_max_m)
        if spot is None or distance(spot, target) * self.mpp < 10:
            return False
        info['moved_for_terrain'] = round(rough, 1)
        self.spot_moved = True
        self.hover_since = None
        if self.land_here:
            self.land_point = spot
        else:
            self.waypoints[-1] = spot
        if nav is not None:
            nav['target'] = spot
        return True

    def _land_anyway(self, now) -> bool:
        """Hovered near the point this long without centring (position noise, wind): land."""
        return (self.land_here and self.land_point is None) or (
            self.hover_since is not None and now - self.hover_since > self.t.land_anyway_s)

    def _fly(self, now, hud, nav) -> Output:
        t = self.t
        dt = 0.1 if self.last_time is None else max(1e-3, min(.5, now - self.last_time))
        self.last_time = now
        neutral = {'collective': 0.0, 'cyclic': 0.0, 'yaw': 0.0}
        if hud is not None and hud.agl_m is not None:
            self.last_hud = now
        if hud is None or hud.agl_m is None:
            self.pitch_target = self.collective = 0.0
            mode = 'hud lost' if now - self.last_hud > t.hud_timeout_s else 'hold'
            return Output(neutral, mode, {})
        if self.landing == 'landed':
            self.pitch_target = 0.0
            return Output(neutral, 'landed', {'agl': hud.agl_m}, done=True)
        agl = hud.agl_m
        speed_ms = nav['speed_ms'] if nav else 0.0
        spd = hud.speed_kmh if hud.speed_kmh is not None else abs(speed_ms) * 3.6
        if speed_ms < -1.0:
            spd = -spd                    # the HUD speed has no sign
        self._rates(now, hud)
        vz_pred = self._predict_vz(now, dt)
        # measured ground velocity (minimap fixes) this tick, or None
        self._vel_now = self.velocity(now) if callable(self.velocity) and nav is not None else None
        error = nav['error'] if nav else 0.0
        dist = nav['dist_m'] if nav else None
        # velocity-vector guidance: everywhere (vector_guidance), or only for the final approach
        # and landing (vector_landing) - and only on a measured velocity and a compass read in
        # the last 2 s (17:00: unreadable compass at 1080p, the nose estimate jumped -> circling)
        in_zone = nav is not None and (self.landing is not None or (
            nav['last'] and (t.auto_land or self.land_here) and dist < t.vector_landing_m))
        if (nav is not None and self._vel_now is not None and t.vector_turnround
                and float(np.hypot(*self._vel_now)) * 3.6 > t.jhook_min_kmh):
            course_now = math.degrees(math.atan2(self._vel_now[0], -self._vel_now[1])) % 360
            if abs(angle_delta(course_now, nav['bearing'])) > t.jhook_trigger_deg:
                self._turning = True
                # the target is behind the way we are going (F7 while flying away): turn round
                # with the vector guidance - brake straight, then pivot; the old yaw-first
                # turn-back drifted 470 m sideways in the measured sliding physics
                in_zone = True
        if self._turning and nav is not None and self._vel_now is not None:
            # stay on it until we are really flying home - handing back mid-turn to the old
            # steering cost 8 s and 250 m of drift in the simulation
            v = float(np.hypot(*self._vel_now)) * 3.6
            course_now = math.degrees(math.atan2(self._vel_now[0], -self._vel_now[1])) % 360
            if abs(angle_delta(course_now, nav['bearing'])) < 25 and v > 120:
                self._turning = False
            else:
                in_zone = True
        compass_ok = self.compass_fresh(now) if callable(self.compass_fresh) else True
        self._vg_on = (self._vel_now is not None and compass_ok
                       and (t.vector_guidance or (t.vector_landing and in_zone)))
        info = dict(heading_error=error, agl=agl, speed_kmh=spd, vz=self.vz, vz_pred=vz_pred,
                    terrain_rate=self.terrain_rate, pitch=hud.pitch, lever=hud.collective)
        if nav:
            info.update(distance_m=dist, waypoint=self.index)

        cruise_agl = t.cruise_agl_m
        if t.auto_land and nav and nav['last'] and self.landing is None and self.airborne:
            # glide path: lose height while braking so we arrive low over the landing point
            cruise_agl = float(np.clip(t.glide_ratio * dist, t.glide_min_agl_m, cruise_agl))
        elif (t.auto_land or self.land_here) and self.landing == 'hover':
            cruise_agl = min(cruise_agl, t.glide_min_agl_m)   # keep going down, never back up
            radius = t.land_here_radius_m if self.land_here else t.hover_radius_m
            if nav is None or nav['dist_m'] < radius:
                cruise_agl = min(cruise_agl, t.land_hover_agl_m)   # over the spot: get low already
        if self.landing is None:
            if agl < t.ground_agl_m:
                self.airborne = False     # (back) on the ground: restart with a level lift-off
                if hud.asl_m is not None:
                    self.takeoff_ground = hud.asl_m - agl
            elif agl >= t.takeoff_agl_m:
                self.airborne = True
        # authority grows with height: full nose-down / speed from ``full_authority_agl_m`` on
        # (was 0.8 x cruise height = 120 m: take-offs crawled); the height-for-speed rule
        # (clearance_m_per_kmh) and the terrain look-ahead keep it safe
        # fast climb-out only where the terrain memory knows the way ahead is not rising (a
        # first flight towards an unknown 60 % mountainside could not out-climb it at speed)
        clear_ahead = (nav is not None and self.terrain is not None and hud.asl_m is not None and
                       self.terrain.known_ahead(nav['pos'], nav['target'], t.known_ahead_m / self.mpp,
                                                hud.asl_m - agl + t.rising_ground_m))
        top = min(.8 * cruise_agl, t.full_authority_agl_m) if clear_ahead else .8 * cruise_agl
        margin = float(np.clip((agl - t.turn_agl_m) / max(1.0, top - t.turn_agl_m), 0.0, 1.0))
        # pedal turn right after lift-off while still slow (the helicopter turns fastest in a
        # hover), so it already faces the target when it reaches the height to fly forward
        hover_turn = agl >= 3.0 and abs(spd) < 30 and self.landing is None
        yaw_ok = agl >= t.turn_agl_m or self.landing is not None or hover_turn
        vz_target = max(-t.vz_max, min(t.vz_max, t.alt_kp * (cruise_agl - agl) + max(0.0, self.terrain_rate)))
        if self.landing == 'hover' and agl > 40:
            # high above the landing area: sink faster than in cruise (the descent from cruise
            # height is the longest part of a landing)
            vz_target = max(-t.land_fast_vz, min(vz_target, t.alt_kp * (cruise_agl - agl)))
        move_to_target = False

        # ------------------------------------------------ phase logic
        if (nav and nav['last'] and t.auto_land and not self.land_here and not self.open_checked
                and self.landing is None and dist < t.open_check_m):
            # final approach: land in the open (meadow, field) rather than in trees or a town
            self.open_checked = True
            spot = self._open_spot(nav['target'], t.open_radius_m)
            if spot is not None:
                info['open_spot'] = [round(spot[0], 1), round(spot[1], 1)]
                self.waypoints[-1] = spot
                nav['target'] = spot
                dist = nav['dist_m'] = distance(nav['pos'], spot) * self.mpp
                nav['nose_error'] = angle_delta(nav['heading'], bearing(nav['pos'], spot))
        if nav and nav['last'] and self.landing in (None, 'jhook'):
            self.closest_m = min(self.closest_m, dist)
        want_hook = (self.landing is None and self.airborne and nav and nav['last'] and t.auto_land
                     and t.jhook_landing and not self.land_here and dist > t.hook_end_m
                     and abs(spd) <= t.hook_max_kmh and dist <= self._hook_distance(spd))
        # only on a steady reading: one garbled HUD speed (534 km/h at 12:54, 3.3 km out) started
        # the hook and flew the helicopter into the ground
        self._hook_votes = self._hook_votes + 1 if want_hook else 0
        if self._hook_votes >= 3:
            self._enter('jhook', now)
        if self.landing == 'jhook' and dist > 1.3 * self._hook_distance(spd) + 150:
            self.landing = None           # nowhere near braking distance: back to the approach
        if self.landing == 'jhook':
            # J-hook (the WARDOGS landing): brake hard and sink with the collective down at the
            # same time, nose locked on the spot, planned to arrive over it slow and low
            mode = 'j-hook'
            # the helicopter answers ~1 s late: plan to stop that much earlier
            room = max(0.0, dist - t.hook_end_m - abs(spd) / 3.6 * t.hook_lag_s)
            # a hook only ever slows down: never more speed than we have
            v_target = min(abs(spd), 3.6 * math.sqrt(2 * t.hook_decel_ms2 * room))
            t_go = max(1.5, 2 * room / max(abs(spd) / 3.6, 2.0))
            # sink on the height we will be at once the collective has taken effect (~3.7 s)
            agl_ahead = max(0.0, agl + min(0.0, vz_pred) * .5 * (t.vert_delay_s + t.vert_tau_s))
            sink_max = max(2.0, min(t.land_fast_vz, .25 * agl))
            vz_target = float(np.clip(-(agl_ahead - t.hook_end_agl_m) / t_go, -sink_max, 2.0))
            error = nav['nose_error'] if dist > 20 else 0.0     # no spinning over the spot
            # pitch straight from the physics: the deceleration that stops us at the spot
            # (v^2 / 2d) minus what drag already gives, as thrust along the nose
            v = abs(spd) / 3.6
            drag = 9.81 * math.tan(math.radians(10 * t.ref_pitch_ticks)) / (t.ref_speed_kmh / 3.6) ** 2
            a_need = -v * v / (2 * max(room, 5.0))
            thrust = a_need + drag * v * v
            # + strong feedback on the real speed: the model expected 4.9 m/s^2 from drag at
            # 280 km/h, the game gave ~2.6 (23:28) - the hook fell behind and overshot
            self._hook_pitch = math.degrees(math.atan(thrust / 9.81)) / 10 + t.hook_kv * (v_target - spd)
            # never more nose-down than it takes to hold the present speed against drag
            hold = math.degrees(math.atan(drag * v * v / 9.81)) / 10
            self._hook_pitch = min(self._hook_pitch, hold)
            info.update(hook_decel=round(a_need, 2))
            passed = dist > self.closest_m + 10.0 and self.closest_m < t.hover_radius_m
            passed = passed or dist > self.closest_m + t.overshoot_m
            if (dist < t.hover_radius_m and abs(spd) < t.descend_max_kmh) or passed \
                    or (abs(spd) < t.descend_max_kmh and dist >= t.hover_radius_m):
                # over the spot and slow: straight down if already low; otherwise (missed or
                # stopped short) the hover logic centres it first
                low = agl < 2 * t.hook_end_agl_m and not passed and dist < t.hover_radius_m
                self._enter('descend' if low else 'hover', now)
                self.hover_since = None
        if self.landing is None and self.airborne and nav and nav['last'] and t.auto_land:
            # stop in the landing zone, or as soon as we fly past the point (real flight
            # 01:52: a lateral miss at 160 km/h never came within the 40 m hover radius)
            passed = self.closest_m < 2 * t.land_zone_m and dist > self.closest_m + t.overshoot_m
            if dist < t.land_zone_m or passed:
                self._enter('hover', now)
                self.hover_since = None
        elif self.landing is None and self.airborne and nav and nav['last'] and dist < t.hover_radius_m:
            self._enter('hover', now)     # auto_land off: hover over the point and hand over
        far = 1.5 * t.land_here_radius_m if self.land_here else 2.5 * t.hover_radius_m
        if self.landing in ('descend', 'flare') and dist is not None and dist > far \
                and agl > 15 and not self._land_anyway(now):
            self._enter('hover', now)     # drifted off: go round and settle again
        if self.landing == 'hover' and not self.land_here and dist is not None and dist > 2.5 * t.land_zone_m:
            self.landing, self.closest_m = None, dist   # far past (or a position jump): approach again
        if self.landing == 'hover':
            mode, v_target = 'hover', 0.0
            move_to_target = True
            if self.spot_checks < 2 and not self.spot_moved and nav is not None and self._flatter_spot(nav, info):
                dist = nav['dist_m'] = distance(nav['pos'], nav['target'] if not self.land_here
                                                else self.land_point) * self.mpp
            if self.hover_since is None or (dist is not None and dist > t.land_zone_m):
                self.hover_since = now        # the land-anyway clock runs inside the zone only
            radius = t.land_here_radius_m if self.land_here and not self.spot_moved else t.hover_radius_m
            centred = dist is None or dist < radius or self._land_anyway(now)
            steady = abs(spd) < t.descend_max_kmh and abs(vz_pred) < 3.0
            if not steady or self.phase_since is None:
                self.phase_since = now
            elif now - self.phase_since > t.hover_settle_s and centred and self._flatter_spot(nav, info):
                self.phase_since = now                # going there first
            elif now - self.phase_since > t.hover_settle_s and centred:
                if t.auto_land or self.land_here:
                    self._enter('descend', now)
                else:
                    self._enter('landed', now)
                    return Output(neutral, 'arrived', info, done=True)
        if self.landing in ('descend', 'flare'):
            move_to_target = True
            v_target = 0.0
            # the climb rate answers ~3.7 s late: schedule the descent on the height we will
            # be at by then, so it has already slowed down when the skids get close
            agl_ahead = max(0.0, agl + min(0.0, vz_pred) * (t.vert_delay_s + t.vert_tau_s))
            if agl_ahead <= t.flare_agl_m or agl <= t.flare_agl_m:
                self.landing = 'flare'
            vz_target = -t.flare_vz if self.landing == 'flare' else \
                -float(np.clip(t.descent_rate_per_m * agl_ahead, t.flare_vz, t.land_vz_max))
            info['agl_ahead'] = agl_ahead
            mode = self.landing
            # the ground under us must hold still (no sliding over a slope, a ledge or trees)
            # and we must be (almost) stopped before the skids go down
            uneven = abs(self.terrain_rate) > t.ground_steady_ms and agl < 25
            moving = abs(spd) > t.touchdown_max_kmh and agl < 2 * t.flare_agl_m
            if uneven or moving:
                self.hold_since = getattr(self, 'hold_since', None) or now
            else:
                self.hold_since = None
            if (uneven or moving) and now - self.hold_since > (8.0 if uneven else 20.0):
                uneven = moving = False       # holding does not settle it: go down anyway, slowly
            if uneven or moving:
                vz_target = max(vz_target, 0.0 if agl > 3 else .5)
                mode += ' (hold: ' + ('uneven ground' if uneven else 'still moving') + ')'
                self.touch_since = None
            # the HUD shows whole metres: 0 = on the skids.  Only then lower the collective.
            if agl <= 0.5 and not moving:
                self.touch_since = self.touch_since or now
                if now - self.touch_since > t.touch_confirm_s:
                    self._enter('settle', now)
            else:
                self.touch_since = None
        if self.landing == 'settle':
            # on the skids: lower the collective fully, keep level, then hand over
            self.pitch_target = 0.0
            self.collective = -1.0
            if now - self.phase_since > t.settle_s:
                self._enter('landed', now)
                return Output(neutral, 'landed', info, done=True)
            return Output({'collective': -1.0, 'cyclic': self.cyclic, 'yaw': 0.0}, 'settle', info)

        if self.landing is None:
            if not self.airborne:
                mode, v_target = ('ground' if agl < t.ground_agl_m else 'takeoff'), 0.0
                vz_target = t.takeoff_vz
            elif nav is None:
                mode, v_target = 'no position', 0.0
            else:
                mode, v_target = 'cruise', t.cruise_kmh
                if nav['last']:      # brake early: the helicopter decelerates slowly
                    stop = max(0.0, dist - t.hover_radius_m * .5)
                    v_target = min(v_target, 3.6 * math.sqrt(2 * t.decel_ms2 * stop))
                    if v_target < t.cruise_kmh:
                        mode = 'approach'
                if abs(error) > t.turn_slow_deg and not self._vg_on:
                    v_target = min(v_target, t.turn_kmh)
                v_target *= max(t.low_speed_frac, margin)          # climb before going fast
                # rising ground: the ground rises at slope x speed.  While we are low, fly
                # only so fast that our real climb capacity still GAINS clearance.  (Real
                # flight 22:32: ground rising 10-13 m/s at 110-140 km/h, 35 m up -> ridge.)
                slope = self.terrain_rate / max(abs(spd) / 3.6, 5.0)
                if slope > t.terrain_slope_min and agl < t.terrain_low_frac * cruise_agl:
                    gain = t.climb_capacity - t.clearance_gain
                    v_allowed = 3.6 * max(0.0, gain) / slope
                    # steep (> 30 %): no 80 km/h floor - at 80 km/h a 60 % slope rises 13 m/s
                    floor = t.terrain_min_kmh if agl > .5 * cruise_agl and slope < .3 else t.turn_kmh
                    if v_allowed < v_target:
                        v_target = max(floor, v_allowed)
                        mode = 'terrain'
        if self.landing is None and self.airborne and nav is not None:
            # clearance grows with speed: no reaction time at 300 km/h 40 m above a slope
            per_kmh = t.known_clearance_m_per_kmh if clear_ahead else t.clearance_m_per_kmh
            v_target = min(v_target, max(t.turn_kmh, agl / per_kmh))
            info['clear_ahead'] = clear_ahead
            # terrain memory: climb early for ground already known along the way
            ahead = self._look_ahead(nav, hud, agl, spd, cruise_agl)
            if ahead is not None:
                deficit, v_allowed = ahead
                vz_target = max(vz_target, min(t.vz_max, t.alt_kp * deficit + 2.0))
                if v_allowed < v_target:
                    v_target = max(t.turn_kmh, v_allowed)
                    mode = 'terrain ahead'
                info['terrain_deficit'] = deficit
        elif self.terrain is not None and nav is not None and hud.asl_m is not None:
            self.terrain.record(nav['pos'], hud.asl_m - agl)
        if move_to_target and nav is not None:
            # creep towards the landing point when facing it (pitch makes the helicopter move)
            cap = t.creep_kmh if dist < t.hover_radius_m or self.landing != 'hover' else t.reposition_kmh
            vectored = self._vg_on
            v_target = min(cap, .5 * max(0.0, dist - (2 if vectored else 5))) if abs(error) < 30 or vectored else 0.0
            near = self.land_here and not self.spot_moved and dist < .7 * t.land_here_radius_m
            if near or (self.landing in ('descend', 'flare') and agl < 10):
                v_target = 0.0          # close enough / nearly down: just stop, do not creep on
        sinking_low = self.airborne and self.landing is None and agl < t.min_agl_m and vz_pred < -1.0
        # ground closing in: dAGL/dt = own climb - ground rise.  This catches rising ground
        # even while we ARE climbing (the crash at 22:32 climbed +5..8 m/s into a ridge).
        agl_rate = self.vz - self.terrain_rate
        closing = self.airborne and self.landing is None and agl < .6 * cruise_agl and agl_rate < -1.0
        tti = (agl - t.min_agl_m) / -agl_rate if closing else math.inf
        info.update(agl_rate=agl_rate, tti=None if math.isinf(tti) else tti)
        pull_up = sinking_low or tti < t.tti_brake_s
        if pull_up:
            mode, vz_target = 'pull up', t.vz_max
            v_target = min(v_target, max(0.0, spd - (80.0 if tti < .5 * t.tti_brake_s else 40.0)))

        hook = None
        if self.landing is None and self.airborne and nav is not None:
            hook = self._jhook(now, nav, spd, agl)
        elif self.jhook is not None:
            self.jhook = None
        if hook is not None and not pull_up:
            mode = 'turn back'
            vz_target = max(vz_target, t.jhook_climb_ms)

        # ------------------------------------------------ speed -> pitch (inner loop flies it)
        moving_low = self.airborne and abs(spd) > 40       # fast below the turn height: must brake, not level
        if self.landing in ('jhook', 'hover', 'descend', 'flare') and abs(spd) > 3:
            moving_low = True             # landing and still drifting: keep braking down to the ground
        if (self.airborne or self.landing) and (moving_low or
                                                agl >= min(t.turn_agl_m, 8.0 if self.landing else t.turn_agl_m)):
            self.spd_i = max(-t.spd_i_max, min(t.spd_i_max, self.spd_i + t.spd_ki * (v_target - spd) * dt))
            if pull_up:
                # emergency: a nose-down integral from cruise must not cancel the braking (it held
                # 32 km/h through a pull-up below a mountainside)
                self.spd_i = min(self.spd_i, 0.0)
            # feed-forward: the pitch that holds v_target against drag (~speed^2), anchored
            # on the player figure "arrow at the bottom notch (~3 ticks) holds ~300 km/h"
            ratio = max(0.0, v_target) / t.ref_speed_kmh
            pitch_ff = math.degrees(math.atan(math.tan(math.radians(10 * t.ref_pitch_ticks)) * ratio * ratio)) / 10
            pitch_target = pitch_ff + t.spd_to_pitch * (v_target - spd) + self.spd_i
            info['pitch_ff'] = pitch_ff
        else:
            self.spd_i, pitch_target = 0.0, 0.0
        fwd = t.pitch_fwd_max * (.3 + .7 * margin)
        back = t.pitch_back_max * (.4 + .6 * margin)
        # brake nose-up only while moving forwards, and less as we slow down: braking past
        # zero would start flying backwards (seen in simulation with the unsigned HUD speed)
        back = min(back, max(0.0, spd) / t.brake_kmh_per_tick)
        fwd = min(fwd, max(.15, fwd if spd >= 0 else .15 + max(0.0, -spd) / t.brake_kmh_per_tick))
        # sinking LOW: nose-down costs lift, keep it gentle (braking nose-up is still allowed).
        # Up high it does not matter - at 250-350 m it held 229 of 407 cruise ticks to 0.4 tick
        # while descending to cruise height (23:12: ~200 km/h instead of 300)
        sinking_low_alt = vz_pred < -4.0 and agl < .6 * t.cruise_agl_m
        if sinking_low_alt:
            fwd = min(fwd, .4)
        if self.landing in ('jhook', 'hover', 'descend', 'flare'):
            # stopping for a landing: brake as hard as players do (270 -> 50 km/h in seconds), and
            # keep a full tick of authority down to ~12 km/h so the last bit does not creep on
            fast = max(0.0, spd) / (t.brake_kmh_per_tick if self.landing != 'jhook' else t.hook_kmh_per_tick)
            slow = min(1.0, max(0.0, spd) / 12.0)
            back = min(t.brake_pitch_max if self.landing in ('jhook', 'hover') and agl > 20 else 1.0, max(fast, slow))
        if self.landing in ('descend', 'flare'):
            fwd = min(fwd, .4)
        if pull_up:                      # full braking authority, never nose-down
            back = min(t.pitch_back_max, max(0.0, spd) / t.brake_kmh_per_tick)
            fwd = 0.0
        pitch_target = max(-back, min(fwd, pitch_target))
        if hook is not None and not pull_up:
            pitch_target = hook[0]          # full authority both ways: steering the velocity vector
            if sinking_low_alt:
                pitch_target = min(pitch_target, .4)
        if self.landing == 'jhook':
            pitch_target = max(-back, min(fwd, self._hook_pitch))
        vector = None
        if (self._vg_on and nav is not None and self.airborne
                and (moving_low or agl >= min(t.turn_agl_m, 8.0 if self.landing else t.turn_agl_m))):
            last_leg = nav['last'] and (t.auto_land or self.land_here)
            stop_m = None
            if last_leg:
                stop_m = t.hook_end_m if self.landing in (None, 'jhook') else 0.0
            vector = self._vector(now, nav, v_target, spd, stop_m)
        if vector is not None:
            v_pitch, v_err, v_info = vector
            info.update(v_info)
            # nose-up may now brake fully at any forward/backward speed: the velocity is measured
            # braking hard is safe high up: the full nose-up authority whenever we are braking
            # for a landing, a pull-up or a turn round (2.5 -> 3.2 ticks)
            hard = self.landing in ('jhook', 'hover') or pull_up or (v_info.get('reversing') and agl > 20)
            up = t.brake_pitch_max if hard else t.pitch_back_max
            down = t.pitch_fwd_max * (.3 + .7 * margin)
            if self.landing in ('descend', 'flare') or agl < 8:
                up, down = min(up, 1.0), min(down, .4)
            if sinking_low_alt:
                down = min(down, .4)
            if pull_up:
                down = 0.0
            pitch_target = max(-up, min(down, v_pitch))
        if sinking_low:
            pitch_target = min(pitch_target, 0.0)
        self.pitch_target = pitch_target
        if self.inner_time is None or now - self.inner_time > .3:   # no fast loop running
            self.inner(hud.pitch, now)

        # ------------------------------------------------ collective on the predicted climb rate
        if not self.airborne and self.landing is None and agl < t.ground_agl_m:
            collective = 1.0               # hold Shift until we are off the ground
        elif self.landing == 'flare' and agl <= 2.0:
            collective = -t.touchdown_collective   # last metres: a gentle, steady sink
        else:
            err = vz_target - vz_pred
            self.vz_i = max(-t.vz_i_max, min(t.vz_i_max, self.vz_i + t.vz_ki * err * dt))
            collective = t.vz_kp * err + self.vz_i
        if collective != collective or abs(collective) >= 1.0:
            self.vz_i *= .98                  # anti wind-up while saturated
        self.collective = max(-1.0, min(1.0, collective))

        # ------------------------------------------------ heading
        near = 20.0 if self.landing in ('descend', 'flare') else 8.0     # going down: hold the heading
        if self._vg_on:
            near = 5.0                    # the thrust can be pointed anywhere: keep centring
        if not yaw_ok or nav is None or (move_to_target and dist is not None and dist < near):
            yaw = 0.0
            if self.landing is None and mode in ('takeoff', 'cruise', 'approach', 'terrain'):
                mode = 'climb'
        else:
            yaw_max = 1.0 if hover_turn else t.yaw_max
            yaw = max(-yaw_max, min(yaw_max, t.yaw_kp * error - t.yaw_kd * nav['yaw_rate']))
        if hook is not None and yaw_ok:
            yaw = hook[1]
            mode = 'pull up' if pull_up else 'turn back'
        if vector is not None and yaw_ok and not (move_to_target and dist is not None and dist < near):
            # the nose goes where the thrust is needed (see _vector), not at the target
            yaw = max(-1.0, min(1.0, t.vg_yaw_kp * vector[1] - t.yaw_kd * nav['yaw_rate']))
        info.update(v_target=v_target, vz_target=vz_target, pitch_target=pitch_target, margin=margin,
                    landing=self.landing, land_here=self.land_here)
        return Output({'collective': self.collective, 'cyclic': self.cyclic, 'yaw': yaw}, mode, info)
