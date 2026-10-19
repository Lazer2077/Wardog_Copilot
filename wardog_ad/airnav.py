"""Helicopter navigation without depending on the minimap.

Once the start position is known (one good minimap fix, usually on the ground, or the
return point marked on the map), the position is carried forward with the HUD airspeed
and the compass heading (dead reckoning).  Minimap fixes are optional corrections: a
fix is blended in only when it is confident and consistent with the estimate, so a
bad match over featureless terrain cannot throw the helicopter off course.
"""
from __future__ import annotations

from collections import deque
import math

import numpy as np

from .estimator import VehicleState
from .geometry import angle_delta, distance


DRAG = 8.2e-4    # 1/m: g*tan(30 deg) holds ~300 km/h (player cruise figure)
SLIDE = 0.02     # 1/s: the slide measured with fit-slip (13:16 flight)


class MotionFilter:
    """Position + velocity (metres, map x / y) under one model of how the motion follows
    the nose: ``align`` 1/s (inf = at once, arcade).  Fed by the rotor thrust (pitch along
    the nose, drag against the motion), the HUD speed (size only) and the minimap fixes.

    A line fit through the raw fixes is far too noisy to hover on (5-13 m fix noise:
    50-180 km/h 'measured' at a 5 km/h hover, sim); the HUD speed is exact to ~1 km/h but
    has no sign and no direction; together with the thrust they give ~1-2 km/h."""

    ACC_NOISE = 1.5      # m/s^2 not explained by the model
    MEMORY = .97         # per fix: how fast the score forgets (~5 s at 7 fixes/s)

    def __init__(self, align: float):
        self.align = align
        self.x = np.zeros(4)
        self.P = np.diag([100.0, 100.0, 25.0, 25.0])
        self.score = 0.0             # log-likelihood of the recent fixes under this model

    def reset(self, pos_m, sigma_m):
        self.x[:2] = pos_m
        self.P[:2, :] = 0.0
        self.P[:, :2] = 0.0
        self.P[0, 0] = self.P[1, 1] = sigma_m ** 2

    def predict(self, dt, nose, pitch, seed=None):
        x = self.x
        v = x[2:].copy()
        if seed is not None and np.hypot(*v) < 1.0:
            v = np.asarray(seed, dtype=float)
        speed = float(np.hypot(*v))
        acc = np.zeros(2)               # pitch unknown: assume it holds the speed
        if pitch is not None:
            acc = 9.81 * math.tan(math.radians(10 * pitch)) * nose - DRAG * speed * v
        v_new = v + acc * dt
        along = float(v_new @ nose)
        if speed > .5 and (along > 0 or math.isinf(self.align)):
            # the motion swings onto the nose axis (arcade: at once, also backwards)
            size = float(np.hypot(*v_new))
            axis = nose if along >= 0 else -nose
            perp = np.array([-axis[1], axis[0]])
            keep = 0.0 if math.isinf(self.align) else math.exp(-self.align * dt)
            ang = math.atan2(float(v_new @ perp), abs(float(v_new @ axis))) * keep
            v_new = size * (math.cos(ang) * axis + math.sin(ang) * perp)
        x[:2] += v_new * dt             # like the dead reckoning: the new heading for the whole step
        x[2:] = v_new
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.P = F @ self.P @ F.T + self.ACC_NOISE ** 2 * G @ G.T

    def update_speed(self, s_hud):
        H = np.hstack([np.zeros((2, 2)), np.eye(2)])
        size = float(np.hypot(*self.x[2:]))
        if s_hud < .2:
            # the HUD shows 0 km/h: hovering, the velocity is near zero in every direction
            self.update(np.zeros(2), np.eye(2) * .3 ** 2, H)
        elif size > max(.3, .5 * s_hud):
            # the size of the motion only: h(v) = |v| (whole km/h on the HUD: +-0.14 m/s)
            u = self.x[2:] / size
            sigma = .25 + .02 * s_hud
            self.update(np.array([s_hud]), np.array([[sigma ** 2]]), np.array([[0.0, 0.0, u[0], u[1]]]),
                        h=np.array([size]))

    def update_fix(self, z, sigma_fix, floor_m=None):
        if floor_m is not None:
            # after a gap: the filter's own spread knows nothing of a biased HUD speed or
            # compass - at least the dead-reckoning drift since the last fix.  (On every fix
            # it cut the link between position and velocity: the fixes moved the position
            # and never corrected a velocity pointing the wrong way.)
            for i in (0, 1):
                self.P[i, i] = max(self.P[i, i], floor_m ** 2)
        H = np.hstack([np.eye(2), np.zeros((2, 2))])
        R = np.eye(2) * sigma_fix ** 2
        y = z - self.x[:2]
        S = H @ self.P @ H.T + R
        d2 = float(y @ np.linalg.solve(S, y))
        self.score = self.MEMORY * self.score - .5 * (min(d2, 50.0) + math.log(np.linalg.det(S)))
        if d2 > 9.0:
            # the fixes keep disagreeing: too sure of itself (a wrong direction at a crawl) -
            # open it up and let them pull it back
            self.P *= min(10.0, d2 / 9.0)
        self.update(z, R, H)

    def update(self, z, R, H, h=None):
        y = z - (H @ self.x if h is None else h)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P


class AirNavigator:
    def __init__(self, meters_per_pixel: float, *, drift: float = 0.03, fix_sigma_m: float = 15.0,
                 min_conf: float = 0.5, min_inliers: int = 15, align: float = math.inf):
        self.mpp = meters_per_pixel
        self.align = align           # 1/s the travel direction swings onto the nose (inf: always along it)
        self.drift, self.fix_sigma, self.min_conf, self.min_inliers = drift, fix_sigma_m, min_conf, min_inliers
        self.reset()

    def reset(self):
        self.pos: np.ndarray | None = None
        self.t: float | None = None
        self.heading: float | None = None
        self.speed_ms = 0.0          # along the nose (signed)
        self.side_ms = 0.0           # sideways, + = right (sideslip after a fast pedal turn)
        self._slip_heading: float | None = None
        self._last_u = 0.0           # along-nose part after the last update (the HUD overwrites speed_ms)
        self._v_meas: float | None = None
        self._track_dir_t = -1e9       # when the fixes last showed forwards/backwards
        self.last_speed_t = -1e9
        self.sigma_m = 0.0
        self.travelled_m = 0.0
        self.headings: deque = deque()
        self.yaw_rate = 0.0
        self.outliers: list = []
        self.source = 'none'
        self.fixes_used = 0
        # learned instrument corrections: true speed = scale * HUD speed, true course = compass + bias
        self.speed_scale = 1.0
        self.heading_bias = 0.0
        self.direction = 1.0         # +1 forwards / -1 backwards (the HUD speed has no sign)
        self.anchor = None           # (fix position, dead-reckoned displacement since then)
        self.last_compass_t = -1e9
        self.track: deque = deque()  # recent accepted fixes (t, position) for the course over ground
        self.heading_source = 'none' # 'compass' | 'track' | 'turn rate'
        self._odometry = np.zeros(2)
        self._motion: deque = deque()  # timestamped displacement, independent of fix corrections
        # motion filters: one per model of the slide; the minimap fixes pick the one that
        # predicts them best (the configured slide_align is 'inf' unless fit-slip was run, and
        # trusting it in a sliding game threw a single filter hundreds of metres off)
        self.filters = [MotionFilter(math.inf), MotionFilter(SLIDE)]
        self.motion = self.filters[0]
        self.align = self.align                 # sets the two models
        self.last_fix_t = -1e9
        self.fix_noise = self.fix_sigma   # m, learned from the scatter of the fixes
        self._thrust_acc = 3.0

    @property
    def align(self) -> float:
        return self._align

    @align.setter
    def align(self, value: float):
        """1/s the travel direction swings onto the nose (inf: always along it).  The motion
        filters run this model (first: it wins ties) and the other one - the slide the real
        flights measured, or arcade - and the fixes pick whichever predicts them better."""
        self._align = value
        if getattr(self, 'filters', None):
            self.filters[0].align = value
            self.filters[1].align = SLIDE if math.isinf(value) else math.inf

    @property
    def initialized(self) -> bool:
        return self.pos is not None

    def start(self, position, now: float, sigma_m: float = 10.0, source: str = 'fix'):
        self.pos = np.array(position, dtype=float)
        self.t, self.sigma_m, self.source = now, sigma_m, source
        self.outliers.clear()
        self.anchor = None
        self._odometry = np.zeros(2)
        self._motion = deque([(now, self._odometry.copy())])
        for f in self.filters:
            f.reset(self.pos * self.mpp, sigma_m)

    def predict(self, now: float, speed_kmh: float | None, heading: float | None, pitch: float | None = None):
        """Advance on airspeed + compass (missing readings: keep the last ones briefly).

        The HUD speed is unsigned.  The direction can only change while passing through
        (almost) zero speed, and then it follows the pitch: nose up -> moving backwards."""
        if heading is None and self.heading is not None and now - self.last_compass_t < 1.5 and self.t:
            # compass box unreadable (over trees / bright sky): keep turning at the last rate
            self.heading = (self.heading + self.yaw_rate * max(0.0, now - self.t) * .8) % 360
            self.heading_source = 'turn rate'
        if heading is not None:
            self.last_compass_t = now
            self.heading_source = 'compass'
            if self.heading is not None:
                unwrapped = self.headings[-1][1] + angle_delta(self.heading, heading) if self.headings else heading
            else:
                unwrapped = heading
            self.heading = heading % 360
            self.headings.append((now, unwrapped))
            while self.headings and now - self.headings[0][0] > 0.8:
                self.headings.popleft()
            if len(self.headings) >= 3:
                ts, hs = np.array(self.headings).T
                if np.ptp(ts) > .25:
                    self.yaw_rate = float(np.clip(np.polyfit(ts - now, hs, 1)[0], -60, 60))
        elif now - self.last_compass_t > 0.8:
            # no fresh compass: the turn rate is unknown, not 'the last one'.  A rate frozen at
            # -60 deg/s (22:48, compass read 14% of the time) cancelled the yaw command for 30 s
            self.yaw_rate *= 0.5
            self.headings.clear()
        self._direction_from_fixes(now)
        if speed_kmh is not None:
            if speed_kmh < 6 and pitch is not None and abs(pitch) > .1 and now - self._track_dir_t > 2.0:
                self.direction = 1.0 if pitch > 0 else -1.0
            elif speed_kmh < 1:
                self.direction = 1.0
            self.speed_ms, self.last_speed_t = self.direction * speed_kmh / 3.6, now
            self._v_meas = speed_kmh / 3.6
        elif now - self.last_speed_t > 1.5:
            self.speed_ms = 0.0
        if self.pos is None or self.t is None:
            self.t = now
            return
        dt = max(0.0, min(1.0, now - self.t))
        self.t = now
        if self.heading is None:
            self._remember_motion(now)
            return
        self._slip(dt, pitch)
        step_m = math.hypot(self.speed_ms, self.side_ms) * self.speed_scale * dt
        r = math.radians(self.heading + self.heading_bias)
        move = (np.array([math.sin(r), -math.cos(r)]) * self.speed_ms
                + np.array([math.cos(r), math.sin(r)]) * self.side_ms) * self.speed_scale * dt / self.mpp
        if self.anchor is not None:
            self.anchor[1] += move          # instrument calibration: pure dead reckoning
        move = self._predict_motion(dt, r, pitch, speed_kmh / 3.6 if speed_kmh is not None else None)
        self.pos = self.pos + move
        self._odometry += move
        self._remember_motion(now)
        self.travelled_m += abs(step_m)
        self.sigma_m += self.drift * abs(step_m) + 0.3 * dt
        self.source = 'dead reckoning'

    def _slip(self, dt, pitch=None):
        """Split the HUD speed (the size of the motion) into along-nose and sideways parts.
        Turning the nose leaves the motion where it was; it then swings onto the nose
        at ``align`` per second (drag and rotor thrust)."""
        last, self._slip_heading = self._slip_heading, self.heading
        # with the slide model the integrated velocity carries its own direction (forwards,
        # backwards, sideways); only the arcade model needs the forwards/backwards flag
        if math.isinf(self.align) or last is None:
            self.side_ms, self._last_u, self._v_meas = 0.0, self.speed_ms, None
            return
        u, w = self._last_u, self.side_ms
        turn = math.radians(angle_delta(last, self.heading))
        u, w = u + w * turn, w - u * turn
        # rotor thrust acts along the nose (HUD pitch, + = nose down), drag against the motion
        v = math.hypot(u, w)
        thrust = 9.81 * math.tan(math.radians(10 * pitch)) if pitch is not None else 0.0
        u += (thrust - DRAG * v * u) * dt
        w -= DRAG * v * w * dt
        if u > 0:
            v, slip = math.hypot(u, w), math.atan2(w, u) * math.exp(-self.align * dt)
            u, w = v * math.cos(slip), v * math.sin(slip)
        size = math.hypot(u, w)
        v_hud = self._v_meas if self._v_meas is not None else size   # the HUD shows the size of the motion
        self._v_meas = None
        if size > 1.0:
            u, w = u * v_hud / size, w * v_hud / size
        else:
            u, w = self.speed_ms, 0.0
        self.speed_ms, self.side_ms, self._last_u = u, w, u

    # ------------------------------------------------------------------ motion filter
    def _predict_motion(self, dt, r, pitch, hud_ms):
        """Advance the motion filters by ``dt`` and fold in the HUD speed.  Returns the
        predicted displacement in map px (the fixes then correct it)."""
        nose = np.array([math.sin(r), -math.cos(r)])
        self._thrust_acc = abs(9.81 * math.tan(math.radians(10 * pitch))) if pitch is not None else 3.0
        seed = None
        if hud_ms is not None and hud_ms * self.speed_scale > 2.0 and self.t - self.last_fix_t > 1.5:
            # dead reckoning only: start the motion from the instruments (with fixes coming
            # in, they and the thrust find the direction - one bad speed reading must not)
            seed = (nose * self.speed_ms + np.array([math.cos(r), math.sin(r)]) * self.side_ms) * self.speed_scale
        old = self.motion.x[:2].copy()
        for f in self.filters:
            f.predict(dt, nose, pitch, seed)
            if hud_ms is not None:
                f.update_speed(hud_ms * self.speed_scale)
        return (self.motion.x[:2] - old) / self.mpp

    @property
    def sliding(self) -> bool:
        """The fixes say the motion slides on (thrust and drag physics), not arcade."""
        return not math.isinf(self.motion.align)

    def velocity(self, now: float, max_age: float = 1.5):
        """Fused ground velocity (m/s, map x / y) while the minimap fixes keep it honest;
        otherwise the dead-reckoned estimate (slide model) or None."""
        if self.pos is not None and now - self.last_fix_t <= max_age:
            return self.motion.x[2:].copy()
        return self.velocity_estimate()

    def _remember_motion(self, now):
        self._motion.append((now, self._odometry.copy()))
        while len(self._motion) > 2 and (now - self._motion[1][0] > 3.0 or len(self._motion) > 400):
            self._motion.popleft()

    def _fix_at_present(self, z, timestamp):
        """Transport an asynchronous camera observation along the recorded flight path.

        Keep raw observation positions for course estimation; only the filter innovation
        is aligned to the present. Interpolation also handles turns and reverse flight.
        """
        if self.t is None or timestamp >= self.t:
            return z
        if not self._motion or timestamp < self._motion[0][0] or self.t - timestamp > 3.0:
            return None
        for (t0, p0), (t1, p1) in zip(self._motion, list(self._motion)[1:]):
            if t0 <= timestamp <= t1:
                fraction = (timestamp - t0) / max(t1 - t0, 1e-9)
                at_capture = p0 + fraction * (p1 - p0)
                return z + self._odometry - at_capture
        return None

    def correct(self, fix) -> bool:
        """Blend in a minimap fix if it is good and plausible.  Returns True if used."""
        if fix is None or not fix.valid or fix.confidence < self.min_conf or fix.inliers < self.min_inliers:
            return False
        z = np.array(fix.position, dtype=float)
        if not np.isfinite([*z, fix.timestamp]).all():
            return False
        if self.pos is None:
            self.start(z, fix.timestamp)
            return True
        raw_z = z.copy()
        z = self._fix_at_present(z, fix.timestamp)
        if z is None:
            return False
        gap_m = float(np.linalg.norm(z - self.pos)) * self.mpp
        if gap_m > max(150.0, 3 * self.sigma_m):
            # far away: a wrong match, unless several agree with each other (estimate is off)
            self.outliers = [o for o in self.outliers if distance(o, z) * self.mpp < 50][-2:] + [tuple(z)]
            if len(self.outliers) >= 3:
                self.start(z, self.t or fix.timestamp, source='fix (reset)')
                return True
            return False
        self.outliers.clear()
        self._calibrate(z)
        self._course(raw_z, fix.timestamp)
        k = self.sigma_m ** 2 / (self.sigma_m ** 2 + self.fix_sigma ** 2)
        fit = self._fix_fit(fix.timestamp, 1.0)
        if fit is not None:
            # how noisy the fixes really are: their scatter about a 1 s line (15 m assumed
            # until then; with ~5 m fixes that let a 19 m error stand for seconds)
            self.fix_noise += .05 * (float(np.clip(fit[2], 2.0, self.fix_sigma)) - self.fix_noise)
        gap = fix.timestamp - self.last_fix_t > 2.0
        track = self._fix_fit(fix.timestamp, 2.0)
        for f in self.filters:
            f.update_fix(z * self.mpp, 1.3 * self.fix_noise, self.sigma_m if gap else None)
            if track is not None:
                # the track's own velocity: an average over the last 2 s, ~1 s old - allow for
                # what the thrust changed meanwhile
                v, se, _ = track
                R = np.eye(2) * (2 * se * se + .5 ** 2 + self._thrust_acc ** 2)
                f.update(v, R, np.hstack([np.zeros((2, 2)), np.eye(2)]))
        best = max(self.filters, key=lambda f: f.score)
        if best is not self.motion and best.score > self.motion.score + 2.0:
            self.motion = best                    # clearly the better model of the motion
        self.pos = self.motion.x[:2] / self.mpp
        self.sigma_m = math.sqrt(1 - k) * self.sigma_m
        self.last_fix_t = max(self.last_fix_t, fix.timestamp)
        self.source = 'fix'
        self.fixes_used += 1
        return True

    def ground_velocity(self, now: float, window: float = 1.5):
        """Velocity over ground (m/s, map x/y) from the recent minimap fixes; None if too few
        or stale.  The game has no wind, so this is the airspeed too - and unlike the HUD
        digits it cannot lose a '1' (real flight 10:56: 167 km/h read as 68, 102 as 2)."""
        fit = self._fix_fit(now, window)
        return None if fit is None else fit[0]

    def _fix_fit(self, now, window):
        """Line fit through the recent fixes -> (velocity m/s, its standard error m/s, fix
        scatter about the line m) or None."""
        pts = [(t, z) for t, z in self.track if now - t <= window]
        if len(pts) < 3 or pts[-1][0] - pts[0][0] < .5 or now - pts[-1][0] > 1.0:
            return None
        ts = np.array([p[0] for p in pts]) - now
        zs = np.array([p[1] for p in pts])
        A = np.vstack([ts, np.ones_like(ts)]).T
        coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
        v = coef[0] * self.mpp
        resid = (zs - A @ coef) * self.mpp
        dof = max(1, len(pts) - 2)
        spread = float(np.sum((ts - ts.mean()) ** 2))
        scatter = math.sqrt(float(np.sum(resid ** 2)) / (2 * dof))
        return v, scatter / math.sqrt(max(spread, 1e-6)), scatter

    def velocity_estimate(self):
        """Dead-reckoned velocity (m/s, map x / y) from the slide model, or None."""
        if self.heading is None or math.isinf(self.align):
            return None
        r = math.radians(self.heading + self.heading_bias)
        return (np.array([math.sin(r), -math.cos(r)]) * self.speed_ms
                + np.array([math.cos(r), math.sin(r)]) * self.side_ms) * self.speed_scale

    def ground_speed(self, now: float, window: float = 1.5) -> float | None:
        v = self.ground_velocity(now, window)
        return None if v is None else float(np.hypot(*v) * 3.6)

    def ground_speed_fit(self, now: float, window: float = 2.0):
        """Ground speed from the fixes and its standard error (both km/h), or None."""
        fit = self._fix_fit(now, window)
        return None if fit is None else (float(np.hypot(*fit[0]) * 3.6), fit[1] * 3.6)

    def _direction_from_fixes(self, now):
        """Forwards or backwards from the fix velocity along the nose (the HUD speed has no
        sign; guessing from the pitch flipped to 'backwards' while braking and ran away)."""
        fit = self._fix_fit(now, 1.5)
        if fit is None or self.heading is None:
            return
        v, se, _ = fit
        r = math.radians(self.heading + self.heading_bias)
        along = float(v[0] * math.sin(r) - v[1] * math.cos(r))
        # only a clear reading: noisy fixes (10 m) put +-5 m/s on a 1.5 s fit, and a wrong
        # 'backwards' at a crawl flies the dead reckoning the wrong way
        if abs(along) > max(0.7, 3 * se):
            self.direction, self._track_dir_t = (1.0 if along > 0 else -1.0), now

    def _course(self, z, t):
        """Heading from the track between minimap fixes: reliable whenever we are moving,
        and the fallback when the compass box cannot be read (it fails over trees)."""
        self.track.append((t, z.copy()))
        while self.track and t - self.track[0][0] > 3.0:
            self.track.popleft()
        if abs(self.speed_ms) < 8.0 or self.direction < 0:
            return
        for t0, z0 in self.track:
            if t - t0 < 1.0:
                break
            d = (z - z0) * self.mpp
            if np.linalg.norm(d) < 25:
                continue
            course = math.degrees(math.atan2(d[0], -d[1])) % 360
            slip = math.degrees(math.atan2(self.side_ms, self.speed_ms)) if self.speed_ms > 1 else 0.0
            if t - self.last_compass_t > .5:
                self.heading = (course - slip - self.heading_bias) % 360
                self.heading_source = 'track'
            elif not math.isinf(self.align) and self.speed_ms > 1:
                # the track shows where we really go: pull the estimated slip towards it
                err = math.radians(angle_delta((self.heading + self.heading_bias + slip) % 360, course))
                v, new = math.hypot(self.speed_ms, self.side_ms), math.radians(slip) + .5 * err
                self.speed_ms, self.side_ms = v * math.cos(new), v * math.sin(new)
                self._last_u = self.speed_ms
            return

    def _calibrate(self, z):
        """Compare the true displacement between two good fixes with the dead-reckoned one
        and learn the airspeed scale and compass bias (slowly, within sane limits)."""
        if self.anchor is None:
            self.anchor = [z.copy(), np.zeros(2)]
            return
        true_d = (z - self.anchor[0]) * self.mpp
        dr_d = self.anchor[1] * self.mpp
        if np.linalg.norm(dr_d) < 300:
            return
        if np.linalg.norm(true_d) > 150:
            ratio = float(np.linalg.norm(true_d) / np.linalg.norm(dr_d))
            turn = angle_delta(math.degrees(math.atan2(dr_d[0], -dr_d[1])),
                               math.degrees(math.atan2(true_d[0], -true_d[1])))
            if .7 < ratio < 1.3 and abs(turn) < 15:
                self.speed_scale = float(np.clip(self.speed_scale * (1 + .5 * (ratio - 1)), .8, 1.2))
                self.heading_bias = float(np.clip(self.heading_bias + .5 * turn, -8, 8))
        self.anchor = [z.copy(), np.zeros(2)]

    def state(self) -> VehicleState | None:
        if self.pos is None or self.heading is None:
            return None
        slip = math.degrees(math.atan2(self.side_ms, self.speed_ms)) if self.speed_ms > 1 else 0.0
        return VehicleState(float(self.pos[0]), float(self.pos[1]), float(self.heading), self.speed_ms,
                            self.yaw_rate, float((self.heading + slip) % 360), float(self.t), self.source)
