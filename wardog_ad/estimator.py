"""Fuse noisy minimap fixes into position, velocity, heading and yaw rate.

Position/velocity come from a least-squares line fit over a short sliding window
of fixes (no lag for constant velocity, much less noise than differencing).
Heading is the low-passed arrow heading, falling back to the course over ground.
Positions are map pixels; speeds are returned in m/s via ``meters_per_pixel``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .geometry import angle_delta, bearing


@dataclass
class VehicleState:
    x: float
    y: float
    heading: float           # compass degrees
    speed_ms: float          # signed ground speed (negative = reversing)
    yaw_rate: float          # deg/s, positive = turning right
    course: float | None     # direction of travel from motion (deg)
    timestamp: float
    heading_source: str      # 'arrow' | 'motion' | 'hold' | 'coast'

    @property
    def position(self):
        return (self.x, self.y)


def _fit(ts: np.ndarray, values: np.ndarray, at: float):
    """Value and slope at ``at`` of a least-squares line through (ts, values)."""
    if len(ts) < 2 or np.ptp(ts) < 1e-6:
        return values[-1], np.zeros_like(values[-1])
    t = ts - at
    a = np.column_stack([np.ones_like(t), t])
    coef, *_ = np.linalg.lstsq(a, values, rcond=None)
    return coef[0], coef[1]


class StateEstimator:
    def __init__(self, meters_per_pixel: float, *, window_s: float = 1.5, heading_window_s: float = 0.6,
                 heading_alpha: float = .5, max_jump_m: float = 80.0, motion_heading_min_ms: float = 2.0):
        self.mpp = meters_per_pixel
        self.window_s, self.heading_window_s = window_s, heading_window_s
        self.heading_alpha = heading_alpha
        self.max_jump_px = max_jump_m / meters_per_pixel
        self.motion_min_px = motion_heading_min_ms / meters_per_pixel
        self.reset()

    def reset(self):
        self.samples: deque = deque()          # (t, x, y)
        self.headings: deque = deque()         # (t, unwrapped heading)
        self.pos: np.ndarray | None = None
        self.vel = np.zeros(2)
        self.heading: float | None = None
        self.yaw_rate = 0.0
        self.t: float | None = None
        self.rejected = 0

    @property
    def initialized(self) -> bool:
        return self.pos is not None and self.heading is not None

    def update(self, fix, now: float | None = None) -> VehicleState | None:
        """Feed a valid fix. Returns the new state, or None while initializing."""
        now = fix.timestamp if now is None else now
        z = np.array([fix.x, fix.y], dtype=float)
        if self.pos is not None and self.t is not None:
            predicted = self.pos + self.vel * (now - self.t)
            # Reject a teleporting match (wrong terrain) but accept it after repeats so a
            # genuine relocation (respawn / long loss) recovers.
            if np.linalg.norm(z - predicted) > self.max_jump_px:
                self.rejected += 1
                if self.rejected < 3:
                    return None
                self.reset()
        self.rejected = 0
        self.samples.append((now, z[0], z[1]))
        while self.samples and now - self.samples[0][0] > self.window_s:
            self.samples.popleft()
        data = np.array(self.samples)
        pos, vel = _fit(data[:, 0], data[:, 1:], now)
        self.pos, self.vel, self.t = np.asarray(pos, float), np.asarray(vel, float), now

        speed_px = float(np.linalg.norm(self.vel))
        course = bearing((0, 0), self.vel) if speed_px > self.motion_min_px else None
        source, target = 'hold', None
        if fix.heading is not None:
            target, source = fix.heading, 'arrow'
        elif course is not None:
            target, source = course, 'motion'
            if self.heading is not None and abs(angle_delta(self.heading, course)) > 100:
                target = (course + 180) % 360   # reversing
        if target is not None:
            if self.heading is None:
                self.heading = target
            else:
                self.heading = (self.heading + self.heading_alpha * angle_delta(self.heading, target)) % 360
        if self.heading is not None:
            unwrapped = self.heading
            if self.headings:
                unwrapped = self.headings[-1][1] + angle_delta(self.headings[-1][1] % 360, self.heading)
            self.headings.append((now, unwrapped))
            while self.headings and now - self.headings[0][0] > self.heading_window_s:
                self.headings.popleft()
            h = np.array(self.headings)
            self.yaw_rate = float(_fit(h[:, 0], h[:, 1], now)[1])
        return self.state(source) if self.heading is not None else None

    def state(self, source='hold') -> VehicleState:
        speed_px = float(np.linalg.norm(self.vel))
        course = bearing((0, 0), self.vel) if speed_px > self.motion_min_px else None
        sign = 1.0
        if (course is not None and self.heading is not None and speed_px * self.mpp > 1.5
                and abs(angle_delta(self.heading, course)) > 100):
            sign = -1.0
        return VehicleState(float(self.pos[0]), float(self.pos[1]), float(self.heading or 0.0),
                            sign * speed_px * self.mpp, self.yaw_rate, course, float(self.t), source)

    def dead_reckon(self, now: float, speed_ms: float, heading: float | None) -> VehicleState | None:
        """Advance the position on airspeed + compass while the minimap cannot be matched."""
        if not self.initialized:
            return None
        if heading is not None:
            self.heading = heading
        dt = max(0.0, now - self.t)
        r = math.radians(self.heading)
        self.vel = np.array([math.sin(r), -math.cos(r)]) * speed_ms / self.mpp
        self.pos = self.pos + self.vel * dt
        self.t = now
        self.samples.clear()            # the next real fix restarts the line fit
        return self.state('dead reckoning')

    def coast(self, now: float) -> VehicleState | None:
        """Dead-reckon briefly while fixes are missing."""
        if not self.initialized:
            return None
        pos = self.pos + self.vel * (now - self.t)
        s = self.state()
        s.x, s.y, s.timestamp, s.heading_source = float(pos[0]), float(pos[1]), now, 'coast'
        return s
