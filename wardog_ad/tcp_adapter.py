"""TCP action semantics and a forward-only adapter for the truck PWM axes.

Pure functions only: this module never captures the screen or sends keys.
TCP predicts Beta concentration parameters for [acceleration, steering], not
literal throttle/brake channels. Negative acceleration means braking, not reverse.
"""
from dataclasses import dataclass
import math

import numpy as np


def target_from_map(position, heading_deg, target, meters_per_pixel):
    """Map pixels (x east, y south) -> TCP (right metres, negative forward metres)."""
    if not all(math.isfinite(v) for v in (*position, heading_deg, *target, meters_per_pixel)) or meters_per_pixel <= 0:
        raise ValueError('target transform requires finite coordinates and positive map scale')
    delta = (np.asarray(target, dtype=float) - position) * meters_per_pixel
    theta = math.radians(heading_deg)
    forward = np.array([math.sin(theta), -math.cos(theta)])
    right = np.array([math.cos(theta), math.sin(theta)])
    return float(delta @ right), -float(delta @ forward)


def decode_beta(alpha, beta, method='mode'):
    """Match upstream TCP mode decoding; mean is diagnostic-only by default."""
    a, b = np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float)
    if a.shape != (2,) or b.shape != (2,):
        raise ValueError('expected two Beta parameters: acceleration, steering')
    if not np.all(np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)):
        raise ValueError('Beta concentrations must be positive and finite')
    if method not in ('mode', 'mean'):
        raise ValueError('decoder must be mode or mean')
    values, cases = [], []
    for x, y in zip(a, b):
        mean = x / (x + y)
        if x > 1 and y > 1:
            mode, case = (x - 1) / (x + y - 2), 'interior'
        elif x <= 1 < y:
            mode, case = 0., 'lower_boundary'
        elif y <= 1 < x:
            mode, case = 1., 'upper_boundary'
        else:
            mode, case = mean, 'mean_fallback'
        values.append(2 * (mode if method == 'mode' else mean) - 1)
        cases.append(case)
    acceleration, steer = values
    return dict(steer=steer, throttle=max(0., acceleration), brake=max(0., -acceleration),
                acceleration=acceleration, alpha=a.tolist(), beta=b.tolist(),
                decoder=method, beta_cases=cases)


@dataclass(frozen=True)
class MappingLimits:
    max_throttle: float = .35
    max_steer: float = .5
    stop_speed_ms: float = .7
    brake_deadband: float = .05
    prediction_timeout_s: float = .8
    speed_timeout_s: float = .25


class TruckOutputMapper:
    """Brake-priority mapping to existing steer and signed-throttle PWM axes.

    Only apply S as a service-brake request while *measured forward speed* is
    above the stop band. At rest, braking maps to neutral, avoiding accidental
    reverse. This is not a parking brake/grade hold and is not a reverse policy.
    Every fast actuation update must call map() with a fresh signed speed.
    W/S semantics and steering gain still require game-specific calibration.
    """
    def __init__(self, limits=MappingLimits()):
        self.t = limits
        values = list(vars(limits).values())
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError('mapping limits must be finite and nonnegative')
        if not (0 <= limits.max_throttle <= 1 and 0 <= limits.max_steer <= 1
                and 0 <= limits.brake_deadband <= 1):
            raise ValueError('axis limits must be in [0, 1]')

    def map(self, control, *, speed_ms, now, prediction_at, speed_at, enabled=False):
        neutral = {'steer': 0., 'throttle': 0.}
        def result(mode, axes=neutral):
            return dict(mode=mode, axes=dict(axes))
        if not enabled:
            return result('disabled')
        if speed_ms is None or not all(math.isfinite(v) for v in (speed_ms, now, prediction_at, speed_at)):
            return result('invalid_state')
        if not 0 <= now - prediction_at <= self.t.prediction_timeout_s:
            return result('stale_prediction')
        if not 0 <= now - speed_at <= self.t.speed_timeout_s:
            return result('stale_speed')
        values = [control.get(k, 0.) for k in ('steer', 'throttle', 'brake')]
        if not all(math.isfinite(v) for v in values):
            return result('invalid_control')
        steer, throttle, brake = values
        steer = float(np.clip(steer, -self.t.max_steer, self.t.max_steer))
        throttle, brake = float(np.clip(throttle, 0, self.t.max_throttle)), float(np.clip(brake, 0, 1))
        if speed_ms < -self.t.stop_speed_ms:
            return result('reverse_motion_requires_takeover')
        if brake > self.t.brake_deadband:
            if speed_ms <= self.t.stop_speed_ms:
                return result('stopped_brake_request')
            return result('brake', {'steer': 0., 'throttle': -brake})
        return result('drive', {'steer': steer, 'throttle': throttle})
