"""2-D helpers in the map-pixel frame: x to the right, y downwards.

Bearings are compass degrees: 0 = north (up, -y), 90 = east (+x), clockwise.
"""
from __future__ import annotations

from bisect import bisect_right
import math

import numpy as np


def distance(a, b) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def bearing(a, b) -> float:
    return math.degrees(math.atan2(b[0] - a[0], a[1] - b[1])) % 360


def wrap180(angle: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def angle_delta(src: float, dst: float) -> float:
    """Signed shortest rotation from ``src`` to ``dst`` (positive = clockwise / right)."""
    return wrap180(dst - src)


def heading_vector(heading: float) -> np.ndarray:
    r = math.radians(heading)
    return np.array([math.sin(r), -math.cos(r)])


def project(p, a, b):
    """Closest point on segment ab: (distance, point, t)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / max(dx * dx + dy * dy, 1e-12)
    t = max(0.0, min(1.0, t))
    q = (a[0] + t * dx, a[1] + t * dy)
    return distance(p, q), q, t


def cumulative(points) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return np.zeros(len(pts))
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])


def point_at(points, lengths, s: float):
    s = max(0.0, min(float(lengths[-1]), s))
    i = min(len(points) - 2, bisect_right(lengths, s) - 1)
    f = (s - lengths[i]) / max(lengths[i + 1] - lengths[i], 1e-9)
    return (points[i][0] + f * (points[i + 1][0] - points[i][0]),
            points[i][1] + f * (points[i + 1][1] - points[i][1]))


def resample(points, spacing: float) -> np.ndarray:
    """Resample a polyline at a fixed arc-length spacing (endpoints kept)."""
    pts = np.asarray(points, dtype=float)
    lengths = cumulative(pts)
    total = float(lengths[-1])
    if total < 1e-9:
        return pts[:1].copy()
    count = max(2, int(math.ceil(total / spacing)) + 1)
    stations = np.linspace(0.0, total, count)
    return np.array([point_at(pts, lengths, s) for s in stations])


def smooth(points, window: int = 5, iterations: int = 2) -> np.ndarray:
    """Moving-average smoothing that pins both endpoints."""
    pts = np.asarray(points, dtype=float).copy()
    if len(pts) < 3 or window < 3:
        return pts
    half = window // 2
    for _ in range(iterations):
        padded = np.concatenate([np.repeat(pts[:1], half, 0), pts, np.repeat(pts[-1:], half, 0)])
        kernel = np.ones(window) / window
        out = np.column_stack([np.convolve(padded[:, k], kernel, mode='valid') for k in (0, 1)])
        out[0], out[-1] = pts[0], pts[-1]
        pts = out
    return pts


def curvature(points, step: int = 2) -> np.ndarray:
    """Unsigned curvature (1/px) at each vertex using a three-point circle fit."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    k = np.zeros(n)
    if n < 3:
        return k
    for i in range(n):
        a, b, c = pts[max(0, i - step)], pts[i], pts[min(n - 1, i + step)]
        ab, bc, ca = np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)
        if ab < 1e-9 or bc < 1e-9 or ca < 1e-9:
            continue
        cross = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        k[i] = 2.0 * cross / (ab * bc * ca)
    return k
