"""Road-network routing (truck) and direct waypoint routing (aircraft).

Roads are polylines in map pixels with ``kind`` in {major, minor, offroad} and an
optional ``bridge`` flag.  Following wardogs-navigator's rules, crossing lines do
NOT form a junction; a junction exists only where a polyline vertex lies on
another road.  Bridge decks only connect to ground roads at their marked ends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math

import numpy as np

from .geometry import cumulative, curvature, distance, point_at, project, resample, smooth

KIND_COST = {'major': 1.0, 'minor': 1.18, 'offroad': 1.6}
JOIN_PX = 1.5


class RouteError(ValueError):
    pass


@dataclass
class Avoid:
    """Axis-aligned rectangle (map px) the route must not touch."""
    x0: float
    y0: float
    x1: float
    y1: float

    def hits(self, a, b) -> bool:
        lo, hi = 0.0, 1.0
        dx, dy = b[0] - a[0], b[1] - a[1]
        for p, q in ((-dx, a[0] - self.x0), (dx, self.x1 - a[0]), (-dy, a[1] - self.y0), (dy, self.y1 - a[1])):
            if abs(p) < 1e-12:
                if q < 0:
                    return False
            elif p < 0:
                lo = max(lo, q / p)
            else:
                hi = min(hi, q / p)
            if lo > hi:
                return False
        return True


def _bridge_ports(road) -> list:
    if not road.get('bridge'):
        return []
    return [road['points'][i] for key, i in (('bridge_start', 0), ('bridge_end', -1)) if road.get(key)]


class RoadGraph:
    def __init__(self, roads: list[dict]):
        self.roads = roads
        self.nodes: list[tuple[float, float]] = []
        self.levels: list[bool] = []
        self.edges: dict[int, list[tuple[int, float, int]]] = {}
        self._node_grid: dict = {}
        self._build()

    # -- construction -------------------------------------------------
    def _node(self, p, bridge_level: bool) -> int:
        gx, gy = math.floor(p[0] / JOIN_PX), math.floor(p[1] / JOIN_PX)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for i in self._node_grid.get((gx + dx, gy + dy, bridge_level), ()):
                    if distance(p, self.nodes[i]) < JOIN_PX:
                        return i
        self.nodes.append((float(p[0]), float(p[1])))
        self.levels.append(bridge_level)
        index = len(self.nodes) - 1
        self.edges[index] = []
        self._node_grid.setdefault((gx, gy, bridge_level), []).append(index)
        return index

    def _build(self):
        segments = []
        for ri, road in enumerate(self.roads):
            pts = road['points']
            for a, b in zip(pts, pts[1:]):
                if distance(a, b) >= .05:
                    segments.append((tuple(a), tuple(b), ri))
        ports = [p for road in self.roads for p in _bridge_ports(road)]
        splits = [[(0.0, a), (1.0, b)] for a, b, _ in segments]
        cell = 24.0
        grid: dict = {}
        for j, (a, b, _) in enumerate(segments):
            for gx in range(math.floor((min(a[0], b[0]) - JOIN_PX) / cell), math.floor((max(a[0], b[0]) + JOIN_PX) / cell) + 1):
                for gy in range(math.floor((min(a[1], b[1]) - JOIN_PX) / cell), math.floor((max(a[1], b[1]) + JOIN_PX) / cell) + 1):
                    grid.setdefault((gx, gy), []).append(j)

        def connects(r1, r2, p):
            b1, b2 = bool(self.roads[r1].get('bridge')), bool(self.roads[r2].get('bridge'))
            return b1 == b2 or any(distance(p, q) < JOIN_PX for q in ports)

        for i, (a, b, ri) in enumerate(segments):
            for p in (a, b):
                for j in grid.get((math.floor(p[0] / cell), math.floor(p[1] / cell)), ()):
                    if i == j:
                        continue
                    c, d, rj = segments[j]
                    if not connects(ri, rj, p):
                        continue
                    gap, _, t = project(p, c, d)
                    if gap < JOIN_PX:
                        splits[j].append((t, p))

        for (_, _, ri), division in zip(segments, splits):
            road = self.roads[ri]
            ids = []
            for _, p in sorted(division, key=lambda item: item[0]):
                level = bool(road.get('bridge')) and not any(distance(p, q) < JOIN_PX for q in ports)
                ids.append(self._node(p, level))
            for u, v in zip(ids, ids[1:]):
                if u != v:
                    w = distance(self.nodes[u], self.nodes[v])
                    self.edges[u].append((v, w, ri))
                    self.edges[v].append((u, w, ri))

    # -- queries ------------------------------------------------------
    def _usable(self, ri: int, kinds) -> bool:
        return self.roads[ri].get('kind', 'minor') in kinds

    def snap(self, p, kinds, avoid=()) -> tuple[float, tuple, int, int, float]:
        """Nearest usable edge: (gap, point, u, v, fraction)."""
        best = None
        for u, adj in self.edges.items():
            for v, _, ri in adj:
                if u > v or not self._usable(ri, kinds):
                    continue
                gap, q, t = project(p, self.nodes[u], self.nodes[v])
                if any(z.hits(q, q) for z in avoid):
                    continue
                if best is None or gap < best[0]:
                    best = (gap, q, u, v, t)
        if best is None:
            raise RouteError('no usable road for the selected road kinds')
        return best

    def route(self, anchors, kinds=('major', 'minor'), avoid=(), max_snap=45.0, preferences=None) -> list:
        """Shortest weighted road path through ``anchors`` (start, waypoints..., goal)."""
        if len(anchors) < 2:
            raise RouteError('need a start and a goal')
        factors = dict(KIND_COST)
        if preferences:
            scale = {'prefer': .5, 'normal': 1.0, 'avoid': 4.0}
            factors = {k: v * scale[preferences.get(k, 'normal')] for k, v in factors.items()}
        snaps = []
        for p in anchors:
            snap = self.snap(p, kinds, avoid)
            if snap[0] > max_snap:
                raise RouteError(f'point {tuple(round(c, 1) for c in p)} is {snap[0]:.0f} px from the nearest road')
            snaps.append(snap)
        path = [snaps[0][1]]
        for s0, s1 in zip(snaps, snaps[1:]):
            path.extend(self._leg(s0, s1, kinds, avoid, factors)[1:])
        return path

    def _leg(self, s0, s1, kinds, avoid, factors) -> list:
        start, goal = -1, -2
        _, q0, u0, v0, _ = s0
        _, q1, u1, v1, _ = s1
        pos = {start: q0, goal: q1}

        def cost_factor(u, v):
            for w, _, ri in self.edges[u]:
                if w == v:
                    return factors.get(self.roads[ri].get('kind', 'minor'), 1.0)
            return 1.0

        extra: dict[int, list] = {start: [], goal: []}
        for virtual, q, u, v in ((start, q0, u0, v0), (goal, q1, u1, v1)):
            f = cost_factor(u, v)
            for n in (u, v):
                w = distance(q, self.nodes[n]) * f
                extra[virtual].append((n, w))
                extra.setdefault(n, []).append((virtual, w))
        if (u0, v0) in ((u1, v1), (v1, u1)):
            direct = distance(q0, q1) * cost_factor(u0, v0)
            extra[start].append((goal, direct))

        def coord(n):
            return pos[n] if n < 0 else self.nodes[n]

        def neighbours(n):
            if n >= 0:
                for m, w, ri in self.edges[n]:
                    if self._usable(ri, kinds):
                        yield m, w * factors.get(self.roads[ri].get('kind', 'minor'), 1.0)
            yield from extra.get(n, ())

        floor = min(factors.values())
        queue = [(0.0, 0.0, start)]
        best = {start: 0.0}
        parent = {}
        while queue:
            _, g, n = heapq.heappop(queue)
            if n == goal:
                break
            if g > best.get(n, math.inf):
                continue
            for m, w in neighbours(n):
                if any(z.hits(coord(n), coord(m)) for z in avoid):
                    continue
                cost = g + w
                if cost < best.get(m, math.inf):
                    best[m] = cost
                    parent[m] = n
                    heapq.heappush(queue, (cost + distance(coord(m), q1) * floor, cost, m))
        if goal not in parent and start != goal:
            raise RouteError('road network is not connected between these points')
        chain = [goal]
        while chain[-1] != start:
            chain.append(parent[chain[-1]])
        return [coord(n) for n in reversed(chain)]


@dataclass
class Path:
    """Dense, smoothed path with arc-length lookup and progress tracking."""
    raw: list
    spacing: float = 1.0
    smoothing: int = 5
    points: np.ndarray = field(init=False)
    s: np.ndarray = field(init=False)
    kappa: np.ndarray = field(init=False)

    def __post_init__(self):
        dense = resample(self.raw, self.spacing)
        self.points = smooth(dense, self.smoothing) if self.smoothing >= 3 else dense
        self.s = cumulative(self.points)
        self.kappa = curvature(self.points, step=max(2, int(round(6 / self.spacing))))

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def at(self, s: float):
        return point_at(self.points, self.s, s)

    def nearest(self, p, s_min: float = 0.0, s_max: float = math.inf) -> tuple[float, float]:
        """(cross-track distance, arc length) of the closest point within [s_min, s_max]."""
        lo = int(np.searchsorted(self.s, max(0.0, s_min), side='right')) - 1
        hi = int(np.searchsorted(self.s, min(self.length, s_max), side='right'))
        lo, hi = max(0, lo), max(lo + 2, min(len(self.points), hi + 1))
        pts = self.points[lo:hi]
        a, b = pts[:-1], pts[1:]
        d = b - a
        denom = np.maximum((d ** 2).sum(1), 1e-12)
        t = np.clip(((np.asarray(p) - a) * d).sum(1) / denom, 0, 1)
        q = a + d * t[:, None]
        gaps = np.linalg.norm(q - np.asarray(p), axis=1)
        k = int(np.argmin(gaps))
        return float(gaps[k]), float(self.s[lo + k] + t[k] * np.linalg.norm(d[k]))

    def signed_offset(self, p, s: float) -> float:
        """Cross-track error, positive when the vehicle is right of the path."""
        a = np.asarray(self.at(s))
        b = np.asarray(self.at(min(self.length, s + 2.0)))
        if np.linalg.norm(b - a) < 1e-9:
            a = np.asarray(self.at(max(0.0, s - 2.0)))
            b = np.asarray(self.at(s))
        tangent = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        rel = np.asarray(p) - a
        # y points down, so tangent × offset > 0 means "to the right of travel".
        return float(tangent[0] * rel[1] - tangent[1] * rel[0])

    def max_curvature(self, s0: float, s1: float) -> float:
        lo = int(np.searchsorted(self.s, s0))
        hi = int(np.searchsorted(self.s, s1, side='right'))
        window = self.kappa[lo:max(lo + 1, hi)]
        return float(window.max()) if len(window) else 0.0


def straight_route(anchors) -> list:
    """Aircraft: fly directly between waypoints."""
    if len(anchors) < 2:
        raise RouteError('need a start and a goal')
    return [tuple(map(float, p)) for p in anchors]
