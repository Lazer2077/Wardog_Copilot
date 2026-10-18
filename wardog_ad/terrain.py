"""Terrain memory: ground heights recorded on every flight, used to look ahead.

The HUD only measures the ground straight below (ALT AGL), which at 300 km/h gives
almost no warning before a mountainside.  Every helicopter tick records the ground
height under the aircraft (ASL - AGL) into a coarse grid per map; the controller then
looks along its route for the highest ground already known and climbs early.  The
return leg of a trip therefore always knows the terrain of the outbound leg, and
routes flown repeatedly become safer every time.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


class TerrainMemory:
    def __init__(self, path: str | Path | None = None, cell_px: float = 8.0):
        self.path = Path(path) if path else None
        self.cell = cell_px
        self.heights: dict[tuple[int, int], float] = {}
        self.dirty = 0
        if self.path and self.path.exists():
            with np.load(self.path) as data:
                for (i, j), h in zip(data['cells'], data['heights']):
                    self.heights[(int(i), int(j))] = float(h)

    def _key(self, x, y):
        return int(math.floor(x / self.cell)), int(math.floor(y / self.cell))

    def record(self, pos, ground_asl: float):
        """Remember the highest ground seen in this cell (trees/roofs count: they are obstacles)."""
        k = self._key(*pos)
        if ground_asl > self.heights.get(k, -1e9) + 0.5:
            self.heights[k] = float(ground_asl)
            self.dirty += 1

    def highest_ahead(self, pos, target, look_px: float, corridor_px: float = 12.0):
        """(height m ASL, distance px) of the highest known ground in a corridor towards target."""
        dx, dy = target[0] - pos[0], target[1] - pos[1]
        length = math.hypot(dx, dy)
        if length < 1e-6 or not self.heights:
            return None
        ux, uy = dx / length, dy / length
        best = None
        steps = int(min(look_px, length) / (self.cell / 2)) + 1
        for n in range(1, steps + 1):
            along = n * self.cell / 2
            for side in (-corridor_px, 0.0, corridor_px):
                x = pos[0] + ux * along - uy * side
                y = pos[1] + uy * along + ux * side
                h = self.heights.get(self._key(x, y))
                if h is not None and (best is None or h > best[0]):
                    best = (h, along)
        return best

    def known_ahead(self, pos, target, look_px: float, below: float) -> bool:
        """True if most of the way ahead (``look_px``) is already known and none of it rises
        above ``below`` m ASL: then a fast, low climb-out is safe."""
        dx, dy = target[0] - pos[0], target[1] - pos[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return False
        ux, uy = dx / length, dy / length
        steps = max(1, int(min(look_px, length) / (self.cell / 2)))
        known = 0
        for n in range(1, steps + 1):
            seen = False
            for side in range(-5, 6):               # ~ +-215 m corridor: outbound and return tracks differ
                x = pos[0] + ux * n * self.cell / 2 - uy * side * self.cell
                y = pos[1] + uy * n * self.cell / 2 + ux * side * self.cell
                h = self.heights.get(self._key(x, y))
                if h is not None:
                    if h > below:
                        return False
                    seen = True
            known += seen
        return known >= .7 * steps

    def roughness(self, x, y) -> float | None:
        """Largest height step (m) from the cell at (x, y) to its known neighbours: a slope,
        a forest edge or a ledge.  None if the cell or too many neighbours are unknown."""
        i, j = self._key(x, y)
        h = self.heights.get((i, j))
        if h is None:
            return None
        steps = [abs(h - self.heights[(i + di, j + dj)]) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                 if (di or dj) and (i + di, j + dj) in self.heights]
        return max(steps) if len(steps) >= 3 else None

    def flattest_near(self, pos, radius_px: float, limit_m: float):
        """Centre (px) of the known cell within ``radius_px`` that is flat enough (roughness
        <= ``limit_m``), preferring flat over near; None if there is none."""
        best = None
        r = int(math.ceil(radius_px / self.cell))
        ci, cj = self._key(*pos)
        for i in range(ci - r, ci + r + 1):
            for j in range(cj - r, cj + r + 1):
                x, y = (i + .5) * self.cell, (j + .5) * self.cell
                d = math.hypot(x - pos[0], y - pos[1])
                if d > radius_px:
                    continue
                rough = self.roughness(x, y)
                if rough is None or rough > limit_m:
                    continue
                score = rough + 2.0 * d / max(radius_px, 1.0)
                if best is None or score < best[0]:
                    best = (score, (x, y))
        return None if best is None else best[1]

    def save(self):
        if self.path and self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            cells = np.array(list(self.heights.keys()), dtype=np.int32).reshape(-1, 2)
            np.savez_compressed(self.path, cells=cells, heights=np.array(list(self.heights.values()), np.float32))
            self.dirty = 0
