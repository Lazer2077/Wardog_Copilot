"""Open ground for landing, read from the base map.

The base map (about 5 m per pixel) shows forest as dark, grainy texture, water as very dark
ribbons, towns as hard edges and bright roofs, and meadows / fields as smooth mid-grey.  A
helicopter needs a clear disk of ~30 m, so every pixel gets a cost from the texture, darkness
and edge density over such a disk; the cost is then spread over the disk (the whole clearing
must be open).  Low cost = open ground.  It cannot see small obstacles (a single tree, a
pole) - the terrain memory and the hold-on-uneven-ground checks still apply on the way down.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


class OpenGround:
    def __init__(self, map_image: np.ndarray, meters_per_pixel: float, clearing_m: float = 30.0):
        self.mpp = meters_per_pixel
        g = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY) if map_image.ndim == 3 else map_image
        g = g.astype(np.float32)
        r = max(2, int(round(clearing_m / 2 / meters_per_pixel)))
        k = 2 * r + 1
        mean = cv2.blur(g, (k, k))
        std = np.sqrt(np.maximum(cv2.blur(g * g, (k, k)) - mean * mean, 0))
        edge = cv2.blur(cv2.Canny(g.astype(np.uint8), 20, 50).astype(np.float32) / 255, (k, k))
        p3, p10, p50, p97 = np.percentile(g, [3, 10, 50, 97])
        dark = np.clip((p50 - mean) / max(p50 - p10, 1.0), 0, 1)           # forest, water
        rough = np.clip(std / max(float(np.percentile(std, 90)), 1e-6), 0, 1)  # tree crowns
        cost = .45 * rough + .35 * dark + .4 * np.clip(edge / .25, 0, 1)  # buildings, roads
        cost += np.where(cv2.blur(g, (3, 3)) < p3, 1.0, 0.0)               # water
        cost += np.where(mean > p97, .3, 0.0)                              # bright roofs / concrete
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.cost_map = cv2.dilate(cost, disk).astype(np.float32)

    def cost(self, x: float, y: float) -> float:
        h, w = self.cost_map.shape
        i, j = int(round(y)), int(round(x))
        if not (0 <= i < h and 0 <= j < w):
            return math.inf
        return float(self.cost_map[i, j])

    def best_near(self, pos, radius_m: float, per_100m: float = .3):
        """(spot, cost) of the most open ground within ``radius_m``, nearer preferred."""
        h, w = self.cost_map.shape
        R = max(1, int(radius_m / self.mpp))
        x, y = int(round(pos[0])), int(round(pos[1]))
        x0, x1, y0, y1 = max(0, x - R), min(w, x + R + 1), max(0, y - R), min(h, y + R + 1)
        if x0 >= x1 or y0 >= y1:
            return None, math.inf
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx - pos[0], yy - pos[1])
        win = self.cost_map[y0:y1, x0:x1]
        score = np.where(d <= R, win + per_100m * d * self.mpp / 100, np.inf)
        j, i = np.unravel_index(int(np.argmin(score)), score.shape)
        return (float(x0 + i), float(y0 + j)), float(win[j, i])
