"""Sub-pixel refinement of a minimap fix.

SIFT + RANSAC places the minimap to about one map pixel (~5 m).  Rendering the base map
the way the minimap should look at that fix and phase-correlating the two images
measures the remaining offset to a fraction of a minimap pixel.  HUD symbols are masked
and both images are high-pass filtered (the game minimap and the base map differ in
brightness, not in structure).  The correction is only applied when the correlation
peak is clear and the shift small, so it can only make a good fix better.
"""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from .localization import feature_mask


def _prepare(gray: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    g = gray.astype(np.float32)
    if mask is not None:
        g[mask == 0] = float(np.mean(g[mask > 0])) if (mask > 0).any() else float(g.mean())
    g = g - cv2.GaussianBlur(g, (0, 0), 3.0)          # keep structure, drop brightness/contrast
    return g / (g.std() + 1e-6)


class FixRefiner:
    def __init__(self, map_image: np.ndarray, *, min_response: float = 0.08, max_shift_px: float = 4.0):
        self.map_gray = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY) if map_image.ndim == 3 else map_image
        self.min_response, self.max_shift = min_response, max_shift_px
        self._window = None

    def render(self, x, y, scale, size, anchor, rotation=0.0):
        """The base map as the minimap would show it with the player at map pixel (x, y)."""
        w, h = size
        ax, ay = anchor[0] * w, anchor[1] * h
        angle = np.deg2rad(rotation)
        inv = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]]) / scale
        a = np.column_stack([inv, np.array([ax, ay]) - inv @ [x, y]]).astype(np.float32)
        return cv2.warpAffine(self.map_gray, a, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def refine(self, image: np.ndarray, fix, anchor=(0.5, 0.5)):
        """Return a refined copy of ``fix`` (or the fix unchanged if refinement is not sure)."""
        if not fix.valid or abs(fix.rotation) > 3 or fix.scale <= 0:
            return fix
        h, w = image.shape[:2]
        mask = feature_mask(image)
        mini = _prepare(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)
        rend = _prepare(self.render(fix.x, fix.y, fix.scale, (w, h), anchor, fix.rotation), mask)
        if self._window is None or self._window.shape != (h, w):
            self._window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        # phaseCorrelate multiplies its inputs by the window in place in OpenCV.
        (dx, dy), response = cv2.phaseCorrelate(rend.copy(), mini.copy(), self._window)
        if not np.isfinite([dx, dy, response]).all() or response < self.min_response or np.hypot(dx, dy) > self.max_shift:
            return fix
        # the minimap content is shifted by (dx, dy) relative to the rendering at the fix,
        # i.e. the player sits (-dx, -dy) minimap pixels from where the fix put him
        angle = np.deg2rad(fix.rotation)
        delta = fix.scale * np.array([[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle), np.cos(angle)]]) @ [dx, dy]
        x, y = fix.x - delta[0], fix.y - delta[1]
        candidate = _prepare(self.render(x, y, fix.scale, (w, h), anchor, fix.rotation), mask)
        keep = mask > 0
        def score(rendered):
            a, b = rendered[keep], mini[keep]
            # Elementwise reductions avoid starting a large BLAS thread pool for
            # every small minimap in the real-time control loop.
            energy = np.sqrt(np.sum(a * a, dtype=np.float64) * np.sum(b * b, dtype=np.float64))
            return float(np.sum(a * b, dtype=np.float64) / max(energy, 1e-6))
        # A peak can come from a repeated road or a HUD edge. Require the corrected
        # rendering to actually agree better with visible terrain before accepting it.
        if score(candidate) <= score(rend) + .002:
            return fix
        return replace(fix, x=float(x), y=float(y),
                       reason=f'ok (refined {np.hypot(dx, dy) * fix.scale:.2f} px)')
