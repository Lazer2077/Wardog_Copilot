"""Minimap -> map-pixel localization and player-arrow heading.

The matching approach (SIFT against pre-computed map features, ratio test,
RANSAC partial affine, local re-tracking around the last fix, convexity-defect
arrow tip) follows greatredo/wardogs-navigator ``vision.py`` (Apache-2.0).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np


@dataclass
class Fix:
    x: float = 0.0
    y: float = 0.0
    heading: float | None = None   # compass degrees from the arrow, None if not found
    confidence: float = 0.0
    inliers: int = 0
    error: float = 0.0
    scale: float = 0.0             # map px per minimap px
    rotation: float = 0.0
    valid: bool = False
    reason: str = 'no fix'
    timestamp: float = 0.0

    @property
    def position(self):
        return (self.x, self.y)


def feature_mask(image: np.ndarray, minimap: bool = True) -> np.ndarray:
    """Exclude borders, coloured HUD symbols and bright lettering/the player marker."""
    h, w = image.shape[:2]
    mask = np.full((h, w), 255, np.uint8)
    bottom = max(5, int(h * .08)) if minimap else 5
    mask[:5] = 0
    mask[-bottom:] = 0
    mask[:, :5] = 0
    mask[:, -5:] = 0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    overlay = ((hsv[:, :, 1] > 80) | (hsv[:, :, 2] > 215)).astype(np.uint8)
    mask[cv2.dilate(overlay, np.ones((7, 7), np.uint8)) > 0] = 0
    return mask


def player_heading(image: np.ndarray, anchor=(0.5, 0.5)) -> float | None:
    """Compass heading of the white chevron near ``anchor`` (fractions of the crop)."""
    h, w = image.shape[:2]
    centre = np.array([anchor[0] * w, anchor[1] * h])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    binary = ((hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 210)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if 12 < cv2.contourArea(c) < w * h * .012]
    if not candidates:
        return None

    def centroid(c):
        m = cv2.moments(c)
        return np.array([m['m10'] / max(m['m00'], 1), m['m01'] / max(m['m00'], 1)])

    contour = min(candidates, key=lambda c: np.linalg.norm(centroid(c) - centre))
    origin = centroid(contour)
    if np.linalg.norm(origin - centre) > min(h, w) * .12:
        return None
    try:
        polygon = cv2.approxPolyDP(contour, .025 * cv2.arcLength(contour, True), True)
        if len(polygon) < 4:
            return None
        hull = cv2.convexHull(polygon, returnPoints=False)
        if hull is None or len(hull) < 3 or len(hull) >= len(polygon):
            return None
        defects = cv2.convexityDefects(polygon, hull)
    except cv2.error:
        return None
    if defects is None:
        return None
    defect = max(defects.reshape(-1, 4), key=lambda d: d[3])
    notch = polygon[defect[2], 0].astype(float)
    tip = max(polygon[:, 0], key=lambda p: np.linalg.norm(p - notch))
    return math.degrees(math.atan2(tip[0] - origin[0], origin[1] - tip[1])) % 360


class MinimapLocalizer:
    def __init__(self, points: np.ndarray, descriptors: np.ndarray, map_shape, *, threads: int = 2):
        cv2.setNumThreads(threads)
        self.points = np.float32(points)
        self.descriptors = np.float32(descriptors)
        self.map_shape = map_shape  # (h, w)
        self.sift = cv2.SIFT_create(nfeatures=24000, contrastThreshold=.008, edgeThreshold=14)
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=96))
        self.matcher.add([self.descriptors])
        self.matcher.train()
        self.last: Fix | None = None
        self._local = None   # (centre, radius, matcher, reference points)

    def _local_index(self, last: Fix, radius: float):
        """FLANN index over map features near the last fix, rebuilt only after moving."""
        cached = self._local
        if cached is not None:
            centre, r, matcher, reference = cached
            if np.hypot(last.x - centre[0], last.y - centre[1]) < r / 3 and r >= radius:
                return matcher, reference
        near = np.where(np.sum((self.points - [last.x, last.y]) ** 2, axis=1) < (radius * 1.4) ** 2)[0]
        if len(near) < 20:
            self._local = None
            return None
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=3), dict(checks=64))
        matcher.add([self.descriptors[near]])
        matcher.train()
        self._local = ((last.x, last.y), radius * 1.4, matcher, self.points[near])
        return matcher, self.points[near]

    @classmethod
    def from_assets(cls, assets) -> 'MinimapLocalizer':
        points, descriptors = assets.load_features()
        return cls(points, descriptors, (assets.height, assets.width))

    def reset(self):
        self.last = None
        self._local = None

    def set_search_prior(self, position):
        """A human-provided search area, never a valid image-derived fix."""
        self.reset()
        self.last = Fix(x=float(position[0]), y=float(position[1]), scale=1.,
                        reason='manual search prior', valid=False)

    def locate(self, image: np.ndarray, anchor=(0.5, 0.5), north_up: bool = True) -> Fix:
        now = time.monotonic()
        if image is None or min(image.shape[:2]) < 60:
            return Fix(reason='capture region too small', timestamp=now)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, desc = self.sift.detectAndCompute(gray, feature_mask(image))
        if desc is None or len(desc) < 8:
            return Fix(reason='not enough terrain features', timestamp=now)
        heading = player_heading(image, anchor)
        if self.last is not None:
            local = self._local_index(self.last, max(65.0, max(image.shape[:2]) * self.last.scale * 1.5))
            if local is not None:
                matcher, reference = local
                fix = self._match(image, keypoints, desc, matcher, reference, anchor, north_up, heading, now)
                if fix.valid:
                    self.last = fix
                    return fix
        fix = self._match(image, keypoints, desc, self.matcher, self.points, anchor, north_up, heading, now)
        self.last = fix if fix.valid else None
        return fix

    def _match(self, image, keypoints, desc, matcher, reference, anchor, north_up, heading, now) -> Fix:
        pairs = matcher.knnMatch(desc, k=2)
        good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < .76 * n.distance]
        if len(good) < 8:
            return Fix(reason=f'too few matches ({len(good)})', timestamp=now)
        src = np.float32([keypoints[m.queryIdx].pt for m in good])
        dst = np.float32([reference[m.trainIdx] for m in good])
        mat, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5,
                                                maxIters=5000, confidence=.995)
        if mat is None:
            return Fix(reason='no consistent transform', timestamp=now)
        inlier = mask.ravel().astype(bool)
        count = int(inlier.sum())
        scale = float(np.linalg.norm(mat[:, 0]))
        rotation = math.degrees(math.atan2(mat[1, 0], mat[0, 0]))
        residual = np.linalg.norm(src @ mat[:, :2].T + mat[:, 2] - dst, axis=1)
        error = float(np.median(residual[inlier])) if count else 99.0
        h, w = image.shape[:2]
        x, y = mat @ [anchor[0] * w, anchor[1] * h, 1]
        spread = float(cv2.contourArea(cv2.convexHull(src[inlier]))) / (w * h) if count >= 3 else 0.0
        confidence = min(1., count / 24) * min(1., (count / len(good)) / .55) * min(1., spread / .1)
        ok = (count >= 8 and confidence >= .35 and error < min(2, 2.5 * scale) and .025 < scale < 6
              and 0 <= x < self.map_shape[1] and 0 <= y < self.map_shape[0])
        if north_up and abs(rotation) > 6:
            ok = False
        if heading is not None:
            heading = (heading + rotation) % 360
        return Fix(float(x), float(y), heading, confidence, count, error, scale, rotation, bool(ok),
                   'ok' if ok else 'unreliable match', now)
