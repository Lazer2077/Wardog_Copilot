"""Screen capture and interactive region calibration."""
from __future__ import annotations

import numpy as np

from .config import Region


class ScreenCapture:
    def __init__(self):
        import mss
        self._mss = mss.MSS() if hasattr(mss, 'MSS') else mss.mss()

    def grab(self, region: Region) -> np.ndarray:
        return np.asarray(self._mss.grab(region.as_mss()))[:, :, :3].copy()

    def monitors(self) -> list[dict]:
        return self._mss.monitors[1:]

    def grab_monitor(self, index: int = 0) -> tuple[np.ndarray, dict]:
        monitor = self.monitors()[index]
        return np.asarray(self._mss.grab(monitor))[:, :, :3].copy(), monitor


def select_region(screen: np.ndarray, monitor: dict, title: str) -> Region | None:
    """Let the user drag a rectangle on a (downscaled) screenshot."""
    import cv2
    h, w = screen.shape[:2]
    scale = min(1.0, 1600 / w, 900 / h)
    preview = cv2.resize(screen, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else screen
    x, y, rw, rh = cv2.selectROI(title, preview, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if rw < 5 or rh < 5:
        return None
    return Region(int(monitor['left'] + x / scale), int(monitor['top'] + y / scale), int(rw / scale), int(rh / scale))


def select_anchor(image: np.ndarray, anchor=(.5, .5)) -> tuple[float, float] | None:
    """Calibrate the vehicle position on a frozen minimap, independent of arrow OCR."""
    import cv2
    h, w = image.shape[:2]
    zoom = max(1., min(3., 800 / max(h, w)))
    preview = cv2.resize(image, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    point = None
    title = 'Click YOUR vehicle position; Enter saves; Esc cancels'
    def clicked(event, x, y, flags, userdata):
        nonlocal point
        if event == cv2.EVENT_LBUTTONDOWN:
            point = (float(np.clip(x / zoom / w, 0, 1)), float(np.clip(y / zoom / h, 0, 1)))
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, clicked)
    try:
        while True:
            view = preview.copy()
            at = point if point is not None else anchor
            centre = (round(at[0] * w * zoom), round(at[1] * h * zoom))
            cv2.drawMarker(view, centre, (0, 255, 0) if point is not None else (0, 180, 255),
                           cv2.MARKER_CROSS, 24, 1)
            cv2.imshow(title, view)
            key = cv2.waitKey(20) & 0xff
            if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key in (10, 13) and point is not None:
                return point
    finally:
        cv2.destroyWindow(title)
