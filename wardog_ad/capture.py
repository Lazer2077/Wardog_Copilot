"""Screen capture and interactive region calibration."""
from __future__ import annotations

import numpy as np
import threading
import time

from .config import Region

_layout_cache = threading.local()


def display_monitors():
    """Refresh display dimensions without relying on MSS's cached monitor list."""
    now = time.monotonic()
    if now - getattr(_layout_cache, 'updated', -1e9) >= 1.:
        import mss
        factory = mss.MSS if hasattr(mss, 'MSS') else mss.mss
        with factory() as cap:
            _layout_cache.monitors = cap.monitors[1:]
        _layout_cache.updated = now
    return _layout_cache.monitors


class CaptureRegionError(ValueError):
    """A saved physical-pixel region no longer fits the display layout."""


def resolve_region(region, monitors):
    """Scale a calibrated ROI with its monitor, preserving the stored calibration.

    Legacy regions have no reference and remain absolute. Ambiguous multi-monitor
    moves require recalibration rather than capturing an unrelated display.
    """
    if not region.reference:
        return region
    if len(region.reference) != 4:
        raise CaptureRegionError('invalid calibration monitor reference; recalibrate')
    left,top,width,height = region.reference
    if width <= 0 or height <= 0:
        raise CaptureRegionError('invalid calibration resolution; recalibrate')
    matches = [m for m in monitors if (m['left'],m['top']) == (left,top)]
    target = matches[0] if len(matches) == 1 else monitors[0] if len(monitors) == 1 else None
    if target is None:
        raise CaptureRegionError('calibration monitor moved or is ambiguous; recalibrate')
    sx,sy = target['width']/width,target['height']/height
    if abs(sx/sy - 1) > .02:
        raise CaptureRegionError('display aspect ratio changed; recalibrate HUD layout')
    x0 = target['left'] + round((region.left-left)*sx)
    y0 = target['top'] + round((region.top-top)*sy)
    x1 = target['left'] + round((region.left+region.width-left)*sx)
    y1 = target['top'] + round((region.top+region.height-top)*sy)
    return Region(x0,y0,x1-x0,y1-y0)


def region_on_screen(region, monitors):
    # Require a single monitor to contain the entire crop; never pad off-screen
    # areas with black and report that as a localization failure.
    return any(region.left >= m['left'] and region.top >= m['top']
               and region.left + region.width <= m['left'] + m['width']
               and region.top + region.height <= m['top'] + m['height'] for m in monitors)


class ScreenCapture:
    def __init__(self):
        import mss
        self._mss = mss.MSS() if hasattr(mss, 'MSS') else mss.mss()

    def grab(self, region: Region) -> np.ndarray:
        region = resolve_region(region, self.monitors())
        if not region_on_screen(region, self.monitors()):
            raise CaptureRegionError(f'capture region outside monitor: {region.as_mss()}; recalibrate after resolution changes')
        return np.asarray(self._mss.grab(region.as_mss()))[:, :, :3].copy()

    def monitors(self) -> list[dict]:
        return display_monitors()

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
    return Region(int(monitor['left'] + x / scale), int(monitor['top'] + y / scale), int(rw / scale), int(rh / scale),
                  reference=tuple(monitor[k] for k in ('left','top','width','height')))


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
