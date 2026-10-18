"""OpenCV debug window: map crop around the vehicle with route, heading and status."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .geometry import heading_vector


def draw_overlay(base: np.ndarray, tel, *, center=None, radius: int = 220, zoom: float = 2.0) -> np.ndarray:
    st = tel.state
    if center is None:
        center = st.position if st is not None else (base.shape[1] / 2, base.shape[0] / 2)
    cx, cy = center
    x0, y0 = int(cx - radius / zoom), int(cy - radius / zoom)
    size = int(2 * radius / zoom)
    pad = np.zeros((size, size, 3), np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(base.shape[1], x0 + size), min(base.shape[0], y0 + size)
    if sx1 > sx0 and sy1 > sy0:
        pad[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = base[sy0:sy1, sx0:sx1]
    view = cv2.resize(pad, (2 * radius, 2 * radius), interpolation=cv2.INTER_LINEAR)

    def to_view(p):
        return int((p[0] - x0) * zoom), int((p[1] - y0) * zoom)

    if len(tel.route) >= 2:
        cv2.polylines(view, [np.int32([to_view(p) for p in tel.route])], False, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.circle(view, to_view(tel.route[-1]), 6, (0, 0, 255), -1)
    if st is not None:
        p = to_view(st.position)
        tip = np.array(p) + heading_vector(st.heading) * 18
        cv2.arrowedLine(view, p, tuple(int(v) for v in tip), (255, 255, 255), 2, tipLength=.4)
        cv2.circle(view, p, 4, (0, 255, 0), -1)
    lines = [f'{tel.status}  engaged={tel.engaged}  {tel.message}']
    if st is not None:
        lines.append(f'v={st.speed_ms:5.1f} m/s  hdg={st.heading:5.1f} ({st.heading_source})  yaw={st.yaw_rate:+5.1f}')
    if tel.fix is not None:
        lines.append(f'fix {tel.fix.reason} conf={tel.fix.confidence:.2f} inl={tel.fix.inliers}')
    if tel.output is not None:
        axes = ' '.join(f'{k}={v:+.2f}' for k, v in tel.output.axes.items())
        extra = ' '.join(f'{k}={v:.1f}' for k, v in tel.output.info.items() if isinstance(v, (int, float)))
        lines += [axes, extra[:90]]
    if tel.lock_state != 'none' or tel.flares:
        lines.append(f'MISSILE {tel.lock_state.upper()}  LOCK={tel.locked}  flares={tel.flares}')
    for i, text in enumerate(lines):
        cv2.putText(view, text, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, text, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, .48,
                    (80, 80, 255) if 'LOCK=True' in text else (255, 255, 255), 1, cv2.LINE_AA)
    return view


class DebugView:
    def __init__(self, assets, title='wardog_ad'):
        self.base = assets.load_image()
        self.title = title

    def show(self, tel) -> bool:
        cv2.imshow(self.title, draw_overlay(self.base, tel))
        return (cv2.waitKey(1) & 0xFF) == 27  # Esc closes


def route_preview(base: np.ndarray, route, path_out, margin: int = 40):
    pts = np.int32(route)
    x0, y0 = np.maximum(pts.min(0) - margin, 0)
    x1, y1 = np.minimum(pts.max(0) + margin, [base.shape[1], base.shape[0]])
    crop = base[y0:y1, x0:x1].copy()
    scale = max(1.0, 700 / max(crop.shape[:2]))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    shifted = np.int32((pts - [x0, y0]) * scale)
    cv2.polylines(crop, [shifted], False, (0, 200, 255), max(2, int(scale)), cv2.LINE_AA)
    cv2.circle(crop, tuple(shifted[0]), 7, (0, 255, 0), -1)
    cv2.circle(crop, tuple(shifted[-1]), 7, (0, 0, 255), -1)
    cv2.imencode('.png', crop)[1].tofile(str(path_out))
    return math.hypot(*(pts[-1] - pts[0]))
