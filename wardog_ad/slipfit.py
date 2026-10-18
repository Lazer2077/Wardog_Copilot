"""Measure how the helicopter's motion follows its nose (sideslip) from a flight log.

After a fast pedal turn the motion either swings onto the nose at once (arcade) or keeps
sliding (thrust/drag physics).  Which one the game does decides how the J-hook and the
approach must steer, so it is measured, not guessed: every turn at speed in the log is
replayed through the navigator's dead reckoning (HUD speed, compass, pitch - no fixes)
with candidate alignment rates, and the rate whose predicted track stays closest to the
minimap fixes wins.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .airnav import AirNavigator
from .geometry import angle_delta

CANDIDATES = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, math.inf)


@dataclass
class SlipFit:
    best: float | None             # alignment rate 1/s (inf = always along the nose)
    errors: dict                   # candidate -> median end-of-window error (m)
    windows: int                   # turns at speed that were replayed
    message: str


def _samples(rows):
    """(t, heading, speed_kmh, pitch) HUD samples and (t, x_m, y_m) fixes, in metres."""
    from .assets import load_map
    scale, hud, fixes = {}, [], []
    for r in rows:
        if r.get('profile') != 'aircraft' or not r.get('map_id'):
            continue
        mid = r['map_id']
        if mid not in scale:
            try:
                scale[mid] = load_map(mid).meters_per_pixel
            except Exception:
                continue
        h, f = r.get('hud') or {}, r.get('fix') or {}
        if h.get('timestamp') is not None and h.get('speed_kmh') is not None and h.get('heading') is not None:
            hud.append((h['timestamp'], h['heading'], h['speed_kmh'], h.get('pitch'), scale[mid]))
        if f.get('valid') and f.get('timestamp') is not None:
            fixes.append((f['timestamp'], f['x'], f['y']))
    hud = sorted({s[0]: s for s in hud}.values())
    fixes = sorted({s[0]: s for s in fixes}.values())
    return hud, fixes, (next(iter(scale.values())) if scale else None)


def _turn_windows(hud, fixes, *, length_s=8.0, min_kmh=80.0, min_turn_deg=30.0):
    """(start time, fix index) of turns at speed: straight for 3 s, then the nose swings
    >= ``min_turn_deg`` within the window.  Each replay starts at the last fix before the
    turn, when the motion still lies along the nose."""
    ft = np.array([f[0] for f in fixes])
    starts, last = [], -1e9
    for i, (t, h, v, *_ ) in enumerate(hud):
        if t - last < length_s or v < min_kmh:
            continue
        before = [s for s in hud[max(0, i - 60):i] if t - 3.0 <= s[0] < t]
        soon = [s for s in hud[i:i + 30] if s[0] <= t + 1.0]
        after = [s for s in hud[i:i + 300] if s[0] <= t + length_s]
        if len(before) < 6 or before[-1][0] - before[0][0] < 2.0 or not soon or len(after) < 10:
            continue
        if max(abs(angle_delta(h, s[1])) for s in before) > 5:                 # not straight before
            continue
        if max(abs(angle_delta(h, s[1])) for s in soon) < 6:                   # no turn starting
            continue
        if max(abs(angle_delta(h, s[1])) for s in after) < min_turn_deg:
            continue
        k = int(np.searchsorted(ft, t)) - 1                                    # last fix before
        if k < 0 or t - ft[k] > 1.2 or np.sum((ft > ft[k] + 2.0) & (ft <= ft[k] + length_s)) < 3:
            continue
        starts.append((float(ft[k]), k))
        last = t
    return starts


def _replay(hud, fixes, mpp, start, fix_index, align, length_s):
    nav = AirNavigator(1.0, align=align)               # positions in metres
    t0 = start
    _, x0, y0 = fixes[fix_index]
    nav.start((x0 * mpp, y0 * mpp), t0)
    errs = []
    fi = fix_index + 1
    for t, heading, speed, pitch, _ in hud:
        if t < t0 - 1.0:
            continue
        if t > t0 + length_s:
            break
        if t < t0:                                      # settle heading/speed before the start
            nav.predict(t, speed, heading, pitch)
            nav.pos = np.array((x0 * mpp, y0 * mpp))
            continue
        nav.predict(t, speed, heading, pitch)
        while fi < len(fixes) and fixes[fi][0] <= t:
            ft, fx, fy = fixes[fi]
            if ft > t0 + 2.0:
                errs.append(math.hypot(nav.pos[0] - fx * mpp, nav.pos[1] - fy * mpp))
            fi += 1
    return float(np.median(errs)) if errs else None


def fit_slip(rows, *, length_s: float = 8.0, candidates=CANDIDATES) -> SlipFit:
    hud, fixes, mpp = _samples(rows)
    if mpp is None or len(hud) < 50 or len(fixes) < 20:
        return SlipFit(None, {}, 0, 'not enough helicopter data with minimap fixes in this log')
    windows = _turn_windows(hud, fixes, length_s=length_s)
    if len(windows) < 2:
        return SlipFit(None, {}, len(windows),
                       f'only {len(windows)} fast turn(s) with fixes - fly a few J-hooks (press return while '
                       'cruising away from home) with the minimap visible, then run this again')
    errors = {}
    for align in candidates:
        per = [_replay(hud, fixes, mpp, t0, fi, align, length_s) for t0, fi in windows]
        per = [e for e in per if e is not None]
        errors[align] = float(np.median(per)) if per else math.inf
    best = min(errors, key=errors.get)
    ranked = sorted(errors.values())
    note = '' if len(ranked) < 2 or ranked[1] > 1.15 * ranked[0] else ' (the next best is close: fly more turns)'
    return SlipFit(best, errors, len(windows),
                   f'best alignment {best}/s over {len(windows)} turns, median error {errors[best]:.0f} m{note}')


def load_rows(path) -> list:
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def latest_log(directory='logs') -> Path | None:
    logs = sorted(Path(directory).glob('flight_*.jsonl'), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None
