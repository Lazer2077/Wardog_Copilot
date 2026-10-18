"""Image cues besides localization: main-view motion and the missile-lock warning."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class MotionMonitor:
    """How much the 3-D main view changes between frames (0 = frozen picture).

    A truck pushing against a wall still shows a moving minimap arrow jitter but an
    almost static world view; combined with near-zero estimated speed this flags
    "stuck" robustly.  Uses phase correlation on a downscaled grayscale frame.
    """

    def __init__(self, size: int = 160, history: int = 8):
        self.size = size
        self.previous: np.ndarray | None = None
        self.values: deque[float] = deque(maxlen=history)

    def update(self, frame: np.ndarray | None) -> float | None:
        if frame is None:
            return None
        h, w = frame.shape[:2]
        scale = self.size / max(h, w)
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.previous is None or self.previous.shape != gray.shape:
            self.previous = gray
            return None
        diff = float(np.mean(np.abs(gray - self.previous))) / 255.0
        (dx, dy), _ = cv2.phaseCorrelate(self.previous, gray)
        self.previous = gray
        value = diff * 10 + min(1.0, np.hypot(dx, dy) / 4)
        self.values.append(value)
        return value

    @property
    def level(self) -> float:
        return float(np.median(self.values)) if self.values else 1.0


@dataclass
class LockReading:
    positive: bool
    score: float
    color_ratio: float
    template_score: float | None


class MissileLockDetector:
    """Detects the missile-lock warning inside a calibrated HUD region.

    Two cues, either of which can be used:
      * template: a cropped screenshot of the warning icon/text (multi-scale NCC)
      * colour:   fraction of ROI pixels in the warning colour (red by default)
    Blinking warnings are bridged by ``hold_s``; ``confirm_frames`` consecutive
    positives are required to switch on, which suppresses single-frame noise.
    """

    def __init__(self, tuning, template_path: str | Path | None = None):
        self.t = tuning
        self.template = None
        if template_path and Path(template_path).exists():
            img = cv2.imdecode(np.fromfile(str(template_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                self.template = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.streak = 0
        self.last_positive = -1e9
        self.active = False

    def measure(self, roi: np.ndarray) -> LockReading:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.t.lock_color_hsv_low), np.array(self.t.lock_color_hsv_high))
        mask |= cv2.inRange(hsv, np.array(self.t.lock_color_hsv_low2), np.array(self.t.lock_color_hsv_high2))
        ratio = float(np.count_nonzero(mask)) / mask.size
        template_score = None
        if self.template is not None:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            template_score = 0.0
            for scale in (0.8, 0.9, 1.0, 1.1, 1.25):
                tpl = cv2.resize(self.template, None, fx=scale, fy=scale)
                if tpl.shape[0] > gray.shape[0] or tpl.shape[1] > gray.shape[1] or min(tpl.shape) < 4:
                    continue
                template_score = max(template_score, float(cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED).max()))
        if template_score is not None:
            score = template_score
            positive = template_score >= self.t.lock_threshold
        else:
            score = min(1.0, ratio / max(self.t.lock_color_ratio, 1e-6))
            positive = ratio >= self.t.lock_color_ratio
        return LockReading(positive, score, ratio, template_score)

    def update(self, roi: np.ndarray | None, now: float) -> bool:
        reading = self.measure(roi) if roi is not None and roi.size else LockReading(False, 0, 0, None)
        self.reading = reading
        self.streak = self.streak + 1 if reading.positive else 0
        if self.streak >= self.t.lock_confirm_frames:
            self.last_positive = now
            self.active = True
        elif reading.positive and self.active:
            self.last_positive = now
        elif now - self.last_positive > self.t.lock_hold_s:
            self.active = False
        return self.active


class FlareController:
    """Fires flare bursts while a lock is active: delay -> burst -> cooldown -> burst ...

    Community guides for WARDOGS: flares are limited and on cooldown, and work best
    1-2 s before impact, so nothing is fired during the acquiring beeps (unless
    ``flare_on='acquire'``) and the first burst waits ``flare_delay_s`` after lock.
    """

    def __init__(self, tuning, fire):
        self.t = tuning
        self.fire = fire              # callable() that taps the flare key
        self.remaining = 0
        self.next_shot = 0.0
        self.cooldown_until = 0.0
        self.fired = 0
        self.lock_start: float | None = None
        self.last_warning = -1e9      # last time a lock warning (beeping or tone) was heard
        self.locked_since: float | None = None
        self.can_hear_missile = False # the audio profile has a recorded missile sound
        self.reason = 'none'

    def reset_life(self):
        self.fired = 0

    @property
    def empty(self) -> bool:
        return self.t.flare_capacity > 0 and self.fired >= self.t.flare_capacity

    def update_state(self, state: str, now: float, missile: bool = False):
        """``flare_on``:
          * 'missile' (default): after a lock warning, fire as soon as the incoming missile is
            HEARD (how players do it: alarm -> missile sound -> flares, 1-2 s before impact).
            If the steady lock tone lasts ``locked_fallback_s`` without a missile being heard,
            fire anyway (the missile may be inaudible over the noise).
          * 'locked': ``flare_delay_s`` after the steady lock tone;  'acquire': already on beeps.
        Without a recorded missile sound, 'missile' behaves like 'locked'."""
        if state != 'none':
            self.last_warning = now
        self.locked_since = (self.locked_since or now) if state == 'locked' else None
        if self.t.flare_on == 'missile' and self.can_hear_missile:
            warned = now - self.last_warning <= self.t.lock_memory_s
            heard = missile and warned
            fallback = self.locked_since is not None and now - self.locked_since >= self.t.locked_fallback_s
            self.reason = 'missile heard' if heard else 'lock tone too long' if fallback else ''
            self.update(heard or fallback, now, delay=0.0 if heard else None)
            return
        trigger = state == 'locked' or (state == 'acquiring' and self.t.flare_on == 'acquire')
        self.reason = 'lock' if trigger else ''
        self.update(trigger, now)

    def update(self, locked: bool, now: float, delay: float | None = None):
        if not self.t.flare_enabled:
            return
        if not locked:
            self.lock_start = None
        elif self.lock_start is None:
            self.lock_start = now
        wait = self.t.flare_delay_s if delay is None else delay
        ready = locked and now - self.lock_start >= wait
        if ready and self.remaining == 0 and now >= self.cooldown_until and not self.empty:
            self.remaining = self.t.flare_burst
            self.next_shot = now
        if self.remaining and now >= self.next_shot and not self.empty:
            self.fire()
            self.fired += 1
            self.remaining -= 1
            self.next_shot = now + self.t.flare_interval_s
            if self.remaining == 0:
                self.cooldown_until = now + self.t.flare_cooldown_s
