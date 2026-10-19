"""Read the WARDOGS helicopter HUD: SPD (km/h), ALT AGL / ASL (m) and the compass heading.

Layout (MH-6, 3840x2160, from a real screenshot): the instrument cluster sits at the
bottom centre::

    SPD             ALT
    KM/H  00        AGL    0
                    ASL   84

and the compass tape at the top shows the heading in a box ("207SW").

Reading strategy (no training):
  * Windows OCR (WinRT, in-process, ~5 ms) reads label+value lines such as "ASL 84".
    It is unreliable on a lone digit, and reads 0 as O.
  * Every successful OCR read teaches per-digit glyph templates (the HUD font is
    fixed).  Afterwards values are read by template matching first; OCR is only the
    fallback, and its results keep enriching the template library (saved to disk).
  * A plausibility filter rejects jumps that are physically impossible.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import re
import threading
import time

import cv2
import numpy as np

GLYPH_SIZE = (16, 24)   # w, h
OCR_FIX = str.maketrans({'O': '0', 'o': '0', 'D': '0', 'Q': '0', 'U': '0', 'l': '1', 'I': '1', '|': '1',
                         'i': '1', '!': '1', 'S': '5', 's': '5', 'B': '8', 'Z': '2', 'z': '2', 'G': '6', 'b': '6',
                         'q': '9', 'g': '9', 'T': '7'})


@dataclass
class HudBox:
    """Sub-box in fractions of the instrument-cluster region."""
    x0: float
    y0: float
    x1: float
    y1: float

    def crop(self, img):
        h, w = img.shape[:2]
        return img[int(self.y0 * h):int(self.y1 * h), int(self.x0 * w):int(self.x1 * w)]


# measured on the 720x270 cluster crop of a 3840x2160 screenshot
LAYOUT = {
    # the speed digits are right-aligned against 'KM/H' (ending ~x 162): a 3-digit number starts
    # at ~113, and the instrument arc begins at ~192 - the old 120..200 box cut the first digit
    # (167 read as 68) and caught the arc (no reading)
    'spd': (HudBox(30 / 720, 65 / 270, 107 / 720, 108 / 270), HudBox(108 / 720, 65 / 270, 178 / 720, 108 / 270)),
    'agl': (HudBox(552 / 720, 65 / 270, 625 / 720, 108 / 270), HudBox(626 / 720, 65 / 270, 706 / 720, 108 / 270)),
    'asl': (HudBox(552 / 720, 110 / 270, 625 / 720, 152 / 270), HudBox(626 / 720, 110 / 270, 706 / 720, 152 / 270)),
}


# Attitude instruments on the same crop (measured on level and nose-down frames):
#   * pitch: the two triangles on the outer arcs; level = centre y 122, one tick
#     mark = 33.7 px, they move DOWN when the nose goes down (clamped at ~3.5 ticks).
#   * collective lever: the lit part of the inner arcs runs from the centre (y 121)
#     towards the lever end: centre..bottom = lever minimum, top..centre = maximum.
PITCH_COLUMNS = ((183 / 720, 210 / 720), (508 / 720, 535 / 720))
ARC_COLUMNS = ((214 / 720, 276 / 720), (444 / 720, 506 / 720))
CENTRE_Y, TICK_PX, ARC_HALF_PX = 122 / 270, 33.7 / 270, 97 / 270


@dataclass
class HudReading:
    speed_kmh: float | None = None
    agl_m: float | None = None
    asl_m: float | None = None
    heading: float | None = None
    timestamp: float = 0.0
    pitch: float | None = None        # tick marks, + = nose DOWN (one tick is probably 10 deg)
    collective: float | None = None   # lever position -1 (down) .. +1 (up)

    @property
    def complete(self) -> bool:
        return self.speed_kmh is not None and self.agl_m is not None


# --------------------------------------------------------------------------- OCR
class WindowsOcr:
    def __init__(self):
        from winrt.windows.media.ocr import OcrEngine
        self.engine = OcrEngine.try_create_from_user_profile_languages()
        if self.engine is None:
            raise RuntimeError('Windows OCR is not available (install an OCR language in Windows settings)')
        self.loop = asyncio.new_event_loop()
        self.lock = threading.Lock()

    def __call__(self, img: np.ndarray) -> str:
        from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
        from winrt.windows.storage.streams import DataWriter
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA if img.ndim == 3 else cv2.COLOR_GRAY2BGRA)
        writer = DataWriter()
        writer.write_bytes(bgra.tobytes())
        bitmap = SoftwareBitmap.create_copy_from_buffer(writer.detach_buffer(), BitmapPixelFormat.BGRA8,
                                                        bgra.shape[1], bgra.shape[0])

        async def run():
            return (await self.engine.recognize_async(bitmap)).text

        with self.lock:
            return self.loop.run_until_complete(run())


def glue(label: np.ndarray, value: np.ndarray, scale: float = 2.0, gap: int = 14) -> np.ndarray:
    """Label and value side by side with a small gap: OCR then sees one text line."""
    h = max(label.shape[0], value.shape[0])

    def pad(x):
        return cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_REPLICATE)

    fill = np.full((h, gap, 3), np.median(value.reshape(-1, 3), 0), np.uint8)
    img = cv2.resize(np.hstack([pad(label), fill, pad(value)]), None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_REPLICATE)


def parse_number(text: str, label: str | None = None) -> str | None:
    """Digits after the label in an OCR line, with letter/digit confusions fixed."""
    text = text.strip()
    if label:
        m = re.search(re.escape(label) + r'\s*(.*)$', text.replace('KWH', 'KM/H'), re.I)
        if not m:
            return None
        text = m.group(1)
    digits = text.replace(' ', '').translate(OCR_FIX)
    m = re.fullmatch(r'\d{1,5}', digits)
    return m.group(0) if m else None


def mask_for_ocr(img: np.ndarray, height: int = 60) -> np.ndarray:
    """The HUD digits as black text on white, scaled to ~``height`` px, padded: what OCR reads best."""
    mask = text_mask(img)
    ys, xs = np.nonzero(mask)
    if len(xs):
        mask = mask[max(0, ys.min() - 1):ys.max() + 2, max(0, xs.min() - 1):xs.max() + 2]
    scale = height / max(1, mask.shape[0])
    big = cv2.resize((1 - mask) * 255, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    big = cv2.copyMakeBorder(big, 20, 20, 30, 30, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------- glyphs
def text_mask(img: np.ndarray) -> np.ndarray:
    """Binary mask of the light, unsaturated HUD text."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    bg = float(np.median(v))
    thr = max(150.0, bg + 45.0, cv2.threshold(v.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    return ((v >= thr) & (hsv[:, :, 1] < 70)).astype(np.uint8)


def glyphs(img: np.ndarray) -> list[np.ndarray]:
    """Character images (left to right), normalised to GLYPH_SIZE."""
    mask = text_mask(img)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    h = img.shape[0]
    # join the parts of a glyph first, THEN drop small specks: at 1080p the top bar of a '5'
    # comes apart from its body and was thrown away on its own ('53' read as '3')
    boxes = [stats[i, :4] for i in range(1, n) if stats[i, 4] >= 3]
    boxes.sort(key=lambda b: b[0])
    merged = []
    for x, y, w, hh in boxes:   # parts of one glyph overlap horizontally
        if merged and x < merged[-1][0] + merged[-1][2] - 1:
            mx, my, mw, mh = merged[-1]
            x1, y1 = max(mx + mw, x + w), max(my + mh, y + hh)
            merged[-1] = [min(mx, x), min(my, y), x1 - min(mx, x), y1 - min(my, y)]
        else:
            merged.append([x, y, w, hh])
    out = []
    for x, y, w, hh in merged:
        if hh < .35 * h or int(mask[y:y + hh, x:x + w].sum()) < 6:
            continue                      # a speck, not a digit
        g = mask[y:y + hh, x:x + w].astype(np.float32)
        side = max(w, int(hh * GLYPH_SIZE[0] / GLYPH_SIZE[1]))   # keep '1' thin
        canvas = np.zeros((hh, side), np.float32)
        canvas[:, (side - w) // 2:(side - w) // 2 + w] = g
        out.append(cv2.resize(canvas, GLYPH_SIZE, interpolation=cv2.INTER_AREA))
    return out


class GlyphLibrary:
    def __init__(self, path: str | Path | None = None, per_class: int = 6):
        self.path = Path(path) if path else None
        self.per_class = per_class
        self.templates: dict[str, list[np.ndarray]] = {}
        self.dirty = False
        if self.path and self.path.exists():
            with np.load(self.path) as data:
                for key in data.files:
                    self.templates[key] = list(data[key])
            self._prune()

    def _prune(self):
        """Drop templates that look the same as another digit's (learned from an OCR misread)."""
        bad = set()
        keys = list(self.templates)
        for a in keys:
            for i, g in enumerate(self.templates[a]):
                for b in keys:
                    if b != a and any(_ncc(g, t) > .9 for t in self.templates[b]):
                        bad.add((a, i))
        for a, i in sorted(bad, reverse=True):
            del self.templates[a][i]
            self.dirty = True

    def learn(self, text: str, imgs: list[np.ndarray]):
        if len(text) != len(imgs) or not text.isdigit():
            return
        for ch, g in zip(text, imgs):
            other, score = self.classify(g)
            if other is not None and other != ch and score > .85:
                continue              # looks like another digit: the OCR probably misread
            bank = self.templates.setdefault(ch, [])
            if all(_ncc(g, t) < .97 for t in bank):   # keep distinct exemplars only
                bank.append(g)
                del bank[:-self.per_class]
                self.dirty = True

    def classify(self, g: np.ndarray) -> tuple[str | None, float]:
        best, score = None, -1.0
        for ch, bank in self.templates.items():
            for t in bank:
                s = _ncc(g, t)
                if s > score:
                    best, score = ch, s
        return best, score

    def read(self, imgs: list[np.ndarray], min_score: float = .8) -> str | None:
        if not imgs:
            return None
        chars = []
        for g in imgs:
            ch, s = self.classify(g)
            if ch is None or s < min_score:
                return None
            chars.append(ch)
        return ''.join(chars)

    def save(self):
        if self.path and self.dirty:
            np.savez_compressed(self.path, **{k: np.stack(v) for k, v in self.templates.items()})
            self.dirty = False


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def _bright(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (hsv[:, :, 2] > 150) & (hsv[:, :, 1] < 70)


def read_pitch(cluster: np.ndarray) -> float | None:
    """Pitch in tick marks from the triangles (both sides must agree)."""
    h, w = cluster.shape[:2]
    scale = h / 270
    found = []
    for x0, x1 in PITCH_COLUMNS:
        m = _bright(cluster[:, int(x0 * w):int(x1 * w)]).astype(np.uint8)
        n, _, st, cen = cv2.connectedComponentsWithStats(m, 8)
        tri = [cen[i][1] for i in range(1, n)
               if 14 * scale <= st[i, 3] <= 30 * scale and 7 * scale <= st[i, 2] <= 16 * scale
               and st[i, 4] >= 60 * scale * scale]
        if len(tri) != 1:
            return None
        found.append(tri[0])
    if abs(found[0] - found[1]) > 6 * scale:
        return None
    return (float(np.mean(found)) / h - CENTRE_Y) / TICK_PX


def read_collective(cluster: np.ndarray) -> float | None:
    """Collective lever from the lit arc segment (-1 bottom .. +1 top)."""
    h, w = cluster.shape[:2]
    centre, half = CENTRE_Y * h, ARC_HALF_PX * h
    values = []
    for x0, x1 in ARC_COLUMNS:
        rows = _bright(cluster[:, int(x0 * w):int(x1 * w)]).sum(1) >= 2
        runs, start = [], None
        for y, lit in enumerate(list(rows) + [False]):
            if lit and start is None:
                start = y
            elif not lit and start is not None:
                runs.append((start, y - 1))
                start = None
        runs = [r for r in runs if r[1] - r[0] >= 8 * h / 270 and r[0] < centre + half * .9]
        if not runs:
            values.append(0.0)       # nothing lit: lever exactly centred
            continue
        y0, y1 = max(runs, key=lambda r: r[1] - r[0])
        if abs(y0 - centre) < abs(y1 - centre):      # centre .. down
            values.append(-min(1.0, (y1 - centre) / half))
        else:                                        # up .. centre
            values.append(min(1.0, (centre - y0) / half))
    if abs(values[0] - values[1]) > .2:
        return None
    return float(np.mean(values))


# --------------------------------------------------------------------------- reader
class HudReader:
    """Reads one HUD frame.  ``cluster`` = instrument-cluster crop, ``compass`` = heading box crop."""

    LABELS = {'spd': 'KM/H', 'agl': 'AGL', 'asl': 'ASL'}
    LIMITS = {'spd': 380.0, 'agl': 3000.0, 'asl': 5000.0}

    def __init__(self, glyph_path: str | Path | None = None, ocr=None, layout=LAYOUT):
        self.library = GlyphLibrary(glyph_path)
        self.layout = layout
        self._ocr = ocr
        self.last: dict[str, tuple[float, float]] = {}   # field -> (value, time)
        self.ocr_calls = 0

    @property
    def ocr(self):
        if self._ocr is None:
            self._ocr = WindowsOcr()
        return self._ocr

    def read_field(self, cluster: np.ndarray, name: str) -> float | None:
        label_box, value_box = self.layout[name]
        value_img = value_box.crop(cluster)
        imgs = glyphs(value_img)
        text = self.library.read(imgs)
        if text is None:
            self.ocr_calls += 1
            text = parse_number(self.ocr(glue(label_box.crop(cluster), value_img)), self.LABELS[name])
            if text is None or len(text) != len(imgs):
                # OCR the clean digit mask (black on white, enlarged): at 1080p the digits are
                # ~8 px and the raw crop over trees was misread (48 -> 3, 366 -> 30)
                self.ocr_calls += 1
                text = parse_number(self.ocr(mask_for_ocr(value_img)))
                if text is not None and len(text) != len(imgs):
                    text = None           # OCR and the segmentation disagree on the digit count
            if text is not None:
                self.library.learn(text, imgs)
        try:
            return float(text) if text is not None else None
        except ValueError:
            return None

    def read_heading(self, compass: np.ndarray) -> float | None:
        img = cv2.resize(compass, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        self.ocr_calls += 1
        m = re.match(r'\s*([0-9OoDlI|SB]{1,3})', self.ocr(img))
        if not m:
            return None
        text = m.group(1).translate(OCR_FIX)
        return float(text) % 360 if text.isdigit() else None

    def read(self, cluster: np.ndarray | None, compass: np.ndarray | None, now: float) -> HudReading:
        out = HudReading(timestamp=now)
        if cluster is not None and cluster.size:
            for name, attr, max_rate in (('spd', 'speed_kmh', 120.0), ('agl', 'agl_m', 80.0), ('asl', 'asl_m', 60.0)):
                value = self.read_field(cluster, name)
                setattr(out, attr, self._plausible(name, value, now, max_rate))
            out.pitch = read_pitch(cluster)
            out.collective = read_collective(cluster)
        if compass is not None and compass.size:
            out.heading = self._plausible_heading(self.read_heading(compass), now)
        self.library.save()
        return out

    def _plausible_heading(self, heading, now, max_rate=90.0):
        """Accept a compass reading if it fits the last one (yaw rate limit) or repeats."""
        if heading is None:
            return None
        last, candidate = self.last.get('hdg'), self.last.get('hdg_candidate')
        ok = last is None or now - last[1] > 3.0 or \
            abs((heading - last[0] + 180) % 360 - 180) <= max_rate * (now - last[1]) + 15
        if not ok and candidate is not None and abs((heading - candidate[0] + 180) % 360 - 180) <= 5:
            ok = True                   # the same "jump" twice in a row: it is real
        if ok:
            self.last['hdg'] = (heading, now)
            self.last.pop('hdg_candidate', None)
            return heading
        self.last['hdg_candidate'] = (heading, now)
        return None

    def _plausible(self, name, value, now, max_rate):
        """Reject a reading that changed faster than ``max_rate`` units/s (OCR glitch)."""
        if value is None:
            return None
        if value > self.LIMITS.get(name, 1e9):
            return None                 # beyond the aircraft (MH-6 tops out at 350 km/h)
        previous = self.last.get(name)
        if previous is not None:
            dt = max(0.05, now - previous[1])
            if abs(value - previous[0]) > max_rate * dt + 3 and now - previous[1] < 2.0:
                return None
            if name in ('agl', 'asl') and abs(value - previous[0]) > 25 and now - previous[1] < 10.0:
                # a big height jump must be read twice in a row (a cliff, or a garbled digit:
                # AGL 68 -> 6 -> 65 at 16:43, each '6' a pull-up)
                candidate = self.last.get(name + '_candidate')
                if candidate is None or abs(candidate[0] - value) > 5 or now - candidate[1] > 1.0:
                    self.last[name + '_candidate'] = (value, now)
                    return None
        self.last.pop(name + '_candidate', None)
        self.last[name] = (value, now)
        return value
