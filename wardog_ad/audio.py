"""Missile-lock detection from the game's audio (WASAPI loopback).

WARDOGS warns pilots by sound (community guides, Early Access build):
  * beeping           -> a launcher is ACQUIRING a lock
  * hard steady tone  -> lock COMPLETE, a missile may already be in the air

Nothing is hard-coded about the tones.  The user records short reference clips
(``audio-record``) and ``fit_profile`` learns, per warning kind, the narrow
frequency bins that carry the tone and a detection threshold.  At run time the
score is *tonal prominence*: how far those bins stand above the local spectral
floor.  Rotor noise, engines, gunfire and explosions are broadband, so they raise
the floor too and do not trigger.  The on/off envelope then separates a beeping
pattern from a continuous tone.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import threading
import time
import wave

import numpy as np

SAMPLE_RATE = 48000
N_FFT = 2048          # 23.4 Hz bins @ 48 kHz
HOP = 512             # 10.7 ms
KINDS = ('acquire', 'locked')
SMOOTH = 4            # frames (~43 ms) of score averaging; warning tones last >= 100 ms


# --------------------------------------------------------------------------- wav io
def write_wav(path, audio: np.ndarray, sr: int = SAMPLE_RATE):
    data = np.clip(np.asarray(audio, dtype=np.float32), -1, 1)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((data * 32767).astype('<i2').tobytes())


def read_wav(path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), 'rb') as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: '<i2', 4: '<i4'}[width]
    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    data = (data - 128) / 128 if width == 1 else data / float(2 ** (8 * width - 1))
    if ch > 1:
        data = data.reshape(-1, ch).mean(1)
    return data, sr


# --------------------------------------------------------------------------- spectra
_WINDOW = np.hanning(N_FFT).astype(np.float32)


def spectrogram(audio: np.ndarray) -> np.ndarray:
    """Log-power frames (n_frames, N_FFT//2+1) in dB."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < N_FFT:
        audio = np.pad(audio, (0, N_FFT - len(audio)))
    n = 1 + (len(audio) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = audio[idx] * _WINDOW
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return 10 * np.log10(power + 1e-10)


def prominence(spec_db: np.ndarray, half_width: int = 12) -> np.ndarray:
    """dB above the local spectral floor (running median over ±half_width bins)."""
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(spec_db, ((0, 0), (half_width, half_width)), mode='edge')
    floor = np.median(sliding_window_view(pad, 2 * half_width + 1, axis=1), axis=-1)
    return spec_db - floor


@dataclass
class ToneSignature:
    kind: str
    bins: list                 # FFT bin indices carrying the tone (fundamental + harmonics)
    freqs: list                # Hz, for humans
    threshold: float           # prominence (dB) above which a frame is "tone on"
    on_ratio: float            # fraction of frames 'on' in the reference clip
    period_s: float | None     # beep period if pulsed
    duty: float | None

    def score(self, prom_frames: np.ndarray) -> np.ndarray:
        """Per-frame score: mean prominence of the strongest half of the signature bins."""
        values = np.stack([prom_frames[:, max(0, b - 1):b + 2].max(1) for b in self.bins], 1)
        k = max(1, len(self.bins) // 2 + (len(self.bins) % 2))
        return np.sort(values, axis=1)[:, -k:].mean(1)

    def smoothed(self, prom_frames: np.ndarray) -> np.ndarray:
        raw = self.score(prom_frames)
        return np.convolve(np.pad(raw, (SMOOTH - 1, 0), mode='edge'), np.ones(SMOOTH) / SMOOTH, 'valid')


# --------------------------------------------------------------------------- missile sound
# The incoming missile is not a tone but a broadband rush/whoosh.  It is described by the
# SHAPE of its spectrum across 32 log-spaced bands, relative to the normal cockpit noise:
# score = projection of the band z-scores on the learned missile direction.  Gunfire and
# explosions are broadband too, but with a different shape and only short, so the score
# must stay up for ``min_s``; the flare logic also only listens after a lock warning.
BAND_EDGES = np.geomspace(150, 16000, 33)
_BAND_BINS = [(int(lo * N_FFT / SAMPLE_RATE), max(int(lo * N_FFT / SAMPLE_RATE) + 1, int(hi * N_FFT / SAMPLE_RATE)))
              for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]
MISSILE_SMOOTH = 12   # frames (~130 ms)


def band_energies(spec_db: np.ndarray) -> np.ndarray:
    """(frames, 32) band energies in dB from a dB spectrogram."""
    power = 10 ** (spec_db / 10)
    return 10 * np.log10(np.stack([power[:, a:b].sum(1) for a, b in _BAND_BINS], 1) + 1e-10)


@dataclass
class BroadbandSignature:
    kind: str
    bg_mean: list
    bg_std: list
    template: list            # unit vector in band z-score space
    threshold: float
    min_s: float = 0.25

    def score(self, bands: np.ndarray) -> np.ndarray:
        z = (bands - np.asarray(self.bg_mean)) / np.asarray(self.bg_std)
        # shape only: subtract the overall level, so broadband bangs (gunfire, explosions,
        # rotor gusts) that lift every band do not look like the missile's band
        z = z - z.mean(1, keepdims=True)
        return z @ np.asarray(self.template)


def _smooth(x, n):
    x = np.asarray(x, float)
    return np.convolve(np.pad(x, (n - 1, 0), mode='edge'), np.ones(n) / n, 'valid')


def fit_broadband(kind: str, reference: np.ndarray, background: np.ndarray) -> BroadbandSignature:
    """Learn the spectral shape of the missile sound against the normal cockpit noise."""
    bg = band_energies(spectrogram(background))
    mean, std = bg.mean(0), np.maximum(bg.std(0), 1.0)
    z = (band_energies(spectrogram(reference)) - mean) / std
    activity = z.mean(1)
    active = z[activity >= np.percentile(activity, 60)]
    direction = active.mean(0)
    direction = direction - direction.mean()
    if np.linalg.norm(direction) < 2.0:
        raise ValueError(f'"{kind}" sound is not distinct from the background: record it louder / closer')
    template = direction / np.linalg.norm(direction)
    sig = BroadbandSignature(kind, mean.tolist(), std.tolist(), template.tolist(), 0.0)
    on = np.percentile(_smooth(sig.score(band_energies(spectrogram(reference))), MISSILE_SMOOTH)[
        activity >= np.percentile(activity, 60)], 50)
    floor = np.percentile(_smooth(sig.score(bg), MISSILE_SMOOTH), 99.5)
    if on - floor < 1.5:
        raise ValueError(f'"{kind}" sound is not separable from the background ({on:.1f} vs {floor:.1f})')
    sig.threshold = float(floor + .35 * (on - floor))
    return sig


@dataclass
class LockProfile:
    sample_rate: int = SAMPLE_RATE
    signatures: dict = field(default_factory=dict)     # kind -> ToneSignature
    missile: BroadbandSignature | None = None           # incoming-missile sound (optional)

    def save(self, path):
        data = {'sample_rate': self.sample_rate,
                'signatures': {k: asdict(v) for k, v in self.signatures.items()},
                'missile': asdict(self.missile) if self.missile else None}
        Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path) -> 'LockProfile':
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        missile = data.get('missile')
        return cls(data['sample_rate'], {k: ToneSignature(**v) for k, v in data['signatures'].items()},
                   BroadbandSignature(**missile) if missile else None)


def _resample(audio, sr):
    if sr == SAMPLE_RATE:
        return np.asarray(audio, dtype=np.float32)
    n = int(round(len(audio) * SAMPLE_RATE / sr))
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


def _pulse_stats(on: np.ndarray) -> tuple[float | None, float | None]:
    edges = np.flatnonzero(np.diff(on.astype(int)) == 1)
    if len(edges) < 3:
        return None, None
    period = float(np.median(np.diff(edges))) * HOP / SAMPLE_RATE
    return period, float(on[edges[0]:edges[-1]].mean())


def fit_signature(kind: str, reference: np.ndarray, background: np.ndarray, *, max_bins: int = 4,
                  min_hz: float = 200, max_hz: float = 8000, margin_db: float = 3.0,
                  freqs: list | None = None) -> ToneSignature:
    """Learn the tone bins of ``kind`` from a reference clip and a background clip."""
    ref = prominence(spectrogram(reference))
    bg = prominence(spectrogram(background))
    lo, hi = int(min_hz * N_FFT / SAMPLE_RATE), int(max_hz * N_FFT / SAMPLE_RATE)
    # frames where *something* tonal is present in the reference (beeps are on only part-time)
    strength = ref[:, lo:hi].max(1)
    active = ref[strength >= np.percentile(strength, 60)]
    contrast = np.percentile(active, 75, axis=0) - np.percentile(bg, 95, axis=0)
    contrast[:lo] = contrast[hi:] = -np.inf
    bins = []
    if freqs:
        # frequencies already known (e.g. found by audio-harvest): do not search, the
        # search can latch onto music or other tones in a video soundtrack
        bins = [int(round(f * N_FFT / SAMPLE_RATE)) for f in freqs]
    for b in ([] if freqs else np.argsort(contrast)[::-1]):
        if contrast[b] < 6 or len(bins) >= max_bins:
            break
        if all(abs(b - c) > 3 for c in bins):
            bins.append(int(b))
    if not bins:
        raise ValueError(f'no distinct tone found for "{kind}": record the warning louder / with less background')
    sig = ToneSignature(kind, sorted(bins), [round(b * SAMPLE_RATE / N_FFT, 1) for b in sorted(bins)], 0, 0, None, None)
    ref_scores, bg_scores = sig.smoothed(ref), sig.smoothed(bg)
    on_level = np.percentile(ref_scores, 90)
    floor = np.percentile(bg_scores, 99.5)
    if on_level - floor < 4:
        raise ValueError(f'"{kind}" tone is not separable from the background ({on_level:.1f} vs {floor:.1f} dB)')
    # A fixed margin above the background's noise ceiling (not a fraction of the
    # reference loudness) keeps detection working when game volume changes.
    sig.threshold = float(floor + min(margin_db, 0.5 * (on_level - floor)))
    on = ref_scores > sig.threshold
    sig.on_ratio = float(on.mean())
    sig.period_s, sig.duty = _pulse_stats(on)
    return sig


def fit_profile(clips: dict, background: np.ndarray, freqs: dict | None = None) -> LockProfile:
    """``clips``: {'acquire': audio, 'locked': audio, 'missile': audio}; any subset is fine.
    ``freqs``: optional {'acquire': [Hz, ...], 'locked': [...]} to pin the tone frequencies."""
    freqs = freqs or {}
    tones = {k: fit_signature(k, a, background, freqs=freqs.get(k)) for k, a in clips.items() if k != 'missile'}
    missile = fit_broadband('missile', clips['missile'], background) if 'missile' in clips else None
    return LockProfile(SAMPLE_RATE, tones, missile)


# --------------------------------------------------------------------------- live detection
class LockState:
    NONE, ACQUIRING, LOCKED = 'none', 'acquiring', 'locked'


class AudioLockDetector:
    """Feeds on audio blocks; ``state`` is none / acquiring / locked.

    * LOCKED: the 'locked' tone is on (almost) continuously for ``steady_s``; if only an
      'acquire' signature exists, a continuous acquire tone also counts as locked.
    * ACQUIRING: the warning tone pulses (several on/off transitions in ``window_s``).
    Each state is held for ``hold_s`` after its evidence disappears.
    """

    def __init__(self, profile: LockProfile, *, steady_s: float = 0.35, window_s: float = 1.2,
                 hold_s: float = 0.5, steady_fill: float = 0.7):
        self.profile = profile
        self.steady_frames = max(3, int(steady_s * SAMPLE_RATE / HOP))
        self.window_frames = int(window_s * SAMPLE_RATE / HOP)
        self.hold_s, self.steady_fill = hold_s, steady_fill
        self.buffer = np.zeros(0, np.float32)
        self.history = {k: deque(maxlen=self.window_frames) for k in profile.signatures}
        self.scores = {k: 0.0 for k in profile.signatures}
        self.raw = {k: deque(maxlen=SMOOTH) for k in profile.signatures}
        self.state = LockState.NONE
        self.since = 0.0
        self.last_evidence = {LockState.ACQUIRING: -1e9, LockState.LOCKED: -1e9}
        # incoming-missile sound
        self.missile_scores: deque = deque(maxlen=MISSILE_SMOOTH)
        self.missile_on: deque = deque(maxlen=max(3, int((profile.missile.min_s if profile.missile else .25)
                                                        * SAMPLE_RATE / HOP)))
        self.missile_score = 0.0
        self.missile_last = -1e9
        self.now = 0.0

    @property
    def has_missile(self) -> bool:
        return self.profile.missile is not None

    @property
    def missile(self) -> bool:
        """Incoming-missile sound heard (sustained), held briefly."""
        return self.now - self.missile_last <= .3

    def feed(self, audio: np.ndarray, now: float) -> str:
        self.buffer = np.concatenate([self.buffer, np.asarray(audio, np.float32)])
        if len(self.buffer) >= N_FFT:
            usable = N_FFT + HOP * ((len(self.buffer) - N_FFT) // HOP)
            spec = spectrogram(self.buffer[:usable])
            prom = prominence(spec)
            self.buffer = self.buffer[usable - N_FFT + HOP:]
            if self.profile.missile is not None:
                sig = self.profile.missile
                for value in sig.score(band_energies(spec)):
                    self.missile_scores.append(float(value))
                    self.missile_score = sum(self.missile_scores) / len(self.missile_scores)
                    self.missile_on.append(self.missile_score > sig.threshold)
                if len(self.missile_on) == self.missile_on.maxlen and \
                        sum(self.missile_on) >= .7 * len(self.missile_on):
                    self.missile_last = now
            for kind, sig in self.profile.signatures.items():
                raw = self.raw[kind]
                for value in sig.score(prom):
                    raw.append(float(value))
                    smoothed = sum(raw) / len(raw)
                    self.history[kind].append(smoothed > sig.threshold)
                self.scores[kind] = smoothed
        return self._decide(now)

    def _steady(self, kind) -> bool:
        h = self.history.get(kind)
        if not h or len(h) < self.steady_frames:
            return False
        recent = list(h)[-self.steady_frames:]
        return sum(recent) >= self.steady_fill * len(recent)

    def _pulsing(self, kind) -> bool:
        h = self.history.get(kind)
        if not h:
            return False
        arr = np.array(h, dtype=int)
        return int((np.diff(arr) == 1).sum()) >= 2 and 0.1 < arr.mean() < self.steady_fill

    def _decide(self, now: float) -> str:
        self.now = now
        sigs = self.profile.signatures
        locked = self._steady('locked') if 'locked' in sigs else self._steady('acquire')
        # if both warnings share one frequency, a steady acquire tone IS the locked tone
        if 'locked' in sigs and 'acquire' in sigs and set(sigs['locked'].bins) & set(sigs['acquire'].bins):
            locked = locked or self._steady('acquire')
        acquiring = self._pulsing('acquire' if 'acquire' in sigs else 'locked')
        if locked:
            self.last_evidence[LockState.LOCKED] = now
        if acquiring:
            self.last_evidence[LockState.ACQUIRING] = now
        if now - self.last_evidence[LockState.LOCKED] <= self.hold_s:
            state = LockState.LOCKED
        elif now - self.last_evidence[LockState.ACQUIRING] <= self.hold_s:
            state = LockState.ACQUIRING
        else:
            state = LockState.NONE
        if state != self.state:
            self.state, self.since = state, now
        return state


class LoopbackTap:
    """Background thread capturing what the speakers play (WASAPI loopback)."""

    def __init__(self, device: str | None = None, block: int = 1024):
        self.device, self.block = device, block
        self.queue: deque = deque(maxlen=400)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name='audio-tap', daemon=True)
        self.error: Exception | None = None

    def start(self) -> 'LoopbackTap':
        self.thread.start()
        return self

    def _run(self):
        try:
            import soundcard as sc
            speaker = sc.get_speaker(self.device) if self.device else sc.default_speaker()
            mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            with mic.recorder(samplerate=SAMPLE_RATE, channels=2, blocksize=self.block) as rec:
                while not self.stop_event.is_set():
                    data = rec.record(numframes=self.block)
                    self.queue.append(data.mean(1).astype(np.float32))
        except Exception as error:   # surfaced to the caller via .error
            self.error = error

    def read(self) -> np.ndarray:
        chunks = []
        while self.queue:
            chunks.append(self.queue.popleft())
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def close(self):
        self.stop_event.set()
        self.thread.join(1.0)


def record_loopback(seconds: float, device: str | None = None) -> np.ndarray:
    tap = LoopbackTap(device).start()
    end = time.monotonic() + seconds
    chunks = []
    while time.monotonic() < end:
        time.sleep(0.05)
        chunks.append(tap.read())
        if tap.error:
            raise tap.error
    tap.close()
    chunks.append(tap.read())
    return np.concatenate(chunks)
