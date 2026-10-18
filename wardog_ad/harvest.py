"""Find the lock-warning tones in a long recording (e.g. a YouTube video played through
the speakers and captured with ``audio-record --kind raw``) and cut clean clips for
``audio-fit``.

Warning tones are electronic: a fixed frequency (plus harmonics), switched on and off.
Speech and engine noise drift in pitch.  So we look for narrow spectral peaks that stay
on the SAME FFT bin for a while, group them by frequency, and classify each frequency:
  * many short repeats at a regular period -> beeping -> 'acquire'
  * long continuous runs                    -> steady  -> 'locked'
Stretches without any such tone become 'background'.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .audio import HOP, N_FFT, SAMPLE_RATE, prominence, spectrogram

FRAME_S = HOP / SAMPLE_RATE


@dataclass
class ToneRun:
    bin: int
    start: float          # s
    end: float            # s

    @property
    def freq(self) -> float:
        return self.bin * SAMPLE_RATE / N_FFT

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass
class ToneGroup:
    freq: float
    runs: list
    kind: str                 # 'acquire' | 'locked' | 'other'
    period_s: float | None
    total_s: float


def tone_runs(audio: np.ndarray, *, min_db: float = 14.0, min_len_s: float = 0.06,
              lo_hz: float = 300, hi_hz: float = 6000, chunk_s: float = 30.0) -> list[ToneRun]:
    """Continuous narrow peaks that hold one FFT bin (+-1) for at least ``min_len_s``."""
    lo, hi = int(lo_hz * N_FFT / SAMPLE_RATE), int(hi_hz * N_FFT / SAMPLE_RATE)
    runs, open_runs = [], {}                 # bin -> start frame
    frame0 = 0
    step = int(chunk_s * SAMPLE_RATE)
    for offset in range(0, len(audio), step):
        part = audio[offset:offset + step + N_FFT]
        if len(part) < N_FFT:
            break
        prom = prominence(spectrogram(part))[:, lo:hi]
        n = min(len(prom), step // HOP)
        for i in range(n):
            row = prom[i]
            peaks = set(np.flatnonzero((row >= min_db) & (row >= np.roll(row, 1)) & (row >= np.roll(row, -1))) + lo)
            frame = frame0 + i
            for b in list(open_runs):
                if not ({b - 1, b, b + 1} & peaks):
                    start = open_runs.pop(b)
                    if (frame - start) * FRAME_S >= min_len_s:
                        runs.append(ToneRun(b, start * FRAME_S, frame * FRAME_S))
                else:
                    peaks -= {b - 1, b, b + 1}
            for b in peaks:
                open_runs[b] = frame
        frame0 += n
    for b, start in open_runs.items():
        if (frame0 - start) * FRAME_S >= min_len_s:
            runs.append(ToneRun(b, start * FRAME_S, frame0 * FRAME_S))
    return sorted(runs, key=lambda r: r.start)


def classify(runs: list[ToneRun], *, min_repeats: int = 4, steady_s: float = 0.6) -> list[ToneGroup]:
    groups = defaultdict(list)
    for r in runs:
        groups[r.bin // 2].append(r)          # merge neighbouring bins
    out = []
    for key, rs in groups.items():
        rs.sort(key=lambda r: r.start)
        # a steady tone broken up by a gunshot is still one tone: bridge gaps < 0.1 s
        merged = []
        for r in rs:
            if merged and r.start - merged[-1].end < .1:
                merged[-1] = ToneRun(merged[-1].bin, merged[-1].start, max(merged[-1].end, r.end))
            else:
                merged.append(ToneRun(r.bin, r.start, r.end))
        rs = merged
        total = sum(r.length for r in rs)
        freq = float(np.median([r.freq for r in rs]))
        long_runs = [r for r in rs if r.length >= steady_s]
        short = [r for r in rs if r.length < steady_s]
        period = None
        if len(short) >= min_repeats:
            gaps = np.diff([r.start for r in short])
            gaps = gaps[(gaps > .08) & (gaps < 1.5)]
            if len(gaps) >= min_repeats - 1:
                # beeps masked by noise leave gaps of 2x, 3x the period: test multiples
                period = float(np.percentile(gaps, 25))
                ratio = gaps / period
                regular = np.mean(np.abs(ratio - np.round(ratio)) < .2) > .7
                if not regular:
                    period = None
        kind = 'acquire' if period else 'locked' if long_runs else 'other'
        out.append(ToneGroup(freq, rs, kind, period, total))
    return sorted(out, key=lambda g: -g.total_s)


def best_segments(groups: list[ToneGroup], kind: str, *, pad_s: float = 0.3, max_s: float = 8.0):
    """Time windows (start, end) that contain the most of one kind of warning."""
    windows = []
    for g in groups:
        if g.kind != kind:
            continue
        rs = g.runs if kind == 'acquire' else [r for r in g.runs if r.length >= .6]
        cluster = []
        for r in rs:
            if cluster and r.start - cluster[-1].end > (2.0 if kind == 'acquire' else .5):
                windows.append(cluster)
                cluster = []
            cluster.append(r)
        if cluster:
            windows.append(cluster)
    scored = []
    for c in windows:
        start, end = c[0].start - pad_s, min(c[-1].end + pad_s, c[0].start - pad_s + max_s)
        on = sum(min(r.end, end) - max(r.start, start) for r in c if r.start < end)
        if kind == 'acquire' and len(c) < 4:
            continue
        scored.append((on, max(0.0, start), end))
    scored.sort(reverse=True)
    return [(s, e) for _, s, e in scored]


def quiet_segments(audio: np.ndarray, runs: list[ToneRun], *, length_s: float = 6.0, count: int = 3):
    """Stretches with sound but no warning tone at all (for the 'background' clip)."""
    busy = np.zeros(int(len(audio) / SAMPLE_RATE) + 2, bool)
    for r in runs:
        busy[int(r.start):int(r.end) + 1] = True
    out, t = [], 0
    while t + length_s < len(busy) and len(out) < count:
        if not busy[t:int(t + length_s) + 1].any():
            seg = audio[int(t * SAMPLE_RATE):int((t + length_s) * SAMPLE_RATE)]
            if np.sqrt(np.mean(seg ** 2)) > 1e-3:        # not silence
                out.append((float(t), float(t + length_s)))
                t += int(length_s)
                continue
        t += 1
    return out


def missile_candidates(audio: np.ndarray, locked_windows, *, before_s: float = 3.0, count: int = 3):
    """The seconds before a steady lock tone ends (impact or flare) are where the incoming
    missile is heard.  Ranked by how much the 2-8 kHz band rises over that time."""
    from .audio import band_energies, BAND_EDGES
    hi = (BAND_EDGES[:-1] >= 2000) & (BAND_EDGES[1:] <= 8000)
    scored = []
    for _, end in locked_windows:
        start = max(0.0, end - before_s)
        seg = audio[int(start * SAMPLE_RATE):int((end + .3) * SAMPLE_RATE)]
        if len(seg) < SAMPLE_RATE:
            continue
        bands = band_energies(spectrogram(seg))[:, hi].mean(1)
        third = max(1, len(bands) // 3)
        scored.append((float(bands[-third:].mean() - bands[:third].mean()), start, end + .3))
    scored.sort(reverse=True)
    return [(s, e, rise) for rise, s, e in scored[:count]]


def harvest(audio: np.ndarray):
    runs = tone_runs(audio)
    groups = classify(runs)
    return runs, groups, {k: best_segments(groups, k) for k in ('acquire', 'locked')}, quiet_segments(audio, runs)
