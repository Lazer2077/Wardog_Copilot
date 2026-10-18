"""Cutting lock-warning clips out of a long recording (e.g. a YouTube video)."""
import numpy as np
import pytest

from wardog_ad.harvest import harvest, missile_candidates
from wardog_ad.sim import synth_battle_audio, synth_lock_tone, synth_missile_sound

SR = 48000


def soundtrack(seed):
    """Combat noise + a narrator with drifting pitch + one beeping and one steady lock."""
    a = synth_battle_audio(60, seed=seed) * .7
    t = np.arange(len(a)) / SR
    f0 = 140 + 30 * np.sin(2 * np.pi * .7 * t) + 15 * np.sin(2 * np.pi * 3.1 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    talking = (np.sin(2 * np.pi * 4 * t) > 0) * (np.sin(2 * np.pi * .23 * t) > -.3)
    a += (sum(.03 / k * np.sin(k * phase) for k in range(1, 12)) * talking).astype(np.float32)
    a[10 * SR:15 * SR] += synth_lock_tone(5, 'acquire', level=.03)
    a[30 * SR:33 * SR] += synth_lock_tone(3, 'locked', level=.03)
    a[31 * SR:33 * SR] += synth_missile_sound(2, level=.06)
    return a


@pytest.mark.parametrize('seed', [7, 8])
def test_finds_beeping_and_steady_tones_but_not_the_voice(seed):
    audio = soundtrack(seed)
    runs, groups, segs, quiet = harvest(audio)
    kinds = {}
    for g in groups:                                   # strongest first; harmonics come later
        kinds.setdefault(g.kind, g)
    assert abs(kinds['acquire'].freq - 1000) < 30 and abs(kinds['acquire'].period_s - .3) < .03
    assert abs(kinds['locked'].freq - 1500) < 30
    (a0, a1), (l0, l1) = segs['acquire'][0], segs['locked'][0]
    assert 9 < a0 < 10.5 and 14.5 < a1 < 16
    assert 29 < l0 < 30.5 and 32.5 < l1 < 34
    for s, e in quiet:                                  # background never overlaps a warning
        assert e <= 10 or 15 <= s <= 30 and e <= 30 or s >= 33
    assert all(g.freq > 250 for g in groups)           # the narrator's low voice is not a tone


def test_missile_candidate_is_the_end_of_the_lock_tone():
    audio = soundtrack(7)
    _, _, segs, _ = harvest(audio)
    (s, e, rise), *_ = missile_candidates(audio, segs['locked'])
    assert 29.5 < s < 31.5 and 33 < e < 34.5 and rise > 1
