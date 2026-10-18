"""Audio lock-warning detection on synthetic cockpit noise (no game or sound card needed)."""
import numpy as np
import pytest

from wardog_ad.audio import SAMPLE_RATE as SR, AudioLockDetector, LockProfile, fit_profile, read_wav, write_wav
from wardog_ad.config import Config
from wardog_ad.perception import FlareController
from wardog_ad.sim import synth_battle_audio, synth_lock_tone


@pytest.fixture(scope='module')
def profile():
    bg = synth_battle_audio(20, seed=1)
    return fit_profile({'acquire': synth_battle_audio(6, seed=2) + synth_lock_tone(6, 'acquire'),
                        'locked': synth_battle_audio(6, seed=3) + synth_lock_tone(6, 'locked')}, bg)


def run(profile, scene):
    det, states = AudioLockDetector(profile), []
    for i in range(0, len(scene), 1024):
        now = i / SR
        states.append((now, det.feed(scene[i:i + 1024], now)))
    return states


def test_fit_finds_tone_frequencies(profile):
    acq, lck = profile.signatures['acquire'], profile.signatures['locked']
    assert any(abs(f - 1000) < 30 for f in acq.freqs)
    assert any(abs(f - 1500) < 30 for f in lck.freqs)
    assert acq.period_s is not None and lck.on_ratio > .8


@pytest.mark.parametrize('level', [0.03, 0.015])
@pytest.mark.parametrize('seed', [11, 12])
def test_states_through_a_lock_sequence(profile, level, seed):
    scene = synth_battle_audio(40, seed=seed)
    scene[8 * SR:12 * SR] += synth_lock_tone(4, 'acquire', level=level)
    scene[12 * SR:16 * SR] += synth_lock_tone(4, 'locked', level=level)
    states = run(profile, scene)
    acquiring = [t for t, s in states if s == 'acquiring' and 8 <= t < 12.5]
    locked = [t for t, s in states if s == 'locked']
    assert acquiring and acquiring[0] < 9.5
    assert locked and 12 < locked[0] < 13.2 and locked[-1] < 17
    # no locked state while only beeping, and nothing at all in pure combat noise
    assert not [t for t, s in states if s == 'locked' and t < 12]
    assert not [t for t, s in states if s != 'none' and (t < 8 or t > 17)]


def test_profile_and_wav_roundtrip(profile, tmp_path):
    profile.save(tmp_path / 'p.json')
    loaded = LockProfile.load(tmp_path / 'p.json')
    assert loaded.signatures['locked'].bins == profile.signatures['locked'].bins
    audio = synth_lock_tone(0.5, 'locked')
    write_wav(tmp_path / 'a.wav', audio)
    back, sr = read_wav(tmp_path / 'a.wav')
    assert sr == SR and np.allclose(back, audio, atol=1e-4)


def test_cli_fit_from_recorded_clips(tmp_path, monkeypatch):
    from wardog_ad.__main__ import main
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'audio_clips').mkdir()
    write_wav(tmp_path / 'audio_clips' / 'background_00.wav', synth_battle_audio(15, seed=4))
    write_wav(tmp_path / 'audio_clips' / 'locked_00.wav', synth_battle_audio(5, seed=6) + synth_lock_tone(5, 'locked'))
    main(['--config', str(tmp_path / 'c.json'), 'audio-fit', '--output', 'lock.json'])
    assert Config.load(tmp_path / 'c.json').audio_profile.endswith('lock.json')
    only_locked = LockProfile.load(tmp_path / 'lock.json')
    assert list(only_locked.signatures) == ['locked']


def test_flare_policy_waits_for_lock_and_delay():
    t = Config().aircraft
    shots = []
    fc = FlareController(t, lambda: shots.append(now))
    now = 0.0
    for state, until in (('acquiring', 2.0), ('locked', 3.5), ('none', 4.0)):
        while now < until:
            fc.update_state(state, now)
            now = round(now + 0.05, 3)
    assert shots and min(shots) >= 2.0 + t.flare_delay_s          # nothing while only beeping
    assert len(shots) == t.flare_burst
    t2 = Config().aircraft
    t2.flare_on, t2.flare_capacity = 'acquire', 1
    shots.clear()
    fc = FlareController(t2, lambda: shots.append(now))
    now = 0.0
    while now < 5:
        fc.update_state('acquiring', now)
        now = round(now + 0.05, 3)
    assert len(shots) == 1                                          # capacity respected
