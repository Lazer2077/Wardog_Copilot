"""Map switch in the helicopter, sub-pixel refinement, missile-sound flares."""
import logging

import numpy as np
import pytest

from wardog_ad.assets import DEFAULT_ASSETS
from wardog_ad.config import Config

needs_maps = pytest.mark.skipif(not (DEFAULT_ASSETS / 'maps.json').exists(), reason='map assets not cloned')


@needs_maps
def test_helicopter_switches_to_the_map_of_a_new_round():
    """The 23:15 session flew another map for minutes while the program kept matching BAKURANI."""
    from wardog_ad.assets import load_map
    from wardog_ad.autopilot import Autopilot
    from wardog_ad.input_win import FakeKeyboard
    from wardog_ad.localization import MinimapLocalizer
    from wardog_ad.map_manager import MapManager
    from wardog_ad.sim import Body, SimSensors, SimWorld
    logging.disable(logging.WARNING)
    ozeti, bakurani = load_map('ozeti'), load_map('bakurani')
    loc_o, loc_b = MinimapLocalizer.from_assets(ozeti), MinimapLocalizer.from_assets(bakurani)
    cfg = Config(profile='aircraft', map_id='ozeti')
    cfg.aircraft.lock_source = 'none'
    kb = FakeKeyboard()
    # the game is actually on BAKURANI
    w = SimWorld(cfg, bakurani.load_image(), bakurani.meters_per_pixel, Body(1380, 378, 90.0), kb)
    pilot = Autopilot(cfg, ozeti, kb, SimSensors(w), locator=loc_o)
    pilot.map_manager = MapManager(ozeti, loc_o, cfg.auto_map, candidates=[(bakurani, loc_b)])
    t = 0.0
    while t < 30 and cfg.map_id == 'ozeti':
        pilot.last_hud = t                       # in the helicopter, HUD on screen
        pilot._air_map_watch(pilot._locate(), t)
        t += .5
    logging.disable(logging.NOTSET)
    assert cfg.map_id == 'bakurani' and pilot.assets is bakurani
    fix = pilot._locate()
    assert fix.valid and np.hypot(fix.x - 1380, fix.y - 378) < 2      # localized on the new map


@needs_maps
def test_no_map_search_outside_the_game():
    from wardog_ad.assets import load_map
    from wardog_ad.autopilot import Autopilot
    from wardog_ad.input_win import FakeKeyboard
    from wardog_ad.localization import Fix
    from wardog_ad.sim import Body, SimSensors, SimWorld, TruthLocator

    class Never:
        def update(self, *a):
            raise AssertionError('must not search maps while the HUD is not on screen')

    ozeti = load_map('ozeti')
    cfg = Config(profile='aircraft')
    cfg.aircraft.lock_source = 'none'
    w = SimWorld(cfg, ozeti.load_image(), ozeti.meters_per_pixel, Body(500, 500, 0), FakeKeyboard())
    pilot = Autopilot(cfg, ozeti, FakeKeyboard(), SimSensors(w), locator=TruthLocator(w))
    pilot.map_manager = Never()
    pilot.map_manager.t = cfg.auto_map
    pilot.last_hud = -100.0                      # lobby / loading screen
    for t in range(20):
        pilot._air_map_watch(Fix(reason='too few matches'), float(t))


@needs_maps
def test_refinement_halves_the_fix_error():
    import cv2
    from wardog_ad.assets import load_map
    from wardog_ad.input_win import FakeKeyboard
    from wardog_ad.localization import MinimapLocalizer
    from wardog_ad.refine import FixRefiner
    from wardog_ad.sim import Body, SimWorld
    assets = load_map('bakurani')
    base = assets.load_image()
    loc, ref = MinimapLocalizer.from_assets(assets), FixRefiner(base)
    w = SimWorld(Config(), base, assets.meters_per_pixel, Body(1380.3, 378.7, 37), FakeKeyboard())
    w.minimap_zoom = 2.1
    rng = np.random.default_rng(1)
    before, after = [], []
    for _ in range(12):
        w.body.x += rng.uniform(-2, 2)
        w.body.y += rng.uniform(-2, 2)
        img = cv2.resize(w.render_minimap(), None, fx=.85, fy=.85, interpolation=cv2.INTER_AREA)
        fix = loc.locate(img)
        better = ref.refine(img, fix)
        before.append(np.hypot(fix.x - w.body.x, fix.y - w.body.y))
        after.append(np.hypot(better.x - w.body.x, better.y - w.body.y))
    assert np.median(after) < .7 * np.median(before)


def _detector_and_flares(flare_on='missile'):
    from wardog_ad.audio import AudioLockDetector
    from wardog_ad.perception import FlareController
    from wardog_ad.sim import sim_lock_profile
    profile = sim_lock_profile()
    t = Config().aircraft
    t.flare_on, t.flare_burst = flare_on, 1
    shots = []
    flares = FlareController(t, lambda: shots.append(1))
    det = AudioLockDetector(profile)
    flares.can_hear_missile = det.has_missile
    return det, flares, shots


def _play(det, flares, audio, sr=48000):
    states = []
    for i in range(0, len(audio), 1024):
        now = i / sr
        state = det.feed(audio[i:i + 1024], now)
        flares.update_state(state, now, det.missile)
        states.append((now, state, det.missile))
    return states


def test_flares_wait_for_the_missile_after_the_warning():
    from wardog_ad.sim import synth_battle_audio, synth_lock_tone, synth_missile_sound
    sr = 48000
    det, flares, shots = _detector_and_flares()
    audio = synth_battle_audio(14, seed=21)
    audio[2 * sr:4 * sr] += synth_lock_tone(2, 'acquire', level=.03)
    audio[4 * sr:5 * sr] += synth_lock_tone(1, 'locked', level=.03)     # lock (short) ...
    audio[8 * sr:10 * sr] += synth_missile_sound(2, level=.08)          # ... missile heard later
    states = _play(det, flares, audio)
    heard = [t for t, _, m in states if m]
    assert heard and 8 <= heard[0] < 8.8                                # missile recognised quickly
    assert flares.fired >= 1 and not [t for t, _, m in states if m and t < 8]


def test_no_flares_for_missile_like_noise_without_a_warning():
    from wardog_ad.sim import synth_battle_audio, synth_missile_sound
    sr = 48000
    det, flares, _ = _detector_and_flares()
    audio = synth_battle_audio(10, seed=22)
    audio[3 * sr:6 * sr] += synth_missile_sound(3, level=.08)           # someone else's missile
    _play(det, flares, audio)
    assert flares.fired == 0


def test_no_false_missile_in_pure_combat_noise():
    from wardog_ad.sim import synth_battle_audio
    det, flares, _ = _detector_and_flares()
    states = _play(det, flares, synth_battle_audio(40, seed=23))
    assert not any(m for _, _, m in states)
