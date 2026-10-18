"""Terrain safety (real crash 2026-10-17 22:32: climbing 35 m above a mountainside into a
ridge at 140 km/h), terrain memory and heading without the compass."""
import logging
import math

import numpy as np
import pytest

from wardog_ad.airnav import AirNavigator
from wardog_ad.assets import DEFAULT_ASSETS
from wardog_ad.config import Config
from wardog_ad.localization import Fix
from wardog_ad.terrain import TerrainMemory

needs_maps = pytest.mark.skipif(not (DEFAULT_ASSETS / 'maps.json').exists(), reason='map assets not cloned')


def _fly(start, goal, ridge, memory):
    from wardog_ad.assets import load_map
    from wardog_ad.autopilot import Autopilot
    from wardog_ad.input_win import FakeKeyboard
    from wardog_ad.sim import Body, SimSensors, SimWorld, TruthLocator
    assets = load_map('ozeti')
    cfg = Config(profile='aircraft')
    cfg.aircraft.lock_source = 'none'
    cfg.destinations = {'ozeti': list(goal)}
    kb = FakeKeyboard()
    w = SimWorld(cfg, assets.load_image(), assets.meters_per_pixel, Body(*start, 90.0), kb, ridge=ridge)
    pilot = Autopilot(cfg, assets, kb, SimSensors(w), locator=TruthLocator(w))
    pilot._terrain, pilot._terrain_map = memory, 'ozeti'
    assert pilot.go()
    lowest, next_slow = math.inf, 0.0
    while w.time < 600 and not (pilot.arrived and not pilot.engaged) and not w.crashes:
        if w.time >= next_slow:
            status = pilot.tick(w.time).status
            next_slow += 1 / cfg.loop_hz
            if w.body.x > ridge[0] and status.split()[0] not in ('ground', 'j-hook', 'hover', 'descend', 'flare', 'settle', 'landed'):
                lowest = min(lowest, w.body.alt)
        pilot.fast_control(w.time)
        pilot.actuate(w.time)
        w.step(1 / 30)
    return pilot, w, lowest


@needs_maps
@pytest.mark.parametrize('slope', [0.3, 0.45, 0.6])
def test_takeoff_below_a_mountainside_keeps_clearance(slope):
    """Replay of the crash: still low when the ground starts rising steeply."""
    logging.disable(logging.WARNING)
    pilot, w, lowest = _fly((380, 700), (1450, 700), (400, slope, 400), TerrainMemory(None))
    logging.disable(logging.NOTSET)
    assert pilot.arrived and w.crashes == 0 and lowest > 20


@needs_maps
def test_terrain_memory_makes_the_second_flight_safe():
    logging.disable(logging.WARNING)
    memory = TerrainMemory(None)
    _fly((150, 700), (1450, 700), (400, 0.6, 400), memory)    # first pass at full speed: no warning
    assert len(memory.heights) > 20
    pilot, w, lowest = _fly((150, 700), (1450, 700), (400, 0.6, 400), memory)
    logging.disable(logging.NOTSET)
    assert pilot.arrived and w.crashes == 0 and lowest > 20


def test_terrain_memory_round_trip(tmp_path):
    mem = TerrainMemory(tmp_path / 'm.npz')
    mem.record((100, 100), 250.0)
    mem.record((100, 100), 240.0)           # keeps the highest
    mem.record((180, 100), 400.0)
    mem.save()
    again = TerrainMemory(tmp_path / 'm.npz')
    h, d = again.highest_ahead((50, 100), (300, 100), 200)
    assert h == 400.0 and 120 < d < 140
    assert again.highest_ahead((50, 100), (50, 0), 50) is None   # nothing known that way


def test_heading_from_track_when_the_compass_fails():
    nav = AirNavigator(5.0)
    true = np.array([500.0, 500.0])
    t = 0.0
    nav.predict(t, 0, 90.0)                  # one compass reading, then none
    nav.correct(Fix(*true, None, .8, 40, .5, .5, 0, True, 'ok', t))
    for _ in range(40):                      # flying north-east (45 deg) at 180 km/h, compass unreadable
        t += .1
        r = math.radians(45.0)
        true += np.array([math.sin(r), -math.cos(r)]) * 50 * .1 / 5.0
        nav.predict(t, 180, None)
        nav.correct(Fix(*true, None, .8, 40, .5, .5, 0, True, 'ok', t))
    assert nav.heading_source == 'track'
    assert abs((nav.heading - 45 + 180) % 360 - 180) < 5
