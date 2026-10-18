"""Map identity, ambiguity, recovery and map-coordinate isolation."""
from types import SimpleNamespace

import numpy as np
import pytest

from wardog_ad.config import Config, MapTuning
from wardog_ad.localization import Fix
from wardog_ad.map_manager import MapManager


def good(x=10, scale=1):
    return Fix(x=x, y=10, heading=90, confidence=.9, inliers=30, error=.3,
               scale=scale, valid=True)


class Locator:
    def __init__(self, fix=None):
        self.fix = fix or Fix()

    def reset(self):
        pass

    def locate(self, *args):
        return self.fix


def asset(name, mpp=1):
    return SimpleNamespace(map_id=name, name=name, meters_per_pixel=mpp, roads=[])


def manager():
    a, b = asset('a'), asset('b', 2)
    la, lb = Locator(good()), Locator()
    m = MapManager(a, la, MapTuning(scan_interval_s=1), candidates=[(b, lb)])
    return m, la, lb


IMAGE = np.zeros((80, 80, 3), np.uint8)


def test_startup_needs_distinct_confirmation_scans():
    m, _, _ = manager()
    assert not m.update(IMAGE, 0).ready
    assert not m.update(IMAGE, .1).ready
    assert not m.update(IMAGE, 1).ready
    result = m.update(IMAGE, 2)
    assert result.ready and result.map_id == 'a'


def test_map_transition_and_loss_never_reuse_old_identity():
    m, la, lb = manager()
    for now in range(3):
        m.update(IMAGE, now)
    la.fix, lb.fix = Fix(), good()
    assert not m.update(IMAGE, 2.1).ready
    assert not m.update(IMAGE, 3.1).ready
    result = m.update(IMAGE, 4.1)
    assert result.ready and result.map_id == 'b'
    assert not m.update(None, 4.2).ready
    assert not m.update(IMAGE, 5.1).ready


def test_ambiguous_and_weak_matches_are_rejected():
    m, la, lb = manager()
    lb.fix = good()
    assert not m.update(IMAGE, 0).ready
    lb.fix = Fix()
    la.fix = good()
    la.fix.inliers = 8
    assert not m.update(IMAGE, 1).ready


def test_candidate_position_or_scale_jump_restarts_confirmation():
    m, la, _ = manager()
    m.update(IMAGE, 0)
    la.fix = good(x=1000)
    assert not m.update(IMAGE, 1).ready
    assert m.count == 1
    la.fix = good(x=1000, scale=2)
    assert not m.update(IMAGE, 2).ready
    assert m.count == 1


def test_map_change_clears_mission_and_loads_per_map_points():
    from wardog_ad.autopilot import Autopilot
    from wardog_ad.input_win import FakeKeyboard
    a, b = asset('a'), asset('b', 2)
    cfg = Config(map_id='a', destinations={'b': [20, 30]}, homes={'b': [40, 50]})
    sensors = SimpleNamespace(minimap=lambda: IMAGE, main_view=lambda: None)
    pilot = Autopilot(cfg, a, FakeKeyboard(), sensors, locator=Locator(), graph=object())
    pilot.goal, pilot.home, pilot.destination = (1, 2), (3, 4), (5, 6)
    pilot.mission, pilot.engaged = 'outbound', True
    pilot.via, pilot.avoid, pilot.telemetry.route = [(9, 9)], [object()], [(1, 2)]
    pilot.keyboard.press('W')
    pilot._switch_map(b, Locator(good()))
    assert cfg.map_id == 'b' and pilot.mpp == 2
    assert pilot.destination == (20, 30) and pilot.home == (40, 50)
    assert pilot.goal is None and pilot.controller is None and pilot.mission == 'idle'
    assert not pilot.engaged and not pilot.keyboard.down
    assert not pilot.via and not pilot.avoid and not pilot.telemetry.route
    assert pilot.telemetry.state is None


def test_unconfirmed_map_blocks_engagement_and_tick_releases_keys():
    from wardog_ad.autopilot import Autopilot
    from wardog_ad.input_win import FakeKeyboard
    m, _, _ = manager()
    cfg = Config(map_id='a', destinations={'a': [20, 30]})
    sensors = SimpleNamespace(minimap=lambda: IMAGE, main_view=lambda: None)
    pilot = Autopilot(cfg, m.assets['a'], FakeKeyboard(), sensors,
                      locator=m.locators['a'], graph=object())
    pilot.map_manager = m
    assert not pilot.go()
    pilot.engaged = True  # the fast actuator must gate independently of the slow scan
    pilot.actuator.command(throttle=1)
    pilot.actuate(0)
    assert not pilot.keyboard.down
    pilot.keyboard.press('W')
    tel = pilot.tick(0)
    assert tel.status == 'map search' and tel.state is None
    assert not pilot.engaged and not pilot.keyboard.down


def test_map_and_camera_settings_roundtrip(tmp_path):
    cfg = Config()
    cfg.auto_map.confirm_scans = 4
    cfg.truck.camera_view = 'third_person'
    path = tmp_path / 'config.json'
    cfg.save(path)
    loaded = Config.load(path)
    assert loaded.auto_map.confirm_scans == 4
    assert loaded.truck.camera_view == 'third_person'


def test_tick_switches_map_without_resuming_old_mission():
    from wardog_ad.autopilot import Autopilot
    from wardog_ad.input_win import FakeKeyboard
    m, la, lb = manager()
    cfg = Config(map_id='a', destinations={'b': [20, 30]})
    sensors = SimpleNamespace(minimap=lambda: IMAGE, main_view=lambda: None)
    pilot = Autopilot(cfg, m.assets['a'], FakeKeyboard(), sensors, locator=la, graph=object())
    pilot.map_manager = m
    for now in range(3):
        pilot.tick(now)
    pilot.goal, pilot.engaged, pilot.mission = (99, 99), True, 'return'
    pilot.keyboard.press('W')
    la.fix, lb.fix = Fix(), good()
    assert pilot.tick(2.1).status == 'map search'
    assert not pilot.engaged and not pilot.keyboard.down
    pilot.tick(3.1)
    tel = pilot.tick(4.1)
    assert cfg.map_id == 'b' and pilot.destination == (20, 30)
    # The new map's saved destination may populate the idle route preview;
    # this must never resume the old return mission or enable key output.
    assert pilot.goal == (20, 30) and pilot.mission == 'idle' and not pilot.engaged
    assert not pilot.keyboard.down
    assert tel is pilot.telemetry and tel.map_status == 'map confirmed: b'


def test_real_map_minimaps_select_and_switch():
    from wardog_ad.assets import DEFAULT_ASSETS, load_map
    from wardog_ad.input_win import FakeKeyboard
    from wardog_ad.localization import MinimapLocalizer
    from wardog_ad.sim import Body, SimWorld
    if not (DEFAULT_ASSETS / 'maps.json').exists():
        pytest.skip('map assets not cloned')
    initial = load_map('ozeti')
    m = MapManager(initial, MinimapLocalizer.from_assets(initial),
                   MapTuning(scan_interval_s=.1, confirm_scans=2))
    now = 0
    for name, position in [('ozeti', (505, 1245)), ('bakurani', (1381, 378)),
                           ('zestafona', (1024, 900))]:
        assets = load_map(name)
        w = SimWorld(Config(), assets.load_image(), assets.meters_per_pixel,
                     Body(*position, 37), FakeKeyboard())
        image = w.render_minimap()
        for _ in range(3):
            result = m.update(image, now)
            now += .2
        assert result.ready and result.map_id == name, result.reason
        assert np.hypot(result.fix.x - position[0], result.fix.y - position[1]) < 2
