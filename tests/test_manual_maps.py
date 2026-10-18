"""Manual map selection pins both vehicle modes and discards old-map state."""
import pytest

from wardog_ad.assets import DEFAULT_ASSETS, load_map
from wardog_ad.config import Config
from wardog_ad.localization import Fix


@pytest.mark.skipif(not (DEFAULT_ASSETS / 'maps.json').exists(), reason='map assets not cloned')
@pytest.mark.parametrize('profile', ['truck', 'aircraft'])
def test_manual_map_selection_and_return_to_auto(profile, tmp_path):
    from wardog_ad.autopilot import Autopilot
    from wardog_ad.input_win import FakeKeyboard
    from wardog_ad.sim import Body, SimSensors, SimWorld, TruthLocator
    a = load_map('bakurani')
    cfg = Config(profile=profile, map_id=a.map_id,
                 destinations={'ozeti': [500, 600]}, homes={'ozeti': [300, 400]})
    cfg.aircraft.lock_source = 'none'
    kb = FakeKeyboard()
    world = SimWorld(cfg, a.load_image(), a.meters_per_pixel, Body(1381, 378, 140.), kb)
    pilot = Autopilot(cfg, a, kb, SimSensors(world), locator=TruthLocator(world))
    pilot.config_path = tmp_path / 'config.json'
    pilot.goal, pilot.engaged, pilot.mission = (1400, 400), True, 'outbound'
    pilot._fix_latest = Fix(1381, 378, valid=True)
    kb.press('W')
    epoch = pilot._map_generation
    pilot.set_map_mode('ozeti')
    assert pilot.assets.map_id == cfg.map_id == 'ozeti'
    assert not cfg.auto_map.enabled and pilot.map_manager is None
    assert not pilot.engaged and not kb.down
    assert pilot.goal is None and pilot.mission == 'idle'
    assert pilot.destination == (500, 600) and pilot.home == (300, 400)
    assert pilot.telemetry.state is None and pilot._fix_latest is None
    assert pilot._map_generation > epoch
    saved = Config.load(pilot.config_path)
    assert saved.map_id == 'ozeti' and not saved.auto_map.enabled
    pilot.last_hud = 100.
    pilot._air_map_watch(Fix(reason='no match'), 100.)
    pilot._air_map_watch(Fix(reason='no match'), 120.)
    assert pilot.assets.map_id == 'ozeti'
    pilot.set_map_mode(None)
    assert cfg.auto_map.enabled and pilot.map_manager is not None
    assert not pilot.map_manager.ready and not pilot.engaged
    assert Config.load(pilot.config_path).auto_map.enabled


@pytest.mark.skipif(not (DEFAULT_ASSETS / 'maps.json').exists(), reason='map assets not cloned')
def test_unavailable_manual_map_preserves_current_map(tmp_path):
    from wardog_ad.autopilot import Autopilot
    from wardog_ad.input_win import FakeKeyboard
    from wardog_ad.sim import Body, SimSensors, SimWorld, TruthLocator
    a = load_map('bakurani')
    cfg = Config(map_id=a.map_id)
    kb = FakeKeyboard()
    world = SimWorld(cfg, a.load_image(), a.meters_per_pixel, Body(1381, 378, 140.), kb)
    pilot = Autopilot(cfg, a, kb, SimSensors(world), locator=TruthLocator(world))
    pilot.engaged = True
    kb.press('W')
    with pytest.raises(ValueError, match='unknown map'):
        pilot.set_map_mode('missing')
    assert pilot.assets is a and cfg.map_id == a.map_id
    assert cfg.auto_map.enabled and not pilot.engaged and not kb.down
