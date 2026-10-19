"""Manual map selection pins both vehicle modes and discards old-map state."""
import pytest

from wardog_ad.assets import DEFAULT_ASSETS, load_map
from wardog_ad.config import Config
from wardog_ad.localization import Fix


def test_localizer_construction_never_reconfigures_live_opencv_threads(monkeypatch):
    import cv2
    import numpy as np
    from wardog_ad.localization import MinimapLocalizer
    def unsafe(_):
        raise AssertionError('runtime map loading must not change global OpenCV threading')
    monkeypatch.setattr(cv2,'setNumThreads',unsafe)
    rng=np.random.default_rng(17)
    for _ in range(2):
        MinimapLocalizer(rng.uniform(0,100,(20,2)).astype('float32'),
                         rng.uniform(0,255,(20,128)).astype('float32'),(100,100))


@pytest.mark.skipif(not (DEFAULT_ASSETS / 'maps.json').exists(), reason='map assets not cloned')
def test_manual_switch_with_background_localization_keeps_worker_alive(caplog):
    import threading
    from wardog_ad.autopilot import Autopilot
    from wardog_ad.input_win import FakeKeyboard
    from wardog_ad.localization import MinimapLocalizer
    from wardog_ad.sim import Body,SimSensors,SimWorld
    assets=load_map('ozeti')
    cfg=Config(profile='aircraft',map_id='ozeti',refine_fixes=False)
    cfg.aircraft.lock_source='none'
    keyboard=FakeKeyboard()
    world=SimWorld(cfg,assets.load_image(),assets.meters_per_pixel,Body(505,1245,37),keyboard)
    p=Autopilot(cfg,assets,keyboard,SimSensors(world),locator=MinimapLocalizer.from_assets(assets))
    stop=threading.Event()
    p.start_fix_worker(stop)
    try:
        for name in ('bakurani','ozeti'):
            p.set_map_mode(name)
            assert p.assets.map_id==name and p.fix_worker.is_alive()
            assert not p.engaged and not keyboard.down
    finally:
        stop.set()
        p.fix_worker.join(5)
    assert not p.fix_worker.is_alive()
    assert 'minimap matching failed' not in caplog.text


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
