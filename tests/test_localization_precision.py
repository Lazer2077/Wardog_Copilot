"""Precision regressions: rotated registration, pipeline parity and delayed fixes."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from wardog_ad.airnav import AirNavigator
from wardog_ad.assets import DEFAULT_ASSETS, load_map
from wardog_ad.config import Config, Region
from wardog_ad.localization import Fix
from wardog_ad.refine import FixRefiner


def test_vehicle_calibrations_are_independent_and_keep_legacy_fallback(tmp_path):
    cfg = Config()
    cfg.minimap = Region(10, 20, 300, 280)
    cfg.anchor = (.49, .51)
    assert cfg.localization_region == cfg.minimap
    assert cfg.localization_anchor == cfg.anchor
    cfg.truck_minimap.region = Region(30, 40, 320, 300)
    cfg.truck_minimap.anchor = (.47, .53)
    cfg.aircraft_minimap.region = Region(50, 60, 350, 310)
    cfg.aircraft_minimap.anchor = (.51, .48)
    path = tmp_path / 'config.json'
    cfg.save(path)
    loaded = Config.load(path)
    assert loaded.localization_region == cfg.truck_minimap.region
    assert loaded.localization_anchor == (.47, .53)
    loaded.profile = 'aircraft'
    assert loaded.localization_region == cfg.aircraft_minimap.region
    assert loaded.localization_anchor == (.51, .48)


def test_capture_switches_minimap_with_vehicle_profile():
    from wardog_ad.autopilot import ScreenSensors
    cfg = Config()
    cfg.truck_minimap.region = Region(30, 40, 320, 300)
    cfg.aircraft_minimap.region = Region(50, 60, 350, 310)
    sensors = ScreenSensors(cfg)
    sensors._cap = lambda: SimpleNamespace(grab=lambda region: region)
    assert sensors.minimap() == cfg.truck_minimap.region
    cfg.profile = 'aircraft'
    assert sensors.minimap() == cfg.aircraft_minimap.region


@pytest.mark.parametrize('rotation', [0., 2.5, -2.5])
def test_refinement_handles_rotation_and_noncentral_player(rotation):
    rng = np.random.default_rng(13)
    base = cv2.GaussianBlur(rng.integers(30, 180, (1000, 1000), dtype=np.uint8), (3, 3), .7)
    ref = FixRefiner(base)
    anchor = (.43, .54)
    image = cv2.cvtColor(ref.render(500, 500, .8, (300, 280), anchor, rotation), cv2.COLOR_GRAY2BGR)
    fix = Fix(500.6, 499.6, scale=.8, rotation=rotation, valid=True)
    result = ref.refine(image, fix, anchor)
    assert np.hypot(result.x - 500, result.y - 500) < .25
    assert np.hypot(fix.x - 500, fix.y - 500) > .7


def test_blank_terrain_cannot_create_a_refinement():
    image = np.full((100, 100, 3), 60, np.uint8)
    ref = FixRefiner(np.full((300, 300), 60, np.uint8))
    fix = Fix(150, 150, scale=1., valid=True)
    assert ref.refine(image, fix) is fix


def camera_fix(x, y, timestamp):
    return Fix(x=x, y=y, confidence=.9, inliers=40, valid=True, timestamp=timestamp)


def test_delayed_camera_fix_does_not_pull_helicopter_backwards():
    nav = AirNavigator(1.)
    nav.start((100., 100.), 0.)
    for t in np.arange(.1, 1.01, .1):
        nav.predict(float(t), 180., 90.)  # 50 m/s east
    assert nav.pos[0] == pytest.approx(150.)
    assert nav.correct(camera_fix(125., 100., .5))  # half-second-old screenshot
    assert nav.pos == pytest.approx([150., 100.])
    # The raw camera track must retain its real observation time and position.
    assert nav.track[-1][0] == .5
    assert nav.track[-1][1] == pytest.approx([125., 100.])


def test_delayed_camera_fix_uses_the_turn_history():
    nav = AirNavigator(1.)
    nav.start((100., 100.), 0.)
    nav.predict(.5, 72., 90.)  # east 10 m
    nav.predict(1., 72., 0.)   # north 10 m
    assert nav.correct(camera_fix(105., 100., .25))
    assert nav.pos == pytest.approx([110., 90.])
    for t in np.arange(1.1, 5.01, .1):
        nav.predict(float(t), 0., 0.)
    assert not nav.correct(camera_fix(105., 100., .25))


def test_reverse_flight_increases_uncertainty_and_distance():
    nav = AirNavigator(1.)
    nav.start((100., 100.), 0.)
    nav.predict(.1, 3., 90., pitch=-.5)
    initial = nav.sigma_m
    nav.predict(1., 72., 90.)
    assert nav.pos[0] < 100.
    assert nav.sigma_m > initial and nav.travelled_m > 0


@pytest.mark.skipif(not (DEFAULT_ASSETS / 'maps.json').exists(), reason='map assets not cloned')
def test_truck_and_recorder_both_apply_precision_refinement():
    from wardog_ad.autopilot import Autopilot
    from wardog_ad.driving_data import NavigationObserver
    from wardog_ad.input_win import FakeKeyboard
    from wardog_ad.localization import MinimapLocalizer
    from wardog_ad.sim import Body, SimSensors, SimWorld

    cfg = Config(map_id='bakurani')
    cfg.auto_map.enabled = False
    assets = load_map('bakurani')
    kb = FakeKeyboard()
    world = SimWorld(cfg, assets.load_image(), assets.meters_per_pixel, Body(1380, 378, 37), kb)
    pilot = Autopilot(cfg, assets, kb, SimSensors(world), locator=MinimapLocalizer.from_assets(assets))
    calls = []
    def refine(image, fix, anchor):
        calls.append(fix.position)
        return fix
    stub = SimpleNamespace(refine=refine)
    pilot._refiner = lambda: stub
    pilot.tick(0.)
    assert len(calls) == 1
    nav = NavigationObserver(cfg, assets)
    nav._refine, nav._refine_map = stub, assets.map_id
    image = world.render_minimap()
    cfg.minimap = Region(0, 0, image.shape[1], image.shape[0])
    nav.observe(image, cfg.minimap, 'first_person', 0.)
    assert len(calls) == 2
