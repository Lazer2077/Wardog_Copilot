"""HQ references are approximate user seeds, never synthetic vision observations."""
import numpy as np
import pytest

from wardog_ad.assets import DEFAULT_ASSETS, load_map
from wardog_ad.bases import team_bases
from wardog_ad.config import Config

pytestmark = pytest.mark.skipif(not (DEFAULT_ASSETS / 'maps.json').exists(), reason='map assets not cloned')


@pytest.mark.parametrize('map_id', ['ozeti', 'bakurani', 'zestafona'])
def test_each_map_has_three_distinct_in_bounds_hq_references(map_id):
    assets = load_map(map_id)
    bases = team_bases(assets, Config())
    assert {b.id for b in bases} == {'valkyra', 'manticore', 'lonestar'}
    assert len({b.point for b in bases}) == 3
    for base in bases:
        assert assets.contains(base.point)
        assert base.sigma_m == 200 and not base.custom
        assert 'wardogs-calculator' in base.source
    if map_id == 'ozeti':
        # The upstream faction icon is misplaced; use its actual deployment board.
        assert assets.map_to_game(bases[0].point) == pytest.approx((137.88, 67.26))
    if map_id == 'bakurani':
        assert assets.map_to_game(bases[2].point) == pytest.approx((66.8, 35.24))


def pilot_at_base(profile):
    from wardog_ad.autopilot import Autopilot
    from wardog_ad.input_win import FakeKeyboard
    from wardog_ad.sim import Body, SimSensors, SimWorld, TruthLocator
    assets = load_map('ozeti')
    cfg = Config(profile=profile)
    cfg.aircraft.lock_source = 'none'
    base = team_bases(assets, cfg)[0]
    kb = FakeKeyboard()
    world = SimWorld(cfg, assets.load_image(), assets.meters_per_pixel, Body(*base.point, 90.), kb)
    pilot = Autopilot(cfg, assets, kb, SimSensors(world), locator=TruthLocator(world, noise_px=0., heading_noise=0.))
    return pilot, world, base


@pytest.mark.parametrize('profile', ['truck', 'aircraft'])
def test_manual_initialization_and_visual_reacquisition(profile):
    pilot, world, base = pilot_at_base(profile)
    pilot.engaged, pilot.goal = True, (600., 800.)
    pilot.keyboard.press('W')
    pilot.set_initial_position(base.point, source=base.name, map_id='ozeti', sigma_m=base.sigma_m)
    assert not pilot.engaged and not pilot.keyboard.down and pilot.goal is None
    assert pilot.manual_position.point == base.point
    assert pilot.telemetry.fix is None and pilot.telemetry.state is None
    assert pilot.estimator.pos is None  # no invented speed or heading measurements
    assert pilot.airnav.initialized == (profile == 'aircraft')
    if profile == 'aircraft':
        assert pilot.airnav.heading is None and pilot.airnav.sigma_m == 200
    world.time = pilot.manual_position.timestamp
    world.body.x += 3.
    world.body.y -= 2.
    pilot.tick(world.time)
    assert pilot.manual_position is None
    assert pilot.telemetry.state is not None
    assert pilot.telemetry.state.position == pytest.approx((world.body.x, world.body.y), abs=.1)
    assert not pilot.engaged


def test_calibrated_base_is_persistent_and_does_not_move_vehicle(tmp_path):
    pilot, _, base = pilot_at_base('truck')
    pilot.config_path = tmp_path / 'config.json'
    pilot.set_initial_position(base.point, map_id='ozeti')
    manual = pilot.manual_position
    corrected = (base.point[0] + 1., base.point[1] + 2.)
    pilot.set_base_position(base.id, corrected, map_id='ozeti')
    assert pilot.manual_position is manual
    saved = Config.load(pilot.config_path)
    edited = team_bases(pilot.assets, saved)[0]
    assert edited.point == corrected and edited.custom
    assert not team_bases(load_map('bakurani'), saved)[0].custom
    assert 'manual_position' not in pilot.config_path.read_text()
    pilot.set_profile('aircraft')
    assert pilot.manual_position is None


def test_fresh_manual_seed_survives_first_hud_without_any_visual_fix():
    from wardog_ad.localization import Fix
    pilot, world, base = pilot_at_base('aircraft')
    pilot.set_initial_position(base.point, map_id='ozeti')
    world.time = pilot.manual_position.timestamp + .1
    pilot.locator.locate = lambda *a: Fix(reason='no terrain', timestamp=world.time)
    pilot.tick(world.time)
    assert pilot.manual_position is not None and pilot.airnav.initialized
    assert pilot.telemetry.state.position == pytest.approx(base.point)
    assert pilot.telemetry.state.heading == pytest.approx(90.)
    # Returning after a prolonged HUD absence invalidates the old manual start.
    pilot.last_hud = world.time
    world.time += pilot.cfg.aircraft.relocalize_after_s + 1
    pilot.tick(world.time)
    assert pilot.manual_position is None and not pilot.airnav.initialized


def test_invalid_or_old_map_initialization_is_rejected():
    pilot, _, base = pilot_at_base('truck')
    with pytest.raises(ValueError, match='map changed'):
        pilot.set_initial_position(base.point, map_id='bakurani')
    with pytest.raises(ValueError, match='invalid initial'):
        pilot.set_initial_position((np.nan, 1.))
    assert pilot.manual_position is None


def test_manual_search_prior_is_not_a_valid_fix():
    from wardog_ad.localization import MinimapLocalizer
    assets = load_map('ozeti')
    loc = MinimapLocalizer.from_assets(assets)
    point = team_bases(assets, Config())[0].point
    loc.set_search_prior(point)
    assert loc.last.position == point
    assert not loc.last.valid and loc.last.heading is None and loc.last.confidence == 0
