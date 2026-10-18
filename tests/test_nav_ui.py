"""Terrain following, flying through minimap dropouts, sparse config, map picker."""
import json
import logging

import numpy as np
import pytest

from wardog_ad.assets import DEFAULT_ASSETS
from wardog_ad.config import Config, Region

needs_maps = pytest.mark.skipif(not (DEFAULT_ASSETS / 'maps.json').exists(), reason='map assets not cloned')


def test_config_saves_only_changes(tmp_path):
    cfg = Config(profile='aircraft', map_id='bakurani')
    cfg.minimap = Region(65, 1510, 510, 460)
    cfg.aircraft.flare_burst = 1
    cfg.save(tmp_path / 'c.json')
    data = json.loads((tmp_path / 'c.json').read_text(encoding='utf-8'))
    assert data == {'profile': 'aircraft', 'map_id': 'bakurani',
                    'minimap': {'left': 65, 'top': 1510, 'width': 510, 'height': 460},
                    'aircraft': {'flare_burst': 1}}
    loaded = Config.load(tmp_path / 'c.json')
    assert loaded.aircraft.cruise_agl_m == Config().aircraft.cruise_agl_m   # new defaults still apply


@needs_maps
@pytest.mark.parametrize('hills', [60.0, 120.0])
def test_helicopter_follows_rising_terrain(hills):
    from wardog_ad.assets import load_map
    from wardog_ad.sim import run_sim
    cfg = Config(profile='aircraft')
    cfg.aircraft.lock_source = 'none'
    res = run_sim(cfg, load_map('ozeti'), (300, 1200), (1300, 300), heading=200, max_time=600,
                  world_options=dict(hills=hills))
    assert res.arrived and res.crashes == 0
    # cruise after the climb-out (forward flight now starts from 10 m, still climbing)
    rows = [r[6] for r in res.trace if r[6] and r[5] in ('cruise', 'terrain') and r[0] > 30]
    agl = np.array([o['agl'] for o in rows])
    # at least the cruise height; terrain look-ahead may keep it higher over valleys
    # (240 m peak-to-valley hills: the look-ahead keeps the next peak cleared, so the median
    #  over the valleys sits higher - errs on the safe side; the low tail is what matters)
    assert -10 < np.median(agl) - cfg.aircraft.cruise_agl_m < 60 and np.percentile(agl, 5) > 50


@needs_maps
def test_helicopter_flies_through_minimap_dropout():
    """The real flight lost the minimap match for ~3 s and then had no control at all."""
    from wardog_ad.assets import load_map
    from wardog_ad.autopilot import Autopilot
    from wardog_ad.input_win import FakeKeyboard
    from wardog_ad.localization import Fix
    from wardog_ad.sim import Body, SimSensors, SimWorld, TruthLocator
    logging.disable(logging.WARNING)
    assets = load_map('ozeti')
    cfg = Config(profile='aircraft')
    cfg.aircraft.lock_source = 'none'
    cfg.destinations = {'ozeti': [1000, 600]}
    kb = FakeKeyboard()
    w = SimWorld(cfg, assets.load_image(), assets.meters_per_pixel, Body(400, 1100, 0.0), kb)
    truth = TruthLocator(w)

    class Dropout:
        def locate(self, image, anchor, north_up):
            if 60.0 <= w.time < 66.0:
                return Fix(reason='too few matches (3)', timestamp=w.time)
            return truth.locate(image, anchor, north_up)

    pilot = Autopilot(cfg, assets, kb, SimSensors(w), locator=Dropout())
    assert pilot.go()
    statuses, alts, error = [], [], []
    next_slow = 0.0
    while w.time < 90:
        if w.time >= next_slow:
            tel = pilot.tick(w.time)
            next_slow += 1 / cfg.loop_hz
            if 60 <= w.time < 66:
                statuses.append(tel.status)
                alts.append(w.body.alt)
                if tel.state is not None:
                    error.append(np.hypot(tel.state.x - w.body.x, tel.state.y - w.body.y) * assets.meters_per_pixel)
        pilot.fast_control(w.time)
        pilot.actuate(w.time)
        w.step(1 / 30)
    logging.disable(logging.NOTSET)
    assert pilot.engaged and w.crashes == 0
    assert 'lost' not in statuses and 'searching' not in statuses   # it kept flying
    assert min(alts) > 60                                            # and kept its height
    assert error and max(error) < 60                                 # dead reckoning stayed close


@needs_maps
def test_map_picker_sets_destination_home_and_favourites(tmp_path):
    tk = pytest.importorskip('tkinter')
    from wardog_ad.assets import load_map
    from wardog_ad.autopilot import Autopilot
    from wardog_ad.input_win import FakeKeyboard
    from wardog_ad.picker import MapPicker
    from wardog_ad.sim import Body, SimSensors, SimWorld, TruthLocator
    logging.disable(logging.WARNING)
    assets = load_map('bakurani')
    cfg = Config(profile='aircraft', map_id='bakurani')
    cfg.aircraft.lock_source = 'none'
    kb = FakeKeyboard()
    w = SimWorld(cfg, assets.load_image(), assets.meters_per_pixel, Body(1381, 378, 140.0), kb)
    pilot = Autopilot(cfg, assets, kb, SimSensors(w), locator=TruthLocator(w))
    pilot.config_path = tmp_path / 'c.json'
    pilot.tick(0.0)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('no display')
    root.withdraw()
    picker = MapPicker(root, pilot)

    class Event:
        def __init__(self, x, y, delta=0):
            self.x, self.y, self.delta = x, y, delta

    picker._press(Event(500, 300))
    picker._release(Event(500, 300))
    clicked = pilot.destination
    assert clicked == pytest.approx(picker.to_map(500, 300))
    picker._press(Event(450, 360))
    picker._motion_drag(Event(380, 320))
    picker._release(Event(380, 320))
    assert pilot.destination == clicked                     # a drag only pans
    picker._right(Event(300, 400))
    assert pilot.home is not None
    picker.coord.insert(0, '110, 95')
    picker._coord_dest()
    assert pilot.destination == pytest.approx(assets.game_to_map((110, 95)))
    pilot.add_favorite('机场', pilot.destination)
    picker._load_favs()
    assert picker.favs.get(0) == '机场'
    picker.cruise.set(180)
    picker._cruise_changed()
    picker.close()
    root.destroy()
    logging.disable(logging.NOTSET)
    saved = json.loads((tmp_path / 'c.json').read_text(encoding='utf-8'))
    assert saved['destinations']['bakurani'] == pytest.approx(list(assets.game_to_map((110, 95))), abs=.01)
    assert 'bakurani' in saved['homes'] and saved['favorites']['bakurani'][0]['name'] == '机场'
    assert saved['aircraft']['cruise_agl_m'] == 180 and cfg.aircraft.cruise_agl_m == 180
