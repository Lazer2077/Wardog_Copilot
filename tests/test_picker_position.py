"""Map selection must show known coordinates even without a vehicle heading."""
from types import SimpleNamespace

import numpy as np
import pytest

from wardog_ad.config import Config
from wardog_ad.estimator import VehicleState
from wardog_ad.localization import Fix


@pytest.fixture(scope='module')
def tk_root():
    tk = pytest.importorskip('tkinter')
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('no display')
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def picker(monkeypatch, tk_root):
    import tkinter as tk
    from wardog_ad.picker import MapPicker
    root = tk_root
    original = tk.Toplevel
    def hidden(*args, **kwargs):
        window = original(*args, **kwargs)
        window.withdraw()
        return window
    monkeypatch.setattr(tk, 'Toplevel', hidden)
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr('wardog_ad.picker.time.monotonic', lambda: clock.now)
    assets = SimpleNamespace(map_id='test', name='TEST', width=512, height=512,
                             load_image=lambda: np.zeros((512, 512, 3), np.uint8),
                             map_to_game=lambda p: p)
    pilot = SimpleNamespace(assets=assets, cfg=Config(map_id='test'), destination=None, home=None,
                            mpp=5., go=lambda: None,
                            telemetry=SimpleNamespace(state=None, fix=None, route=[]),
                            estimator=SimpleNamespace(pos=None, t=None, heading=None))
    window = MapPicker(root, pilot)
    yield window, pilot, clock
    window.close()


def test_fix_without_heading_draws_a_labeled_position_and_centers(picker):
    window, pilot, _ = picker
    assert not window.canvas.find_withtag('vehicle')
    assert '等待小地图定位' in window.info.cget('text')
    pilot.telemetry.fix = Fix(123., 234., valid=True, timestamp=100.)
    window._tick()
    assert (window.cx, window.cy) == (123., 234.)
    items = window.canvas.find_withtag('vehicle')
    assert [window.canvas.type(i) for i in items] == ['oval', 'oval']
    assert '当前位置' in window.info.cget('text')
    assert '朝向：尚未识别' in window.info.cget('text')
    window.cx, window.cy = 400., 400.
    window.center_on_vehicle()
    assert (window.cx, window.cy) == (123., 234.)


def test_dead_reckoning_and_stale_fix_are_distinguishable(picker):
    window, pilot, clock = picker
    pilot.telemetry.state = VehicleState(123., 234., 90., 20., 0., 90., 100., 'dead reckoning')
    window._draw_overlay()
    items = window.canvas.find_withtag('vehicle')
    assert window.canvas.type(items[-1]) == 'polygon'
    assert window.canvas.itemcget(items[-1], 'fill') == '#40c4ff'
    pilot.telemetry.state = None
    clock.now = 110.
    window._draw_overlay()
    window._update_info()
    assert '最后定位（10 秒前）' in window.info.cget('text')
    items = window.canvas.find_withtag('vehicle')
    assert window.canvas.itemcget(items[-1], 'fill') == '#ffb300'


def test_last_accepted_fix_available_when_opened_during_dropout(picker):
    window, pilot, _ = picker
    pilot.estimator.pos, pilot.estimator.t = np.array([150., 200.]), 90.
    window._draw_overlay()
    window._update_info()
    assert window.canvas.find_withtag('vehicle')
    assert '最后定位（10 秒前）' in window.info.cget('text')


def test_new_map_clears_old_position_and_invalid_fixes_do_not_draw(picker):
    window, pilot, _ = picker
    pilot.telemetry.fix = Fix(123., 234., valid=True, timestamp=100.)
    window._tick()
    assert window.canvas.find_withtag('vehicle')
    pilot.assets = SimpleNamespace(map_id='new', name='NEW', width=512, height=512,
                                  load_image=lambda: np.zeros((512, 512, 3), np.uint8),
                                  map_to_game=lambda p: p)
    pilot.telemetry.fix = Fix(100., 100., valid=False, timestamp=100.)
    window._tick()
    assert not window.canvas.find_withtag('vehicle')
    assert '等待小地图定位' in window.info.cget('text')


def test_waiting_for_position_does_not_override_user_pan(picker):
    window, pilot, _ = picker
    window._press(SimpleNamespace(x=300, y=300))
    before = (window.cx, window.cy)
    pilot.telemetry.fix = Fix(123., 234., valid=True, timestamp=100.)
    window._tick()
    assert (window.cx, window.cy) == before
    assert window.canvas.find_withtag('vehicle')


def test_map_dropdown_dispatches_manual_and_auto_and_reports_failure(picker):
    window, pilot, _ = picker
    calls = []
    pilot.set_map_mode = calls.append
    choice, map_id = next(iter(window._map_choices.items()))
    window.map_choice.set(choice)
    window._change_map()
    window.map_choice.set('自动识别')
    window._change_map()
    assert calls == [map_id, None]
    def fail(map_id):
        raise ValueError('missing map')
    pilot.set_map_mode = fail
    window.map_choice.set(choice)
    window._change_map()
    assert '切图失败' in window.map_error.cget('text')
    assert window.map_choice.get() == '自动识别'


def test_base_selection_and_manual_position_label(picker):
    from wardog_ad.bases import TeamBase, ManualPosition
    window, pilot, clock = picker
    base = TeamBase('valkyra', 'red', '#ef5350', (100., 200.), 200., 'test')
    window._team_bases = [base]
    window._base_choices = {'red': base}
    window.base_choice.set('red')
    def seed(point, *, source, map_id, sigma_m):
        pilot.manual_position = ManualPosition(map_id, point, clock.now, source, sigma_m)
    pilot.set_initial_position = seed
    window._use_base()
    assert pilot.manual_position.point == base.point
    assert pilot.destination is None
    assert window.canvas.find_withtag('team_base')
    assert (window.cx, window.cy) == base.point
    assert '手动初始位置' in window.info.cget('text')
    items = window.canvas.find_withtag('vehicle')
    assert window.canvas.itemcget(items[-1], 'fill') == '#b388ff'
    pilot.telemetry.fix = Fix(300., 300., valid=True, timestamp=clock.now)
    clock.now += 5.
    window._update_info()
    assert '手动初始位置' in window.info.cget('text')  # unaccepted fix cannot replace a manual seed


def test_point_selection_seeds_position_without_changing_destination(picker):
    window, pilot, _ = picker
    calls = []
    pilot.set_initial_position = lambda point, **kwargs: calls.append((point, kwargs))
    window._start_position_pick('initial')
    expected_point = window.to_map(450, 360)
    window._press(SimpleNamespace(x=450, y=360))
    window._release(SimpleNamespace(x=450, y=360))
    assert calls[0][0] == pytest.approx(expected_point)
    assert calls[0][1]['map_id'] == 'test'
    assert pilot.destination is None and window._position_action is None
    window._start_position_pick('initial')
    window._right(SimpleNamespace(x=450, y=360))
    assert window._position_action is None and pilot.home is None
