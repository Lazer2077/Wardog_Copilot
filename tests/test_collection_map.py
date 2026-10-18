from types import SimpleNamespace

from PIL import Image
import pytest

from wardog_ad.config import Config


@pytest.fixture
def embedded_map(tmp_path,monkeypatch):
    import tkinter as tk
    import wardog_ad.collection_map as maps
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('no graphical display')
    root.withdraw()
    assets = {}
    for name in ('alpha','beta'):
        image = tmp_path/f'{name}.png'
        Image.new('RGB',(100,100),'#1a2938').save(image)
        assets[name] = SimpleNamespace(root=tmp_path,map_id=name,image_path=image)
    monkeypatch.setattr(maps,'available_maps',lambda path:list(assets))
    monkeypatch.setattr(maps,'load_map',lambda name,path:assets[name])
    calls = []
    cfg = Config(map_id='alpha')
    cfg.auto_map.enabled = True
    c = SimpleNamespace(cfg=cfg,assets=assets['alpha'],latest_nav=None,context_id=0,
                        paused=False,now=lambda:10.,reference_routes={})
    def pause():
        calls.append('pause')
        c.paused = not c.paused
    def destination(map_id,point,**kwargs):
        calls.append((map_id,point))
        cfg.destinations[map_id] = point
    c.toggle_pause,c.set_destination = pause,destination
    picker = maps.CollectionMapPicker(root,c,embedded=True)
    try:
        yield root,picker,c,calls
    finally:
        root.destroy()


def test_embedded_map_is_present_without_opening_a_window_or_pausing(embedded_map):
    import tkinter as tk
    root,picker,c,calls = embedded_map
    assert isinstance(picker.win,tk.Frame)
    assert not any(isinstance(w,tk.Toplevel) for w in root.winfo_children())
    assert picker.canvas.find_all()
    assert c.paused is False and calls == []


def test_map_follows_fresh_navigation_and_keeps_manual_inspection(embedded_map):
    root,picker,c,calls = embedded_map
    c.latest_nav = dict(t=9.8,context_id=0,map_id='beta',state=dict(x=10,y=20),route_id=7,
                        local_plan=dict(reference_map_px=[[10,20],[20,30]]))
    c.reference_routes[7] = dict(map_id='beta',points=[[10,20],[20,30],[40,50]])
    picker.refresh()
    assert picker.assets.map_id == 'beta'
    assert sum(picker.canvas.type(item)=='line' for item in picker.canvas.find_withtag('overlay')) == 2
    picker.choice.set('alpha')
    picker.select_map()
    picker.refresh()
    assert picker.assets.map_id == 'alpha'
    picker.follow.set(True)
    picker.refresh()
    assert picker.assets.map_id == 'beta'
    assert calls == []  # automatic display never pauses or edits navigation


def test_stale_map_does_not_switch_and_destination_edit_pauses(embedded_map):
    root,picker,c,calls = embedded_map
    c.latest_nav = dict(t=5.,context_id=0,map_id='beta')
    picker.refresh()
    assert picker.assets.map_id == 'alpha'
    picker.choose(SimpleNamespace(x=10*picker.scale,y=20*picker.scale))
    assert calls[0] == 'pause'
    assert c.cfg.destinations['alpha'] == pytest.approx([10,20])
