import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from wardog_ad.config import Config
from wardog_ad.localization import Fix


def test_control_panel_automatically_embeds_map_and_follows_both_vehicle_modes(monkeypatch):
    import tkinter as tk
    import wardog_ad.panel as panels
    original = tk.Tk
    def hidden_root(*args,**kwargs):
        root = original(*args,**kwargs)
        root.withdraw()
        return root
    monkeypatch.setattr(tk,'Tk',hidden_root)
    monkeypatch.setattr(panels,'_no_activate',lambda root:None)
    def assets(name):
        return SimpleNamespace(map_id=name,name=name.upper(),width=512,height=512,
                               load_image=lambda:np.zeros((512,512,3),np.uint8),map_to_game=lambda p:p)
    calls = []
    pilot = SimpleNamespace(cfg=Config(map_id='test',profile='truck'),assets=assets('test'),
                            keyboard=SimpleNamespace(down=set()),home=None,destination=None,mpp=5.,
                            telemetry=SimpleNamespace(state=None,fix=None,route=[]))
    for method in ('toggle','return_home','land_here','toggle_profile','set_home_here','go'):
        setattr(pilot,method,lambda method=method:calls.append(method))
    stop = threading.Event()
    try:
        panel = panels.ControlPanel(pilot,stop)
    except tk.TclError as exc:
        pytest.skip(f'no graphical display: {exc}')
    picker = panel.picker
    try:
        assert picker is not None and picker.alive
        assert picker.embedded and isinstance(picker.win,tk.Frame)
        assert picker.win.winfo_toplevel() == panel.root
        assert not any(isinstance(w,tk.Toplevel) for w in panel.root.winfo_children())
        assert picker.opacity is None  # shared top-level opacity
        panel.open_picker()
        assert panel.picker is picker
        assert calls == []  # displaying the map must never engage or switch modes
        pilot.telemetry.fix = Fix(123,234,valid=True,timestamp=time.monotonic())
        picker._tick()
        assert picker.canvas.find_withtag('vehicle')
        # Map/profile changes update the same embedded widget, with no Toplevel API calls.
        pilot.cfg.profile,pilot.cfg.map_id = 'aircraft','test2'
        pilot.assets = assets('test2')
        picker._tick()
        assert picker.assets.map_id == 'test2'
        assert picker.canvas.find_withtag('vehicle')
        assert calls == []
    finally:
        panel.close()
    assert stop.is_set() and picker.alive is False
