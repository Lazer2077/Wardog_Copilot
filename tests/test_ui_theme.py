from types import SimpleNamespace

import pytest

from wardog_ad.config import Config
from wardog_ad.ui_theme import opacity_value, save_opacity


def test_opacity_is_bounded_and_persisted_independently(tmp_path):
    cfg = Config(destinations={'ozeti': [50, 60]})
    path = tmp_path / 'config.json'
    cfg.save(path)
    save_opacity(cfg, 'panel_opacity', .7, path=path)
    save_opacity(cfg, 'map_opacity', 4., path=path)
    saved = Config.load(path)
    assert saved.ui.panel_opacity == .7 and saved.ui.map_opacity == 1.
    assert saved.ui.collection_opacity == Config().ui.collection_opacity
    assert saved.destinations == cfg.destinations
    assert opacity_value(-1) == .55
    assert opacity_value(float('nan')) == .9


def test_theme_preserves_widget_actions_and_opacity_slider_commits_on_release():
    import tkinter as tk
    from tkinter import ttk
    from wardog_ad.ui_theme import OpacityControl, ScrollPanel, style_widgets, window_theme
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('no display')
    root.withdraw()
    try:
        window_theme(root, .86)
        calls, saved = [], []
        rail = ScrollPanel(root, height=160)
        rail.pack()
        combo = ttk.Combobox(rail.body, values=['a', 'b'], state='readonly')
        combo.set('a')
        combo.pack()
        entry = tk.Entry(rail.body)
        entry.insert(0, '42')
        entry.pack()
        spin = ttk.Spinbox(rail.body, from_=30, to=500)
        spin.pack()
        button = tk.Button(rail.body, text='GO', command=lambda: calls.append('go'))
        button.pack()
        opacity = OpacityControl(rail.body, root, .86, on_save=saved.append)
        opacity.pack()
        style_widgets(root)
        rail.bind_wheel()
        button.invoke()
        assert calls == ['go'] and combo.get() == 'a' and entry.get() == '42'
        combo.event_generate('<MouseWheel>', delta=-120)
        assert combo.get() == 'a'  # scrolling the sidebar must not select another map
        opacity.value.set(65)
        opacity._apply()
        assert opacity.percent.cget('text') == '65%' and not saved
        opacity._save(SimpleNamespace())
        assert saved == [.65]
        opacity.value.set(100)
        opacity._save()
        assert saved[-1] == 1.
    finally:
        root.destroy()
