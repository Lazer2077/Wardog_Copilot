"""Run the actual UI against a stationary simulation. Never sends game input."""
import sys
from pathlib import Path
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wardog_ad.assets import load_map
from wardog_ad.autopilot import Autopilot
from wardog_ad.config import Config
from wardog_ad.input_win import FakeKeyboard, set_dpi_aware
from wardog_ad.panel import ControlPanel
from wardog_ad.sim import Body, SimSensors, SimWorld, TruthLocator


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('map', 'control', 'collection'), default='map')
    parser.add_argument('--seconds', type=int, default=90)
    parser.add_argument('--collection-map',action='store_true',help='also preview collector destination picker')
    args = parser.parse_args()
    set_dpi_aware()
    assets = load_map('bakurani')
    cfg = Config(map_id='bakurani', destinations={'bakurani': [1420, 530]}, homes={'bakurani': [1381, 378]})
    cfg.aircraft.lock_source = 'none'
    keyboard = FakeKeyboard()
    world = SimWorld(cfg, assets.load_image(), assets.meters_per_pixel, Body(1381, 378, 140.), keyboard)
    pilot = Autopilot(cfg, assets, keyboard, SimSensors(world), locator=TruthLocator(world))
    import time
    world.time = time.monotonic()
    pilot.tick(world.time)
    pilot.telemetry.route = [(1381, 378), (1380, 440), (1420, 530)]
    pilot.telemetry.message = 'UI PREVIEW / simulation only'
    # Preview windows are ordinary taskbar windows so visual QA can target them.
    # The real panels retain their non-activating behavior.
    import wardog_ad.panel as panel_module
    panel_module._no_activate = lambda root: None
    if args.mode == 'collection':
        from types import SimpleNamespace
        from wardog_ad.driving_data import CollectionPanel
        c = SimpleNamespace(cfg=cfg, stop_event=threading.Event(), keys={'W'}, vehicle='URAL',
                            view='first_person', lock=threading.RLock(), exclusions=[], latest_nav=None,
                            context_id=0, intervention_reason='obstacle', intervention=dict(active=False,reason=None),
                            assets=assets,
                            observer_args=(cfg, assets, None), paused=True, foreground=False, now=time.monotonic,
                            writer=SimpleNamespace(written=0, dropped=0, directory=Path('UI_PREVIEW')))
        c.set_view = lambda view: setattr(c, 'view', view)
        c.set_vehicle = lambda vehicle: setattr(c, 'vehicle', vehicle)
        c.toggle_pause = lambda: setattr(c, 'paused', not c.paused)
        c.toggle_intervention = lambda: c.intervention.update(active=not c.intervention['active'],reason='obstacle')
        c.mark_incident = lambda kind: None
        c.set_destination = lambda map_id,point,**kwargs: cfg.destinations.update({map_id:point})
        panel = CollectionPanel(c)
        panel.root.title('WARDOGS UI 预览 · 采集面板')
        panel.root.geometry('+100+100')
        if args.collection_map:
            panel.open_map()
        panel.root.after(args.seconds * 1000, c.stop_event.set)
        panel.run()
        return
    panel = ControlPanel(pilot, threading.Event())
    panel.root.title('WARDOGS UI 预览 · 不控制游戏')
    panel.root.geometry('+100+100' if args.mode == 'control' else '+1600+100')
    if args.mode == 'map':
        panel.open_picker()
        panel.root.title('WARDOGS UI 预览 · 战术地图与控制')
        panel.root.geometry('+100+100')
    panel.root.after(args.seconds * 1000, panel.close)
    panel.run()


if __name__ == '__main__':
    main()
