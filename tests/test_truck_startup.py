from types import SimpleNamespace

import numpy as np
import pytest

from wardog_ad.autopilot import Autopilot
from wardog_ad.config import Config
from wardog_ad.input_win import FakeKeyboard
from wardog_ad.localization import Fix


def pilot(profile='truck',heading=90.):
    cfg = Config(map_id='test',profile=profile,destinations={'test':[200,0]})
    cfg.aircraft.lock_source = 'none'
    cfg.refine_fixes = False
    assets = SimpleNamespace(map_id='test',name='TEST',meters_per_pixel=1.,
                             roads=[dict(points=[[0,0],[300,0]],kind='major')])
    sensors = SimpleNamespace(minimap=lambda:np.zeros((64,64,3),np.uint8),main_view=lambda:None,
                              truck_heading=lambda:heading)
    locator = SimpleNamespace(locate=lambda *a:Fix(x=0,y=0,heading=None,valid=True))
    return Autopilot(cfg,assets,FakeKeyboard(),sensors,locator=locator)


def test_route_preview_does_not_require_heading_or_engagement():
    p = pilot(heading=None)
    tel = p.tick(1.)
    assert len(tel.route)>=2 and tel.route[-1] == (200,0)
    assert tel.state is None and tel.status == 'need heading'
    assert not p.engaged and not p.keyboard.down
    assert tel.output is None


def test_stationary_truck_can_start_from_compass_without_motion_heading():
    p = pilot()
    assert p.go()
    tel = p.tick(1.)
    assert tel.state.speed_ms == 0 and tel.state.heading == pytest.approx(90.)
    assert tel.state.heading_source == 'compass'
    assert tel.output.axes['throttle'] > 0
    p.actuate(1.)
    assert 'W' in p.keyboard.down


def test_pause_previews_route_but_does_not_advance_recovery_or_send_keys():
    p = pilot()
    p.go()
    p.tick(1.)
    p.actuate(1.)
    p.disengage('paused')
    for i in range(100):
        tel = p.tick(2.+i*.1)
        p.actuate(2.+i*.1)
        assert tel.output is None and tel.status == 'paused'
        assert not p.keyboard.down
    assert p.controller.mode == 'drive'
    assert p.controller.last_time is None and p.controller.slow_since is None
    p.toggle()
    tel = p.tick(13.)
    assert tel.output.mode == 'drive' and tel.output.axes['throttle']>0


def test_vehicle_switch_discards_helicopter_landing_goal_and_mission():
    p = pilot(profile='aircraft')
    p.mission,p.goal,p.waypoints = 'land',(50,50),[(20,20)]
    p.telemetry.route = [(50,50),(60,60)]
    p.keyboard.press('W')
    p.set_profile('truck')
    assert p.mission == 'idle' and p.goal is None
    assert p.waypoints == [] and p.controller is None
    assert not p.keyboard.down and not p.engaged
    assert p.telemetry.route == []
    p.toggle()
    assert p.mission == 'outbound' and p.goal == (200,0)


def test_changing_destination_while_idle_refreshes_preview_without_driving():
    p = pilot()
    p.tick(1.)
    p.set_destination((80,0))
    tel = p.tick(1.2)
    assert tel.route[-1] == (80,0)
    assert not p.engaged and not p.keyboard.down


def test_go_can_wait_for_initial_map_confirmation_but_cancel_and_dropout_do_not_resume():
    p = pilot()
    manager = SimpleNamespace(ready=False,active='test')
    manager.update = lambda *a:SimpleNamespace(ready=manager.ready,map_id='test' if manager.ready else None,
                        fix=Fix(x=0,y=0,heading=None,valid=True),reason='confirmed' if manager.ready else 'identifying')
    p.map_manager = manager
    p.go()
    p.tick(1.)
    assert p.pending_go and not p.engaged
    p.toggle()
    assert not p.pending_go and not p.engaged
    p.go()
    manager.ready = True
    tel = p.tick(2.)
    assert not p.pending_go and p.engaged and tel.output.axes['throttle']>0
    p.actuate(2.)
    assert 'W' in p.keyboard.down
    manager.ready = False
    p.tick(3.)
    assert not p.engaged and not p.pending_go and not p.keyboard.down
    manager.ready = True
    p.tick(4.)
    assert not p.engaged and not p.keyboard.down
