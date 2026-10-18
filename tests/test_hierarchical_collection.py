import json
import time

import numpy as np
import pytest

from wardog_ad.config import Config, Region
from wardog_ad.driving_data import DrivingCollector, navigation_validity, inspect_dataset
from wardog_ad.localization import Fix
from test_driving_data import observer, lines, until


def test_navigation_snapshot_causality_context_and_missing_fields():
    nav = dict(t=1.,processed_t=1.2,context_id=4,camera_view='first_person',destination=[5,6],map_id='test',route_id=1,
               reference_route=dict(route_id=1,map_id='test'),local_plan=dict(valid=True,route_id=1),
               rule_control=dict(valid=True,route_id=1,applied=False))
    def validity(**kw):
        return navigation_validity(nav,**dict(view='first_person',context_id=4,now=1.3,**kw))
    assert all(validity().values())
    assert not any(navigation_validity(nav,view='first_person',context_id=4,now=1.1).values())
    assert not any(navigation_validity(nav,view='third_person',context_id=4,now=1.3).values())
    assert not any(navigation_validity(nav,view='first_person',context_id=5,now=1.3).values())
    assert not any(navigation_validity(nav,view='first_person',context_id=4,now=2.1).values())
    nav['destination'] = None
    assert not validity()['local_plan_valid']
    nav['destination'] = [5,6]
    nav['local_plan']['route_id'] = 99
    assert not validity()['rule_control_valid']


def test_observer_generates_route_local_plan_and_stateless_rule_advice():
    cfg = Config(map_id='test',destinations={'test':[50,0]})
    cfg.minimap = Region(0,0,100,100)
    nav = observer(cfg)
    nav.assets.roads = [dict(points=[[50,100],[50,0]],kind='major')]
    image = np.zeros((100,100,3),np.uint8)
    results = []
    for i in range(3):
        nav.locator.locate = lambda *a,i=i: Fix(x=50,y=50-i*2,heading=0,valid=True,timestamp=500000+i)
        results.append(nav.observe(image,Region(0,0,100,100),'first_person',i*.25))
    assert results[0]['route']['points']
    assert 'route' not in results[1]
    assert not results[0]['rule_control']['valid']
    result = results[-1]
    assert result['local_plan']['target_ego_m'][1] > 0
    assert result['local_target_m'][1] < 0
    assert result['fix']['timestamp'] == .5
    assert result['reference_route']['route_id'] == result['route_id']
    assert result['rule_control']['valid'] and result['rule_control']['applied'] is False
    assert not result['local_plan']['obstacle_checked']
    # A long acquisition gap must not preserve a usable speed/controller label.
    later = nav.observe(image,Region(0,0,100,100),'first_person',2.)
    assert not later['rule_control']['valid']


def test_collector_preview_annotations_routes_and_context_survive_save(tmp_path,monkeypatch):
    import wardog_ad.driving_data as data
    import wardog_ad.input_win as inputs
    import wardog_ad.capture as capture
    client = Region(0,0,128,96)
    state = dict(keys={'W'})
    monkeypatch.setattr(inputs,'foreground_client',lambda *a:(1,'WARDOGS',client))
    monkeypatch.setattr(inputs,'key_is_down',lambda k:k in state['keys'])
    monkeypatch.setattr(inputs,'WindowsKeyboard',lambda:pytest.fail('collector attempted actuation'))
    class Capture:
        def grab(self,*a):
            return np.zeros((96,128,3),np.uint8)
    monkeypatch.setattr(capture,'ScreenCapture',Capture)
    class Nav:
        def __init__(self,*a):
            self.route_id = 0
        def reset_temporal(self):
            self.route_id += 1
            self.new = True
        def observe(self,image,client,view,now):
            result = dict(t=now,map_id='test',destination=[20,30],route_id=self.route_id,
                          reference_route=dict(route_id=self.route_id,map_id='test'),
                          local_plan=dict(valid=True,obstacle_checked=False,route_id=self.route_id),
                          rule_control=dict(valid=True,route_id=self.route_id,applied=False,axes=dict(steer=.1,throttle=.3)))
            if self.new:
                result['route'] = dict(route_id=self.route_id,map_id='test',points=[[0,0],[20,30]])
                self.new = False
            return result
    monkeypatch.setattr(data,'NavigationObserver',Nav)
    cfg = Config()
    cfg.collection.max_width = 128
    c = DrivingCollector(cfg,object(),view='first_person',vehicle='KODIAK',output=tmp_path)
    try:
        c.start()
        until(lambda:c.latest_nav is not None)
        assert c.writer.written == 0
        assert c.latest_nav['frame_id'] is None  # preflight preview has no saved image
        c.toggle_pause()
        until(lambda:c.writer.written >= 12)
        state['keys'] = {'W','F8'}
        until(lambda:c.intervention['active'])
        state['keys'] = {'W'}
        until(lambda:c.writer.written >= 17)
        c.mark_incident('collision')
        # View changes invalidate the previously published advice immediately.
        c.set_view('third_person')
        assert c.latest_nav is None
        until(lambda:c.writer.written >= 24)
        state['keys'] = {'I'}
        until(lambda:c.paused)
        written = c.writer.written
        time.sleep(.15)
        assert c.writer.written == written
        # Deliberately leave correction active: closing must emit an explicit end.
    finally:
        c.close()
    rows = lines(c.writer.directory/'frames.jsonl')
    assert any(r['hierarchical_training_candidate'] for r in rows)
    assert any(r['intervention']['active'] for r in rows)
    assert any(r['camera_view']=='third_person' for r in rows)
    for r in rows:
        if r['label_validity']['rule_control_valid']:
            assert r['navigation']['processed_t'] <= r['t']
            assert r['navigation']['context_id'] == r['context_id']
    events = lines(c.writer.directory/'events.jsonl')
    assert any(e['type']=='intervention_start' and e['marker_source']=='F8' for e in events)
    assert any(e['type']=='intervention_end' and e['end_reason']=='session_closed_not_human_success' for e in events)
    assert any(e['type']=='incident' and e['kind']=='collision' for e in events)
    assert json.loads((c.writer.directory/'manifest.json').read_text())['schema_version'] == 2
    report = inspect_dataset(c.writer.directory)
    assert report['ok'],report['problems']
    assert report['reference_routes'] >= 2
    assert report['hierarchical_coverage']['hierarchical_candidates'] > 0
