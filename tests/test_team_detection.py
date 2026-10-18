"""Faction inference ignores territory colour, score bars and one-frame noise."""
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from wardog_ad.team_detection import TeamDetector, TeamReading, detect_team


COLOURS = {'lonestar': (240,160,30), 'valkyra': (30,40,240), 'manticore': (45,220,45)}


def minimap(faction='lonestar', count=4):
    image = np.full((270,300,3),35,np.uint8)
    for i in range(count):
        x,y = 50+i*45,80+(i%2)*65
        cv2.fillPoly(image,[np.array([[x,y],[x-6,y+13],[x+6,y+10]])],COLOURS[faction])
    return image


@pytest.mark.parametrize('faction', list(COLOURS))
@pytest.mark.parametrize('scale', [.8,1.,2.])
def test_detects_multiple_friendly_markers_across_resolutions(faction, scale):
    reading = detect_team(cv2.resize(minimap(faction),None,fx=scale,fy=scale),1.)
    assert reading.faction_id == faction and reading.markers >= 3


def test_zone_tint_and_control_or_score_bars_do_not_choose_team():
    image = minimap(count=0)
    image[20:200,20:250] = COLOURS['manticore']
    image[250:,:] = COLOURS['valkyra']
    assert detect_team(image,1.).faction_id is None
    assert detect_team(minimap(count=1),1.).faction_id is None


def test_temporal_confirmation_expiry_and_context_reset():
    d = TeamDetector()
    image = minimap()
    assert d.update(image,10.,'map1').faction_id is None
    assert d.update(image,10.01,'map1').faction_id is None  # same frame/time window
    assert d.update(image,10.2,'map1').faction_id is None
    assert d.update(image,10.4,'map1').faction_id == 'lonestar'
    assert d.update(None,14.,'map1').faction_id is None
    assert d.update(image,14.2,'map2').faction_id is None


def test_conflicting_markers_never_confirm():
    image = minimap(count=3)
    for x,y in ((50,200),(110,200),(170,200)):
        cv2.fillPoly(image,[np.array([[x,y],[x-6,y+13],[x+6,y+10]])],COLOURS['valkyra'])
    assert detect_team(image,1.).faction_id is None


def test_sensor_does_not_match_or_classify_pixels_from_another_app(monkeypatch):
    from wardog_ad.autopilot import ScreenSensors
    from wardog_ad.config import Config
    monkeypatch.setattr('wardog_ad.input_win.foreground_is',lambda _:False)
    sensors = ScreenSensors(Config())
    sensors._grab = Mock(side_effect=AssertionError('another app must not be captured'))
    assert sensors.minimap() is None and sensors.team_reading.faction_id is None


def test_automatic_base_selection_never_sets_vehicle_position_or_overrides_manual_choice(monkeypatch):
    from wardog_ad.picker import MapPicker
    monkeypatch.setattr('wardog_ad.picker.time.monotonic',lambda:10.)
    class Value:
        def __init__(self,value): self.value=value
        def get(self): return self.value
        def set(self,value): self.value=value
    p = MapPicker.__new__(MapPicker)
    p.pilot = SimpleNamespace(sensors=SimpleNamespace(team_reading=TeamReading('lonestar',.9,10.)),
                              set_initial_position=Mock())
    p.base_auto,p.base_choice = Value(True),Value('red')
    p._base_choices = {'red':SimpleNamespace(id='valkyra'),'blue':SimpleNamespace(id='lonestar')}
    p.team_hint,p.use_base_button = Mock(),Mock()
    p._sync_team()
    assert p.base_choice.get() == 'blue'
    p.pilot.set_initial_position.assert_not_called()
    p.base_auto.set(False);p.base_choice.set('red');p._sync_team()
    assert p.base_choice.get() == 'red'
    p.base_auto.set(True)
    p.pilot.sensors.team_reading = TeamReading('lonestar',.9,0.)
    p._sync_team()
    assert p.base_choice.get() == ''
    p.use_base_button.config.assert_called_with(state='disabled')
