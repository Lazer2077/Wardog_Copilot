"""Resolution changes must not silently feed off-screen black crops to navigation."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from wardog_ad.capture import CaptureRegionError, ScreenCapture, region_on_screen, resolve_region
from wardog_ad.config import Config, Region


HD = dict(left=0, top=0, width=1920, height=1080)


def test_saved_calibration_scales_and_round_trips_without_mutating_source(tmp_path):
    cfg = Config(minimap=Region(65, 1510, 510, 460, reference=(0, 0, 3840, 2160)))
    cfg.save(tmp_path / 'config.json')
    restored = Config.load(tmp_path / 'config.json').minimap
    assert resolve_region(restored, [HD]) == Region(32, 755, 256, 230)
    assert resolve_region(restored, [dict(HD, width=2560, height=1440)]) == Region(43, 1007, 340, 306)
    assert restored == cfg.minimap and restored.top == 1510


def test_legacy_regions_stay_absolute_and_offscreen_capture_never_reaches_mss():
    roi = Region(65, 1510, 510, 460)
    assert resolve_region(roi, [HD]) is roi
    cap = ScreenCapture.__new__(ScreenCapture)
    cap.monitors = lambda: [HD]
    cap._mss = SimpleNamespace(grab=lambda _: pytest.fail('must reject offscreen capture'))
    with pytest.raises(CaptureRegionError, match='outside monitor'):
        cap.grab(roi)


def test_resolution_mapping_rejects_ambiguous_displays_and_aspect_changes():
    roi = Region(100, 100, 200, 200, reference=(0, 0, 3840, 2160))
    with pytest.raises(CaptureRegionError, match='aspect ratio'):
        resolve_region(roi, [dict(HD, width=1280, height=1024)])
    with pytest.raises(CaptureRegionError, match='ambiguous'):
        resolve_region(roi, [dict(HD, left=-1920), dict(HD, left=1920)])
    shifted = dict(HD, left=-1920)
    assert resolve_region(roi, [shifted]) == Region(-1870, 50, 100, 100)
    assert not region_on_screen(Region(-10, 0, 100, 100), [HD])


def test_sensor_reports_bad_region_and_recovers_after_recalibration(monkeypatch):
    from wardog_ad.autopilot import ScreenSensors
    monkeypatch.setattr('wardog_ad.input_win.foreground_is', lambda _: True)
    cfg = Config(minimap=Region(65, 1510, 510, 460))
    s = ScreenSensors(cfg)
    cap = ScreenCapture.__new__(ScreenCapture)
    cap.monitors = lambda: [HD]
    cap._mss = SimpleNamespace(grab=lambda r: np.zeros((r['height'], r['width'], 4), np.uint8))
    s._factory = lambda: cap
    assert s.minimap() is None
    assert 'outside monitor' in s.capture_errors['minimap']
    cfg.minimap.reference = (0, 0, 3840, 2160)
    assert s.minimap().shape == (230, 256, 3)
    assert not s.capture_errors


def test_collector_crop_uses_same_resolution_mapping(monkeypatch):
    from wardog_ad.driving_data import crop_region
    monkeypatch.setattr('wardog_ad.capture.display_monitors', lambda: [HD])
    image = np.zeros((1080, 1920, 3), np.uint8)
    image[755:985, 32:288] = 73
    roi = Region(65, 1510, 510, 460, reference=(0, 0, 3840, 2160))
    crop = crop_region(image, Region(0, 0, 1920, 1080), roi)
    assert crop.shape == (230, 256, 3) and np.all(crop == 73)


def test_zone_tint_keeps_terrain_but_masks_small_symbols_and_white_arrow():
    from wardog_ad.localization import feature_mask
    hsv = np.zeros((160, 180, 3), np.uint8)
    hsv[:] = (0, 0, 75)
    hsv[15:100, 15:125] = (60, 140, 95)  # translucent green zone
    hsv[105:112, 140:147] = (110, 200, 200)  # small blue symbol
    hsv[70:77, 70:77] = (0, 0, 255)  # white player marker
    mask = feature_mask(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    assert mask[35, 35] == 255
    assert mask[108, 143] == 0 and mask[73, 73] == 0
