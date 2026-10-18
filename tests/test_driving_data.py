"""Read-only capture pipeline, file integrity, temporal metadata and camera labels."""
import json
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from wardog_ad.config import Config, Region
from wardog_ad.driving_data import DatasetWriter, DrivingCollector, NavigationObserver, crop_region, inspect_dataset, mask_regions
from wardog_ad.localization import Fix


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def until(condition, timeout=4):
    end = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > end:
            raise AssertionError('condition not reached')
        time.sleep(.01)


def test_writer_video_and_frame_index_survive_segmentation(tmp_path):
    directory = tmp_path / 'session'
    writer = DatasetWriter(directory, fps=2, segment_s=1, queue_size=10)
    for i in range(5):
        assert writer.submit(np.full((48, 64, 3), i * 40, np.uint8),
                             dict(t=i * .6, capture_end_t=i * .6 + .01, frame_id=i, camera_view='first_person'))
    writer.close()
    rows = lines(directory / 'frames.jsonl')
    assert [(r['video'], r['video_frame']) for r in rows] == [
        ('video_0000.avi', 0), ('video_0000.avi', 1),
        ('video_0001.avi', 0), ('video_0001.avi', 1), ('video_0002.avi', 0)]
    assert [r['t'] for r in rows] == pytest.approx([i * .6 for i in range(5)])
    for path, expected in zip(sorted(directory.glob('*.avi')), [2, 2, 1]):
        video = cv2.VideoCapture(str(path))
        count = 0
        while True:
            ok, frame = video.read()
            if not ok:
                break
            assert frame.shape[:2] == (48, 64)
            count += 1
        video.release()
        assert count == expected
    assert json.loads((directory / 'manifest.json').read_text())['complete']
    result = inspect_dataset(directory)
    assert result['ok'] and result['indexed_frames'] == 5 and result['video_segments'] == 3


def test_bounded_encoder_queue_reports_drops(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class BlockedVideo:
        def __init__(self, *args):
            pass
        def isOpened(self):
            return True
        def write(self, image):
            entered.set()
            release.wait(3)
        def release(self):
            pass
    writer = DatasetWriter(tmp_path / 'session', queue_size=1, video_factory=BlockedVideo)
    try:
        image = np.zeros((32, 32, 3), np.uint8)
        assert writer.submit(image, dict(t=0, frame_id=0))
        assert entered.wait(2)
        assert writer.submit(image, dict(t=.05, frame_id=1))
        assert not writer.submit(image, dict(t=.1, frame_id=2))
    finally:
        release.set()
        writer.close()
    assert writer.written == 2 and writer.dropped == 1
    assert lines(writer.directory / 'events.jsonl')[0]['frame_id'] == 2


def test_encoder_open_failure_marks_session_incomplete(tmp_path):
    class FailedVideo:
        def __init__(self, *args):
            pass
        def isOpened(self):
            return False
        def release(self):
            pass
    writer = DatasetWriter(tmp_path / 'session', video_factory=FailedVideo)
    writer.submit(np.zeros((32, 32, 3), np.uint8), dict(t=0, frame_id=0))
    until(lambda: writer.error is not None)
    writer.close()
    assert not json.loads((writer.directory / 'manifest.json').read_text())['complete']


def test_crop_uses_absolute_original_coordinates():
    image = np.zeros((100, 200, 3), np.uint8)
    assert crop_region(image, Region(300, 400, 200, 100), Region(310, 420, 40, 30)).shape == (30, 40, 3)
    assert crop_region(image, Region(300, 400, 200, 100), Region(0, 0, 40, 30)) is None


def test_wasd_overlay_is_masked_from_training_pixels():
    image = np.full((100, 200, 3), 255, np.uint8)
    masked = mask_regions(image, Region(300, 400, 200, 100), [Region(400, 400, 150, 30)])
    assert not masked[:30, 100:].any()
    assert masked[50, 150, 0] == 255 and masked[10, 50, 0] == 255
    assert image.min() == 255


def observer(cfg):
    from wardog_ad.estimator import StateEstimator
    item = NavigationObserver.__new__(NavigationObserver)
    item.cfg, item.config_path = cfg, None
    item.assets = SimpleNamespace(map_id='test', meters_per_pixel=2)
    item.locator = SimpleNamespace(locate=lambda *a: Fix(x=50, y=50, heading=0, valid=True))
    item.manager = None
    item.estimator = StateEstimator(2)
    item.map_id, item.path, item.route_key = 'test', None, None
    item.progress, item.route_id = 0, 0
    item.ocr = lambda image: '36'
    item.last_speed = item.last_view = None
    item.last_config = -1e9
    return item


def test_speed_by_view_and_manual_map_recovers_after_loss():
    cfg = Config(map_id='test')
    cfg.minimap = Region(0, 0, 80, 80)
    cfg.collection.speed_first_person = Region(0, 0, 25, 25)
    cfg.collection.speed_third_person = Region(30, 0, 25, 25)
    nav = observer(cfg)
    image = np.zeros((100, 100, 3), np.uint8)
    result = nav.observe(image, Region(0, 0, 100, 100), 'first_person', 0)
    assert result['speed'] == dict(kmh=36, source='hud_ocr', signed_ms=None)
    assert result['map_id'] == 'test'
    assert nav.observe(image, Region(0, 0, 100, 100), 'third_person', .2)['speed']['kmh'] == 36
    nav.locator.locate = lambda *a: Fix(valid=False)
    assert nav.observe(image, Region(0, 0, 100, 100), 'unknown', .4)['speed'] is None
    nav.locator.locate = lambda *a: Fix(x=50, y=50, heading=0, valid=True)
    assert nav.observe(image, Region(0, 0, 100, 100), 'first_person', .6)['map_id'] == 'test'


def test_collector_captures_manual_keys_views_and_excludes_other_apps(tmp_path, monkeypatch):
    import wardog_ad.input_win as inputs
    import wardog_ad.capture as capture
    import wardog_ad.driving_data as data
    state = dict(foreground=True, keys={'W'})
    client = Region(0, 0, 128, 96)
    monkeypatch.setattr(inputs, 'foreground_client', lambda *args: (1, 'WARDOGS', client) if state['foreground'] else (2, 'other app', client))
    monkeypatch.setattr(inputs, 'key_is_down', lambda key: key in state['keys'])
    # Any accidental construction of a keyboard output object fails this integration test.
    monkeypatch.setattr(inputs, 'WindowsKeyboard', lambda: pytest.fail('collector attempted keyboard output'))
    class Capture:
        def grab(self, region):
            assert state['foreground']
            return np.zeros((96, 128, 3), np.uint8)
    monkeypatch.setattr(capture, 'ScreenCapture', Capture)
    class Nav:
        def __init__(self, *args):
            pass
        def observe(self, image, client, view, now):
            return dict(t=now, map_id='test', speed=dict(kmh=12, source='fake'), command='straight')
    monkeypatch.setattr(data, 'NavigationObserver', Nav)
    cfg = Config()
    cfg.collection.max_width = 128
    collector = DrivingCollector(cfg, object(), view='first_person', vehicle='URAL', output=tmp_path)
    try:
        collector.start()
        assert collector.paused
        collector.toggle_pause()
        until(lambda: collector.writer.written >= 3)
        state['keys'] = {'C'}
        until(lambda: collector.view == 'unknown')
        state['keys'] = {'W', 'D'}
        collector.set_view('third_person')
        until(lambda: collector.writer.written >= 8)
        state['foreground'] = False
        until(lambda: not collector.foreground)
        before = collector.frame_id
        time.sleep(.15)
        assert collector.frame_id == before
    finally:
        collector.close()
    rows = lines(collector.writer.directory / 'frames.jsonl')
    assert any(row['observed_keys'] == ['W'] for row in rows)
    assert any(row['camera_view'] == 'third_person' for row in rows)
    assert all(row['keys_age_s'] >= 0 for row in rows)
    assert any(row['navigation'] is not None for row in rows)
    events = lines(collector.writer.directory / 'events.jsonl')
    assert any(e.get('key') == 'C' and e['down'] for e in events)
    assert any(e.get('camera_view') == 'unknown' for e in events)


def test_wasd_widget_distinguishes_system_and_sent_states():
    import tkinter as tk
    from wardog_ad.key_display import KeyStateDisplay
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('no graphical display')
    root.withdraw()
    try:
        display = KeyStateDisplay(root, observed=lambda: {'W'}, sent=lambda: {'A'})
        assert display.labels['observed', 'W'].cget('bg') == '#2e7d32'
        assert display.labels['sent', 'A'].cget('bg') == '#1565c0'
        from wardog_ad.ui_theme import KEY_OFF
        assert display.labels['sent', 'W'].cget('bg') == KEY_OFF
    finally:
        root.destroy()


def test_collection_settings_keep_separate_view_speed_regions(tmp_path):
    cfg = Config()
    cfg.collection.speed_first_person = Region(10, 20, 30, 40)
    cfg.collection.speed_third_person = Region(50, 60, 70, 80)
    cfg.save(tmp_path / 'config.json')
    restored = Config.load(tmp_path / 'config.json')
    assert restored.collection.speed_first_person == cfg.collection.speed_first_person
    assert restored.collection.speed_third_person == cfg.collection.speed_third_person
