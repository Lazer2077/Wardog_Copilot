"""Read-only manual-driving dataset capture, independent of Autopilot/input output.

Capture, key sampling, perception and encoding use separate loops. All JSONL
timestamps are seconds on one session-relative monotonic clock; video timestamps
must be reconstructed using frames.jsonl, not the AVI nominal frame rate.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import statistics
import threading
import time
import uuid

import cv2

from .config import Config, Region
from .recorder import _plain

VIEWS = ('first_person', 'third_person', 'unknown')
DRIVING_KEYS = ('W', 'A', 'S', 'D', 'SPACE', 'R', 'C', 'ALT', 'LSHIFT', 'LCTRL', 'ESC', 'M', 'TAB', 'I', 'F', '0')
INTERVENTION_REASONS = ('obstacle', 'route_correction', 'localization_error', 'emergency_stop', 'other')


def navigation_validity(nav, *, view, context_id, now):
    fresh = bool(nav and nav.get('camera_view') == view and nav.get('context_id') == context_id
                 and 0 <= now-nav['t'] <= 1. and nav.get('processed_t', math.inf) <= now)
    ref = (nav or {}).get('reference_route') or {}
    plan = (nav or {}).get('local_plan') or {}
    control = (nav or {}).get('rule_control') or {}
    reference = bool(fresh and nav.get('route_id') is not None and nav.get('destination') is not None
                     and ref.get('route_id') == nav['route_id'] and ref.get('map_id') == nav.get('map_id'))
    local = bool(reference and plan.get('valid') and plan.get('route_id') == nav['route_id'])
    rule = bool(local and control.get('valid') and control.get('route_id') == nav['route_id'] and control.get('applied') is False)
    return dict(navigation_fresh=fresh, reference_route_valid=reference,
                local_plan_valid=local, rule_control_valid=rule)


def crop_region(image, client: Region, roi: Region):
    """Absolute physical screen ROI -> original, unscaled client image."""
    x, y = roi.left - client.left, roi.top - client.top
    if not roi.valid or x < 0 or y < 0 or x + roi.width > image.shape[1] or y + roi.height > image.shape[0]:
        return None
    return image[y:y + roi.height, x:x + roi.width].copy()


def mask_regions(image, client, regions):
    """Remove our key display from saved pixels to prevent action-label leakage."""
    if not regions:
        return image
    output = image.copy()
    for region in regions:
        x0, y0 = max(0, region.left - client.left), max(0, region.top - client.top)
        x1 = min(image.shape[1], region.left + region.width - client.left)
        y1 = min(image.shape[0], region.top + region.height - client.top)
        if x1 > x0 and y1 > y0:
            output[y0:y1, x0:x1] = 0
    return output


class DatasetWriter:
    def __init__(self, directory, *, fps=20., segment_s=120., queue_size=8, metadata=None,
                 video_factory=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.fps, self.segment_frames = fps, max(1, int(fps * segment_s))
        self.metadata = metadata or {}
        self.frames = open(self.directory / 'frames.jsonl', 'w', encoding='utf-8')
        self.events = open(self.directory / 'events.jsonl', 'w', encoding='utf-8')
        self.navigation = open(self.directory / 'navigation.jsonl', 'w', encoding='utf-8')
        self.routes = open(self.directory / 'routes.jsonl', 'w', encoding='utf-8')
        self.route_records = 0
        self.lock = threading.Lock()
        self.queue = queue.Queue(maxsize=queue_size)
        self.factory = video_factory or cv2.VideoWriter
        self.accepted = self.written = self.dropped = 0
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._encode, name='dataset-encoder', daemon=True)
        self._manifest(False)
        self.thread.start()

    def _manifest(self, complete):
        manifest = dict(schema_version=2, complete=complete, error=self.error, route_records=self.route_records,
                        frames_accepted=self.accepted, frames_written=self.written, frames_dropped=self.dropped,
                        video_format='MJPG AVI; use frames.jsonl timestamps for variable-rate timing',
                        key_source='GetAsyncKeyState polling; may include input injected by other processes',
                        metadata=self.metadata)
        (self.directory / 'manifest.json').write_text(json.dumps(_plain(manifest), indent=2, ensure_ascii=False), encoding='utf-8')

    def event(self, record):
        with self.lock:
            self.events.write(json.dumps(_plain(record), ensure_ascii=False) + '\n')
            self.events.flush()

    def nav(self, record):
        with self.lock:
            self.navigation.write(json.dumps(_plain(record), ensure_ascii=False) + '\n')
            self.navigation.flush()

    def route(self, record):
        with self.lock:
            self.routes.write(json.dumps(_plain(record), ensure_ascii=False) + '\n')
            self.routes.flush()
            self.route_records += 1

    def submit(self, image, record):
        if self.closed or self.error:
            return False
        try:
            self.queue.put_nowait((image, record))
            self.accepted += 1
            return True
        except queue.Full:
            self.dropped += 1
            self.event(dict(t=record['t'], type='frame_dropped', frame_id=record['frame_id'], reason='encoder_queue_full'))
            return False

    def _encode(self):
        video = None
        shape, segment, index = None, -1, 0
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                image, record = item
                size = (image.shape[1], image.shape[0])
                if video is None or size != shape or index >= self.segment_frames:
                    if video is not None:
                        video.release()
                    segment += 1
                    index, shape = 0, size
                    filename = f'video_{segment:04d}.avi'
                    video = self.factory(str(self.directory / filename), cv2.VideoWriter_fourcc(*'MJPG'), self.fps, size)
                    if not video.isOpened():
                        raise RuntimeError('MJPG encoder could not open video file')
                video.write(image)
                self.frames.write(json.dumps(_plain(dict(record, video=filename, video_frame=index,
                                                        width=size[0], height=size[1])), ensure_ascii=False) + '\n')
                self.frames.flush()
                self.written += 1
                index += 1
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            if video is not None:
                video.release()

    def close(self):
        if self.closed:
            return
        self.closed = True
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=.1)
                break
            except queue.Full:
                continue
        self.thread.join(30)
        if self.thread.is_alive():
            raise RuntimeError('encoder did not finish; session remains marked incomplete')
        self.frames.close()
        self.events.close()
        self.navigation.close()
        self.routes.close()
        self._manifest(self.error is None)


class NavigationObserver:
    """Position, route and stateless rule advice; never constructs an actuator."""
    def __init__(self, cfg, assets, config_path=None):
        from .localization import MinimapLocalizer
        from .map_manager import MapManager
        from .estimator import StateEstimator
        self.cfg, self.assets, self.config_path = cfg, assets, config_path
        self.locator = MinimapLocalizer.from_assets(assets)
        self.manager = MapManager(assets, self.locator, cfg.auto_map) if cfg.auto_map.enabled else None
        self.estimator = StateEstimator(assets.meters_per_pixel)
        self.map_id = assets.map_id
        self.path = self.route_key = None
        self.progress = 0.
        self.route_id = 0
        self.ocr = None
        self.last_speed = None
        self.last_view = None
        self.last_config = -math.inf
        self.configured_mode = cfg.map_id, cfg.auto_map.enabled
        self.last_observation = None

    def reset_temporal(self):
        self.estimator.reset()
        self.path = self.route_key = None
        self.progress = 0.
        self.last_speed = self.last_view = None

    def _sync_mode(self):
        mode = self.cfg.map_id, self.cfg.auto_map.enabled
        if mode != getattr(self, 'configured_mode', mode):
            from .assets import load_map
            from .localization import MinimapLocalizer
            from .map_manager import MapManager
            from .estimator import StateEstimator
            self.assets = load_map(self.cfg.map_id, self.cfg.assets_dir or self.assets.root)
            self.locator = MinimapLocalizer.from_assets(self.assets)
            self.manager = MapManager(self.assets, self.locator, self.cfg.auto_map) if self.cfg.auto_map.enabled else None
            self.estimator = StateEstimator(self.assets.meters_per_pixel)
            self.map_id = self.assets.map_id
            self.reset_temporal()
        self.configured_mode = mode

    def observe(self, image, client, view, now):
        from .planner import RoadGraph, Path as RoadPath, RouteError
        from .estimator import StateEstimator
        from .local_planner import ReferenceLocalPlanner, shadow_rule_control
        self._sync_mode()
        previous = getattr(self, 'last_observation', None)
        if previous is not None and not 0 < now-previous <= 1.:
            self.reset_temporal()
        self.last_observation = now
        roi = crop_region(image, client, self.cfg.localization_region)
        if roi is not None and max(roi.shape[:2]) > self.cfg.minimap_max_px:
            scale = self.cfg.minimap_max_px / max(roi.shape[:2])
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        out = dict(t=now, map_id=None, map_status='no calibrated minimap', fix=None,
                   state=None, speed=None, destination=None, route_id=None,
                   reference_route=None, local_plan=None,
                   rule_control=dict(t=now, valid=False, reason='missing_state_or_route', applied=False, axes=None))
        if self.config_path and now - self.last_config > 2:
            saved = Config.load(self.config_path)
            self.cfg.destinations = saved.destinations
            self.last_config = now
        fix = None
        if self.manager:
            selection = self.manager.update(roi, now, self.cfg.localization_anchor, self.cfg.north_up)
            out['map_status'] = selection.reason
            if selection.ready:
                self.assets = self.manager.assets[selection.map_id]
                out['map_id'], fix = selection.map_id, selection.fix
        elif roi is not None:
            fix = self.locator.locate(roi, self.cfg.localization_anchor, self.cfg.north_up)
            out['map_status'] = fix.reason
            out['map_id'] = self.assets.map_id if fix.valid else None
        if out['map_id'] != self.map_id:
            self.map_id = out['map_id']
            self.estimator = StateEstimator(self.assets.meters_per_pixel)
            self.path = self.route_key = None
        if fix is not None and fix.valid and self.cfg.refine_fixes and hasattr(self.assets, 'load_image'):
            from .refine import FixRefiner
            if getattr(self, '_refine_map', None) != self.assets.map_id:
                self._refine = FixRefiner(self.assets.load_image())
                self._refine_map = self.assets.map_id
            fix = self._refine.refine(roi, fix, self.cfg.localization_anchor)
        state = self.estimator.update(fix, now) if fix is not None and fix.valid else None
        if fix is None or not fix.valid:
            self.estimator.reset()  # do not bridge gaps with invented speed labels
        out['fix'], out['state'] = _plain(fix), _plain(state)
        if out['fix'] is not None:
            out['fix']['timestamp'] = now  # unify nested fix with the session clock
        history = self.estimator.samples
        speed_ready = state is not None and len(history) >= 3 and history[-1][0] - history[0][0] >= .3
        out['speed_estimate_ready'] = speed_ready
        if out['state'] is not None and not speed_ready:
            out['state']['speed_ms'] = None
        speed_kmh, speed_error = self._speed(image, client, view, now)
        if speed_kmh is not None:
            out['speed'] = dict(kmh=speed_kmh, source='hud_ocr', signed_ms=None)
        elif speed_ready:
            out['speed'] = dict(kmh=abs(state.speed_ms) * 3.6, source='minimap_estimate', signed_ms=state.speed_ms)
        out['speed_ocr_status'] = speed_error
        if state is not None:
            goal = self.cfg.destinations.get(self.map_id)
            out['destination'] = goal
            if goal:
                key = self.map_id, tuple(goal)
                try:
                    if self.route_key != key:
                        graph = RoadGraph(self.assets.roads)
                        points = graph.route([state.position, goal], kinds=self.cfg.truck.road_kinds,
                                             max_snap=self.cfg.truck.max_snap_px)
                        self.path = RoadPath(points)
                        self.route_key, self.progress = key, 0.
                        self.route_id += 1
                        out['route'] = dict(t=now, route_id=self.route_id, map_id=self.map_id,
                                            points=self.path.points.tolist(), raw_points=points, destination=goal,
                                            coordinate_frame='map_px', meters_per_pixel=self.assets.meters_per_pixel)
                    plan = ReferenceLocalPlanner(self.cfg.truck, self.assets.meters_per_pixel).plan(
                        self.path, state, progress_px=self.progress, route_id=self.route_id, now=now)
                    out['route_id'] = self.route_id
                    out['reference_route'] = dict(route_id=self.route_id, map_id=self.map_id,
                        destination=goal, coordinate_frame='map_px', points_file='routes.jsonl')
                    out['local_plan'] = _plain(plan)
                    if not plan.valid:
                        self.path = self.route_key = None
                        out['route_error'] = plan.reason
                        out['rule_control']['reason'] = plan.reason
                        return out
                    self.progress = plan.progress_m/self.assets.meters_per_pixel
                    out.update(remaining_m=plan.remaining_m, cross_track_m=plan.cross_track_m,
                               local_target_m=[plan.target_ego_m[0], -plan.target_ego_m[1]],
                               local_target_coordinate_frame='legacy_TCP_right_negative_forward_m', command=plan.command)
                    if speed_ready:
                        out['rule_control'] = shadow_rule_control(self.path,state,plan,self.cfg.truck,self.assets.meters_per_pixel,now)
                    else:
                        out['rule_control']['reason'] = 'speed_estimate_not_ready'
                except RouteError as exc:
                    self.path = self.route_key = None
                    out['route_error'] = str(exc)
            else:
                self.path = self.route_key = None
        return out

    def _speed(self, image, client, view, now):
        if view != self.last_view:
            self.last_speed, self.last_view = None, view
        if view not in VIEWS[:2]:
            return None, 'unknown_camera'
        region = getattr(self.cfg.collection, f'speed_{view}')
        crop = crop_region(image, client, region)
        if crop is None:
            return None, 'speed_digits_not_calibrated'
        try:
            from .hud import WindowsOcr, parse_number
            if self.ocr is None:
                self.ocr = WindowsOcr()
            text = self.ocr(cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
            number = parse_number(text)
            value = float(number) if number is not None else None
            if value is None or not 0 <= value <= 180:
                return None, 'unreadable'
            if self.last_speed is not None and abs(value - self.last_speed[1]) > 10 + 100 * (now - self.last_speed[0]):
                return None, 'jump_rejected'
            self.last_speed = now, value
            return value, 'ok'
        except Exception as exc:
            return None, f'ocr_unavailable: {type(exc).__name__}'


class DrivingCollector:
    def __init__(self, cfg, assets, *, view='unknown', vehicle='unknown', output=None, config_path=None):
        if view not in VIEWS:
            raise ValueError('invalid camera view')
        t = cfg.collection
        if not (1 <= t.fps <= 60 and 10 <= t.input_hz <= 500 and .1 <= t.nav_hz <= 20
                and t.queue_size >= 1 and t.max_width >= 64 and t.segment_s >= 1):
            raise ValueError('invalid collection rates, queue size or image size')
        self.cfg, self.assets = cfg, assets
        self.view, self.last_confirmed_view, self.vehicle = view, view, vehicle
        self.start_time = time.monotonic()
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.paused, self.transition_until = True, 0.
        self.keys, self.keys_at, self.foreground = set(), 0., False
        self.latest_nav, self.nav_queue = None, queue.Queue(maxsize=1)
        self.reference_routes = {}
        self.context_id = 0
        self.intervention_id = 0
        self.intervention_reason = 'obstacle'
        self.intervention = dict(active=False, episode_id=None, reason=None, started_t=None,
                                 source='human_annotation', actual_control_source='manual',
                                 semantics='manual_correction_demonstration; no autonomous handover')
        self.frame_id = 0
        self.exclusions = []
        self.require_panel_mask = False
        self.workers = []
        self.observer_args = cfg, assets, config_path
        directory = Path(output or t.directory) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
        metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), vehicle=vehicle, initial_view=view,
                        clock='session-relative time.monotonic seconds', read_only=True, config=asdict(cfg),
                        frame_target_fps=t.fps, input_poll_hz=t.input_hz, navigation_target_hz=t.nav_hz,
                        operator='manual', key_origin_unverified=True,
                        hierarchy_schema='reference_route -> local_plan -> shadow_rule_control',
                        ego_coordinates='right_forward_m; legacy local_target_m remains right_negative_forward_m',
                        rule_control='stateless counterfactual advice only; never applied; not expert ground truth',
                        intervention_semantics='explicit human correction annotation; all driving remains manual',
                        training_eligibility='Frames are candidates only; human review required for menus, seat changes, OCR and externally injected inputs')
        self.writer = DatasetWriter(directory, fps=t.fps, segment_s=t.segment_s, queue_size=t.queue_size, metadata=metadata)

    def now(self):
        return time.monotonic() - self.start_time

    def _invalidate_context(self):
        self.context_id += 1
        self.latest_nav = None
        self.transition_until = self.now() + .5

    def toggle_intervention(self, reason=None, *, source='panel'):
        reason = reason or self.intervention_reason
        if reason not in INTERVENTION_REASONS:
            raise ValueError('unknown intervention reason')
        with self.lock:
            now = self.now()
            if self.intervention['active']:
                self.writer.event(dict(t=now, type='intervention_end', **dict(self.intervention,active=False),
                                       marker_source=source, end_reason='human_marker'))
                self.intervention = dict(self.intervention, active=False)
            else:
                self.intervention_id += 1
                self.intervention = dict(self.intervention, active=True, episode_id=self.intervention_id,
                                         reason=reason, started_t=now)
                self.writer.event(dict(t=now, type='intervention_start', **self.intervention, marker_source=source))

    def mark_incident(self, kind, *, source='panel'):
        if kind not in ('obstacle', 'collision'):
            raise ValueError('unknown incident')
        with self.lock:
            self.writer.event(dict(t=self.now(), type='incident', kind=kind, source=source,
                frame_id=self.frame_id-1 if self.frame_id else None, context_id=self.context_id,
                episode_id=self.intervention.get('episode_id') if self.intervention['active'] else None,
                navigation_id=(self.latest_nav or {}).get('navigation_id')))

    def set_destination(self, map_id, point, *, pin_map=False):
        from .assets import load_map
        assets = load_map(map_id, self.cfg.assets_dir or self.assets.root)
        if point is not None and (len(point) != 2 or not all(math.isfinite(float(v)) for v in point) or not assets.contains(point)):
            raise ValueError('destination is outside the selected map')
        point = None if point is None else list(map(float,point))
        with self.lock:
            if point is None:
                self.cfg.destinations.pop(map_id,None)
            else:
                self.cfg.destinations[map_id] = point
            if pin_map:
                self.cfg.map_id, self.cfg.auto_map.enabled = map_id, False
            path = self.observer_args[2]
            if path:
                saved = Config.load(path)
                saved.destinations = dict(self.cfg.destinations)
                if pin_map:
                    saved.map_id, saved.auto_map.enabled = map_id, False
                saved.save(path)
            self._invalidate_context()
            self.writer.event(dict(t=self.now(), type='destination_changed', map_id=map_id,
                                   destination=point, pin_map=pin_map, context_id=self.context_id))

    def set_view(self, view):
        if view not in VIEWS:
            raise ValueError('invalid camera view')
        with self.lock:
            self.view = view
            if view != 'unknown':
                self.last_confirmed_view = view
            self._invalidate_context()
            self.writer.event(dict(t=self.now(), type='camera_label', camera_view=view))

    def toggle_view_label(self):
        self.set_view('third_person' if self.last_confirmed_view == 'first_person' else 'first_person')

    def set_vehicle(self, vehicle):
        with self.lock:
            self.vehicle = vehicle
            self._invalidate_context()
            self.writer.event(dict(t=self.now(), type='vehicle_label', vehicle=vehicle))

    def toggle_pause(self):
        with self.lock:
            self.paused = not self.paused
            self._invalidate_context()
            self.writer.event(dict(t=self.now(), type='pause', paused=self.paused))

    def start(self):
        for function in (self._inputs, self._navigation, self._capture):
            worker = threading.Thread(target=self._guard, args=(function,), name=function.__name__, daemon=True)
            self.workers.append(worker)
            worker.start()

    def _guard(self, function):
        try:
            function()
        except Exception as exc:
            self.writer.error = f'{function.__name__}: {type(exc).__name__}: {exc}'
            self.stop_event.set()

    def _inputs(self):
        from .input_win import key_is_down, foreground_client
        previous, previous_hotkeys = set(), set()
        last_focus = None
        polls, max_gap, last_poll = 0, 0., self.now()
        while not self.stop_event.is_set():
            started = time.monotonic()
            client = foreground_client(self.cfg.collection.process_name)
            foreground = bool(client and self.cfg.window_title.casefold() in client[1].casefold())
            down = {k for k in DRIVING_KEYS if key_is_down(k)} if foreground else set()
            hotkeys = {k for k in ('F7', 'F8', 'F9', 'F10', 'F11', 'F12') if key_is_down(k)} if foreground else set()
            now = self.now()
            polls += 1
            max_gap = max(max_gap, now - last_poll)
            last_poll = now
            with self.lock:
                self.keys, self.keys_at, self.foreground = down, now, foreground
                if foreground != last_focus:
                    self._invalidate_context()
                    self.writer.event(dict(t=now, type='focus', game_foreground=foreground))
                    last_focus = foreground
                for key in sorted(previous | down):
                    if (key in down) != (key in previous):
                        self.writer.event(dict(t=now, type='key', key=key, down=key in down,
                                               foreground=foreground, paused=self.paused))
                if 'C' in down - previous:
                    self.set_view('unknown')  # a request does not prove the camera changed
                if (down - previous) & {'ESC', 'M', 'I', 'TAB', 'F', '0'}:
                    self.paused = True
                    self._invalidate_context()
                    self.writer.event(dict(t=now, type='pause_for_ui_or_seat_change', keys=sorted(down - previous)))
                if 'F10' in hotkeys - previous_hotkeys:
                    self.toggle_pause()
                if 'F11' in hotkeys - previous_hotkeys:
                    self.toggle_view_label()
                if 'F12' in hotkeys - previous_hotkeys:
                    self.stop_event.set()
                if 'F8' in hotkeys - previous_hotkeys:
                    self.toggle_intervention(source='F8')
                if 'F7' in hotkeys - previous_hotkeys:
                    self.mark_incident('obstacle',source='F7')
                if 'F9' in hotkeys - previous_hotkeys:
                    self.mark_incident('collision',source='F9')
            previous, previous_hotkeys = down, hotkeys
            self.stop_event.wait(max(0, 1 / self.cfg.collection.input_hz - (time.monotonic() - started)))
        self.writer.event(dict(t=self.now(), type='input_poll_summary', polls=polls, max_gap_s=max_gap))

    def _navigation(self):
        observer = NavigationObserver(*self.observer_args)
        last_context, navigation_id = None, 0
        while not self.stop_event.is_set():
            try:
                image, client, view, t, frame_id, context_id = self.nav_queue.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                if context_id != last_context and hasattr(observer,'reset_temporal'):
                    observer.reset_temporal()
                last_context = context_id
                out = observer.observe(image, client, view, t)
                navigation_id += 1
                out.update(frame_id=frame_id, camera_view=view, processed_t=self.now(),
                           context_id=context_id, navigation_id=navigation_id)
                out['processing_latency_s'] = out['processed_t']-t
                if out.get('route'):
                    route = dict(out.pop('route'), navigation_id=navigation_id, context_id=context_id)
                    self.writer.route(route)
                    with self.lock:
                        self.reference_routes[route['route_id']] = route
                        if len(self.reference_routes)>16:
                            self.reference_routes.pop(next(iter(self.reference_routes)))
                self.writer.nav(out)
                with self.lock:
                    if context_id == self.context_id:
                        self.latest_nav = out
            except Exception as exc:
                self.writer.event(dict(t=self.now(), type='navigation_error', error=f'{type(exc).__name__}: {exc}'))

    def _capture(self):
        from .capture import ScreenCapture
        from .input_win import foreground_client
        cap = ScreenCapture()
        next_nav = 0.
        while not self.stop_event.is_set():
            started = time.monotonic()
            client_info = foreground_client(self.cfg.collection.process_name)
            with self.lock:
                paused = self.paused
                masks_ready = not self.require_panel_mask or bool(self.exclusions)
            preview_due = self.now() >= next_nav
            if (not paused or preview_due) and masks_ready and client_info and self.cfg.window_title.casefold() in client_info[1].casefold():
                handle, _, client = client_info
                with self.lock:
                    view, keys, keys_at, nav = self.view, sorted(self.keys), self.keys_at, self.latest_nav
                    vehicle = self.vehicle
                    context_id = self.context_id
                    intervention = dict(self.intervention)
                    transition_until = self.transition_until
                    exclusions = list(self.exclusions)
                    before = self.now()
                image = cap.grab(client)
                after = self.now()
                # Do not save a desktop frame when focus changed during capture.
                check = foreground_client(self.cfg.collection.process_name)
                if check and check[0] == handle and check[2] == client:
                    with self.lock:
                        if context_id != self.context_id or paused != self.paused:
                            self.writer.event(dict(t=before,type='frame_discarded',reason='collection_context_changed'))
                            continue
                        candidate = (view != 'unknown' and vehicle != 'unknown' and before >= transition_until
                                     and not {'ALT', 'TAB'} & set(keys)
                                     and 0 <= before - keys_at <= .1)
                    frame_id = None if paused else self.frame_id
                    if not paused:
                        self.frame_id += 1
                    if before >= next_nav:
                        packet = image, client, view, before, frame_id, context_id
                        try:
                            self.nav_queue.put_nowait(packet)
                        except queue.Full:
                            try:
                                self.nav_queue.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                self.nav_queue.put_nowait(packet)
                            except queue.Full:
                                pass
                        next_nav = before + 1 / self.cfg.collection.nav_hz
                    if not paused:
                        validity = navigation_validity(nav,view=view,context_id=context_id,now=before)
                        record = dict(frame_id=frame_id, t=before, capture_end_t=after, camera_view=view,
                                      vehicle=vehicle, context_id=context_id, client_region=asdict(client), observed_keys=keys,
                                      keys_t=keys_at, keys_age_s=before-keys_at,
                                      excluded_screen_regions=[asdict(r) for r in exclusions],
                                      training_candidate=candidate, navigation=nav, intervention=intervention,
                                      hierarchical_training_candidate=candidate and validity['local_plan_valid'] and validity['rule_control_valid'],
                                      label_validity=dict(camera_confirmed=view != 'unknown', input_fresh=0 <= before-keys_at <= .1, **validity),
                                      navigation_age_s=None if nav is None else before-nav['t'])
                        scale = min(1., self.cfg.collection.max_width / image.shape[1])
                        width = max(2, int(image.shape[1]*scale)//2*2)
                        height = max(2, int(image.shape[0]*scale)//2*2)
                        resized = cv2.resize(mask_regions(image,client,exclusions),(width,height),interpolation=cv2.INTER_AREA)
                        self.writer.submit(resized,record)
                else:
                    self.writer.event(dict(t=before, type='frame_discarded', reason='focus_or_geometry_changed'))
            if self.writer.error:
                self.stop_event.set()
            self.stop_event.wait(max(0, 1 / self.cfg.collection.fps - (time.monotonic() - started)))

    def close(self):
        self.stop_event.set()
        for worker in self.workers:
            worker.join(20)
            if worker.is_alive():
                raise RuntimeError(f'{worker.name} did not stop; leaving session incomplete')
        if self.intervention['active']:
            self.writer.event(dict(t=self.now(),type='intervention_end',**dict(self.intervention,active=False),
                                   marker_source='system',end_reason='session_closed_not_human_success'))
            self.intervention = dict(self.intervention,active=False)
        self.writer.event(dict(t=self.now(), type='session_end', observed_keys=sorted(self.keys),
                               camera_view=self.view, vehicle=self.vehicle))
        self.writer.close()


class CollectionPanel:
    def __init__(self, collector):
        import tkinter as tk
        from tkinter import ttk
        from .key_display import KeyStateDisplay
        from .panel import _no_activate
        from .ui_theme import (ACCENT, GREEN, SURFACE, MUTED, OpacityControl, window_theme,
                               style_widgets, dark_titlebar, save_opacity, ScrollPanel)
        self.c = collector
        self.root = root = tk.Tk()
        root.title('WARDOGS 手动驾驶采集（只读）')
        root.attributes('-topmost', True)
        root.protocol('WM_DELETE_WINDOW', collector.stop_event.set)
        window_theme(root, collector.cfg.ui.collection_opacity)
        root.resizable(True, True)
        body = tk.Frame(root)
        body.pack(fill='both',expand=True)
        map_host = tk.Frame(body)
        map_host.pack(side='left',fill='both',expand=True,padx=(4,0))
        rail = ScrollPanel(body, height=580)
        rail.pack(side='right',fill='y')
        controls = rail.body
        from .collection_map import CollectionMapPicker
        self.map_picker = CollectionMapPicker(map_host,collector,embedded=True)
        brand = tk.Label(controls, text='WARDOGS  /  驾驶数据采集', font=('Microsoft YaHei UI', 13, 'bold'))
        brand.pack(anchor='w', padx=14, pady=(12, 4))
        self.record_badge = tk.Label(controls, text='PAUSED  /  暂停', font=('Microsoft YaHei UI', 10, 'bold'), padx=12, pady=4)
        self.record_badge.pack(fill='x', padx=12, pady=4)
        self.keys = KeyStateDisplay(controls, observed=lambda: set(collector.keys))
        self.keys.pack(fill='x', padx=12, pady=4)
        vehicle_bar = tk.Frame(controls)
        vehicle_bar.pack()
        tk.Label(vehicle_bar, text='车型标签：').pack(side='left')
        vehicle = tk.StringVar(value=collector.vehicle)
        tk.OptionMenu(vehicle_bar, vehicle, *dict.fromkeys([collector.vehicle, 'KODIAK', 'URAL', 'URAL_DEFENDER', 'DUNE_BUGGY', 'unknown']),
                      command=collector.set_vehicle).pack(side='left')
        bar = tk.Frame(controls)
        bar.pack(padx=6, pady=5)
        tk.Button(bar, text='开始 / 暂停 (F10)', bg='#2e7d32', fg='white', command=collector.toggle_pause).pack(side='left', padx=3)
        self.first_button = tk.Button(bar, text='第一视角', command=lambda: collector.set_view('first_person'))
        self.first_button.pack(side='left', padx=3)
        self.third_button = tk.Button(bar, text='第三视角', command=lambda: collector.set_view('third_person'))
        self.third_button.pack(side='left', padx=3)
        tk.Button(bar, text='结束保存 (F12)', bg='#c62828', fg='white', command=collector.stop_event.set).pack(side='left', padx=3)
        correction = tk.Frame(controls)
        correction.pack(fill='x',padx=12,pady=4)
        tk.Label(correction,text='纠偏原因：').pack(side='left')
        reason_labels = dict(zip(INTERVENTION_REASONS,('避障','路线纠偏','定位错误','紧急停车','其他')))
        reason_keys = {label:key for key,label in reason_labels.items()}
        reason = tk.StringVar(value=reason_labels[collector.intervention_reason])
        reasons = ttk.Combobox(correction,textvariable=reason,values=list(reason_keys),state='readonly',width=14)
        reasons.pack(side='left')
        reasons.bind('<<ComboboxSelected>>',lambda event:setattr(collector,'intervention_reason',reason_keys[reason.get()]))
        self.intervention_button = tk.Button(correction,text='开始 / 结束纠偏 (F8)',command=collector.toggle_intervention)
        self.intervention_button.pack(side='left',padx=5)
        incidents = tk.Frame(controls)
        incidents.pack(fill='x',padx=12,pady=3)
        tk.Button(incidents,text='障碍标记 (F7)',command=lambda:collector.mark_incident('obstacle')).pack(side='left')
        tk.Button(incidents,text='碰撞标记 (F9)',command=lambda:collector.mark_incident('collision')).pack(side='left',padx=6)
        self.label = tk.Label(controls, justify='left', anchor='w', font=('Microsoft YaHei UI', 10), padx=10, pady=8, wraplength=560)
        self.label.pack(fill='x', padx=12, pady=5)
        tk.Label(controls, text='全程手动驾驶；规则建议不发送按键。F8 标记人工纠偏段。\nF11 切换视角标签；菜单/换座位后需 F10 恢复。请退出其他自动驾驶程序。', fg='#777').pack(pady=4)
        self.opacity = OpacityControl(controls, root, collector.cfg.ui.collection_opacity,
            on_save=lambda value: save_opacity(collector.cfg, 'collection_opacity', value,
                                               path=collector.observer_args[2]))
        self.opacity.pack(fill='x', padx=12, pady=(2, 10))
        style_widgets(root)
        brand.configure(fg=ACCENT)
        dark_titlebar(root)
        root.update_idletasks()
        rail.canvas.configure(width=controls.winfo_reqwidth())
        rail.bind_wheel()
        root.minsize(controls.winfo_reqwidth() + 360, 420)
        _no_activate(root)
        root.geometry(f'+{max(0, root.winfo_screenwidth() - root.winfo_reqwidth() - 20)}+30')
        root.after(100, self.refresh)

    def open_map(self):
        self.map_picker.win.focus_set()

    def refresh(self):
        c = self.c
        from .ui_theme import SURFACE, MUTED, TEXT, GREEN, BLUE, CARD
        if c.stop_event.is_set():
            self.root.destroy()
            return
        with c.lock:
            # Include the window border/title as well as the client area.
            c.exclusions = [Region(self.root.winfo_rootx() - 10, self.root.winfo_rooty() - 40,
                                   self.root.winfo_width() + 20, self.root.winfo_height() + 50)]
        nav = c.latest_nav or {}
        fresh = nav.get('t') is not None and 0 <= c.now() - nav['t'] <= 1 and nav.get('camera_view') == c.view
        speed = nav.get('speed') if fresh else None
        speed_text = '--' if speed is None else f'{speed["kmh"]:.1f} km/h ({speed["source"]})'
        views = {'first_person': '第一视角', 'third_person': '第三视角', 'unknown': '未确认视角'}
        self.record_badge.configure(text='PAUSED  /  暂停' if c.paused else 'REC  /  录制中',
                                    bg=SURFACE if c.paused else GREEN, fg=MUTED if c.paused else TEXT)
        self.first_button.configure(bg=BLUE if c.view == 'first_person' else CARD)
        self.third_button.configure(bg=BLUE if c.view == 'third_person' else CARD)
        plan, rule = nav.get('local_plan') or {}, nav.get('rule_control') or {}
        ready = fresh and c.view in VIEWS[:2] and c.vehicle != 'unknown' and plan.get('valid') and rule.get('valid')
        if c.view == 'unknown':
            readiness = '请确认第一 / 第三视角'
        elif c.vehicle == 'unknown':
            readiness = '请确认车型'
        elif not fresh:
            readiness = '等待游戏前台与导航更新'
        elif not nav.get('state'):
            readiness = '等待可靠定位和朝向，可先手动缓行数米'
        elif nav.get('destination') is None:
            readiness = '请在地图上设置目的地'
        elif not plan.get('valid'):
            readiness = '参考路线尚未就绪，请检查定位和目的地'
        elif not rule.get('valid'):
            readiness = '等待稳定速度估计，可先手动缓行数米'
        else:
            readiness = '就绪（路线参考，视觉避障待接入）'
        axes = rule.get('axes') or {}
        advice = f'转向 {axes["steer"]:+.2f} / 前后 {axes["throttle"]:+.2f}' if ready else '--'
        self.intervention_button.configure(text='结束纠偏 (F8)' if c.intervention['active'] else '开始纠偏 (F8)',
                                            bg=BLUE if c.intervention['active'] else CARD)
        self.label.config(text=f'{"暂停" if c.paused else "录制中"} · {views[c.view]} · {c.vehicle}\n'
                          f'游戏前台: {c.foreground}  写入: {c.writer.written} 帧  丢帧: {c.writer.dropped}\n'
                          f'速度: {speed_text}\n地图: {nav.get("map_id") or "识别中"}  导航: {nav.get("command") or "--"}\n'
                          f'分层标签: {readiness}\n'
                          f'规则建议（不发送）: {advice}\n'
                          f'人工纠偏: {c.intervention["reason"] if c.intervention["active"] else "未标记"}\n'
                          f'文件夹: {c.writer.directory.name}')
        self.root.after(100, self.refresh)

    def run(self):
        self.root.mainloop()


def run_collection(cfg, assets, *, view, vehicle, output=None, config_path=None, seconds=None, panel=True):
    from .input_win import set_dpi_aware
    set_dpi_aware()
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        raise ValueError('seconds must be positive and finite')
    collector = DrivingCollector(cfg, assets, view=view, vehicle=vehicle, output=output, config_path=config_path)
    collector.require_panel_mask = panel
    print(f'Dataset: {collector.writer.directory.resolve()}\nF10 start/pause; F11 view label; F12 finish; F8 correction episode; F7 obstacle; F9 collision. Initially paused; navigation preview remains active.')
    if seconds is not None:
        timer = threading.Timer(seconds, collector.stop_event.set)
        timer.daemon = True
        timer.start()
    collector.start()
    try:
        if panel:
            CollectionPanel(collector).run()
        else:
            while not collector.stop_event.wait(.2):
                pass
    except KeyboardInterrupt:
        collector.stop_event.set()
    finally:
        collector.close()
    print(f'Saved {collector.writer.written} frames; dropped={collector.writer.dropped}; error={collector.writer.error}')


def inspect_dataset(directory):
    """Check indexes and segment readability without loading all video into RAM."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    problems, counts, views, intervals = [], {}, {}, []
    coverage = dict(reference_route_valid=0, local_plan_valid=0, rule_control_valid=0,
                    hierarchical_candidates=0, intervention_frames=0)
    route_path = directory / 'routes.jsonl'
    routes = [json.loads(line) for line in route_path.read_text(encoding='utf-8').splitlines()] if route_path.exists() else []
    route_lookup = {r['route_id']:r for r in routes}
    if len(route_lookup) != len(routes):
        problems.append('duplicate reference route ID')
    if manifest.get('schema_version',1) >= 2 and manifest.get('route_records') != len(routes):
        problems.append('manifest/route-index count mismatch')
    previous_id, previous_t, count = -1, None, 0
    def records():
        with open(directory / 'frames.jsonl', encoding='utf-8') as file:
            for line in file:
                yield json.loads(line)
    for row in records():
        name = row['video']
        if (directory / name).resolve().parent != directory:
            problems.append('video path outside session')
            continue
        expected = counts.get(name, 0)
        if row['video_frame'] != expected:
            problems.append(f'{name}: noncontiguous video index')
        counts[name] = expected + 1
        if row['frame_id'] <= previous_id:
            problems.append('nonmonotonic frame id')
        previous_id = row['frame_id']
        if row['capture_end_t'] < row['t'] or (previous_t is not None and row['t'] < previous_t):
            problems.append('nonmonotonic capture time')
        if previous_t is not None and row['t'] > previous_t:
            intervals.append(row['t'] - previous_t)
        previous_t = row['t']
        count += 1
        views[row['camera_view']] = views.get(row['camera_view'], 0) + 1
        validity = row.get('label_validity',{})
        for key in ('reference_route_valid','local_plan_valid','rule_control_valid'):
            coverage[key] += bool(validity.get(key))
        coverage['hierarchical_candidates'] += bool(row.get('hierarchical_training_candidate'))
        coverage['intervention_frames'] += bool((row.get('intervention') or {}).get('active'))
        if validity.get('reference_route_valid'):
            nav = row.get('navigation') or {}
            ref = nav.get('reference_route') or {}
            route = route_lookup.get(ref.get('route_id'))
            if not route or route['map_id'] != nav.get('map_id'):
                problems.append('frame references missing or mismatched route')
            actual = navigation_validity(nav,view=row['camera_view'],context_id=row.get('context_id'),now=row['t'])
            if not actual['reference_route_valid']:
                problems.append('claimed valid route has stale/future/mismatched navigation')
    for name, expected in counts.items():
        cap = cv2.VideoCapture(str(directory / name))
        try:
            actual = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            first_ok, _ = cap.read()
            if expected == 1:
                last_ok = first_ok
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, expected - 1)
                last_ok, _ = cap.read()
            if actual != expected or not first_ok or not last_ok:
                problems.append(f'{name}: encoded frames={actual}, indexed={expected}, endpoints readable={first_ok}/{last_ok}')
        finally:
            cap.release()
    if count != manifest['frames_written']:
        problems.append('manifest/frame-index count mismatch')
    if not manifest['complete']:
        problems.append('session incomplete or still recording')
    if not count:
        problems.append('no frames recorded')
    events_path = directory / 'events.jsonl'
    events = [json.loads(line) for line in events_path.read_text(encoding='utf-8').splitlines()] if events_path.exists() else []
    event_counts = {kind:sum(e.get('type')==kind for e in events) for kind in ('intervention_start','intervention_end','incident')}
    warnings = []
    if not coverage['hierarchical_candidates']:
        warnings.append('No valid hierarchical training candidates; check view, vehicle, map, destination and speed readiness')
    return dict(directory=str(directory), indexed_frames=count, video_segments=len(counts),
                reference_routes=len(routes), hierarchical_coverage=coverage, annotation_events=event_counts, warnings=warnings,
                dropped_frames=manifest['frames_dropped'], camera_views=views,
                median_sample_fps=None if not intervals else round(1 / statistics.median(intervals), 2),
                timing_note='Median sample rate; gaps can include pauses and loss of game focus. Use frame timestamps, not AVI time.',
                problems=problems, ok=not problems)
