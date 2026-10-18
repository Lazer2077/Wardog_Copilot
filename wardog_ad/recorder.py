"""Flight data recorder: one JSON line per control tick in logs/, for diagnosing real runs."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import time


def _plain(value):
    if is_dataclass(value):
        return {k: _plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float):
        return round(value, 3)
    if hasattr(value, 'item'):          # numpy scalar
        return _plain(value.item())
    return value


class FlightRecorder:
    def __init__(self, directory: str | Path = 'logs'):
        Path(directory).mkdir(exist_ok=True)
        self.path = Path(directory) / time.strftime('flight_%Y%m%d_%H%M%S.jsonl')
        self.file = open(self.path, 'w', encoding='utf-8')

    def write(self, pilot, tel, now: float, tick_s: float):
        from .input_win import key_is_down
        fix = tel.fix
        record = {
            't': round(now, 3), 'tick_ms': round(tick_s * 1000), 'profile': pilot.cfg.profile,
            'map_id': pilot.cfg.map_id, 'map_status': tel.map_status,
            'camera_view': pilot.cfg.truck.camera_view if pilot.cfg.profile == 'truck' else None,
            'engaged': pilot.engaged, 'foreground': pilot.foreground_ok, 'mission': pilot.mission,
            'actuator_enabled': pilot.actuator.enabled,
            'actuator_values': dict(pilot.actuator.values),
            'goal': pilot.goal, 'destination': pilot.destination,
            'route_points': len(tel.route),
            'status': tel.status, 'message': tel.message,
            'fix': None if fix is None else {'valid': fix.valid, 'x': fix.x, 'y': fix.y, 'heading': fix.heading,
                                             'conf': fix.confidence, 'reason': fix.reason,
                                             'inliers': fix.inliers, 'error': fix.error,
                                             'scale': fix.scale, 'rotation': fix.rotation,
                                             'timestamp': fix.timestamp,
                                             'age_s': max(0., now - fix.timestamp)},
            'minimap_region': pilot.cfg.localization_region, 'minimap_anchor': pilot.cfg.localization_anchor,
            'manual_position': getattr(pilot, 'manual_position', None),
            'state': tel.state, 'hud': tel.hud,
            'axes': None if tel.output is None else tel.output.axes,
            'info': None if tel.output is None else tel.output.info,
            'keys': sorted(pilot.keyboard.down), 'lock': tel.lock_state, 'flares': tel.flares,
            'observed_wasd': [key for key in 'WASD' if key_is_down(key)],
            'fast': [],                                          # [t, pitch, pitch_target, cyclic] @30 Hz
        }
        if hasattr(pilot, 'fast_samples'):
            with pilot.lock:
                record['fast'] = list(pilot.fast_samples)
                pilot.fast_samples.clear()
        self.file.write(json.dumps(_plain(record), ensure_ascii=False) + '\n')
        self.file.flush()

    def close(self):
        self.file.close()
