"""Supervisor: perception -> estimation -> control -> keys, with safety gating.

The same ``Autopilot`` runs against the real screen/keyboard (``run_live``) and
against the simulator (``sim.run_sim``); only the ``Sensors`` object and the
keyboard differ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Protocol

import cv2
import numpy as np

from .actuator import Axis, PWMActuator
from .airnav import AirNavigator
from .config import Config
from .controllers import AircraftPilot, HeliPilot, Output, TruckDriver, TruckRecoveryState
from .estimator import StateEstimator, VehicleState
from .localization import Fix, MinimapLocalizer
from .perception import FlareController, MissileLockDetector, MotionMonitor
from .planner import Avoid, Path, RoadGraph, RouteError, straight_route

log = logging.getLogger('wardog_ad')


class Sensors(Protocol):
    def minimap(self) -> np.ndarray | None: ...
    def main_view(self) -> np.ndarray | None: ...
    def lock_roi(self) -> np.ndarray | None: ...
    def audio(self) -> np.ndarray | None: ...      # new mono 48 kHz samples since last call
    def hud(self): ...                               # HudReading | None (helicopter)
    def attitude(self): ...                          # pitch ticks | None, cheap (fast loop)


class FixSource(Protocol):
    def locate(self, image, anchor, north_up) -> Fix: ...


def build_axes(cfg: Config) -> dict[str, Axis]:
    k = cfg.keys
    if cfg.profile == 'aircraft' and cfg.aircraft.flight_model == 'helicopter':
        p = cfg.aircraft.pwm_period_s
        return {'collective': Axis(k.throttle_down, k.throttle_up, p),
                'cyclic': Axis(k.pitch_up, k.pitch_down, cfg.aircraft.cyclic_pwm_s),   # + = W (nose down)
                'yaw': Axis(k.yaw_left, k.yaw_right, p)}
    if cfg.profile == 'aircraft':
        p = cfg.aircraft.pwm_period_s
        return {'roll': Axis(k.roll_left, k.roll_right, p), 'yaw': Axis(k.yaw_left, k.yaw_right, p),
                'pitch': Axis(k.pitch_down, k.pitch_up, p)}
    p = cfg.truck.pwm_period_s
    return {'steer': Axis(k.left, k.right, p), 'throttle': Axis(k.backward, k.forward, p, deadband=.08)}


@dataclass
class Telemetry:
    status: str = 'standby'
    engaged: bool = False
    fix: Fix | None = None
    state: VehicleState | None = None
    output: Output | None = None
    locked: bool = False
    lock_state: str = 'none'        # none | acquiring | locked
    missile_heard: bool = False     # incoming-missile sound detected
    flares: int = 0
    message: str = ''
    route: list = field(default_factory=list)
    hud: object = None              # HudReading (helicopter)
    nav_sigma: float | None = None  # helicopter dead-reckoning uncertainty (m)
    mission: str = 'idle'           # idle | outbound | return
    home: tuple | None = None
    destination: tuple | None = None
    remaining_m: float | None = None
    map_status: str = ''
    speed_source: str = 'hud'       # helicopter: 'hud' | 'fixes' (HUD digits disagreed)


class Autopilot:
    def __init__(self, cfg: Config, assets, keyboard, sensors: Sensors, *, locator: FixSource | None = None,
                 graph: RoadGraph | None = None, lock_profile=None):
        self.cfg, self.assets, self.keyboard, self.sensors = cfg, assets, keyboard, sensors
        self.mpp = assets.meters_per_pixel
        self.locator = locator or MinimapLocalizer.from_assets(assets)
        self.estimator = StateEstimator(self.mpp)
        self.airnav = AirNavigator(self.mpp, drift=cfg.aircraft.dr_drift, min_conf=cfg.aircraft.fix_min_conf,
                                   align=cfg.aircraft.slip_align)
        self.last_fix_try = -1e9
        self._loc_lock = threading.Lock()
        self._control_lock = threading.RLock()
        self._map_generation = 0
        self.fix_worker = None
        self._needs_plan = False
        self.actuator = PWMActuator(keyboard, build_axes(cfg))
        self.motion = MotionMonitor()
        self.graph = graph
        self.config_path = None         # set by the CLI: profile switches are remembered
        self.controller: TruckDriver | AircraftPilot | None = None
        self.goal: tuple | None = None
        self.waypoints: list = []
        self.avoid: list[Avoid] = []
        self.engaged = False
        self.pending_go = False
        self.mission = 'idle'           # idle | outbound | return
        self.arrived = False
        dest, home = cfg.destinations.get(cfg.map_id), cfg.homes.get(cfg.map_id)
        self.destination: tuple | None = tuple(dest) if dest else None
        self.home: tuple | None = tuple(home) if home else None
        self.via: list = []             # waypoints of the outbound leg (flown reversed on return)
        self._capture_home = False
        self.foreground_ok = True
        self.last_valid = -1e9
        self.speed_mismatches = 0
        self._last_ground = None
        self._last_mismatch_log = -1e9
        self.last_speed = 0.0
        self.telemetry = Telemetry()
        self.manual_position = None
        self.lock = threading.RLock()
        self.detector = self.audio_detector = self.flares = None
        self.countermeasures_on = True
        self.lock_profile = lock_profile
        self.last_hud = -1e9
        self._hud_grace = False
        self.fast_samples: list = []
        self.last_fast = -1e9           # last fast-loop pitch update    # (t, pitch, pitch_target, cyclic) since the last log line
        self._apply_profile()
        self.map_manager = None        # enabled by run_live, truck only
        self.truck_recovery = TruckRecoveryState()

    def _switch_map(self, assets, locator):
        """Discard all old-map coordinates before allowing another mission."""
        with self.lock:
            self.disengage(f'map changed to {assets.name}; press GO to start a new mission')
            graph = RoadGraph(assets.roads)
            self.assets, self.locator = assets, locator
            self.truck_recovery = TruckRecoveryState()
            self.cfg.map_id = assets.map_id
            self.mpp = assets.meters_per_pixel
            self.graph = graph
            self.estimator = StateEstimator(self.mpp)
            self.airnav = AirNavigator(self.mpp, drift=self.cfg.aircraft.dr_drift,
                                       min_conf=self.cfg.aircraft.fix_min_conf,
                                       align=self.cfg.aircraft.slip_align)
            self.controller = self.goal = None
            self.waypoints, self.via, self.avoid = [], [], []
            self.mission, self.arrived = 'idle', False
            self._capture_home = self._needs_plan = False
            dest, home = self.cfg.destinations.get(assets.map_id), self.cfg.homes.get(assets.map_id)
            self.destination = tuple(dest) if dest else None
            self.home = tuple(home) if home else None
            self.last_valid, self.last_fix_try, self.last_speed = -1e9, -1e9, 0.0
            self.motion = MotionMonitor()
            self.telemetry = Telemetry(message=self.telemetry.message)
            self.manual_position = None
            self._map_generation += 1
            self._fix_latest = None
            log.warning('map switched to %s; old mission cleared', assets.map_id)
        if getattr(self, 'background_open', False):
            self.open_ground()                # build the new map's open-ground map now

    def set_map_mode(self, map_id=None):
        """None enables map identification; a map id pins localization to that map."""
        from .assets import load_map
        from .map_manager import MapManager
        self.disengage('changing map; press GO again after localization')
        assets, locator = self.assets, self.locator
        if map_id is not None and map_id != assets.map_id:
            # Validate before replacing the current map. Loading failures leave its
            # settings and coordinates intact, with driving paused.
            assets = load_map(map_id, self.assets.root)
            assets.roads
            assets.load_image()
            locator = MinimapLocalizer.from_assets(assets)
        with self._control_lock, self._loc_lock:
            if assets.map_id != self.assets.map_id:
                self._switch_map(assets, locator)
            self.map_manager = MapManager(self.assets, self.locator, self.cfg.auto_map) if map_id is None else None
            automatic = map_id is None
            def remember(c):
                c.map_id = self.assets.map_id
                c.auto_map.enabled = automatic
            self._remember(remember)
            self.telemetry.map_status = 'identifying map' if automatic else f'manual map: {self.assets.map_id}'

    def set_initial_position(self, point, *, source='map click', map_id=None, sigma_m=200.):
        """Seed navigation from a user statement without inventing a visual fix/heading."""
        from .bases import ManualPosition
        with self._control_lock, self._loc_lock, self.lock:
            if map_id is not None and map_id != self.cfg.map_id:
                raise ValueError('map changed; select an initial position on the current map')
            if not self.assets.contains(point) or not np.isfinite(sigma_m) or sigma_m <= 0:
                raise ValueError('invalid initial position or uncertainty')
            self.disengage('manual initial position set; waiting for heading/localization')
            now = time.monotonic()
            self.manual_position = ManualPosition(self.cfg.map_id, tuple(map(float, point)), now, source, sigma_m)
            self._map_generation += 1
            self._fix_latest = None
            self.estimator.reset()
            self.airnav.reset()
            if self.cfg.profile == 'aircraft':
                self.airnav.start(point, now, sigma_m=sigma_m, source='manual initial position')
            if hasattr(self.locator, 'set_search_prior'):
                self.locator.set_search_prior(point)
            self.goal = self.controller = None
            self.mission, self.arrived = 'idle', False
            self.telemetry.fix = self.telemetry.state = self.telemetry.output = None
            self.telemetry.route = []
            self.last_valid, self.last_speed = -1e9, 0.
            self._needs_plan, self._capture_home = False, False

    def set_base_position(self, faction_id, point, *, map_id=None):
        from .bases import team_bases
        with self._control_lock, self.lock:
            if map_id is not None and map_id != self.cfg.map_id:
                raise ValueError('map changed; select the base again')
            if faction_id not in {b.id for b in team_bases(self.assets, self.cfg)} or not self.assets.contains(point):
                raise ValueError('invalid base or map coordinates')
            active_map = self.cfg.map_id
            def remember(c):
                c.base_overrides.setdefault(active_map, {})[faction_id] = list(map(float, point))
            self._remember(remember)

    # -- vehicle profile -------------------------------------------------
    def _apply_profile(self):
        """(Re)build everything that depends on truck vs helicopter."""
        cfg = self.cfg
        self.actuator.release()
        self.actuator = PWMActuator(self.keyboard, build_axes(cfg))
        self.controller = None
        self.goal = None
        self.waypoints = []
        self.mission, self.arrived = 'idle', False
        self.pending_go = False
        self._needs_plan = self._capture_home = False
        self._map_generation += 1
        self._fix_latest = None
        self.last_valid, self.last_speed = -1e9, 0.
        self.estimator.reset()
        self.airnav.reset()
        self.manual_position = None
        self.telemetry.fix = self.telemetry.state = None
        self.telemetry.route = []
        self.telemetry.output = None
        self.telemetry.remaining_m = None
        self.telemetry.mission, self.telemetry.status = 'idle','standby'
        if cfg.profile == 'truck' and self.graph is None:
            self.graph = RoadGraph(self.assets.roads)
        self.detector = self.audio_detector = self.flares = None
        if cfg.profile != 'aircraft':
            return
        source = cfg.aircraft.lock_source
        if source in ('visual', 'both'):
            if cfg.lock_warning.valid:
                self.detector = MissileLockDetector(cfg.aircraft, cfg.lock_template)
            else:
                log.warning('visual lock warning region not calibrated: visual lock detection off')
        if source in ('audio', 'both'):
            from .audio import AudioLockDetector, LockProfile
            if self.lock_profile is None and cfg.audio_profile:
                self.lock_profile = LockProfile.load(cfg.audio_profile)
            if self.lock_profile is not None:
                self.audio_detector = AudioLockDetector(self.lock_profile)
            else:
                log.warning('lock tones not recorded (audio-record / audio-fit): AUTO-FLARES OFF, press %s yourself',
                            cfg.keys.flare)
        self.flares = FlareController(cfg.aircraft, self._fire_flare)

    def set_profile(self, profile: str):
        """Switch between 'truck' and 'aircraft' (helicopter) at run time."""
        with self._control_lock, self.lock:
            if profile == self.cfg.profile:
                return
            self.disengage(f'switched to {profile}')
            self.cfg.profile = profile
            self._apply_profile()
            if self.config_path:
                try:
                    from .config import Config
                    saved = Config.load(self.config_path)
                    saved.profile = profile
                    saved.save(self.config_path)
                except Exception:
                    log.exception('could not remember the profile')
            log.warning('vehicle: %s', 'HELICOPTER' if profile == 'aircraft' else 'TRUCK')

    # -- destination / return point / favourites (from the map picker) ---------
    def _remember(self, change):
        """Apply ``change(cfg)`` to the running config and to config.json (if known)."""
        change(self.cfg)
        if self.config_path:
            try:
                from .config import Config
                saved = Config.load(self.config_path)
                change(saved)
                saved.save(self.config_path)
            except Exception:
                log.exception('could not save the settings')

    def set_destination(self, point):
        point = (float(point[0]), float(point[1]))
        with self.lock:
            self.destination = point
            m = self.cfg.map_id
            self._remember(lambda c: c.destinations.__setitem__(m, [round(point[0], 2), round(point[1], 2)]))
            if self.mission == 'outbound' and not self.arrived:
                self.set_goal(point, self.via)          # re-plan the running trip
            elif self.cfg.profile == 'truck' and not self.engaged and self.mission == 'idle':
                self.goal, self.controller = point, None
                self.waypoints = [tuple(p) for p in self.via]
                self.telemetry.route = []
        log.warning('destination set to %s', tuple(round(c) for c in point))

    def set_home(self, point):
        point = (float(point[0]), float(point[1]))
        with self.lock:
            self.home = point
            self._capture_home = False
            m = self.cfg.map_id
            self._remember(lambda c: c.homes.__setitem__(m, [round(point[0], 2), round(point[1], 2)]))
            if self.mission == 'return' and not self.arrived:
                self.set_goal(point, list(reversed(self.via)))
        log.warning('home set to %s', tuple(round(c) for c in point))

    def add_favorite(self, name, point):
        entry = {'name': name, 'point': [round(float(point[0]), 2), round(float(point[1]), 2)]}
        m = self.cfg.map_id

        def change(c):
            favs = [f for f in c.favorites.get(m, []) if f['name'] != name]
            c.favorites[m] = favs + [entry]
        self._remember(change)

    def remove_favorite(self, name):
        m = self.cfg.map_id

        def change(c):
            c.favorites[m] = [f for f in c.favorites.get(m, []) if f['name'] != name]
        self._remember(change)

    def set_cruise_height(self, meters: float):
        self._remember(lambda c: setattr(c.aircraft, 'cruise_agl_m', float(meters)))
        log.warning('helicopter cruise height %.0f m', meters)

    def terrain_memory(self):
        """Ground heights learned on earlier flights over this map (terrain/<map>.npz)."""
        from .terrain import TerrainMemory
        if getattr(self, '_terrain', None) is None or self._terrain_map != self.cfg.map_id:
            path = None
            if self.config_path:
                from pathlib import Path
                path = Path(self.config_path).resolve().parent / 'terrain' / f'{self.cfg.map_id}.npz'
            self._terrain, self._terrain_map = TerrainMemory(path), self.cfg.map_id
        return self._terrain

    def _velocity(self, now):
        """Measured ground velocity (minimap fixes); without fixes the slide model's estimate."""
        v = self.airnav.ground_velocity(now)
        return v if v is not None else self.airnav.velocity_estimate()

    def open_ground(self):
        """Open-ground cost map of this map (built once per map from the base image, ~0.5 s).
        Live it is built in the background and None until ready; in the simulator at once."""
        from .openground import OpenGround
        key = self.cfg.map_id
        built = getattr(self, '_open', None)
        if built is not None and built[0] == key:
            return built[1]
        if getattr(self, '_open_building', None) == key:
            return None
        assets, mpp = self.assets, self.mpp

        def build():
            try:
                self._open = (key, OpenGround(assets.load_image(), mpp))
            except Exception:
                log.exception('open-ground map unavailable')
                self._open = (key, None)
            self._open_building = None
        self._open_building = key
        if getattr(self, 'background_open', False):
            threading.Thread(target=build, name='open-ground', daemon=True).start()
            return None
        build()
        return self._open[1]

    def set_cruise_speed(self, kmh: float):
        self._remember(lambda c: setattr(c.aircraft, 'cruise_kmh', float(kmh)))
        log.warning('helicopter cruise speed %.0f km/h', kmh)

    def toggle_profile(self):
        self.set_profile('truck' if self.cfg.profile == 'aircraft' else 'aircraft')

    # -- mission ---------------------------------------------------------
    def set_goal(self, goal, waypoints=(), start=None):
        """Store the destination; the route is planned from ``start`` or the first good fix."""
        with self.lock:
            self.goal, self.waypoints = tuple(goal), [tuple(w) for w in waypoints]
            if isinstance(self.controller, HeliPilot):
                self._needs_plan = True     # keep the flight state, only change the route
            else:
                self.controller = None
            if start is not None:
                self._plan(tuple(start))

    def _plan(self, start):
        anchors = [start, *self.waypoints, self.goal]
        if self.cfg.profile == 'aircraft':
            points = straight_route(anchors)
            if isinstance(self.controller, HeliPilot):
                self.controller.set_waypoints(points)
            else:
                pilot = HeliPilot if self.cfg.aircraft.flight_model == 'helicopter' else AircraftPilot
                self.controller = pilot(points, self.cfg.aircraft, self.mpp)
            if isinstance(self.controller, HeliPilot):
                self.controller.terrain = self.terrain_memory()
                self.controller.open_ground = self.open_ground   # asked when the final approach starts
                self.controller.velocity = self._velocity
            self._needs_plan = False
        else:
            t = self.cfg.truck
            points = self.graph.route(anchors, kinds=t.road_kinds, max_snap=t.max_snap_px, avoid=self.avoid)
            # leave the first road point from where we actually are
            points = [start, *points] if np.hypot(points[0][0] - start[0], points[0][1] - start[1]) > 2 else points
            path = Path(points)
            if isinstance(self.controller, TruckDriver):
                self.controller.set_path(path)
            else:
                self.controller = TruckDriver(path, t, self.mpp, recovery=self.truck_recovery)
        self.telemetry.route = [tuple(p) for p in points]
        log.info('route planned: %d points', len(points))

    # -- one-button missions ----------------------------------------------
    def go(self, destination=None):
        """Drive to the destination.  The first press also remembers home (current spot)."""
        with self.lock:
            if destination is not None:
                self.destination = tuple(destination)
            if self.destination is None:
                self.telemetry.message = 'no destination: run "set-dest" or pass --to'
                log.warning(self.telemetry.message)
                return False
            if self.home is None:
                self._capture_home = True
            self.mission, self.arrived = 'outbound', False
            self.set_goal(self.destination, self.via)
            self.countermeasures_on = True
            self.engage()
            log.warning('GO -> destination %s', tuple(round(c) for c in self.destination))
            return self.engaged

    def return_home(self):
        """Drive back to the home point."""
        with self.lock:
            if self.home is None:
                self.telemetry.message = 'no home yet: press go once (or set-home) first'
                log.warning(self.telemetry.message)
                return False
            self._capture_home = False
            self.mission, self.arrived = 'return', False
            self.set_goal(self.home, list(reversed(self.via)))
            self.countermeasures_on = True
            self.engage()
            log.warning('RETURN -> home %s', tuple(round(c) for c in self.home))
            return self.engaged

    def land_here(self):
        """Helicopter: stop, hover and land on the spot.  Works on or off a mission and
        needs only the HUD (AGL, speed, pitch), not the position."""
        with self.lock:
            if self.cfg.profile != 'aircraft' or self.cfg.aircraft.flight_model != 'helicopter':
                self.telemetry.message = 'land is for the helicopter'
                return False
            st = self.telemetry.state
            here = tuple(st.position) if st is not None else (self.goal or (0.0, 0.0))
            if not isinstance(self.controller, HeliPilot):
                self.controller = HeliPilot([here, here], self.cfg.aircraft, self.mpp)
                self.controller.terrain = self.terrain_memory()
            self.controller.open_ground = self.open_ground
            self.controller.velocity = self._velocity
            # land close to where the key was pressed (when the position is known)
            self.controller.land_now(here if st is not None else None)
            self._needs_plan = False
            self.goal = here
            self.mission, self.arrived = 'land', False
            self.countermeasures_on = True
            self.engage()
            log.warning('LAND here')
            return self.engaged

    def toggle(self):
        """One button: pause when driving, resume an unfinished mission, else go."""
        with self.lock:
            if self.pending_go:
                self.disengage('start cancelled')
            elif self.engaged:
                self.disengage('paused')
            elif (self.mission in ('outbound','return') or
                  (self.cfg.profile == 'aircraft' and self.mission == 'land')) and not self.arrived and self.goal is not None:
                self.countermeasures_on = True
                self.engage()
            elif self.mission == 'outbound' and self.arrived and self.goal == self.destination:
                # already at the destination: flying there again would only take off and land
                self.telemetry.message = f'already at the destination - {self.cfg.keys.return_home} returns home'
                log.warning(self.telemetry.message)
            else:
                self.go()

    def set_home_here(self):
        with self.lock:
            st = self.telemetry.state
            if st is None:
                self._capture_home = True
            else:
                self.home = st.position
                log.warning('home set to %s', tuple(round(c) for c in self.home))

    # -- engagement --------------------------------------------------------
    def engage(self):
        with self.lock:
            if self.cfg.profile == 'truck' and self.map_manager is not None and not self.map_manager.ready:
                self.disengage('waiting for map confirmation; GO queued (press again to cancel)')
                self.pending_go = True
                return
            self.pending_go = False
            self._hud_grace = True
            self.engaged = True
            self.actuator.enabled = True
            self.telemetry.message = 'engaged'

    def disengage(self, reason='disengaged'):
        with self.lock:
            if getattr(self, '_terrain', None) is not None:
                try:
                    self._terrain.save()
                except OSError:
                    log.exception('could not save the terrain memory')
            self.engaged = False
            self.pending_go = False
            self.actuator.release()
            self.telemetry.engaged = False
            self.telemetry.output = None
            if self.cfg.profile == 'truck' and isinstance(self.controller, TruckDriver):
                # Manual motion cannot advance an autonomous recovery state or PI history.
                self.controller = None
            self.telemetry.message = reason

    def emergency_stop(self):
        self.countermeasures_on = False
        self.disengage('EMERGENCY STOP')
        self.keyboard.release_all()

    # -- loops ---------------------------------------------------------------
    def start_fix_worker(self, stop_event):
        """Live helicopter: match the minimap continuously in the background (~10 Hz), so
        the control loop never waits for it and gets a fresh position every few ticks."""
        self._fix_latest = None

        def loop():
            while not stop_event.is_set():
                if self.cfg.profile == 'aircraft' and self.cfg.aircraft.nav == 'dead_reckoning':
                    try:
                        generation = self._map_generation
                        fix = self._locate()
                        self._air_map_watch(fix, time.monotonic())
                    except Exception:
                        log.exception('minimap matching failed')
                        fix = None
                    if fix is not None:
                        with self._control_lock:
                            if generation == self._map_generation:
                                self._fix_latest = fix
                    stop_event.wait(.02)
                else:
                    stop_event.wait(.2)

        self.fix_worker = threading.Thread(target=loop, name='minimap', daemon=True)
        self.fix_worker.start()

    def _take_fix(self, now):
        """Newest minimap fix: from the background worker (live) or matched here (sim/tests)."""
        if getattr(self, 'fix_worker', None) is not None:
            fix, self._fix_latest = self._fix_latest, None
            return fix
        if not self.airnav.initialized or now - self.last_fix_try >= self.cfg.aircraft.fix_interval_s:
            self.last_fix_try = now
            return self._locate()
        return None

    def _locate(self):
        with self._loc_lock:
            return self._locate_now()

    def _minimap_small(self):
        image = self.sensors.minimap()
        cfg = self.cfg
        if image is not None and max(image.shape[:2]) > cfg.minimap_max_px:
            f = cfg.minimap_max_px / max(image.shape[:2])
            image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        self._last_minimap = image
        return image

    def _locate_now(self):
        image = self._minimap_small()
        if image is None:
            error = getattr(self.sensors, 'capture_errors', {}).get('minimap')
            if error:
                return Fix(reason=error, timestamp=time.monotonic())
            return None
        fix = self.locator.locate(image, self.cfg.localization_anchor, self.cfg.north_up)
        if fix.valid and self.cfg.refine_fixes:
            fix = self._refiner().refine(image, fix, self.cfg.localization_anchor)
        return fix

    def _refiner(self):
        """Sub-pixel refinement against the base map of the current map (cached per map)."""
        from .refine import FixRefiner
        if getattr(self, '_refine_map', None) != self.cfg.map_id:
            self._refine, self._refine_map = FixRefiner(self.assets.load_image()), self.cfg.map_id
        return self._refine

    def _air_map_watch(self, fix, now):
        with self._control_lock:
            return self._air_map_watch_locked(fix, now)

    def _air_map_watch_locked(self, fix, now):
        """Helicopter: if the current map stops matching while we are in the game (HUD on
        screen), a new round may be on another map.  Identify the map from the minimap,
        switch to it and let the navigator localize itself from scratch."""
        mm = self.map_manager
        if mm is None:
            return
        good = fix is not None and fix.valid and fix.confidence >= mm.t.min_confidence
        in_game = now - self.last_hud < 2.0
        if not in_game:
            self._air_fail_since = None
            return
        if good:
            self._air_fail_since = None
            if mm.ready and now < mm.next_scan:
                return
        else:
            self._air_fail_since = getattr(self, '_air_fail_since', None) or now
            if now - self._air_fail_since < self.cfg.auto_map.air_fail_s:
                return
        image = getattr(self, '_last_minimap', None)
        if image is None:
            return
        with self._loc_lock:
            selection = mm.update(image, now, self.cfg.localization_anchor, self.cfg.north_up)
        self.telemetry.map_status = selection.reason
        if selection.ready and selection.map_id:
            if selection.map_id != self.cfg.map_id:
                log.warning('new map detected in flight: %s', selection.map_id)
                self._switch_map(mm.assets[selection.map_id], mm.locators[selection.map_id])
                self._remember(lambda c: setattr(c, 'map_id', selection.map_id))
            self._air_fail_since = None

    def _tick_air(self, now: float) -> Telemetry:
        """Helicopter: dead reckoning on compass + airspeed, minimap fixes only correct it."""
        cfg, tel, nav = self.cfg, self.telemetry, self.airnav
        hud = self.sensors.hud()
        tel.hud = hud
        self._watch_hud(hud, now)
        fix = self._take_fix(now)
        if fix is not None:
            tel.fix = fix
        hud = self._check_speed(hud, now)
        tel.hud = hud
        speed = hud.speed_kmh if hud is not None else None
        heading = hud.heading if hud is not None else None
        nav.predict(now, speed, heading, hud.pitch if hud is not None else None)
        if fix is not None and fix.valid:
            self.last_valid = now
            if nav.correct(fix):
                self.manual_position = None
        if not nav.initialized and self.home is not None and fix is None:
            nav.start(self.home, now, sigma_m=30.0, source='return point')   # no minimap configured
        state = nav.state()
        tel.state = state
        if state is not None:
            self.last_speed = state.speed_ms
            if self._capture_home:
                self.home, self._capture_home = state.position, False
                log.warning('home remembered at %s', tuple(round(c) for c in self.home))
        tel.mission, tel.home, tel.destination = self.mission, self.home, self.destination
        tel.nav_sigma = nav.sigma_m if nav.initialized else None
        with self.lock:
            fast = now - self.last_fast < .3
            if self.goal is None:
                tel.status = 'no goal'
                self.actuator.neutral()
                return tel
            if state is None:
                tel.status = 'need position' if not nav.initialized else 'need heading'
                if (self.engaged and not isinstance(self.controller, HeliPilot) and hud is not None
                        and hud.agl_m is not None and cfg.aircraft.flight_model == 'helicopter'):
                    # engaged but no position yet (e.g. the wrong map: 12:54 flew 90 s with no keys
                    # at all and hit the ground): at least hold the height, level out and hover
                    self.controller = HeliPilot([(0.0, 0.0), (0.0, 0.0)], cfg.aircraft, self.mpp)
                    self._needs_plan = True
                if isinstance(self.controller, HeliPilot) and hud is not None and hud.agl_m is not None:
                    if self.engaged and hud.agl_m > 3:
                        tel.message = 'no position (minimap not matching - right map?): holding height, hovering'
                    out = self.controller.hold(now, hud)
                    tel.output = out
                    self.actuator.command(**{k: v for k, v in out.axes.items() if not (fast and k == 'cyclic')})
                else:
                    self.actuator.neutral()
                tel.engaged = self.engaged
                return tel
            if tel.message.startswith('no position'):
                tel.message = 'engaged'
            if self.controller is None or self._needs_plan:
                self._plan(state.position)
            out = self.controller.step(state, now, hud=hud)
            tel.output = out
            tel.remaining_m = out.info.get('distance_m')
            self.actuator.command(**{k: v for k, v in out.axes.items() if not (fast and k == 'cyclic')})
            tel.status = out.mode
            if out.done:
                self.arrived = True
                if self.engaged:
                    tel.message = {'return': 'arrived home', 'land': 'landed'}.get(self.mission, 'arrived')
                    self.disengage(tel.message + ' - your controls')
        tel.engaged = self.engaged
        return tel

    def _check_speed(self, hud, now):
        """Replace a HUD speed that disagrees with the ground speed from the minimap fixes."""
        if hud is None:
            return None
        ground = self.airnav.ground_speed(now) if self.airnav.initialized else None
        if ground is not None:
            self._last_ground = (ground, now)
        elif self._last_ground is not None and now - self._last_ground[1] < 1.5:
            ground = self._last_ground[0]     # a thin moment between fixes: the last one still holds
        self.telemetry.speed_source = 'hud'
        if ground is None:
            return hud
        hud_speed = hud.speed_kmh
        if hud_speed is not None and abs(hud_speed - ground) <= max(self.cfg.aircraft.speed_check_kmh, .2 * ground):
            return hud
        from dataclasses import replace
        self.telemetry.speed_source = 'fixes'
        self.speed_mismatches += 1
        if now - self._last_mismatch_log > 5.0:
            self._last_mismatch_log = now
            log.warning('HUD speed %s km/h but the minimap track says %.0f km/h: using the track',
                        hud_speed, ground)
            save = getattr(self.sensors, 'save_hud_crop', None)
            if save is not None:
                try:
                    save(f'spd{hud_speed}_true{ground:.0f}')
                except Exception:
                    log.exception('could not save the HUD crop')
        return replace(hud, speed_kmh=float(round(ground)))

    def tick(self, now: float) -> Telemetry:
        with self._control_lock:
            tel = self._tick_locked(now)
        # keys only go out while the game has the focus: say so instead of silently coasting
        # (real flight 01:52: 13 s without the game in front on the final approach)
        if self.engaged and not self.foreground_ok:
            tel.message = NOT_FOCUSED
        elif tel.message == NOT_FOCUSED:
            tel.message = 'engaged' if self.engaged else ''
        return tel

    def _tick_locked(self, now: float) -> Telemetry:
        """One perception/control cycle (≈10 Hz)."""
        cfg, tel = self.cfg, self.telemetry
        tel.engaged, tel.output = self.engaged, None
        if (cfg.profile == 'aircraft' and cfg.aircraft.flight_model == 'helicopter'
                and cfg.aircraft.nav == 'dead_reckoning'):
            return self._tick_air(now)
        image = self.sensors.minimap()
        if image is not None and max(image.shape[:2]) > cfg.minimap_max_px:
            f = cfg.minimap_max_px / max(image.shape[:2])
            image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        if cfg.profile == 'truck' and self.map_manager is not None:
            selection = self.map_manager.update(image, now, cfg.localization_anchor, cfg.north_up)
            if not selection.ready:
                pending = self.pending_go
                self.disengage(selection.reason)
                self.pending_go = pending
                self.estimator.reset()
                self.last_valid, self.last_speed = -1e9, 0.0
                tel.fix, tel.state, tel.output = selection.fix, None, None
                tel.status, tel.map_status, tel.engaged = 'map search', selection.reason, False
                return tel
            if selection.map_id != cfg.map_id:
                self._switch_map(self.map_manager.assets[selection.map_id],
                                 self.map_manager.locators[selection.map_id])
                tel = self.telemetry
            tel.map_status = selection.reason
            fix = selection.fix
            if self.pending_go:
                self.engage()
        else:
            fix = self.locator.locate(image, cfg.localization_anchor, cfg.north_up) if image is not None else Fix(
                reason=getattr(self.sensors, 'capture_errors', {}).get('minimap', 'no image'))
        if (fix.valid and cfg.refine_fixes and image is not None
                and isinstance(self.locator, MinimapLocalizer)):
            fix = self._refiner().refine(image, fix, cfg.localization_anchor)
        hud = None
        compass_heading = None
        if cfg.profile == 'truck' and fix.valid:
            read_heading = getattr(self.sensors,'truck_heading',None)
            if read_heading is not None:
                try:
                    compass_heading = read_heading()
                except Exception as error:
                    if not getattr(self,'_truck_compass_failed',False):
                        log.warning('truck compass unavailable: %s',error)
                    self._truck_compass_failed = True
                    compass_heading = None
                if compass_heading is not None and np.isfinite(compass_heading):
                    from dataclasses import replace
                    fix = replace(fix,heading=float(compass_heading)%360)
                else:
                    compass_heading = None
        if cfg.profile == 'aircraft' and cfg.aircraft.flight_model == 'helicopter':
            hud = self.sensors.hud()
            if hud is not None and hud.heading is not None:
                fix.heading = hud.heading   # the compass tape beats the minimap icon
        tel.fix, tel.hud = fix, hud
        if hud is not None or (cfg.profile == 'aircraft' and cfg.aircraft.flight_model == 'helicopter'):
            self._watch_hud(hud, now)
        state = None
        if fix.valid:
            self.last_valid = now
            state = self.estimator.update(fix, now)
            if state is not None and compass_heading is not None:
                state.heading_source = 'compass'
            if self.estimator.t == now:
                self.manual_position = None
        elif now - self.last_valid < cfg.lost_release_s:
            state = self.estimator.coast(now)
        motion = self.motion.update(self.sensors.main_view())
        tel.state = state
        if state is not None:
            self.last_speed = state.speed_ms
            if self._capture_home and fix.valid:
                self.home, self._capture_home = state.position, False
                log.warning('home remembered at %s', tuple(round(c) for c in self.home))
        tel.mission, tel.home, tel.destination = self.mission, self.home, self.destination

        heli = isinstance(self.controller, HeliPilot)
        if state is None and heli and hud is not None and hud.speed_kmh is not None \
                and now - self.last_valid < cfg.aircraft.dead_reckon_s:
            state = self.estimator.dead_reckon(now, hud.speed_kmh / 3.6, hud.heading)
            tel.state = state
        with self.lock:
            if cfg.profile == 'truck':
                # Route preview needs position, not heading or throttle permission.
                if self.goal is None and self.destination is not None:
                    self.goal, self.waypoints = self.destination, [tuple(p) for p in self.via]
                origin = (state.position if state is not None else fix.position if fix.valid
                          else self.manual_position.point if self.manual_position is not None else None)
                if self.goal is not None and self.controller is None and origin is not None:
                    try:
                        self._plan(origin)
                    except RouteError as error:
                        tel.status = 'route error'
                        self.disengage(str(error))
                        return tel
                if not self.engaged:
                    self.actuator.neutral()
                    if self.arrived:
                        tel.status = 'arrived'
                    elif self.goal is None:
                        tel.status = 'no goal'
                    elif state is None:
                        tel.status = 'need heading' if fix.valid else 'need position'
                    else:
                        tel.status = 'standby' if self.mission == 'idle' else 'paused'
                    return tel
                if state is None and fix.valid:
                    self.actuator.neutral()
                    tel.status = 'need heading'
                    tel.message = 'position confirmed; waiting for compass heading (check compass calibration)'
                    return tel
            if state is None and heli and hud is not None and hud.agl_m is not None:
                out = self.controller.hold(now, hud)        # no position: hold height, hover
                tel.output, tel.status = out, out.mode
                fast = now - self.last_fast < .3
                self.actuator.command(**{k: v for k, v in out.axes.items() if not (fast and k == 'cyclic')})
                tel.engaged = self.engaged
                return tel
            if state is None:
                lost = now - self.last_valid
                tel.status = 'searching' if lost < cfg.lost_stop_s else 'lost'
                if self.cfg.profile == 'truck':
                    self.actuator.command(steer=0.0, throttle=-1.0 if lost > cfg.lost_stop_s and self._moving() else 0.0)
                else:
                    self.actuator.neutral()   # helicopter: the game's auto-hover holds us
                if lost > cfg.lost_stop_s:
                    self.estimator.reset()
                return tel
            if self.goal is None:
                tel.status = 'no goal'
                self.actuator.neutral()
                return tel
            if self.controller is None:
                try:
                    self._plan(state.position)
                except RouteError as error:
                    tel.status, tel.message = 'route error', str(error)
                    self.disengage(str(error))
                    return tel
            out = self.controller.step(state, now, None if motion is None else self.motion.level, hud=hud)
            tel.output = out
            if out.takeover:
                self.disengage('automatic reverse limit reached; manual intervention required')
                tel.output,tel.status,tel.engaged = out,out.mode,False
                return tel
            info = out.info
            tel.remaining_m = info.get('remaining_m', info.get('distance_m'))
            if out.replan:
                blocked = out.info.get('blocked_at')
                if blocked is not None:
                    r = self.cfg.truck.blocked_radius_m / self.mpp
                    self.avoid.append(Avoid(blocked[0] - r, blocked[1] - r, blocked[0] + r, blocked[1] + r))
                    log.warning('road blocked at %s, re-routing around it', blocked)
                else:
                    log.info('off route, re-planning')
                try:
                    self._plan(state.position)
                except RouteError as error:
                    tel.message = str(error)
                out = Output({k: 0.0 for k in self.actuator.axes}, 'replan')
            # the fast loop owns the helicopter cyclic while it runs
            fast = heli and now - self.last_fast < .3
            axes = {k: v for k, v in out.axes.items() if not (fast and k == 'cyclic')}
            self.actuator.command(**axes)
            tel.status = out.mode
            if out.done:
                self.arrived = True
                if self.engaged:
                    tel.message = 'arrived home' if self.mission == 'return' else 'arrived'
                    # the truck brakes to a stop first; the helicopter has no hover mode yet,
                    # so control goes straight back to the pilot
                    if self.cfg.profile == 'aircraft' or not self._moving():
                        self.disengage(tel.message + ' - your controls')
                        tel.output = out  # preserve the terminal completion event for this tick
        tel.engaged = self.engaged
        return tel

    def _watch_hud(self, hud, now):
        """Pause when the helicopter HUD disappears (got out, map open, died, alt-tab)."""
        visible = hud is not None and (hud.agl_m is not None or hud.speed_kmh is not None or hud.pitch is not None)
        recent_manual = (self.manual_position is not None
                         and now - self.manual_position.timestamp <= self.cfg.aircraft.relocalize_after_s)
        if (visible and now - self.last_hud > self.cfg.aircraft.relocalize_after_s
                and self.airnav.initialized and not recent_manual):
            # back in a helicopter after a long break (new round / respawn): the old position
            # and terrain context are stale -> localize from the minimap again
            self.airnav.reset()
            self.manual_position = None
            self._needs_plan = self.controller is not None
            log.warning('HUD back after %.0fs: re-localizing', now - self.last_hud)
        if visible or self._hud_grace:
            self.last_hud = now
            self._hud_grace = False
        elif self.engaged and now - self.last_hud > self.cfg.aircraft.hud_pause_s:
            self.disengage('HUD gone - paused (F8 resumes)')
            log.warning('helicopter HUD not visible for %.1fs: paused', now - self.last_hud)

    def _moving(self) -> bool:
        return abs(self.last_speed) > 1.0

    def fast_control(self, now: float):
        """Helicopter pitch hold at the fast-loop rate (the slow loop only sets the target)."""
        if not (self.engaged and isinstance(self.controller, HeliPilot)):
            return
        try:
            pitch = self.sensors.attitude()
        except Exception:
            log.exception('attitude read failed')
            pitch = None
        with self.lock:
            if isinstance(self.controller, HeliPilot):
                self.last_fast = now
                cyclic = self.controller.inner(pitch, now)
                self.actuator.command(cyclic=cyclic)
                self.fast_samples.append((round(now, 3), None if pitch is None else round(pitch, 3),
                                          round(self.controller.pitch_target, 3), round(cyclic, 3)))
                del self.fast_samples[:-60]

    def actuate(self, now: float):
        with self.lock:
            map_ready = (self.cfg.profile != 'truck' or self.map_manager is None
                         or (self.map_manager.ready and self.map_manager.active == self.cfg.map_id))
            self.actuator.enabled = self.engaged and self.foreground_ok and map_ready
            self.actuator.update(now)

    def countermeasures(self, now: float) -> str:
        """Missile-lock watch (helicopter).  Runs regardless of route engagement.

        Returns 'none' | 'acquiring' | 'locked'.  Audio (lock tones) and the optional
        visual HUD detector are OR-ed; the visual cue only knows "locked".
        """
        if self.flares is None:
            return 'none'
        state = 'none'
        if self.audio_detector is not None:
            samples = self.sensors.audio()
            state = self.audio_detector.feed(samples if samples is not None else np.zeros(0, np.float32), now)
        if self.detector is not None and self.detector.update(self.sensors.lock_roi(), now):
            state = 'locked'
        if state != self.telemetry.lock_state and state != 'none':
            log.warning('missile warning: %s', state)
        self.telemetry.lock_state = state
        self.telemetry.locked = state == 'locked'
        if self.foreground_ok and self.countermeasures_on:
            self.flares.can_hear_missile = bool(self.audio_detector and self.audio_detector.has_missile)
            missile = bool(self.audio_detector and self.audio_detector.missile)
            self.telemetry.missile_heard = missile
            self.flares.update_state(state, now, missile)
        self.telemetry.flares = self.flares.fired
        return state

    def _fire_flare(self):
        log.warning('missile lock -> flare')
        self.keyboard.tap(self.cfg.keys.flare, 0.04)


# ---------------------------------------------------------------------------
class ScreenSensors:
    def __init__(self, cfg: Config):
        from .capture import ScreenCapture
        self.cfg = cfg
        self._local = threading.local()
        self._factory = ScreenCapture
        self._tap = None
        self._hud = None
        self.capture_errors = {}
        from .team_detection import TeamDetector
        self._team_detector = TeamDetector()
        self.team_reading = self._team_detector.reading
        self._team_lock = threading.Lock()

    def _cap(self):
        # mss handles are per-thread on Windows
        if not hasattr(self._local, 'cap'):
            self._local.cap = self._factory()
        return self._local.cap

    def _grab(self, region, name):
        from .capture import CaptureRegionError
        if not region.valid:
            self.capture_errors[name] = f'{name}: capture region not calibrated'
            return None
        try:
            image = self._cap().grab(region)
        except CaptureRegionError as error:
            self.capture_errors[name] = f'{name}: {error}'
            return None
        self.capture_errors.pop(name, None)
        return image

    def minimap(self):
        from .input_win import foreground_is
        if self.cfg.require_foreground and not foreground_is(self.cfg.window_title):
            image = None  # Never classify another foreground app as game terrain/team.
        else:
            image = self._grab(self.cfg.localization_region, 'minimap')
        with self._team_lock:
            self.team_reading = self._team_detector.update(image, time.monotonic(),
                (self.cfg.map_id,self.cfg.profile))
        return image

    def main_view(self):
        return self._grab(self.cfg.main_view, 'main_view') if self.cfg.main_view.valid else None

    def lock_roi(self):
        return self._grab(self.cfg.lock_warning, 'lock_warning') if self.cfg.lock_warning.valid else None

    def hud(self):
        if not self.cfg.hud.valid:
            return None
        if self._hud is None:
            from .hud import HudReader
            self._hud = HudReader(self.cfg.hud_glyphs)
        compass = self._grab(self.cfg.compass, 'compass') if self.cfg.compass.valid else None
        return self._hud.read(self._grab(self.cfg.hud, 'hud'), compass, time.monotonic())

    def truck_heading(self):
        """Read the shared top compass only; truck dashboards are not helicopter HUDs."""
        if not self.cfg.compass.valid:
            return None
        from .hud import HudReader
        if not hasattr(self._local,'truck_compass'):
            self._local.truck_compass = HudReader(self.cfg.hud_glyphs)
        return self._local.truck_compass.read(None,self._grab(self.cfg.compass, 'compass'),time.monotonic()).heading

    def save_hud_crop(self, tag: str):
        """Keep the speed digits that disagreed with the minimap track (to fix the reader)."""
        from pathlib import Path
        folder = Path('logs') / 'hud_mismatch'
        if not self.cfg.hud.valid or (folder.exists() and len(list(folder.glob('*.png'))) >= 60):
            return
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f'{time.strftime("%Y%m%d_%H%M%S")}_{tag}.png'), self._cap().grab(self.cfg.hud))

    def attitude(self):
        if not self.cfg.hud.valid:
            return None
        from .hud import read_pitch
        image = self._grab(self.cfg.hud, 'hud')
        return read_pitch(image) if image is not None else None

    def audio(self):
        if self._tap is None:
            from .audio import LoopbackTap
            self._tap = LoopbackTap(self.cfg.audio_device or None).start()
        if self._tap.error:
            raise RuntimeError(f'audio loopback failed: {self._tap.error}')
        return self._tap.read()


NOT_FOCUSED = 'game window not in front - keys released (click into the game)'


def run_live(pilot: Autopilot, *, view: bool = False, engage_now: bool = False, flares_only: bool = False,
             panel: bool = False, record: bool = True):
    """Real game loop.  F8 = go/pause/resume, F7 = return home, F9 = emergency stop, Ctrl+C = quit."""
    from .input_win import EdgeDetector, foreground_is

    cfg = pilot.cfg
    if cfg.auto_map.enabled and not flares_only:
        from .map_manager import MapManager
        pilot.map_manager = MapManager(pilot.assets, pilot.locator, cfg.auto_map)
    toggle, estop = EdgeDetector(cfg.keys.toggle), EdgeDetector(cfg.keys.emergency_stop)
    home_key = EdgeDetector(cfg.keys.return_home)
    land_key = EdgeDetector(cfg.keys.land)
    switch_key = EdgeDetector(cfg.keys.switch_vehicle)
    stop = threading.Event()

    def fast_loop():
        period = 1.0 / max(30.0, cfg.aircraft.detector_hz)
        while not stop.is_set():
            now = time.monotonic()
            pilot.foreground_ok = (not cfg.require_foreground) or foreground_is(cfg.window_title)
            if estop.pressed():
                pilot.emergency_stop()
            if toggle.pressed():
                if flares_only:
                    pilot.countermeasures_on = not pilot.countermeasures_on
                    log.warning('auto-flares %s', 'ON' if pilot.countermeasures_on else 'OFF')
                else:
                    pilot.toggle()
            if home_key.pressed() and not flares_only:
                pilot.return_home()
            if land_key.pressed() and not flares_only:
                pilot.land_here()
            if switch_key.pressed() and not flares_only:
                pilot.toggle_profile()
            try:
                pilot.countermeasures(now)
            except Exception:
                log.exception('countermeasure error')
            pilot.fast_control(now)
            pilot.actuate(now)
            stop.wait(max(0.0, period - (time.monotonic() - now)))

    from .recorder import FlightRecorder
    recorder = FlightRecorder() if record else None

    def control_loop():
        viewer = None
        if view:
            from .debug_view import DebugView
            viewer = DebugView(pilot.assets)
        viewed_map = pilot.cfg.map_id
        period = 1.0 / cfg.loop_hz
        while not stop.is_set():
            now = time.monotonic()
            if not flares_only:
                try:
                    tel = pilot.tick(now)
                except Exception:
                    log.exception('control loop error')
                    pilot.disengage('error')
                    tel = pilot.telemetry
                if recorder is not None:
                    recorder.write(pilot, tel, now, time.monotonic() - now)
                if viewer is not None:
                    if viewed_map != pilot.cfg.map_id:
                        viewer = DebugView(pilot.assets)
                        viewed_map = pilot.cfg.map_id
                    if viewer.show(tel):
                        stop.set()
            stop.wait(max(0.0, period - (time.monotonic() - now)))

    pilot.background_open = True
    pilot.open_ground()                       # start building the open-ground map now
    worker = threading.Thread(target=fast_loop, name='actuator', daemon=True)
    worker.start()
    if not flares_only:
        pilot.start_fix_worker(stop)          # continuous minimap position for the helicopter
    if engage_now:
        pilot.go()
    try:
        if panel:
            # tkinter must own the main thread; perception/control runs beside it
            controller = threading.Thread(target=control_loop, name='control', daemon=True)
            controller.start()
            from .panel import ControlPanel
            ControlPanel(pilot, stop, flares_only=flares_only).run()
        else:
            control_loop()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.join(1.0)
        pilot.actuator.release()
        pilot.keyboard.release_all()
        if recorder is not None:
            recorder.close()
            print(f'flight log: {recorder.path}')
