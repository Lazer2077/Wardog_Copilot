"""Persistent settings (JSON).  Screen regions are physical pixels as used by mss."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / 'config.json'


@dataclass
class Region:
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0
    reference: tuple = ()              # calibration monitor: left, top, width, height

    @property
    def valid(self) -> bool:
        return self.width >= 20 and self.height >= 20

    def as_mss(self) -> dict:
        return {'left': self.left, 'top': self.top, 'width': self.width, 'height': self.height}


@dataclass
class KeyBindings:
    """DirectInput key names (see input_win.SCANCODES)."""
    forward: str = 'W'
    backward: str = 'S'
    left: str = 'A'
    right: str = 'D'
    handbrake: str = 'SPACE'
    reverse_gear: str = 'R'             # "Gear Shift R" in WARDOGS ground-vehicle binds (verify in game)
    # helicopter (WARDOGS default PC binds: cyclic W/S + A/D, yaw Q/E, collective LShift/LCtrl)
    pitch_up: str = 'S'                 # cyclic back
    pitch_down: str = 'W'               # cyclic forward (nose down -> speed)
    roll_left: str = 'A'
    roll_right: str = 'D'
    yaw_left: str = 'Q'
    yaw_right: str = 'E'
    throttle_up: str = 'LSHIFT'         # collective raise
    throttle_down: str = 'LCTRL'        # collective lower
    flare: str = 'V'                    # "Activate Counter Measure"
    # autopilot hotkeys (polled with GetAsyncKeyState, never injected)
    toggle: str = 'F8'                  # go to destination / pause / resume
    return_home: str = 'F7'             # drive back to the home point
    switch_vehicle: str = 'F6'          # switch truck <-> helicopter
    emergency_stop: str = 'F9'
    land: str = 'F10'                   # helicopter: land right here (HUD only, no position needed)


@dataclass
class MinimapCalibration:
    region: Region = field(default_factory=Region)
    anchor: tuple = ()                 # empty means use the legacy shared anchor


@dataclass
class TruckTuning:
    camera_view: str = 'unknown'        # manual dataset label: first_person | third_person | unknown
    max_speed_ms: float = 16.0          # straight-line cruise (≈58 km/h)
    min_speed_ms: float = 4.0
    lateral_accel: float = 2.5          # m/s² allowed in curves -> curve speed limit
    goal_slowdown_m: float = 60.0
    arrive_radius_m: float = 15.0
    lookahead_base_m: float = 14.0
    lookahead_gain_s: float = 1.0       # lookahead += gain * speed
    max_curvature: float = 1 / 9.0      # 1/m, full steering lock
    steer_gain: float = 1.0
    yaw_kp: float = 0.03                # steer per deg/s of yaw-rate error
    yaw_ki: float = 0.06                # adaptive steering integral gain
    yaw_i_limit: float = 0.6
    speed_kp: float = 0.25
    speed_ki: float = 0.05
    brake_margin_ms: float = 2.5
    pwm_period_s: float = 0.20
    off_route_m: float = 45.0
    off_route_s: float = 3.0
    stuck_s: float = 2.5
    stuck_move_m: float = 5.0          # less displacement than this in stuck_s => stuck
    startup_grace_s: float = 6.0       # no automatic reverse immediately after starting/replanning
    recovery_cooldown_s: float = 8.0   # measured after the recovery manoeuvre ends
    max_reverse_attempts: int = 2      # reverse + K-turn starts per recovery area
    recovery_reset_m: float = 30.0     # leave this area before another attempt budget is granted
    stuck_max_speed_ms: float = .75
    stuck_motion_threshold: float = .15
    reverse_s: float = 1.6
    bypass_s: float = 1.5
    stuck_attempts: int = 3             # failures at one spot before re-routing around it
    blocked_radius_m: float = 25.0
    road_kinds: tuple = ('major', 'minor')
    max_snap_px: float = 45.0


@dataclass
class AircraftTuning:
    flight_model: str = 'helicopter'    # 'helicopter' (HUD altitude/speed hold) | 'plane' (heading only)
    # --- helicopter, identified from real flight logs (2026-10-17)
    cruise_agl_m: float = 150.0         # height above ground in cruise (UI adjustable)
    cruise_kmh: float = 300.0           # players cruise the Little Bird at ~300 km/h (top 350)
    takeoff_agl_m: float = 10.0         # climb vertically to this before flying forward
    turn_agl_m: float = 10.0            # no heading control below this height
    ground_agl_m: float = 3.0           # below this we are on the ground
    takeoff_vz: float = 6.0             # m/s vertical climb during takeoff
    min_agl_m: float = 20.0             # below this at speed: pull up
    turn_slow_deg: float = 35.0         # big heading error -> slow to turn_kmh first
    turn_kmh: float = 40.0
    low_speed_frac: float = 0.25        # speed fraction allowed while far below the cruise height
    terrain_slope_min: float = 0.08     # rising-ground speed limit only for a real slope (8 %) ...
    terrain_low_frac: float = 0.7       # ... while lower than 70 % of the cruise height ...
    terrain_min_kmh: float = 80.0       # ... and, above half the cruise height, never slower than this
    climb_capacity: float = 7.0         # m/s we can really climb at speed (real log: 5-9)
    clearance_gain: float = 2.0         # while low, keep gaining at least this much clearance (m/s)
    tti_brake_s: float = 6.0            # ground closing in: brake hard when impact is nearer than this
    clearance_m_per_kmh: float = 0.5    # fly only as fast as the height allows: 150 m -> 300 km/h ...
    known_clearance_m_per_kmh: float = 0.3   # ... but 90 m -> 300 km/h where the terrain ahead is known and flat
    known_ahead_m: float = 1500.0       # how far ahead the terrain memory must know the ground for that
    full_authority_agl_m: float = 50.0  # full speed/pitch authority from this height on
    rising_ground_m: float = 10.0       # look-ahead: only ground this much above the ground here counts
    lookahead_s: float = 20.0           # terrain memory: look this far ahead (time at current speed) ...
    lookahead_min_m: float = 600.0      # ... but at least this far
    lookahead_clear_frac: float = 0.8   # pass known high ground with 80 % of the cruise height (0.65 dipped to 3 m on a known ridge)
    decel_ms2: float = 3.0              # planned braking (players dump 270 -> 50 km/h nose-up)
    # landing at the destination / return point
    auto_land: bool = True              # False: hover at the destination and hand over
    hover_radius_m: float = 40.0        # descend only this close to the point ...
    land_zone_m: float = 150.0          # ... but stop and hover from this far (fixes are ~5 m/px)
    overshoot_m: float = 40.0           # final leg: this much farther than the closest pass -> hover
    land_anyway_s: float = 25.0         # cannot centre within this time -> land where we are
    reposition_kmh: float = 80.0        # max speed while moving back over the point from the zone edge
    # J-hook landing (WARDOGS players' technique): pass beside the spot, then brake hard and
    # sink with the collective down at once, nose on the spot, arriving slow and low over it
    jhook_landing: bool = True
    # velocity-vector guidance: point the thrust (nose) where acceleration is needed - the game's
    # helicopter barely turns its motion with the nose (fit-slip on 13:16: 0.02/s)
    vector_guidance: bool = False       # off: at 17:00 the compass was unreadable most of the time, the
                                        # nose heading wrong, and it circled worse than the old steering
    vector_landing: bool = True         # ... but use it for the final approach and landing: in the measured
                                        # sliding physics the default steering cannot land (sim: circles 340 s)
    vector_landing_m: float = 1500.0    # from this far out
    vector_turnround: bool = True       # turning round at speed (F7 while flying away) with it too
    vg_pivot_kmh: float = 45.0          # turning round: start the 180 deg pivot below this speed
    vg_gain: float = 0.5                # 1/s: velocity error -> acceleration
    vg_decel_ms2: float = 3.5           # braking curve onto the landing point
    vg_dead_ms2: float = 0.3            # smaller accelerations: hold the nose where it is
    vg_yaw_kp: float = 0.03             # yaw keys per degree: the nose must keep up with the thrust direction
    hook_decel_ms2: float = 3.5         # planned braking during the hook
    hook_end_m: float = 10.0            # ... to stop this far from the spot
    hook_end_agl_m: float = 10.0        # ... this high (then a short vertical touchdown)
    hook_offset_m: float = 40.0         # approach passing this far beside the spot
    hook_max_kmh: float = 360.0         # a faster speed reading is a misread: never start a hook on it
    hook_lag_s: float = 1.0             # plan the stop this many seconds of flight early (response lag)
    hook_kmh_per_tick: float = 20.0     # nose-up braking allowed in the hook: 1 tick per 20 km/h
    hook_kv: float = 0.05               # ticks nose-up per km/h above the hook's braking curve
    land_here_radius_m: float = 150.0   # F10: touch down within this of where the key was pressed
    land_hover_agl_m: float = 15.0      # over the landing area: hover this low before the final descent
    land_fast_vz: float = 10.0          # sink rate allowed while still above 40 m over the landing area
    # open ground (base map): forest, water and town are avoided for the touchdown
    open_ground: bool = True
    open_check_m: float = 2000.0        # pick the spot when the final approach starts
    open_radius_m: float = 150.0        # ... within this of the destination / home / F10 point
    open_ok_cost: float = 0.45          # a spot this open is kept as it is
    open_max_cost: float = 0.6          # a spot must be at least this open to move there
    open_gain: float = 0.25             # ... and clearly more open than the original
    flat_max_m: float = 6.0             # terrain memory: landing cell may differ this much from its neighbours
    spot_shift_m: float = 80.0          # destination / home: move the touchdown this far for flatter ground
    brake_pitch_max: float = 3.2        # ticks nose-up when stopping for a landing (cruise: pitch_back_max)
    touchdown_max_kmh: float = 6.0      # never put the skids down faster than this (over the ground)
    ground_steady_ms: float = 1.5       # ground under us changing faster than this: hold the descent
    speed_check_kmh: float = 25.0       # HUD speed this far off the fix ground speed: use the fixes
    # J-hook: reverse at speed (e.g. return pressed while flying away) - full pedal turn while
    # the cyclic steers the velocity vector onto 'cruise speed towards the target'
    jhook: bool = True
    jhook_trigger_deg: float = 100.0    # target this far off the direction of travel ...
    jhook_min_kmh: float = 90.0         # ... at this speed or more
    jhook_exit_deg: float = 30.0        # nose and travel both within this of the target: done
    jhook_max_s: float = 30.0
    jhook_gain: float = 0.5             # 1/s: velocity error -> acceleration along the nose
    jhook_climb_ms: float = 3.0         # climb a little meanwhile (clearance while sliding)
    slip_align: float = float('inf')    # navigator: 1/s the travel direction swings onto the nose
                                        # (measured: fit-slip on the 12:54 / 13:16 flights; inf = arcade)
    hover_settle_s: float = 1.0         # steady this long over the point before descending
    descend_max_kmh: float = 20.0       # "steady enough" to start the descent
    glide_ratio: float = 0.12           # final leg: target height = 12 % of the remaining distance ...
    glide_min_agl_m: float = 30.0       # ... but not below this
    creep_kmh: float = 15.0             # max speed while centring over the landing point
    descent_rate_per_m: float = 0.15    # descent rate (m/s) per metre of (predicted) height ...
    land_vz_max: float = 6.0            # ... capped here
    flare_agl_m: float = 5.0            # below this (predicted): final slow descent
    flare_vz: float = 1.5
    touchdown_collective: float = 0.15  # gentle constant sink for the last 2 m
    touch_confirm_s: float = 0.5        # AGL reads 0 this long -> on the skids
    settle_s: float = 1.5               # collective down on the skids before handing over
    # vertical: Shift/Ctrl = climb command, measured ~1.2 s delay + ~2.5 s lag
    alt_kp: float = 0.12                # m/s of climb per m of altitude error
    vz_max: float = 8.0
    vz_kp: float = 0.12                 # Shift/Ctrl per m/s of (predicted) climb-rate error
    vz_ki: float = 0.03                 # integral: removes the steady climb-rate shortfall
    vz_i_max: float = 0.4
    vert_gain: float = 12.0             # m/s of climb per unit of collective (model)
    vert_tau_s: float = 2.5
    vert_delay_s: float = 1.2
    # cyclic: W/S tilt the nose ~3.7 ticks/s (1 tick ~ 10 deg); acceleration ~ g*tan(pitch) - drag
    spd_to_pitch: float = 0.02          # ticks of nose-down per km/h of speed error
    spd_ki: float = 0.004
    spd_i_max: float = 1.0
    ref_speed_kmh: float = 300.0        # feed-forward anchor: this speed is held ...
    ref_pitch_ticks: float = 3.0        # ... with this much nose-down (player guide)
    pitch_fwd_max: float = 3.2          # ticks nose down: players cruise with the arrow at the bottom notch
    pitch_back_max: float = 2.5         # ticks nose up (braking limit)
    brake_kmh_per_tick: float = 40.0    # nose-up braking allowed: 1 tick per 40 km/h of forward speed
    pitch_kp: float = 0.6               # keys per tick of pitch error (fast loop)
    pitch_kd: float = 0.12              # per tick/s of pitch rate
    cyclic_pwm_s: float = 0.08          # shorter PWM period for the fast pitch loop
    yaw_kp: float = 0.012               # yaw keys per degree of heading error (Q/E turn ~40 deg/s)
    yaw_kd: float = 0.012               # per deg/s of yaw rate
    yaw_max: float = 0.6
    hud_timeout_s: float = 1.0
    hud_pause_s: float = 2.0            # HUD gone this long while engaged -> pause
    relocalize_after_s: float = 20.0    # HUD gone this long (new round / respawn) -> localize from scratch
    dead_reckon_s: float = 8.0          # (minimap nav mode) fly on compass + HUD speed this long, then hover
    nav: str = 'dead_reckoning'         # 'dead_reckoning': compass + airspeed, minimap only corrects
                                        # 'minimap': position from every minimap match (old behaviour)
    fix_interval_s: float = 0.15        # dead_reckoning: try a minimap correction this often
    fix_min_conf: float = 0.5           # ... and use it only above this match confidence
    dr_drift: float = 0.03              # assumed dead-reckoning error per metre flown
    # plane / generic
    arrive_radius_m: float = 120.0
    waypoint_radius_m: float = 150.0
    steering: str = 'bank'              # 'bank' (roll keys) | 'yaw' (rudder keys only)
    max_turn_rate_dps: float = 10.0     # commanded turn rate limit
    heading_to_rate: float = 0.35       # deg/s of turn rate per degree of heading error
    rate_gain: float = 0.12             # roll command per deg/s of turn-rate error
    rudder_gain: float = 0.05           # yaw command per degree of heading error
    rudder_threshold_deg: float = 8.0   # below this error, use rudder only
    turn_pitch_gain: float = 0.0        # pitch-up per unit roll command (holds altitude in turns)
    pwm_period_s: float = 0.15
    cruise_throttle_taps: int = 0       # extra throttle-up taps when engaging
    detector_hz: float = 30.0
    # missile countermeasures
    lock_source: str = 'audio'          # 'audio' (lock tones) | 'visual' (HUD region) | 'both' | 'none'
    flare_on: str = 'missile'           # 'missile': lock warning, then the missile is heard -> fire
                                        # 'locked': steady lock tone (+ flare_delay_s); 'acquire': on beeps
    lock_memory_s: float = 15.0         # 'missile': a missile sound counts only this long after a warning
    locked_fallback_s: float = 1.5      # 'missile': steady lock tone this long without a missile heard -> fire
    flare_delay_s: float = 0.6          # wait after lock: flares work best 1-2 s before impact
    flare_capacity: int = 0             # flares per life, 0 = unknown / do not count
    flare_enabled: bool = True
    flare_burst: int = 2
    flare_interval_s: float = 0.25
    flare_cooldown_s: float = 2.0
    lock_confirm_frames: int = 2        # consecutive positive frames before firing
    lock_hold_s: float = 0.6            # treat a blinking warning as continuous
    lock_threshold: float = 0.70        # template score / colour-ratio threshold
    lock_color_hsv_low: tuple = (0, 120, 150)    # red warning by default
    lock_color_hsv_high: tuple = (12, 255, 255)
    lock_color_hsv_low2: tuple = (168, 120, 150)
    lock_color_hsv_high2: tuple = (180, 255, 255)
    lock_color_ratio: float = 0.04      # fraction of ROI pixels matching the colour


@dataclass
class MapTuning:
    enabled: bool = True              # live truck only; simulation keeps its explicit map
    scan_interval_s: float = 2.0
    confirm_scans: int = 3
    min_confidence: float = .55
    min_inliers: int = 12
    score_ratio: float = 1.3
    air_fail_s: float = 4.0           # helicopter: current map not matching this long -> identify map


@dataclass
class CollectionTuning:
    process_name: str = 'WardogsClient-Win64-Shipping.exe'
    fps: float = 15.0
    input_hz: float = 100.0
    nav_hz: float = 4.0
    max_width: int = 512
    queue_size: int = 8
    segment_s: float = 120.0
    directory: str = 'data/driving'
    speed_first_person: Region = field(default_factory=Region)
    speed_third_person: Region = field(default_factory=Region)


@dataclass
class UiTuning:
    panel_opacity: float = .86
    map_opacity: float = .90
    collection_opacity: float = .86


@dataclass
class Config:
    profile: str = 'truck'              # 'truck' | 'aircraft'
    map_id: str = 'ozeti'
    auto_map: MapTuning = field(default_factory=MapTuning)
    collection: CollectionTuning = field(default_factory=CollectionTuning)
    ui: UiTuning = field(default_factory=UiTuning)
    assets_dir: str = ''
    minimap: Region = field(default_factory=Region)
    truck_minimap: MinimapCalibration = field(default_factory=MinimapCalibration)
    aircraft_minimap: MinimapCalibration = field(default_factory=MinimapCalibration)
    main_view: Region = field(default_factory=Region)
    lock_warning: Region = field(default_factory=Region)
    hud: Region = field(default_factory=Region)          # helicopter instrument cluster (SPD / ALT)
    compass: Region = field(default_factory=Region)      # heading box of the compass tape
    hud_glyphs: str = 'hud_glyphs.npz'                   # learned HUD digit templates
    lock_template: str = ''             # png path of the lock warning icon (optional)
    audio_profile: str = ''             # json from "audio-fit" (lock tone signatures)
    audio_device: str = ''              # speaker name for loopback, '' = default output
    anchor: tuple = (0.5, 0.5)          # player arrow position inside the minimap crop
    north_up: bool = True
    destinations: dict = field(default_factory=dict)   # map_id -> [x, y] map px, set with "set-dest"
    homes: dict = field(default_factory=dict)          # map_id -> fixed return point; absent = first go spot
    favorites: dict = field(default_factory=dict)      # map_id -> [{"name", "point": [x, y]}]
    base_overrides: dict = field(default_factory=dict) # map_id -> faction id -> calibrated map pixels
    window_title: str = 'WARDOGS'       # inputs only go out while this window is foreground
    require_foreground: bool = True
    loop_hz: float = 8.0
    refine_fixes: bool = True           # terrain alignment refinement, accepted only when agreement improves
    minimap_max_px: int = 300           # downscale big minimaps before matching (4K: 394 -> 79 ms)
    lost_release_s: float = 0.8
    lost_stop_s: float = 3.0
    keys: KeyBindings = field(default_factory=KeyBindings)
    truck: TruckTuning = field(default_factory=TruckTuning)
    aircraft: AircraftTuning = field(default_factory=AircraftTuning)

    @property
    def localization_region(self) -> Region:
        calibration = self.aircraft_minimap if self.profile == 'aircraft' else self.truck_minimap
        return calibration.region if calibration.region.valid else self.minimap

    @property
    def localization_anchor(self) -> tuple:
        calibration = self.aircraft_minimap if self.profile == 'aircraft' else self.truck_minimap
        return calibration.anchor if len(calibration.anchor) == 2 else self.anchor

    def save(self, path: Path | str = DEFAULT_PATH):
        """Write only what differs from the defaults, so improved defaults (tuning) still
        reach existing installs; screen regions, keys and destinations are kept."""
        data = _diff(asdict(self), asdict(Config()))
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> 'Config':
        path = Path(path)
        if not path.exists():
            return cls()
        return _from_dict(cls, json.loads(path.read_text(encoding='utf-8')))


def _diff(values: dict, defaults: dict) -> dict:
    out = {}
    for key, value in values.items():
        default = defaults.get(key)
        if isinstance(value, dict) and isinstance(default, dict) and key not in ('destinations', 'homes', 'favorites'):
            sub = _diff(value, default)
            if sub:
                out[key] = sub
        elif _norm(value) != _norm(default):
            out[key] = value
    return out


def _norm(value):
    return [_norm(v) for v in value] if isinstance(value, (list, tuple)) else value


def _from_dict(kind, data):
    values = {}
    for f in fields(kind):
        if f.name not in data:
            continue
        value = data[f.name]
        default = f.default_factory() if callable(f.default_factory) else f.default
        if is_dataclass(default):
            value = _from_dict(type(default), value)
        elif isinstance(default, tuple):
            value = tuple(value)
        values[f.name] = value
    return kind(**values)
