"""Offline closed-loop simulator.

Renders a north-up minimap (with a white player chevron) from the real base map,
a synthetic main view, and a red lock warning panel, and integrates simple truck /
aircraft dynamics from the keys the autopilot presses.  With ``vision=True`` the
autopilot localizes from the rendered minimap exactly as in the game.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import random

import cv2
import numpy as np

from .autopilot import Autopilot
from .config import Config
from .geometry import heading_vector
from .input_win import FakeKeyboard
from .localization import Fix


@dataclass
class Body:
    x: float
    y: float
    heading: float
    speed: float = 0.0     # m/s (negative = reversing)
    side: float = 0.0      # m/s sideways (+ = right), helicopter sideslip
    steer: float = 0.0     # truck wheel angle, deg
    bank: float = 0.0      # aircraft bank, deg
    alt: float = 0.0       # helicopter height above ground, m
    vz: float = 0.0        # climb rate, m/s
    yaw_rate: float = 0.0  # deg/s
    pitch: float = 0.0     # deg, + = nose down
    lever: float = -1.0    # collective indicator, -1 (down) .. +1
    spool: float = 0.0     # seconds of Shift held on the ground
    asl: float | None = None


@dataclass
class LockEvent:
    """Acquire (beeping) -> locked (steady tone) -> launch -> impact.

    A flare defeats the missile only inside ``flare_window`` seconds before impact
    (WARDOGS guides: fire 1-2 s before impact; early flares are wasted).
    """
    start: float
    acquire_s: float = 1.5
    launch_after_lock_s: float = 0.4
    flight_s: float = 2.5
    flare_window: float = 2.0
    defeated: bool = False
    resolved: bool = False
    wasted_flares: int = 0

    @property
    def lock_time(self):
        return self.start + self.acquire_s

    @property
    def impact(self):
        return self.lock_time + self.launch_after_lock_s + self.flight_s

    def phase(self, t: float) -> str:
        if self.resolved or t < self.start:
            return 'none'
        return 'acquiring' if t < self.lock_time else 'locked'


@dataclass
class SimWorld:
    cfg: Config
    base: np.ndarray
    mpp: float
    body: Body
    keyboard: FakeKeyboard
    minimap_size: int = 300
    minimap_zoom: float = 2.0          # minimap px per map px
    walls: list = field(default_factory=list)       # [(x0, y0, x1, y1)] map px — impassable boxes
    locks: list = field(default_factory=list)       # [LockEvent]
    time: float = 0.0
    flares: int = 0
    hits: int = 0
    blocked: bool = False
    tone_level: float = 0.025
    missile_level: float = 0.12        # loudness of the incoming-missile rush at impact (0 = silent)
    ground_asl: float = 84.0
    hills: float = 0.0                 # terrain amplitude (m); 0 = flat
    ridge: tuple | None = None         # (x0 px, slope, height m): rising mountainside
    crashes: int = 0
    heli_auto_level: bool = True       # stability assist levels the nose when W/S are released
    heli_pitch_rate: float = 37.0      # deg/s with W/S held (measured: ~3.7 ticks/s)
    heli_vert_delay: float = 1.2       # s before Shift/Ctrl show in the climb rate (measured)
    heli_vert_tau: float = 2.5         # s lag of the climb rate (measured)
    heli_yaw_scale: float = 1.0        # < 1: a sluggish yaw (real flight 01:52 turned ~3-20 deg/s)
    # how fast the direction of travel swings onto the nose (1/s).  inf = always along the
    # nose (arcade); 0 = pure physics (thrust along the nose, drag against the motion), so a
    # fast pedal turn leaves the helicopter sliding on.  Not yet measured in the game.
    heli_align: float = math.inf       # (fit-slip on 13:16 suggested 0.02 = sliding; tests choose it explicitly)
    hud_drops_hundreds: bool = False   # HUD reader bug seen at 10:56: 100-199 km/h read without the '1'
    hud_speed_glitch: float = 0.0      # chance per reading of a garbled speed (534 km/h at 12:54)
    hard_landings: int = 0
    touchdowns: list = field(default_factory=list)
    _col_hist: deque = field(default_factory=deque)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    _taps_seen: int = 0
    _background: np.ndarray | None = None

    # ---------------------------------------------------------------- dynamics
    def step(self, dt: float):
        k, keys = self.cfg.keys, self.keyboard.down
        b = self.body
        if self.cfg.profile == 'aircraft' and self.cfg.aircraft.flight_model == 'helicopter':
            self._heli(dt, k, keys, b)
        elif self.cfg.profile == 'aircraft':
            roll = (k.roll_right in keys) - (k.roll_left in keys)
            b.bank = float(np.clip(b.bank + roll * 45 * dt - (0 if roll else np.sign(b.bank) * 8 * dt), -70, 70))
            yaw = (k.yaw_right in keys) - (k.yaw_left in keys)
            b.speed = 80.0
            turn = math.degrees(9.81 * math.tan(math.radians(b.bank)) / b.speed) + yaw * 3.0
            b.heading = (b.heading + turn * dt) % 360
        else:
            steer = (k.right in keys) - (k.left in keys)
            target = steer * 32.0
            rate = 110.0 if steer else 180.0
            b.steer += float(np.clip(target - b.steer, -rate * dt, rate * dt))
            fwd, back = k.forward in keys, k.backward in keys
            if fwd and not back:
                acc = 3.2 if b.speed >= 0 else 9.0
            elif back and not fwd:
                acc = -9.0 if b.speed > 0.3 else -2.5
            else:
                acc = -np.sign(b.speed) * min(abs(b.speed) / dt, 1.0)
            b.speed = float(np.clip(b.speed + acc * dt - 0.004 * b.speed * abs(b.speed) * dt, -6.0, 22.0))
            b.heading = (b.heading + math.degrees(b.speed / 3.6 * math.tan(math.radians(b.steer))) * dt) % 360
        step = (heading_vector(b.heading) * b.speed + heading_vector(b.heading + 90) * b.side) * dt / self.mpp
        nx, ny = b.x + step[0], b.y + step[1]
        self.blocked = any(x0 <= nx <= x1 and y0 <= ny <= y1 for x0, y0, x1, y1 in self.walls)
        if self.blocked and self.cfg.profile == 'truck':
            b.speed = 0.0 if b.speed > 0 else b.speed  # can still reverse out
            if b.speed < 0:
                b.x, b.y = nx, ny
        else:
            b.x, b.y = nx, ny
        self._missiles()
        self.time += dt

    def _heli(self, dt, k, keys, b):
        """MH-6 model identified from the real flight logs: W/S tilt the nose ~37 deg/s
        (``heli_auto_level``: slowly levels when released); pitch sets acceleration
        (~0.9 m/s^2 per 10 deg); Shift/Ctrl command climb/descent with ``heli_vert_delay``
        s delay and 2.5 s lag (auto-hover holds when released); on the ground Shift
        must be held ~1 s before the skids leave the ground."""
        col = (k.throttle_up in keys) - (k.throttle_down in keys)
        b.lever += ((col if col else (-1.0 if b.alt <= 0 else 0.5)) - b.lever) * min(1.0, dt / 0.3)
        self._col_hist.append((self.time, col))
        while self._col_hist and self.time - self._col_hist[0][0] > self.heli_vert_delay + 0.5:
            self._col_hist.popleft()
        delayed = next((u for ts, u in reversed(self._col_hist) if ts <= self.time - self.heli_vert_delay), 0)
        cyclic = (k.pitch_down in keys) - (k.pitch_up in keys)
        if cyclic:
            b.pitch += self.heli_pitch_rate * cyclic * dt
        elif self.heli_auto_level:
            b.pitch -= np.sign(b.pitch) * min(abs(b.pitch), 8.0 * dt)
        b.pitch = float(np.clip(b.pitch, -45.0, 45.0))
        ticks = b.pitch / 10.0
        ground = self.ground_height(b.x, b.y)
        if b.alt <= 0.0:
            b.spool = b.spool + dt if col > 0 else max(0.0, b.spool - 2 * dt)
            vz_target = 6.0 if b.spool > 1.0 else min(0.0, 12.0 * delayed)
        else:
            vz_target = 12.0 * delayed - 9.0 * (1 - math.cos(math.radians(b.pitch)))
        b.vz += (vz_target - b.vz) * min(1.0, dt / self.heli_vert_tau)
        if b.asl is None:
            b.asl = ground + b.alt
        b.asl += b.vz * dt
        b.alt = b.asl - ground
        if b.alt <= 0.0:
            if b.vz < -6.0 or (abs(b.pitch) > 20 and abs(b.speed) > 5) or (b.alt < -1.0 and abs(b.speed) > 10):
                self.crashes += 1
            elif b.vz < -2.5:
                self.hard_landings += 1
            if b.vz < -0.5:
                self.touchdowns.append((self.time, b.vz, b.speed))
            b.alt, b.vz, b.asl = 0.0, max(0.0, b.vz), ground
        if b.alt < 0.5:
            b.speed *= max(0.0, 1 - 3 * dt)                       # skids on the ground
        else:
            # g*tan(pitch) - drag; drag calibrated so ~3 ticks (30 deg) holds ~300 km/h
            # (player cruise) and the measured weak response at small pitch is kept
            v = math.hypot(b.speed, b.side)
            b.speed += (9.81 * math.tan(math.radians(b.pitch)) - 8.2e-4 * b.speed * v) * dt
            b.side -= 8.2e-4 * b.side * v * dt
        b.speed = float(np.clip(b.speed, -25.0, 97.0))     # 350 km/h top speed
        yaw = (k.yaw_right in keys) - (k.yaw_left in keys)
        max_rate = (45.0 if abs(b.speed) < 15 else 30.0) * self.heli_yaw_scale
        b.yaw_rate += (yaw * max_rate - b.yaw_rate) * min(1.0, dt / 0.3)
        b.heading = (b.heading + b.yaw_rate * dt) % 360
        if math.isinf(self.heli_align):
            b.side = 0.0
        else:
            # the nose turns under the motion: the world velocity stays, its body-frame parts rotate
            turn = math.radians(b.yaw_rate * dt)
            b.speed, b.side = b.speed + b.side * turn, b.side - b.speed * turn
            if b.speed > 0 and self.heli_align > 0:
                v, slip = math.hypot(b.speed, b.side), math.atan2(b.side, b.speed)
                slip *= math.exp(-self.heli_align * dt)
                b.speed, b.side = v * math.cos(slip), v * math.sin(slip)

    def ground_height(self, x: float, y: float) -> float:
        """Terrain height (m ASL) under map pixel (x, y): flat unless ``hills`` / ``ridge``."""
        h = self.ground_asl
        if self.hills:
            h += self.hills * (1 + math.sin(x / 45.0) * math.cos(y / 38.0))
        if self.ridge:
            # mountainside like the 22:32 crash: ground rises with ``slope`` east of x0
            # (then steepens), up to ``height`` metres, then a plateau
            x0, slope, height = self.ridge
            run = max(0.0, x - x0) * self.mpp
            h += min(height, slope * run + max(0.0, run - 600) * slope)
        return h

    def hud_reading(self):
        from .hud import HudReading
        b = self.body
        shown = float(round(math.hypot(b.speed, b.side) * 3.6))
        if self.hud_drops_hundreds and 100 <= shown < 200:
            shown -= 100
        if self.hud_speed_glitch and self.rng.random() < self.hud_speed_glitch:
            shown = float(self.rng.integers(300, 999))
        return HudReading(shown, float(round(b.alt)),
                          float(round(b.alt + self.ground_height(b.x, b.y))),
                          float(round(b.heading)) % 360, self.time,
                          pitch=b.pitch / 10.0 + self.rng.normal(0, .03), collective=b.lever + self.rng.normal(0, .01))

    def _missiles(self):
        new_taps = self.keyboard.taps[self._taps_seen:]
        self._taps_seen = len(self.keyboard.taps)
        flare = self.cfg.keys.flare in self.keyboard.down or self.cfg.keys.flare in new_taps
        if flare:
            self.flares += 1
        for ev in self.locks:
            if ev.resolved or self.time < ev.start:
                continue
            if flare:
                if ev.impact - ev.flare_window <= self.time <= ev.impact:
                    ev.defeated = ev.resolved = True
                else:
                    ev.wasted_flares += 1
            if not ev.resolved and self.time >= ev.impact:
                self.hits += 1
                ev.resolved = True

    def lock_phase(self, t: float | None = None) -> str:
        phases = {ev.phase(self.time if t is None else t) for ev in self.locks}
        return 'locked' if 'locked' in phases else 'acquiring' if 'acquiring' in phases else 'none'

    def lock_active(self) -> bool:
        return self.lock_phase() == 'locked'

    def audio_chunk(self, t0: float, t1: float, sr: int = 48000) -> np.ndarray:
        """Cockpit audio between t0 and t1: looped battle noise plus warning tones."""
        if self._background is None:
            self._background = synth_battle_audio(30.0, seed=5)
        idx = np.arange(int(t0 * sr), int(t1 * sr))
        out = self._background[idx % len(self._background)].copy()
        t = idx / sr
        for ev in self.locks:
            if ev.resolved and ev.impact < t0:
                continue
            acq = (t >= ev.start) & (t < ev.lock_time)
            lck = (t >= ev.lock_time) & (t < ev.impact)
            if acq.any():
                gate = ((t - ev.start) % 0.30) < 0.12
                out += acq * gate * self.tone_level * (np.sin(2 * np.pi * 1000 * t) + .4 * np.sin(4 * np.pi * 1000 * t))
            if lck.any():
                out += lck * self.tone_level * (np.sin(2 * np.pi * 1500 * t) + .4 * np.sin(4 * np.pi * 1500 * t))
            launch = ev.lock_time + ev.launch_after_lock_s
            flying = (t >= launch) & (t < ev.impact)
            if flying.any() and self.missile_level > 0:
                # incoming missile: a band-limited rush that gets louder as it closes in
                closing = np.clip((t - launch) / max(ev.flight_s, 1e-3), 0, 1)
                envelope = self.missile_level * (.08 + .92 * closing ** 2)
                out += flying * envelope * missile_noise(idx)
        return out.astype(np.float32)

    # ---------------------------------------------------------------- rendering
    def render_minimap(self) -> np.ndarray:
        n, z, b = self.minimap_size, self.minimap_zoom, self.body
        # affine: minimap pixel -> map pixel
        m = np.float32([[1 / z, 0, b.x - n / (2 * z)], [0, 1 / z, b.y - n / (2 * z)]])
        img = cv2.warpAffine(self.base, m, (n, n), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))
        c = np.array([n / 2, n / 2])
        f, r = heading_vector(b.heading), heading_vector(b.heading + 90)
        chevron = np.int32([c + f * 11, c - f * 7 + r * 8, c - f * 2, c - f * 7 - r * 8])
        cv2.fillPoly(img, [chevron], (255, 255, 255), cv2.LINE_AA)
        return img

    def render_main_view(self) -> np.ndarray:
        # a coarse "forward camera": rotated close-up of the terrain ahead
        b = self.body
        rot = cv2.getRotationMatrix2D((b.x, b.y), b.heading, 1.0)
        rot[:, 2] += [80 - b.x, 80 - b.y]
        return cv2.warpAffine(self.base, rot, (160, 120), borderMode=cv2.BORDER_REFLECT)

    def render_lock_roi(self) -> np.ndarray:
        roi = np.full((60, 200, 3), 30, np.uint8)
        # blinking red "MISSILE LOCK" panel while a missile is tracking us
        if self.lock_active() and int(self.time * 4) % 2 == 0:
            cv2.rectangle(roi, (20, 15), (180, 45), (0, 0, 230), -1)
            cv2.putText(roi, 'LOCK', (70, 38), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        return roi


class SimSensors:
    def __init__(self, world: SimWorld):
        self.world = world
        self._audio_t = 0.0

    def minimap(self):
        return self.world.render_minimap()

    def main_view(self):
        return self.world.render_main_view()

    def lock_roi(self):
        return self.world.render_lock_roi()

    def hud(self):
        return self.world.hud_reading()

    def attitude(self):
        hud = self.world.hud_reading()
        return None if hud is None else hud.pitch

    def audio(self):
        t1 = self.world.time
        chunk = self.world.audio_chunk(self._audio_t, t1) if t1 > self._audio_t else None
        self._audio_t = t1
        return chunk


class TruthLocator:
    """Fast stand-in for SIFT: noisy ground truth (heading from the 'arrow')."""

    def __init__(self, world: SimWorld, noise_px: float = 0.4, heading_noise: float = 3.0, seed: int = 1):
        self.world, self.noise, self.hnoise = world, noise_px, heading_noise
        self.rng = random.Random(seed)

    def locate(self, image, anchor, north_up) -> Fix:
        b = self.world.body
        return Fix(b.x + self.rng.gauss(0, self.noise), b.y + self.rng.gauss(0, self.noise),
                   (b.heading + self.rng.gauss(0, self.hnoise)) % 360, 1.0, 50, .5, .5, 0.0, True, 'truth',
                   self.world.time)


@dataclass
class SimResult:
    arrived: bool
    time: float
    distance_to_goal_m: float
    max_cross_track_m: float
    flares: int
    missiles_defeated: int
    missiles_hit: int
    wasted_flares: int
    crashes: int
    trace: list


def run_sim(cfg: Config, assets, start, goal, *, heading: float | None = None, waypoints=(), vision: bool = False,
            max_time: float = 600.0, walls=(), locks=(), view: bool = False, graph=None,
            lock_profile=None, world_options: dict | None = None) -> SimResult:
    base = assets.load_image()
    kb = FakeKeyboard()
    if heading is None:
        heading = 0.0
    world = SimWorld(cfg, base, assets.meters_per_pixel, Body(*start, heading), kb,
                     walls=list(walls), locks=[LockEvent(t) for t in locks])
    for key, value in (world_options or {}).items():
        setattr(world, key, value)
    if cfg.profile == 'aircraft':
        world.minimap_zoom = 1.0
        if cfg.aircraft.flight_model != 'helicopter':
            world.body.speed = 80.0
    locator = None if vision else TruthLocator(world)
    if cfg.profile == 'aircraft' and cfg.aircraft.lock_source != 'visual' and lock_profile is None:
        lock_profile = sim_lock_profile()
    pilot = Autopilot(cfg, assets, kb, SimSensors(world), locator=locator, graph=graph, lock_profile=lock_profile)
    pilot.set_goal(goal, waypoints)
    pilot.engage()
    viewer = None
    if view:
        from .debug_view import DebugView
        viewer = DebugView(assets, 'wardog_ad sim')

    physics_dt, control_dt, fast_dt = 0.02, 1.0 / cfg.loop_hz, 1.0 / 30
    next_control = next_fast = 0.0
    trace, max_cross, arrived = [], 0.0, False
    while world.time < max_time:
        if world.time >= next_fast:
            pilot.countermeasures(world.time)
            pilot.fast_control(world.time)
            pilot.actuate(world.time)
            next_fast += fast_dt
        if world.time >= next_control:
            tel = pilot.tick(world.time)
            next_control += control_dt
            if tel.output is not None:
                max_cross = max(max_cross, tel.output.info.get('cross_m', 0.0))
                if tel.output.done:
                    arrived = True
            trace.append((world.time, world.body.x, world.body.y, world.body.heading, world.body.speed, tel.status,
                          None if tel.output is None else dict(tel.output.axes, **tel.output.info)))
            if tel.output is not None and tel.output.takeover:
                break
            if viewer is not None and viewer.show(tel):
                break
            pending = any(not ev.resolved for ev in world.locks)
            if arrived and (abs(world.body.speed) < 0.5 or cfg.profile == 'aircraft') and not pending:
                break
            if tel.status == 'route error':
                break
        world.step(physics_dt)
    pilot.actuator.release()
    gx, gy = goal
    return SimResult(arrived, world.time, math.hypot(world.body.x - gx, world.body.y - gy) * assets.meters_per_pixel,
                     max_cross, pilot.telemetry.flares, sum(e.defeated for e in world.locks), world.hits,
                     sum(e.wasted_flares for e in world.locks), world.crashes, trace)


# --------------------------------------------------------------------------- audio
def synth_battle_audio(duration: float, *, seed: int = 0, sr: int = 48000, tone_level: float = 0.05) -> np.ndarray:
    """Helicopter cockpit-like background: rotor harmonics, wind, gunfire, explosions."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    rotor = sum(0.08 / k * np.sin(2 * np.pi * 21.0 * k * t + rng.uniform(0, 6)) for k in range(1, 12))
    rotor *= 1 + 0.5 * np.sin(2 * np.pi * 21.0 * t)                     # blade slap modulation
    wind = np.convolve(rng.normal(0, 0.06, len(t)), np.ones(8) / 8, 'same')
    audio = rotor + wind
    for start in rng.uniform(0, duration, int(duration * 1.5)):       # gunfire bursts
        i = int(start * sr)
        for shot in range(rng.integers(3, 12)):
            j = i + int(shot * 0.07 * sr)
            n = min(int(0.03 * sr), len(audio) - j)
            if n > 0:
                audio[j:j + n] += rng.normal(0, 0.35, n) * np.exp(-np.arange(n) / (0.006 * sr))
    for start in rng.uniform(0, duration, int(duration / 4)):         # explosions
        i, n = int(start * sr), int(0.8 * sr)
        n = min(n, len(audio) - i)
        audio[i:i + n] += np.convolve(rng.normal(0, 0.6, n), np.ones(30) / 30, 'same') * np.exp(-np.arange(n) / (0.2 * sr))
    return audio.astype(np.float32)


def synth_lock_tone(duration: float, kind: str, *, sr: int = 48000, level: float = 0.05,
                    freq: float | None = None, period: float = 0.30, duty: float = 0.4) -> np.ndarray:
    """'acquire' = beeping, 'locked' = steady tone (both with a 2nd harmonic)."""
    freq = freq or (1000.0 if kind == 'acquire' else 1500.0)
    t = np.arange(int(duration * sr)) / sr
    tone = level * (np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(4 * np.pi * freq * t))
    if kind == 'acquire':
        gate = ((t % period) < period * duty).astype(np.float32)
        gate = np.convolve(gate, np.ones(96) / 96, 'same')                 # soft edges
        tone *= gate
    return tone.astype(np.float32)


_MISSILE_NOISE = None


def missile_noise(idx: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Deterministic looped rushing noise (2-7 kHz band) for the incoming missile."""
    global _MISSILE_NOISE
    if _MISSILE_NOISE is None:
        rng = np.random.default_rng(42)
        n = 2 * sr
        spec = np.fft.rfft(rng.normal(0, 1, n))
        f = np.fft.rfftfreq(n, 1 / sr)
        spec *= ((f > 2000) & (f < 7000)) * (1 + 0.5 * np.sin(f / 300))
        noise = np.fft.irfft(spec, n)
        _MISSILE_NOISE = (noise / noise.std()).astype(np.float32)
    return _MISSILE_NOISE[idx % len(_MISSILE_NOISE)]


def synth_missile_sound(duration: float, *, sr: int = 48000, level: float = 0.12, start: int = 0) -> np.ndarray:
    idx = np.arange(start, start + int(duration * sr))
    return (level * missile_noise(idx)).astype(np.float32)


def sim_lock_profile(missile: bool = True):
    """Lock-tone (+ missile sound) profile fitted on synthetic clips matching ``audio_chunk``."""
    from .audio import fit_profile
    bg = synth_battle_audio(20, seed=1)
    clips = {'acquire': synth_battle_audio(6, seed=2) + synth_lock_tone(6, 'acquire', period=.3, duty=.4),
             'locked': synth_battle_audio(6, seed=3) + synth_lock_tone(6, 'locked')}
    if missile:
        clips['missile'] = synth_battle_audio(4, seed=4) + synth_missile_sound(4, level=.06, start=12345)
    return fit_profile(clips, bg)
