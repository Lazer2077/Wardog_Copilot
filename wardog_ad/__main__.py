"""Command line: python -m wardog_ad [command] ...   (no command = start)

  start            one launcher for truck and helicopter with the button panel
                   (F8 go/pause, F7 return, F6 truck<->helicopter, F9 stop)

  calibrate        select minimap / main view / lock-warning regions on screen
  collect          read-only manual driving video, actual key state and navigation
  collect-calibrate select truck speed digits separately for first/third person
  dataset-check    inspect a recorded session's video and frame index
  lock-template    save the current lock-warning region as the matching template
  lock-test        print the live (visual) missile-lock detector score
  hud-test         print the live helicopter HUD reading (speed / altitude / heading)
  audio-record     record a lock-warning / background clip from the speakers
  audio-harvest    cut lock-tone clips out of a long recording (e.g. a YouTube video)
  audio-fit        learn the lock tones from the recorded clips
  audio-test       print the live audio lock state (none / acquiring / locked)
  locate           test minimap localization (live or --image)
  route            plan and preview a route (no input sent)
  set-dest         save the destination (then just press F8 in the vehicle)
  set-home         fix the return point (default: where you first press F8)
  drive            run the autopilot (F8 go/pause/resume, F7 return home, F9 e-stop)
  flares           aircraft: only auto-flares on missile lock, you fly manually
  sim              closed-loop offline simulation on the real map
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

from .assets import DEFAULT_ASSETS, load_map
from .config import DEFAULT_PATH, Config


def _point(text: str):
    parts = text.replace(';', ',').split(',')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('expected "x,y"')
    return float(parts[0]), float(parts[1])


def _resolve(assets, args, point):
    return assets.game_to_map(point) if args.game else point


def _setup(args) -> tuple[Config, object]:
    cfg = Config.load(args.config)
    if getattr(args, 'profile', None):
        cfg.profile = args.profile
    if getattr(args, 'map', None):
        cfg.map_id = args.map
    assets = load_map(cfg.map_id, cfg.assets_dir or DEFAULT_ASSETS)
    return cfg, assets


def cmd_calibrate(args):
    from .capture import ScreenCapture, select_region, select_anchor
    from .input_win import set_dpi_aware
    set_dpi_aware()
    cfg = Config.load(args.config)
    cap = ScreenCapture()
    calibration = getattr(cfg, f'{args.profile}_minimap') if args.profile else None
    print('Switch to the game; screenshot in 3 s ...')
    time.sleep(3)
    screen, monitor = cap.grab_monitor(args.monitor)
    if args.anchor_only:
        from .driving_data import crop_region
        from .config import Region
        client = Region(monitor['left'], monitor['top'], monitor['width'], monitor['height'])
        region = calibration.region if calibration is not None and calibration.region.valid else cfg.minimap
        from .capture import resolve_region
        region = resolve_region(region, cap.monitors())
        roi = crop_region(screen, client, region)
        if roi is None:
            sys.exit('Calibrate the minimap region on this monitor first.')
        old_anchor = calibration.anchor if calibration is not None and calibration.anchor else cfg.anchor
        anchor = select_anchor(roi, old_anchor)
        if anchor is not None:
            if calibration is None:
                cfg.anchor = anchor
            else:
                calibration.anchor = anchor
            cfg.save(args.config)
            print(f'saved player position anchor: {anchor}')
        return
    targets = [('minimap', 'Drag the MINIMAP map area (no border), Enter to confirm'),
               ('main_view', 'Drag a central part of the 3-D view (for stuck detection), Enter; Esc to skip')]
    if args.lock or (args.profile or cfg.profile) == 'aircraft':
        targets.append(('lock_warning', 'Drag the MISSILE-LOCK warning area of the HUD, Enter; Esc to skip'))
    for name, title in targets:
        region = select_region(screen, monitor, title)
        if region is not None:
            if name == 'minimap' and calibration is not None:
                calibration.region = region
                calibration.anchor = ()
            else:
                setattr(cfg, name, region)
            print(f'{name}: {region}')
    cfg.save(args.config)
    print(f'saved {args.config}')


def cmd_lock_template(args):
    import cv2
    from .capture import ScreenCapture
    from .input_win import set_dpi_aware
    set_dpi_aware()
    cfg = Config.load(args.config)
    if not cfg.lock_warning.valid:
        sys.exit('calibrate the lock_warning region first (calibrate --lock)')
    print(f'Get locked in game; capturing in {args.delay:.0f} s ...')
    time.sleep(args.delay)
    roi = ScreenCapture().grab(cfg.lock_warning)
    out = Path(args.output)
    cv2.imencode('.png', roi)[1].tofile(str(out))
    cfg.lock_template = str(out.resolve())
    cfg.save(args.config)
    print(f'template saved to {out}; tighten it in an image editor to only the warning symbol for best results')


def cmd_lock_test(args):
    from .autopilot import ScreenSensors
    from .input_win import set_dpi_aware
    from .perception import MissileLockDetector
    set_dpi_aware()
    cfg = Config.load(args.config)
    sensors = ScreenSensors(cfg)
    det = MissileLockDetector(cfg.aircraft, cfg.lock_template)
    try:
        while True:
            active = det.update(sensors.lock_roi(), time.monotonic())
            r = det.reading
            tpl = '-' if r.template_score is None else f'{r.template_score:.2f}'
            print(f'\rscore={r.score:.2f} colour={r.color_ratio:.3f} template={tpl} LOCK={active}   ', end='')
            time.sleep(1 / cfg.aircraft.detector_hz)
    except KeyboardInterrupt:
        print()


AUDIO_DIR = Path('audio_clips')


def cmd_audio_record(args):
    from .audio import SAMPLE_RATE, record_loopback, write_wav
    AUDIO_DIR.mkdir(exist_ok=True)
    hints = {'acquire': 'let a launcher ACQUIRE you (beeping) for the whole clip',
             'locked': 'stay LOCKED (steady tone) for the whole clip',
             'missile': 'record while a missile is INCOMING (the rushing missile sound; flares off)',
             'background': 'fly normally WITHOUT any lock warning (rotor, guns, radio are welcome)',
             'raw': 'long take (e.g. play a YouTube video); cut it with audio-harvest'}
    print(f'{args.kind}: {hints[args.kind]}. Recording {args.seconds:.0f} s in {args.delay:.0f} s ...')
    time.sleep(args.delay)
    audio = record_loopback(args.seconds, Config.load(args.config).audio_device or None)
    index = 0
    while (AUDIO_DIR / f'{args.kind}_{index:02d}.wav').exists():
        index += 1
    path = AUDIO_DIR / f'{args.kind}_{index:02d}.wav'
    write_wav(path, audio, SAMPLE_RATE)
    print(f'saved {path} ({len(audio) / SAMPLE_RATE:.1f} s, peak {abs(audio).max():.2f})')


def _load_clips(kind):
    import numpy as np
    from .audio import _resample, read_wav
    paths = [p for p in sorted(AUDIO_DIR.glob(f'{kind}_*.wav')) if '_candidate_' not in p.name]
    clips = [_resample(*read_wav(p)) for p in paths]      # unconfirmed candidates are never used
    return np.concatenate(clips) if clips else None


def cmd_audio_harvest(args):
    """Cut lock-warning clips out of a long recording (e.g. a YouTube video)."""
    from .audio import SAMPLE_RATE, _resample, read_wav, write_wav
    from .harvest import harvest, missile_candidates
    audio = _resample(*read_wav(args.input))
    s0 = int(args.start * SAMPLE_RATE)
    audio = audio[s0:int(args.end * SAMPLE_RATE) if args.end else None]   # only the wanted video
    runs, groups, segs, quiet = harvest(audio)
    if args.start:
        print(f'(times below are relative to {args.start:.1f} s in the recording)')
    print(f'{len(audio) / SAMPLE_RATE:.0f} s analysed, {len(runs)} stable tone runs')
    for g in groups[:8]:
        extra = f' period {g.period_s:.2f}s' if g.period_s else ''
        print(f'  {g.freq:7.1f} Hz  {g.kind:8s} {len(g.runs):3d} runs, {g.total_s:5.1f} s{extra}')
    if not args.write:
        print('add --write to save the clips into audio_clips/')
    AUDIO_DIR.mkdir(exist_ok=True)
    tag = args.tag
    plan = [('acquire', segs['acquire'][:args.clips]), ('locked', segs['locked'][:args.clips]),
            ('background', quiet[:args.clips])]
    for kind, windows in plan:
        for i, (s, e) in enumerate(windows):
            name = AUDIO_DIR / f'{kind}_{tag}{i:02d}.wav'
            print(f'  {kind:10s} {s:7.1f}-{e:7.1f} s -> {name if args.write else "(not written)"}')
            if args.write:
                write_wav(name, audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)])
    cands = missile_candidates(audio, segs['locked'])
    for i, (s, e, rise) in enumerate(cands):
        name = AUDIO_DIR / f'missile_candidate_{tag}{i:02d}.wav'
        print(f'  missile?   {s:7.1f}-{e:7.1f} s (2-8 kHz rise {rise:+.1f} dB) -> '
              f'{name if args.write else "(not written)"}')
        if args.write:
            write_wav(name, audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)])
    if cands:
        print('listen to the missile candidates; rename the real ones to missile_*.wav before audio-fit')


def cmd_audio_fit(args):
    from .audio import fit_profile
    background = _load_clips('background')
    if background is None:
        sys.exit('record a background clip first: audio-record --kind background')
    clips = {k: a for k in ('acquire', 'locked', 'missile') if (a := _load_clips(k)) is not None}
    if not clips:
        sys.exit('record at least one warning clip: audio-record --kind locked')
    pinned = {k: v for k, v in (('acquire', args.acquire_hz), ('locked', args.locked_hz)) if v}
    profile = fit_profile(clips, background, pinned)
    out = Path(args.output)
    profile.save(out)
    cfg = Config.load(args.config)
    cfg.audio_profile = str(out.resolve())
    cfg.save(args.config)
    for kind, sig in profile.signatures.items():
        pulse = f'period {sig.period_s:.2f}s duty {sig.duty:.2f}' if sig.period_s else 'continuous'
        print(f'{kind:8s} tones {sig.freqs} Hz  threshold {sig.threshold:.1f} dB  {pulse}')
    if profile.missile:
        print(f'missile  broadband sound learned (threshold {profile.missile.threshold:.2f}); '
              'flares fire when it is heard after a lock warning')
    else:
        print('no missile sound recorded: flares use the lock tone (audio-record --kind missile to add it)')
    print(f'profile saved to {out}')


def cmd_audio_test(args):
    from .audio import AudioLockDetector, LockProfile, LoopbackTap
    cfg = Config.load(args.config)
    if not cfg.audio_profile:
        sys.exit('run audio-fit first')
    det = AudioLockDetector(LockProfile.load(cfg.audio_profile))
    tap = LoopbackTap(cfg.audio_device or None).start()
    try:
        while True:
            time.sleep(0.05)
            if tap.error:
                raise tap.error
            state = det.feed(tap.read(), time.monotonic())
            scores = ' '.join(f'{k}={v:5.1f}/{det.profile.signatures[k].threshold:.1f}dB' for k, v in det.scores.items())
            print(f'\r{state.upper():10s} {scores}   ', end='')
    except KeyboardInterrupt:
        print()
    finally:
        tap.close()


def cmd_hud_test(args):
    from .autopilot import ScreenSensors
    from .input_win import set_dpi_aware
    set_dpi_aware()
    cfg = Config.load(args.config)
    if not cfg.hud.valid:
        sys.exit('no HUD region in config.json')
    sensors = ScreenSensors(cfg)
    try:
        while True:
            t0 = time.monotonic()
            h = sensors.hud()
            fmt = lambda v: '  --' if v is None else f'{v:4.0f}'
            print(f'\rSPD {fmt(h.speed_kmh)} km/h  AGL {fmt(h.agl_m)} m  ASL {fmt(h.asl_m)} m  '
                  f'HDG {fmt(h.heading)}  ({(time.monotonic() - t0) * 1000:3.0f} ms, '
                  f'{len(sensors._hud.library.templates)}/10 digits learned)   ', end='')
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()


def cmd_locate(args):
    import cv2
    import numpy as np
    from .localization import MinimapLocalizer
    cfg, assets = _setup(args)
    loc = MinimapLocalizer.from_assets(assets)
    if args.image:
        image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
        fix = loc.locate(image, cfg.localization_anchor, cfg.north_up)
        print(fix)
        if fix.valid:
            print('game coords: %.2f, %.2f' % assets.map_to_game(fix.position))
        return
    from .autopilot import ScreenSensors
    from .input_win import set_dpi_aware
    set_dpi_aware()
    sensors = ScreenSensors(cfg)
    try:
        while True:
            t0 = time.monotonic()
            fix = loc.locate(sensors.minimap(), cfg.localization_anchor, cfg.north_up)
            hdg = '-' if fix.heading is None else f'{fix.heading:5.1f}'
            print(f'\r{fix.reason:18s} x={fix.x:7.1f} y={fix.y:7.1f} hdg={hdg} conf={fix.confidence:.2f} '
                  f'{(time.monotonic() - t0) * 1000:4.0f} ms   ', end='')
    except KeyboardInterrupt:
        print()


def cmd_route(args):
    from .debug_view import route_preview
    from .planner import RoadGraph, straight_route
    cfg, assets = _setup(args)
    anchors = [_resolve(assets, args, p) for p in [args.start, *args.via, args.to]]
    if cfg.profile == 'aircraft':
        route = straight_route(anchors)
    else:
        route = RoadGraph(assets.roads).route(anchors, kinds=cfg.truck.road_kinds, max_snap=cfg.truck.max_snap_px)
    from .geometry import cumulative
    length = cumulative(route)[-1] * assets.meters_per_pixel
    route_preview(assets.load_image(), route, args.output)
    print(f'{len(route)} points, {length:.0f} m -> {args.output}')


def cmd_drive(args, flares_only=False):
    from .autopilot import Autopilot, ScreenSensors, run_live
    from .input_win import WindowsKeyboard, set_dpi_aware
    set_dpi_aware()
    cfg, assets = _setup(args)
    if flares_only:
        cfg.profile = 'aircraft'
    elif not cfg.localization_region.valid:
        sys.exit('run "python -m wardog_ad calibrate" first')
    if getattr(args, 'no_flares', False):
        cfg.aircraft.lock_source = 'none'
    pilot = Autopilot(cfg, assets, WindowsKeyboard(), ScreenSensors(cfg))
    pilot.config_path = args.config
    if not flares_only:
        if args.to:
            pilot.destination = _resolve(assets, args, args.to)
        if args.home:
            pilot.home = _resolve(assets, args, args.home)
        pilot.via = [_resolve(assets, args, p) for p in args.via]
        if pilot.destination is None:
            print('warning: no destination yet (set-dest or --to); only return-home works')
        k = cfg.keys
        vehicle = 'HELICOPTER' if cfg.profile == 'aircraft' else 'TRUCK'
        print(f'[{vehicle}] get in, then {k.toggle} = go / pause / resume, {k.return_home} = return home, '
              f'{k.switch_vehicle} = truck <-> helicopter, {k.emergency_stop} = emergency stop, Ctrl+C = quit')
    else:
        source = cfg.aircraft.lock_source
        if source in ('visual', 'both') and not cfg.lock_warning.valid:
            sys.exit('run "python -m wardog_ad calibrate --lock" first')
        if source in ('audio', 'both') and not cfg.audio_profile:
            sys.exit('run "audio-record" (acquire / locked / background) and "audio-fit" first')
        print(f'auto-flares on missile lock (key {cfg.keys.flare}); {cfg.keys.toggle} toggles, '
              f'{cfg.keys.emergency_stop} stops; Ctrl+C quits')
    run_live(pilot, view=args.view, engage_now=getattr(args, 'engage', False), flares_only=flares_only,
             panel=args.panel)


def cmd_set_dest(args):
    cfg, assets = _setup(args)
    point = _resolve(assets, args, args.to)
    if not assets.contains(point):
        sys.exit(f'{point} is outside the {cfg.map_id} map')
    cfg.destinations[cfg.map_id] = [round(point[0], 2), round(point[1], 2)]
    cfg.save(args.config)
    print(f'destination saved for {cfg.map_id}: map px {cfg.destinations[cfg.map_id]}')


def cmd_set_home(args):
    cfg, assets = _setup(args)
    if args.clear:
        cfg.homes.pop(cfg.map_id, None)
        print('home cleared: it will be remembered where you first press go')
    else:
        if not args.at:
            sys.exit('give --at x,y or --clear')
        point = _resolve(assets, args, args.at)
        cfg.homes[cfg.map_id] = [round(point[0], 2), round(point[1], 2)]
        print(f'home saved for {cfg.map_id}: map px {cfg.homes[cfg.map_id]}')
    cfg.save(args.config)


def cmd_sim(args):
    from .sim import run_sim
    cfg, assets = _setup(args)
    start = _resolve(assets, args, args.start)
    goal = _resolve(assets, args, args.to)
    result = run_sim(cfg, assets, start, goal, heading=args.heading, vision=args.vision, view=args.view,
                     waypoints=[_resolve(assets, args, p) for p in args.via], max_time=args.max_time,
                     locks=args.lock_at)
    print(f'arrived={result.arrived} time={result.time:.1f}s goal_dist={result.distance_to_goal_m:.1f}m '
          f'max_cross={result.max_cross_track_m:.1f}m flares={result.flares} '
          f'missiles defeated={result.missiles_defeated} hit={result.missiles_hit}')


def cmd_fit_slip(args):
    """Measure the helicopter sideslip (how fast the motion follows the nose) from flight logs."""
    from .config import Config
    from .slipfit import fit_slip, latest_log, load_rows
    paths = [Path(p) for p in args.log] if args.log else [latest_log()]
    if not paths or paths[0] is None:
        sys.exit('no flight log in logs/ - fly first (the recorder writes logs/flight_*.jsonl)')
    rows = [r for path in paths for r in load_rows(path)]
    fit = fit_slip(rows)
    print(f'logs: {", ".join(p.name for p in paths)}')
    for align, err in fit.errors.items():
        print(f'  align {align:>5}/s: median error {err:6.1f} m{"   <- best" if align == fit.best else ""}')
    print(fit.message)
    if args.write and fit.best is not None:
        cfg = Config.load(args.config)
        cfg.aircraft.slip_align = fit.best
        cfg.save(args.config)
        print(f'saved aircraft.slip_align = {fit.best}')


def cmd_collect(args):
    from .driving_data import run_collection
    cfg, assets = _setup(args)
    cfg.profile = 'truck'
    if args.map:
        cfg.auto_map.enabled = False  # an explicit map pins this collection only
    if args.to is not None:
        point = _resolve(assets,args,args.to)
        if not assets.contains(point):
            raise ValueError('collection destination is outside the selected map')
        cfg.destinations[cfg.map_id] = list(point)
        saved = Config.load(args.config)
        saved.destinations[cfg.map_id] = list(point)
        saved.save(args.config)
    if args.fps is not None:
        cfg.collection.fps = args.fps
    if args.max_width is not None:
        cfg.collection.max_width = args.max_width
    run_collection(cfg, assets, view=args.camera_view, vehicle=args.vehicle, output=args.output,
                   config_path=args.config, seconds=args.seconds, panel=not args.no_panel)


def cmd_collect_calibrate(args):
    from .capture import ScreenCapture, select_region
    from .input_win import set_dpi_aware
    set_dpi_aware()
    cfg = Config.load(args.config)
    print('Switch to the requested game camera in 3 seconds. Select speed DIGITS ONLY (exclude SPD / KM/H).')
    time.sleep(3)
    screen, monitor = ScreenCapture().grab_monitor(args.monitor)
    region = select_region(screen, monitor, f'{args.camera_view}: SPEED DIGITS ONLY; Enter to save; Esc to cancel')
    if region is not None:
        setattr(cfg.collection, f'speed_{args.camera_view}', region)
        cfg.save(args.config)
        print(f'saved speed digits for {args.camera_view}: {region}')


def cmd_dataset_check(args):
    import json
    from .driving_data import inspect_dataset
    result = inspect_dataset(args.directory)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['ok']:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='wardog_ad', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=str(DEFAULT_PATH))
    parser.add_argument('-v', '--verbose', action='store_true')
    sub = parser.add_subparsers(dest='command')

    def common(p, route=True, to_required=True):
        p.add_argument('--profile', choices=('truck', 'aircraft'))
        p.add_argument('--map', help='ozeti | bakurani | zestafona')
        if route:
            p.add_argument('--to', type=_point, required=to_required, help='destination "x,y"')
            p.add_argument('--via', type=_point, nargs='*', default=[], help='waypoints "x,y" ...')
            p.add_argument('--game', action='store_true', help='points are in-game x/y coordinates, not map pixels')

    p = sub.add_parser('collect', help='read-only manual driving video/input/navigation recording')
    p.add_argument('--map')
    p.add_argument('--to', type=_point, help='navigation destination x,y (map pixels unless --game)')
    p.add_argument('--game', action='store_true', help='--to is in game coordinates')
    p.add_argument('--camera-view', choices=('first_person', 'third_person', 'unknown'), default='unknown')
    p.add_argument('--vehicle', required=True, help='vehicle name, e.g. URAL or DUNE_BUGGY')
    p.add_argument('--fps', type=float, help='saved video target FPS; config default: 15')
    p.add_argument('--max-width', type=int, help='maximum saved image width; config default: 512, aspect ratio preserved')
    p.add_argument('--output', help='parent folder for a unique session directory')
    p.add_argument('--seconds', type=float, help='finish after this many wall-clock seconds, including pauses')
    p.add_argument('--no-panel', action='store_true')
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser('collect-calibrate', help='calibrate truck speed digits separately for each view')
    p.add_argument('--camera-view', choices=('first_person', 'third_person'), required=True)
    p.add_argument('--monitor', type=int, default=0)
    p.set_defaults(func=cmd_collect_calibrate)

    p = sub.add_parser('dataset-check', help='verify recorded video segments and temporal frame index')
    p.add_argument('directory')
    p.set_defaults(func=cmd_dataset_check)

    p = sub.add_parser('calibrate')
    p.add_argument('--profile', choices=('truck', 'aircraft'), help='store minimap calibration separately for this vehicle mode')
    p.add_argument('--monitor', type=int, default=0)
    p.add_argument('--lock', action='store_true', help='also select the missile-lock warning region')
    p.add_argument('--anchor-only', action='store_true', help='click the exact player position inside the existing minimap crop')
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser('lock-template')
    p.add_argument('--delay', type=float, default=5.0)
    p.add_argument('--output', default='lock_template.png')
    p.set_defaults(func=cmd_lock_template)

    sub.add_parser('lock-test').set_defaults(func=cmd_lock_test)
    sub.add_parser('hud-test').set_defaults(func=cmd_hud_test)

    p = sub.add_parser('audio-record')
    p.add_argument('--kind', choices=('acquire', 'locked', 'missile', 'background', 'raw'), required=True,
                   help='raw = long take (e.g. a YouTube video) for audio-harvest')
    p.add_argument('--seconds', type=float, default=8.0)
    p.add_argument('--delay', type=float, default=3.0)
    p.set_defaults(func=cmd_audio_record)

    p = sub.add_parser('audio-harvest')
    p.add_argument('--input', required=True, help='long recording, e.g. audio_clips/raw_00.wav')
    p.add_argument('--write', action='store_true', help='write the clips into audio_clips/')
    p.add_argument('--clips', type=int, default=2, help='clips per kind')
    p.add_argument('--tag', default='yt', help='file name tag for the written clips')
    p.add_argument('--start', type=float, default=0.0, help='use the recording from here (s)')
    p.add_argument('--end', type=float, default=None, help='... up to here (s)')
    p.set_defaults(func=cmd_audio_harvest)

    p = sub.add_parser('audio-fit')
    p.add_argument('--output', default='lock_audio.json')
    p.add_argument('--acquire-hz', type=float, nargs='+', help='pin the "being locked" beep frequency (Hz)')
    p.add_argument('--locked-hz', type=float, nargs='+', help='pin the "locked" tone frequency (Hz)')
    p.set_defaults(func=cmd_audio_fit)

    sub.add_parser('audio-test').set_defaults(func=cmd_audio_test)

    p = sub.add_parser('locate')
    common(p, route=False)
    p.add_argument('--image')
    p.set_defaults(func=cmd_locate)

    p = sub.add_parser('route')
    common(p)
    p.add_argument('--from', dest='start', type=_point, required=True)
    p.add_argument('--output', default='route_preview.png')
    p.set_defaults(func=cmd_route)

    p = sub.add_parser('set-dest')
    common(p, route=False)
    p.add_argument('--to', type=_point, required=True)
    p.add_argument('--game', action='store_true', help='in-game x/y coordinates')
    p.set_defaults(func=cmd_set_dest)

    p = sub.add_parser('set-home')
    common(p, route=False)
    p.add_argument('--at', type=_point)
    p.add_argument('--clear', action='store_true')
    p.add_argument('--game', action='store_true', help='in-game x/y coordinates')
    p.set_defaults(func=cmd_set_home)

    p = sub.add_parser('drive')
    common(p, to_required=False)
    p.add_argument('--home', type=_point, help='return point (default: where you first press go)')
    p.add_argument('--view', action='store_true', help='show the debug window')
    p.add_argument('--panel', action='store_true', help='show the button panel')
    p.add_argument('--engage', action='store_true', help='go immediately instead of waiting for F8')
    p.add_argument('--no-flares', action='store_true', help='aircraft: no automatic flares')
    p.set_defaults(func=cmd_drive)

    p = sub.add_parser('start')
    p.add_argument('--no-panel', dest='panel', action='store_false', help='hotkeys only')
    p.add_argument('--view', action='store_true', help='show the debug window')
    p.add_argument('--no-flares', action='store_true')
    p.set_defaults(func=cmd_drive, to=None, home=None, via=[], game=False, engage=False, profile=None, map=None)

    p = sub.add_parser('flares')
    common(p, route=False)
    p.add_argument('--panel', action='store_true')
    p.set_defaults(func=lambda a: cmd_drive(a, flares_only=True), view=False)

    p = sub.add_parser('fit-slip', help='helicopter: measure the sideslip after fast turns from flight logs')
    p.add_argument('--log', nargs='*', help='flight logs (default: the newest in logs/)')
    p.add_argument('--write', action='store_true', help='save the result as aircraft.slip_align')
    p.set_defaults(func=cmd_fit_slip)

    p = sub.add_parser('sim')
    common(p)
    p.add_argument('--from', dest='start', type=_point, required=True)
    p.add_argument('--heading', type=float, default=0.0)
    p.add_argument('--vision', action='store_true', help='localize from rendered minimaps (slow, realistic)')
    p.add_argument('--view', action='store_true')
    p.add_argument('--max-time', type=float, default=600.0)
    p.add_argument('--lock-at', type=float, nargs='*', default=[], help='aircraft: simulated missile lock times (s)')
    p.set_defaults(func=cmd_sim)

    argv = sys.argv[1:] if argv is None else list(argv)
    if not any(a in sub.choices for a in argv) and not any(a in ('-h', '--help') for a in argv):
        argv.append('start')        # double-click / no command: start the launcher
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format='%(asctime)s %(levelname)s %(message)s')
    args.func(args)


if __name__ == '__main__':
    main()
