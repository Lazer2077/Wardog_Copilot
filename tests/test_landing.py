"""Helicopter auto-landing and the return-home flow (simulator with the identified dynamics)."""
import logging
import math

import pytest

from wardog_ad.assets import DEFAULT_ASSETS
from wardog_ad.config import Config

pytestmark = pytest.mark.skipif(not (DEFAULT_ASSETS / 'maps.json').exists(), reason='map assets not cloned')

START, GOAL = (400, 1100), (1000, 500)


def _setup(hills=60.0):
    from wardog_ad.assets import load_map
    from wardog_ad.autopilot import Autopilot
    from wardog_ad.input_win import FakeKeyboard
    from wardog_ad.sim import Body, SimSensors, SimWorld, TruthLocator
    assets = load_map('ozeti')
    cfg = Config(profile='aircraft')
    cfg.aircraft.lock_source = 'none'
    cfg.destinations = {'ozeti': list(GOAL)}
    kb = FakeKeyboard()
    w = SimWorld(cfg, assets.load_image(), assets.meters_per_pixel, Body(*START, 200.0), kb, hills=hills)
    pilot = Autopilot(cfg, assets, kb, SimSensors(w), locator=TruthLocator(w))
    return assets, cfg, w, pilot


def _landed_at_home(pilot, w, cfg, m=5.357339770866236):
    """On the touchdown spot, which may sit in the nearest open ground around home."""
    spot = pilot.controller.waypoints[-1]
    return (math.hypot(w.body.x - spot[0], w.body.y - spot[1]) * m < 40
            and math.hypot(spot[0] - START[0], spot[1] - START[1]) * m
            <= cfg.aircraft.open_radius_m + cfg.aircraft.spot_shift_m + 1)


def _run(cfg, w, pilot, until, limit, log=None):
    next_slow = w.time
    end = w.time + limit
    while w.time < end:
        if w.time >= next_slow:
            tel = pilot.tick(w.time)
            next_slow += 1 / cfg.loop_hz
            if log is not None:
                log.append((w.time, tel.status, w.body.alt))
            if until():
                return True
        pilot.fast_control(w.time)
        pilot.actuate(w.time)
        w.step(1 / 30)
    return False


def test_go_land_return_land():
    logging.disable(logging.WARNING)
    assets, cfg, w, pilot = _setup()
    m = assets.meters_per_pixel
    assert pilot.go()
    statuses = []
    assert _run(cfg, w, pilot, lambda: pilot.arrived and not pilot.engaged, 600, statuses)
    assert w.body.alt < 0.5 and abs(w.body.speed) < 2                       # on the ground, stopped

    def landed_near(goal):
        # on the touchdown spot, which the terrain check may have moved to flatter ground nearby
        spot = pilot.controller.waypoints[-1]
        return (math.hypot(w.body.x - spot[0], w.body.y - spot[1]) * m < 40
                and math.hypot(spot[0] - goal[0], spot[1] - goal[1]) * m <= cfg.aircraft.open_radius_m + cfg.aircraft.spot_shift_m + 1)
    assert landed_near(GOAL)
    assert {'hover', 'descend', 'flare', 'settle'} <= {s for _, s, _ in statuses}
    assert pilot.return_home()                                             # F7 from the landing spot
    assert _run(cfg, w, pilot, lambda: pilot.arrived and not pilot.engaged, 600)
    assert landed_near(START)
    logging.disable(logging.NOTSET)
    assert w.crashes == 0 and w.hard_landings == 0
    assert len(w.touchdowns) >= 1 and all(vz > -2.5 for _, vz, _ in w.touchdowns)


def test_return_pressed_mid_air_keeps_flying():
    """The real flight dropped all control when F7 was pressed in the air."""
    logging.disable(logging.WARNING)
    assets, cfg, w, pilot = _setup()
    assert pilot.go()
    _run(cfg, w, pilot, lambda: False, 50)                  # mid-cruise (trips are fast now)
    height = w.body.alt
    controller = pilot.controller
    assert height > 60 and pilot.engaged
    assert pilot.return_home()
    log = []
    assert _run(cfg, w, pilot, lambda: pilot.arrived and not pilot.engaged, 600, log)
    logging.disable(logging.NOTSET)
    assert pilot.controller is controller                                 # same flight state, new route
    cruise = [alt for _, s, alt in log if s not in ('hover', 'descend', 'flare', 'settle', 'landed')]
    assert min(cruise[:50]) > 0.6 * height                                # no sag right after the switch
    assert _landed_at_home(pilot, w, cfg)
    assert w.crashes == 0 and w.hard_landings == 0


def test_hover_only_when_auto_land_is_off():
    logging.disable(logging.WARNING)
    assets, cfg, w, pilot = _setup(hills=0.0)
    cfg.aircraft.auto_land = False
    assert pilot.go()
    assert _run(cfg, w, pilot, lambda: pilot.arrived and not pilot.engaged, 600)
    logging.disable(logging.NOTSET)
    assert w.body.alt > 15                                                # handed over in a hover


def test_go_again_at_destination_does_nothing():
    logging.disable(logging.WARNING)
    assets, cfg, w, pilot = _setup(hills=0.0)
    assert pilot.go()
    assert _run(cfg, w, pilot, lambda: pilot.arrived and not pilot.engaged, 600)
    pilot.toggle()                                   # F8 while parked at the destination
    logging.disable(logging.NOTSET)
    assert not pilot.engaged and 'already at the destination' in pilot.telemetry.message


def test_cruises_at_player_speed_on_a_long_trip():
    """Benchmark from player guides: the Little Bird cruises at ~300 km/h (top speed 350)."""
    logging.disable(logging.WARNING)
    assets, cfg, w, pilot = _setup(hills=0.0)
    start, goal = (100, 1350), (1500, 150)
    w.body.x, w.body.y = start
    pilot.destination = goal
    assert pilot.go()
    cruise_speeds = []
    next_slow = 0.0
    while w.time < 600 and not (pilot.arrived and not pilot.engaged):
        if w.time >= next_slow:
            if pilot.tick(w.time).status == 'cruise':
                cruise_speeds.append(w.body.speed * 3.6)
            next_slow += 1 / cfg.loop_hz
        pilot.fast_control(w.time)
        pilot.actuate(w.time)
        w.step(1 / 30)
    logging.disable(logging.NOTSET)
    assert pilot.arrived and w.crashes == 0 and w.hard_landings == 0
    assert sorted(cruise_speeds)[len(cruise_speeds) // 2] > 270      # median cruise speed
    trip_km = math.hypot(goal[0] - start[0], goal[1] - start[1]) * assets.meters_per_pixel / 1000
    assert trip_km / (w.time / 3600) > 150                           # door to door, incl. takeoff + landing


def test_land_button_mid_flight():
    """F10 while cruising: brake, hover, land on the spot (no destination needed)."""
    logging.disable(logging.WARNING)
    assets, cfg, w, pilot = _setup()
    assert pilot.go()
    _run(cfg, w, pilot, lambda: False, 45)                   # cruising
    assert w.body.alt > 60 and pilot.engaged
    pressed_at = (w.body.x, w.body.y)
    statuses = []
    assert pilot.land_here()
    assert _run(cfg, w, pilot, lambda: pilot.arrived and not pilot.engaged, 300, statuses)
    logging.disable(logging.NOTSET)
    assert pilot.mission == 'land' and 'landed' in pilot.telemetry.message
    assert w.body.alt < 0.5 and abs(w.body.speed) < 2
    assert w.crashes == 0 and w.hard_landings == 0 and all(vz > -2.5 for _, vz, _ in w.touchdowns)
    assert {'hover', 'descend', 'settle'} <= {s for _, s, _ in statuses}
    # braking from cruise speed: lands well within a kilometre of where the key was pressed
    assert math.hypot(w.body.x - pressed_at[0], w.body.y - pressed_at[1]) * assets.meters_per_pixel < 1000
    pilot.toggle()                                            # F8 afterwards: on to the destination
    assert pilot.engaged and pilot.mission == 'outbound'


def test_land_button_on_the_ground_and_without_position():
    logging.disable(logging.WARNING)
    assets, cfg, w, pilot = _setup(hills=0.0)
    assert pilot.land_here()                                  # parked, nothing planned yet
    assert _run(cfg, w, pilot, lambda: pilot.arrived and not pilot.engaged, 30)
    logging.disable(logging.NOTSET)
    assert w.body.alt < 0.5 and w.crashes == 0


def test_lands_after_a_lateral_miss():
    """Real flight 01:52: the helicopter passed ~200 m beside the point at 160 km/h and never
    came within the 40 m hover radius.  Now it stops, turns back, centres and lands."""
    logging.disable(logging.WARNING)
    assets, cfg, w, pilot = _setup(hills=0.0)
    m = assets.meters_per_pixel
    assert pilot.go()
    goal = GOAL
    while w.time < 400:
        _run(cfg, w, pilot, lambda: False, 1)
        if math.hypot(w.body.x - goal[0], w.body.y - goal[1]) * m < 700:
            break
    # move the landing point 250 m to the side of the track: a guaranteed lateral miss
    from wardog_ad.geometry import heading_vector
    hx, hy = heading_vector(w.body.heading)
    goal = (goal[0] - hy * 250 / m, goal[1] + hx * 250 / m)
    pilot.set_destination(goal)
    statuses = []
    assert _run(cfg, w, pilot, lambda: pilot.arrived and not pilot.engaged, 400, statuses)
    logging.disable(logging.NOTSET)
    assert w.crashes == 0 and w.hard_landings == 0 and w.body.alt < 0.5
    assert math.hypot(w.body.x - goal[0], w.body.y - goal[1]) * m < 60


def _return_mid_cruise(jhook, align=math.inf, hills=0.0, vector=True):
    """Cruise away from home, press return: time until the helicopter really heads home."""
    from wardog_ad.geometry import bearing, heading_vector
    logging.disable(logging.WARNING)
    assets, cfg, w, pilot = _setup(hills=hills)
    cfg.aircraft.jhook = jhook
    cfg.aircraft.vector_guidance = vector
    w.heli_align, pilot.airnav.align = align, align
    assert pilot.go()
    _run(cfg, w, pilot, lambda: False, 45)
    assert w.body.speed * 3.6 > 150
    pressed, statuses, reversed_at = w.time, [], []
    assert pilot.return_home()

    def heading_home():
        v = heading_vector(w.body.heading) * w.body.speed + heading_vector(w.body.heading + 90) * w.body.side
        toward = float(v @ heading_vector(bearing((w.body.x, w.body.y), START)))
        if not reversed_at and toward * 3.6 > 150:
            reversed_at.append(w.time - pressed)
        return pilot.arrived and not pilot.engaged
    arrived = _run(cfg, w, pilot, heading_home, 400, statuses)
    w.pilot = pilot
    logging.disable(logging.NOTSET)
    return arrived, reversed_at[0] if reversed_at else math.inf, statuses, w


def test_return_mid_cruise_turns_round_and_lands_home():
    """Return pressed at ~290 km/h flying away: with the measured slide the vector guidance
    brakes nose-up, turns round slow and heads home (no separate turn-back mode needed)."""
    arrived, t_back, statuses, w = _return_mid_cruise(jhook=True, align=0.02)
    assert arrived and w.crashes == 0 and w.hard_landings == 0
    assert t_back < 32                              # sim: ~30 s to 150 km/h homewards
    assert _landed_at_home(w.pilot, w, w.cfg)


@pytest.mark.parametrize('align', [0.3, math.inf])
def test_return_mid_cruise_also_works_if_the_game_were_arcade(align):
    arrived, t_back, statuses, w = _return_mid_cruise(jhook=True, align=align, hills=60.0)
    assert arrived and w.crashes == 0 and w.hard_landings == 0 and t_back < 60
    assert _landed_at_home(w.pilot, w, w.cfg)


def test_navigator_follows_a_slide_without_fixes():
    """Dead reckoning with the slip model (HUD speed, compass, pitch) through a J-hook."""
    from wardog_ad.airnav import AirNavigator
    logging.disable(logging.WARNING)
    assets, cfg, w, pilot = _setup(hills=0.0)
    w.heli_align = pilot.airnav.align = 0.02            # the measured slide
    cfg.aircraft.vector_guidance = False                # nose-pointing: a big sideslip to follow
    assert pilot.go()
    _run(cfg, w, pilot, lambda: False, 45)
    m = assets.meters_per_pixel
    navs = {a: AirNavigator(m, align=a) for a in (0.02, math.inf)}
    for nav in navs.values():
        nav.start((w.body.x, w.body.y), w.time)
    assert pilot.return_home()
    end = w.time + 12
    while w.time < end:
        _run(cfg, w, pilot, lambda: False, .1)
        hud = w.hud_reading()
        for nav in navs.values():
            nav.predict(w.time, hud.speed_kmh, hud.heading, hud.pitch)
    logging.disable(logging.NOTSET)
    err = {a: math.hypot(nav.pos[0] - w.body.x, nav.pos[1] - w.body.y) * m for a, nav in navs.items()}
    assert err[0.02] < 40 and err[0.02] < err[math.inf] / 3


def test_fit_slip_measures_the_slide_from_a_flight_log(tmp_path):
    from wardog_ad.recorder import FlightRecorder
    from wardog_ad.slipfit import fit_slip, load_rows
    logging.disable(logging.WARNING)
    assets, cfg, w, pilot = _setup(hills=0.0)
    w.heli_align = 0.5
    cfg.aircraft.vector_guidance = False      # nose-pointing makes the sharp turns the fit needs
    rec = FlightRecorder(tmp_path)
    nxt = w.time

    def run(seconds):
        nonlocal nxt
        end = w.time + seconds
        while w.time < end:
            if w.time >= nxt:
                rec.write(pilot, pilot.tick(w.time), w.time, 0.0)
                nxt += 1 / cfg.loop_hz
            pilot.fast_control(w.time)
            pilot.actuate(w.time)
            w.step(1 / 30)
    assert pilot.go()
    run(45)
    for i in range(3):                                  # J-hooks back and forth
        (pilot.return_home if i % 2 == 0 else pilot.go)()
        run(30)
    rec.close()
    logging.disable(logging.NOTSET)
    fit = fit_slip(load_rows(rec.path))
    assert fit.windows >= 2 and fit.best in (0.3, 0.5, 1.0), fit.message


def _press_land(bug, check=True, press_after=25):
    logging.disable(logging.WARNING)
    assets, cfg, w, pilot = _setup()
    w.hud_drops_hundreds = bug
    if not check:
        pilot.airnav.ground_speed = lambda now, window=1.2: None
    assert pilot.go()
    _run(cfg, w, pilot, lambda: False, press_after)
    pressed = (w.body.x, w.body.y)
    assert pilot.land_here()
    assert _run(cfg, w, pilot, lambda: pilot.arrived and not pilot.engaged, 300)
    logging.disable(logging.NOTSET)
    far = math.hypot(w.body.x - pressed[0], w.body.y - pressed[1]) * assets.meters_per_pixel
    return cfg, w, pilot, far, abs(w.touchdowns[-1][2]) * 3.6


def test_land_button_survives_the_hud_dropping_the_hundreds():
    """Real flight 10:56: 167 km/h read as 68, 112 as 0 -> braked softly, 'hovered' at
    110 km/h and slid 1.4 km.  The minimap ground speed now overrules such digits."""
    cfg, w, pilot, far, touchdown_kmh = _press_land(bug=True, press_after=45)
    assert pilot.speed_mismatches > 0
    assert w.crashes == 0 and w.hard_landings == 0
    assert far <= cfg.aircraft.land_here_radius_m and touchdown_kmh < cfg.aircraft.touchdown_max_kmh


def test_land_button_lands_near_the_press_point_from_cruise():
    cfg, w, pilot, far, touchdown_kmh = _press_land(bug=False, press_after=45)
    assert w.crashes == 0 and w.hard_landings == 0
    assert far <= cfg.aircraft.land_here_radius_m and touchdown_kmh < cfg.aircraft.touchdown_max_kmh


def test_flatter_spot_from_terrain_memory():
    from wardog_ad.terrain import TerrainMemory
    mem = TerrainMemory(None, cell_px=8.0)
    for i in range(-6, 7):
        for j in range(-6, 7):
            ground = 100.0 + (25.0 * i if i < 0 else 0.0)   # steep slope west of x = 0, flat east
            mem.record(((i + .5) * 8, (j + .5) * 8), ground)
    assert mem.roughness(-12, 4) > 20 and mem.roughness(30, 4) == 0
    spot = mem.flattest_near((-12, 4), 40, limit_m=6)
    assert spot is not None and spot[0] > 0              # moved onto the flat side, nearby
    assert math.hypot(spot[0] + 12, spot[1] - 4) <= 40


def _time_from(distance_m, hills=0.0, align=math.inf, yaw=1.0, vector=True):
    logging.disable(logging.WARNING)
    assets, cfg, w, pilot = _setup(hills=hills)
    cfg.aircraft.vector_guidance = vector       # the option under test (off by default since 17:00)
    w.heli_align = pilot.airnav.align = align
    w.heli_yaw_scale = yaw
    m = assets.meters_per_pixel
    assert pilot.go()
    started, statuses = [], []

    def until():
        if not started and math.hypot(w.body.x - GOAL[0], w.body.y - GOAL[1]) * m < distance_m:
            started.append(w.time)
        return pilot.arrived and not pilot.engaged
    assert _run(cfg, w, pilot, until, 500, statuses)
    logging.disable(logging.NOTSET)
    spot = pilot.controller.waypoints[-1]
    w.pilot_spot = spot
    miss = math.hypot(w.body.x - spot[0], w.body.y - spot[1]) * m
    return w.time - started[0], miss, statuses, w


def test_default_guidance_lands_as_before():
    """Default (nose-pointing) guidance, arcade sim: as before the vector-guidance trial."""
    seconds, miss, statuses, w = _time_from(1500, 0.0, math.inf, 1.0, vector=False)
    assert 'j-hook' in {s for _, s, _ in statuses}
    assert seconds < 56 and miss < 40 and w.crashes == 0 and w.hard_landings == 0


@pytest.mark.parametrize('hills, align, yaw', [(0.0, .02, 1.0), (60.0, .02, .3), (60.0, .02, 1.0)])
def test_j_hook_landing_is_fast_and_soft(hills, align, yaw):
    """The WARDOGS J-hook: brake and sink together, nose on the spot, arrive slow and low.
    (Before: brake, hover 30 m up, turn round, then a slow descent - 56-73 s from 1.5 km.)"""
    seconds, miss, statuses, w = _time_from(1500, hills, align, yaw)
    assert 'j-hook' in {s for _, s, _ in statuses}
    assert seconds < 56 and miss < 20                 # sim with the measured slide: 44-53 s, 9-13 m
    assert w.crashes == 0 and w.hard_landings == 0
    assert all(vz > -2.0 and abs(v) < 2.0 for _, vz, v in w.touchdowns)


def _to_100(pilot, cfg, w):
    start, reached = w.time, []

    def until():
        if not reached and math.hypot(w.body.speed, w.body.side) * 3.6 > 100:
            reached.append(w.time - start)
        return bool(reached) or w.time - start > 40
    _run(cfg, w, pilot, until, 41)
    return reached[0] if reached else math.inf


@pytest.mark.parametrize('hills, align', [(0.0, math.inf), (60.0, .3), (60.0, .1)])
def test_landing_also_works_if_the_game_were_less_slidey(hills, align):
    seconds, miss, statuses, w = _time_from(1500, hills, align, 1.0)
    assert seconds < 150 and miss < 40 and w.crashes == 0 and w.hard_landings == 0


def test_takeoff_gets_going_before_cruise_height():
    """Turn from 3 m, fly forward from 10 m.  Over unknown ground it still climbs before it is
    fast (a mountainside may be ahead); where the terrain memory knows the way is flat - e.g.
    the way home - it climbs out fast.  (Before: 100 km/h after ~24 s either way.)"""
    logging.disable(logging.WARNING)
    assets, cfg, w, pilot = _setup(hills=0.0)
    assert pilot.go()
    first = _to_100(pilot, cfg, w)
    assert _run(cfg, w, pilot, lambda: pilot.arrived and not pilot.engaged, 400)
    assert pilot.return_home()
    known = _to_100(pilot, cfg, w)
    logging.disable(logging.NOTSET)
    assert first < 19 and known < 15 and w.crashes == 0


def test_a_garbled_speed_reading_does_not_start_the_j_hook():
    """12:54: one HUD frame read 534 km/h 3.3 km out at 50 km/h; the J-hook started, pitched
    the nose down to 'brake' towards a spot 3 km away and flew into the ground."""
    logging.disable(logging.WARNING)
    assets, cfg, w, pilot = _setup()
    w.hud_speed_glitch = 0.03
    pilot.airnav.ground_speed = lambda now, window=1.2: None   # as live: too few fixes to cross-check
    m = assets.meters_per_pixel
    assert pilot.go()
    hooked_at = []

    def until():
        if pilot.telemetry.status == 'j-hook' and not hooked_at:
            hooked_at.append(math.hypot(w.body.x - GOAL[0], w.body.y - GOAL[1]) * m)
        return pilot.arrived and not pilot.engaged
    assert _run(cfg, w, pilot, until, 500)
    logging.disable(logging.NOTSET)
    assert w.crashes == 0 and w.hard_landings == 0
    assert hooked_at and hooked_at[0] < 1500            # only within real braking distance


def test_engaged_without_a_position_holds_height_and_hovers():
    """12:54: the wrong map was selected, the minimap never matched, and the autopilot showed
    'engaged' for 90 s without pressing a key while the helicopter flew into the ground."""
    from wardog_ad.localization import Fix
    logging.disable(logging.WARNING)
    assets, cfg, w, pilot = _setup(hills=0.0)

    class WrongMap:
        def locate(self, image, anchor, north_up):
            return Fix(valid=False, reason='too few matches (3)', timestamp=w.time)
    pilot.locator = WrongMap()
    w.body.alt, w.body.speed, w.body.pitch = 120.0, 60.0, 25.0     # handed over fast, nose down
    assert pilot.go()
    _run(cfg, w, pilot, lambda: False, 60)
    logging.disable(logging.NOTSET)
    assert w.crashes == 0 and w.body.alt > 40 and abs(w.body.speed) < 4
    assert 'no position' in pilot.telemetry.message


def test_open_ground_is_found_on_the_base_map():
    """BAKURANI: the saved destination sits in forest; the meadow just north is chosen."""
    from wardog_ad.assets import load_map
    from wardog_ad.openground import OpenGround
    a = load_map('bakurani')
    og = OpenGround(a.load_image(), a.meters_per_pixel)
    forest = (957, 997)
    spot, cost = og.best_near(forest, 150)
    assert og.cost(*forest) > .8 and cost < .4
    assert 50 < math.hypot(spot[0] - forest[0], spot[1] - forest[1]) * a.meters_per_pixel <= 150


def test_destination_in_the_trees_lands_in_the_nearest_open_ground():
    from wardog_ad.openground import OpenGround
    logging.disable(logging.WARNING)
    assets, cfg, w, pilot = _setup(hills=0.0)
    goal = (1000, 600)                                     # OZETI: trees; a clearing ~70 m east
    pilot.destination = goal
    assert pilot.go()
    statuses = []
    assert _run(cfg, w, pilot, lambda: pilot.arrived and not pilot.engaged, 500, statuses)
    logging.disable(logging.NOTSET)
    og = OpenGround(assets.load_image(), assets.meters_per_pixel)
    m = assets.meters_per_pixel
    spot = pilot.controller.waypoints[-1]
    assert og.cost(*spot) < og.cost(*goal) - cfg.aircraft.open_gain
    assert math.hypot(spot[0] - goal[0], spot[1] - goal[1]) * m <= cfg.aircraft.open_radius_m + cfg.aircraft.spot_shift_m
    assert math.hypot(w.body.x - spot[0], w.body.y - spot[1]) * m < 40
    assert w.crashes == 0 and w.hard_landings == 0
