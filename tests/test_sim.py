"""Closed-loop tests on the real OZETI map (skipped when the map assets are absent)."""
import numpy as np
import pytest

from wardog_ad.assets import DEFAULT_ASSETS
from wardog_ad.config import Config
from wardog_ad.geometry import angle_delta, bearing

pytestmark = pytest.mark.skipif(not (DEFAULT_ASSETS / 'maps.json').exists(), reason='map assets not cloned')


@pytest.fixture(scope='module')
def world():
    from wardog_ad.assets import load_map
    from wardog_ad.planner import RoadGraph
    assets = load_map('ozeti')
    graph = RoadGraph(assets.roads)
    majors = [r for r in assets.roads if r['kind'] == 'major']
    start, goal = tuple(majors[0]['points'][0]), tuple(majors[3]['points'][-1])
    route = graph.route([start, goal])
    return assets, graph, start, goal, route


def test_localization_on_rendered_minimap(world):
    from wardog_ad.input_win import FakeKeyboard
    from wardog_ad.localization import MinimapLocalizer
    from wardog_ad.sim import Body, SimWorld
    assets = world[0]
    loc = MinimapLocalizer.from_assets(assets)
    sim = SimWorld(Config(), assets.load_image(), assets.meters_per_pixel, Body(505, 1245, 37), FakeKeyboard())
    for heading in (37, 200):
        sim.body.heading = heading
        fix = loc.locate(sim.render_minimap())
        assert fix.valid
        assert np.hypot(fix.x - 505, fix.y - 1245) < 1.0
        assert abs(angle_delta(heading, fix.heading)) < 6


def test_truck_follows_road(world):
    from wardog_ad.planner import Path
    from wardog_ad.sim import run_sim
    assets, graph, start, goal, route = world
    res = run_sim(Config(), assets, start, goal, heading=bearing(route[0], route[5]), graph=graph, max_time=300)
    assert res.arrived and res.distance_to_goal_m < 30
    path = Path(route)
    cross = [path.nearest((x, y))[0] * assets.meters_per_pixel for _, x, y, *_ in res.trace]
    assert np.percentile(cross, 90) < 12 and max(cross) < 20


def test_truck_turns_around_and_hands_over_after_bounded_recovery(world):
    from wardog_ad.planner import Path
    from wardog_ad.sim import run_sim
    assets, graph, start, goal, route = world
    res = run_sim(Config(), assets, start, goal, heading=(bearing(route[0], route[5]) + 180) % 360,
                  graph=graph, max_time=300)
    assert res.arrived and 'kturn' in {row[5] for row in res.trace}
    wx, wy = Path(route).at(300 / assets.meters_per_pixel)
    res = run_sim(Config(), assets, start, goal, heading=bearing(route[0], route[5]), graph=graph,
                  max_time=400, walls=[(wx - 1.5, wy - 1.5, wx + 1.5, wy + 1.5)])
    # The default policy now hands over after two attempts instead of continuing
    # to reverse until a third failure permits a speculative detour.
    starts = [row for i,row in enumerate(res.trace) if row[5]=='reverse'
              and (i==0 or res.trace[i-1][5]!='reverse')]
    assert not res.arrived and len(starts)==2
    assert res.trace[-1][5]=='blocked' and res.trace[-1][6]['throttle']==0
    assert res.trace[-1][6]['recovery_reason']=='reverse_limit'
    assert res.time < 400


def test_aircraft_waypoints_and_flares(world):
    from wardog_ad.sim import run_sim
    assets = world[0]
    res = run_sim(Config(profile='aircraft'), assets, (300, 1200), (1300, 300), heading=90,
                  waypoints=[(800, 1100)], max_time=400, locks=[20, 60])
    assert res.arrived and res.distance_to_goal_m < 200
    assert res.missiles_defeated == 2 and res.missiles_hit == 0


def _mission(cfg, assets, start, heading, graph=None, max_time=400):
    """Press go, wait for arrival, press return, wait for arrival (one-button flow)."""
    from wardog_ad.autopilot import Autopilot
    from wardog_ad.input_win import FakeKeyboard
    from wardog_ad.sim import Body, SimSensors, SimWorld, TruthLocator, sim_lock_profile
    kb = FakeKeyboard()
    world = SimWorld(cfg, assets.load_image(), assets.meters_per_pixel, Body(*start, heading), kb)
    if cfg.profile == 'aircraft' and cfg.aircraft.flight_model != 'helicopter':
        world.body.speed = 80.0
    pilot = Autopilot(cfg, assets, kb, SimSensors(world), locator=TruthLocator(world), graph=graph,
                      lock_profile=sim_lock_profile() if cfg.profile == 'aircraft' else None)
    def run(until):
        next_control = world.time
        while world.time < until:
            if world.time >= next_control:
                pilot.actuate(world.time)
                pilot.tick(world.time)
                next_control += 1 / cfg.loop_hz
                if pilot.arrived and not pilot.engaged:
                    return True
            pilot.actuate(world.time)
            world.step(0.02)
        return False

    assert pilot.go() is True
    outbound = run(world.time + max_time)
    assert not pilot.engaged            # control is handed back on arrival
    assert pilot.return_home() is True
    back = run(world.time + max_time)
    return outbound, back, pilot, world


def test_one_button_go_and_return_truck(world):
    assets, graph, start, goal, route = world
    cfg = Config()
    cfg.destinations = {'ozeti': list(goal)}
    outbound, back, pilot, w = _mission(cfg, assets, start, bearing(route[0], route[5]), graph)
    assert outbound and back and pilot.mission == 'return'
    assert np.hypot(pilot.home[0] - start[0], pilot.home[1] - start[1]) < 1.5      # home = where go was pressed
    assert np.hypot(w.body.x - start[0], w.body.y - start[1]) * assets.meters_per_pixel < 30


def test_one_button_go_and_return_helicopter(world):
    assets = world[0]
    cfg = Config(profile='aircraft')
    cfg.destinations = {'ozeti': [1000, 600]}
    outbound, back, pilot, w = _mission(cfg, assets, (400, 1100), 0.0)
    assert outbound and back
    assert np.hypot(w.body.x - 400, w.body.y - 1100) * assets.meters_per_pixel < 200


def test_go_without_destination_is_refused(world):
    from wardog_ad.autopilot import Autopilot
    from wardog_ad.input_win import FakeKeyboard
    from wardog_ad.sim import Body, SimSensors, SimWorld, TruthLocator
    assets, graph = world[0], world[1]
    kb = FakeKeyboard()
    w = SimWorld(Config(), assets.load_image(), assets.meters_per_pixel, Body(500, 1200, 0), kb)
    pilot = Autopilot(Config(), assets, kb, SimSensors(w), locator=TruthLocator(w), graph=graph)
    assert pilot.go() is False and not pilot.engaged
    assert pilot.return_home() is False and not pilot.engaged


def _live_like(cfg, assets, graph=None):
    from wardog_ad.autopilot import Autopilot
    from wardog_ad.input_win import FakeKeyboard
    from wardog_ad.sim import Body, SimSensors, SimWorld, TruthLocator
    kb = FakeKeyboard()
    w = SimWorld(cfg, assets.load_image(), assets.meters_per_pixel, Body(400, 1100, 0.0), kb)
    pilot = Autopilot(cfg, assets, kb, SimSensors(w), locator=TruthLocator(w), graph=graph)
    return w, pilot


def test_pauses_when_helicopter_hud_disappears(world):
    assets = world[0]
    cfg = Config(profile='aircraft')
    cfg.aircraft.lock_source = 'none'
    cfg.destinations = {'ozeti': [1000, 600]}
    w, pilot = _live_like(cfg, assets)
    assert pilot.go()
    for _ in range(100):                                     # 10 s of normal flight
        pilot.tick(w.time)
        for _ in range(5):
            pilot.actuate(w.time)
            w.step(0.02)
    assert pilot.engaged and w.body.alt > 5
    w.hud_reading = lambda: None                             # got out / map opened
    for _ in range(25):
        pilot.tick(w.time)
        pilot.actuate(w.time)
        w.step(0.1)
    assert not pilot.engaged and 'HUD' in pilot.telemetry.message
    assert not pilot.keyboard.down                           # all keys released
    assert pilot.mission == 'outbound' and not pilot.arrived  # F8 resumes the same trip


def test_switch_vehicle_at_run_time(world, tmp_path):
    from wardog_ad.recorder import FlightRecorder
    assets, graph = world[0], world[1]
    cfg = Config()
    cfg.aircraft.lock_source = 'none'
    w, pilot = _live_like(cfg, assets, graph)
    assert set(pilot.actuator.axes) == {'steer', 'throttle'}
    pilot.toggle_profile()
    assert cfg.profile == 'aircraft' and set(pilot.actuator.axes) == {'collective', 'cyclic', 'yaw'}
    assert pilot.flares is not None
    pilot.toggle_profile()
    assert cfg.profile == 'truck' and pilot.flares is None and pilot.graph is graph
    rec = FlightRecorder(tmp_path)
    rec.write(pilot, pilot.tick(0.0), 0.0, 0.01)
    rec.close()
    import json
    line = json.loads(rec.path.read_text(encoding='utf-8').splitlines()[0])
    assert line['profile'] == 'truck' and 'state' in line and 'hud' in line
