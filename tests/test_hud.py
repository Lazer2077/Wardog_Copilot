"""HUD reading (OCR bootstrap + learned digit templates) and helicopter holds."""
import cv2
import numpy as np
import pytest

from wardog_ad.config import Config
from wardog_ad.hud import HudReader, HudReading, glyphs, parse_number


def cluster_image(spd: str, agl: str, asl: str, bg: int = 70) -> np.ndarray:
    """Synthetic 720x270 instrument cluster laid out like the WARDOGS HUD."""
    img = np.full((270, 720, 3), bg, np.uint8)
    font, color = cv2.FONT_HERSHEY_SIMPLEX, (225, 225, 225)
    (lw, _), _ = cv2.getTextSize('KM/H', font, 1.0, 2)
    cv2.putText(img, 'KM/H', (104 - lw, 100), font, 1.0, color, 2, cv2.LINE_AA)  # ends at x~106 like the game
    (w, _), _ = cv2.getTextSize(spd, font, 1.0, 2)
    cv2.putText(img, spd, (162 - w, 100), font, 1.0, color, 2, cv2.LINE_AA)   # right-aligned like the game
    cv2.putText(img, 'AGL', (555, 100), font, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(img, 'ASL', (555, 145), font, 1.0, color, 2, cv2.LINE_AA)
    for text, y in ((agl, 100), (asl, 145)):   # right-aligned like the game
        w = cv2.getTextSize(text, font, 1.0, 2)[0][0]
        cv2.putText(img, text, (700 - w, y), font, 1.0, color, 2, cv2.LINE_AA)
    return img


class ScriptedOcr:
    """Stands in for Windows OCR: returns what the label+value line says, 0 written as O."""

    def __init__(self, truth):
        self.truth, self.calls = truth, 0

    def __call__(self, img):
        self.calls += 1
        return self.truth.pop(0)


def test_parse_number_fixes_ocr_confusions():
    assert parse_number('KM/H OO', 'KM/H') == '00'
    assert parse_number('KWH 1O5', 'KM/H') == '105'
    assert parse_number('ASL 84', 'ASL') == '84'
    assert parse_number('AGL', 'AGL') is None
    assert parse_number('ASL 8x4', 'ASL') is None


def test_glyph_segmentation_counts_digits():
    img = cluster_image('120', '7', '1350')
    from wardog_ad.hud import LAYOUT
    assert [len(glyphs(LAYOUT[k][1].crop(img))) for k in ('spd', 'agl', 'asl')] == [3, 1, 4]


def test_reader_learns_templates_then_reads_without_ocr(tmp_path):
    ocr = ScriptedOcr(['KM/H 12O', 'AGL 34', 'ASL 567', 'KM/H 298'])
    reader = HudReader(tmp_path / 'g.npz', ocr=ocr)
    first = reader.read(cluster_image('120', '34', '567'), None, 0.0)
    assert (first.speed_kmh, first.agl_m, first.asl_m) == (120, 34, 567)
    second = reader.read(cluster_image('298', '10', '234'), None, 5.0)
    assert (second.speed_kmh, second.agl_m, second.asl_m) == (298, 10, 234)
    # "10" and "234" were already readable from the templates learned in frame 1
    assert ocr.calls == 4
    # every digit has been seen once: from now on no OCR at all, even for a lone digit
    reader2 = HudReader(tmp_path / 'g.npz', ocr=ScriptedOcr([]))
    third = reader2.read(cluster_image('75', '6', '908'), None, 10.0)
    assert (third.speed_kmh, third.agl_m, third.asl_m) == (75, 6, 908)


def test_reader_rejects_impossible_jumps(tmp_path):
    reader = HudReader(tmp_path / 'g.npz', ocr=ScriptedOcr(['KM/H 5O', 'AGL 2O', 'ASL 3O', 'AGL 9OO']))
    reader.read(cluster_image('50', '20', '30'), None, 0.0)
    jump = reader.read(cluster_image('50', '900', '30'), None, 0.2)
    assert jump.agl_m is None and jump.speed_kmh == 50


def _heli_world(start=(300, 1200), heading=200.0):
    from wardog_ad.input_win import FakeKeyboard
    from wardog_ad.sim import Body, SimWorld
    cfg = Config(profile='aircraft')
    return cfg, SimWorld(cfg, np.zeros((10, 10, 3), np.uint8), 5.0, Body(*start, heading), FakeKeyboard())


@pytest.mark.parametrize('auto_level,pitch_rate,loop_hz', [(True, 37.0, 8.0), (False, 50.0, 3.5), (True, 25.0, 3.5)])
def test_helicopter_sim_flight(auto_level, pitch_rate, loop_hz):
    """Plant fitted to the real flight log (pitch-rate cyclic, auto-hover collective, ground
    spool-up), including the 3.5 Hz slow loop the real run had."""
    from wardog_ad.assets import DEFAULT_ASSETS
    if not (DEFAULT_ASSETS / 'maps.json').exists():
        pytest.skip('map assets not cloned')
    from wardog_ad.assets import load_map
    from wardog_ad.sim import run_sim
    cfg = Config(profile='aircraft')
    cfg.loop_hz = loop_hz
    res = run_sim(cfg, load_map('ozeti'), (300, 1200), (1300, 300), heading=200, waypoints=[(800, 1100)],
                  max_time=500, world_options=dict(heli_auto_level=auto_level, heli_pitch_rate=pitch_rate))
    assert res.arrived and res.crashes == 0 and res.distance_to_goal_m < cfg.aircraft.hover_radius_m
    rows = [r for r in res.trace if r[6]]
    assert next(r[0] for r in rows if r[6]['agl'] >= 3) < 5          # lifted off quickly (log: 16 s)
    cruise = [r[6] for r in rows if r[5] == 'cruise']
    agl = np.array([o['agl'] for o in cruise][50:])
    assert abs(np.median(agl) - cfg.aircraft.cruise_agl_m) < 3 and agl.min() > 40
    assert np.median([o['speed_kmh'] for o in cruise][200:]) > 90
    pitch = np.array([o['pitch'] for o in (r[6] for r in rows) if o.get('pitch') is not None])
    assert np.abs(pitch).max() < cfg.aircraft.pitch_fwd_max + .4    # log: swung to +-4 ticks and crashed
    low = [r[6] for r in rows if r[6]['agl'] < 2 and r[5] in ('ground', 'takeoff', 'climb')]
    assert low and all(o['yaw'] == 0 for o in low)                   # no pedal on / at the ground


def test_pitch_inner_loop_settles_without_oscillation():
    """Step response of the fast loop on the measured plant (37 deg/s per held key, 30 Hz)."""
    from wardog_ad.actuator import Axis, PWMActuator
    from wardog_ad.controllers import HeliPilot
    from wardog_ad.input_win import FakeKeyboard
    t = Config().aircraft
    pilot = HeliPilot([(0, 0), (1, 0)], t, 5.0)
    kb = FakeKeyboard()
    act = PWMActuator(kb, {'cyclic': Axis('S', 'W', t.cyclic_pwm_s)})
    act.enabled = True
    pitch, history = 0.0, []
    pilot.pitch_target = 0.8
    for i in range(int(6 / 0.005)):
        now = i * 0.005
        if i % 7 == 0:                                   # ~30 Hz fast loop
            act.command(cyclic=pilot.inner(pitch, now))
        act.update(now)
        pitch += 3.7 * (('W' in kb.down) - ('S' in kb.down)) * 0.005
        history.append(pitch)
    h = np.array(history)
    assert h.max() < 0.8 + 0.35                          # small overshoot
    assert abs(h[-200:].mean() - 0.8) < 0.15 and np.ptp(h[-200:]) < 0.3   # settled, no limit cycle
    assert pilot.inner(None, 7.0) == 0.0                 # unreadable pitch: hands off


def test_ground_liftoff_holds_shift_continuously():
    from wardog_ad.controllers import HeliPilot
    from wardog_ad.estimator import VehicleState
    pilot = HeliPilot([(0, 0), (500, 0)], Config().aircraft, 5.0)
    st = VehicleState(0, 0, 0.0, 0, 0, None, 0, 'arrow')
    for i, lever in enumerate((-0.99, 0.4, 1.0, -0.99)):   # the flickering arc from the log
        out = pilot.step(st, i * .3, hud=HudReading(0, 0, 84, 0, i * .3, pitch=0.0, collective=lever))
        assert out.axes['collective'] == 1.0 and out.mode == 'ground'


def test_compass_rejects_single_misread():
    reader = HudReader(None, ocr=ScriptedOcr([]))
    assert reader._plausible_heading(195.0, 0.0) == 195.0
    assert reader._plausible_heading(95.0, 0.1) is None          # '195' read as '95' (seen in the log)
    assert reader._plausible_heading(196.0, 0.2) == 196.0
    assert reader._plausible_heading(300.0, 0.3) is None
    assert reader._plausible_heading(301.0, 0.4) == 301.0       # repeated twice: a real change


def test_minus_sign_never_learned(tmp_path):
    reader = HudReader(tmp_path / 'g.npz', ocr=ScriptedOcr(['KM/H', 'AGL -6', 'ASL']))
    assert parse_number('AGL -6', 'AGL') is None
    reader.read(cluster_image('', '-6', ''), None, 0.0)
    assert '-' not in reader.library.templates


def test_helicopter_holds_neutral_without_hud():
    from wardog_ad.controllers import HeliPilot
    from wardog_ad.estimator import VehicleState
    pilot = HeliPilot([(0, 0), (500, 0)], Config().aircraft, 5.0)
    st = VehicleState(0, 0, 0.0, 0, 0, None, 0, 'arrow')     # target is 90 deg to the right
    out = pilot.step(st, 0.0, hud=None)
    assert out.axes == {'collective': 0.0, 'cyclic': 0.0, 'yaw': 0.0}
    assert pilot.step(st, 2.0, hud=HudReading()).mode == 'hud lost'
    lift = pilot.step(st, 2.1, hud=HudReading(0, 0, 84, 0, 2.1, pitch=0.0, collective=-1.0))
    assert lift.mode == 'ground' and lift.axes['collective'] > .5 and lift.axes['cyclic'] == 0
    assert lift.axes['yaw'] == 0                                   # no turning on the ground


def test_helicopter_climbs_before_turning_and_levels_pitch():
    from wardog_ad.controllers import HeliPilot
    from wardog_ad.estimator import VehicleState
    t = Config().aircraft
    pilot = HeliPilot([(0, 0), (500, 0)], t, 5.0)
    st = VehicleState(0, 0, 0.0, 0, 0, None, 0, 'arrow')
    # on/near the ground: no pedal (from 3 m it may turn in the hover, see take-off)
    low = pilot.step(st, 0.0, hud=HudReading(0, 2, 100, 0, 0.0, pitch=0.0, collective=0.1))
    assert low.axes['yaw'] == 0
    high = pilot.step(st, 0.2, hud=HudReading(0, t.turn_agl_m + 5, 110, 0, 0.2, pitch=0.0, collective=0.1))
    assert high.axes['yaw'] > .5
    # nose far down (the crash seen in the game): cyclic must pull back hard
    tipped = pilot.step(st, 0.4, hud=HudReading(20, 40, 120, 0, 0.4, pitch=3.4, collective=1.0))
    assert tipped.axes['cyclic'] <= -1.0 + 1e-9


def test_glyph_library_drops_templates_learned_from_misreads(tmp_path):
    import numpy as np
    from wardog_ad.hud import GlyphLibrary
    one = np.zeros((24, 16), np.float32); one[:, 7:9] = 1
    four = np.zeros((24, 16), np.float32); four[:14, 3:5] = 1; four[12:14, 3:13] = 1; four[:, 10:12] = 1
    np.savez_compressed(tmp_path / 'g.npz', **{'1': np.stack([one]), '4': np.stack([four, one])})  # a '1' filed as '4'
    lib = GlyphLibrary(tmp_path / 'g.npz')
    assert len(lib.templates['4']) == 1 and lib.read([one]) != '4'   # ambiguous copies dropped
    lib.learn('1', [one])                                            # relearned from a good OCR read
    lib.learn('4', [one])                                            # a later misread is refused
    assert lib.read([one]) == '1' and len(lib.templates['4']) == 1
