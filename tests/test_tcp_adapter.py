"""TCP semantic decoding and the model-to-PWM boundary, without game input."""
import math

import numpy as np
import pytest

from wardog_ad.actuator import Axis, PWMActuator
from wardog_ad.input_win import FakeKeyboard
from wardog_ad.tcp_adapter import decode_beta, TruckOutputMapper, target_from_map


@pytest.mark.parametrize('heading,target,expected', [
    (0, (0, -10), (0, -20)), (90, (10, 0), (0, -20)),
    (180, (0, 10), (0, -20)), (270, (-10, 0), (0, -20)),
    (0, (10, 0), (20, 0)), (90, (0, 10), (20, 0)),
])
def test_navigation_transform_uses_negative_forward_and_positive_right(heading, target, expected):
    assert target_from_map((0, 0), heading, target, 2) == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize('a,b,expected', [
    ([3, 2], [2, 3], [1/3, -1/3]),
    ([.5, 2], [2, .5], [-1, 1]),
    ([.5, .5], [.5, .5], [0, 0]),
    ([1, 1], [2, 2], [-1, -1]),
])
def test_upstream_beta_semantics(a, b, expected):
    out = decode_beta(a, b)
    assert [out['acceleration'], out['steer']] == pytest.approx(expected)
    assert out['throttle'] * out['brake'] == 0


def test_boundary_mode_is_not_a_sign_error_and_mean_is_only_diagnostic():
    mode = decode_beta([.9, 2], [1.1, 2])
    mean = decode_beta([.9, 2], [1.1, 2], 'mean')
    assert mode['brake'] == 1 and mode['beta_cases'][0] == 'lower_boundary'
    assert mean['brake'] == pytest.approx(.1)


@pytest.mark.parametrize('a,b', [([0, 1], [1, 1]), ([math.nan, 1], [1, 1]),
                                ([1], [1]), ([1, 2], [-1, 1])])
def test_bad_distribution_rejected(a, b):
    with pytest.raises(ValueError):
        decode_beta(a, b)


def mapped(control, speed=4, **kwargs):
    params = dict(speed_ms=speed, now=1., prediction_at=1., speed_at=1., enabled=True)
    params.update(kwargs)
    return TruckOutputMapper().map(control, **params)


def test_braking_overrides_simultaneous_throttle():
    out = mapped(dict(steer=.8, throttle=.7, brake=.3))
    assert out['mode'] == 'brake' and out['axes'] == dict(steer=0., throttle=-.3)


def test_stop_does_not_translate_model_braking_into_reverse():
    kb = FakeKeyboard()
    pwm = PWMActuator(kb, {'steer': Axis('A', 'D'), 'throttle': Axis('S', 'W')})
    pwm.enabled = True
    for i, speed in enumerate([5, 3, 1.5, .7, .2, 0, -.2]):
        out = mapped(dict(brake=1), speed=speed)
        pwm.command(**out['axes'])
        pwm.update(i * .1)
        if speed > .7:
            assert kb.down == {'S'}
        else:
            assert not kb.down


def test_limited_forward_command_reaches_w_not_s():
    out = mapped(dict(steer=1, throttle=1), speed=0)
    assert out['axes'] == dict(steer=.5, throttle=.35)
    kb = FakeKeyboard()
    pwm = PWMActuator(kb, {'steer': Axis('A', 'D'), 'throttle': Axis('S', 'W')})
    pwm.enabled = True
    pwm.command(**out['axes'])
    active = []
    for now in np.arange(0, 1, .01):
        pwm.update(float(now))
        assert 'S' not in kb.down and 'A' not in kb.down
        active.append('W' in kb.down)
    assert np.mean(active) == pytest.approx(.35, abs=.05)
    pwm.release()
    assert not kb.down


@pytest.mark.parametrize('kwargs,mode', [
    ({'enabled': False}, 'disabled'),
    ({'prediction_at': 0.}, 'stale_prediction'),
    ({'speed_at': .5}, 'stale_speed'),
    ({'speed_ms': None}, 'invalid_state'),
    ({'prediction_at': 2.}, 'stale_prediction'),
    ({'speed_ms': -3}, 'reverse_motion_requires_takeover'),
])
def test_no_acceleration_on_invalid_or_unhandled_state(kwargs, mode):
    out = mapped(dict(throttle=1), **kwargs)
    assert out['mode'] == mode and out['axes'] == dict(steer=0., throttle=0.)


def test_nonfinite_command_never_reaches_pwm():
    assert mapped(dict(throttle=float('nan')))['mode'] == 'invalid_control'
