import numpy as np

from wardog_ad.imitation import eligible, features, make_samples, split_samples, metrics


def row(t, **kw):
    result = dict(t=t, capture_end_t=t+.05, keys_t=t-.01, keys_age_s=.01,
                  camera_view='first_person', observed_keys=['W'],
                  label_validity=dict(camera_confirmed=True, input_fresh=True, navigation_fresh=True))
    result.update(kw)
    return result


def test_missing_navigation_is_masked_and_future_navigation_rejected():
    assert np.array_equal(features(row(1))[:5], np.zeros(5))
    nav = dict(processed_t=1.01, speed=dict(signed_ms=10), local_target_m=[5,-20])
    assert features(row(1, navigation=nav))[1] == 0
    nav['processed_t'] = .9
    f = features(row(1, navigation=nav))
    assert f[1] == f[4] == 1
    assert np.allclose(f[2:4], [.1,.4])


def test_stale_future_keys_and_conflicting_axes_rejected():
    assert eligible(row(1), 'KODIAK')
    assert not eligible(row(1, keys_t=1.02), 'KODIAK')
    assert not eligible(row(1, keys_age_s=-.01), 'KODIAK')
    assert not eligible(row(1, observed_keys=['W','S']), 'KODIAK')
    assert not eligible(row(1, camera_view='unknown'), 'KODIAK')


def test_sequences_never_cross_rejected_frames_gaps_or_view_changes():
    rows = [row(i*.2) for i in range(20)]
    rows[12]['t'] += .7
    for r in rows[13:]:
        r['t'] += .7
        r['camera_view'] = 'third_person'
    good = np.ones(20,bool)
    good[6] = False
    samples = make_samples(rows, good)
    assert all(6 not in s for s in samples)
    assert not any(s[0] < 6 < s[-1] for s in samples)
    assert not any(s[0] < 12 <= s[-1] for s in samples)
    assert not any(s[0] < 13 <= s[-1] for s in samples)


def test_temporal_splits_have_no_history_overlap_and_purged_boundaries():
    rows = [row(i*.2) for i in range(500)]
    samples = make_samples(rows,np.ones(500,bool))
    times = np.array([r['t'] for r in rows])
    splits = split_samples(samples,times)
    for a,b in [('train','val'),('val','test')]:
        left = samples[splits[a]]
        right = samples[splits[b]]
        assert not set(left.flat).intersection(right.flat)
        assert times[right[:,0]].min()-times[left[:,-1]].max() > 4


def test_metrics_expose_majority_failure_on_turns():
    y = np.array([[1,0,0,0,0],[1,1,0,0,0],[1,0,0,1,0]])
    p = np.tile([1,0,0,0,0],(3,1))
    score = metrics(y,p)
    assert score['per_key']['W']['f1'] == 1
    assert score['per_key']['A']['f1'] == 0
    assert score['macro_f1_supported'] < .5


def test_model_has_finite_gradient_and_weights_only_roundtrip(tmp_path):
    import pytest
    torch = pytest.importorskip('torch')
    from wardog_ad.imitation import build_model
    torch.set_num_threads(1)
    model = build_model()
    x, f = torch.rand(2,4,3,90,160), torch.zeros(2,4,7)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x,f), torch.zeros(2,5))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    path = tmp_path/'weights.pt'
    torch.save({'model':model.state_dict()},path)
    other = build_model()
    other.load_state_dict(torch.load(path,weights_only=True)['model'])
    torch.testing.assert_close(model(x,f),other(x,f))
