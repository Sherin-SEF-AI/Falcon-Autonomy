"""Why the cheap and the CARLA criticality agree so little (core.fastsim.diagnose, 4.2): planted pairs with one cause
each — too few pairs, a saturated cheap value, disagreement that grows with one parameter, a tiny camera, an inverted
order — and the pre-4.2 prescreen.json and results rows still read. Pure functions: no CARLA, no Qt."""
import json
import os

import numpy as np
import pytest

from falcon_autonomy.core import fastsim as F


def _pairs(low, high, values=None):
    values = values or [{} for _ in low]
    return [{'low': float(a), 'high': float(b), 'values': v, 'run': f'/r/{i:02d}', 'role': 'top', 'fidelity': 'fake'}
            for i, (a, b, v) in enumerate(zip(low, high, values))]


def test_agreeing_pairs_are_not_low_and_list_no_cause():
    x = np.linspace(0.0, 1.0, 16)
    d = F.diagnose(_pairs(x, x ** 2))
    assert d['spearman'] == pytest.approx(1.0) and not d['low'] and d['pairs'] == 16
    assert [f['key'] for f in d['findings']] == [] and d['disagreements'] == [] and d['parameter'] is None


def test_too_few_pairs_is_said_with_the_trust_rule():
    d = F.diagnose(_pairs([0.1, 0.5, 0.4, 0.9], [0.2, 0.3, 0.6, 0.8]), min_pairs=12)
    keys = [f['key'] for f in d['findings']]
    assert 'few_pairs' in keys and '4 of the 12 pairs' in d['findings'][keys.index('few_pairs')]['text']
    assert F.diagnose(_pairs([0.1, 0.2], [0.3, 0.4]))['spearman'] is None and F.diagnose([])['low']


def test_a_saturated_cheap_value_is_found():
    rng = np.random.default_rng(1)
    low = [0.0] * 9 + list(np.linspace(0.3, 0.9, 5))
    high = list(rng.random(14))
    d = F.diagnose(_pairs(low, high))
    t = d['ties']
    assert t['distinct'] == 6 and t['at'] == 0.0 and t['at_zero'] == 9 and t['largest_share'] == pytest.approx(9 / 14, abs=1e-3) and t['count'] == 9
    f = next(f for f in d['findings'] if f['key'] == 'saturation')
    assert '9 of 14 pairs share one cheap value at 0' in f['text'] and f['tone'] == 'warning'


def test_the_largest_disagreements_and_the_parameter_they_grow_with():
    # the screen orders every small-gap pair right; it swaps the four pairs with the largest gaps (most at 40 m)
    high = np.arange(14) / 13.0
    low = high.copy()
    for a, b in ((3, 10), (5, 8)):
        low[a], low[b] = high[b], high[a]
    gaps = [5, 6, 7, 40, 8, 30, 9, 10, 28, 11, 38, 12, 13, 14]
    speed = [20, 12, 25, 18, 22, 15, 11, 24, 19, 13, 21, 16, 23, 17]     # unrelated to the swaps
    vals = [{'gap': float(g), 'speed': float(s), 'side': 'left'} for g, s in zip(gaps, speed)]
    d = F.diagnose(_pairs(low, high, vals), min_spearman=0.95, top=3)
    assert d['low'] and d['spearman'] == pytest.approx(0.745, abs=1e-3) and len(d['disagreements']) == 3
    gaps_ = [x['rank_diff'] for x in d['disagreements']]
    assert gaps_ == [7.0, 7.0, 3.0]
    assert {x['run'] for x in d['disagreements'][:2]} == {'/r/03', '/r/10'}
    worst = d['disagreements'][0]
    assert worst['run'].startswith('/r/') and worst['fidelity'] == 'fake' and 'gap' in worst['values']
    assert d['parameter']['name'] == 'gap' and d['parameter']['rho'] >= F.CONCENTRATED_RHO
    assert any(f['key'] == 'parameter' and 'grows with gap' in f['text'] for f in d['findings'])
    assert any(f['key'] == 'disagreement' and 'places apart' in f['text'] for f in d['findings'])


def test_an_inverted_order_and_a_tiny_camera():
    x = np.linspace(0.0, 1.0, 14)
    rig = {'sensors': [{'type': 'camera.pinhole', 'name': 'front', 'width': 64, 'height': 36},
                       {'type': 'lidar.ray_cast', 'name': 'top'}]}
    d = F.diagnose(_pairs(x, 1.0 - x), rig=rig)
    assert d['spearman'] == pytest.approx(-1.0) and d['findings'][0]['key'] == 'inverted'
    assert d['findings'][0]['tone'] == 'critical' and d['cameras'] == [('front', 64, 36)]
    assert any(f['key'] == 'camera' and '64 × 36' in f['text'] for f in d['findings'])
    big = {'sensors': [{'type': 'camera.pinhole', 'name': 'front', 'width': 1280, 'height': 720}]}
    assert F.diagnose(_pairs(x, 1.0 - x), rig=big)['cameras'] == []


def test_a_prescreen_file_from_before_4_2_is_read(tmp_path):
    """Before 4.2 the fidelity gap was the "fidelity" block and each row's record was "fidelity": both still read."""
    gap = {'pairs': 2, 'spearman': None, 'trust': 'not judged yet'}
    old = {'schema': F.PRESCREEN_SCHEMA, 'fidelity': gap, 'pairs': [{'low': 0.1, 'high': 0.2, 'run': ''}]}
    with open(tmp_path / 'prescreen.json', 'w') as fh:
        json.dump(old, fh)
    d = F.read_prescreen(str(tmp_path))
    assert d['gap'] == gap and d['fidelity'] == 'unknown' and d['screen_fidelity'] == 'kinematic'
    assert F.screen_record({'fidelity': {'low': 0.3, 'role': 'top'}}) == {'low': 0.3, 'role': 'top'}
    assert F.screen_record({'screen': {'low': 0.4}, 'fidelity': 'fake'}) == {'low': 0.4}
    assert F.screen_record({'fidelity': 'fake'}) == {}
    assert not os.path.exists(tmp_path / 'search-prescreen.json')
