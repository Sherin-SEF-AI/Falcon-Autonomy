"""The scenario model: the built-in library, seeded sampling, expressions, search spaces, validation."""
import pytest

from falcon_autonomy.core import scenarios as SC


@pytest.mark.parametrize('key', sorted(SC.LIBRARY))
def test_every_builtin_is_valid_and_samples_deterministically(key):
    s = SC.builtin(key)
    assert SC.validate(s) == []
    a, b, c = SC.sample(s, 3), SC.sample(s, 3), SC.sample(s, 4)
    assert a == b and a['values'] != c['values']
    for name, p in s['params'].items():
        v = a['values'][name]
        if p['dist'] == 'uniform':
            assert p['lo'] <= v <= p['hi']
    assert '$' not in repr({k: v for k, v in a.items() if k not in ('params',)})


def test_the_indian_scenarios_are_in_the_library():
    assert set(SC.INDIAN) <= set(SC.LIBRARY) and len(SC.LIBRARY) == 11


def test_expressions():
    v = {'ride': 2.0, 'start_behind': 12.0, 'distance': 30.0}
    assert SC.evaluate('$ride * 3.6', v) == pytest.approx(7.2)
    assert SC.evaluate('-$start_behind', v) == -12.0
    assert SC.evaluate('$distance - 7', v) == 23.0
    assert SC.evaluate('car', v) == 'car' and SC.evaluate(5, v) == 5
    with pytest.raises(ValueError):
        SC.evaluate('__import__("os").system("x") + $ride', v)


def test_overrides_and_adding_a_parameter_keeps_the_others():
    s = SC.builtin('lead_brake')
    a = SC.sample(s, 7)
    s2 = SC.builtin('lead_brake')
    s2['params']['extra'] = SC.U(0, 1)
    b = SC.sample(s2, 7)
    assert all(b['values'][k] == a['values'][k] for k in a['values'])
    c = SC.sample(s, 7, {'gap': 10.0})
    assert c['actors'][0]['place']['ahead_m'] == 10.0


def test_search_space_and_validation():
    sp = SC.search_space(SC.builtin('stalled_vehicle'))
    assert {'name': 'kind', 'choices': ['car', 'truck', 'van']} in sp
    bad = SC.builtin('cut_in')
    bad['timeline'][0]['actions'][0]['actor'] = 'ghost'
    assert any('ghost' in e for e in SC.validate(bad))


@pytest.mark.parametrize('key', sorted(SC.LIBRARY))
def test_openscenario_round_trip(key):
    from falcon_autonomy.core import openscenario as X
    conc = SC.sample(SC.builtin(key), 3)
    xml = X.export_xosc(conc)
    assert xml.startswith('<?xml') and '<OpenSCENARIO>' in xml and 'RelativeLanePosition' in xml
    back = X.import_xosc(xml)
    assert back['duration_s'] == pytest.approx(conc['duration_s'])
    assert back['ego']['speed_kmh'] == pytest.approx(conc['ego']['speed_kmh'])
    for a, b in zip(conc['actors'], back['actors']):
        assert (a['id'], a['kind'], a['heading'], a['start']) == (b['id'], b['kind'], b['heading'], b['start'])
        assert b['place']['lane'] == a['place']['lane']
        assert b['place']['ahead_m'] == pytest.approx(float(a['place']['ahead_m']))
        assert b['speed_kmh'] == pytest.approx(float(a['speed_kmh']) if a['start'] == 'moving' else 0.0)
    assert len(back['timeline']) == len(conc['timeline'])
    for ea, eb in zip(conc['timeline'], back['timeline']):
        assert ea['trigger']['type'] == eb['trigger']['type']
        assert [x['do'] for x in ea['actions']] == [x['do'] for x in eb['actions']]
    assert set(back['values']) == set(conc['values'])


def test_openscenario_refuses_outside_the_subset():
    from falcon_autonomy.core import openscenario as X
    xml = X.export_xosc(SC.sample(SC.builtin('lead_brake'), 1)).replace('RelativeLanePosition', 'WorldPosition')
    with pytest.raises(ValueError, match='subset'):
        X.import_xosc(xml)
