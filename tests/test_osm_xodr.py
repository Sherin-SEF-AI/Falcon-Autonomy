"""OSM input (read/crop/write/PBF/lanes/projection/geocode) and OpenDRIVE (parse/geometry/graph/LHT/lane check).

Fixtures: tests/data/blr_small.osm (a 300 m crop of central Bengaluru) and tests/data/sumo_small.xodr (three
junctions with their roads, cut from carla.Osm2Odr output). Tests on the full-size files run only where they exist.
"""
import json
import math
import os
import re
import struct
import time
import zlib

import numpy as np
import pytest

from falcon_autonomy.core import osm, xodr

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SMALL_OSM = os.path.join(DATA, 'blr_small.osm')
SMALL_XODR = os.path.join(DATA, 'sumo_small.xodr')
BIG_OSM = '/home/jo/Desktop/LabeloxAV/.scratch/maps/bangalore-roads.osm'
BIG_XODR = ('/tmp/claude-1000/-home-jo-Desktop-FalconAutonomy/62da7238-0b01-4122-bd39-3110fdc3b02f/scratchpad/osm/'
            'blr.xodr')                     # carla.Osm2Odr of BIG_OSM (2026-10-29: 7554 roads), centre from its summary
big_osm = pytest.mark.skipif(not os.path.isfile(BIG_OSM), reason='full Bengaluru OSM export not on this machine')
big_xodr = pytest.mark.skipif(not os.path.isfile(BIG_XODR), reason='full Bengaluru OpenDRIVE not on this machine')


def _text(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


# ------------------------------------------------------------------------------------------------ protobuf writer
# An independent, minimal PBF writer (OSM-binary fileformat.proto / osmformat.proto field numbers) for the reader.
def _v(n):
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _zz(n):
    return n << 1 if n >= 0 else ((-n) << 1) - 1


def _vi(f, n):
    return _v(f << 3) + _v(n)


def _ld(f, b):
    return _v(f << 3 | 2) + _v(len(b)) + b


def _packed(f, vals):
    return _ld(f, b''.join(_v(x) for x in vals))


def _deltas(xs):
    out, prev = [], 0
    for x in xs:
        out.append(x - prev)
        prev = x
    return out


def _blob(btype, payload, compress):
    blob = (_vi(2, len(payload)) + _ld(3, zlib.compress(payload))) if compress else _ld(1, payload)
    hdr = _ld(1, btype.encode()) + _vi(3, len(blob))
    return struct.pack('>I', len(hdr)) + hdr + blob


class _Strings:
    def __init__(self):
        self.s = ['']

    def __call__(self, x):
        if x not in self.s:
            self.s.append(x)
        return self.s.index(x)

    def table(self):
        """The StringTable message: repeated bytes s = 1."""
        return b''.join(_ld(1, x.encode()) for x in self.s)


def _nano(deg):
    return round(deg * 1e9)


def _dense_block(nodes, tags, gran=100, lat_off=0, lon_off=0):
    st = _Strings()
    ids = sorted(nodes)
    lat = [(_nano(nodes[i][0]) - lat_off) // gran for i in ids]
    lon = [(_nano(nodes[i][1]) - lon_off) // gran for i in ids]
    kv = []
    for i in ids:
        for k, v in tags.get(i, {}).items():
            kv += [st(k), st(v)]
        kv.append(0)
    dense = (_packed(1, [_zz(d) for d in _deltas(ids)]) + _packed(8, [_zz(d) for d in _deltas(lat)])
             + _packed(9, [_zz(d) for d in _deltas(lon)]) + _packed(10, kv))
    group = _ld(2, dense)
    return _ld(1, st.table()) + _ld(2, group) + _vi(17, gran) + _vi(19, lat_off) + _vi(20, lon_off)


def _plain_block(nodes, tags, gran=100, lat_off=0, lon_off=0):
    st = _Strings()
    msgs = b''
    for i in sorted(nodes):
        keys = [st(k) for k in tags.get(i, {})]
        vals = [st(v) for v in tags.get(i, {}).values()]
        body = _vi(1, _zz(i))
        if keys:
            body += _packed(2, keys) + _packed(3, vals)
        body += _vi(8, _zz((_nano(nodes[i][0]) - lat_off) // gran))
        body += _vi(9, _zz((_nano(nodes[i][1]) - lon_off) // gran))
        msgs += _ld(1, body)
    return _ld(1, st.table()) + _ld(2, msgs) + _vi(17, gran) + _vi(19, lat_off) + _vi(20, lon_off)


def _way_block(ways, relation=False):
    st = _Strings()
    msgs = b''
    for w in ways:
        keys = [st(k) for k in w['tags']]
        vals = [st(v) for v in w['tags'].values()]
        msgs += _ld(3, _vi(1, w['id']) + _packed(2, keys) + _packed(3, vals)
                    + _packed(8, [_zz(d) for d in _deltas(w['nodes'])]))
    if relation:
        msgs += _ld(4, _vi(1, 99) + _packed(2, [st('type')]) + _packed(3, [st('restriction')]))
    return _ld(1, st.table()) + _ld(2, msgs)


def _header(bounds, features=('OsmSchema-V0.6', 'DenseNodes')):
    s, w, n, e = bounds
    bbox = _vi(1, _zz(_nano(w))) + _vi(2, _zz(_nano(e))) + _vi(3, _zz(_nano(n))) + _vi(4, _zz(_nano(s)))
    return _ld(1, bbox) + b''.join(_ld(4, f.encode()) for f in features) + _ld(16, b'falcon-tests')


def write_pbf(d, path, features=('OsmSchema-V0.6', 'DenseNodes')):
    """The data as PBF: tagged nodes split over a dense block (granularity 100), a dense block with granularity 10 and
    lat/lon offsets, and a plain-Node block with offsets; ways in a zlib block and a raw block (with a relation)."""
    ids = sorted(d.nodes)
    thirds = [ids[:len(ids) // 3], ids[len(ids) // 3:2 * len(ids) // 3], ids[2 * len(ids) // 3:]]
    part = [{i: d.nodes[i] for i in t} for t in thirds]
    ways = sorted(d.ways, key=lambda w: w['id'])
    out = _blob('OSMHeader', _header(d.bounds, features), True)
    out += _blob('OSMData', _dense_block(part[0], d.node_tags), True)
    out += _blob('OSMData', _dense_block(part[1], d.node_tags, gran=10, lat_off=12_000_000_000,
                                         lon_off=77_000_000_000), False)
    out += _blob('OSMData', _plain_block(part[2], d.node_tags, lat_off=12_900_000_000, lon_off=77_500_000_000), True)
    out += _blob('OSMData', _way_block(ways[:len(ways) // 2]), True)
    out += _blob('OSMData', _way_block(ways[len(ways) // 2:], relation=True), False)
    with open(path, 'wb') as fh:
        fh.write(out)


def _same(a, b):
    assert set(a.nodes) == set(b.nodes)
    for i, p in a.nodes.items():
        assert abs(p[0] - b.nodes[i][0]) < 1e-9 and abs(p[1] - b.nodes[i][1]) < 1e-9
    key = lambda w: w['id']                                                       # noqa: E731
    assert sorted(a.ways, key=key) == sorted(b.ways, key=key)
    assert a.node_tags == b.node_tags


# ------------------------------------------------------------------------------------------------ osm
def test_varints_are_decoded_vectorised():
    rng = np.random.default_rng(0)
    vals = [0, 1, 127, 128, 300, 2 ** 32, 2 ** 63 - 1, 2 ** 64 - 1] + rng.integers(0, 2 ** 62, 500).tolist()
    got = osm.decode_varints(b''.join(_v(x) for x in vals))
    assert got.dtype == np.uint64 and got.tolist() == vals
    signed = [0, -1, 1, -2 ** 40, 2 ** 40, -2 ** 63, 2 ** 63 - 1]
    assert osm.unzigzag(osm.decode_varints(b''.join(_v(_zz(x)) for x in signed))).tolist() == signed
    assert osm.decode_varints(b'').size == 0
    with pytest.raises(ValueError):
        osm.decode_varints(b'\x80')


def test_read_xml_fixture():
    d = osm.read_osm(SMALL_OSM)
    assert d.counts['highways'] == 88 and d.counts['nodes'] == 599 and d.counts['missing_node_refs'] == 0
    assert d.bounds == pytest.approx((12.9753441, 77.5981174, 12.9780559, 77.6008826))
    assert all(r in d.nodes for w in d.ways for r in w['nodes'])
    assert all(w['kind'] == 'highway' for w in d.ways)
    assert any(t.get('highway') == 'traffic_signals' for t in d.node_tags.values())
    names = {w['tags'].get('name') for w in d.highways}
    assert {'Kasturba Road', "Queen's Road", 'Mahatma Gandhi Road'} <= names


def test_crop_keeps_whole_ways_touching_the_box():
    d = osm.read_osm(SMALL_OSM)
    bb = osm.bbox_around(12.9767, 77.5995, 100)
    c = osm.crop(d, bb)
    s, w, n, e = bb

    def inside(r):
        return s <= d.nodes[r][0] <= n and w <= d.nodes[r][1] <= e
    expect = {x['id'] for x in d.ways if any(inside(r) for r in x['nodes'])}
    assert {x['id'] for x in c.ways} == expect and 0 < len(expect) < len(d.ways)
    for x in c.ways:                                              # complete ways, sticking out of the box
        assert x['nodes'] == next(y for y in d.ways if y['id'] == x['id'])['nodes']
    assert set(c.nodes) == {r for x in c.ways for r in x['nodes']}
    assert any(not inside(r) for r in c.nodes)
    assert c.bounds == bb
    _same(osm.read_osm(SMALL_OSM, bbox=bb), c)
    with pytest.raises(ValueError):
        osm.crop(d, (13.0, 179.0, 13.1, -179.0))


def test_write_osm_round_trips(tmp_path):
    d = osm.read_osm(SMALL_OSM)
    d.ways[0]['tags']['note'] = 'a "quoted" <tag> & more\nline'
    p = str(tmp_path / 'out.osm')
    osm.write_osm(d, p)
    text = _text(p)
    assert text.index('<bounds') < text.index('<node') < text.index('<way')
    _same(osm.read_osm(p), d)


def test_pbf_reader_matches_xml(tmp_path):
    d = osm.read_osm(SMALL_OSM)
    p = str(tmp_path / 'small.osm.pbf')
    write_pbf(d, p)
    seen = []
    q = osm.read_osm(p, progress=lambda f, t: seen.append(f))
    _same(q, d)
    assert q.counts['relations_skipped'] == 1 and any('relations' in n for n in q.notes)
    assert q.bounds == pytest.approx(d.bounds, abs=1e-9)
    assert seen and seen[-1] == 1.0 and all(0 <= f <= 1 for f in seen)
    bb = osm.bbox_around(12.9767, 77.5995, 100)
    _same(osm.read_osm(p, bbox=bb), osm.crop(d, bb))


def test_pbf_refuses_unsupported_features(tmp_path):
    d = osm.read_osm(SMALL_OSM)
    p = str(tmp_path / 'hist.osm.pbf')
    write_pbf(d, p, features=('OsmSchema-V0.6', 'HistoricalInformation'))
    with pytest.raises(ValueError, match='HistoricalInformation'):
        osm.read_osm(p)
    with open(p, 'wb') as fh:
        fh.write(b'<?xml version="1.0"?><osm/>')
    with pytest.raises(ValueError):
        osm.read_osm(p)


@pytest.mark.parametrize('tags, want', [
    ({'highway': 'primary', 'lanes': '4'}, (2, 2)),
    ({'highway': 'primary', 'lanes': '3'}, None),                          # odd, split unspecified
    ({'highway': 'primary', 'lanes': '3', 'lanes:forward': '2'}, (2, 1)),
    ({'highway': 'primary', 'lanes': '3', 'lanes:backward': '2'}, (1, 2)),
    ({'highway': 'primary', 'lanes:forward': '2', 'lanes:backward': '1'}, (2, 1)),
    ({'highway': 'primary', 'lanes': '5', 'lanes:both_ways': '1'}, (2, 2)),
    ({'highway': 'primary', 'lanes': '2', 'oneway': 'yes'}, (2, 0)),
    ({'highway': 'primary', 'lanes': '2', 'oneway': 'true'}, (2, 0)),
    ({'highway': 'primary', 'lanes': '2', 'oneway': '1'}, (2, 0)),
    ({'highway': 'primary', 'lanes': '2', 'oneway': '-1'}, (0, 2)),
    ({'highway': 'primary', 'lanes': '2', 'oneway': 'reverse'}, (0, 2)),
    ({'highway': 'primary', 'lanes': '2', 'oneway': 'reversible'}, None),
    ({'highway': 'motorway', 'lanes': '3'}, (3, 0)),                       # oneway implied
    ({'highway': 'primary', 'junction': 'roundabout', 'lanes': '2'}, (2, 0)),
    ({'highway': 'primary', 'oneway': 'no', 'lanes': '2'}, (1, 1)),
    ({'highway': 'primary'}, None),
    ({'highway': 'primary', 'oneway': 'yes'}, None),
    ({'highway': 'primary', 'lanes': '2;3'}, None),
])
def test_expected_lanes(tags, want):
    assert osm.expected_lanes(tags) == want
    assert osm.expected_lanes({'id': 1, 'nodes': [], 'tags': tags}) == want


def test_projection_matches_proj():
    lat0, lon0 = 12.9702655, 77.5861
    P = osm.local_projection(lat0, lon0)
    assert P.proj == osm.proj_string(lat0, lon0) == ('+proj=tmerc +lat_0=12.9702655 +lon_0=77.5861 +k=1 +x_0=0 '
                                                     '+y_0=0 +ellps=WGS84 +units=m +no_defs')
    # reference values: PROJ 9.4.1 (pyproj) with that proj string
    ref = [((12.9846, 77.6048), (2028.7899796305144, 1585.9043437325101)),
           ((12.9559, 77.5674), (-2029.022566319536, -1589.1835091440985)),
           ((13.0152, 77.6311), (4881.517203836788, 4971.554523528058))]
    for (lat, lon), (e, n) in ref:
        x, y = P.to_local(lat, lon)
        assert isinstance(x, float) and abs(x - e) < 1e-3 and abs(y - n) < 1e-3
        la, lo = P.to_geo(x, y)
        assert abs(la - lat) < 1e-10 and abs(lo - lon) < 1e-10
    assert P.to_local(lat0, lon0) == pytest.approx((0.0, 0.0), abs=1e-9)
    xs, ys = P.to_local(np.array([lat0, 12.98]), np.array([lon0, 77.59]))
    assert xs.shape == (2,)


def test_bbox_around_and_summary():
    s, w, n, e = osm.bbox_around(12.9767, 77.5995, 1000)
    P = osm.local_projection(12.9767, 77.5995)                         # ellipsoidal metres through the centre
    assert P.to_local(n, 77.5995)[1] - P.to_local(s, 77.5995)[1] == pytest.approx(1000, abs=0.01)
    assert P.to_local(12.9767, e)[0] - P.to_local(12.9767, w)[0] == pytest.approx(1000, abs=0.01)
    assert osm._haversine(s, 77.5995, n, 77.5995) == pytest.approx(1000, rel=0.006)   # spherical, within 0.5 %
    d = osm.read_osm(SMALL_OSM)
    sm = osm.summary(d)
    assert sm['highways'] == 88 and sm['by_highway']['footway'] == 56 and sm['by_highway']['secondary'] == 12
    assert 0 < sm['road_length_m'] < sm['highway_length_m']
    assert sm['oneway'] == 21 and sm['with_lanes_tag'] == 19 and sm['buildings'] == 0
    assert sm['centre'] == pytest.approx((12.9767, 77.5995), abs=1e-6)


def test_buildings_are_kept_with_a_flag(tmp_path):
    p = tmp_path / 'b.osm'
    p.write_text('<?xml version="1.0"?><osm version="0.6">'
                 '<node id="1" lat="12.0" lon="77.0"/><node id="2" lat="12.001" lon="77.0"/>'
                 '<node id="3" lat="12.001" lon="77.001"/>'
                 '<way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="1"/><tag k="building" v="yes"/></way>'
                 '<way id="11"><nd ref="1"/><nd ref="3"/><nd ref="99"/><tag k="highway" v="service"/></way>'
                 '<way id="12"><nd ref="1"/><nd ref="2"/><tag k="waterway" v="river"/></way>'
                 '<relation id="5"><member type="way" ref="10" role="outer"/></relation></osm>')
    d = osm.read_osm(str(p))
    assert [(w['id'], w['kind']) for w in d.ways] == [(10, 'building'), (11, 'highway')]
    assert d.counts['missing_node_refs'] == 1 and d.counts['relations_skipped'] == 1
    assert d.ways[1]['nodes'] == [1, 3]
    assert osm.summary(d)['buildings'] == 1


def test_geocode_needs_permission_and_parses(monkeypatch):
    with pytest.raises(PermissionError):
        osm.geocode('Bengaluru', allowed=False)
    canned = [{'display_name': 'Bengaluru, Karnataka, India', 'lat': '12.9767936', 'lon': '77.590082',
               'boundingbox': ['12.8340125', '13.1436649', '77.4601025', '77.7840515']},
              {'display_name': 'broken'}]
    got = osm.parse_geocode(canned)
    assert got == [('Bengaluru, Karnataka, India', 12.9767936, 77.590082,
                    (12.8340125, 77.4601025, 13.1436649, 77.7840515))]
    seen = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(canned).encode()

    def fake_urlopen(req, timeout):
        seen['url'], seen['ua'] = req.full_url, req.get_header('User-agent')
        return _Resp()
    monkeypatch.setattr(osm.urllib.request, 'urlopen', fake_urlopen)
    assert osm.geocode('Bengaluru', allowed=True) == got
    assert seen['ua'] == 'FalconAutonomy/0.1 (+local desktop app)' and 'q=Bengaluru' in seen['url']


@big_osm
def test_full_bengaluru_osm():
    t = time.perf_counter()
    d = osm.read_osm(BIG_OSM)
    assert time.perf_counter() - t < 5
    assert d.counts['highways'] == 2591 and d.counts['missing_node_refs'] == 0
    bb = osm.bbox_around(12.9702, 77.5861, 500)
    _same(osm.read_osm(BIG_OSM, bbox=bb), osm.crop(d, bb))


# ------------------------------------------------------------------------------------------------ xodr geometry
def _geom(kind, length, x=0.0, y=0.0, hdg=0.0, **params):
    return xodr.Geometry(0.0, x, y, hdg, length, kind, params)


def test_line_and_arc():
    x, y, h = xodr.eval_geometry(_geom('line', 10, 1, 2, math.pi / 2), [0, 10])
    assert np.allclose(x, [1, 1]) and np.allclose(y, [2, 12]) and np.allclose(h, math.pi / 2)
    q = 10 * math.pi / 2                                       # quarter circle, radius 10, turning left
    x, y, h = xodr.eval_geometry(_geom('arc', q, curvature=0.1), [q])
    assert x[0] == pytest.approx(10, abs=1e-9) and y[0] == pytest.approx(10, abs=1e-9)
    assert h[0] == pytest.approx(math.pi / 2)
    x, y, h = xodr.eval_geometry(_geom('arc', q, curvature=-0.1), [q])
    assert (x[0], y[0]) == pytest.approx((10, -10), abs=1e-9)


def test_spiral_against_arc_line_and_fresnel_series():
    q = 10 * math.pi / 2
    ds = np.linspace(0, q, 7)
    a = xodr.eval_geometry(_geom('arc', q, 3, 4, 0.3, curvature=0.1), ds)
    s = xodr.eval_geometry(_geom('spiral', q, 3, 4, 0.3, curvStart=0.1, curvEnd=0.1), ds)
    assert np.allclose(a, s, atol=1e-9)
    s0 = xodr.eval_geometry(_geom('spiral', 50, curvStart=0.0, curvEnd=0.0), [50])
    assert (s0[0][0], s0[1][0]) == pytest.approx((50, 0), abs=1e-9)
    L, k1 = 40.0, 0.05                                         # clothoid from straight: theta = c u^2 / 2
    c = k1 / L

    def fresnel(sv):
        x = sum((-1) ** n * (c / 2) ** (2 * n) * sv ** (4 * n + 1) / (math.factorial(2 * n) * (4 * n + 1))
                for n in range(20))
        y = sum((-1) ** n * (c / 2) ** (2 * n + 1) * sv ** (4 * n + 3) / (math.factorial(2 * n + 1) * (4 * n + 3))
                for n in range(20))
        return x, y
    x, y, h = xodr.eval_geometry(_geom('spiral', L, curvStart=0.0, curvEnd=k1), [L / 3, L])
    for i, sv in enumerate((L / 3, L)):
        assert (x[i], y[i]) == pytest.approx(fresnel(sv), abs=1e-9)
    assert h[1] == pytest.approx(k1 * L / 2)


def test_parametric_curves():
    hdg = 0.7
    for prange, b in (('normalized', 10.0), ('arcLength', 1.0)):
        g = _geom('paramPoly3', 10, 1, 1, hdg, aU=0, bU=b, cU=0, dU=0, aV=0, bV=0, cV=0, dV=0, pRange=prange)
        x, y, h = xodr.eval_geometry(g, [0, 5, 10])
        assert np.allclose(x, 1 + np.array([0, 5, 10]) * math.cos(hdg), atol=1e-9)
        assert np.allclose(y, 1 + np.array([0, 5, 10]) * math.sin(hdg), atol=1e-9)
        assert np.allclose(h, hdg)
    # a straight line traced non-uniformly (u = 10 p^2): s must map to the arc length, not to p
    g = _geom('paramPoly3', 10, aU=0, bU=0, cU=10, dU=0, aV=0, bV=0, cV=0, dV=0, pRange='normalized')
    x, y, _ = xodr.eval_geometry(g, [2.5, 5])
    assert np.allclose(x, [2.5, 5], atol=1e-4) and np.allclose(y, 0)
    # poly3 offset laterally by a: a straight line 1 m left of the start point
    x, y, h = xodr.eval_geometry(_geom('poly3', 10, a=1, b=0, c=0, d=0), [0, 10])
    assert np.allclose(x, [0, 10]) and np.allclose(y, [1, 1]) and np.allclose(h, 0)


def test_sample_reference_line_covers_the_road():
    r = xodr.Road('1', 'r', 25.0, '-1', None, None, None,
                  [xodr.Geometry(0, 0, 0, 0, 10, 'line'), xodr.Geometry(10, 10, 0, 0, 15, 'arc', {'curvature': 0.0})],
                  [], [])
    pts = xodr.sample_reference_line(r, step=2.0)
    assert pts[0].tolist() == [0, 0, 0] and pts[-1][:2].tolist() == pytest.approx([25, 0])
    gaps = np.hypot(*np.diff(pts[:, :2], axis=0).T)
    assert gaps.max() <= 2.0 + 1e-9 and gaps.min() > 0


# ------------------------------------------------------------------------------------------------ xodr structure
def test_parse_sumo_fixture():
    od = xodr.parse(_text(SMALL_XODR))
    assert len(od.roads) == 26 and len(od.junctions) == 3 and od.signals == 1
    assert od.header['revMinor'] == '4' and '+proj=tmerc' in od.header['geoReference']
    assert od.header['offset']['x'] == pytest.approx(-11783042.32)
    assert xodr.handedness(od) == 'RHT'
    for r in od.roads:                                         # SUMO: one travel direction per road, right lanes only
        assert r.rule is None and all(not s.left for s in r.sections)
        assert [ln.id for ln in r.sections[0].right] == list(range(-1, -len(r.sections[0].right) - 1, -1))
    conn = [r for r in od.roads if r.in_junction]
    assert conn and all(r.name.startswith(':') for r in conn)
    assert sum(1 for r in od.roads if r.lane_offsets) == 1
    lanes = [ll for j in od.junctions for c in j.connections for ll in c.lane_links]
    assert any(a != b for a, b in lanes) and all(a < 0 and b < 0 for a, b in lanes)
    assert any(r.signal_refs for r in od.roads)
    assert any(g.kind == 'paramPoly3' for r in od.roads for g in r.geometries)


def test_road_graph_preview():
    od = xodr.parse(_text(SMALL_XODR))
    g = xodr.road_graph(od)
    t = g['totals']
    assert t['roads'] + t['connecting_roads'] == 26 and t['junctions'] == 3
    assert t['road_km'] == pytest.approx(sum(r.length for r in od.roads if not r.in_junction) / 1000, abs=1e-3)
    assert t['driving_lanes_per_direction'] and t['handedness'] == 'RHT'
    names = {r['name'] for r in g['roads']}
    assert "Queen's Road" in names                             # SUMO's double-escaped &amp;apos; decoded once
    for item in g['roads'] + g['connections']:
        assert len(item['x']) == len(item['y']) >= 2
    r = next(r for r in od.roads if not r.in_junction and len(r.geometries) > 1)
    item = next(i for i in g['roads'] if i['id'] == r.id)
    full = xodr.sample_reference_line(r)
    assert len(item['x']) <= len(full)
    assert (item['x'][0], item['y'][0]) == pytest.approx(tuple(full[0, :2]), abs=0.01)
    assert (item['x'][-1], item['y'][-1]) == pytest.approx(tuple(full[-1, :2]), abs=0.01)
    assert len(json.dumps(g)) < 60_000


def _structure(od):
    roads = {}
    for r in od.roads:
        lanes = [(s.s, [(ln.id, ln.type, ln.widths, ln.predecessor, ln.successor) for ln in s.left + s.right])
                 for s in r.sections]
        roads[r.id] = (lanes, r.lane_offsets, [(g.kind, g.x, g.y, g.hdg, g.length) for g in r.geometries])
    links = {(j.id, c.id): c.lane_links for j in od.junctions for c in j.connections}
    return roads, links


def test_to_lht_converts_the_sumo_fixture():
    text = _text(SMALL_XODR)
    before = xodr.parse(text)
    new, notes = xodr.to_lht(text)
    after = xodr.parse(new)
    assert xodr.handedness(after) == 'LHT' and all(r.rule == 'LHT' for r in after.roads)
    assert new.startswith('<?xml') and '<!-- Cut from carla.Osm2Odr' in new and '<![CDATA[' in new
    assert any('Limit' in n for n in notes)
    paired = xodr._paired(before)
    for b, a in zip(before.roads, after.roads):
        assert a.id == b.id and a.geometries == b.geometries       # travel direction (reference line) unchanged
        if b.id in paired:                                         # mirrored across the shared reference line
            assert a.lane_offsets == [(s, -p, -q, -r, -t) for s, p, q, r, t in b.lane_offsets]
        for sb, sa in zip(b.sections, a.sections):
            assert not sa.right and [ln.id for ln in sa.left] == sorted((-ln.id for ln in sb.right), reverse=True)
            for ln in sa.left:
                old = next(x for x in sb.right if x.id == -ln.id)
                assert (ln.type, ln.widths) == (old.type, old.widths)
                assert ln.predecessor == (None if old.predecessor is None else -old.predecessor)
                assert ln.successor == (None if old.successor is None else -old.successor)
            assert all(ln.id > 0 for ln in sa.left if ln.type == 'driving')
    raw_gaps, gaps = xodr.lane_link_gaps(before), xodr.lane_link_gaps(after)
    assert len(gaps) == len(raw_gaps) == 38                         # linked lanes still meet (pure mirroring: 13.4 m)
    assert np.percentile(gaps, 95) < 1.5 and gaps.max() < 2.3 and (gaps > 0.5).sum() <= (raw_gaps > 0.5).sum()
    assert any('8 one-way roads kept in place' in n for n in notes)
    for jb, ja in zip(before.junctions, after.junctions):
        for cb, ca in zip(jb.connections, ja.connections):
            assert ca.lane_links == [(-f, -t) for f, t in cb.lane_links]
    for m in re.finditer(r'<validity fromLane="(-?\d+)" toLane="(-?\d+)"', new):
        assert int(m.group(1)) <= int(m.group(2))
    with pytest.raises(ValueError, match='already'):
        xodr.to_lht(new)


def test_to_lht_twice_with_rule_reset_is_the_identity():
    text = _text(SMALL_XODR)
    once, _ = xodr.to_lht(text)
    twice, _ = xodr.to_lht(once.replace(' rule="LHT"', ''))
    a, b = xodr.parse(twice), xodr.parse(text)
    got, want = _structure(a), _structure(b)
    assert got[1] == want[1] and set(got[0]) == set(want[0])
    for rid, (lanes, _, geoms) in want[0].items():
        assert got[0][rid][0] == lanes and got[0][rid][2] == geoms
        r = b.road(rid)
        if r.in_junction:                     # a connector's laneOffset is made to meet its links, which the
            continue                          # converter's own connectors miss by up to 1.5 m here (checked below)
        for s in np.linspace(0, r.length, 9):
            assert xodr._cubic_at(a.road(rid).lane_offsets, s) == pytest.approx(xodr._cubic_at(r.lane_offsets, s),
                                                                                abs=1e-6), rid
    assert np.percentile(xodr.lane_link_gaps(a), 95) <= np.percentile(xodr.lane_link_gaps(b), 95) + 1e-6


def _road_xml(rid, name, x, y, hdg, length, right=(), left=(), junction='-1', extra='', rule=''):
    def lanes(ids):
        return ''.join(f'<lane id="{i}" type="{t}" level="false"><link/><width sOffset="0" a="{w}" b="0" c="0" d="0"/>'
                       f'</lane>' for i, t, w in ids)
    sides = ''
    if left:
        sides += f'<left>{lanes(sorted(left, key=lambda z: -z[0]))}</left>'
    sides += '<center><lane id="0" type="none" level="false"/></center>'
    if right:
        sides += f'<right>{lanes(sorted(right, key=lambda z: -z[0]))}</right>'
    rule = f' rule="{rule}"' if rule else ''
    return (f'<road name="{name}" length="{length}" id="{rid}" junction="{junction}"{rule}><link/>'
            f'<planView><geometry s="0" x="{x}" y="{y}" hdg="{hdg}" length="{length}"><line/></geometry></planView>'
            f'<lanes><laneSection s="0">{sides}</laneSection></lanes>{extra}</road>')


def _doc(*roads, junctions=''):
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<OpenDRIVE><header revMajor="1" revMinor="4" name="t"/>'
            + ''.join(roads) + junctions + '</OpenDRIVE>\n')


def test_to_lht_signals_objects_and_refusals():
    extra = ('<objects><object id="o" type="crosswalk" s="5" t="1.5" hdg="0.25"><outline id="0">'
             '<cornerLocal u="1" v="-2" z="0" height="0" id="0"/></outline></object></objects>'
             '<signals><signal id="s" s="3" t="-4.5" orientation="+" hOffset="0.1">'
             '<validity fromLane="-2" toLane="-1"/></signal></signals>'
             '<lateralProfile><superelevation s="0" a="0.02" b="0" c="0" d="0"/></lateralProfile>')
    text = _doc(_road_xml(1, 'A', 0, 0, 0, 50, right=[(-1, 'driving', 3.5), (-2, 'driving', 3.5)], extra=extra),
                _road_xml(3, 'A', 50, 0, math.pi, 50, right=[(-1, 'driving', 3.5)]))   # the other direction
    new, notes = xodr.to_lht(text)
    # its lanes are now on the other side, so is its orientation (kept, CARLA's lights controlled no lane: measured)
    assert 'signal id="s" s="3" t="4.5"' in new and 'hOffset="0.1"' in new and 'orientation="-"' in new
    assert '<validity fromLane="1" toLane="2"' in new
    assert 't="-1.5" hdg="-0.25"' in new and 'u="1" v="2"' in new
    assert 'superelevation s="0" a="-0.02"' in new
    mixed = _doc(_road_xml(1, 'A', 0, 0, 0, 50, right=[(-1, 'driving', 3.5)]),
                 _road_xml(2, 'B', 0, 0, 0, 50, left=[(1, 'driving', 3.5)], rule='LHT'),
                 junctions='<junction id="9" name="j"><connection id="0" incomingRoad="1" connectingRoad="2" '
                           'contactPoint="start"><laneLink from="-1" to="1"/></connection></junction>')
    assert xodr.handedness(xodr.parse(mixed)) == 'mixed'
    with pytest.raises(ValueError, match='mixed'):
        xodr.to_lht(mixed)
    forced, notes = xodr.to_lht(mixed, force=True)
    od = xodr.parse(forced)
    assert xodr.handedness(od) == 'LHT' and [ln.id for ln in od.roads[0].sections[0].left] == [1]
    assert od.roads[1].sections[0].left[0].id == 1                  # the already-LHT road is untouched
    assert od.junctions[0].connections[0].lane_links == [(1, 1)]
    assert any('already LHT' in n for n in notes)
    with pytest.raises(ValueError, match='shape'):
        xodr.to_lht(_doc(_road_xml(1, 'A', 0, 0, 0, 50, right=[(-1, 'driving', 3.5)],
                                   extra='<lateralProfile><shape s="0" t="0" a="0" b="0" c="0" d="0"/>'
                                         '</lateralProfile>')))


# ------------------------------------------------------------------------------------------------ lane check
LAT0, LON0 = 12.9767, 77.5995


def _synthetic(offset=(0.0, 0.0)):
    """OSM ways laid out in local metres, and an OpenDRIVE (x east, y north, plus offset) built to match some."""
    P = osm.local_projection(LAT0, LON0)
    nodes, ways = {}, []

    def way(wid, pts, **tags):
        refs = []
        for e, n in pts:
            nid = len(nodes) + 1
            nodes[nid] = P.to_geo(e, n)
            refs.append(nid)
        ways.append({'id': wid, 'nodes': refs, 'tags': {'highway': tags.pop('highway', 'secondary'), **tags},
                     'kind': 'highway'})
    way(1, [(-100, 0), (0, 0), (100, 0)], name='Main Road', lanes='4')
    way(2, [(0, 0), (0, 150)], name='Side Street', oneway='yes', lanes='1')
    way(3, [(200, 0), (200, 150)], name='Wrong Way', oneway='yes', lanes='2')
    way(4, [(-100, 100), (100, 100)], name='Bad Road', **{'lanes:forward': '2', 'lanes:backward': '1'})
    way(5, [(-100, -100), (100, -100)], name='Ghost Road', lanes='2')
    way(6, [(-100, -200), (100, -200)], name='Far Road', lanes='2')
    way(7, [(-100, 300), (100, 300)], name='Odd Road', lanes='3')                  # not eligible: odd total
    way(8, [(-100, 400), (100, 400)], lanes='2')                                   # not eligible: no name
    way(9, [(-100, 500), (100, 500)], name='Path', lanes='2', highway='footway')   # not eligible: type
    d = osm.OsmData(nodes, ways, (LAT0 - 0.01, LON0 - 0.01, LAT0 + 0.01, LON0 + 0.01), 'synthetic')
    dx, dy = offset
    sw = (-3, 'sidewalk', 2.8)
    two = [(-1, 'driving', 3.35), (-2, 'driving', 3.35), sw]
    one = [(-1, 'driving', 3.35), (-2, 'sidewalk', 2.8)]
    text = _doc(
        _road_xml(1, 'Main Road', -100 + dx, 0 + dy, 0, 200, right=two),
        _road_xml(2, 'Main Road', 100 + dx, 0 + dy, math.pi, 200, right=two),
        _road_xml(3, 'Side Street', 0 + dx, 0 + dy, math.pi / 2, 150, right=one),
        _road_xml(4, 'Wrong Way', 200 + dx, 150 + dy, -math.pi / 2, 150, right=two),
        _road_xml(5, 'Bad Road', -100 + dx, 100 + dy, 0, 200, right=one),
        _road_xml(6, 'Bad Road', 100 + dx, 100 + dy, math.pi, 200, right=one),
        _road_xml(7, 'Far Road', -100 + dx, 300 + dy, 0, 200, right=one),
        _road_xml(8, 'Main Road', -10 + dx, 0 + dy, 0, 20, right=[(-1, 'driving', 3), (-2, 'driving', 3),
                                                                  (-3, 'driving', 3)], junction='1'))
    return d, text, P


def test_lane_check_on_synthetic_network():
    d, text, P = _synthetic()
    res = xodr.lane_check(xodr.parse(text), d, P)
    rows = {r['way']: r for r in res['rows']}
    assert res['eligible'] == 6 and set(rows) == {1, 2, 3, 4, 5, 6}
    assert rows[1]['found'] == (2, 2) and rows[1]['matched'] and sorted(rows[1]['roads']) == ['1', '2']
    assert rows[2]['found'] == (1, 0) and rows[2]['matched']
    assert rows[3]['found'] == (0, 2) and rows[3]['matched'] is False
    assert rows[4]['expected'] == (2, 1) and rows[4]['found'] == (1, 1) and rows[4]['matched'] is False
    assert rows[5]['found'] is None and 'no OpenDRIVE road' in rows[5]['reason']
    assert rows[6]['found'] is None and 'within' in rows[6]['reason']
    assert (res['comparable'], res['matched'], res['agreement']) == (4, 2, 0.5)
    assert 'sample' in res['method']
    lht, _ = xodr.to_lht(text)                                         # same answer once the network drives left
    res2 = xodr.lane_check(xodr.parse(lht), d, P)
    assert [(r['way'], r['found'], r['matched']) for r in res2['rows']] == \
        [(r['way'], r['found'], r['matched']) for r in res['rows']]
    few = xodr.lane_check(xodr.parse(text), d, P.to_local, sample=3, seed=1)
    assert len(few['rows']) == 3 and few == xodr.lane_check(xodr.parse(text), d, P, sample=3, seed=1)


def _sumo_like():
    """A network written the way carla.Osm2Odr writes one (measured on central Bengaluru): a two-way way as two
    roads sharing the way as reference line, each with its lanes on its right; a one-way way as one road whose
    reference line is its carriageway's left edge (the way runs down the carriageway's middle); a junction road from
    the first to the second; and a two-way way only one of whose directions became a road. OSM ways in local metres."""
    P = osm.local_projection(LAT0, LON0)
    nodes, ways = {}, []

    def way(wid, pts, **tags):
        refs = []
        for e, n in pts:
            nodes[len(nodes) + 1] = P.to_geo(e, n)
            refs.append(len(nodes))
        ways.append({'id': wid, 'nodes': refs, 'tags': {'highway': 'secondary', **tags}, 'kind': 'highway'})
    way(1, [(-100, 0), (40, 0)], name='Two Way', lanes='2')
    way(2, [(50, 6.65), (50, 150)], name='One Way', oneway='yes', lanes='2')
    way(3, [(-100, -60), (100, -60)], name='Half Way', lanes='2')
    d = osm.OsmData(nodes, ways, (LAT0 - 0.01, LON0 - 0.01, LAT0 + 0.01, LON0 + 0.01), 'synthetic')
    one = [(-1, 'driving', 3.35), (-2, 'sidewalk', 2.8)]
    R = 6.65                  # a quarter circle, tangent at both ends, whose lane joins both roads' lanes
    link = ('<lane id="-1" type="driving" level="false"><link><predecessor id="-1"/><successor id="-1"/></link>'
            '<width sOffset="0" a="3.35" b="0" c="0" d="0"/></lane>')
    conn = (f'<road name="" length="{math.pi * R / 2}" id="4" junction="1"><link>'
            '<predecessor elementType="road" elementId="1" contactPoint="end"/>'
            '<successor elementType="road" elementId="3" contactPoint="start"/></link>'
            f'<planView><geometry s="0" x="40" y="0" hdg="0" length="{math.pi * R / 2}"><arc curvature="{1 / R}"/>'
            '</geometry></planView><lanes><laneSection s="0"><center><lane id="0" type="none" level="false"/></center>'
            f'<right>{link}</right></laneSection></lanes></road>')
    text = _doc(_road_xml(1, 'Two Way', -100, 0, 0, 140, right=one),
                _road_xml(2, 'Two Way', 40, 0, math.pi, 140, right=one),
                _road_xml(3, 'One Way', 50 - 3.35, R, math.pi / 2, 140,
                          right=[(-1, 'driving', 3.35), (-2, 'driving', 3.35)]),
                conn,
                _road_xml(5, 'Half Way', -100, -60, 0, 200, right=one))
    return d, text, P


def _span(road, x=None, y=None):
    """Median position (x or y) of a road's driving lanes' middle."""
    lc = xodr.lane_centres(road)
    return float(np.median(lc[:, 0] if x else lc[:, 1]))


def test_to_lht_keeps_one_way_roads_on_their_way_and_lanes_joined():
    d, text, P = _sumo_like()
    raw = xodr.parse(text)
    assert _span(raw.road('3'), x=True) == pytest.approx(50.0) and xodr.lane_link_gaps(raw).max() < 1e-6
    for ways in (xodr.osm_lines(d, P), None):
        new, notes = xodr.to_lht(text, ways=ways)
        od = xodr.parse(new)
        assert _span(od.road('3'), x=True) == pytest.approx(50.0, abs=1e-6)      # the one-way road stays on its way
        assert _span(od.road('1'), y=True) == pytest.approx(1.675)               # two-way: driving on the left
        assert _span(od.road('2'), y=True) == pytest.approx(-1.675)
        assert xodr.lane_link_gaps(od).max() < 1e-6                              # the junction road meets both
        check = xodr.lane_check(od, d, P)
        rows = {r['way']: r for r in check['rows']}
        assert rows[1]['position_error_m'] == pytest.approx(0, abs=1e-6)
        assert rows[2]['position_error_m'] == pytest.approx(0, abs=1e-6)
        # the half-built two-way way: its road's reference line is the way (mirror there) - which only the OSM way
        # can tell; without it the road is taken for a lone one-way road and kept where it was
        assert rows[3]['position_error_m'] == pytest.approx(0 if ways else 3.35, abs=1e-6)
        assert rows[3]['misplaced'] is (ways is None)
        assert check['position']['misplaced'] == (0 if ways else 1)
        assert any('kept in place' in n and 'junction connecting roads' in n for n in notes)
    rht = xodr.lane_check(raw, d, P)['position']
    assert rht['median_m'] == pytest.approx(0, abs=1e-6) and rht['misplaced'] == 0


def test_lane_check_with_an_offset():
    d, text, P = _synthetic(offset=(1000.0, -2000.0))
    od = xodr.parse(text)
    assert xodr.lane_check(od, d, P)['comparable'] == 0
    res = xodr.lane_check(od, d, P, transform=(1000.0, -2000.0))
    assert (res['comparable'], res['matched']) == (4, 2)


# ------------------------------------------------------------------------------------------------ real SUMO output
@big_xodr
def test_full_sumo_output_smoke():
    text = _text(BIG_XODR)
    t = time.perf_counter()
    od = xodr.parse(text)
    assert time.perf_counter() - t < 10
    assert len(od.roads) == 7554 and len(od.junctions) == 828 and xodr.handedness(od) == 'RHT'
    g = xodr.road_graph(od)
    assert g['totals']['roads'] == 2422 and g['totals']['connecting_roads'] == 5132
    t = time.perf_counter()
    new, notes = xodr.to_lht(text)
    assert time.perf_counter() - t < 20
    after = xodr.parse(new)
    assert xodr.handedness(after) == 'LHT'
    assert all(ln.id > 0 for r in after.roads for s in r.sections for ln in s.left) and \
        not any(s.right for r in after.roads for s in r.sections)
    assert sum(len(r.lane_offsets) for r in after.roads) == 1599
    gaps, raw = xodr.lane_link_gaps(after), xodr.lane_link_gaps(od)
    assert (gaps > 0.5).sum() <= (raw > 0.5).sum() and np.percentile(gaps, 95) < 2.5     # mirroring alone: 1214, 6.6 m


def test_sanitize_clamps_objects_past_the_road_end():
    from falcon_autonomy.core import xodr
    text = ('<OpenDRIVE><header/><road name="r" length="29.92954051" id="1428" junction="-1"><objects>'
            '<object id="7" type="crosswalk" s="29.93300628" t="0"/></objects><signals>'
            '<signal id="2" s="0" t="0"/></signals></road></OpenDRIVE>')
    out, notes = xodr.sanitize(text)
    assert len(notes) == 1 and 'mm' in notes[0] and 's="29.92954051"' in out
    same, none = xodr.sanitize(out)
    assert none == []
    with pytest.raises(ValueError):
        xodr.sanitize(text.replace('29.93300628', '31.0'))


# ------------------------------------------------------------------------------------------------ road features
def test_import_places_traffic_calming_as_elevation(tmp_path):
    """The import worker (fake converter) turns a traffic_calming=hump node into an IRC hump on both travel directions'
    roads, keeps it through the left-hand conversion, and reports a rumble strip as not represented."""
    import json
    import subprocess
    import sys
    from conftest import worker_env
    from falcon_autonomy.core import protocol as P
    from falcon_autonomy.core import roadfeatures as RF
    from falcon_autonomy.core.worlds import import_config, world_entry
    lat0, lon0 = 12.9716, 77.5946
    dlon = 1.0 / (111319.49 * np.cos(np.radians(lat0)))
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6">']
    for k in range(9):                               # 0 … 400 m east, a node every 50 m
        lines.append(f'<node id="{k + 1}" lat="{lat0:.7f}" lon="{lon0 + 50.0 * k * dlon:.9f}"/>')
    lines.append(f'<node id="100" lat="{lat0:.7f}" lon="{lon0 + 200.0 * dlon:.9f}">'
                 '<tag k="traffic_calming" v="hump"/></node>')
    lines.append(f'<node id="101" lat="{lat0:.7f}" lon="{lon0 + 320.0 * dlon:.9f}">'
                 '<tag k="traffic_calming" v="rumble_strip"/></node>')
    refs = [1, 2, 3, 4, 5, 100, 6, 7, 101, 8, 9]
    lines.append('<way id="1">' + ''.join(f'<nd ref="{r}"/>' for r in refs) +
                 '<tag k="highway" v="residential"/><tag k="name" v="Breaker Road"/></way></osm>')
    src = tmp_path / 'area.osm'
    src.write_text('\n'.join(lines))
    out = tmp_path / 'w'
    cfg = import_config(str(src), str(out), None, True, centre=(lat0, lon0 + 200.0 * dlon))
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'osm_import', str(tmp_path / 'c.json')],
                       capture_output=True, text=True, env=worker_env(), timeout=120)
    assert r.returncode == P.OK, r.stderr
    rep = json.loads((out / 'import.json').read_text())
    assert rep['handedness'] == 'LHT'
    assert rep['road_features']['counts'] == {'irc_hump': 2} and rep['road_features']['not_represented'] == \
        {'rumble_strip': 1}
    text = (out / 'world.xodr').read_text()
    feats = RF.read(text)
    assert len(feats) == 2 and {f['kind'] for f in feats} == {'irc_hump'}
    assert all(abs(f['x']) < 1.0 and abs(f['y']) < 1.0 for f in feats)          # at the node: the area's centre
    assert xodr.handedness(xodr.parse(text)) == 'LHT' and RF.read((out / 'world.base.xodr').read_text()) == []
    e = world_entry('w', 'W', cfg, rep)
    assert e['osm']['mesh']['vertex_distance'] == RF.FEATURE_VERTEX_DISTANCE_M
    cfg['settings']['speed_breakers'] = 'off'
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'osm_import', str(tmp_path / 'c.json')],
                       capture_output=True, text=True, env=worker_env(), timeout=120)
    assert r.returncode == P.OK and RF.read((out / 'world.xodr').read_text()) == []
