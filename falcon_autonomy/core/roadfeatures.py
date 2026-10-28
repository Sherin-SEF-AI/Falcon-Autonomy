"""Road features as real geometry: speed breakers and dips written into an OpenDRIVE world's elevation profiles
(standard library + numpy), placed from OpenStreetMap's traffic_calming tags or at a spacing the user chooses.

CARLA 0.9.16's packaged release cannot import a mesh, but generate_opendrive_world builds the road mesh from the
OpenDRIVE file, elevation included: a breaker written as an <elevationProfile> is a bump of the road surface that the
vehicle physics drives over. docs/indian-assets.md gives what was measured on the reference server (the mesh's height
along the road with world.ground_projection, a vehicle's z, pitch and vertical acceleration over each profile) and at
which mesh vertex distance a profile is rendered.

Profiles (PROFILES; every one spans the full width of the road — an elevation profile is a function of s only):

* irc_hump — IRC:99-1988 ("Tentative guidelines on the provision of speed breakers for control of vehicular speeds on
  minor roads", Indian Roads Congress) §5.1: a rounded hump of 17 m radius, 3.7 m wide (along the road) and 0.10 m
  high, for an advisory crossing speed of 25 km/h. Written as the parabola z = h·(1 − (2u/L)²) through the same
  chord and height (its curvature 8h/L² = 1/17.1 m; the circle's and the parabola's heights differ by under 0.3 mm).
* osm_bump — OpenStreetMap's traffic_calming=bump: "a short bump, with extent, in the direction of travel, of about
  30 cm or shorter" (wiki Tag:traffic_calming=bump). OSM gives no height: 0.075 m is Falcon's choice (heuristic). A
  parabola 0.30 m long.
* table — traffic_calming=table ("even longer in direction of travel with a flat middle", OSM wiki): 0.10 m high,
  2.0 m ramps (1:20, IRC:99-1988 §5.1's ramp slope) and a 4.0 m flat top (the top length is Falcon's choice).
* dip — a full-width depression, 0.08 m deep and 1.2 m long (a parabola; Falcon's choice). A *pothole proxy*: a real
  pothole is local across the lane, and OpenDRIVE can express that only with lateralProfile <shape>, which is checked
  on the server (docs/indian-assets.md says what CARLA renders) and which core.xodr.to_lht refuses to mirror.

OSM: traffic_calming is tagged on a highway node (or a short highway way). bump → osm_bump or, when the user maps it
so, irc_hump (many Indian "speed breakers" tagged bump are humps; whether one is cannot be told from the tag); hump →
irc_hump; table → table; yes → irc_hump; rumble_strip, cushion, chicane, island, choker and the rest are reported as
not represented (rumble strips are 1–2 cm ridges, below what the mesh resolves; the others are not elevation).

Placement. A feature at a point goes onto every non-junction road whose reference line passes within PLACE_TOL_M of
it (in an imported OSM world each travel direction is its own road with its reference line at the street's centre:
both get it), at the s of the nearest reference-line point, if the whole profile fits on the road with END_MARGIN_M
to spare at both ends (an elevation that is not zero at a road's end would step against the next road); otherwise it
is skipped and the report says why. Features closer than their lengths on one road are merged into the first.
Roads whose elevation profile is not flat already are left alone (reported). Each placed feature is also written as
<userData code="falcon:road_feature" value="kind=…;…"/> in its road (CARLA's parser ignores userData; core.rules
reads it: the "speed over a breaker" rule).

Files of a world (worlds/<id>/): world.xodr is what the server loads; world.base.xodr the same without features
(written the first time features are applied, so they can be changed or removed); features.json what was placed
(falcon.road_features/1).
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np

SCHEMA = 'falcon.road_features/1'
USERDATA_CODE = 'falcon:road_feature'
PLACE_TOL_M = 3.0
END_MARGIN_M = 0.5
IRC_99 = ('IRC:99-1988 (Indian Roads Congress), "Tentative guidelines on the provision of speed breakers for control '
          'of vehicular speeds on minor roads", §5.1 and §5.2')
OSM_WIKI = 'OpenStreetMap wiki, Key:traffic_calming and Tag:traffic_calming=bump/hump/table'

# kind: name, shape, along-road length (m), height (m, negative: a dip), ramp (table), advisory crossing speed, source
PROFILES = {
    'irc_hump': {'name': 'speed breaker (IRC hump)', 'shape': 'parabola', 'length_m': 3.7, 'height_m': 0.10,
                 'crossing_kmh': 25.0, 'source': f'{IRC_99}: 17 m radius, 3.7 m, 0.10 m, 25 km/h'},
    'osm_bump': {'name': 'speed bump (short)', 'shape': 'parabola', 'length_m': 0.30, 'height_m': 0.075,
                 'crossing_kmh': 25.0, 'source': f'{OSM_WIKI}: about 30 cm or shorter along the road; the height '
                                                 '0.075 m and the 25 km/h crossing speed are Falcon\'s (heuristic)'},
    'table': {'name': 'speed table', 'shape': 'table', 'length_m': 8.0, 'ramp_m': 2.0, 'height_m': 0.10,
              'crossing_kmh': 25.0, 'source': f'{OSM_WIKI} (a long hump with a flat middle); 1:20 ramps as '
                                              f'{IRC_99.split(",")[0]} §5.1; the 4 m top is Falcon\'s (heuristic)'},
    'dip': {'name': 'dip (pothole proxy, full width)', 'shape': 'parabola', 'length_m': 1.2, 'height_m': -0.08,
            'crossing_kmh': None, 'source': 'Falcon\'s (heuristic): a full-width depression; a pothole local across '
                                            'the lane is not expressible here (docs/indian-assets.md)'},
}
OSM_KINDS = {'bump': 'osm_bump', 'hump': 'irc_hump', 'table': 'table', 'yes': 'irc_hump'}
NOT_ELEVATION = ('rumble_strip', 'cushion', 'chicane', 'choker', 'island', 'painted_table', 'mini_bumps',
                 'double_dip', 'dip', 'chicane;choker')


# ------------------------------------------------------------------------------------------------ profiles
def pieces(kind: str, s0: float) -> list:
    """The feature as cubic elevation pieces [(s, a, b, c, d)] from s0 (its start) to its end, then a flat zero piece
    starting at its end: z(ds) = a + b·ds + c·ds² + d·ds³ (OpenDRIVE's <elevation>)."""
    p = PROFILES[kind]
    L, h = float(p['length_m']), float(p['height_m'])
    if p['shape'] == 'parabola':
        out = [(s0, 0.0, 4.0 * h / L, -4.0 * h / (L * L), 0.0)]
    elif p['shape'] == 'table':
        r = float(p['ramp_m'])
        out = [(s0, 0.0, h / r, 0.0, 0.0), (s0 + r, h, 0.0, 0.0, 0.0), (s0 + L - r, h, -h / r, 0.0, 0.0)]
    else:
        raise ValueError(f'unknown profile shape {p["shape"]!r}')
    return out + [(s0 + L, 0.0, 0.0, 0.0, 0.0)]


def height_at(entries: list, s) -> np.ndarray:
    """OpenDRIVE's elevation at s (array) from [(s, a, b, c, d)] sorted by s (before the first: the first's a)."""
    s = np.atleast_1d(np.asarray(s, float))
    out = np.zeros_like(s)
    if not entries:
        return out
    starts = np.array([e[0] for e in entries])
    k = np.clip(np.searchsorted(starts, s + 1e-9, side='right') - 1, 0, len(entries) - 1)
    E = np.array([e[1:] for e in entries], float)[k]
    u = np.maximum(s - starts[k], 0.0)
    return E[:, 0] + E[:, 1] * u + E[:, 2] * u * u + E[:, 3] * u ** 3


def profile_entries(features: list) -> list:
    """The road's elevation entries for its features [{'kind', 's'} (s: the feature's centre)], flat elsewhere."""
    out = [(0.0, 0.0, 0.0, 0.0, 0.0)]
    for f in sorted(features, key=lambda f: f['s']):
        L = float(PROFILES[f['kind']]['length_m'])
        ps = pieces(f['kind'], float(f['s']) - L / 2.0)
        if out and abs(out[-1][0] - ps[0][0]) < 1e-9:
            out.pop()
        out += ps
    return out


def _fmt(v: float) -> str:
    return f'{v:.10g}'


# ------------------------------------------------------------------------------------------------ the file
def flat_profile(road) -> bool:
    """Whether a <road> element's elevation profile is absent or zero everywhere."""
    for e in road.findall('elevationProfile/elevation'):
        if any(abs(float(e.get(k, '0') or 0.0)) > 1e-12 for k in 'abcd'):
            return False
    return True


def _road_xy(road_el, s: float):
    """(x, y, heading) on a <road> element's reference line at s, and its lane widths (left, right) at s."""
    from . import xodr as X
    r = X._road(road_el)
    geoms = [g for g in r.geometries if g.length > 0]
    g = geoms[0]
    for c in geoms:
        if c.s <= s + 1e-9:
            g = c
    x, y, h = X.eval_geometry(g, np.array([s - g.s]))
    sec = r.sections[0] if r.sections else None
    for c in r.sections:
        if c.s <= s + 1e-9:
            sec = c

    def width(lanes):
        tot = 0.0
        for ln in lanes:
            if ln.type in ('driving', 'shoulder', 'parking', 'bidirectional', 'border', 'restricted', 'biking'):
                w = 0.0
                for e in ln.widths:
                    if e[0] <= s - (sec.s if sec else 0) + 1e-9:
                        u = s - (sec.s if sec else 0) - e[0]
                        w = e[1] + e[2] * u + e[3] * u * u + e[4] * u ** 3
                tot += max(w, 0.0)
        return tot
    wl, wr = (width(sec.left), width(sec.right)) if sec else (0.0, 0.0)
    return float(x[0]), float(y[0]), float(h[0]), wl, wr


def apply(text: str | bytes, placements: list) -> tuple[str, list, list]:
    """The OpenDRIVE document with the features written into its roads' elevation profiles.

    placements: [{'road': id, 's': centre s, 'kind', 'source'}]. Returns (text, placed, skipped): placed rows add
    the feature's centre (x, y, heading in OpenDRIVE's frame) and the carriageway half-widths left/right; skipped rows
    say why (unknown road, too near an end, overlapping, a non-flat road)."""
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    m = re.search(r'<OpenDRIVE[\s>]', text)
    if not m:
        raise ValueError('Not an OpenDRIVE document')
    prolog = text[:m.start()]
    root = ET.fromstring(text, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    roads = {r.get('id'): r for r in root.findall('road')}
    by_road: dict[str, list] = {}
    skipped = []
    for p in placements:
        kind = p['kind']
        if kind not in PROFILES:
            raise ValueError(f'unknown road feature {kind!r} (known: {", ".join(PROFILES)})')
        el = roads.get(str(p['road']))
        if el is None:
            skipped.append(dict(p, why='no such road'))
            continue
        L = float(PROFILES[kind]['length_m'])
        length = float(el.get('length', '0'))
        s = float(p['s'])
        if s - L / 2.0 < END_MARGIN_M or s + L / 2.0 > length - END_MARGIN_M:
            skipped.append(dict(p, why=f'the {L:g} m profile does not fit on road {p["road"]} ({length:.2f} m) at '
                                       f's = {s:.2f} m with {END_MARGIN_M:g} m to its ends'))
            continue
        if not flat_profile(el):
            skipped.append(dict(p, why=f'road {p["road"]} already has a non-flat elevation profile'))
            continue
        lst = by_road.setdefault(str(p['road']), [])
        if any(abs(q['s'] - s) < (L + float(PROFILES[q['kind']]['length_m'])) / 2.0 + 0.1 for q in lst):
            skipped.append(dict(p, why=f'overlaps another feature on road {p["road"]}'))
            continue
        lst.append(dict(p, s=s))
    placed = []
    for rid, feats in by_road.items():
        el = roads[rid]
        for old in el.findall('elevationProfile'):
            el.remove(old)
        ep = ET.Element('elevationProfile')
        for s0, a, b, c, d in profile_entries(feats):
            ET.SubElement(ep, 'elevation', {'s': _fmt(s0), 'a': _fmt(a), 'b': _fmt(b), 'c': _fmt(c), 'd': _fmt(d)})
        kids = list(el)
        pv = next((i for i, k in enumerate(kids) if k.tag == 'planView'), len(kids) - 1)
        el.insert(pv + 1, ep)
        for f in sorted(feats, key=lambda f: f['s']):
            x, y, h, wl, wr = _road_xy(el, f['s'])
            pr = PROFILES[f['kind']]
            row = {'road': rid, 's': round(f['s'], 4), 'kind': f['kind'], 'name': pr['name'],
                   'length_m': pr['length_m'], 'height_m': pr['height_m'], 'crossing_kmh': pr['crossing_kmh'],
                   'x': round(x, 4), 'y': round(y, 4), 'hdg': round(h, 6), 'half_width_left_m': round(wl, 3),
                   'half_width_right_m': round(wr, 3), 'source': f.get('source', '')}
            placed.append(row)
            ET.SubElement(el, 'userData', {'code': USERDATA_CODE, 'value': encode_userdata(row)})
    ET.indent(root, space='    ')
    body = ET.tostring(root, encoding='unicode')
    body = re.sub(r'<geoReference>(.*?)</geoReference>',
                  lambda mm: '<geoReference><![CDATA[' + _unescape(mm.group(1)) + ']]></geoReference>'
                  if ']]>' not in _unescape(mm.group(1)) else mm.group(0), body, count=1, flags=re.S)
    if not prolog.strip():
        prolog = '<?xml version="1.0" encoding="UTF-8"?>\n'
    return prolog + body + '\n', placed, skipped


def _unescape(s: str) -> str:
    import html
    return html.unescape(s)


USERDATA_KEYS = ('kind', 's', 'length_m', 'height_m', 'crossing_kmh', 'x', 'y', 'hdg', 'half_width_left_m',
                 'half_width_right_m')


def encode_userdata(row: dict) -> str:
    return ';'.join(f'{k}={row[k]}' for k in USERDATA_KEYS if row.get(k) is not None)


def read(text: str | bytes) -> list:
    """The features a file carries (its falcon:road_feature userData): [{'road', 'kind', 's', 'length_m', 'height_m',
    'crossing_kmh', 'x', 'y', 'hdg', 'half_width_left_m', 'half_width_right_m'}] (OpenDRIVE's frame)."""
    root = ET.fromstring(text if isinstance(text, (bytes, bytearray)) else text.encode('utf-8'))
    out = []
    for road in root.findall('road'):
        for ud in road.findall('userData'):
            if ud.get('code') != USERDATA_CODE:
                continue
            row = {'road': road.get('id')}
            for part in (ud.get('value') or '').split(';'):
                k, _, v = part.partition('=')
                if not k:
                    continue
                if k == 'kind':
                    row[k] = v
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
            out.append(row)
    return out


def strip(text: str | bytes) -> str:
    """The document without any feature this module wrote: flat elevation and no falcon:road_feature userData on the
    roads that had them (used when features are re-applied to a file that has them, without its base)."""
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    root = ET.fromstring(text)
    if not any(ud.get('code') == USERDATA_CODE for ud in root.iter('userData')):
        return text
    feats = {r['road'] for r in read(text)}
    for road in root.findall('road'):
        if road.get('id') not in feats:
            continue
        for ud in list(road.findall('userData')):
            if ud.get('code') == USERDATA_CODE:
                road.remove(ud)
        for ep in road.findall('elevationProfile'):
            for e in list(ep):
                ep.remove(e)
            ET.SubElement(ep, 'elevation', {'s': '0', 'a': '0', 'b': '0', 'c': '0', 'd': '0'})
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding='unicode') + '\n'


# ------------------------------------------------------------------------------------------------ placing
def _ref_lines(od, step: float = 0.5) -> list:
    from . import xodr as X
    out = []
    for r in od.roads:
        if r.in_junction:
            continue
        pts = X.sample_reference_line(r, step)
        if len(pts) < 2:
            continue
        seg = np.hypot(*np.diff(pts[:, :2], axis=0).T)
        out.append((r, pts, np.r_[0.0, np.cumsum(seg)]))
    return out


def snap(od, points: list, tol: float = PLACE_TOL_M) -> tuple[list, list]:
    """Points [{'x', 'y' (OpenDRIVE frame), 'kind', 'source'}] onto the non-junction roads passing within tol:
    (placements [{'road', 's', 'kind', 'source'}], misses [{…, 'why'}])."""
    lines = _ref_lines(od)
    out, miss = [], []
    for p in points:
        hit = False
        for r, pts, s in lines:
            d = np.hypot(pts[:, 0] - p['x'], pts[:, 1] - p['y'])
            k = int(np.argmin(d))
            if d[k] <= tol:
                # the exact s: the nearest point's s refined by the projection on its segment
                s_k = float(s[k] * r.length / s[-1]) if s[-1] > 0 else 0.0
                out.append({'road': r.id, 's': s_k, 'kind': p['kind'], 'source': p.get('source', '')})
                hit = True
        if not hit:
            miss.append(dict(p, why=f'no non-junction road within {tol:g} m (in a junction, or a way type not '
                                    'converted)'))
    return out, miss


def osm_points(data, proj, bump_kind: str = 'osm_bump') -> tuple[list, dict]:
    """traffic_calming nodes on the area's highways as points in OpenDRIVE's frame (proj: core.osm.local_projection
    of the import's centre, so x = east, y = north), with the feature kind; and counts of the values not represented.
    A tagged highway *way* contributes its middle node."""
    from . import osm as O
    on_road = {n for w in data.ways if w['kind'] == 'highway' and w['tags'].get('highway') in O.CARLA_HIGHWAYS
               for n in w['nodes']}
    pts, other = [], {}

    def add(nid, val, src):
        kind = bump_kind if val == 'bump' else OSM_KINDS.get(val)
        if kind is None:
            other[val] = other.get(val, 0) + 1
            return
        la, lo = data.nodes[nid]
        x, y = proj.to_local(la, lo) if hasattr(proj, 'to_local') else proj(la, lo)
        pts.append({'x': float(x), 'y': float(y), 'kind': kind, 'source': src, 'osm_value': val})
    for nid, tags in sorted(data.node_tags.items()):
        val = tags.get('traffic_calming')
        if val and nid in on_road:
            add(nid, val, f'osm node {nid} traffic_calming={val}')
    for w in data.ways:
        val = w['tags'].get('traffic_calming')
        if w['kind'] == 'highway' and val:
            add(w['nodes'][len(w['nodes']) // 2], val, f'osm way {w["id"]} traffic_calming={val}')
    return pts, other


def spaced(od, kind: str = 'irc_hump', spacing_m: float = 110.0, min_road_m: float = 60.0, offset_m: float = 0.0,
           roads: list | None = None) -> list:
    """The user's choice: a feature every spacing_m along every non-junction road at least min_road_m long (roads:
    only these ids), the first at offset_m + spacing_m / 2 from the road's start (110 m: IRC:99-1988 §5.2's 100–120
    m between repeated humps). Placements as snap() gives."""
    L = float(PROFILES[kind]['length_m'])
    out = []
    for r in od.roads:
        if r.in_junction or r.length < min_road_m or (roads is not None and r.id not in set(map(str, roads))):
            continue
        s = offset_m + spacing_m / 2.0
        while s + L / 2.0 <= r.length - END_MARGIN_M:
            if s - L / 2.0 >= END_MARGIN_M:
                out.append({'road': r.id, 's': s, 'kind': kind, 'source': f'every {spacing_m:g} m (user choice)'})
            s += spacing_m
    return out


# ------------------------------------------------------------------------------------------------ a world
def world_paths(world_dir: str) -> dict:
    return {'xodr': os.path.join(world_dir, 'world.xodr'), 'base': os.path.join(world_dir, 'world.base.xodr'),
            'features': os.path.join(world_dir, 'features.json'), 'osm': os.path.join(world_dir, 'area.osm'),
            'import': os.path.join(world_dir, 'import.json')}


def apply_to_world(world_dir: str, opts: dict) -> dict:
    """Write an imported world's features: world.xodr from world.base.xodr (made from world.xodr the first time) with
    the breakers opts asks for, and features.json. opts: {'osm': bool (traffic_calming tags), 'osm_bump': 'osm_bump'
    | 'irc_hump' (what traffic_calming=bump becomes), 'spacing_m': float or 0 (the user's spacing), 'spaced_kind':
    kind, 'min_road_m'}. Returns the features.json document."""
    from . import osm as O
    from . import xodr as X
    P = world_paths(world_dir)
    if not os.path.isfile(P['base']):
        if not os.path.isfile(P['xodr']):
            raise FileNotFoundError(f'{P["xodr"]}: the world has no OpenDRIVE file')
        with open(P['xodr'], encoding='utf-8') as fh:
            cur = fh.read()
        with open(P['base'], 'w', encoding='utf-8') as fh:
            fh.write(strip(cur))
    with open(P['base'], encoding='utf-8') as fh:
        base = fh.read()
    od = X.parse(base)
    placements, misses, other, notes = [], [], {}, []
    if opts.get('osm', True) and os.path.isfile(P['osm']):
        data = O.read_osm(P['osm'])
        centre = None
        if os.path.isfile(P['import']):
            with open(P['import'], encoding='utf-8') as fh:
                centre = json.load(fh).get('centre')
        if not centre:
            centre = O.summary(data)['centre']
        proj = O.local_projection(*centre)
        pts, other = osm_points(data, proj, opts.get('osm_bump', 'osm_bump'))
        got, miss = snap(od, pts)
        placements += got
        misses += miss
        if not pts:
            notes.append('the area\'s OpenStreetMap data has no traffic_calming bump, hump, table or yes on its roads')
    elif opts.get('osm', True):
        notes.append('no area.osm in the world\'s folder: OpenStreetMap features not placed')
    if float(opts.get('spacing_m') or 0) > 0:
        placements += spaced(od, opts.get('spaced_kind', 'irc_hump'), float(opts['spacing_m']),
                             float(opts.get('min_road_m', 60.0)))
    text, placed, skipped = apply(base, placements)
    with open(P['xodr'] + '.tmp', 'w', encoding='utf-8') as fh:
        fh.write(text)
    os.replace(P['xodr'] + '.tmp', P['xodr'])
    by_kind = {}
    for p in placed:
        by_kind[p['kind']] = by_kind.get(p['kind'], 0) + 1
    doc = {'schema': SCHEMA, 'options': dict(opts), 'placed': placed, 'skipped': skipped + misses,
           'not_represented': other, 'counts': by_kind, 'notes': notes,
           'profiles': {k: PROFILES[k] for k in sorted({p['kind'] for p in placed})},
           'mesh_note': MESH_NOTE}
    with open(P['features'], 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1)
    return doc


def remove_from_world(world_dir: str) -> bool:
    """Put the world's file back as it was before features (world.base.xodr); False when it never had any."""
    P = world_paths(world_dir)
    if not os.path.isfile(P['base']):
        return False
    shutil.copyfile(P['base'], P['xodr'])
    if os.path.isfile(P['features']):
        os.remove(P['features'])
    return True


# what the mesh renders (measured: docs/indian-assets.md); a world with features is generated at this vertex distance
FEATURE_VERTEX_DISTANCE_M = 0.25
MESH_NOTE = ('CARLA builds the road mesh with vertices every `vertex_distance` metres along the road: a profile '
             'shorter than a few vertex distances is not rendered as written. A world with features is generated '
             f'at {FEATURE_VERTEX_DISTANCE_M} m (docs/indian-assets.md: what was measured at each distance).')


def mesh_for(mesh: dict | None, has_features: bool) -> dict:
    """The world's mesh settings with the vertex distance a world with features needs."""
    m = dict(mesh or {})
    if has_features:
        m['vertex_distance'] = min(float(m.get('vertex_distance', 2.0)), FEATURE_VERTEX_DISTANCE_M)
    return m


# ------------------------------------------------------------------------------------------------ a test road
def test_road(length_m: float = 360.0, features: list | None = None, lane_width: float = 3.5, rule: str = 'RHT',
              shape: dict | None = None) -> str:
    """A straight two-lane road along +x (one lane each way, sidewalks), with features [{'kind', 's'}] and optionally
    a lateralProfile <shape> {'s', 't', 'a', 'b', 'c', 'd'} (to check what CARLA renders): the verification road the
    assets worker drives over."""
    ent = profile_entries(features or [])
    elev = '\n'.join(f'            <elevation s="{_fmt(s)}" a="{_fmt(a)}" b="{_fmt(b)}" c="{_fmt(c)}" d="{_fmt(d)}"/>'
                     for s, a, b, c, d in ent)
    shp = ''
    if shape:
        shp = (f'        <lateralProfile>\n            <shape s="{_fmt(shape["s"])}" t="{_fmt(shape["t"])}" '
               f'a="{_fmt(shape["a"])}" b="{_fmt(shape["b"])}" c="{_fmt(shape["c"])}" d="{_fmt(shape["d"])}"/>\n'
               '        </lateralProfile>\n')
    lane = ('                    <lane id="{id}" type="{ty}" level="false">\n                        <link/>\n'
            '                        <width sOffset="0" a="{w}" b="0" c="0" d="0"/>\n'
            '                        <roadMark sOffset="0" type="{rm}" weight="standard" color="standard" '
            'width="0.13"/>\n                    </lane>\n')
    left = lane.format(id=2, ty='sidewalk', w=2.0, rm='solid') + lane.format(id=1, ty='driving', w=lane_width,
                                                                              rm='solid')
    right = lane.format(id=-1, ty='driving', w=lane_width, rm='solid') + lane.format(id=-2, ty='sidewalk', w=2.0,
                                                                                   rm='solid')
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<OpenDRIVE>
    <header revMajor="1" revMinor="4" name="falcon-feature-test" version="1" north="0" south="0" east="0" west="0"/>
    <road name="test" length="{_fmt(length_m)}" id="1" junction="-1" rule="{rule}">
        <link/>
        <type s="0" type="town"><speed max="50" unit="km/h"/></type>
        <planView>
            <geometry s="0" x="0" y="0" hdg="0" length="{_fmt(length_m)}"><line/></geometry>
        </planView>
        <elevationProfile>
{elev}
        </elevationProfile>
{shp}        <lanes>
            <laneSection s="0">
                <left>
{left}                </left>
                <center>
                    <lane id="0" type="none" level="false"><link/><roadMark sOffset="0" type="broken" '''\
        f'''weight="standard" color="standard" width="0.13"/></lane>
                </center>
                <right>
{right}                </right>
            </laneSection>
        </lanes>
    </road>
</OpenDRIVE>
'''
