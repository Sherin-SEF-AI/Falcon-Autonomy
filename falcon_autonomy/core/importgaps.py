"""What an OpenStreetMap import cannot represent, counted from the import (standard library + numpy).

compute(data, way_types, centre, lane_width) -> {'schema', 'items': [...]} is recorded by workers.osm_import in the
world's import.json ('import_gaps') from 4.2 on; for a world imported earlier the Worlds inspector computes it from
the world's stored area.osm (marked 'computed later'). Each item is {'key', 'title', 'count', 'handled', 'tip',
'detail'}; count None means "not counted" (the tip says why). with_city() and with_features() add what the city
stand-ins and the speed breakers left out, from their own reports (they can be rebuilt after the import).

The items and how each is counted:

* elevation — the converted roads tagged bridge (other than "no"), tunnel (other than "no") or layer other than 0.
  An OpenDRIVE world from carla.Osm2Odr has no terrain and Falcon leaves Osm2OdrSettings.elevation_layer_height at
  its default, 0.0 m (CARLA's PythonAPI/docs/osm2odr.yml), so every such road is flattened onto the ground plane: a
  bridge and the road under it lie at the same height and overlap without a junction. ele=* tags are not read.
* buildings_on_road — building footprints that a converted road's surface reaches: the road taken as a band of
  half-width lanes × lane width / 2 around its OSM centreline (lanes from the tags as core.osm.expected_lanes reads
  them, else 1 lane for a one-way and 2 for a two-way road). An estimate: the converter's own widths and offsets
  differ by up to a lane. Usually arcades, gates, roofs over fuel stations or mapping errors.
* roads_dropped — highway ways whose type the converter does not convert (footways, service roads, tracks, paths …),
  by type: vehicles in the world cannot use them.
* relations — OSM relations skipped when reading (turn restrictions, multipolygon buildings): core.osm reads none.
* cut_ways — node references missing from the file (ways cut at the extract's edge), dropped when reading.
* buildings_not_placed / props_not_placed (with_city) — footprints and point features the city stand-ins left out,
  with the reasons core.cityprops gives; without a city every building is absent.
* speed_breakers_skipped (with_features) — traffic_calming features not placed (core.roadfeatures' skipped and
  not-represented counts).
"""
from __future__ import annotations

import math

import numpy as np

SCHEMA = 'falcon.import_gaps/1'

TIPS = {
    'elevation': 'The world is flat: roads OpenStreetMap puts on bridges, in tunnels or on another layer are drawn at '
                 'ground level (CARLA\'s converter is left at elevation_layer_height 0.0 m), so a flyover and the road '
                 'under it overlap at the same height with no junction between them. Elevation (ele) tags are not '
                 'read either.',
    'buildings_on_road': 'OSM building footprints that reach a converted road\'s surface (the road as a band of its '
                         'lane count × lane width around the OSM centreline; an estimate). Often arcades, gates, '
                         'canopies or mapping errors. The world has no buildings of its own; the city stand-ins are '
                         'kept off the lanes, so these are left out or moved.',
    'roads_dropped': 'Highway ways of types the converter does not turn into roads (footways, service roads, tracks, '
                     'paths …): nothing in the world drives on them.',
    'relations': 'OSM relations are not read: turn restrictions are unknown to the converted network (every turn a '
                 'junction\'s geometry allows is allowed) and multipolygon buildings are missing.',
    'cut_ways': 'Node references missing from the file — ways cut at the extract\'s edge — were dropped when reading.',
    'buildings_not_placed': 'OSM building footprints with no stand-in in the world (City around the roads): no town '
                            'mesh fits, it would cover a lane or sidewalk, or overlap another building.',
    'props_not_placed': 'Trees, lamps, bus stops and street furniture from OSM the city left out, with the reason.',
    'speed_breakers_skipped': 'traffic_calming features (bump, hump, table …) not written into the road: in a '
                              'junction, near a road\'s end, overlapping another, or of a kind not expressible as a '
                              'road elevation profile.',
}
TITLES = {'elevation': 'Flat terrain', 'buildings_on_road': 'Buildings on the road', 'roads_dropped': 'Roads dropped',
          'relations': 'Relations not read', 'cut_ways': 'Ways cut at the edge',
          'buildings_not_placed': 'Buildings not placed', 'props_not_placed': 'Props not placed',
          'speed_breakers_skipped': 'Speed breakers skipped'}


def _item(key, count, handled, detail=None) -> dict:
    return {'key': key, 'title': TITLES[key], 'count': count, 'handled': handled, 'tip': TIPS[key],
            'detail': detail or {}}


def _yes(v) -> bool:
    return v not in (None, '', 'no', '0', 'false')


def _layer(tags) -> float:
    try:
        return float(str(tags.get('layer', '0')).split(';')[0])
    except ValueError:
        return 0.0


def _half_width(tags, lane_width: float) -> float:
    from .osm import expected_lanes, oneway
    lanes = expected_lanes(tags)
    n = sum(lanes) if lanes else (1 if oneway(tags) in (1, -1) else 2)
    return max(n, 1) * lane_width / 2.0


def _seg_dist(p, q, a, b) -> float:
    """Distance between segments pq and ab (0 when they cross)."""
    def cross(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
    d1, d2, d3, d4 = cross(a, b, p), cross(a, b, q), cross(p, q, a), cross(p, q, b)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return 0.0

    def pt(c, u, v):
        ux, uy = v[0] - u[0], v[1] - u[1]
        L = ux * ux + uy * uy
        t = 0.0 if L == 0 else max(0.0, min(1.0, ((c[0] - u[0]) * ux + (c[1] - u[1]) * uy) / L))
        return math.hypot(c[0] - u[0] - t * ux, c[1] - u[1] - t * uy)
    return min(pt(p, a, b), pt(q, a, b), pt(a, p, q), pt(b, p, q))


def _inside(pt, poly) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def buildings_on_roads(data, way_types, proj, lane_width: float = 3.35, cell: float = 30.0) -> dict:
    """Count building footprints within the band of a converted road (module doc): {'count', 'raised' (of them,
    tagged layer > 0 or building=roof), 'ids' (the first 20)}."""
    def xy(refs):
        lat = np.array([data.nodes[r][0] for r in refs])
        lon = np.array([data.nodes[r][1] for r in refs])
        x, y = proj.to_local(lat, lon)
        return list(zip(np.atleast_1d(x).tolist(), np.atleast_1d(y).tolist()))

    grid: dict = {}
    segs = []
    for w in data.ways:
        if w.get('kind') != 'highway' or w['tags'].get('highway') not in way_types:
            continue
        refs = [r for r in w['nodes'] if r in data.nodes]
        if len(refs) < 2:
            continue
        hw = _half_width(w['tags'], lane_width)
        pts = xy(refs)
        for a, b in zip(pts, pts[1:]):
            k = len(segs)
            segs.append((a, b, hw))
            for cx in range(int(math.floor((min(a[0], b[0]) - hw) / cell)),
                            int(math.floor((max(a[0], b[0]) + hw) / cell)) + 1):
                for cy in range(int(math.floor((min(a[1], b[1]) - hw) / cell)),
                                int(math.floor((max(a[1], b[1]) + hw) / cell)) + 1):
                    grid.setdefault((cx, cy), []).append(k)
    count, raised, ids = 0, 0, []
    for w in data.ways:
        if w.get('kind') != 'building':
            continue
        refs = [r for r in w['nodes'] if r in data.nodes]
        if len(refs) < 3:
            continue
        poly = xy(refs)
        if poly[0] == poly[-1]:
            poly = poly[:-1]
        if len(poly) < 3:
            continue
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        cand = set()
        for cx in range(int(math.floor(min(xs) / cell)), int(math.floor(max(xs) / cell)) + 1):
            for cy in range(int(math.floor(min(ys) / cell)), int(math.floor(max(ys) / cell)) + 1):
                cand.update(grid.get((cx, cy), ()))
        hit = False
        for k in cand:
            a, b, hw = segs[k]
            if _inside(a, poly) or _inside(b, poly):
                hit = True
            else:
                hit = any(_seg_dist(a, b, poly[i], poly[(i + 1) % len(poly)]) < hw for i in range(len(poly)))
            if hit:
                break
        if hit:
            count += 1
            if _layer(w['tags']) > 0 or w['tags'].get('building') == 'roof':
                raised += 1
            if len(ids) < 20:
                ids.append(w['id'])
    return {'count': count, 'raised': raised, 'ids': ids}


def compute(data, way_types=None, centre=None, lane_width: float = 3.35) -> dict:
    """The import's gaps from what it read (module doc). data: core.osm.OsmData as converted (after map edits);
    way_types: the converted highway types (default the converter's); centre: (lat, lon) of the projection."""
    from . import osm as O
    types = tuple(way_types or O.CARLA_HIGHWAYS)
    roads = [w for w in data.ways if w.get('kind') == 'highway' and w['tags'].get('highway') in types]
    bridge = sum(_yes(w['tags'].get('bridge')) for w in roads)
    tunnel = sum(_yes(w['tags'].get('tunnel')) for w in roads)
    layered = [w for w in roads if _yes(w['tags'].get('bridge')) or _yes(w['tags'].get('tunnel'))
               or _layer(w['tags']) != 0]
    items = [_item('elevation', len(layered), 'flattened to ground level (elevation_layer_height 0.0 m)',
                   {'bridge': bridge, 'tunnel': tunnel,
                    'layer_only': sum(1 for w in layered if not _yes(w['tags'].get('bridge'))
                                      and not _yes(w['tags'].get('tunnel'))),
                    'ele_tagged_nodes': sum(1 for t in data.node_tags.values() if 'ele' in t)})]
    if centre is None:
        b = data.bounds
        centre = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) if b else (0.0, 0.0)
    if data.buildings:
        on = buildings_on_roads(data, types, O.local_projection(*centre), lane_width)
        items.append(_item('buildings_on_road', on['count'], 'not in the world; city stand-ins kept off the lanes',
                           {'raised_or_roof': on['raised'], 'of_buildings': len(data.buildings),
                            'osm_ids': on['ids']}))
    else:
        items.append(_item('buildings_on_road', None, 'no building footprints in the extract',
                           {'of_buildings': 0}))
    dropped: dict = {}
    for w in data.ways:
        if w.get('kind') == 'highway' and w['tags'].get('highway') not in types:
            t = w['tags'].get('highway') or '?'
            dropped[t] = dropped.get(t, 0) + 1
    items.append(_item('roads_dropped', sum(dropped.values()), 'not converted (their type is not among the '
                                                                 'converter\'s)',
                       {'by_type': dict(sorted(dropped.items(), key=lambda kv: (-kv[1], kv[0])))}))
    rel = data.counts.get('relations_skipped')
    items.append(_item('relations', rel if rel is not None else None, 'skipped when reading'))
    cut = int(data.counts.get('missing_node_refs') or 0)
    items.append(_item('cut_ways', cut, 'dropped when reading'))
    return {'schema': SCHEMA, 'items': items, 'lane_width_m': float(lane_width), 'way_types': list(types),
            'method': 'counted from the OSM data as converted (core.importgaps)'}


def with_city(gaps: dict, city: dict | None, buildings_in_osm: int | None = None) -> dict:
    """gaps plus what the city stand-ins left out (city: a world's 'city' entry, core.cityprops.build's summary)."""
    out = dict(gaps, items=[i for i in gaps.get('items') or [] if i['key'] not in ('buildings_not_placed',
                                                                                    'props_not_placed')])
    c = city or {}
    s = c.get('summary') or {}
    if not c.get('plan'):
        n = buildings_in_osm
        out['items'].append(_item('buildings_not_placed', n, 'absent: no city built (Worlds ▸ City around the '
                                                             'roads)'))
        return out
    sk = s.get('skipped') or {}
    b = sk.get('building') or {}
    placed = sum(((s.get('by_kind') or {}).get('building') or {}).values())
    fb = (c.get('features') or {}).get('buildings')
    n = sum(b.values()) if b else (max(fb - placed, 0) if fb is not None else 0)
    out['items'].append(_item('buildings_not_placed', n, 'left out of the city', {'why': dict(b), 'placed': placed,
                                                                                    'footprints': fb}))
    props = {k: v for k, v in sk.items() if k != 'building'}
    out['items'].append(_item('props_not_placed', sum(sum(v.values()) for v in props.values()),
                              'left out of the city', {'why': props}))
    return out


def with_features(gaps: dict, features: dict | None) -> dict:
    """gaps plus the speed breakers not placed (features: a world's features.json, core.roadfeatures)."""
    out = dict(gaps, items=[i for i in gaps.get('items') or [] if i['key'] != 'speed_breakers_skipped'])
    if not features:
        return out
    sk = features.get('skipped') or []
    nr = features.get('not_represented') or {}
    n = (len(sk) if isinstance(sk, list) else int(sk or 0)) + sum(int(v or 0) for v in nr.values())
    out['items'].append(_item('speed_breakers_skipped', n, 'not written into the road',
                              {'skipped': len(sk) if isinstance(sk, list) else sk, 'not_represented': dict(nr)}))
    return out


def detail_text(item: dict) -> str:
    """One line of an item's breakdown, for its tooltip."""
    d = item.get('detail') or {}
    k = item['key']
    if k == 'elevation':
        return (f'{d.get("bridge", 0)} bridge, {d.get("tunnel", 0)} tunnel, {d.get("layer_only", 0)} on another layer '
                f'only; {d.get("ele_tagged_nodes", 0)} nodes with an ele tag (not read)')
    if k == 'buildings_on_road' and d.get('of_buildings'):
        return f'of {d["of_buildings"]} footprints; {d.get("raised_or_roof", 0)} tagged roof or layer > 0'
    if k == 'roads_dropped' and d.get('by_type'):
        return ', '.join(f'{v} {t}' for t, v in list(d['by_type'].items())[:8])
    if k in ('buildings_not_placed', 'props_not_placed') and d.get('why'):
        w = d['why']
        if k == 'props_not_placed':
            return '; '.join(f'{kind}: ' + ', '.join(f'{n} {why}' for why, n in sorted(v.items()))
                             for kind, v in sorted(w.items()))
        return ', '.join(f'{n} {why}' for why, n in sorted(w.items()))
    if k == 'speed_breakers_skipped':
        nr = d.get('not_represented') or {}
        return f'{d.get("skipped", 0)} skipped' + (', not representable: ' + ', '.join(
            f'{v} {t}' for t, v in sorted(nr.items())) if nr else '')
    return ''
