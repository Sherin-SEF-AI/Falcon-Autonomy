"""Map edits for an imported world (standard library + numpy): an ordered list of changes to the OpenStreetMap data the
world is converted from, applied deterministically before the conversion, so a re-import keeps them.

An imported world is OSM ways converted by carla.Osm2Odr (SUMO's netconvert). The editor changes what the converter
reads — the way's tags, the way itself, a node's tags — rather than the OpenDRIVE it writes: lanes, junctions, lane
links, signals and the road-feature profiles are then all made by the same pipeline as any import, and the lane check
measures the result. The world entry keeps the list in osm.edits; import.json records what was applied, what was
skipped (a way the area no longer has) and the list's SHA-256.

Edits (each a dict with "op"):

| op | fields | what changes in OSM |
|---|---|---|
| lanes | way, forward, backward | lanes (and lanes:forward / lanes:backward on a two-way way) |
| maxspeed | way, kmh (None: the converter's default for the road type) | maxspeed |
| oneway | way, value: "no", "yes" or "-1" | oneway; lanes re-tagged per direction |
| highway | way, value (one of the converted types) | highway (the road type: the converter's default lanes and speed) |
| remove | way | the way is left out |
| speed_breaker | way, lat, lon, value: bump, hump or table | a node on the way at that point, tagged traffic_calming |
| remove_feature | node | traffic_calming taken off an OSM node |
| signal | node, on | highway=traffic_signals on a junction node (added or taken off) |

What the converter does with each was measured with CARLA 0.9.16's carla.Osm2Odr (docs/worlds.md, *Editing the map*).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os

from . import osm as O

SCHEMA = 'falcon.map_edits/1'
OPS = ('lanes', 'maxspeed', 'oneway', 'highway', 'remove', 'speed_breaker', 'remove_feature', 'signal')
WAY_OPS = ('lanes', 'maxspeed', 'oneway', 'highway', 'remove')
BREAKERS = ('bump', 'hump', 'table')
MAX_LANES = 6
SPEED_KMH = (5, 150)
ON_WAY_TOL_M = 15.0                    # a speed breaker's point must lie this close to its way

# what CARLA does with each edit (measured on 0.9.16: docs/worlds.md, Editing the map)
HONOURED = {
    'lanes': 'geometry: the converter builds the lanes; CARLA drives them',
    'oneway': 'geometry: the converter builds one direction; CARLA drives it',
    'highway': 'the converter\'s default lanes and speed for the type (when the tags do not say)',
    'remove': 'the road and its junction connections are gone; a dead end may remain where it joined',
    'maxspeed': 'written as the road\'s speed record (<type><speed>); CARLA\'s traffic manager does not read it — '
                'traffic keeps the vehicle\'s 30 km/h limit (measured)',
    'speed_breaker': 'an elevation profile the vehicle physics drives over (core.roadfeatures)',
    'remove_feature': 'the elevation profile is not written',
    'signal': 'the converter writes a traffic-light signal per approach; CARLA makes traffic lights that control '
              'their lanes (left-hand worlds: converted by this version or later)',
}


def edits_hash(edits: list) -> str:
    """SHA-256 of the list's canonical JSON (sorted keys, no spaces): the world's provenance names it."""
    return hashlib.sha256(json.dumps(edits or [], sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _key(e: dict):
    if e['op'] in WAY_OPS:
        return e['op'], int(e['way'])
    if e['op'] in ('signal', 'remove_feature'):
        return e['op'], int(e['node'])
    return None                                  # speed breakers accumulate


def upsert(edits: list, e: dict) -> list:
    """A new list with e: replacing the edit of the same op on the same way or node (one lanes edit per way, …), else
    appended. Removing a way drops the other edits of that way (they would not apply)."""
    out = [dict(x) for x in edits or []]
    k = _key(e)
    if e['op'] == 'remove':
        on_way = WAY_OPS + ('speed_breaker',)
        out = [x for x in out if not (x['op'] in on_way and int(x.get('way', -1)) == int(e['way']))]
    if k is not None:
        for i, x in enumerate(out):
            if _key(x) == k:
                out[i] = dict(e)
                return out
    out.append(dict(e))
    return out


def describe(e: dict, data: O.OsmData | None = None) -> str:
    """One line for the edit list."""
    def way(w):
        name = ''
        if data is not None:
            ww = way_by_id(data, w)
            name = (ww or {}).get('tags', {}).get('name', '') if ww else ''
        return f'way {w}' + (f' ({name})' if name else '')
    op = e.get('op')
    if op == 'lanes':
        return f'{way(e["way"])}: {e["forward"]} lane(s) forward, {e["backward"]} backward'
    if op == 'maxspeed':
        return f'{way(e["way"])}: speed limit ' + (f'{e["kmh"]:g} km/h' if e.get('kmh') else 'the type\'s default')
    if op == 'oneway':
        return f'{way(e["way"])}: ' + {'no': 'two-way', 'yes': 'one-way along the way',
                                       '-1': 'one-way against the way'}.get(str(e['value']), str(e['value']))
    if op == 'highway':
        return f'{way(e["way"])}: road type {e["value"]}'
    if op == 'remove':
        return f'{way(e["way"])}: removed'
    if op == 'speed_breaker':
        return f'{way(e["way"])}: speed breaker ({e["value"]}) at {e["lat"]:.6f}, {e["lon"]:.6f}'
    if op == 'remove_feature':
        return f'node {e["node"]}: traffic calming removed'
    if op == 'signal':
        return f'node {e["node"]}: traffic signals ' + ('added' if e.get('on', True) else 'removed')
    return json.dumps(e)


def way_by_id(data: O.OsmData, way_id) -> dict | None:
    wid = int(way_id)
    return next((w for w in data.ways if int(w['id']) == wid), None)


def junction_nodes(data: O.OsmData, types=O.CARLA_HIGHWAYS) -> dict:
    """{node id: [way ids]} for the nodes two or more converted highway ways share (or one way passes twice)."""
    seen: dict = {}
    for w in data.ways:
        if w['kind'] != 'highway' or w['tags'].get('highway') not in types:
            continue
        for n in set(w['nodes']):
            seen.setdefault(n, []).append(int(w['id']))
    return {n: ws for n, ws in seen.items() if len(ws) >= 2}


def components(data: O.OsmData, types=O.CARLA_HIGHWAYS) -> list:
    """The converted road network's connected parts: lists of way ids (ways sharing a node are connected), largest
    first."""
    ways = [w for w in data.ways if w['kind'] == 'highway' and w['tags'].get('highway') in types]
    parent = list(range(len(ways)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    by_node: dict = {}
    for i, w in enumerate(ways):
        for n in w['nodes']:
            j = by_node.setdefault(n, i)
            a, b = find(i), find(j)
            if a != b:
                parent[a] = b
    groups: dict = {}
    for i, w in enumerate(ways):
        groups.setdefault(find(i), []).append(int(w['id']))
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), g[0]))


def _local(lat0):
    kx = 111319.49 * math.cos(math.radians(lat0))
    return kx, 110574.0


def _project_on_way(data: O.OsmData, way: dict, lat: float, lon: float):
    """The nearest point of the way to (lat, lon): (index of the segment's first node, lat, lon, distance m)."""
    kx, ky = _local(lat)
    best = None
    for i, (a, b) in enumerate(zip(way['nodes'], way['nodes'][1:])):
        (la, lo), (lb, lob) = data.nodes[a], data.nodes[b]
        ax, ay = (lo - lon) * kx, (la - lat) * ky
        bx, by = (lob - lon) * kx, (lb - lat) * ky
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, -(ax * dx + ay * dy) / L2))
        px, py = ax + t * dx, ay + t * dy
        d = math.hypot(px, py)
        if best is None or d < best[3]:
            best = (i, lat + py / ky, lon + px / kx, d)
    return best


def check_edit(data: O.OsmData, e: dict, types=O.CARLA_HIGHWAYS) -> list:
    """Why an edit cannot be made (an empty list: it can). Checked against the data as it stands before the edit."""
    op = e.get('op')
    if op not in OPS:
        return [f'unknown edit {op!r}']
    errs = []
    if op in WAY_OPS + ('speed_breaker',):
        w = way_by_id(data, e.get('way', -1))
        if w is None:
            return [f'way {e.get("way")} is not in the area']
        if op == 'lanes':
            f, b = int(e.get('forward', 0)), int(e.get('backward', 0))
            ow = O.oneway(w['tags'])
            if ow == 1 and (f < 1 or b != 0):
                errs.append('a one-way road (along the way) needs at least one lane forward and none backward')
            elif ow == -1 and (b < 1 or f != 0):
                errs.append('a one-way road (against the way) needs at least one lane backward and none forward')
            elif ow in (0, None) and (f < 1 or b < 1):
                errs.append('a two-way road needs at least one lane in each direction (make it one-way instead)')
            if max(f, b) > MAX_LANES:
                errs.append(f'at most {MAX_LANES} lanes per direction')
        elif op == 'maxspeed':
            v = e.get('kmh')
            if v is not None and not (SPEED_KMH[0] <= float(v) <= SPEED_KMH[1]):
                errs.append(f'a speed limit between {SPEED_KMH[0]} and {SPEED_KMH[1]} km/h')
        elif op == 'oneway':
            if str(e.get('value')) not in ('no', 'yes', '-1'):
                errs.append('oneway is "no", "yes" or "-1"')
        elif op == 'highway':
            if e.get('value') not in types:
                errs.append(f'{e.get("value")!r} is not a road type this world converts ({", ".join(types)})')
        elif op == 'speed_breaker':
            if e.get('value') not in BREAKERS:
                errs.append(f'a speed breaker is one of {", ".join(BREAKERS)}')
            else:
                hit = _project_on_way(data, w, float(e['lat']), float(e['lon']))
                if hit is None or hit[3] > ON_WAY_TOL_M:
                    errs.append(f'the point is not on the way (more than {ON_WAY_TOL_M:g} m from it)')
    elif op == 'signal':
        n = int(e.get('node', -1))
        if n not in data.nodes:
            return [f'node {n} is not in the area']
        if e.get('on', True) and n not in junction_nodes(data, types):
            errs.append('traffic signals go on a junction (a node two or more roads share)')
    elif op == 'remove_feature':
        n = int(e.get('node', -1))
        if 'traffic_calming' not in (data.node_tags.get(n) or {}):
            errs.append(f'node {n} has no traffic_calming tag')
    return errs


def _lanes_tags(tags: dict, f: int, b: int):
    for k in ('lanes:forward', 'lanes:backward', 'lanes:both_ways'):
        tags.pop(k, None)
    ow = O.oneway(tags)
    if ow == 1:
        tags['lanes'] = str(f)
    elif ow == -1:
        tags['lanes'] = str(b)
    else:
        tags['lanes'] = str(f + b)
        tags['lanes:forward'] = str(f)
        tags['lanes:backward'] = str(b)


def apply_edits(data: O.OsmData, edits: list, types=O.CARLA_HIGHWAYS) -> tuple[O.OsmData, dict]:
    """The data with the edits applied in order (the input is not changed), and a report: applied (index, text),
    skipped (index, text, reasons — an edit whose way or node the area does not have, or that check_edit refuses),
    the list's SHA-256, the network's connected parts before and after. Deterministic: the same data and list give
    the same result (new nodes are numbered after the data's largest id, in the list's order)."""
    nodes = dict(data.nodes)
    node_tags = copy.deepcopy(data.node_tags)
    ways = copy.deepcopy(data.ways)
    cur = O.OsmData(nodes, ways, data.bounds, data.source, node_tags, dict(data.counts), list(data.notes))
    next_id = max([int(n) for n in nodes] + [0]) + 1
    applied, skipped, removed, touched = [], [], set(), set()
    for i, e in enumerate(edits or []):
        errs = check_edit(cur, e, types)
        if errs:
            skipped.append({'index': i, 'edit': e, 'text': describe(e, data), 'reasons': errs})
            continue
        op = e['op']
        if op in WAY_OPS + ('speed_breaker',):
            w = way_by_id(cur, e['way'])
            touched.add(int(w['id']))
            t = w['tags']
            if op == 'lanes':
                _lanes_tags(t, int(e['forward']), int(e['backward']))
            elif op == 'maxspeed':
                if e.get('kmh'):
                    t['maxspeed'] = f'{float(e["kmh"]):g}'
                else:
                    t.pop('maxspeed', None)
            elif op == 'oneway':
                fb = O.expected_lanes(w)
                n = (sum(fb) if fb else None) or O._count_tag(t, 'lanes') or 1
                t['oneway'] = str(e['value'])
                if str(e['value']) == 'no':
                    _lanes_tags(t, max(1, (n + 1) // 2), max(1, n // 2))
                else:
                    _lanes_tags(t, n, n)
            elif op == 'highway':
                t['highway'] = e['value']
            elif op == 'remove':
                cur.ways.remove(w)
                removed.add(int(w['id']))
            elif op == 'speed_breaker':
                k, la, lo, _ = _project_on_way(cur, w, float(e['lat']), float(e['lon']))
                nid = next_id
                next_id += 1
                nodes[nid] = (round(la, 7), round(lo, 7))
                node_tags[nid] = {'traffic_calming': e['value'], 'note': 'falcon map edit'}
                w['nodes'].insert(k + 1, nid)
        elif op == 'signal':
            n = int(e['node'])
            tg = node_tags.setdefault(n, {})
            if e.get('on', True):
                tg['highway'] = 'traffic_signals'
            elif tg.get('highway') == 'traffic_signals':
                del tg['highway']
            if not tg:
                del node_tags[n]
        elif op == 'remove_feature':
            n = int(e['node'])
            node_tags[n].pop('traffic_calming', None)
            if not node_tags[n]:
                del node_tags[n]
        applied.append({'index': i, 'text': describe(e, data)})
    out = O.subset(cur, cur.ways, bounds=data.bounds)
    before, after = components(data, types), components(out, types)
    rep = {'schema': SCHEMA, 'count': len(edits or []), 'sha256': edits_hash(edits), 'applied': applied,
           'skipped': skipped, 'touched_ways': sorted(touched - removed), 'removed_ways': sorted(removed),
           'connectivity': {'parts_before': len(before), 'parts_after': len(after),
                            'cut_off': _cut_off(before, after)}}
    return out, rep


def _cut_off(before: list, after: list) -> list:
    """Ways the edits separated from the network's main part (they were in the largest part before, not after)."""
    if not before or not after:
        return []
    main_before, main_after = set(before[0]), set(after[0])
    return sorted(w for g in after[1:] for w in g if w in main_before and w not in main_after)


def check_list(data: O.OsmData, edits: list, types=O.CARLA_HIGHWAYS) -> dict:
    """What the list would do, without converting: the apply report (skipped edits, connectivity)."""
    return apply_edits(data, edits, types)[1]


# ------------------------------------------------------------------------------------------------ for the editor
def geometry(data: O.OsmData, proj, types=O.CARLA_HIGHWAYS) -> dict:
    """The area's converted roads for drawing, in the world's frame (proj: osm.local_projection of the import's
    centre: x east, y north): ways (id, name, highway, oneway, lanes (forward, backward) or None, maxspeed, xy),
    junction nodes, signal nodes and traffic_calming nodes with their positions."""
    import numpy as np

    def xy_of(ids):
        lat = np.array([data.nodes[n][0] for n in ids])
        lon = np.array([data.nodes[n][1] for n in ids])
        x, y = proj.to_local(lat, lon)
        return np.round(np.column_stack((np.atleast_1d(x), np.atleast_1d(y))), 2).tolist()
    ways = []
    for w in sorted(data.ways, key=lambda w: int(w['id'])):
        if w['kind'] != 'highway' or w['tags'].get('highway') not in types:
            continue
        ways.append({'id': int(w['id']), 'name': w['tags'].get('name', ''), 'highway': w['tags'].get('highway'),
                     'oneway': O.oneway(w['tags']), 'lanes': O.expected_lanes(w), 'maxspeed': w['tags'].get('maxspeed'),
                     'tags': dict(w['tags']), 'xy': xy_of(w['nodes'])})
    on_road = {n for w in data.ways if w['kind'] == 'highway' and w['tags'].get('highway') in types
               for n in w['nodes']}

    def pts(ids):
        ids = sorted(ids)
        return [{'id': int(n), 'xy': p} for n, p in zip(ids, xy_of(ids))] if ids else []
    junc = junction_nodes(data, types)
    sig = [n for n, t in data.node_tags.items() if t.get('highway') == 'traffic_signals' and n in on_road]
    calm = [n for n, t in data.node_tags.items() if t.get('traffic_calming') and n in on_road]
    out = {'ways': ways, 'junctions': pts(junc), 'signals': pts(sig), 'calming': pts(calm)}
    for item in out['junctions']:
        item['ways'] = junc[item['id']]
    for item in out['calming']:
        item['value'] = data.node_tags[item['id']]['traffic_calming']
    return out


# ------------------------------------------------------------------------------------------------ re-imports
def source_for(world: dict, world_dir: str) -> tuple[str, list | None, str]:
    """Where an imported world's roads can be read again: (path, bbox, what it is). The import's own source and box
    when the file is still there; else the world's cropped, unedited copy (area.source.osm, written when edits were
    applied), else its area.osm (only when no edits were applied to it). Raises FileNotFoundError otherwise."""
    osm = world.get('osm') or {}
    src = osm.get('source') or ''
    if src and os.path.isfile(src):
        return src, osm.get('bbox'), 'the import\'s source file'
    p = os.path.join(world_dir, 'area.source.osm')
    if os.path.isfile(p):
        return p, None, 'the world\'s own copy of its area (unedited)'
    p = os.path.join(world_dir, 'area.osm')
    applied = 0
    try:
        with open(os.path.join(world_dir, 'import.json'), encoding='utf-8') as fh:
            applied = len(((json.load(fh).get('map_edits') or {}).get('applied')) or [])
    except (OSError, ValueError):
        pass
    if os.path.isfile(p) and not applied:
        return p, None, 'the world\'s own copy of its area'
    raise FileNotFoundError(f'neither the import\'s source ({src or "none"}) nor a copy of the area in {world_dir}')


def import_config_for(world: dict, world_dir: str, out_dir: str, edits: list | None = None, seed: int = 0) -> dict:
    """The import config that converts the world again with the current converter, into out_dir: same area, centre
    (so positions in the world do not move), handedness and settings; edits (default: the world's own) applied."""
    from .worlds import import_config
    osm = world.get('osm') or {}
    src, bbox, _ = source_for(world, world_dir)
    cfg = import_config(src, out_dir, bbox, bool(osm.get('lht', True)), osm.get('settings'), osm.get('centre'),
                        seed=seed)
    cfg['edits'] = list(osm.get('edits') or []) if edits is None else list(edits)
    if osm.get('road_features'):
        cfg['road_features'] = dict(osm['road_features'])
    return cfg
