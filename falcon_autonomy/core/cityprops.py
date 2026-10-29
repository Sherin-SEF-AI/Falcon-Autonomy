"""An imported world dressed as a city: buildings, trees, street lamps, bus stops and street furniture from
OpenStreetMap, placed at load time as CARLA's own cooked meshes (numpy and the standard library only: it runs in
CARLA's Python too).

What CARLA 0.9.16's packaged release allows (measured on Falcon's own server, docs/worlds.md "A city around the
roads"): a world built by generate_opendrive_world has roads and sidewalks only, and a new mesh cannot be imported
without the Unreal editor — but `static.prop.mesh` spawns any mesh cooked into the package by its path, at a uniform
scale, and that includes the towns' whole buildings (/Game/Carla/Static/Building/…), trees, bushes and street
lamps. They are drawn, collide (cast_ray hits them) and are labelled by their folder: buildings 3, vegetation 9,
poles 6 in the semantic and instance cameras, the instance id being the actor's id. So:

* buildings: each OSM footprint gets the measured town building closest in size (length, width, height — the
  height from `height`, `building:levels` × LEVEL_M, or a typical one by type), turned to the footprint's
  minimum-area rectangle and uniformly scaled within SCALE_RANGE; never over a lane (the world's own lane geometry)
  nor over another building. It is a stand-in of about the right size, not the real building: its façade, style,
  roof and shape (an L-shaped footprint gets a box) are the town mesh's;
* trees: OSM trees where mapped, and trees spread over parks, grass and woods (positions seeded, spacing
  TREE_SPACING_M), as the package's palms and broad-leaved trees (stand-in species);
* street lamps: OSM lamps where mapped, else along the sidewalks of the bigger roads every LAMP_SPACING_M
  (procedural, said so in the summary);
* bus stops, benches, bins, kiosks, ATMs…: the package's props at OSM's points, moved off the carriageway;
* the ground: one flat mesh under the whole area, 6 cm under the road, when the catalogue names one (label cameras
  see it, so the ground relabelling of workers.camgt names its pixels from the map's lanes). The shipped catalogue
  names none: no mesh of the package was a plausible flat ground (measured, docs/worlds.md), so the space beside the
  roads stays as CARLA draws it.

Everything is deterministic: the same features, map, catalogue, seed and options give the same plan (its SHA-256 is in
the run's provenance). A plan is spawned in one batch per run and destroyed with the run's other actors.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import zlib

import numpy as np

FEATURES_SCHEMA = 'falcon.city-features/1'
PLAN_SCHEMA = 'falcon.city-plan/1'
CATALOGUE_SCHEMA = 'falcon.city-catalogue/1'
RESOURCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources', 'carla',
                        'city-0.9.16.json')

LEVEL_M = 3.2                      # storey height (typical)
TYPICAL_LEVELS = {'house': 2, 'detached': 2, 'semidetached_house': 2, 'bungalow': 1, 'residential': 3,
                  'apartments': 5, 'commercial': 4, 'retail': 2, 'office': 5, 'hotel': 5, 'hospital': 4,
                  'school': 3, 'college': 3, 'university': 3, 'public': 3, 'civic': 3, 'government': 3,
                  'temple': 2, 'church': 2, 'mosque': 2, 'religious': 2, 'industrial': 2, 'warehouse': 2,
                  'garage': 1, 'garages': 1, 'shed': 1, 'roof': 1, 'kiosk': 1, 'service': 1, 'hut': 1,
                  'construction': 3, 'yes': 3}
SCALE_RANGE = (0.6, 1.6)
FIT_SLACK = (1.0, 0.6)             # a placed building may exceed its footprint's rectangle by 0.3 m on each side
MIN_FOOTPRINT_M2 = 12.0
VARIETY = 0.03                     # candidates within this size score of the best are drawn from (seeded)
LANE_MARGIN_M = 0.3                # a building keeps this far from any lane (sidewalks included)
TREE_SPACING_M = 9.0
TREE_CAP_PER_AREA = 400
TREE_CAP = 2000                    # trees spread over areas, the ones nearest a road kept (visible from it)
NUDGE_M = (1.0, 2.0, 3.0)          # a building over a lane is first moved away from it by up to this much
LAMP_SPACING_M = 30.0
GROUND_DROP_M = 0.06
ITEM_CAP = 6000                    # a plan never spawns more (measured: 2500 meshes cost 3 ms a step)

DEFAULT_OPTIONS = {'buildings': True, 'trees': True, 'lamps': 'osm+procedural', 'props': True, 'ground': True}
# 'ground' places the catalogue's ground mesh when it has one: the shipped catalogue has none (measured: no flat mesh
# in the 0.9.16 package with a plausible ground surface — SM_Plane is an upright debug plane, WaterPlane water,
# Big_Plane flowers, the towns' terrain pieces 5 m high and their sidewalk pieces irregular)


class CityError(Exception):
    pass


# --------------------------------------------------------------------------------------------- catalogue
_PIECES = ('Body', 'Corner', 'Top', 'Win', 'Lights', 'Logo', 'Name', 'PlantPot', 'Tower', 'lettering', 'Gate',
           'Platform', 'Ramp', 'Pier', 'Statue', 'Railing', 'Structure', 'Parking_', 'access', 'flat', 'mid',
           'station', 'machine', 'block14White', 'Floor', 'Base', 'BaseGr')


def catalogue_from_probe(probe: dict, measured: str = '') -> dict:
    """The catalogue (resources/carla/city-0.9.16.json) from a server probe: every cooked mesh's measured box, the
    whole buildings among them (not facade pieces; standing on their origin; 4–80 m), the trees, bushes and lamps
    chosen for a south-Indian street (stand-ins), and the tags the cameras gave the samples."""
    boxes = probe.get('boxes') or {}
    out = {'schema': CATALOGUE_SCHEMA, 'carla': '0.9.16', 'measured': measured,
           'method': 'static.prop.mesh spawned once per path on Falcon\'s own server; bounding_box read (extent, '
                     'centre in the mesh frame, unscaled); paths from the package\'s Content folder',
           'buildings': [], 'trees': [], 'bushes': [], 'lamps': [], 'ground': None,
           'tags': {}, 'props': {}}
    seen = set()
    for path, b in sorted(boxes.items()):
        if not b.get('valid'):
            continue
        name = path.rsplit('.', 1)[-1]
        e, c = b['extent'], b['center']
        size = [round(2 * e[0], 3), round(2 * e[1], 3), round(2 * e[2], 3)]
        bottom = c[2] - e[2]
        row = {'path': path, 'size': size, 'center': c, 'bottom': round(bottom, 3)}
        if '/Building/' in path:
            stem = name.replace('_Opt', '').replace('_oPT', '').replace('_merged', '').replace('_Merged', '')
            if any(p in name for p in _PIECES) or abs(bottom) > 1.0 or min(size[:2]) < 4.0 or size[2] < 3.0 \
                    or max(size) > 80.0 or 'Kiosk' in name or 'GuardShelter' in name or 'SkyEvo' in path:
                continue
            key = (stem.lower(), tuple(round(v) for v in size))
            if key in seen:                     # the optimised copy (00_Opt) of the same building is kept, sorted first
                continue
            seen.add(key)
            out['buildings'].append(row)
        elif '/Vegetation/Trees/' in path and any(k in name for k in TREE_KINDS):
            out['trees'].append(row)
        elif '/Vegetation/Bushes/' in path and size[2] >= 0.9:
            out['bushes'].append(row)
        elif name in LAMPS:
            row['arm'] = LAMPS[name]
            out['lamps'].append(row)
    out['buildings'].sort(key=lambda r: ('/00_Opt/' not in r['path'], r['path']))
    for item, got in (probe.get('labels') or {}).items():
        sem = got.get('semantic') or {}
        tags = {int(k): v for k, v in sem.items() if int(k) != 11}
        if tags:
            out['tags'][item] = max(tags, key=tags.get)
    out['props'] = {p: PROPS[p] for p in PROPS if p in set(probe.get('static_props') or [])}
    return out


TREE_KINDS = ('CoconutPalm_01', 'CoconutPalm_02.', 'FanPalm', 'Aporosa.', 'Sassafras', 'Beech', 'Oak_L_v4',
              'Oak_L_v6', 'Oak_M_v4', 'Ash_01', 'Japanese_Maple_01', 'Maple_L_v5', 'Acer_02')
LAMPS = {'SM_Streetlight1_group1': 'x', 'SM_Streetlight2_group1': None, 'SM_Streetlight3_group3': 'x',
         'SM_parkLightV2': None}
# OSM point → a library prop (kind, blueprint, how it stands): measured tags in the catalogue's 'tags'
PROPS = {'static.prop.busstop': 'bus_stop', 'static.prop.bench01': 'bench', 'static.prop.bench02': 'bench',
         'static.prop.bin': 'bin', 'static.prop.trashcan01': 'bin', 'static.prop.kiosk_01': 'kiosk',
         'static.prop.atm': 'atm', 'static.prop.mailbox': 'post_box', 'static.prop.vendingmachine': 'vending',
         'static.prop.streetfountain': 'drinking_water'}


def save_catalogue(doc: dict, path: str = RESOURCE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.write('\n')


def load_catalogue(path: str = RESOURCE) -> dict:
    with open(path, encoding='utf-8') as fh:
        doc = json.load(fh)
    if doc.get('schema') != CATALOGUE_SCHEMA:
        raise CityError(f'{path}: not a city catalogue')
    return doc


# --------------------------------------------------------------------------------------------- features
def _point_kind(t: dict) -> str | None:
    if t.get('natural') == 'tree':
        return 'tree'
    if t.get('highway') == 'street_lamp':
        return 'lamp'
    if t.get('highway') == 'bus_stop':
        return 'bus_stop'
    a = t.get('amenity')
    if a == 'bench':
        return 'bench'
    if a in ('waste_basket', 'waste_disposal', 'recycling'):
        return 'bin'
    if a == 'atm':
        return 'atm'
    if a == 'post_box':
        return 'post_box'
    if a == 'vending_machine':
        return 'vending'
    if a == 'drinking_water':
        return 'drinking_water'
    if t.get('shop') == 'kiosk':
        return 'kiosk'
    return None


def _area_kind(t: dict) -> str | None:
    if t.get('natural') == 'water':
        return 'water'
    if t.get('natural') in ('wood',) or t.get('landuse') in ('forest', 'orchard'):
        return 'wood'
    if t.get('natural') == 'tree_row':
        return 'tree_row'
    if t.get('leisure') in ('park', 'garden', 'common') or t.get('landuse') in ('recreation_ground', 'village_green',
                                                                                 'cemetery'):
        return 'park'
    if t.get('leisure') in ('pitch', 'playground'):
        return 'open'
    return 'grass'


def source_key(path: str) -> dict:
    st = os.stat(path)
    return {'path': os.path.abspath(path), 'size': int(st.st_size), 'mtime': int(st.st_mtime)}


def extract(source: str, bbox, centre, progress=None) -> dict:
    """The city features of an OSM file within bbox (south, west, north, east), in the world's local frame (metres
    east and north of `centre`, OpenDRIVE's): buildings (closed footprints), areas (parks, grass, woods, water,
    tree rows) and point features. The source is read, never written."""
    from . import osm as O
    data = O.read_osm(source, bbox=bbox, progress=progress, city=True)
    proj = O.local_projection(*centre)

    def xy(refs):
        la = np.array([data.nodes[r][0] for r in refs], float)
        lo = np.array([data.nodes[r][1] for r in refs], float)
        e, n = proj.to_local(la, lo)
        return np.round(np.stack([e, n], 1), 3)

    buildings, areas, notes = [], [], list(data.notes)
    open_fp = 0
    for w in data.ways:
        if w['kind'] == 'building':
            if len(w['nodes']) < 4 or w['nodes'][0] != w['nodes'][-1]:
                open_fp += 1
                continue
            t = w['tags']
            keep = {k: t[k] for k in ('building', 'building:levels', 'height', 'name', 'roof:levels') if k in t}
            buildings.append({'id': int(w['id']), 'poly': xy(w['nodes'][:-1]).tolist(), 'tags': keep})
        elif w['kind'] == 'area':
            k = _area_kind(w['tags'])
            pts = xy(w['nodes'])
            closed = w['nodes'][0] == w['nodes'][-1]
            if k != 'tree_row' and (not closed or len(w['nodes']) < 4):
                continue
            areas.append({'id': int(w['id']), 'kind': k, 'poly': (pts[:-1] if closed else pts).tolist()})
    points = []
    for p in data.points:
        k = _point_kind(p['tags'])
        if not k:
            continue
        e, n = proj.to_local(p['lat'], p['lon'])
        points.append({'id': int(p['id']), 'kind': k, 'xy': [round(float(e), 3), round(float(n), 3)]})
    if open_fp:
        notes.append(f'{open_fp} building ways were not closed footprints and are left out')
    counts = {'buildings': len(buildings), 'areas': len(areas), 'points': len(points),
              'by_point': {k: sum(p['kind'] == k for p in points) for k in sorted({p['kind'] for p in points})},
              'by_area': {k: sum(a['kind'] == k for a in areas) for k in sorted({a['kind'] for a in areas})}}
    return {'schema': FEATURES_SCHEMA, 'source': source_key(source), 'bbox': list(bbox), 'centre': list(centre),
            'frame': 'metres east, north of the centre (the world\'s OpenDRIVE frame)', 'buildings': buildings,
            'areas': areas, 'points': points, 'counts': counts, 'notes': notes}


# --------------------------------------------------------------------------------------------- geometry
def poly_area(P) -> float:
    P = np.asarray(P, float)
    x, y = P[:, 0], P[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _hull(P: np.ndarray) -> np.ndarray:
    pts = sorted(set(map(tuple, np.round(P, 6))))
    if len(pts) <= 2:
        return np.asarray(pts, float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return np.asarray(lo[:-1] + hi[:-1], float)


def min_rect(P) -> tuple:
    """The minimum-area rectangle around points: (cx, cy, length, width, angle) with length ≥ width and angle the
    length's direction (radians, from east, counter-clockwise) in [0, π)."""
    H = _hull(np.asarray(P, float))
    if len(H) < 3:
        c = H.mean(0) if len(H) else np.zeros(2)
        return float(c[0]), float(c[1]), 0.0, 0.0, 0.0
    best = None
    for i in range(len(H)):
        d = H[(i + 1) % len(H)] - H[i]
        n = math.hypot(*d)
        if n < 1e-9:
            continue
        u = d / n
        v = np.array([-u[1], u[0]])
        a, b = H @ u, H @ v
        area = (a.max() - a.min()) * (b.max() - b.min())
        if best is None or area < best[0] - 1e-9:
            best = (area, u, v, a.min(), a.max(), b.min(), b.max())
    _, u, v, a0, a1, b0, b1 = best
    c = u * (a0 + a1) / 2 + v * (b0 + b1) / 2
    L, W, ang = a1 - a0, b1 - b0, math.atan2(u[1], u[0])
    if W > L:
        L, W, ang = W, L, ang + math.pi / 2
    return float(c[0]), float(c[1]), float(L), float(W), float(ang % math.pi)


def rect_corners(cx, cy, L, W, ang) -> np.ndarray:
    u = np.array([math.cos(ang), math.sin(ang)])
    v = np.array([-u[1], u[0]])
    c = np.array([cx, cy])
    return np.array([c + u * L / 2 + v * W / 2, c - u * L / 2 + v * W / 2, c - u * L / 2 - v * W / 2,
                     c + u * L / 2 - v * W / 2])


def rects_overlap(r1, r2) -> bool:
    """Two rectangles (cx, cy, L, W, angle) overlap (separating axis test; touching is not overlapping)."""
    A, B = rect_corners(*r1), rect_corners(*r2)
    for R in (A, B):
        for i in range(2):
            ax = R[i + 1] - R[i]
            n = np.array([-ax[1], ax[0]])
            pa, pb = A @ n, B @ n
            if pa.max() <= pb.min() + 1e-9 or pb.max() <= pa.min() + 1e-9:
                return False
    return True


def point_rect_distance(P, rect) -> np.ndarray:
    """Distances of points (N, 2) to a rectangle (0 inside)."""
    cx, cy, L, W, ang = rect
    P = np.asarray(P, float) - (cx, cy)
    u = np.array([math.cos(ang), math.sin(ang)])
    v = np.array([-u[1], u[0]])
    du = np.maximum(np.abs(P @ u) - L / 2, 0.0)
    dv = np.maximum(np.abs(P @ v) - W / 2, 0.0)
    return np.hypot(du, dv)


def in_poly(P, poly) -> np.ndarray:
    """Points (N, 2) inside a polygon (even-odd rule)."""
    P = np.atleast_2d(np.asarray(P, float))
    Q = np.asarray(poly, float)
    x, y = P[:, 0][:, None], P[:, 1][:, None]
    x0, y0 = Q[:, 0][None], Q[:, 1][None]
    x1, y1 = np.roll(Q[:, 0], -1)[None], np.roll(Q[:, 1], -1)[None]
    cond = (y0 > y) != (y1 > y)
    with np.errstate(divide='ignore', invalid='ignore'):
        xs = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return ((cond & (x < xs)).sum(1) % 2) == 1


# --------------------------------------------------------------------------------------------- the map's lanes
class Lanes:
    """The world's lanes as sampled centre points (OpenDRIVE frame) — rows x, y, half width, lane id, road index —
    in a grid for nearness tests: `paved` every lane type (a building keeps off sidewalks too), `driving` the
    carriageways, `sidewalk` the sidewalks."""
    CELL = 8.0

    def __init__(self, xodr_text: str, step: float = 1.0):
        from . import xodr as X
        od = X.parse(xodr_text)
        rows = {'paved': [], 'driving': [], 'sidewalk': []}
        self.roads = []
        for road in od.roads:
            types = set()
            for sec in road.sections:
                types |= {ln.type for ln in sec.left + sec.right}
            types.discard('none')
            if not types:
                continue
            c = X.lane_centres(road, step, types=tuple(types))
            if not len(c):
                continue
            ids = c[:, 3].astype(int)
            lane_type = {}
            for sec in road.sections:
                for ln in sec.left + sec.right:
                    lane_type.setdefault(ln.id, ln.type)
            kinds = np.array([lane_type.get(i, 'driving') for i in ids])
            pts = np.c_[c[:, 0], c[:, 1], c[:, 4] / 2.0, c[:, 3], np.full(len(c), len(self.roads))]
            rows['paved'].append(pts)
            rows['driving'].append(pts[kinds == 'driving'])
            rows['sidewalk'].append(pts[kinds == 'sidewalk'])
            self.roads.append({'id': road.id, 'junction': road.in_junction, 'length': road.length})
        self.sets = {k: (np.concatenate(v) if v else np.zeros((0, 5))) for k, v in rows.items()}
        self.grids = {k: self._grid(v) for k, v in self.sets.items()}
        self.max_half = {k: (float(v[:, 2].max()) if len(v) else 0.0) for k, v in self.sets.items()}

    def _grid(self, pts):
        g = {}
        for i, (x, y) in enumerate(pts[:, :2]):
            g.setdefault((int(x // self.CELL), int(y // self.CELL)), []).append(i)
        return {k: np.asarray(v) for k, v in g.items()}

    def near(self, kind: str, x0, y0, x1, y1) -> np.ndarray:
        """Rows of lane points of a kind within an axis-aligned box."""
        g, pts = self.grids[kind], self.sets[kind]
        idx = [g[(i, j)] for i in range(int(x0 // self.CELL), int(x1 // self.CELL) + 1)
               for j in range(int(y0 // self.CELL), int(y1 // self.CELL) + 1) if (i, j) in g]
        return pts[np.concatenate(idx)] if idx else np.zeros((0, 5))

    def clearance(self, kind: str, rect) -> float:
        """The least gap between a rectangle and the lanes of a kind (negative: it covers a lane); inf if none
        near."""
        cx, cy, L, W, _ = rect
        r = math.hypot(L, W) / 2 + self.max_half[kind] + 2.0
        P = self.near(kind, cx - r, cy - r, cx + r, cy + r)
        if not len(P):
            return math.inf
        return float((point_rect_distance(P[:, :2], rect) - P[:, 2]).min())

    def nearest(self, kind: str, x, y, radius: float = 30.0):
        """The nearest lane point of a kind (row) and its distance to the lane's edge, or (None, inf)."""
        P = self.near(kind, x - radius, y - radius, x + radius, y + radius)
        if not len(P):
            return None, math.inf
        d = np.hypot(P[:, 0] - x, P[:, 1] - y)
        i = int(np.argmin(d))
        return P[i], float(d[i] - P[i, 2])


# --------------------------------------------------------------------------------------------- the plan
def _rng(seed: int, *key) -> np.random.Generator:
    return np.random.default_rng([int(seed) & 0xFFFFFFFF] + [zlib.crc32(str(k).encode()) for k in key])


def building_height(tags: dict) -> tuple[float, str]:
    """(metres, where from): `height`, else `building:levels` × LEVEL_M, else typical levels for the type."""
    h = tags.get('height')
    if h:
        try:
            v = float(str(h).lower().replace('m', '').strip().split()[0])
            if 2.0 <= v <= 300.0:
                return v, 'height'
        except (ValueError, IndexError):
            pass
    lv = tags.get('building:levels')
    if lv:
        try:
            v = float(str(lv).split(';')[0])
            if 1 <= v <= 100:
                return v * LEVEL_M + (0.0 if v > 1 else 0.6), 'levels'
        except ValueError:
            pass
    return TYPICAL_LEVELS.get(tags.get('building', 'yes'), TYPICAL_LEVELS['yes']) * LEVEL_M, 'typical'


def _fit(mesh: dict, L: float, W: float, H: float):
    """The best orientation and scale of a mesh for a footprint L × W (L ≥ W) and height H:
    (score, scale, quarter turn) — size_score of the scaled box, the scale within SCALE_RANGE and the footprint's
    slack, a quarter turn when the mesh's y lies along the length."""
    from .assets import best_scale, size_score
    sx, sy, sz = mesh['size']
    best = None
    for quarter, (a, b) in ((0, (sx, sy)), (1, (sy, sx))):
        s = best_scale((a, b, sz), (L, W, H), *SCALE_RANGE)
        s = min(s, (L * FIT_SLACK[0] + FIT_SLACK[1]) / a, (W * FIT_SLACK[0] + FIT_SLACK[1]) / b)
        if s < SCALE_RANGE[0] - 1e-9:
            continue
        sc = size_score((a * s, b * s, sz * s), (L, W, H))
        if best is None or sc < best[0]:
            best = (sc, s, quarter)
    return best


def _place_mesh(mesh: dict, cx: float, cy: float, ang: float, scale: float, z: float = 0.0) -> dict:
    """The CARLA transform that puts a mesh's (scaled) box centre over (cx, cy) of the OpenDRIVE frame, turned by
    `ang` (OpenDRIVE, radians), its bottom at z: the mesh scales about its origin, so its box centre moves with the
    scale (measured: docs/carla-lessons.md)."""
    ox, oy, oz = (scale * v for v in mesh['center'])
    bottom = scale * mesh['bottom']
    # OpenDRIVE (x east, y north, CCW) → CARLA (x east, y south, yaw clockwise as seen from above)
    yaw = -math.degrees(ang)
    cy_c = -cy
    r = math.radians(yaw)
    x = cx - (ox * math.cos(r) - oy * math.sin(r))
    y = cy_c - (ox * math.sin(r) + oy * math.cos(r))
    return {'x': round(x, 3), 'y': round(y, 3), 'z': round(z - bottom, 3), 'yaw': round(((yaw + 180) % 360) - 180, 3)}


def plan(features: dict, xodr_text: str, catalogue: dict, seed: int = 0, options: dict | None = None) -> dict:
    """Where every building, tree, lamp, prop and the ground go: {'items': [{kind, mesh | blueprint, scale, x, y,
    z, yaw (CARLA), source, osm_id, …}], 'summary', 'notes'}. Deterministic in (features, map, catalogue, seed,
    options)."""
    opt = dict(DEFAULT_OPTIONS, **(options or {}))
    lanes = Lanes(xodr_text)
    items, skipped, notes = [], {}, []

    def skip(kind, why):
        skipped.setdefault(kind, {}).setdefault(why, 0)
        skipped[kind][why] += 1

    # buildings, largest first (a big footprint's stand-in is chosen before small ones crowd it)
    placed = []                              # (rect, grid cell keys)
    pgrid: dict = {}
    fits, sources = [], {}
    blds = sorted(features.get('buildings') or [], key=lambda b: (-poly_area(b['poly']), b['id']))
    cat_b = catalogue.get('buildings') or []
    for b in blds if opt['buildings'] and cat_b else ():
        P = np.asarray(b['poly'], float)
        area = poly_area(P)
        if area < MIN_FOOTPRINT_M2:
            skip('building', 'footprint under %d m²' % MIN_FOOTPRINT_M2)
            continue
        cx, cy, L, W, ang = min_rect(P)
        H, h_from = building_height(b.get('tags') or {})
        cands = []
        for m in cat_b:
            f = _fit(m, L, W, H)
            if f:
                cands.append((f[0], m['path'], m, f[1], f[2]))
        if not cands:
            skip('building', 'no stand-in fits')
            continue
        cands.sort(key=lambda c: (c[0], c[1]))
        near = [c for c in cands if c[0] <= cands[0][0] + VARIETY]
        rng = _rng(seed, 'building', b['id'])
        sc, _, m, s, quarter = near[int(rng.integers(len(near)))]
        flip = int(rng.integers(2))                       # the façade's side is not known: either way along the length
        a = ang + quarter * math.pi / 2 + flip * math.pi
        ok = None
        away, _ = lanes.nearest('paved', cx, cy, radius=60.0)
        nx = ny = 0.0
        if away is not None:
            d = math.hypot(cx - away[0], cy - away[1])
            nx, ny = ((cx - away[0]) / d, (cy - away[1]) / d) if d > 1e-6 else (0.0, 0.0)
        for shrink in (1.0, 0.9, 0.8, 0.7):
            s2 = s * shrink
            if s2 < SCALE_RANGE[0] - 1e-9:
                break
            lx, ly = m['size'][0] * s2, m['size'][1] * s2
            for nudge in (0.0,) + NUDGE_M:
                rect = (cx + nx * nudge, cy + ny * nudge, lx, ly, a)
                if lanes.clearance('paved', rect) >= LANE_MARGIN_M:
                    break
            else:
                why = 'would cover a lane or sidewalk'
                continue
            cells = {(int(x // 20), int(y // 20)) for x, y in rect_corners(*rect)}
            if any(rects_overlap(rect, placed[k][0]) for c in cells for k in pgrid.get(c, ())):
                why = 'would overlap another building'
                continue
            ok = (s2, rect, cells)
            break
        if ok is None:
            skip('building', why)
            continue
        s2, rect, cells = ok
        k = len(placed)
        placed.append((rect, cells))
        for c in cells:
            pgrid.setdefault(c, []).append(k)
        moved = round(math.hypot(rect[0] - cx, rect[1] - cy), 2)
        tf = _place_mesh(m, rect[0], rect[1], a, s2)
        from .assets import size_score
        score = size_score((m['size'][0] * s2, m['size'][1] * s2, m['size'][2] * s2),
                           (L, W, H) if quarter == 0 else (L, W, H))
        fits.append(score)
        sources[h_from] = sources.get(h_from, 0) + 1
        items.append({'kind': 'building', 'mesh': m['path'], 'scale': round(s2, 4), **tf, 'source': 'osm',
                      'osm_id': b['id'], 'footprint': [round(L, 2), round(W, 2), round(area, 1)],
                      'height': [round(H, 2), h_from], 'placed': [round(m['size'][0] * s2, 2),
                                                                  round(m['size'][1] * s2, 2),
                                                                  round(m['size'][2] * s2, 2)],
                      'score': round(score, 4), **({'moved_m': moved} if moved else {})})

    def on_building(x, y, pad=0.0):
        cells = {(int(x // 20), int(y // 20))}
        return any(point_rect_distance([[x, y]], placed[k][0])[0] <= pad for c in cells for k in pgrid.get(c, ()))

    # trees: OSM's, then spread over parks, grass, woods and along tree rows
    trees = catalogue.get('trees') or []
    tree_pts = []
    if opt['trees'] and trees:
        for p in features.get('points') or []:
            if p['kind'] == 'tree':
                tree_pts.append((p['xy'][0], p['xy'][1], 'osm', p['id']))
        for a in features.get('areas') or []:
            if a['kind'] == 'water':
                continue
            P = np.asarray(a['poly'], float)
            rng = _rng(seed, 'area', a['id'])
            if a['kind'] == 'tree_row':
                seg = np.hypot(*np.diff(P, axis=0).T)
                total = float(seg.sum())
                for d in np.arange(TREE_SPACING_M / 2, total, TREE_SPACING_M * 0.8):
                    i = int(np.searchsorted(np.cumsum(seg), d))
                    i = min(i, len(seg) - 1)
                    t = (d - (np.cumsum(seg)[i] - seg[i])) / max(seg[i], 1e-9)
                    q = P[i] + t * (P[i + 1] - P[i])
                    tree_pts.append((float(q[0]), float(q[1]), 'osm-area', a['id']))
                continue
            spacing = TREE_SPACING_M * (0.7 if a['kind'] == 'wood' else 1.0 if a['kind'] == 'park' else 1.6)
            x0, y0 = P.min(0)
            x1, y1 = P.max(0)
            n_try = int(min(4 * poly_area(P) / spacing ** 2, 8 * TREE_CAP_PER_AREA))
            cand = np.c_[rng.uniform(x0, x1, n_try), rng.uniform(y0, y1, n_try)] if n_try else np.zeros((0, 2))
            cand = cand[in_poly(cand, P)] if len(cand) else cand
            kept = []
            for q in cand:                                  # dart throwing: a seeded Poisson-disc sample
                if all(math.hypot(q[0] - k[0], q[1] - k[1]) >= spacing for k in kept):
                    kept.append(q)
                    if len(kept) >= TREE_CAP_PER_AREA:
                        break
            tree_pts += [(float(q[0]), float(q[1]), 'osm-area', a['id']) for q in kept]
        area_pts = [t for t in tree_pts if t[1] is not None and t[2] != 'osm']
        if len(area_pts) > TREE_CAP:
            near = []
            for t in area_pts:
                _, g = lanes.nearest('paved', t[0], t[1], radius=60.0)
                near.append(g)
            order = np.argsort(np.asarray(near), kind='stable')
            keep = {area_pts[i] for i in order[:TREE_CAP]}
            skipped.setdefault('tree', {})['over the cap (farthest from a road)'] = len(area_pts) - TREE_CAP
            tree_pts = [t for t in tree_pts if t[2] == 'osm' or t in keep]
        for x, y, src, oid in tree_pts:
            _, gap = lanes.nearest('driving', x, y, radius=10.0)
            if gap < 0.8:
                skip('tree', 'on a carriageway')
                continue
            if src != 'osm':
                _, g2 = lanes.nearest('paved', x, y, radius=10.0)
                if g2 < 1.0:
                    skip('tree', 'on a sidewalk')
                    continue
            if on_building(x, y, 1.0):
                skip('tree', 'in a building')
                continue
            rng = _rng(seed, 'tree', oid, round(x, 1), round(y, 1))
            m = trees[int(rng.integers(len(trees)))]
            s = float(rng.uniform(0.8, 1.15))
            items.append({'kind': 'tree', 'mesh': m['path'], 'scale': round(s, 3),
                          **_place_mesh(m, x, y, float(rng.uniform(0, 2 * math.pi)), s), 'source': src,
                          'osm_id': oid})

    # street lamps: OSM's; else along the bigger roads' sidewalks
    arm_lamps = [m for m in catalogue.get('lamps') or [] if m.get('arm') == 'x']
    park_lamps = [m for m in catalogue.get('lamps') or [] if not m.get('arm')]
    lamp_mode = str(opt['lamps'] or 'off')
    if (arm_lamps or park_lamps) and lamp_mode != 'off':
        m = (arm_lamps or park_lamps)[0]
        lamp_pts = []
        if 'osm' in lamp_mode:
            lamp_pts = [(p['xy'][0], p['xy'][1], 'osm', p['id']) for p in features.get('points') or []
                        if p['kind'] == 'lamp']
        if 'procedural' in lamp_mode:
            S = lanes.sets['sidewalk']
            last: dict = {}
            for row in S:
                ri = int(row[4])
                if lanes.roads[ri]['junction']:
                    continue
                key = (ri, 1 if row[3] > 0 else 0)
                prev = last.get(key)
                if prev is not None and math.hypot(row[0] - prev[0], row[1] - prev[1]) < LAMP_SPACING_M:
                    continue
                last[key] = (row[0], row[1])
                lamp_pts.append((float(row[0]), float(row[1]), 'procedural', None))
        kept = []
        for x, y, src, oid in lamp_pts:
            d, gap = lanes.nearest('driving', x, y, radius=25.0)
            if d is None and src == 'osm' and park_lamps:     # a lamp in a park or a yard: a lamp post, no arm
                if on_building(x, y, 0.3):
                    skip('lamp', 'in a building')
                    continue
                pm = park_lamps[0]
                kept.append((x, y))
                items.append({'kind': 'lamp', 'mesh': pm['path'], 'scale': 1.0,
                              **_place_mesh(dict(pm, center=[0.0, 0.0, pm['center'][2]]), x, y,
                                            float(_rng(seed, 'lamp', oid).uniform(0, 2 * math.pi)), 1.0),
                              'source': src, 'osm_id': oid})
                continue
            if d is None or gap < 0.4:
                skip('lamp', 'no carriageway near' if d is None else 'on a carriageway')
                continue
            if any(math.hypot(x - k[0], y - k[1]) < LAMP_SPACING_M * 0.6 for k in kept):
                skip('lamp', 'too close to another lamp')
                continue
            if on_building(x, y, 0.5):
                skip('lamp', 'in a building')
                continue
            # the pole at the kerb, its arm (the mesh's +x) towards the carriageway
            ang = math.atan2(d[1] - y, d[0] - x)
            kept.append((x, y))
            items.append({'kind': 'lamp', 'mesh': m['path'], 'scale': 1.0, **_place_mesh(
                dict(m, center=[0.0, 0.0, m['center'][2]]), x, y, ang, 1.0), 'source': src, 'osm_id': oid})

    # street furniture at OSM's points (the library's props), moved onto the nearest sidewalk when on a carriageway
    by_kind = {}
    for bp, k in (catalogue.get('props') or {}).items():
        by_kind.setdefault(k, []).append(bp)
    if opt['props']:
        for p in features.get('points') or []:
            bps = by_kind.get(p['kind'])
            if not bps:
                continue
            x, y = p['xy']
            d, gap = lanes.nearest('driving', x, y, radius=25.0)
            if d is not None and gap < 0.3:
                sw, _ = lanes.nearest('sidewalk', x, y, radius=12.0)
                if sw is None:
                    skip(p['kind'], 'on a carriageway, no sidewalk near')
                    continue
                x, y = float(sw[0]), float(sw[1])
            if on_building(x, y, 0.2):
                skip(p['kind'], 'in a building')
                continue
            ang = math.atan2(d[1] - y, d[0] - x) - math.pi / 2 if d is not None else 0.0
            rng = _rng(seed, 'prop', p['id'])
            items.append({'kind': p['kind'], 'blueprint': bps[int(rng.integers(len(bps)))],
                          'x': round(x, 3), 'y': round(-y, 3), 'z': 0.0, 'yaw': round(-math.degrees(ang), 3),
                          'scale': 1.0, 'source': 'osm', 'osm_id': p['id']})

    # the ground under everything
    g = catalogue.get('ground')
    if opt['ground'] and g:
        P = lanes.sets['paved'][:, :2]
        pts = [P] + [np.asarray(b['poly'], float) for b in features.get('buildings') or []]
        allp = np.concatenate([q for q in pts if len(q)]) if any(len(q) for q in pts) else np.zeros((1, 2))
        x0, y0 = allp.min(0) - 60.0
        x1, y1 = allp.max(0) + 60.0
        side = max(x1 - x0, y1 - y0)
        s = side / max(g['size'][0], g['size'][1])
        m = dict(g)
        tf = _place_mesh(m, (x0 + x1) / 2, (y0 + y1) / 2, 0.0, s, z=0.0)
        tf['z'] = round(-GROUND_DROP_M - s * (g['center'][2] + g['size'][2] / 2), 4)
        items.append({'kind': 'ground', 'mesh': g['path'], 'scale': round(s, 4), **tf, 'source': 'procedural',
                      'osm_id': None})

    if len(items) > ITEM_CAP:
        notes.append(f'{len(items) - ITEM_CAP} items over the cap of {ITEM_CAP} left out (farthest from the centre '
                     'first)')
        keep_ground = [i for i in items if i['kind'] == 'ground']
        rest = sorted((i for i in items if i['kind'] != 'ground'), key=lambda i: (math.hypot(i['x'], i['y']),
                                                                                   i['kind']))
        items = keep_ground + rest[:ITEM_CAP - len(keep_ground)]
    counts = {}
    for i in items:
        c = counts.setdefault(i['kind'], {})
        c[i['source']] = c.get(i['source'], 0) + 1
    summary = {'items': len(items), 'by_kind': counts, 'skipped': skipped,
               'buildings_in_features': len(features.get('buildings') or []),
               'fit_score_median': round(float(np.median(fits)), 3) if fits else None,
               'height_from': sources}
    doc = {'schema': PLAN_SCHEMA, 'seed': int(seed), 'options': opt,
           'features': {'source': features.get('source'), 'counts': features.get('counts')},
           'xodr_sha256': hashlib.sha256(xodr_text.encode()).hexdigest(),
           'catalogue': {'carla': catalogue.get('carla'), 'measured': catalogue.get('measured')},
           'frame': 'CARLA world (x east, y south, yaw degrees)', 'items': items, 'summary': summary,
           'notes': notes + list(features.get('notes') or [])}
    return doc


def plan_sha(doc: dict) -> str:
    return hashlib.sha256(json.dumps(doc.get('items'), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


# --------------------------------------------------------------------------------------------- a world's city
def city_dir(world_dir: str) -> str:
    return os.path.join(world_dir, 'city')


def build(world_dir: str, source: str, bbox, centre, seed: int = 0, options: dict | None = None,
          margin_m: float = 150.0, catalogue: dict | None = None, progress=None) -> dict:
    """Extract a world's city features from `source` (its box plus margin_m) and plan them on its world.xodr; writes
    city/features.json and city/plan.json in the world's folder and returns the summary the world entry keeps."""
    import time
    t0 = time.monotonic()
    s, w, n, e = (float(v) for v in bbox)
    dlat = margin_m / 110574.0
    dlon = margin_m / (111319.49 * max(math.cos(math.radians((s + n) / 2)), 0.1))
    feats = extract(source, (s - dlat, w - dlon, n + dlat, e + dlon), centre, progress)
    return plan_world(world_dir, feats, seed, options, catalogue, read_s=time.monotonic() - t0)


def plan_world(world_dir: str, feats: dict, seed: int = 0, options: dict | None = None,
               catalogue: dict | None = None, read_s: float = 0.0) -> dict:
    """Plan extracted features on a world's world.xodr; writes city/features.json and city/plan.json and returns the
    summary the world entry keeps ('city')."""
    import time
    t1 = time.monotonic()
    with open(os.path.join(world_dir, 'world.xodr'), encoding='utf-8') as fh:
        text = fh.read()
    cat = catalogue or load_catalogue()
    doc = plan(feats, text, cat, seed, options)
    t2 = time.monotonic()
    out = city_dir(world_dir)
    os.makedirs(out, exist_ok=True)
    for name, d in (('features.json', feats), ('plan.json', doc)):
        tmp = os.path.join(out, name + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(d, fh, separators=(',', ':'))
        os.replace(tmp, os.path.join(out, name))
    return {'enabled': True, 'source': feats['source'], 'seed': int(seed), 'options': doc['options'],
            'plan': 'city/plan.json', 'plan_sha256': plan_sha(doc), 'summary': doc['summary'],
            'features': feats['counts'], 'notes': doc['notes'],
            'timings': {'read_s': round(read_s, 2), 'plan_s': round(t2 - t1, 2)}}


def attach(cfg: dict, world: dict, workspace: str):
    """Put an imported world's city plan into a worker config (cfg['city']) when the world has one and it is on;
    absent otherwise, so other configs' cache keys are unchanged."""
    c = world.get('city') or {}
    if not c.get('enabled') or not c.get('plan') or not workspace or not world.get('xodr'):
        return
    path = os.path.join(workspace, os.path.dirname(world['xodr']), c['plan'])
    if not os.path.isfile(path):
        cfg.setdefault('notes', []).append(f'the world\'s city plan {c["plan"]} is missing: the world is drawn '
                                           'without it (Worlds ▸ City: build it again)')
        return
    cfg['city'] = {'plan': path, 'plan_sha256': c.get('plan_sha256'), 'summary': (c.get('summary') or {}).get('by_kind')}


# --------------------------------------------------------------------------------------------- spawning (worker)
def spawn(w, world, ccfg: dict) -> dict:
    """Spawn a city plan into the server's world (one batch), own the actors so the run's tidy-up removes them, and
    tell the ground relabelling which actors are ground (w.city_ground_ids). Returns what the run records."""
    import time
    with open(ccfg['plan'], encoding='utf-8') as fh:
        doc = json.load(fh)
    carla = w.carla
    lib = world.get_blueprint_library()
    cmds, kinds = [], []
    for it in doc.get('items') or []:
        if it.get('mesh'):
            bp = lib.find('static.prop.mesh')              # a copy per item: the command keeps its blueprint
            bp.set_attribute('mesh_path', it['mesh'])
            bp.set_attribute('scale', str(it.get('scale', 1.0)))
        else:
            try:
                bp = lib.find(it['blueprint'])
            except (IndexError, RuntimeError):
                kinds.append(None)
                continue
        tf = carla.Transform(carla.Location(x=float(it['x']), y=float(it['y']), z=float(it['z'])),
                             carla.Rotation(yaw=float(it.get('yaw', 0.0))))
        cmds.append(carla.command.SpawnActor(bp, tf))
        kinds.append(it['kind'])
    t0 = time.monotonic()
    res = w.client.apply_batch_sync(cmds, False) if cmds else []
    spawn_s = time.monotonic() - t0
    ids, failed = [], 0
    ground = []
    kk = [k for k in kinds if k is not None]
    for r, k in zip(res, kk):
        if r.error:
            failed += 1
            continue
        ids.append(r.actor_id)
        if k == 'ground':
            ground.append(r.actor_id)
    for a in world.get_actors(ids) if ids else ():
        w.own(a)
    w.city_ground_ids = frozenset(ground)
    w.alive()
    out = {'plan_sha256': plan_sha(doc), 'spawned': len(ids), 'failed': failed,
           'missing_blueprints': sum(1 for k in kinds if k is None), 'spawn_s': round(spawn_s, 3),
           'by_kind': (doc.get('summary') or {}).get('by_kind')}
    return out


def populate(w, world, cfg: dict):
    """The worker's hook (workers.scene.load_map): spawn cfg['city'], recorded in run.json's 'city'. A failure is
    noted and the run goes on without the city (the roads are unaffected)."""
    ccfg = cfg['city']
    w.emit.progress(0.02, 'placing the city around the roads', force=True)
    try:
        got = spawn(w, world, ccfg)
    except (OSError, ValueError, RuntimeError) as e:
        w.note(f'the city around the roads was not placed: {e}')
        w.run['city'] = {'plan_sha256': ccfg.get('plan_sha256'), 'error': str(e)}
        return
    if got['failed']:
        w.note(f'{got["failed"]} of {got["spawned"] + got["failed"]} city meshes did not spawn')
    w.run['city'] = dict(got, plan=os.path.basename(os.path.dirname(os.path.dirname(ccfg['plan']))) + '/city/plan.json')
