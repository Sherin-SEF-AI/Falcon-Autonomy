"""OpenDRIVE: parse, reference lines, preview graph, handedness, right→left conversion, lane check (stdlib + numpy).

parse() reads what a preview and a check need: the header (with its geoReference text and offset), roads (id, name,
length, junction, rule, links, planView geometry, laneOffsets, lane sections with lane ids, types, widths and lane
links, signal counts) and junctions (connections and laneLinks). Elevation, lateral profile, objects and road marks
are not modelled (to_lht works on the XML tree and keeps them).

Geometry (sample_reference_line): line and arc are exact. Spirals (clothoids) are integrated with 5-point
Gauss–Legendre quadrature on pieces of at most 1 m and 0.05 rad: position error below 1e-9 m per 100 m.
poly3 and paramPoly3 (pRange arcLength or normalized) map s to the curve parameter by inverting an arc-length table
(trapezoid rule on pieces of at most 5 cm, at least 64 per geometry): position error below 0.1 mm for curvature up to
0.2 /m. Where a parametric curve's own length differs from the geometry's `length`, s is scaled to the curve's length.

to_lht() turns a right-hand-traffic file (SUMO netconvert / carla.Osm2Odr output) into left-hand traffic that CARLA
0.9.16 drives on the left (verified on a real server: with rule="LHT" on a road, lane +1 runs along +s and the traffic
manager drives in it). Per road: lanes are mirrored across the reference line (lane k -> -k, <right> <-> <left>, lanes
inside <left> ordered by descending id; widths, types, road marks, speeds kept; road-mark laneChange increase <->
decrease), rule="LHT" is set, laneOffset a b c d and superelevation a b c d are negated, lane-link ids (lane
<link> predecessor/successor, junction <laneLink from to>) are negated, signals' and signal references' t is negated
and their validity fromLane/toLane swapped and negated (from <= to still holds) and their orientation flipped ("+"
<-> "-": the lanes it applies to are now on the other side; with the orientation kept, CARLA 0.9.16's traffic lights
controlled no lane and traffic-manager cars drove through red — measured), objects' t, hdg, cornerLocal v and cornerRoad t are negated.
Each road's travel direction is unchanged; only its side changes. Signal positionInertial and hOffset are kept (SUMO
writes t=0 everywhere, so its signals do not move). lateralProfile <shape> is
refused. Known limit: on approaches with two or more lanes per direction, which lane turns where is the right-hand
converter's assignment mirrored (the inner lane keeps the turns it had), not a true left-hand assignment;
single-lane approaches are unaffected.

Where the lanes go: mirroring across the reference line is right for a road whose reference line is its OSM way
(carla.Osm2Odr writes the two directions of a two-way way as two roads sharing the way as reference line). A lone
one-way road's reference line is its carriageway's left edge instead, so mirrored there the carriageway would move a
whole carriageway width sideways (measured on central Bengaluru: 3.35–10 m, 33 of 72 one-way roads). Such roads (one
side of lanes only, no opposite road along the same line) are mirrored about their carriageway's centre with a
laneOffset, and junction connecting roads get a cubic laneOffset that meets the lanes they link at both ends.
lane_link_gaps() measures how well linked lanes meet.

lane_check() compares lane counts on a seeded sample of OSM ways with the network converted from them, and where
their lanes lie against the ways (see there).
"""
from __future__ import annotations

import html
import math
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

_GL_X, _GL_W = np.polynomial.legendre.leggauss(5)


# ------------------------------------------------------------------------------------------------ model
@dataclass(slots=True)
class Geometry:
    s: float
    x: float
    y: float
    hdg: float
    length: float
    kind: str                                    # line, arc, spiral, poly3, paramPoly3
    params: dict = field(default_factory=dict)


@dataclass(slots=True)
class Lane:
    id: int
    type: str
    widths: list = field(default_factory=list)   # [(sOffset, a, b, c, d)]
    predecessor: int | None = None
    successor: int | None = None


@dataclass(slots=True)
class LaneSection:
    s: float
    left: list = field(default_factory=list)     # Lane, descending id
    right: list = field(default_factory=list)    # Lane, descending id (-1, -2, ...)


@dataclass(slots=True)
class Road:
    id: str
    name: str
    length: float
    junction: str
    rule: str | None
    predecessor: dict | None
    successor: dict | None
    geometries: list
    lane_offsets: list                           # [(s, a, b, c, d)]
    sections: list
    signals: int = 0
    signal_refs: int = 0

    @property
    def in_junction(self) -> bool:
        return self.junction not in ('-1', '', None)

    @property
    def lht(self) -> bool:
        return (self.rule or '').upper() == 'LHT'


@dataclass(slots=True)
class Connection:
    id: str
    incoming: str
    connecting: str
    contact: str
    lane_links: list                             # [(from, to)]


@dataclass(slots=True)
class Junction:
    id: str
    name: str
    connections: list


@dataclass
class OpenDrive:
    header: dict
    roads: list
    junctions: list
    signals: int = 0

    def road(self, rid: str) -> Road | None:
        if not hasattr(self, '_by_id'):
            self._by_id = {r.id: r for r in self.roads}
        return self._by_id.get(rid)


# ------------------------------------------------------------------------------------------------ parse
def _f(el, k, default=0.0) -> float:
    v = el.get(k)
    return float(v) if v not in (None, '') else default


def _link(el) -> dict | None:
    if el is None:
        return None
    return {'type': el.get('elementType'), 'id': el.get('elementId'), 'contact': el.get('contactPoint')}


def _lane(el) -> Lane:
    ln = Lane(int(el.get('id')), el.get('type', ''))
    ln.widths = [(_f(w, 'sOffset'), _f(w, 'a'), _f(w, 'b'), _f(w, 'c'), _f(w, 'd')) for w in el.findall('width')]
    lk = el.find('link')
    if lk is not None:
        p, s = lk.find('predecessor'), lk.find('successor')
        ln.predecessor = int(p.get('id')) if p is not None and p.get('id') not in (None, '') else None
        ln.successor = int(s.get('id')) if s is not None and s.get('id') not in (None, '') else None
    return ln


def _geometry(el) -> Geometry:
    g = Geometry(_f(el, 's'), _f(el, 'x'), _f(el, 'y'), _f(el, 'hdg'), _f(el, 'length'), 'line')
    for c in el:
        if c.tag == 'line':
            g.kind = 'line'
        elif c.tag == 'arc':
            g.kind, g.params = 'arc', {'curvature': _f(c, 'curvature')}
        elif c.tag == 'spiral':
            g.kind, g.params = 'spiral', {'curvStart': _f(c, 'curvStart'), 'curvEnd': _f(c, 'curvEnd')}
        elif c.tag == 'poly3':
            g.kind, g.params = 'poly3', {k: _f(c, k) for k in 'abcd'}
        elif c.tag == 'paramPoly3':
            g.kind = 'paramPoly3'
            g.params = {k: _f(c, k) for k in ('aU', 'bU', 'cU', 'dU', 'aV', 'bV', 'cV', 'dV')}
            g.params['pRange'] = c.get('pRange', 'normalized')
        else:
            continue
        break
    return g


def _road(el) -> Road:
    link = el.find('link')
    pv = el.find('planView')
    lanes = el.find('lanes')
    sections, offsets = [], []
    if lanes is not None:
        offsets = [(_f(o, 's'), _f(o, 'a'), _f(o, 'b'), _f(o, 'c'), _f(o, 'd')) for o in lanes.findall('laneOffset')]
        for sec in lanes.findall('laneSection'):
            ls = LaneSection(_f(sec, 's'))
            left, right = sec.find('left'), sec.find('right')
            ls.left = sorted((_lane(x) for x in left.findall('lane')) if left is not None else [], key=lambda x: -x.id)
            ls.right = sorted((_lane(x) for x in right.findall('lane')) if right is not None else [],
                              key=lambda x: -x.id)
            sections.append(ls)
    sig = el.find('signals')
    return Road(
        id=el.get('id', ''), name=el.get('name', ''), length=_f(el, 'length'), junction=el.get('junction', '-1'),
        rule=el.get('rule'),
        predecessor=_link(link.find('predecessor')) if link is not None else None,
        successor=_link(link.find('successor')) if link is not None else None,
        geometries=[_geometry(g) for g in pv.findall('geometry')] if pv is not None else [],
        lane_offsets=offsets, sections=sections,
        signals=len(sig.findall('signal')) if sig is not None else 0,
        signal_refs=len(sig.findall('signalReference')) if sig is not None else 0)


def parse(text: str | bytes) -> OpenDrive:
    """An OpenDRIVE document (text or bytes) as an OpenDrive."""
    root = ET.fromstring(text)
    if root.tag != 'OpenDRIVE':
        raise ValueError('Not an OpenDRIVE document')
    hdr = root.find('header')
    header = dict(hdr.attrib) if hdr is not None else {}
    if hdr is not None:
        geo = hdr.find('geoReference')
        header['geoReference'] = (geo.text or '').strip() if geo is not None else ''
        off = hdr.find('offset')
        header['offset'] = {k: _f(off, k) for k in ('x', 'y', 'z', 'hdg')} if off is not None else None
    roads = [_road(r) for r in root.findall('road')]
    junctions = []
    for j in root.findall('junction'):
        conns = [Connection(c.get('id', ''), c.get('incomingRoad', ''), c.get('connectingRoad', ''),
                            c.get('contactPoint', ''),
                            [(int(ll.get('from')), int(ll.get('to'))) for ll in c.findall('laneLink')])
                 for c in j.findall('connection')]
        junctions.append(Junction(j.get('id', ''), j.get('name', ''), conns))
    return OpenDrive(header, roads, junctions, sum(r.signals for r in roads))


# ------------------------------------------------------------------------------------------------ geometry
def _spiral(g: Geometry, ds: np.ndarray):
    k0 = g.params['curvStart']
    c = (g.params['curvEnd'] - k0) / g.length if g.length > 0 else 0.0

    def theta(u):
        return g.hdg + k0 * u + 0.5 * c * u * u

    dmax = max(abs(k0), abs(g.params['curvEnd']), 1e-9)
    piece = min(1.0, 0.05 / dmax)
    n = max(1, math.ceil(g.length / piece))
    grid = np.linspace(0.0, g.length, n + 1)
    h = np.diff(grid)

    def integrate(a, b):                         # ∫_a^b (cos θ, sin θ) du, 5-point Gauss–Legendre, vectorised
        mid, half = (a + b) / 2, (b - a) / 2
        u = mid[:, None] + half[:, None] * _GL_X[None, :]
        t = theta(u)
        return half * (np.cos(t) @ _GL_W), half * (np.sin(t) @ _GL_W)

    gx, gy = integrate(grid[:-1], grid[1:])
    cx, cy = np.concatenate(([0.0], np.cumsum(gx))), np.concatenate(([0.0], np.cumsum(gy)))
    i = np.clip(np.searchsorted(grid, ds, side='right') - 1, 0, max(len(h) - 1, 0))
    rx, ry = integrate(grid[i], ds)
    return g.x + cx[i] + rx, g.y + cy[i] + ry, theta(ds)


def _parametric(g: Geometry, ds: np.ndarray):
    p = g.params
    if g.kind == 'poly3':
        coef_u, coef_v = (0.0, 1.0, 0.0, 0.0), (p['a'], p['b'], p['c'], p['d'])
        pmax = g.length                          # arc length >= u, so u never exceeds length
    else:
        coef_u, coef_v = (p['aU'], p['bU'], p['cU'], p['dU']), (p['aV'], p['bV'], p['cV'], p['dV'])
        pmax = g.length if p.get('pRange', 'normalized') == 'arcLength' else 1.0
    n = max(64, math.ceil(g.length / 0.05))
    t = np.linspace(0.0, pmax, n + 1)

    def d1(cf, x):
        return cf[1] + 2 * cf[2] * x + 3 * cf[3] * x * x

    speed = np.hypot(d1(coef_u, t), d1(coef_v, t))
    arc = np.concatenate(([0.0], np.cumsum((speed[1:] + speed[:-1]) / 2 * np.diff(t))))
    if g.kind == 'poly3':
        target = np.clip(ds, 0.0, arc[-1])
    else:
        target = ds * (arc[-1] / g.length) if g.length > 0 else ds * 0
    q = np.interp(target, arc, t)
    u = coef_u[0] + coef_u[1] * q + coef_u[2] * q * q + coef_u[3] * q ** 3
    v = coef_v[0] + coef_v[1] * q + coef_v[2] * q * q + coef_v[3] * q ** 3
    ch, sh = math.cos(g.hdg), math.sin(g.hdg)
    return g.x + u * ch - v * sh, g.y + u * sh + v * ch, g.hdg + np.arctan2(d1(coef_v, q), d1(coef_u, q))


def eval_geometry(g: Geometry, ds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x, y, heading at distances ds (array) from the start of one planView geometry."""
    ds = np.clip(np.asarray(ds, float), 0.0, g.length)
    if g.kind == 'arc' and abs(g.params['curvature']) > 1e-12:
        k = g.params['curvature']
        h = g.hdg + k * ds
        return g.x + (np.sin(h) - math.sin(g.hdg)) / k, g.y - (np.cos(h) - math.cos(g.hdg)) / k, h
    if g.kind == 'spiral':
        return _spiral(g, ds)
    if g.kind in ('poly3', 'paramPoly3'):
        return _parametric(g, ds)
    return g.x + ds * math.cos(g.hdg), g.y + ds * math.sin(g.hdg), np.full_like(ds, g.hdg)


def sample_reference_line(road: Road, step: float = 2.0) -> np.ndarray:
    """(N, 3) x, y, heading along the road's reference line, at most `step` metres apart, ends included."""
    parts = []
    geoms = [g for g in road.geometries if g.length > 0]
    for i, g in enumerate(geoms):
        n = max(1, math.ceil(g.length / step))
        ds = np.linspace(0.0, g.length, n + 1)
        if i < len(geoms) - 1:
            ds = ds[:-1]
        x, y, h = eval_geometry(g, ds)
        parts.append(np.column_stack((x, y, h)))
    return np.concatenate(parts) if parts else np.zeros((0, 3))


def _cubic_at(entries, s: float, key=0) -> float:
    """The record in force at s (the last with start <= s) of [(start, a, b, c, d)], evaluated; 0 when none is."""
    best = None
    for e in entries:
        if e[key] <= s + 1e-9:
            best = e
    if best is None:
        return 0.0
    u = s - best[0]
    return best[1] + best[2] * u + best[3] * u * u + best[4] * u ** 3


def lane_centres(road: Road, step: float = 2.0, types=('driving',)) -> np.ndarray:
    """(N, 6) x, y, travel heading, lane id, width, s of the centres of the road's lanes of the given types, every
    `step` metres of reference line: the reference line moved by laneOffset and the lane widths. Lanes travel along +s
    when they are right lanes of a right-hand road or left lanes of a left-hand one."""
    ref = sample_reference_line(road, step)
    if not len(ref) or not road.sections:
        return np.zeros((0, 6))
    ss = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(ref[:, :2], axis=0).T))))
    if ss[-1] > 0 and road.length > 0:
        ss *= road.length / ss[-1]
    starts = [sec.s for sec in road.sections]
    rows = []
    for (x, y, h), s in zip(ref, ss):
        sec = road.sections[max(0, int(np.searchsorted(starts, s + 1e-9, side='right')) - 1)]
        ls = s - sec.s
        nx, ny = -math.sin(h), math.cos(h)
        for lanes, sign in ((sec.right, -1.0), (sorted(sec.left, key=lambda q: q.id), 1.0)):
            t = _cubic_at(road.lane_offsets, s)
            for ln in lanes:
                w = _cubic_at(ln.widths, ls)
                c = t + sign * w / 2.0
                t += sign * w
                if ln.type not in types or w <= 0.05:
                    continue
                along = (sign < 0) != road.lht
                rows.append((x + nx * c, y + ny * c, h if along else h + math.pi, ln.id, w, s))
    return np.array(rows, float).reshape(-1, 6)


def _rdp(xy: np.ndarray, tol: float) -> np.ndarray:
    """Indices kept by Ramer–Douglas–Peucker with tolerance tol (iterative)."""
    n = len(xy)
    if n <= 2:
        return np.arange(n)
    keep = np.zeros(n, bool)
    keep[[0, n - 1]] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        p, q = xy[a], xy[b]
        seg = q - p
        L = math.hypot(*seg)
        pts = xy[a + 1:b] - p
        d = np.abs(pts[:, 0] * seg[1] - pts[:, 1] * seg[0]) / L if L > 1e-9 else np.hypot(pts[:, 0], pts[:, 1])
        i = int(np.argmax(d))
        if d[i] > tol:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return np.flatnonzero(keep)


def _driving(lanes) -> int:
    return sum(1 for ln in lanes if ln.type == 'driving')


def _section_lengths(road: Road) -> list:
    ss = [s.s for s in road.sections] + [road.length]
    return [max(0.0, ss[i + 1] - ss[i]) for i in range(len(road.sections))]


def driving_lanes(road: Road) -> tuple[int, int]:
    """(left, right) driving lanes of the road's longest lane section."""
    if not road.sections:
        return 0, 0
    sec = road.sections[int(np.argmax(_section_lengths(road)))]
    return _driving(sec.left), _driving(sec.right)


def road_graph(od: OpenDrive, step: float = 2.0, tol: float = 0.2) -> dict:
    """A compact preview: each non-junction road's reference line (decimated to tol metres, coordinates to 1 cm) with
    its driving lanes left/right, length and name; junction connecting roads' lines separately; totals."""
    roads, conns = [], []
    hist: dict[int, int] = {}
    km = jkm = 0.0
    lo, hi = np.array([np.inf, np.inf]), np.array([-np.inf, -np.inf])
    for r in od.roads:
        pts = sample_reference_line(r, step)
        if len(pts) == 0:
            continue
        xy = pts[:, :2][_rdp(pts[:, :2], tol)]
        lo, hi = np.minimum(lo, xy.min(0)), np.maximum(hi, xy.max(0))
        item = {'id': r.id, 'x': np.round(xy[:, 0], 2).tolist(), 'y': np.round(xy[:, 1], 2).tolist()}
        if r.in_junction:
            item['junction'] = r.junction
            conns.append(item)
            jkm += r.length / 1000.0
            continue
        left, right = driving_lanes(r)
        item.update(name=html.unescape(r.name), length=round(r.length, 2), left=left, right=right,
                    rule='LHT' if r.lht else 'RHT')
        roads.append(item)
        km += r.length / 1000.0
        for n in (left, right):
            if n:
                hist[n] = hist.get(n, 0) + 1
    bounds = (round(float(lo[0]), 2), round(float(lo[1]), 2), round(float(hi[0]), 2), round(float(hi[1]), 2)) \
        if np.isfinite(lo).all() else None
    return {'roads': roads, 'connections': conns,
            'totals': {'roads': len(roads), 'road_km': round(km, 3), 'junctions': len(od.junctions),
                       'connecting_roads': len(conns), 'connecting_km': round(jkm, 3),
                       'driving_lanes_per_direction': dict(sorted(hist.items())), 'signals': od.signals,
                       'handedness': handedness(od), 'bounds': bounds}}


def handedness(od: OpenDrive) -> str:
    """'RHT' when no road says rule="LHT" (OpenDRIVE's default is right-hand), 'LHT' when every road does, else
    'mixed'."""
    rules = {'LHT' if r.lht else 'RHT' for r in od.roads}
    return rules.pop() if len(rules) == 1 else ('RHT' if not rules else 'mixed')


# ------------------------------------------------------------------------------------------------ right -> left
def _neg(v: str | None) -> str | None:
    """Negate a number kept as text (exact: no float round trip); zero stays as written."""
    if v is None:
        return None
    s = v.strip()
    try:
        if float(s) == 0:
            return s.lstrip('-+')
    except ValueError:
        raise ValueError(f'Not a number: {v!r}') from None
    return s[1:] if s.startswith('-') else '-' + s.lstrip('+')


def _neg_attrs(el, keys):
    for k in keys:
        if el.get(k) is not None:
            el.set(k, _neg(el.get(k)))


_CHANGE = {'increase': 'decrease', 'decrease': 'increase'}


def _mirror_section(sec, negate_pred, negate_succ, counts):
    left, center, right = sec.find('left'), sec.find('center'), sec.find('right')
    old = [(lt, el) for lt, el in (('left', left), ('right', right)) if el is not None]
    lanes = []
    for _, side in old:
        for ln in side.findall('lane'):
            ln.set('id', _neg(ln.get('id')))
            lk = ln.find('link')
            if lk is not None:
                for tag, neg in (('predecessor', negate_pred), ('successor', negate_succ)):
                    e = lk.find(tag)
                    if e is not None and e.get('id') is not None and neg:
                        e.set('id', _neg(e.get('id')))
                        counts['lane_links'] += 1
            for rm in ln.iter('roadMark'):
                if rm.get('laneChange') in _CHANGE:
                    rm.set('laneChange', _CHANGE[rm.get('laneChange')])
            lanes.append(ln)
            counts['lanes'] += 1
    for _, side in old:
        sec.remove(side)
    children = list(sec)
    ci = children.index(center) if center is not None else len(children)
    for name, sign, at in (('right', -1, ci + 1), ('left', 1, ci)):      # right first: inserting left shifts it
        group = sorted((ln for ln in lanes if int(ln.get('id')) * sign > 0), key=lambda ln: -int(ln.get('id')))
        if group:
            el = ET.SubElement(sec, name)
            sec.remove(el)
            el.extend(group)
            sec.insert(at, el)


def _shift_poly(rec, s0: float) -> tuple:
    """A cubic record (start, a, b, c, d) re-expanded about s0 (>= start): the same curve with start s0."""
    st, a, b, c, d = rec
    h = s0 - st
    return (s0, a + b * h + c * h * h + d * h ** 3, b + 2 * c * h + 3 * d * h * h, c + 3 * d * h, d)


def _pw_add(*parts) -> list:
    """The sum of piecewise cubics [(start, a, b, c, d)] (each record in force from its start to the next; zero
    before the first), as one piecewise cubic over the union of their breakpoints."""
    starts = sorted({r[0] for p in parts for r in p})
    out = []
    for s0 in starts:
        acc = [0.0, 0.0, 0.0, 0.0]
        for p in parts:
            rec = None
            for r in p:
                if r[0] <= s0 + 1e-9:
                    rec = r
            if rec is not None:
                q = _shift_poly(rec, s0)
                for i in range(4):
                    acc[i] += q[i + 1]
        if out and all(abs(x - y) < 1e-12 for x, y in zip(_shift_poly(out[-1], s0)[1:], acc)):
            continue                                       # the previous record already continues this way
        out.append((s0, *acc))
    return out


def _carriageway(road: Road, side: str) -> list:
    """The width of a road's carriageway on one side as a piecewise cubic over s: per lane section, the lanes from
    the centre out to the outermost driving lane (every lane when the section has no driving lane)."""
    out = []
    for k, sec in enumerate(road.sections):
        end = road.sections[k + 1].s if k + 1 < len(road.sections) else road.length
        lanes = sorted(sec.right if side == 'right' else sec.left, key=lambda q: abs(q.id))
        drive = [i for i, ln in enumerate(lanes) if ln.type == 'driving']
        lanes = lanes[:drive[-1] + 1] if drive else lanes
        pieces = [[(sec.s + so, a, b, c, d) for so, a, b, c, d in ln.widths if sec.s + so < end - 1e-9]
                  for ln in lanes]
        total = _pw_add(*pieces) if pieces else []
        if not total or total[0][0] > sec.s + 1e-9:
            total = [(sec.s, 0.0, 0.0, 0.0, 0.0)] + total
        out += total
    return out


def _ref_pose(road: Road, s: float) -> tuple[float, float, float]:
    """x, y, heading of a parsed road's reference line at s."""
    geoms = [g for g in road.geometries if g.length > 0] or road.geometries
    g = geoms[0]
    for cand in geoms:
        if cand.s <= s + 1e-9:
            g = cand
    x, y, h = eval_geometry(g, np.array([s - g.s]))
    return float(x[0]), float(y[0]), float(h[0])


def _lane_t(road: Road, s: float, lane_id: int) -> float | None:
    """Lateral position (left of the reference line +) of a lane's centre at s; None when the lane is not there."""
    if not road.sections:
        return None
    sec = road.sections[0]
    for c in road.sections:
        if c.s <= s + 1e-9:
            sec = c
    ls = s - sec.s
    t = _cubic_at(road.lane_offsets, s)
    lanes, sign = (sec.left, 1.0) if lane_id > 0 else (sec.right, -1.0)
    for ln in sorted(lanes, key=lambda q: abs(q.id)):
        w = _cubic_at(ln.widths, ls)
        if ln.id == lane_id:
            return t + sign * w / 2.0
        t += sign * w
    return None


def _lane_xy(road: Road, s: float, lane_id: int):
    t = _lane_t(road, s, lane_id)
    if t is None:
        return None
    x, y, h = _ref_pose(road, s)
    return np.array([x - math.sin(h) * t, y + math.cos(h) * t])


def lane_link_gaps(od: OpenDrive) -> np.ndarray:
    """For every lane link between roads (a lane's predecessor or successor on the road its road links to), the
    distance in metres between the two lanes' centres where they meet: 0 for a network whose lanes join up."""
    gaps = []
    for r in od.roads:
        if not r.sections:
            continue
        for link, s_here, sec in ((r.predecessor, 0.0, r.sections[0]), (r.successor, r.length, r.sections[-1])):
            other = od.road(link['id']) if link and link.get('type') == 'road' else None
            if other is None:
                continue
            s_other = 0.0 if link.get('contact') == 'start' else other.length
            for ln in sec.left + sec.right:
                to = ln.predecessor if s_here == 0.0 else ln.successor
                if to is None:
                    continue
                p, q = _lane_xy(other, s_other, to), _lane_xy(r, s_here, ln.id)
                if p is not None and q is not None:
                    gaps.append(float(np.hypot(*(p - q))))
    return np.array(gaps)


def _paired(od: OpenDrive, tol: float = 0.3, step: float = 2.0) -> set:
    """Ids of the non-junction roads whose reference line another non-junction road runs along in the opposite
    direction for at least half its length: the two directions (or a direction and its sidewalk) of one OSM way,
    whose shared reference line is the way itself. carla.Osm2Odr (SUMO) writes a lone one-way road's reference line
    at its carriageway's left edge instead."""
    roads = [r for r in od.roads if not r.in_junction]
    pts = {r.id: sample_reference_line(r, step) for r in roads}
    cell = 10.0
    grid: dict = {}
    for r in roads:
        for key in {(int(math.floor(x / cell)), int(math.floor(y / cell))) for x, y, _ in pts[r.id]}:
            grid.setdefault(key, set()).add(r.id)
    by_id = {r.id: r for r in roads}
    out = set()
    for r in roads:
        a = pts[r.id]
        if len(a) < 2:
            continue
        near = set()
        for x, y, _ in a:
            i, j = int(math.floor(x / cell)), int(math.floor(y / cell))
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    near |= grid.get((i + di, j + dj), set())
        for q in near - {r.id}:
            b = pts[q]
            if len(b) < 2 or by_id[q].in_junction:
                continue
            d, u = _seg_dist(a[:, :2], b[:, :2])
            dot = np.cos(a[:, 2]) * u[:, 0] + np.sin(a[:, 2]) * u[:, 1]
            if ((d <= tol) & (dot < -0.9)).mean() >= 0.5:
                out.add(r.id)
                break
    return out


def _fmt(v: float) -> str:
    v = 0.0 if abs(v) < 1e-12 else v
    return f'{v:.10g}'


def _set_offsets(road_el, records: list):
    """Replace a road element's laneOffset records (they come first inside <lanes>)."""
    lanes = road_el.find('lanes')
    for lo in lanes.findall('laneOffset'):
        lanes.remove(lo)
    for i, (s, a, b, c, d) in enumerate(records):
        el = ET.Element('laneOffset', {'s': _fmt(s), 'a': _fmt(a), 'b': _fmt(b), 'c': _fmt(c), 'd': _fmt(d)})
        lanes.insert(i, el)


def _one_side(road: Road) -> str | None:
    """'right' or 'left' when every lane section has lanes on that side only, else None."""
    sides = {('right' if s.right else '') + ('left' if s.left else '') for s in road.sections}
    return sides.pop() if len(sides) == 1 and next(iter(sides)) in ('right', 'left') else None


def _mirror_line(road: Road, side: str, ways, grid, cell: float, paired: set, step: float = 2.0):
    """Where a one-sided road should be mirrored for left-hand traffic, as 'ref' (its reference line) or 'centre'
    (its carriageway's centre), and why. With OSM ways (to_lht's `ways`): whichever of the two the nearest parallel
    way runs along (median within 0.75 m; the way is what both conversions are faithful to). Without one: 'ref' when
    an opposite road shares the reference line (the two directions of one way), else 'centre'."""
    fallback = ('ref', 'shares its reference line with an opposite road') if road.id in paired else \
        ('centre', 'a lone one-way road: its reference line is its carriageway\'s edge')
    if not ways:
        return fallback
    ref = sample_reference_line(road, step)
    if len(ref) < 2:
        return fallback
    keys = {(int(math.floor(x / cell)), int(math.floor(y / cell))) for x, y, _ in ref}
    cands = set()
    for i, j in keys:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                cands |= grid.get((i + di, j + dj), set())
    ss = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(ref[:, :2], axis=0).T))))
    if ss[-1] > 0:
        ss *= road.length / ss[-1]
    sign = -1.0 if side == 'right' else 1.0
    centre = np.array([_cubic_at(road.lane_offsets, s) + sign * _cubic_at(_carriageway(road, side), s) / 2.0
                       for s in ss])
    nrm = np.column_stack((-np.sin(ref[:, 2]), np.cos(ref[:, 2])))
    best = None
    for k in cands:
        poly = ways[k]
        d, u = _seg_dist(ref[:, :2], poly)
        par = np.abs(np.cos(ref[:, 2]) * u[:, 0] + np.sin(ref[:, 2]) * u[:, 1]) > 0.95
        a, b = poly[:-1], poly[1:]
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(1), 1e-12)
        ap = ref[:, None, :2] - a[None]
        t = np.clip((ap * ab[None]).sum(2) / L2[None], 0.0, 1.0)
        foot = a[None] + t[..., None] * ab[None]
        kk = np.hypot(*(ref[:, None, :2] - foot).transpose(2, 0, 1)).argmin(1)
        lat = ((foot[np.arange(len(ref)), kk] - ref[:, :2]) * nrm).sum(1)      # the way, left of the ref line +
        ok = par & (d < 15.0)
        if ok.mean() < 0.6:
            continue
        e_ref, e_centre = float(np.median(np.abs(lat[ok]))), float(np.median(np.abs(lat[ok] - centre[ok])))
        e = min(e_ref, e_centre)
        if best is None or e < best[0]:
            best = (e, 'ref' if e_ref <= e_centre else 'centre')
    if best is None or best[0] > 0.75:
        return fallback
    return best[1], f'the OSM way runs along its {"reference line" if best[1] == "ref" else "carriageway's centre"}'


def osm_lines(osm_data, proj, transform=None) -> list:
    """The polylines (OpenDRIVE x, y) of the OSM highways carla.Osm2Odr converts, for to_lht's `ways`."""
    from . import osm as O
    to_local = proj.to_local if hasattr(proj, 'to_local') else proj
    tf = _transformer(transform)
    out = []
    for w in osm_data.ways:
        if w.get('kind') != 'highway' or w['tags'].get('highway') not in O.CARLA_HIGHWAYS:
            continue
        nodes = [n for n in w['nodes'] if n in osm_data.nodes]
        if len(nodes) < 2:
            continue
        lat = np.array([osm_data.nodes[n][0] for n in nodes])
        lon = np.array([osm_data.nodes[n][1] for n in nodes])
        x, y = tf(*to_local(lat, lon))
        out.append(np.column_stack((np.asarray(x, float), np.asarray(y, float))))
    return out


def _place(root, original: str, conv: set, ways=None) -> dict:
    """After mirroring (to_lht), put lanes where left-hand traffic needs them. Mirroring across the reference line is
    right for a road whose reference line is its OSM way (both directions of a two-way way share it), but a lone
    one-way road's reference line is its carriageway's edge: mirrored there, the carriageway would move a whole
    carriageway width sideways. Such roads (_mirror_line) get a laneOffset that mirrors them about their
    carriageway's centre instead: the carriageway stays where it was. Junction connecting roads then get a
    laneOffset, a smooth cubic from one end to the other, that makes their lanes meet the lanes they link to at both
    ends (a connector built for right-hand traffic, mirrored about its own reference line, would otherwise not).
    Objects and signals move with the laneOffset added. Returns counts."""
    od0 = parse(original)
    paired = _paired(od0)
    cell = 20.0
    grid: dict = {}
    for k, poly in enumerate(ways or []):
        dense = np.concatenate([np.linspace(a, b, max(2, int(math.ceil(np.hypot(*(b - a)) / (cell / 2))) + 1))
                                for a, b in zip(poly[:-1], poly[1:])]) if len(poly) > 1 else poly
        for key in {(int(math.floor(x / cell)), int(math.floor(y / cell))) for x, y in dense}:
            grid.setdefault(key, set()).add(k)
    els = {r.get('id'): r for r in root.findall('road')}
    corr: dict = {}
    why: dict = {}
    for r in od0.roads:
        if r.id not in conv or r.in_junction:
            continue
        side = _one_side(r)
        if side is None:
            continue
        line, reason = _mirror_line(r, side, ways, grid, cell, paired)
        why[reason] = why.get(reason, 0) + 1
        if line == 'ref':
            continue
        sign = -1.0 if side == 'right' else 1.0       # reflection about the centre m = o ∓ W/2: t -> 2m - t
        corr[r.id] = _pw_add([(s, 2 * a, 2 * b, 2 * c, 2 * d) for s, a, b, c, d in r.lane_offsets],
                             [(s, sign * a, sign * b, sign * c, sign * d) for s, a, b, c, d in _carriageway(r, side)])
    for rid, c in corr.items():
        _set_offsets(els[rid], _pw_add(parse_offsets(els[rid]), c))
    kept = len(corr)
    od1 = parse(ET.tostring(root, encoding='unicode'))
    moved = 0
    for r in od1.roads:
        if r.id not in conv or not r.in_junction or not r.sections:
            continue
        ends = []
        for link, s_here, sec in ((r.predecessor, 0.0, r.sections[0]), (r.successor, r.length, r.sections[-1])):
            other = od1.road(link['id']) if link and link.get('type') == 'road' else None
            vals = []
            if other is not None:
                s_other = 0.0 if link.get('contact') == 'start' else other.length
                x, y, h = _ref_pose(r, s_here)
                n = np.array([-math.sin(h), math.cos(h)])
                for ln in sec.left + sec.right:
                    to = ln.predecessor if s_here == 0.0 else ln.successor
                    if to is None:
                        continue
                    p, q = _lane_xy(other, s_other, to), _lane_xy(r, s_here, ln.id)
                    if p is not None and q is not None:
                        vals.append(float((p - q) @ n))
            ends.append(float(np.mean(vals)) if vals else None)
        c0, c1 = ends
        if c0 is None and c1 is None:
            continue
        c0 = c1 if c0 is None else c0
        c1 = c0 if c1 is None else c1
        if max(abs(c0), abs(c1)) < 1e-3:
            continue
        L = max(r.length, 1e-6)
        dc = c1 - c0
        corr[r.id] = [(0.0, c0, 0.0, 3 * dc / L ** 2, -2 * dc / L ** 3)]
        _set_offsets(els[r.id], _pw_add(r.lane_offsets, corr[r.id]))
        moved += 1
    for rid, c in corr.items():                   # objects and signals are placed by t, which laneOffset does not move
        el = els[rid]
        for ob in el.findall('signals/signal') + el.findall('signals/signalReference') + \
                el.findall('objects/object') + el.findall('objects/objectReference'):
            if ob.get('t') is not None:
                ob.set('t', _fmt(float(ob.get('t')) + _cubic_at(c, float(ob.get('s', '0')))))
            for cr in ob.iter('cornerRoad'):
                if cr.get('t') is not None:
                    cr.set('t', _fmt(float(cr.get('t')) + _cubic_at(c, float(cr.get('s', '0')))))
    return {'kept': kept, 'moved': moved, 'why': why}


def parse_offsets(road_el) -> list:
    """A road element's laneOffset records [(s, a, b, c, d)]."""
    lanes = road_el.find('lanes')
    return [(_f(o, 's'), _f(o, 'a'), _f(o, 'b'), _f(o, 'c'), _f(o, 'd'))
            for o in (lanes.findall('laneOffset') if lanes is not None else [])]


def to_lht(text: str | bytes, force: bool = False, ways=None) -> tuple[str, list]:
    """A right-hand-traffic OpenDRIVE document converted to left-hand traffic (see the module docstring) and notes
    saying what was changed. A file whose roads all say rule="LHT" is refused (ValueError), so converting twice is
    an error, never a silent flip back. A mixed file is refused unless force=True, which converts only the roads not
    already LHT (lane links into an already-LHT road are left as they are). `ways` (osm_lines(): the OSM highways
    in OpenDRIVE coordinates) lets each one-way road be mirrored about the line its OSM way runs along; without it,
    whether an opposite road shares its reference line decides (_mirror_line)."""
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    m = re.search(r'<OpenDRIVE[\s>]', text)
    if not m:
        raise ValueError('Not an OpenDRIVE document')
    prolog = text[:m.start()]
    root = ET.fromstring(text, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    roads = root.findall('road')
    lht = {r.get('id') for r in roads if (r.get('rule') or '').upper() == 'LHT'}
    if roads and len(lht) == len(roads):
        raise ValueError('Every road is already left-hand traffic (rule="LHT"); refusing to convert again')
    if lht and not force:
        raise ValueError(f'{len(lht)} of {len(roads)} roads are already left-hand traffic (mixed file); '
                         'pass force=True to convert only the others')
    if root.find('road/lateralProfile/shape') is not None:
        raise ValueError('lateralProfile <shape> cannot be mirrored by this converter')
    ids = {r.get('id') for r in roads}
    conv = ids - lht
    counts = dict.fromkeys(('roads', 'lanes', 'lane_links', 'lane_offsets', 'lane_link_junction', 'signals',
                            'orientations', 'objects', 'superelevations'), 0)

    def negate_for(road, tag):
        """Negate lane links toward the predecessor/successor when that road is converted (or, for a junction or a
        missing element, when this road is)."""
        lk = road.find(f'link/{tag}')
        if lk is not None and lk.get('elementType') == 'road' and lk.get('elementId') in ids:
            return lk.get('elementId') in conv
        return road.get('id') in conv

    for road in roads:
        if road.get('id') not in conv:
            continue
        counts['roads'] += 1
        road.set('rule', 'LHT')
        lanes = road.find('lanes')
        if lanes is not None:
            for lo in lanes.findall('laneOffset'):
                _neg_attrs(lo, 'abcd')
                counts['lane_offsets'] += 1
            np_, ns = negate_for(road, 'predecessor'), negate_for(road, 'successor')
            for sec in lanes.findall('laneSection'):
                _mirror_section(sec, np_, ns, counts)
        for se in road.findall('lateralProfile/superelevation'):
            _neg_attrs(se, 'abcd')
            counts['superelevations'] += 1
        for sig in road.findall('signals/signal') + road.findall('signals/signalReference'):
            _neg_attrs(sig, ('t',))
            o = sig.get('orientation')          # the lanes a signal applies to are mirrored: so is its side
            if o in ('+', '-'):
                sig.set('orientation', '-' if o == '+' else '+')
                counts['orientations'] += 1
            for v in sig.findall('validity'):
                f, t = v.get('fromLane'), v.get('toLane')
                v.set('fromLane', _neg(t))
                v.set('toLane', _neg(f))
            counts['signals'] += 1
        for ob in road.findall('objects/object') + road.findall('objects/objectReference'):
            _neg_attrs(ob, ('t', 'hdg'))
            for c in ob.iter('cornerLocal'):
                _neg_attrs(c, ('v',))
            for c in ob.iter('cornerRoad'):
                _neg_attrs(c, ('t',))
            for v in ob.findall('validity'):
                f, t = v.get('fromLane'), v.get('toLane')
                v.set('fromLane', _neg(t))
                v.set('toLane', _neg(f))
            counts['objects'] += 1
    for j in root.findall('junction'):
        for c in j.findall('connection'):
            nf, nt = c.get('incomingRoad') in conv, c.get('connectingRoad') in conv
            for ll in c.findall('laneLink'):
                if nf:
                    ll.set('from', _neg(ll.get('from')))
                if nt:
                    ll.set('to', _neg(ll.get('to')))
                counts['lane_link_junction'] += 1
    placed = _place(root, text, conv, ways)
    ET.indent(root, space='    ')
    body = ET.tostring(root, encoding='unicode')
    body = re.sub(r'<geoReference>(.*?)</geoReference>',
                  lambda mm: '<geoReference><![CDATA[' + html.unescape(mm.group(1)) + ']]></geoReference>'
                  if ']]>' not in html.unescape(mm.group(1)) else mm.group(0), body, count=1, flags=re.S)
    if not prolog.strip():
        prolog = '<?xml version="1.0" encoding="UTF-8"?>\n'
    notes = [
        f'{counts["roads"]} roads set to rule="LHT" with their lanes mirrored across the reference line '
        f'({counts["lanes"]} lanes; travel direction along each road unchanged)',
        f'{counts["lane_offsets"]} laneOffsets negated; {counts["lane_links"]} lane links and '
        f'{counts["lane_link_junction"]} junction laneLinks negated',
        f'{counts["signals"]} signals/signal references: t negated, validity swapped and negated, orientation '
        f'flipped on {counts["orientations"]} (positionInertial and hOffset kept)',
        f'{counts["objects"]} objects mirrored (t, hdg, cornerLocal v, cornerRoad t); '
        f'{counts["superelevations"]} superelevations negated',
        f'{placed["kept"]} one-way roads kept in place, mirrored about their carriageway\'s centre by a laneOffset '
        f'(the converter draws a lone one-way road\'s reference line at its carriageway\'s edge); one-sided roads: '
        + '; '.join(f'{v} {k}' for k, v in sorted(placed['why'].items())) +
        f'; {placed["moved"]} junction connecting roads given a laneOffset that meets the lanes they link at both '
        f'ends',
        'Limit: on approaches with two or more lanes per direction, which lane turns where is the right-hand '
        'converter\'s assignment mirrored (the inner lane keeps the turns it had), not a true left-hand assignment; '
        'single-lane approaches are unaffected.',
    ]
    if lht:
        notes.insert(1, f'{len(lht)} roads were already LHT and were left unchanged (force=True)')
    return prolog + body + '\n', notes


# ------------------------------------------------------------------------------------------------ lane check
def _seg_dist(pts: np.ndarray, poly: np.ndarray):
    """Distance from each point to a polyline and the unit direction of the nearest segment."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    ap = pts[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None]).sum(2) / L2[None], 0.0, 1.0)
    d = np.hypot(*(ap - t[..., None] * ab[None]).transpose(2, 0, 1))
    k = d.argmin(1)
    u = ab[k] / np.sqrt(L2[k])[:, None]
    return d[np.arange(len(pts)), k], u


def _norm_name(s: str) -> str:
    return html.unescape(html.unescape(s or '')).strip().casefold()     # SUMO escapes names twice (&amp;apos;)


def _transformer(transform):
    if transform is None:
        return lambda e, n: (e, n)
    if callable(transform):
        return transform
    dx, dy = transform
    return lambda e, n: (np.asarray(e) + dx, np.asarray(n) + dy)


def _position_errors(road: Road, poly: np.ndarray, one_way: int, tol: float, step: float) -> list:
    """[(error, lane width)] per s sample and travel direction of a road matched to an OSM way: how far the centre of
    the road's driving lanes in that direction is from where OSM puts it. OSM draws a one-way way along its
    carriageway's middle and a two-way way between the directions, so the expected centre is on the way (one-way) or
    half the direction's driving width out on its driving side (two-way: left for LHT, right for RHT). one_way: 0 for
    a two-way way, 1 or -1 (OSM oneway) for a one-way one, whose lanes in the other direction belong to another way
    (a dual carriageway's other half) and are left out."""
    lc = lane_centres(road, step)
    if not len(lc):
        return []
    d, u = _seg_dist(lc[:, :2], poly)
    a, b = poly[:-1], poly[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    ap = lc[:, None, :2] - a[None]
    t = np.clip((ap * ab[None]).sum(2) / L2[None], 0.0, 1.0)
    foot = a[None] + t[..., None] * ab[None]
    k = np.hypot(*(lc[:, None, :2] - foot).transpose(2, 0, 1)).argmin(1)
    v = lc[:, :2] - foot[np.arange(len(lc)), k]
    along = np.sign(np.cos(lc[:, 2]) * u[:, 0] + np.sin(lc[:, 2]) * u[:, 1])
    lateral = np.sign(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) * along * d     # left of the lane's travel +
    out = []
    keep = d <= tol + 15.0
    for s in np.unique(lc[keep, 5]):
        for dirn in ((1.0, -1.0) if not one_way else (float(one_way),)):
            m = keep & (lc[:, 5] == s) & (along == dirn)
            if not m.any():
                continue
            w = lc[m, 4]
            centre = float((lateral[m] * w).sum() / w.sum())
            expected = 0.0 if one_way else (w.sum() / 2.0) * (1.0 if road.lht else -1.0)
            out.append((abs(centre - expected), float(w.mean())))
    return out


def lane_check(od: OpenDrive, osm_data, proj, sample: int = 50, seed: int = 0, transform=None,
               tol: float = 8.0, step: float = 2.0, ways=None) -> dict:
    """Whether the road network matches OSM's lane tags on a checked SAMPLE of ways (not the whole map).

    Eligible ways: highways of the types carla.Osm2Odr converts, with a name and lane tags that give
    expected_lanes(). `sample` of them are drawn with random.Random(seed). For each, the OSM nodes are projected
    with proj (a Projection from osm.local_projection, or a callable (lat, lon) -> (east, north)), then through
    `transform` (None, an (dx, dy) offset added to east/north, or a callable) into OpenDRIVE x, y: with
    proj_string(lat0, lon0) and center_map=False OpenDRIVE x = east, y = north, so no transform is needed.
    The way's roads are the non-junction roads with the same name (HTML entities decoded, case-folded) whose
    reference-line samples lie within tol metres of the way for at least half their length. A road's direction is
    the sign of its heading's dot product with the nearest way segment; lanes that travel along +s (right lanes for
    RHT roads, left lanes for LHT) count forward when the road runs along the way and backward otherwise, and lanes
    along -s the opposite. Per direction the length-weighted most common count over the way's roads is "found"
    (0 when no road carries that direction).

    A way is comparable when at least one road was found; matched when found == expected. agreement = matched /
    comparable. `position`: where the matched roads' driving lanes lie against the way (_position_errors): the
    median and 95th percentile error over every sample, and the ways where most samples are off by more than 0.75
    of a lane width (misplaced).

    ways: check exactly these way ids instead of a sample (the ways a map edit changed): every one of them whose tags
    give expected_lanes(), named or not — an unnamed way is matched to unnamed roads near it."""
    from . import osm as O
    to_local = proj.to_local if hasattr(proj, 'to_local') else proj
    tf = _transformer(transform)
    by_name: dict[str, list] = {}
    for r in od.roads:
        if not r.in_junction and r.name:
            by_name.setdefault(_norm_name(r.name), []).append(r)
    samples: dict[str, np.ndarray] = {}
    if ways is not None:
        want = {int(i) for i in ways}
        eligible = sorted((w for w in osm_data.ways if int(w['id']) in want and w['kind'] == 'highway'
                           and w['tags'].get('highway') in O.CARLA_HIGHWAYS and O.expected_lanes(w) is not None),
                          key=lambda w: w['id'])
        chosen = list(eligible)
        for r in od.roads:
            if not r.in_junction and not r.name:
                by_name.setdefault('', []).append(r)
    else:
        eligible = sorted((w for w in osm_data.ways if w['kind'] == 'highway' and w['tags'].get('name')
                           and w['tags'].get('highway') in O.CARLA_HIGHWAYS and O.expected_lanes(w) is not None),
                          key=lambda w: w['id'])
        chosen = random.Random(seed).sample(eligible, min(sample, len(eligible)))
    rows, all_errs = [], []
    for w in sorted(chosen, key=lambda w: w['id']):
        exp = O.expected_lanes(w)
        row = {'way': w['id'], 'name': w['tags'].get('name', ''), 'highway': w['tags'].get('highway'),
               'expected': exp, 'found': None, 'roads': [], 'matched': None, 'reason': None}
        rows.append(row)
        cands = by_name.get(_norm_name(w['tags'].get('name', '')), [])
        if not cands:
            row['reason'] = 'no OpenDRIVE road with this name'
            continue
        lat = np.array([osm_data.nodes[n][0] for n in w['nodes']])
        lon = np.array([osm_data.nodes[n][1] for n in w['nodes']])
        x, y = tf(*to_local(lat, lon))
        poly = np.column_stack((np.asarray(x, float), np.asarray(y, float)))
        votes = {1: {}, -1: {}}                  # direction -> {lane count: metres}
        one_way = O.oneway(w['tags']) if O.oneway(w['tags']) in (1, -1) else 0
        errs = []
        for r in cands:
            pts = samples.get(r.id)
            if pts is None:
                pts = samples[r.id] = sample_reference_line(r, step)
            if len(pts) < 2:
                continue
            d, u = _seg_dist(pts[:, :2], poly)
            near = d <= tol
            if near.mean() < 0.5:
                continue
            dot = (np.cos(pts[near, 2]) * u[near, 0] + np.sin(pts[near, 2]) * u[near, 1]).mean()
            sgn = 1 if dot > 0 else -1
            left, right = driving_lanes(r)
            plus, minus = (left, right) if r.lht else (right, left)
            metres = float(near.sum()) * step
            for direction, n in ((sgn, plus), (-sgn, minus)):
                if n:
                    votes[direction][n] = votes[direction].get(n, 0.0) + metres
            row['roads'].append(r.id)
            errs += _position_errors(r, poly, one_way, tol, step)
        row['position_error_m'] = round(float(np.median([e for e, _ in errs])), 3) if errs else None
        row['misplaced'] = bool(errs) and float(np.median([e > 0.75 * lw for e, lw in errs])) > 0.5
        all_errs += errs
        if not row['roads']:
            row['reason'] = f'no road with this name within {tol:g} m of the way for half its length'
            continue
        found = tuple(max(votes[dn].items(), key=lambda kv: (kv[1], -kv[0]))[0] if votes[dn] else 0 for dn in (1, -1))
        row['found'] = found
        row['matched'] = found == tuple(exp)
    comparable = [r for r in rows if r['found'] is not None]
    matched = sum(1 for r in comparable if r['matched'])
    e = np.array([x for x, _ in all_errs])
    position = {
        'median_m': round(float(np.median(e)), 3) if len(e) else None,
        'p95_m': round(float(np.percentile(e, 95)), 3) if len(e) else None,
        'checked': sum(1 for r in comparable if r['position_error_m'] is not None),
        'misplaced': sum(1 for r in comparable if r['misplaced']),
        'method': 'per comparable way, every 2 m of each matched road: the centre of its driving lanes in each '
                  'direction against where OSM puts them (on the way for a one-way way; beside it on the driving '
                  'side, half that direction\'s lanes\' width out, for a two-way way); a way is misplaced when most '
                  'of its samples are off by more than 0.75 of a lane width',
    }
    return {
        'rows': rows, 'eligible': len(eligible), 'sampled': len(rows), 'comparable': len(comparable),
        'matched': matched, 'agreement': (matched / len(comparable)) if comparable else None, 'seed': seed,
        'method': (f'the {len(rows)} given ways with lane tags' if ways is not None else
                   f'a seeded sample (seed {seed}) of {len(rows)} of {len(eligible)} named, lane-tagged OSM ways')
                  + '; '
                  f'comparable = a same-named road within {tol:g} m along the way; per-direction driving-lane '
                  f'counts compared with OSM expected_lanes',
        'position': position,
    }


def signal_lanes(text: str | bytes) -> dict:
    """Signals and signal references with an orientation ("+" or "-") whose side of their road has no driving lane:
    CARLA 0.9.16 gives such a traffic light no lane to control, so traffic drives through it. The side: "+" the lanes
    with positive ids, "-" the negative ones — measured on a real server (a right-hand road's lane -1 with "-", and
    after the left-hand mirror lane +1 with "+", control their lanes; lane +1 with "-" controls nothing)."""
    root = ET.fromstring(text.encode() if isinstance(text, str) else text)
    total = none = 0
    for road in root.iter('road'):
        ids = [int(ln.get('id')) for ln in road.iter('lane') if ln.get('type') == 'driving']
        for sig in road.findall('signals/signal') + road.findall('signals/signalReference'):
            o = sig.get('orientation')
            if o not in ('+', '-'):
                continue
            total += 1
            if not any((i > 0) if o == '+' else (i < 0) for i in ids):
                none += 1
    return {'oriented': total, 'no_lane': none,
            'method': 'signals and signal references with an orientation whose side of the road has no driving lane '
                      '("+": positive lane ids, "-": negative; measured on CARLA 0.9.16)'}


SANITIZE_TOL_M = 0.05


def sanitize(text: str | bytes) -> tuple[str, list]:
    """Positions a road's objects and signals must keep for CARLA: s within [0, the road's length].

    CARLA 0.9.16's generate_opendrive_world ends the server (signal 11, "Exception thrown: s <= road->GetLength()")
    when an object lies past its road's end. carla.Osm2Odr (SUMO) writes some crosswalks a few millimetres past it
    (measured: a crosswalk at s = 29.933 m on a road 29.930 m long, central Bengaluru). Positions out by at most
    SANITIZE_TOL_M are clamped onto the road and each is reported; anything further out is refused (ValueError), since
    moving it would misplace it."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(text if isinstance(text, (bytes, bytearray)) else text.encode('utf-8'))
    notes, clamped = [], 0
    for road in root.findall('road'):
        length = float(road.get('length', '0'))
        for el in list(road.findall('objects/object')) + list(road.findall('signals/signal')) + \
                list(road.findall('signals/signalReference')):
            s = float(el.get('s', '0'))
            if 0.0 <= s <= length:
                continue
            new = min(max(s, 0.0), length)
            if abs(new - s) > SANITIZE_TOL_M:
                raise ValueError(f'road {road.get("id")}: {el.tag} {el.get("id")} at s = {s} lies {abs(new - s):.3f} m '
                                 f'outside the road (length {length}); refusing to move it')
            el.set('s', f'{new:.8f}')
            clamped += 1
            notes.append(f'road {road.get("id")} ({road.get("name") or "unnamed"}): {el.tag} {el.get("id") or ""} '
                         f'moved from s = {s} to the road\'s end {new:.4f} ({1000 * abs(new - s):.1f} mm; CARLA crashes '
                         'on positions past a road\'s end)')
    if not clamped:
        return (text.decode('utf-8') if isinstance(text, (bytes, bytearray)) else text), []
    return ET.tostring(root, encoding='unicode', xml_declaration=True), notes
