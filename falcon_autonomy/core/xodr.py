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
and their validity fromLane/toLane swapped and negated (from <= to still holds), objects' t, hdg, cornerLocal v and
cornerRoad t are negated. Each road's travel direction is unchanged; only its side changes. Signal positionInertial,
hOffset and orientation are kept (SUMO writes t=0 everywhere, so its signals do not move). lateralProfile <shape> is
refused. Known limit: on approaches with two or more lanes per direction, which lane turns where is the right-hand
converter's assignment mirrored (the inner lane keeps the turns it had), not a true left-hand assignment;
single-lane approaches are unaffected.

lane_check() compares lane counts on a seeded sample of OSM ways with the network converted from them (see there).
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


def to_lht(text: str | bytes, force: bool = False) -> tuple[str, list]:
    """A right-hand-traffic OpenDRIVE document converted to left-hand traffic (see the module docstring) and notes
    saying what was changed. A file whose roads all say rule="LHT" is refused (ValueError), so converting twice is
    an error, never a silent flip back. A mixed file is refused unless force=True, which converts only the roads not
    already LHT (lane links into an already-LHT road are left as they are)."""
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
                            'objects', 'superelevations'), 0)

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
        f'{counts["signals"]} signals/signal references: t negated, validity swapped and negated (positionInertial, '
        f'hOffset and orientation kept)',
        f'{counts["objects"]} objects mirrored (t, hdg, cornerLocal v, cornerRoad t); '
        f'{counts["superelevations"]} superelevations negated',
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


def lane_check(od: OpenDrive, osm_data, proj, sample: int = 50, seed: int = 0, transform=None,
               tol: float = 8.0, step: float = 2.0) -> dict:
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
    comparable."""
    from . import osm as O
    to_local = proj.to_local if hasattr(proj, 'to_local') else proj
    tf = _transformer(transform)
    by_name: dict[str, list] = {}
    for r in od.roads:
        if not r.in_junction and r.name:
            by_name.setdefault(_norm_name(r.name), []).append(r)
    samples: dict[str, np.ndarray] = {}
    eligible = sorted((w for w in osm_data.ways if w['kind'] == 'highway' and w['tags'].get('name')
                       and w['tags'].get('highway') in O.CARLA_HIGHWAYS and O.expected_lanes(w) is not None),
                      key=lambda w: w['id'])
    chosen = random.Random(seed).sample(eligible, min(sample, len(eligible)))
    rows = []
    for w in sorted(chosen, key=lambda w: w['id']):
        exp = O.expected_lanes(w)
        row = {'way': w['id'], 'name': w['tags']['name'], 'highway': w['tags'].get('highway'), 'expected': exp,
               'found': None, 'roads': [], 'matched': None, 'reason': None}
        rows.append(row)
        cands = by_name.get(_norm_name(w['tags']['name']), [])
        if not cands:
            row['reason'] = 'no OpenDRIVE road with this name'
            continue
        lat = np.array([osm_data.nodes[n][0] for n in w['nodes']])
        lon = np.array([osm_data.nodes[n][1] for n in w['nodes']])
        x, y = tf(*to_local(lat, lon))
        poly = np.column_stack((np.asarray(x, float), np.asarray(y, float)))
        votes = {1: {}, -1: {}}                  # direction -> {lane count: metres}
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
        if not row['roads']:
            row['reason'] = f'no road with this name within {tol:g} m of the way for half its length'
            continue
        found = tuple(max(votes[dn].items(), key=lambda kv: (kv[1], -kv[0]))[0] if votes[dn] else 0 for dn in (1, -1))
        row['found'] = found
        row['matched'] = found == tuple(exp)
    comparable = [r for r in rows if r['found'] is not None]
    matched = sum(1 for r in comparable if r['matched'])
    return {
        'rows': rows, 'eligible': len(eligible), 'sampled': len(rows), 'comparable': len(comparable),
        'matched': matched, 'agreement': (matched / len(comparable)) if comparable else None, 'seed': seed,
        'method': f'a seeded sample (seed {seed}) of {len(rows)} of {len(eligible)} named, lane-tagged OSM ways; '
                  f'comparable = a same-named road within {tol:g} m along the way; per-direction driving-lane '
                  f'counts compared with OSM expected_lanes',
    }


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
