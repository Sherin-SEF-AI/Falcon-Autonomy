"""OpenStreetMap input: read .osm / .osm.pbf, crop, write, summarise, expected lanes, local projection (stdlib + numpy).

What is read. Ways tagged highway=* (the roads carla.Osm2Odr converts) and building=* (footprints, counted for the
report), every node they reference with its coordinates, and the tags of those nodes (traffic signals and crossings
are node tags, and the converter uses them). Everything else is dropped. Relations are NOT kept, in either format:
turn restrictions (type=restriction) are therefore unknown to a network converted from what write_osm writes, and
multipolygon buildings are not counted. A way's references to nodes missing from the file are dropped and counted
(counts['missing_node_refs']); a way left with fewer than 2 nodes is dropped.

Formats. OSM XML 0.6 (.osm, also .osm.gz / .osm.bz2), parsed incrementally (iterparse). PBF (.osm.pbf) with a
pure-Python reader: BlobHeader/Blob framing, raw, zlib and lzma blobs (zstd on Python >= 3.14; lz4 is refused),
HeaderBlock, PrimitiveBlock string tables, granularity and lat/lon offsets, DenseNodes and plain Nodes, Ways. Packed
varints are decoded with numpy (vectorised), then zigzag- and delta-decoded. Files with required features other than
OsmSchema-V0.6 and DenseNodes (e.g. HistoricalInformation) are refused.

Bounding boxes are (south, west, north, east) in degrees; boxes across the antimeridian are refused. The crop rule,
the same for read_osm(bbox=...) and crop(): keep a highway or building way when at least one of its nodes is inside
the box, with ALL its nodes (ways stay complete, so a crop's edges stick out of the box), and drop everything else.

Projection. local_projection() is transverse Mercator on WGS84 with scale 1 at (lat0, lon0), computed with Krüger's
series to sixth order in n (Karney 2011), the algorithm PROJ's tmerc uses by default: it agrees with PROJ 9.4.1 on
`proj_string(lat0, lon0)` to within 10 nm out to 300 km from the centre (checked with pyproj at five centres
worldwide), so far better than 1 mm over 5 km. proj_string() is what is passed to carla.Osm2OdrSettings.proj_string
with center_map=False, so that OpenDRIVE x, y are metres east and north of (lat0, lon0). Osm2Odr's default,
'+proj=tmerc', centres the projection on 0°E: converted at that default, central Bengaluru (77.6°E, about
4.1 km × 3.2 km of OSM nodes) came out 11.7 km × 13.4 km: stretched about 2.9× east-west and 4.2× north-south.

Lengths are spherical (haversine, R = 6 371 008.8 m): within 0.5 % of the ellipsoidal length.
"""
from __future__ import annotations

import bz2
import gzip
import json
import lzma
import math
import os
import re
import struct
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field
from typing import Callable, NamedTuple
from xml.sax.saxutils import quoteattr

import numpy as np

from .paths import atomic_write

# carla.Osm2OdrSettings' default way types (the converter's own config, written as a comment at the top of its output)
CARLA_HIGHWAYS = ('motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link', 'secondary',
                  'secondary_link', 'tertiary', 'tertiary_link', 'unclassified', 'residential')
EARTH_RADIUS_M = 6371008.8                  # mean radius (IUGG), for haversine lengths
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
NOMINATIM_URL = 'https://nominatim.openstreetmap.org/search'
USER_AGENT = 'FalconAutonomy/0.1 (+local desktop app)'
PBF_FEATURES = {'OsmSchema-V0.6', 'DenseNodes'}

Progress = Callable[[float, str], None] | None


@dataclass
class OsmData:
    """Roads and buildings from an OSM file. nodes {id: (lat, lon)}; ways [{'id', 'nodes', 'tags', 'kind'}] with
    kind 'highway' or 'building' (and 'area' — parks, grass, woods, water — when read with city=True); bounds (south,
    west, north, east); node_tags {id: {k: v}} for tagged nodes only; points [{'id', 'lat', 'lon', 'tags'}] the
    stand-alone point features (trees, street lamps, bus stops, benches…) inside the bounds when read with city=True."""
    nodes: dict = field(default_factory=dict)
    ways: list = field(default_factory=list)
    bounds: tuple | None = None
    source: str = ''
    node_tags: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    points: list = field(default_factory=list)

    @property
    def highways(self) -> list:
        return [w for w in self.ways if w['kind'] == 'highway']

    @property
    def buildings(self) -> list:
        return [w for w in self.ways if w['kind'] == 'building']


# what city=True keeps besides roads and buildings (core.cityprops dresses a world with them)
CITY_AREAS = {'landuse': {'grass', 'recreation_ground', 'village_green', 'forest', 'meadow', 'orchard', 'cemetery',
                          'flowerbed'},
              'leisure': {'park', 'garden', 'pitch', 'playground', 'common'},
              'natural': {'wood', 'scrub', 'grassland', 'heath', 'water', 'tree_row'}}
CITY_POINTS = {'natural': {'tree'},
               'highway': {'street_lamp', 'bus_stop'},
               'amenity': {'bench', 'waste_basket', 'waste_disposal', 'recycling', 'telephone', 'atm', 'post_box',
                           'drinking_water', 'bicycle_parking', 'vending_machine'},
               'shop': {'kiosk'}}


def _city_match(tags: dict, table: dict) -> bool:
    return any(tags.get(k) in vals for k, vals in table.items())


def _kind(tags: dict, city: bool = False) -> str | None:
    if 'highway' in tags:
        return 'highway'
    if 'building' in tags:
        return 'building'
    if city and _city_match(tags, CITY_AREAS):
        return 'area'
    return None


def _check_bbox(bbox) -> tuple:
    s, w, n, e = (float(v) for v in bbox)
    if not (-90.0 <= s <= n <= 90.0) or not (-180.0 <= w <= 180.0 and -180.0 <= e <= 180.0):
        raise ValueError(f'Bad bounding box {bbox!r}: want (south, west, north, east) in degrees')
    if w > e:
        raise ValueError('Bounding boxes across the antimeridian are not supported')
    return s, w, n, e


def _report(progress: Progress, frac: float, text: str):
    if progress is not None:
        progress(max(0.0, min(1.0, frac)), text)


def _finish(nodes: dict, node_tags: dict, ways: list, bounds, bbox, source: str, counts: dict, notes: list,
            points=None) -> OsmData:
    """Drop missing node references, apply the crop rule, keep only referenced nodes (and the point features inside
    the box)."""
    missing = 0
    kept = []
    for w in ways:
        refs = [r for r in w['nodes'] if r in nodes]
        missing += len(w['nodes']) - len(refs)
        if len(refs) < 2:
            continue
        if bbox is not None:
            s, west, n, e = bbox
            if not any(s <= nodes[r][0] <= n and west <= nodes[r][1] <= e for r in refs):
                continue
        kept.append({'id': w['id'], 'nodes': refs, 'tags': w['tags'], 'kind': w['kind']})
    used = {r for w in kept for r in w['nodes']}
    out_nodes = {r: nodes[r] for r in used}
    out_tags = {r: t for r, t in node_tags.items() if r in used}
    if bbox is not None:
        bounds = tuple(bbox)
    elif bounds is None and out_nodes:
        lat = [p[0] for p in out_nodes.values()]
        lon = [p[1] for p in out_nodes.values()]
        bounds = (min(lat), min(lon), max(lat), max(lon))
    counts = dict(counts)
    counts['missing_node_refs'] = counts.get('missing_node_refs', 0) + missing
    pts = [p for p in points or () if bbox is None or (bbox[0] <= p['lat'] <= bbox[2] and bbox[1] <= p['lon'] <= bbox[3])]
    d = OsmData(out_nodes, kept, bounds, source, out_tags, counts, list(notes), sorted(pts, key=lambda p: p['id']))
    d.counts.update(_counts(d))
    return d


def _counts(d: OsmData) -> dict:
    out = {'nodes': len(d.nodes), 'ways': len(d.ways), 'highways': sum(w['kind'] == 'highway' for w in d.ways),
           'buildings': sum(w['kind'] == 'building' for w in d.ways)}
    if d.points or any(w['kind'] == 'area' for w in d.ways):
        out['areas'] = sum(w['kind'] == 'area' for w in d.ways)
        out['points'] = len(d.points)
    return out


RELATIONS_NOTE = 'relations were skipped (turn restrictions and multipolygon buildings are not kept)'


# ------------------------------------------------------------------------------------------------ XML
def _read_xml(path: str, bbox, progress: Progress, city: bool = False) -> OsmData:
    size = max(os.path.getsize(path), 1)
    nodes, node_tags, ways, points = {}, {}, [], []
    bounds, relations, n = None, 0, 0
    with open(path, 'rb') as raw:
        fh = gzip.GzipFile(fileobj=raw) if path.endswith('.gz') else bz2.BZ2File(raw) if path.endswith('.bz2') else raw
        root = None
        for ev, el in ET.iterparse(fh, events=('start', 'end')):
            if ev == 'start':
                if root is None:
                    root = el
                continue
            tag = el.tag
            if tag == 'node':
                nid = int(el.get('id'))
                nodes[nid] = (float(el.get('lat')), float(el.get('lon')))
                if len(el):
                    t = {c.get('k'): c.get('v') for c in el if c.tag == 'tag'}
                    if t:
                        node_tags[nid] = t
                        if city and _city_match(t, CITY_POINTS):
                            points.append({'id': nid, 'lat': nodes[nid][0], 'lon': nodes[nid][1], 'tags': t})
            elif tag == 'way':
                t = {c.get('k'): c.get('v') for c in el if c.tag == 'tag'}
                k = _kind(t, city)
                if k:
                    ways.append({'id': int(el.get('id')), 'nodes': [int(c.get('ref')) for c in el if c.tag == 'nd'],
                                 'tags': t, 'kind': k})
            elif tag == 'relation':
                relations += 1
            elif tag == 'bounds':
                bounds = (float(el.get('minlat')), float(el.get('minlon')), float(el.get('maxlat')),
                          float(el.get('maxlon')))
            else:
                continue
            root.clear()
            n += 1
            if n % 50000 == 0:
                _report(progress, raw.tell() / size, f'Reading OSM: {len(nodes):,} nodes, {len(ways):,} ways')
    if root is None or root.tag != 'osm':
        raise ValueError(f'{path}: not an OSM XML file')
    notes = [RELATIONS_NOTE] if relations else []
    d = _finish(nodes, node_tags, ways, bounds, bbox, path, {'relations_skipped': relations}, notes, points)
    _report(progress, 1.0, f'Read {d.counts["highways"]:,} roads, {d.counts["buildings"]:,} buildings')
    return d


# ------------------------------------------------------------------------------------------------ PBF
def _uvarint(buf, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if b < 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError('PBF: varint too long')


def _fields(buf):
    """(field number, wire type, value) of a protobuf message: an int for varint/fixed, a memoryview for bytes."""
    buf = memoryview(buf)
    pos, n = 0, len(buf)
    while pos < n:
        key, pos = _uvarint(buf, pos)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, pos = _uvarint(buf, pos)
        elif wt == 2:
            ln, pos = _uvarint(buf, pos)
            v = buf[pos:pos + ln]
            pos += ln
        elif wt == 1:
            v = int.from_bytes(buf[pos:pos + 8], 'little')
            pos += 8
        elif wt == 5:
            v = int.from_bytes(buf[pos:pos + 4], 'little')
            pos += 4
        else:
            raise ValueError(f'PBF: unsupported wire type {wt}')
        if pos > n:
            raise ValueError('PBF: truncated message')
        yield f, wt, v


def decode_varints(buf) -> np.ndarray:
    """All varints in a packed buffer as uint64, vectorised: each byte's 7 payload bits shifted by its position in its
    varint, OR-reduced per varint (values wider than 64 bits are truncated, as protobuf does)."""
    a = np.frombuffer(buf, dtype=np.uint8)
    if a.size == 0:
        return np.zeros(0, np.uint64)
    if a[-1] >= 0x80:
        raise ValueError('PBF: truncated packed varint')
    ends = np.flatnonzero(a < 0x80)
    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    pos = np.arange(a.size, dtype=np.int64) - np.repeat(starts, ends - starts + 1)
    if pos.max() > 9:
        raise ValueError('PBF: varint too long')
    vals = (a & 0x7F).astype(np.uint64) << (pos.astype(np.uint64) * np.uint64(7))
    return np.bitwise_or.reduceat(vals, starts)


def unzigzag(u: np.ndarray) -> np.ndarray:
    u = u.astype(np.uint64, copy=False)
    return (u >> np.uint64(1)).astype(np.int64) ^ -((u & np.uint64(1)).astype(np.int64))


def _signed(v: int) -> int:
    return v - (1 << 64) if v >= 1 << 63 else v


def _zz(v: int) -> int:
    return (v >> 1) ^ -(v & 1)


class _Packed:
    """Repeated scalar fields, packed (wire type 2) or not (wire type 0), concatenated as uint64."""

    def __init__(self):
        self.parts = {}

    def add(self, f: int, wt: int, v):
        self.parts.setdefault(f, []).append(decode_varints(v) if wt == 2 else np.array([v], np.uint64))

    def get(self, f: int) -> np.ndarray:
        p = self.parts.get(f)
        if not p:
            return np.zeros(0, np.uint64)
        return p[0] if len(p) == 1 else np.concatenate(p)


def _blob_data(blob) -> bytes:
    raw = zdata = None
    kind, raw_size = None, None
    for f, _, v in _fields(blob):
        if f == 1:
            raw = bytes(v)
        elif f == 2:
            raw_size = v
        elif f in (3, 4, 6, 7):
            kind, zdata = f, bytes(v)
        elif f == 5:
            raise ValueError('PBF: bzip2 blobs are obsolete and not supported')
    if raw is not None:
        return raw
    if kind == 3:
        out = zlib.decompress(zdata)
    elif kind == 4:
        out = lzma.decompress(zdata)
    elif kind == 7:
        try:
            from compression import zstd       # Python >= 3.14
        except ImportError as e:
            raise ValueError('PBF: zstd-compressed blobs need Python >= 3.14') from e
        out = zstd.decompress(zdata)
    elif kind == 6:
        raise ValueError('PBF: lz4-compressed blobs are not supported (standard library only)')
    else:
        raise ValueError('PBF: empty blob')
    if raw_size is not None and len(out) != raw_size:
        raise ValueError('PBF: blob size mismatch')
    return out


def _blobs(path: str, offsets=None):
    """(offset, type, decompressed data, end position) for each blob, or only those at the given offsets."""
    with open(path, 'rb') as fh:
        todo = sorted(offsets) if offsets is not None else None
        i = 0
        while True:
            if todo is not None:
                if i >= len(todo):
                    return
                fh.seek(todo[i])
                i += 1
            off = fh.tell()
            h = fh.read(4)
            if not h:
                return
            if len(h) < 4:
                raise ValueError('PBF: truncated file')
            hl = struct.unpack('>I', h)[0]
            if hl > 64 * 1024:
                raise ValueError('PBF: BlobHeader too large (not a PBF file?)')
            btype, dsize = '', 0
            for f, _, v in _fields(fh.read(hl)):
                if f == 1:
                    btype = bytes(v).decode('utf-8')
                elif f == 3:
                    dsize = v
            if dsize > 32 * 1024 * 1024:
                raise ValueError('PBF: blob too large')
            blob = fh.read(dsize)
            if len(blob) < dsize:
                raise ValueError('PBF: truncated blob')
            yield off, btype, _blob_data(blob), fh.tell()


def _header_block(data) -> tuple | None:
    bounds = None
    for f, _, v in _fields(data):
        if f == 1:
            bb = {g: _zz(x) * 1e-9 for g, _, x in _fields(v)}
            bounds = (bb.get(4, 0.0), bb.get(1, 0.0), bb.get(3, 0.0), bb.get(2, 0.0))   # bottom, left, top, right
        elif f == 4:
            feat = bytes(v).decode('utf-8')
            if feat not in PBF_FEATURES:
                raise ValueError(f'PBF: required feature {feat!r} is not supported')
    return bounds


class _Block:
    """A decoded PrimitiveBlock: its string table, coordinate scaling and groups."""

    def __init__(self, data):
        self.strings, self.groups = [], []
        self.gran, self.lat_off, self.lon_off = 100, 0, 0
        for f, _, v in _fields(data):
            if f == 1:
                self.strings = [bytes(s).decode('utf-8', 'replace') for g, _, s in _fields(v) if g == 1]
            elif f == 2:
                self.groups.append(v)
            elif f == 17:
                self.gran = v
            elif f == 19:
                self.lat_off = _signed(v)
            elif f == 20:
                self.lon_off = _signed(v)

    def kinds(self) -> set:
        out = set()
        for g in self.groups:
            for f, _, _ in _fields(g):
                out.add({1: 'nodes', 2: 'nodes', 3: 'ways', 4: 'relations'}.get(f, 'other'))
        return out

    def _deg(self, raw: np.ndarray, off: int) -> np.ndarray:
        return (off + self.gran * raw.astype(np.int64)) * 1e-9

    def nodes(self):
        """(ids, lat, lon, tags) per group with nodes; tags {index: [(k, v)]} for tagged nodes only."""
        for g in self.groups:
            plain_ids, plain_lat, plain_lon, plain_tags = [], [], [], {}
            for f, wt, v in _fields(g):
                if f == 2:
                    p = _Packed()
                    for h, hwt, x in _fields(v):
                        if h in (1, 8, 9, 10):
                            p.add(h, hwt, x)
                    ids = np.cumsum(unzigzag(p.get(1)), dtype=np.int64)
                    lat = self._deg(np.cumsum(unzigzag(p.get(8)), dtype=np.int64), self.lat_off)
                    lon = self._deg(np.cumsum(unzigzag(p.get(9)), dtype=np.int64), self.lon_off)
                    yield ids, lat, lon, self._dense_tags(p.get(10).astype(np.int64), len(ids))
                elif f == 1:
                    p = _Packed()
                    nid = la = lo = 0
                    for h, hwt, x in _fields(v):
                        if h == 1:
                            nid = _zz(x)
                        elif h == 8:
                            la = _zz(x)
                        elif h == 9:
                            lo = _zz(x)
                        elif h in (2, 3):
                            p.add(h, hwt, x)
                    keys, vals = p.get(2), p.get(3)
                    if len(keys):
                        plain_tags[len(plain_ids)] = [(self.strings[int(k)], self.strings[int(w)])
                                                      for k, w in zip(keys, vals)]
                    plain_ids.append(nid)
                    plain_lat.append(la)
                    plain_lon.append(lo)
            if plain_ids:
                yield (np.array(plain_ids, np.int64), self._deg(np.array(plain_lat, np.int64), self.lat_off),
                       self._deg(np.array(plain_lon, np.int64), self.lon_off), plain_tags)

    def _dense_tags(self, kv: np.ndarray, n: int):
        """Lazy tags of DenseNodes: keys_vals holds (k, v)* 0 per node; only tagged nodes get an entry."""
        if kv.size == 0:
            return {}
        zeros = np.flatnonzero(kv == 0)
        if len(zeros) != n:
            raise ValueError('PBF: DenseNodes keys_vals does not match the node count')
        starts = np.empty_like(zeros)
        starts[0] = 0
        starts[1:] = zeros[:-1] + 1
        tagged = np.flatnonzero(zeros > starts)
        return _DenseTags(kv, starts, zeros, tagged, self.strings)

    def ways(self):
        for g in self.groups:
            for f, _, v in _fields(g):
                if f != 3:
                    continue
                p = _Packed()
                wid = 0
                for h, hwt, x in _fields(v):
                    if h == 1:
                        wid = _signed(x)
                    elif h in (2, 3, 8):
                        p.add(h, hwt, x)
                tags = {self.strings[int(k)]: self.strings[int(w)] for k, w in zip(p.get(2), p.get(3))}
                yield wid, tags, p.get(8)

    def relations(self) -> int:
        return sum(1 for g in self.groups for f, _, _ in _fields(g) if f == 4)


class _DenseTags(dict):
    """{node index: [(k, v)]} built on demand for the indices asked for."""

    def __init__(self, kv, starts, ends, tagged, strings):
        super().__init__()
        self.kv, self.starts, self.ends, self.strings = kv, starts, ends, strings
        self.tagged = set(tagged.tolist())

    def __contains__(self, i):
        return i in self.tagged

    def __getitem__(self, i):
        seg = self.kv[self.starts[i]:self.ends[i]]
        return [(self.strings[int(seg[j])], self.strings[int(seg[j + 1])]) for j in range(0, len(seg) - 1, 2)]

    def __bool__(self):
        return bool(self.tagged)

    def __iter__(self):
        return iter(sorted(self.tagged))


def _member(ids: np.ndarray, sorted_set: np.ndarray) -> np.ndarray:
    if sorted_set.size == 0 or ids.size == 0:
        return np.zeros(ids.shape, bool)
    i = np.clip(np.searchsorted(sorted_set, ids), 0, sorted_set.size - 1)
    return sorted_set[i] == ids


def _read_pbf(path: str, bbox, progress: Progress, city: bool = False) -> OsmData:
    size = max(os.path.getsize(path), 1)
    npass = 3 if bbox is not None else 2
    bounds, relations = None, 0
    points = []

    def city_points(ids, lat, lon, tags, idx):
        """The point features among nodes idx of a group (city=True only)."""
        if not city or not tags:
            return
        for i in idx:
            if i in tags:
                t = dict(tags[i])
                if _city_match(t, CITY_POINTS):
                    points.append({'id': int(ids[i]), 'lat': round(float(lat[i]), 9), 'lon': round(float(lon[i]), 9),
                                   'tags': t})
    kinds: dict[int, set] = {}

    def scan(k: int, want: str):
        """Blocks for pass k (0-based): all of them on the first pass, then only those holding `want`."""
        nonlocal bounds
        offs = None if k == 0 else [o for o, ks in kinds.items() if want in ks]
        for off, btype, data, end in _blobs(path, offs):
            _report(progress, (k + end / size) / npass, f'Reading PBF (pass {k + 1} of {npass})')
            if btype == 'OSMHeader':
                bounds = _header_block(data) if bounds is None else bounds
                continue
            if btype != 'OSMData':
                continue
            b = _Block(data)
            if k == 0:
                kinds[off] = b.kinds()
            yield b

    def collect_ways(k: int, inside: np.ndarray | None):
        nonlocal relations
        ways = []
        for b in scan(k, 'ways'):
            if k == 0:
                relations += b.relations()
            for wid, tags, refs in b.ways():
                kind = _kind(tags, city)
                if not kind:
                    continue
                ids = np.cumsum(unzigzag(refs), dtype=np.int64)
                if inside is not None and not _member(ids, inside).any():
                    continue
                ways.append({'id': wid, 'nodes': ids, 'tags': tags, 'kind': kind})
        return ways

    def collect_nodes(k: int, needed: np.ndarray):
        nodes, node_tags = {}, {}
        for b in scan(k, 'nodes'):
            for ids, lat, lon, tags in b.nodes():
                if bbox is None and city:
                    city_points(ids, lat, lon, tags, list(tags) if tags else ())
                hit = np.flatnonzero(_member(ids, needed))
                for i, nid, la, lo in zip(hit.tolist(), ids[hit].tolist(), lat[hit].tolist(), lon[hit].tolist()):
                    nodes[nid] = (round(la, 9), round(lo, 9))
                    if tags and i in tags:
                        node_tags[nid] = dict(tags[i])
        return nodes, node_tags

    if bbox is not None:
        s, w, n, e = bbox
        inside = []
        for b in scan(0, 'nodes'):
            relations += b.relations()
            for ids, lat, lon, tags in b.nodes():
                m = (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)
                if m.any():
                    inside.append(ids[m])
                    city_points(ids, lat, lon, tags, np.flatnonzero(m).tolist())
        inside = np.unique(np.concatenate(inside)) if inside else np.zeros(0, np.int64)
        ways = collect_ways(1, inside)
        k_nodes = 2
    else:
        ways = collect_ways(0, None)
        k_nodes = 1
    needed = np.unique(np.concatenate([w['nodes'] for w in ways])) if ways else np.zeros(0, np.int64)
    nodes, node_tags = collect_nodes(k_nodes, needed)
    for wy in ways:
        wy['nodes'] = wy['nodes'].tolist()
    notes = [RELATIONS_NOTE] if relations else []
    d = _finish(nodes, node_tags, ways, bounds, bbox, path, {'relations_skipped': relations}, notes, points)
    _report(progress, 1.0, f'Read {d.counts["highways"]:,} roads, {d.counts["buildings"]:,} buildings')
    return d


# ------------------------------------------------------------------------------------------------ public API
def read_osm(path: str, bbox=None, progress: Progress = None, city: bool = False) -> OsmData:
    """Roads and buildings from an .osm (XML, optionally .gz/.bz2) or .osm.pbf file, optionally cropped to bbox
    (south, west, north, east) with the module's crop rule. progress(fraction, text) is called while reading.
    city=True also keeps parks, grass, woods and water (ways of kind 'area') and the point features of CITY_POINTS
    (OsmData.points): what core.cityprops dresses a world with."""
    bbox = _check_bbox(bbox) if bbox is not None else None
    if path.endswith('.pbf'):
        return _read_pbf(path, bbox, progress, city)
    return _read_xml(path, bbox, progress, city)


def crop(data: OsmData, bbox) -> OsmData:
    """The ways with at least one node inside bbox, each with all its nodes (the module's crop rule)."""
    bbox = _check_bbox(bbox)
    counts = {k: v for k, v in data.counts.items() if k in ('relations_skipped', 'missing_node_refs')}
    return _finish(data.nodes, data.node_tags, data.ways, None, bbox, data.source, counts, data.notes, data.points)


def subset(data: OsmData, ways: list, bounds=None, notes=()) -> OsmData:
    """The given ways (dicts as data.ways', node ids from data.nodes) with only the nodes they use; no crop rule.
    bounds: the result's <bounds> (default: the nodes' extent)."""
    counts = {k: v for k, v in data.counts.items() if k == 'relations_skipped'}
    return _finish(data.nodes, data.node_tags, ways, tuple(bounds) if bounds else None, None, data.source, counts,
                   list(data.notes) + list(notes))


def _radii(lat: float) -> tuple[float, float]:
    """Meridional and prime-vertical radii of curvature of WGS84 at lat."""
    e2 = WGS84_F * (2 - WGS84_F)
    s2 = math.sin(math.radians(lat)) ** 2
    return WGS84_A * (1 - e2) / (1 - e2 * s2) ** 1.5, WGS84_A / math.sqrt(1 - e2 * s2)


def bbox_around(lat: float, lon: float, size_m: float) -> tuple:
    """(south, west, north, east) of a box centred on (lat, lon) whose sides measure size_m through its centre
    (along the meridian and the parallel; size_m=1000 is about 1 km²)."""
    m, n = _radii(lat)
    h = size_m / 2.0
    dlat = math.degrees(h / m)
    dlon = math.degrees(h / (n * math.cos(math.radians(lat))))
    return _check_bbox((lat - dlat, lon - dlon, lat + dlat, lon + dlon))


def write_osm(data: OsmData, path: str):
    """OSM XML 0.6 that carla.Osm2Odr accepts: <bounds>, nodes (with their tags), then ways (nd refs and tags)."""
    out = ["<?xml version='1.0' encoding='UTF-8'?>", '<osm version="0.6" generator="Falcon Autonomy">']
    if data.bounds:
        s, w, n, e = data.bounds
        out.append(f'  <bounds minlat="{s:.7f}" minlon="{w:.7f}" maxlat="{n:.7f}" maxlon="{e:.7f}"/>')

    def tags(t: dict) -> list:
        return [f'    <tag k={quoteattr(str(k))} v={quoteattr(str(v))}/>' for k, v in t.items()]

    for nid in sorted(data.nodes):
        la, lo = data.nodes[nid]
        t = data.node_tags.get(nid)
        if t:
            out.append(f'  <node id="{nid}" lat="{la:.7f}" lon="{lo:.7f}">')
            out += tags(t)
            out.append('  </node>')
        else:
            out.append(f'  <node id="{nid}" lat="{la:.7f}" lon="{lo:.7f}"/>')
    for w in sorted(data.ways, key=lambda w: w['id']):
        out.append(f'  <way id="{w["id"]}">')
        out += [f'    <nd ref="{r}"/>' for r in w['nodes']]
        out += tags(w['tags'])
        out.append('  </way>')
    out.append('</osm>\n')
    atomic_write(path, '\n'.join(out))


def _haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def way_length_m(data: OsmData, way: dict) -> float:
    """A way's length along its nodes (haversine)."""
    p = np.array([data.nodes[r] for r in way['nodes']], float)
    return float(_haversine(p[:-1, 0], p[:-1, 1], p[1:, 0], p[1:, 1]).sum()) if len(p) > 1 else 0.0


def oneway(tags: dict) -> int | None:
    """1 traffic only along the way, -1 only against it, 0 both ways, None unknown (reversible, alternating, other
    values). Untagged: oneway implied on highway=motorway and junction=roundabout/circular (OSM wiki, Key:oneway)."""
    v = tags.get('oneway')
    if v is None:
        return 1 if tags.get('highway') == 'motorway' or tags.get('junction') in ('roundabout', 'circular') else 0
    v = v.strip().lower()
    if v in ('yes', '1', 'true'):
        return 1
    if v in ('-1', 'reverse'):
        return -1
    if v in ('no', '0', 'false'):
        return 0
    return None


def _count_tag(tags: dict, key: str) -> int | None:
    v = tags.get(key)
    return int(v) if v is not None and re.fullmatch(r'\s*[0-9]+\s*', v) else None


def expected_lanes(way) -> tuple[int, int] | None:
    """(forward, backward) driving lanes from OSM tags (OSM wiki, Key:lanes), relative to the way's node order; None
    when the tags do not say. Oneway: (lanes, 0), or (0, lanes) for oneway=-1. Two-way: lanes:forward and
    lanes:backward when tagged (the other one = lanes − lanes:both_ways − the given one), else an even `lanes`
    (minus lanes:both_ways) split evenly; an odd total without forward/backward is None (the wiki leaves the split
    unspecified). `lanes` counts every lane (bus and turn lanes too). way: a way dict or its tags."""
    tags = way['tags'] if isinstance(way, dict) and isinstance(way.get('tags'), dict) else way
    ow = oneway(tags)
    if ow is None:
        return None
    n = _count_tag(tags, 'lanes')
    f, b = _count_tag(tags, 'lanes:forward'), _count_tag(tags, 'lanes:backward')
    if ow == 1:
        n = n if n is not None else f
        return (n, 0) if n else None
    if ow == -1:
        n = n if n is not None else b
        return (0, n) if n else None
    if f is not None and b is not None:
        return (f, b) if f + b > 0 else None
    if n is None:
        return None
    rest = n - (_count_tag(tags, 'lanes:both_ways') or 0)
    if f is not None:
        return (f, rest - f) if 0 <= rest - f else None
    if b is not None:
        return (rest - b, b) if 0 <= rest - b else None
    if rest > 0 and rest % 2 == 0:
        return rest // 2, rest // 2
    return None


def summary(data: OsmData) -> dict:
    """Counts by highway type, road lengths (CARLA's converted types, and all highways), oneway ways, ways with a lanes
    tag, buildings, bounds and centre."""
    by_type: dict[str, int] = {}
    road_m = all_m = 0.0
    oneways = with_lanes = 0
    for w in data.highways:
        t = w['tags'].get('highway', '')
        by_type[t] = by_type.get(t, 0) + 1
        ln = way_length_m(data, w)
        all_m += ln
        if t in CARLA_HIGHWAYS:
            road_m += ln
        if oneway(w['tags']) in (1, -1):
            oneways += 1
        if any(k == 'lanes' or k.startswith('lanes:') for k in w['tags']):
            with_lanes += 1
    b = data.bounds
    return {
        'highways': sum(by_type.values()),
        'by_highway': dict(sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0]))),
        'road_length_m': round(road_m, 1),
        'road_types': list(CARLA_HIGHWAYS),
        'highway_length_m': round(all_m, 1),
        'oneway': oneways,
        'with_lanes_tag': with_lanes,
        'buildings': len(data.buildings),
        'nodes': len(data.nodes),
        'bbox': tuple(b) if b else None,
        'centre': ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) if b else None,
        'length_method': 'haversine (spherical, within 0.5 % of the ellipsoidal length)',
    }


# ------------------------------------------------------------------------------------------------ projection
def _tm_series():
    n = WGS84_F / (2 - WGS84_F)
    n2, n3, n4, n5, n6 = n ** 2, n ** 3, n ** 4, n ** 5, n ** 6
    big_a = WGS84_A / (1 + n) * (1 + n2 / 4 + n4 / 64 + n6 / 256)
    alpha = (n / 2 - 2 * n2 / 3 + 5 * n3 / 16 + 41 * n4 / 180 - 127 * n5 / 288 + 7891 * n6 / 37800,
             13 * n2 / 48 - 3 * n3 / 5 + 557 * n4 / 1440 + 281 * n5 / 630 - 1983433 * n6 / 1935360,
             61 * n3 / 240 - 103 * n4 / 140 + 15061 * n5 / 26880 + 167603 * n6 / 181440,
             49561 * n4 / 161280 - 179 * n5 / 168 + 6601661 * n6 / 7257600,
             34729 * n5 / 80640 - 3418889 * n6 / 1995840,
             212378941 * n6 / 319334400)
    beta = (n / 2 - 2 * n2 / 3 + 37 * n3 / 96 - n4 / 360 - 81 * n5 / 512 + 96199 * n6 / 604800,
            n2 / 48 + n3 / 15 - 437 * n4 / 1440 + 46 * n5 / 105 - 1118711 * n6 / 3870720,
            17 * n3 / 480 - 37 * n4 / 840 - 209 * n5 / 4480 + 5569 * n6 / 90720,
            4397 * n4 / 161280 - 11 * n5 / 504 - 830251 * n6 / 7257600,
            4583 * n5 / 161280 - 108847 * n6 / 3991680,
            20648693 * n6 / 638668800)
    return big_a, alpha, beta


_TM_A, _TM_ALPHA, _TM_BETA = _tm_series()
_E = math.sqrt(WGS84_F * (2 - WGS84_F))


def _tm_forward(lat, lam):
    """Krüger series: (eta, xi), the TM coordinates divided by A, of latitude lat and longitude lam from the central
    meridian (radians, arrays)."""
    s = np.sin(lat)
    t = np.sinh(np.arctanh(s) - _E * np.arctanh(_E * s))
    cl = np.cos(lam)
    xi_p = np.arctan2(t, cl)
    eta_p = np.arcsinh(np.sin(lam) / np.sqrt(t * t + cl * cl))
    xi, eta = xi_p.copy(), eta_p.copy()
    for j, a in enumerate(_TM_ALPHA, 1):
        xi += a * np.sin(2 * j * xi_p) * np.cosh(2 * j * eta_p)
        eta += a * np.cos(2 * j * xi_p) * np.sinh(2 * j * eta_p)
    return eta, xi


def _tm_inverse(eta, xi):
    xi_p, eta_p = xi.copy(), eta.copy()
    for j, b in enumerate(_TM_BETA, 1):
        xi_p -= b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        eta_p -= b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    she = np.sinh(eta_p)
    taup = np.sin(xi_p) / np.sqrt(she * she + np.cos(xi_p) ** 2)
    lam = np.arctan2(she, np.cos(xi_p))
    e2m = 1 - _E * _E
    tau = taup.copy()
    for _ in range(5):                       # Newton on tau' (Karney 2011, eqs. 7 and 19-21); converges in 2-3
        t1 = np.sqrt(1 + tau * tau)
        sig = np.sinh(_E * np.arctanh(_E * tau / t1))
        tp = tau * np.sqrt(1 + sig * sig) - sig * t1
        tau = tau + (taup - tp) / ((e2m * np.sqrt(1 + tp * tp) * t1) / (1 + e2m * tau * tau))
    return np.arctan(tau), lam


class Projection(NamedTuple):
    """to_local(lat, lon) -> (east, north) metres; to_geo(east, north) -> (lat, lon) degrees. Scalars or arrays."""
    to_local: Callable
    to_geo: Callable
    lat0: float
    lon0: float
    proj: str


def _out(v, scalar: bool):
    return float(v) if scalar else v


def local_projection(lat0: float, lon0: float) -> Projection:
    """Transverse Mercator (WGS84, k=1) centred on (lat0, lon0): the same map as PROJ with proj_string(lat0, lon0)
    (within 10 nm out to 300 km)."""
    lat0, lon0 = float(lat0), float(lon0)
    _, xi0 = _tm_forward(np.array(math.radians(lat0)), np.array(0.0))
    north0 = _TM_A * float(xi0)

    def to_local(lat, lon):
        scalar = np.ndim(lat) == 0 and np.ndim(lon) == 0
        lam = np.radians((np.asarray(lon, float) - lon0 + 180.0) % 360.0 - 180.0)
        eta, xi = _tm_forward(np.radians(np.asarray(lat, float)), lam)
        return _out(_TM_A * eta, scalar), _out(_TM_A * xi - north0, scalar)

    def to_geo(east, north):
        scalar = np.ndim(east) == 0 and np.ndim(north) == 0
        lat, lam = _tm_inverse(np.asarray(east, float) / _TM_A, (np.asarray(north, float) + north0) / _TM_A)
        lon = (np.degrees(lam) + lon0 + 180.0) % 360.0 - 180.0
        return _out(np.degrees(lat), scalar), _out(lon, scalar)

    return Projection(to_local, to_geo, lat0, lon0, proj_string(lat0, lon0))


def _num(v: float) -> str:
    s = f'{float(v):.9f}'.rstrip('0').rstrip('.')
    return '0' if s in ('-0', '') else s


def proj_string(lat0: float, lon0: float) -> str:
    """The PROJ string of local_projection(lat0, lon0), for carla.Osm2OdrSettings.proj_string (with center_map=False
    OpenDRIVE x = metres east, y = metres north of (lat0, lon0))."""
    return (f'+proj=tmerc +lat_0={_num(lat0)} +lon_0={_num(lon0)} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m '
            f'+no_defs')


# ------------------------------------------------------------------------------------------------ geocoding
def parse_geocode(results) -> list:
    """Nominatim search results (JSON list) as [(display_name, lat, lon, (south, west, north, east) | None)]."""
    out = []
    for r in results or ():
        try:
            lat, lon = float(r['lat']), float(r['lon'])
        except (KeyError, TypeError, ValueError):
            continue
        bb = r.get('boundingbox')
        try:
            bbox = (float(bb[0]), float(bb[2]), float(bb[1]), float(bb[3])) if bb and len(bb) == 4 else None
        except (TypeError, ValueError):
            bbox = None
        out.append((str(r.get('display_name', '')), lat, lon, bbox))
    return out


def geocode(query: str, allowed: bool, limit: int = 5, timeout: float = 15.0) -> list:
    """Place search on OpenStreetMap's Nominatim (sends the query to nominatim.openstreetmap.org; its usage policy
    allows at most one request a second). Raises PermissionError when network lookups are not allowed."""
    if not allowed:
        raise PermissionError('Place search is off: it sends the query to nominatim.openstreetmap.org')
    q = (query or '').strip()
    if not q:
        return []
    url = NOMINATIM_URL + '?' + urllib.parse.urlencode({'q': q, 'format': 'jsonv2', 'limit': int(limit)})
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return parse_geocode(json.loads(r.read().decode('utf-8')))
