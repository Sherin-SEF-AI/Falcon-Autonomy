"""Dashcam ingest: a consumer dashcam's MP4 clips, their embedded GPS track, and frames extracted with the GPS
position each was taken at (standard library and numpy at import; ffmpeg as a subprocess, OpenCV only to write
images). This module's public functions are a stable API: core.recon's dashcam odometry imports them.

Clips. Files named YYYYMMDDHHMMSS_NNNNNN<F|R>.MP4 (the start in local time, a running number, front or rear camera;
the "Movie_F", "Park_F" and "Photo_R" folders of a Novatek-based camera). Anything else ending in .mp4 is listed with
no start time from its name.

GPS ("freeGPS" boxes, YOUQING variant). The clip's moov box holds a box "gps " whose body is: u32 version/flags
(0x00000101), u32 count, then count big-endian (u32 file offset, u32 size) pairs — one per second of video. Each pair
points to a 16 384-byte block inside mdat: u32 BE size, "freeGPS ", u32 LE payload length (0x90 = 144), then the
payload. A block with no fix is all zeros after the header. With a fix, payload offsets (little-endian):

     0  "YOUQINGGPS\\0\\0"                      signature
    24  f32 latitude,  ddmm.mmmm (degrees·100 + minutes)
    28  f32 longitude, dddmm.mmmm
    32  i32 hour, minute, second              UTC (GPS time)
    44  i32 year − 2000, month, day           UTC date
    56  char status 'A' (valid) | 'V', char 'N'|'S', char 'E'|'W', 0
    64  f64, f64                              not identified (the first varies at random from fix to fix, the
                                              second grows slowly while moving); not used
    80  f32 speed over ground, knots
    84  f32 course over ground, degrees from north
    88  "1234567890123456"                    constant; 104 "YSKJ" constant
   108  i32 ×3                                small integers that change every second (−6 … 6): probably the
                                              G-sensor; not used
   120  i32 hour, minute, second, year, month, day   the camera's own clock (local time)

Verified on the user's 262 clips — 15 539 blocks, 15 033 fixes (after dropping 414 repeated UTC seconds), every one
with status 'A', 'N', 'E' (a missing fix is an empty block, never 'V'); positions plausible along the roads driven
and continuous from second to second; the distance between consecutive
one-second fixes divided by the mean reported speed read as knots has median 1.002 (10th–90th percentile 0.969–1.041,
6 271 pairs above 5 m/s) — read as km/h it would be 1.86, so the unit is knots (the camera's burned-in text shows the
same speed in km/h); the heading between consecutive fixes minus the reported course has median 0.00° (10th–90th
percentile −1.5° … +1.5°). The UTC time + 5 h 30 min equals the camera clock to within ±1 s, and the camera clock of
block k equals the file name's time + k s (+0 or +1 s). Block k is written into mdat right after the video packets of
0 … k + 0.93 s. Float32 coordinates carry ≈ 0.2 m (latitude) and ≈ 0.9 m (longitude) of quantisation. GPS_LAYOUT
holds the same layout for the code; anything that fails its checks (signature, ranges, minutes < 60) is skipped and
counted, never guessed.

Time base. A fix's video time is t = UTC − t0, with t0 = the clip's median of (UTC_k − k) over its fixes (block k
covers video second k, as above). Using the GPS time handles a skipped or repeated second (≈ 3.7 % of blocks repeat or
skip a UTC second relative to the block index); a repeated UTC second keeps its first fix. The sub-second offset
between a fix and the video frame is not known from the file: about ±0.5 s (block written at ≈ k + 0.93 s; fix
latency unknown), i.e. ±7 m along the track at 50 km/h; the GPS speed against the video's own motion supports it
to about ±2 s (10 stop-and-go clips with a clear cross-correlation peak: −1.4 … +2.4 s, median 0.0 s; flat curves
on others). A frame's position is linearly interpolated between the fixes around its time (course on the circle);
frames more than MAX_GAP_S from any fix get no position.

Frames. ffmpeg decodes (NVDEC with -hwaccel cuda when it works — measured on HEVC 3840×2160 at 30 fps: 60 s of video
at 2 frames/s in 3.9 s with every frame decoded, 1.0 s decoding only the key frames, which this camera writes every
0.5 s; 2.4 s key frames on 24 CPU threads); scaled to the output size; a frame's index n gives its video time n /
fps (constant frame rate, checked). An image set is a folder: frames/<clip>_<n>.jpg, frames.jsonl (one record per
frame: image, clip, frame index, video time, UTC, local time, lat/lon, speed, course, sun altitude) and run.json
(provenance; each clip fingerprinted by size, mtime and the SHA-256 of its first MiB — hashing 59 GB whole is too slow,
and the first MiB holds the moov box with the GPS index, so an edited clip almost surely changes it).

Privacy. A dashcam's frames and track show where its owner drives. Nothing here copies them anywhere but the folder
the user names; tests use synthetic clips.
"""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import time

import numpy as np

SET_SCHEMA = 'falcon.dashcam-set/1'
NAME_RE = re.compile(r'^(\d{14})_(\d+)([FR])?\.mp4$', re.IGNORECASE)
TZ_OFFSET_H = 5.5                     # IST, the default local time of the clip names
KNOT_KMH = 1.852
MAX_GAP_S = 2.5                       # a frame further than this from a fix gets no position
BLOCK_SIZE = 16384
SIGNATURE = b'YOUQINGGPS'
GPS_LAYOUT = {
    'box': "moov/'gps ': u32 BE version, u32 BE count, count × (u32 BE offset, u32 BE size); one per video second",
    'block': "u32 BE size, 'freeGPS ', u32 LE payload length (0x90), payload (all zero when there is no fix)",
    'payload': {
        'signature': (0, '12s', b'YOUQINGGPS\0\0'),
        'lat_ddmm': (24, '<f'), 'lon_ddmm': (28, '<f'),
        'utc_hms': (32, '<3i'), 'utc_ymd': (44, '<3i'),       # year − 2000
        'status_ns_ew': (56, '3s'),
        'unknown_f64': (64, '<2d'),
        'speed_knots': (80, '<f'), 'course_deg': (84, '<f'),
        'gsensor': (108, '<3i'),
        'camera_local': (120, '<6i'),                          # h, m, s, year, month, day
    },
    'verified': 'see the module docstring and docs/dashcam.md',
}
DEFAULT_ROOT = os.path.expanduser('~/Videos/dashcam-videos')
DEFAULT_SETS = os.path.expanduser('~/FalconWorkspace/realsets')
NIGHT_SUN_DEG = -6.0                  # civil twilight ends: the sun 6° below the horizon (`--night`: lower than this)
DARK_LUMA, CLIPPED_LUMA = 16, 250     # exposure stats: crushed shadows and clipped highlights, Rec. 601 luma levels


class DashcamError(Exception):
    pass


# ================================================================================================ clips
def parse_name(path: str, tz_offset_h: float = TZ_OFFSET_H) -> dict:
    """{'start_local', 'start_utc' (aware datetimes or None), 'number', 'camera' ('F'|'R'|None)} from a clip's
    file name."""
    m = NAME_RE.match(os.path.basename(path))
    if not m:
        return {'start_local': None, 'start_utc': None, 'number': None, 'camera': None}
    tz = _dt.timezone(_dt.timedelta(hours=tz_offset_h))
    try:
        loc = _dt.datetime.strptime(m.group(1), '%Y%m%d%H%M%S').replace(tzinfo=tz)
    except ValueError:
        return {'start_local': None, 'start_utc': None, 'number': None, 'camera': None}
    return {'start_local': loc, 'start_utc': loc.astimezone(_dt.timezone.utc), 'number': int(m.group(2)),
            'camera': (m.group(3) or '').upper() or None}


def _boxes(buf: bytes, start: int = 0, end: int | None = None):
    end = len(buf) if end is None else end
    i = start
    while i + 8 <= end:
        size, typ = struct.unpack('>I4s', buf[i:i + 8])
        hs = 8
        if size == 1:
            size, hs = struct.unpack('>Q', buf[i + 8:i + 16])[0], 16
        elif size == 0:
            size = end - i
        if size < hs:
            return
        yield typ, i + hs, i + size
        i += size


def _moov(path: str) -> bytes:
    """The moov box's body (read without the media data)."""
    size = os.path.getsize(path)
    with open(path, 'rb') as fh:
        pos = 0
        while pos + 8 <= size:
            fh.seek(pos)
            h = fh.read(16)
            sz, typ = struct.unpack('>I4s', h[:8])
            hs = 8
            if sz == 1:
                sz, hs = struct.unpack('>Q', h[8:16])[0], 16
            elif sz == 0:
                sz = size - pos
            if sz < hs:
                break
            if typ == b'moov':
                fh.seek(pos + hs)
                return fh.read(sz - hs)
            pos += sz
    raise DashcamError(f'{path}: no moov box (not an MP4, or still being written)')


def probe(path: str) -> dict:
    """Duration, frame rate, size and frame count from the MP4 header, and the number of GPS blocks."""
    m = _moov(path)
    out = {'duration_s': None, 'fps': None, 'width': None, 'height': None, 'n_frames': None, 'gps_blocks': 0}
    for typ, a, b in _boxes(m):
        if typ == b'mvhd':
            v = m[a]
            ts, dur = (struct.unpack('>IQ', m[a + 20:a + 32]) if v == 1 else struct.unpack('>II', m[a + 12:a + 20]))
            out['duration_s'] = dur / ts if ts else None
        elif typ == b'gps ':
            out['gps_blocks'] = struct.unpack('>I', m[a + 4:a + 8])[0] if b - a >= 8 else 0
        elif typ == b'trak' and out['fps'] is None:
            v = _video_trak(m, a, b)
            if v:
                out.update(v)
    return out


def _video_trak(m: bytes, a: int, b: int) -> dict | None:
    def find(path, a, b):
        for typ, x, y in _boxes(m, a, b):
            if typ == path[0]:
                return (x, y) if len(path) == 1 else find(path[1:], x, y)
        return None
    hd = find([b'mdia', b'hdlr'], a, b)
    if not hd or m[hd[0] + 8:hd[0] + 12] != b'vide':
        return None
    md = find([b'mdia', b'mdhd'], a, b)
    v = m[md[0]]
    ts = struct.unpack('>I', m[md[0] + (20 if v == 1 else 12):md[0] + (24 if v == 1 else 16)])[0]
    st = find([b'mdia', b'minf', b'stbl', b'stts'], a, b)
    n = struct.unpack('>I', m[st[0] + 4:st[0] + 8])[0]
    ents = [struct.unpack('>II', m[st[0] + 8 + 8 * i:st[0] + 16 + 8 * i]) for i in range(n)]
    frames = sum(c for c, _d in ents)
    fps = ts / ents[0][1] if ents and ents[0][1] else None
    tk = find([b'tkhd'], a, b)
    v = m[tk[0]]
    o = tk[0] + (88 if v == 1 else 76)
    w, h = struct.unpack('>II', m[o:o + 8])
    return {'fps': fps, 'constant_rate': len(ents) == 1, 'n_frames': frames, 'width': w >> 16, 'height': h >> 16}


def list_clips(root: str = DEFAULT_ROOT, pattern: str = '**/*', tz_offset_h: float = TZ_OFFSET_H,
               headers: bool = False) -> list:
    """Every .mp4 under root (recursively), by start time then name: [{'path', 'name', 'folder', 'start_local',
    'start_utc', 'number', 'camera', 'size', 'mtime'}] (+ probe() fields with headers=True). `root` may also be one
    file."""
    if os.path.isfile(root):
        paths = [root]
    else:
        if not os.path.isdir(root):
            raise DashcamError(f'{root} does not exist')
        paths = [p for p in glob.glob(os.path.join(root, pattern), recursive=True)
                 if p.lower().endswith('.mp4') and os.path.isfile(p)]
    out = []
    for p in paths:
        st = os.stat(p)
        c = {'path': os.path.abspath(p), 'name': os.path.basename(p), 'folder': os.path.basename(os.path.dirname(p)),
             'size': st.st_size, 'mtime': st.st_mtime}
        c.update(parse_name(p, tz_offset_h))
        if headers:
            try:
                c.update(probe(p))
            except (DashcamError, OSError, struct.error, TypeError, IndexError) as e:
                c['error'] = str(e)
        out.append(c)
    far = _dt.datetime.max.replace(tzinfo=_dt.timezone.utc)
    out.sort(key=lambda c: (c['start_utc'] or far, c['name']))
    return out


def fingerprint(path: str, head: int = 1 << 20) -> dict:
    """{'path', 'size', 'mtime', 'sha256_head', 'head_bytes'}: a clip's identity without hashing all of it."""
    st = os.stat(path)
    with open(path, 'rb') as fh:
        h = hashlib.sha256(fh.read(head)).hexdigest()
    return {'path': os.path.abspath(path), 'size': st.st_size,
            'mtime': _dt.datetime.fromtimestamp(st.st_mtime, _dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'sha256_head': h, 'head_bytes': min(head, st.st_size)}


# ================================================================================================ GPS
def gps_index(path: str) -> list:
    """The (offset, size) of each GPS block, from moov/'gps ' (empty when the clip has none)."""
    m = _moov(path)
    for typ, a, b in _boxes(m):
        if typ == b'gps ':
            if b - a < 8:
                return []
            n = struct.unpack('>I', m[a + 4:a + 8])[0]
            n = min(n, (b - a - 8) // 8)
            return [struct.unpack('>II', m[a + 8 + 8 * i:a + 16 + 8 * i]) for i in range(n)]
    return []


def decode_block(block: bytes) -> dict | None:
    """One GPS block (with its 16-byte header) → a fix, or None for an empty or unrecognised block. Raises nothing."""
    if len(block) < 16 + 144 or block[4:12] != b'freeGPS ':
        return None
    p = block[16:]
    if p[:10] != SIGNATURE:
        return None
    try:
        lat, lon = struct.unpack('<ff', p[24:32])
        hh, mi, ss, yy, mo, dd = struct.unpack('<6i', p[32:56])
        status, ns, ew = (chr(c) for c in p[56:59])
        knots, course = struct.unpack('<ff', p[80:88])
        loc = struct.unpack('<6i', p[120:144])
        utc = _dt.datetime(2000 + yy, mo, dd, hh, mi, ss, tzinfo=_dt.timezone.utc)
    except (struct.error, ValueError):
        return None
    if ns not in 'NS' or ew not in 'EW' or status not in 'AV':
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)) or not (0 <= lat <= 9000 and 0 <= lon <= 18000):
        return None
    la_d, lo_d = int(lat // 100), int(lon // 100)
    if lat - 100 * la_d >= 60 or lon - 100 * lo_d >= 60:
        return None
    la = (la_d + (lat - 100 * la_d) / 60) * (-1 if ns == 'S' else 1)
    lo = (lo_d + (lon - 100 * lo_d) / 60) * (-1 if ew == 'W' else 1)
    try:
        cam = _dt.datetime(loc[3], loc[4], loc[5], loc[0], loc[1], loc[2]).isoformat()
    except ValueError:
        cam = None
    sp = float(knots) if math.isfinite(knots) and 0 <= knots < 300 else None
    return {'utc': utc, 'lat': la, 'lon': lo, 'valid': status == 'A',
            'speed_knots': sp, 'speed_kmh': None if sp is None else sp * KNOT_KMH,
            'course_deg': float(course) % 360 if math.isfinite(course) else None, 'camera_local': cam}


def read_gps(path: str) -> dict:
    """{'blocks', 'fixes': [fix + 'block' (its index)], 'empty', 'unrecognised'} for one clip: every block read."""
    idx = gps_index(path)
    fixes, empty, bad = [], 0, 0
    if idx:
        with open(path, 'rb') as fh:
            for k, (off, size) in enumerate(idx):
                fh.seek(off)
                b = fh.read(min(size, 512))
                f = decode_block(b)
                if f is None:
                    if len(b) >= 16 and not b[16:].strip(b'\0'):
                        empty += 1
                    else:
                        bad += 1
                    continue
                f['block'] = k
                fixes.append(f)
    return {'blocks': len(idx), 'fixes': fixes, 'empty': empty, 'unrecognised': bad}


def track(path: str, gps: dict | None = None, valid_only: bool = True) -> dict:
    """A clip's GPS track on its video time: {'clip', 't0_utc' (video time 0 in UTC, ISO), 'fixes': [{'t', 'utc',
    'lat', 'lon', 'speed_kmh', 'course_deg'}], 'blocks', 'n_fixes', 'repeated', 'notes'}; t0 = median(UTC_k − k)."""
    g = gps or read_gps(path)
    fx = [f for f in g['fixes'] if f['valid'] or not valid_only]
    out = {'clip': os.path.basename(path), 'path': os.path.abspath(path), 'blocks': g['blocks'], 'fixes': [],
           't0_utc': None, 'n_fixes': 0, 'repeated': 0, 'empty': g['empty'], 'unrecognised': g['unrecognised'],
           'notes': []}
    if not fx:
        return out
    ep = np.array([f['utc'].timestamp() for f in fx])
    t0 = float(np.median(ep - np.array([f['block'] for f in fx], float)))
    seen, rep = set(), 0
    for f, e in zip(fx, ep):
        if e in seen:
            rep += 1
            continue
        seen.add(e)
        out['fixes'].append({'t': round(e - t0, 3), 'utc': f['utc'].strftime('%Y-%m-%dT%H:%M:%SZ'),
                             'lat': f['lat'], 'lon': f['lon'], 'speed_kmh': f['speed_kmh'],
                             'course_deg': f['course_deg']})
    out['fixes'].sort(key=lambda r: r['t'])
    out['t0_utc'] = _dt.datetime.fromtimestamp(t0, _dt.timezone.utc).isoformat(timespec='milliseconds')
    out['n_fixes'] = len(out['fixes'])
    out['repeated'] = rep
    if rep:
        out['notes'].append(f'{rep} repeated UTC second(s): first fix kept')
    return out


def position_at(trk: dict, t: float, max_gap: float = MAX_GAP_S) -> dict | None:
    """The interpolated position at video time t: {'lat', 'lon', 'speed_kmh', 'course_deg', 'utc', 'gps':
    'interpolated'|'nearest'}, or None when no fix is within max_gap seconds."""
    fx = trk.get('fixes') or ()
    if not fx:
        return None
    ts = [f['t'] for f in fx]
    import bisect
    j = bisect.bisect_left(ts, t)
    t0 = _dt.datetime.fromisoformat(trk['t0_utc']).timestamp()
    if 0 < j < len(fx) and ts[j] - ts[j - 1] <= 2 * max_gap:
        a, b = fx[j - 1], fx[j]
        w = (t - a['t']) / (b['t'] - a['t']) if b['t'] > a['t'] else 0.0
        lerp = (lambda k: None if a[k] is None or b[k] is None else a[k] + w * (b[k] - a[k]))
        ca, cb = a['course_deg'], b['course_deg']
        crs = None if ca is None or cb is None else (ca + w * (((cb - ca + 180) % 360) - 180)) % 360
        r = {'lat': lerp('lat'), 'lon': lerp('lon'), 'speed_kmh': lerp('speed_kmh'), 'course_deg': crs,
             'gps': 'interpolated'}
    else:
        k = min(range(max(0, j - 1), min(len(fx), j + 1)), key=lambda i: abs(ts[i] - t))
        if abs(ts[k] - t) > max_gap:
            return None
        f = fx[k]
        r = {'lat': f['lat'], 'lon': f['lon'], 'speed_kmh': f['speed_kmh'], 'course_deg': f['course_deg'],
             'gps': 'nearest'}
    r['utc'] = _dt.datetime.fromtimestamp(t0 + t, _dt.timezone.utc).isoformat(timespec='milliseconds')
    return r


def check_track(trk: dict, max_kmh: float = 160.0) -> dict:
    """Plausibility of a decoded track: per consecutive one-second pair, distance / mean reported speed and heading
    minus course; implied speeds above max_kmh counted as jumps."""
    ratios, dh, jumps = [], [], 0
    fx = trk.get('fixes') or ()
    for a, b in zip(fx, fx[1:]):
        dt_ = b['t'] - a['t']
        if dt_ <= 0 or dt_ > 1.5:
            continue
        d, hd = _dist_heading(a['lat'], a['lon'], b['lat'], b['lon'])
        if d / dt_ * 3.6 > max_kmh:
            jumps += 1
        if a['speed_kmh'] is not None and b['speed_kmh'] is not None:
            v = (a['speed_kmh'] + b['speed_kmh']) / 2 / 3.6
            if v > 5:
                ratios.append(d / dt_ / v)
                if b['course_deg'] is not None:
                    dh.append(((hd - b['course_deg'] + 180) % 360) - 180)
    q = (lambda v: None if not v else [round(float(x), 4) for x in np.percentile(v, [10, 50, 90])])
    return {'pairs': len(ratios), 'dist_over_speed_p10_50_90': q(ratios), 'heading_minus_course_p10_50_90': q(dh),
            'jumps': jumps}


def distance_m(la1, lo1, la2, lo2) -> float:
    """The distance between two positions in metres (equirectangular about their mean latitude: within 0.1 % up to
    tens of kilometres)."""
    return _dist_heading(la1, lo1, la2, lo2)[0]


def _dist_heading(la1, lo1, la2, lo2) -> tuple:
    R = 6371008.8
    y = math.radians(la2 - la1) * R
    x = math.radians(lo2 - lo1) * R * math.cos(math.radians((la1 + la2) / 2))
    return math.hypot(x, y), math.degrees(math.atan2(x, y)) % 360


def sun_altitude(lat: float, lon: float, when: _dt.datetime) -> float:
    """The sun's altitude above the horizon in degrees (NOAA's low-precision formulae; ≈ 0.5°; refraction left
    out)."""
    when = when.astimezone(_dt.timezone.utc)
    d = (when - _dt.datetime(2000, 1, 1, 12, tzinfo=_dt.timezone.utc)).total_seconds() / 86400.0
    g = math.radians((357.529 + 0.98560028 * d) % 360)
    q = (280.459 + 0.98564736 * d) % 360
    L = math.radians((q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360)
    e = math.radians(23.439 - 0.00000036 * d)
    dec = math.asin(math.sin(e) * math.sin(L))
    ra = math.atan2(math.cos(e) * math.sin(L), math.cos(L))
    gmst = (18.697374558 + 24.06570982441908 * d) % 24
    ha = math.radians(gmst * 15 + lon) - ra
    la = math.radians(lat)
    return math.degrees(math.asin(math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(ha)))


def clip_summary(c: dict, trk: dict | None = None) -> dict:
    """A one-line view of a clip and its track: fixes, distance, speeds, bounding box, sun altitude at the start."""
    trk = trk or track(c['path'])
    fx = trk['fixes']
    s = {'name': c['name'], 'folder': c.get('folder'), 'start_local': c['start_local'].isoformat()
         if c.get('start_local') else None, 'blocks': trk['blocks'], 'fixes': len(fx), 'distance_m': 0.0}
    if fx:
        s['distance_m'] = round(sum(_dist_heading(a['lat'], a['lon'], b['lat'], b['lon'])[0]
                                    for a, b in zip(fx, fx[1:])), 1)
        sp = [f['speed_kmh'] for f in fx if f['speed_kmh'] is not None]
        s['speed_kmh_mean'] = round(float(np.mean(sp)), 1) if sp else None
        s['moving_share'] = round(float(np.mean([v >= 2.0 for v in sp])), 3) if sp else None
        s['bbox'] = [round(min(f['lat'] for f in fx), 6), round(min(f['lon'] for f in fx), 6),
                     round(max(f['lat'] for f in fx), 6), round(max(f['lon'] for f in fx), 6)]
        s['sun_deg'] = round(sun_altitude(fx[0]['lat'], fx[0]['lon'], _dt.datetime.fromisoformat(
            fx[0]['utc'].replace('Z', '+00:00'))), 1)
    return s


# ================================================================================================ frames
_HW = {}


def hwaccel_works(path: str) -> bool:
    """Whether ffmpeg's -hwaccel cuda decodes this clip's codec (tried once per codec and cached)."""
    try:
        codec = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                'stream=codec_name', '-of', 'csv=p=0', path], capture_output=True, text=True,
                               timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return False
    if codec not in _HW:
        try:
            r = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-xerror', '-hwaccel', 'cuda',
                                '-hwaccel_flags', '+unsafe_output', '-i', path, '-frames:v', '1', '-f', 'null', '-'],
                               capture_output=True, text=True, timeout=60)
            _HW[codec] = r.returncode == 0 and 'Failed' not in r.stderr and 'Error' not in r.stderr
        except (OSError, subprocess.TimeoutExpired):
            _HW[codec] = False
    return _HW[codec]


def keyframe_times(path: str, seconds: float = 4.0) -> list:
    """The key frames' times in the first `seconds` of a clip (ffprobe, key frames only)."""
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-skip_frame', 'nokey',
                        '-show_entries', 'frame=pts_time', '-of', 'csv=p=0', '-read_intervals', f'%+{seconds}', path],
                       capture_output=True, text=True, timeout=60)
    out = []
    for line in r.stdout.split():
        try:
            out.append(float(line.strip(',')))
        except ValueError:
            pass
    return out


def iter_frames(path: str, rate: float | None = 2.0, size: tuple = (640, 360), start: float = 0.0,
                duration: float | None = None, hwaccel: str = 'auto', keyframes: str | bool = 'auto', info=None,
                crop_bottom: float = 0.0):
    """Yield (frame index n, video time n / fps, RGB uint8 array H×W×3) from a clip: every frame (rate None) or
    every round(fps / rate)-th frame from `start`. keyframes 'auto' decodes only the key frames when they fall
    exactly on the requested cadence (much faster: nothing between them is decoded). crop_bottom cuts that share of
    the frame's height off its bottom before scaling (the camera's burned-in date, speed and position line); `size`
    is then the output after the cut. `info` (a dict) receives what was used: hwaccel, keyframes, the decimation
    step."""
    pr = probe(path)
    fps = pr.get('fps') or 30.0
    if pr.get('constant_rate') is False:
        raise DashcamError(f'{path}: variable frame rate; frame times would be guesses')
    W, H = int(size[0]), int(size[1])
    step = 1 if not rate else max(1, int(round(fps / float(rate))))
    n0 = int(round(start * fps))
    use_hw = hwaccel == 'cuda' or (hwaccel == 'auto' and hwaccel_works(path))
    kf = False
    kstep = 1
    if keyframes and step > 1:
        kt = keyframe_times(path, 4.0)
        if len(kt) >= 3:
            gaps = np.diff(kt)
            gap = int(round(float(np.median(gaps)) * fps))
            regular = gap > 0 and np.allclose(gaps * fps, gap, atol=0.01)
            if regular and step % gap == 0 and n0 % gap == 0:
                kf, kstep = True, step // gap
        if keyframes is True and not kf:
            raise DashcamError(f'{path}: key frames are not on a cadence that gives every {step}th frame')
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'info', '-nostats', '-nostdin']   # info: showinfo's lines
    if kf:
        cmd += ['-skip_frame', 'nokey']
    if use_hw:
        cmd += ['-hwaccel', 'cuda']
    if n0:
        cmd += ['-ss', f'{n0 / fps:.6f}']
    if duration:
        cmd += ['-t', f'{float(duration):.6f}']
    sel = f'not(mod(n\\,{kstep}))' if kf else f'not(mod(n\\,{step}))'
    crop = f'crop=iw:trunc(ih*{1.0 - float(crop_bottom):.6f}/2)*2:0:0,' if crop_bottom else ''
    cmd += ['-i', path, '-an', '-vf', f'select={sel},{crop}scale={W}:{H}:flags=area,showinfo',
            '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    if info is not None:
        info.update({'hwaccel': 'cuda' if use_hw else 'none', 'keyframes': kf, 'step': step, 'fps': fps})
    fb = W * H * 3
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    import threading
    pts, err = [], []

    def _read_err():
        for line in p.stderr:
            line = line.decode('utf-8', 'replace')
            m = re.search(r'pts_time:\s*([-\d.]+)', line)
            if m and 'Parsed_showinfo' in line:
                pts.append(float(m.group(1)))
            elif 'Parsed_showinfo' not in line:
                err.append(line)
    th = threading.Thread(target=_read_err, daemon=True)
    th.start()
    i = 0
    try:
        while True:
            buf = p.stdout.read(fb)
            if len(buf) < fb:
                break
            # the frame index: from showinfo's time when it has arrived, else from the cadence
            for _w in range(2000):
                if len(pts) > i or p.poll() is not None:
                    break
                time.sleep(0.0005)
            if len(pts) > i:
                n = n0 + int(round(pts[i] * fps))
            else:
                n = n0 + i * step
            yield n, n / fps, np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            i += 1
    finally:
        if p.poll() is None:
            p.kill()
        p.wait()
        th.join(timeout=5)
    if p.returncode not in (0, -9) and i == 0:
        raise DashcamError(f'ffmpeg failed on {path}: {"".join(err)[-400:]}')


# ================================================================================================ image sets
def plan(clips: list, rate: float = 2.0, min_speed_kmh: float | None = 2.0, min_sun_deg: float | None = None,
         require_gps: bool = False, max_frames: int | None = None, seed: int = 0, tracks: dict | None = None,
         max_sun_deg: float | None = None) -> dict:
    """Which frames to extract: {'items': [{'clip', 'path', 'frame', 't', position…}], 'counts': {…}}. Frames are on
    the rate's cadence; left out: no position (require_gps), slower than min_speed_kmh (frames with no position
    are kept unless require_gps), sun lower than min_sun_deg or higher than max_sun_deg (at the frame's position
    or, without one, at the clip's first fix / Bengaluru with the name's time; a frame whose sun altitude is unknown
    is left out by either bound); then max_frames are taken evenly across the whole set (seeded jitter)."""
    counts = {'clips': 0, 'candidates': 0, 'no_gps': 0, 'static': 0, 'dark': 0, 'bright': 0, 'kept': 0}
    items = []
    for c in clips:
        try:
            pr = probe(c['path'])
        except (DashcamError, OSError, struct.error):
            continue
        fps, n = pr.get('fps') or 30.0, pr.get('n_frames') or 0
        trk = (tracks or {}).get(c['path']) or track(c['path'])
        counts['clips'] += 1
        step = max(1, int(round(fps / rate)))
        ref = (trk['fixes'][0]['lat'], trk['fixes'][0]['lon']) if trk['fixes'] else (12.9716, 77.5946)
        su = c.get('start_utc')
        for fr in range(0, n, step):
            t = fr / fps
            counts['candidates'] += 1
            pos = position_at(trk, t) if trk['fixes'] else None
            if pos is None:
                counts['no_gps'] += 1
                if require_gps:
                    continue
            if min_speed_kmh is not None and pos is not None and pos.get('speed_kmh') is not None \
                    and pos['speed_kmh'] < min_speed_kmh:
                counts['static'] += 1
                continue
            when = (_dt.datetime.fromisoformat(pos['utc']) if pos else
                    (su + _dt.timedelta(seconds=t) if su else None))
            sun = None
            if when is not None:
                sun = sun_altitude(pos['lat'] if pos else ref[0], pos['lon'] if pos else ref[1], when)
            if min_sun_deg is not None and (sun is None or sun < min_sun_deg):
                counts['dark'] += 1
                continue
            if max_sun_deg is not None and (sun is None or sun > max_sun_deg):
                counts['bright'] += 1
                continue
            it = {'clip': c['name'], 'path': c['path'], 'frame': fr, 't': round(t, 4),
                  'sun_deg': None if sun is None else round(sun, 2)}
            if pos:
                it.update({k: pos[k] for k in ('lat', 'lon', 'speed_kmh', 'course_deg', 'utc', 'gps')})
            else:
                it.update({'lat': None, 'lon': None, 'speed_kmh': None, 'course_deg': None,
                           'utc': when.isoformat(timespec='milliseconds') if when else None, 'gps': None})
            items.append(it)
    if max_frames and len(items) > max_frames:
        rng = np.random.default_rng(seed)
        edges = np.linspace(0, len(items), int(max_frames) + 1).astype(int)     # strictly increasing here
        items = [items[a + int(rng.integers(0, b - a))] for a, b in zip(edges, edges[1:])]
    counts['kept'] = len(items)
    return {'items': items, 'counts': counts}


def extract(clips: list, out: str, rate: float = 2.0, size: tuple = (640, 360), min_speed_kmh: float | None = 2.0,
            min_sun_deg: float | None = None, require_gps: bool = False, max_frames: int | None = None,
            seed: int = 0, image_format: str = 'jpg', quality: int = 95, hwaccel: str = 'auto',
            crop_bottom: float = 0.0, progress=None, cancelled=None, argv: list | None = None,
            max_sun_deg: float | None = None) -> dict:
    """Write an image set: out/frames/*.jpg|png, out/frames.jsonl, out/run.json. Returns the run.json document.
    `progress(done, total, message)`; `cancelled()` stops between clips (the set written so far is kept and
    marked). crop_bottom (a share of the height, e.g. 0.06 for the burned-in text line of the camera verified in
    docs/dashcam.md) is cut before scaling: the images are then size[0] × round(size[1] · (1 − crop_bottom)).
    min_sun_deg / max_sun_deg bound the sun's altitude (a day set: min 5; a night set: max NIGHT_SUN_DEG). Every
    frame's exposure (exposure()) is recorded in frames.jsonl, and its medians in run.json."""
    from . import provenance as PV
    from .paths import tilde
    import cv2
    if image_format not in ('jpg', 'png'):
        raise DashcamError('image_format is jpg or png')
    if not clips:
        raise DashcamError('no clips')
    os.makedirs(os.path.join(out, 'frames'), exist_ok=True)
    if os.path.exists(os.path.join(out, 'frames.jsonl')):
        raise DashcamError(f'{out} already holds an image set (frames.jsonl); choose a new folder')
    settings = {'rate': rate, 'size': list(size), 'min_speed_kmh': min_speed_kmh, 'min_sun_deg': min_sun_deg,
                'max_sun_deg': max_sun_deg, 'require_gps': require_gps, 'max_frames': max_frames, 'image_format': image_format,
                'quality': quality, 'hwaccel': hwaccel, 'crop_bottom': crop_bottom, 'tz_offset_h': TZ_OFFSET_H,
                'max_gap_s': MAX_GAP_S}
    if not 0.0 <= float(crop_bottom) < 0.5:
        raise DashcamError('crop_bottom is a share of the height in [0, 0.5)')
    size = (int(size[0]), int(round(size[1] * (1.0 - float(crop_bottom)) / 2)) * 2)
    settings['image_size'] = list(size)
    run = PV.new_run('dashcam-extract', list(argv or ['dashcam', 'extract']), seed, settings)
    run['schema_set'] = SET_SCHEMA
    t_start = time.time()
    tracks = {c['path']: track(c['path']) for c in clips}
    pl = plan(clips, rate, min_speed_kmh, min_sun_deg, require_gps, max_frames, seed, tracks, max_sun_deg)
    by = {}
    for it in pl['items']:
        by.setdefault(it['path'], []).append(it)
    run['provenance']['inputs'] = {}
    run['clips'] = []
    n_done, decode_s, video_s, used = 0, 0.0, 0.0, {}
    expo = []
    status = 'ok'
    with open(os.path.join(out, 'frames.jsonl'), 'w', encoding='utf-8') as fj:
        for ci, c in enumerate(clips):
            want = by.get(c['path'])
            fp = fingerprint(c['path'])
            fp['path'] = tilde(fp['path'])
            trk = tracks[c['path']]
            run['clips'].append({'name': c['name'], 'fingerprint': fp, 'frames': len(want or ()),
                                 'fixes': trk['n_fixes'], 't0_utc': trk['t0_utc']})
            if not want:
                continue
            if cancelled and cancelled():
                status = 'cancelled'
                break
            wanted = {it['frame']: it for it in want}
            first, last = min(wanted), max(wanted)
            fps = probe(c['path']).get('fps') or 30.0
            t0 = time.time()
            info = {}
            for n, t, rgb in iter_frames(c['path'], rate, size, start=first / fps,
                                         duration=(last - first) / fps + 0.5 / fps + 1e-3, hwaccel=hwaccel,
                                         info=info, crop_bottom=crop_bottom):
                it = wanted.pop(n, None)
                if it is None:
                    continue
                stem = f'{os.path.splitext(c["name"])[0]}_{n:05d}.{image_format}'
                rel = os.path.join('frames', stem)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if image_format == 'jpg' else []
                if not cv2.imwrite(os.path.join(out, rel), bgr, params):
                    raise DashcamError(f'cannot write {os.path.join(out, rel)}')
                rec = {'image': rel, 'clip': c['name'], 'frame': n, 't': round(t, 4)}
                rec.update({k: it.get(k) for k in ('utc', 'lat', 'lon', 'speed_kmh', 'course_deg', 'gps',
                                                   'sun_deg')})
                if c.get('start_local') is not None:
                    rec['local'] = (c['start_local'] + _dt.timedelta(seconds=t)).isoformat(timespec='milliseconds')
                rec['exposure'] = exposure(rgb)
                expo.append(rec['exposure'])
                fj.write(json.dumps(rec) + '\n')
                n_done += 1
            decode_s += time.time() - t0
            video_s += (last - first) / fps
            for k, v in info.items():
                used.setdefault(k, set()).add(str(v))
            if wanted:
                run['notes'].append(f'{c["name"]}: {len(wanted)} planned frame(s) not decoded (end of stream)')
            if progress:
                progress(ci + 1, len(clips), f'{c["name"]}: {n_done} frames')
    run['status'] = status
    run['exit_code'] = 0
    run['finished'] = PV.utc_now()
    run['counts'] = dict(pl['counts'], written=n_done)
    run['timings'] = {'total_s': round(time.time() - t_start, 2), 'decode_s': round(decode_s, 2),
                      'frames_per_s': round(n_done / decode_s, 2) if decode_s else None,
                      'video_s': round(video_s, 1), 'video_s_per_s': round(video_s / decode_s, 1) if decode_s else None}
    run['decoder'] = {k: sorted(v) for k, v in used.items()}
    run['exposure'] = {'median': {k: round(float(np.median([e[k] for e in expo])), 5) for k in expo[0]},
                       'p10': {k: round(float(np.percentile([e[k] for e in expo], 10)), 5) for k in expo[0]},
                       'p90': {k: round(float(np.percentile([e[k] for e in expo], 90)), 5) for k in expo[0]},
                       'method': EXPOSURE_METHOD} if expo else None
    run['notes'].append('clips fingerprinted by size, mtime and the SHA-256 of their first MiB (not the whole file)')
    run['outputs'] = {'frames.jsonl': PV.sha256_file(os.path.join(out, 'frames.jsonl')), 'frames': n_done}
    PV.write_json(os.path.join(out, 'run.json'), run)
    return run


EXPOSURE_METHOD = (f'Rec. 601 luma of the written frame (after the crop and the scaling): mean, 1st/50th/99th '
                   f'percentiles, the share at or below {DARK_LUMA} (crushed shadows) and at or above {CLIPPED_LUMA} '
                   '(clipped highlights)')


def exposure(rgb: np.ndarray) -> dict:
    """One frame's exposure: luma mean, percentiles and the crushed and clipped shares (EXPOSURE_METHOD)."""
    y = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    p1, p50, p99 = np.percentile(y, [1, 50, 99])
    return {'luma_mean': round(float(y.mean()), 3), 'luma_p01': round(float(p1), 2), 'luma_p50': round(float(p50), 2),
            'luma_p99': round(float(p99), 2), 'dark_share': round(float((y <= DARK_LUMA).mean()), 5),
            'clipped_share': round(float((y >= CLIPPED_LUMA).mean()), 5)}


def read_set(folder: str) -> dict:
    """An image set written by extract(): {'path', 'run', 'items': [frames.jsonl records + 'image' as an absolute
    path]}."""
    fj = os.path.join(folder, 'frames.jsonl')
    if not os.path.isfile(fj):
        raise DashcamError(f'{folder}: no frames.jsonl (not a dashcam image set)')
    items = []
    with open(fj, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                r['image'] = os.path.join(folder, r['image'])
                items.append(r)
    run = {}
    if os.path.isfile(os.path.join(folder, 'run.json')):
        with open(os.path.join(folder, 'run.json'), encoding='utf-8') as fh:
            run = json.load(fh)
    return {'path': os.path.abspath(folder), 'run': run, 'items': items}


def default_set_dir(name: str) -> str:
    return os.path.join(DEFAULT_SETS, name)


# ================================================================================================ the imported world
MATCH_MARGIN_M = 50.0                 # a fix this close to the imported lanes' extent counts as "near the world"
LANE_NEAR_M = 8.0                     # = core.logreplay.mapmatch.SEARCH_M


def world_geo(project_path: str, world_id: str) -> dict:
    """A project's imported world as the matching needs it: {'id', 'lat0', 'lon0', 'bbox' [s w n e], 'xodr',
    'lanes' (N, 4) x, y, yaw, width in CARLA's frame, 'lanes_from'}."""
    from .logreplay.mapmatch import lanes_from_xodr
    from .logreplay.plan import world_info
    from .project import Project
    p = Project.load(project_path)
    w = world_info(p, world_id)
    if not w.get('geo'):
        raise DashcamError(f'world {world_id!r} has no geographic reference (a CARLA town is not on the map)')
    out = {'id': world_id, 'name': w.get('name'), 'lat0': w['geo']['lat0'], 'lon0': w['geo']['lon0'],
           'bbox': w.get('bbox'), 'xodr': w.get('xodr'), 'lanes': np.zeros((0, 4)), 'lanes_from': None}
    if w.get('xodr') and os.path.isfile(w['xodr']):
        with open(w['xodr'], encoding='utf-8') as fh:
            out['lanes'] = lanes_from_xodr(fh.read())
        out['lanes_from'] = os.path.basename(w['xodr'])
    return out


def to_world(world: dict, lat, lon, course=None) -> tuple:
    """Latitude/longitude (and course from north) → CARLA's frame of the world: x east, y south, yaw degrees."""
    from .osm import local_projection
    pr = local_projection(world['lat0'], world['lon0'])
    e, n = pr.to_local(np.asarray(lat, float), np.asarray(lon, float))
    x, y = np.asarray(e, float), -np.asarray(n, float)
    yaw = None if course is None else (np.asarray(course, float) - 90.0 + 180.0) % 360.0 - 180.0
    return x, y, yaw


def _dense(fixes: list, step_s: float = 0.2) -> tuple:
    t = np.array([f['t'] for f in fixes], float)
    la = np.array([f['lat'] for f in fixes], float)
    lo = np.array([f['lon'] for f in fixes], float)
    if len(t) < 2:
        return t, la, lo
    tq = np.arange(t[0], t[-1] + 1e-9, step_s)
    gap = np.diff(t)
    keep = np.ones(len(tq), bool)
    j = np.clip(np.searchsorted(t, tq) - 1, 0, len(gap) - 1)
    keep &= gap[j] <= 2 * MAX_GAP_S                      # no interpolation across a hole in the track
    return tq[keep], np.interp(tq, t, la)[keep], np.interp(tq, t, lo)[keep]


def match_clip(world: dict, trk: dict) -> dict:
    """How a clip's track relates to the imported world: fixes inside the import box, distance to it, the
    stretches near the imported lanes and their lateral distance to the nearest lane (and, when a stretch is long
    enough, core.logreplay.mapmatch's verdict on it)."""
    from .logreplay import mapmatch as MM
    fx = trk['fixes']
    out = {'clip': trk['clip'], 'fixes': len(fx), 'inside_box': 0, 'box_distance_m': None, 'near_lanes': 0,
           'lane_distance_m': None, 'segments': [], 'match': None}
    if not fx:
        return out
    lat = np.array([f['lat'] for f in fx])
    lon = np.array([f['lon'] for f in fx])
    x, y, _ = to_world(world, lat, lon)
    if world.get('bbox'):
        s, w, n, e = world['bbox']
        inside = (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)
        out['inside_box'] = int(inside.sum())
        bx0, by_n = to_world(world, s, w)[:2]
        bx1, by_s = to_world(world, n, e)[:2]
        dx = np.maximum(0, np.maximum(min(bx0, bx1) - x, x - max(bx0, bx1)))
        dy = np.maximum(0, np.maximum(min(by_n, by_s) - y, y - max(by_n, by_s)))
        out['box_distance_m'] = round(float(np.min(np.hypot(dx, dy))), 1)
    L = world['lanes']
    if not len(L):
        return out
    from scipy.spatial import cKDTree
    tree = cKDTree(L[:, :2])
    d, _ = tree.query(np.column_stack([x, y]))
    out['lane_distance_m'] = round(float(d.min()), 1)
    near = d <= MATCH_MARGIN_M
    out['near_lanes'] = int(near.sum())
    if not near.any():
        return out
    # stretches of consecutive fixes near the lanes
    idx = np.flatnonzero(near)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    for r in runs:
        seg = [fx[i] for i in r]
        tq, la, lo = _dense(seg)
        sx, sy, _ = to_world(world, la, lo)
        dd, _ = tree.query(np.column_stack([sx, sy]))
        row = {'t0': seg[0]['t'], 't1': seg[-1]['t'], 'fixes': len(seg),
               'lane_distance_m': {'median': round(float(np.median(dd)), 2),
                                   'p95': round(float(np.percentile(dd, 95)), 2), 'max': round(float(dd.max()), 2)},
               'speed_kmh_mean': round(float(np.mean([f['speed_kmh'] or 0 for f in seg])), 1)}
        if len(seg) >= 3:
            crs = np.interp(tq, [f['t'] for f in seg], np.unwrap(np.radians([f['course_deg'] or 0 for f in seg])))
            yaw = (np.degrees(crs) - 90.0 + 180.0) % 360.0 - 180.0
            m = MM.match(sx, sy, yaw, L, world.get('lanes_from') or '')
            row['match'] = {k: m.get(k) for k in ('verdict', 'why', 'share_on_lane_in_direction', 'share_on_lane',
                                                  'median_lateral_m', 'p95_lateral_m')}
        out['segments'].append(row)
    best = max(out['segments'], key=lambda r: r['fixes'])
    out['match'] = (best.get('match') or {}).get('verdict')
    return out


def suggest_area(tracks: list, size_m: float = 1000.0, min_speed_kmh: float = 2.0, min_sun_deg: float | None = None
                 ) -> dict | None:
    """The size_m square (sliding on a size_m / 4 grid) holding the most moving fixes of the tracks: where an imported
    world would match the most footage. {'centre' [lat, lon], 'fixes', 'clips'}."""
    from .osm import local_projection
    rows = []
    for trk in tracks:
        for f in trk['fixes']:
            if f['speed_kmh'] is None or f['speed_kmh'] < min_speed_kmh:
                continue
            dark = min_sun_deg is not None and sun_altitude(f['lat'], f['lon'], _dt.datetime.fromisoformat(
                f['utc'].replace('Z', '+00:00'))) < min_sun_deg
            rows.append((f['lat'], f['lon'], trk['clip'], dark))
    if not rows:
        return None
    # one grid for every filter: projected about the median of all moving fixes
    pr = local_projection(float(np.median([r[0] for r in rows])), float(np.median([r[1] for r in rows])))
    rows = [r for r in rows if not r[3]]
    if not rows:
        return None
    e, n = pr.to_local(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]))
    q = 4                                                 # the square slides on a size_m / q grid
    h = size_m / q
    ke, kn = np.floor(e / h).astype(int), np.floor(n / h).astype(int)
    cells = {}
    for i, k in enumerate(zip(ke, kn)):
        cells.setdefault(k, []).append(i)
    best, best_n = None, -1
    for (a, b) in cells:
        for a0 in range(a - q + 1, a + 1):
            for b0 in range(b - q + 1, b + 1):
                m = sum(len(cells.get((a0 + i, b0 + j), ())) for i in range(q) for j in range(q))
                if m > best_n:
                    best, best_n = (a0, b0), m
    a0, b0 = best
    members = [i for i_ in range(q) for j_ in range(q) for i in cells.get((a0 + i_, b0 + j_), ())]
    c_lat, c_lon = pr.to_geo((a0 + q / 2) * h, (b0 + q / 2) * h)
    return {'centre': [round(float(c_lat), 6), round(float(c_lon), 6)], 'size_m': size_m, 'fixes': best_n,
            'clips': sorted({rows[i][2] for i in members})}


def match_world(project_path: str, world_id: str, clips: list) -> dict:
    """Every clip against the imported world (match_clip) and where an import would match the most footage."""
    w = world_geo(project_path, world_id)
    tracks = [track(c['path']) for c in clips]
    rows = [match_clip(w, t) for t in tracks]
    return {'world': {k: w[k] for k in ('id', 'name', 'lat0', 'lon0', 'bbox', 'lanes_from')},
            'lane_points': int(len(w['lanes'])), 'clips': rows,
            'inside': [r['clip'] for r in rows if r['inside_box']],
            'near_lanes': [r['clip'] for r in rows if r['near_lanes']],
            'nearest_m': min((r['box_distance_m'] for r in rows if r['box_distance_m'] is not None), default=None),
            'suggest': suggest_area(tracks), 'suggest_daytime': suggest_area(tracks, min_sun_deg=0.0),
            'rule': f'near: within {MATCH_MARGIN_M:g} m of an imported driving lane centre; per stretch the lateral '
                    f'distance to the nearest lane centre (any direction) and core.logreplay.mapmatch\'s verdict '
                    f'(its heuristic rule) on the track interpolated every 0.2 s'}


# ================================================================================================ the command
def _select(a) -> list:
    clips = list_clips(a.root)
    if getattr(a, 'include', None):
        import fnmatch
        clips = [c for c in clips if any(fnmatch.fnmatch(c['name'], p) for p in a.include)]
    if getattr(a, 'holdout', None):
        k = int(a.holdout)
        clips = [c for i, c in enumerate(clips) if (i % k == 0) == (a.split == 'holdout')]
    return clips


def main_dashcam(argv) -> int:
    """`falcon dashcam list|gps|extract|match …`: 0 ok, 2 failed."""
    import argparse
    from . import protocol as P
    from .paths import tilde
    ap = argparse.ArgumentParser(prog='falcon dashcam', description='a dashcam\'s clips, GPS track and frames '
                                                                    '(docs/dashcam.md)')
    sub = ap.add_subparsers(dest='action', required=True)

    def common(sp):
        sp.add_argument('--root', default=DEFAULT_ROOT, help='a folder of clips (recursively) or one clip')
        sp.add_argument('--include', action='append', default=None, metavar='GLOB', help='only clips whose name '
                                                                                         'matches (repeatable)')
        sp.add_argument('--holdout', type=int, help='split by clip: every Nth clip (index %% N == 0) is held out')
        sp.add_argument('--split', choices=('train', 'holdout'), default='train')
    ls_ = sub.add_parser('list', help='the clips with their GPS fixes, distance, speed and sun altitude')
    common(ls_)
    ls_.add_argument('--json', action='store_true')
    gp = sub.add_parser('gps', help='a clip\'s decoded GPS track (and its plausibility check)')
    gp.add_argument('clip')
    gp.add_argument('--json', action='store_true')
    gp.add_argument('--raw', action='store_true', help='every block as decoded, before the time base')
    ex = sub.add_parser('extract', help='write an image set (frames + frames.jsonl + run.json)')
    common(ex)
    ex.add_argument('--out', '-o', default='', help=f'the set folder (default {tilde(DEFAULT_SETS)}/NAME)')
    ex.add_argument('--name', default='')
    ex.add_argument('--rate', type=float, default=2.0, help='frames per second of video (0: every frame)')
    ex.add_argument('--size', default='640x360')
    ex.add_argument('--min-speed', type=float, default=2.0, help='km/h; slower frames are left out (-1: keep all)')
    tod = ex.add_mutually_exclusive_group()
    tod.add_argument('--day', action='store_true', help='daytime only: the sun at least --min-sun degrees high')
    tod.add_argument('--night', action='store_true', help=f'night only: the sun lower than --max-sun degrees '
                                                          f'(default {NIGHT_SUN_DEG:g}: civil twilight excluded)')
    tod.add_argument('--twilight', action='store_true', help=f'civil twilight only: the sun between '
                                                             f'{NIGHT_SUN_DEG:g}° and 0°')
    ex.add_argument('--min-sun', type=float, default=0.0)
    ex.add_argument('--max-sun', type=float, default=NIGHT_SUN_DEG)
    ex.add_argument('--require-gps', action='store_true')
    ex.add_argument('--max-frames', type=int)
    ex.add_argument('--seed', type=int, default=0)
    ex.add_argument('--png', action='store_true')
    ex.add_argument('--hwaccel', choices=('auto', 'cuda', 'none'), default='auto')
    ex.add_argument('--crop-bottom', type=float, default=0.0, help='cut this share of the height off the bottom '
                                                                   '(the burned-in text line: 0.06 on the camera '
                                                                   'verified in docs/dashcam.md)')
    mt = sub.add_parser('match', help='which clips drive inside a project\'s imported world, and how far from its '
                                      'lanes')
    common(mt)
    mt.add_argument('--project', required=True)
    mt.add_argument('--world', required=True)
    mt.add_argument('--json', action='store_true')
    mt.add_argument('--out', default='', help='write match.json here')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    em = P.Emitter()
    em.hello('dashcam')
    try:
        if a.action == 'list':
            rows = [clip_summary(c) for c in _select(a)]
            if a.json:
                print(json.dumps(rows, indent=1), flush=True)
            else:
                for r in rows:
                    print(f'{r["name"]:30s} {r["folder"] or "":8s} fixes {r["fixes"]:3d}/{r["blocks"]:<3d} '
                          f'{r["distance_m"]:7.0f} m  {_fmt(r.get("speed_kmh_mean"), "km/h")}  moving '
                          f'{_fmt(r.get("moving_share"))}  sun {_fmt(r.get("sun_deg"), "°")}', flush=True)
                print(f'{len(rows)} clips, {sum(r["fixes"] for r in rows)} fixes, '
                      f'{sum(r["distance_m"] for r in rows) / 1000:.1f} km', flush=True)
            em.result('ok', clips=len(rows))
            return P.OK
        if a.action == 'gps':
            if a.raw:
                g = read_gps(a.clip)
                for f in g['fixes']:
                    f['utc'] = f['utc'].isoformat()
                doc = g
            else:
                doc = track(a.clip)
                doc['check'] = check_track(doc)
            if a.json or a.raw:
                print(json.dumps(doc, indent=1, default=str), flush=True)
            else:
                print(f'{doc["clip"]}: {doc["n_fixes"]} fixes of {doc["blocks"]} blocks ({doc["empty"]} empty, '
                      f'{doc["unrecognised"]} unrecognised); video time 0 = {doc["t0_utc"]}', flush=True)
                for f in doc['fixes']:
                    print(f'  t {f["t"]:6.1f} s  {f["utc"]}  {f["lat"]:.6f} {f["lon"]:.6f}  '
                          f'{_fmt(f["speed_kmh"], "km/h")}  course {_fmt(f["course_deg"], "°")}', flush=True)
                c = doc['check']
                print(f'check: distance / speed (p10 p50 p90) {c["dist_over_speed_p10_50_90"]}, heading − course '
                      f'{c["heading_minus_course_p10_50_90"]}, jumps {c["jumps"]} ({c["pairs"]} pairs)', flush=True)
            em.result('ok')
            return P.OK
        if a.action == 'extract':
            clips = _select(a)
            out = a.out or default_set_dir(a.name or time.strftime('%Y%m%d-%H%M%S-dashcam'))
            w, h = (int(v) for v in a.size.lower().split('x'))
            em.log(f'{len(clips)} clips → {tilde(out)}')
            doc = extract(clips, out, rate=a.rate or None, size=(w, h),
                          min_speed_kmh=None if a.min_speed < 0 else a.min_speed,
                          min_sun_deg=a.min_sun if a.day else NIGHT_SUN_DEG if a.twilight else None,
                          max_sun_deg=a.max_sun if a.night else 0.0 if a.twilight else None,
                          require_gps=a.require_gps,
                          max_frames=a.max_frames, seed=a.seed, image_format='png' if a.png else 'jpg',
                          hwaccel=a.hwaccel, crop_bottom=a.crop_bottom, progress=lambda i, n, m: em.progress(i / n, m),
                          argv=['dashcam'] + list(argv))
            c, t = doc['counts'], doc['timings']
            print(f'{tilde(out)}: {c["written"]} frames from {c["clips"]} clips ({c["candidates"]} candidates; left '
                  f'out {c["static"]} static, {c["dark"]} dark, {c["bright"]} too bright, {c["no_gps"]} without a fix'
                  f'{" (kept)" if not a.require_gps else ""}); {t["frames_per_s"]} frames/s decoding, '
                  f'{t["total_s"]} s in all; decoder {doc["decoder"]}', flush=True)
            em.result('ok', set=out, frames=c['written'])
            return P.OK
        # match
        doc = match_world(a.project, a.world, _select(a))
        if a.out:
            from .provenance import block, write_json
            doc['provenance'] = block({'project': a.project}, None, {'world': a.world, 'root': tilde(a.root)})
            write_json(os.path.join(a.out, 'match.json'), doc)
        if a.json:
            print(json.dumps(doc, indent=1, default=str), flush=True)
        else:
            print(match_text(doc), flush=True)
        em.result('ok', inside=len(doc['inside']), near=len(doc['near_lanes']))
        return P.OK
    except (DashcamError, OSError, ValueError) as e:
        print(f'failed: {e}', flush=True)
        em.result('failed', error=str(e))
        return P.FAILED


def match_text(doc: dict) -> str:
    w = doc['world']
    lines = [f'world {w["id"]} ({w.get("name") or ""}): centre {w["lat0"]:.5f} {w["lon0"]:.5f}, box '
             f'{w.get("bbox")}, {doc["lane_points"]} lane points from {w.get("lanes_from")}']
    for r in doc['clips']:
        if not r['near_lanes'] and not r['inside_box']:
            continue
        lines.append(f'  {r["clip"]}: {r["inside_box"]} fixes inside the box, {r["near_lanes"]} near the lanes')
        for s in r['segments']:
            m = s.get('match') or {}
            lines.append(f'    t {s["t0"]:.0f}–{s["t1"]:.0f} s ({s["fixes"]} fixes, {s["speed_kmh_mean"]} km/h): lane '
                         f'distance median {s["lane_distance_m"]["median"]} m, p95 {s["lane_distance_m"]["p95"]} m'
                         f'{"; " + m["verdict"] + " — " + m["why"] if m else ""}')
    lines.append(f'{len(doc["inside"])} of {len(doc["clips"])} clips drive inside the imported box, '
                 f'{len(doc["near_lanes"])} come within {MATCH_MARGIN_M:g} m of its lanes; the nearest fix is '
                 f'{_fmt(doc["nearest_m"], "m")} from the box')
    for k, what in (('suggest', 'most moving footage'), ('suggest_daytime', 'most moving daytime footage')):
        s = doc.get(k)
        if s:
            lines.append(f'a {s["size_m"]:g} m import centred on {s["centre"][0]:.5f} {s["centre"][1]:.5f} would hold '
                         f'the {what}: {s["fixes"]} fixes from {len(s["clips"])} clips')
    return '\n'.join(lines)


def _fmt(v, unit: str = '') -> str:
    if v is None:
        return '—'
    return f'{v:.1f}{" " + unit if unit and unit != "°" else unit}'


# ================================================================================================ synthetic clips
def encode_block(utc: _dt.datetime, lat: float, lon: float, speed_knots: float = 0.0, course_deg: float = 0.0,
                 tz_offset_h: float = TZ_OFFSET_H, valid: bool = True, size: int = BLOCK_SIZE) -> bytes:
    """A GPS block in the layout decode_block() reads (the inverse, for synthetic test clips)."""
    def ddmm(v):
        d = int(abs(v))
        return d * 100 + (abs(v) - d) * 60
    utc = utc.astimezone(_dt.timezone.utc)
    loc = utc + _dt.timedelta(hours=tz_offset_h)
    p = bytearray(144)
    p[0:12] = SIGNATURE + b'\0\0'
    struct.pack_into('<ff', p, 24, ddmm(lat), ddmm(lon))
    struct.pack_into('<6i', p, 32, utc.hour, utc.minute, utc.second, utc.year - 2000, utc.month, utc.day)
    p[56:60] = (b'A' if valid else b'V') + (b'N' if lat >= 0 else b'S') + (b'E' if lon >= 0 else b'W') + b'\0'
    struct.pack_into('<ff', p, 80, speed_knots, course_deg)
    p[88:104] = b'1234567890123456'
    p[104:108] = b'YSKJ'
    struct.pack_into('<6i', p, 120, loc.hour, loc.minute, loc.second, loc.year, loc.month, loc.day)
    head = struct.pack('>I', size) + b'freeGPS ' + struct.pack('<I', len(p))
    return head + bytes(p) + bytes(size - 16 - len(p))


def empty_block(size: int = BLOCK_SIZE) -> bytes:
    return struct.pack('>I', size) + b'freeGPS ' + struct.pack('<I', 144) + bytes(size - 16)


def add_gps(path: str, blocks: list):
    """Append GPS blocks to an MP4 whose moov box is its last box (as ffmpeg writes it): each block is a valid
    top-level 'free' box (its type reads 'free'), and a moov/'gps ' index pointing to them is added. For synthetic
    test clips; the media data does not move, so its sample offsets stay valid."""
    size = os.path.getsize(path)
    with open(path, 'rb') as fh:
        buf = fh.read()
    last = None
    for typ, a, b in _boxes(buf):
        last = (typ, a, b)
    if last is None or last[0] != b'moov' or last[2] != size:
        raise DashcamError(f'{path}: moov is not the last box')
    start = last[1] - 8
    body = buf[last[1]:last[2]]
    entries_len = 8 + 8 + 8 * len(blocks)
    base = size + entries_len                  # the blocks follow the enlarged moov
    gps = struct.pack('>I4sII', entries_len, b'gps ', 0x101, len(blocks))
    off = base
    for b in blocks:
        gps += struct.pack('>II', off, len(b))
        off += len(b)
    moov = struct.pack('>I4s', 8 + len(body) + len(gps), b'moov') + body + gps
    with open(path, 'wb') as fh:
        fh.write(buf[:start] + moov + b''.join(blocks))
