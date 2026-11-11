"""The road users a dashcam ingest tracked, as a reconstruction's moving actors (numpy; docs/reconstruction.md,
"Moving road users from a dashcam").

`falcon dashcam ingest` (core.dashcam_ingest, core.nearmiss) leaves, per clip, a perception record — every processed
frame's ByteTrack boxes with their fused monocular range in the camera's levelled frame (forward, left; core.
dashcam_depth) — and the mined ego motion (the GPS through a Kalman/RTS smoother, the odometry's heading on the
candidate windows). A reconstruction posed from the same clip (core.recon.fromdashcam) has its own world: the
segment's visual odometry, metric by the GPS speed and adjusted towards the GPS. This module carries the tracks from
the first into the second, frame by frame:

    A_recon(t) = C_recon(t) · E_nearmiss(t)⁻¹ · A_nearmiss(t)

* A_nearmiss(t): the track's box in the ingest's world (local east-north-up about the clip's first fix) — its
  RTS-smoothed trajectory (core.nearmiss.actor_traj: the range's σ along the ray, 0.3 m + 1 % across it, 3 m/s²
  process noise for vehicles, 1 m/s² for walkers; a vehicle's centre half its typical length beyond its near face),
  heading from the smoothed velocity above 1 m/s (else the ego's), its centre half the kind's typical height above
  the road (z = 0);
* E_nearmiss(t): the ego's pose that the ranges were attached to (core.nearmiss.ego_pose: the camera's levelled
  frame — x the camera's horizontal forward direction, the origin on the road under the camera);
* C_recon(t): the same levelled camera frame in the reconstruction, from the posed frames' camera poses (x the
  optical axis made horizontal, the origin on z = 0 under the camera centre), interpolated linearly in time
  (position and unwrapped yaw) between the posed frames.

So each box keeps where the ingest measured it relative to the camera (smoothed over the track), and is placed
where the reconstruction's own camera was — the images the actor is learned from are the reconstruction's, posed by
its odometry, not by the ingest's GPS track. Sizes are the kind's typical ones (core.scenario_plan.SIZES, HEIGHTS
here), not measured; the learned rigid correction of core.recon.actors absorbs a box that is a little off.

The tracks are written into the posed folder (posed/tracks.json, falcon.recon_tracks/1: core.recon.actors.Track
documents on the posed frames' clock) and named in posed.json ('tracks'); core.recon.data.posed_capture reads them in
dynamic mode (learned as actors) and masks their boxes in static mode.

Privacy: everything read and written here is derived from the user's footage (the ingest folder, the posed folder of
a reconstruction with dashcam.json): it stays on this machine; nothing here is sent anywhere.
"""
from __future__ import annotations

import math
import os

import numpy as np

SCHEMA = 'falcon.recon_tracks/1'
FILE = 'tracks.json'
MIN_OBS = 3                     # core.nearmiss.MIN_OBS: fewer observations make no trajectory
PAD_S = 0.5                     # track observations kept this far outside the posed frames' span
# a typical road user's height (m) by kind — heuristic, like core.scenario_plan.SIZES' lengths and widths (a
# two-wheeler's and a bicycle's with their rider)
HEIGHTS = {'car': 1.5, 'van': 2.0, 'truck': 3.0, 'bus': 3.2, 'autorickshaw': 1.7, 'motorcycle': 1.5, 'bicycle': 1.6,
           'pedestrian': 1.7, 'child': 1.2, 'cattle': 1.5, 'cycle_rickshaw': 1.8, 'handcart': 1.0, 'vendor': 1.6}
DEFAULT_HEIGHT = 1.5
CATEGORY = {'car': 'vehicle.car', 'van': 'vehicle.car', 'truck': 'vehicle.truck', 'bus': 'vehicle.bus.rigid',
            'autorickshaw': 'vehicle.autorickshaw', 'motorcycle': 'vehicle.motorcycle', 'bicycle': 'vehicle.bicycle',
            'pedestrian': 'human.pedestrian.adult', 'child': 'human.pedestrian.child', 'cattle': 'animal'}
CHAIN = ('A_recon(t) = C_recon(t) · E_nearmiss(t)⁻¹ · A_nearmiss(t): the ingest\'s RTS-smoothed box relative to the '
         'camera it was measured from (E: core.nearmiss.ego_pose, the camera\'s levelled frame on the road), placed '
         'at the reconstruction\'s own levelled camera frame (C: the posed frames\' optical axis made horizontal, '
         'the origin on z = 0 under the camera; linear in time between posed frames)')


class TracksError(ValueError):
    pass


def size_of(kind: str) -> tuple:
    """(length, width, height) m of a kind: core.nearmiss's typical length and width (core.scenario_plan.SIZES, the
    ego's for a car) and HEIGHTS."""
    from ..scenario_plan import EGO_SIZE, SIZES
    L, W = SIZES.get(kind, EGO_SIZE)
    return float(L), float(W), float(HEIGHTS.get(kind, DEFAULT_HEIGHT))


# ================================================================================================ the ingest
def find_ingest(clip: str, where: str | None = None) -> dict:
    """The ingest that perceived `clip`: {'ingest' (its falcon-ingest folder), 'clip_key', 'perception',
    'nearmiss' (or None)}. where: an ingested folder, its falcon-ingest/ folder or an exported event folder
    (falcon-ingest/events/<id>/); None looks for falcon-ingest/ beside the clip and up to three folders above it."""
    from .. import dashcam as DC
    from .. import dashcam_ingest as I
    clip = os.path.abspath(os.path.expanduser(clip))
    ck = I.clip_key(clip)
    if where:
        w = os.path.abspath(os.path.expanduser(where))
        if os.path.isfile(os.path.join(w, 'event.json')) and os.path.basename(os.path.dirname(w)) == 'events':
            cands = [os.path.dirname(os.path.dirname(w))]
        elif os.path.basename(w) == I.INGEST_DIR:
            cands = [w]
        else:
            cands = [I.out_dir(w)]
    else:
        cands, d = [], os.path.dirname(clip)
        for _ in range(4):
            cands.append(I.out_dir(d))
            up = os.path.dirname(d)
            if up == d:
                break
            d = up
    for od in cands:
        p = os.path.join(od, 'clips', ck, 'perception.json')
        if os.path.isfile(p):
            from ..provenance import read_json
            perc = read_json(p)
            fp = DC.fingerprint(clip)
            got = perc.get('fingerprint') or {}
            if got.get('sha256_head') != fp['sha256_head'] or got.get('size') != fp['size']:
                raise TracksError('the ingest perceived another version of this clip (its fingerprint differs): '
                                  'run `falcon dashcam ingest` on its folder again')
            mp = os.path.join(od, 'clips', ck, 'nearmiss.json')
            return {'ingest': od, 'clip_key': ck, 'perception': p, 'nearmiss': mp if os.path.isfile(mp) else None}
    raise TracksError('no ingest of this clip ' + (f'in {where}' if where else 'beside it') +
                      ': run `falcon dashcam ingest FOLDER` on the folder of clips first (its tracks are what '
                      'dynamic mode learns), or give --tracks')


def nearmiss_world(perc: dict, mined: dict | None = None) -> tuple:
    """(ego, tracks) of a clip in the ingest's world: ego as core.nearmiss.ego_motion gives it — the mined
    ego_track when it belongs to this perception (the odometry's heading on the candidate windows included), else
    the GPS's again — and {track id: {'kind', 'class_source', 'rows': [(frame k, t, box)]}} of every track with a
    range, its kind the perception's majority."""
    from .. import nearmiss as NM
    times = np.array([f['t'] for f in perc['frames']], float)
    eg = (mined or {}).get('ego_track')
    if eg and mined.get('perception_created') == perc.get('created') and len(eg.get('t') or ()) == len(times):
        ego = {k: (np.asarray(v, float) if isinstance(v, list) and k != 'ok' else v) for k, v in eg.items()}
        ego['ok'] = np.asarray(eg.get('ok') or [True] * len(times), bool)
        ego['heading'] = 'gps and the odometry on the candidate windows (core.nearmiss)'
    else:
        try:
            ego = NM.ego_motion(perc.get('gps') or {}, times)
        except NM.NearMissError as e:
            raise TracksError(f'no ego motion for the clip: {e}') from None
        ego['heading'] = 'gps'
    out = {}
    for tid, rows in NM._tracks(perc).items():
        rows = [r for r in rows if bool(ego['ok'][r[0]])]
        if not rows:
            continue
        kinds = [r[2].get('kind') for r in rows]
        kind = max(set(kinds), key=kinds.count)
        src = 'heuristic' if any(r[2].get('class_source') == 'heuristic' and r[2].get('kind') == kind
                                 for r in rows) else 'model'
        out[int(tid)] = {'kind': kind, 'class_source': src, 'rows': rows}
    return ego, out


# ================================================================================================ the chain
def level_frame(viewmat) -> tuple:
    """(x, y, yaw rad) of a camera's levelled frame in its world: under the camera centre on z = 0, x the optical
    axis made horizontal."""
    V = np.asarray(viewmat, float).reshape(4, 4)
    R = V[:3, :3]
    C = -R.T @ V[:3, 3]
    ax = R[2]
    return float(C[0]), float(C[1]), math.atan2(ax[1], ax[0])


def _mat(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = [x, y, z]
    return T


class LevelPath:
    """The reconstruction's levelled camera frames over time (video seconds), interpolated linearly (position and
    unwrapped yaw) between the posed frames, held at the ends."""

    def __init__(self, frames: list):
        rows = sorted(((float(f['video_t']),) + level_frame(f['viewmat']) for f in frames), key=lambda r: r[0])
        a = np.array(rows, float)
        self.t, self.x, self.y = a[:, 0], a[:, 1], a[:, 2]
        self.yaw = np.unwrap(a[:, 3])

    def at(self, t: float) -> np.ndarray:
        return _mat(float(np.interp(t, self.t, self.x)), float(np.interp(t, self.t, self.y)),
                    float(np.interp(t, self.t, self.yaw)))


def recon_tracks(frames: list, perc: dict, mined: dict | None = None, pad_s: float = PAD_S) -> tuple:
    """({track id: core.recon.actors.Track}, report) of a clip's tracked road users in a reconstruction's world
    (module doc). frames: the posed frames (posed.json's: 'video_t', 't', 'viewmat'); times on the posed frames'
    clock (Frame.t: seconds from the first posed frame)."""
    from .. import nearmiss as NM
    from .actors import build_tracks, rot_to_quat_np
    if len(frames) < 2:
        raise TracksError('fewer than two posed frames')
    path = LevelPath(frames)
    f0 = min(frames, key=lambda f: float(f['video_t']))
    clock0 = float(f0['video_t']) - float(f0['t'])            # video time of the posed clock's zero
    lo, hi = float(path.t[0]) - pad_s, float(path.t[-1]) + pad_s
    ego, tracks = nearmiss_world(perc, mined)
    obs, info, meta = {}, {}, {}
    left = {'too_few_observations': 0, 'outside_the_segment': 0}
    for tid, tr in sorted(tracks.items()):
        rows = [r for r in tr['rows'] if lo <= r[1] <= hi]
        if not rows:
            left['outside_the_segment'] += 1
            continue
        if len(rows) < MIN_OBS:
            left['too_few_observations'] += 1
            continue
        kind = tr['kind']
        L, W, H = size_of(kind)
        traj = NM.actor_traj(rows, ego, kind)                 # (N, 8): t x y yaw° vx vy L W, the ingest's world
        out = []
        for (k, t, _b), row in zip(rows, traj):
            A = _mat(row[1], row[2], math.radians(row[3]), H / 2.0)
            B = path.at(t) @ np.linalg.inv(NM.ego_pose(ego, k)) @ A
            out.append((t - clock0, B[:3, 3], rot_to_quat_np(B[:3, :3])[0]))
        obs[tid] = out
        info[tid] = (kind, CATEGORY.get(kind, f'dashcam.{kind}'), True, (L, W, H))
        sig = [float(b.get('sigma') or 0.0) for _k, _t, b in rows]
        meta[tid] = {'source': 'dashcam ingest (ByteTrack, fused monocular ranges, RTS-smoothed: core.nearmiss)',
                     'track': int(tid), 'class_source': tr['class_source'], 'observations': len(rows),
                     'range_sigma_m_median': round(float(np.median(sig)), 3),
                     'size': 'typical for the kind (not measured)'}
    built = build_tracks(obs, info, by_motion=True)
    for tid, t in built.items():
        t.meta = meta[tid]
    kinds = {}
    for t in built.values():
        kinds[t.cls] = kinds.get(t.cls, 0) + 1
    rep = {'tracks': len(built), 'kinds': kinds, 'moving': sum(1 for t in built.values() if t.dynamic),
           'left_out': left, 'ego_heading': ego.get('heading'), 'chain': CHAIN, 'pad_s': pad_s,
           'sizes': {k: list(size_of(k)) for k in sorted(kinds)},
           'heuristic': [{'what': 'sizes', 'how': 'typical for the kind (core.scenario_plan.SIZES, '
                                                 'core.recon.dashcam_tracks.HEIGHTS)'},
                         {'what': 'box centre', 'how': 'half the kind\'s height above z = 0 (a flat road)'}]}
    return built, rep


# ================================================================================================ the posed folder
def write(posed_dir: str, tracks: dict, report: dict) -> str:
    """posed/tracks.json (falcon.recon_tracks/1); returns its name relative to the folder."""
    from ..provenance import write_json
    write_json(os.path.join(posed_dir, FILE), {
        'schema': SCHEMA, 'clock': 'seconds from the first posed frame (the frames\' t)',
        'frame': 'the posed folder\'s world: box x along the length, y left, z up, origin at the box centre',
        'source': {'format': 'dashcam', 'kind': 'ingest tracks'}, 'report': report,
        'tracks': [t.doc() for _k, t in sorted(tracks.items())]})
    return FILE


def read(path: str) -> dict:
    """{id: Track} of a tracks.json."""
    from ..provenance import read_json
    from .actors import Track
    doc = read_json(path)
    if doc.get('schema') != SCHEMA:
        raise TracksError(f'{path}: schema {doc.get("schema")!r}, expected {SCHEMA}')
    out = {}
    for d in doc.get('tracks') or ():
        t = Track.from_doc(d)
        out[t.id] = t
    return out
