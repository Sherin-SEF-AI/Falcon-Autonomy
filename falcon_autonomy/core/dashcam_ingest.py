"""Dashcam ingest: a folder of clips perceived and mined for near-misses, in place, resumably (numpy and OpenCV at
import; the models only when asked for). `falcon dashcam ingest DIR` and `falcon dashcam nearmiss …`.

Everything is written inside the folder, under DIR/falcon-ingest/ (the footage itself is never modified):

    ingest.json                 the state (worker dashcam-ingest, source dashcam: assistant.privacy keeps the whole
                                folder private), the settings and their key, per clip what is done, counts
    clips/<key>/perception.json pass 1 of a clip (key: a hash of its path — never its name, which carries its date)
    clips/<key>/nearmiss.json   pass 2 of a clip (core.nearmiss.mine_clip)
    ranking.json                every event of every clip, best first (core.nearmiss.rank)
    events/<id>/                the top 20: clip.mp4 (stream-copied), event.json, detections.json, posed/, run.json

Pass 1 — perception (perceive), per clip, streaming (no frame is kept): frames at `rate` (10 a second) and `size`
(960 × 540) from core.dashcam.iter_frames (NVDEC when it works); in batches the detector (core.dashcam_detect:
RT-DETRv2, FP16 on CUDA) and the depth network (core.dashcam_depth: Depth Anything V2 Small by default; Metric3D v2
opt-in; or none); per frame the COCO classes mapped and riders folded, ByteTrack (core.bytetrack) at the frame rate,
the road plane of the depth map (RANSAC; independent of the horizon), each tracked box's depth value, and — while the
car drives straight above 4 m/s by its GPS — the focus of expansion. After the clip: the horizon (the foci's median
row), then per box the autorickshaw heuristic, the geometric and network ranges and their fusion, and each track's
kind by majority. A clip without at least two GPS fixes is skipped (no ego motion: nothing to mine); a rear camera's
clip (name ending R) is skipped.

Pass 2 — mining (core.nearmiss.mine_clip), per clip: ego motion, candidate windows, the odometry on them, trajectories,
criticality with the depth-scale bootstrap. Then the folder's ranking and the top events exported.

Resuming. A clip whose perception.json has the same settings key (every setting but the device, batch, models folder,
top, clip cutting, precision and GPU share) and the same clip fingerprint is not processed again, nor its
nearmiss.json; a cancel loses only the clip in progress (files are written whole, atomically). --redo starts over.

The performance profile (core.profiles.cap_perception): under low_vram the detector and the depth network run in
batches of 2, rtdetr-v2-r50 becomes rtdetr-v2-r18 (--keep-detector keeps it), Depth Anything V2 works at 364 px,
both models get 0.35 of the GPU, FP16 on CUDA unless inference_precision is fp32; the profile and what it changed are
recorded in ingest.json's settings and every perception.json. A swapped detector or a capped depth resolution changes
the settings key, so a folder perceived under the other profile is perceived again.

A near-miss as a 4D reconstruction (`nearmiss reconstruct FOLDER EVENT`, reconstruct_spec for the GUI): the event's
window ± 2 s of its clip through `falcon recon from-dashcam --dynamic --tracks EVENT_DIR` — the road users tracked
here become the reconstruction's moving actors (core.recon.dashcam_tracks), so `falcon recon closedloop` can drive a
policy through the real incident. The reconstruction is private like everything here (its dashcam.json).

Privacy: dashcam-derived — the tracks, events, clips and the scenarios made from them stay on this machine;
summary() (what a tool may report) is counts and event ids only.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

from . import dashcam as DC
from . import dashcam_depth as DD
from . import dashcam_detect as DT
from . import nearmiss as NM
from .provenance import read_json, sha256_json, utc_now, write_json

SCHEMA = 'falcon.dashcam_ingest/1'
PERCEPTION_SCHEMA = 'falcon.dashcam_perception/1'
INGEST_DIR = 'falcon-ingest'
DEFAULTS = {'detector': 'rtdetr-v2-r18', 'depth': 'da2-small', 'rate': 10.0, 'size': [960, 540], 'crop_bottom': 0.0,
            'batch': 8, 'device': 'auto', 'camera_height_m': DD.DEFAULT_HEIGHT_M, 'hfov_deg': DD.DEFAULT_HFOV_DEG,
            'focal_px': None, 'horizon_v': None, 'top': NM.TOP, 'models_dir': '', 'vo': True, 'cut_clips': True,
            'accept_unstated_weights_terms': False, 'include_rear': False,
            # the profile's (core.profiles.cap_perception): precision on CUDA, the GPU share, Depth Anything V2's
            # working resolution (None: its own 518), keeping r50 under low_vram, and the record of what was used
            'precision': 'fp16', 'gpu_memory_fraction': 0.5, 'depth_input_px': None, 'keep_detector': False,
            'profile': None}
VOLATILE = ('device', 'batch', 'models_dir', 'top', 'cut_clips', 'precision', 'gpu_memory_fraction',
            'keep_detector', 'profile')
STRAIGHT_DEG_S = 2.0
STRAIGHT_MIN_MPS = 4.0


class IngestError(Exception):
    pass


def out_dir(folder: str) -> str:
    return os.path.join(folder, INGEST_DIR)


def clip_key(path: str) -> str:
    return hashlib.sha1(os.path.realpath(path).encode()).hexdigest()[:12]


def settings_key(s: dict) -> str:
    """What the perception depends on (every setting but VOLATILE); depth_input_px only when set — a capped depth
    resolution perceives again, the model's own one keeps the key of ingests made before it existed."""
    return sha256_json({k: v for k, v in s.items() if k not in VOLATILE and not (k == 'depth_input_px' and v is None)})


def normalise(settings: dict | None) -> dict:
    s = dict(DEFAULTS, **{k: v for k, v in (settings or {}).items() if v is not None or k in ('focal_px',
                                                                                              'horizon_v')})
    s['size'] = [int(v) for v in s['size']]
    if len(s['size']) != 2 or min(s['size']) < 64:
        raise IngestError('size is [width, height], each at least 64 px')
    if not 0.5 <= float(s['rate']) <= 30:
        raise IngestError('rate is 0.5–30 frames a second')
    if not 0 <= float(s['crop_bottom']) < 0.5:
        raise IngestError('crop_bottom is a share of the height below 0.5')
    if not 0.3 <= float(s['camera_height_m']) <= 4.0:
        raise IngestError('camera_height_m must lie in 0.3–4 m')
    return s


def camera(s: dict) -> DD.Camera:
    W, H = s['size']
    cy = H / (2.0 * (1.0 - float(s['crop_bottom'])))       # the optical centre of the uncut frame, scaled
    return DD.Camera(W, H, f=s['focal_px'], hfov_deg=float(s['hfov_deg']), height_m=float(s['camera_height_m']),
                     horizon_v=s['horizon_v'], cy=cy)


def _course_rate(trk: dict, t: float) -> tuple:
    """(speed m/s, |course rate| °/s) of the GPS around video time t, or (None, None)."""
    a, b = DC.position_at(trk, t - 0.5), DC.position_at(trk, t + 0.5)
    if not a or not b or a.get('course_deg') is None or b.get('course_deg') is None:
        return None, None
    sp = ((a.get('speed_kmh') or 0.0) + (b.get('speed_kmh') or 0.0)) / 2.0 / 3.6
    return sp, abs(((b['course_deg'] - a['course_deg'] + 180.0) % 360.0) - 180.0)


# ================================================================================================ pass 1
def perceive(clip: str, s: dict, detector, depth=None, trk: dict | None = None, emitter=None, cancel=None,
             frac=(0.0, 1.0)) -> dict:
    """Pass 1 of one clip (module doc). detector(frames [(n, rgb)]) → per frame raw boxes; depth(frames) → maps
    (its .kind 'relative' or 'metric'), or None."""
    import cv2
    from .bytetrack import ByteTracker
    s = normalise(s)
    trk = trk or DC.track(clip)
    cam = camera(s)
    tracker = ByteTracker(frame_rate=float(s['rate']))
    info = {}
    recs, foes = [], []
    prev = None
    v_road = cam.cy + 0.05 * cam.height                  # the road plane's rows before the horizon is known
    kind = getattr(depth, 'kind', None)
    dur = (DC.probe(clip).get('duration_s') or 0.0)
    t_start = time.perf_counter()
    counts = {'frames': 0, 'detections': 0, 'riders_folded': 0, 'tracked_boxes': 0, 'plane_fits': 0}

    def run(batch):
        nonlocal prev
        raws = detector([(n, rgb) for n, _t, rgb in batch])
        maps = depth([(n, rgb) for n, _t, rgb in batch]) if depth is not None else [None] * len(batch)
        for (n, t, rgb), raw, dm in zip(batch, raws, maps):
            dets, folded = DT.postprocess(raw, None)
            counts['frames'] += 1
            counts['detections'] += len(dets)
            counts['riders_folded'] += folded
            out = tracker.update(DT.boxes_array(dets), [d['score'] for d in dets], [d['kind'] for d in dets])
            boxes = [d['box'] for d in dets]
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if prev is not None:
                sp, cr = _course_rate(trk, t)
                if sp is not None and sp >= STRAIGHT_MIN_MPS and cr <= STRAIGHT_DEG_S:     # °/s over 1 s
                    f = DD.foe(prev[1], gray, boxes)
                    if f:
                        foes.append(dict(f, t=round(t, 3)))
            prev = (t, gray)
            plane = None
            if dm is not None:
                plane = DD.fit_plane(np.asarray(dm, np.float32), cam, boxes, v_min=v_road, metric=kind == 'metric',
                                     seed=int(n))
                counts['plane_fits'] += plane is not None
            brs = []
            for r in out:
                d = dets[r['det']]
                brs.append({'track': r['track'], 'box': [round(v, 2) for v in d['box']], 'score': round(d['score'], 4),
                            'coco': d['coco'], 'kind': d['kind'], 'class_source': d['class_source'],
                            'value': None if dm is None else DD.box_value(np.asarray(dm), d['box'])})
            counts['tracked_boxes'] += len(brs)
            recs.append({'n': int(n), 't': round(float(t), 4), 'detections': len(dets), 'riders': folded,
                         'plane': plane, 'boxes': brs})
        if emitter is not None and dur:
            f = min(1.0, (batch[-1][1] + 1e-6) / dur)
            emitter.progress(frac[0] + (frac[1] - frac[0]) * f, f'perceiving {clip_key(clip)}: {f:.0%}')

    batch = []
    for n, t, rgb in DC.iter_frames(clip, float(s['rate']), tuple(s['size']), info=info,
                                    crop_bottom=float(s['crop_bottom'])):
        if cancel is not None and cancel():
            raise KeyboardInterrupt
        batch.append((n, t, rgb))
        if len(batch) >= int(s['batch']):
            run(batch)
            batch = []
    if batch:
        run(batch)
    horizon = ({'horizon_v': cam.horizon_v, 'pairs': 0, 'given': True} if s['horizon_v'] is not None else
               DD.estimate_horizon(foes, cam))
    # ranges, now that the horizon is known
    votes = {}
    for rec in recs:
        fit = DD.scale_plane(rec.pop('plane'), cam)
        rec['fit'] = None if fit is None else {k: (round(v, 6) if isinstance(v, float) else v)
                                               for k, v in fit.items()}
        for b in rec['boxes']:
            b.update({k: v for k, v in DT.autorickshaw({'box': b['box'], 'kind': b['kind'], 'coco': b['coco'],
                                                       'class_source': b['class_source'], 'score': b['score']},
                                                      cam).items() if k in ('kind', 'class_source', 'heuristic',
                                                                            'size_m')})
            geo = DD.geo_point(cam, b['box'])
            net = DD.net_point(b.pop('value'), kind or 'relative', fit, cam, b['box'])
            fz = DD.fuse(geo, net)
            if fz is not None:
                b.update(forward=round(fz['forward'], 3), left=round(fz['left'], 3), sigma=round(fz['sigma'], 3),
                         source=fz['source'], disagree=fz['disagree'], geo=fz['geo'], net=fz['net'])
            v = votes.setdefault(b['track'], {})
            v[(b['kind'], b['class_source'])] = v.get((b['kind'], b['class_source']), 0) + 1
    major = {tid: max(v, key=v.get) for tid, v in votes.items()}
    for rec in recs:
        for b in rec['boxes']:
            b['kind'], b['class_source'] = major[b['track']]
    return {'schema': PERCEPTION_SCHEMA, 'clip_key': clip_key(clip), 'fingerprint': DC.fingerprint(clip),
            'settings_key': settings_key(s), 'decoder': info, 'camera': cam.doc(), 'horizon': horizon,
            'foe_pairs': len(foes), 'detector': getattr(detector, 'info', {}),
            'depth': getattr(depth, 'info', {'model': None}) if depth is not None else {'model': None},
            'tracker': tracker.params(), 'counts': counts, 'gps': trk, 'frames': recs,
            'batch': int(s['batch']), 'profile': s.get('profile'),
            'seconds': round(time.perf_counter() - t_start, 2), 'created': utc_now()}


def camera_of(perc: dict) -> DD.Camera:
    c = perc['camera']
    cam = DD.Camera(c['size'][0], c['size'][1], f=c['f_px'], height_m=c['height_m'], horizon_v=c['horizon_v'],
                    cy=c['cy'])
    cam.horizon_source = c['horizon_source']
    return cam


# ================================================================================================ the folder
def _clips(folder: str, s: dict) -> tuple:
    od = os.path.realpath(out_dir(folder))
    clips, rear = [], 0
    for c in DC.list_clips(folder):
        if os.path.realpath(c['path']).startswith(od + os.sep):
            continue
        if c.get('camera') == 'R' and not s['include_rear']:
            rear += 1
            continue
        clips.append(c['path'])
    return clips, rear


def _state(folder: str) -> dict:
    p = os.path.join(out_dir(folder), 'ingest.json')
    return read_json(p) if os.path.isfile(p) else {}


def _decoder(clip: str, s: dict):
    def decode(t0, t1):
        import cv2
        out = []
        for n, t, rgb in DC.iter_frames(clip, float(s['rate']), tuple(s['size']), start=t0,
                                        duration=t1 - t0 + 1.5 / float(s['rate']), crop_bottom=float(s['crop_bottom'])):
            out.append((n, round(t, 3), cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)))
        return out
    return decode


def ingest(folder: str, settings: dict | None = None, detector=None, depth=None, emitter=None, cancel=None,
           redo: bool = False, app_settings=None) -> dict:
    """Both passes over a folder's clips, then the ranking and the top events (module doc). detector / depth:
    objects (tests) or None to build them from the settings. The performance profile in force (app_settings: a
    core.settings.Settings, default the user's) caps the perception (core.profiles.cap_perception); what it used is
    recorded in ingest.json's settings ('profile') and every perception.json. Returns summary()."""
    from .profiles import cap_perception
    asked = dict(settings or {})
    s = normalise(cap_perception(asked, app_settings, keep_detector=bool(asked.get('keep_detector'))))
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise IngestError(f'{folder} is not a folder')
    od = out_dir(folder)
    os.makedirs(os.path.join(od, 'clips'), exist_ok=True)
    key = settings_key(s)
    st = {} if redo else _state(folder)
    if st.get('settings_key') not in (None, key):
        st = {}                                        # other settings: a fresh state (the files are rewritten)
    st.update(schema=SCHEMA, worker='dashcam-ingest', source={'format': 'dashcam', 'kind': 'ingest'},
              settings={k: v for k, v in s.items() if k != 'models_dir'}, settings_key=key, status='running',
              updated=utc_now(), thresholds=NM.THRESHOLDS)
    st.setdefault('created', utc_now())
    st.setdefault('clips', {})
    write_json(os.path.join(od, 'ingest.json'), st)
    clips, rear = _clips(folder, s)
    counts = {'clips': len(clips), 'rear_skipped': rear, 'no_gps': 0, 'perceived': 0, 'reused': 0, 'mined': 0,
              'failed': 0}
    try:
        for i, clip in enumerate(clips):
            ck = clip_key(clip)
            cd = os.path.join(od, 'clips', ck)
            pp, mp = os.path.join(cd, 'perception.json'), os.path.join(cd, 'nearmiss.json')
            rec = st['clips'].setdefault(ck, {})
            fp = DC.fingerprint(clip)
            try:
                old = read_json(pp) if os.path.isfile(pp) and not redo else None
                if old and old.get('settings_key') == key and old.get('fingerprint', {}).get('sha256_head') == \
                        fp['sha256_head'] and old.get('fingerprint', {}).get('size') == fp['size']:
                    perc = old
                    counts['reused'] += 1
                else:
                    trk = DC.track(clip)
                    if trk['n_fixes'] < 2:
                        counts['no_gps'] += 1
                        rec.update(perceived=False, mined=False, why='fewer than two GPS fixes')
                        continue
                    fp16 = s['precision'] != 'fp32'
                    if detector is None:
                        detector = DT.make_detector(s['detector'], s['models_dir'], s['device'], fp16,
                                                    float(s['gpu_memory_fraction']))
                    if depth is None and s['depth'] not in ('none', '', None):
                        depth = DD.make_depth(s['depth'], camera(s).f, s['models_dir'], s['device'],
                                              bool(s['accept_unstated_weights_terms']), fp16, s['depth_input_px'],
                                              float(s['gpu_memory_fraction']),
                                              limit_onnx=(s.get('profile') or {}).get('name') == 'low_vram')
                    perc = perceive(clip, s, detector, depth, trk, emitter, cancel,
                                    (i / len(clips) * 0.9, (i + 0.8) / len(clips) * 0.9))
                    os.makedirs(cd, exist_ok=True)
                    write_json(pp, perc)
                    counts['perceived'] += 1
                    if os.path.isfile(mp):
                        os.remove(mp)
                rec.update(perceived=True, frames=len(perc['frames']), why=None)
                if os.path.isfile(mp) and read_json(mp).get('perception_created') == perc['created']:
                    rec['mined'] = True
                    continue
                cam = camera_of(perc)
                mined = NM.mine_clip(perc, perc['gps'], clip, cam, _decoder(clip, s) if s['vo'] else None, emitter)
                mined['perception_created'] = perc['created']
                mined['clip_key'] = ck
                write_json(mp, mined)
                counts['mined'] += 1
                rec.update(mined=True, events=len(mined['events']))
            except (DC.DashcamError, DT.DetectError, DD.DepthError, NM.NearMissError, OSError) as e:
                counts['failed'] += 1
                rec.update(error=f'{type(e).__name__}: {e}')
                if isinstance(e, (DT.DetectError, DD.DepthError)):
                    raise IngestError(str(e)) from None
            finally:
                st['updated'] = utc_now()
                write_json(os.path.join(od, 'ingest.json'), st)
            if emitter is not None:
                emitter.progress((i + 1) / len(clips) * 0.9, f'{i + 1} of {len(clips)} clips')
    except KeyboardInterrupt:
        st.update(status='cancelled', counts=counts)
        write_json(os.path.join(od, 'ingest.json'), st)
        raise
    rk = rerank(folder, s, emitter)
    st.update(status='ok', counts=counts, events=rk['events'], exported=rk['exported'], finished=utc_now())
    write_json(os.path.join(od, 'ingest.json'), st)
    return summary(folder)


def rerank(folder: str, s: dict, emitter=None) -> dict:
    """ranking.json from every clip's nearmiss.json, and the top events exported (stale event folders removed)."""
    import shutil
    od = out_dir(folder)
    events, by_clip = [], {}
    clips, _ = _clips(folder, s)
    path_of = {clip_key(c): c for c in clips}
    for ck in sorted(os.listdir(os.path.join(od, 'clips'))):
        mp = os.path.join(od, 'clips', ck, 'nearmiss.json')
        if not os.path.isfile(mp) or ck not in path_of:
            continue
        m = read_json(mp)
        by_clip[ck] = m
        for e in m['events']:
            events.append(dict({k: v for k, v in e.items() if k != 'trajectory'}, clip_key=ck))
    ranked = NM.rank(events)
    write_json(os.path.join(od, 'ranking.json'), {'schema': 'falcon.nearmiss_ranking/1', 'created': utc_now(),
                                                  'events': ranked, 'thresholds': NM.THRESHOLDS})
    ev_dir = os.path.join(od, 'events')
    os.makedirs(ev_dir, exist_ok=True)
    top = ranked[:int(s['top'])]
    keep = {e['id'] for e in top}
    for d in os.listdir(ev_dir):
        if d not in keep:
            shutil.rmtree(os.path.join(ev_dir, d), ignore_errors=True)
    exported = 0
    for e in top:
        d = os.path.join(ev_dir, e['id'])
        ej = os.path.join(d, 'event.json')
        if os.path.isfile(ej) and read_json(ej).get('score') == e['score']:
            ev = read_json(ej)
            ev['rank'] = e['rank']
            write_json(ej, ev)
            continue
        ck = e['clip_key']
        perc = read_json(os.path.join(od, 'clips', ck, 'perception.json'))
        full = next(x for x in by_clip[ck]['events'] if x['id'] == e['id'])
        NM.export_event(dict(full, rank=e['rank']), by_clip[ck], perc, camera_of(perc), path_of[ck], d,
                        cut=bool(s['cut_clips']))
        exported += 1
    return {'events': len(ranked), 'exported': exported}


def summary(folder: str) -> dict:
    """What may be said about an ingest outside the machine (a tool, a log): counts and event ids only — no clip
    names, places, times of day, positions or images."""
    od = out_dir(folder)
    st = _state(folder)
    rk = os.path.join(od, 'ranking.json')
    ranked = read_json(rk)['events'] if os.path.isfile(rk) else []
    kinds = {}
    for e in ranked:
        kinds[e['kind']] = kinds.get(e['kind'], 0) + 1
    return {'status': st.get('status'), 'counts': st.get('counts') or {}, 'events': len(ranked), 'kinds': kinds,
            'top': [{'rank': e['rank'], 'id': e['id'], 'kind': e['kind'], 'score_median': round(e['score_median'], 3)}
                    for e in ranked[:int((st.get('settings') or {}).get('top') or NM.TOP)]]}


def ranking(folder: str) -> list:
    p = os.path.join(out_dir(folder), 'ranking.json')
    return read_json(p)['events'] if os.path.isfile(p) else []


def event_dir(folder: str, event: str) -> str:
    """An exported event's folder by id or rank."""
    ev = os.path.join(out_dir(folder), 'events')
    if event.isdigit():
        hit = next((e for e in ranking(folder) if e['rank'] == int(event)), None)
        if hit is None:
            raise IngestError(f'no event ranked {event}')
        event = hit['id']
    d = os.path.join(ev, event)
    if not os.path.isfile(os.path.join(d, 'event.json')):
        raise IngestError(f'no exported event {event!r} (only the top events are exported)')
    return d


RECON_PAD_S = 2.0               # a near-miss reconstruction's segment: the event's window and this much either side


def event_segment(folder: str, event: str, pad_s: float = RECON_PAD_S) -> dict:
    """The clip and segment a near-miss event is reconstructed from: {'event' (its ranking row), 'event_dir', 'clip',
    'start', 'seconds'} — the event's window with pad_s either side, inside the clip."""
    folder = os.path.abspath(os.path.expanduser(folder))
    d = event_dir(folder, event)
    eid = os.path.basename(d)
    ev = next((e for e in ranking(folder) if e['id'] == eid), None)
    if ev is None:
        raise IngestError(f'event {eid} is not in the ranking (ingest the folder again)')
    st = _state(folder)
    clips, _ = _clips(folder, normalise(st.get('settings') or {}))
    clip = next((c for c in clips if clip_key(c) == ev['clip_key']), None)
    if clip is None:
        raise IngestError(f'the clip of event {eid} is no longer in the folder')
    dur = DC.probe(clip).get('duration_s') or float(ev['window_s'][1]) + pad_s
    start = max(0.0, float(ev['window_s'][0]) - pad_s)
    end = min(float(dur), float(ev['window_s'][1]) + pad_s)
    return {'event': ev, 'event_dir': d, 'clip': clip, 'start': round(start, 3), 'seconds': round(end - start, 3)}


def reconstruct_argv(folder: str, event: str, out: str, pad_s: float = RECON_PAD_S, extra=()) -> list:
    """`falcon recon from-dashcam` of a near-miss event's segment with its tracked road users (--dynamic --tracks:
    core.recon.dashcam_tracks), as recon's argv."""
    seg = event_segment(folder, event, pad_s)
    return ['from-dashcam', '--clip', seg['clip'], '--start', f'{seg["start"]:g}', '--seconds',
            f'{seg["seconds"]:g}', '--out', out, '--dynamic', '--tracks', seg['event_dir']] + list(extra)


def reconstruct_spec(folder: str, event: str, out: str, device: str = 'auto', iterations: int | None = None,
                     width: int | None = None, title: str = ''):
    """`falcon dashcam nearmiss reconstruct FOLDER EVENT --out OUT …` as a queued job: a GPU job unless on the CPU
    (the queue holds the GPU lock for it), local only (dashcam-derived)."""
    from .jobs import JobSpec
    argv = [sys.executable, '-m', 'falcon_autonomy', 'dashcam', 'nearmiss', 'reconstruct', folder, str(event),
            '--out', out, '--device', device]
    if iterations:
        argv += ['--iterations', str(int(iterations))]
    if width:
        argv += ['--width', str(int(width))]
    return JobSpec('recon', title or f'Reconstruct near-miss {event} with its road users', argv,
                   env=dict(os.environ, FALCON_GPU_LOCK_HELD='1'), needs_gpu=device != 'cpu', memory_gb=6.0,
                   meta={'source': {'format': 'dashcam'}, 'folder': folder, 'event': str(event), 'out': out})


def job_spec(folder: str, argv_extra=(), device: str = 'auto', title: str = ''):
    """`falcon dashcam ingest FOLDER …` as a queued job: a GPU job unless on the CPU, resumable, local only."""
    from .jobs import JobSpec
    argv = [sys.executable, '-m', 'falcon_autonomy', 'dashcam', 'ingest', folder, '--device', device] + \
        list(argv_extra)
    return JobSpec('dashcam-ingest', title or 'Near-miss mining', argv, needs_gpu=device != 'cpu', memory_gb=3.0,
                   resumable=True, meta={'source': {'format': 'dashcam'}, 'folder': folder})


# ================================================================================================ the commands
def _size(text: str) -> list:
    try:
        w, h = text.lower().split('x')
        return [int(w), int(h)]
    except ValueError:
        raise IngestError(f'a size is WIDTHxHEIGHT (not {text!r})') from None


def main_ingest(argv) -> int:
    """`falcon dashcam ingest DIR …` and `falcon dashcam nearmiss list|scenario|reconstruct|variations|models|fetch
    …` (module doc; docs/nearmiss.md). argv starts with 'ingest' or 'nearmiss'."""
    import argparse
    from . import perception_models as PM
    from . import protocol as P
    from .profiles import PERCEPTION_CAPS
    argv = list(argv)
    if argv[:1] == ['dashcam']:
        argv = argv[1:]
    what = argv[0] if argv and argv[0] in ('ingest', 'nearmiss') else 'ingest'
    rest = argv[1:] if argv and argv[0] in ('ingest', 'nearmiss') else argv
    if what == 'ingest':
        ap = argparse.ArgumentParser(prog='falcon dashcam ingest', description='perceive a folder of dashcam clips '
                                     'and mine it for near-misses, in place and resumably (docs/nearmiss.md)')
        ap.add_argument('folder')
        ap.add_argument('--detector', default=DEFAULTS['detector'],
                        help='rtdetr-v2-r18 (default), rtdetr-v2-r50 or file:PATH (detections from a file)')
        ap.add_argument('--depth', default=DEFAULTS['depth'], help='da2-small (default), metric3d-small (opt-in) or '
                                                                   'none (flat-ground ranges only)')
        ap.add_argument('--accept-unstated-weights-terms', action='store_true',
                        help='use metric3d-small, whose weights\' terms are not stated (recorded)')
        ap.add_argument('--rate', type=float, default=DEFAULTS['rate'], help='frames a second (default %(default)g)')
        ap.add_argument('--size', default='960x540', help='processing size (default %(default)s)')
        ap.add_argument('--crop-bottom', type=float, default=0.0, help='share of the height cut off the bottom (the '
                                                                       'burned-in text line: 0.06 on the verified '
                                                                       'camera)')
        ap.add_argument('--batch', type=int, default=DEFAULTS['batch'],
                        help='frames a batch (default %(default)d; the low_vram profile caps it at '
                             f'{PERCEPTION_CAPS["batch"]})')
        ap.add_argument('--keep-detector', action='store_true',
                        help='under the low_vram profile, keep rtdetr-v2-r50 when asked for (the profile swaps in '
                             'rtdetr-v2-r18)')
        ap.add_argument('--device', default='auto', help='auto, cuda or cpu')
        ap.add_argument('--camera-height', type=float, default=DEFAULTS['camera_height_m'],
                        help='the camera above the road, metres (default %(default)g)')
        ap.add_argument('--hfov', type=float, default=DEFAULTS['hfov_deg'], help='horizontal field of view, degrees '
                                                                               '(default %(default)g)')
        ap.add_argument('--focal-px', type=float, default=None, help='the focal length in processed pixels (instead '
                                                                     'of --hfov)')
        ap.add_argument('--horizon-row', type=float, default=None, help='the horizon\'s row in processed pixels '
                                                                        '(default: estimated per clip)')
        ap.add_argument('--top', type=int, default=NM.TOP, help='events exported (default %(default)d)')
        ap.add_argument('--models', default='', help=f'model folder (default {PM.DEFAULT_DIR})')
        ap.add_argument('--include-rear', action='store_true', help='also the rear camera\'s clips')
        ap.add_argument('--no-vo', action='store_true', help='GPS heading only on the candidate windows')
        ap.add_argument('--no-clips', action='store_true', help='do not cut the events\' video')
        ap.add_argument('--redo', action='store_true', help='start over (ignore what was done)')
    else:
        ap = argparse.ArgumentParser(prog='falcon dashcam nearmiss', description='the near-misses an ingest found '
                                     '(docs/nearmiss.md)')
        sub = ap.add_subparsers(dest='cmd', required=True)
        ls = sub.add_parser('list', help='the ranked near-misses of an ingested folder')
        ls.add_argument('folder')
        ls.add_argument('--top', type=int, default=NM.TOP)
        sc = sub.add_parser('scenario', help='an event as a Falcon scenario, variations and OpenSCENARIO')
        sc.add_argument('folder')
        sc.add_argument('event', help='its id or rank')
        sc.add_argument('--project', required=True)
        sc.add_argument('--world', required=True, help='the imported world whose lanes the event is on')
        sc.add_argument('--name', default='')
        rc = sub.add_parser('reconstruct', help='reconstruct an event\'s segment with its tracked road users as '
                                                'moving actors (falcon recon from-dashcam --dynamic), on this machine')
        rc.add_argument('folder')
        rc.add_argument('event', help='its id or rank')
        rc.add_argument('--out', '-o', help='the reconstruction folder (default: '
                                            '~/FalconWorkspace/recon/nearmiss-<event id>)')
        rc.add_argument('--pad', type=float, default=RECON_PAD_S,
                        help='seconds of the clip either side of the event\'s window (default %(default)g)')
        rc.add_argument('--config', help='a JSON file of from-dashcam settings (core.recon.fromdashcam.DEFAULTS)')
        rc.add_argument('--intrinsics', help='as for falcon recon from-dashcam')
        for k, typ in (('iterations', int), ('width', int), ('seed', int)):
            rc.add_argument(f'--{k}', type=typ)
        rc.add_argument('--device')
        rc.add_argument('--no-train', action='store_true', help='pose the segment and place its road users only')
        va = sub.add_parser('variations', help='the rare-case search over a scenario\'s variations, on this machine')
        va.add_argument('project')
        va.add_argument('name')
        va.add_argument('--budget', type=int, default=40)
        va.add_argument('--method', default='bayes')
        va.add_argument('--fake-carla', default='')
        mo = sub.add_parser('models', help='the perception models: candidates (licence, state) and refusals')
        mo.add_argument('--models', default='')
        fe = sub.add_parser('fetch', help='download a model, after its licence is shown and accepted')
        fe.add_argument('name')
        fe.add_argument('--accept-licence', action='store_true', help='accept the licence shown')
        fe.add_argument('--accept-unstated-weights-terms', action='store_true',
                        help='for metric3d-small: accept weights whose terms are not stated (recorded)')
        fe.add_argument('--models', default='')
    try:
        a = ap.parse_args(rest)
    except SystemExit as ex:
        return P.OK if ex.code == 0 else P.FAILED
    try:
        if what == 'ingest':
            em = P.Emitter()
            em.hello('dashcam-ingest')
            from .training import CancelFlag
            cancel = CancelFlag().install()
            s = {'detector': a.detector, 'depth': a.depth, 'rate': a.rate, 'size': _size(a.size),
                 'crop_bottom': a.crop_bottom, 'batch': a.batch, 'device': a.device,
                 'camera_height_m': a.camera_height, 'hfov_deg': a.hfov, 'focal_px': a.focal_px,
                 'horizon_v': a.horizon_row, 'top': a.top, 'models_dir': a.models, 'vo': not a.no_vo,
                 'cut_clips': not a.no_clips, 'accept_unstated_weights_terms': a.accept_unstated_weights_terms,
                 'include_rear': a.include_rear, 'keep_detector': a.keep_detector}
            try:
                res = ingest(a.folder, s, emitter=em, cancel=cancel, redo=a.redo)
            except KeyboardInterrupt:
                em.result('cancelled')
                return P.CANCELLED
            em.result('ok', summary=res)
            print(json.dumps(res, indent=1))
            return P.OK
        if a.cmd == 'list':
            rows = ranking(a.folder)[:a.top]
            for e in rows:
                ttc = '—' if e['min_ttc_s'] is None else f'{e["min_ttc_s"]:.2f} s'
                pet = '—' if e['pet_s'] is None else f'{e["pet_s"]:.2f} s'
                print(f'{e["rank"]:3d}  {e["id"]}  {e["kind"]:13s} score {e["score_median"]:.3f} '
                      f'[{e["score_ci"][0]:.3f}, {e["score_ci"][1]:.3f}]  TTC {ttc}  PET {pet}  '
                      f'DRAC {e["max_drac_mps2"]:.1f} m/s²  ego braking {e["ego_peak_decel_mps2"]:.1f} m/s²'
                      + ('  (class: heuristic)' if e.get('class_source') == 'heuristic' else ''))
            if not rows:
                print('no near-misses (has the folder been ingested? falcon dashcam ingest DIR)')
            return P.OK
        if a.cmd == 'scenario':
            from .project import Project
            proj = Project.load(a.project)
            res = NM.make_scenario(event_dir(a.folder, a.event), proj, a.world, a.name)
            print(json.dumps(res, indent=1))
            return P.OK
        if a.cmd == 'reconstruct':
            from .recon.commands import main_recon
            eid = os.path.basename(event_dir(a.folder, a.event))
            out = a.out or os.path.expanduser(f'~/FalconWorkspace/recon/nearmiss-{eid}')
            extra = []
            for flag, v in (('--config', a.config), ('--intrinsics', a.intrinsics), ('--iterations', a.iterations),
                            ('--width', a.width), ('--seed', a.seed), ('--device', a.device)):
                if v is not None:
                    extra += [flag, str(v)]
            if a.no_train:
                extra.append('--no-train')
            return main_recon(reconstruct_argv(a.folder, a.event, out, a.pad, extra))
        if a.cmd == 'variations':
            from .incident import search_argv
            from .project import Project
            from ..sim.studio import main_study
            proj = Project.load(a.project)
            return main_study(search_argv(proj.path, proj.workspace, a.name, a.budget, a.method,
                                          fake_carla=a.fake_carla))
        if a.cmd == 'models':
            for r in PM.list_models(a.models):
                size = '' if r['bytes'] is None else f'{r["bytes"] / 2 ** 20:7.1f} MiB'
                state = f'REFUSED: {r["refused"]}' if r['refused'] else r['state'] + (
                    ' (opt-in: weights\' terms unstated)' if r['opt_in'] else '')
                print(f'{r["name"]:15s} {r["licence"]:22s} {size:12s} {state}')
            return P.OK
        print(PM.describe(a.name))
        folder = PM.fetch(a.name, a.models, a.accept_licence, a.accept_unstated_weights_terms)
        print(f'downloaded and pinned: {folder}')
        return P.OK
    except (IngestError, NM.NearMissError, DC.DashcamError, PM.PerceptionModelError, OSError, ValueError) as ex:
        print(f'failed: {ex}', flush=True)
        return P.FAILED
    except Exception as ex:          # noqa: BLE001 (IncidentError and the like: one sentence, exit 2)
        if type(ex).__name__ in ('IncidentError', 'ProjectError'):
            print(f'failed: {ex}', flush=True)
            return P.FAILED
        raise

