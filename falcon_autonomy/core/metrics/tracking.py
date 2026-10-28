"""Multi-object tracking metrics (numpy): CLEAR MOT (MOTA, MOTP, IDSW, FP, FN, Frag, MT/PT/ML) and HOTA (DetA,
AssA, LocA, HOTA over localisation thresholds).

Input. A sequence is a list of frames (or {'frames': [...]}) in time order, a frame {'gt': [{'id', 'bbox': [x, y, w, h]
or 'center': (x, y[, z])}], 'det': [{'id', ...}]} (det = the tracker's output); ids are any hashables, per sequence.
Several sequences are evaluated separately and combined as TrackEval combines them. Similarity between a GT and a
tracker box: 'iou' (box IoU) or 'center' (TrackEval's Euclidean similarity max(0, 1 − d / zero_distance), d the centre
distance in the units given; with threshold 0.5 a pair matches when d ≤ zero_distance / 2).

CLEAR MOT (Bernardin & Stiefelhagen 2008, Evaluating multiple object tracking performance: the CLEAR MOT metrics,
EURASIP JIVP; the reference is TrackEval's metrics/clear.py, Luiten & Hoffhues 2020, whose rules are reproduced):
per frame, pairs with similarity ≥ threshold are matched by an optimal assignment (Hungarian, assignment.py)
maximising 1000 · [the pair continues a match of the previous frame] + similarity, so existing tracks are kept first.
IDSW: a matched GT whose tracker id differs from the one it was last matched to (any number of frames before).
MOTA = 1 − (FN + FP + IDSW) / #GT; MOTP = the mean similarity of the matches; Frag = the number of times a GT track is
resumed after an interruption (TrackEval: fragments − 1 per GT with any match; a frame with no tracker output at all
does not interrupt, as in TrackEval); MT / PT / ML: GTs tracked in > 80 %, ≥ 20 % and < 20 % of their frames.

HOTA (Luiten et al. 2020, HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking, IJCV 129; the reference is
TrackEval's metrics/hota.py, reproduced): for α = 0.05, 0.10, ..., 0.95, per frame an optimal assignment maximising
(global alignment score of the id pair) × similarity, where the alignment score of a GT id and a tracker id is
Σ_t sim'_t / (n_gt + n_tr − Σ_t sim'_t), sim' the similarity normalised as a soft IoU over each frame's rows and
columns; pairs with similarity ≥ α are TPs. DetA = TP / (TP + FN + FP); AssA = the mean over TPs of
TPA / (TPA + FNA + FPA) of their id pair (TPA = the frames where the pair is matched, FNA / FPA = the GT's / tracker's
other frames); LocA = the mean similarity of the TPs; HOTA(α) = √(DetA · AssA). Reported values are means over α,
plus HOTA(0) / LocA(0) at α = 0.05, DetRe, DetPr, AssRe, AssPr, OWTA, and the per-α arrays.

Limits. One class at a time (call per class); no TrackEval preprocessing (MOTChallenge's distractor removal, crowd
regions, min visibility): filter GT before. Equal-score assignments are resolved by assignment.py, which may pick a
different but equally optimal matching than scipy's in TrackEval; MOTA of a sequence without GT is −FP (TrackEval
reports 0 for that sequence alone, but uses the counts when combining).
"""
from __future__ import annotations

import numpy as np

from .assignment import linear_assignment
from .coco import box_iou

ALPHAS = np.arange(0.05, 0.99, 0.05)              # TrackEval's array_labels
EPS = np.finfo('float').eps


def _frames(seq):
    return seq['frames'] if isinstance(seq, dict) else seq


def _is_single(seqs) -> bool:
    if isinstance(seqs, dict):
        return True
    return bool(seqs) and isinstance(seqs[0], dict) and 'frames' not in seqs[0]


def _similarity(gts, trs, similarity, zero_distance):
    if similarity == 'iou':
        return box_iou(np.array([d['bbox'] for d in trs], float).reshape(-1, 4),
                       np.array([g['bbox'] for g in gts], float).reshape(-1, 4)).T
    if similarity == 'center':
        a = np.array([list(g['center']) for g in gts], float)
        b = np.array([list(d['center']) for d in trs], float)
        if not len(a) or not len(b):
            return np.zeros((len(gts), len(trs)))
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        return np.maximum(0.0, 1.0 - d / zero_distance)
    raise ValueError("similarity is 'iou' or 'center'")


def prepare_sequence(seq, similarity: str = 'iou', zero_distance: float = 2.0) -> dict:
    """TrackEval's per-sequence data: contiguous id indices per frame and the GT × tracker similarity matrices."""
    gmap, tmap = {}, {}
    gt_ids, tr_ids, sims = [], [], []
    for fr in _frames(seq):
        gts, trs = list(fr.get('gt') or ()), list(fr.get('det') or ())
        g = np.array([gmap.setdefault(o['id'], len(gmap)) for o in gts], dtype=np.int64)
        t = np.array([tmap.setdefault(o['id'], len(tmap)) for o in trs], dtype=np.int64)
        if len(set(g.tolist())) != len(g) or len(set(t.tolist())) != len(t):
            raise ValueError('an id appears twice in one frame')
        gt_ids.append(g)
        tr_ids.append(t)
        sims.append(_similarity(gts, trs, similarity, zero_distance))
    return {'gt_ids': gt_ids, 'tracker_ids': tr_ids, 'similarity': sims, 'num_gt_ids': len(gmap),
            'num_tracker_ids': len(tmap), 'num_gt_dets': int(sum(len(g) for g in gt_ids)),
            'num_tracker_dets': int(sum(len(t) for t in tr_ids)), 'num_frames': len(gt_ids)}


def _clear_counts(data: dict, threshold: float) -> dict:
    n_gt = data['num_gt_ids']
    res = {'CLR_TP': 0, 'CLR_FN': 0, 'CLR_FP': 0, 'IDSW': 0, 'MOTP_sum': 0.0, 'CLR_Frames': data['num_frames']}
    gt_id_count = np.zeros(n_gt)
    gt_matched_count = np.zeros(n_gt)
    gt_frag_count = np.zeros(n_gt)
    prev_tracker_id = np.full(n_gt, np.nan)                     # last matched tracker id, for IDSW
    prev_timestep_tracker_id = np.full(n_gt, np.nan)            # the previous frame's, for continuing matches
    for g, t, sim in zip(data['gt_ids'], data['tracker_ids'], data['similarity']):
        if len(g) == 0:
            res['CLR_FP'] += len(t)
            continue
        if len(t) == 0:
            res['CLR_FN'] += len(g)
            gt_id_count[g] += 1
            continue
        score = 1000.0 * (t[None, :] == prev_timestep_tracker_id[g[:, None]]) + sim
        score[sim < threshold - EPS] = 0
        rows, cols = linear_assignment(score, maximize=True)
        ok = score[rows, cols] > 0 + EPS
        rows, cols = rows[ok], cols[ok]
        mg, mt = g[rows], t[cols]
        prev = prev_tracker_id[mg]
        res['IDSW'] += int(np.sum(~np.isnan(prev) & (mt != prev)))
        gt_id_count[g] += 1
        gt_matched_count[mg] += 1
        not_prev = np.isnan(prev_timestep_tracker_id)
        prev_tracker_id[mg] = mt
        prev_timestep_tracker_id[:] = np.nan
        prev_timestep_tracker_id[mg] = mt
        gt_frag_count += not_prev & ~np.isnan(prev_timestep_tracker_id)
        res['CLR_TP'] += len(mg)
        res['CLR_FN'] += len(g) - len(mg)
        res['CLR_FP'] += len(t) - len(mg)
        res['MOTP_sum'] += float(sim[rows, cols].sum())
    seen = gt_id_count > 0
    ratio = gt_matched_count[seen] / gt_id_count[seen]
    res['MT'] = int(np.sum(ratio > 0.8))
    res['PT'] = int(np.sum(ratio >= 0.2)) - res['MT']
    res['ML'] = n_gt - res['MT'] - res['PT']
    res['Frag'] = int(np.sum(gt_frag_count[gt_frag_count > 0] - 1))
    res['num_gt_ids'] = n_gt
    return res


def _clear_final(r: dict) -> dict:
    tp, fn, fp, idsw = r['CLR_TP'], r['CLR_FN'], r['CLR_FP'], r['IDSW']
    n_ids = r['num_gt_ids']
    out = {'MOTA': (tp - fp - idsw) / max(1.0, tp + fn), 'MOTP': r['MOTP_sum'] / max(1.0, tp),
           'MODA': (tp - fp) / max(1.0, tp + fn), 'IDSW': idsw, 'FP': fp, 'FN': fn, 'TP': tp, 'Frag': r['Frag'],
           'recall': tp / max(1.0, tp + fn), 'precision': tp / max(1.0, tp + fp),
           'MT': r['MT'], 'PT': r['PT'], 'ML': r['ML'], 'MTR': r['MT'] / n_ids if n_ids else 0.0,
           'MLR': r['ML'] / n_ids if n_ids else 0.0, 'sMOTA': (r['MOTP_sum'] - fp - idsw) / max(1.0, tp + fn),
           'n_gt': tp + fn, 'n_det': tp + fp, 'n_gt_ids': n_ids, 'n_frames': r['CLR_Frames']}
    return out


def clear_mot(sequences, similarity: str = 'iou', threshold: float = 0.5, zero_distance: float = 2.0) -> dict:
    """CLEAR MOT of one sequence or a list of them (counts summed over sequences): {'MOTA', 'MOTP', 'MODA', 'IDSW',
    'FP', 'FN', 'TP', 'Frag', 'recall', 'precision', 'MT', 'PT', 'ML', 'MTR', 'MLR', 'sMOTA', 'n_gt', 'n_det',
    'n_gt_ids', 'n_frames', 'n_sequences'}."""
    seqs = [sequences] if _is_single(sequences) else list(sequences)
    tot = None
    for s in seqs:
        r = _clear_counts(prepare_sequence(s, similarity, zero_distance), threshold)
        tot = r if tot is None else {k: tot[k] + r[k] for k in r}
    if tot is None:
        raise ValueError('no sequences')
    out = _clear_final(tot)
    out['n_sequences'] = len(seqs)
    return out


def _hota_counts(data: dict, alphas=ALPHAS) -> dict:
    A = len(alphas)
    res = {k: np.zeros(A) for k in ('HOTA_TP', 'HOTA_FN', 'HOTA_FP', 'AssA', 'AssRe', 'AssPr', 'LocA')}
    if data['num_tracker_dets'] == 0 or data['num_gt_dets'] == 0:
        res['HOTA_FN'][:] = data['num_gt_dets']
        res['HOTA_FP'][:] = data['num_tracker_dets']
        res['LocA'][:] = 1.0
        return res
    ng, nt = data['num_gt_ids'], data['num_tracker_ids']
    potential = np.zeros((ng, nt))
    gt_count = np.zeros((ng, 1))
    tr_count = np.zeros((1, nt))
    for g, t, sim in zip(data['gt_ids'], data['tracker_ids'], data['similarity']):
        denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
        soft = np.zeros_like(sim)
        ok = denom > EPS
        soft[ok] = sim[ok] / denom[ok]
        potential[g[:, None], t[None, :]] += soft
        gt_count[g] += 1
        tr_count[0, t] += 1
    align = potential / (gt_count + tr_count - potential)
    matches = np.zeros((A, ng, nt))
    for g, t, sim in zip(data['gt_ids'], data['tracker_ids'], data['similarity']):
        if len(g) == 0:
            res['HOTA_FP'] += len(t)
            continue
        if len(t) == 0:
            res['HOTA_FN'] += len(g)
            continue
        score = align[g[:, None], t[None, :]] * sim
        rows, cols = linear_assignment(score, maximize=True)
        s = sim[rows, cols]
        for a, alpha in enumerate(alphas):
            ok = s >= alpha - EPS
            r, c = rows[ok], cols[ok]
            n = len(r)
            res['HOTA_TP'][a] += n
            res['HOTA_FN'][a] += len(g) - n
            res['HOTA_FP'][a] += len(t) - n
            if n:
                res['LocA'][a] += s[ok].sum()
                matches[a, g[r], t[c]] += 1
    for a in range(A):
        m = matches[a]
        tp = max(1.0, res['HOTA_TP'][a])
        res['AssA'][a] = np.sum(m * (m / np.maximum(1, gt_count + tr_count - m))) / tp
        res['AssRe'][a] = np.sum(m * (m / np.maximum(1, gt_count))) / tp
        res['AssPr'][a] = np.sum(m * (m / np.maximum(1, tr_count))) / tp
    res['LocA'] = np.maximum(1e-10, res['LocA']) / np.maximum(1e-10, res['HOTA_TP'])
    return res


def _hota_combine(all_res: list) -> dict:
    tp = np.sum([r['HOTA_TP'] for r in all_res], axis=0)
    out = {'HOTA_TP': tp, 'HOTA_FN': np.sum([r['HOTA_FN'] for r in all_res], axis=0),
           'HOTA_FP': np.sum([r['HOTA_FP'] for r in all_res], axis=0)}
    for k in ('AssRe', 'AssPr', 'AssA'):
        out[k] = np.sum([r[k] * r['HOTA_TP'] for r in all_res], axis=0) / np.maximum(1.0, tp)
    loc = np.sum([r['LocA'] * r['HOTA_TP'] for r in all_res], axis=0)
    out['LocA'] = np.maximum(1e-10, loc) / np.maximum(1e-10, tp)
    return out


def hota(sequences, similarity: str = 'iou', zero_distance: float = 2.0, alphas=ALPHAS) -> dict:
    """HOTA of one sequence or a list of them: {'HOTA', 'DetA', 'AssA', 'LocA', 'DetRe', 'DetPr', 'AssRe', 'AssPr',
    'OWTA', 'HOTA(0)', 'LocA(0)', 'HOTALocA(0)'} (means over α), 'per_alpha': {name: [per α]}, 'alphas', and the
    counts 'TP', 'FN', 'FP' (per α), 'n_gt', 'n_det', 'n_sequences'."""
    seqs = [sequences] if _is_single(sequences) else list(sequences)
    if not seqs:
        raise ValueError('no sequences')
    datas = [prepare_sequence(s, similarity, zero_distance) for s in seqs]
    r = _hota_combine([_hota_counts(d, alphas) for d in datas])
    tp, fn, fp = r['HOTA_TP'], r['HOTA_FN'], r['HOTA_FP']
    per = {'DetRe': tp / np.maximum(1.0, tp + fn), 'DetPr': tp / np.maximum(1.0, tp + fp),
           'DetA': tp / np.maximum(1.0, tp + fn + fp), 'AssA': r['AssA'], 'AssRe': r['AssRe'], 'AssPr': r['AssPr'],
           'LocA': r['LocA']}
    per['HOTA'] = np.sqrt(per['DetA'] * per['AssA'])
    per['OWTA'] = np.sqrt(per['DetRe'] * per['AssA'])
    out = {k: float(np.mean(v)) for k, v in per.items()}
    out['HOTA(0)'] = float(per['HOTA'][0])
    out['LocA(0)'] = float(per['LocA'][0])
    out['HOTALocA(0)'] = out['HOTA(0)'] * out['LocA(0)']
    out['per_alpha'] = {k: [float(x) for x in v] for k, v in per.items()}
    out['alphas'] = [float(a) for a in alphas]
    out['TP'] = [int(x) for x in tp]
    out['FN'] = [int(x) for x in fn]
    out['FP'] = [int(x) for x in fp]
    out['n_gt'] = sum(d['num_gt_dets'] for d in datas)
    out['n_det'] = sum(d['num_tracker_dets'] for d in datas)
    out['n_sequences'] = len(seqs)
    return out


def tracking_evaluate(sequences, similarity: str = 'iou', threshold: float = 0.5, zero_distance: float = 2.0) -> dict:
    """Both families: {'clear': clear_mot(...), 'hota': hota(...)} plus the headline numbers at the top level
    (HOTA, DetA, AssA, LocA, MOTA, MOTP, IDSW)."""
    c = clear_mot(sequences, similarity, threshold, zero_distance)
    h = hota(sequences, similarity, zero_distance)
    out = {k: h[k] for k in ('HOTA', 'DetA', 'AssA', 'LocA')}
    out.update({k: c[k] for k in ('MOTA', 'MOTP', 'IDSW', 'FP', 'FN', 'Frag')})
    out['clear'] = c
    out['hota'] = h
    return out

