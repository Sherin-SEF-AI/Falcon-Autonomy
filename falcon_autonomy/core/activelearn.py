"""Active learning for 2D box detectors: choose which recorded frames to add to training by what the model does not
yet know, in rounds with a fixed budget, and measure the data efficiency honestly against random choice (numpy at
import; torch inside the functions, through core.training).

In Falcon the labels are free: every frame of a drive comes labelled from the simulation. "Labelling" a frame here
means ADDING IT TO TRAINING. The point is data efficiency — the same AP from fewer frames (shorter trainings, smaller
datasets, fewer drives to record) — not saving annotation cost.

Acquisition (score_frames, select). Each round the model trained so far scores every unlabelled frame of the pool,
and the budget's B frames are chosen:

    random    a seeded draw (the baseline every other strategy is compared with)
    entropy   Faster R-CNN: the normalised entropy H(q) / ln(K + 1) of each proposal's softmax over the K classes and
              the background; FCOS / RetinaNet (per-class sigmoids): the binary entropy of the highest class score,
              / ln 2. Over the proposals (anchors) whose best foreground score is at least the detector's own score
              threshold (0.05); per frame their maximum (AGGREGATES: max, mean or sum); 0 when none
    margin    1 − (p₁ − p₂), the two highest class probabilities of a proposal (the background included for Faster
              R-CNN), over the same proposals, aggregated the same way
    det_mean  the mean over the frame's final detections scoring at least DET_FLOOR (0.3, heuristic) of 1 − score
              (score = the detection's class probability, the maximum over classes); 0 with no such detection
    det_max   the same, the maximum
    kcenter   diversity without a learned loss (Sener & Savarese 2018, k-centre greedy): each frame's feature is the
              backbone's feature pyramid, each level averaged over the image and the levels concatenated, L2
              normalised; repeatedly the frame farthest (Euclidean) from every labelled or chosen frame is chosen
    hybrid    uncertainty, then diversity: the HYBRID_FACTOR × B frames of highest entropy, then k-centre greedy
              among them

Not implemented, and why: MC dropout — torchvision's detection heads have no dropout layers (Faster R-CNN's
TwoMLPHead is two linear layers; FCOS and RetinaNet heads are convolutions and GroupNorm), so it would need another
model trained with dropout; ensembles — K trainings per round, K times the cost of the experiment.

The experiment (run_active; `falcon experiment active`). A pool of frames (falcon drives, dataset versions), a test
set (real or held-out drives) that shares no image and no drive/sequence with the pool or the validation set (refused
otherwise), an initial set of `initial` frames, `rounds` rounds of `budget` frames, `seeds` seeds. Per seed the initial
set is the first `initial` frames of a seeded permutation of the pool, the same for every strategy, and its model
(round 0) is trained once and shared. Every round's model is trained from the same initialisation with the same
settings on all the frames chosen so far (no warm start: the rounds differ only by their data, at the cost of a full
training per round), so at a given round every strategy trains on the same number of frames and iterations. Every
model is scored on the same test set.

Honest intervals (analyse; the mixed experiment's method, core.experiments). A strategy's AP at a round is the mean over
seeds of COCO AP on the test set; its 95 % interval a bootstrap over the test set's units (drives, or sequences; images
for a real set without sequences), all runs resampled together. Each strategy is compared with random at the same
round by the difference on the same resamples (paired), and the verdict uses the combined interval
diff ± 1.96 √(sd_boot² + se² + se_random²) with se the seed spread's standard error (a normal approximation that treats
test-sample and training noise as independent). Two summaries per strategy: the area under its AP-vs-frames curve
(trapezoids over the rounds, divided by the frames spanned: a mean AP) against random's, paired the same way; and the
frames it needs to reach random's final-round AP (linear interpolation between rounds; "not reached" when it does not),
with the share of frames saved and a bootstrap interval over test resamples (training noise not included there).
"Better" or "worse" only when the interval excludes zero; otherwise "no detectable difference at 95 %".

Resumable: a finished run is skipped; a round's selection is kept in its run (selection.json) with the hash of the
model that scored it, so a resumed experiment never re-scores. Disk: a finished training keeps only its chosen weights
(no optimizer state), and they are removed once no later selection needs them — round 0's after every strategy's
first choice (keep_checkpoints keeps them).

select_frames() serves the loop (core.loop, step 5 with generate.acquisition): the frames of generated drives chosen
by a trained model's acquisition score instead of all of them.
"""
from __future__ import annotations

import math
import os

import numpy as np

SCHEMA = 'falcon.active_results/1'
STRATEGIES = ('random', 'entropy', 'margin', 'det_mean', 'det_max', 'kcenter', 'hybrid')
UNCERTAINTY = ('entropy', 'margin', 'det_mean', 'det_max')
AGGREGATES = ('max', 'mean', 'sum')
DET_FLOOR = 0.3                  # heuristic: detections below it are left out of det_mean / det_max
HYBRID_FACTOR = 4                # heuristic: hybrid's uncertainty short-list is this many times the budget
HEADLINE = ('AP', 'AP50', 'AP75', 'AR100')
ACTIVE_DEFAULTS = {
    'kind': 'active', 'name': None, 'pool': [], 'val': [], 'test': [], 'initial': 32, 'budget': 32, 'rounds': 3,
    'strategies': ['random', 'entropy', 'kcenter', 'hybrid'], 'seeds': 3, 'seed': 0, 'aggregate': 'max',
    'det_floor': DET_FLOOR, 'hybrid_factor': HYBRID_FACTOR, 'n_boot': 1000, 'boot_seed': 0, 'score_batch': 8,
    'keep_checkpoints': False, 'train': {},
}
VOLATILE = ('score_batch', 'keep_checkpoints')


class ActiveError(Exception):
    pass


# ================================================================================================ scoring
def _family(s: dict) -> str:
    from .training import ARCHS
    return ARCHS[s['arch']]['family']


def _agg(v: np.ndarray, how: str) -> float:
    if not len(v):
        return 0.0
    return float(v.max() if how == 'max' else v.mean() if how == 'mean' else v.sum())


def score_frames(model, samples: list, s: dict, device, aggregate: str = 'max', det_floor: float = DET_FLOOR,
                 batch: int = 8, cancel=None) -> dict:
    """Every acquisition score of every sample with the model in eval mode: {'entropy', 'margin', 'det_mean',
    'det_max', 'proposals', 'detections'} as arrays aligned with samples, and 'features' (n, d) float32 L2-normalised
    (the module doc defines each)."""
    import torch
    from .models import preprocess
    from .training import Cancelled, input_spec, load_rgb
    if aggregate not in AGGREGATES:
        raise ActiveError(f'aggregate is one of {", ".join(AGGREGATES)}')
    fam = _family(s)
    spec = input_spec(s)
    thr = float(s['score_threshold'])
    model.eval()
    out = {k: np.zeros(len(samples)) for k in ('entropy', 'margin', 'det_mean', 'det_max', 'proposals',
                                               'detections')}
    feats = []
    with torch.inference_mode():
        for j in range(0, len(samples), batch):
            if cancel and cancel():
                raise Cancelled()
            chunk = samples[j:j + batch]
            xs = [torch.from_numpy(preprocess(load_rgb(sm['image']), spec)[0][0]).to(device) for sm in chunk]
            images, _ = model.transform(xs)
            fmap = model.backbone(images.tensors)
            levels = list(fmap.values())
            pooled = torch.cat([f.float().mean((2, 3)) for f in levels], 1)
            feats.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy().astype(np.float32))
            if fam == 'fasterrcnn':
                rh = model.roi_heads
                props, _ = model.rpn(images, fmap)
                bf = rh.box_head(rh.box_roi_pool(fmap, props, images.image_sizes))
                logits, _ = rh.box_predictor(bf)
                probs = torch.softmax(logits.float(), -1).split([len(p) for p in props])
                per = []
                for q in probs:
                    q = q.cpu().numpy().astype(np.float64)
                    keep = q[:, 1:].max(1) >= thr if q.shape[1] > 1 else np.zeros(len(q), bool)
                    q = q[keep]
                    ent = -(q * np.log(np.clip(q, 1e-12, 1.0))).sum(1) / math.log(q.shape[1]) if len(q) else \
                        np.zeros(0)
                    top = np.sort(q, 1)[:, ::-1] if len(q) else np.zeros((0, 2))
                    mar = 1.0 - (top[:, 0] - top[:, 1]) if len(q) else np.zeros(0)
                    per.append((ent, mar, int(keep.sum())))
            else:
                head = model.head(levels)
                cls = torch.sigmoid(head['cls_logits'].float())
                if fam == 'fcos':
                    cls = torch.sqrt(cls * torch.sigmoid(head['bbox_ctrness'].float()))
                per = []
                for p in cls:
                    p = p.cpu().numpy().astype(np.float64)
                    p = p[p.max(1) >= thr] if len(p) else p
                    top = np.sort(p, 1)[:, ::-1] if len(p) else np.zeros((0, 2))
                    m1 = np.clip(top[:, 0], 1e-12, 1 - 1e-12) if len(p) else np.zeros(0)
                    ent = -(m1 * np.log(m1) + (1 - m1) * np.log(1 - m1)) / math.log(2.0) if len(p) else np.zeros(0)
                    second = top[:, 1] if top.shape[1] > 1 else np.zeros(len(top))
                    mar = 1.0 - (top[:, 0] - second) if len(p) else np.zeros(0)
                    per.append((ent, mar, len(p)))
            dets = model(xs)
            for k, ((ent, mar, n_prop), d) in enumerate(zip(per, dets)):
                i = j + k
                sc = d['scores'].float().cpu().numpy()
                sc = sc[sc >= det_floor]
                out['entropy'][i] = _agg(ent, aggregate)
                out['margin'][i] = _agg(mar, aggregate)
                out['det_mean'][i] = float((1.0 - sc).mean()) if len(sc) else 0.0
                out['det_max'][i] = float((1.0 - sc).max()) if len(sc) else 0.0
                out['proposals'][i] = n_prop
                out['detections'][i] = len(sc)
    out['features'] = np.concatenate(feats) if feats else np.zeros((0, 0), np.float32)
    return out


def kcenter_greedy(F: np.ndarray, labelled: list, candidates: list, k: int) -> list:
    """k-centre greedy (Sener & Savarese 2018): candidates' indices into F chosen one by one, each the farthest from
    every labelled or already chosen row. With nothing labelled the first is the candidate farthest from the
    candidates' mean. Ties: the lowest index."""
    cand = np.asarray(candidates, int)
    if k <= 0 or not len(cand):
        return []
    C = F[cand].astype(np.float64)
    if len(labelled):
        L = F[np.asarray(labelled, int)].astype(np.float64)
        d = np.full(len(cand), np.inf)
        for a in range(0, len(L), 512):
            blk = L[a:a + 512]
            dd = np.sqrt(np.maximum(((C[:, None, :] - blk[None, :, :]) ** 2).sum(-1), 0.0))
            d = np.minimum(d, dd.min(1))
    else:
        d = np.linalg.norm(C - C.mean(0), axis=1)
    chosen = []
    for _ in range(min(k, len(cand))):
        j = int(np.argmax(d))
        chosen.append(int(cand[j]))
        d = np.minimum(d, np.linalg.norm(C - C[j], axis=1))
        d[j] = -np.inf
    return chosen


def select(strategy: str, scores: dict, labelled: list, candidates: list, k: int, rng=None,
           hybrid_factor: int = HYBRID_FACTOR) -> list:
    """k of the candidates (indices into the scored samples) by a strategy (STRATEGIES); uncertainty strategies take
    the highest scores (ties: the lower index)."""
    cand = sorted(int(c) for c in candidates)
    k = min(int(k), len(cand))
    if strategy == 'random':
        rng = rng if rng is not None else np.random.default_rng(0)
        return sorted(int(cand[i]) for i in rng.permutation(len(cand))[:k])
    if strategy in UNCERTAINTY:
        v = np.asarray(scores[strategy])[cand]
        order = np.lexsort((np.asarray(cand), -v))          # highest score first, then the lower index
        return sorted(int(cand[i]) for i in order[:k])
    if strategy == 'kcenter':
        return sorted(kcenter_greedy(scores['features'], labelled, cand, k))
    if strategy == 'hybrid':
        v = np.asarray(scores['entropy'])[cand]
        order = np.lexsort((np.asarray(cand), -v))
        short = [cand[i] for i in order[:max(k, int(hybrid_factor) * k)]]
        return sorted(kcenter_greedy(scores['features'], labelled, short, k))
    raise ActiveError(f'unknown strategy {strategy!r}: one of {", ".join(STRATEGIES)}')


def load_trained(train_dir: str, workspace: str = '', device: str | None = None):
    """(model, settings, device) of a finished core.training run: its architecture built as it was (with its pinned
    initial weights, checked by their SHA-256), then the chosen epoch's weights (best.pt, else last.pt)."""
    import torch
    from . import provenance as PV
    from . import training as T
    run = PV.read_json(os.path.join(train_dir, 'run.json'))
    s = T.normalise({k: v for k, v in run['provenance']['settings'].items()})
    if device:
        s['device'] = device
    chosen = str((run.get('summary') or {}).get('chosen') or '')
    ck = os.path.join(train_dir, 'checkpoints', 'best.pt' if chosen.startswith('best') else 'last.pt')
    if not os.path.isfile(ck):
        raise ActiveError(f'{train_dir}: no {os.path.basename(ck)} (checkpoints removed?): retrain it to score with it')
    wname = T.weights_needed(s)
    wpath = None
    if wname:
        wpath, _ = T.find_weights(wname, T.weights_dirs(workspace, list(s.get('weights_dirs') or ())))
        if wpath is None:
            raise ActiveError(f'the pinned weights {wname} that built this model are not here (falcon train --fetch '
                              f'{wname})')
    dev = T._device(s)
    model, _ = T.build_model(s, wpath)
    model.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['model'])
    model.to(dev).eval()
    return model, s, dev


def select_frames(train_dir: str, runs: list, n: int, strategy: str = 'entropy', workspace: str = '',
                  aggregate: str = 'max', seed: int = 0, out: str = '', cancel=None) -> dict:
    """The loop's acquisition (core.loop step 5): the frames of runs (falcon drives) scored by the trained model of
    train_dir, the n best by the strategy kept. Writes out (selection JSON) when given: {'ids': [...], 'strategy',
    'model', 'frames', 'chosen', 'scores'} — a training source's select= reads it."""
    from . import provenance as PV
    from . import training as T
    model, s, dev = load_trained(train_dir, workspace)
    samples = []
    for r in runs:
        samples += T.load_source({'kind': 'falcon', 'path': r}, s['classes'])[0]
    if strategy not in STRATEGIES:
        raise ActiveError(f'unknown strategy {strategy!r}: one of {", ".join(STRATEGIES)}')
    sc = score_frames(model, samples, s, dev, aggregate, batch=int(s.get('val_batch') or 8), cancel=cancel) \
        if strategy != 'random' else {}
    idx = select(strategy, sc, [], list(range(len(samples))), n, np.random.default_rng(seed))
    doc = {'schema': 'falcon.acquisition/1', 'strategy': strategy, 'aggregate': aggregate,
           'model': {'training': os.path.abspath(train_dir), 'run_json_sha256': PV.sha256_file(
               os.path.join(train_dir, 'run.json'))},
           'frames': len(samples), 'chosen': len(idx), 'ids': [samples[i]['id'] for i in idx],
           'scores': {k: [float(sc[k][i]) for i in idx] for k in UNCERTAINTY if k in sc},
           'note': 'Falcon\'s labels are free: "chosen" means added to training (data efficiency, not labelling cost)'}
    if out:
        PV.write_json(out, doc)
    return doc


# ================================================================================================ the experiment
def normalise_active(d: dict) -> dict:
    from . import training as T
    m = {**ACTIVE_DEFAULTS, **{k: v for k, v in (d or {}).items() if v is not None}}
    m['strategies'] = list(dict.fromkeys(m['strategies']))
    bad = [x for x in m['strategies'] if x not in STRATEGIES]
    if bad:
        raise ActiveError(f'unknown strategies {", ".join(bad)}: one of {", ".join(STRATEGIES)}')
    if 'random' not in m['strategies']:
        raise ActiveError('the strategies must include random (the baseline every other strategy is compared with)')
    m['strategies'] = ['random'] + [x for x in m['strategies'] if x != 'random']
    for k in ('initial', 'budget', 'rounds', 'seeds'):
        if int(m[k]) < 1:
            raise ActiveError(f'{k} must be at least 1')
    if m['aggregate'] not in AGGREGATES:
        raise ActiveError(f'aggregate is one of {", ".join(AGGREGATES)}')
    for k in ('pool', 'test'):
        if not m[k]:
            raise ActiveError(f'{k} needs at least one source')
    srcs = lambda xs: [T.parse_source(x) if isinstance(x, str) else dict(x) for x in xs]     # noqa: E731
    for k in ('pool', 'val', 'test'):
        m[k] = srcs(m[k])
    tr = dict(m['train'])
    tr.update(train=[], val=[], test=[], patience=0, keep_last=True, export=False)
    m['train'] = T.normalise(tr)
    return m


def run_name(strategy: str | None, seed: int, rnd: int) -> str:
    return f'seed{seed}-r0' if rnd == 0 else f'{strategy}-seed{seed}-r{rnd}'


def _perm(seed: int, salt: int, n: int) -> np.ndarray:
    from .training import _mix
    return np.random.default_rng(_mix(seed, salt)).permutation(n)


def _weights_file(train_dir: str) -> str:
    from . import provenance as PV
    run = PV.read_json(os.path.join(train_dir, 'run.json'))
    chosen = str((run.get('summary') or {}).get('chosen') or '')
    return os.path.join(train_dir, 'checkpoints', 'best.pt' if chosen.startswith('best') else 'last.pt')


def _slim(train_dir: str):
    """A finished training keeps only the chosen epoch's weights (no optimizer state): best.pt when it was chosen
    (last.pt removed), else last.pt rewritten with the model alone. Its run is finished, so nothing resumes from it."""
    import torch
    d = os.path.join(train_dir, 'checkpoints')
    w = _weights_file(train_dir)
    last = os.path.join(d, 'last.pt')
    if os.path.basename(w) == 'best.pt':
        if os.path.isfile(last):
            os.remove(last)
    elif os.path.isfile(last):
        ck = torch.load(last, map_location='cpu', weights_only=False)
        tmp = last + '.part'
        torch.save({'settings_key': ck.get('settings_key'), 'model': ck['model']}, tmp)
        os.replace(tmp, last)


def _drop_checkpoints(train_dir: str):
    d = os.path.join(train_dir, 'checkpoints')
    for f in ('best.pt', 'last.pt'):
        p = os.path.join(d, f)
        if os.path.isfile(p):
            os.remove(p)


def run_active(root: str, exp_id: str, definition: dict, emitter=None, cancel=None, workspace: str = '',
               weights_dirs_extra=()) -> dict:
    """Run (or resume) the active-learning experiment experiments/<exp_id>/ and analyse it: results.json's document.
    Raises ActiveError, training.TrainingError or training.Cancelled."""
    from . import provenance as PV
    from . import training as T
    from .experiments import Experiment, _SubEmitter
    em = emitter or T._Quiet()
    cancel = cancel or (lambda: False)
    m = normalise_active(definition)
    fixed = {k: v for k, v in m.items() if k not in VOLATILE}
    fixed['train'] = {k: v for k, v in m['train'].items() if k not in T.VOLATILE}
    exp = Experiment.create(root, exp_id, fixed, m.get('name'))
    classes = m['train']['classes']
    pool, n1, i1 = T.load_sources(m['pool'], classes)
    test, n2, i2 = T.load_sources(m['test'], classes)
    val, n3, i3 = T.load_sources(m['val'], classes)
    notes, ins = n1 + n2 + n3, {**i1, **i2, **i3}
    t_ids, t_units = {s['id'] for s in test}, {s['unit'] for s in test}
    for name, part in (('pool', pool), ('validation', val)):
        if t_ids & {s['id'] for s in part} or t_units & {s['unit'] for s in part}:
            raise ActiveError(f'the test set shares images or drives/sequences with the {name}: a held-out test set '
                              f'may share neither')
    if val and ({s['id'] for s in val} & {s['id'] for s in pool} or {s['unit'] for s in val} &
                {s['unit'] for s in pool}):
        raise ActiveError('the validation set shares images or drives with the pool')
    N = len(pool)
    need = int(m['initial']) + int(m['rounds']) * int(m['budget'])
    if N < need:
        raise ActiveError(f'{m["initial"]} initial + {m["rounds"]} rounds × {m["budget"]} frames need {need} pool '
                          f'frames; the pool has {N}')
    seeds = [int(m['seed']) + j for j in range(int(m['seeds']))]
    strategies = m['strategies']
    R = int(m['rounds'])
    sizes = [int(m['initial']) + r * int(m['budget']) for r in range(R + 1)]
    plan = [(None, sd, 0) for sd in seeds] + [(st, sd, r) for sd in seeds for st in strategies
                                              for r in range(1, R + 1)]
    em.log(f'{len(strategies)} strategies × {len(seeds)} seeds × {R} rounds of {m["budget"]} frames from '
           f'{m["initial"]}; pool {N} frames, test {len(test)} images; {len(plan)} trainings')
    pool_ids = [s['id'] for s in pool]
    done = [0, 0]

    def train_run(name, idx, k, why, st_name, rnd, sd):
        tr = exp.run(name)
        tdir = os.path.join(tr.dir, 'training')
        if tr.status == 'ok' and os.path.isfile(os.path.join(tdir, 'test_predictions.jsonl')):
            done[1] += 1
            return tr, tdir
        if cancel():
            raise T.Cancelled()
        s = dict(m['train'], seed=sd)
        params = {'strategy': st_name, 'seed': sd, 'round': rnd, 'frames': len(idx), 'budget': m['budget'],
                  **{f'train.{kk}': vv for kk, vv in s.items() if kk not in ('train', 'val', 'test')}}
        tr.start(params, {'strategy': st_name, 'seed': sd, 'round': rnd, 'kind': 'active'}, sd, ins)
        PV.write_json(os.path.join(tr.dir, 'labelled.json'), {'ids': [pool_ids[i] for i in idx], 'why': why})
        sub = _SubEmitter(em, k / len(plan), (k + 1) / len(plan), name)
        try:
            summ = T.train(s, tdir, data={'train': [pool[i] for i in idx], 'val': val, 'test': test, 'inputs': ins,
                                          'notes': notes + [f'{name}: {len(idx)} frames ({why})']},
                           workspace=workspace, emitter=sub, cancel=cancel, argv=['experiment', 'active', exp.id, name],
                           weights_dirs_extra=weights_dirs_extra)
        except T.Cancelled:
            tr.finish('cancelled')
            raise
        except T.TrainingError as e:
            tr.finish('failed', str(e))
            raise
        _record(tr, tdir, summ)
        _slim(tdir)
        tr.finish('ok')
        done[0] += 1
        return tr, tdir

    at = {x: i for i, x in enumerate(pool_ids)}

    def selection(st, sd, r, labelled, prev_dir, k):
        """Round r's choice by st (selection.json; made with prev_dir's model unless already made from exactly these
        frames in training)."""
        tr = exp.run(run_name(st, sd, r))
        sel_p = os.path.join(tr.dir, 'selection.json')
        sel = PV.read_json(sel_p) if os.path.isfile(sel_p) else None
        if sel is None or sel.get('labelled_before') != [pool_ids[i] for i in labelled]:
            if cancel():
                raise T.Cancelled()
            em.progress(k / len(plan), f'{tr.name}: choosing {m["budget"]} frames by {st}', force=True)
            sel = _choose(m, st, sd, r, prev_dir, _weights_file(prev_dir), pool, labelled, workspace, cancel)
            os.makedirs(tr.dir, exist_ok=True)
            sel['labelled_before'] = [pool_ids[i] for i in labelled]
            PV.write_json(sel_p, sel)
        return [at[x] for x in sel['ids']]

    keep = m['keep_checkpoints']
    k = 0
    for sd in seeds:
        init = sorted(int(i) for i in _perm(sd, 3, N)[:int(m['initial'])])
        _, r0dir = train_run(run_name(None, sd, 0), init, k, f'the initial {m["initial"]} frames of seed {sd}\'s '
                                                             f'permutation, shared by every strategy', 'initial', 0, sd)
        k += 1
        for st in strategies:                    # every strategy's round-1 choice first: then round 0's weights go
            selection(st, sd, 1, init, r0dir, k)
        if not keep:
            _drop_checkpoints(r0dir)
        for st in strategies:
            labelled = list(init)
            prev_dir = r0dir
            for r in range(1, R + 1):
                chosen = selection(st, sd, r, labelled, prev_dir, k)
                if prev_dir != r0dir and not keep:
                    _drop_checkpoints(prev_dir)              # no later choice needs it
                labelled = sorted(set(labelled) | set(chosen))
                _, tdir = train_run(run_name(st, sd, r), labelled, k, f'{st}, round {r}: {len(chosen)} chosen by {st}',
                                    st, r, sd)
                k += 1
                prev_dir = tdir
            if not keep:
                _drop_checkpoints(prev_dir)
    em.progress(1.0, 'analysing', force=True)
    res = analyse(exp, m, seeds, sizes, notes)
    res['runs_trained'], res['runs_skipped'] = done
    return res


def _choose(m, st, sd, r, prev_dir, prev_w, pool, labelled, workspace, cancel) -> dict:
    """One round's selection by strategy st with the previous round's model (selection.json)."""
    from . import provenance as PV
    cand = sorted(set(range(len(pool))) - set(labelled))
    sc = {}
    model_doc = {'training': prev_dir}
    if st != 'random':
        if not os.path.isfile(prev_w):
            raise ActiveError(f'{prev_dir}: its weights were removed before round {r}\'s selection; delete that run\'s '
                              f'folder to retrain it')
        model_doc['weights_sha256'] = PV.sha256_file(prev_w)
        model, s, dev = load_trained(prev_dir, workspace)
        sc = score_frames(model, pool, s, dev, m['aggregate'], float(m['det_floor']), int(m['score_batch']), cancel)
    from .training import _mix
    idx = select(st, sc, labelled, cand, int(m['budget']), np.random.default_rng(_mix(sd, r, 0xA11)),
                 int(m['hybrid_factor']))
    return {'strategy': st, 'round': r, 'model': model_doc, 'candidates': len(cand),
            'ids': [pool[i]['id'] for i in idx],
            'scores': {k: [float(sc[k][i]) for i in idx] for k in UNCERTAINTY if k in sc},
            'pool_scores': {k: {'mean': float(np.mean(sc[k][cand])), 'max': float(np.max(sc[k][cand]))}
                            for k in UNCERTAINTY if k in sc and cand}}


def _record(tr, tdir: str, summ: dict):
    from .training import _read_metrics
    for row in _read_metrics(os.path.join(tdir, 'metrics.jsonl')):
        tr.log_metrics({'train_loss': row.get('train_loss'), **{f'val.{k}': (row.get('val') or {}).get(k)
                                                                 for k in HEADLINE}}, step=row['epoch'])
    t = summ.get('test') or {}
    tr.log_metrics({f'test.{k}': t.get(k) for k in HEADLINE}, step=summ.get('epochs_run') or 0)
    for f in ('run.json', 'metrics.jsonl', 'test_predictions.jsonl'):
        if os.path.isfile(os.path.join(tdir, f)):
            tr.log_artifact(os.path.join(tdir, f))


def _frames_to_reach(sizes: list, curve: list, target) -> float | None:
    """The first frame count at which the curve reaches target (linear between rounds); None when it does not."""
    if target is None or any(v is None for v in curve):
        return None
    if curve[0] >= target:
        return float(sizes[0])
    for a in range(1, len(curve)):
        if curve[a] >= target:
            y0, y1 = curve[a - 1], curve[a]
            f = (target - y0) / (y1 - y0) if y1 != y0 else 1.0
            return float(sizes[a - 1] + f * (sizes[a] - sizes[a - 1]))
    return None


def _aulc(sizes: list, curve: list) -> float | None:
    if any(v is None for v in curve):
        return None
    if len(sizes) < 2:
        return float(curve[0])
    return float(np.trapezoid(curve, sizes) / (sizes[-1] - sizes[0])) if hasattr(np, 'trapezoid') else \
        float(np.trapz(curve, sizes) / (sizes[-1] - sizes[0]))


def analyse(exp, m: dict, seeds: list, sizes: list, notes=()) -> dict:
    """AP per strategy and round with intervals over the test set's units, paired against random at each round, the
    area under the curve and the frames to reach random's final AP (module doc): results.json."""
    from . import metrics as M
    from . import provenance as PV
    from .experiments import _frames, _unit_description
    strategies = m['strategies']
    R = len(sizes) - 1

    def rn(st, sd, r):
        return run_name(st, sd, r)
    keys = [(st, sd, r) for st in strategies for sd in seeds for r in range(R + 1)]
    paths = {kk: os.path.join(exp.path, 'runs', rn(kk[0], kk[1], kk[2]), 'training', 'test_predictions.jsonl')
             for kk in keys}
    missing = sorted({rn(*kk) for kk, p in paths.items() if not os.path.isfile(p)})
    if missing:
        raise ActiveError(f'runs not finished: {", ".join(missing)}')
    key = PV.sha256_json({'preds': {rn(*kk): PV.sha256_file(p) for kk, p in paths.items()}, 'n_boot': m['n_boot'],
                          'boot_seed': m['boot_seed']})
    out_p = os.path.join(exp.path, 'results.json')
    if os.path.isfile(out_p):
        old = PV.read_json(out_p)
        if old.get('key') == key:
            return old
    uniq = {}
    for kk, p in paths.items():
        uniq.setdefault(p, kk)
    frames = {p: {f['image_id']: f for f in _frames(p)} for p in uniq}
    ids = sorted(next(iter(frames.values())))
    for p, fr in frames.items():
        if sorted(fr) != ids:
            raise ActiveError(f'{p} was tested on other images than the others')
    recs = {p: {r['image_id']: r for r in M.coco_match([fr[i] for i in ids])} for p, fr in frames.items()}
    first = frames[next(iter(frames))]
    items = [{'image_id': i, 'drive': first[i]['drive'], 'r': {p: recs[p][i] for p in recs}} for i in ids]
    n_obj = sum(1 for i in ids for g in first[i]['gt'] if not g['ignore'])

    def ap_of(its, p, metric='AP'):
        try:
            return M.coco_summarise([it['r'][p] for it in its]).get(metric)
        except ValueError:
            return None

    def curves(its, metric='AP'):
        """{strategy: [mean over seeds per round]} and {strategy: [[per seed] per round]}"""
        cache = {}
        per = {}
        for st in strategies:
            rows = []
            for r in range(R + 1):
                vals = []
                for sd in seeds:
                    p = paths[(st, sd, r)]
                    if p not in cache:
                        cache[p] = ap_of(its, p, metric)
                    vals.append(cache[p])
                rows.append(vals)
            per[st] = rows
        mean = {st: [float(np.mean(v)) if all(x is not None for x in v) else None for v in rows]
                for st, rows in per.items()}
        return mean, per

    def metric_fn(its):
        mean, per = curves(its)
        out = {}
        target = mean['random'][-1]
        for st in strategies:
            for r in range(R + 1):
                out[f'{st}|{r}'] = mean[st][r]
                if st != 'random' and mean[st][r] is not None and mean['random'][r] is not None:
                    out[f'{st}-random|{r}'] = mean[st][r] - mean['random'][r]
            a = _aulc(sizes, mean[st])
            out[f'{st}|aulc'] = a
            if st != 'random' and a is not None and out.get('random|aulc') is not None:
                out[f'{st}-random|aulc'] = a - out['random|aulc']
            out[f'{st}|reach'] = _frames_to_reach(sizes, mean[st], target)
        return out
    n_boot, bseed = int(m['n_boot']), int(m['boot_seed'])
    boot = M.bootstrap(metric_fn, items, n=n_boot, seed=bseed, unit='drive')
    bm = boot['metrics']
    _, per_full = curves(items)
    K = len(seeds)

    def seed_se(st, r):
        v = [x for x in per_full[st][r] if x is not None]
        sd_ = float(np.std(v, ddof=1)) if len(v) > 1 else None
        return sd_, (sd_ / math.sqrt(len(v)) if sd_ is not None else None)

    def aulc_seed(st):
        v = []
        for j in range(K):
            c = [per_full[st][r][j] for r in range(R + 1)]
            a = _aulc(sizes, c)
            if a is not None:
                v.append(a)
        sd_ = float(np.std(v, ddof=1)) if len(v) > 1 else None
        return v, sd_, (sd_ / math.sqrt(len(v)) if sd_ is not None else None)

    def verdict(diff, sd_boot, se_a, se_b, lo, hi):
        comb = None
        if diff is not None and sd_boot is not None and se_a is not None and se_b is not None:
            h = 1.96 * math.sqrt(sd_boot ** 2 + se_a ** 2 + se_b ** 2)
            comb = [diff - h, diff + h]
        a, b = comb if comb else (lo, hi)
        if a is None or b is None:
            v = 'not measurable (nothing to score)'
        elif a > 0:
            v = 'better than random'
        elif b < 0:
            v = 'worse than random'
        else:
            v = 'no detectable difference at 95 %'
        if not comb and K == 1:
            v += ' (one seed: training noise not measured)'
        if boot['n_units'] < 5 and v.startswith(('better', 'worse')):
            v += f' (only {boot["n_units"]} test unit(s): not reliable)'
        return comb, v

    rows = []
    for st in strategies:
        rounds = []
        for r in range(R + 1):
            b = bm[f'{st}|{r}']
            sd_, se = seed_se(st, r)
            row = {'round': r, 'frames': sizes[r], 'AP': {kk: b[kk] for kk in ('value', 'ci_low', 'ci_high')},
                   'per_seed': per_full[st][r], 'seed_sd': sd_, 'vs_random': None}
            if st != 'random':
                d = bm.get(f'{st}-random|{r}') or {}
                _, se_r = seed_se('random', r)
                comb, v = verdict(d.get('value'), d.get('sd'), se, se_r, d.get('ci_low'), d.get('ci_high'))
                if r == 0:
                    comb, v = [0.0, 0.0], 'the same model (round 0 is shared)'
                row['vs_random'] = {'diff': d.get('value'), 'ci_low': d.get('ci_low'), 'ci_high': d.get('ci_high'),
                                    'sd_boot': d.get('sd'), 'combined_ci': comb, 'verdict': v}
            rounds.append(row)
        av, asd, ase = aulc_seed(st)
        ab = bm[f'{st}|aulc']
        summ = {'strategy': st, 'rounds': rounds,
                'aulc': {'value': ab['value'], 'ci_low': ab['ci_low'], 'ci_high': ab['ci_high'], 'per_seed': av,
                         'seed_sd': asd}, 'aulc_vs_random': None}
        if st != 'random':
            d = bm.get(f'{st}-random|aulc') or {}
            _, _, ase_r = aulc_seed('random')
            comb, v = verdict(d.get('value'), d.get('sd'), ase, ase_r, d.get('ci_low'), d.get('ci_high'))
            summ['aulc_vs_random'] = {'diff': d.get('value'), 'ci_low': d.get('ci_low'), 'ci_high': d.get('ci_high'),
                                      'combined_ci': comb, 'verdict': v.replace('than random', 'than random: '
                                                                                'more data-efficient')
                                      if v.startswith('better') else v}
        rb = bm[f'{st}|reach']
        reach = rb['value']
        summ['frames_to_reach_random_final'] = {
            'value': reach, 'ci_low': rb['ci_low'], 'ci_high': rb['ci_high'],
            'reached_share': rb['n_valid'] / max(n_boot, 1),
            'saving': (1.0 - reach / sizes[-1]) if reach is not None else None,
            'note': 'the mean curve over seeds; interval over test resamples where reached (training noise not in '
                    'it)'}
        rows.append(summ)
    n_units = boot['n_units']
    warn = []
    if n_units < 5:
        warn.append(f'only {n_units} test units: the bootstrap intervals are not reliable')
    if K == 1:
        warn.append('one seed: training noise is not measured')
    if n_obj < 50:
        warn.append(f'{n_obj} scored test objects: AP differences of a few points are noise at this size')
    top = max((r['AP']['value'] or 0.0 for s_ in rows for r in s_['rounds']), default=0.0)
    if top < 0.01:
        warn.append(f'AP below 0.01 at every point (at most {top:.4f}): the models detect almost nothing, so no '
                    f'difference here means anything about the strategies')
    unit_desc = _unit_description(list(first.values()))
    doc = {'schema': SCHEMA, 'experiment': exp.id, 'key': key, 'strategies': strategies, 'frames': sizes,
           'initial': m['initial'], 'budget': m['budget'], 'rounds': R, 'seeds': seeds, 'pool': None,
           'classes': m['train']['classes'], 'aggregate': m['aggregate'],
           'test': {'images': len(ids), 'objects': n_obj, 'unit': unit_desc, 'n_units': n_units},
           'n_boot': n_boot, 'boot_seed': bseed, 'results': rows, 'warnings': warn, 'notes': list(notes),
           'labels': 'Falcon\'s labels are free (simulation): a chosen frame is added to training; the measure is '
                     'data efficiency — AP per training frame — against random choice from the same pool',
           'method': {
               'value': 'mean over seeds of COCO AP on the test set (core.metrics)',
               'interval': f'95 % percentile bootstrap over test {unit_desc}s ({n_boot} resamples), every run '
                           f'resampled together; the test sample only, not training noise',
               'paired': 'strategy − random at the same round on the same resamples',
               'combined': 'diff ± 1.96 √(sd_boot² + se² + se_random²), se = seed sd / √K (normal approximation)',
               'aulc': 'area under AP vs frames (trapezoids over the rounds) / frames spanned',
               'reach': 'frames for the mean curve to reach random\'s final-round mean AP (linear between rounds)',
               'verdict': 'better/worse only when the combined interval excludes 0'},
           'provenance': PV.block({rn(*kk): p for kk, p in paths.items()}, bseed, {'n_boot': n_boot})}
    PV.write_json(out_p, doc)
    tr = exp.run('analysis')
    tr.start({'n_boot': n_boot, 'boot_seed': bseed, 'seeds': len(seeds), 'rounds': R}, {'kind': 'active-analysis'})
    for s_ in rows:
        for rr in s_['rounds']:
            tr.log_metrics({f'{s_["strategy"]}.AP': rr['AP']['value'], f'{s_["strategy"]}.AP.ci_low':
                            rr['AP']['ci_low'], f'{s_["strategy"]}.AP.ci_high': rr['AP']['ci_high']},
                           step=rr['frames'])
        tr.log_metrics({f'{s_["strategy"]}.aulc': s_['aulc']['value']})
    tr.log_artifact(out_p)
    tr.finish('ok')
    return doc


def table(res: dict) -> str:
    """results.json as text: AP per strategy and frame count, then the summaries."""
    head = f'{"strategy":10s}' + ''.join(f'{n:>24d}' for n in res['frames'])
    lines = [f'AP [95 % CI] by training frames ({len(res["seeds"])} seeds; test {res["test"]["images"]} images, '
             f'{res["test"]["objects"]} objects, unit {res["test"]["unit"]} ({res["test"]["n_units"]}))', head,
             '-' * len(head)]
    for s in res['results']:
        cells = []
        for r in s['rounds']:
            a = r['AP']
            cells.append('—' if a['value'] is None else
                         f'{a["value"]:.4f} [{a["ci_low"]:.4f},{a["ci_high"]:.4f}]' if a['ci_low'] is not None
                         else f'{a["value"]:.3f}')
        lines.append(f'{s["strategy"]:10s}' + ''.join(f'{c:>24s}' for c in cells))
    lines.append('')
    for s in res['results']:
        au = s['aulc']
        line = f'{s["strategy"]:10s} area under the curve {au["value"]:.4f}' if au['value'] is not None else \
            f'{s["strategy"]:10s} area under the curve —'
        if s.get('aulc_vs_random'):
            d = s['aulc_vs_random']
            if d['diff'] is not None:
                line += f'; vs random {d["diff"]:+.4f} [{d["ci_low"]:+.4f}, {d["ci_high"]:+.4f}] → {d["verdict"]}'
        rf = s['frames_to_reach_random_final']
        line += ('; reaches random\'s final AP at ' + (f'{rf["value"]:.0f} frames (saves {100 * rf["saving"]:.0f} %)'
                                                       if rf['value'] is not None else 'no frame count tried'))
        lines.append(line)
    lines += [f'note: {w}' for w in res.get('warnings') or ()]
    return '\n'.join(lines)


def main_active(argv) -> int:
    """`falcon experiment active --id ID --pool SRC... --test SRC... [--strategies random,entropy,…] [--initial N]
    [--budget B] [--rounds R] [--seeds K] --workspace W`: protocol lines; 0 ok, 2 failed, 3 cancelled."""
    import argparse
    from . import protocol as P
    from . import provenance as PV
    from . import training as T
    ap = argparse.ArgumentParser(prog='falcon experiment active', description='active learning against random choice '
                                                                              '(docs/active-learning.md)')
    ap.add_argument('--root', default='', help='the experiments folder (default WORKSPACE/experiments)')
    ap.add_argument('--workspace', default='')
    ap.add_argument('--id', required=True)
    ap.add_argument('--config', help='a JSON file of the definition (the flags below override it)')
    ap.add_argument('--pool', action='append', default=None, metavar='KIND:PATH[?k=v&…]')
    ap.add_argument('--val', action='append', default=None, metavar='KIND:PATH[?k=v&…]')
    ap.add_argument('--test', action='append', default=None, metavar='KIND:PATH[?k=v&…]')
    ap.add_argument('--strategies', help=f'comma-separated, of {",".join(STRATEGIES)} (random is always included)')
    for k in ('initial', 'budget', 'rounds', 'seeds', 'seed', 'boot'):
        ap.add_argument(f'--{k}', type=int)
    ap.add_argument('--aggregate', choices=AGGREGATES)
    ap.add_argument('--keep-checkpoints', action='store_true')
    for k, t in (('epochs', int), ('batch', int), ('lr', float), ('workers', int)):
        ap.add_argument(f'--{k}', type=t)
    ap.add_argument('--arch', choices=tuple(T.ARCHS))
    ap.add_argument('--init', choices=T.INITS)
    ap.add_argument('--size', help='WIDTHxHEIGHT')
    ap.add_argument('--classes')
    ap.add_argument('--device')
    ap.add_argument('--weights-dir', action='append', default=[])
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    em = P.Emitter()
    em.hello('experiment')
    flag = T.CancelFlag().install()
    locks = []
    try:
        d = PV.read_json(a.config) if a.config else {}
        over = {'pool': a.pool, 'val': a.val, 'test': a.test, 'initial': a.initial, 'budget': a.budget,
                'rounds': a.rounds, 'seeds': a.seeds, 'seed': a.seed, 'n_boot': a.boot, 'aggregate': a.aggregate}
        d.update({k: v for k, v in over.items() if v is not None})
        if a.strategies:
            d['strategies'] = [x.strip() for x in a.strategies.split(',') if x.strip()]
        if a.keep_checkpoints:
            d['keep_checkpoints'] = True
        trn = dict(d.get('train') or {})
        trn.update({k: v for k, v in {'epochs': a.epochs, 'batch': a.batch, 'lr': a.lr, 'workers': a.workers,
                                      'arch': a.arch, 'init': a.init, 'device': a.device}.items() if v is not None})
        if a.size:
            trn['size'] = T._size(a.size)
        if a.classes:
            trn['classes'] = [c.strip() for c in a.classes.split(',') if c.strip()]
        d['train'] = trn
        root = a.root or (os.path.join(a.workspace, 'experiments') if a.workspace else '')
        if not root:
            raise ActiveError('--root DIR or --workspace W is needed')
        m = normalise_active(d)
        if T._device(m['train']).type == 'cuda':
            locks = T.gpu_locks(em, flag, held_by_queue=os.environ.get('FALCON_GPU_LOCK_HELD') == '1')
        res = run_active(root, a.id, d, em, flag, a.workspace, a.weights_dir)
        print(table(res), flush=True)
        from .experiments import _safe
        em.result('ok', results=os.path.join(root, _safe(a.id), 'results.json'),
                  summary={s['strategy']: s['aulc']['value'] for s in res['results']})
        return P.OK
    except T.Cancelled:
        em.result('cancelled')
        return P.CANCELLED
    except (ActiveError, T.TrainingError, ValueError, OSError, KeyError) as e:
        print(f'failed: {e}', flush=True)
        em.result('failed', error=str(e))
        return P.FAILED
    finally:
        for lk in locks:
            lk.release()
