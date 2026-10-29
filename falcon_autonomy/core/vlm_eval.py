"""Evaluate a vision-language model's driving explanations against the simulator's truth, and the `falcon vlm` command
(standard library and numpy at import; a runner's torch only inside it).

    evaluate(settings, out_dir, ...) -> summary     answers.jsonl, scores.json, run.json in out_dir (resumable)
    main_vlm(argv)                                  falcon vlm list | verify | fetch | eval | finetune
    job_spec(argv, needs_gpu)                       the JobSpec the GUI submits
    report(src) -> core.reports.Page                the HTML report of an evaluation (falcon report vlm DIR)

An evaluation reads language folders (core.language: `falcon derive language RUN --out DIR`, or derived here from a
run with --run into OUT/language/<drive>/). Each folder is one drive; its manifest names the run, whose frames.jsonl
gives the camera image of each step (the first "<camera>/rgb" unless a camera is chosen). Every n-th record (every,
default 5) up to max_frames per drive is asked core.vlm.questions(); each question goes to the runner with its images
(clip = how many frames, oldest first, clip_step steps apart, ending at the frame) and the prompt (question +
core.vlm.PROMPT_SUFFIX). Each answer is scored (core.vlm.score_answer) and written to answers.jsonl as it comes, so a
cancelled evaluation resumes where it stopped (same settings; another settings hash in the same folder is refused).
scores.json holds core.vlm.aggregate() — per question type, accuracy with 95 % bootstrap intervals over drives (2 s
blocks under five drives, said so), grounding, the hallucination rate, the explanation's precision and recall — with
core.vlm.method() and the runner's description including its licence. run.json is the provenance: the language
files' and model files' hashes, settings, seed, versions, timings.

An optional judge for the explanations: a client the user writes and approves — an object with .complete(prompt) or
a callable, named as module:Class with judge_approved true (--judge M:C --approve-judge). Falcon opens no connection
of its own and never uses a judge by default. The judge's verdict (FAITHFUL / UNFAITHFUL as its first word) is stored
beside the rule-based one, and scores.json reports its faithful rate and its agreement with the rules; the rules stay
the primary score.

GPU: an hf runner on CUDA is a needs_gpu job; run from the CLI it takes core.training's gpu_locks (waits while a CARLA
job holds the GPU) unless the job queue holds them (FALCON_GPU_LOCK_HELD).
"""
from __future__ import annotations

import json
import os
import time

SCHEMA = 'falcon.vlm.evaluation/1'
DEFAULTS = {'runner': 'oracle', 'lang': [], 'runs': [], 'workspace': '', 'every': 5, 'max_frames': 0, 'types': None,
            'max_objects': 2, 'clip': 1, 'clip_step': 5, 'camera': None, 'n_boot': 1000, 'seed': 0, 'tolerance': None,
            'judge': None, 'judge_approved': False, 'name': ''}
VOLATILE = ('n_boot', 'workspace', 'name')
JUDGE_PROMPT = ('You check whether a driving explanation is faithful to the facts. Facts of the scene, from the '
                'simulator (distances in metres, CARLA track ids in brackets): {facts}. Ground-truth reasons to brake: '
                '{reasons}. The explanation: "{answer}". Answer FAITHFUL if every reason it states is true of the '
                'facts '
                'and it states at least one of the ground-truth reasons (or, when there is none, says that nothing '
                'calls for braking); otherwise answer UNFAITHFUL. Answer with that one word first.')


class EvalError(Exception):
    pass


# ================================================================================================ reading
def load_language(folder: str) -> dict:
    """A language folder (core.language): its manifest, records, drive name, run folder and frames by step."""
    from .provenance import read_json
    man_p = os.path.join(folder, 'language.json')
    if not os.path.isfile(man_p):
        raise EvalError(f'{folder}: no language.json — derive it first (falcon derive language RUN --out DIR)')
    man = read_json(man_p)
    with open(os.path.join(folder, 'language.jsonl'), encoding='utf-8') as fh:
        recs = [json.loads(x) for x in fh if x.strip()]
    run = os.path.expanduser(man.get('run') or '')
    frames = {}
    fp = os.path.join(run, 'frames.jsonl')
    if run and os.path.isfile(fp):
        with open(fp, encoding='utf-8') as fh:
            for x in fh:
                if x.strip():
                    f = json.loads(x)
                    frames[int(f['step'])] = f
    name = os.path.basename(os.path.normpath(run)) if run else os.path.basename(os.path.normpath(folder))
    return {'folder': os.path.abspath(folder), 'manifest': man, 'records': recs, 'name': name, 'run': run,
            'frames': frames}


def images_for(L: dict, step: int, camera=None, clip: int = 1, clip_step: int = 5) -> list:
    """The camera images of a clip ending at step (absolute paths, oldest first; steps the run lacks are skipped)."""
    out = []
    for k in range(max(int(clip), 1) - 1, -1, -1):
        f = L['frames'].get(int(step) - k * max(int(clip_step), 1))
        if not f:
            continue
        files = f.get('files') or {}
        key = f'{camera}/rgb' if camera else next((k2 for k2 in files if k2.endswith('/rgb')), None)
        if key and key in files:
            p = os.path.join(L['run'], files[key])
            if os.path.isfile(p):
                out.append(p)
    return out


def _facts_text(ctx: dict) -> str:
    return '; '.join(f'{a["noun"]} [{a["id"]}] {a["distance_m"]:.1f} m {a["direction"]}, lane {a.get("lane_rel")}, '
                     f'speed {a["speed_mps"]:.1f} m/s, TTC {a["ttc_s"] if a.get("ttc_s") is not None else "none"}'
                     for a in ctx['agents']) or 'no vehicles or pedestrians within range'


def make_judge(entry: str):
    import importlib
    mod, _, cls = entry.partition(':')
    if not mod or not cls:
        raise EvalError('a judge is module:Class')
    return getattr(importlib.import_module(mod), cls)()


def ask_judge(judge, q: dict, answer: str, ctx: dict):
    """The judge's verdict (True faithful, False not, None unreadable) and its raw text."""
    reasons = '; '.join(f'{r["kind"]}' + (f' [{r["id"]}]' if r.get('id') is not None else '') +
                        (f' {r["state"]}' if r.get('state') else '') for r in q['truth']['reasons']) or 'none'
    prompt = JUDGE_PROMPT.format(facts=_facts_text(ctx), reasons=reasons, answer=answer)
    out = judge.complete(prompt) if hasattr(judge, 'complete') else judge(prompt)
    w = str(out or '').strip().split()
    first = w[0].strip('.,:;!').upper() if w else ''
    return {'FAITHFUL': True, 'UNFAITHFUL': False}.get(first), str(out)


# ================================================================================================ evaluate
def normalise(settings: dict) -> dict:
    s = dict(DEFAULTS, **(settings or {}))
    unknown = sorted(set(s) - set(DEFAULTS))
    if unknown:
        raise EvalError(f'unknown settings: {", ".join(unknown)}')
    s['lang'] = [os.path.abspath(d) for d in s['lang']]
    s['runs'] = [os.path.abspath(d) for d in s['runs']]
    if s['judge'] and not s['judge_approved']:
        raise EvalError('a judge runs only when you approve it (--approve-judge): it sends the facts and answers to '
                        'whatever client the judge class talks to')
    return s


def settings_key(s: dict) -> str:
    from .provenance import sha256_json
    return sha256_json({k: v for k, v in s.items() if k not in VOLATILE})


def _derive(runs: list, out_dir: str) -> list:
    from .language import derive_language
    out = []
    for r in runs:
        d = os.path.join(out_dir, 'language', os.path.basename(os.path.normpath(r)))
        if not os.path.isfile(os.path.join(d, 'language.json')):
            derive_language(r, d)                 # template text only: no paraphrase client
        out.append(d)
    return out


def evaluate(settings: dict, out_dir: str, runner=None, emitter=None, cancel=None, argv=None) -> dict:
    """Ask the runner every question of the chosen drives and score the answers (module doc). runner: a core.vlm_models
    Runner (else made from settings['runner']). Returns the summary (also in run.json). Raises training.Cancelled
    after keeping what was answered."""
    from . import provenance as PV
    from . import vlm as V
    from .training import Cancelled, _Quiet
    from .vlm_models import make_runner
    s = normalise(settings)
    em = emitter or _Quiet()
    cancel = cancel or (lambda: False)
    os.makedirs(out_dir, exist_ok=True)
    key = settings_key(s)
    run_p = os.path.join(out_dir, 'run.json')
    prev = PV.read_json(run_p) if os.path.isfile(run_p) else None
    if prev is not None and prev.get('settings_key') != key:
        raise EvalError(f'{out_dir} holds an evaluation with other settings; use another folder')
    if prev is not None and prev.get('status') == 'ok' and os.path.isfile(os.path.join(out_dir, 'scores.json')):
        return prev['summary']
    langs = list(s['lang']) + _derive(s['runs'], out_dir)
    if not langs:
        raise EvalError('no drives: give language folders (--lang) or runs (--run)')
    drives = [load_language(d) for d in langs]
    names = [L['name'] for L in drives]
    if len(set(names)) != len(names):
        raise EvalError(f'two folders are the same drive: {names}')
    runner = runner or make_runner(s['runner'], s['workspace'])
    judge = make_judge(s['judge']) if s['judge'] else None
    opts = {'max_objects': s['max_objects']}
    if s['types']:
        bad = sorted(set(s['types']) - set(V.QTYPES))
        if bad:
            raise EvalError(f'unknown question types {bad}; of {", ".join(V.QTYPES)}')
        opts['types'] = list(s['types'])
    if s['tolerance']:
        opts['tolerance'] = s['tolerance']
    tol = {k: tuple(v) for k, v in (s['tolerance'] or {}).items()} or None
    # the plan: every question of every chosen frame
    plan = []
    for L in drives:
        recs = [r for k, r in enumerate(L['records']) if k % max(int(s['every']), 1) == 0]
        if s['max_frames']:
            recs = recs[:int(s['max_frames'])]
        for rec in recs:
            ctx = V.context_of(rec, L['manifest'], L['name'])
            if tol:
                ctx['tolerance'] = dict(V.TOLERANCE, **tol)
            imgs = images_for(L, rec['step'], s['camera'], s['clip'], s['clip_step'])
            for q in V.questions(rec, ctx, opts):
                plan.append((L, rec, ctx, imgs, q))
    if not plan:
        raise EvalError('no questions: the drives have no records at this stride')
    if runner.kind == 'hf' and not all(p[3] for p in plan):
        missing = sorted({p[0]['name'] for p in plan if not p[3]})
        raise EvalError(f'no camera images for {", ".join(missing)} (its run folder or frames are missing): a VLM '
                        'needs them')
    ans_p = os.path.join(out_dir, 'answers.jsonl')
    done = {}
    if prev is not None and os.path.isfile(ans_p):
        with open(ans_p, encoding='utf-8') as fh:
            for x in fh:
                if x.strip():
                    r = json.loads(x)
                    done[r['qid']] = r
    ins = {f'lang:{L["name"]}': os.path.join(L['folder'], 'language.jsonl') for L in drives}
    ins.update({f'lang_manifest:{L["name"]}': os.path.join(L['folder'], 'language.json') for L in drives})
    info = runner.info()
    folder = info.get('folder')
    if folder and os.path.isfile(os.path.join(folder, 'pins.json')):
        ins['model_pins'] = os.path.join(folder, 'pins.json')
    if info.get('adapter'):
        ins['adapter'] = os.path.join(info['adapter'], 'adapter.pt')
    run = PV.new_run('vlm-eval', argv or [], s['seed'], s, ins)
    run.update(settings_key=key, runner=info, schema=SCHEMA)
    PV.write_json(run_p, run)
    t0 = time.time()
    lat = []
    mode = 'a' if done else 'w'
    with open(ans_p, mode, encoding='utf-8') as fh:
        for k, (L, rec, ctx, imgs, q) in enumerate(plan):
            if q['qid'] in done:
                continue
            if cancel():
                run.update(status='cancelled', exit_code=3, finished=PV.utc_now())
                PV.write_json(run_p, run)
                raise Cancelled()
            prompt = f'{q["text"]} {V.PROMPT_SUFFIX}'
            meta = {'drive': L['name'], 'step': rec['step'], 'frame': rec.get('frame'), 'sim_t': rec.get('sim_t'),
                    'question': q, 'context': ctx}
            t1 = time.perf_counter()
            a = runner.answer(imgs, prompt, meta)
            ms = (time.perf_counter() - t1) * 1000.0
            lat.append(ms)
            sc = V.score_answer(q, a, ctx)
            row = {'qid': q['qid'], 'drive': L['name'], 'step': rec['step'], 'frame': rec.get('frame'),
                   'sim_t': rec.get('sim_t'), 'type': q['type'], 'question': q['text'], 'answer': a,
                   'truth': q['truth'], 'target': q['target'], 'template': q['template'], 'images': imgs,
                   'latency_ms': round(ms, 3), 'score': sc, 'note': runner.note(q['qid'])}
            if judge is not None and q['type'] == 'brake_reason':
                v, raw = ask_judge(judge, q, a, ctx)
                row['judge'] = {'faithful': v, 'text': raw}
            fh.write(json.dumps(row, separators=(',', ':'), ensure_ascii=False, default=str) + '\n')
            fh.flush()
            done[q['qid']] = row
            em.progress((k + 1) / len(plan), f'{k + 1}/{len(plan)} answers')
    rows = [done[q[4]['qid']] for q in plan]
    flat = [dict(r['score'], drive=r['drive'], sim_t=r['sim_t']) for r in rows]
    agg = V.aggregate(flat, int(s['n_boot']), int(s['seed']))
    judged = [r for r in rows if r.get('judge')]
    judge_doc = None
    if judge is not None:
        ok = [r for r in judged if r['judge']['faithful'] is not None]
        judge_doc = {'client': s['judge'], 'approved': True, 'judged': len(judged), 'readable': len(ok),
                     'faithful_rate': sum(r['judge']['faithful'] for r in ok) / len(ok) if ok else None,
                     'agreement_with_rules': sum(r['judge']['faithful'] == r['score']['faithful'] for r in ok) /
                     len(ok) if ok else None, 'note': 'a second opinion; the rule-based faithfulness is the score'}
    tab = table(agg)
    scores = {'schema': V.SCHEMA_SCORES, 'runner': info, 'not_a_model': info.get('reads_truth', False),
              'drives': names, 'n_questions': len(rows), 'scores': agg, 'table': tab, 'method': V.method(opts),
              'judge': judge_doc, 'latency_ms': _lat(lat or [r['latency_ms'] for r in rows]),
              'settings': {k: v for k, v in s.items()},
              'provenance': PV.block(ins, s['seed'], {k: v for k, v in s.items()})}
    PV.write_json(os.path.join(out_dir, 'scores.json'), scores)
    summary = {'macro_accuracy': agg['metrics'].get('macro_accuracy', {}).get('value'),
               'hallucination_rate': agg['metrics'].get('hallucination_rate', {}).get('value'),
               'grounding': agg['metrics'].get('grounding', {}).get('value'), 'questions': len(rows),
               'drives': len(names), 'reliable_intervals': agg['reliable'], 'runner': info['name'],
               'licence': info.get('licence')}
    run.update(status='ok', exit_code=0, finished=PV.utc_now(), summary=summary,
               timings={'total_s': time.time() - t0, 'answer_s': sum(lat) / 1000.0, 'answers_this_run': len(lat)},
               outputs=PV.hash_tree(out_dir, skip=('run.json',)))
    PV.write_json(run_p, run)
    runner.close()
    return summary


def _lat(v: list) -> dict:
    v = sorted(float(x) for x in v)
    if not v:
        return {'median': None, 'p90': None, 'n': 0}
    return {'median': v[len(v) // 2], 'p90': v[min(len(v) - 1, int(0.9 * len(v)))], 'n': len(v)}


def table(agg: dict) -> list:
    """Rows {'type', 'metric', 'value', 'ci_low', 'ci_high', 'n'} for the UI and the report: per type its accuracy
    first, then its other numbers; the overall numbers last."""
    from . import vlm as V
    m = agg['metrics']
    rows = []
    for t in V.QTYPES:
        if f'{t}.accuracy' not in m:
            continue
        n = agg['counts'].get(t, {}).get('n')
        for k in ('accuracy', 'grounding', 'median_abs_error', 'precision', 'recall', 'decision', 'parsed'):
            r = m.get(f'{t}.{k}')
            if r is not None and r.get('value') is not None:
                rows.append({'type': t, 'metric': k, 'value': r['value'], 'ci_low': r['ci_low'],
                             'ci_high': r['ci_high'], 'n': n})
    for k in ('macro_accuracy', 'grounding', 'hallucination_rate', 'answers_with_hallucination'):
        r = m.get(k)
        if r is not None and r.get('value') is not None:
            rows.append({'type': 'all', 'metric': k, 'value': r['value'], 'ci_low': r['ci_low'],
                         'ci_high': r['ci_high'], 'n': agg['n_answers']})
    return rows


# ================================================================================================ jobs and folders
def evaluations_dir(ws: str) -> str:
    return os.path.join(ws, 'vlm', 'evaluations')


def list_evaluations(ws: str) -> list:
    d = evaluations_dir(ws)
    if not os.path.isdir(d):
        return []
    return sorted((os.path.join(d, x) for x in os.listdir(d) if os.path.isfile(os.path.join(d, x, 'scores.json'))),
                  reverse=True)


def job_spec(argv: list, title: str = 'VLM evaluation', needs_gpu: bool = False, memory_gb: float = 2.0):
    """The JobSpec of `falcon vlm …` (eval or finetune): an hf runner is a needs_gpu job, and the process knows the
    queue holds the GPU lock (FALCON_GPU_LOCK_HELD)."""
    import sys
    from .jobs import JobSpec
    env = dict(os.environ, FALCON_GPU_LOCK_HELD='1') if needs_gpu else None
    return JobSpec('vlm', title, [sys.executable, '-m', 'falcon_autonomy', 'vlm'] + list(argv), env=env,
                   needs_gpu=needs_gpu, memory_gb=memory_gb, resumable=True)


def needs_gpu(spec: str) -> bool:
    """Whether a runner spec would run torch on CUDA (an hf runner, unless device=cpu, on a machine with CUDA)."""
    if not spec.startswith('hf:') or 'device=cpu' in spec:
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


# ================================================================================================ report
def report(src: str):
    """An evaluation's HTML report (core.reports' page): per type the numbers with their intervals, the method, the
    runner and its licence, and the provenance. It computes nothing new."""
    from . import reports as R
    d = src if os.path.isdir(src) else os.path.dirname(src)
    pg = R.Page('vlm', 'VLM driving explanations', d)
    sc = pg.json(os.path.join(d, 'scores.json'))
    run = pg.json(os.path.join(d, 'run.json')) if os.path.isfile(os.path.join(d, 'run.json')) else {}
    r = sc['runner']
    agg = sc['scores']
    pg.title = f'VLM explanations: {r.get("name")}'
    pg.add(R.kv([('runner', R.esc(f'{r.get("name")} ({r.get("kind")})')), ('licence', R.esc(r.get('licence'))),
                 ('drives', R.esc(', '.join(sc.get('drives') or []))), ('questions', R.esc(sc.get('n_questions'))),
                 ('interval', R.esc(f'95 % bootstrap over {agg["unit"]}s ({agg["n_units"]}), {agg["n_boot"]} '
                                    f'resamples, seed {agg["seed"]}')),
                 ('latency', R.esc(f'median {R.num((sc.get("latency_ms") or {}).get("median"), 2)} ms per answer '
                                   '(this machine)'))]))
    if sc.get('not_a_model'):
        pg.note('NOT A MODEL: this runner reads the ground truth (the oracle or the noisy baseline). It validates the '
                'scorer; its numbers say nothing about any vision-language model.')
    pg.note(agg.get('note'))
    rows = sc['table']
    acc = [x for x in rows if x['metric'] == 'accuracy']
    pg.h2('Accuracy per question type')
    pg.add(R.legend('estimate', None),
           R.interval_chart([{'label': x['type'], 'before': x} for x in acc], title='accuracy',
                            names=('estimate', None)),
           R.table(['type', 'metric', 'value', '95 % interval', 'answers'], [
               [R.esc(x['type']), R.esc(x['metric']), R.num(x['value']),
                f'[{R.num(x["ci_low"])}, {R.num(x["ci_high"])}]', R.esc(x['n'])] for x in rows], numeric=(2, 3, 4)))
    m = sc['method']
    pg.h2('How it is scored')
    pg.add(R.table(['number', 'band'], [[R.esc(k), R.esc(v['rule'])] for k, v in m['tolerance'].items()]))
    pg.add(R.kv([('grounding', R.esc(m['grounding'])), ('hallucination', R.esc(m['hallucination'])),
                 ('explanation', R.esc(m['explanation'])), ('heuristic', R.esc(', '.join(m['heuristic'])))]))
    if sc.get('judge'):
        j = sc['judge']
        pg.note(f'An approved judge ({j["client"]}) also read {j["judged"]} explanations: faithful rate '
                f'{R.num(j["faithful_rate"])}, agreement with the rules {R.num(j["agreement_with_rules"])}.')
    pg.provenance(run.get('provenance') or sc.get('provenance'), 'the evaluation')
    return pg


# ================================================================================================ the command
def _confirm(text: str, yes: bool) -> bool:
    import sys
    print(text, flush=True)
    if yes:
        return True
    if not sys.stdin.isatty():
        print('not confirmed: pass --yes after reading the licence above (no terminal to ask on)', flush=True)
        return False
    return input('Download it? Type yes: ').strip().lower() == 'yes'


def main_vlm(argv) -> int:
    """`falcon vlm list|verify|fetch|eval|finetune …` (docs/vlm.md): protocol lines on stdout; 0 ok, 2 failed,
    3 cancelled, 6 refused (a download not confirmed)."""
    import argparse
    from . import protocol as P
    from .training import CancelFlag, Cancelled, gpu_locks
    from . import vlm_models as VM
    ap = argparse.ArgumentParser(prog='falcon vlm', description='vision-language driving explanations scored against '
                                                                'ground truth (docs/vlm.md)')
    sub = ap.add_subparsers(dest='action', required=True)
    li = sub.add_parser('list', help='the candidate models, their licences, sizes and state; the refused ones')
    li.add_argument('--workspace', default='')
    ve = sub.add_parser('verify', help='check a downloaded candidate against its pins and the publisher\'s checksums')
    ve.add_argument('name')
    ve.add_argument('--workspace', required=True)
    fe = sub.add_parser('fetch', help='download a candidate (licence, sizes and checksums shown; asks first)')
    fe.add_argument('name')
    fe.add_argument('--workspace', required=True)
    fe.add_argument('--yes', action='store_true', help='confirm the download non-interactively')
    ev = sub.add_parser('eval', help='ask a runner the grounded questions and score the answers')
    ev.add_argument('--runner', default='oracle', help='oracle | noisy[:p=0.3,h=0.1,seed=0] | plugin:module:Class | '
                                                       'hf:NAME|DIR[,adapter=DIR,device=cpu]')
    ev.add_argument('--lang', action='append', default=[], help='a language folder (repeatable)')
    ev.add_argument('--run', action='append', default=[], help='a run: its language data are derived first')
    ev.add_argument('--out', '-o', required=True)
    ev.add_argument('--workspace', default='')
    for k, t in (('every', int), ('max-frames', int), ('max-objects', int), ('clip', int), ('clip-step', int),
                 ('boot', int), ('seed', int)):
        ev.add_argument(f'--{k}', type=t)
    ev.add_argument('--types', help='comma-separated question types (default all)')
    ev.add_argument('--camera')
    ev.add_argument('--tolerance', action='append', default=[], metavar='TYPE=ABS,REL',
                    help='a tolerance band: distance, speed or ttc = absolute, relative (default distance=2,0.15 '
                         'speed=1,0.15 ttc=0.5,0.2)')
    ev.add_argument('--judge', help='module:Class of an explanation judge YOU configure (never by default)')
    ev.add_argument('--approve-judge', action='store_true', help='approve sending facts and answers to the judge')
    ft = sub.add_parser('finetune', help='LoRA fine-tuning on the grounded QA (needs downloaded weights)')
    ft.add_argument('--model', required=True, help='a downloaded candidate or a model folder')
    ft.add_argument('--lang', action='append', default=[], required=True)
    ft.add_argument('--out', '-o', required=True)
    ft.add_argument('--workspace', default='')
    for k, t in (('epochs', int), ('lr', float), ('r', int), ('alpha', float), ('accumulate', int), ('every', int),
                 ('seed', int), ('val-share', float), ('val-generate', int), ('max-examples', int)):
        ft.add_argument(f'--{k}', type=t)
    ft.add_argument('--targets', help='comma-separated module names (default q_proj,k_proj,v_proj,o_proj)')
    ft.add_argument('--scope', choices=('text', 'all'))
    ft.add_argument('--device')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    if a.action == 'list':
        for r in VM.list_candidates(a.workspace):
            size = f'{r["bytes"] / 2 ** 30:6.2f} GiB' if r['bytes'] else '        —'
            print(f'{r["name"]:22s} {r["params"] / 1e9:5.2f} B  {size}  {r["licence"]:40s} {r["state"]}')
            print(f'{"":22s} {r["repo"]}' + (f' @ {r["revision"][:12]}' if r['revision'] else ''))
            if r['refused']:
                print(f'{"":22s} refused: {r["refused"]}')
        print('\nWeights are downloaded only by falcon vlm fetch NAME --workspace W, after the licence is shown and '
              'confirmed; each file is checked against the publisher\'s checksum and its SHA-256 pinned on that first '
              'download.')
        return P.OK
    em = P.Emitter()
    em.hello('vlm')
    if a.action == 'verify':
        if a.name not in VM.CANDIDATES:
            print(f'failed: not a candidate: {a.name}', flush=True)
            return P.FAILED
        v = VM.verify(a.name, VM.candidate_dir(a.workspace, a.name))
        for p in v['problems']:
            print(f'  -- {p}', flush=True)
        em.result('ok' if v['ok'] else 'failed', problems=v['problems'])
        return P.OK if v['ok'] else P.FAILED
    if a.action == 'fetch':
        if a.name not in VM.CANDIDATES:
            why = VM.REFUSED[a.name][3] if a.name in VM.REFUSED else f'one of {", ".join(VM.CANDIDATES)}'
            print(f'failed: {a.name} is not a candidate: {why}', flush=True)
            return P.REFUSED if a.name in VM.REFUSED else P.FAILED
        if not _confirm(VM.describe_candidate(a.name), a.yes):
            em.result('refused', error='download not confirmed')
            return P.REFUSED
        try:
            path = VM.fetch(a.name, a.workspace, lambda f: em.progress(f, 'download'))
        except (OSError, VM.VLMError) as e:
            print(f'failed: {e}', flush=True)
            em.result('failed', error=str(e))
            return P.FAILED
        em.result('ok', path=path)
        return P.OK
    flag = CancelFlag().install()
    locks = []
    try:
        if a.action == 'eval':
            s = {'runner': a.runner, 'lang': a.lang, 'runs': a.run, 'workspace': a.workspace,
                 'judge': a.judge, 'judge_approved': a.approve_judge}
            for k, v in (('every', a.every), ('max_frames', a.max_frames), ('max_objects', a.max_objects),
                         ('clip', a.clip), ('clip_step', a.clip_step), ('n_boot', a.boot), ('seed', a.seed),
                         ('camera', a.camera)):
                if v is not None:
                    s[k] = v
            if a.types:
                s['types'] = [t.strip() for t in a.types.split(',') if t.strip()]
            if a.tolerance:
                tol = {}
                for kv in a.tolerance:
                    k, _, v = kv.partition('=')
                    ab, _, rel = v.partition(',')
                    if k not in ('distance', 'speed', 'ttc') or not rel:
                        raise EvalError(f'--tolerance {kv!r}: TYPE=ABS,REL with TYPE distance, speed or ttc')
                    tol[k] = [float(ab), float(rel)]
                s['tolerance'] = tol
            if needs_gpu(a.runner):
                locks = gpu_locks(em, flag, held_by_queue=os.environ.get('FALCON_GPU_LOCK_HELD') == '1')
            res = evaluate(s, a.out, emitter=em, cancel=flag, argv=['vlm'] + list(argv))
            em.result('ok', run_json=os.path.join(a.out, 'run.json'), summary=res)
            return P.OK
        from .vlm_lora import finetune
        s = {'model': a.model, 'lang': [os.path.abspath(x) for x in a.lang], 'workspace': a.workspace}
        for k, v in (('epochs', a.epochs), ('lr', a.lr), ('r', a.r), ('alpha', a.alpha), ('accumulate', a.accumulate),
                     ('every', a.every), ('seed', a.seed), ('val_share', a.val_share), ('val_generate', a.val_generate),
                     ('max_examples', a.max_examples), ('scope', a.scope), ('device', a.device)):
            if v is not None:
                s[k] = v
        if a.targets:
            s['targets'] = [t.strip() for t in a.targets.split(',') if t.strip()]
        if needs_gpu('hf:' + a.model + (f',device={a.device}' if a.device else '')):
            locks = gpu_locks(em, flag, held_by_queue=os.environ.get('FALCON_GPU_LOCK_HELD') == '1')
        res = finetune(s, a.out, emitter=em, cancel=flag, argv=['vlm'] + list(argv))
        em.result('ok', run_json=os.path.join(a.out, 'run.json'), summary=res)
        return P.OK
    except Cancelled:
        em.result('cancelled', run_json=os.path.join(a.out, 'run.json'))
        return P.CANCELLED
    except Exception as e:                 # noqa: BLE001 — every failure is reported as a protocol result
        from .vlm_lora import LoRAError
        if not isinstance(e, (EvalError, LoRAError, VM.VLMError, ValueError, OSError, KeyError, ImportError)):
            raise
        print(f'failed: {e}', flush=True)
        em.result('failed', error=str(e))
        return P.FAILED
    finally:
        for lk in locks:
            lk.release()
