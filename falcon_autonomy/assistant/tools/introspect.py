"""Introspection tools (the formal group): causal probes of driving models — what a token policy attends to, what
changes under an intervention on its tokens, heads or inputs (core.introspect), and closed-loop bias audits of any ego
with reliance ratios and bootstrap intervals (core.biasaudit) — queued as jobs exactly as Models ▸ Introspection
queues them (`falcon introspect attention|intervene|audit`), their outputs in the workspace's introspect/ folder; and
the reader of those outputs. Nothing here sends images to the model: results are numbers and token tables.

A module adds its tools' one-line descriptions to _compact.SHORT itself (MINE below; tools/__init__ explains why)."""
from __future__ import annotations

import json
import os

from ..tool_api import ToolError, tool
from . import _common as C
from . import _compact
from .jobs import _cli_preview, _fake, _queue, _saved, _stamp

MINE = {
    'formal_causal_probe': 'Queue a causal probe of a driving policy: attention, an intervention, or a bias audit.',
    'formal_causal_result': 'A causal probe\'s result: attention by token type, intervention deltas, audit R and '
                            'flags.',
}
assert not set(MINE) & set(_compact.SHORT), set(MINE) & set(_compact.SHORT)
_compact.SHORT.update(MINE)

KINDS = ('attention', 'intervene', 'audit')
ARMS = ('brake_light', 'velocity', 'autorickshaw')
SCENARIOS = ('lead_brake', 'cut_in', 'autorickshaw_cut_in', 'free')


def _policy(ctx, policy: str, black_box_ok: bool) -> str:
    if black_box_ok and policy in ('follower', 'none'):
        return policy
    return C.resolve_in(ctx, 'policies', policy, 'policy', must=('policy.json',))


def _token_policy(ctx, policy: str) -> str:
    p = _policy(ctx, policy, False)
    try:
        with open(os.path.join(p, 'policy.json'), encoding='utf-8') as fh:
            kind = json.load(fh).get('net', {}).get('kind')
    except (OSError, ValueError) as e:
        raise ToolError(f'policies/{policy}: unreadable ({type(e).__name__})') from None
    if kind != 'token':
        raise ToolError(f'policies/{policy} is a {kind} policy: attention and interventions need a token policy '
                        '(falcon policy bc --arch token); an audit works on any ego')
    return p


def _probe(ctx, kind, policy, scenario='lead_brake', frames='', ops=None, arms=None, seeds=6, dry=False):
    root = os.path.join(C.workspace(ctx), 'introspect')
    if kind == 'attention':
        pol = _token_policy(ctx, policy)
        path = C.project(ctx).path if dry else _saved(ctx)
        out = os.path.join(root, f'attention-{_stamp()}')
        argv = ['introspect', 'attention', path or '<the project file>', '--policy', pol, '--scenario', scenario,
                '--out', out]
        title = f'Attention of {policy} ({scenario})'
    elif kind == 'intervene':
        pol = _token_policy(ctx, policy)
        if not frames:
            raise ToolError('intervene needs frames: an attention output id (introspect/attention-…, from '
                            'formal_causal_result or a previous attention probe)')
        src = C.resolve_in(ctx, 'introspect', frames, 'introspect output', must=('frames.npz',))
        if not ops:
            raise ToolError('intervene needs ops: e.g. [{"op": "mask_class", "class": "autorickshaw"}]')
        from ...core.introspect import check_ops
        try:
            check_ops(ops)
        except ValueError as e:
            raise ToolError(str(e)) from None
        out = os.path.join(root, f'intervene-{_stamp()}')
        argv = ['introspect', 'intervene', '--policy', pol, '--frames', os.path.join(src, 'frames.npz'), '--out', out]
        for o in ops:
            argv += ['--op', json.dumps(o, separators=(',', ':'))]
        title = f'Intervene on {policy}: {", ".join(o["op"] for o in ops)}'
    else:
        pol = _policy(ctx, policy, True)
        path = C.project(ctx).path if dry else _saved(ctx)
        a = list(arms or ARMS)
        out = os.path.join(root, f'audit-{_stamp()}')
        argv = ['introspect', 'audit', path or '<the project file>', '--policy', pol, '--out', out, '--arms',
                ','.join(a), '--seeds', str(int(seeds))]
        title = f'Bias audit of {policy} ({", ".join(a)})'
    if _fake(ctx) and kind != 'intervene':
        argv += ['--fake-carla', _fake(ctx)]
    return 'introspect', title, argv, {'memory_gb': 2.0}


def _estimate(ctx, **a):
    kind = a.get('kind')
    if kind == 'audit':
        arms = a.get('arms') or ARMS
        n = sum({'brake_light': 4, 'velocity': 3, 'autorickshaw': 3}[x] for x in arms) * int(a.get('seeds', 6))
        return {'seconds': None, 'what': f'{n} CARLA episodes of up to 8 s (arms × cells × seeds), then the '
                                         'bootstrap'}
    if kind == 'intervene':
        return {'seconds': None, 'what': 'one forward pass per frame and op on the CPU: seconds'}
    return {'seconds': None, 'what': 'one CARLA episode recorded with depth, then attention every 5th frame'}


@tool('formal_causal_probe', 'formal', 'act',
      'A causal probe of a driving model (Models ▸ Introspection), queued as a job: attention — record one episode of '
      'a token policy (core.tokenpolicy) with depth and compute what it attends to per frame (rollout relevance per '
      'token, patch points in 3D, object boxes); intervene — on an attention output\'s frames, mask a token by '
      'object id or class, zero a head, patch activations, flip a feature or shift the speed history, giving '
      'steering and acceleration deltas; audit — closed-loop arms with common random numbers (brake-light shortcut, '
      'velocity extrapolation, autorickshaw token dropout) with reliance R = |cue effect| / |cause effect| and '
      'bootstrap intervals; black-box egos (follower, none, policies without tokens) get the simulation-side arm. '
      'Output introspect/<kind>-<stamp>; read it with formal_causal_result.',
      C.obj({'kind': C.s('attention, intervene or audit', enum=list(KINDS)),
             'policy': C.s('a policy id in policies/ (audit: also follower or none)', minLength=1),
             'scenario': C.s('attention: the scenario to record', enum=list(SCENARIOS)),
             'frames': C.s('intervene: an attention output id (introspect/…)'),
             'ops': C.arr({'type': 'object'}, 'intervene: interventions (op dicts, docs/introspection.md)',
                          maxItems=8),
             'arms': C.arr(C.s('arm', enum=list(ARMS)), 'audit: the arms (default all)', maxItems=3),
             'seeds': C.i('audit: seeds per cell (default 6)', minimum=2, maximum=200)}, ['kind', 'policy']),
      estimate=_estimate, preview=_cli_preview(_probe))
def formal_causal_probe(ctx, **a):
    return dict(_queue(ctx, _probe(ctx, **a)), read='formal_causal_result with the output id when the job is done')


def _brief_audit(doc: dict) -> dict:
    out = {'ego': os.path.basename(str(doc.get('ego'))), 'seeds': len(doc.get('seeds') or ()),
           'flagged': doc.get('flagged') or [], 'arms': {}}
    for arm, v in (doc.get('arms') or {}).items():
        if not v.get('available'):
            out['arms'][arm] = {'available': False, 'why': C.clip(v.get('why', ''), 160)}
            continue
        out['arms'][arm] = {'flagged': v['flagged'], 'outcomes': {
            m: {'R': C.rounded(o['R']), 'R_ci95': [C.rounded(x) for x in o['R_ci95'] or ()],
                'cue': C.rounded(o['cue_effect']['mean']), 'cue_ci95': [C.rounded(x) for x in
                                                                        o['cue_effect']['ci95'] or ()],
                'cause': C.rounded(o['cause_effect']['mean']), 'flagged': o['flagged']}
            for m, o in v['outcomes'].items()}}
    return out


@tool('formal_causal_result', 'formal', 'read',
      'The result of a causal probe (formal_causal_probe; Models ▸ Introspection) by its output id in introspect/: an '
      'audit\'s arms with R per outcome (min TTC, brake onset, collision), the cue and cause effects with 95 % '
      'intervals and which arms flag a shortcut; an intervention\'s mean and largest steering and acceleration '
      'deltas; an attention record\'s relevance by token type and its most attended objects. Without an id, the '
      'outputs there, newest first. Numbers only: no images.',
      C.obj({'id': C.s('an output id in introspect/ (empty: list them)')}))
def formal_causal_result(ctx, id=''):                                        # noqa: A002 (the schema's name)
    root = os.path.join(C.workspace(ctx), 'introspect')
    if not id:
        rows = sorted((d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))), reverse=True) \
            if os.path.isdir(root) else []
        return C.capped([{'id': d, 'kind': d.split('-')[0]} for d in rows])
    d = C.resolve_in(ctx, 'introspect', id, 'introspect output')
    for name, kind in (('audit.json', 'audit'), ('intervene.json', 'intervene'), ('attention.json', 'attention')):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            doc = C.read_json(p, {})
            break
    else:
        return {'id': id, 'state': 'no result yet (the job may still be running: jobs_get)'}
    if kind == 'audit':
        return dict(_brief_audit(doc), id=id, kind=kind)
    if kind == 'intervene':
        return {'id': id, 'kind': kind, 'ops': doc.get('ops'), 'summary': doc.get('summary')}
    frames = doc.get('frames') or []
    by = {t: C.rounded(sum(f['by_type'][t] for f in frames) / max(len(frames), 1))
          for t in ('patch', 'object', 'route', 'ego')}
    top = {}
    for f in frames:
        for o in f.get('objects') or ():
            k = f'{o["class"]} #{o["id"]}'
            top[k] = max(top.get(k, 0.0), float(o.get('weight') or 0.0))
    return {'id': id, 'kind': kind, 'frames': len(frames), 'mean_relevance_by_type': by,
            'most_attended_objects': [{'object': k, 'max_relevance': C.rounded(v)} for k, v in
                                      sorted(top.items(), key=lambda kv: -kv[1])[:5]],
            'note': 'attention is not explanation: use intervene or audit for what the policy relies on'}
