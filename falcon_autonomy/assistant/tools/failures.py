"""Failure tools: the workspace's failure modes (core.failclusters), its triage queue and comments (core.team), and a
regression bisect of a stack's git history through CI mode (core.regbisect).

Privacy: failures whose run the privacy rules call private (anything from the dashcam footage) are left out before
clustering, so they shape no cluster and no name; a triage item linking a private run is listed by id and state only
(its title, links and comments withheld); comments on a private run are refused. What the assistant writes (a
comment, a triage change) carries the user's name and `via: assistant`.
"""
from __future__ import annotations

import os

from ..privacy import private_reason
from ..tool_api import ToolError, tool
from . import _common as C
from .jobs import _cli_preview, _fake, _queue, _stamp


def _public(f: dict) -> bool:
    return not f.get('run') or not private_reason(f['run'])


def _ref_private(ctx, ref: str) -> bool:
    kind, _, rest = str(ref).partition(':')
    if kind != 'run':
        return False
    p = os.path.join(C.workspace(ctx), rest)
    return bool(private_reason(p))


def _cluster_row(ctx, c: dict, members: int = 5) -> dict:
    return {'id': c['id'], 'name': c['name'], 'size': c['size'], 'sources': c['sources'],
            'scenarios': dict(sorted(c['scenarios'].items(), key=lambda kv: -kv[1])[:5]),
            'features': [C.rounded({k: f.get(k) for k in ('text', 'score', 'share_in', 'share_out')
                                    if f.get(k) is not None}, 3) for f in c['features'][:5]],
            'cohesion': c['cohesion'], 'members': c['members'][:members],
            **({'more_members': c['size'] - members} if c['size'] > members else {})}


@tool('research_clusters', 'research', 'read',
      'Failure modes (core.failclusters, docs/failure-clusters.md): every failure of the workspace — rare-case and '
      'sweep studies\' failures, stack suites\' failed runs, CI runs\' failed runs — clustered by a Gower distance '
      'over scenario, parameters, conditions, outcome and root-cause factors (average linkage, cut 0.3), each '
      'cluster named from the features that set it apart, with its size, sources, scenarios and members. With '
      'cluster: that cluster\'s members (run ids, scenario, values, outcome). With ci: which of that CI run\'s '
      'failures fall in a known mode and which are new failure modes. Private runs are left out.',
      C.obj({'cluster': C.s('a cluster id (C-…): its members'),
             'ci': C.s('a CI run id (folder in ci/): its new failure modes')}))
def research_clusters(ctx, cluster='', ci=''):
    from ...core import failclusters as FC
    ws = C.workspace(ctx)
    if ci:
        d = C.resolve_in(ctx, 'ci', ci, 'CI run', ('ci-runs.jsonl',))
        doc = FC.check_ci(ws, d, write=False, keep=_public)
        return C.rounded({'ci': ci, 'failures': doc['failures'], 'reference': doc['reference'],
                          'known': doc['known'][:15], 'new': doc['new'][:15],
                          'new_modes': [_cluster_row(ctx, m) for m in doc['new_modes'][:10]]}, 3)
    fails = FC.collect(ws, keep=_public)
    hidden = len(FC.collect(ws)) - len(fails)
    cl = FC.cluster(fails)
    if cluster:
        c = next((x for x in cl if x['id'] == cluster), None)
        if c is None:
            raise ToolError(f'no cluster {cluster!r} (ids: {", ".join(x["id"] for x in cl[:15]) or "none"})')
        by = {f['key']: f for f in fails}
        rows = []
        for k in c['members'][:25]:
            f = by[k]
            rows.append(C.rounded({'key': k, 'source': f['source'], 'origin': f['origin'], 'scenario': f['scenario'],
                                   'seed': f.get('seed'), 'values': f.get('values'), 'outcome': f.get('outcome'),
                                   'conditions': f.get('conditions'), 'causes': f.get('causes'),
                                   'run': C.run_id(ctx, f['run']) if f.get('run') and C.inside(f['run'], ws)
                                   else None}, 3))
        return {'cluster': _cluster_row(ctx, c, 0), 'members': C.capped(rows, 25)}
    out = {'failures': len(fails), 'clusters': [_cluster_row(ctx, c) for c in cl[:12]], 'total_clusters': len(cl),
           'singletons': sum(1 for c in cl if c['size'] == 1), 'cut': FC.CUT}
    if hidden:
        out['private_left_out'] = hidden
    return out


def _item_row(ctx, it: dict, full: bool = False) -> dict:
    from ...core import team as TM
    if any(_ref_private(ctx, r) for r in it['links']):
        return {'id': it['id'], 'state': it['state'], 'private': 'links a run from the dashcam footage: withheld'}
    row = {'id': it['id'], 'title': it['title'], 'kind': it['kind'], 'state': it['state'],
           'assignee': it['assignee'] or None, 'links': it['links'][:10], 'updated': it['updated'],
           'comments': len(TM.comments(C.workspace(ctx), f'triage:{it["id"]}'))}
    if full:
        row['history'] = [{k: h.get(k) for k in ('time', 'user', 'what', 'ok', 'note', 'via') if h.get(k) is not None}
                          for h in it['history'][-20:]]
        row['comment_texts'] = [{'user': c['user'], 'time': c['time'], 'text': C.clip(c['text'], 400)}
                                for c in TM.comments(C.workspace(ctx), f'triage:{it["id"]}')[-10:]]
        row['created_by'] = it['created_by']
    return row


@tool('research_triage', 'research', 'read',
      'The triage queue (core.team, docs/team.md): failure and issue items with state (open, assigned, '
      'investigating, fixed, verified), assignee, links (runs, scenarios, cases, clusters, CI runs, commits) and '
      'comment counts, newest first; with item, its history and comments; with ref (kind:id, e.g. run:runs/x), the '
      'comments on that object. Items linking private runs are withheld.',
      C.obj({'item': C.s('triage item id (T-3)'), 'state': C.s('only this state',
                                                               enum=['open', 'assigned', 'investigating', 'fixed',
                                                                     'verified']),
             'ref': C.s('kind:id (its comments)')}))
def research_triage(ctx, item='', state='', ref=''):
    from ...core import team as TM
    ws = C.workspace(ctx)
    try:
        if ref:
            ref = TM.check_ref(ref)
            if _ref_private(ctx, ref):
                raise ToolError('private: a run from the dashcam footage — its comments are not read')
            return {'ref': ref, 'comments': C.capped([{'user': c['user'], 'time': c['time'], 'text': C.clip(c['text'],
                                                                                                            400)}
                                                      for c in TM.comments(ws, ref)], 20)}
        if item:
            return _item_row(ctx, TM.get_item(ws, item), full=True)
        rows = [_item_row(ctx, it) for it in TM.items(ws, state)]
    except TM.TeamError as e:
        raise ToolError(str(e)) from e
    return dict(C.capped(rows, 25, 'filter with state'), team=TM.is_team(ws))


def _ident(ctx) -> dict:
    from ...core.team import identity
    st = ctx.settings if ctx.win is None else ctx.win.settings
    return dict(identity(st), via='assistant')


def _triage_preview(ctx, action, item='', ref='', text='', title='', to='', links=(), **_a) -> str:
    who = _ident(ctx)['user']
    if action == 'comment':
        return f'Comment on {ref} as {who} (via the assistant):\n{C.clip(text, 400)}'
    if action == 'add':
        return f'New triage item "{title}" by {who}' + (f', assigned to {to}' if to else '') + \
            (f'\nlinks: {", ".join(links)}' if links else '')
    if action == 'assign':
        return f'{item}: assign to {to or who}'
    if action == 'state':
        return f'{item}: state → {to}' + (f' ({C.clip(text, 200)})' if text else '')
    return f'{item}: add links {", ".join(links)}'


@tool('research_triage_edit', 'research', 'act',
      'Write to the triage queue or comment (core.team; append-only files in the workspace\'s team folder, with the '
      'user\'s name and "via assistant"): comment on an object (ref kind:id: run:runs/<folder>, case:<study>#<row>, '
      'scenario:<id>, cluster:<id>, ci:<name>, triage:T-3), add an item (title, links, optional assignee), assign '
      'one, move its state (assigned → investigating → fixed → verified; back to investigating when a fix does not '
      'hold), or link more objects. Nothing is deleted; a wrong comment is answered by another.',
      C.obj({'action': C.s('what to do', enum=['comment', 'add', 'assign', 'state', 'link']),
             'item': C.s('triage item id (assign, state, link)'), 'ref': C.s('kind:id (comment)'),
             'text': C.s('comment, or a state change\'s note'), 'title': C.s('new item\'s title (add)'),
             'to': C.s('assignee or state'), 'links': C.arr(C.s('kind:id'), 'links', maxItems=20)},
            ['action']),
      preview=_triage_preview)
def research_triage_edit(ctx, action, item='', ref='', text='', title='', to='', links=()):
    from ...core import team as TM
    ws = C.workspace(ctx)
    ident = _ident(ctx)
    for r in list(links) + ([ref] if ref else []):
        if _ref_private(ctx, r):
            raise ToolError('private: a run from the dashcam footage — the assistant does not write about it')
    try:
        if action == 'comment':
            if not ref or not text:
                raise ToolError('comment needs ref and text')
            c = TM.add_comment(ws, ref, text, ident)
            ctx.emit('team', {'comment': c['id'], 'ref': c['object']})
            return {'comment': c['id'], 'ref': c['object'], 'by': c['user'],
                    'undo': 'comments are append-only: answer it with another comment'}
        if action == 'add':
            it = TM.create_item(ws, title, ident, links, 'failure', to, text)
        elif not item:
            raise ToolError(f'{action} needs item')
        elif action == 'assign':
            it = TM.assign(ws, item, to or ident['user'], ident)
        elif action == 'state':
            if not to:
                raise ToolError('state needs to (the new state)')
            it = TM.set_state(ws, item, to, ident, text)
        else:
            it = TM.link(ws, item, links, (), ident)
    except TM.TeamError as e:
        raise ToolError(str(e)) from e
    ctx.emit('team', {'item': it['id'], 'state': it['state']})
    return {'item': it['id'], 'state': it['state'], 'assignee': it['assignee'] or None, 'links': it['links'][:10],
            'undo': 'events are append-only: move it back with another state change'}


def _bisect(ctx, repo, config, good, bad, repeats=0, scenarios=(), name='', dry=False):
    r = os.path.realpath(os.path.expanduser(str(repo)))
    if not os.path.isabs(os.path.expanduser(str(repo))) or not os.path.isdir(os.path.join(r, '.git')):
        raise ToolError('repo must be the absolute path of the stack\'s git repository (a folder with .git)')
    if private_reason(r):
        raise ToolError('private: that folder holds the user\'s own footage')
    if C._bad_name(config):
        raise ToolError('config is the CI config\'s path inside the repository (e.g. falcon-ci.yml)')
    if not os.path.isfile(os.path.join(r, config)):
        raise ToolError(f'no {config} in {r} (at its checked-out commit)')
    nm = name or f'{_stamp()}-{str(good)[:12]}-{str(bad)[:12]}'.replace('/', '_')
    out = os.path.join(C.workspace(ctx), 'bisect', nm) if dry else C.out_dir(ctx, 'bisect', nm)
    argv = ['bisect', 'run', r, '--config', config, '--good', good, '--bad', bad, '--out', out]
    if repeats:
        argv += ['--repeats', str(int(repeats))]
    for s in scenarios:
        argv += ['--scenario', s]
    if _fake(ctx):
        argv += ['--fake-carla', _fake(ctx)]
    return 'bisect', f'Bisect {str(good)[:12]}..{str(bad)[:12]}', argv, {'memory_gb': 0.5}


@tool('research_bisect', 'research', 'act',
      'Regression bisect (`falcon bisect run`, docs/bisect.md): run a stack\'s CI config (falcon.ci/1, read from each '
      'commit) at a good and a bad git ref, find the scenarios that got worse (paired runs when deterministic, else '
      'Fisher\'s exact test over repeats), then bisect the first-parent history between them through CI mode to the '
      'first bad commit of each (unbuildable commits skipped as git bisect skip). Output bisect/<name>; read with '
      'analysis_outputs kind bisect and the id.',
      C.obj({'repo': C.s('absolute path (of the stack\'s git repository)'),
             'config': C.s('CI config path in the repo'), 'good': C.s('older ref'), 'bad': C.s('newer ref'),
             'repeats': C.i('runs per commit (default 5; deterministic: 1)', minimum=1, maximum=20),
             'scenarios': C.arr(C.s('scenario id'), 'only these', maxItems=20),
             'name': C.s('folder in bisect/')}, ['repo', 'config', 'good', 'bad']),
      estimate=lambda ctx, **a: {'seconds': None, 'what': 'two CI runs of the suite at the ends, then about '
                                                          'log2(commits) more per regressed scenario (repeats each '
                                                          'when not deterministic)'},
      preview=_cli_preview(_bisect))
def research_bisect(ctx, **a):
    out = _queue(ctx, _bisect(ctx, **a))
    out['bisect'] = os.path.basename(out['command'].split('--out ', 1)[1].split(' ', 1)[0])
    return out


def bisect_result(ctx, id):
    """A bisect's answer (analysis_outputs kind bisect with an id): the endpoints' test per scenario, the regressed
    scenarios and each one's first bad commit (or the candidates), the skipped commits and the notes."""
    from .analysis import compact
    d = C.resolve_in(ctx, 'bisect', id, 'bisect', ('bisect.json',))
    doc = C.read_json(os.path.join(d, 'bisect.json'), {}) or {}
    brief = (lambda c: {k: c.get(k) for k in ('short', 'subject', 'author', 'date')} if c else None)  # noqa: E731
    return compact({'good': brief(doc.get('good')), 'bad': brief(doc.get('bad')), 'commits': len(doc.get('commits')
                                                                                               or []),
                    'rule': doc.get('rule'), 'repeats': doc.get('repeats'), 'deterministic': doc.get('deterministic'),
                    'endpoints': [{k: r.get(k) for k in ('scenario', 'good', 'bad', 'regressed', 'why')}
                                  for r in doc.get('endpoints') or []],
                    'bisect': {s: {'status': r.get('status'), 'first_bad': brief(r.get('first_bad')),
                                   'candidates': [c.get('short') for c in r.get('candidates') or []],
                                   'tested': r.get('tested')} for s, r in (doc.get('bisect') or {}).items()},
                    'skipped': doc.get('skipped'), 'notes': doc.get('notes')}, cap=15)
