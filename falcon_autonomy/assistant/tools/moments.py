"""4.1 assistant tools: moment search (runs_moments_*) and overnight campaigns (research_campaign_*). Registered at
import (see tools/__init__).

    runs_moments_search     read  moments by words and/or a structured filter (core.moments; local, no model)
    runs_moments_stats      read  how often each moment type occurs per hour of indexed driving
    research_campaign_new   act   draft a campaign, show its whole scope and budget on ONE approval card (in every
                                  permission mode), then queue its runner (`falcon campaign run`)
    research_campaign_status read budgets used, studies, the last decisions
    research_campaign_brief read  the brief's text

Privacy. Moments of runs derived from the user's dashcam footage (a route replay, a near-miss ranking, anything
assistant.privacy.private_reason names) are in the index on this machine but are never returned here: the tools read
with public_only and report only how many were left out. A campaign's plan, status and brief are built from public
facts only (core.campaign). The query's words are parsed locally (core.moments.parse_query); nothing is sent anywhere.

Approval. research_campaign_new asks on a card in every mode, Autopilot included, with no "approve all of this
kind": the card is the campaign's one approval and shows everything it covers (core.campaign.approval_text). The
approval the agent hands the tool (ToolContext.approval) is stored in the campaign and is what core.campaign and the
research autopilot check; without one the tool refuses. Nothing here can widen a scope: the budgets and kinds are
those on the card, and the accountant in core.campaign enforces them whatever a model says later.

A module adds its tools' one-line descriptions to _compact.SHORT itself (MINE below; tools/__init__ explains why)."""
from __future__ import annotations

from ..tool_api import ToolError, tool
from . import _common as C
from . import _compact

MINE = {
    'runs_moments_search': 'Find recorded moments (cut-in, near-miss, crossing…) by plain words or a filter; local.',
    'runs_moments_stats': 'How often each moment type occurs per hour of indexed driving (an exposure table).',
    'research_campaign_new': 'Start an overnight campaign: a goal, budgets and study kinds on ONE approval card.',
    'research_campaign_status': 'A campaign\'s state: budgets used, studies and results, needs-you items, decisions.',
    'research_campaign_brief': 'A campaign\'s one-page brief: answer, confidence, what was tried, next step.',
}
assert not set(MINE) & set(_compact.SHORT), set(MINE) & set(_compact.SHORT)
_compact.SHORT.update(MINE)

ROW = ('id', 'type', 't0', 't1', 'actors', 'side', 'min_ttc_s', 'gap_m', 'rel_speed_kmh', 'ego_speed_kmh',
       'decel_mps2', 'severity', 'road', 'tod', 'weather', 'world', 'source')


def _M():
    from ...core import moments as M
    return M


def _K():
    from ...core import campaign as K
    return K


def _filter(M, query, types, actors, side, tod, weather, road, where):
    pq = M.parse_query(query or '')
    try:
        f = M.merge_filters(pq['filter'], M.check_filter(M.new_filter(
            types=list(types or ()), actors=list(actors or ()), side=side or None, tod=tod or None,
            weather=weather or None, road=road or None, where=[M.parse_where(w) for w in where or ()])))
    except M.MomentsError as e:
        raise ToolError(str(e)) from None
    return pq, f


@tool('runs_moments_search', 'runs', 'read',
      'Search the workspace\'s moment index (core.moments, docs/moments.md): events found in every recorded run from '
      'its tracks — cut_in, cut_out, hard_brake_ego, hard_brake_lead, close_pass, overtake, crossing, jaywalking, '
      'oncoming_in_lane, stopped_obstacle, red_light_violation, low_ttc, high_decel_needed, near_miss, collision, '
      'lane_departure, sensor_fault, mrm. query is plain words parsed by a fixed vocabulary on this machine ("two-'
      'wheeler cuts in from the left at night in rain, TTC under 1.5 s"); the structured arguments add to it. Returns '
      'understood (the filter as parsed — show it to the user), not_understood (words outside the vocabulary), and '
      'hits ranked by match then severity: run (use it in the run tools), time span, actor classes, side, min TTC, '
      'gap, speeds, context; match < 1 marks a partial hit with what it misses. Types and numeric constraints are '
      'required; the rest is scored. Moments of runs from the user\'s dashcam footage are counted, never returned. '
      'An empty index says so: the user runs `falcon moments index` (or the Moments tab\'s Index).',
      C.obj({'query': C.s('plain words (may be empty when a filter is given)', maxLength=400),
             'types': C.arr(C.s('a moment type'), 'moment types (any of them)', maxItems=8),
             'actors': C.arr(C.s('a class or group'), 'car, truck, bus, van, motorcycle, bicycle, autorickshaw, '
                             'cycle_rickshaw, pedestrian, child, cattle, handcart, vendor, or two_wheeler, vru, animal',
                             maxItems=8),
             'side': C.s('the side it came from or was on', enum=['left', 'right']),
             'tod': C.s('light', enum=['day', 'low sun', 'twilight', 'night']),
             'weather': C.s('weather', enum=['clear', 'cloudy', 'wet', 'rain']),
             'road': C.s('road context', enum=['road', 'junction', 'speed_breaker']),
             'where': C.arr(C.s('FIELD<N'), 'numeric constraints such as ttc<1.5, gap<10, speed>40 (km/h), decel>=4, '
                            'severity>0.5, duration>2', maxItems=6),
             'strict': C.b('only hits matching every term'),
             'limit': C.i('rows (default 15, at most 40)', minimum=1, maximum=40)}))
def runs_moments_search(ctx, query='', types=None, actors=None, side='', tod='', weather='', road='', where=None,
                        strict=False, limit=15):
    import os
    M = _M()
    ws = C.workspace(ctx)
    pq, f = _filter(M, query, types, actors, side, tod, weather, road, where)
    if not os.path.isfile(M.db_path(ws)):
        return {'understood': M.understood(f), 'not_understood': pq['not_understood'], 'items': [], 'total': 0,
                'note': 'no moment index in this workspace yet: `falcon moments index` (or the Replay page ▸ Moments '
                        '▸ Index) builds it'}
    res = M.search(ws, f, limit=0, strict=bool(strict), public_only=True)
    rows = []
    for m in res['hits']:
        r = {k: C.rounded(m.get(k), 3) for k in ROW if m.get(k) not in (None, [], '')}
        r['run'] = m['rel']
        if m['match'] < 1.0:
            r['match'], r['missed'] = m['match'], m['missed']
        rows.append(r)
    out = C.capped(rows, int(limit), 'add terms or raise limit (≤ 40)')
    out.update(understood=res['understood'], not_understood=pq['not_understood'], full_matches=res['full'])
    if res['private_left_out']:
        out['private_left_out'] = res['private_left_out']
    return out


@tool('runs_moments_stats', 'runs', 'read',
      'The exposure table of the workspace\'s moment index: per moment type its count, the rate per hour of indexed '
      'driving with an approximate 95 % Poisson interval, and per 100 km; hours, km and runs behind it. by splits it '
      '(tod, weather, world, source). Rates describe this workspace\'s recorded driving — mostly simulated scenarios '
      'chosen to be eventful — not real traffic. Private (dashcam-derived) sources are left out and counted.',
      C.obj({'by': C.s('split the table', enum=['tod', 'weather', 'world', 'source'])}))
def runs_moments_stats(ctx, by=''):
    M = _M()
    try:
        st = M.stats(C.workspace(ctx), public_only=True, by=by or None)
    except M.MomentsError as e:
        raise ToolError(str(e)) from None
    rows = [r for r in st['rows'] if r['count']]
    out = C.capped(C.rounded(rows, 3), 40)
    out.update(hours=round(st['hours'], 4), km=round(st['km'], 2), runs=st['runs'], note=st['note'])
    if st['private_left_out']:
        out['private_left_out'] = st['private_left_out']
    return out


# ================================================================================================ campaigns
_BUDGET = {'gpu_hours': C.n('GPU-hours the campaign may use (default 2)', minimum=0, maximum=200),
           'until': C.s('the deadline: 07:00, +6h, +90m or a UTC time YYYY-MM-DDTHH:MM:SSZ', maxLength=24),
           'max_jobs': C.i('at most this many jobs (default 8)', minimum=1, maximum=100),
           'disk_gb': C.n('disk its studies may write, GB (default 20)', exclusiveMinimum=0, maximum=2000)}


def _proj(ctx):
    try:
        return C.project(ctx)
    except ToolError:
        return None


def _draft(ctx, goal, gpu_hours=None, until='', max_jobs=None, disk_gb=None, kinds=None, name='', dry=False):
    """The record core.campaign.new() makes — written (dry False), or only built for the card's preview."""
    K = _K()
    from ..agent import compose
    _text, why = compose(goal)
    if why:
        raise ToolError(why)
    ws = C.workspace(ctx)
    proj = _proj(ctx)
    path = getattr(proj, 'path', '') or ''
    budget = {'gpu_hours': gpu_hours, 'max_jobs': max_jobs, 'disk_gb': disk_gb}
    try:
        budget['deadline'] = K.parse_until(until) if until else None
        return K.new(ws, goal, budget, kinds or K.DEFAULT_KINDS, name, path, proj, write=not dry)
    except K.CampaignError as e:
        raise ToolError(str(e)) from None


def _campaign_gate(ctx, **_a):
    return {'ask': True, 'allow_all': False,
            'why': 'this is the campaign\'s one approval: everything it may run unattended is on this card'}


def _campaign_preview(ctx, goal, gpu_hours=None, until='', max_jobs=None, disk_gb=None, kinds=None, name='', **_a):
    return _K().approval_text(_draft(ctx, goal, gpu_hours, until, max_jobs, disk_gb, kinds, name, dry=True))


@tool('research_campaign_new', 'research', 'act',
      'Start an overnight campaign (core.campaign, docs/campaign.md): the user\'s goal ("find why night pedestrian '
      'recall dropped") with budgets — GPU-hours, a deadline, a job count, disk — and the study kinds it may run '
      '(clusters, moments, rootcause, ablation by default; also failrate, search). Falcon '
      'plans the studies by rule from the workspace, shows the whole scope, the plan and its estimated costs on ONE '
      'approval card (asked in every mode), and after approval queues the runner job, which works unattended inside '
      'that scope: a hard accountant refuses any job past the budgets and cancels at the deadline; cloud spend, model '
      'downloads, trash and training are never done and are listed as "needs you". Returns the campaign id, its plan '
      'and the runner\'s job; follow with research_campaign_status, and read research_campaign_brief when it is done. '
      'Do not describe results before the brief exists.',
      C.obj(dict({'goal': C.s('what should be found out, in the user\'s words', minLength=8, maxLength=600),
                  'kinds': C.arr(C.s('a study kind'), 'allowed study kinds (default clusters, moments, rootcause, '
                                 'ablation)', maxItems=9),
                  'name': C.s('the campaign\'s id (default: made from the goal)',
                              pattern=r'^[A-Za-z0-9][\w.-]{0,47}$')},
                 **_BUDGET), ['goal']),
      approval=_campaign_gate, preview=_campaign_preview)
def research_campaign_new(ctx, goal, gpu_hours=None, until='', max_jobs=None, disk_gb=None, kinds=None, name=''):
    K = _K()
    ap = dict(getattr(ctx, 'approval', None) or {})
    from ...core.autoresearch import APPROVED_HOW
    if ap.get('how') not in APPROVED_HOW:
        raise ToolError('a campaign starts only after the user approves its card (its scope and budget): nothing was '
                        'created')
    rec = _draft(ctx, goal, gpu_hours, until, max_jobs, disk_gb, kinds, name)
    ws = C.workspace(ctx)
    try:
        K.approve(ws, rec['id'], ap)
    except K.CampaignError as e:
        raise ToolError(str(e)) from None
    from .jobs import _fake
    spec = K.runner_job_spec(ws, rec['id'], _fake(ctx))
    job = C.cli_job(ctx, spec.kind, spec.title, spec.argv[3:], spec=spec)
    return {'campaign': rec['id'], 'state': 'approved', 'budget': rec['budget'], 'kinds': rec['scope']['kinds'],
            'plan': [{'study': s['id'], 'kind': s['kind'], 'title': C.clip(s['title'], 120),
                      'estimate_s': s['est']['seconds']} for s in rec['studies']][:12],
            'needs_you': [C.clip(n['what'], 120) for n in rec['needs_you']][:8], 'job': job['job'],
            'next': f'research_campaign_status with campaign {rec["id"]}; the brief when its state is done or stopped'}


_ID = C.obj({'campaign': C.s('the campaign id (omit: the newest)', maxLength=48)})


def _load(ctx, campaign=''):
    K = _K()
    ws = C.workspace(ctx)
    try:
        if not campaign:
            recs = sorted(K.campaigns(ws), key=lambda r: r.get('created') or '')
            if not recs:
                raise ToolError('no campaign in this workspace yet (research_campaign_new starts one)')
            campaign = recs[-1]['id']
        return ws, campaign, K.status(ws, campaign)
    except K.CampaignError as e:
        raise ToolError(str(e)) from None


@tool('research_campaign_status', 'research', 'read',
      'A campaign at a glance: its state (draft, approved, running, stopped, done), the budgets given, used and left '
      '(GPU-hours, jobs, disk, time to the deadline), each study with its state, estimate and result, the hypotheses, '
      'the "needs you" items and the last decisions of its log with their reasons. Reads only.', _ID)
def research_campaign_status(ctx, campaign=''):
    _ws, _cid, st = _load(ctx, campaign)
    st = dict(st)
    st.pop('brief', None)
    st['studies'] = [dict(s, title=C.clip(s['title'], 120), result=C.clip(s['result'], 300) if s.get('result')
                          else None) for s in st['studies']][:20]
    st['hypotheses'] = [dict(h, text=C.clip(h['text'], 200)) for h in st['hypotheses']][:20]
    st['log'] = [{'time': r.get('time'), 'decision': r.get('decision'), 'reason': C.clip(r.get('reason'), 240),
                  'study': r.get('study')} for r in st['log']]
    return C.rounded(st, 4)


@tool('research_campaign_brief', 'research', 'read',
      'A campaign\'s brief (Markdown): the goal, the answer in two sentences with its confidence in plain words, what '
      'was tried (study, cost, result), what was ruled out, the evidence, budget used against given, the "needs you" '
      'items and the recommended next step. final is false while the campaign still runs (the brief so far). Quote '
      'its numbers as they are; its conclusions are only as good as the studies that ran.', _ID)
def research_campaign_brief(ctx, campaign=''):
    ws, cid, st = _load(ctx, campaign)
    K = _K()
    md = K.brief_markdown(ws, K.load(ws, cid))
    return {'campaign': cid, 'state': st['state'], 'final': st['state'] in ('done', 'stopped'),
            'markdown': C.clip(md, 6000)}
