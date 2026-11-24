"""The overnight campaign (core.campaign) in its deterministic mode and with a fake provider: the plan and the one
approval, the hard budget accountant, the loop with its decision log, resumption after a restart, Stop, the scope and
the "needs you" items, the brief, and the command line with real (CPU-only) jobs. No model, no network, no CARLA."""
import json
import os

import pytest

from falcon_autonomy.core import ablation as AB
from falcon_autonomy.core import autoresearch as R
from falcon_autonomy.core import campaign as C
from falcon_autonomy.core import moments as M
from test_moments import NIGHT_RAIN, actor, write_run
from test_research_autopilot import spec

GOAL = 'Find why night pedestrian recall dropped'
T0 = 1_800_000_000.0


class Em:
    def log(self, *a, **k):
        pass

    progress = metric = log


class FakeIO:
    """A job queue with a clock of its own: a job takes durations[kind] seconds and ends as outcomes[kind] (done by
    default), doing what the real command would have written (effects)."""

    def __init__(self, ws, durations=None, outcomes=None, t=T0):
        self.ws, self.t = ws, t
        self.durations = dict({'clusters': 20.0, 'moments': 40.0, 'rootcause': 60.0, 'ablation': 1800.0},
                              **(durations or {}))
        self.outcomes = outcomes or {}
        self.jobs, self.sent, self.submitted, self.cancelled = {}, [], [], []

    def now(self):
        return self.t

    def advance(self, dt):
        self.t += dt

    def submit(self, cid, study):
        jid = len(self.jobs) + 1
        self.jobs[jid] = {'study': dict(study), 'start': self.t, 'state': 'running', 'end': None}
        self.submitted.append(study['id'])
        return {'id': jid, 'submitted': self.t}

    def _effect(self, st):
        if st['kind'] == 'moments':
            M.index(self.ws, xodr='')
        elif st['kind'] == 'rootcause':
            os.makedirs(st['out'], exist_ok=True)
            with open(os.path.join(st['out'], 'rootcause.json'), 'w') as fh:
                json.dump({'summary': 'misses go with lower brightness (Spearman ρ -0.41, 95 % [-0.55, -0.22])',
                           'miss_rate': 0.31, 'n_objects': 420, 'note': None,
                           'attributes': [{'attribute': 'brightness', 'clear': True, 'spearman': -0.41},
                                          {'attribute': 'range', 'clear': False, 'spearman': 0.05}]}, fh)
        elif st['kind'] == 'ablation':
            AB.run(self.ws, st['argv'][3], emitter=Em(), cancel=lambda: False)

    def state(self, ref):
        j = self.jobs.get((ref or {}).get('id'))
        if j is None or abs(j['start'] - ref['submitted']) > 1e-3:
            return None, 0.0
        kind = j['study']['kind']
        if j['state'] == 'running' and self.t >= j['start'] + self.durations[kind]:
            j['state'], j['end'] = self.outcomes.get(kind, 'done'), j['start'] + self.durations[kind]
            if j['state'] == 'done':
                self._effect(j['study'])
        return j['state'], (j['end'] or self.t) - j['start']

    def cancel(self, ref):
        j = self.jobs.get((ref or {}).get('id'))
        if j and j['state'] == 'running':
            j['state'], j['end'] = 'cancelled', self.t
            self.cancelled.append(j['study']['id'])

    def cancel_all(self):
        for jid in list(self.jobs):
            self.cancel({'id': jid})

    def notify(self, title, body):
        self.sent.append((title, body))

    def running(self):
        return [j for j in self.jobs.values() if j['state'] == 'running']


def suggestion(k=1, about='rain at night', add=0.10):
    return {'id': f'cell:c{k}', 'kind': 'odd_cell', 'about': about,
            'question': f'Is perception worse in the operating-domain cell {about}?',
            'facts': {'cell': f'c{k}', 'failures': 4}, 'note': '',
            'template': spec(add=add, noise=0.02, sd=0.04)}


MODE = {'id': 'cluster:m1', 'kind': 'failure_mode', 'about': 'night, pedestrian crossing',
        'question': 'Does perception degrade in the conditions of the failure mode?', 'template': None, 'note': '',
        'facts': {'cluster': 'm1', 'size': 5, 'scenarios': ['ped'], 'features': ['light night', 'actor pedestrian']}}


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    w = str(tmp_path / 'ws')
    os.makedirs(os.path.join(w, 'evaluations', 'ev1'))
    with open(os.path.join(w, 'evaluations', 'ev1', 'evaluation.json'), 'w') as fh:
        json.dump({'model': 'm'}, fh)
    write_run(w, 'ped-night', 8.0, [actor('pedestrian', 60.0, lambda t: -6 + 1.5 * (t - 0.5), yaw=90.0,
                                          size=(0.5, 0.5))], weather=NIGHT_RAIN)
    monkeypatch.setattr(R, 'suggestions', lambda workspace, project=None, limit=8: [MODE, suggestion()])
    return w


def new(ws, **kw):
    kw.setdefault('budget', {'gpu_hours': 2.0, 'max_jobs': 8, 'disk_gb': 5.0, 'deadline': C.utc(T0 + 8 * 3600)})
    return C.new(ws, GOAL, cid='night', now=T0, **kw)


def decisions(ws, cid='night'):
    return [r['decision'] for r in C.read_log(ws, cid)]


# ------------------------------------------------------------------------------------------------ plan and approval
def test_a_draft_shows_its_plan_and_nothing_runs_without_the_one_approval(ws):
    rec = new(ws)
    assert [(s['id'], s['kind'], s['state']) for s in rec['studies']] == [
        ('s1', 'clusters', 'planned'), ('s2', 'moments', 'planned'), ('s3', 'rootcause', 'planned'),
        ('s4', 'ablation', 'planned')]
    assert [h['prior'] for h in rec['hypotheses']] == [0.9, 0.8, 0.7, 0.65]          # the goal's conditions: +0.15
    assert rec['approval'] is None and C.state_of(rec) == 'draft' and decisions(ws) == ['new', 'plan']
    assert all('not measured' in s['est']['source'] or 'simulated' in s['est']['source'] for s in rec['studies'])
    text = C.approval_text(rec)
    for part in ('Find why night pedestrian recall dropped', '2 GPU-hours', 'at most 8 jobs', '5 GB of disk',
                 'clusters, moments, rootcause, ablation', 's4 [ablation]', 'Never done unattended', 'cloud',
                 'download', 'trash', 'rule-based, no model'):
        assert part in text, part
    io = FakeIO(ws)
    with pytest.raises(C.CampaignError, match='not approved'):
        C.step(ws, 'night', io)
    for bad in ({}, {'how': 'model'}, None):
        with pytest.raises(C.CampaignError, match='approves'):
            C.approve(ws, 'night', bad)
    assert io.jobs == {} and not os.path.exists(os.path.join(ws, 'experiments'))
    rec = C.approve(ws, 'night', {'how': 'approved', 'by': 'user'})
    assert rec['approval']['how'] == 'approved' and rec['scope_sha256'] == C.scope_hash(rec)
    with pytest.raises(C.CampaignError, match='approved already'):
        C.approve(ws, 'night', {'how': 'approved'})
    # the approved scope cannot be widened afterwards
    rec['budget']['gpu_hours'] = 99.0
    C.save(ws, rec)
    with pytest.raises(C.CampaignError, match='changed after it was approved'):
        C.step(ws, 'night', io)
    assert io.jobs == {}


def test_bad_drafts_are_refused(ws):
    for kw, why in (({'kinds': ['cloud']}, 'cloud spend'), ({'kinds': []}, 'study kinds'),
                    ({'budget': {'gpu_hours': -1}}, 'gpu_hours'), ({'budget': {'money': 3}}, 'unknown budget'),
                    ({'budget': {'deadline': C.utc(T0 - 60)}}, 'in the past'), ({'planner': 'oracle'}, 'planner')):
        with pytest.raises(C.CampaignError, match=why):
            C.new(ws, GOAL, cid='x', now=T0, **kw)
    with pytest.raises(C.CampaignError, match='goal'):
        C.new(ws, 'why?', cid='x')
    new(ws)
    with pytest.raises(C.CampaignError, match='exists already'):
        new(ws)
    with pytest.raises(C.CampaignError):
        C.load(ws, '../etc')
    assert C.parse_until('+90m', T0) == C.utc(T0 + 5400) and C.parse_until('+6h', T0) == C.utc(T0 + 21600)
    assert T0 < C.epoch(C.parse_until('07:00', T0)) <= T0 + 86400
    assert C.parse_until('2027-01-15T07:00:00Z') == '2027-01-15T07:00:00Z'
    with pytest.raises(C.CampaignError):
        C.parse_until('tomorrow')


# ------------------------------------------------------------------------------------------------ the loop
def test_the_whole_loop_ends_with_an_answer_a_log_and_a_brief(ws):
    new(ws)
    C.approve(ws, 'night', {'how': 'approved'})
    io = FakeIO(ws)
    st = C.run(ws, 'night', io, poll_s=30.0, sleep=io.advance)
    assert io.submitted == ['s1', 's2', 's3', 's4'] and io.running() == []
    assert st['state'] == 'done' and [s['state'] for s in st['studies']] == ['done'] * 4
    assert [s['verdict'] for s in st['studies']] == ['evidence', 'evidence', 'evidence', 'supported']
    assert 'night, pedestrian crossing' in st['studies'][0]['result'] and '5 failures' in st['studies'][0]['result']
    assert st['studies'][1]['result'].startswith('1 recorded moment(s) match "actor pedestrian · light night"')
    assert 'lower brightness' in st['studies'][2]['result'] and 'not a cause' in st['studies'][2]['result']
    assert st['answer']['confidence'] == 'high' and st['answer']['study'] == 's4'
    assert st['stopped']['why'].startswith('the goal is answered')
    assert st['used'] == {'gpu_hours': 0.5, 'jobs': 4, 'disk_gb': pytest.approx(0.0, abs=0.01)}
    assert st['left']['gpu_hours'] == 1.5 and st['left']['jobs'] == 4
    # every decision is in the append-only log with its reason
    rows = C.read_log(ws, 'night')
    assert [r['decision'] for r in rows] == ['new', 'plan', 'approved', 'start', 'run', 'result', 'run', 'result',
                                             'run', 'result', 'run', 'result', 'stop']
    assert all(r['reason'] and r['time'] for r in rows)
    assert 'within budget' in rows[4]['reason'] and 'no GPU' in rows[4]['reason'] and rows[4]['study'] == 's1'
    # the research autopilot ran the ablation under the campaign's approval: its own record says so
    study = R.load(ws, 'night-a1')
    assert study['rounds'][0]['approval']['how'] == 'approved' and study['source']['kind'] == 'campaign'
    # the brief: one page, Markdown and HTML, and a notification
    md = open(os.path.join(ws, 'campaigns', 'night', 'brief.md')).read()
    for part in ('# Morning brief', '**Goal.** Find why night pedestrian recall dropped', '**Confidence: high**',
                 '## What was tried', '| s4 Ablation: rain at night | ablation | 30 min GPU |', '## Ruled out',
                 '## Evidence', '## Budget', '| GPU-hours | 0.50 | 2 |', '## Needs you', '- Nothing.',
                 '## Recommended next step', 'only as', 'campaigns/night/log.jsonl'):
        assert part in md, part
    html = open(os.path.join(ws, 'campaigns', 'night', 'brief.html')).read()
    assert '<h1>Morning brief' in html and '<table>' in html and '<strong>Confidence: high</strong>' in html
    assert len(io.sent) == 1 and 'finished' in io.sent[0][0] and 'high' in io.sent[0][1]
    assert C.step(ws, 'night', io) == 'done' and len(io.sent) == 1          # finished stays finished
    assert C.brief(ws, 'night')['final'] is True


def test_without_a_decisive_study_the_brief_says_how_little_it_knows(ws, monkeypatch):
    monkeypatch.setattr(R, 'suggestions', lambda workspace, project=None, limit=8: [])
    new(ws, kinds=['clusters', 'moments'])
    C.approve(ws, 'night', {'how': 'command line'})
    io = FakeIO(ws)
    st = C.run(ws, 'night', io, sleep=io.advance, poll_s=30.0)
    assert [s['verdict'] for s in st['studies']] == ['inconclusive', 'evidence']
    assert st['answer']['confidence'] == 'low' and 'not an established cause' in st['answer']['second']
    assert st['stopped']['why'].startswith('nothing is left that is worth its cost')
    md = C.brief(ws, 'night')['markdown']
    assert '**Confidence: low** — only an association' in md
    assert 'Decide the first "needs you" item: Root cause of evaluation ev1' in md       # it was outside the scope
    # and with nothing at all
    C.new(ws, 'Find out whether the lidar mount vibrates', cid='none', kinds=['clusters'], now=T0)
    C.approve(ws, 'none', {'how': 'approved'})
    st = C.run(ws, 'none', io, sleep=io.advance, poll_s=30.0)
    assert st['answer']['confidence'] == 'none' and 'did not find an answer' in st['answer']['text']


# ------------------------------------------------------------------------------------------------ the accountant
def test_the_accountant_refuses_by_numbers_alone():
    hist = [{'kind': 'campaign-ablation', 'status': 'done', 'elapsed_s': s} for s in (3000.0, 3600.0, 4200.0)] + \
           [{'kind': 'campaign-ablation', 'status': 'failed', 'elapsed_s': 5.0}, {'kind': 'experiment', 'status': 'done',
                                                                                  'elapsed_s': 3600.0}]
    free = [50.0]
    acc = C.Accountant({'gpu_hours': 2.5, 'max_jobs': 3, 'disk_gb': 1.0, 'deadline': C.utc(T0 + 7200)}, hist,
                       free_gb=lambda: free[0], size_gb=lambda p: 0.4)
    e = acc.estimate('ablation')
    assert e == {'seconds': 3600.0, 'gpu_seconds': 3600.0, 'n': 4,
                 'source': 'measured: the median of 4 finished job(s) of this kind in the job history'}
    assert acc.estimate('clusters')['gpu_seconds'] == 0.0 and 'not measured' in acc.estimate('clusters')['source']
    assert acc.estimate('failrate', 600.0)['seconds'] == 600.0
    st = lambda **kw: dict({'kind': 'ablation', 'state': 'planned', 'est': e}, **kw)       # noqa: E731
    rec = {'studies': []}
    assert acc.may_start(rec, st(), T0) == (True, acc.may_start(rec, st(), T0)[1])
    assert 'within budget: about 60 min, 1.00 GPU-hours (measured' in acc.may_start(rec, st(), T0)[1]
    # the deadline: it would end after it
    ok, why = acc.may_start(rec, st(), T0 + 3700)
    assert not ok and 'would end after the deadline' in why and '58 min left' in why
    # GPU-hours: 1.8 used + 1.0 > 2.5
    rec = {'studies': [st(state='done', launches=1, elapsed_s=6480.0, out='x')]}
    ok, why = acc.may_start(rec, st(), T0)
    assert not ok and '1.00 GPU-hours would exceed what is left (0.70 of 2.5)' in why
    assert acc.may_start(rec, {'kind': 'clusters', 'est': acc.estimate('clusters')}, T0)[0]      # no GPU: still fine
    # the job limit, the disk budget, the free space
    rec = {'studies': [st(state='done', launches=2, elapsed_s=10.0), st(state='done', launches=1, elapsed_s=10.0)]}
    assert acc.may_start(rec, st(), T0) == (False, 'the job limit is reached (3 of 3)')
    rec = {'studies': [st(state='done', launches=1, elapsed_s=10.0, out='a'), st(state='done', launches=1, out='b',
                                                                                 elapsed_s=10.0)]}
    assert acc.used(rec)['disk_gb'] == 0.8
    acc.size_gb = lambda p: 0.6
    assert acc.may_start(rec, st(), T0) == (False, 'the disk budget is used up (1.20 of 1 GB)')
    acc.size_gb = lambda p: 0.0
    free[0] = 0.5
    assert acc.may_start(rec, st(), T0)[1] == 'less than 1 GB is free on the workspace\'s disk'
    # while a job runs: cancelled at the deadline, or when the GPU-hours are used up
    free[0] = 50.0
    run = st(state='running', launches=1, started_t=T0)
    rec = {'studies': [run]}
    assert acc.must_cancel(rec, run, T0 + 3600) is None
    assert 'deadline' in acc.must_cancel(rec, run, T0 + 7200)
    acc.b['deadline'] = None
    assert acc.must_cancel(rec, run, T0 + 9000) is None
    assert acc.must_cancel(rec, run, T0 + 9100) == 'the GPU-hours are used up (2.53 of 2.5)'


def test_budget_out_and_the_deadline_stop_it_and_leave_no_job_running(ws, monkeypatch, tmp_path):
    monkeypatch.setattr(R, 'suggestions', lambda workspace, project=None, limit=8: [
        suggestion(1, add=0.0), suggestion(2, 'fog at dusk', add=0.0)])
    os.makedirs(os.path.join(ws, 'jobs'))
    with open(os.path.join(ws, 'jobs', 'jobs.jsonl'), 'w') as fh:           # measured: an ablation takes an hour here
        fh.write(json.dumps({'schema': 'falcon.job-history/1', 'kind': 'campaign-ablation', 'status': 'done',
                             'elapsed_s': 3600.0}) + '\n')
    rec = new(ws, kinds=['ablation'], budget={'gpu_hours': 1.5, 'max_jobs': 8, 'disk_gb': 5.0})
    assert [s['est']['gpu_seconds'] for s in rec['studies']] == [3600.0, 3600.0]
    assert 'measured' in C.approval_text(rec)
    C.approve(ws, 'night', {'how': 'approved'})
    io = FakeIO(ws, durations={'ablation': 3600.0})
    st = C.run(ws, 'night', io, sleep=io.advance, poll_s=600.0)
    assert io.submitted == ['s1'] and [s['state'] for s in st['studies']] == ['done', 'refused']
    assert 'would exceed what is left (0.50 of 1.5)' in st['studies'][1]['why']
    assert st['stopped']['why'].startswith('the budget is used up') and st['answer']['confidence'] == 'none'
    assert 'Run the controlled comparison that did not fit the budget' in C.brief(ws, 'night')['markdown']
    assert st['studies'][0]['verdict'] in ('inconclusive', 'ruled out')
    # an estimate that alone exceeds the budget never enters the plan: it needs you
    rec = C.new(ws, GOAL, cid='tiny', kinds=['ablation'], budget={'gpu_hours': 0.5}, now=T0)
    over = [n for n in rec['needs_you'] if 'alone exceeds the campaign\'s 0.5 GPU-hours' in n['why']]
    assert rec['studies'] == [] and [n['what'] for n in over] == ['Ablation: rain at night', 'Ablation: fog at dusk']
    # the deadline: the running job is cancelled through the queue, nothing is left running
    with open(os.path.join(ws, 'jobs', 'jobs.jsonl'), 'a') as fh:           # (the history says the index is quick)
        fh.write(json.dumps({'schema': 'falcon.job-history/1', 'kind': 'campaign-moments', 'status': 'done',
                             'elapsed_s': 10.0}) + '\n')
    C.new(ws, GOAL, cid='late', kinds=['clusters', 'moments'], now=T0,
          budget={'gpu_hours': 1.0, 'deadline': C.utc(T0 + 100)})
    C.approve(ws, 'late', {'how': 'approved'})
    io = FakeIO(ws, durations={'clusters': 20.0, 'moments': 500.0})
    st = C.run(ws, 'late', io, sleep=io.advance, poll_s=30.0)
    assert io.cancelled == ['s2'] and io.running() == []
    assert [s['state'] for s in st['studies']] == ['done', 'cancelled'] and 'deadline' in st['studies'][1]['why']
    assert st['stopped']['why'].startswith('the deadline')
    assert decisions(ws, 'late')[-4:] == ['run', 'cancel', 'result', 'stop']
    # a study that would end after the deadline is not started at all
    C.new(ws, GOAL, cid='short', kinds=['moments'], now=T0, budget={'deadline': C.utc(T0 + 5)})
    C.approve(ws, 'short', {'how': 'approved'})
    io = FakeIO(ws)
    st = C.run(ws, 'short', io, sleep=io.advance)
    assert io.jobs == {} and st['studies'][0]['state'] == 'refused' and 'after the deadline' in st['studies'][0]['why']


def test_a_crash_cancels_its_job_and_a_restart_resumes(ws, monkeypatch):
    monkeypatch.setattr(R, 'suggestions', lambda workspace, project=None, limit=8: [])
    new(ws, kinds=['clusters', 'moments'])
    C.approve(ws, 'night', {'how': 'approved'})
    io = FakeIO(ws)

    def boom(dt):
        raise RuntimeError('power cut')
    with pytest.raises(RuntimeError):
        C.run(ws, 'night', io, sleep=boom)
    assert io.cancelled == ['s1'] and io.running() == [] and decisions(ws)[-1] == 'interrupted'
    assert C.status(ws, 'night', T0)['state'] == 'running'
    # only one runner at a time
    from falcon_autonomy.core.jobs import FileLock
    lock = FileLock('run.lock', C.folder(ws, 'night'))
    assert lock.acquire()
    with pytest.raises(C.CampaignError, match='being run already'):
        C.run(ws, 'night', io)
    lock.release()
    # a new process: its queue does not know the old job -> interrupted, run once more, then on to the end
    io2 = FakeIO(ws, t=T0 + 600)
    st = C.run(ws, 'night', io2, sleep=io2.advance, poll_s=30.0)
    assert io2.submitted == ['s1', 's2'] and st['state'] == 'done'
    assert [s['state'] for s in st['studies']] == ['done', 'done'] and st['used']['jobs'] == 3
    d = decisions(ws)
    assert d.count('interrupted') == 2 and d[-1] == 'stop'
    # lost twice: not run a third time
    C.new(ws, GOAL, cid='twice', kinds=['clusters'], now=T0)
    C.approve(ws, 'twice', {'how': 'approved'})
    for k in range(2):
        C.run(ws, 'twice', FakeIO(ws, t=T0 + 10 * k), max_steps=2 if k == 0 else 3)
    st = C.run(ws, 'twice', FakeIO(ws, t=T0 + 100))
    assert st['studies'][0]['state'] == 'failed' and 'lost' in st['studies'][0]['why'] and st['used']['jobs'] == 2


def test_stop_cancels_the_running_job_and_writes_the_brief(ws):
    new(ws)
    C.approve(ws, 'night', {'how': 'approved'})
    io = FakeIO(ws)
    C.run(ws, 'night', io, max_steps=2)                     # 'start' + s1 queued, then the runner is gone
    io.jobs[1]['state'] = 'running'                         # (this test keeps the job alive to see Stop cancel it)
    st = C.stop(ws, 'night', why='changed my mind', io=io)
    assert io.running() == [] and st['state'] == 'stopped' and st['stopped']['why'] == 'stopped: changed my mind'
    assert [s['state'] for s in st['studies']] == ['cancelled', 'not run', 'not run', 'not run']
    assert os.path.isfile(st['brief']) and not os.path.exists(os.path.join(C.folder(ws, 'night'), 'stop.json'))
    assert decisions(ws)[-3:] == ['stop_requested', 'cancel', 'stop']
    assert C.stop(ws, 'night')['state'] == 'stopped'
    # a draft can be stopped too; a runner sees a stop request at its next step
    C.new(ws, GOAL, cid='draft', now=T0)
    assert C.stop(ws, 'draft')['state'] == 'stopped'
    C.new(ws, GOAL, cid='live', kinds=['clusters', 'moments'], now=T0)
    C.approve(ws, 'live', {'how': 'approved'})
    io = FakeIO(ws)
    assert C.step(ws, 'live', io) == 'progress' and io.running()
    from falcon_autonomy.core.jobs import FileLock
    lock = FileLock('run.lock', C.folder(ws, 'live'))
    assert lock.acquire()                                   # a runner holds the campaign
    assert C.stop(ws, 'live')['state'] == 'running' and io.running()
    lock.release()
    assert C.step(ws, 'live', io) == 'done' and io.running() == [] and io.cancelled == ['s1']


# ------------------------------------------------------------------------------------------------ scope, needs you
def test_what_is_outside_the_scope_is_not_done_but_listed(ws):
    rec = new(ws, kinds=['clusters'])
    assert [s['kind'] for s in rec['studies']] == ['clusters']
    assert [(n['what'], n['why']) for n in rec['needs_you']] == [
        ('Moments matching the goal in recorded runs (moments)', "the study kind 'moments' is not in the approved scope"),
        ('Root cause of evaluation ev1 (rootcause)', "the study kind 'rootcause' is not in the approved scope"),
        ('Ablation: rain at night (ablation)', "the study kind 'ablation' is not in the approved scope")]
    assert 'Already outside the scope (needs you)' in C.approval_text(rec)
    C.approve(ws, 'night', {'how': 'approved'})
    io = FakeIO(ws)
    st = C.run(ws, 'night', io, sleep=io.advance, poll_s=30.0)
    assert io.submitted == ['s1'] and len(st['needs_you']) == 3
    md = C.brief(ws, 'night')['markdown']
    assert '- Ablation: rain at night (ablation) — the study kind' in md
    assert 'Decide the first "needs you" item' in md
    # a study smuggled into an approved record is refused by its kind, whatever its command
    rec = C.new(ws, GOAL, cid='smuggle', kinds=['clusters'], now=T0)
    C.approve(ws, 'smuggle', {'how': 'approved'})
    rec = C.load(ws, 'smuggle')
    rec['studies'].append({'id': 's9', 'key': 'x', 'hypothesis': 'h1', 'kind': 'train', 'title': 'Train a model',
                           'argv': ['train', '--out', 'x'], 'est': {'seconds': 1.0, 'gpu_seconds': 0.0, 'source': 'x'},
                           'state': 'planned', 'planned_at_approval': False})
    C.save(ws, rec)
    io = FakeIO(ws)
    st = C.run(ws, 'smuggle', io, sleep=io.advance, poll_s=30.0)
    assert io.submitted == ['s1'] and st['studies'][1]['state'] == 'refused'


def test_simulator_studies_are_planned_only_for_a_project_scenario_the_goal_names(ws, tmp_path):
    class Proj:
        doc = {'scenarios': [{'id': 'autorickshaw_cut_in'}, {'id': 'lead_brake'}]}
    pp = str(tmp_path / 'p.falcon.json')
    goal = 'Find how often the auto cuts in dangerously at night'
    rec = C.new(ws, goal, cid='sim', kinds=['moments', 'failrate', 'search'], project_path=pp, project=Proj(), now=T0,
                budget={'gpu_hours': 3.0})
    assert [(s['kind'], s['est']['gpu_seconds']) for s in rec['studies']] == [('moments', 0.0), ('failrate', 3600.0),
                                                                              ('search', 2400.0)]
    assert rec['studies'][1]['argv'] == ['failrate', 'run', pp, '--scenario', 'autorickshaw_cut_in', '--name',
                                         'sim-fr-autorickshaw_cut_in']
    assert rec['studies'][2]['argv'][:4] == ['search', pp, '--scenario', 'autorickshaw_cut_in']
    assert set(C.KINDS) == {'clusters', 'moments', 'rootcause', 'ablation', 'failrate', 'search'}
    C.approve(ws, 'sim', {'how': 'approved'})
    io = FakeIO(ws, durations={'failrate': 3600.0, 'search': 2400.0})
    st = C.run(ws, 'sim', io, sleep=io.advance, poll_s=600.0)
    assert io.submitted == ['s1', 's2', 's3'] and st['used']['gpu_hours'] == pytest.approx(1.6667, abs=1e-3)
    assert 'its folder holds the result' in st['studies'][1]['result'] and st['answer']['confidence'] == 'none'
    # without a project, or a scenario the project does not have, nothing of the kind is planned
    rec = C.new(ws, goal, cid='noproj', kinds=['failrate', 'search'], now=T0)
    assert rec['studies'] == []
    rec = C.new(ws, 'Find why a cow standing in the lane is hit', cid='other', kinds=['failrate'], project_path=pp,
                project=Proj(), now=T0)
    assert rec['studies'] == []


def test_a_fake_provider_orders_the_candidates_and_cannot_add_anything(ws):
    seen = []

    def complete(messages):
        seen.append(messages)
        return 'Here you go: ' + json.dumps({
            'order': ['ablation:cell:c1', 'moments', 'made-up-study'],
            'extra': [{'action': 'rent a cloud GPU for a bigger sweep', 'why': 'faster'},
                      {'action': 'download a larger detector', 'why': 'better recall'}]})
    rec = C.new(ws, GOAL, cid='llm', planner='assistant', complete=complete, now=T0)
    assert [s['kind'] for s in rec['studies']] == ['ablation', 'moments', 'clusters', 'rootcause']
    why = {n['what']: n['why'] for n in rec['needs_you']}
    assert why['rent a cloud GPU for a bigger sweep'].startswith('any cloud spend needs its own priced approval')
    assert why['download a larger detector'].startswith('a model download is never started unattended')
    assert why['made-up-study'].startswith('outside the approved scope')
    assert 'the model orders the candidates' in C.approval_text(rec)
    sent = json.dumps(seen)
    assert len(seen) == 1 and GOAL in sent and 'ablation:cell:c1' in sent
    assert ws not in sent and 'runs/' not in sent and 'ped-night' not in sent        # no paths, no run names
    assert any('ordered by the assistant' in r['reason'] for r in C.read_log(ws, 'llm'))
    # a reply that cannot be read: the playbook's order, said in the log
    rec = C.new(ws, GOAL, cid='llm2', planner='assistant', complete=lambda m: 'I think you should rest.', now=T0)
    assert [s['kind'] for s in rec['studies']] == ['clusters', 'moments', 'rootcause', 'ablation']
    assert any('could not be used' in r['reason'] for r in C.read_log(ws, 'llm2'))

    def broken(m):
        raise OSError('no network')
    rec = C.new(ws, GOAL, cid='llm3', planner='assistant', complete=broken, now=T0)
    assert len(rec['studies']) == 4 and rec['needs_you'] == []
    # the loop is the same loop: the accountant, not the model, decides what starts
    C.approve(ws, 'llm', {'how': 'approved'})
    io = FakeIO(ws)
    st = C.run(ws, 'llm', io, sleep=io.advance, poll_s=30.0, complete=complete)
    assert io.submitted == ['s1'] and st['answer']['confidence'] == 'high'           # the ablation answered it
    assert [s['state'] for s in st['studies']] == ['done', 'not run', 'not run', 'not run']


def test_private_runs_never_reach_the_brief(ws, tmp_path):
    write_run(ws, 'route-ped', 8.0, [actor('pedestrian', 60.0, lambda t: -6 + 1.5 * (t - 0.5), yaw=90.0,
                                           size=(0.5, 0.5))], weather=NIGHT_RAIN, settings={'dashcam_route': {'id': 'r'}})
    new(ws, kinds=['moments'])
    C.approve(ws, 'night', {'how': 'approved'})
    io = FakeIO(ws)
    st = C.run(ws, 'night', io, sleep=io.advance, poll_s=30.0)
    assert st['studies'][0]['result'].startswith('1 recorded moment(s)')
    assert '(1 more in private runs, not read)' in st['studies'][0]['result']
    blob = json.dumps(st) + C.brief(ws, 'night')['markdown'] + json.dumps(C.read_log(ws, 'night'))
    assert 'route-ped' not in blob and 'ped-night' in blob


# ------------------------------------------------------------------------------------------------ the command line
def test_cli_with_real_cpu_jobs(ws, capsys, monkeypatch):
    monkeypatch.setattr(R, 'suggestions', lambda workspace, project=None, limit=8: [])
    assert C.main_campaign(['new', ws, GOAL, '--id', 'cli', '--kinds', 'moments', '--gpu-hours', '0.5', '--until',
                            '+2h', '--max-jobs', '3']) == 0
    out = capsys.readouterr().out
    assert 'Campaign cli:' in out and 'Nothing runs yet' in out and 's1 [moments]' in out
    assert C.main_campaign(['run', ws, 'cli']) == 2
    assert 'not approved' in capsys.readouterr().out
    monkeypatch.setattr('builtins.input', lambda prompt='': 'n')
    assert C.main_campaign(['approve', ws, 'cli']) == 1
    assert C.main_campaign(['approve', ws, 'cli', '--yes']) == 0
    out = capsys.readouterr().out
    assert 'systemd-run --user --collect --unit=falcon-campaign-cli' in out and 'campaign run' in out
    assert C.main_campaign(['run', ws, 'cli', '--detached']) == 0
    assert capsys.readouterr().out.startswith('systemd-run --user')
    assert C.load(ws, 'cli')['approval']['how'] == 'command line'
    assert C.main_campaign(['run', ws, 'cli', '--poll', '0.2']) == 0          # `falcon moments index` as a subprocess
    out = capsys.readouterr().out
    assert 'campaign cli: done' in out and 'recorded moment(s) match' in out and 'brief:' in out
    assert os.path.isfile(M.db_path(ws))
    hist = [json.loads(x) for x in open(os.path.join(ws, 'jobs', 'jobs.jsonl'))]
    assert [(h['kind'], h['status']) for h in hist] == [('campaign-moments', 'done')]
    assert C.accountant(ws, C.load(ws, 'cli')).estimate('moments')['n'] == 1      # the next estimate is measured
    assert C.main_campaign(['status', ws, 'cli', '--json']) == 0
    st = json.loads(capsys.readouterr().out)
    assert st['state'] == 'done' and st['used']['jobs'] == 1 and st['answer']['confidence'] == 'low'
    assert C.main_campaign(['brief', ws, 'cli']) == 0
    assert '# Morning brief' in capsys.readouterr().out
    assert C.main_campaign(['stop', ws, 'cli']) == 0 and C.main_campaign(['list', ws]) == 0
    assert 'cli' in capsys.readouterr().out
    assert C.main_campaign(['status', ws, 'nope']) == 2
