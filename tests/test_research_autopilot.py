"""The research autopilot (core.autoresearch, assistant/tools/autopilot.py) against a fake Groq server with scripted
model replies and the simulated ablation executor (no network, no CARLA, no model download): the whole loop — a
question, an approved plan, runs, an inconclusive analysis, a follow-up, a conclusive one, a checked report — in each
permission mode, with an approval card at each step that queues compute (Autopilot too) and none for what the user
approved ahead; resumption after a restart in the middle of a run; Stop; a report with a number no tool returned kept
a draft; a failure mode from the dashcam footage never reaching the model; the follow-up rule; the tools' size and
the command-line job."""
import itertools
import json
import os
import threading

import pytest

from conftest import FAKE
from fake_groq import FAKE_KEY, FakeGroq, text, tools
from falcon_autonomy.assistant import agent as A
from falcon_autonomy.assistant import tool_api as T
from falcon_autonomy.assistant.keystore import KeyStore
from falcon_autonomy.assistant.service import AssistantService
from falcon_autonomy.assistant.session import Session
from falcon_autonomy.core import ablation as AB
from falcon_autonomy.core import autoresearch as R
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core.jobs import Job, JobQueue, JobSpec, Launcher
from falcon_autonomy.core.settings import Secrets, Settings

Q = 'Does the fine-tuned detector raise recall?'


def spec(add=0.04, noise=0.06, sd=0.04, **kw):
    """Round 1 (4 replicates, planned at sd 0.04) is inconclusive for want of power at the observed spread; the
    follow-up (12 replicates) supports it — measured with these seeds."""
    s = {'hypothesis': {'text': 'the fine-tuned detector raises recall', 'direction': 'increase'}, 'metric': 'AR100',
         'factors': [{'name': 'model', 'kind': 'value', 'levels': ['base', 'ft']},
                     {'name': 'weather', 'kind': 'value', 'levels': ['clear', 'rain']}],
         'contrasts': [{'factor': 'model', 'a': 'base', 'b': 'ft'}],
         'design': {'effect': 0.05, 'sd': sd}, 'analysis': {'n_boot': 500},
         'run': {'executor': 'simulated', 'simulate': {'base': 0.6, 'noise_sd': noise, 'unit_sd': 0.0,
                                                        'effects': [{'when': {'model': 'ft'}, 'add': add}]}}}
    s.update(kw)
    return s


# ------------------------------------------------------------------------------------------------ the harness
class _Handle:
    def __init__(self, job, q):
        self.job, self.q = job, q

    def cancel(self):
        if self.job.state == 'running' and getattr(self, 'hold', False):
            self.q.on_exit(self.job, P.CANCELLED)

    def kill(self):
        self.cancel()

    def send(self, cmd):
        return True


class InlineLauncher(Launcher):
    """Runs `falcon experiment ablation run ID --workspace WS` in this process (core.ablation.run) on a thread, once
    the job is released (the agent has recorded it as its own: the host's 'job' event) — or never (hold)."""

    def __init__(self, hold=False, auto=False):
        self.hold, self.auto = hold, auto
        self.released: dict = {}
        self.logs: dict = {}

    def release(self, jid):
        self.released.setdefault(jid, threading.Event()).set()

    def start(self, job, queue):
        h = _Handle(job, queue)
        h.hold = self.hold
        argv = [str(a) for a in job.spec.argv]
        if self.hold or 'run' not in argv:
            return h
        i = argv.index('run')
        exp, ws = argv[i + 1], argv[argv.index('--workspace') + 1]
        ev = self.released.setdefault(job.id, threading.Event())
        if self.auto:
            ev.set()
        logs = self.logs.setdefault(job.id, [])

        class Em:
            def log(self, t, *a, **k):
                logs.append(t)

            def progress(self, *a, **k):
                pass

            def metric(self, *a, **k):
                pass

        def work():
            ev.wait(30)
            try:
                AB.run(ws, exp, emitter=Em(), cancel=lambda: job.cancel_requested)
                queue.on_exit(job, P.OK)
            except AB.Cancelled:
                queue.on_exit(job, P.CANCELLED)
            except Exception as e:                          # noqa: BLE001
                job.log.append(('error', str(e)))
                queue.on_exit(job, P.FAILED)
        threading.Thread(target=work, daemon=True).start()
        return h


def make_queue(tmp_path, launcher):
    return JobQueue(launcher, lock_dir=str(tmp_path / 'locks'),
                    meminfo_fn=lambda: {'available_gb': 64.0, 'total_gb': 64.0})


class Host(A.Host):
    """Headless: records events, answers cards by a rule (choice, options), releases the jobs the agent queued."""

    def __init__(self, ctx, launcher, decide=None):
        super().__init__(ctx)
        self.events, self.cards = [], []
        self.launcher = launcher
        self.decide = decide or (lambda req: ('approve', {}))

    def event(self, kind, payload):
        self.events.append((kind, payload))
        if kind == 'approval':
            self.cards.append(payload)
            choice, opts = self.decide(payload)
            payload.decide(choice, opts)
        elif kind == 'job':
            self.launcher.release(payload)


@pytest.fixture
def srv():
    s = FakeGroq().start()
    yield s
    s.stop()


@pytest.fixture
def ids(monkeypatch):
    monkeypatch.setattr(Job, '_ids', itertools.count(700))       # job ids the scripts can name


def make(tmp_path, srv, queue, launcher, mode='ask', decide=None):
    st = Settings(str(tmp_path / 'config' / 'settings.json'))
    ks = KeyStore(Secrets(str(tmp_path / 'config' / 'secrets.json')), use_keyring=False)
    ws = str(tmp_path / 'proj.falcon')
    os.makedirs(ws, exist_ok=True)
    svc = AssistantService(st, ks, workspace_fn=lambda: ws)
    svc.set_cfg(base_url=srv.base_url, enabled=True, tpm_limit=0)
    svc.accept_notice()
    ks.set(FAKE_KEY)
    host = Host(T.ToolContext(queue=queue, workspace=ws), launcher, decide)
    ag = A.Agent(svc, Session(ws), host, mode)
    return ag, host, ws


def turn(ag, content, timeout=120.0):
    assert ag.start(content)
    assert ag.wait(timeout), 'the turn did not end'


def tool_msgs(ag, name=None):
    return [m for m in ag.session.messages if m.get('role') == 'tool' and (name is None or m.get('_name') == name)]


def results(ag, name):
    return [json.loads(m['content']) for m in tool_msgs(ag, name)]


def loop_script(study='s1', job1=700, job2=701, prose='The follow-up supported the hypothesis.'):
    return [tools(('research_autopilot', {'action': 'start', 'study': study, 'question': Q, 'spec': spec()}),
                  content='I will plan the study; the card shows it.'),
            tools(('plan_wait_jobs', {'job_ids': [job1]})),
            tools(('research_studies', {'study': study})),
            tools(('research_autopilot', {'action': 'follow_up', 'study': study})),
            tools(('plan_wait_jobs', {'job_ids': [job2]})),
            tools(('research_studies', {'study': study})),
            tools(('research_autopilot', {'action': 'report', 'study': study, 'text': prose})),
            text('The study is reported: the hypothesis is supported at the second look.')]


# ------------------------------------------------------------------------------------------------ the whole loop
@pytest.mark.parametrize('mode', ['ask', 'auto', 'autopilot'])
def test_the_whole_loop_with_a_card_at_each_compute_step(tmp_path, srv, ids, mode):
    la = InlineLauncher()
    q = make_queue(tmp_path, la)
    ag, host, ws = make(tmp_path, srv, q, la, mode)
    srv.push(*loop_script())
    turn(ag, Q + ' Run it as a study.')
    # a card for each step that queues compute, in every mode; the report asks only in Ask
    asked = [(c.tool, c.args.get('action')) for c in host.cards]
    want = [('research_autopilot', 'start'), ('research_autopilot', 'follow_up')]
    assert asked == want + ([('research_autopilot', 'report')] if mode == 'ask' else []), asked
    first, second = host.cards[0], host.cards[1]
    assert not first.allow_all and not second.allow_all                # never "approve all of this kind"
    assert 'follow_ups' in first.options and not second.options
    assert 'Research study s1' in first.preview and '2 cells × ' not in first.preview
    assert '4 cells × 4 replicates = 16 runs' in first.preview and 'power analysis' in first.preview
    assert 'Follow-up s1-f1 (more replicates)' in second.preview and 'inconclusive' in second.preview
    assert first.why.startswith('it queues compute')
    how = [m.get('_approval') for m in tool_msgs(ag, 'research_autopilot')]
    assert how == ['approved', 'approved', 'approved' if mode == 'ask' else 'auto'], how
    # the orchestrator's results the model read: inconclusive and a follow-up proposed, then conclusive
    st1, st2 = results(ag, 'research_studies')
    assert st1['state'] == 'follow_up' and st1['follow_up']['kind'] == 'more_replicates'
    assert st1['rounds'][0]['verdict'] == 'not supported' and 'underpowered' not in st1['why']
    assert st2['state'] == 'analysed' and st2['rounds'][1]['verdict'] == 'supported'
    assert st2['rounds'][1]['plan']['replicates'] == 12 and st2['rounds'][1]['alpha'] == 0.025
    rep = results(ag, 'research_autopilot')[-1]
    assert rep['final'] and rep['verdict'] == 'supported' and rep['ungrounded'] == []
    assert rep['paper'] == {'s1': 'paper/s1', 's1-f1': 'paper/s1-f1'}
    md = open(os.path.join(ws, 'experiments', 's1', 'report.md'), encoding='utf-8').read()
    for part in ('# Research study s1', '**Status: final**', '## Hypothesis', '## Design', '## Pins', '## Results',
                 '## Verdict', '## Discussion', 'The follow-up supported the hypothesis.', '## Limits',
                 '## Paper outputs', 'SIMULATED', 'stays below 0.1'):
        assert part in md, part
    assert os.path.isfile(os.path.join(ws, 'paper', 's1-f1', 'contrasts.tex'))
    rec = R.load(ws, 's1')
    assert rec['state'] == 'reported' and [r['experiment'] for r in rec['rounds']] == ['s1', 's1-f1']
    states = [h['state'] for h in rec['history']]
    assert states[:3] == ['proposed', 'approved', 'running'] and 'follow_up' in states and states[-1] == 'reported'
    # the jobs are the agent's own: in the activity log, each with Undo (cancelling it)
    acts = [a for a in ag.session.activity if a['kind'] == 'job']
    assert [a['job_id'] for a in acts] == [700, 701] and all(a['tool'] == 'research_autopilot' for a in acts)
    assert [j.spec.meta.get('study') for j in q.jobs] == ['s1', 's1']
    assert all('--workspace' in j.spec.argv for j in q.jobs)
    assert not [n for m in ag.session.messages if m.get('role') == 'assistant' for n in m.get('_ungrounded') or ()]


def test_follow_ups_approved_ahead_run_without_a_card_within_the_plan(tmp_path, srv, ids):
    la = InlineLauncher()
    q = make_queue(tmp_path, la)
    ag, host, ws = make(tmp_path, srv, q, la, 'autopilot',
                        decide=lambda req: ('approve', {'follow_ups': True} if req.options else {}))
    srv.push(*loop_script())
    turn(ag, Q)
    assert [c.args['action'] for c in host.cards] == ['start']              # the follow-up asked no card
    how = [m.get('_approval') for m in tool_msgs(ag, 'research_autopilot')]
    assert how == ['approved', 'preapproved', 'auto']
    rec = R.load(ws, 's1')
    assert rec['follow_ups']['approved'] and rec['rounds'][1]['approval']['how'] == 'preapproved'
    assert 'first card' in rec['rounds'][1]['approval']['why']
    # bounded: a follow-up outside the approved levels is refused, approved ahead or not
    wide = spec()
    wide['factors'][1]['levels'] = ['clear', 'rain', 'fog']
    s2 = AB.normalise(dict(wide, id='x'))
    assert not R.within_bounds(rec, s2)[0] and R.preapproved(rec, 'follow_up', s2) == (False, '')
    with pytest.raises(R.StudyError, match='approved levels'):
        R.follow_up(ws, 's1', {'how': 'approved'}, lambda e: {'id': 1, 'submitted': 0.0}, spec=wide)


def test_compute_is_refused_without_the_users_approval(tmp_path):
    la = InlineLauncher(auto=True)
    q = make_queue(tmp_path, la)
    ws = str(tmp_path / 'ws')
    os.makedirs(ws)
    from falcon_autonomy.assistant.privacy import redact
    ctx = T.ToolContext(queue=q, workspace=ws, redact=redact)
    args = {'action': 'start', 'study': 's1', 'question': Q, 'spec': spec()}
    t = T.REGISTRY['research_autopilot']
    g = t.approval(ctx, **args)
    assert g['ask'] is True and g['allow_all'] is False and 'follow_ups' in g['options']
    r = T.call(ctx, 'research_autopilot', args)                       # Auto mode's 'auto' is not an approval
    assert 'approves its card' in r['error'] and q.jobs == [] and not os.path.exists(R.path_of(ws, 's1'))
    ctx.approval = {'how': 'auto'}
    assert 'approves its card' in T.call(ctx, 'research_autopilot', args)['error']
    assert t.approval(ctx, action='report', study='s1') is None         # report and stop: the mode decides
    bad = dict(args, spec=dict(spec(), metric=''))
    assert t.approval(ctx, **bad) is None and 'metric' in T.call(ctx, 'research_autopilot', bad)['error']
    ctx.approval = {'how': 'approved', 'options': {}}
    r = T.call(ctx, 'research_autopilot', args)
    assert r['state'] in ('running', 'follow_up', 'analysed') and r['study'] == 's1', r
    assert 'exists already' in T.call(ctx, 'research_autopilot', args)['error']
    assert 'no suggestion' in T.call(ctx, 'research_autopilot', dict(args, study='s2', source='cluster:C-0'))['error']


# ------------------------------------------------------------------------------------------------ restart, stop
def test_resume_after_a_restart_in_the_middle_of_a_run(tmp_path, srv, ids):
    ws = str(tmp_path / 'proj.falcon')
    os.makedirs(ws)
    hold = InlineLauncher(hold=True)
    q1 = make_queue(tmp_path, hold)
    from falcon_autonomy.assistant.privacy import redact
    ctx = T.ToolContext(queue=q1, workspace=ws, redact=redact, approval={'how': 'approved'})
    r = T.call(ctx, 'research_autopilot', {'action': 'start', 'study': 's1', 'question': Q, 'spec': spec()})
    assert r['state'] == 'running' and r['job'] == 700
    done = {'n': 0}

    def cancel():                                  # the job's process gets 6 of 16 runs done, then Falcon is closed
        done['n'] += 1
        return done['n'] > 6
    with pytest.raises(AB.Cancelled):
        AB.run(ws, 's1', cancel=cancel)
    # a restart: a new queue (job ids start again: an old id must not be taken for a new job), the files on disk
    la = InlineLauncher()
    q2 = make_queue(tmp_path, la)
    other = q2.submit(JobSpec('x', 'another job of the new session', ['true']))
    assert other.id == 701
    from falcon_autonomy.assistant.tools.autopilot import job_state
    ctx2 = T.ToolContext(queue=q2, workspace=ws, redact=redact)
    st = T.call(ctx2, 'research_studies', {'study': 's1'})
    assert st['state'] == 'interrupted' and 'restarted' in st['why'] and st['rounds'][0]['runs']['ok'] == 6
    assert 'resume' in st['next']
    ag, host, ws2 = make(tmp_path, srv, q2, la, 'autopilot')
    assert ws2 == ws
    srv.push(tools(('research_studies', {})),
             tools(('research_autopilot', {'action': 'resume', 'study': 's1'})),
             tools(('plan_wait_jobs', {'job_ids': [702]})),
             tools(('research_studies', {'study': 's1'})),
             text('Resumed; round one finished and is inconclusive.'))
    turn(ag, 'Continue my research study.')
    assert [(c.tool, c.args['action']) for c in host.cards] == [('research_autopilot', 'resume')]
    assert 'resume s1: queue' in host.cards[0].preview and '10 of 16 runs left' in host.cards[0].preview
    assert any('6 finished already' in t for t in la.logs[702])         # the finished runs were not repeated
    final = results(ag, 'research_studies')[-1]
    assert final['state'] == 'follow_up' and final['rounds'][0]['runs']['ok'] == 16
    assert R.derive(ws, R.load(ws, 's1'), job_state(ctx2))[0] == 'follow_up'


def test_stop_cancels_the_job_and_revokes_the_approval_ahead(tmp_path, srv, ids):
    from falcon_autonomy.assistant.privacy import redact
    from falcon_autonomy.assistant.tools.autopilot import job_state
    ws = str(tmp_path / 'proj.falcon')
    os.makedirs(ws)
    hold = InlineLauncher(hold=True)
    q = make_queue(tmp_path, hold)
    ctx = T.ToolContext(queue=q, workspace=ws, redact=redact,
                        approval={'how': 'approved', 'options': {'follow_ups': True}})
    T.call(ctx, 'research_autopilot', {'action': 'start', 'study': 's1', 'question': Q, 'spec': spec()})
    assert q.get(700).state == 'running' and R.load(ws, 's1')['follow_ups']['approved']
    # the agent waiting on the study's job: Stop ends the wait and the turn
    ag, host, _ws = make(tmp_path, srv, q, hold, 'autopilot')
    srv.push(tools(('plan_wait_jobs', {'job_ids': [700]})))
    assert ag.start('wait for my study')
    import time
    t0 = time.monotonic()
    while not any(k == 'waiting' and p for k, p in host.events) and time.monotonic() - t0 < 10:
        time.sleep(0.02)
    ag.stop()
    assert ag.wait(10) and ('turn', {'state': 'stopped'}) in host.events
    # the study's Stop (the Research view's button, or the tool): its job cancelled, the stop recorded
    ctx.approval = {}
    r = T.call(ctx, 'research_autopilot', {'action': 'stop', 'study': 's1'})
    assert r['state'] == 'stopped' and q.get(700).state == 'cancelled'
    rec = R.load(ws, 's1')
    assert rec['stopped']['cancelled'] == [700] and not rec['follow_ups']['approved'] and rec['follow_ups']['revoked']
    assert 'do not resume' in r['next']
    t = T.REGISTRY['research_autopilot']
    g = t.approval(ctx, action='resume', study='s1')
    assert g['ask'] is True                                    # after Stop, resuming asks again
    assert 'approves its card' in T.call(ctx, 'research_autopilot', {'action': 'resume', 'study': 's1'})['error']
    ctx.approval = {'how': 'approved'}
    r = T.call(ctx, 'research_autopilot', {'action': 'resume', 'study': 's1'})
    assert r['state'] == 'running' and r['job'] == 701
    # a job cancelled from the activity log (Undo) stops the study too
    A.undo_activity({'kind': 'job', 'job_id': 701}, queue=q)
    assert R.derive(ws, R.load(ws, 's1'), job_state(ctx))[0] == 'stopped'


# ------------------------------------------------------------------------------------------------ the report
def _conclusive(ws):
    n = itertools.count(1)

    def submit(e):
        AB.run(ws, e)
        return {'id': next(n), 'submitted': 0.0}
    p = R.draft(ws, 'c1', Q, spec(add=0.08, noise=0.02, sd=0.04))
    R.start(ws, p, {'how': 'approved'}, submit)
    return lambda j: 'done'


def test_a_report_with_a_number_no_tool_returned_is_not_final(tmp_path):
    ws = str(tmp_path / 'ws')
    js = _conclusive(ws)
    assert R.derive(ws, R.load(ws, 'c1'), js)[0] == 'analysed'
    res = json.load(open(os.path.join(ws, 'experiments', 'c1', 'results.json')))
    c = res['contrasts'][0]
    doc = R.report(ws, 'c1', f'The difference was {c["diff"]:.3f}, an effect of 0.4321 in plain terms.', js)
    assert not doc['final'] and doc['ungrounded'] == ['0.4321']
    md = open(os.path.join(ws, 'experiments', 'c1', 'report.md'), encoding='utf-8').read()
    assert '**Status: draft**' in md and "0.4321 [not from Falcon's data]" in md
    assert f'{c["diff"]:.3f} [not' not in md                       # the study's own number passes
    assert R.load(ws, 'c1')['report']['final'] is False
    st = R.status(ws, 'c1', js)
    assert st['state'] == 'reported' and 'draft' in st['why'] and 'write it again' in st['next']
    doc = R.report(ws, 'c1', f'B − A was {c["diff"]:.4f} with a 95 % interval of [{c["ci_low"]:.4f}, '
                             f'{c["ci_high"]:.4f}]; dz {c["dz"]:.2f}, {100 * c["power_at_planned"]:.0f} % power.', js)
    assert doc['final'] and doc['ungrounded'] == [], doc['ungrounded']
    assert json.load(open(os.path.join(ws, 'experiments', 'c1', 'report.json')))['final'] is True
    # every number Falcon itself wrote into the report passes the check (it is generated from the facts)
    md = open(os.path.join(ws, 'experiments', 'c1', 'report.md'), encoding='utf-8').read()
    assert R.ground(ws, R.load(ws, 'c1'), md.split('**Status: final**')[1]) == []


def test_the_follow_up_rule(tmp_path):
    ws = str(tmp_path / 'ws')
    n = itertools.count(1)

    def submit(e):
        AB.run(ws, e)
        return {'id': next(n), 'submitted': 0.0}
    js = (lambda j: 'done')
    # well powered and nothing found: an answer, not a follow-up
    p = R.draft(ws, 'n1', Q, spec(add=0.0, noise=0.02, sd=0.03))
    R.start(ws, p, {'how': 'approved'}, submit)
    st, why = R.derive(ws, R.load(ws, 'n1'), js)
    assert st == 'analysed' and 'had the target power' in why
    # no effect planned (replicates given): the power cannot be judged
    p = R.draft(ws, 'n2', Q, spec(add=0.0, noise=0.06, design={'replicates': 3}))
    R.start(ws, p, {'how': 'approved'}, submit)
    assert 'no effect was planned' in R.derive(ws, R.load(ws, 'n2'), js)[1]
    # a mixed "by" split: supported in one weather, inconclusive in the other -> narrowed to it
    s = spec(add=0.0, noise=0.05, sd=0.03,
             contrasts=[{'factor': 'model', 'a': 'base', 'b': 'ft', 'by': ['weather']}])
    s['run']['simulate']['effects'] = [{'when': {'model': 'ft', 'weather': 'clear'}, 'add': 0.2},
                                       {'when': {'model': 'ft', 'weather': 'rain'}, 'add': 0.02}]
    p = R.draft(ws, 'm1', Q, s)
    R.start(ws, p, {'how': 'approved'}, submit)
    rec = R.load(ws, 'm1')
    fu = R.follow_up_proposal(ws, rec)
    assert fu['kind'] == 'narrower' and fu['spec']['factors'][1]['levels'] == ['rain'], fu.get('why')
    assert fu['spec']['design']['alpha'] == 0.025 and fu['spec']['design']['seed'] == 1000
    assert fu['spec']['design']['sd'] > 0.03 and 'look 2' in fu['why']
    rec = R.follow_up(ws, 'm1', {'how': 'approved'}, submit, job_state=js)
    assert rec['rounds'][1]['kind'] == 'narrower' and rec['rounds'][1]['experiment'] == 'm1-f1'
    with pytest.raises(R.StudyError, match='approves its card'):        # the model's own follow-up: still a card
        R.follow_up(ws, 'm1', None, submit, spec=s, job_state=js)
    with pytest.raises(R.StudyError, match='exists already'):
        R.draft(ws, 'm1', Q, spec())


# ------------------------------------------------------------------------------------------------ privacy
def test_a_private_failure_mode_never_reaches_the_model(tmp_path, srv, ids):
    from test_failclusters import _w, lead_brake_row, make_workspace
    ws = str(tmp_path / 'proj.falcon')
    os.makedirs(ws)
    make_workspace(ws)
    # a failure mode from the dashcam footage: an incident made a scenario, failing in a distinctive way
    rows = []
    for i in range(6):
        r = lead_brake_row(ws, 40 + i, 60.0, private=True)
        r['run'] = r['run'].replace('search-lead_brake', 'incident-secret')
        os.makedirs(r['run'], exist_ok=True)
        os.replace(os.path.join(ws, 'studies', 'search-lead_brake', 'runs', f'r{40 + i}', 'run.json'),
                   os.path.join(r['run'], 'run.json'))
        r['scenario'] = dict(r['scenario'], id='dashcam_incident_secret')
        rows.append(r)
    _w(os.path.join(ws, 'studies', 'incident-secret', 'study.json'), {'scenario': 'dashcam_incident_secret'})
    _w(os.path.join(ws, 'studies', 'incident-secret', 'results.jsonl'), rows)
    from falcon_autonomy.core import failclusters as FC
    assert any(f['scenario'] == 'dashcam_incident_secret' for f in FC.collect(ws))   # it is there, unfiltered
    sug = R.suggestions(ws)
    assert sug and all('dashcam_incident_secret' not in json.dumps(s) for s in sug)
    assert any(s['kind'] == 'failure_mode' and 'lead_brake' in json.dumps(s['facts']) for s in sug)
    la = InlineLauncher()
    q = make_queue(tmp_path, la)
    ag, host, _ = make(tmp_path, srv, q, la, 'autopilot')
    srv.push(tools(('research_studies', {'suggest': True})),
             text('Two failure modes could be studied.'))
    turn(ag, 'What could I research next?')
    got = results(ag, 'research_studies')[0]
    assert got['suggestions'] and 'next' in got
    sent = json.dumps([r['body'] for r in srv.chats()])
    assert 'dashcam_incident_secret' not in sent and 'incident-secret' not in sent
    assert 'lead_brake' in sent
    # a source that is not a suggestion (a private cluster id cannot be one) is refused
    from falcon_autonomy.assistant.privacy import redact
    ctx = T.ToolContext(queue=q, workspace=ws, redact=redact, approval={'how': 'approved'})
    priv = [c for c in FC.cluster(FC.collect(ws)) if 'dashcam_incident_secret' in c['scenarios']][0]
    r = T.call(ctx, 'research_autopilot', {'action': 'start', 'study': 'p1', 'question': Q, 'spec': spec(),
                                           'source': f'cluster:{priv["id"]}'})
    assert 'no suggestion' in r['error']


def test_suggestions_carry_templates_for_conditions(tmp_path, project):
    from test_failclusters import make_workspace
    ws = project.workspace
    make_workspace(ws)
    project.doc['models'] = [{'id': 'base', 'name': 'Base', 'kind': 'detector', 'runtime': 'python', 'entry': 'x:y'},
                             {'id': 'ft', 'name': 'Fine-tuned', 'kind': 'detector', 'runtime': 'python',
                              'entry': 'x:y'}]
    sug = R.suggestions(ws, project)
    fm = [s for s in sug if s['kind'] == 'failure_mode']
    night = next(s for s in fm if s['template'] is not None)
    t = night['template']
    assert [f['name'] for f in t['factors']] == ['model', 'conditions'] and t['run']['executor'] == 'detection'
    mode = t['factors'][1]['levels']['mode']
    assert mode['sun_altitude_angle'] < 0 and 'manual\'s example' in night['note']
    s = AB.normalise(dict(t, id='tpl'), project)                # the template is a valid spec for this project
    assert AB.plan(s)['counts']['drives'] > 0
    cells = [s for s in sug if s['kind'] == 'odd_cell']
    assert all(c['template'] is None or c['template']['factors'][1]['kind'] == 'weather' for c in cells)


# ------------------------------------------------------------------------------------------------ the tools
def test_the_tools_are_registered_documented_and_small():
    from falcon_autonomy.assistant.agent import GROUP_ABOUT, guess_groups
    from falcon_autonomy.assistant.cli import schema_tokens
    from falcon_autonomy.assistant.tools._compact import DETAILS
    for n, tier in (('research_autopilot', 'act'), ('research_studies', 'read')):
        t = T.REGISTRY[n]
        assert t.group == 'research' and t.tier == tier and len(DETAILS[n]['description']) > 200
    t = T.REGISTRY['research_autopilot']
    assert t.preview is not None and t.estimate is not None and t.approval is not None
    assert 'research autopilot' in GROUP_ABOUT['research']
    assert guess_groups('run a research study of this hypothesis')[0] == 'research'
    toks = schema_tokens()
    assert toks['research'] < 3500 and toks['all'] < 14000, toks


def test_the_command_line_job(tmp_path, project):
    """The study's job is `falcon experiment ablation run ID --workspace WS` through the app's queue: run for real
    (a subprocess), waited for with jobs_wait, then analysed and read back."""
    from falcon_autonomy.assistant import tools as TOOLS
    ctx = TOOLS.context(project.path, FAKE)
    try:
        ctx.approval = {'how': 'command line'}
        r = T.call(ctx, 'research_autopilot', {'action': 'start', 'study': 'cli1', 'question': Q,
                                               'spec': spec(design={'replicates': 2, 'effect': 0.05, 'sd': 0.04})})
        assert r['state'] == 'running', r
        job = ctx.queue.get(r['job'])
        assert job.spec.argv[-4:] == ['run', 'cli1', '--workspace', project.workspace] or \
            job.spec.argv[-2:] == ['--workspace', os.path.realpath(project.workspace)]
        w = T.call(ctx, 'jobs_wait', {'jobs': [r['job']], 'timeout_s': 240})
        assert w['jobs'][0]['state'] == 'done', w
        st = T.call(ctx, 'research_studies', {'study': 'cli1'})
        assert st['state'] in ('analysed', 'follow_up') and st['rounds'][0]['runs']['ok'] == 8
        lst = T.call(ctx, 'research_studies', {})
        assert lst['items'][0]['study'] == 'cli1'
    finally:
        ctx.service.shutdown(grace_s=10.0)


def test_the_eval_task_runs_end_to_end(srv, tmp_path):
    """The shipped eval task for the autopilot (assistant/eval_tasks.json) with its scripted model: the study's card
    approved, its real job (`falcon experiment ablation run … --workspace`) done, the study read back, nothing
    ungrounded."""
    from falcon_autonomy.assistant import evaluate as E
    from test_assistant_eval import cfg_for, shipped, sink_for
    doc = E.run_eval(shipped('research_study_simulated'), cfg_for(srv), FAKE_KEY, str(tmp_path / 'out'),
                     fake_carla=FAKE, script_sink=sink_for(srv), log=None)
    r = doc['results'][0]
    assert r['success'], r['failures']
    assert r['approvals'] == [{'tool': 'research_autopilot', 'tier': 'act', 'choice': 'approve'}]
    assert r['tool_calls'] == ['research_autopilot', 'research_studies'] and r['grounding_flags'] == []
