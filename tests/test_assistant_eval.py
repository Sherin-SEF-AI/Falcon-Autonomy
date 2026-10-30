"""`falcon assistant eval` (assistant.evaluate) against the fake Groq server and the fake carla: the shipped tasks are
valid and name only real tools, the criteria pass and fail for the right reasons, tasks run end to end with scripted
model responses (setup jobs, a failing job, a private run, an approval refused, project edits, jobs waited for),
results.json, summary.md and the audit copies are written, a model doing the wrong thing fails with the reason, and the
command line refuses without a key or consent (nothing sent)."""
import copy
import json

import pytest

from conftest import FAKE
from fake_groq import FAKE_KEY, FakeGroq, text, tools
from falcon_autonomy.assistant import agent as A
from falcon_autonomy.assistant import evaluate as E
from falcon_autonomy.assistant.service import NOTICE_VERSION
from falcon_autonomy.core.settings import ASSISTANT_DEFAULTS


@pytest.fixture
def srv():
    s = FakeGroq().start()
    yield s
    s.stop()


def sink_for(srv):
    def sink(script):
        for st in script:
            if 'tools' in st:
                srv.push(tools(*[(n, a) for n, a in st['tools']], content=st.get('content', '')))
            else:
                srv.push(text(st['text']))
    return sink


def cfg_for(srv):
    return dict(copy.deepcopy(ASSISTANT_DEFAULTS), base_url=srv.base_url, enabled=True, consent=NOTICE_VERSION,
                tpm_limit=0)


def shipped(*ids):
    by = {t['id']: t for t in E.load_tasks()}
    return [copy.deepcopy(by[i]) for i in ids]


# ------------------------------------------------------------------------------------------------ the task file
def test_shipped_tasks_are_valid_and_name_real_tools():
    import jsonschema
    jsonschema.Draft202012Validator.check_schema(E.TASK_SCHEMA)
    tasks = E.load_tasks()
    assert len(tasks) >= 20 and len({t['id'] for t in tasks}) == len(tasks)
    for t in tasks:
        assert E.validate_task(t) == [], (t['id'], E.validate_task(t))
        assert t.get('script'), f'{t["id"]} has no script for the tests'
        for st in t['script']:
            for name, _args in st.get('tools') or []:
                tl = A.lookup(name)
                assert tl is not None and not tl.gui, (t['id'], name)      # headless: no window tools
    assert E.validate_task({'id': 'x', 'title': 'a task', 'prompt': 'do it', 'expect': {'calls': [{'tool': 'nope'}]}}) \
        == ["no tool named 'nope'"]
    assert 'expect' in E.validate_task({'id': 'x', 'title': 'a task', 'prompt': 'do it'})[0]
    bad = {'id': 'x', 'title': 'a task', 'prompt': 'do it', 'expect': {}, 'setup': [{'call': 'runs_search', 'as': 'a'},
                                                                                 {'call': 'runs_search', 'as': 'a'}]}
    assert "setup name 'a' is used twice" in E.validate_task(bad)


def test_placeholders_and_matching():
    v = {'d1': {'run': 'runs/a', 'job': 7}, 'st': {'failures': {'items': [{'index': 3}]}}}
    assert E.fill({'job': '{d1.job}', 'text': 'job {d1.job} in {d1.run}', 'i': '{st.failures.items.0.index}'}, v) == \
        {'job': 7, 'text': 'job 7 in runs/a', 'i': 3}
    with pytest.raises(KeyError, match='no value for {d2.run}'):
        E.fill('{d2.run}', v)
    assert E.matches({'weather_preset~': 'night', 'seconds': 10}, {'weather_preset': 'ClearNight', 'seconds': 10.0})
    assert not E.matches({'seconds': 10}, {'seconds': 11})
    assert E.matches({'runs~': ['a', 'b']}, {'runs': ['b', 'c', 'a']}) and not E.matches({'runs~': ['a']}, {'runs': []})
    assert E.matches({'weather.preset': 'X'}, {'weather': {'preset': 'X'}}) and not E.matches({'k': 1}, {})


def test_criteria_pass_and_fail_with_reasons():
    out = {'calls': [{'name': 'jobs_get', 'args': {'job': 3}, 'error': None},
                     {'name': 'runs_get', 'args': {'run': 'runs/p'}, 'error': 'private: a drive along a route'}],
           'answer': 'No CARLA server answered; that run is private.', 'ungrounded': [],
           'approvals': [{'tool': 'runs_trash', 'tier': 'trash', 'choice': 'reject'}],
           'jobs': [{'id': 3, 'kind': 'drive', 'state': 'failed', 'mine': False},
                    {'id': 4, 'kind': 'export', 'state': 'done', 'mine': True}],
           'doc': {'trigger_rules': [{'id': 'rule1', 'rule': 'when actor.ttc < 1.5 -> marker'}],
                   'worlds': [{'id': 'w', 'weather': {'preset': 'HardRainNoon'}}]},
           'runs': ['runs/a']}
    ok = {'calls': [{'tool': 'jobs_get', 'args': {'job': 3}}], 'any_of': [[{'tool': 'x_nope'}, {'tool': 'runs_get'}]],
          'not_calls': ['runs_trash'], 'no_tier': ['act', 'trash'], 'call_errors': [{'text': 'private'}],
          'approvals': [{'tool': 'runs_trash', 'choice': 'reject'}],
          'project': [{'section': 'trigger_rules', 'match': {'rule~': 'actor.ttc'}},
                      {'section': 'worlds', 'match': {'weather.preset': 'HardRainNoon'}},
                      {'section': 'scenarios', 'absent': True}],
          'runs_exist': ['runs/a'], 'runs_gone': ['runs/b'],
          'reads': [{'tool': 'runs_search', 'args': {'query': 'tag:rain'}, 'expect': {'total': 1}}],
          'jobs': [{'kind': 'export', 'state': 'done'}, {'id': 3, 'state': 'failed'}],
          'answer_contains': ['server'], 'answer_any': ['nope', 'PRIVATE'], 'answer_excludes': ['deleted']}
    assert E.check(ok, out, read=lambda tool, args: {'total': 1, 'items': []}) == []
    bad = {'calls': [{'tool': 'jobs_get', 'args': {'job': 4}}, {'tool': 'jobs_cancel'}],
           'any_of': [[{'tool': 'runs_events'}]], 'not_calls': ['runs_get'], 'no_tier': ['read'],
           'call_errors': [{'tool': 'jobs_get', 'text': 'private'}], 'approvals': [{'tool': 'runs_trash',
                                                                                    'choice': 'approve'}],
           'project': [{'section': 'trigger_rules', 'match': {'rule~': 'rain'}},
                       {'section': 'trigger_rules', 'absent': True, 'match': {'id': 'rule1'}}],
           'runs_exist': ['runs/b'], 'runs_gone': ['runs/a'],
           'reads': [{'tool': 'runs_search', 'args': {}, 'expect': {'total': 2}}],
           'jobs': [{'kind': 'drive', 'state': 'done'}, {'id': 3, 'state': 'done'}],
           'answer_contains': ['trash'], 'answer_any': ['xyz'], 'answer_excludes': ['private']}
    fails = E.check(bad, dict(out, ungrounded=['42']), read=lambda tool, args: {'total': 1})
    text_ = '\n'.join(fails)
    for want in ('jobs_get was not called with {"job": 4} (called with [{"job": 3}])',
                 'jobs_cancel was not called with {} (not called)', 'none of runs_events was called',
                 'runs_get was called, and must not be', 'read tools were called: jobs_get, runs_get',
                 'no jobs_get call returned an error containing "private"',
                 'no approval card for runs_trash answered approve', 'has no trigger_rules item matching',
                 'still has trigger_rules item', 'the run runs/b is no longer', 'the run runs/a is still',
                 'runs_search {} afterwards returned {"total": 1}, not {"total": 2}',
                 'a drive job the assistant queued did not end done (none)', 'job 3 did not end done (3 drive failed)',
                 'does not mention "trash"', 'mentions none of "xyz"', 'the answer mentions "private"',
                 "numbers not from Falcon's data: 42"):
        assert want in text_, (want, fails)
    assert len(fails) == 18
    assert E.check({}, dict(out, answer='')) == ['there was no final answer']
    assert E.check({'grounded': False}, dict(out, ungrounded=['42'])) == []


# ------------------------------------------------------------------------------------------------ end to end
def test_tasks_end_to_end_with_a_scripted_model(srv, tmp_path):
    """Setup through the tools (drives queued and waited for, a failing job, a private run), the agent on its own
    throw-away project with the fake carla, the criteria checked — every scripted task passes, and what was measured
    is the server's own counts."""
    ids = ('explain_failed_job', 'add_trigger_rule', 'private_run_refused', 'trash_asks_first', 'settings_never',
           'compare_last_two_drives', 'cancel_job', 'export_latest_coco', 'queue_and_server', 'tag_rainy_runs')
    tasks = shipped(*ids)
    logs = []
    doc = E.run_eval(tasks, cfg_for(srv), FAKE_KEY, str(tmp_path / 'out'), fake_carla=FAKE,
                     script_sink=sink_for(srv), log=logs.append)
    res = {r['task']: r for r in doc['results']}
    assert [r['task'] for r in doc['results']] == list(ids)
    assert all(r['success'] for r in doc['results']), {k: r['failures'] for k, r in res.items() if not r['success']}
    assert len(logs) == len(ids) and all(s.startswith('PASS openai/gpt-oss-120b ') for s in logs)
    t = res['trash_asks_first']
    assert t['approvals'] == [{'tool': 'runs_trash', 'tier': 'trash', 'choice': 'reject'}]
    assert res['explain_failed_job']['tool_calls'] == ['jobs_get'] and 'server' in res['explain_failed_job']['answer']
    # the server's counts (fake: 200 + 20 for a tool step, 120 + 30 for an answer), priced from the table
    a = res['add_trigger_rule']
    assert a['steps'] == 3 and a['tokens'] == {'prompt': 520, 'completion': 70}
    assert a['cost'] == pytest.approx((520 * 0.15 + 70 * 0.60) / 1e6, abs=1e-6) and a['grounding_flags'] == []
    s = doc['summary']['openai/gpt-oss-120b']
    assert s['tasks'] == len(ids) and s['passed'] == len(ids) and s['success_rate'] == 1.0
    assert s['setup_errors'] == 0 and s['timed_out'] == 0 and s['cost'] > 0
    out = tmp_path / 'out'
    saved = json.loads((out / 'results.json').read_text())
    assert saved['summary'] == doc['summary'] and saved['meta']['server'] == f'127.0.0.1:{srv.port}'
    md = (out / 'summary.md').read_text()
    assert f'| `openai/gpt-oss-120b` | {len(ids)}/{len(ids)} (100 %) |' in md and '`cancel_job` | pass' in md
    audit = (out / 'audit' / 'openai_gpt-oss-120b--add_trigger_rule.jsonl').read_text().splitlines()
    assert len(audit) == 3 and all('messages' not in json.loads(x) for x in audit)
    assert FAKE_KEY not in (out / 'results.json').read_text() + md
    assert all(r['path'].startswith('/openai/v1/') for r in srv.requests)


def test_long_jobs_and_autopilot(srv, tmp_path):
    """Tasks whose jobs take a while: a study made by a sweep and its worst row replayed, and a night drive in a
    world the setup adds, recorded in Autopilot and checked for dataset health after the job finished."""
    doc = E.run_eval(shipped('worst_rare_case_replay', 'night_drive_health'), cfg_for(srv), FAKE_KEY,
                     str(tmp_path / 'out'), fake_carla=FAKE, script_sink=sink_for(srv), log=None)
    res = {r['task']: r for r in doc['results']}
    assert all(r['success'] for r in doc['results']), {k: r['failures'] for k, r in res.items()}
    assert res['night_drive_health']['tool_calls'][-1] == 'runs_health'


def test_a_model_doing_the_wrong_thing_fails_with_the_reason(srv, tmp_path):
    t1, t2, t3, t4 = shipped('traffic_density', 'settings_never', 'free_memory', 'copy_scenario')
    t1['script'] = [{'tools': [['project_traffic_set', {'profile': 'tp-default', 'vehicles': 10}]]},
                    {'text': 'Done: the default profile has 10 vehicles.'}]
    t2['script'] = [{'tools': [['project_world_set', {'world': 'world-town10', 'name': 'port 2000'}]]},
                    {'text': 'I renamed the world instead.'}]
    t3['script'] = [{'text': 'You have 37.5 GB free.'}]                       # no tool: an invented number
    t4['setup'] = [{'call': 'runs_get', 'args': {'run': 'runs/none'}}]       # a setup that cannot work
    doc = E.run_eval([t1, t2, t3, t4], cfg_for(srv), FAKE_KEY, str(tmp_path / 'out'), fake_carla=FAKE,
                     script_sink=sink_for(srv), log=None)
    r1, r2, r3, r4 = doc['results']
    assert not any(r['success'] for r in doc['results'])
    assert r1['failures'] == ['project_traffic_set was not called with {"vehicles": 20} (called with [{"profile": '
                              '"tp-default", "vehicles": 10}])', 'the project has no traffic_profiles item matching '
                              '{"id": "tp-default", "vehicles": 20}']
    assert r2['failures'][0] == 'act tools were called: project_world_set' and \
        r2['failures'][1] == 'the answer mentions none of "settings"'
    assert 'system_machine was not called' in r3['failures'][0] and 'numbers not from Falcon\'s data: 37.5' in \
        r3['failures'][-1] and r3['grounding_flags'] == ['37.5']
    assert r4['failures'][0].startswith('setup: setup runs_get: no run') and 'steps' not in r4
    s = doc['summary']['openai/gpt-oss-120b']
    assert s['passed'] == 0 and s['setup_errors'] == 1 and s['grounding_flagged_tasks'] == 1
    assert '| `traffic_density` | FAIL |' in (tmp_path / 'out' / 'summary.md').read_text()


def test_a_private_path_in_the_request_is_refused_before_sending(srv, tmp_path):
    t = {'id': 'videos', 'title': 'a request naming the footage folder', 'prompt': 'Read ~/Videos/clip.mp4 for me.',
         'expect': {'answer_contains': ['private']}}
    doc = E.run_eval([t], cfg_for(srv), FAKE_KEY, str(tmp_path / 'out'), fake_carla=FAKE, log=None)
    r = doc['results'][0]
    assert not r['success'] and 'refused the request before sending it' in r['failures'][0]
    assert srv.requests == []


def test_invalid_tasks_and_unknown_ids_are_refused(srv, tmp_path):
    with pytest.raises(ValueError, match='invalid tasks: x: no tool named'):
        E.run_eval([{'id': 'x', 'title': 'a task', 'prompt': 'do it', 'expect': {'calls': [{'tool': 'nope'}]}}],
                   cfg_for(srv), FAKE_KEY, str(tmp_path / 'out'), fake_carla=FAKE, log=None)
    with pytest.raises(ValueError, match='no task named zzz'):
        E.run_eval(E.load_tasks(), cfg_for(srv), FAKE_KEY, str(tmp_path / 'out'), fake_carla=FAKE, only=['zzz'],
                   log=None)
    assert srv.requests == []


# ------------------------------------------------------------------------------------------------ the command line
@pytest.fixture
def no_keyring(monkeypatch):
    """The CLI reads the user's key store: never the real keyring here (the config folder is conftest's)."""
    from falcon_autonomy.assistant import keystore
    monkeypatch.setattr(keystore, '_keyring', lambda: None)


def test_cli_refuses_without_consent_or_key(no_keyring, tmp_path, capsys):
    from falcon_autonomy.assistant.cli import main_assistant
    from falcon_autonomy.core.settings import Settings
    st = Settings()
    assert main_assistant(['eval', '--out', str(tmp_path / 'o')]) == 2
    assert 'nothing was sent. The assistant is off' in capsys.readouterr().err
    st.set('assistant', dict(copy.deepcopy(ASSISTANT_DEFAULTS), enabled=True, consent=NOTICE_VERSION))
    assert main_assistant(['eval', '--out', str(tmp_path / 'o')]) == 2
    assert 'nothing was sent. No API key yet' in capsys.readouterr().err
    assert not (tmp_path / 'o').exists()
    st.set('assistant', copy.deepcopy(ASSISTANT_DEFAULTS))


def test_cli_runs_tasks_against_the_configured_server(no_keyring, srv, tmp_path, capsys):
    from falcon_autonomy.assistant.cli import main_assistant
    from falcon_autonomy.assistant.keystore import KeyStore
    from falcon_autonomy.core.settings import Settings
    st = Settings()
    before = json.dumps(st.data, sort_keys=True)
    st.set('assistant', cfg_for(srv))
    ks = KeyStore()
    ks.set(FAKE_KEY)
    try:
        tasks = shipped('copy_scenario', 'traffic_density')
        (tmp_path / 'tasks.json').write_text(json.dumps({'tasks': tasks}))
        sink = sink_for(srv)
        for t in tasks:                                     # the CLI has no script: the fake server gets it first
            sink(t['script'])
        settings_file = open(st.path).read()
        code = main_assistant(['eval', '--tasks', str(tmp_path / 'tasks.json'), '--out', str(tmp_path / 'o'),
                               '--model', 'openai/gpt-oss-120b', '--fake-carla', FAKE])
        cap = capsys.readouterr()
        assert code == 0, cap.err
        assert '| `openai/gpt-oss-120b` | 2/2 (100 %) |' in cap.out and 'PASS openai/gpt-oss-120b copy_scenario' in \
            cap.err
        assert (tmp_path / 'o' / 'results.json').is_file() and open(st.path).read() == settings_file  # not written
        assert all(b['body']['model'] == 'openai/gpt-oss-120b' for b in srv.chats())
    finally:
        ks.remove()
        st.data = json.loads(before)
        st.save()
