"""Suggestions, "Ask about this", "Explain this error" and show-me in the window, offscreen (a fake Groq for the model):
the chips on the pages and Home's next best action offline, a click opening an approval card, dismissing, turning
them off, the model's wording; a failed job's Explain button in the Jobs panel and in its log window, with the
obvious next step; "Ask about this" on a run (left in the box until the assistant is ready, private runs refused);
the show-me toggle; and the new parts fitting a 1280 px window in both themes (screenshots kept in
$FALCON_SHOTS when set)."""
import json
import os
import sys
import time

import pytest

pytestmark = pytest.mark.gui

from fake_groq import FAKE_KEY  # noqa: E402
from test_gui_assistant import pump, qapp, ready, slot_exceptions, srv, window  # noqa: E402,F401


def _quick(content, model='qwen/qwen3.8-27b'):
    return {'json': {'model': model, 'choices': [{'message': {'role': 'assistant', 'content': content},
                                                  'finish_reason': 'stop'}],
                     'usage': {'prompt_tokens': 90, 'completion_tokens': 40, 'total_tokens': 130}}}


def _settle(win, qapp):
    """A gather now, and its result delivered."""
    from falcon_autonomy.ui import tasks
    win.suggestions.pacer.last_gather = None
    win.suggestions.refresh(forced=True)
    assert tasks.wait_all(20)
    assert pump(qapp, lambda: not win.suggestions.busy, 10)
    pump(qapp, timeout=0.1)


def _fail_job(win, qapp, text='the rig has no microphone array (audio_array)'):
    from falcon_autonomy.core.jobs import JobSpec
    code = f'import sys; print({text!r}, file=sys.stderr); sys.exit(2)'
    job = win.queue.submit(JobSpec('derive', 'Synthetic audio of drive-1', [sys.executable, '-c', code]))
    assert pump(qapp, lambda: job.state == 'failed', 20), job.state
    pump(qapp, timeout=0.2)
    return job


def _shot(widget, name):
    d = os.environ.get('FALCON_SHOTS')
    if d:
        os.makedirs(d, exist_ok=True)
        widget.grab().save(os.path.join(d, name))


def test_chips_and_home_card_offline_click_dismiss_and_off(window, qapp):
    from falcon_autonomy.ui import assistant_dock as D
    from falcon_autonomy.ui import suggestions as SG
    eng = window.suggestions
    assert set(window.suggestion_bars) == set(SG.BAR_PAGES)
    _settle(window, qapp)
    keys = [s.key for s in eng.cands]
    assert 'first_drive:project' in keys and 'no_scenarios:project' in keys and 'no_rules:project' in keys
    home = window.go('home')
    pump(qapp, timeout=0.1)
    assert home.next_kind.text() == 'Next best action' and home.next_head.text() == 'Record a first 10 s drive'
    assert home.next_btn.text() == 'Review…' and home.next_dismiss.isVisibleTo(home)
    data = window.suggestion_bars['data']
    assert data.isVisibleTo(window.page('data')) and [c.s.key for c in data.chips] == ['first_drive:project']
    chip = data.chips[0]
    assert 'Nothing is recorded in this project yet' in chip.btn.toolTip() and 'jobs_drive' in chip.btn.toolTip()
    scn = window.suggestion_bars['scenarios']
    assert {c.s.trigger for c in scn.chips} == {'no_scenarios', 'no_rules'} and len(scn.chips) <= 3
    window.go('data')
    chip.btn.click()                                          # Ask mode: an approval card, nothing runs yet
    panel = window.assistant_panel
    assert window.assistant_dock.isVisible() and panel.pages.currentIndex() == 1
    cards = panel.findChildren(D.ApprovalCard)
    assert cards and cards[-1].req.tool == 'jobs_drive' and 'record a drive' in cards[-1].req.preview
    assert window.queue.jobs == []
    cards[-1].reject.click()
    rule = next(c for c in scn.chips if c.s.trigger == 'no_rules')
    rule.x.click()                                            # dismissed for this project, and kept
    pump(qapp, timeout=0.1)
    assert 'no_rules:project' not in [c.s.key for c in scn.chips]
    with open(os.path.join(window.workspace(), 'assistant', 'suggestions.json'), encoding='utf-8') as fh:
        assert 'no_rules:project' in json.load(fh)['dismissed']
    home.next_dismiss.click()
    pump(qapp, timeout=0.1)
    assert home.next_head.text() != 'Record a first 10 s drive'
    window.assistant.set_cfg(suggestions=False)
    eng.refresh()
    pump(qapp, timeout=0.1)
    assert not data.isVisibleTo(window.page('data')) and not scn.chips and home.next_kind.text() == 'Next step'


def test_model_wording_on_chips_and_home(window, qapp, srv):
    ready(window, srv)
    window.assistant.set_cfg(fast_model='qwen/qwen3.8-27b')
    srv.push(_quick(json.dumps({'suggestions': [
        {'key': 'no_scenarios:project', 'text': 'Copy lead_brake into an editable scenario to search it.'},
        {'key': 'first_drive:project', 'text': 'Record 10 s in Town 10 with the default rig to start a dataset.'}]})))
    _settle(window, qapp)
    eng = window.suggestions
    assert pump(qapp, lambda: not eng.phrasing and any(s.phrased for s in eng.cands), 10), eng.last_error
    first = eng.best()
    assert first.key == 'no_scenarios:project' and first.phrased                # the model ranked it first
    home = window.go('home')
    assert 'worded by the assistant' in home.next_kind.text() and home.next_why.text().startswith('Copy lead_brake')
    assert srv.chats()[-1]['body']['model'] == 'qwen/qwen3.8-27b'
    n = len(srv.chats())
    _settle(window, qapp)                                     # nothing new: the model is not asked again
    pump(qapp, timeout=0.3)
    assert len(srv.chats()) == n


def test_explain_a_failed_job_in_the_jobs_panel_and_its_log(window, qapp, srv, monkeypatch):
    from falcon_autonomy.ui import assistant_dock as D
    window.suggestions.refresh(forced=True)
    job = _fail_job(window, qapp)                             # its end brings a gather forward (after the pause)
    assert pump(qapp, lambda: any(s.key == f'job_failed:{job.id}' for s in window.suggestions.cands), 10)
    assert window.go('home').next_head.text() == f'Look into failed job {job.id}'
    dock = window.jobs_dock.widget()
    acts = dock.actions(job.id)
    assert acts.explain.isVisibleTo(acts) and acts.main.text() == 'Open'
    acts.explain.click()                                      # the assistant is off: worded from the log
    panel = window.assistant_panel
    assert pump(qapp, lambda: panel.findChildren(D.ExplainCard) and
                panel.findChildren(D.ExplainCard)[-1].result is not None, 10)
    card = panel.findChildren(D.ExplainCard)[-1]
    assert 'microphone array' in card.body.raw and 'no model was asked' in card.foot.text()
    assert 'The last error: the rig has no microphone array' in card.body.raw   # what the job said on stderr
    assert card.btn.isVisibleTo(card) and card.btn.text() == 'Add a microphone array to the rig…'
    card.btn.click()
    appr = panel.findChildren(D.ApprovalCard)[-1]
    assert appr.req.tool == 'project_sensor_add' and appr.req.args == {'type': 'audio_array', 'rig': 'rig-default'}
    appr.approve.click()
    assert pump(qapp, lambda: any(s.get('type') == 'audio_array' for s in window.project.doc['rigs'][0]['sensors']),
                10)
    window.undo.undo()                                        # one undo takes it back
    assert not any(s.get('type') == 'audio_array' for s in window.project.doc['rigs'][0]['sensors'])
    m = dock._menu(dock.tree.visualItemRect(dock.items[job.id]).center())
    assert [a.text() for a in m.actions() if a.text()] == ['Open', 'Log', 'Ask about this', 'Explain this error']
    m.close()
    ready(window, srv)                                        # now from the fast model, from the log window
    srv.push(_quick('Cause: the rig has no microphone array.\nFix: add an audio_array sensor.\nAction: audio_array'))
    d = dock.show_log(job)
    btn = next(b for b in d.findChildren(type(acts.explain)) if b.text() == 'Explain this error')
    btn.click()
    assert pump(qapp, lambda: panel.findChildren(D.ExplainCard)[-1].result is not None, 10)
    card = panel.findChildren(D.ExplainCard)[-1]
    assert card.result['model'] == 'qwen/qwen3.8-27b' and 'Worded by' in card.foot.text()
    assert card.cta['key'] == 'audio_array' and 'Action:' not in card.body.raw
    body = json.dumps(srv.chats()[-1]['body'])
    assert 'microphone array' in body and FAKE_KEY not in body


def test_ask_about_a_run_waits_in_the_box_and_private_runs_are_refused(window, qapp, srv):
    from falcon_autonomy.ui import assistant_dock as D
    ws = window.workspace()
    r = os.path.join(ws, 'runs', 'drive-a')
    os.makedirs(r)
    with open(os.path.join(r, 'run.json'), 'w') as fh:
        json.dump({'name': 'drive-a', 'status': 'ok'}, fh)
    home = window.go('home')
    t = home.runs._ask_target
    from PyQt6.QtWidgets import QTreeWidgetItem
    it = QTreeWidgetItem(['x', 'drive-a', 'ok'])
    it.setData(0, 256, r)
    kind, ids = t(it)
    assert window.ask_about(kind, **ids) is False                # not set up: left in the box
    panel = window.assistant_panel
    assert 'runs/drive-a' in panel.composer.toPlainText() and panel.pending_context == {'about': 'run',
                                                                                         'run': 'runs/drive-a'}
    ready(window, srv)
    from fake_groq import text
    srv.push(text('It recorded nothing unusual.'))
    assert panel.send()
    assert pump(qapp, lambda: not panel.agent.busy, 10)
    sent = srv.chats()[-1]['body']['messages'][-1]['content']
    assert '"run": "runs/drive-a"' in sent and ws not in sent
    priv = os.path.join(ws, 'runs', 'route-b')
    os.makedirs(priv)
    with open(os.path.join(priv, 'run.json'), 'w') as fh:
        json.dump({'name': 'route-b', 'provenance': {'settings': {'dashcam_route': {'clip': 'q'}}}}, fh)
    n = len(srv.chats())
    assert window.ask_about('run', run=priv) is False
    assert any(b.kind == 'error' and 'private' in b.raw for b in panel.findChildren(D.Bubble))
    assert len(srv.chats()) == n


def test_show_me_toggle(window, qapp, srv):
    panel = ready(window, srv)
    assert not panel.show_btn.isChecked() and not panel.agent.show_me
    panel.show_btn.click()
    assert panel.agent.show_me and window.assistant.cfg()['show_me'] and 'Show-me mode on' in panel.status.text()
    panel.load_session()                                      # kept across conversations (a setting)
    assert panel.agent.show_me and panel.show_btn.isChecked()
    panel.show_btn.click()
    assert not panel.agent.show_me and not window.assistant.cfg()['show_me']


@pytest.mark.parametrize('theme', ['dark', 'light'])
def test_the_new_parts_fit_1280_in_both_themes(window, qapp, theme):
    from falcon_autonomy.ui import assistant_dock as D
    window.set_theme(theme)
    window.resize(1280, 800)
    pump(qapp, timeout=0.2)
    assert window.width() <= 1280
    _settle(window, qapp)
    job = _fail_job(window, qapp)
    for key in ('home', 'data', 'scenarios', 'rarecases', 'stack'):
        page = window.go(key)
        pump(qapp, timeout=0.15)
        _shot(window, f'{theme}-{key}.png')
        bar = window.suggestion_bars.get(key)
        if bar is not None and bar.chips:
            for c in bar.chips:                               # every chip inside the page, none clipped
                g = c.geometry()
                assert g.right() <= bar.width() and c.width() >= min(c.sizeHint().width(), bar.width() - 4), key
            assert bar.mapTo(window, bar.rect().topRight()).x() <= window.width()
        if key == 'home':
            card = page.next_card
            assert card.mapTo(window, card.rect().topRight()).x() <= window.width()
            assert page.next_btn.isVisible() and page.next_btn.width() >= page.next_btn.sizeHint().width() - 2
    acts = window.jobs_dock.widget().actions(job.id)
    assert all(b.width() >= b.sizeHint().width() - 2 for b in (acts.main, acts.log, acts.explain))
    acts.explain.click()
    panel = window.assistant_panel
    assert pump(qapp, lambda: panel.findChildren(D.ExplainCard)[-1].result is not None, 10)
    pump(qapp, timeout=0.3)
    _shot(window, f'{theme}-explain.png')
    _shot(window.assistant_dock, f'{theme}-explain-panel.png')
    card = panel.findChildren(D.ExplainCard)[-1]
    assert card.width() <= panel.width() and card.btn.width() >= card.btn.sizeHint().width() - 2
    top = panel.show_btn.mapTo(panel, panel.show_btn.rect().topRight()).x()
    assert top <= panel.width() and window.width() <= 1280
    window.set_theme('dark')
