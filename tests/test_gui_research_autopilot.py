"""The research autopilot in the Assistant panel, offscreen against a fake Groq server: the first card offers to approve
the study's follow-ups ahead and has no "approve all"; the Research tab shows the study's state machine; Undo cancels
its job (the study stops); Resume asks on a card even in Auto; Stop cancels the job and records the stop; a suggested
question goes to the assistant."""
import json

import pytest

pytestmark = pytest.mark.gui

from fake_groq import text, tools  # noqa: E402
from test_gui_assistant import pump, qapp, ready, slot_exceptions, srv, window  # noqa: E402,F401
from test_research_autopilot import Q, spec  # noqa: E402


def test_research_tab_cards_stop_and_undo(window, qapp, srv):
    from falcon_autonomy.core import autoresearch as R
    from falcon_autonomy.ui import assistant_dock as D
    window.assistant.set_cfg(suggestions=False)          # no fast-model wording request takes a scripted reply
    panel = ready(window, srv)
    window.assistant.set_cfg(tpm_limit=0)               # not a test of pacing: research + analysis + project loaded
    ws = window.workspace()
    old = window.queue.before_start
    window.queue.before_start = lambda job: 'held by the test'          # jobs wait: nothing runs
    try:
        srv.push(tools(('research_autopilot', {'action': 'start', 'study': 'g1', 'question': Q, 'spec': spec()}),
                       content='The plan is on the card.'),
                 text('Queued the study.'))
        assert panel.send(Q)
        assert pump(qapp, lambda: panel.approvals, 15), panel.status.text()
        card = next(iter(panel.approvals.values()))
        assert isinstance(card, D.ApprovalCard) and card.all is None and set(card.boxes) == {'follow_ups'}
        assert 'Research study g1' in card.req.preview and 'queues compute' in card.req.why
        panel.tabs.setCurrentIndex(2)                                    # the Research tab: proposed, from the card
        panel.research.refresh()
        assert panel.research.rows[0]['state'] == 'proposed' and panel.research.rows[0]['study'] == 'g1'
        assert '[proposed]' in panel.research.chain.text()
        card.approve.click()
        assert pump(qapp, lambda: not panel.agent.busy, 20)
        rec = R.load(ws, 'g1')
        assert rec['rounds'][0]['approval']['how'] == 'approved' and not rec['follow_ups']
        job = window.queue.jobs[-1]
        assert job.spec.meta['study'] == 'g1' and job.state == 'waiting'
        panel.research.refresh()
        assert panel.research.rows[0]['state'] == 'running' and '[running]' in panel.research.chain.text()
        assert panel.research.undo_btn.isEnabled() and panel.research.stop_btn.isEnabled()
        # Undo: the activity log's entry for the study's job — cancelled, the study stopped
        assert panel.research.undo().startswith('Cancelled job')
        assert job.state == 'cancelled' and panel.research.rows[0]['state'] == 'stopped'
        # Resume asks on a card even in Auto (it queues compute), then Stop cancels and records it
        panel.mode.setCurrentIndex(1)
        assert panel.agent.mode == 'auto'
        rc = panel.research.resume()
        assert isinstance(rc, D.ApprovalCard) and 'resume g1' in rc.req.preview
        rc.approve.click()
        assert pump(qapp, lambda: window.queue.jobs[-1] is not job, 15)
        assert pump(qapp, lambda: panel.research.refresh() or panel.research.rows[0]['state'] == 'running', 10)
        msg = panel.research.stop()
        assert msg.startswith('Stopped g1 (cancelled job') and window.queue.jobs[-1].state == 'cancelled'
        assert R.load(ws, 'g1')['stopped']['by'] == 'user' and panel.research.rows[0]['state'] == 'stopped'
        assert 'Follow-up' not in panel.research.detail.text() and 'stopped' in panel.research.detail.text()
        # a suggested question goes to the assistant as a request
        srv.push(text('I will look at it.'))
        panel.research.show_suggestions([{'id': 'cell:rain/junction/car/40-60/night', 'question': 'Is it worse?',
                                          'note': ''}])
        assert panel.research.sug.topLevelItemCount() == 1
        panel.research.study_this(panel.research.suggested[0])
        assert pump(qapp, lambda: not panel.agent.busy and len(srv.chats()) >= 3, 15), panel.status.text()
        sent = srv.chats()[-1]['body']['messages']
        assert sent[-1]['content'].startswith('Make this a research study: Is it worse?')
        assert json.dumps(sent).count('cell:rain/junction/car/40-60/night') >= 1
    finally:
        window.queue.before_start = old
