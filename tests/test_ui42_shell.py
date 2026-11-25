"""4.2 window, assistant and worlds (offscreen): focus mode hides the navigator, the docks and the pages' side panes and
puts them back; the context pills follow win.context and clear; the top bar fits 1280 px; the Assistant panel's run
and time links, its verified numbers and the not-a-live-model banner (pure functions and the panel against the fake
Groq server); the Assistant dock floating over a narrow window; the import gaps of a synthetic OpenStreetMap area
(core.importgaps) and their Worlds inspector; the command palette's moments; the command blocks."""
import json
import math
import os

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget

pytestmark = pytest.mark.gui

from fake_groq import FAKE_KEY, FakeGroq, text, tools  # noqa: E402
from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401  (fixtures)

LAT0, LON0 = 10.0, 20.0                          # a synthetic place
KX, KY = 111319.49 * math.cos(math.radians(LAT0)), 110574.0


def ll(x, y):
    return LAT0 + y / KY, LON0 + x / KX


def synthetic_osm(path):
    """Two crossing residential streets (ways 10, 11), a bridge (13, layer 1) and a tunnel (14, layer -1) crossing
    them without shared nodes, a footway (15) and a service road (16) the converter drops, a building across East
    Street (100), a roof over North Street (102) and a building well away from every road (101)."""
    nodes = {1: (-200, 0), 21: (-100, 0), 2: (0, 0), 22: (100, 0), 3: (200, 0), 4: (0, -200), 7: (0, 100),
             5: (0, 200), 30: (-150, -150), 31: (150, 150), 40: (-150, 60), 41: (150, 60), 50: (-80, -80),
             51: (-60, -120), 52: (80, -80), 53: (120, -120),
             60: (50, -4), 61: (70, -4), 62: (70, 12), 63: (50, 12),
             64: (-180, 140), 65: (-160, 140), 66: (-160, 160), 67: (-180, 160),
             68: (-4, -110), 69: (4, -110), 70: (4, -90), 71: (-4, -90)}
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6">']
    for n, (x, y) in sorted(nodes.items()):
        la, lo = ll(x, y)
        lines.append(f'<node id="{n}" lat="{la:.8f}" lon="{lo:.8f}"/>')

    def way(i, refs, tags):
        lines.append(f'<way id="{i}">' + ''.join(f'<nd ref="{r}"/>' for r in refs) +
                     ''.join(f'<tag k="{k}" v="{v}"/>' for k, v in tags.items()) + '</way>')
    way(10, [1, 21, 2, 22, 3], {'highway': 'residential', 'name': 'East Street', 'lanes': '2'})
    way(11, [4, 2, 7, 5], {'highway': 'residential', 'name': 'North Street', 'lanes': '2'})
    way(13, [30, 31], {'highway': 'primary', 'bridge': 'yes', 'layer': '1', 'name': 'Flyover', 'lanes': '2'})
    way(14, [40, 41], {'highway': 'tertiary', 'tunnel': 'yes', 'layer': '-1', 'name': 'Underpass', 'lanes': '2'})
    way(15, [50, 51], {'highway': 'footway'})
    way(16, [52, 53], {'highway': 'service'})
    way(100, [60, 61, 62, 63, 60], {'building': 'yes'})
    way(101, [64, 65, 66, 67, 64], {'building': 'yes'})
    way(102, [68, 69, 70, 71, 68], {'building': 'roof'})
    lines.append('<relation id="900"><member type="way" ref="10" role="from"/><tag k="type" v="restriction"/>'
                 '</relation>')
    lines.append('</osm>')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines))
    return path


# ----------------------------------------------------------------------------------------------- import gaps (pure)
def test_import_gaps_of_a_synthetic_area(tmp_path):
    from falcon_autonomy.core import importgaps as G
    from falcon_autonomy.core import osm as O
    data = O.read_osm(synthetic_osm(str(tmp_path / 'a.osm')))
    g = G.compute(data, centre=(LAT0, LON0))
    by = {i['key']: i for i in g['items']}
    assert by['elevation']['count'] == 2 and by['elevation']['detail']['bridge'] == 1 and \
        by['elevation']['detail']['tunnel'] == 1 and 'flattened' in by['elevation']['handled']
    assert by['buildings_on_road']['count'] == 2 and by['buildings_on_road']['detail']['raised_or_roof'] == 1
    assert sorted(by['buildings_on_road']['detail']['osm_ids']) == [100, 102]
    assert by['roads_dropped']['count'] == 2 and by['roads_dropped']['detail']['by_type'] == {'footway': 1,
                                                                                              'service': 1}
    assert by['relations']['count'] == 1 and by['cut_ways']['count'] == 0
    assert all(i['tip'] and i['title'] for i in g['items'])
    # converting service roads too: one fewer dropped
    g2 = G.compute(data, O.CARLA_HIGHWAYS + ('service',), (LAT0, LON0))
    assert {i['key']: i for i in g2['items']}['roads_dropped']['count'] == 1
    # the city and the speed breakers, from their own reports
    no_city = {i['key']: i for i in G.with_city(g, None, 3)['items']}
    assert no_city['buildings_not_placed']['count'] == 3 and 'no city' in no_city['buildings_not_placed']['handled']
    city = {'plan': 'city/plan.json', 'features': {'buildings': 3},
            'summary': {'by_kind': {'building': {'osm': 1}, 'tree': {'osm': 4}},
                        'skipped': {'building': {'would cover a lane or sidewalk': 2},
                                    'tree': {'in a building': 1}}}}
    full = G.with_features(G.with_city(g, city), {'skipped': [{'x': 1}], 'not_represented': {'rumble_strip': 2}})
    by = {i['key']: i for i in full['items']}
    assert by['buildings_not_placed']['count'] == 2 and by['props_not_placed']['count'] == 1
    assert by['speed_breakers_skipped']['count'] == 3
    assert 'would cover a lane' in G.detail_text(by['buildings_not_placed'])
    assert '2 rumble_strip' in G.detail_text(by['speed_breakers_skipped'])


# ----------------------------------------------------------------------------------------------- the window
def test_focus_mode_hides_and_restores(window, qapp):
    from PyQt6.QtWidgets import QSplitter
    for key in ('scenarios', 'rigs', 'models', 'data', 'live', 'training'):    # a page showing a split view
        page = window.go(key)
        pump(qapp, timeout=0.2)
        splits = [s for s in page.findChildren(QSplitter) if s.isVisibleTo(page) and s.count() > 1 and
                  sum(s.widget(i).isVisibleTo(page) for i in range(s.count())) > 1]
        if splits:
            break
    assert splits, 'no page shows a split view'
    before = {id(s): s.sizes() for s in splits}
    vis_before = {id(s): [s.widget(i).isVisibleTo(page) for i in range(s.count())] for s in splits}
    window.show_jobs()
    pump(qapp, timeout=0.1)
    assert window.jobs_dock.isVisible()
    a = window.actions_['focus']
    assert a.shortcut().toString() == 'Ctrl+Shift+F' and a.isCheckable()
    assert window.actions_['focus_exit'].shortcut().toString() == 'Esc' and not window.actions_['focus_exit'].isEnabled()
    window.header.focus_btn.click()
    pump(qapp, timeout=0.2)
    assert window.focus_mode and a.isChecked() and window.header.focus_btn.isChecked()
    assert not window.side.isVisible() and not window.jobs_dock.isVisible() and not window.log_dock.isVisible()
    assert not window.assistant_dock.isVisible() and window.statusBar().height() <= 4
    assert not window.header.about.isVisible() and window.header.title.isVisible()
    assert window.actions_['focus_exit'].isEnabled()
    outer = splits[0]
    assert sum(outer.widget(i).isVisibleTo(page) for i in range(outer.count())) == 1
    kept = next(i for i in range(outer.count()) if outer.widget(i).isVisibleTo(page))
    assert before[id(outer)][kept] == max(before[id(outer)][i] for i in range(outer.count()) if
                                          vis_before[id(outer)][i])
    window.show_jobs()                                      # a job meanwhile: shown on leaving, not now
    pump(qapp, timeout=0.05)
    assert not window.jobs_dock.isVisible()
    window.go('worlds')                                     # another page in focus mode: this one's panes come back
    pump(qapp, timeout=0.1)
    assert all(outer.widget(i).isVisibleTo(page) == vis_before[id(outer)][i] for i in range(outer.count()))
    window.go(key)
    pump(qapp, timeout=0.1)
    window.actions_['focus_exit'].trigger()                 # Esc
    pump(qapp, timeout=0.3)
    assert not window.focus_mode and window.side.isVisible() and window.jobs_dock.isVisible()
    assert window.statusBar().height() > 4 and window.header.about.isVisible()
    for s in splits:
        assert [s.widget(i).isVisibleTo(page) for i in range(s.count())] == vis_before[id(s)]
        assert sum(abs(x - y) for x, y in zip(s.sizes(), before[id(s)])) <= 4 * s.count()
    window.actions_['focus'].trigger()                      # Ctrl+Shift+F twice: on and off
    assert window.focus_mode
    window.actions_['focus'].trigger()
    assert not window.focus_mode


def test_context_pills_follow_the_context_and_clear(window, qapp, tmp_path):
    top = window.top
    assert not top.pills.isVisible()
    run = tmp_path / 'runs' / '20261125-101112-drive-town10-seed3'
    run.mkdir(parents=True)
    window.context.set(run=str(run), seed=3, world='w-default')
    pump(qapp, timeout=0.1)
    pills = top.pills.pills
    assert top.pills.isVisible() and set(pills) == {'run', 'seed', 'world'}
    assert pills['run'].text.full.startswith('Run ') and '…' in pills['run'].text.full     # a short id
    assert str(run) in pills['run'].toolTip() and pills['seed'].text.full == 'Seed 3'
    order = [top.pills.layout().indexOf(pills[k]) for k in ('run', 'seed', 'world')]
    assert order == sorted(order)
    opened = []
    data = window.page('data')
    data.show_run = lambda p: opened.append(p)
    top.pills.open('run')
    assert opened == [str(run)] and window.current_key() == 'data'
    pills['seed'].close_btn.click()
    pump(qapp, timeout=0.1)
    assert window.context.get('seed') == '' and 'seed' not in top.pills.pills
    window.context.clear()
    pump(qapp, timeout=0.1)
    assert not top.pills.pills and not top.pills.isVisible()


def test_top_bar_fits_1280(window, qapp, tmp_path):
    top = window.top
    window.context.set(run=str(tmp_path / 'runs' / ('20261125-101112-a-very-long-run-name-' + 'x' * 40)), seed=12345,
                       rig='rig-with-a-long-name-for-the-roof-rack', scenario='cut_in_rain_at_dusk_with_traffic',
                       world='bengaluru-central-imported-2026')
    for w in (1280, 1440, 1920):
        window.resize(w, 800)
        pump(qapp, timeout=0.3)
        assert top.overflow() == 0, w
        kids = sorted((c for c in top.children() if isinstance(c, QWidget) and c.isVisible()),
                      key=lambda c: c.geometry().left())
        for a, b in zip(kids, kids[1:]):                         # in a row, none over the next, all inside
            assert a.geometry().right() < b.geometry().left(), (w, a.objectName(), b.objectName())
        assert kids[0].geometry().left() >= 0 and kids[-1].geometry().right() <= top.width(), w
        assert top.assistant.geometry().right() <= top.width()
    window.resize(1280, 800)
    pump(qapp, timeout=0.3)
    path = top.path
    assert path.toolTip() and path.full == path.toolTip()
    if path.fontMetrics().horizontalAdvance(path.full) > path.width():
        assert '…' in path.text() and path.text()[-2:] == path.full[-2:]          # cut in the middle
    shown = [p for p in top.pills.pills.values() if p.isVisible()]
    assert shown and all(p.text.width() <= p.text.natural() for p in shown)
    folded = len(top.pills.pills) - len(shown)
    assert (top.pills.more.text() == f'+{folded}') if folded else not top.pills.more.isVisible()
    window.resize(1440, 900)
    pump(qapp, timeout=0.2)


# ----------------------------------------------------------------------------------------------- the assistant
def test_assistant_facts_pure():
    from falcon_autonomy.ui import assistant_facts as F
    tools_ = [('runs_count', json.dumps({'runs': 7, 'ap': 0.4213})), ('runs_list', {'items': [1, 2, 3, 4]})]
    v = F.verified_numbers('7 runs, AP 0.42 (42.1 %), 4 items, 9 others, version 0.9.16, 2 options', tools_)
    assert v == {'7': 'runs_count', '0.42': 'runs_count', '42.1 %': 'runs_count', '4': 'runs_list'}
    assert F.verified_numbers('7 runs', []) == {}
    spans, last = F.annotate('see runs/20261125-a at t = 12.5 s and job 3, then 2.0–4.5 s', jobs={3},
                             verified={'12.5': 'x'})
    kinds = [(k, v) for _a, _b, k, v in spans]
    assert kinds == [('run', '20261125-a'), ('time', ('20261125-a', '12.5')), ('job', '3'),
                     ('time', ('20261125-a', '2.0'))] and last == '20261125-a'
    spans, _ = F.annotate('at 3.0 s nothing is open', ())             # a time with no run named: plain
    assert spans == []
    spans, _ = F.annotate('r_7 had 0.5 recall &#39;x&#39; and 39', runs={'r_7': '/w/runs/r_7'},
                          verified={'0.5': 'runs_count'}, ungrounded=['39'])
    assert [(k, v) for _a, _b, k, v in spans] == [('run', 'r_7'), ('verified', 'runs_count'), ('ungrounded', '')]
    s = 'r_7 had 0.5 recall &#39;x&#39; and 39'
    assert s[spans[-1][0]:spans[-1][1]] == '39' and spans[-1][0] > s.index('and')     # entities never cut
    assert F.liveness(True, True, True) is None
    assert F.liveness(True, True, False)[0] == 'no_key' and 'missing' in F.liveness(True, True, False)[1]
    assert F.liveness(True, True, True, last_error='auth')[0] == 'refused'
    assert F.liveness(False, True, False, scripted=True)[1].startswith('Not a live model')
    assert F.liveness(False, True, False) is None                     # off: the empty state says so
    msgs = [{'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'tool_calls': [{'id': 'c1', 'function': {'name': 'runs_count'}}]},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': '{"runs": 7}'},
            {'role': 'assistant', 'content': '7 runs'},
            {'role': 'user', 'content': 'again'}, {'role': 'assistant', 'content': 'no tools'}]
    assert F.turn_tools(msgs, 3) == [('runs_count', '{"runs": 7}')] and F.turn_tools(msgs, 5) == []


@pytest.fixture
def srv():
    s = FakeGroq().start()
    yield s
    s.stop()


@pytest.fixture
def reg(monkeypatch):
    from falcon_autonomy.assistant import tool_api as T
    monkeypatch.setattr(T, 'REGISTRY', {})
    return T.REGISTRY


@pytest.fixture
def keys(window, tmp_path):
    from falcon_autonomy.assistant.keystore import KeyStore
    from falcon_autonomy.core.settings import Secrets
    window.assistant.keys = KeyStore(Secrets(str(tmp_path / 'secrets.json')), use_keyring=False)
    window.assistant.claude_keys = KeyStore(Secrets(str(tmp_path / 'secrets2.json')), use_keyring=False)
    return window.assistant.keys


def test_assistant_links_verified_numbers_and_banner(window, qapp, srv, reg, keys, tmp_path):
    from falcon_autonomy.assistant import tool_api as T
    from falcon_autonomy.ui import assistant_dock as D
    T.register(T.Tool('runs_count', 'runs', 'read', 'count runs', {'type': 'object', 'properties': {}},
                      lambda ctx: {'runs': 7, 'path': 'runs/20261125-abc'}))
    svc = window.assistant
    svc.set_cfg(enabled=True, base_url=srv.base_url)
    svc.accept_notice()
    panel = window.assistant_panel
    panel.refresh_state()
    assert not panel.live_banner.isHidden() and 'key for' in panel.live_banner.text()       # no key yet
    keys.set(FAKE_KEY)
    svc.scripted = True                                          # the fake server: a test double
    window.show_assistant(True)
    pump(qapp, timeout=0.2)
    assert not panel.live_banner.isHidden() and panel.live_banner.text().startswith('Not a live model')
    srv.push(tools(('runs_count', {}), content=''),
             text('There are 7 runs; runs/20261125-abc stops at t = 12.5 s with an AP of 0.42.'))
    assert panel.send('how many runs?')
    assert pump(qapp, lambda: not panel.agent.busy, 10)
    pump(qapp, timeout=0.2)
    b = [x for x in panel.findChildren(D.Bubble) if x.kind == 'assistant' and x.raw.strip()][-1]
    assert b.verified == {'7': 'runs_count'} and '0.42' in b.ungrounded and '7' not in b.ungrounded
    html = b.text()
    assert 'falcon://fact/runs_count' in html and 'falcon://run/20261125-abc?t=12.5' in html
    assert 'falcon://run/20261125-abc"' in html and 'Tools this turn: runs_count' in html
    assert 'Not from Falcon' in html                             # the unverified number is still flagged
    card = [c for c in panel.findChildren(D.ToolCard)][-1]
    assert card.head.text().strip().startswith('runs_count()') and card.line.text().startswith('→ ')
    assert D.short_args('{"name": "x", "n": 3}') == 'name="x", n=3'
    opened, seek = [], []
    data = window.page('data')
    data.show_run = lambda p: opened.append(p)
    run = os.path.join(window.workspace(), 'runs', '20261125-abc')
    os.makedirs(run, exist_ok=True)

    class _B:
        def seek_time(self, t):
            seek.append(t)
    data.browse, real = _B(), data.browse
    try:
        panel.navigate('falcon://run/20261125-abc?t=12.5')
        assert opened == [run] and seek == [12.5] and window.context.get('run') == run
        panel.navigate('falcon://fact/runs_count')
        assert card.head.isChecked() and card.body.isVisibleTo(card)
    finally:
        data.browse = real
    # the rebuilt conversation marks the same (the turn's tools are read back from the session)
    panel.load_session()
    b2 = [x for x in panel.findChildren(D.Bubble) if x.kind == 'assistant' and x.raw.strip()][-1]
    assert b2.verified == {'7': 'runs_count'}
    svc.scripted = False
    panel.refresh_state()
    assert panel.live_banner.isHidden()
    panel.on_event('error', {'kind': 'auth', 'message': 'The server refused the API key (401).'})
    assert not panel.live_banner.isHidden() and 'refused' in panel.live_banner.text()
    assert not panel.grab().isNull()


def test_assistant_dock_floats_over_a_narrow_window(window, qapp, srv, keys):
    from falcon_autonomy.ui import assistant_dock as D
    svc = window.assistant
    svc.set_cfg(enabled=True, base_url=srv.base_url)
    svc.accept_notice()
    keys.set(FAKE_KEY)
    dock = window.assistant_dock
    window.resize(1700, 900)
    pump(qapp, timeout=0.2)
    window.show_assistant(True)
    pump(qapp, timeout=0.3)
    assert dock.isVisible() and not dock.isFloating() and not D.squeezed(window)
    window.resize(1280, 800)              # the pages would get less than PAGES_MIN: an overlay over them instead
    assert pump(qapp, lambda: dock.isFloating(), 2) and dock._auto_floated
    pump(qapp, timeout=0.2)
    window.resize(1260, 800)                                     # floating, it no longer holds the window wide
    pump(qapp, timeout=0.2)
    assert window.width() == 1260 and dock.isFloating()
    assert window.frameGeometry().contains(dock.geometry().topLeft())
    window.resize(1700, 900)                                     # room again: docked
    assert pump(qapp, lambda: not dock.isFloating(), 2)
    dock.hide()
    pump(qapp, timeout=0.1)


# ----------------------------------------------------------------------------------------------- worlds
def test_worlds_inspector_shows_import_gaps(window, qapp, tmp_path):
    from falcon_autonomy.core.jobs import FINAL
    from falcon_autonomy.core.project import Edit
    from falcon_autonomy.core.worlds import import_config, world_dir, world_entry
    from falcon_autonomy.ui import import_gaps as IG
    from falcon_autonomy.ui import tasks
    src = synthetic_osm(str(tmp_path / 'syn.osm'))
    cfg = import_config(src, world_dir(window.workspace(), 'w-syn'), None, True, centre=(LAT0, LON0), city=False)
    job = window.service.submit_osm_import(cfg)
    assert pump(qapp, lambda: job.state in FINAL, 60) and job.state == 'done', job.log
    with open(os.path.join(cfg['out'], 'import.json')) as fh:
        rep = json.load(fh)
    by = {i['key']: i['count'] for i in rep['import_gaps']['items']}       # recorded by the import
    assert by['elevation'] == 2 and by['buildings_on_road'] == 2 and by['roads_dropped'] == 2
    window.edit(Edit('insert', ['worlds'], world_entry('w-syn', 'Synthetic', cfg, rep), text='import'))
    page = window.go('worlds')
    page.list.setCurrentRow(page.list.count() - 1)
    pump(qapp, timeout=0.3)
    g = page.gaps
    assert g.count('elevation') == '2' and g.count('buildings_on_road') == '2' and g.count('roads_dropped') == '2'
    assert g.count('buildings_not_placed') == '3'                # no city built: every footprint absent
    assert 'bridge' in g.rows['elevation'][1].toolTip() and 'flat' in g.rows['elevation'][0].toolTip()
    assert not g.state.isVisible()
    kv = page.place_kv.vals
    assert len(kv['Road network'].text()) < 80 and kv['Not in the world'].toolTip()
    assert page.place_details.toggle_btn.text().startswith('Notes and method') and not page.place_details.expanded
    # a world imported before 4.2: counted from its area.osm in the background
    rep.pop('import_gaps')
    with open(os.path.join(cfg['out'], 'import.json'), 'w') as fh:
        json.dump(rep, fh)
    IG._cache.clear()
    page.show_world()
    assert 'Counting' in g.state.text()
    assert tasks.wait_all(30)
    assert pump(qapp, lambda: g.count('elevation') == '2', 5) and 'before 4.2' in g.state.text()
    os.remove(os.path.join(cfg['out'], 'area.osm'))
    IG._cache.clear()
    page.show_world()
    assert 'Not recorded for this world' in g.state.text() and not g.rows
    assert not page.place.grab().isNull()


# ----------------------------------------------------------------------------------------------- palette, commands
def test_palette_moments_and_command_blocks(window, qapp, monkeypatch, tmp_path):
    from falcon_autonomy.core import moments as M
    from falcon_autonomy.ui import search
    from falcon_autonomy.ui.palette import CommandPalette
    from falcon_autonomy.ui.widgets import CommandBlock
    run = str(tmp_path / 'r1')
    hit = {'id': 'm1', 'type': 'cut_in', 'actors': ['car'], 'min_ttc_s': 1.2, 'run': run, 'rel': 'runs/r1',
           't0': 4.5, 't1': 6.0, 'source': 'run', 'match': 1.0, 'missed': []}
    asked = []

    def fake_search(ws, flt, limit=50, strict=False, public_only=False):
        asked.append(flt)
        return {'hits': [hit], 'total': 1, 'full': 1}
    monkeypatch.setattr(M, 'search', fake_search)
    assert search.moment_entries(window, 'settings') == [] and not asked       # every moment: no entries
    es = search.moment_entries(window, 'cut-in ttc<1.5')
    assert len(es) == 1 and es[0]['kind'] == 'moment' and 'TTC 1.20 s' in es[0]['title'] and '4.5 s' in es[0]['sub']
    opened, seek = [], []
    data = window.page('data')
    monkeypatch.setattr(data, 'show_run', lambda p: opened.append(p))
    monkeypatch.setattr(data.browse, 'seek_time', lambda t: seek.append(t))
    es[0]['open'](window)
    assert opened == [run] and seek == [4.5]
    pal = CommandPalette(window)
    pal.input.setText('cut-in ttc<1.5')
    titles = [pal.list.item(i).data(Qt.ItemDataRole.UserRole)['kind'] for i in range(pal.list.count())]
    assert 'moment' in titles
    pal.close()
    # commands to copy: HIL setup with its undo, a workstation node, the deploy job
    hw = window.go('hardware')
    hw.setup_details.set_expanded(True)
    assert 'sudo modprobe vcan' in hw.setup_cmd.command and 'Undo' not in hw.setup_cmd.command
    assert 'sudo ip link delete vcan0' in hw.setup_undo.command
    st = window.go('settings')
    blocks = [b.command for b in st.findChildren(CommandBlock)]
    assert any('systemctl --user enable --now falcon-worker-daemon@0' in c for c in blocks)
    assert any('systemctl --user disable --now' in c for c in blocks)
    assert any(c.startswith('falcon cluster nodes test') for c in blocks) and 'falcon hil setup' in blocks
    from falcon_autonomy.ui.deploy_tab import DeployTab
    dt = next(iter(window.findChildren(DeployTab)), None)
    if dt is not None:
        dt.cmd_details.set_expanded(True)
        assert dt.cmd_all.command.startswith('falcon deploy all ') and '--device' in dt.cmd_all.command
        assert dt.cmd_check.command.startswith('falcon deploy check --device ')
