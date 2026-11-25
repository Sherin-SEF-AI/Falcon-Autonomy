"""4.2 robustness and compliance offscreen (no simulator runs; planted results): the world and rig pickers of Fault
tolerance, Regulation packs and Latency budget (and the fake simulator's cost warning on a large rig), the flags with
their next steps — Jump to config, Attach evidence (undoable in the safety case), Run sensitivity check —, the
fidelity chips beside verdicts, the GSN evidence node's fidelity and its amber, Send to ▸ Fault tolerance and
Reports (nothing starts or changes without a click), the reports list's fidelity column, and the monitor's command
blocks (dry run, deploy, rollback)."""
import json
import os

import pytest

from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401  (fixtures)

pytestmark = pytest.mark.gui


def _regpack_result(ws, name='ncap-a', impact=14.0):
    """A planted Euro NCAP result on the fake: CCRs 30 km/h, 100 % overlap, hit at `impact` km/h."""
    from falcon_autonomy.core import regpacks as RP
    from falcon_autonomy.core.provenance import write_json
    p = RP.pack('ncap_aeb_c2c')
    c = next(c for c in p['cases'] if c['id'] == 'ccrs-30-100')
    out = RP.result_dir(ws, name)
    run = os.path.join(out, 'runs', 'abcdef0123456789')
    os.makedirs(run)
    write_json(os.path.join(run, 'run.json'), {'status': 'ok', 'provenance': {'fidelity': 'fake'}})
    word = RP.band(impact, c['criteria'][0]['bands'])
    row = RP._row(c, result=word, word=word, tone='warning', factor=RP.BAND_SCORE['value'][word], speed=30,
                  overlap=100, run='runs/abcdef0123456789', fidelity='fake',
                  measured={'impact_speed_kmh': impact, 'relative_impact_kmh': impact, 'collision': True},
                  checks=[{'name': 'impact speed band', 'ok': False, 'detail': f'{impact} km/h at contact: {word}'}])
    doc = {'schema': RP.SCHEMA, 'pack': 'ncap_aeb_c2c', 'title': p['title'], 'kind': p['kind'], 'name': name,
           'created': '2026-11-25T00:00:00Z', 'disclaimer': RP.DISCLAIMER, 'about': p['about'], 'limits': p['limits'],
           'sources': {k: RP.SOURCES[k] for k in p['sources']}, 'inputs': {'cases_selected': 1}, 'cases': [row],
           'unverified': RP.unverified(p), 'choices': RP.choices(p), 'fidelity': 'fake'}
    doc.update(RP.verdict_of(p, doc, False))
    write_json(os.path.join(out, 'regpack.json'), doc)
    return out, run


def _campaign(ws, name='planted'):
    from falcon_autonomy.core import faulttol as FT
    cam = {'id': 'cam', 'type': 'camera.pinhole'}
    lid = {'id': 'lid', 'type': 'lidar'}
    cells = []
    for s, f, fn, thr in ((cam, 'cam_yaw', lambda x: 20.0 * x, 5.0), (lid, 'lidar_sector', lambda x: x / 3.6, 10.0)):
        c = FT._cell(s, f, 'recall_drop', thr, 'points lost')
        c.update(FT.tolerance(fn, thr, 0.0, FT.MODELS[f]['max']))
        cells.append(c)
    cells.append(FT._refused(cam, 'noise', 'noise needs a detector (--model)'))
    FT.write(FT.result(cells, 'offline', name, requirements={'cam_yaw': 0.5}, fidelity='recorded',
                       source_fidelity='fake', inputs={'runs': ['runs/r1']}), FT.campaign_dir(ws, name))
    return cells


def test_regulation_packs_flags_pickers_and_evidence(window, qapp, tmp_path):
    from falcon_autonomy.core import gsn as G
    from falcon_autonomy.ui.theme import THEME
    window.project.save(str(tmp_path / 'gui.falcon.json'))
    ws = window.workspace()
    out, run = _regpack_result(ws)
    page = window.go('reports')
    tab = page.regpacks
    tab.packs.setCurrentItem(tab.packs.topLevelItem(1))                   # Euro NCAP AEB C2C
    tab.refresh()
    assert tab.result.currentData() == 'ncap-a' and tab.doc['fidelity'] == 'fake'
    assert tab.chip.isVisible() and tab.chip.text() == 'Fake simulator' and tab.chip.property('tone') == 'warning'
    # the unverified values: compact rows (document · clause), flags with three next steps
    assert set(tab.flag_widgets) == {'bands_ccrs', 'bands_ccrm', 'overlap_sign'}
    fl = tab.flag_widgets['bands_ccrs']
    assert [b.text() for b in fl.buttons] == ['Jump to config', 'Attach evidence', 'Run sensitivity check']
    assert 'Euro NCAP Assessment Protocol' in fl.label.text() and '3.3.2' in fl.label.text()
    assert 'NOT VERIFIED' in fl.label.toolTip()
    fl.buttons[0].click()                                                 # Jump to config
    assert tab.values.currentItem().data(0, 256) == 'bands_ccrs'
    fl.buttons[2].click()                                                 # Run sensitivity check
    s = tab.sens['bands_ccrs']
    assert s['flips'] and 'FLIPS' in tab.sens_label.text() and 'yellow' in tab.sens_label.toolTip()
    assert tab.sens_label.styleSheet().endswith(THEME.status('warning'))
    tab.flag_widgets['overlap_sign'].buttons[2].click()
    assert 'HOLDS' in tab.sens_label.text()
    # a case row with its run: the active run, and Send to ▸ in its menu
    tab.cases.setCurrentItem(tab.cases.topLevelItem(0))
    assert window.context.get('run') == run and tab.run_of(tab.cases.topLevelItem(0)) == run
    # Attach evidence: a Solution in the safety case, amber (a fake result), undoable
    nid = tab.flag_widgets['bands_ccrs'].buttons[1].click() or page.gsn.selected
    assert window.stack.currentWidget() is page and nid and nid.startswith('Sn')
    doc = G.load(ws)
    n = G.node(doc, nid)
    assert n['evidence']['kind'] == 'regpack' and n['evidence']['ref'] == 'regpacks/ncap-a'
    assert 'bands_ccrs' in n['text']
    v = page.gsn.ver['nodes'][nid]
    assert v['status'] == 'amber' and v['fidelity'] == 'fake'          # a partial score: no share to claim on
    assert n['evidence']['claim'] == {'measure': 'field', 'path': 'score.share', 'op': '>=', 'limit': 1.0}
    item = page.gsn.items[nid]
    assert 'Fake simulator' in item.toolTip()
    page.gsn.select(nid)
    assert page.gsn.info.vals['Fidelity'].text() == 'Fake simulator'
    page.gsn.undo.undo()
    assert nid not in {x['id'] for x in G.load(ws)['nodes']}
    # the pickers: the project's world and rig in the run's command line
    tab.name.setText('x')
    a = tab.argv()
    assert a[a.index('--world') + 1] == window.project.doc['worlds'][0]['id']
    assert a[a.index('--rig') + 1] == window.project.doc['rigs'][0]['id']


def test_the_fake_simulator_warns_on_a_large_rig(window, qapp, tmp_path):
    from falcon_autonomy.ui import robustkit as RK
    window.project.save(str(tmp_path / 'gui.falcon.json'))
    tab = window.page('reports').regpacks
    assert not tab.where.check()                                          # 160 × 90: cheap on the fake
    window.project.doc['rigs'][0]['sensors'][0].update(width=1280, height=720, labels=['semantic'])
    lat = window.page('hardware').latency
    for picker in (tab.where, lat.where, window.page('stack').faulttol.where):
        picker.refresh()
        assert picker.warning.isVisibleTo(picker.parentWidget() or picker)
        assert 'Mpx per step' in picker.warning.label.text() and '320 × 180' in picker.warning.label.text()
    assert RK.camera_mpx(window.project.doc['rigs'][0]) == pytest.approx(1.8432)
    assert RK.camera_mpx(window.project.doc['rigs'][0], labels=False) == pytest.approx(0.9216)
    assert RK.fake_cost_note({'sensors': [{'type': 'camera.pinhole', 'width': 320, 'height': 180}]}, True) == ''
    window.fake_carla = ''                                                # a real CARLA: no warning
    window.service.fake_carla = ''
    assert not tab.where.check()


def test_fault_tolerance_flags_send_to_and_evidence(window, qapp, tmp_path):
    from falcon_autonomy.core import gsn as G
    window.project.save(str(tmp_path / 'gui.falcon.json'))
    ws = window.workspace()
    _campaign(ws)
    tab = window.page('stack').faulttol
    tab.refresh()
    assert tab.campaign.currentData() == 'planted'
    assert tab.chip.text() == 'Recorded · of Fake simulator runs' and tab.chip.property('tone') == 'warning'
    kinds = [f['kind'] for f in tab.flag_list]
    assert kinds[0] == 'not_met' and 'missing' in kinds and 'no_requirement' in kinds
    box = tab.flags_box.itemAt(0).widget()
    assert [b.text() for b in box.buttons] == ['Jump to config', 'Attach evidence', 'Run sensitivity check']
    target = tab.jump_to(tab.flag_list[0])
    assert target is tab.req and tab.table.currentItem().data(0, 256)['fault'] == 'cam_yaw'
    assert tab.req.value() == 0.5
    s = tab.run_sensitivity(tab.flag_list[0])
    assert s['applicable'] and s['nominal'] == 'not met' and 'NOT MET' in tab.sens_label.text().upper()
    missing = next(f for f in tab.flag_list if f['kind'] == 'missing')
    assert tab.jump_to(missing) is tab.table and not missing['sensitivity']
    # compact derived rows, the caveat behind Details
    assert tab.derived.vals['cam'].text().startswith('extrinsic yaw must hold within')
    assert 'no safety factor' in tab.derived_more.body.text() and not tab.derived_more.expanded
    # Attach evidence: the claim is the flagged cell's tolerance against its requirement
    nid = tab.attach(tab.flag_list[0])
    n = G.node(G.load(ws), nid)
    assert n['evidence']['claim'] == {'measure': 'field', 'path': 'tolerance.cam.cam_yaw.tolerated', 'op': '>=',
                                      'limit': 0.5}
    assert window.page('reports').gsn.ver['nodes'][nid]['status'] == 'red'          # not met stays red
    # Send to ▸ Fault tolerance: the run preselected as the offline input, nothing started
    assert {'faulttol', 'reports'} <= set(window.send_targets)
    run = os.path.join(ws, 'runs', 'r1')
    os.makedirs(run)
    before = len(window.queue.jobs) if hasattr(window.queue, 'jobs') else None
    window.send_targets['faulttol'][1](run)
    assert window.stack.currentWidget().key == 'stack' and tab.offline_run.full == run
    assert tab.offline_btn.isEnabled() and tab.name.text() == 'ft-r1' and tab.job is None
    if before is not None:
        assert len(window.queue.jobs) == before
    a = tab.argv(offline=True)
    assert a[a.index('--runs') + 1] == run and '--closed-loop' not in a
    a = tab.argv()
    assert '--world' in a and '--rig' in a


def test_send_to_reports_offers_the_run_and_attaches_on_a_click(window, qapp, tmp_path):
    from falcon_autonomy.core import gsn as G
    from falcon_autonomy.core import reports as R
    window.project.save(str(tmp_path / 'gui.falcon.json'))
    ws = window.workspace()
    run = os.path.join(ws, 'runs', 'drive-1')
    os.makedirs(run)
    with open(os.path.join(run, 'run.json'), 'w') as fh:
        json.dump({'status': 'ok', 'scenario': {'passed': True}, 'provenance': {'fidelity': 'kinematic'}}, fh)
    page = window.page('reports')
    gsn = page.gsn
    window.send_targets['reports'][1](run)
    assert window.stack.currentWidget() is page and gsn.offer.isVisibleTo(page) and gsn.offered == run
    assert gsn.offer_chip.text() == 'Kinematic' and not os.path.isfile(G.gsn_path(ws))     # nothing changed yet
    nid = gsn.attach_offered()                                       # the click
    doc = G.load(ws)
    n = G.node(doc, nid)
    assert n['evidence']['kind'] == 'run' and n['evidence']['ref'] == os.path.join('runs', 'drive-1')
    assert n['evidence']['claim'] == {'measure': 'field', 'path': 'scenario.passed', 'equals': True}
    assert gsn.ver['nodes'][nid]['status'] == 'amber' and 'Kinematic' in gsn.ver['nodes'][nid]['why']
    assert not gsn.offer.isVisibleTo(page)
    # the reports list: a report's fidelity when its source records it
    out, _run = _regpack_result(ws, 'ncap-b')
    rp = os.path.join(R.reports_dir(ws), 'ncap.html')
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    meta = R.write_report('regpack', out, rp)
    assert meta['fidelity'] == 'fake'
    page.refresh()
    row = next(page.list.topLevelItem(i) for i in range(page.list.topLevelItemCount())
               if page.list.topLevelItem(i).data(0, 256) == rp)
    assert row.text(3) == 'Fake simulator' and 'test double' in row.toolTip(3)
    page.list.setCurrentItem(row)
    assert page.info.vals['Fidelity'].text() == 'Fake simulator'


def test_latency_budget_chip_and_monitor_commands(window, qapp, tmp_path):
    from falcon_autonomy.core import latencyloop as LL
    from falcon_autonomy.core.provenance import write_json
    from PyQt6.QtWidgets import QApplication
    window.project.save(str(tmp_path / 'gui.falcon.json'))
    ws = window.workspace()
    os.makedirs(os.path.join(ws, 'latency', 'b'))
    run = os.path.join(ws, 'runs', 'base')
    os.makedirs(run)
    case = {'case': 'lead_brake#0', 'scenario': 'lead_brake', 'name': 'Lead vehicle brakes', 'values': {}, 'runs': 2,
            'baseline_criticality': 0.4, 'status': 'fails only when late', 'budget': 150.0, 'first_failure': 200.0,
            'why': 'passes up to 150 ms, fails from 200 ms', 'tested': [{'x': 0.0, 'run': run}, {'x': 200.0}]}
    write_json(os.path.join(ws, 'latency', 'b', 'latency-budget.json'), {
        'schema': LL.BUDGET_SCHEMA, 'name': 'b', 'unit': 'ms', 'grid_step': 50.0, 'grid_max': 500.0, 'step_ms': 50.0,
        'sweep': 'constant latency', 'quantisation': 'latencies are rounded up to whole steps of 50 ms',
        'assumes': 'common random numbers', 'fails_only_when_late': ['lead_brake#0'], 'cases': [case], 'runs': 2,
        'fidelity': 'fake', 'world': 'w', 'rig': 'r'})
    lat = window.page('hardware').latency
    lat.refresh()
    assert lat.chip.isVisibleTo(lat) and lat.chip.text() == 'Fake simulator'
    assert 'common random numbers' in lat.note.toolTip() and 'rounded up' in lat.method.body.text()
    lat.tree.setCurrentItem(lat.tree.topLevelItem(0))
    assert window.context.get('run') == run
    lat.scenario.setCurrentIndex(lat.scenario.findData('lead_brake'))
    a = lat.argv()
    assert a[a.index('--world') + 1] == window.project.doc['worlds'][0]['id'] and '--rig' in a
    mon = window.page('hardware').monitors
    mon.ruleset.setCurrentIndex(mon.ruleset.findData('india'))
    mon.generate()
    pkg = mon.current()
    assert mon.dry_cmd.command == f'falcon monitor deploy {pkg} --dry-run'
    assert mon.deploy_cmd.command == f'falcon monitor deploy {pkg}'
    assert mon.rollback_cmd.command == f'falcon monitor rollback {pkg}'
    assert 'No SSH' in mon.dry_cmd.toolTip() and 'try-restart' in mon.rollback_cmd.toolTip()
    assert 'systemctl --user enable --now falcon-monitor-india.service' in mon.device_cmd.command
    assert not mon.device.expanded and len(mon.manifest_id.text()) < len(mon.manifest_id.full)
    mon.dry_cmd.copy_btn.click()
    assert QApplication.clipboard().text() == mon.dry_cmd.command
