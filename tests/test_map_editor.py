"""The map editor: edits to an imported world's OpenStreetMap data (core.mapedit) — applied in order and
deterministically, refused when they would leave a road without lanes, checked for roads they cut off — the import
worker converting with them (fake converter), and the editor and the re-import in the GUI (offscreen)."""
import json
import math
import os
import subprocess
import sys

import pytest

from conftest import worker_env
from falcon_autonomy.core import mapedit as ME
from falcon_autonomy.core import osm as O
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core.worlds import import_config, world_entry

LAT0, LON0 = 12.9716, 77.5946                    # a synthetic place (the test world's centre)
KX, KY = 111319.49 * math.cos(math.radians(LAT0)), 110574.0


def ll(x, y):
    return LAT0 + y / KY, LON0 + x / KX


def crossing_osm(path):
    """A crossroads: East Street (way 10, west–east through node 2), North Street (way 11, south–north through node
    2), a spur (way 12) that joins only North Street's north end, and a traffic_calming=bump node on North Street."""
    nodes = {1: (-200, 0), 21: (-100, 0), 2: (0, 0), 22: (100, 0), 3: (200, 0), 4: (0, -200), 7: (0, 100),
             5: (0, 200), 6: (100, 300)}
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6">']
    for n, (x, y) in sorted(nodes.items()):
        la, lo = ll(x, y)
        tag = '<tag k="traffic_calming" v="bump"/>' if n == 7 else ''
        lines.append(f'<node id="{n}" lat="{la:.8f}" lon="{lo:.8f}">{tag}</node>')

    def way(i, refs, name):
        lines.append(f'<way id="{i}">' + ''.join(f'<nd ref="{r}"/>' for r in refs) +
                     f'<tag k="highway" v="residential"/><tag k="name" v="{name}"/><tag k="lanes" v="2"/></way>')
    way(10, [1, 21, 2, 22, 3], 'East Street')
    way(11, [4, 2, 7, 5], 'North Street')
    way(12, [5, 6], 'Spur Lane')
    lines.append('</osm>')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines))
    return path


@pytest.fixture
def area(tmp_path):
    return O.read_osm(crossing_osm(str(tmp_path / 'cross.osm')))


def breaker(way, x, y, value='hump'):
    la, lo = ll(x, y)
    return {'op': 'speed_breaker', 'way': way, 'lat': la, 'lon': lo, 'value': value}


# ------------------------------------------------------------------------------------------------ core
def test_edits_apply_in_order_and_deterministically(area, tmp_path):
    O.write_osm(area, str(tmp_path / 'before.osm'))
    edits = [{'op': 'lanes', 'way': 10, 'forward': 2, 'backward': 1},
             {'op': 'maxspeed', 'way': 11, 'kmh': 30},
             {'op': 'oneway', 'way': 12, 'value': 'yes'},
             {'op': 'highway', 'way': 10, 'value': 'tertiary'},
             breaker(10, 50, 0.4),
             {'op': 'signal', 'node': 2, 'on': True},
             {'op': 'remove_feature', 'node': 7}]
    out, rep = ME.apply_edits(area, edits)
    assert [a['index'] for a in rep['applied']] == list(range(7)) and not rep['skipped']
    w10, w11, w12 = (ME.way_by_id(out, i) for i in (10, 11, 12))
    assert (w10['tags']['lanes'], w10['tags']['lanes:forward'], w10['tags']['lanes:backward']) == ('3', '2', '1')
    assert O.expected_lanes(w10) == (2, 1) and w10['tags']['highway'] == 'tertiary'
    assert w11['tags']['maxspeed'] == '30'
    assert w12['tags']['oneway'] == 'yes' and O.expected_lanes(w12) == (2, 0)       # both lanes one way
    new = max(area.nodes) + 1                         # new nodes are numbered after the data's largest id
    assert w10['nodes'] == [1, 21, 2, new, 22, 3] and out.node_tags[new]['traffic_calming'] == 'hump'
    la, lo = out.nodes[new]
    assert abs((la - LAT0) * KY) < 0.01 and abs((lo - LON0) * KX - 50) < 0.01      # on the way, not off it
    assert out.node_tags[2]['highway'] == 'traffic_signals' and 7 not in out.node_tags
    assert rep['sha256'] == ME.edits_hash(edits) and rep['touched_ways'] == [10, 11, 12]
    # the input is untouched, and the same list gives the same file
    O.write_osm(area, str(tmp_path / 'again.osm'))
    assert (tmp_path / 'again.osm').read_text() == (tmp_path / 'before.osm').read_text()
    O.write_osm(out, str(tmp_path / 'a.osm'))
    O.write_osm(ME.apply_edits(area, edits)[0], str(tmp_path / 'b.osm'))
    assert (tmp_path / 'a.osm').read_text() == (tmp_path / 'b.osm').read_text()


def test_edits_that_would_break_a_road_are_refused(area):
    def errs(e):
        return ME.check_edit(area, e)
    assert errs({'op': 'lanes', 'way': 11, 'forward': 0, 'backward': 1})            # a two-way road with no lane
    assert errs({'op': 'lanes', 'way': 11, 'forward': 7, 'backward': 1})
    assert not errs({'op': 'lanes', 'way': 11, 'forward': 1, 'backward': 1})
    assert errs({'op': 'signal', 'node': 21, 'on': True})                            # not a junction
    assert not errs({'op': 'signal', 'node': 2, 'on': True})
    assert errs(breaker(10, 50, 40.0))                                                # 40 m off the way
    assert errs({'op': 'highway', 'way': 10, 'value': 'footway'})                     # not converted
    assert errs({'op': 'maxspeed', 'way': 10, 'kmh': 300})
    assert errs({'op': 'remove_feature', 'node': 21})
    one = ME.apply_edits(area, [{'op': 'oneway', 'way': 11, 'value': 'yes'}])[0]
    assert ME.check_edit(one, {'op': 'lanes', 'way': 11, 'forward': 1, 'backward': 1})   # one-way: none backward
    # an edit the area no longer has is skipped, with its reason, not an error
    out, rep = ME.apply_edits(area, [{'op': 'remove', 'way': 99}, {'op': 'maxspeed', 'way': 10, 'kmh': 20}])
    assert [s['index'] for s in rep['skipped']] == [0] and 'not in the area' in rep['skipped'][0]['reasons'][0]
    assert ME.way_by_id(out, 10)['tags']['maxspeed'] == '20'


def test_one_edit_per_road_and_property_and_the_list_hash(area):
    e1 = {'op': 'lanes', 'way': 10, 'forward': 2, 'backward': 1}
    e2 = {'op': 'lanes', 'way': 10, 'forward': 1, 'backward': 1}
    lst = ME.upsert(ME.upsert([], e1), {'op': 'maxspeed', 'way': 10, 'kmh': 40})
    lst = ME.upsert(lst, e2)
    assert lst == [e2, {'op': 'maxspeed', 'way': 10, 'kmh': 40}]
    assert ME.upsert(lst + [breaker(10, 50, 0)], {'op': 'remove', 'way': 10}) == [{'op': 'remove', 'way': 10}]
    assert ME.edits_hash(lst) == ME.edits_hash(json.loads(json.dumps(lst)))
    assert ME.edits_hash(lst) != ME.edits_hash(lst[::-1])                             # the order is part of it


def test_removing_a_road_reports_what_it_cuts_off(area):
    assert len(ME.components(area)) == 1
    _, rep = ME.apply_edits(area, [{'op': 'remove', 'way': 11}])
    assert rep['connectivity'] == {'parts_before': 1, 'parts_after': 2, 'cut_off': [12]}
    assert rep['removed_ways'] == [11]
    _, rep = ME.apply_edits(area, [{'op': 'remove', 'way': 12}])
    assert rep['connectivity']['cut_off'] == []


def test_geometry_for_the_editor(area):
    geo = ME.geometry(area, O.local_projection(LAT0, LON0))
    assert [w['id'] for w in geo['ways']] == [10, 11, 12]
    assert [j['id'] for j in geo['junctions']] == [2, 5] and geo['junctions'][0]['ways'] == [10, 11]
    assert [c['id'] for c in geo['calming']] == [7] and geo['calming'][0]['value'] == 'bump'
    x, y = geo['junctions'][0]['xy']
    assert abs(x) < 0.05 and abs(y) < 0.05


# ------------------------------------------------------------------------------------------------ the worker
def _import(tmp_path, cfg):
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'osm_import', str(tmp_path / 'c.json')],
                       capture_output=True, text=True, env=worker_env(), timeout=120)
    assert r.returncode == P.OK, r.stdout + r.stderr
    return json.loads((tmp_path / 'w' / 'import.json').read_text())


def test_the_import_converts_with_the_edits(tmp_path):
    from falcon_autonomy.core import roadfeatures as RF
    from falcon_autonomy.core import xodr as X
    src = crossing_osm(str(tmp_path / 'cross.osm'))
    cfg = import_config(src, str(tmp_path / 'w'), None, True, centre=(LAT0, LON0))
    plain = _import(tmp_path, cfg)
    assert plain['map_edits'] is None and plain['graph_totals']['roads'] == 6     # 3 two-way ways
    assert plain['lane_links']['links'] == 0                  # the fake writes no links: nothing to measure
    cfg['edits'] = [{'op': 'lanes', 'way': 10, 'forward': 2, 'backward': 1}, {'op': 'remove', 'way': 12},
                    breaker(10, 50, 0), {'op': 'remove', 'way': 77}]
    rep = _import(tmp_path, cfg)
    me = rep['map_edits']
    assert len(me['applied']) == 3 and [s['index'] for s in me['skipped']] == [3]
    assert me['sha256'] == ME.edits_hash(cfg['edits'])
    row = me['lane_check']['rows'][0]
    assert row['way'] == 10 and tuple(row['found']) == (2, 1) and row['matched']   # built as edited
    assert rep['graph_totals']['roads'] == 4 and rep['handedness'] == 'LHT'
    assert any('3 of 4 map edits applied' in n for n in rep['notes'])
    w = tmp_path / 'w'
    assert ME.way_by_id(O.read_osm(str(w / 'area.source.osm')), 12) is not None    # the area as read, unedited
    assert ME.way_by_id(O.read_osm(str(w / 'area.osm')), 12) is None
    # the new hump on both directions of East Street, and North Street's own bump
    assert rep['road_features']['counts'] == {'irc_hump': 2, 'osm_bump': 2} and rep['road_features']['placed'] == 4
    assert len(RF.read((w / 'world.xodr').read_text())) == 4 and \
        X.handedness(X.parse((w / 'world.xodr').read_text())) == 'LHT'


def test_reimport_config_reads_the_area_again(tmp_path):
    src = crossing_osm(str(tmp_path / 'cross.osm'))
    cfg = import_config(src, str(tmp_path / 'w'), (12.96, 77.58, 12.98, 77.61), True, centre=(LAT0, LON0))
    w = world_entry('w1', 'W', cfg, {'centre': [LAT0, LON0], 'handedness': 'LHT'})
    w['osm']['edits'] = [{'op': 'maxspeed', 'way': 10, 'kmh': 30}]
    folder = tmp_path / 'w'
    folder.mkdir()
    c = ME.import_config_for(w, str(folder), str(tmp_path / 'out'))
    assert c['source'] == src and c['bbox'] == cfg['bbox'] and c['centre'] == [LAT0, LON0] and c['lht']
    assert c['edits'] == w['osm']['edits'] and ME.import_config_for(w, str(folder), 'o', [])['edits'] == []
    w['osm']['source'] = str(tmp_path / 'gone.osm')           # the source moved: the world's own copy
    (folder / 'area.osm').write_text(open(src).read())
    assert ME.source_for(w, str(folder))[0] == str(folder / 'area.osm')
    (folder / 'import.json').write_text(json.dumps({'map_edits': {'applied': [{'index': 0}]}}))
    with pytest.raises(FileNotFoundError):                    # area.osm has edits in it: not a source
        ME.source_for(w, str(folder))
    (folder / 'area.source.osm').write_text(open(src).read())
    p, bbox, _ = ME.source_for(w, str(folder))
    assert p == str(folder / 'area.source.osm') and bbox is None


# ------------------------------------------------------------------------------------------------ the GUI
from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401


def _imported_world(window, qapp, tmp_path):
    from falcon_autonomy.core.jobs import FINAL
    from falcon_autonomy.core.project import Edit
    from falcon_autonomy.core.worlds import world_dir
    src = crossing_osm(str(tmp_path / 'cross.osm'))
    cfg = import_config(src, world_dir(window.workspace(), 'w-cross'), None, True, centre=(LAT0, LON0))
    job = window.service.submit_osm_import(cfg)
    assert pump(qapp, lambda: job.state in FINAL, 60) and job.state == 'done', job.log
    with open(os.path.join(cfg['out'], 'import.json')) as fh:
        rep = json.load(fh)
    window.edit(Edit('insert', ['worlds'], world_entry('w-cross', 'Crossroads', cfg, rep), text='import'))
    page = window.go('worlds')
    page.list.setCurrentRow(page.list.count() - 1)
    return page


@pytest.mark.gui
def test_map_editor_edits_undoes_and_applies(window, qapp, tmp_path):
    from falcon_autonomy.core.jobs import FINAL
    page = _imported_world(window, qapp, tmp_path)
    assert page.edit_map_btn.isEnabled() and page.reimport_btn.isEnabled()
    dlg = page.edit_map()
    dlg.resize(1280, 820)
    pump(qapp, timeout=0.2)
    c = dlg.canvas
    assert {w['id'] for w in c.geo['ways']} == {10, 11, 12} and c.lines            # OSM ways over the lanes
    assert 'up to date' in dlg.state.text() or 'No edits' in dlg.state.text()
    # click the crossroads: a junction, which can get signals
    c.click(c.to_px(0.0, 0.0))
    assert c.sel == ('node', 2) and dlg.signal_cb.isVisibleTo(dlg)
    dlg.signal_cb.setChecked(True)
    assert dlg.edits() == [{'op': 'signal', 'node': 2, 'on': True}]
    # click East Street, give it 2 + 1 lanes
    c.click(c.to_px(-150.0, 0.5))
    assert c.sel == ('way', 10) and dlg.form_w.isVisibleTo(dlg)
    assert (dlg.fwd.value(), dlg.bwd.value()) == (1, 1)
    dlg.fwd.setValue(2)
    dlg.edit_lanes()
    assert dlg.edits()[-1] == {'op': 'lanes', 'way': 10, 'forward': 2, 'backward': 1}
    # a two-way road with no lane backward is refused (a message, no edit)
    from PyQt6.QtWidgets import QMessageBox
    shown = []
    QMessageBox.warning, orig = (lambda *a, **k: shown.append(a[2])), QMessageBox.warning
    try:
        dlg.bwd.setValue(0)
        dlg.edit_lanes()
    finally:
        QMessageBox.warning = orig
    assert shown and 'at least one lane in each direction' in shown[0] and len(dlg.edits()) == 2
    # a speed breaker by clicking in breaker mode
    dlg.set_mode('breaker')
    c.click(c.to_px(60.0, 0.0))
    assert dlg.edits()[-1]['op'] == 'speed_breaker' and dlg.edits()[-1]['way'] == 10
    dlg.set_mode('select')
    # removing North Street cuts the spur off: said before applying
    c.click(c.to_px(0.5, -120.0))
    assert c.sel == ('way', 11)
    dlg.remove_way()
    assert 'no longer connect' in dlg.warn.text() and 'Spur Lane' in dlg.warn.text()
    assert dlg.edits_list.count() == 4
    window.undo.undo()                                        # undo the removal
    assert len(dlg.edits()) == 3 and not dlg.warn.text()
    page.show_world()
    assert 'not applied yet' in page.place_kv.vals['Map edits'].text()
    # apply: a new folder, the world pointed at it, the checks shown
    old_xodr = window.project.doc['worlds'][-1]['xodr']
    dlg.apply()
    assert pump(qapp, lambda: dlg.job.state in FINAL, 60) and dlg.job.state == 'done', dlg.job.log
    pump(qapp, timeout=0.2)
    w = window.project.doc['worlds'][-1]
    assert w['xodr'] != old_xodr and '/edit-' in w['xodr'] and w['osm']['applied_edits'] == ME.edits_hash(dlg.edits())
    assert os.path.isfile(os.path.join(window.workspace(), w['xodr']))
    assert 'up to date' in dlg.state.text() and not dlg.apply_btn.isEnabled()
    assert '1 of 1 as edited' in dlg.results.vals['Lanes of the edited roads'].text()
    assert 'LHT' in dlg.results.vals['Driving side'].text()
    page.show_world()
    assert 'applied' in page.place_kv.vals['Map edits'].text()
    window.undo.undo()                                        # back to the files before: still there
    assert window.project.doc['worlds'][-1]['xodr'] == old_xodr
    assert os.path.isfile(os.path.join(window.workspace(), old_xodr)) and 'do not have them' in dlg.state.text()
    dlg.close()
    dlg.deleteLater()
    pump(qapp, timeout=0.1)


@pytest.mark.gui
def test_reimport_adds_a_new_world_beside_the_old(window, qapp, tmp_path):
    from falcon_autonomy.core.jobs import FINAL
    page = _imported_world(window, qapp, tmp_path)
    n = len(window.project.doc['worlds'])
    dlg = page.reimport()
    assert 'source file' in dlg.kv.vals['Source'].text()
    assert pump(qapp, lambda: 'measuring' not in dlg.kv.vals['Before'].text(), 30)
    dlg.convert()
    assert pump(qapp, lambda: dlg.job.state in FINAL, 60) and dlg.job.state == 'done', dlg.job.log
    pump(qapp, timeout=0.2)
    assert dlg.add_btn.isEnabled() and 'lane positions' in dlg.kv.vals['After'].text()
    assert 'Lane positions' in dlg.kv.vals and 'LHT' in dlg.kv.vals['Driving side'].text()
    dlg.add_world()
    worlds = window.project.doc['worlds']
    assert len(worlds) == n + 1 and worlds[-2]['id'] == 'w-cross'
    assert worlds[-1]['name'] == 'Crossroads (re-imported)' and worlds[-1]['osm']['reimported_from'] == 'w-cross'
    assert worlds[-1]['osm']['centre'] == [LAT0, LON0]
    from falcon_autonomy.core.project import validate
    assert validate(window.project.doc) == []
    window.undo.undo()
    assert len(window.project.doc['worlds']) == n


def test_signal_lanes_finds_lights_that_control_nothing():
    from falcon_autonomy.core import xodr as X
    road = ('<road id="{rid}" length="10" junction="1"><planView><geometry s="0" x="0" y="0" hdg="0" length="10">'
            '<line/></geometry></planView><lanes><laneSection s="0">{lanes}<center><lane id="0" type="none"/>'
            '</center></laneSection></lanes><signals><signalReference id="3" s="0" t="0" orientation="{o}"/>'
            '</signals></road>')
    left = '<left><lane id="1" type="driving"/></left>'
    right = '<right><lane id="-1" type="driving"/></right>'

    def doc(*roads):
        return '<OpenDRIVE><header/>' + ''.join(roads) + '</OpenDRIVE>'
    ok = doc(road.format(rid=1, lanes=right, o='-'), road.format(rid=2, lanes=left, o='+'))
    assert X.signal_lanes(ok)['oriented'] == 2 and X.signal_lanes(ok)['no_lane'] == 0
    bad = doc(road.format(rid=1, lanes=left, o='-'))          # a mirrored road whose orientation was kept
    assert X.signal_lanes(bad)['no_lane'] == 1
    fixed, _ = X.to_lht(doc(road.format(rid=1, lanes=right, o='-')))
    assert X.signal_lanes(fixed)['no_lane'] == 0
