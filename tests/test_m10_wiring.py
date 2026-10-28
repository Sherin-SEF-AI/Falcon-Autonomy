"""Milestone 10's wiring: a world's sensor weather physics switched on the Worlds page reaches the drive config; the
workspace's ODD (odd.json, its reference) is attached to drive, scenario, gym and HIL configs and left out of a
drive's cache key (monitoring only observes)."""
import json
import os

import pytest

from falcon_autonomy.core.datasets import drive_key
from falcon_autonomy.core.drive import drive_config

ODD = {'schema': 'falcon.odd/1', 'name': 'clear days', 'weather': {'fog_density': [0, 10]}}


def test_workspace_odd_is_attached_and_not_in_the_cache_key(project, tmp_path):
    ws = tmp_path / 'ws'
    ws.mkdir()
    before = drive_config(project, 'world-town10', 'rig-default', str(tmp_path / 'a'), 2310, 8310, workspace=str(ws))
    assert 'odd' not in before
    (ws / 'odd.json').write_text(json.dumps(ODD))
    (ws / 'odd-reference.json').write_text('{}')
    cfg = drive_config(project, 'world-town10', 'rig-default', str(tmp_path / 'a'), 2310, 8310, workspace=str(ws))
    assert cfg['odd'] == ODD and cfg['odd_workspace'] == str(ws)
    assert cfg['odd_reference'] == os.path.join(str(ws), 'odd-reference.json')
    assert drive_key(cfg) == drive_key(before)
    from falcon_autonomy.core.gymenv import gym_config
    g = gym_config(project, 'world-town10', str(tmp_path / 'g'), 2310, 8310, workspace=str(ws))
    assert g['odd'] == ODD
    from falcon_autonomy.core.hil import hil_config
    h = hil_config(project, 'world-town10', 'rig-default', str(tmp_path / 'h'), 2310, 8310, workspace=str(ws))
    assert h['odd'] == ODD and h['odd_reference'].endswith('odd-reference.json')


def test_the_loop_odd_is_monitored_when_the_workspace_has_none(project, tmp_path):
    loop_odd = {'id': 'odd-1', 'name': 'loop'}
    project.doc['loop']['odd'] = loop_odd
    cfg = drive_config(project, 'world-town10', 'rig-default', str(tmp_path / 'a'), 2310, 8310,
                       workspace=str(tmp_path / 'empty'))
    assert cfg['odd'] == loop_odd


pytest.importorskip('PyQt6')
from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401  (fixtures)


def test_worlds_page_switches_sensor_physics(window, qapp, tmp_path):
    page = window.go('worlds')
    page.list.setCurrentRow(0) if hasattr(page, 'list') else None
    pump(qapp, timeout=0.2)
    assert not page.physics['lidar'].isEnabled()                 # off: only the main switch is live
    page.physics['enabled'].setChecked(True)
    w = window.project.doc['worlds'][0]
    assert w['sensor_physics']['enabled'] and w['sensor_physics']['lidar']
    assert page.physics['lidar'].isEnabled()
    page.physics['radar'].setChecked(False)
    assert window.project.doc['worlds'][0]['sensor_physics']['radar'] is False
    cfg = drive_config(window.project, w['id'], 'rig-default', str(tmp_path / 'a'), 2310, 8310)
    assert cfg['sensor_physics']['enabled'] and cfg['sensor_physics']['radar'] is False
    window.undo.undo()
    window.undo.undo()
    assert not (window.project.doc['worlds'][0].get('sensor_physics') or {}).get('enabled')
