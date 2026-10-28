"""The live worker against the fake carla, with a socket standing in for the GUI: frames arrive with their stats,
commands drive the ego, events happen, a recording becomes a falcon-raw run with labels, stop ends it cleanly."""
import json
import os
import socket
import subprocess
import sys
import threading
import time

import pytest

from conftest import worker_env
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core.live import live_config
from falcon_autonomy.core.livelink import Reader, Sender, pack


def test_link_round_trip_and_latest_only(tmp_path):
    path = str(tmp_path / 's.sock')
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(1)
    snd = Sender(path)
    conn, _ = srv.accept()
    snd.put_urgent({'t': 'map', 'x': [1.0]})
    for i in range(50):                        # nobody reads quickly: most are replaced (dropped), none queued up
        snd.put({'t': 'frame', 'frame': i}, b'\xff' * 1000)
    time.sleep(0.2)
    snd.close('done')
    r = Reader()
    msgs = []
    conn.settimeout(2)
    while True:
        b = conn.recv(65536)
        if not b:
            break
        msgs += r.feed(b)
    assert msgs[0][0]['t'] == 'map' and msgs[-1][0] == {'t': 'bye', 'reason': 'done'}
    frames = [h['frame'] for h, _ in msgs if h['t'] == 'frame']
    assert frames == sorted(frames) and frames[-1] == 49 and len(frames) + snd.dropped == 50
    assert Reader().feed(pack({'t': 'x'}, b'abc')) == [({'t': 'x'}, b'abc')]


@pytest.fixture
def gui_socket(tmp_path):
    path = str(tmp_path / 'live.sock')
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(1)
    got = {'msgs': [], 'conn': None}

    def serve():
        conn, _ = srv.accept()
        got['conn'] = conn
        r = Reader()
        while True:
            try:
                b = conn.recv(1 << 20)
            except OSError:
                break
            if not b:
                break
            got['msgs'] += r.feed(b)
    threading.Thread(target=serve, daemon=True).start()
    yield path, got
    srv.close()


def _frames(got):
    return [h for h, _ in got['msgs'] if h['t'] == 'frame']


def _wait(cond, timeout=60):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_live_session(project, tmp_path, gui_socket):
    path, got = gui_socket
    cfg = live_config(project, 'world-town10', 'rig-default', str(tmp_path / 'live'), 2310, 8310, path,
                      str(tmp_path / 'recs'), fps=20.0, mode='manual',
                      view={'kind': 'chase', 'width': 96, 'height': 64, 'fov': 90.0})
    cp = tmp_path / 'live.json'
    cp.write_text(json.dumps(cfg))
    p = subprocess.Popen([sys.executable, '-m', 'falcon_autonomy.workers', 'live', str(cp)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=worker_env())

    def send(**c):
        p.stdin.write(json.dumps(c) + '\n')
        p.stdin.flush()
    try:
        assert _wait(lambda: len(_frames(got)) > 5), p.stderr.read() if p.poll() is not None else 'no frames'
        maps = [h for h, _ in got['msgs'] if h['t'] == 'map']
        assert maps and len(maps[0]['x']) > 50 and maps[0]['rule'] == 'RHT'
        f = _frames(got)[-1]
        assert f['w'] == 96 and f['mode'] == 'manual' and any(a[1] == 'e' for a in f['actors'])
        assert 0.3 < f['rtf'] and f['ticks_per_s'] > 0
        payload = [pl for h, pl in got['msgs'] if h['t'] == 'frame'][-1]
        assert payload[:2] == b'\xff\xd8'                                  # a JPEG
        for _ in range(10):
            send(cmd='control', throttle=1.0, steer=0.0, brake=0.0)
            time.sleep(0.05)
        assert _wait(lambda: _frames(got)[-1]['control'] and _frames(got)[-1]['control']['throttle'] == 1.0, 10)
        send(cmd='record', on=True)
        send(cmd='event', kind='sensor_fail', sensor='cam-front', mode='black', seconds=0.3)
        send(cmd='event', kind='weather', fields={'precipitation': 80.0})
        send(cmd='mode', mode='policy')
        send(cmd='view', kind='free')
        ok = _wait(lambda: p.poll() is not None or (_frames(got)[-1]['view'] == 'free' and
                                                     _frames(got)[-1]['mode'] == 'policy'), 20)
        assert ok and p.poll() is None, p.stderr.read()[-3000:] if p.poll() is not None else 'no view change'
        send(cmd='free_camera', move=[2.0, 0, 1.0], turn=[10, -5])
        assert _wait(lambda: _frames(got)[-1]['recording'] is not None, 10)
        time.sleep(1.0)
        send(cmd='record', on=False)
        assert _wait(lambda: _frames(got)[-1]['recording'] is None, 10)
        send(cmd='stop')
        p.wait(60)
    finally:
        if p.poll() is None:
            p.kill()
    out = p.stdout.read()
    assert p.returncode == P.OK, p.stderr.read()
    run = json.load(open(tmp_path / 'live' / 'run.json'))
    assert run['status'] == 'ok' and run['performance']['steps'] > 20
    kinds = [e['kind'] for e in run['events']]
    assert {'record', 'sensor_fail', 'weather', 'mode'} <= set(kinds)
    rec = run['recordings'][0]
    assert '"recording"' in out and rec.startswith(str(tmp_path / 'recs'))
    rr = json.load(open(os.path.join(rec, 'run.json')))
    assert rr['status'] == 'ok' and rr['summary']['steps'] > 5
    frames = [json.loads(x) for x in open(os.path.join(rec, 'frames.jsonl'))]
    assert all('cam-front/semantic' in fr['files'] for fr in frames)       # reopens with all its labels
    assert any(fr.get('faults', {}).get('cam-front') == 'black' for fr in frames)
    assert os.path.isfile(os.path.join(rec, frames[-1]['files']['cam-front/semantic']))
    assert json.load(open(os.path.join(rec, 'dataset.json')))['frames'] == len(frames)


def test_manual_input_loss_brakes(project, tmp_path, gui_socket):
    """With no control for 0.5 s the ego brakes: a lost input never drives on."""
    path, got = gui_socket
    cfg = live_config(project, 'world-town10', 'rig-default', str(tmp_path / 'live'), 2310, 8310, path,
                      str(tmp_path / 'recs'), mode='manual',
                      view={'kind': 'chase', 'width': 64, 'height': 48, 'fov': 90.0})
    cp = tmp_path / 'live.json'
    cp.write_text(json.dumps(cfg))
    p = subprocess.Popen([sys.executable, '-m', 'falcon_autonomy.workers', 'live', str(cp)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=worker_env())
    try:
        p.stdin.write(json.dumps({'cmd': 'control', 'throttle': 1.0}) + '\n')
        p.stdin.flush()
        assert _wait(lambda: len(_frames(got)) > 30, 60)
        assert _frames(got)[-1]['control']['brake'] == 1.0 and _frames(got)[-1]['control']['throttle'] == 0.0
        p.stdin.write('{"cmd": "cancel"}\n')
        p.stdin.flush()
        p.wait(60)
        assert p.returncode == P.CANCELLED
    finally:
        if p.poll() is None:
            p.kill()
