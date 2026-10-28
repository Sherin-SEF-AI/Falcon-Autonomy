"""Workers against the fake carla: the drive's outputs, determinism, exit codes, cancel, the watchdog, tidy-up."""
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

from conftest import worker_env
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core.drive import drive_config
from falcon_autonomy.core.reproduce import compare_drives


def _cfg(project, out, **kw):
    cfg = drive_config(project, 'world-town10', 'rig-default', str(out), 2310, 8310, seconds=kw.pop('seconds', 1.0),
                       seed=kw.pop('seed', 1))
    cfg.update(kw)
    p = os.path.join(os.path.dirname(str(out)), os.path.basename(str(out)) + '.json')
    with open(p, 'w') as fh:
        json.dump(cfg, fh)
    return p


def _run(cfg_path, env=None, timeout=120):
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', cfg_path], capture_output=True,
                       text=True, env=worker_env(env), timeout=timeout)
    return r.returncode, [P.parse_line(x) for x in r.stdout.splitlines()], r.stderr


def test_drive_records_frames_labels_and_run_json(project, tmp_path):
    code, msgs, err = _run(_cfg(project, tmp_path / 'run'))
    assert code == P.OK, err
    out = tmp_path / 'run'
    run = json.loads((out / 'run.json').read_text())
    assert run['status'] == 'ok' and run['exit_code'] == 0 and run['provenance']['seed'] == 1
    assert run['provenance']['host']['carla_server'] == '0.9.16'
    perf = run['performance']
    assert perf['steps'] == 20 and perf['source'] == 'measured' and perf['mpx_per_step'] == pytest.approx(0.0288, abs=1e-3)
    frames = [json.loads(x) for x in (out / 'frames.jsonl').read_text().splitlines()]
    assert [f['step'] for f in frames] == list(range(20))
    assert all(b['frame'] == a['frame'] + 1 for a, b in zip(frames, frames[1:]))       # one tick per step
    assert frames[1]['sim_t'] == pytest.approx(0.05)
    import cv2
    rgb = cv2.imread(str(out / frames[5]['files']['cam-front/rgb']))
    sem = cv2.imread(str(out / frames[5]['files']['cam-front/semantic']), cv2.IMREAD_UNCHANGED)
    assert rgb.shape == (90, 160, 3) and sem.shape == (90, 160) and sem.dtype == np.uint8
    assert {1, 11} <= set(np.unique(sem).tolist())                 # the fake's road and sky
    ds = json.loads((out / 'dataset.json').read_text())
    assert ds['frames'] == 20 and ds['sensors'][0]['streams'] == ['rgb', 'semantic']
    assert ds['sensors'][0]['K'][0][0] == pytest.approx(80.0)    # f = W / (2 tan(fov / 2)) at 90°
    kinds = [m['t'] for m in msgs]
    assert kinds[0] == 'hello' and kinds[-1] == 'result' and 'progress' in kinds and 'metric' in kinds


def test_labels_are_the_same_tick_as_the_image(project, tmp_path):
    """The fake's RGB draws the sky blue (B 230) and everything else not: the semantic image's sky (tag 11) must be
    the RGB's sky pixel for pixel (JPEG aside), frame by frame: the label camera is at the image's pose and tick."""
    project.doc['rigs'][0]['sensors'][0]['pose'].update(pitch=8.0)        # horizon in view
    code, _, err = _run(_cfg(project, tmp_path / 'run'))
    assert code == P.OK, err
    import cv2
    out = tmp_path / 'run'
    frames = [json.loads(x) for x in (out / 'frames.jsonl').read_text().splitlines()]

    def sky_rgb(f):
        im = cv2.imread(str(out / f['files']['cam-front/rgb']))
        return (im[:, :, 0] > 210) & (im[:, :, 1] > 145)

    def sky_sem(f):
        return cv2.imread(str(out / f['files']['cam-front/semantic']), cv2.IMREAD_UNCHANGED) == 11
    for f in frames[2::6]:
        assert sky_sem(f).any() and np.mean(sky_rgb(f) == sky_sem(f)) > 0.98


def test_same_seed_reproduces_bit_for_bit(project, tmp_path):
    assert _run(_cfg(project, tmp_path / 'a', seed=4))[0] == P.OK
    assert _run(_cfg(project, tmp_path / 'b', seed=4))[0] == P.OK
    rep = compare_drives(str(tmp_path / 'a'), str(tmp_path / 'b'), {'pose_tol': 0.0, 'rgb_tol': 0.0,
                                                                     'label_agree': 1.0})
    assert rep['ok'], rep['checks']


def test_over_budget_is_refused_before_connecting(project, tmp_path):
    project.doc['rigs'][0]['sensors'][0].update(width=8000, height=8000, labels=['semantic', 'instance', 'depth'])
    code, msgs, err = _run(_cfg(project, tmp_path / 'run'))
    assert code == P.REFUSED and 'budget' in err
    run = json.loads((tmp_path / 'run' / 'run.json').read_text())
    assert run['status'] == 'refused' and os.listdir(tmp_path / 'run') == ['run.json']


def test_no_server_exits_no_server(project, tmp_path):
    code, _, err = _run(_cfg(project, tmp_path / 'run', connect_wait_s=2.0), {'FAKE_CARLA_NO_SERVER': '1'})
    assert code == P.NO_SERVER and 'no CARLA server answered' in err


def test_watchdog_ends_a_worker_whose_server_hangs(project, tmp_path):
    """A crashed server leaves the client hanging in its call: the watchdog writes run.json and exits SERVER_LOST."""
    cfg = _cfg(project, tmp_path / 'run', timeout=1.0, seconds=5.0)
    t0 = time.monotonic()
    code, _, err = _run(cfg, {'FAKE_CARLA_HANG_AT_TICK': '15'}, timeout=60)
    assert code == P.SERVER_LOST and time.monotonic() - t0 < 30
    run = json.loads((tmp_path / 'run' / 'run.json').read_text())
    assert run['status'] == 'server_lost' and 'stopped answering' in run['error']


def test_sigterm_cancels_tidies_and_leaves_only_run_json(project, tmp_path):
    cfg = _cfg(project, tmp_path / 'run', seconds=60.0)
    p = subprocess.Popen([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', cfg], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, env=worker_env())
    for line in p.stdout:                                # wait until it is recording
        if '"step ' in line:
            break
    # numpy (and its BLAS threads) are loaded by now: the signal must still reach the watchdog, not kill the worker
    p.send_signal(signal.SIGTERM)
    p.wait(60)
    assert p.returncode == P.CANCELLED
    run = json.loads((tmp_path / 'run' / 'run.json').read_text())
    assert run['status'] == 'cancelled' and os.listdir(tmp_path / 'run') == ['run.json']


def test_stdin_cancel(project, tmp_path):
    cfg = _cfg(project, tmp_path / 'run', seconds=60.0)
    p = subprocess.Popen([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', cfg], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=worker_env())
    for line in p.stdout:
        if '"step ' in line:
            break
    p.stdin.write('{"cmd": "cancel"}\n')
    p.stdin.flush()
    p.wait(60)
    assert p.returncode == P.CANCELLED


def test_tidy_destroys_in_a_safe_order(fake_module):
    """Children before parents, vehicles that were on the autopilot by id (Actor.destroy() on one aborts libcarla),
    and the world back in asynchronous mode."""
    from falcon_autonomy.workers.common import Worker
    carla = fake_module
    w = Worker(['x'])
    w.import_carla()
    w.client = carla.Client('127.0.0.1', 2310)
    w.world = world = w.client.get_world()
    st = world.get_settings()
    st.synchronous_mode, st.fixed_delta_seconds = True, 0.05
    world.apply_settings(st)
    w.tm = w.client.get_trafficmanager(8310)
    lib = world.get_blueprint_library()
    sp = world.get_map().get_spawn_points()
    ego = w.own(world.spawn_actor(lib.find('vehicle.lincoln.mkz_2020'), sp[0]))
    ego.set_autopilot(True, 8310)
    cam = w.own(world.spawn_actor(lib.find('sensor.camera.rgb'), carla.Transform(), attach_to=ego))
    cam.listen(lambda d: None)
    w.tidy()
    gone = [e[1] for e in carla.LOG if e[0] == 'destroy']
    assert gone == [cam.id, ego.id]
    assert world.get_settings().synchronous_mode is False and world.get_settings().fixed_delta_seconds is None
    assert ('tm_sync', False) in carla.LOG


def test_tidy_after_a_killed_worker(fake_module):
    from falcon_autonomy.workers.tidy import tidy_server
    carla = fake_module
    c = carla.Client('127.0.0.1', 2310)
    world = c.get_world()
    lib = world.get_blueprint_library()
    sp = world.get_map().get_spawn_points()
    bp = lib.find('vehicle.lincoln.mkz_2020')
    bp.set_attribute('role_name', 'hero')
    ego = world.spawn_actor(bp, sp[0])
    s = lib.find('sensor.camera.rgb')
    s.set_attribute('role_name', 'falcon_sensor')
    world.spawn_actor(s, carla.Transform(), attach_to=ego)
    other = lib.find('vehicle.audi.a2')
    other.set_attribute('role_name', 'someone_elses')
    keep = world.spawn_actor(other, sp[1])
    st = world.get_settings()
    st.synchronous_mode = True
    world.apply_settings(st)
    res = tidy_server(c, world)
    assert res['destroyed'] == 2 and res['was_synchronous'] and keep.is_alive
    assert not world.get_settings().synchronous_mode


def test_probe(tmp_path):
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'probe', '--port', '2310'],
                       capture_output=True, text=True, env=worker_env(), timeout=60)
    assert r.returncode == 0
    info = [P.parse_line(x) for x in r.stdout.splitlines() if '"server"' in x][0]
    assert info['server_version'] == '0.9.16' and 'Town10HD_Opt' in info['maps']


def test_unknown_worker_is_a_usage_error():
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'nope'], capture_output=True, text=True,
                       env=worker_env(), timeout=60)
    assert r.returncode == P.FAILED and 'usage' in r.stderr
