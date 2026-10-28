"""`falcon hil ...`: hardware-in-the-loop sessions from the command line (docs/hardware.md), on the same code paths
as the Hardware page.

    falcon hil device [--address A] [--port P] [--ssh-user U] [--ssh-key K] [--ssh-port N] [--remote-dir D]
                      [--python P]                    show or set the device (the user's settings only)
    falcon hil deploy [--start] [--keep-running] [--stop]   copy the agent, check the device (and start / stop it)
    falcon hil session PROJECT [--device local|remote] [--seconds S] [--fps F] [--width W --height H]
                      [--camera h264-rtp|raw-tcp] [--encoder auto|nvh264enc|x264enc] [--lidar tcp|udp] ...
    falcon hil report RUN [--out DIR]                  (re)write report.html and waterfall.png from a session's files
    falcon hil acceptance PROJECT [--seconds S]        the real-device acceptance in one command: deploy, start the
                                                       agent for one session, 60 s at 1280 × 720 and 20 Hz with a
                                                       32-channel lidar, control back, the report

Exit codes: 0 ok; a session's worker's code (docs/cli.md); 2 for a usage error or a device that could not be reached.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from ..core import protocol as P


def _clock(text: str) -> dict:
    """'OFFSET_MS,DRIFT_PPM,JITTER_US' → an emulated clock."""
    parts = [float(x) for x in text.split(',')] + [0.0, 0.0, 0.0]
    return {'offset_ms': parts[0], 'drift_ppm': parts[1], 'jitter_us': parts[2]}


def _session_args(ap):
    ap.add_argument('project')
    ap.add_argument('--world', default='')
    ap.add_argument('--rig', default='')
    ap.add_argument('--seconds', type=float, default=30.0, help='0: until Ctrl+C')
    ap.add_argument('--fps', type=float, default=20.0)
    ap.add_argument('--width', type=int, default=0)
    ap.add_argument('--height', type=int, default=0)
    ap.add_argument('--camera', choices=('h264-rtp', 'raw-tcp'), default='h264-rtp')
    ap.add_argument('--encoder', choices=('auto', 'nvh264enc', 'x264enc'), default='auto')
    ap.add_argument('--bitrate', type=int, default=8000, help='kbit/s')
    ap.add_argument('--lidar', choices=('tcp', 'udp'), default='tcp')
    ap.add_argument('--lidar-channels', type=int, default=0)
    ap.add_argument('--lidar-pps', type=int, default=0, help='points per second')
    ap.add_argument('--lidar-rotation', type=float, default=0.0, help='rotations per second')
    ap.add_argument('--ros2-topic', default='', help='the device republishes the lidar as PointCloud2 here')
    ap.add_argument('--camera-clock', default='0,0,0', help='emulated camera clock: OFFSET_MS,DRIFT_PPM,JITTER_US')
    ap.add_argument('--lidar-clock', default='0,0,0', help='emulated lidar clock: OFFSET_MS,DRIFT_PPM,JITTER_US')
    ap.add_argument('--sim-offset-ms', type=float, default=0.0, help='simulated device: emulated clock offset')
    ap.add_argument('--sim-drift-ppm', type=float, default=0.0, help='simulated device: emulated clock drift')
    ap.add_argument('--sim-decoder', default='auto', help='simulated device: H.264 decoder element')
    ap.add_argument('--traffic', type=int, default=None, help='background vehicles (default: the world\'s profile)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='')
    ap.add_argument('--keep-server', action='store_true')
    ap.add_argument('--fake-carla', default='', help=argparse.SUPPRESS)


def _config(a, svc, project, device: str, dev: dict) -> dict:
    from ..core.drive import run_dir_name
    from ..core.hil import hil_config
    world = a.world or project.doc['worlds'][0]['id']
    rig = a.rig or project.doc['rigs'][0]['id']
    lo = {k: v for k, v in (('channels', a.lidar_channels), ('points_per_second', a.lidar_pps),
                            ('rotation_frequency', a.lidar_rotation)) if v}
    out = a.out or os.path.join(svc.runs_dir() or os.getcwd(), run_dir_name('hil'))
    return hil_config(project, world, rig, out, svc.port, svc.tm_port, seconds=a.seconds, fps=a.fps, seed=a.seed,
                      width=a.width or None, height=a.height or None, camera=a.camera, encoder=a.encoder,
                      bitrate_kbps=a.bitrate, lidar=a.lidar, ros2_topic=a.ros2_topic, lidar_overrides=lo,
                      device=device, device_port=int(dev['port']), camera_clock=_clock(a.camera_clock),
                      lidar_clock=_clock(a.lidar_clock), traffic_vehicles=a.traffic,
                      sim_device={'clock_offset_ms': a.sim_offset_ms, 'clock_drift_ppm': a.sim_drift_ppm,
                                  'decoder': a.sim_decoder}, workspace=svc.workspace)


def _print_summary(run_dir: str):
    from ..core.provenance import read_json
    try:
        run = read_json(os.path.join(run_dir, 'run.json'))
    except (OSError, ValueError):
        return
    h = run.get('hil') or {}
    if not h:
        return
    print(f'{"stage":36s} {"median":>9s} {"p95":>9s} {"p99":>9s}  (ms)')
    for k, st in h['stages'].items():
        flag = '' if st['clock'] != 'cross' or st.get('trusted') else '  (one-way, offset not trusted)'
        print(f'{st["label"]:36s} {st.get("median", float("nan")):9.3f} {st.get("p95", float("nan")):9.3f} '
              f'{st.get("p99", float("nan")):9.3f}{flag}')
    for k in ('e2e', 'processing', 'round_trip', 'on_device', 'network_rtt'):
        m = h['measures'][k]
        if m.get('n'):
            print(f'{m["label"][:36]:36s} {m["median"]:9.3f} {m["p95"]:9.3f} {m["p99"]:9.3f}')
    d = h['deadline']
    print(f'frames {d["frames"]}: on time {d["on_time"]}, late {d["late"]}, superseded {d["superseded"]}, '
          f'no control {d["missing"]}')
    est = h.get('clock_offset') or {}
    if est:
        print(f'device clock offset {est["offset_ns"] / 1e6:.4f} ms ± {est["uncertainty_ns"] / 1e6:.4f} ms, drift '
              f'{est["drift_ppm"]:.2f} ppm')
    for k in ('html', 'png'):
        p = os.path.join(run_dir, 'report.html' if k == 'html' else 'waterfall.png')
        if os.path.isfile(p):
            print(p)


def run_session(a, device: str, address: str = '') -> int:
    from ..cli import _run_job, _service
    from ..core.hil import device_settings, submit_session
    from ..core.project import Project, ProjectError
    svc, st = _service(a.project, a.fake_carla)
    try:
        project = Project.load(a.project)
        cfg = _config(a, svc, project, device, device_settings(st))
        job = submit_session(svc, cfg, address)
    except (OSError, ValueError, ProjectError, RuntimeError) as e:
        print(f'failed: {e}', file=sys.stderr)
        svc.shutdown()
        return P.FAILED
    try:
        code = _run_job(svc, job)
    finally:
        if not a.keep_server:
            svc.shutdown()
    _print_summary(cfg['out'])
    return code


def main_hil(argv) -> int:
    from ..core.hil import DEVICE_DEFAULTS, device_settings
    from ..core.settings import Settings
    from . import deploy as D
    ap = argparse.ArgumentParser(prog='falcon hil', description='hardware in the loop (docs/hardware.md)')
    sub = ap.add_subparsers(dest='what', required=True)
    dv = sub.add_parser('device', help='show or set the device (stored in your settings only)')
    for k in ('address', 'ssh_user', 'ssh_key', 'remote_dir', 'python'):
        dv.add_argument('--' + k.replace('_', '-'), default=None)
    dv.add_argument('--port', type=int, default=None)
    dv.add_argument('--ssh-port', type=int, default=None)
    dp = sub.add_parser('deploy', help='copy the agent to the device and check it')
    dp.add_argument('--start', action='store_true', help='start the agent afterwards')
    dp.add_argument('--keep-running', action='store_true', help='with --start: serve sessions until stopped')
    dp.add_argument('--stop', action='store_true', help='stop the agent this machine started')
    se = sub.add_parser('session', help='a HIL session')
    _session_args(se)
    se.add_argument('--device', choices=('local', 'remote'), default='local')
    rp = sub.add_parser('report', help='(re)write a session\'s report')
    rp.add_argument('run')
    rp.add_argument('--out', default='')
    ac = sub.add_parser('acceptance', help='the real-device acceptance run in one command')
    _session_args(ac)
    ac.set_defaults(seconds=60.0)
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else P.FAILED
    st = Settings()
    dev = device_settings(st)
    if a.what == 'device':
        changed = {k: getattr(a, k) for k in ('address', 'ssh_user', 'ssh_key', 'remote_dir', 'python', 'port',
                                              'ssh_port') if getattr(a, k) is not None}
        if changed:
            if changed.get('ssh_key'):
                changed['ssh_key'] = os.path.abspath(os.path.expanduser(changed['ssh_key']))
            dev.update(changed)
            st.set('hil_device', {k: dev[k] for k in DEVICE_DEFAULTS})
        shown = dict(dev, ssh_key=dev['ssh_key'] or '(none)', agent_pid=dev.get('agent_pid'))
        print(json.dumps(shown, indent=1))
        return 0
    if a.what == 'report':
        from .report import write_report
        try:
            files = write_report(a.run, a.out)
        except (OSError, ValueError) as e:
            print(f'failed: {e}', file=sys.stderr)
            return P.FAILED
        _print_summary(a.run)
        print('\n'.join(files.values()))
        return 0
    if a.what == 'deploy':
        try:
            if a.stop:
                pid = dev.get('agent_pid')
                if not pid:
                    print('no agent started from this machine is recorded', file=sys.stderr)
                    return P.FAILED
                print('stopped' if D.stop(dev, pid) else 'it had already ended')
                st.set('hil_device', dict(dev, agent_pid=None))
                return 0
            out = D.deploy(dev, start_agent=a.start, once=not a.keep_running)
        except D.DeployError as e:
            print(f'failed: {e}', file=sys.stderr)
            return P.FAILED
        env = out['env']
        print(f'device: {env.get("model") or env.get("platform")}; L4T {env.get("l4t") or "?"}; Python '
              f'{env.get("python")}; numpy {env.get("numpy")}; {env.get("gstreamer")}; decoder '
              f'{env.get("decoder") or "none"}; rclpy {"yes" if env.get("rclpy") else "no"}')
        for p in out['problems']:
            print(f'  warning: {p}')
        if out['pid']:
            st.set('hil_device', dict(dev, agent_pid=out['pid']))
            print(f'agent started (pid {out["pid"]} on the device)')
        return 0
    if a.what == 'session':
        address = ''
        if a.device == 'remote':
            address = dev['address']
            if not address:
                print('failed: no device address (falcon hil device --address ...)', file=sys.stderr)
                return P.FAILED
        return run_session(a, a.device, address)
    if a.what == 'acceptance':
        return acceptance(a, st, dev)
    return P.FAILED


def acceptance(a, st, dev) -> int:
    """Deploy and check, start the agent for one session, run a session against it at the acceptance's settings
    (unless overridden: 1280 × 720 at 20 Hz, H.264, a 32-channel lidar), write the report, stop the agent."""
    from . import deploy as D
    a.width, a.height = a.width or 1280, a.height or 720
    a.lidar_channels = a.lidar_channels or 32
    try:
        out = D.deploy(dev, start_agent=True, once=True)
    except D.DeployError as e:
        print(f'failed: {e}', file=sys.stderr)
        return P.FAILED
    env = out['env']
    print(f'device: {env.get("model") or env.get("platform")}; L4T {env.get("l4t") or "?"}; {env.get("gstreamer")}; '
          f'decoder {env.get("decoder") or "none"}')
    for p in out['problems']:
        print(f'  warning: {p}')
    if a.camera == 'h264-rtp' and not env.get('decoder'):
        print('failed: the device cannot decode H.264 (see the warnings); use --camera raw-tcp', file=sys.stderr)
        D.stop(dev, out['pid'])
        return P.FAILED
    try:
        code = run_session(a, 'remote', dev['address'])
    finally:
        try:
            D.stop(dev, out['pid'])                       # normally it has already exited (--once)
        except D.DeployError:
            pass
    return code
