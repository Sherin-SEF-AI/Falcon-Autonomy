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
    falcon hil setup [--can-interface can0] [--ptp-interface IF]   PRINT the privileged commands that create vcan,
                                                       v4l2loopback, a CAN FD adapter's link and PTP (never run here)
    falcon hil can dbc [--out FILE] | interfaces | confirm IF --statement "bench network, not a vehicle" |
                   revoke IF | bench IF [--seconds S] [--surface S] [--fd auto|on|off]
    falcon hil ptp conf [--profile software|gptp] [--interface IF] [--timestamping software|hardware] | status
    falcon hil v4l2 devices | pattern DEVICE [--seconds S] [--format YUYV|UYVY|RGB3|NV12]

Session options for the 4.0 interfaces: --camera v4l2 [--v4l2-device /dev/video40 --v4l2-format YUYV
--v4l2-method ioctl|gstreamer --no-barcode], --can IF [--can-fd auto|on|off --can-control can|record --surface S
--mu M --dbc FILE --dbc-map ROLE=MESSAGE.SIGNAL …], --clock-source auto|ptp|ntp-style.

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
    ap.add_argument('--camera', choices=('h264-rtp', 'raw-tcp', 'v4l2'), default='h264-rtp')
    ap.add_argument('--v4l2-device', default='/dev/video40', help='camera v4l2: the output node (v4l2loopback)')
    ap.add_argument('--v4l2-format', choices=('YUYV', 'UYVY', 'RGB3', 'NV12'), default='YUYV')
    ap.add_argument('--v4l2-method', choices=('ioctl', 'gstreamer'), default='ioctl')
    ap.add_argument('--no-barcode', action='store_true', help='camera v4l2: no frame-number barcode in the top rows')
    ap.add_argument('--can', default='', help='the CAN loop on this interface (vcan0, can0, sim:NAME); allowed by '
                                              'settings hil_can')
    ap.add_argument('--can-fd', choices=('auto', 'on', 'off'), default='auto')
    ap.add_argument('--can-control', choices=('can', 'record'), default='can',
                    help='can: the CAN commands drive the ego; record: CAN runs beside the session\'s control')
    ap.add_argument('--surface', default='', help='the chassis\'s road surface (core.chassis.SURFACES)')
    ap.add_argument('--mu', type=float, default=None, help='its friction coefficient (within the surface\'s range)')
    ap.add_argument('--dbc', default='', help='a user DBC for the CAN loop (with --dbc-map)')
    ap.add_argument('--dbc-map', action='append', default=[], metavar='ROLE=MESSAGE.SIGNAL')
    ap.add_argument('--clock-source', choices=('auto', 'ptp', 'ntp-style'), default='',
                    help='default: settings hil_ptp.clock_source')
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


def _can_args(a) -> dict | None:
    if not a.can:
        return None
    chassis = {}
    if a.surface:
        chassis['surface'] = a.surface
    if a.mu is not None:
        chassis['mu'] = a.mu
    can = {'interface': a.can, 'fd': a.can_fd, 'control': a.can_control, 'chassis': chassis}
    if a.dbc:
        mp = {}
        for kv in a.dbc_map:
            k, _, v = kv.partition('=')
            mp[k.strip()] = v.strip()
        can.update(dbc=os.path.abspath(a.dbc), dbc_map=mp)
    return can


def _config(a, svc, project, device: str, dev: dict, settings=None) -> dict:
    from ..core.drive import run_dir_name
    from ..core.hil import hil_config
    world = a.world or project.doc['worlds'][0]['id']
    rig = a.rig or project.doc['rigs'][0]['id']
    lo = {k: v for k, v in (('channels', a.lidar_channels), ('points_per_second', a.lidar_pps),
                            ('rotation_frequency', a.lidar_rotation)) if v}
    out = a.out or os.path.join(svc.runs_dir() or os.getcwd(), run_dir_name('hil'))
    v4l2 = {'device': a.v4l2_device, 'format': a.v4l2_format, 'method': a.v4l2_method,
            'barcode': not a.no_barcode} if a.camera == 'v4l2' else None
    return hil_config(project, world, rig, out, svc.port, svc.tm_port, seconds=a.seconds, fps=a.fps, seed=a.seed,
                      width=a.width or None, height=a.height or None, camera=a.camera, encoder=a.encoder,
                      bitrate_kbps=a.bitrate, lidar=a.lidar, ros2_topic=a.ros2_topic, lidar_overrides=lo,
                      device=device, device_port=int(dev['port']), camera_clock=_clock(a.camera_clock),
                      lidar_clock=_clock(a.lidar_clock), traffic_vehicles=a.traffic,
                      sim_device={'clock_offset_ms': a.sim_offset_ms, 'clock_drift_ppm': a.sim_drift_ppm,
                                  'decoder': a.sim_decoder}, workspace=svc.workspace, can=_can_args(a), v4l2=v4l2,
                      clock_source=a.clock_source, settings=settings)


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
        cfg = _config(a, svc, project, device, device_settings(st), st)
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
    dp.add_argument('--agent-can', default='', help='with --start: the device\'s CAN interface (its --can)')
    dp.add_argument('--agent-camera-device', default='', help='with --start: the device\'s V4L2 capture node '
                                                              '(a GMSL2 camera)')
    su = sub.add_parser('setup', help='print (never run) the privileged commands for vcan, v4l2loopback, a CAN FD '
                                      'adapter and PTP')
    su.add_argument('--can-interface', default='can0', help='the USB-CAN adapter\'s interface')
    su.add_argument('--ptp-interface', default='', help='the network interface for PTP (default: settings, or the '
                                                        'default route\'s)')
    su.add_argument('--bitrate', type=int, default=500000)
    su.add_argument('--dbitrate', type=int, default=2000000, help='CAN FD data phase bit rate')
    cn = sub.add_parser('can', help='the CAN loop: the DBC, interfaces, bench confirmation, the chassis on a bus')
    cs = cn.add_subparsers(dest='can_what', required=True)
    cd = cs.add_parser('dbc', help='write Falcon\'s chassis DBC')
    cd.add_argument('--out', default='', help='a file (default: standard output)')
    cs.add_parser('interfaces', help='the CAN interfaces here and whether Falcon may open them')
    cc = cs.add_parser('confirm', help='record that a real adapter is on a bench network, not a vehicle')
    cc.add_argument('interface')
    cc.add_argument('--statement', required=True, help='must read: bench network, not a vehicle')
    cr = cs.add_parser('revoke', help='withdraw a confirmation')
    cr.add_argument('interface')
    cb = cs.add_parser('bench', help='the chassis model alone on a bus, for a device on the bench')
    cb.add_argument('interface')
    cb.add_argument('--seconds', type=float, default=10.0, help='0: until Ctrl+C')
    cb.add_argument('--fd', choices=('auto', 'on', 'off'), default='auto')
    cb.add_argument('--surface', default='dry_asphalt')
    cb.add_argument('--mu', type=float, default=None)
    pt = sub.add_parser('ptp', help='PTP: a ptp4l.conf to install, this machine\'s status')
    ps = pt.add_subparsers(dest='ptp_what', required=True)
    pc = ps.add_parser('conf', help='print a ptp4l.conf (software timestamping or the gPTP profile)')
    pc.add_argument('--profile', choices=('software', 'gptp'), default='software')
    pc.add_argument('--interface', default='')
    pc.add_argument('--timestamping', choices=('software', 'hardware'), default=None)
    pc.add_argument('--domain', type=int, default=0)
    pc.add_argument('--slave-only', action='store_true')
    ps.add_parser('status', help='what pmc, ptp4l and ethtool -T say here')
    vv = sub.add_parser('v4l2', help='V4L2 devices; a test pattern into an output device')
    vs = vv.add_subparsers(dest='v4l2_what', required=True)
    vs.add_parser('devices', help='the V4L2 nodes here')
    vp = vs.add_parser('pattern', help='write a moving test pattern with the frame barcode')
    vp.add_argument('device')
    vp.add_argument('--seconds', type=float, default=5.0)
    vp.add_argument('--fps', type=float, default=20.0)
    vp.add_argument('--width', type=int, default=640)
    vp.add_argument('--height', type=int, default=360)
    vp.add_argument('--format', choices=('YUYV', 'UYVY', 'RGB3', 'NV12'), default='YUYV')
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
    if a.what == 'setup':
        try:
            print(setup_text(a.can_interface, a.ptp_interface or (st.get('hil_ptp') or {}).get('interface') or '',
                             a.bitrate, a.dbitrate))
        except ValueError as e:
            print(f'failed: {e}', file=sys.stderr)
            return P.FAILED
        return 0
    if a.what == 'can':
        return can_command(a, st)
    if a.what == 'ptp':
        return ptp_command(a, st)
    if a.what == 'v4l2':
        return v4l2_command(a)
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
            extra = (['--can', a.agent_can] if a.agent_can else []) + \
                (['--camera-device', a.agent_camera_device] if a.agent_camera_device else [])
            out = D.deploy(dev, start_agent=a.start, once=not a.keep_running, extra=extra)
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


# ============================================================================================ the 4.0 interfaces
def setup_text(can_if: str = 'can0', ptp_if: str = '', bitrate: int = 500000, dbitrate: int = 2000000) -> str:
    """The privileged commands for the HIL interfaces, to be copied into a terminal by the user. Falcon runs none."""
    import re
    from ..core import ptp
    for name in (can_if, ptp_if):
        if name and not re.match(r'^[A-Za-z0-9_.:-]{1,15}$', name):
            raise ValueError(f'{name!r} is not an interface name')
    ptp_if = ptp_if or ptp.default_interface() or 'eth0'
    lines = [
        '# Falcon Autonomy never runs these: each needs root, so copy the ones you want into a terminal yourself.',
        '# Everything here is optional; `falcon doctor` shows what is in place.',
        '',
        '## Virtual CAN (vcan0): the CAN loop on this machine, with the simulated device',
        'sudo modprobe vcan',
        'sudo ip link add dev vcan0 type vcan',
        'sudo ip link set vcan0 mtu 72              # CAN FD frames (leave out for classical CAN)',
        'sudo ip link set up vcan0',
        '',
        f'## A USB-CAN / CAN FD adapter ({can_if}: gs_usb, peak_usb, kvaser_usb …) on a BENCH network',
        f'sudo ip link set {can_if} down',
        f'sudo ip link set {can_if} up type can bitrate {int(bitrate)} dbitrate {int(dbitrate)} fd on '
        'restart-ms 100',
        f'falcon hil can confirm {can_if} --statement "bench network, not a vehicle"   # refused until then',
        '',
        '## On the Jetson AGX Orin (its own controller, mttcan; the header needs a CAN transceiver)',
        'sudo modprobe mttcan',
        f'sudo ip link set can0 up type can bitrate {int(bitrate)} dbitrate {int(dbitrate)} fd on restart-ms 100',
        'python3 ~/falcon-hil/device_agent.py --can can0      # or: falcon hil deploy --start --agent-can can0',
        '',
        '## V4L2 loopback devices (/dev/video40, /dev/video41) for the camera (--camera v4l2)',
        'sudo apt install v4l2loopback-dkms',
        'sudo modprobe v4l2loopback devices=2 video_nr=40,41 card_label=falcon exclusive_caps=1',
        '',
        f'## PTP on {ptp_if} (software timestamping; the gPTP profile: falcon hil ptp conf --profile gptp)',
    ] + ptp.commands(ptp_if, 'software') + [
        '',
        '## Undo',
        'sudo ip link delete vcan0',
        'sudo modprobe -r v4l2loopback',
    ]
    return '\n'.join(lines)


def doctor_rows(settings=None) -> list:
    """(name, ok, detail) of the HIL interfaces for `falcon doctor` — informational: none is required."""
    from ..core import ptp
    from . import canbridge, v4l2out
    rows = []
    for fn in (canbridge.doctor_rows, ptp.doctor_rows):
        try:
            rows += fn(settings)
        except Exception as e:                           # noqa: BLE001 (a check that fails says so, never stops doctor)
            rows.append((fn.__module__.rsplit('.', 1)[-1], False, f'{type(e).__name__}: {e}'))
    try:
        rows += v4l2out.doctor_rows()
    except Exception as e:                               # noqa: BLE001
        rows.append(('v4l2', False, f'{type(e).__name__}: {e}'))
    return rows


def can_command(a, st) -> int:
    from ..core import canmsgs as C
    from . import canbridge as B
    if a.can_what == 'dbc':
        text = C.dbc_text()
        if a.out:
            with open(a.out, 'w', encoding='utf-8') as fh:
                fh.write(text)
            print(a.out)
        else:
            sys.stdout.write(text)
        return 0
    if a.can_what == 'interfaces':
        ifs = B.interfaces()
        if not ifs:
            print('no CAN interfaces (falcon hil setup prints how to create vcan0)')
        for i in ifs:
            ok, why, _ = B.check_interface(i['name'], st)
            print(f'{i["name"]:10s} {"virtual" if i["virtual"] else "adapter":8s} '
                  f'{"CAN FD" if i["fd"] else "classical":10s} {"up" if i["up"] else "down":5s} '
                  f'{"allowed" if ok else "refused"}: {why}')
        return 0
    try:
        if a.can_what == 'confirm':
            B.confirm_bench(st, a.interface, a.statement)
            print(f'{a.interface}: confirmed as a {B.CONFIRM_TEXT}')
            return 0
        if a.can_what == 'revoke':
            B.revoke_bench(st, a.interface)
            print(f'{a.interface}: no longer allowed')
            return 0
        if a.can_what == 'bench':
            chassis = {'surface': a.surface}
            if a.mu is not None:
                chassis['mu'] = a.mu

            def show(snap, fb):
                print(f'\r{snap["mode"]:9s} tx {snap["tx_frames"]:7d} commands {snap["commands"]:6d} watchdog '
                      f'{snap["watchdog_trips"]:3d} speed {fb["VEHICLE_SPEED"]:6.1f} km/h brake '
                      f'{fb["BRAKE_PRESSURE"]:5.1f} bar steer {fb["STEER_ANGLE"]:7.1f}°', end='', flush=True)
            out = B.bench(a.interface, st, a.seconds, a.fd, chassis, show)
            print()
            print(json.dumps(out, indent=1, default=str))
            return 0 if not out.get('aborted') else P.FAILED
    except (PermissionError, ValueError, OSError, RuntimeError) as e:
        print(f'failed: {e}', file=sys.stderr)
        return P.FAILED
    return P.FAILED


def ptp_command(a, st) -> int:
    from ..core import ptp
    if a.ptp_what == 'conf':
        iface = a.interface or (st.get('hil_ptp') or {}).get('interface') or ptp.default_interface() or 'eth0'
        try:
            print(ptp.ptp4l_conf(a.profile, iface, a.domain, a.slave_only, a.timestamping))
        except ValueError as e:
            print(f'failed: {e}', file=sys.stderr)
            return P.FAILED
        return 0
    iface = (st.get('hil_ptp') or {}).get('interface') or ptp.default_interface()
    for name, ok, detail in ptp.doctor_rows(st):
        print(f'  {"ok " if ok else "-- "} {name:22s} {detail}')
    q = ptp.query_pmc()
    s = q.get('summary') or {}
    if s:
        print(f'pmc: grandmaster {"present" if s.get("gm_present") else "absent"} ({s.get("gm_identity")}), port '
              f'{s.get("port_state")}, offset {s.get("offset_ns")} ns, path delay {s.get("mean_path_delay_ns")} ns')
    else:
        print(f'pmc: {q.get("error") or "no answer"}')
    print(f'clock source: {(st.get("hil_ptp") or {}).get("clock_source", "auto")} (interface {iface or "?"})')
    return 0


def v4l2_command(a) -> int:
    import time

    import numpy as np

    from . import v4l2out as V
    if a.v4l2_what == 'devices':
        devs = V.devices()
        if not devs:
            print('no V4L2 devices (falcon hil setup prints how to create v4l2loopback nodes)')
        for d in devs:
            print(f'{d["device"]:14s} {d["name"]}{"  (v4l2loopback)" if d["loopback"] else ""}')
        return 0
    try:
        w = V.V4L2Writer(a.device, a.width, a.height, a.format, barcode=True)
    except V.V4L2Error as e:
        print(f'failed: {e}', file=sys.stderr)
        return P.FAILED
    n = int(round(a.seconds * a.fps))
    xs = np.arange(a.width)[None, :, None]
    t0 = time.monotonic()
    try:
        for k in range(n):
            img = np.empty((a.height, a.width, 3), np.uint8)
            img[:] = ((xs + 8 * k) % 256).astype(np.uint8)
            w.write(img, k)
            delay = t0 + (k + 1) / a.fps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        w.close()
    print(f'{w.frames} frames of {a.width} × {a.height} {a.format} written to {a.device} ({w.short_writes} short)')
    return 0
