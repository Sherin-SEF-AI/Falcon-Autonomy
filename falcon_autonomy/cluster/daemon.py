"""The worker daemon of a compute node (standard library; docs/cluster.md, "Nodes").

    falcon worker-daemon serve --slot N [--workspace DIR] [--no-render]
        slot N of this machine: listens on 127.0.0.1:2350+N only (another machine reaches it through an SSH tunnel
        with the user's key), runs the jobs it is sent with its own Settings, CarlaService (CARLA on 2310+10N, its
        traffic manager on 8310+N) and JobQueue, in a scratch workspace (~/.cache/FalconAutonomy/worker-N), and sends
        each run folder back. Packaged as the systemd user unit falcon-worker-daemon@N (packaging/cluster).
    falcon worker-daemon oneshot --bundle FILE --out DIR [--hold-s S]
        one bundle, then exit with the worker's code (a Slurm array task, a Kubernetes Job, a cloud instance): its
        protocol lines on stdout, the run (or the study) in DIR, DIR/exit.json at the end. --hold-s keeps the process
        (and its pod) alive until DIR/.collected appears, so the run can be copied out first.

A client proves it may use the slot with the token the daemon writes when it starts (cluster/daemon-N.token in its
config folder, readable only by the user; a client reads it over SSH). The daemon runs only the workers on its
allow-list (bundle.WORKERS) and builds their command lines itself from the bundle's config: a client cannot run a
command of its choosing. A study bundle runs only failrate, sweep or search with allow-listed options.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import shutil
import signal
import socket
import sys
import threading
import time

from ..core import protocol as CP
from ..core.jobs import FINAL, JobQueue, SubprocessLauncher
from . import bundle as B
from . import protocol as W

RID = re.compile(r'^[A-Za-z0-9_.-]{1,80}$')
RETAIN_S = 3600.0                    # an artifact nobody collected is deleted after this long


def slot_ports(slot: int) -> tuple:
    """(daemon port, CARLA RPC port, traffic manager port) of a slot."""
    return W.PORT_BASE + int(slot), 2310 + 10 * int(slot), 8310 + int(slot)


def node_info() -> dict:
    from .. import __version__
    from ..core.sysinfo import meminfo
    try:
        mem = round(float(meminfo().get('total_gb') or 0.0), 1)
    except (OSError, ValueError, TypeError):
        mem = 0.0
    return {'node': socket.gethostname(), 'app_version': __version__, 'codecs': W.codecs(), 'memory_gb': mem,
            'backends': ['carla-0.9.16'], 'carla': '0.9.16'}


def use_backend(svc, b: dict):
    """Run the bundle's backend (sim.backends: CARLA 0.9.16 unless it names another): the node switches when its
    server is idle; BundleError when it cannot (a fake carla is the one backend it is, a busy server)."""
    want = str((b.get('needs') or {}).get('backend') or svc.backend)
    if want == svc.backend:
        return
    if svc.fake_carla:
        raise B.BundleError(f'this node runs {svc.backend}, not {want}')
    try:
        svc.use_backend(want)
    except (KeyError, ValueError, RuntimeError) as e:
        raise B.BundleError(f'this node cannot run {want} now: {e}') from None


class _ForwardQueue(JobQueue):
    """The node's queue: every protocol message of a job also goes to whoever submitted it."""

    forward = None

    def on_message(self, job, msg):
        super().on_message(job, msg)
        if self.forward is not None:
            self.forward(job, msg)


class Daemon:
    def __init__(self, slot: int = 0, port: int = 0, workspace: str = '', fake_carla: str = '', no_render=False,
                 heartbeat_s: float = W.HEARTBEAT_S, settings=None):
        from ..core.paths import cache_dir
        from ..core.settings import MEMORY_RESERVE_GB, Settings, port_allowed
        from ..sim.service import NodeService
        self.slot = int(slot)
        dport, cport, tport = slot_ports(self.slot)
        self.port = int(port) or dport
        st = settings if settings is not None else Settings()
        for p in (cport, tport):
            if not port_allowed(p, st):
                raise SystemExit(f'failed: port {p} is not one Falcon may use')
        self.workspace = workspace or os.path.join(cache_dir(), f'worker-{self.slot}')
        os.makedirs(os.path.join(self.workspace, 'runs'), exist_ok=True)
        self.queue = _ForwardQueue(SubprocessLauncher(), memory_reserve_gb=MEMORY_RESERVE_GB)
        self.queue.forward = self._forward
        self.svc = NodeService(st, self.queue, self.workspace, fake_carla=fake_carla, port=cport, tm_port=tport,
                               no_render=no_render)
        self.queue.subscribe(self._on_job)
        self.heartbeat_s = float(heartbeat_s)
        self.token = W.write_token(self.slot)
        self.info = dict(node_info(), slot=self.slot, backends=[self.svc.backend])
        self.lock = threading.RLock()
        self.records: dict = {}          # rid -> {'job', 'chan', 'dir', 'done': (head, payload) | None, 't'}
        self.chans: set = set()
        self.stop = threading.Event()
        self.sock = None

    # ---------------------------------------------------------------- serving
    def bind(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', self.port))                  # loopback only: other machines come through SSH
        s.listen(8)
        s.settimeout(0.5)
        self.sock = s
        return s

    def serve(self):
        if self.sock is None:
            self.bind()
        threading.Thread(target=self._ticker, daemon=True, name='falcon-daemon-ticks').start()
        print(json.dumps({'t': 'daemon', 'state': 'listening', 'slot': self.slot, 'port': self.port}), flush=True)
        while not self.stop.is_set():
            try:
                conn, _addr = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=self._client, args=(W.Channel(conn),), daemon=True).start()
        self.shutdown()

    def shutdown(self):
        self.stop.set()
        try:
            if self.sock is not None:
                self.sock.close()
        except OSError:
            pass
        for c in list(self.chans):
            c.close()
        self.svc.shutdown(grace_s=10.0)

    def _ticker(self):
        last = 0.0
        while not self.stop.wait(0.5):
            self.queue.poll()
            now = time.monotonic()
            if now - last >= self.heartbeat_s:
                last = now
                self._heartbeat()
                self._expire()

    def _heartbeat(self):
        with self.lock:
            running = [r for r, d in self.records.items() if d['done'] is None]
            queued = sum(1 for j in self.queue.active() if j.state != 'running')
            chans = list(self.chans)
        for c in chans:
            try:
                c.send({'t': 'heartbeat', 'time': time.time(), 'running': running, 'queued': queued})
            except W.ProtocolError:
                pass

    def _expire(self):
        now = time.monotonic()
        with self.lock:
            old = [r for r, d in self.records.items() if d['done'] is not None and now - d['t'] > RETAIN_S]
        for r in old:
            self._release(r)

    # ---------------------------------------------------------------- one client
    def _client(self, ch: W.Channel):
        try:
            msg, _ = ch.recv()
            if msg.get('t') != 'hello' or int(msg.get('protocol') or 0) != W.PROTOCOL or \
                    not hmac.compare_digest(str(msg.get('token') or ''), self.token):
                ch.send({'t': 'error', 'why': 'not allowed: a wrong token or protocol (read the slot\'s token again)'})
                ch.close()
                return
            ch.codec = W.pick_codec(msg.get('codecs'))
            with self.lock:
                self.chans.add(ch)
                jobs = sum(1 for d in self.records.values() if d['done'] is None)
            ch.send(dict(self.info, t='hello', protocol=W.PROTOCOL, jobs=jobs))
            while not self.stop.is_set():
                msg, payload = ch.recv()
                self._handle(ch, msg, payload)
        except W.ProtocolError:
            pass
        finally:
            with self.lock:
                self.chans.discard(ch)
                for d in self.records.values():
                    if d['chan'] is ch:
                        d['chan'] = None            # the job goes on; a reconnecting client attaches to it
            ch.close()

    def _handle(self, ch, msg, payload):
        t = msg.get('t')
        rid = str(msg.get('rid') or '')
        if t == 'ping':
            ch.send({'t': 'heartbeat', 'time': time.time(), 'running': [], 'queued': 0})
        elif t == 'submit':
            self._submit(ch, rid, payload)
        elif t == 'cancel':
            with self.lock:
                d = self.records.get(rid)
            if d is not None and d['done'] is None:
                if msg.get('force') and d['job'].handle is not None:
                    d['job'].handle.kill()
                else:
                    self.queue.cancel(d['job'])
        elif t == 'attach':
            for r in msg.get('rids') or ():
                with self.lock:
                    d = self.records.get(str(r))
                    if d is not None:
                        d['chan'] = ch
                if d is None:
                    ch.send({'t': 'state', 'rid': r, 'state': 'unknown'})
                elif d['done'] is not None:
                    ch.send(*d['done'])
                else:
                    ch.send({'t': 'state', 'rid': r, 'state': d['job'].state})
        elif t == 'release':
            self._release(rid)

    def _submit(self, ch, rid, payload):
        if not RID.match(rid):
            ch.send({'t': 'refused', 'rid': rid, 'why': 'a bad job id'})
            return
        with self.lock:
            if rid in self.records:
                ch.send({'t': 'accepted', 'rid': rid})       # a resend after a reconnect: the job is here already
                return
        run_dir = os.path.join(self.workspace, 'runs', rid)
        try:
            b = B.decode(payload)
            if b.get('kind') != 'job' or b.get('worker') not in B.WORKERS:
                raise B.BundleError(f'the {b.get("worker") or b.get("kind")} worker is not on this node\'s allow-list')
            use_backend(self.svc, b)
            shutil.rmtree(run_dir, ignore_errors=True)
            os.makedirs(run_dir)
            cfg = B.materialise(b, run_dir, '127.0.0.1', self.svc.port, self.svc.tm_port)
            w = B.WORKERS[b['worker']]
            spec = self.svc.worker_spec(b['worker'], [cfg], str(b.get('title') or rid)[:200],
                                        str(b.get('job_kind') or b['worker']), run_dir, needs_carla=w['needs_carla'],
                                        memory_gb=max(float(w['memory_gb']), 0.3), meta={'config': cfg, 'rid': rid})
        except (B.BundleError, OSError, ValueError, RuntimeError) as e:
            shutil.rmtree(run_dir, ignore_errors=True)
            ch.send({'t': 'refused', 'rid': rid, 'why': str(e)[:500]})
            return
        with self.lock:
            self.records[rid] = {'job': None, 'chan': ch, 'dir': run_dir, 'done': None, 't': time.monotonic()}
        ch.send({'t': 'accepted', 'rid': rid})
        job = self.queue.submit(spec)
        with self.lock:
            self.records[rid]['job'] = job
            if job.state in FINAL and self.records[rid]['done'] is None:
                threading.Thread(target=self._finished, args=(rid, job), daemon=True).start()

    def _forward(self, job, msg):
        rid = job.spec.meta.get('rid')
        with self.lock:
            d = self.records.get(rid)
            ch = d['chan'] if d else None
        if ch is not None:
            try:
                ch.send({'t': 'msg', 'rid': rid, 'msg': msg})
            except W.ProtocolError:
                pass

    def _on_job(self, ev, job):
        rid = job.spec.meta.get('rid')
        if ev == 'finished' and rid:
            threading.Thread(target=self._finished, args=(rid, job), daemon=True).start()

    def _finished(self, rid, job):
        with self.lock:
            d = self.records.get(rid)
            if d is None or d['done'] is not None:
                return
            d['done'] = ()                                 # being packed
        for lvl, text in job.log:                          # what the node's queue itself said (refusals, waits)
            if lvl == 'error':
                self._forward(job, {'t': 'log', 'level': 'error', 'text': text})
        codec = 'zlib'
        with self.lock:
            ch = d['chan']
        if ch is not None:
            codec = getattr(ch, 'codec', 'zlib')
        try:
            data = W.compress(W.pack_dir(d['dir'], skip=('drive.json', 'inputs')), codec)
        except OSError as e:
            data, codec = W.compress(b'', 'zlib'), 'zlib'
            self._forward(job, {'t': 'log', 'level': 'error', 'text': f'packing the run failed: {e}'})
        code = job.exit_code if job.exit_code is not None else CP.CRASHED
        head = {'t': 'done', 'rid': rid, 'code': code, 'codec': codec, 'sha256': W.sha256(data),
                'scratch': os.path.abspath(d['dir'])}
        with self.lock:
            d['done'] = (head, data)
            d['t'] = time.monotonic()
            ch = d['chan']
        if ch is not None:
            try:
                ch.send(head, data)
            except W.ProtocolError:
                pass

    def _release(self, rid):
        with self.lock:
            d = self.records.pop(rid, None)
        if d is not None:
            shutil.rmtree(d['dir'], ignore_errors=True)


# ================================================================================================ oneshot
def _study_argv(b: dict, project_path: str, scenario_arg: str, fake_carla: str) -> list:
    cmd = b['command']
    opts = b.get('options') or {}
    bad = sorted(set(opts) - set(B.STUDY_OPTIONS[cmd]))
    if bad:
        raise B.BundleError(f'options a {cmd} study does not take here: {", ".join(bad)}')
    head = ['run', project_path] if cmd == 'failrate' else [cmd, project_path]
    argv = head + ['--scenario', scenario_arg, '--name', str(b['name']), '--world', str(b['world']), '--rig',
                   str(b['rig'])]
    for k, v in sorted(opts.items()):
        if isinstance(v, bool):
            if v:
                argv.append('--' + k.replace('_', '-'))
        elif v is not None:
            argv += ['--' + k.replace('_', '-'), str(v)]
    return argv + (['--fake-carla', fake_carla] if fake_carla else [])


TOP_FAILURES = 10


def _summary(study_dir: str, out: str) -> list:
    """What a cloud burst brings back: the study's own files, and the drive.json, run.json and recorder log of its ten
    most critical runs (by objective)."""
    os.makedirs(out, exist_ok=True)
    names = []
    for f in sorted(os.listdir(study_dir)):
        p = os.path.join(study_dir, f)
        if os.path.isfile(p) and f.endswith(('.json', '.jsonl')):
            shutil.copy2(p, os.path.join(out, f))
            names.append(f)
    rows = []
    try:
        with open(os.path.join(study_dir, 'results.jsonl'), encoding='utf-8') as fh:
            rows = [json.loads(x) for x in fh if x.strip()]
    except (OSError, ValueError):
        pass
    rows.sort(key=lambda r: -float(r.get('objective') or 0.0))
    for r in rows[:TOP_FAILURES]:
        run = r.get('run') or ''
        if not os.path.isdir(run):
            continue
        dst = os.path.join(out, 'runs', os.path.basename(run))
        os.makedirs(dst, exist_ok=True)
        for f in sorted(os.listdir(run)):
            if f in ('drive.json', 'run.json') or f.endswith(('.log', '.rec')):
                shutil.copy2(os.path.join(run, f), os.path.join(dst, f))
                names.append(os.path.join('runs', os.path.basename(run), f))
    return names


def oneshot(bundle_path: str, out: str, fake_carla: str = '', hold_s: float = 0.0, slot: int | None = None) -> int:
    if slot is None:
        slot = int(os.environ.get('FALCON_SLOT') or os.environ.get('SLURM_LOCALID') or 0)
    _d, cport, tport = slot_ports(slot)
    em = CP.Emitter()
    em.hello('oneshot', slot=slot)
    try:
        with open(bundle_path, 'rb') as fh:
            b = B.decode(fh.read(), cap=B.LAN_MAX)
    except (OSError, B.BundleError) as e:
        em.log(f'refused: {e}', 'error')
        return _end(out, CP.REFUSED, hold_s)
    os.makedirs(out, exist_ok=True)
    if b['kind'] == 'job':
        code = _oneshot_job(b, out, fake_carla, cport, tport)
    else:
        code = _oneshot_study(b, out, fake_carla, cport, tport, em)
    return _end(out, code, hold_s)


def _end(out: str, code: int, hold_s: float) -> int:
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'exit.json'), 'w', encoding='utf-8') as fh:
        json.dump({'code': code, 'status': CP.status_of(code)}, fh)
    print(json.dumps({'t': 'oneshot', 'state': 'holding' if hold_s > 0 else 'done', 'code': code}), flush=True)
    t0 = time.monotonic()
    while hold_s > 0 and time.monotonic() - t0 < hold_s and not os.path.exists(os.path.join(out, '.collected')):
        time.sleep(0.5)
    return code


def _oneshot_job(b, out, fake_carla, cport, tport) -> int:
    from ..core.settings import Settings
    from ..sim.service import NodeService
    q = JobQueue(SubprocessLauncher())
    svc = NodeService(Settings(), q, os.path.dirname(os.path.abspath(out)), fake_carla=fake_carla, port=cport,
                      tm_port=tport)
    try:
        use_backend(svc, b)
        cfg = B.materialise(b, out, '127.0.0.1', cport, tport)
        w = B.WORKERS[b['worker']]
        spec = svc.worker_spec(b['worker'], [cfg], str(b.get('title') or 'job')[:200], str(b.get('job_kind') or
                                                                                        b['worker']),
                               out, needs_carla=w['needs_carla'], memory_gb=w['memory_gb'], meta={'config': cfg})
    except (B.BundleError, OSError, RuntimeError) as e:
        print(json.dumps({'t': 'log', 'level': 'error', 'text': f'refused: {e}'}), flush=True)
        svc.shutdown()
        return CP.REFUSED

    orig = q.on_message

    def echo(job, msg):                                    # the worker's lines on stdout: the batch system's log
        print(json.dumps(msg, separators=(',', ':')), flush=True)
        orig(job, msg)
    q.on_message = echo
    job = q.submit(spec)
    q.wait(job)
    code = job.exit_code if job.exit_code is not None else CP.CRASHED
    svc.shutdown()
    return code


def _oneshot_study(b, out, fake_carla, cport, tport, em) -> int:
    from ..core import scenarios as SC
    from ..core.project import Project
    from ..sim import service as SV
    if b.get('command') not in B.STUDY_COMMANDS:
        em.log(f'refused: {b.get("command")} is not a study a node runs', 'error')
        return CP.REFUSED
    pdir = os.path.join(out, 'project')
    os.makedirs(pdir, exist_ok=True)
    ppath = os.path.join(pdir, 'study.falcon.json')
    try:
        proj = Project.new(str(b['project'].get('name') or 'study'))
        for k in ('worlds', 'rigs', 'traffic_profiles', 'scenarios', 'models'):
            if k in b['project']:
                proj.doc[k] = b['project'][k]
        proj.save(ppath)
        scn = b['scenario']
        if scn.get('id') in SC.LIBRARY and scn.get('builtin'):
            scenario_arg = scn['id']
        else:
            scenario_arg = os.path.join(pdir, 'scenario.json')
            with open(scenario_arg, 'w', encoding='utf-8') as fh:
                json.dump(scn, fh)
        argv = _study_argv(b, ppath, scenario_arg, fake_carla)
    except (B.BundleError, OSError, KeyError, ValueError, TypeError) as e:
        em.log(f'refused: {e}', 'error')
        return CP.REFUSED
    SV.NODE_OVERRIDES.update(port=cport, tm_port=tport, no_render=bool(b.get('no_render')))
    try:
        if b['command'] == 'failrate':
            from ..sim.estimate import main_failrate
            code = main_failrate(argv)
        else:
            from ..sim.studio import main_study
            code = main_study(argv)
    finally:
        SV.NODE_OVERRIDES.clear()
    ws = os.path.join(pdir, 'study.falcon')
    sdir = os.path.join(ws, 'studies', str(b['name']))
    if os.path.isdir(sdir):
        names = _summary(sdir, os.path.join(out, 'summary'))
        with open(os.path.join(out, 'summary', 'workspace.json'), 'w', encoding='utf-8') as fh:
            json.dump({'workspace': os.path.abspath(ws), 'files': names}, fh)
    return int(code)


# ================================================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='falcon worker-daemon',
                                 description='a compute node\'s job daemon (docs/cluster.md)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('serve', help='serve slot N on 127.0.0.1:2350+N until stopped')
    s.add_argument('--slot', type=int, default=0)
    s.add_argument('--port', type=int, default=0, help='listen here instead of 2350+N (tests)')
    s.add_argument('--workspace', default='', help='the scratch workspace (default ~/.cache/FalconAutonomy/worker-N)')
    s.add_argument('--no-render', action='store_true', help='start CARLA without rendering (-nullrhi)')
    s.add_argument('--heartbeat', type=float, default=W.HEARTBEAT_S, help='seconds between heartbeats')
    s.add_argument('--fake-carla', default='')
    o = sub.add_parser('oneshot', help='run one bundle and exit with its code')
    o.add_argument('--bundle', required=True)
    o.add_argument('--out', required=True)
    o.add_argument('--slot', type=int, default=None, help='the CARLA ports of slot N (default $FALCON_SLOT, '
                                                         '$SLURM_LOCALID or 0)')
    o.add_argument('--hold-s', type=float, default=0.0, help='then wait up to this long for OUT/.collected')
    o.add_argument('--fake-carla', default='')
    a = ap.parse_args(argv)
    if a.cmd == 'oneshot':
        return oneshot(a.bundle, a.out, a.fake_carla, a.hold_s, a.slot)
    if not 0 <= a.slot < 50:
        print('failed: --slot is 0–49', file=sys.stderr)
        return 2
    d = Daemon(a.slot, a.port, a.workspace, a.fake_carla, a.no_render, a.heartbeat)
    try:
        d.bind()
    except OSError as e:
        print(f'failed: cannot listen on 127.0.0.1:{d.port}: {e.strerror}', file=sys.stderr)
        return 2

    def stop(*_):
        d.stop.set()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    d.serve()
    return 0


if __name__ == '__main__':
    sys.exit(main())
