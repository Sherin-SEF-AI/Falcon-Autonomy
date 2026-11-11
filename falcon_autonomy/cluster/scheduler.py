"""The cluster's scheduler: which node slot runs a job, and what happens when a node goes quiet (standard library;
docs/cluster.md, "Scheduling").

* Slots. A daemon node (ssh, local) has slots, each one daemon with one CARLA (slot N: 127.0.0.1:2350+N on the node);
  a batch node (slurm, k8s) has max_jobs. A slot takes one of this process's jobs at a time.
* Matching. A job goes to a node of its pool (placement 'cluster:<pool>'; plain 'cluster' is every pool) that has its
  backend (meta['backend'], default carla-0.9.16; a node that lists no backends has the default) and at least its
  video memory (meta['vram_gb'] against the node's vram_gb). No such node: refused. Busy: it waits in the queue.
* Pools and quotas. A pool may cap the jobs it runs at once (max_running: they wait) and per day (max_per_day: refused
  past it; counted in cluster/usage.json in the config folder).
* Ids. A job keeps one global id (JobSpec.gid, a random 20-hex id); each dispatch of it is "<gid>.<attempt>".
* Heartbeats. A daemon sends one every 15 s. A slot not heard from for 60 s (LOSS_S) is lost: each job it ran is
  queued again when it is resumable — a drive is: it writes only its own folder — at most MAX_REQUEUES times, and
  fails otherwise. A dropped connection is retried meanwhile; a daemon that comes back is told to cancel what was
  queued elsewhere. A batch node's job lost by its system (NODE_FAIL, PREEMPTED, an evicted pod) is lost the same way.
* Results. A finished job's run folder comes back as the daemon's artifact and is unpacked into the job's own run
  folder (core.jobs: before the queue hears that it ended), with the node's folder replaced by this one in its text
  files. run.json keeps the config as it was written here (drive.json) in provenance.settings and says in "node" where
  it ran: the node, the slot, the dispatch id and the values that were the node's own (its CARLA ports, its paths).

Everything network happens on the scheduler's threads: can_start() and start() (called by JobQueue.poll on the GUI's
thread) never wait on a node.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import socket
import threading
import time
import uuid

from ..core import protocol as CP
from ..core.jobs import Refused
from . import bundle as B
from . import nodes as N
from . import protocol as W

LOSS_S = 60.0
MAX_REQUEUES = 2
RETRY_S = 5.0


def _today() -> str:
    return _dt.date.today().isoformat()


class Usage:
    """Jobs started per pool and day (the max_per_day quota), in cluster/usage.json of the config folder."""

    def __init__(self, path: str = ''):
        from ..core.paths import config_dir
        self.path = path or os.path.join(config_dir(), 'cluster', 'usage.json')
        self.lock = threading.Lock()

    def _read(self) -> dict:
        try:
            with open(self.path, encoding='utf-8') as fh:
                d = json.load(fh)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def today(self, pool: str) -> int:
        return int((self._read().get(_today()) or {}).get(pool, 0))

    def add(self, pool: str):
        from ..core.paths import atomic_write
        with self.lock:
            d = self._read()
            day = _today()
            d = {day: d.get(day) or {}}                     # only today's counts are kept
            d[day][pool] = int(d[day].get(pool, 0)) + 1
            atomic_write(self.path, json.dumps(d))


# ================================================================================================ results
def import_run(job, tar: bytes, scratch: str, where: dict) -> list:
    """Unpack a finished run (an uncompressed tar of its folder) into the job's run folder, the node's folder made
    this one in its text files, and run.json told where it ran (module doc). The files' names."""
    from ..core.provenance import sha256_file, sha256_json
    run_dir = job.spec.run_dir
    if not run_dir:
        return []
    names = W.unpack_tar(tar, run_dir)
    W.translate_paths(run_dir, names, scratch, run_dir)
    rj = os.path.join(run_dir, 'run.json')
    cfg_path = job.spec.meta.get('config') or ''
    try:
        with open(rj, encoding='utf-8') as fh:
            run = json.load(fh)
    except (OSError, ValueError):
        return names
    node = dict(where)
    local = None
    try:
        with open(cfg_path, encoding='utf-8') as fh:
            local = json.load(fh)
    except (OSError, ValueError):
        pass
    prov = run.get('provenance') if isinstance(run.get('provenance'), dict) else None
    if prov is not None and isinstance(local, dict) and isinstance(prov.get('settings'), dict):
        theirs = prov['settings']
        node['settings'] = {k: theirs.get(k) for k in sorted(set(theirs) | set(local))
                            if theirs.get(k) != local.get(k)}
        prov['settings'] = local
        cfg_in = (prov.get('inputs') or {}).get('config')
        if isinstance(cfg_in, dict) and cfg_in.get('path') == cfg_path and os.path.isfile(cfg_path):
            node['config_file_sha256'] = cfg_in.get('sha256')
            cfg_in['sha256'] = sha256_file(cfg_path)
        if 'config_sha256' in run:                              # the worker's hash of the config it ran
            node['config_sha256'] = run['config_sha256']
            run['config_sha256'] = sha256_json(local)
    run['node'] = node
    from ..core.paths import atomic_write
    atomic_write(rj, json.dumps(run, indent=1) + '\n')
    return names


# ================================================================================================ slots
class Slot:
    """One daemon slot of a node: its connection (made and remade by the scheduler's thread) and its jobs."""

    def __init__(self, sched, node: dict, slot: int, port: int):
        self.sched, self.node, self.slot, self.port = sched, node, int(slot), int(port)
        self.key = f'{node["name"]}/{slot}'
        self.state = 'down'
        self.chan: W.Channel | None = None
        self.info: dict = {}
        self.last_rx = 0.0
        self.down_since = sched.clock()
        self.last_try = -1e9
        self.error = ''
        self.running: dict = {}             # rid -> job (dispatched here, not finished)
        self.lost: set = set()              # rids queued elsewhere after a loss: cancelled when the slot comes back
        self.conn = None                    # transport.Connection (ssh nodes)

    @property
    def pool(self) -> str:
        return self.node.get('pool', 'default')

    def _token_and_address(self) -> tuple:
        if self.node['kind'] == 'local':
            tok = W.read_token(self.slot, os.path.expanduser(self.node.get('token_dir') or ''))
            return tok, ('127.0.0.1', self.port)
        from . import transport as T
        if self.conn is None:
            self.conn = T.Connection(self.node)
        code, out, _err = self.conn.exec(T.remote_token_command(self.slot), timeout=20)
        if code != 0:
            raise OSError('the slot\'s daemon is not running (no token on the node: systemctl --user start '
                          f'falcon-worker-daemon@{self.slot})')
        return out.decode().strip(), ('127.0.0.1', self.conn.tunnel(self.port))

    def connect(self):
        tok, addr = self._token_and_address()
        if not tok:
            raise OSError(f'slot {self.slot}\'s daemon has written no token (is it running?)')
        s = socket.create_connection(addr, timeout=10)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        ch = W.Channel(s)
        from .. import __version__
        ch.send({'t': 'hello', 'protocol': W.PROTOCOL, 'token': tok, 'app_version': __version__,
                 'codecs': W.codecs()})
        msg, _ = ch.recv()
        if msg.get('t') != 'hello':
            ch.close()
            raise OSError(str(msg.get('why') or 'the daemon refused the connection'))
        s.settimeout(None)
        self.chan, self.info = ch, msg
        self.state, self.error = 'up', ''
        self.last_rx = self.sched.clock()
        threading.Thread(target=self._reader, args=(ch,), daemon=True, name=f'falcon-slot-{self.key}').start()
        if self.lost:
            for rid in list(self.lost):
                self._send({'t': 'cancel', 'rid': rid, 'force': True})
                self._send({'t': 'release', 'rid': rid})
            self.lost.clear()
        if self.running:
            self._send({'t': 'attach', 'rids': list(self.running)})

    def _send(self, msg, payload=b'') -> bool:
        ch = self.chan
        if ch is None:
            return False
        try:
            ch.send(msg, payload)
            return True
        except W.ProtocolError:
            self.drop('connection lost')
            return False

    def drop(self, why: str):
        ch, self.chan = self.chan, None
        if ch is not None:
            ch.close()
        if self.state == 'up':
            self.down_since = self.last_rx
        self.state, self.error = 'down', why

    def _reader(self, ch):
        while True:
            try:
                msg, payload = ch.recv()
            except W.ProtocolError as e:
                if self.chan is ch:
                    self.drop(str(e))
                return
            self.last_rx = self.sched.clock()
            self.sched._on_slot_message(self, msg, payload)


class RemoteHandle:
    def __init__(self, sched, rid: str):
        self.sched, self.rid = sched, rid

    def send(self, cmd: dict) -> bool:
        return False                         # remote jobs take no commands while they run (none is interactive)

    def cancel(self):
        self.sched.cancel(self.rid, force=False)

    def kill(self):
        self.sched.cancel(self.rid, force=True)


# ================================================================================================ the scheduler
class Scheduler:
    def __init__(self, cfg: dict, clock=time.monotonic, loss_s: float = LOSS_S, usage: Usage | None = None,
                 tick_s: float = 0.5):
        self.cfg = cfg
        self.clock = clock
        self.loss_s = float(loss_s)
        self.usage = usage or Usage()
        self.lock = threading.RLock()
        self.slots: list = []
        self.batch: list = []               # slurm.Backend, k8s.Backend
        self.jobs: dict = {}                # rid -> {'job', 'queue', 'where': Slot | backend, 'pool'}
        self.pools = N.pools(cfg)
        for n in cfg.get('nodes') or ():
            if n.get('kind') in ('ssh', 'local'):
                self.slots += [Slot(self, n, s, p) for s, p in N.slots(n)]
            elif n.get('kind') == 'slurm':
                from .slurm import Backend
                self.batch.append(Backend(self, n))
            elif n.get('kind') == 'k8s':
                from .k8s import Backend
                self.batch.append(Backend(self, n))
        self.stop = threading.Event()
        self.tick_s = float(tick_s)
        self.thread = threading.Thread(target=self._run, daemon=True, name='falcon-cluster')
        self.thread.start()

    # ---------------------------------------------------------------- placement
    def _candidates(self, job) -> tuple:
        _w, pool = N.parse_placement(job.spec.placement)
        backend = job.spec.meta.get('backend') or N.DEFAULT_BACKEND
        vram = float(job.spec.meta.get('vram_gb') or 0.0)

        def fits(n):
            return (not pool or n.get('pool', 'default') == pool) and \
                backend in (n.get('backends') or [N.DEFAULT_BACKEND]) and float(n.get('vram_gb') or 1e9) >= vram
        slots = [s for s in self.slots if fits(s.node)]
        batch = [b for b in self.batch if fits(b.node)]
        if not slots and not batch:
            what = f'pool {pool!r}' if pool else 'the cluster'
            need = f' with {backend}' + (f' and {vram:g} GB of video memory' if vram else '')
            raise Refused(f'no node of {what}{need} (Settings ▸ Compute, or falcon cluster nodes add)')
        return pool, slots, batch

    def _pool_wait(self, pool_name: str) -> str:
        p = self.pools.get(pool_name) or {}
        lim = int(p.get('max_running') or 0)
        if lim and sum(1 for j in self.jobs.values() if j['pool'] == pool_name) >= lim:
            return f'waiting: pool {pool_name} runs its quota of {lim} jobs at once'
        return ''

    def _day_used(self, pool_name: str) -> int:
        """The pool's max_per_day when today's jobs reached it, else 0."""
        per_day = int((self.pools.get(pool_name) or {}).get('max_per_day') or 0)
        return per_day if per_day and self.usage.today(pool_name) >= per_day else 0

    def _pick(self, job):
        """(slot or backend, '') or (None, why it waits). Refused when nothing could ever take it."""
        pool, slots, batch = self._candidates(job)
        used = {n.get('pool', 'default'): self._day_used(n.get('pool', 'default'))
                for n in [s.node for s in slots] + [b.node for b in batch]}
        if all(used.values()):
            name, lim = next(iter(used.items()))
            raise Refused(f'pool {name} has used its quota of {lim} jobs today' if len(used) == 1 else
                          'every pool that could run it has used its quota of jobs for today')
        slots = [s for s in slots if not used[s.pool]]
        batch = [b for b in batch if not used[b.node.get('pool', 'default')]]
        free_slots = [s for s in slots if s.state == 'up' and not s.running]
        free_batch = [b for b in batch if b.free() > 0]
        waits = []
        for s in sorted(free_slots, key=lambda s: (int(s.info.get('jobs') or 0), s.key)):
            w = self._pool_wait(s.pool)
            if not w:
                return s, ''
            waits.append(w)
        for b in free_batch:
            w = self._pool_wait(b.node.get('pool', 'default'))
            if not w:
                return b, ''
            waits.append(w)
        if waits:
            return None, waits[0]
        if slots and not batch and all(s.state != 'up' for s in slots):
            down = slots[0]
            return None, (f'waiting for a node to answer ({len(slots)} slot{"s" * (len(slots) > 1)} down: '
                          f'{down.key}: {down.error or "connecting"})')
        return None, f'waiting for a free slot on the cluster{f" (pool {pool})" if pool else ""}'

    def can_start(self, job) -> str:
        with self.lock:
            if getattr(job, '_bundle', None) is None:
                try:
                    job._bundle = B.encode(B.make(job.spec))
                except B.BundleError as e:
                    raise Refused(str(e)) from None
            _where, why = self._pick(job)
            return why

    def start(self, job, queue):
        with self.lock:
            where, _why = self._pick(job)
            if where is None:
                return None                             # a slot went since can_start(): the job waits again
            if not job.spec.gid:
                job.spec.gid = uuid.uuid4().hex[:20]
            rid = f'{job.spec.gid}.{job.requeues + 1}'
            pool = where.node.get('pool', 'default')
            self.jobs[rid] = {'job': job, 'queue': queue, 'where': where, 'pool': pool}
            data = job._bundle
            if isinstance(where, Slot):
                where.running[rid] = job
                job.node = where.key
            else:
                job.node = where.key
            self.usage.add(pool)
        job.log.append(('info', f'on {job.node} ({rid})'))
        threading.Thread(target=self._dispatch, args=(where, rid, job, data), daemon=True).start()
        return RemoteHandle(self, rid)

    def _dispatch(self, where, rid, job, data):
        if isinstance(where, Slot):
            if not where._send({'t': 'submit', 'rid': rid}, data):
                pass                         # the slot dropped: the loss check queues the job again
        else:
            try:
                where.submit(rid, job, data)
            except (OSError, B.BundleError) as e:
                self._end(rid, CP.FAILED, f'could not submit to {where.key}: {e}')

    # ---------------------------------------------------------------- events
    def _job(self, rid):
        with self.lock:
            return self.jobs.get(rid)

    def _on_slot_message(self, slot: Slot, msg: dict, payload: bytes):
        t = msg.get('t')
        rid = str(msg.get('rid') or '')
        if t == 'heartbeat':
            return
        rec = self._job(rid)
        if rec is None or rec['where'] is not slot:
            if t == 'done':                           # a job queued elsewhere after a loss: nobody waits for it
                slot._send({'t': 'release', 'rid': rid})
            return
        job, q = rec['job'], rec['queue']
        if t == 'msg' and isinstance(msg.get('msg'), dict):
            q.on_message(job, msg['msg'])
        elif t == 'refused':
            self._end(rid, CP.REFUSED, f'the node refused it: {msg.get("why")}')
        elif t == 'state' and msg.get('state') == 'unknown':
            self._lost(rid, f'{slot.key} no longer knows the job')
        elif t == 'done':
            code = int(msg.get('code') if msg.get('code') is not None else CP.CRASHED)
            err = ''
            try:
                if W.sha256(payload) != msg.get('sha256'):
                    raise W.ProtocolError('the artifact does not match its hash')
                tar = W.decompress(payload, str(msg.get('codec') or 'zlib'))
                import_run(job, tar, str(msg.get('scratch') or ''), self._where(slot, rid))
            except (W.ProtocolError, OSError) as e:
                err = f'the run folder from {slot.key} could not be unpacked: {e}'
            slot._send({'t': 'release', 'rid': rid})
            self._end(rid, CP.FAILED if err and code == CP.OK else code, err)

    def _where(self, where, rid: str) -> dict:
        d = {'name': where.node['name'], 'kind': where.node['kind'], 'pool': where.node.get('pool', 'default'),
             'dispatch': rid}
        if isinstance(where, Slot):
            d.update(slot=where.slot, host_name=where.info.get('node'), app_version=where.info.get('app_version'))
        return d

    def on_batch_done(self, backend, rid: str, code: int, tar: bytes | None, scratch: str, err: str = ''):
        rec = self._job(rid)
        if rec is None:
            return
        if tar is not None:
            try:
                import_run(rec['job'], tar, scratch, self._where(backend, rid))
            except (W.ProtocolError, OSError) as e:
                err = err or f'the run folder from {backend.key} could not be unpacked: {e}'
                code = CP.FAILED if code == CP.OK else code
        self._end(rid, code, err)

    def on_batch_message(self, rid: str, msg: dict):
        rec = self._job(rid)
        if rec is not None:
            rec['queue'].on_message(rec['job'], msg)

    def _end(self, rid: str, code: int, why: str = ''):
        with self.lock:
            rec = self.jobs.pop(rid, None)
            if rec is None:
                return
            if isinstance(rec['where'], Slot):
                rec['where'].running.pop(rid, None)
        if why:
            rec['job'].log.append(('error', why))
        rec['queue'].on_exit(rec['job'], code)

    def _lost(self, rid: str, why: str):
        """A job whose node was lost: queued again when it is resumable (at most MAX_REQUEUES times), else failed."""
        with self.lock:
            rec = self.jobs.pop(rid, None)
            if rec is None:
                return
            where = rec['where']
            if isinstance(where, Slot):
                where.running.pop(rid, None)
                where.lost.add(rid)
        job, q = rec['job'], rec['queue']
        if job.cancel_requested:
            q.on_exit(job, CP.CANCELLED)
        elif job.spec.resumable and job.requeues < MAX_REQUEUES:
            q.requeue(job, f'{why}: queued again (attempt {job.requeues + 2} of {MAX_REQUEUES + 1})')
            q.poll()
        else:
            job.log.append(('error', f'{why}: '
                            + ('not resumable, so not queued again' if not job.spec.resumable else
                               f'already queued again {MAX_REQUEUES} times')))
            q.on_exit(job, CP.FAILED)

    def cancel(self, rid: str, force: bool = False):
        rec = self._job(rid)
        if rec is None:
            return
        where = rec['where']
        if isinstance(where, Slot):
            if not where._send({'t': 'cancel', 'rid': rid, 'force': bool(force)}) and where.state != 'up':
                self._end(rid, CP.CANCELLED, f'{where.key} is not reachable: cancelled here; it is told when it '
                                             'answers again')
                where.lost.add(rid)
        else:
            threading.Thread(target=where.cancel, args=(rid,), daemon=True).start()

    # ---------------------------------------------------------------- the thread
    def _run(self):
        while not self.stop.wait(self.tick_s):
            now = self.clock()
            for s in list(self.slots):
                if s.state == 'up' and now - s.last_rx > self.loss_s:
                    s.drop(f'silent for {now - s.last_rx:.0f} s')
                if s.state != 'up':
                    if s.running and now - s.down_since > self.loss_s:
                        for rid in list(s.running):
                            self._lost(rid, f'node {s.key} lost ({s.error or "no heartbeat"})')
                    if now - s.last_try >= RETRY_S:
                        s.last_try = now
                        s.state = 'connecting' if s.state == 'down' else s.state
                        try:
                            s.connect()
                        except (OSError, W.ProtocolError) as e:
                            s.state, s.error = 'down', str(e)[:300]
            for b in list(self.batch):
                try:
                    b.poll()
                except Exception as e:              # noqa: BLE001 (a batch system's hiccup: retried next tick)
                    b.error = str(e)[:300]

    def status(self) -> list:
        out = []
        for s in self.slots:
            out.append({'slot': s.key, 'pool': s.pool, 'state': s.state, 'running': len(s.running),
                        'error': s.error, 'node': s.info.get('node', ''), 'app_version': s.info.get('app_version')})
        for b in self.batch:
            out.append({'slot': b.key, 'pool': b.node.get('pool', 'default'), 'state': 'batch',
                        'running': b.running(), 'error': b.error})
        return out

    def close(self):
        self.stop.set()
        for s in self.slots:
            s.drop('closed')
            if s.conn is not None:
                s.conn.close()

