"""A Slurm cluster as a batch node (standard library; docs/cluster.md, "Slurm").

Jobs dispatched within a tick go together as one sbatch array. Each array lives in its own folder under the node's
workdir — a folder its compute nodes share — holding bundles/<i>.bundle, logs/<i>.out and out/<i>/, and runs

    #SBATCH --array=0-<n-1>  (--partition, --account, --gres, --time from the node; sbatch_extra lines as given)
    <falcon> oneshot --bundle <dir>/bundles/$SLURM_ARRAY_TASK_ID.bundle --out <dir>/out/$SLURM_ARRAY_TASK_ID

(<falcon> is the node's `falcon` command, falcon-worker-daemon by default; each task takes its CARLA ports from
$SLURM_LOCALID, so tasks sharing a machine do not collide). Falcon reaches the login host over SSH with the user's
key (transport.Connection), or runs sbatch here when the node has no host (Falcon on the login node). Every
poll_s it asks squeue and sacct for the tasks' states and reads the new part of each task's log (its protocol lines:
progress in the Jobs dock); a finished task's out/<i> comes back as a tar and is unpacked into the job's run folder.
NODE_FAIL, PREEMPTED and BOOT_FAIL lose the task (queued again, as a lost daemon's); scancel cancels one.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import threading
import time
import uuid

from ..core import protocol as CP
from . import protocol as W

LOST_STATES = ('NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL')
END_STATES = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'DEADLINE') + LOST_STATES


class Backend:
    def __init__(self, sched, node: dict, poll_s: float = 5.0):
        self.sched, self.node = sched, node
        self.key = f'{node["name"]}/slurm'
        self.error = ''
        self.poll_s = float(node.get('poll_s') or poll_s)
        self.pending: list = []             # (rid, job, bundle bytes) not yet in an array
        self.tasks: dict = {}               # rid -> {'batch', 'i', 'jobid', 'offset', 'state'}
        self.batches: dict = {}             # batch -> {'dir', 'jobid', 'rids': [..]}
        self.lock = threading.Lock()
        self.last = 0.0
        self.conn = None
        self.host = bool(node.get('host'))

    # ---------------------------------------------------------------- the scheduler's side
    def free(self) -> int:
        return max(int(self.node.get('max_jobs') or 4) - self.running(), 0)

    def running(self) -> int:
        """This process's jobs on the node (counted by the scheduler from the moment it assigns one)."""
        with self.sched.lock:
            return sum(1 for r in self.sched.jobs.values() if r['where'] is self)

    def submit(self, rid, job, data):
        with self.lock:
            self.pending.append((rid, job, data))

    def cancel(self, rid):
        with self.lock:
            pend = [p for p in self.pending if p[0] == rid]
            self.pending = [p for p in self.pending if p[0] != rid]
            t = self.tasks.get(rid)
        if pend:
            self.sched.on_batch_done(self, rid, CP.CANCELLED, None, '', '')
            return
        if t is not None and t.get('jobid'):
            self._run(['scancel', f'{t["jobid"]}_{t["i"]}'])

    # ---------------------------------------------------------------- running commands here or on the login host
    def _run(self, argv: list, stdin: bytes = b'', timeout: float = 60.0) -> tuple:
        if self.host:
            from .transport import Connection
            if self.conn is None:
                self.conn = Connection(self.node)
            return self.conn.exec(' '.join(shlex.quote(a) for a in argv), stdin=stdin, timeout=timeout)
        try:
            r = subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout)
        except FileNotFoundError:
            raise OSError(f'{argv[0]} is not installed here (a Slurm node without a host runs sbatch on this '
                          'machine)') from None
        except subprocess.TimeoutExpired:
            raise OSError(f'{argv[0]} did not answer within {timeout:.0f} s') from None
        return r.returncode, r.stdout, r.stderr

    def _write(self, path: str, data: bytes):
        if self.host:
            from .transport import Connection
            if self.conn is None:
                self.conn = Connection(self.node)
            self.conn.put(data, path)
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as fh:
            fh.write(data)

    # ---------------------------------------------------------------- arrays
    def script(self, d: str, n: int) -> str:
        nd = self.node
        lines = ['#!/bin/bash', f'#SBATCH --job-name=falcon-{os.path.basename(d)}', f'#SBATCH --array=0-{n - 1}',
                 f'#SBATCH --output={d}/logs/%a.out']
        for opt, key in (('partition', 'partition'), ('account', 'account'), ('gres', 'gres'),
                         ('time', 'time_limit')):
            if nd.get(key):
                lines.append(f'#SBATCH --{opt}={nd[key]}')
        for extra in nd.get('sbatch_extra') or ():
            if str(extra).startswith('--') and '\n' not in str(extra):
                lines.append(f'#SBATCH {extra}')
        falcon = str(nd.get('falcon') or 'falcon-worker-daemon')
        lines.append(f'exec {falcon} oneshot --bundle {d}/bundles/"$SLURM_ARRAY_TASK_ID".bundle '
                     f'--out {d}/out/"$SLURM_ARRAY_TASK_ID"')
        return '\n'.join(lines) + '\n'

    def _flush(self):
        with self.lock:
            batch, self.pending = self.pending, []
        if not batch:
            return
        bid = uuid.uuid4().hex[:12]
        d = os.path.join(self.node['workdir'], f'falcon-{bid}')
        try:
            for i, (_rid, _job, data) in enumerate(batch):
                self._write(f'{d}/bundles/{i}.bundle', data)
            self._write(f'{d}/logs/.keep', b'')
            self._write(f'{d}/array.sh', self.script(d, len(batch)).encode())
            code, out, err = self._run(['sbatch', '--parsable', f'{d}/array.sh'])
            if code != 0:
                raise OSError(f'sbatch failed: {err.decode("utf-8", "replace").strip()[-300:]}')
            jobid = out.decode().strip().split(';')[0]
            if not jobid.isdigit():
                raise OSError(f'sbatch printed no job id ({out[:80]!r})')
        except OSError as e:
            for rid, _job, _data in batch:
                self.sched.on_batch_done(self, rid, CP.FAILED, None, '', f'could not submit to {self.key}: {e}')
            return
        with self.lock:
            self.batches[bid] = {'dir': d, 'jobid': jobid, 'rids': [r for r, _j, _d in batch]}
            for i, (rid, _job, _data) in enumerate(batch):
                self.tasks[rid] = {'batch': bid, 'i': i, 'jobid': jobid, 'offset': 0, 'state': 'PENDING'}
        for i, (rid, _job, _data) in enumerate(batch):
            self.sched.on_batch_message(rid, {'t': 'log', 'level': 'info',
                                              'text': f'Slurm array {jobid}, task {i} (on {self.key})'})

    # ---------------------------------------------------------------- polling
    def poll(self):
        now = time.monotonic()
        if self.pending:
            self._flush()
        if now - self.last < self.poll_s or not self.tasks:
            return
        self.last = now
        with self.lock:
            jobids = sorted({t['jobid'] for t in self.tasks.values()})
        states = {}
        for jid in jobids:
            states.update(self._states(jid))
        self.error = ''
        for rid, t in list(self.tasks.items()):
            self._read_log(rid, t)
            st = states.get(f'{t["jobid"]}_{t["i"]}', t['state'])
            t['state'] = st
            if st in LOST_STATES:
                self._forget(rid)
                self.sched._lost(rid, f'{self.key} lost task {t["jobid"]}_{t["i"]} ({st})')
            elif st in END_STATES:
                self._collect(rid, t, st)

    def _states(self, jid: str) -> dict:
        out = {}
        code, so, _se = self._run(['squeue', '-h', '-r', '-j', jid, '-o', '%i|%T'])
        if code == 0:
            for ln in so.decode().splitlines():
                parts = ln.strip().split('|')
                if len(parts) == 2:
                    out[parts[0]] = parts[1]
        code, so, _se = self._run(['sacct', '-n', '-P', '-X', '-j', jid, '-o', 'JobID,State,ExitCode'])
        if code == 0:
            for ln in so.decode().splitlines():
                parts = ln.strip().split('|')
                if len(parts) >= 2 and parts[0] not in out:
                    out[parts[0]] = parts[1].split()[0] if parts[1] else ''
        return out

    def _read_log(self, rid, t):
        d = self.batches[t['batch']]['dir']
        path = f'{d}/logs/{t["i"]}.out'
        if self.host:
            code, data, _e = self._run(['tail', '-c', f'+{t["offset"] + 1}', path])
            if code != 0:
                return
        else:
            try:
                with open(path, 'rb') as fh:
                    fh.seek(t['offset'])
                    data = fh.read()
            except OSError:
                return
        end = data.rfind(b'\n')
        if end < 0:
            return
        t['offset'] += end + 1
        for ln in data[:end].decode('utf-8', 'replace').splitlines():
            msg = CP.parse_line(ln)
            if msg.get('t') in ('oneshot', 'hello'):
                continue
            self.sched.on_batch_message(rid, msg)

    def _collect(self, rid, t, state):
        d = self.batches[t['batch']]['dir']
        out = f'{d}/out/{t["i"]}'
        code = {'COMPLETED': CP.OK, 'CANCELLED': CP.CANCELLED}.get(state, CP.FAILED)
        tar, err = None, ''
        try:
            if self.host:
                c, data, _e = self._run(['cat', f'{out}/exit.json'])
                ex = json.loads(data) if c == 0 else {}
                c, tar, e = self._run(['tar', '-C', out, '--exclude=./drive.json', '--exclude=./inputs',
                                       '--exclude=./exit.json', '-cf', '-', '.'], timeout=600)
                if c != 0:
                    raise OSError(f'tar failed: {e.decode("utf-8", "replace")[-200:]}')
            else:
                with open(os.path.join(out, 'exit.json'), encoding='utf-8') as fh:
                    ex = json.load(fh)
                tar = W.pack_dir(out, skip=('drive.json', 'inputs', 'exit.json'))
            if state != 'CANCELLED' and isinstance(ex.get('code'), int):
                code = ex['code']
        except (OSError, ValueError) as e:
            err = f'{self.key}: the task ended ({state}) without its run folder: {e}'
            tar = None
        if state in ('TIMEOUT', 'OUT_OF_MEMORY', 'DEADLINE'):
            err = err or f'{self.key}: Slurm ended the task ({state})'
        self._forget(rid)
        self.sched.on_batch_done(self, rid, code, tar, out, err)

    def _forget(self, rid):
        with self.lock:
            t = self.tasks.pop(rid, None)
            if t is None:
                return
            b = self.batches.get(t['batch'])
            if b is not None:
                b['rids'].remove(rid)
                if not b['rids']:
                    self.batches.pop(t['batch'], None)
                    self._cleanup(b['dir'])

    def _cleanup(self, d: str):
        if not os.path.basename(d).startswith('falcon-'):
            return
        if self.host:
            try:
                self._run(['rm', '-rf', '--', d])
            except OSError:
                pass
        else:
            shutil.rmtree(d, ignore_errors=True)
