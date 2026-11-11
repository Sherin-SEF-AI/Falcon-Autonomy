"""A Kubernetes cluster as a batch node, through kubectl (standard library; docs/cluster.md, "Kubernetes").

Each job is a Kubernetes Job of one pod running the Falcon worker image (packaging/cluster/Dockerfile: CARLA 0.9.16
and Falcon) with `falcon-worker-daemon oneshot --bundle /bundle/bundle --out /work/run --hold-s H` (as the
container's args, so the image's entry point runs it as its user carla: CARLA refuses root): the bundle in a
ConfigMap of its own (Kubernetes caps one at 1 MiB, so a larger bundle is refused here), the node's GPU limit
(nvidia.com/gpu), activeDeadlineSeconds from the node's deadline_s, no retries (backoffLimit 0: Falcon decides) and
ttlSecondsAfterFinished. Falcon follows the pod's log (`kubectl logs -f`: its protocol lines, progress in the Jobs
dock); when the worker is done the pod holds, Falcon copies /work/run out (`kubectl exec … tar`), tells the pod it may
end (/work/run/.collected) and deletes the Job and its ConfigMap. A pod that fails or is evicted before that loses the
job (queued again, as a lost daemon's); cancelling deletes the Job.

The node's fields: context and namespace (kubectl's own defaults when empty), image, gpu (default 1), deadline_s
(default 3600), hold_s (default 900), kubectl (the program, default kubectl), max_jobs.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time

from ..core import protocol as CP

CONFIGMAP_MAX = 1 << 20


def job_name(rid: str) -> str:
    return 'falcon-' + rid.lower().replace('.', '-').replace('_', '-')[:50]


def manifest(node: dict, name: str, rid: str) -> dict:
    """The Job (JSON, which kubectl applies as it does YAML; packaging/cluster/k8s-job.yaml is the same Job)."""
    gpu = int(node.get('gpu') if node.get('gpu') is not None else 1)
    limits = {'nvidia.com/gpu': gpu} if gpu > 0 else {}
    labels = {'app': 'falcon-worker', 'falcon-dispatch': rid.replace('.', '-')}
    return {
        'apiVersion': 'batch/v1', 'kind': 'Job', 'metadata': {'name': name, 'labels': labels},
        'spec': {'backoffLimit': 0, 'activeDeadlineSeconds': int(node.get('deadline_s') or 3600),
                 'ttlSecondsAfterFinished': 600,
                 'template': {'metadata': {'labels': labels}, 'spec': {
                     'restartPolicy': 'Never',
                     'containers': [{'name': 'worker', 'image': str(node['image']),
                                     'args': ['falcon-worker-daemon', 'oneshot', '--bundle', '/bundle/bundle',
                                                 '--out', '/work/run', '--hold-s',
                                                 str(int(node.get('hold_s') or 900))],
                                     'resources': {'limits': limits},
                                     'volumeMounts': [{'name': 'bundle', 'mountPath': '/bundle', 'readOnly': True},
                                                      {'name': 'work', 'mountPath': '/work'}]}],
                     'volumes': [{'name': 'bundle', 'configMap': {'name': name}},
                                 {'name': 'work', 'emptyDir': {}}]}}}}


class Backend:
    def __init__(self, sched, node: dict, poll_s: float = 5.0):
        self.sched, self.node = sched, node
        self.key = f'{node["name"]}/k8s'
        self.error = ''
        self.poll_s = float(node.get('poll_s') or poll_s)
        self.jobs: dict = {}                # rid -> {'name', 'pod', 'state', 'thread', 'holding': code | None}
        self.lock = threading.Lock()
        self.last = 0.0

    def free(self) -> int:
        return max(int(self.node.get('max_jobs') or 4) - self.running(), 0)

    def running(self) -> int:
        with self.sched.lock:
            return sum(1 for r in self.sched.jobs.values() if r['where'] is self)

    # ---------------------------------------------------------------- kubectl
    def kubectl(self, *args, stdin: bytes = b'', timeout: float = 60.0) -> tuple:
        argv = [str(self.node.get('kubectl') or 'kubectl')]
        if self.node.get('context'):
            argv += ['--context', str(self.node['context'])]
        if self.node.get('namespace'):
            argv += ['-n', str(self.node['namespace'])]
        try:
            r = subprocess.run(argv + list(args), input=stdin, capture_output=True, timeout=timeout)
        except FileNotFoundError:
            raise OSError(f'{argv[0]} is not installed') from None
        except subprocess.TimeoutExpired:
            raise OSError(f'kubectl did not answer within {timeout:.0f} s') from None
        return r.returncode, r.stdout, r.stderr

    def _must(self, *args, stdin: bytes = b'', timeout: float = 60.0) -> bytes:
        code, out, err = self.kubectl(*args, stdin=stdin, timeout=timeout)
        if code != 0:
            raise OSError(f'kubectl {args[0]} failed: {err.decode("utf-8", "replace").strip()[-300:]}')
        return out

    # ---------------------------------------------------------------- the scheduler's side
    def submit(self, rid, job, data):
        if len(data) > CONFIGMAP_MAX - 4096:
            raise OSError(f'the bundle ({len(data) >> 10} KiB) is over what a Kubernetes ConfigMap holds (1 MiB)')
        name = job_name(rid)
        with tempfile.TemporaryDirectory(prefix='falcon-k8s-') as d:
            p = os.path.join(d, 'bundle')
            with open(p, 'wb') as fh:
                fh.write(data)
            self._must('create', 'configmap', name, f'--from-file=bundle={p}')
        try:
            self._must('apply', '-f', '-', stdin=json.dumps(manifest(self.node, name, rid)).encode())
        except OSError:
            self.kubectl('delete', 'configmap', name, '--ignore-not-found')
            raise
        with self.lock:
            self.jobs[rid] = {'name': name, 'pod': '', 'state': 'Pending', 'thread': None, 'holding': None}
        self.sched.on_batch_message(rid, {'t': 'log', 'level': 'info', 'text': f'Kubernetes Job {name} (on {self.key})'})

    def cancel(self, rid):
        with self.lock:
            j = self.jobs.pop(rid, None)
        if j is None:
            return
        self._delete(j['name'])
        self.sched.on_batch_done(self, rid, CP.CANCELLED, None, '', '')

    def _delete(self, name):
        try:
            self.kubectl('delete', 'job', name, '--ignore-not-found', '--wait=false')
            self.kubectl('delete', 'configmap', name, '--ignore-not-found')
        except OSError:
            pass

    # ---------------------------------------------------------------- polling
    def poll(self):
        now = time.monotonic()
        if now - self.last < self.poll_s or not self.jobs:
            return
        self.last = now
        for rid, j in list(self.jobs.items()):
            if j['holding'] is not None:
                self._collect(rid, j)
                continue
            out = self._must('get', 'pods', '-l', f'job-name={j["name"]}', '-o', 'json')
            items = (json.loads(out or b'{}').get('items') or [])
            if not items:
                continue
            pod = items[0]
            j['pod'] = pod['metadata']['name']
            phase = (pod.get('status') or {}).get('phase') or 'Pending'
            j['state'] = phase
            if phase in ('Running', 'Succeeded') and j['thread'] is None:
                t = threading.Thread(target=self._follow, args=(rid, j), daemon=True)
                j['thread'] = t
                t.start()
            elif phase == 'Failed' and j['holding'] is None:
                reason = (pod.get('status') or {}).get('reason') or 'failed'
                with self.lock:
                    self.jobs.pop(rid, None)
                self._delete(j['name'])
                if reason in ('DeadlineExceeded',):
                    self.sched.on_batch_done(self, rid, CP.FAILED, None, '', f'{self.key}: the Job passed its '
                                                                              'deadline')
                else:
                    self.sched._lost(rid, f'{self.key} lost pod {j["pod"]} ({reason})')

    def _follow(self, rid, j):
        """kubectl logs -f: the worker's protocol lines, until the pod holds (its oneshot line says the code)."""
        argv = [str(self.node.get('kubectl') or 'kubectl')]
        if self.node.get('context'):
            argv += ['--context', str(self.node['context'])]
        if self.node.get('namespace'):
            argv += ['-n', str(self.node['namespace'])]
        try:
            p = subprocess.Popen(argv + ['logs', '-f', j['pod']], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 text=True)
        except OSError:
            return
        for line in p.stdout:
            msg = CP.parse_line(line)
            if msg.get('t') == 'oneshot':
                j['holding'] = int(msg.get('code') if msg.get('code') is not None else CP.CRASHED)
                break
            if msg.get('t') != 'hello':
                self.sched.on_batch_message(rid, msg)
        p.terminate()
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()

    def _collect(self, rid, j):
        with self.lock:
            if self.jobs.pop(rid, None) is None:
                return
        code, tar, err = j['holding'], None, ''
        try:
            tar = self._must('exec', j['pod'], '--', 'tar', '-C', '/work/run', '--exclude=./drive.json',
                             '--exclude=./inputs', '--exclude=./exit.json', '-cf', '-', '.', timeout=600)
            self.kubectl('exec', j['pod'], '--', 'touch', '/work/run/.collected')
        except OSError as e:
            err = f'{self.key}: the run folder could not be copied out of the pod: {e}'
            tar = None
            if code == CP.OK:
                code = CP.FAILED
        self._delete(j['name'])
        self.sched.on_batch_done(self, rid, code, tar, '/work/run', err)
