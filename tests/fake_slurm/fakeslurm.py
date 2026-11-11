#!/usr/bin/env python3
"""A fake Slurm for the tests (sbatch, squeue, sacct, scancel are links to this file): array tasks run here, at once,
as background processes; their state lives in $FAKE_SLURM_DIR. $FAKE_SLURM_NODE_FAIL=<task index> makes sacct report
NODE_FAIL for that task of the first array (once), killing it — a lost compute node. `squeue -h -o %j` (no -j) lists
the names of the arrays with a task still running; $FAKE_SLURM_SQUEUE_FAIL makes squeue fail (no controller)."""
import json
import os
import re
import signal
import subprocess
import sys

D = os.environ['FAKE_SLURM_DIR']
os.makedirs(D, exist_ok=True)


def _next_id() -> int:
    p = os.path.join(D, 'next')
    n = int(open(p).read()) if os.path.exists(p) else 1000
    open(p, 'w').write(str(n + 1))
    return n


def _state(task: str) -> str:
    p = os.path.join(D, task)
    if os.path.exists(p + '.nodefail'):
        return 'NODE_FAIL'
    if os.path.exists(p + '.cancelled'):
        return 'CANCELLED'
    if os.path.exists(p + '.rc'):
        rc = open(p + '.rc').read().strip()
        return 'COMPLETED' if rc == '0' else 'FAILED'
    return 'RUNNING'


def sbatch(args):
    script = args[-1]
    text = open(script).read()
    arr = re.search(r'#SBATCH --array=(\d+)-(\d+)', text)
    out = re.search(r'#SBATCH --output=(\S+)', text).group(1)
    jid = _next_id()
    lo, hi = int(arr.group(1)), int(arr.group(2))
    for i in range(lo, hi + 1):
        task = f'{jid}_{i}'
        log = out.replace('%a', str(i)).replace('%A', str(jid))
        os.makedirs(os.path.dirname(log), exist_ok=True)
        env = dict(os.environ, SLURM_ARRAY_TASK_ID=str(i), SLURM_ARRAY_JOB_ID=str(jid), SLURM_JOB_ID=str(jid + i + 1),
                   SLURM_LOCALID='0')
        rc = os.path.join(D, task + '.rc')
        p = subprocess.Popen(['bash', '-c', f'bash {script}; echo $? > {rc}'], env=env, stdout=open(log, 'w'),
                             stderr=subprocess.STDOUT, start_new_session=True)
        json.dump({'pid': p.pid}, open(os.path.join(D, task + '.json'), 'w'))
        with open(os.path.join(D, 'pids'), 'a') as fh:
            fh.write(f'{p.pid}\n')
        if os.environ.get('FAKE_SLURM_NODE_FAIL') == str(i) and not os.path.exists(os.path.join(D, 'failed-once')):
            open(os.path.join(D, 'failed-once'), 'w').write(task)
            open(os.path.join(D, task + '.failnext'), 'w').write('')
    with open(os.path.join(D, 'submitted'), 'a') as fh:
        fh.write(json.dumps({'job': jid, 'script': script, 'text': text, 'tasks': hi - lo + 1}) + '\n')
    print(jid)


def _tasks(jid):
    return sorted(f[:-5] for f in os.listdir(D) if f.startswith(f'{jid}_') and f.endswith('.json'))


def _maybe_fail(task):
    p = os.path.join(D, task)
    if os.path.exists(p + '.failnext') and os.path.getsize(os.path.join(D, task + '.json')) and \
            os.path.exists(os.path.join(D, 'squeue-seen-' + task)):
        os.remove(p + '.failnext')
        _kill(task)
        open(p + '.nodefail', 'w').write('')


def _kill(task):
    try:
        pid = json.load(open(os.path.join(D, task + '.json')))['pid']
        os.killpg(pid, signal.SIGKILL)
    except (OSError, ValueError, KeyError):
        pass


def squeue(args):
    if os.environ.get('FAKE_SLURM_SQUEUE_FAIL'):
        print('squeue: error: slurm_load_jobs error: Unable to contact slurm controller', file=sys.stderr)
        sys.exit(1)
    if '-j' not in args:                        # squeue -h -o %j: the names of the jobs in the queue
        for ln in open(os.path.join(D, 'submitted')) if os.path.exists(os.path.join(D, 'submitted')) else ():
            sub = json.loads(ln)
            name = re.search(r'#SBATCH --job-name=(\S+)', sub['text'])
            if name and any(_state(t) == 'RUNNING' for t in _tasks(sub['job'])):
                print(name.group(1))
        return
    jid = args[args.index('-j') + 1]
    for t in _tasks(jid):
        open(os.path.join(D, 'squeue-seen-' + t), 'w').write('')
        _maybe_fail(t)
        st = _state(t)
        if st == 'RUNNING':
            print(f'{t}|RUNNING')


def sacct(args):
    jid = args[args.index('-j') + 1]
    for t in _tasks(jid):
        st = _state(t)
        if st != 'RUNNING':
            print(f'{t}|{st}|0:0')


def scancel(args):
    task = args[-1]
    _kill(task)
    open(os.path.join(D, task + '.cancelled'), 'w').write('')


if __name__ == '__main__':
    {'sbatch': sbatch, 'squeue': squeue, 'sacct': sacct, 'scancel': scancel}[os.path.basename(sys.argv[0])](
        sys.argv[1:])
