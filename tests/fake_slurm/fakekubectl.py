#!/usr/bin/env python3
"""A fake kubectl for the tests: a Job's pod is a background process here running the container's args, with /bundle
and /work mapped into $FAKE_KUBE_DIR and its first word (falcon-worker-daemon) replaced by $FAKE_KUBE_FALCON.
$FAKE_KUBE_EVICT=1 evicts the first pod once (its phase Failed, reason Evicted): a lost node."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time

D = os.environ['FAKE_KUBE_DIR']
os.makedirs(D, exist_ok=True)


def _strip(argv):
    out, i = [], 0
    while i < len(argv):
        if argv[i] in ('--context', '-n', '--namespace'):
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def _pod_dir(pod):
    return os.path.join(D, 'pods', pod)


def _map(arg, name, pod):
    return arg.replace('/bundle/bundle', os.path.join(D, 'cm', name, 'bundle')).replace(
        '/work/run', os.path.join(_pod_dir(pod), 'work', 'run'))


def main(argv):
    a = _strip(argv)
    with open(os.path.join(D, 'calls.jsonl'), 'a') as fh:
        fh.write(json.dumps(a) + '\n')
    if a[:2] == ['auth', 'can-i']:
        print('yes')
        return 0
    if a[:2] == ['create', 'configmap']:
        name = a[2]
        src = next(x for x in a if x.startswith('--from-file=')).split('=', 2)[2]
        os.makedirs(os.path.join(D, 'cm', name), exist_ok=True)
        shutil.copy(src, os.path.join(D, 'cm', name, 'bundle'))
        print(f'configmap/{name} created')
        return 0
    if a[:2] == ['apply', '-f']:
        job = json.load(sys.stdin)
        name = job['metadata']['name']
        pod = name + '-pod'
        c = job['spec']['template']['spec']['containers'][0]
        args = [_map(x, name, pod) for x in c['args']]
        args[0] = os.environ['FAKE_KUBE_FALCON']
        os.makedirs(os.path.join(_pod_dir(pod), 'work'), exist_ok=True)
        rc = os.path.join(_pod_dir(pod), 'rc')
        cmd = ' '.join(f"'{x}'" for x in args)
        p = subprocess.Popen(['bash', '-c', f'{cmd}; echo $? > {rc}'], stdout=open(os.path.join(_pod_dir(pod), 'log'),
                                                                                   'w'),
                             stderr=subprocess.STDOUT, start_new_session=True)
        json.dump({'pod': pod, 'pid': p.pid, 'job': job}, open(os.path.join(D, name + '.json'), 'w'))
        with open(os.path.join(D, 'pids'), 'a') as fh:
            fh.write(f'{p.pid}\n')
        print(f'job.batch/{name} created')
        return 0
    if a[:2] == ['get', 'pods']:
        name = a[a.index('-l') + 1].split('=', 1)[1]
        try:
            rec = json.load(open(os.path.join(D, name + '.json')))
        except OSError:
            print(json.dumps({'items': []}))
            return 0
        pod = rec['pod']
        status = {'phase': 'Running'}
        rcf = os.path.join(_pod_dir(pod), 'rc')
        if os.path.exists(os.path.join(_pod_dir(pod), 'evicted')):
            status = {'phase': 'Failed', 'reason': 'Evicted'}
        elif os.environ.get('FAKE_KUBE_EVICT') == '1' and not os.path.exists(os.path.join(D, 'evicted-once')):
            open(os.path.join(D, 'evicted-once'), 'w').write(pod)
            open(os.path.join(_pod_dir(pod), 'evicted'), 'w').write('')
            _kill(rec['pid'])
            status = {'phase': 'Failed', 'reason': 'Evicted'}
        elif os.path.exists(rcf):
            status = {'phase': 'Succeeded' if open(rcf).read().strip() == '0' else 'Failed'}
        print(json.dumps({'items': [{'metadata': {'name': pod}, 'status': status}]}))
        return 0
    if a[:2] == ['logs', '-f']:
        pod = a[2]
        log = os.path.join(_pod_dir(pod), 'log')
        rcf = os.path.join(_pod_dir(pod), 'rc')
        with open(log) as fh:
            while True:
                line = fh.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    continue
                if os.path.exists(rcf) or os.path.exists(os.path.join(_pod_dir(pod), 'evicted')):
                    return 0
                time.sleep(0.05)
    if a[0] == 'exec':
        pod = a[1]
        rest = [x.replace('/work/run', os.path.join(_pod_dir(pod), 'work', 'run')) for x in a[a.index('--') + 1:]]
        r = subprocess.run(rest, capture_output=True)
        sys.stdout.buffer.write(r.stdout)
        sys.stderr.buffer.write(r.stderr)
        return r.returncode
    if a[0] == 'delete':
        name = a[2]
        if a[1] == 'job':
            try:
                rec = json.load(open(os.path.join(D, name + '.json')))
                _kill(rec['pid'])
                open(os.path.join(D, name + '.deleted'), 'w').write('')
            except OSError:
                pass
        else:
            shutil.rmtree(os.path.join(D, 'cm', name), ignore_errors=True)
        return 0
    print(f'fake kubectl: {a}', file=sys.stderr)
    return 1


def _kill(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
