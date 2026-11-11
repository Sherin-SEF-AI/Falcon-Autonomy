"""What the cluster tests share: free loopback ports, worker daemons started as children of the test (each with its
own config and cache folders, as separate machines would have; on 127.0.0.1 only, against the fake carla) and
stopped at the end."""
import os
import socket
import subprocess
import sys
import time

from conftest import FAKE, ROOT


def free_port() -> int:
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Daemons:
    def __init__(self, base, heartbeat=1.0):
        self.base, self.heartbeat = str(base), heartbeat
        self.procs: dict = {}
        self.homes: dict = {}
        self.nodes: list = []

    def start(self, name: str, pool: str = 'default', slot: int = 0, fake: str = FAKE) -> dict:
        port = free_port()
        home = os.path.join(self.base, name)
        os.makedirs(home, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
        env.update(XDG_CONFIG_HOME=os.path.join(home, 'config'), XDG_CACHE_HOME=os.path.join(home, 'cache'),
                   PYTHONPATH=ROOT, PYTHONUNBUFFERED='1')
        log = open(os.path.join(home, 'daemon.log'), 'w')
        p = subprocess.Popen([sys.executable, '-m', 'falcon_autonomy.cluster.daemon', 'serve', '--slot', str(slot),
                              '--port', str(port), '--fake-carla', fake, '--heartbeat', str(self.heartbeat)],
                             env=env, stdout=log, stderr=subprocess.STDOUT, cwd=home)
        t0 = time.monotonic()
        while 'listening' not in open(os.path.join(home, 'daemon.log')).read():
            assert p.poll() is None, open(os.path.join(home, 'daemon.log')).read()
            assert time.monotonic() - t0 < 30, 'the daemon did not start'
            time.sleep(0.05)
        self.procs[name] = p
        self.homes[name] = home
        node = {'name': name, 'kind': 'local', 'pool': pool, 'slots': [{'slot': slot, 'port': port}],
                'token_dir': os.path.join(home, 'config', 'FalconAutonomy')}
        self.nodes.append(node)
        return node

    def kill(self, name: str):
        p = self.procs[name]
        p.kill()
        p.wait(10)

    def stop(self):
        for p in self.procs.values():
            if p.poll() is None:
                p.terminate()
        for p in self.procs.values():
            try:
                p.wait(20)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(5)


def wait_for(cond, timeout=60.0, poll=None, interval=0.1):
    t0 = time.monotonic()
    while not cond():
        if poll is not None:
            poll()
        assert time.monotonic() - t0 < timeout, 'timed out'
        time.sleep(interval)
