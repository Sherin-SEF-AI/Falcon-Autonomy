"""SSH to a node (cluster.transport) against an SSH server in this process on 127.0.0.1 (paramiko's server side):
keys only, a new host key remembered and a changed one refused, commands with stdin and exit codes, a tunnel, and a
whole `ssh` node — its slot's token read over SSH and a drive run through the tunnel on a daemon."""
import json
import os
import socket
import subprocess
import threading

import pytest

paramiko = pytest.importorskip('paramiko')

from cluster_helpers import Daemons, wait_for  # noqa: E402
from conftest import FAKE  # noqa: E402
from falcon_autonomy.cluster import transport as T  # noqa: E402

USER = 'tester'


def _ed25519(path):
    from cryptography.hazmat.primitives import serialization as S
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    k = Ed25519PrivateKey.generate()
    path.write_bytes(k.private_bytes(S.Encoding.PEM, S.PrivateFormat.OpenSSH, S.NoEncryption()))
    path.chmod(0o600)
    return paramiko.Ed25519Key.from_private_key_file(str(path))


class SSHServer:
    """Accepts one user with one public key; runs exec requests with bash here (env: extra_env); forwards
    direct-tcpip channels to 127.0.0.1 only."""

    def __init__(self, allowed, extra_env=None, port=0, host_key=None):
        self.allowed, self.env = allowed, dict(os.environ, **(extra_env or {}))
        self.host_key = host_key or paramiko.RSAKey.generate(2048)
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('127.0.0.1', port))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.commands, self.forwards = [], []
        self.transports = []
        self.stop = False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self.stop:
            try:
                s, _a = self.sock.accept()
            except OSError:
                return
            t = paramiko.Transport(s)
            t.add_server_key(self.host_key)
            self.transports.append(t)
            srv = _Iface(self)
            try:
                t.start_server(server=srv)
            except (paramiko.SSHException, EOFError):
                continue
            threading.Thread(target=self._channels, args=(t, srv), daemon=True).start()

    def _channels(self, t, srv):
        while t.is_active():
            ch = t.accept(1.0)
            if ch is None:
                continue
            dest = srv.dest.pop(ch.get_id(), None)
            if dest is not None:
                try:
                    out = socket.create_connection(dest, timeout=5)
                except OSError:
                    ch.close()
                    continue
                threading.Thread(target=T._pump, args=(ch, out), daemon=True).start()
                threading.Thread(target=T._pump, args=(out, ch), daemon=True).start()

    def run(self, ch, command):
        self.commands.append(command)
        p = subprocess.Popen(['bash', '-c', command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, env=self.env)

        def feed():
            while True:
                data = ch.recv(65536)
                if not data:
                    break
                p.stdin.write(data)
            p.stdin.close()

        def errs():
            for chunk in iter(lambda: p.stderr.read(4096), b''):
                ch.sendall_stderr(chunk)
        tf, te = threading.Thread(target=feed, daemon=True), threading.Thread(target=errs, daemon=True)
        tf.start()
        te.start()
        for chunk in iter(lambda: p.stdout.read1(65536), b''):
            ch.sendall(chunk)
        te.join(10)
        ch.send_exit_status(p.wait())
        ch.close()

    def close(self):
        self.stop = True
        self.sock.close()
        for t in self.transports:
            t.close()


class _Iface(paramiko.ServerInterface):
    def __init__(self, server):
        self.server = server
        self.dest = {}

    def get_allowed_auths(self, username):
        return 'publickey'

    def check_auth_publickey(self, username, key):
        ok = username == USER and key.get_base64() == self.server.allowed.get_base64()
        return paramiko.AUTH_SUCCESSFUL if ok else paramiko.AUTH_FAILED

    def check_auth_password(self, username, password):
        return paramiko.AUTH_FAILED

    def check_channel_request(self, kind, chanid):
        return paramiko.OPEN_SUCCEEDED if kind == 'session' else paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def check_channel_direct_tcpip_request(self, chanid, origin, destination):
        if destination[0] not in ('127.0.0.1', 'localhost'):
            return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED
        self.server.forwards.append(destination)
        self.dest[chanid] = destination
        return paramiko.OPEN_SUCCEEDED

    def check_channel_exec_request(self, channel, command):
        threading.Thread(target=self.server.run, args=(channel, command.decode()), daemon=True).start()
        return True


@pytest.fixture
def ssh(tmp_path):
    key = tmp_path / 'id_ed25519'
    pub = _ed25519(key)
    servers = []

    def make(**kw):
        s = SSHServer(pub, **kw)
        servers.append(s)
        node = {'name': 'ws1', 'kind': 'ssh', 'host': '127.0.0.1', 'ssh_port': s.port, 'ssh_user': USER,
                'ssh_key': str(key), 'known_hosts': str(tmp_path / 'known_hosts'), 'slots': [0]}
        return s, node
    yield make
    for s in servers:
        s.close()


def test_commands_files_and_a_tunnel(ssh, tmp_path):
    srv, node = ssh()
    c = T.Connection(node)
    try:
        assert c.exec('echo hello; echo oops >&2; exit 3') == (3, b'hello\n', b'oops\n')
        c.put(b'bundle bytes', str(tmp_path / 'remote' / 'b.bin'))
        assert (tmp_path / 'remote' / 'b.bin').read_bytes() == b'bundle bytes'
        lines = []
        assert c.stream('printf "a\\nb\\n"; exit 0', lines.append) == 0 and lines == ['a', 'b']
        echo = socket.socket()
        echo.bind(('127.0.0.1', 0))
        echo.listen(1)

        def serve():
            s, _ = echo.accept()
            s.sendall(s.recv(100).upper())
            s.close()
        threading.Thread(target=serve, daemon=True).start()
        local = c.tunnel(echo.getsockname()[1])
        s = socket.create_connection(('127.0.0.1', local), timeout=5)
        s.sendall(b'through the tunnel')
        assert s.recv(100) == b'THROUGH THE TUNNEL'
        assert srv.forwards == [('127.0.0.1', echo.getsockname()[1])]
    finally:
        c.close()
    kh = open(node['known_hosts']).read()
    assert f'[127.0.0.1]:{srv.port}' in kh                 # remembered on first contact (accept-new)


def test_keys_only_and_a_changed_host_key_refused(ssh, tmp_path):
    srv, node = ssh()
    T.Connection(node).exec('true')
    wrong = tmp_path / 'other_key'
    _ed25519(wrong)
    with pytest.raises(T.TransportError, match='could not log in with the key'):
        T.Connection(dict(node, ssh_key=str(wrong))).exec('true')
    with pytest.raises(T.TransportError, match='key file is not set or does not exist'):
        T.Connection(dict(node, ssh_key=str(tmp_path / 'none'))).exec('true')
    kh = node['known_hosts']                                # another key on record for this address: refused
    other = paramiko.RSAKey.generate(2048)
    with open(kh, 'w') as fh:
        fh.write(f'[127.0.0.1]:{srv.port} ssh-rsa {other.get_base64()}\n')
    n = len(srv.commands)
    with pytest.raises(T.TransportError, match='host key changed'):
        T.Connection(node).exec('true')
    assert len(srv.commands) == n                           # nothing ran
    with pytest.raises(T.TransportError) as e:
        T.Connection(dict(node, ssh_port=1)).exec('true')
    assert '127.0.0.1' not in str(e.value) and USER not in str(e.value)    # neither address nor user in a message


def test_an_ssh_node_runs_a_drive_on_its_daemon(ssh, tmp_path):
    from falcon_autonomy.cluster.cli import test_node
    from falcon_autonomy.cluster.launcher import RemoteLauncher, close_all
    from falcon_autonomy.core.drive import drive_config
    from falcon_autonomy.core.jobs import DONE, JobQueue, SubprocessLauncher
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.core.settings import Settings
    from falcon_autonomy.sim.service import CarlaService
    d = Daemons(tmp_path / 'nodes')
    try:
        dn = d.start('ws1')
        home = os.path.join(str(tmp_path / 'nodes'), 'ws1')
        srv, node = ssh(extra_env={'XDG_CONFIG_HOME': os.path.join(home, 'config')})
        node['slots'] = dn['slots']                         # the daemon's own port, reached through the tunnel
        r = test_node(node)
        assert r['ok'] and r['slots'][0]['ok'], r
        assert any('daemon-0.token' in cmd for cmd in srv.commands)
        st = Settings(str(tmp_path / 's.json'))
        st.set('cluster', {'nodes': [node], 'pools': {}})
        p = Project.new('T')
        p.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
        p.save(str(tmp_path / 'p.falcon.json'))
        q = JobQueue(SubprocessLauncher())
        svc = CarlaService(st, q, p.workspace, fake_carla=FAKE)
        q.remote = RemoteLauncher.for_settings(st)
        svc.placement = 'cluster'
        try:
            cfg = drive_config(p, 'world-town10', 'rig-default', str(tmp_path / 'run'), 2310, 8310, seconds=2.0)
            job = svc.submit_drive(cfg, title='over ssh')
            wait_for(lambda: job.state in ('done', 'failed', 'cancelled'), 120, q.poll)
            assert job.state == DONE and job.node == 'ws1/0', job.log
            run = json.load(open(tmp_path / 'run' / 'run.json'))
            assert run['status'] == 'ok' and run['node']['kind'] == 'ssh'
            assert ('127.0.0.1', dn['slots'][0]['port']) in srv.forwards
        finally:
            svc.shutdown()
            close_all()
    finally:
        d.stop()
