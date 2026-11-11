"""SSH to the cluster's nodes with the user's key (paramiko, imported only when a node needs it: the `cluster` extra;
docs/cluster.md, "SSH").

The rules are hil/deploy.py's: keys only (the node's ssh_key is a path to the user's private key; no password is
asked for, stored or tried, no agent, no other key), a new host key accepted on first contact and kept in Falcon's own
known_hosts file (cluster/known_hosts in the config folder), a changed one refused; messages name neither the address
nor the user.

    Connection(node)              the SSH session (connect() on first use)
      .exec(cmd, stdin, timeout)  (exit code, stdout bytes, stderr bytes)
      .put(data, path)            a file on the node (through `cat`, so any sshd will do)
      .tunnel(port)               a local port on 127.0.0.1 forwarded to 127.0.0.1:port on the node
      .close()
"""
from __future__ import annotations

import os
import shlex
import socket
import threading


class TransportError(OSError):
    pass


def _paramiko():
    try:
        import paramiko
    except ImportError as e:
        raise TransportError('SSH to a node needs paramiko (pip install "falcon-autonomy[cluster]")') from e
    return paramiko


def known_hosts_path(node: dict) -> str:
    from ..core.paths import config_dir
    return os.path.expanduser(str(node.get('known_hosts') or os.path.join(config_dir(), 'cluster', 'known_hosts')))


def clean(text: str, node: dict) -> str:
    """A message without the node's address or user."""
    text = str(text)
    for k in ('host', 'ssh_user'):
        if node.get(k):
            text = text.replace(str(node[k]), f'<{k}>')
    return text.strip()[-400:]


class Connection:
    def __init__(self, node: dict, timeout: float = 10.0):
        self.node, self.timeout = node, float(timeout)
        self.client = None
        self.tunnels: list = []
        self._lock = threading.Lock()

    def connect(self):
        with self._lock:
            if self.client is not None and self.client.get_transport() is not None and \
                    self.client.get_transport().is_active():
                return self.client
            pm = _paramiko()
            n = self.node
            key = os.path.expanduser(str(n.get('ssh_key') or ''))
            if not key or not os.path.isfile(key):
                raise TransportError('the node\'s SSH key file is not set or does not exist')
            kh = known_hosts_path(n)
            os.makedirs(os.path.dirname(kh), mode=0o700, exist_ok=True)
            c = pm.SSHClient()
            if os.path.isfile(kh):
                c.load_host_keys(kh)

            class AcceptNew(pm.MissingHostKeyPolicy):
                """accept-new: a host never seen is remembered; a changed key raises BadHostKeyException."""

                def missing_host_key(self, client, hostname, k):
                    client.get_host_keys().add(hostname, k.get_name(), k)
                    client.save_host_keys(kh)
            c.set_missing_host_key_policy(AcceptNew())
            try:
                c.connect(str(n['host']), port=int(n.get('ssh_port') or 22), username=str(n['ssh_user']),
                          key_filename=key, look_for_keys=False, allow_agent=False, timeout=self.timeout,
                          banner_timeout=self.timeout, auth_timeout=self.timeout)
            except pm.BadHostKeyException:
                raise TransportError('the node\'s host key changed since Falcon first saw it: refused (if the node '
                                     'was reinstalled, remove its line from Falcon\'s cluster/known_hosts)') from None
            except pm.AuthenticationException:
                raise TransportError('SSH could not log in with the key (is it installed on the node? ssh-copy-id)'
                                     ) from None
            except (pm.SSHException, OSError) as e:
                raise TransportError(f'cannot reach the node: {clean(e, n)}') from None
            tr = c.get_transport()
            if tr is not None:
                tr.set_keepalive(15)
            self.client = c
            return c

    def exec(self, cmd: str, stdin: bytes = b'', timeout: float = 60.0) -> tuple:
        c = self.connect()
        tr = c.get_transport()
        try:
            ch = tr.open_session(timeout=self.timeout)
            ch.settimeout(timeout)
            ch.exec_command(cmd)
            if stdin:
                ch.sendall(stdin)
            ch.shutdown_write()
            err = bytearray()

            def read_err():
                f = ch.makefile_stderr('rb')
                err.extend(f.read())
            te = threading.Thread(target=read_err, daemon=True)
            te.start()
            out = ch.makefile('rb').read()
            te.join(timeout)
            code = ch.recv_exit_status()
            ch.close()
            return code, bytes(out), bytes(err)
        except (socket.timeout, TimeoutError):
            raise TransportError(f'the node did not answer within {timeout:.0f} s') from None
        except Exception as e:                      # noqa: BLE001 (paramiko's own errors)
            if isinstance(e, TransportError):
                raise
            raise TransportError(f'SSH failed: {clean(e, self.node)}') from None

    def stream(self, cmd: str, on_line, timeout: float | None = None) -> int:
        """Run cmd, calling on_line(text) per stdout line as it comes; its exit code."""
        c = self.connect()
        ch = c.get_transport().open_session(timeout=self.timeout)
        ch.settimeout(timeout)
        ch.exec_command(cmd)
        ch.shutdown_write()
        buf = b''
        try:
            while True:
                data = ch.recv(1 << 16)
                if not data:
                    break
                buf += data
                *lines, buf = buf.split(b'\n')
                for ln in lines:
                    on_line(ln.decode('utf-8', 'replace'))
        except (socket.timeout, TimeoutError):
            ch.close()
            raise TransportError('the node stopped answering') from None
        if buf.strip():
            on_line(buf.decode('utf-8', 'replace'))
        code = ch.recv_exit_status()
        ch.close()
        return code

    def put(self, data: bytes, path: str, timeout: float = 120.0):
        d = os.path.dirname(path) or '.'
        code, _o, err = self.exec(f'mkdir -p {shlex.quote(d)} && cat > {shlex.quote(path)}', stdin=data,
                                  timeout=timeout)
        if code != 0:
            raise TransportError(f'writing a file on the node failed: {clean(err.decode("utf-8", "replace"), self.node)}')

    def tunnel(self, remote_port: int) -> int:
        """A local port (127.0.0.1) whose connections go to 127.0.0.1:remote_port on the node."""
        t = _Tunnel(self, int(remote_port))
        self.tunnels.append(t)
        return t.port

    def close(self):
        for t in self.tunnels:
            t.close()
        self.tunnels = []
        with self._lock:
            if self.client is not None:
                try:
                    self.client.close()
                except Exception:                   # noqa: BLE001
                    pass
                self.client = None


class _Tunnel:
    def __init__(self, conn: Connection, remote_port: int):
        self.conn, self.remote_port = conn, remote_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('127.0.0.1', 0))
        self.sock.listen(4)
        self.sock.settimeout(0.5)
        self.port = self.sock.getsockname()[1]
        self.closed = False
        threading.Thread(target=self._accept, daemon=True, name='falcon-ssh-tunnel').start()

    def _accept(self):
        while not self.closed:
            try:
                s, addr = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                tr = self.conn.connect().get_transport()
                ch = tr.open_channel('direct-tcpip', ('127.0.0.1', self.remote_port), addr, timeout=10)
            except Exception:                       # noqa: BLE001 (the client sees its connection close)
                s.close()
                continue
            threading.Thread(target=_pump, args=(s, ch), daemon=True).start()
            threading.Thread(target=_pump, args=(ch, s), daemon=True).start()

    def close(self):
        self.closed = True
        try:
            self.sock.close()
        except OSError:
            pass


def _pump(src, dst):
    try:
        while True:
            data = src.recv(1 << 16)
            if not data:
                break
            dst.sendall(data)
    except Exception:                               # noqa: BLE001 (either end went away)
        pass
    for x in (src, dst):
        try:
            x.close()
        except Exception:                           # noqa: BLE001
            pass


def remote_token_command(slot: int) -> str:
    """The shell command that prints slot N's token on the node (written by its daemon, readable only by the user)."""
    return f'cat "${{XDG_CONFIG_HOME:-$HOME/.config}}/FalconAutonomy/cluster/daemon-{int(slot)}.token"'
