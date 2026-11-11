"""The worker daemon's wire format (standard library; docs/cluster.md, "Protocol").

A connection is a TCP stream to a daemon slot on 127.0.0.1 of its node (reached through an SSH tunnel from another
machine; the daemon never listens on another address). It carries frames: a 4-byte big-endian length, that many bytes
of a JSON object (the message, with a type "t"), then the message's "size" bytes of payload when it has one.

client -> daemon
    hello     {"protocol": 1, "token": "...", "app_version": "...", "codecs": ["zstd", "zlib"]}   first, always
    submit    {"rid": "<gid>.<attempt>", "bundle_size": n}  + the bundle (bundle.encode)
    cancel    {"rid", "force": false}            polite (the worker tidies up), force: killed
    attach    {"rids": [...]}                    after a reconnect: what became of these
    release   {"rid"}                            the artifact arrived: the daemon deletes its scratch folder
    gc        {"older_than_s", "dry_run", "keep": [gid, …], "only": [name, …] | null}   collect the slot's scratch
                                                 (cluster.gc; never a job running there or one the client keeps)
    ping      {}
daemon -> client
    hello     {"protocol": 1, "node": "<host name>", "slot": n, "app_version", "codecs": [...], "carla": "0.9.16",
               "backends": [id, …], "capabilities": {id: {"caps", "maps", "probed", "measured", "title"}},
               "memory_gb": .., "jobs": n}       the backends this node has (sim.backends.advertised)
    accepted  {"rid"}
    refused   {"rid", "why"}                     never started (not on the allow-list, a bundle it cannot read, …)
    msg       {"rid", "msg": {…}}                one of the worker's protocol lines (core.protocol), as it came
    done      {"rid", "code", "codec", "size", "sha256", "scratch"}  + the artifact: the run folder, packed
    heartbeat {"time", "running": [rid, …], "queued": n}     every HEARTBEAT_S seconds
    state     {"rid", "state": "running" | "queued" | "unknown"}  the answer to attach for a job not finished
    gc        {"items": [{"name", "what", "bytes", "age_h"}], "kept": [{"name", "why"}], "removed", "bytes_freed"}
               (or {"error"})                    the answer to gc: what was (or, dry, would be) removed
    error     {"why"}                            and the daemon closes the connection (a bad hello, a bad frame)

An artifact is a tar of the run folder (regular files and folders only), compressed with zstd when both ends have it
(Python 3.14's compression.zstd) else zlib. Unpacking refuses absolute names, "..", links and devices, and stops at a
size cap.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import socket
import struct
import tarfile
import threading

PROTOCOL = 1
HEADER_MAX = 1 << 20                 # a message (its JSON) is at most 1 MiB
PAYLOAD_MAX = 4 << 30                # a payload (a bundle or an artifact) at most 4 GiB
ARTIFACT_MAX = 8 << 30               # an artifact unpacks to at most 8 GiB
HEARTBEAT_S = 15.0
PORT_BASE = 2350                     # slot N listens on 127.0.0.1:2350+N (Falcon's CARLA uses 2310-2312)


class ProtocolError(Exception):
    pass


def codecs() -> list:
    """The compressions this Python has, best first."""
    try:
        import compression.zstd  # noqa: F401 (Python 3.14+)
        return ['zstd', 'zlib']
    except ImportError:
        return ['zlib']


def pick_codec(theirs) -> str:
    mine = codecs()
    return next((c for c in mine if c in (theirs or ())), 'zlib')


def compress(data: bytes, codec: str) -> bytes:
    if codec == 'zstd':
        from compression import zstd
        return zstd.compress(data, level=6)
    if codec == 'zlib':
        import zlib
        return zlib.compress(data, 6)
    raise ProtocolError(f'unknown codec {codec!r}')


def decompress(data: bytes, codec: str, max_size: int = ARTIFACT_MAX) -> bytes:
    if codec == 'zstd':
        try:
            from compression import zstd
        except ImportError as e:
            raise ProtocolError('zstd payload but this Python has no compression.zstd') from e
        d = zstd.ZstdDecompressor()
        out = d.decompress(data, max_length=max_size + 1)
    elif codec == 'zlib':
        import zlib
        d = zlib.decompressobj()
        out = d.decompress(data, max_size + 1)
    else:
        raise ProtocolError(f'unknown codec {codec!r}')
    if len(out) > max_size:
        raise ProtocolError(f'the payload unpacks to more than {max_size >> 20} MiB')
    return out


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------------------------------ frames
class Channel:
    """Frames over a connected socket; send() is thread-safe, recv() is for one reader thread."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._wlock = threading.Lock()
        self.closed = False

    def send(self, msg: dict, payload: bytes = b''):
        if payload:
            msg = dict(msg, size=len(payload))
        head = json.dumps(msg, separators=(',', ':')).encode()
        if len(head) > HEADER_MAX:
            raise ProtocolError('message too large')
        with self._wlock:
            try:
                self.sock.sendall(struct.pack('>I', len(head)) + head)
                if payload:
                    self.sock.sendall(payload)
            except OSError as e:
                self.closed = True
                raise ProtocolError(f'connection lost: {e}') from e

    def _exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = self.sock.recv(min(n - len(buf), 1 << 20))
            except OSError as e:
                self.closed = True
                raise ProtocolError(f'connection lost: {e}') from e
            if not chunk:
                self.closed = True
                raise ProtocolError('connection closed')
            buf += chunk
        return bytes(buf)

    def recv(self) -> tuple:
        """(message, payload). ProtocolError when the connection ends or a frame is malformed."""
        (n,) = struct.unpack('>I', self._exact(4))
        if n > HEADER_MAX:
            raise ProtocolError('message too large')
        try:
            msg = json.loads(self._exact(n))
        except ValueError as e:
            raise ProtocolError('a frame that is not JSON') from e
        if not isinstance(msg, dict) or not isinstance(msg.get('t'), str):
            raise ProtocolError('a frame without a type')
        size = int(msg.get('size') or 0)
        if size < 0 or size > PAYLOAD_MAX:
            raise ProtocolError('payload too large')
        return msg, (self._exact(size) if size else b'')

    def close(self):
        self.closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


# ------------------------------------------------------------------------------------------------ artifacts
def pack_dir(root: str, skip=('drive.json',)) -> bytes:
    """A tar of the folder's regular files (names relative to it; skip: top-level names left out)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w', format=tarfile.PAX_FORMAT) as tf:
        for d, dirs, files in os.walk(root):
            dirs.sort()
            for f in sorted(files):
                p = os.path.join(d, f)
                rel = os.path.relpath(p, root)
                if rel in skip or os.path.islink(p) or not os.path.isfile(p):
                    continue
                ti = tf.gettarinfo(p, arcname=rel)
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = ''
                with open(p, 'rb') as fh:
                    tf.addfile(ti, fh)
    return buf.getvalue()


def pack_files(files: dict) -> bytes:
    """A tar of {relative name: path or bytes}."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w', format=tarfile.PAX_FORMAT) as tf:
        for rel, src in sorted(files.items()):
            data = src if isinstance(src, bytes) else open(src, 'rb').read()
            ti = tarfile.TarInfo(rel)
            ti.size = len(data)
            ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def safe_name(name: str) -> bool:
    if not name or name.startswith(('/', '\\')) or '\x00' in name:
        return False
    parts = name.replace('\\', '/').split('/')
    return '..' not in parts and not any(p == '' for p in parts[:-1])


def unpack_tar(data: bytes, dest: str, max_bytes: int = ARTIFACT_MAX) -> list:
    """Extract a tar (uncompressed bytes) into dest: regular files and folders only, names inside dest; the files'
    names. ProtocolError on anything else (absolute names, '..', links, devices) or past max_bytes."""
    names, total = [], 0
    os.makedirs(dest, exist_ok=True)
    root = os.path.realpath(dest)
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as tf:
        for ti in tf:
            if not safe_name(ti.name):
                raise ProtocolError(f'an artifact entry with an unsafe name: {ti.name!r}')
            target = os.path.realpath(os.path.join(root, ti.name))
            if target != root and not target.startswith(root + os.sep):
                raise ProtocolError(f'an artifact entry outside its folder: {ti.name!r}')
            if ti.isdir():
                os.makedirs(target, exist_ok=True)
                continue
            if not ti.isfile():
                raise ProtocolError(f'an artifact entry that is not a regular file: {ti.name!r}')
            total += ti.size
            if total > max_bytes:
                raise ProtocolError(f'the artifact is larger than {max_bytes >> 20} MiB')
            os.makedirs(os.path.dirname(target), exist_ok=True)
            src = tf.extractfile(ti)
            tmp = target + '.part'
            with open(tmp, 'wb') as fh:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            os.replace(tmp, target)
            names.append(ti.name)
    return names


TEXT_SUFFIXES = ('.json', '.jsonl', '.txt', '.csv', '.log', '.md', '.yaml', '.yml', '.xosc', '.xodr', '.html')


def translate_paths(root: str, names, old: str, new: str, limit: int = 256 << 20):
    """Replace the node's folder (old) with this machine's (new) in the text files among names: a run's records name
    their own folder. Binary files are left as they are."""
    if not old or old == new:
        return
    a, b = old.encode(), new.encode()
    for n in names:
        if not n.endswith(TEXT_SUFFIXES):
            continue
        p = os.path.join(root, n)
        try:
            if os.path.getsize(p) > limit:
                continue
            with open(p, 'rb') as fh:
                data = fh.read()
        except OSError:
            continue
        if a in data:
            tmp = p + '.part'
            with open(tmp, 'wb') as fh:
                fh.write(data.replace(a, b))
            os.replace(tmp, p)


# ------------------------------------------------------------------------------------------------ the daemon's token
def token_path(slot: int, config: str = '') -> str:
    from ..core.paths import config_dir
    return os.path.join(config or config_dir(), 'cluster', f'daemon-{int(slot)}.token')


def write_token(slot: int) -> str:
    """A new random token for the slot, in a file only the user can read (the client reads it over SSH)."""
    p = token_path(slot)
    os.makedirs(os.path.dirname(p), mode=0o700, exist_ok=True)
    tok = secrets.token_hex(24)
    fd = os.open(p + '.tmp', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='ascii') as fh:
        fh.write(tok)
    os.replace(p + '.tmp', p)
    return tok


def read_token(slot: int, config: str = '') -> str:
    try:
        with open(token_path(slot, config), encoding='ascii') as fh:
            return fh.read().strip()
    except OSError:
        return ''
