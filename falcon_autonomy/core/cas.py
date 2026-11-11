"""A content-addressed store: files kept once under the hash of their bytes (standard library; blake3 when it is
installed, the extra `lineage`).

Addresses. An address is the hash algorithm's prefix, a colon and the digest in lower-case hex: "b3:<64 hex>" for
BLAKE3 (the `blake3` package) or "s256:<64 hex>" for SHA-256 (hashlib). The algorithm is BLAKE3 when `blake3` imports,
else SHA-256 (default_alg()); an address always says which, so a store, an index or a Merkle root made with one is
checked with the same one (a BLAKE3 address cannot be checked without the package: CASError says so).

Layout. <root>/objects/<first 2 hex>/<64 hex> (mode 0444), <root>/tmp/ (files being put), <root>/lock (flock: puts
hold it shared, gc exclusive). The default root is <workspace>/cas (store_dir()).

Putting a file (Store.put). The file is first brought into tmp/ the cheapest way the file system allows:
1. a reflink — the FICLONE ioctl (Linux; btrfs, XFS, bcachefs: a copy-on-write clone, no data copied, and the
   original stays writable and independent);
2. else a hard link (the same inode: no data copied, but it IS the source file — so it becomes read-only (0444)
   with the object, the "0444 guard": an object must never change under its address, and a recorded frame is never
   edited in place by Falcon; os.replace()-style rewrites make a new inode and leave the object alone);
3. else a copy (another file system: EXDEV; or `how='copy'`).
The copy in tmp/ is made 0444, hashed (so the address is that of the bytes stored, not of the source read earlier),
and renamed to its object path; when the object already exists the tmp copy is dropped. On ext4 FICLONE fails with
EOPNOTSUPP (measured on this machine's ext4): a put is a hard link.

Garbage collection (Store.gc). Mark and sweep: the caller passes every address still referenced (the roots — a
lineage index's frames.parquet, a dataset version's splits/<v>.parquet, a training's data.json: core.lineage.roots());
objects not among them are removed, except objects whose inode changed (ctime) within `grace_s` (a put whose
reference is not yet written: a link or a chmod updates ctime). It holds the lock exclusively, so no put runs during
it. An address that cannot be parsed in the roots aborts the sweep (CASError): a bad root never means "collect
everything". dry_run lists what would go.

Verification. Store.verify(address) re-hashes an object; fsck() every object (and restores 0444 on one that lost it).
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import re
import shutil
import stat
import time
import uuid

FICLONE = 0x40049409                     # _IOW(0x94, 9, int): linux/fs.h
MODE = 0o444
CHUNK = 1 << 20
PREFIXES = {'b3': 'blake3', 's256': 'sha256'}
_ADDR = re.compile(r'^(b3|s256):([0-9a-f]{64})$')
DEFAULT_GRACE_S = 3600.0


class CASError(Exception):
    pass


# ------------------------------------------------------------------------------------------------ hashing
def blake3_available() -> bool:
    try:
        import blake3  # noqa: F401
        return True
    except ImportError:
        return False


def default_alg() -> str:
    """'b3' when the blake3 package imports, else 's256'."""
    return 'b3' if blake3_available() else 's256'


def new_hasher(alg: str | None = None):
    alg = alg or default_alg()
    if alg == 's256':
        return hashlib.sha256()
    if alg == 'b3':
        try:
            import blake3
        except ImportError:
            raise CASError('this address is BLAKE3 (b3:…) and the blake3 package is not installed '
                           '(pip install blake3, or the extra falcon-autonomy[lineage])') from None
        return blake3.blake3()
    raise CASError(f'unknown hash algorithm {alg!r} (b3 or s256)')


def digest(data: bytes, alg: str | None = None) -> bytes:
    h = new_hasher(alg)
    h.update(data)
    return h.digest()


def hash_bytes(data: bytes, alg: str | None = None) -> str:
    alg = alg or default_alg()
    h = new_hasher(alg)
    h.update(data)
    return f'{alg}:{h.hexdigest()}'


def hash_file(path: str, alg: str | None = None) -> str:
    alg = alg or default_alg()
    h = new_hasher(alg)
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
    return f'{alg}:{h.hexdigest()}'


def parse(address: str) -> tuple:
    """(alg, hex) of an address; CASError when it is not one."""
    m = _ADDR.match(str(address or ''))
    if not m:
        raise CASError(f'not a content address: {address!r} (b3:<64 hex> or s256:<64 hex>)')
    return m.group(1), m.group(2)


def is_address(s) -> bool:
    return bool(_ADDR.match(str(s or '')))


# ------------------------------------------------------------------------------------------------ linking
def reflink(src: str, dst: str):
    """A copy-on-write clone (FICLONE); OSError when the file system cannot (EOPNOTSUPP, EXDEV, EINVAL …)."""
    with open(src, 'rb') as s:
        fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            fcntl.ioctl(fd, FICLONE, s.fileno())
        except OSError:
            os.close(fd)
            os.unlink(dst)
            raise
        os.close(fd)


def link_or_copy(src: str, dst: str, how: str = 'auto') -> str:
    """Bring src to dst (which must not exist): 'reflink', 'hardlink' or 'copy' as allowed by `how` ('auto' tries
    them in that order; 'hardlink' / 'reflink' / 'copy' only that one, falling back to a copy for the first two —
    except that how='reflink-or-copy' never hard-links). Returns the method used."""
    order = {'auto': ('reflink', 'hardlink', 'copy'), 'reflink': ('reflink', 'copy'),
             'reflink-or-copy': ('reflink', 'copy'), 'hardlink': ('hardlink', 'copy'), 'copy': ('copy',)}.get(how)
    if order is None:
        raise CASError(f'how is auto, reflink, reflink-or-copy, hardlink or copy (not {how!r})')
    for m in order:
        try:
            if m == 'reflink':
                reflink(src, dst)
            elif m == 'hardlink':
                os.link(src, dst)
            else:
                shutil.copyfile(src, dst)
            return m
        except OSError as e:
            if m == 'copy' or e.errno in (errno.ENOENT, errno.EEXIST):
                raise
            continue
    raise CASError(f'could not bring {src} to {dst}')      # not reached: copy raises


# ------------------------------------------------------------------------------------------------ the store
def store_dir(workspace: str) -> str:
    return os.path.join(workspace, 'cas')


class Store:
    """A content-addressed store at root (module doc)."""

    def __init__(self, root: str, alg: str | None = None):
        self.root = os.path.abspath(root)
        self.alg = alg or default_alg()
        new_hasher(self.alg)                       # refuse an algorithm that cannot run here, now
        self.methods = {'reflink': 0, 'hardlink': 0, 'copy': 0, 'present': 0}

    # -- paths
    def object_path(self, address: str) -> str:
        _alg, hx = parse(address)
        return os.path.join(self.root, 'objects', hx[:2], hx)

    def has(self, address: str) -> bool:
        return os.path.isfile(self.object_path(address))

    def _lock(self, exclusive: bool):
        os.makedirs(self.root, exist_ok=True)
        fd = os.open(os.path.join(self.root, 'lock'), os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        return fd

    @staticmethod
    def _unlock(fd):
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    # -- putting
    def put(self, path: str, how: str = 'auto') -> str:
        """Store a file's bytes; returns its address (module doc: reflink → hard link → copy, 0444, hashed as
        stored). An object already present is not touched (the duplicate brought in is dropped)."""
        return self.put_detail(path, how)['address']

    def put_detail(self, path: str, how: str = 'auto') -> dict:
        if not os.path.isfile(path):
            raise CASError(f'{path}: not a file')
        tmpd = os.path.join(self.root, 'tmp')
        os.makedirs(tmpd, exist_ok=True)
        fd = self._lock(False)
        try:
            tmp = os.path.join(tmpd, uuid.uuid4().hex)
            method = link_or_copy(path, tmp, how)
            try:
                os.chmod(tmp, MODE)
                address = hash_file(tmp, self.alg)
                dst = self.object_path(address)
                if os.path.isfile(dst):
                    os.unlink(tmp)
                    self.methods['present'] += 1
                    return {'address': address, 'method': 'present', 'path': dst}
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.replace(tmp, dst)
            except BaseException:
                if os.path.lexists(tmp):
                    os.unlink(tmp)
                raise
            self.methods[method] += 1
            return {'address': address, 'method': method, 'path': dst}
        finally:
            self._unlock(fd)

    def put_bytes(self, data: bytes) -> str:
        address = hash_bytes(data, self.alg)
        dst = self.object_path(address)
        if os.path.isfile(dst):
            return address
        tmpd = os.path.join(self.root, 'tmp')
        os.makedirs(tmpd, exist_ok=True)
        fd = self._lock(False)
        try:
            tmp = os.path.join(tmpd, uuid.uuid4().hex)
            with open(tmp, 'wb') as fh:
                fh.write(data)
            os.chmod(tmp, MODE)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(tmp, dst)
        finally:
            self._unlock(fd)
        return address

    def read_bytes(self, address: str) -> bytes:
        with open(self.object_path(address), 'rb') as fh:
            return fh.read()

    # -- materialising
    def materialise(self, address: str, dest: str, how: str = 'hardlink') -> str:
        """The object at dest: 'hardlink' (the store's own read-only inode; falls back to a copy across file
        systems), 'reflink' (a clone; a copy where the file system cannot), 'copy' (writable) or 'symlink'. A dest
        that already holds these bytes is left; one that holds others is refused. Returns the method used."""
        src = self.object_path(address)
        if not os.path.isfile(src):
            raise CASError(f'{address}: not in the store {self.root}')
        if os.path.lexists(dest):
            if os.path.isfile(dest) and hash_file(dest, parse(address)[0]) == address:
                return 'present'
            raise CASError(f'{dest} exists and holds other bytes than {address}')
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        if how == 'symlink':
            os.symlink(src, dest)
            return 'symlink'
        if how == 'reflink':
            how = 'reflink-or-copy'
        m = link_or_copy(src, dest, how)
        if m == 'copy' and how == 'copy':
            os.chmod(dest, 0o644)
        return m

    # -- checking
    def verify(self, address: str) -> bool:
        """Whether the object's bytes still hash to its address."""
        p = self.object_path(address)
        return os.path.isfile(p) and hash_file(p, parse(address)[0]) == address

    def objects(self):
        """Every object's address — the algorithm read back from the digest's length is ambiguous (both are 64
        hex), so objects are listed as their hex; addresses() pairs them with the store's algorithm."""
        base = os.path.join(self.root, 'objects')
        if not os.path.isdir(base):
            return
        for d in sorted(os.listdir(base)):
            sub = os.path.join(base, d)
            if not os.path.isdir(sub):
                continue
            for f in sorted(os.listdir(sub)):
                if re.fullmatch(r'[0-9a-f]{64}', f) and f[:2] == d:
                    yield f

    def fsck(self, algs=None) -> dict:
        """Re-hash every object with each algorithm in algs (default: the store's, then the other when available)
        until one matches its name; restore 0444 where lost. {'objects', 'ok', 'bad': [hex], 'chmod': n}."""
        algs = list(algs or [self.alg] + [a for a in PREFIXES if a != self.alg and
                                         (a != 'b3' or blake3_available())])
        out = {'objects': 0, 'ok': 0, 'bad': [], 'chmod': 0}
        for hx in self.objects():
            out['objects'] += 1
            p = os.path.join(self.root, 'objects', hx[:2], hx)
            if stat.S_IMODE(os.stat(p).st_mode) != MODE:
                os.chmod(p, MODE)
                out['chmod'] += 1
            if any(hash_file(p, a) == f'{a}:{hx}' for a in algs):
                out['ok'] += 1
            else:
                out['bad'].append(hx)
        return out

    def stats(self) -> dict:
        n = size = 0
        for hx in self.objects():
            n += 1
            size += os.path.getsize(os.path.join(self.root, 'objects', hx[:2], hx))
        return {'root': self.root, 'alg': self.alg, 'objects': n, 'bytes': size}

    # -- garbage collection
    def gc(self, referenced, dry_run: bool = False, grace_s: float = DEFAULT_GRACE_S, now: float | None = None
           ) -> dict:
        """Mark and sweep (module doc). referenced: every address still in use. {'kept', 'removed': [hex],
        'young': n (spared by the grace period), 'bytes_freed', 'dry_run', 'tmp_removed'}."""
        keep = set()
        for a in referenced:
            keep.add(parse(a)[1])                   # a malformed root aborts before anything is removed
        now = time.time() if now is None else now
        fd = self._lock(True)
        try:
            out = {'kept': 0, 'removed': [], 'young': 0, 'bytes_freed': 0, 'dry_run': bool(dry_run),
                   'tmp_removed': 0}
            for hx in list(self.objects()):
                if hx in keep:
                    out['kept'] += 1
                    continue
                p = os.path.join(self.root, 'objects', hx[:2], hx)
                st = os.stat(p)
                if now - st.st_ctime < grace_s:
                    out['young'] += 1
                    continue
                out['removed'].append(hx)
                out['bytes_freed'] += st.st_size
                if not dry_run:
                    os.unlink(p)
            tmpd = os.path.join(self.root, 'tmp')
            if os.path.isdir(tmpd):
                for f in os.listdir(tmpd):
                    p = os.path.join(tmpd, f)
                    try:
                        if now - os.lstat(p).st_ctime >= grace_s:
                            out['tmp_removed'] += 1
                            if not dry_run:
                                os.unlink(p)
                    except FileNotFoundError:
                        continue
            if not dry_run:                           # empty fan-out folders
                base = os.path.join(self.root, 'objects')
                for d in os.listdir(base) if os.path.isdir(base) else ():
                    try:
                        os.rmdir(os.path.join(base, d))
                    except OSError:
                        pass
            return out
        finally:
            self._unlock(fd)
