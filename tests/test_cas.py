"""The content-addressed store and the Merkle tree (core.cas, core.lineage): RFC 6962's published test vectors (roots
and audit paths), the SHA-256 path when blake3 is missing (it is not installed on the reference machine) and BLAKE3
when it is, putting a file by reflink → hard link → copy (ext4 has no reflink: a hard link), the 0444 guard,
deduplication, materialising, mark-and-sweep garbage collection that never collects a referenced object, fsck."""
import errno
import os
import shutil
import stat
import time

import pytest

from falcon_autonomy.core import cas as C
from falcon_autonomy.core import lineage as L

# RFC 6962's reference inputs (certificate-transparency's merkle tree tests) and the roots of their first n leaves
LEAVES = [b'', b'\x00', b'\x10', b'\x20\x21', b'\x30\x31', b'\x40\x41\x42\x43', b'\x50\x51\x52\x53\x54\x55\x56\x57',
          bytes(range(0x60, 0x70))]
ROOTS = ['6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d',
         'fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125',
         'aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77',
         'd37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7',
         '4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4',
         '76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef',
         'ddb89be403809e325750d3d263cd78929c2942b7942a34b77e122c9594a74c8c',
         '5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328']
PATHS = {(0, 8): ['96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7',
                  '5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e',
                  '6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4'],
         (2, 3): ['fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125'],
         (1, 5): ['6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d',
                  '5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e',
                  'bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b']}


@pytest.fixture
def no_blake3(monkeypatch):
    """As on the reference machine: `import blake3` fails."""
    import sys
    monkeypatch.setitem(sys.modules, 'blake3', None)
    assert not C.blake3_available()


# ------------------------------------------------------------------------------------------------ Merkle (RFC 6962)
def test_rfc6962_roots_and_the_empty_tree():
    for n in range(1, 9):
        assert L.merkle_root(LEAVES[:n], 's256').hex() == ROOTS[n - 1]
    assert L.merkle_root([], 's256').hex() == 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'


def test_rfc6962_audit_paths_and_their_check():
    for (m, n), want in PATHS.items():
        assert [h.hex() for h in L.audit_path(LEAVES[:n], m, 's256')] == want
    for n in range(1, 9):
        root = L.merkle_root(LEAVES[:n], 's256')
        for m in range(n):
            p = L.audit_path(LEAVES[:n], m, 's256')
            assert L.verify_path(LEAVES[m], m, n, p, root, 's256')
            assert not L.verify_path(LEAVES[m] + b'x', m, n, p, root, 's256')       # another leaf
            if n > 1:
                assert not L.verify_path(LEAVES[m], (m + 1) % n, n, p, root, 's256')  # another place
    big = [bytes([k % 251, k // 251]) for k in range(1000)]                     # odd nodes promoted at many levels
    root = L.merkle_root(big, 's256')
    for m in (0, 511, 512, 999):
        assert L.verify_path(big[m], m, 1000, L.audit_path(big, m, 's256'), root, 's256')


def test_dataset_root_is_independent_of_order_and_counts_duplicates():
    a = [('s256:' + '1' * 64, 's256:' + '2' * 64), ('s256:' + '3' * 64, None)]
    assert L.pairs_root(a, 's256') == L.pairs_root(a[::-1], 's256')
    assert L.pairs_root(a + a[:1], 's256') != L.pairs_root(a, 's256')
    assert L.leaf_bytes(None, None) == b'-|-'


# ------------------------------------------------------------------------------------------------ hashing
def test_sha256_when_blake3_is_missing(no_blake3, tmp_path):
    assert C.default_alg() == 's256'
    p = tmp_path / 'f'
    p.write_bytes(b'abc')
    a = C.hash_file(str(p))
    assert a == 's256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad' == C.hash_bytes(b'abc')
    assert C.parse(a) == ('s256', a[5:]) and C.is_address(a) and not C.is_address('md5:abc')
    with pytest.raises(C.CASError, match='blake3 package is not installed'):
        C.hash_bytes(b'abc', 'b3')
    with pytest.raises(C.CASError, match='blake3 package'):
        C.Store(str(tmp_path / 's'), 'b3')
    with pytest.raises(C.CASError, match='not a content address'):
        C.parse('s256:xyz')
    s = C.Store(str(tmp_path / 's'))
    assert s.alg == 's256' and s.put(str(p)) == a
    assert L.Hasher().alg == 's256'


def test_blake3_when_installed(tmp_path):
    pytest.importorskip('blake3')
    assert C.default_alg() == 'b3'
    assert C.hash_bytes(b'', 'b3') == 'b3:af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262'
    root = L.merkle_root(LEAVES, 'b3')
    assert root.hex() != ROOTS[-1] and L.verify_path(LEAVES[5], 5, 8, L.audit_path(LEAVES, 5, 'b3'), root, 'b3')
    p = tmp_path / 'f'
    p.write_bytes(b'abc')
    assert C.Store(str(tmp_path / 's')).put(str(p)).startswith('b3:')


# ------------------------------------------------------------------------------------------------ putting
def _file(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_put_on_ext4_is_a_hard_link_and_the_object_is_read_only(no_blake3, tmp_path):
    s = C.Store(str(tmp_path / 'cas'))
    src = _file(tmp_path, 'frame.jpg', b'\xff\xd8 a frame')
    d = s.put_detail(src)
    try:                                       # what this file system does with FICLONE
        C.reflink(src, str(tmp_path / 'probe'))
        reflinks = True
    except OSError as e:
        reflinks = False
        assert e.errno in (errno.EOPNOTSUPP, errno.EINVAL, errno.EXDEV, errno.ENOTTY)
    assert d['method'] == ('reflink' if reflinks else 'hardlink')
    obj = s.object_path(d['address'])
    assert obj == os.path.join(s.root, 'objects', d['address'][5:7], d['address'][5:])
    assert stat.S_IMODE(os.stat(obj).st_mode) == 0o444
    if not reflinks:
        assert os.path.samefile(obj, src)                        # no data copied …
        assert stat.S_IMODE(os.stat(src).st_mode) == 0o444       # … so the source is guarded with it
    if os.geteuid() != 0:
        with pytest.raises(PermissionError):
            open(obj, 'r+b')
    assert s.verify(d['address']) and s.read_bytes(d['address']) == b'\xff\xd8 a frame'
    assert os.listdir(os.path.join(s.root, 'tmp')) == []


def test_fallbacks_reflink_then_hardlink_then_copy(no_blake3, tmp_path, monkeypatch):
    calls = []

    def fake_reflink(src, dst):                # a file system that clones (btrfs, XFS)
        calls.append('reflink')
        shutil.copyfile(src, dst)
    monkeypatch.setattr(C, 'reflink', fake_reflink)
    s = C.Store(str(tmp_path / 'cas'))
    src = _file(tmp_path, 'a', b'one')
    assert s.put_detail(src)['method'] == 'reflink' and calls == ['reflink']
    assert os.access(src, os.W_OK) and not os.path.samefile(src, s.object_path(C.hash_bytes(b'one')))

    def no_reflink(src, dst):
        raise OSError(errno.EOPNOTSUPP, 'Operation not supported')

    def no_link(src, dst):                     # another file system: hard links refused
        raise OSError(errno.EXDEV, 'Invalid cross-device link')
    monkeypatch.setattr(C, 'reflink', no_reflink)
    assert s.put_detail(_file(tmp_path, 'b', b'two'))['method'] == 'hardlink'
    monkeypatch.setattr(os, 'link', no_link)
    src3 = _file(tmp_path, 'c', b'three')
    d = s.put_detail(src3)
    assert d['method'] == 'copy' and os.access(src3, os.W_OK)
    assert stat.S_IMODE(os.stat(d['path']).st_mode) == 0o444
    assert s.put_detail(_file(tmp_path, 'c2', b'three'))['method'] == 'present'      # the same bytes: stored once
    assert s.methods == {'reflink': 1, 'hardlink': 1, 'copy': 1, 'present': 1} and s.stats()['objects'] == 3
    assert s.put_bytes(b'three') == d['address'] and s.stats()['objects'] == 3
    with pytest.raises(C.CASError, match='how is'):
        C.link_or_copy(src3, str(tmp_path / 'z'), 'teleport')


def test_materialise_and_refusals(no_blake3, tmp_path):
    s = C.Store(str(tmp_path / 'cas'))
    a = s.put_bytes(b'labels')
    out = tmp_path / 'out'
    assert s.materialise(a, str(out / 'h' / 'x.json')) == 'hardlink'
    assert os.path.samefile(out / 'h' / 'x.json', s.object_path(a))
    assert s.materialise(a, str(out / 'h' / 'x.json')) == 'present'
    assert s.materialise(a, str(out / 'c.json'), 'copy') == 'copy' and os.access(out / 'c.json', os.W_OK)
    assert s.materialise(a, str(out / 's.json'), 'symlink') == 'symlink' and os.path.islink(out / 's.json')
    (out / 'other.json').write_bytes(b'not these bytes')
    with pytest.raises(C.CASError, match='other bytes'):
        s.materialise(a, str(out / 'other.json'))
    with pytest.raises(C.CASError, match='not in the store'):
        s.materialise(C.hash_bytes(b'never stored'), str(out / 'n'))


# ------------------------------------------------------------------------------------------------ gc and fsck
def test_gc_never_collects_a_referenced_object(no_blake3, tmp_path):
    s = C.Store(str(tmp_path / 'cas'))
    keep = [s.put_bytes(f'keep {i}'.encode()) for i in range(3)]
    drop = [s.put_bytes(f'drop {i}'.encode()) for i in range(4)]
    later = time.time() + 2 * C.DEFAULT_GRACE_S
    young = s.gc(keep, now=time.time())                     # everything was just linked: the grace period spares it
    assert young['removed'] == [] and young['young'] == 4 and young['kept'] == 3
    dry = s.gc(keep, dry_run=True, now=later)
    assert sorted(dry['removed']) == sorted(C.parse(a)[1] for a in drop) and dry['dry_run']
    assert all(s.has(a) for a in keep + drop)               # a dry run removes nothing
    with pytest.raises(C.CASError):                         # a malformed root aborts before anything goes
        s.gc(keep + ['garbage'], now=later)
    assert all(s.has(a) for a in drop)
    res = s.gc(keep, now=later)
    assert len(res['removed']) == 4 and res['bytes_freed'] == sum(len(f'drop {i}') for i in range(4))
    assert all(s.has(a) and s.verify(a) for a in keep) and not any(s.has(a) for a in drop)
    assert s.gc([], now=later, grace_s=0)['removed'] and s.stats()['objects'] == 0


def test_fsck_finds_a_changed_object_and_restores_the_guard(no_blake3, tmp_path):
    s = C.Store(str(tmp_path / 'cas'))
    a, b = s.put_bytes(b'good'), s.put_bytes(b'will change')
    p = s.object_path(b)
    os.chmod(p, 0o644)
    with open(p, 'wb') as fh:
        fh.write(b'changed')
    os.chmod(s.object_path(a), 0o600)
    res = s.fsck()
    assert res['objects'] == 2 and res['ok'] == 1 and res['bad'] == [C.parse(b)[1]] and res['chmod'] == 2
    assert stat.S_IMODE(os.stat(s.object_path(a)).st_mode) == 0o444 and not s.verify(b) and s.verify(a)
