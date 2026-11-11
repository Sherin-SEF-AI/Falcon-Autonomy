"""The pretrained perception models Falcon can run on dashcam footage, pinned, and the ones it refuses (standard library
at import; nothing here downloads unless fetch() is called by an explicit, licence-accepting command).

Candidates (CANDIDATES), each pinned to a Hugging Face repository revision (commit) with every file's size and the
publisher's checksum (an LFS file's SHA-256, a small file's git blob id) as the Hub's API states them
(https://huggingface.co/api/models/REPO?blobs=true), and its licence as the model card's metadata states it, with the
URL it was read from and the date (LICENCE_CHECKED). No weights were downloaded to read them.

    rtdetr-v2-r18   RT-DETRv2, ResNet-18-vd backbone, COCO (Lv et al. 2024, arXiv:2407.17140; transformers
                    RTDetrV2ForObjectDetection) — Apache-2.0 (the card; the code at github.com/lyuwenyu/RT-DETR is
                    Apache-2.0). The default detector: 20 M parameters.
    rtdetr-v2-r50   the same with ResNet-50-vd: 43 M parameters, more accurate, slower.
    da2-small       Depth Anything V2 Small (Yang et al. 2024, arXiv:2406.09414; transformers
                    DepthAnythingForDepthEstimation): relative depth (affine-invariant inverse depth) — Apache-2.0
                    (the card, and the project's README: "Depth-Anything-V2-Small model is under the Apache-2.0
                    license"). The default depth model: scaled to metres by the ground plane (core.dashcam_depth).
    metric3d-small  Metric3D v2, ViT-Small, as ONNX (onnx-community/metric3d-vit-small; Hu et al. 2024,
                    arXiv:2404.15506): metric depth. OPT-IN ONLY: the ONNX conversion's card says cc0-1.0 in its
                    metadata and nothing else; the upstream code (github.com/YvanYin/Metric3D) is BSD-2-Clause and
                    its README asks commercial users to contact the authors; the WEIGHTS' terms are not stated
                    anywhere Falcon could read. fetch() refuses it unless accept_unstated_terms is given
                    (`--accept-unstated-weights-terms`), and records that acknowledgement in pins.json.

Refused (REFUSED), with why: every YOLO from Ultralytics (AGPL-3.0: its LICENSE is the GNU Affero GPL — it would bind
the app; Falcon never imports `ultralytics`), Depth Anything V2 Base and Large (CC-BY-NC-4.0: non-commercial), and
anything built on SMPL / SMPL-X body models (their licences are non-commercial).

Downloads. fetch() downloads a candidate's files from https://huggingface.co/REPO/resolve/REVISION/FILE only when
called with accept_licence=True (the command shows describe() first and needs --accept-licence), checks each file's
size and the publisher's checksum (refusing and deleting on a mismatch), and writes pins.json with the SHA-256 it
computed: the first approved download pins the files. Loading (load_folder) checks pins.json, the sizes and the
publisher's checksums again, then the runners load with local_files_only and HF_HUB_OFFLINE set: they never reach
the network.
"""
from __future__ import annotations

import hashlib
import json
import os

HF = 'https://huggingface.co'
LICENCE_CHECKED = '2026-11-12'
PINS = 'pins.json'
PERMISSIVE = ('Apache-2.0', 'MIT', 'BSD-3-Clause', 'BSD-2-Clause')
DEFAULT_DIR = os.path.expanduser('~/FalconWorkspace/models/perception')

# name: (repo, revision, last modified, licence as stated, where it was read, parameters, task, the class to load,
#        [(file, size, LFS SHA-256 or None, git blob id)])
_PINNED = {
    'rtdetr-v2-r18': (
        'PekingU/rtdetr_v2_r18vd', '5650961749fa93567c0d46fc7f43ea4f9e914107', '2025-02-06', 'Apache-2.0',
        'card metadata "license: apache-2.0" (https://huggingface.co/PekingU/rtdetr_v2_r18vd); code: '
        'https://github.com/lyuwenyu/RT-DETR (Apache-2.0)', 20209716, 'detector', 'RTDetrV2ForObjectDetection', [
            ('config.json', 5437, None, '6f984c30c65857419deeeebfdf7dc4e7d7bce6f8'),
            ('model.safetensors', 80904640, 'd18309d0d7ea57048138885c4c6ecfcb1e24506fc6153b94ad484f8ab62c7115',
             '820a37cacd1a40bd83f736222a2f6ddd5e1f4eab'),
            ('preprocessor_config.json', 444, None, 'fcdff16b42e5ebc51d28f59184490c6eff91a88b')]),
    'rtdetr-v2-r50': (
        'PekingU/rtdetr_v2_r50vd', '282494075698cab9faa1096ae26856890030c817', '2025-02-06', 'Apache-2.0',
        'card metadata "license: apache-2.0" (https://huggingface.co/PekingU/rtdetr_v2_r50vd); code: '
        'https://github.com/lyuwenyu/RT-DETR (Apache-2.0)', 43019444, 'detector', 'RTDetrV2ForObjectDetection', [
            ('config.json', 5283, None, '9891f5364789e54553d1cfeed69693f5bb05fa44'),
            ('model.safetensors', 172176856, '3331d977dbc0c7a6cdae9ec0b0b6ad156eb6720d65b7cf0fa710dcc541d88d71',
             '81da7e7c3845592e703b6a14ba30e0237440075e'),
            ('preprocessor_config.json', 444, None, 'fcdff16b42e5ebc51d28f59184490c6eff91a88b')]),
    'da2-small': (
        'depth-anything/Depth-Anything-V2-Small-hf', '5426e4f0f36572d16453bbda7a8389317b1bef99', '2024-07-05',
        'Apache-2.0', 'card metadata "license: apache-2.0" '
        '(https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf); README of '
        'https://github.com/DepthAnything/Depth-Anything-V2: "Depth-Anything-V2-Small model is under the Apache-2.0 '
        'license"', 24785089, 'relative_depth', 'DepthAnythingForDepthEstimation', [
            ('config.json', 950, None, '9207d3a304e86517dab21638c559d8b3195d41b6'),
            ('model.safetensors', 99173660, '3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70',
             '94c4baf1f543643f97b56508cd2b9d03b2529c6b'),
            ('preprocessor_config.json', 775, None, 'a7822a90a0e91a14d50d1996953bac8f389a4377')]),
    'metric3d-small': (
        'onnx-community/metric3d-vit-small', 'ee63b95d92d2f629b3e019e88f57eea7c1e97d4b', '2024-06-20',
        'unstated (weights)', 'the ONNX conversion\'s card metadata says "license: cc0-1.0" and has no text '
        '(https://huggingface.co/onnx-community/metric3d-vit-small); the upstream code '
        'https://github.com/YvanYin/Metric3D is BSD-2-Clause and its README asks commercial users to contact the '
        'authors; no licence for the trained weights is stated', None, 'metric_depth', 'onnxruntime', [
            ('config.json', 30, None, '83189832d06dc29318b62972aa11f58299cf5ac8'),
            ('onnx/model.onnx', 150739433, '674a665052f01bb2b64687200182f3380f0a462f4b6ef2e51fefd06c84d8ee75',
             '098eb83a90e14d130f414219944bf8260c9a0e8b'),
            ('preprocessor_config.json', 288, None, '5ff0694d2ee940bac3e38d57fb17b5214126fe88')]),
}
NOTES = {
    'rtdetr-v2-r18': 'RT-DETRv2-S (Lv et al. 2024, arXiv:2407.17140), COCO\'s 80 classes; the default detector',
    'rtdetr-v2-r50': 'RT-DETRv2-M* (arXiv:2407.17140), COCO; more accurate, about twice the cost',
    'da2-small': 'Depth Anything V2 Small (arXiv:2406.09414): relative depth, scaled by the ground plane',
    'metric3d-small': 'Metric3D v2 ViT-S (arXiv:2404.15506) as ONNX: metric depth; opt-in (weights\' terms not stated)',
}
UNSTATED = ('metric3d-small',)                 # fetched only with an explicit acknowledgement

CANDIDATES = {}
for _n, (_repo, _rev, _mod, _lic, _src, _params, _task, _cls, _files) in _PINNED.items():
    CANDIDATES[_n] = {'repo': _repo, 'revision': _rev, 'last_modified': _mod, 'licence': _lic,
                      'licence_source': _src, 'licence_checked': LICENCE_CHECKED, 'params': _params, 'task': _task,
                      'loader': _cls, 'note': NOTES[_n], 'opt_in': _n in UNSTATED,
                      'bytes': sum(f[1] for f in _files),
                      'files': [{'name': f[0], 'size': f[1], 'lfs_sha256': f[2], 'git_oid': f[3]} for f in _files],
                      'sha256_pin': 'to be pinned on first approved download',
                      'metadata': f'{HF}/api/models/{_repo}?blobs=true (read {LICENCE_CHECKED}; no weights downloaded)'}

# name: (what, licence, where it was read, why refused)
REFUSED = {
    'yolo': ('every Ultralytics YOLO (YOLOv5, YOLOv8, YOLO11, …)', 'AGPL-3.0',
             'https://github.com/ultralytics/ultralytics/blob/main/LICENSE (GNU Affero General Public License v3, '
             f'read {LICENCE_CHECKED})',
             'AGPL-3.0 would bind the app (Falcon refuses AGPL code and weights; it never imports ultralytics)'),
    'da2-base': ('depth-anything/Depth-Anything-V2-Base-hf', 'CC-BY-NC-4.0',
                 f'card metadata, and the project README ("Base/Large/Giant models are under the CC-BY-NC-4.0 '
                 f'license"), read {LICENCE_CHECKED}', 'non-commercial'),
    'da2-large': ('depth-anything/Depth-Anything-V2-Large-hf', 'CC-BY-NC-4.0',
                  f'card metadata and the project README, read {LICENCE_CHECKED}', 'non-commercial'),
    'smpl': ('SMPL / SMPL-X body models and models built on them', 'SMPL Model License (non-commercial)',
             'https://smpl.is.tue.mpg.de/modellicense.html', 'non-commercial research only'),
}
REFUSED_PATTERNS = (('yolo', 'yolo'), ('ultralytics', 'yolo'), ('depth-anything-v2-base', 'da2-base'),
                    ('depth-anything-v2-large', 'da2-large'), ('depth-anything-v2-giant', 'da2-large'),
                    ('smpl', 'smpl'))


class PerceptionModelError(Exception):
    pass


def refused_reason(name: str) -> str | None:
    """Why a model name is refused (a REFUSED key or anything that names one), or None."""
    n = str(name or '').lower()
    key = n if n in REFUSED else next((k for pat, k in REFUSED_PATTERNS if pat in n), None)
    if key is None:
        return None
    what, lic, _src, why = REFUSED[key]
    return f'{what} is refused: {lic} — {why}'


def check_licence(c: dict, accept_unstated: bool = False) -> list:
    """Problems with a candidate's licence: AGPL or non-commercial refused; one not stated needs the explicit
    acknowledgement; anything else not on the permissive list refused."""
    lic = str(c.get('licence') or '')
    up = lic.upper()
    if 'AGPL' in up:
        return ['an AGPL-licensed model is refused (its licence would bind the app)']
    if '-NC' in up or 'NON-COMMERCIAL' in up or 'NONCOMMERCIAL' in up:
        return [f'licence {lic} is non-commercial: refused']
    if lic.startswith('unstated'):
        return [] if accept_unstated else [
            'the weights\' terms are not stated (see the licence source); they are fetched or loaded only with '
            '--accept-unstated-weights-terms, which records your acknowledgement']
    if lic not in PERMISSIVE:
        return [f'licence {lic or "not stated"} is not one Falcon accepts by default ({", ".join(PERMISSIVE)})']
    return []


def candidate(name: str) -> dict:
    why = refused_reason(name)
    if why:
        raise PerceptionModelError(why)
    if name not in CANDIDATES:
        raise PerceptionModelError(f'unknown model {name!r}; one of {", ".join(CANDIDATES)}')
    return CANDIDATES[name]


def candidate_dir(models_dir: str, name: str) -> str:
    return os.path.join(models_dir or DEFAULT_DIR, name)


def git_oid(path: str) -> str:
    h = hashlib.sha1()
    h.update(f'blob {os.path.getsize(path)}\0'.encode())
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def describe(name: str) -> str:
    """The text shown before a download is accepted."""
    c = candidate(name)
    lines = [f'{name}: {c["repo"]} at revision {c["revision"]} ({c["note"]})',
             f'licence: {c["licence"]} — {c["licence_source"]} (checked {c["licence_checked"]})',
             f'download: {c["bytes"] / 2 ** 20:.1f} MiB in {len(c["files"])} files from {HF}/{c["repo"]}/resolve/'
             f'{c["revision"]}/', 'publisher checksums (checked on download; Falcon pins its SHA-256 then):']
    for f in c['files']:
        lines.append(f'  {f["name"]:28s} {f["size"]:>11d} B  ' + (f'LFS sha256 {f["lfs_sha256"]}' if f['lfs_sha256']
                                                                  else f'git blob {f["git_oid"]}'))
    if c['opt_in']:
        lines.append('OPT-IN: the weights\' terms are not stated; --accept-unstated-weights-terms is required')
    return '\n'.join(lines)


def verify(name: str, folder: str, full: bool = True) -> dict:
    """A downloaded candidate checked: pins.json for this revision, every file's size, its SHA-256 against the pin
    and (full) against the publisher's checksum. {'ok', 'problems', 'pins'}."""
    from .provenance import sha256_file
    c = candidate(name)
    problems, pins = [], None
    pp = os.path.join(folder, PINS)
    if os.path.isfile(pp):
        with open(pp, encoding='utf-8') as fh:
            pins = json.load(fh)
        if pins.get('revision') != c['revision']:
            problems.append(f'pins.json is for revision {pins.get("revision")}, not {c["revision"]}')
    else:
        problems.append('no pins.json: not downloaded by Falcon (falcon dashcam nearmiss fetch)')
    for f in c['files']:
        p = os.path.join(folder, f['name'])
        if not os.path.isfile(p):
            problems.append(f'{f["name"]}: missing')
            continue
        if os.path.getsize(p) != f['size']:
            problems.append(f'{f["name"]}: {os.path.getsize(p)} bytes, not {f["size"]}')
            continue
        h = sha256_file(p)
        pin = ((pins or {}).get('files') or {}).get(f['name'], {}).get('sha256')
        if pin and h != pin:
            problems.append(f'{f["name"]}: SHA-256 {h} is not the pinned {pin}')
        if full:
            if f['lfs_sha256'] and h != f['lfs_sha256']:
                problems.append(f'{f["name"]}: SHA-256 is not the publisher\'s {f["lfs_sha256"]}')
            elif not f['lfs_sha256'] and git_oid(p) != f['git_oid']:
                problems.append(f'{f["name"]}: git blob id is not the publisher\'s {f["git_oid"]}')
    return {'ok': not problems, 'problems': problems, 'pins': pins}


def load_folder(name: str, models_dir: str = '', accept_unstated: bool = False) -> str:
    """The verified local folder of a downloaded candidate (refused when missing, unverified, or — for an opt-in
    model — without the acknowledgement, given now or recorded when it was fetched)."""
    c = candidate(name)
    folder = candidate_dir(models_dir, name)
    if not os.path.isdir(folder):
        raise PerceptionModelError(f'{name} is not downloaded: falcon dashcam nearmiss fetch {name} --accept-licence'
                                   f' ({c["licence"]}, {c["bytes"] / 2 ** 20:.0f} MiB) — asked for, never automatic')
    v = verify(name, folder)
    if not v['ok']:
        raise PerceptionModelError(f'{name}: ' + '; '.join(v['problems']))
    recorded = bool((v['pins'] or {}).get('accepted_unstated_terms'))
    bad = check_licence(c, accept_unstated or recorded)
    if bad:
        raise PerceptionModelError(f'{name}: ' + '; '.join(bad))
    return folder


def fetch(name: str, models_dir: str = '', accept_licence: bool = False, accept_unstated: bool = False,
          progress=None, base_url: str = HF) -> str:
    """Download a candidate (module doc). accept_licence must be True (the caller showed describe()); an opt-in
    model also needs accept_unstated. Returns the folder."""
    import urllib.request
    from .provenance import utc_now, write_json
    c = candidate(name)
    if not accept_licence:
        raise PerceptionModelError(f'{name}: the licence must be accepted first (--accept-licence)\n{describe(name)}')
    bad = check_licence(c, accept_unstated)
    if bad:
        raise PerceptionModelError(f'{name}: ' + '; '.join(bad))
    folder = candidate_dir(models_dir, name)
    os.makedirs(folder, exist_ok=True)
    total, done = c['bytes'], 0
    pins = {'name': name, 'repo': c['repo'], 'revision': c['revision'], 'licence': c['licence'],
            'licence_source': c['licence_source'], 'accepted_licence': utc_now(),
            'accepted_unstated_terms': utc_now() if c['opt_in'] else None, 'files': {},
            'source': f'{base_url}/{c["repo"]}/resolve/{c["revision"]}/', 'downloaded': None}
    for f in c['files']:
        p = os.path.join(folder, f['name'])
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + '.part'
        h256, h1 = hashlib.sha256(), hashlib.sha1()
        h1.update(f'blob {f["size"]}\0'.encode())
        got = 0
        with urllib.request.urlopen(f'{base_url}/{c["repo"]}/resolve/{c["revision"]}/{f["name"]}', timeout=60) as r, \
                open(tmp, 'wb') as fh:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                fh.write(b)
                h256.update(b)
                h1.update(b)
                got += len(b)
                if progress and total:
                    progress((done + got) / total)
        why = None
        if got != f['size']:
            why = f'{got} bytes, not {f["size"]}'
        elif f['lfs_sha256'] and h256.hexdigest() != f['lfs_sha256']:
            why = f'SHA-256 {h256.hexdigest()} is not the publisher\'s {f["lfs_sha256"]}'
        elif not f['lfs_sha256'] and h1.hexdigest() != f['git_oid']:
            why = f'git blob id {h1.hexdigest()} is not the publisher\'s {f["git_oid"]}'
        if why:
            os.remove(tmp)
            raise PerceptionModelError(f'{f["name"]}: {why} — not installed')
        os.replace(tmp, p)
        done += f['size']
        pins['files'][f['name']] = {'size': f['size'], 'sha256': h256.hexdigest(),
                                    'checked': 'publisher LFS SHA-256' if f['lfs_sha256'] else 'publisher git blob id'}
    pins['downloaded'] = utc_now()
    write_json(os.path.join(folder, PINS), pins)
    return folder


def list_models(models_dir: str = '') -> list:
    """Rows for a table: every candidate with its state, then the refused ones."""
    rows = []
    for n, c in CANDIDATES.items():
        folder = candidate_dir(models_dir, n)
        rows.append({'name': n, 'repo': c['repo'], 'task': c['task'], 'licence': c['licence'],
                     'licence_source': c['licence_source'], 'checked': c['licence_checked'], 'bytes': c['bytes'],
                     'opt_in': c['opt_in'], 'refused': None,
                     'state': 'downloaded (pinned)' if os.path.isfile(os.path.join(folder, PINS)) else
                     'not downloaded'})
    for n, (what, lic, src, why) in REFUSED.items():
        rows.append({'name': n, 'repo': what, 'task': None, 'licence': lic, 'licence_source': src,
                     'checked': LICENCE_CHECKED, 'bytes': None, 'opt_in': False, 'refused': why, 'state': 'refused'})
    return rows
