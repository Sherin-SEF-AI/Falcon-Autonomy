"""The bird's-eye surround view (core.bevstitch): a synthetic ground rendered into four fisheye cameras through the lens
models (core.lens) and stitched back — to within a pixel —, the pinhole homography against the lens path, seams from
a Vantage rig and by Vantage's rule, exposure gains recovered, the exported tables read by the Apache-2.0 runtime
with the same result, the torch path against OpenCV's, the CLI on a recorded run, and Rigs ▸ Surround view
offscreen. GPU tests (the cuda marker) run with FALCON_CUDA=1 only."""
import json
import math
import os

import numpy as np
import pytest

from falcon_autonomy.core import bevstitch as B
from falcon_autonomy.core import lens as LENS

SIZE = (480, 300)


@pytest.fixture(scope='module')
def rig():
    return B.surround_rig(*SIZE)


@pytest.fixture(scope='module')
def lut(rig):
    return B.build_lut(rig, res=0.04)


def _interior(lut, rig, margin_px=1.5):
    """BEV pixels every contributing camera samples well inside its image and its lens's field (bilinear reads of
    the rendered image never touch its border or the black beyond the image circle)."""
    sens = {s['id']: s for s in rig['sensors']}
    good = np.ones((lut.ny, lut.nx), bool)
    for i in lut.ids:
        intr = LENS.intrinsics(sens[i])
        u, v = lut.map_x[i], lut.map_y[i]
        on = lut.weight[i] > 0
        _, th = intr.unproject(u, v)
        ok = (u >= margin_px) & (u <= intr.width - margin_px) & (v >= margin_px) & (v <= intr.height - margin_px) & \
            (th <= intr.max_theta - 2.0 / intr.f_px)
        good &= ~on | ok
    return good & lut.covered & ~lut.body


def coord_texture(x, y):
    return np.stack([x, y, np.zeros_like(x)], -1)


def test_checkerboard_ground_is_stitched_back_within_a_pixel(rig, lut):
    """Each fisheye's image of the ground carries the ground's own coordinates (x, y as values): stitched back, every
    covered pixel must hold its own position — the remap tables (the forward lens model) against the rendering (its
    inverse), through OpenCV's fixed-point remap and the seam feathering."""
    imgs = {c['id']: B.render_ground(c, coord_texture, supersample=1) for c in rig['sensors']}
    st = B.Stitcher(lut, 'cpu')
    res = st.stitch(imgs)
    X, Y = lut.world_xy()
    good = _interior(lut, rig)
    assert good.sum() > 0.8 * (lut.covered & ~lut.body).sum()
    err = np.hypot(res.image[0] - X, res.image[1] - Y)[good] / lut.res
    assert err.max() < 1.0, err.max()
    assert np.percentile(err, 99) < 0.3 and np.median(err) < 0.05
    # and a real checkerboard: away from the squares' edges every pixel has its square's colour
    tex = B.checkerboard(0.5)
    imgs = {c['id']: np.clip(np.rint(B.render_ground(c, tex, supersample=3)), 0, 255).astype(np.uint8)
            for c in rig['sensors']}
    out = st.stitch(imgs).rgb().astype(float)
    truth = tex(X.ravel(), Y.ravel()).reshape(lut.ny, lut.nx, 3)
    fx = np.abs((X / 0.5) - np.round(X / 0.5)) * 0.5
    fy = np.abs((Y / 0.5) - np.round(Y / 0.5)) * 0.5
    # a camera pixel's footprint on the ground there (the tables' area scale): far from a camera one fisheye pixel
    # covers several centimetres, and its colour mixes the squares within it
    foot = np.zeros((lut.ny, lut.nx))
    for i in lut.ids:
        gu = np.gradient(lut.map_x[i])
        gv = np.gradient(lut.map_y[i])
        # the smallest singular value of d(u, v)/d(row, col): camera pixels per BEV pixel along the worst direction
        a, b, c, d = gu[0], gu[1], gv[0], gv[1]
        s1 = a * a + b * b + c * c + d * d
        s2 = np.sqrt(np.maximum((a * a + b * b - c * c - d * d) ** 2 + 4 * (a * c + b * d) ** 2, 0))
        smin = np.sqrt(np.maximum(0.5 * (s1 - s2), 1e-12))
        f_m = lut.res / smin
        foot = np.where(lut.weight[i] > 0, np.maximum(foot, f_m), foot)
    away = good & (np.minimum(fx, fy) > 1.5 * np.maximum(lut.res, foot))
    assert away.sum() > 0.1 * good.sum()
    d = np.abs(out - truth).max(-1)[away]
    assert np.percentile(d, 99.5) < 12 and d.mean() < 1.5, (np.percentile(d, 99.5), d.mean())


def test_tables_layout_seams_and_body(rig, lut):
    assert lut.seam_source == 'estimated' and len(lut.seams) == 4
    s = lut.summary()
    assert s['covered_share'] > 0.97 and all(c['method'] == 'lens' for c in s['cameras'])
    # each camera owns the ground in its own direction (Vantage's azimuth: counter-clockwise, y to the left)
    X, Y = lut.world_xy()
    for az, cam in ((0, 'svm-front'), (90, 'svm-left'), (180, 'svm-rear'), (-90, 'svm-right')):
        r, c = lut.pixel_of(lut.centre_x + 3 * math.cos(math.radians(az)), -3 * math.sin(math.radians(az)))
        assert lut.ids[lut.owner[int(r), int(c)]] == cam, (az, cam)
        from falcon_autonomy.core.rigcompare import owner
        assert owner(lut.seams, az) == cam
    # weights sum to one where covered, nothing under the body, the front at the top, the left on the left
    tot = sum(lut.weight[i] for i in lut.ids)
    assert np.allclose(tot[lut.covered], 1.0, atol=1e-5) and not lut.covered[lut.body].any()
    assert X[0, 0] > X[-1, 0] and Y[0, 0] < Y[0, -1]
    # a seam is feathered: across it both cameras contribute
    both = sum((lut.weight[i] > 0.05) & (lut.weight[i] < 0.95) for i in lut.ids)
    assert both.sum() > 50


def test_pinhole_homography_equals_the_lens_path():
    sensor = {'id': 'c', 'type': 'camera.pinhole', 'width': 640, 'height': 400, 'fov': 100.0,
              'pose': {'x': 2.0, 'y': 0.3, 'z': 1.2, 'roll': 2.0, 'pitch': -35.0, 'yaw': 10.0}}
    rng = np.random.default_rng(0)
    P = np.c_[rng.uniform(2.5, 10, 2000), rng.uniform(-4, 4, 2000), np.zeros(2000)]
    u, v, ok, _, _, method = B.project_ground(sensor, P)
    assert method == 'homography' and ok.sum() > 500
    intr = LENS.intrinsics(sensor)
    R, pos = B._pose_of(sensor)
    u2, v2, _, ok2 = intr.project_dirs((P - pos) @ R)
    assert (ok == ok2).mean() > 0.999
    m = ok & ok2
    assert np.abs(u[m] - u2[m]).max() < 1e-6 and np.abs(v[m] - v2[m]).max() < 1e-6
    # a pinhole rig's tables use it
    rig = {'id': 'p', 'vehicle': 'vehicle.lincoln.mkz_2020', 'sensors': [
        dict(sensor, id=f'p{k}', name=f'p{k}', pose=dict(sensor['pose'], x=x, y=y, yaw=yaw))
        for k, (x, y, yaw) in enumerate(((2.45, 0, 0), (-2.45, 0, 180), (0.9, -1.0, -90), (0.9, 1.0, 90)))]}
    lt = B.build_lut(rig, res=0.08)
    assert set(lt.method.values()) == {'homography'}


def test_seams_come_from_a_vantage_rig():
    from test_rigcompare import SAMPLE
    from falcon_autonomy.core import vantagerig as VR
    rig = VR.rigs_for(VR.read(SAMPLE), variants=['1'], image_scale=0.1)[0]
    lt = B.build_lut(rig, res=0.08)
    assert lt.seam_source == 'vantage'
    names = {s['id']: s['name'] for s in rig['sensors']}
    want = sorted((names[s['a']] if s['a'] in names else s['a'], s['angle_deg']) for s in lt.seams)
    have = sorted((s['a'], s['angle_deg']) for s in rig['vantage']['seams'])
    assert [w[1] for w in want] == [h[1] for h in have]
    assert lt.centre_x == pytest.approx(rig['vantage']['centre']['falcon_x'])
    assert lt.blend_m == pytest.approx(rig['vantage']['blend_m'])
    # a rig without the block is estimated
    plain = dict(rig)
    plain.pop('vantage')
    assert B.rig_seams(plain)['source'] == 'estimated'


def test_exposure_gains_are_recovered(rig, lut):
    tex = B.checkerboard(0.5, 70, 170)
    imgs = {c['id']: B.render_ground(c, tex, supersample=1) for c in rig['sensors']}
    imgs['svm-left'] = imgs['svm-left'] * 1.35
    imgs = {k: np.clip(np.rint(v), 0, 255).astype(np.uint8) for k, v in imgs.items()}
    st = B.Stitcher(lut, 'cpu')
    res = st.stitch(imgs, gains='auto')
    g = {k: np.mean(v) for k, v in res.gains.items()}
    others = np.mean([g[k] for k in g if k != 'svm-left'])
    assert g['svm-left'] / others == pytest.approx(1 / 1.35, rel=0.05)
    # balanced, the seams between the left camera and its neighbours agree better
    w, v = B.warp_cpu(lut, imgs)
    pairs = B.overlap_means(lut, w, v)
    before = np.mean([abs(ma.mean() - mb.mean()) for a, b, n, ma, mb in pairs if 'svm-left' in (a, b)])
    after = np.mean([abs(ma.mean() * g[a] - mb.mean() * g[b]) for a, b, n, ma, mb in pairs if 'svm-left' in (a, b)])
    assert after < 0.25 * before


def test_exported_tables_run_in_the_apache_runtime(rig, lut, tmp_path):
    out = str(tmp_path / 'tables')
    paths = B.export(lut, out, rig)
    names = {os.path.relpath(p, out) for p in paths}
    assert {'bev_lut.json', 'bev_lut.npz', 'bev_lut.txt', 'body_mask.bin', 'stitcher/bev_stitch.py',
            'stitcher/LICENSE'} <= names
    head = open(os.path.join(out, 'stitcher', 'bev_stitch.py'), encoding='utf-8').read(300)
    assert 'SPDX-License-Identifier: Apache-2.0' in head
    assert 'Apache License' in open(os.path.join(out, 'stitcher', 'LICENSE'), encoding='utf-8').read(400)
    meta = json.load(open(os.path.join(out, 'bev_lut.json')))
    assert meta['width'] == lut.nx and meta['seam_source'] == 'estimated' and 'smeared' in meta['limits']
    rt = B.load_runtime()
    st_rt = rt.Stitcher(out)
    tex = B.checkerboard(0.5)
    imgs = {c['id']: np.clip(np.rint(B.render_ground(c, tex, supersample=2)), 0, 255).astype(np.uint8)
            for c in rig['sensors']}
    by_tag = {c['name']: imgs[c['id']] for c in rig['sensors']}
    a = st_rt.stitch(by_tag).astype(int)
    b = B.Stitcher(lut, 'cpu').stitch(imgs).rgb().astype(int)
    d = np.abs(a - b)
    assert d.max() <= 2 and d.mean() < 0.2, (d.max(), d.mean())
    # half the resolution of the frames: the runtime scales the tables
    small = {k: np.ascontiguousarray(v[::2, ::2]) for k, v in by_tag.items()}
    assert st_rt.stitch(small).shape == a.shape


def test_torch_path_matches_opencv(rig, lut):
    pytest.importorskip('torch')
    tex = B.checkerboard(0.5)
    imgs = {c['id']: np.clip(np.rint(B.render_ground(c, tex, supersample=2)), 0, 255).astype(np.uint8)
            for c in rig['sensors']}
    a = B.Stitcher(lut, 'torch-cpu').stitch(imgs)
    b = B.Stitcher(lut, 'cpu').stitch(imgs)
    assert a.backend == 'torch:cpu' and tuple(a.image.shape) == (3, lut.ny, lut.nx)
    d = np.abs(a.rgb().astype(int) - b.rgb().astype(int))
    # OpenCV quantises sub-pixel positions to 1/32 px: a level or two at sharp edges
    assert d.mean() < 0.5 and np.percentile(d, 99.9) <= 4, (d.mean(), d.max())
    assert np.array_equal(a.valid, b.valid) and np.array_equal(a.owner, b.owner)
    # a camera without a frame: its ground goes to the others that see it
    part = dict(imgs)
    part.pop('svm-rear')
    r = B.Stitcher(lut, 'cpu').stitch(part)
    assert not np.any(r.owner == lut.ids.index('svm-rear'))


@pytest.mark.cuda
def test_gpu_half_precision_equals_the_cpu(rig, lut):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('no CUDA GPU')
    torch.cuda.set_per_process_memory_fraction(0.35)
    tex = B.checkerboard(0.5)
    imgs = {c['id']: np.clip(np.rint(B.render_ground(c, tex, supersample=2)), 0, 255).astype(np.uint8)
            for c in rig['sensors']}
    b = B.Stitcher(lut, 'cpu').stitch(imgs)
    a = B.Stitcher(lut, 'cuda', half=False).stitch(imgs)
    assert a.backend == 'torch:cuda'
    d = np.abs(a.rgb().astype(int) - b.rgb().astype(int))
    assert d.mean() < 0.5 and np.percentile(d, 99.9) <= 4, (d.mean(), d.max())
    assert np.array_equal(a.valid, b.valid)
    # half precision: the sampling grid's 11-bit mantissa moves a sample by up to ~1/4 px of a 480 px image, which
    # shows at the checkerboard's sharp edges only
    h = B.Stitcher(lut, 'cuda').stitch(imgs)
    assert h.backend == 'torch:cuda:fp16'
    d = np.abs(h.rgb().astype(int) - b.rgb().astype(int))
    assert d.mean() < 1.0 and np.percentile(d, 99) <= 6, (d.mean(), np.percentile(d, 99))


@pytest.mark.cuda
def test_fps_four_1080p_fisheyes_on_the_gpu():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('no CUDA GPU')
    torch.cuda.set_per_process_memory_fraction(0.35)
    res = B.bench(B.surround_rig(1920, 1080), frames=60, device='cuda')
    assert res['fps'] >= 30, res
    assert res['peak_gpu_memory_gib'] < 2.0


def make_surround_run(path, rig, steps=3):
    """A finished falcon-raw run of the rig: every camera's view of a checkerboard ground per step."""
    import cv2
    os.makedirs(path, exist_ok=True)
    tex = B.checkerboard(0.5)
    rows = []
    for k in range(steps):
        files = {}
        for c in rig['sensors']:
            img = np.clip(np.rint(B.render_ground(c, lambda x, y: tex(x + 0.25 * k, y), supersample=1)), 0, 255)
            rel = f'rgb/{c["id"]}/{k:06d}.png'
            os.makedirs(os.path.join(path, os.path.dirname(rel)), exist_ok=True)
            cv2.imwrite(os.path.join(path, rel), cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
            files[f'{c["id"]}/rgb'] = rel
        rows.append({'step': k, 'frame': k, 'sim_t': k * 0.1, 'files': files, 'recorded': True,
                     'ego': {'x': 0.25 * k, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}})
    with open(os.path.join(path, 'frames.jsonl'), 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    json.dump({'town': 'Synthetic', 'rig': rig, 'fps': 10.0}, open(os.path.join(path, 'drive.json'), 'w'))
    json.dump({'schema': 'falcon.dataset/1', 'layout': 'falcon-raw', 'frames': steps, 'fps': 10.0,
               'sensors': [dict(c, streams=['rgb']) for c in rig['sensors']]}, open(os.path.join(path, 'dataset.json'),
                                                                                    'w'))
    json.dump({'schema': 'falcon.run/1', 'status': 'ok', 'started': '2026-11-12T00:00:00Z'},
              open(os.path.join(path, 'run.json'), 'w'))
    return path


def test_cli_stitch_export_and_bench(tmp_path, capsys):
    rig = B.surround_rig(240, 150)
    run = make_surround_run(str(tmp_path / 'run'), rig)
    assert B.main_bev(['stitch', run, '--res', '0.08', '--balance', '--seams', '--device', 'cpu']) == 0
    doc = json.load(open(os.path.join(run, 'bev', 'bev.json')))
    assert len(doc['frames']) == 3 and doc['backend'] == 'cv2:cpu' and doc['tables']['seam_source'] == 'estimated'
    assert os.path.isfile(os.path.join(run, 'bev', '000002.png'))
    rj = str(tmp_path / 'rig.json')
    json.dump({'schema': 'falcon.rig/1', 'rig': rig}, open(rj, 'w'))
    assert B.main_bev(['export', rj, '--out', str(tmp_path / 'tables'), '--res', '0.1']) == 0
    assert os.path.isfile(str(tmp_path / 'tables' / 'stitcher' / 'bev_stitch.py'))
    assert B.main_bev(['bench', '--size', '240x150', '--frames', '2', '--device', 'cpu', '--res', '0.1']) == 0
    out = capsys.readouterr().out
    assert '"fps"' in out
    assert B.main_bev(['stitch', str(tmp_path / 'nothing')]) == 2
    for argv in (['--help'], ['stitch', '--help'], ['export', '--help'], ['bench', '--help']):
        assert B.main_bev(argv) == 0


from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401  (fixtures)


@pytest.mark.gui
def test_rigs_surround_view_tab(window, qapp, tmp_path):
    from falcon_autonomy.core.project import Edit
    from falcon_autonomy.ui.bev_tab import BevTab
    from falcon_autonomy.ui.pages.rigs import RigsPage
    rig = B.surround_rig(240, 150)
    window.edit(Edit('set', ['rigs', 0], dict(rig, id=window.project.doc['rigs'][0]['id']), text='surround rig'))
    page = RigsPage(window)
    page.show()
    tab = page.surround
    assert isinstance(tab, BevTab)
    run = make_surround_run(str(tmp_path / 'run'), dict(rig, id=window.project.doc['rigs'][0]['id']), steps=2)
    tab.set_run(run)
    assert pump(qapp, lambda: tab.last is not None, timeout=60)
    assert tab.last['image'].ndim == 3 and tab.last['step'] == 0
    assert 'estimated' in tab.info.value('Seams') if hasattr(tab.info, 'value') else True
    assert pump(qapp, lambda: not tab.busy and not tab.pending, timeout=60)
    tab.last = None
    tab.frame.setValue(1)
    assert pump(qapp, lambda: tab.last is not None, timeout=60) and tab.last['step'] == 1
    paths = tab.export(str(tmp_path / 'tables'))
    assert paths and os.path.isfile(str(tmp_path / 'tables' / 'bev_lut.json'))
    page.close()
    page.deleteLater()
