"""Hybrid worlds (core.recon.hybrid): CARLA's actors composited over a reconstruction of the static scene — the
composite and its rules (CARLA's occlusion, the depth test and its relabelling, hard links never written through),
the falcon-raw run and its provenance, the measures against CARLA's own frames, refusals, the re-drive's log and
config, the command line, a re-drive end to end against the fake carla, and the Replay page's section. Everything
runs on the CPU on the synthetic street of test_recon (rendered here)."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')
cv2 = pytest.importorskip('cv2')

from test_recon import _settings, make_falcon_run  # noqa: E402


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    from falcon_autonomy.core.recon.train import train
    root = tmp_path_factory.mktemp('hybrid')
    run = make_falcon_run(str(root / 'run'), n=24)
    out = str(root / 'recon')
    torch.set_num_threads(2)
    train(f'falcon:{run}', out, _settings())
    return run, out, root


def _rows(path):
    with open(path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def _png(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def test_hybrid_of_the_source_drive_keeps_carlas_actors_and_labels(trained, tmp_path):
    from falcon_autonomy.core.exporters import Run
    from falcon_autonomy.core.provenance import sha256_file
    from falcon_autonomy.core.recon.hybrid import hybrid_run
    run, rec_dir, _ = trained
    res = hybrid_run(rec_dir, run, str(tmp_path / 'h'), device='cpu')
    out = res['run']
    assert res['method'] == 'mask' and res['measured'] and res['frames'] == 24
    r = Run(out)                                    # a falcon-raw run every tool reads
    assert len(r.frames) == 24 and r.run['worker'] == 'recon.hybrid' and r.run['status'] == 'ok'
    hy = r.run['hybrid']
    assert hy['splat_sha256'] == sha256_file(os.path.join(rec_dir, 'point_cloud.ply'))
    assert hy['source_run'] == os.path.abspath(run) and hy['compositing']['method'] == 'mask' and hy['own_source']
    assert r.ds['hybrid']['labels'].startswith('CARLA') and 'shadows' in json.dumps(hy['not_modelled'])
    assert set(hy['inputs']) >= {'source/run.json', 'recon/point_cloud.ply'}
    rows = _rows(os.path.join(out, 'hybrid.jsonl'))
    assert len(rows) == 24 and {x['split'] for x in rows} == {'train', 'test'}
    for k in (0, 3, 20):
        rel = f'rgb/cam-front/{k:06d}.png'
        src, hyb = cv2.imread(os.path.join(run, rel)), cv2.imread(os.path.join(out, rel))
        inst = _png(os.path.join(run, f'instance/cam-front/{k:06d}.png'))
        actor = inst == 5
        assert np.array_equal(src[actor], hyb[actor])           # the actor's pixels are CARLA's, exactly
        for kind in ('semantic', 'instance', 'range'):          # every label is CARLA's, hard-linked
            a, b = (os.path.join(d, f'{kind}/cam-front/{k:06d}.png') for d in (run, out))
            assert os.path.samefile(a, b)
        assert not os.path.samefile(os.path.join(run, rel), os.path.join(out, rel))
    assert any(x['actor_share'] > 0 for x in rows) and all(x['hidden_px'] == 0 for x in rows)
    s = res['summary']
    a = s['all']
    assert a['frames'] == 24 and a['background']['psnr'] > 20 and a['whole']['psnr'] >= a['background']['psnr'] - 1
    assert a['actors']['identical'] and a['actors']['psnr'] is None and a['actors']['ssim'] == 1.0
    assert 'depth' in a['alternative'] and a['alternative']['depth']['actors']['psnr'] is not None
    assert a['labels']['accuracy'] > 0.8 and a['labels']['classes'] >= 2
    assert a['depth_background_median_abs_rel'] < 0.15 and a['shift_px_median'] < 1.0
    assert s['heldout']['frames'] == 3 and s['train']['frames'] == 21
    assert a['render_ms']['mean'] > 0 and abs(a['abs_lateral_m']) < 1e-3
    assert os.listdir(os.path.join(out, 'hybrid-previews'))
    assert r.run['outputs'] and 'hybrid.jsonl' in r.run['outputs']


def test_a_drive_beside_the_path_with_other_traffic(trained, tmp_path):
    from falcon_autonomy.core.recon.hybrid import hybrid_run
    _, rec_dir, root = trained
    other = make_falcon_run(str(tmp_path / 'other'), n=10, lateral=0.5)
    res = hybrid_run(rec_dir, other, str(tmp_path / 'h'), device='cpu')
    a = res['summary']['all']
    assert {'all', 'train', 'heldout'} <= set(res['summary']) and abs(a['lateral_m'] - 0.5) < 0.05
    assert a['background']['psnr'] > 15 and a['actors']['identical']


def test_depth_test_hides_actors_behind_the_background_and_relabels_them(trained, tmp_path):
    """An actor whose range says it lies behind the reconstructed street is hidden by the depth test; its pixels take
    the reconstruction's labels in the hybrid run, and the source's files (hard links) are never written through."""
    from falcon_autonomy.core.recon.hybrid import hybrid_run
    run, rec_dir, _ = trained
    drive = str(tmp_path / 'far')
    shutil.copytree(run, drive)
    k = 2
    rp = os.path.join(drive, f'range/cam-front/{k:06d}.png')
    rng = _png(rp)
    actor = _png(os.path.join(drive, f'instance/cam-front/{k:06d}.png')) == 5
    assert actor.sum() > 10
    rng[actor] = 250 * 256                          # 250 m: behind every wall of the street
    cv2.imwrite(rp, rng)
    before = {kind: _png(os.path.join(drive, f'{kind}/cam-front/{k:06d}.png')).copy()
              for kind in ('semantic', 'instance', 'range')}
    res = hybrid_run(rec_dir, drive, str(tmp_path / 'h'), method='depth', device='cpu')
    out = res['run']
    rows = {x['step']: x for x in _rows(os.path.join(out, 'hybrid.jsonl'))}
    assert res['method'] == 'depth' and rows[k]['hidden_px'] >= 0.9 * actor.sum()
    inst = _png(os.path.join(out, f'instance/cam-front/{k:06d}.png'))
    sem = _png(os.path.join(out, f'semantic/cam-front/{k:06d}.png'))
    r2 = _png(os.path.join(out, f'range/cam-front/{k:06d}.png'))
    hid = actor & (inst == 0)
    assert hid.sum() >= 0.9 * actor.sum() and (sem[hid] != 14).mean() > 0.9 and (r2[hid] < 100 * 256).mean() > 0.9
    for kind, arr in before.items():                # the source's labels are untouched
        assert np.array_equal(_png(os.path.join(drive, f'{kind}/cam-front/{k:06d}.png')), arr)
    assert not os.path.samefile(os.path.join(drive, f'instance/cam-front/{k:06d}.png'),
                                os.path.join(out, f'instance/cam-front/{k:06d}.png'))
    hyb = cv2.imread(os.path.join(out, f'rgb/cam-front/{k:06d}.png'))
    src = cv2.imread(os.path.join(drive, f'rgb/cam-front/{k:06d}.png'))
    assert np.abs(hyb[hid].astype(int) - src[hid].astype(int)).mean() > 10      # the street, not the red actor
    s = res['summary']['all']
    assert s['hidden_share'] > 0 and s['alternative']['mask']['actors']['identical']
    # frames the test changed nothing in keep their hard links
    keep = [x['step'] for x in rows.values() if x['hidden_px'] == 0]
    assert keep and all(os.path.samefile(os.path.join(drive, f'instance/cam-front/{j:06d}.png'),
                                         os.path.join(out, f'instance/cam-front/{j:06d}.png')) for j in keep)


def test_compositor_rules_and_refusals(trained, tmp_path):
    from falcon_autonomy.core.recon.hybrid import Compositor, HybridError, hybrid_run
    from falcon_autonomy.core.recon.scene import Recon
    run, rec_dir, _ = trained
    rec = Recon.load(rec_dir, 'cpu')
    c = Compositor(rec)
    assert c.exact and c.method == 'mask'
    a = torch.tensor([[True, True, True, False]])
    ra = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
    rs = torch.tensor([[10.0, 4.9, 0.0, 1.0]])
    c2 = Compositor(rec, method='depth', tolerance=0.05)
    assert c2.win(a, ra, rs).tolist() == [[True, True, True, False]]     # 5 ≤ 4.9 × 1.05; nothing there: CARLA's
    assert Compositor(rec, method='depth', tolerance=0.0).win(a, ra, rs).tolist() == [[True, False, True, False]]
    assert c.win(a, ra, rs).tolist() == a.tolist()
    with pytest.raises(HybridError, match='range labels'):
        c2.win(a, None, rs)
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 0.0]
    ca = Compositor(rec, align=T)
    assert not ca.exact and ca.method == 'depth'
    assert np.allclose(ca.view(np.eye(4)), np.linalg.inv(T))
    with pytest.raises(HybridError, match='rigid'):
        Compositor(rec, align=np.diag([2.0, 2.0, 2.0, 1.0]))
    with pytest.raises(HybridError, match='method'):
        Compositor(rec, method='nearest')
    rec.views['source'] = dict(rec.views['source'], kind='posed')
    with pytest.raises(HybridError, match='alignment'):
        Compositor(rec)
    other = make_falcon_run(str(tmp_path / 'town'), n=4)
    dj = os.path.join(other, 'drive.json')
    d = json.load(open(dj))
    d['town'] = 'Town01'
    json.dump(d, open(dj, 'w'))
    with pytest.raises(HybridError, match='same world'):
        hybrid_run(rec_dir, other, str(tmp_path / 'x'), device='cpu')
    assert not os.path.exists(tmp_path / 'x')
    with pytest.raises(HybridError, match='folder of its own'):
        hybrid_run(rec_dir, run, run, device='cpu')


def test_compositor_matches_the_reconstruction_off_actor_pixels(trained):
    """frame(): where there is no actor the hybrid is the reconstruction's own render at the camera."""
    from falcon_autonomy.core.recon import data as D
    from falcon_autonomy.core.recon.hybrid import Compositor
    from falcon_autonomy.core.recon.scene import Recon
    run, rec_dir, _ = trained
    rec = Recon.load(rec_dir, 'cpu')
    f = D.falcon_capture(run).frames[1]
    rgb = D.load_rgb(f.image)
    actor = ~D.load_mask(f)
    out = Compositor(rec).frame(rgb, actor, D.load_range(f), f.viewmat, f.K)
    alone = rec.render({'viewmat': f.viewmat, 'K': f.K, 'size': f.size}, semantic=False)['image']
    assert np.array_equal(out['image'][~actor], alone[~actor]) and np.array_equal(out['image'][actor], rgb[actor])
    assert out['render_ms'] > 0 and out['composite_ms'] >= 0


def test_redrive_log_moves_the_ego_right_and_drops_the_actors(trained):
    from falcon_autonomy.core.logreplay import import_log
    from falcon_autonomy.core.recon.hybrid import redrive_log
    run, _, _ = trained
    src = import_log('falcon', run)
    log = redrive_log(run, 1.5)
    assert log.actors == [] and np.allclose(src.ego.yaw, 0.0)
    assert np.allclose(log.ego.y, src.ego.y + 1.5) and np.allclose(log.ego.x, src.ego.x)    # yaw 0: right is +y
    assert 'hybrid re-drive' in json.dumps(log.meta['notes'])        # (other headings: the end-to-end test below)


def test_cli_hybrid(trained, tmp_path, capsys):
    from falcon_autonomy.cli import main
    run, rec_dir, _ = trained
    out = str(tmp_path / 'h')
    assert main(['recon', 'hybrid', rec_dir, '--drive', run, '--out', out, '--device', 'cpu', '--camera',
                 'cam-front', '--no-measure']) == 0
    rj = json.load(open(os.path.join(out, 'run.json')))
    assert rj['hybrid']['measured'] is False and 'not asked' in rj['hybrid']['why_not_measured']
    assert main(['recon', 'hybrid', rec_dir, '--drive', run, '--out', out, '--device', 'cpu']) == 2   # not empty
    assert 'not empty' in capsys.readouterr().out


# ================================================================================================ the re-drive, fake carla
def _project(path):
    from falcon_autonomy.core.project import Project
    p = Project.new('Hybrid')
    p.doc['rigs'][0]['sensors'][0].update(width=96, height=54, labels=['semantic', 'instance', 'depth', 'boxes2d'])
    p.save(str(path))
    return p


def test_redrive_end_to_end_on_the_fake(tmp_path):
    """A drive on the fake carla, reconstructed (briefly, on the CPU), re-driven with traffic and the ego 1 m to
    the right through the replay worker, composited: the config, the log, the CARLA run and the hybrid run."""
    from conftest import worker_env
    from falcon_autonomy.core import protocol as P
    from falcon_autonomy.core.drive import drive_config
    from falcon_autonomy.core.recon.hybrid import HybridError, hybrid_drive_config, hybrid_run
    from falcon_autonomy.core.recon.scene import Recon
    from falcon_autonomy.core.recon.train import train
    proj = _project(tmp_path / 'p.falcon.json')
    cfg = drive_config(proj, 'world-town10', 'rig-default', str(tmp_path / 'src'), 2310, 8310, seconds=1.2, fps=10.0,
                       seed=3)
    cfg['traffic'] = dict(cfg['traffic'], vehicles=4, walkers=0)
    os.makedirs(tmp_path / 'src')
    (tmp_path / 'src' / 'drive.json').write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', str(tmp_path / 'src' / 'drive.json')],
                       capture_output=True, text=True, env=worker_env(), timeout=300)
    assert r.returncode == P.OK, r.stderr[-3000:]
    torch.set_num_threads(2)
    train(f'falcon:{tmp_path / "src"}', str(tmp_path / 'rec'),
          _settings(iterations=30, width=96, init='random', sky_points=100, densify_from=1000))
    rec = Recon.load(str(tmp_path / 'rec'), 'cpu')
    hc = hybrid_drive_config(proj, rec, str(tmp_path / 'carla'), 2310, 8310, lateral_m=1.0, vehicles=3)
    assert hc['traffic']['vehicles'] == 3 and hc['hybrid']['seed'] == 4 and hc['replay']['plan']['actors'] == []
    assert hc['replay']['plan']['ego']['driver'] == 'log' and os.path.isfile(tmp_path / 'carla' / 'log.json')
    src0 = _rows(str(tmp_path / 'src' / 'frames.jsonl'))[0]['ego']
    plan = hc['replay']['plan']['ego']
    yaw = np.radians(src0['yaw'])
    assert abs(plan['x'][0] - (src0['x'] - np.sin(yaw))) < 1e-3 and abs(plan['y'][0] - (src0['y'] + np.cos(yaw))) < 1e-3
    (tmp_path / 'carla' / 'drive.json').write_text(json.dumps(hc))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'replay', str(tmp_path / 'carla' / 'drive.json')],
                       capture_output=True, text=True, env=worker_env(), timeout=300)
    assert r.returncode == P.OK, r.stderr[-3000:]
    run = json.loads((tmp_path / 'carla' / 'run.json').read_text())
    assert run['status'] == 'ok' and (run.get('traffic') or {}).get('placed', run.get('traffic')) is not None
    res = hybrid_run(rec, str(tmp_path / 'carla'), str(tmp_path / 'hybrid'))
    assert res['frames'] > 5 and res['summary']['all']['lateral_m'] == pytest.approx(1.0, abs=0.1)
    with pytest.raises(HybridError, match='instance and depth'):
        proj.doc['rigs'][0]['sensors'][0]['labels'] = ['semantic']
        hybrid_drive_config(proj, rec, str(tmp_path / 'c2'), 2310, 8310, rig='rig-default')


# ================================================================================================ the Replay page
from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401


@pytest.mark.gui
def test_replay_page_hybrid_section(window, qapp, trained, tmp_path):
    from falcon_autonomy.core.jobs import DONE, FINAL
    from falcon_autonomy.core.logreplay import import_log
    from falcon_autonomy.ui.pages.replay import ReplayPage
    run, rec_dir, _ = trained
    page = ReplayPage(window)
    window.stack.addWidget(page)
    window.stack.setCurrentWidget(page)
    drive = os.path.join(window.service.runs_dir(), 'drive-a')
    shutil.copytree(run, drive)
    os.makedirs(page.logs_dir(), exist_ok=True)
    import_log('falcon', run).save(os.path.join(page.logs_dir(), 'synthetic'))
    page.refresh()
    page.logs.setCurrentIndex(page.logs.findText('synthetic'))
    panel = page.recon
    shutil.copytree(rec_dir, panel.rec_dir)
    panel.open_recon()
    assert pump(qapp, lambda: panel.rec is not None, 60)
    assert panel.hy_drive.findData(drive) >= 0
    panel.device.setCurrentIndex(panel.device.findData('cpu'))
    panel.hy_drive.setCurrentIndex(panel.hy_drive.findData(drive))
    out = panel.hybrid()
    job = panel.jobs['hybrid']
    assert job.spec.needs_gpu and '--drive' in job.spec.argv
    assert pump(qapp, lambda: job.state in FINAL, 180), job.summary()
    assert job.state == DONE, job.log[-5:]
    assert pump(qapp, lambda: 'dB' in panel.hy_info.vals['Against CARLA'].text(), 30)
    assert os.path.isfile(os.path.join(out, 'hybrid.jsonl')) and 'ms' in panel.hy_info.vals['Cost'].text()
    # a new CARLA drive needs a world of the source's map: the synthetic street has none, and says so
    assert panel.hybrid_drive() is None and 'cannot drive it' in panel.note.text()
