"""`falcon recon train|from-dashcam|place|eval|render|rerender|hybrid|hybrid-drive|info` (docs/reconstruction.md). Protocol lines on stdout
(core.protocol); exit codes 0 ok, 2 failed, 3 cancelled. A command that uses CUDA takes Falcon's GPU lock
(core.training.gpu_locks) unless the job queue holds it for it (FALCON_GPU_LOCK_HELD=1); --device cpu needs no lock.
hybrid-drive records its CARLA drive first (a CARLA job through CarlaService: the CARLA lock, Falcon's own server,
stopped afterwards) and takes the GPU lock only to composite.
"""
from __future__ import annotations

import json
import os


def _pose_args(ap):
    ap.add_argument('--lateral', type=float, default=0.0, help='metres to the right (negative: left)')
    ap.add_argument('--forward', type=float, default=0.0)
    ap.add_argument('--up', type=float, default=0.0)
    ap.add_argument('--yaw', type=float, default=0.0, help='degrees, CARLA\'s sense (positive turns right)')
    ap.add_argument('--mount', help='the camera\'s mount, CARLA vehicle frame: x,y,z,roll,pitch,yaw')
    ap.add_argument('--fov', type=float, help='a centred pinhole of this horizontal field (degrees)')
    ap.add_argument('--size', help='WIDTHxHEIGHT of the rendered image')
    ap.add_argument('--device', default='auto')


def _pose_kw(a) -> dict:
    mount = None
    if a.mount:
        v = [float(x) for x in a.mount.split(',')]
        if len(v) != 6:
            raise ValueError('--mount is x,y,z,roll,pitch,yaw')
        mount = dict(zip(('x', 'y', 'z', 'roll', 'pitch', 'yaw'), v))
    size = None
    if a.size:
        w, _, h = a.size.lower().partition('x')
        size = (int(w), int(h))
    return {'lateral_m': a.lateral, 'forward_m': a.forward, 'up_m': a.up, 'yaw_deg': a.yaw, 'mount': mount,
            'fov': a.fov, 'size': size}


def _hybrid_args(ap):
    ap.add_argument('--method', choices=('auto', 'mask', 'depth'), default='auto',
                    help='auto: CARLA\'s occlusion (mask) for a reconstruction of a CARLA run of the same world, '
                         'else the per-pixel depth test')
    ap.add_argument('--tolerance', type=float, default=0.05, help='the depth test\'s relative slack')
    ap.add_argument('--camera', action='append', help='a camera to composite (default: every pinhole camera)')
    ap.add_argument('--align', help='a JSON 4 × 4: the reconstruction\'s world ← CARLA\'s world (right-handed), '
                                    'for a reconstruction not of a CARLA run')
    ap.add_argument('--no-measure', action='store_true', help='do not measure against CARLA\'s frames')
    ap.add_argument('--device', default='auto')


def _hybrid_kw(a) -> dict:
    align = None
    if a.align:
        from ..provenance import read_json
        doc = read_json(a.align)
        align = doc.get('matrix') if isinstance(doc, dict) else doc
    return {'cameras': a.camera, 'method': a.method, 'tolerance': a.tolerance, 'align': align,
            'measure': not a.no_measure, 'device': a.device}


def hybrid_drive_command(a, em, flag, argv) -> tuple:
    """Record the re-drive in CARLA, stop the server, then composite: (exit code, summary)."""
    from .. import protocol as P
    from ..jobs import JobQueue, SubprocessLauncher
    from ..logreplay.commands import _wait, submit
    from ..project import Project
    from ..settings import Settings
    from ...sim.service import CarlaService
    from .hybrid import hybrid_drive_config, hybrid_run
    from .scene import Recon
    proj = Project.load(a.project)
    rec = Recon.load(a.recon, a.device)
    carla_dir, hyb_dir = os.path.join(a.out, 'carla'), os.path.join(a.out, 'hybrid')
    svc = CarlaService(Settings(), JobQueue(SubprocessLauncher()), proj.workspace)
    try:
        cfg = hybrid_drive_config(proj, rec, carla_dir, svc.port, svc.tm_port, a.world, a.rig, a.lateral, a.seed,
                                  a.vehicles, a.walkers, float(svc.settings.get('render_budget_mpx')),
                                  proj.workspace)
        em.progress(0.0, f'recording the CARLA drive into {carla_dir}', force=True)
        code = _wait(svc, submit(svc, cfg, cfg['name']), em)
    finally:
        svc.shutdown()
    if code != P.OK:
        return code, {'carla': carla_dir, 'error': f'the CARLA drive ended with code {code}'}
    locks = _locks(a.device, em, flag)
    try:
        kw = _hybrid_kw(a)
        kw.pop('device')
        res = hybrid_run(rec, carla_dir, hyb_dir, emitter=em, cancel=flag, argv=['recon'] + list(argv), **kw)
    finally:
        for lk in locks:
            lk.release()
    return P.OK, dict(res, carla=carla_dir)


def _locks(device: str, em, flag):
    from ..training import gpu_locks
    from .scene import pick_device
    if pick_device(device).type != 'cuda':
        return []
    return gpu_locks(em, flag, held_by_queue=os.environ.get('FALCON_GPU_LOCK_HELD') == '1')


def main_recon(argv) -> int:
    import argparse
    from .. import protocol as P
    from ..training import CancelFlag, Cancelled as TCancelled
    ap = argparse.ArgumentParser(prog='falcon recon', description='neural reconstruction of a recorded drive '
                                                                  '(3D Gaussian Splatting; docs/reconstruction.md)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train', help='reconstruct a drive from its posed images')
    t.add_argument('--source', required=True, metavar='KIND:PATH[?k=v]',
                   help='falcon:RUN, kitti_raw:DRIVE or nuscenes:ROOT?scene=NAME (a bare path is detected)')
    t.add_argument('--out', '-o', required=True)
    t.add_argument('--config', help='a JSON file of settings (the flags below override it)')
    for k, typ in (('iterations', int), ('width', int), ('holdout', int), ('seed', int), ('sh-degree', int),
                   ('max-gaussians', int), ('every', int), ('voxel-m', float), ('sky-points', int),
                   ('gpu-memory-fraction', float)):
        t.add_argument(f'--{k}', type=typ)
    t.add_argument('--init', choices=('auto', 'lidar', 'points', 'sfm', 'range', 'random'))
    t.add_argument('--camera')
    t.add_argument('--device')
    t.add_argument('--no-masks', action='store_true', help='learn from every pixel (moving objects included)')
    t.add_argument('--no-semantic', action='store_true', help='do not vote semantic classes')
    dc = sub.add_parser('from-dashcam', help='pose a dashcam segment (visual odometry + its GPS), reconstruct it and '
                                             'place it on an imported world\'s map')
    dc.add_argument('--clip', required=True, help='the dashcam MP4 (with its embedded GPS)')
    dc.add_argument('--start', type=float, default=0.0, help='seconds into the clip')
    dc.add_argument('--seconds', type=float, default=20.0, help='the segment\'s length (10–30 s is the intended use)')
    dc.add_argument('--out', '-o', help='the reconstruction folder (default: '
                                        '~/FalconWorkspace/recon/dashcam-<clip>-<start>s)')
    dc.add_argument('--config', help='a JSON file of settings (core.recon.fromdashcam.DEFAULTS; the flags override it)')
    dc.add_argument('--intrinsics', help="'auto' (plumb-line k1 and a focal search), a JSON file an earlier run wrote "
                                         "(intrinsics.json) or 'f=PX,k1=…,k2=…' in the clip's native pixels")
    dc.add_argument('--project', help='a Falcon project (.falcon.json) whose imported world to place the segment on')
    dc.add_argument('--world', help='the imported world\'s id in that project')
    for k, typ in (('iterations', int), ('width', int), ('seed', int), ('image-every', int), ('vo-width', int)):
        dc.add_argument(f'--{k}', type=typ)
    dc.add_argument('--device')
    dc.add_argument('--no-train', action='store_true', help='pose and place the segment only')
    dc.add_argument('--keep-raw', action='store_true', help='keep the decoded (distorted) frames')
    pl = sub.add_parser('place', help='place a posed dashcam segment on an imported world after the fact')
    pl.add_argument('recon', help='a from-dashcam reconstruction folder')
    pl.add_argument('--project', required=True)
    pl.add_argument('--world', required=True)
    e = sub.add_parser('eval', help='measure a reconstruction: held-out frames, another run beside the path, '
                                    'lateral shifts against depth-warped CARLA references')
    e.add_argument('recon')
    e.add_argument('--reference', action='append', default=[], metavar='falcon:RUN',
                   help='another Falcon run of the same world whose frames lie beside the path')
    e.add_argument('--warp-shift', action='append', type=float, default=[], metavar='M',
                   help='a lateral shift (m) measured against depth-warped CARLA references (both sides)')
    e.add_argument('--warp-check', action='append', default=[], metavar='DIST_M|falcon:RUN',
                   help='measure the warp itself: frames DIST_M ahead on the path, or another run\'s frames')
    e.add_argument('--limit', type=int, default=20, help='frames per measure')
    e.add_argument('--allow-other-world', action='store_true')
    e.add_argument('--device', default='auto')
    r = sub.add_parser('render', help='render one pose: a frame of the drive, shifted')
    r.add_argument('recon')
    r.add_argument('--frame', type=int, default=0, help='the frame\'s position in views.json')
    r.add_argument('--out', '-o', required=True, help='the image (.png or .jpg)')
    r.add_argument('--depth-out', help='the range as a uint16 PNG (range × 256)')
    _pose_args(r)
    rr = sub.add_parser('rerender', help='a falcon-raw run of the drive re-rendered from a shifted ego path or '
                                         'another camera')
    rr.add_argument('recon')
    rr.add_argument('--out', '-o', required=True)
    rr.add_argument('--every', type=int, default=1)
    rr.add_argument('--split', choices=('all', 'train', 'test'), default='all')
    _pose_args(rr)
    hy = sub.add_parser('hybrid', help='a recorded CARLA drive with its actors composited over the reconstruction '
                                       '(a hybrid world; labels CARLA\'s)')
    hy.add_argument('recon')
    hy.add_argument('--drive', required=True, help='the recorded CARLA drive (a falcon-raw run with instance labels)')
    hy.add_argument('--out', '-o', required=True)
    _hybrid_args(hy)
    hd = sub.add_parser('hybrid-drive', help='drive the reconstruction\'s path in CARLA with new traffic (and the '
                                             'ego moved sideways), then composite it')
    hd.add_argument('recon')
    hd.add_argument('--project', required=True)
    hd.add_argument('--out', '-o', required=True, help='a folder: carla/ (CARLA\'s run, the truth) and hybrid/')
    hd.add_argument('--world', help='the project\'s world (default: the one of the source drive\'s town)')
    hd.add_argument('--rig', help='a rig with a pinhole camera labelled instance and depth (default: the first)')
    hd.add_argument('--lateral', type=float, default=0.0, help='metres to the right of the source\'s path')
    hd.add_argument('--seed', type=int, help='the traffic\'s seed (default: the source drive\'s + 1)')
    hd.add_argument('--vehicles', type=int, help='default: the source drive\'s')
    hd.add_argument('--walkers', type=int, help='default: the source drive\'s')
    _hybrid_args(hd)
    i = sub.add_parser('info', help='what a reconstruction is and how it measured')
    i.add_argument('recon')
    try:
        a = ap.parse_args(argv)
    except SystemExit as ex:
        return P.OK if ex.code == 0 else P.FAILED
    if a.cmd == 'info':
        from ..provenance import read_json
        p = os.path.join(a.recon, 'run.json')
        if not os.path.isfile(p):
            print(f'failed: {a.recon} has no run.json', flush=True)
            return P.FAILED
        run = read_json(p)
        print(json.dumps({'status': run.get('status'), 'summary': run.get('summary'), 'timings': run.get('timings'),
                          'device': run.get('device'), 'source': run.get('source')}, indent=1, default=str))
        return P.OK
    em = P.Emitter()
    em.hello(f'recon {a.cmd}')
    flag = CancelFlag().install()
    locks = []
    from .data import DataError
    from .train import Cancelled, ReconError
    from .vo import VOError
    from .hybrid import HybridError
    try:
        if a.cmd == 'train':
            from ..provenance import read_json
            from .train import train
            s = read_json(a.config) if a.config else {}
            over = {'iterations': a.iterations, 'width': a.width, 'holdout': a.holdout, 'seed': a.seed,
                    'sh_degree': a.sh_degree, 'max_gaussians': a.max_gaussians, 'every': a.every,
                    'voxel_m': a.voxel_m, 'sky_points': a.sky_points, 'gpu_memory_fraction': a.gpu_memory_fraction,
                    'init': a.init, 'camera': a.camera, 'device': a.device}
            s.update({k: v for k, v in over.items() if v is not None})
            if a.no_masks:
                s['masks'] = False
            if a.no_semantic:
                s['semantic'] = False
            locks = _locks(s.get('device') or 'auto', em, flag)
            res = train(a.source, a.out, s, em, flag, argv=['recon'] + list(argv))
            em.result('ok', run_json=os.path.join(a.out, 'run.json'), summary=res)
            return P.OK
        if a.cmd == 'from-dashcam':
            from ..provenance import read_json
            from .fromdashcam import default_out, from_dashcam
            s = read_json(a.config) if a.config else {}
            over = {'iterations': a.iterations, 'width': a.width, 'seed': a.seed, 'image_every': a.image_every,
                    'vo_width': a.vo_width, 'device': a.device, 'intrinsics': a.intrinsics, 'project': a.project,
                    'world': a.world}
            s.update({k: v for k, v in over.items() if v is not None})
            if a.no_train:
                s['train'] = False
            if a.keep_raw:
                s['keep_raw'] = True
            out = a.out or default_out(a.clip, a.start)
            if s.get('train', True):
                locks = _locks(s.get('device') or 'auto', em, flag)
            res = from_dashcam(a.clip, out, a.start, a.seconds, s, em, flag, argv=['recon'] + list(argv))
            em.result('ok', report=os.path.join(out, 'dashcam.json'), summary=res)
            return P.OK
        if a.cmd == 'place':
            from .fromdashcam import place_existing
            res = place_existing(a.recon, a.project, a.world)
            em.result('ok', report=os.path.join(a.recon, 'dashcam.json'), summary=res)
            return P.OK
        if a.cmd == 'hybrid-drive':
            code, res = hybrid_drive_command(a, em, flag, argv)
            em.result(P.status_of(code), run=res.get('run'), summary=res, error=res.get('error'))
            return code
        locks = _locks(a.device, em, flag)
        if a.cmd == 'hybrid':
            from .hybrid import hybrid_run
            res = hybrid_run(a.recon, a.drive, a.out, emitter=em, cancel=flag, argv=['recon'] + list(argv),
                             **_hybrid_kw(a))
            em.result('ok', run=res['run'], summary=res)
            return P.OK
        if a.cmd == 'eval':
            res = evaluate_command(a.recon, a.reference, a.warp_shift, a.warp_check, a.limit, a.device,
                                   a.allow_other_world, em)
            em.result('ok', evaluation=os.path.join(a.recon, 'evaluation.json'), summary=res)
            return P.OK
        if a.cmd == 'render':
            import cv2
            from .scene import Recon
            rec = Recon.load(a.recon, a.device)
            cam = rec.camera(rec.frame(a.frame), **_pose_kw(a))
            out = rec.render(cam)
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            cv2.imwrite(a.out, cv2.cvtColor(out['image'], cv2.COLOR_RGB2BGR))
            if a.depth_out:
                import numpy as np
                cv2.imwrite(a.depth_out, np.where(out['range'] < 256, np.round(out['range'] * 256), 0)
                            .astype(np.uint16))
            em.result('ok', image=a.out, depth=a.depth_out)
            return P.OK
        from .rerender import rerender
        res = rerender(a.recon, a.out, every=a.every, split=a.split, emitter=em, cancel=flag,
                       argv=['recon'] + list(argv), device=a.device, **_pose_kw(a))
        em.result('ok', run=a.out, summary=res)
        return P.OK
    except (Cancelled, TCancelled, KeyboardInterrupt):
        em.result('cancelled')
        return P.CANCELLED
    except (DataError, ReconError, VOError, HybridError, ValueError, OSError, KeyError) as ex:
        print(f'failed: {ex}', flush=True)
        em.result('failed', error=str(ex))
        return P.FAILED
    finally:
        for lk in locks:
            lk.release()


def evaluate_command(recon_dir: str, references=(), shifts=(), checks=(), limit: int = 20, device: str = 'auto',
                     allow_other_world: bool = False, em=None) -> dict:
    """Run the requested measures and merge them into the reconstruction's evaluation.json (key 'novel_view')."""
    import time
    from .. import provenance as PV
    from . import evaluate as EV
    from .scene import Recon
    rec = Recon.load(recon_dir, device)
    p = os.path.join(rec.path, 'evaluation.json')
    doc = PV.read_json(p) if os.path.isfile(p) else {'schema': 'falcon.recon_evaluation/1'}
    nv = doc.setdefault('novel_view', {})
    summary = {}
    t0 = time.perf_counter()
    for ref in references:
        if em:
            em.progress(0.0, f'against {ref}', force=True)
        res = EV.against_run(rec, ref, limit=max(limit, 1) * 2, out_dir=rec.path,
                             allow_other_world=allow_other_world)
        nv.setdefault('reference_runs', {})[os.path.basename(os.path.normpath(ref.partition('?')[0]))] = res
        summary[f'reference {os.path.basename(os.path.normpath(ref))}'] = res.get('by_lateral') or res.get('note')
    if shifts:
        if em:
            em.progress(0.3, f'lateral shifts {list(shifts)} m against depth-warped references', force=True)
        res = EV.lateral_shift(rec, tuple(shifts), limit, rec.path)
        nv['lateral_warped'] = res
        summary['lateral_warped'] = [{k: s.get(k) for k in ('lateral_m', 'psnr', 'ssim', 'coverage',
                                                             'depth_median_abs_rel')} for s in res['shifts']]
    for c in checks:
        if em:
            em.progress(0.7, f'checking the warp: {c}', force=True)
        from . import data as D
        cap = rec.capture()
        try:
            dist = float(c)
            res = EV.warp_check(cap, rec.work_size, (dist,), limit, rec.device)
        except ValueError:
            res = EV.warp_check(cap, rec.work_size, limit=limit, dev=rec.device, targets=D.open_capture(c))
        nv.setdefault('warp_checks', []).extend(res)
        summary.setdefault('warp_checks', []).extend({k: r.get(k) for k in ('what', 'distance_m', 'psnr', 'ssim',
                                                                            'coverage')} for r in res)
    nv['seconds'] = round(time.perf_counter() - t0, 2)
    nv['provenance'] = PV.block({'run.json': os.path.join(rec.path, 'run.json')})
    doc['novel_view'] = nv
    PV.write_json(p, doc)
    return summary
