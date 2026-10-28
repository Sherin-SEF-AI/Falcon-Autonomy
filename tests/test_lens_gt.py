"""core.lens and core.groundtruth (ported from Vantage's simrig and groundtruth): the lens models, a pinhole sensor
rendered as itself, a fisheye's views (their CARLA rotations checked by rendering a direction field the way CARLA
renders, and end to end against the fake carla), the remap reproducing the lens model, the seams feathered, the
vignette taken out; the class table against CARLA 0.9.16's labels and nuScenes-lidarseg's categories, the projection,
boxes, COCO run-length encoding, the camera labels read through the remap tables (nearest view pixel, range along the
output ray), objects and their geometry on a scene ray-cast independently, the traffic lights' and signs' boxes
claiming their pixels, and the reader."""
import json
import math
import os

import numpy as np
import pytest

from falcon_autonomy.core import budget
from falcon_autonomy.core import groundtruth as GT
from falcon_autonomy.core import lens as L
from falcon_autonomy.sim.frames import CARLA_TO_OPTICAL, carla_matrix

# carla.CityObjectLabel of the real carla 0.9.16 client module (printed 2026-10-27): name -> value
CARLA_LABELS = {'NONE': 0, 'Roads': 1, 'Sidewalks': 2, 'Buildings': 3, 'Walls': 4, 'Fences': 5, 'Poles': 6,
                'TrafficLight': 7, 'TrafficSigns': 8, 'Vegetation': 9, 'Terrain': 10, 'Sky': 11, 'Pedestrians': 12,
                'Rider': 13, 'Car': 14, 'Truck': 15, 'Bus': 16, 'Train': 17, 'Motorcycle': 18, 'Bicycle': 19,
                'Static': 20, 'Dynamic': 21, 'Other': 22, 'Water': 23, 'RoadLines': 24, 'Ground': 25, 'Bridge': 26,
                'RailTrack': 27, 'GuardRail': 28}
# ColorConverter.CityScapesPalette's colours read off a 0.9.16 server's semantic image (RGB)
MEASURED_PALETTE = {1: (128, 64, 128), 2: (244, 35, 232), 3: (70, 70, 70), 6: (153, 153, 153), 7: (250, 170, 30),
                    8: (220, 220, 0), 9: (107, 142, 35), 11: (70, 130, 180), 12: (220, 20, 60), 13: (255, 0, 0),
                    14: (0, 0, 142), 18: (0, 0, 230), 19: (119, 11, 32), 20: (110, 190, 160), 21: (170, 120, 50),
                    22: (55, 90, 80), 24: (157, 234, 50), 25: (81, 0, 81)}
CONTRACT = ('unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign',
            'vegetation', 'terrain', 'sky', 'pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
            'bicycle', 'static', 'dynamic', 'other', 'water', 'road_line', 'ground', 'bridge', 'rail_track',
            'guard_rail')
POSE = {'x': 2.0, 'y': 0.0, 'z': 1.0, 'roll': 1.5, 'pitch': -4.0, 'yaw': 2.0}
# a 190° fisheye whose image circle covers the image's width (the corners cut), all label cameras
FISHEYE = {'id': 'cam-f', 'name': 'front_fisheye', 'type': 'camera.fisheye', 'width': 288, 'height': 224,
           'labels': ['semantic', 'instance', 'depth'], 'pose': POSE,
           'lens': {'model': 'equidistant', 'fov': 190.0, 'f_px': 100.0, 'k': [0.0, 0.0, 0.0, 0.0]}}
PINHOLE = {'id': 'cam-p', 'name': 'front', 'type': 'camera.pinhole', 'width': 160, 'height': 90, 'fov': 90.0,
           'labels': ['semantic', 'depth'], 'pose': POSE}


def _fisheye(**lens):
    return dict(FISHEYE, lens=dict(FISHEYE['lens'], **lens))


# ======================================================================= lens models
def test_lens_models_invert_and_resolve():
    t = np.linspace(0.0, 1.5, 200)
    kb = (0.01, -0.002, 0.0003, -0.00002)
    for m in L.MODELS:
        r = L.r_of_theta(t, 2.1, m, kb)
        assert np.allclose(L.theta_of_r(r, 2.1, m, kb), t, atol=1e-9), m
        h = 1e-6                                                    # the derivative is r's
        assert np.allclose(L.dr_dtheta(t[1:-1], 2.1, m, kb),
                           (L.r_of_theta(t[1:-1] + h, 2.1, m, kb) - L.r_of_theta(t[1:-1] - h, 2.1, m, kb)) / (2 * h),
                           rtol=1e-5), m
    i = L.intrinsics(PINHOLE)
    assert i.model == 'rectilinear' and i.f_px == pytest.approx(80.0) and (i.cx, i.cy) == (80.0, 45.0)
    i = L.intrinsics(_fisheye(model='KB', k=[0.1], cx=150.0))
    assert i.model == 'kannala_brandt' and i.k == (0.1, 0.0, 0.0, 0.0) and i.cx == 150.0 and i.cy == 112.0
    assert i.max_theta == pytest.approx(math.radians(95.0)) and not i.f_heuristic
    # no focal length: the heuristic equidistant lens whose circle fits the shorter side (core.budget's), said
    i = L.intrinsics(_fisheye(f_px=None))
    assert i.f_heuristic and i.f_px == pytest.approx(112.0 / math.radians(95.0))
    assert L.plan_camera(_fisheye(f_px=None), tables=False)['notes']
    assert L.intrinsics(_fisheye(model='orthographic', fov=200.0)).max_theta == pytest.approx(math.pi / 2)
    with pytest.raises(ValueError):
        L.intrinsics(_fisheye(model='fancy'))
    with pytest.raises(ValueError):
        L.intrinsics({'type': 'lidar', 'width': 1, 'height': 1})


def test_projection_is_the_lens_model():
    """groundtruth.project of a plan's lens block = Intrinsics.project_dirs, for every model; the output rays project
    back to their own pixel centres (pixel i covers [i, i+1))."""
    for model, fov, k in (('equidistant', 190.0, ()), ('equisolid', 190.0, ()), ('stereographic', 200.0, ()),
                          ('orthographic', 160.0, ()), ('kannala_brandt', 190.0, (0.01, -0.002, 0.0003, -0.00002)),
                          ('rectilinear', 120.0, ())):
        s = _fisheye(model=model, fov=fov, k=list(k), cx=146.3, cy=109.1)
        intr = L.intrinsics(s)
        lens = L.plan_camera(s, tables=False)['lens']
        D = np.random.default_rng(4).normal(size=(500, 3))
        D[:, 2] = np.abs(D[:, 2])
        u, v, th, ok = intr.project_dirs(D)
        u2, v2, th2, ok2 = GT.project(lens, D * 3.7)                 # points along the rays: the same pixels
        assert np.allclose(u2, u) and np.allclose(v2, v), model
        assert np.array_equal(ok2, (th <= intr.max_theta) & ((D[:, 2] > 1e-6) if model == 'rectilinear' else True))
        d, okr = L.output_rays(intr)
        uu, vv, _, _ = GT.project(lens, d[okr])
        ys, xs = np.nonzero(okr)
        assert okr.mean() > 0.4 and np.abs(uu - (xs + 0.5)).max() < 1e-6 and np.abs(vv - (ys + 0.5)).max() < 1e-6
    K = np.array([[200.0, 0, 99.5], [0, 200.0, 79.5], [0, 0, 1]])
    u, v, _, ok = GT.project({'model': 'pinhole', 'K': K.tolist(), 'max_theta_deg': 80.0}, [[0.1, -0.2, 1.0],
                                                                                             [0, 0, -1.0]])
    assert u[0] == pytest.approx(200 * 0.1 + 99.5 + 0.5) and v[0] == pytest.approx(-40 + 79.5 + 0.5) and not ok[1]


# ======================================================================= the plan
def test_pinhole_is_one_view_the_camera_itself_with_the_identity_remap():
    p = L.plan_camera(PINHOLE)
    assert len(p['views']) == 1
    v = p['views'][0]
    assert v['name'] == 'front' and (v['width'], v['height'], v['fov']) == (160, 90, 90.0)
    assert v['rotation'] == {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
    assert all(v['pose'][k] == pytest.approx(POSE[k], abs=1e-9) for k in POSE)
    assert v['attributes'] == {'image_size_x': 160, 'image_size_y': 90, 'fov': 90.0}
    mx, my = p['remap']['map_x'], p['remap']['map_y']
    assert np.array_equal(mx, np.tile(np.arange(160, dtype=np.float32), (90, 1)))
    assert np.array_equal(my, np.tile(np.arange(90, dtype=np.float32)[:, None], (1, 160)))
    assert p['remap']['blend'] is None and p['atlas'] == {'width': 160, 'height': 90, 'offsets': [0]}
    assert p['render_mpx'] == pytest.approx(160 * 90 / 1e6)
    assert L.camera_mpx(PINHOLE, p) == pytest.approx(budget.sensor_mpx(PINHOLE))       # core.budget's figure
    assert p['K'] == [[pytest.approx(80.0), 0.0, 80.0], [0.0, pytest.approx(80.0), 45.0], [0.0, 0.0, 1.0]]
    json.dumps(p['summary'])
    img = np.random.default_rng(0).integers(0, 256, (90, 160, 4), dtype=np.uint8)
    assert np.array_equal(L.remap_image(p, [img]), img[..., :3])
    cg = GT.CameraGT(p)
    assert cg.ok.all() and np.array_equal(cg.gather(np.arange(90 * 160).reshape(90, 160)),
                                          np.arange(90 * 160).reshape(90, 160))
    d, _ = L.output_rays(L.intrinsics(PINHOLE))
    assert np.abs(cg.rays - d).max() < 1e-6


def test_view_rotations_and_poses_in_carlas_frame():
    """A face's CARLA rotation relative to the sensor (right: yaw +90 — CARLA's y is to the right; up: pitch +90) and
    its pose in the vehicle frame: the CARLA camera's forward axis is the face's optical z, its right the face's x, its
    up the face's −y."""
    want = {'front': (0, 0, 0), 'right': (0, 0, 90), 'left': (0, 0, -90), 'up': (0, 90, 0), 'down': (0, -90, 0)}
    for name, (r, p, y) in want.items():
        got = L.view_rotation(L.FACES[name])
        M = carla_matrix(got['pitch'], got['yaw'], got['roll'])
        assert np.allclose(M, carla_matrix(p, y, r), atol=1e-9), name
    got = L.view_rotation(L.FACES['back'])
    assert np.allclose(carla_matrix(got['pitch'], got['yaw'], got['roll']), carla_matrix(0, 180, 0), atol=1e-9)
    O2V = L.optical_to_vehicle(POSE)
    for name, F in L.FACES.items():
        q = L.view_pose(POSE, F)
        M = carla_matrix(q['pitch'], q['yaw'], q['roll'])
        assert np.allclose(M[:, 0], O2V @ F[:, 2]) and np.allclose(M[:, 1], O2V @ F[:, 0])
        assert np.allclose(M[:, 2], -(O2V @ F[:, 1])) and (q['x'], q['y'], q['z']) == (2.0, 0.0, 1.0), name


def _render_views(plan):
    """Each view's image of the direction field in the sensor's optical frame, rendered as CARLA would: the view's pose
    (carla.Rotation in the vehicle frame), CARLA's pinhole (f = W / 2 tan(fov/2), centre at the image centre), its
    actor axes (x forward, y right, z up) — independent of the plan's own 'optical' matrices."""
    W2O = (carla_matrix(POSE['pitch'], POSE['yaw'], POSE['roll']) @ CARLA_TO_OPTICAL.T).T   # vehicle -> optical
    imgs = []
    for v in plan['views']:
        Wd, Hd = v['width'], v['height']
        f = Wd / (2.0 * math.tan(math.radians(v['fov']) / 2))
        u, vv = np.meshgrid(np.arange(Wd, dtype=float), np.arange(Hd, dtype=float))
        d = np.stack([np.full_like(u, f), u - (Wd / 2 - 0.5), -(vv - (Hd / 2 - 0.5))], -1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        q = v['pose']
        imgs.append((d @ carla_matrix(q['pitch'], q['yaw'], q['roll']).T @ W2O.T).astype(np.float32))
    return imgs


@pytest.mark.parametrize('lens', [{}, {'model': 'kannala_brandt', 'k': [0.02, -0.004, 0.0, 0.0], 'cx': 140.0}])
def test_fisheye_remap_reproduces_the_lens_model(lens):
    """The remapped image of a direction field equals the lens model's projection: at every pixel centre, and at random
    directions projected with Intrinsics.project_dirs (sub-pixel)."""
    import cv2
    s = _fisheye(**lens)
    p = L.plan_camera(s)
    assert {x['name'] for x in p['views']} == {'front', 'left', 'right', 'up', 'down'}
    intr = L.intrinsics(s)
    atlas = L.atlas_of(p, _render_views(p))
    out = cv2.remap(atlas, p['remap']['map_x'], p['remap']['map_y'], cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    d, ok = L.output_rays(intr)
    assert np.array_equal(ok, p['remap']['map_x'] >= 0) and ok.mean() > 0.6
    got = out[ok] / np.linalg.norm(out[ok], axis=1, keepdims=True)
    err_px = np.arccos(np.clip(np.sum(got * d[ok], 1), -1, 1)) * intr.f_px
    # cv2.remap reads at 1/32 px steps (its fixed-point maps): about 0.03 px, plus bilinear interpolation
    assert np.percentile(err_px, 99) < 0.1 and err_px.max() < 0.15, (np.percentile(err_px, 99), err_px.max())
    rng = np.random.default_rng(11)
    D = rng.normal(size=(4000, 3))
    D[:, 2] = np.abs(D[:, 2]) + 0.05
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    u, vv, th, inimg = intr.project_dirs(D)
    sel = inimg & (u > 4) & (vv > 4) & (u < intr.width - 4) & (vv < intr.height - 4) & (th < intr.max_theta - 0.02)
    samp = cv2.remap(out, (u[sel] - 0.5).astype(np.float32).reshape(-1, 1),
                     (vv[sel] - 0.5).astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).reshape(-1, 3)
    samp /= np.linalg.norm(samp, axis=1, keepdims=True)
    e = np.arccos(np.clip(np.sum(samp * D[sel], 1), -1, 1)) * intr.f_px
    assert sel.sum() > 1000 and np.percentile(e, 99) < 0.1, np.percentile(e, 99)


def test_cube_faces_their_size_cost_and_feathering_across_seams():
    """Faces of N = 2·⌈f·tan(46.5°) + 1.5⌉ reaching half the band past each seam; the cost from the views planned; near
    a seam a pixel mixes both faces (smoothstep weights summing to 1, the first face at least a third), every face's
    sample the same direction, so the feathered image still reproduces the lens model; a step between two views (CARLA
    renders the views of a moving camera slightly differently) fades over the band instead of showing as a line."""
    import cv2
    p = L.plan_camera(FISHEYE)
    assert p['seam_blend_deg'] == L.SEAM_BLEND_DEG == 3.0
    t = math.tan(math.radians(45 + L.SEAM_BLEND_DEG / 2))
    for x in p['views']:
        assert x['width'] == x['height'] == L.face_px(100.0) == 2 * math.ceil(100.0 * math.tan(math.radians(46.5)) + 1.5)
        assert x['width'] / 2 - L.FACE_MARGIN_PX >= x['f_px'] * t - 1e-6
        assert x['attributes']['lens_k'] == 0.0 and x['attributes']['exposure_mode'] == 'manual'
    assert p['render_mpx'] == pytest.approx(5 * L.face_px(100.0) ** 2 / 1e6)
    assert L.camera_mpx(FISHEYE, p) == pytest.approx(3 * p['render_mpx'])        # image + instance + depth
    blend = L.blend_tuple(p)
    idx, bx, by, bw = blend
    assert len(idx) > 500 and np.allclose(bw.sum(1), 1, atol=1e-6) and bw[:, 0].min() > 1 / 3 - 1e-6
    assert np.all((bx[:, 0] >= 0) & (bw[:, 1] > 0)) and np.all((bx[:, 1] < 0) == (bw[:, 2] == 0))
    intr = L.intrinsics(FISHEYE)
    atlas = L.atlas_of(p, _render_views(p))
    mx, my = p['remap']['map_x'], p['remap']['map_y']
    first = cv2.remap(atlas, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    d, ok = L.output_rays(intr)
    for j in range(2):                                      # each other face's sample: the pixel's own direction
        has = bx[:, j] >= 0
        smp = L._sample(atlas, bx[has, j], by[has, j])
        smp /= np.linalg.norm(smp, axis=1, keepdims=True)
        e = np.arccos(np.clip(np.sum(smp * d.reshape(-1, 3)[idx[has]], 1), -1, 1)) * intr.f_px
        assert has.sum() > 0 and np.percentile(e, 99) < 0.1 and e.max() < 0.15, (j, e.max())
    out = L.feather(first.copy(), atlas, blend)
    got = out[ok] / np.linalg.norm(out[ok], axis=1, keepdims=True)
    e = np.arccos(np.clip(np.sum(got * d[ok], 1), -1, 1)) * intr.f_px
    assert np.percentile(e, 99) < 0.1 and e.max() < 0.15
    # a step of 40 levels between the views: one line without feathering, a ramp with it (the band is 3° wide: about
    # 5 px at 100 px/rad, a smoothstep's steepest slope 1.5× its mean)
    at = p['atlas']
    flat = np.zeros((at['height'], at['width'], 3), np.uint8)
    for k, x in enumerate(p['views']):
        flat[at['offsets'][k]:at['offsets'][k] + x['height'], :x['width']] = 100 + 40 * (k % 2)
    hard = cv2.remap(flat, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    soft = L.feather(hard.copy(), flat, blend)
    inner = cv2.erode(ok.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    jump = lambda img: max(np.abs(np.diff(img[..., 0].astype(int), axis=a))[(inner[:-1] if a == 0 else inner[:, :-1])
                                                                          ].max() for a in (0, 1))
    band_px = math.radians(L.SEAM_BLEND_DEG) * intr.f_px
    assert jump(hard) == 40 and jump(soft) <= 1.5 * 40 / band_px + 2, (jump(hard), jump(soft))


def test_single_view_for_a_narrow_lens_and_six_faces_for_a_270_degree_one():
    s = _fisheye(fov=100.0, f_px=150.0)
    p = L.plan_camera(s)
    assert [v['name'] for v in p['views']] == ['front'] and p['remap']['blend'] is None
    v = p['views'][0]
    mx, my = p['remap']['map_x'], p['remap']['map_y']
    seen = mx >= 0
    assert seen.mean() > 0.6                                # every ray inside its view, the margin kept
    assert mx[seen].min() >= L.FACE_MARGIN_PX - 0.5 and mx[seen].max() <= v['width'] - 0.5 - L.FACE_MARGIN_PX
    assert my[seen].min() >= L.FACE_MARGIN_PX - 0.5 and my[seen].max() <= v['height'] - 0.5 - L.FACE_MARGIN_PX
    assert v['f_px'] == pytest.approx(150.0) and v['fov'] == pytest.approx(math.degrees(2 * math.atan(v['width'] / 300.0)))
    wide = dict(_fisheye(fov=270.0, f_px=None), width=96, height=96)
    pw = L.plan_camera(wide)
    assert {x['name'] for x in pw['views']} == set(L.FACES)
    assert pw['summary']['coverage'] == pytest.approx(math.pi / 4, abs=0.02)        # the circle fits the square


def test_vignette_taken_out_through_the_measured_response():
    """tonemap: UE's vignette mask (its formula, the corners at √2 for any aspect), the mask each output pixel reads,
    and the correction: an image darkened in linear light by the mask (through the response) comes back within a
    level; and a fisheye's views, each darkened towards its own edges, come out flat through devignette_layers."""
    from falcon_autonomy.sim import tonemap as TM
    assert TM.vignette_mask(1.0) == pytest.approx(1 / 1.16 ** 2) and TM.vignette_mask(0.0) == 1.0
    for w, h in ((640, 640), (960, 540), (300, 900)):
        r = TM.vignette_radius(np.array([-0.5, w - 0.5, (w - 1) / 2]), np.array([-0.5, h - 0.5, (h - 1) / 2]), w, h)
        assert r[0] == pytest.approx(math.sqrt(2)) and r[1] == pytest.approx(math.sqrt(2)) and r[2] == pytest.approx(0)
    g = TM.RESPONSE_LN
    assert g.shape == (3, 256) and np.all(np.diff(g, axis=1) > 0)
    sizes = [(64, 48), (32, 32)]
    mx = np.array([[31.5, -0.5, 15.5, -1.0]], np.float32)
    my = np.array([[23.5, -0.5, 48 + 15.5, -1.0]], np.float32)
    m = TM.mask_of_remap(mx, my, [0, 48], sizes)
    assert m[0, 0] == pytest.approx(1.0) and m[0, 1] == pytest.approx(TM.vignette_mask(math.sqrt(2)), rel=1e-5)
    assert m[0, 2] == pytest.approx(1.0) and m[0, 3] == 1.0
    z = np.tile(np.arange(5, 251, dtype=float), (3, 1)).T[None]                # (1, 246, 3)
    mask = np.linspace(TM.M_LO * 1.001, 1.0, z.shape[1])[None]
    dark = np.empty_like(z)
    for c in range(3):
        dark[..., c] = np.interp(np.interp(z[..., c], np.arange(256), g[c]) + np.log(mask), g[c], np.arange(256))
    back = TM.Devignette(mask).apply(np.rint(dark).astype(np.uint8))
    keep = dark.min(2) >= 1.0
    assert keep.mean() > 0.9 and np.abs(back.astype(float) - z)[keep].max() <= 1.5
    lut = TM.correction_lut()
    assert np.array_equal(lut[-1, 0], np.arange(256)) and np.all(lut[:, :, 0] == 0) and np.all(lut[:, :, 255] == 255)
    # a uniform scene (level 160) seen by every view of a fisheye, each view vignetted as Unreal does
    p = L.plan_camera(FISHEYE)
    views = []
    for v in p['views']:
        u, vv = np.meshgrid(np.arange(v['width'], dtype=float), np.arange(v['height'], dtype=float))
        mv = TM.vignette_mask(TM.vignette_radius(u, vv, v['width'], v['height']))
        lv = np.stack([np.interp(np.interp(160.0, np.arange(256), g[c]) + np.log(mv), g[c], np.arange(256))
                       for c in range(3)], -1)
        views.append(np.rint(lv).astype(np.uint8))
    ok = p['remap']['map_x'] >= 0
    raw = L.remap_image(p, views)
    flat = L.remap_image(p, views, L.devignette_layers(p))
    assert len(L.devignette_layers(p)) == 3                   # the first face's, then the two blend samples'
    assert raw[ok].min() < 140 and np.abs(flat[ok].astype(int) - 160).max() <= 2, (raw[ok].min(), flat[ok].min())


# ======================================================================= classes
def test_class_table_is_carlas_labels_with_their_nuscenes_categories():
    from falcon_autonomy.core import labels
    doc = GT.classes_doc()
    names = [c['name'] for c in doc]
    assert set(CONTRACT) <= set(names) and names[:29] == list(CONTRACT) and doc[29]['name'] == 'ego'
    for c in doc[:29]:                                      # the class id is CARLA's own tag value
        assert CARLA_LABELS[c['carla_label']] == c['id'] == c['carla_tags'][0]
        assert labels.CARLA_NAMES[c['id']] == c['carla_label'] and tuple(c['color']) == labels.color(c['id'])
        assert c['nuscenes'] in GT.NUSC_INDEX and c['nuscenes_index'] == GT.NUSC_INDEX[c['nuscenes']]
    pal = GT.palette()
    for tag, rgb in MEASURED_PALETTE.items():
        assert tuple(pal[tag]) == rgb, tag
    assert np.array_equal(GT.CLASS_OF_TAG[:29], np.arange(29)) and GT.CLASS_OF_TAG[200] == 0
    assert [i for i, _, _ in GT.NUSCENES_LIDARSEG] == list(range(32))
    assert GT.NUSC_NAME[0] == 'noise' and GT.NUSC_NAME[17] == 'vehicle.car' and GT.NUSC_NAME[31] == 'vehicle.ego'
    assert GT.NUSC_NAME[24] == 'flat.driveable_surface' and GT.NUSC_NAME[28] == 'static.manmade'
    assert GT.NUSC_OF_TAG[14] == 17 and GT.NUSC_OF_TAG[1] == GT.NUSC_OF_TAG[24] == 24 and GT.NUSC_OF_TAG[11] == 0
    rows = GT.nuscenes_categories(lambda n: 'tok:' + n, [('vehicle.car', 'x'), ('my.extra', 'y')])
    assert [r['index'] for r in rows] == list(range(33)) and rows[-1]['name'] == 'my.extra'
    assert len({r['name'] for r in rows}) == 33
    assert GT.SCHEMA == 'falcon.ground_truth/1' and GT.EGO_CLASS == 29 and GT.RANGE_SCALE == 256.0


def test_actor_categories():
    assert GT.category_of('walker.pedestrian.0009', {'age': 'child'}) == 'human.pedestrian.child'
    assert GT.category_of('walker.pedestrian.0001', {'age': 'adult'}) == 'human.pedestrian.adult'
    assert GT.category_of('vehicle.ford.ambulance', {'base_type': 'van'}) == 'vehicle.emergency.ambulance'
    assert GT.category_of('vehicle.diamondback.century', {'base_type': 'bicycle'}) == 'vehicle.bicycle'
    assert GT.category_of('vehicle.yamaha.yzf', {'base_type': 'motorcycle'}) == 'vehicle.motorcycle'
    assert GT.category_of('vehicle.carlamotors.carlacola', {'base_type': 'truck'}) == 'vehicle.truck'
    assert GT.category_of('vehicle.mitsubishi.fusorosa', {'base_type': 'Bus'}) == 'vehicle.bus.rigid'
    assert GT.category_of('static.prop.trafficcone01') == 'movable_object.trafficcone'
    assert GT.category_of('static.prop.streetbarrier') == 'movable_object.barrier'
    assert GT.category_of('vehicle.tesla.model3', {'base_type': 'car'}) == 'vehicle.car'


# ======================================================================= CARLA images
def test_depth_and_instance_decoding():
    rng = np.random.default_rng(1)
    d = rng.uniform(0.5, 900.0, (37, 51))
    img = GT.encode_depth(d)
    assert np.abs(GT.decode_depth(img) - d).max() < 1.2e-4         # a step (6e-5 m) and float32's rounding at 900 m
    raw = img.tobytes()
    assert np.allclose(GT.unpack_depth(GT.packed(raw, 51, 37)), GT.decode_depth(img), atol=1e-6)
    bgra = rng.integers(0, 256, (20, 30, 4), dtype=np.uint8)
    tag, ids = GT.decode_instance(bgra)
    assert np.array_equal(tag, bgra[..., 2]) and np.array_equal(ids, bgra[..., 1].astype(int) +
                                                                256 * bgra[..., 0].astype(int))
    t2, i2 = GT.unpack_instance(GT.packed(bgra.tobytes(), 30, 20))
    assert np.array_equal(t2, tag) and np.array_equal(i2, ids)
    s = np.zeros(3, GT.SEMANTIC_LIDAR)
    s['idx'] = [0, 7, 70000]
    assert list(GT.semantic_lidar(s.tobytes())['idx']) == [0, 7, 70000] and GT.SEMANTIC_LIDAR.itemsize == 24


# ======================================================================= boxes
def test_box_helpers_against_sampling():
    rng = np.random.default_rng(3)
    assert np.allclose(GT.box_matrix(37.0), carla_matrix(0.0, 37.0, 0.0))        # a yaw: CARLA's own matrix
    for _ in range(20):
        c = rng.uniform(-5, 5, 3)
        size = rng.uniform(0.3, 4.0, 3)
        R = GT.box_matrix(rng.uniform(-180, 180))
        E = GT.box_edges(c, size, R)
        F = GT.box_faces(c, size, R)
        assert len(E) == 12 * 17 and len(F) == 6 * 81
        q = np.abs((E - c) @ R) / (size / 2)
        assert np.allclose(np.sort(q, 1)[:, 1:], 1.0)              # every edge point: on two faces
        q = np.abs((F - c) @ R) / (size / 2)
        assert np.allclose(q.max(1), 1.0)
        assert GT.in_box(F, c, size, R, 1e-9).all() and not GT.in_box(c + R @ (size * 0.51 * [1, 0, 0]), c, size, R)
        o = c + rng.uniform(-12, 12, 3)
        D = rng.normal(size=(400, 3))
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        tin, tout = GT.ray_box(o, D, c, size, R)
        s = np.linspace(0, 40, 8001)
        for k in range(0, 400, 7):                                 # brute force along the ray
            inside = GT.in_box(o + s[:, None] * D[k], c, size, R)
            if inside.any():
                assert abs(tin[k] - s[np.argmax(inside)]) < 0.01 and \
                    abs(tout[k] - s[len(s) - 1 - np.argmax(inside[::-1])]) < 0.01
            else:
                assert not np.isfinite(tin[k]) or tin[k] > 39.99


# ======================================================================= COCO RLE
def test_rle_round_trip_and_from_pixels():
    rng = np.random.default_rng(2)
    for _ in range(300):
        h, w = rng.integers(1, 30, 2)
        m = rng.random((h, w)) < rng.random()
        r = GT.rle_encode(m)
        assert r['size'] == [h, w] and np.array_equal(GT.rle_decode(r), m)
        assert GT.rle_of_pixels(np.flatnonzero(m.ravel()), h, w) == r
        flat = m.ravel(order='F')                                  # an independent run count, from the zeros
        runs, cur, n = [], False, 0
        for x in flat:
            if bool(x) != cur:
                runs.append(n)
                cur, n = bool(x), 0
            n += 1
        runs.append(n)
        assert GT.rle_counts(r) == runs
    m = np.array([[0, 1, 1, 0], [0, 1, 0, 0], [1, 1, 0, 1]], bool)
    assert GT.rle_counts(GT.rle_encode(m)) == [2, 5, 4, 1]                 # 0 0 1 | 1 1 1 | 1 0 0 | 0 0 1


# ======================================================================= camera labels through the remap
@pytest.fixture(scope='module')
def cam():
    p = L.plan_camera(FISHEYE)
    return p, GT.CameraGT(p)


def _view_rays(plan):
    """Each atlas pixel's ray in the camera's optical frame, from the views alone (as CARLA renders them)."""
    at = plan['atlas']
    out = np.zeros((at['height'], at['width'], 3))
    for k, v in enumerate(plan['views']):
        u, vv = np.meshgrid(np.arange(v['width'], dtype=float), np.arange(v['height'], dtype=float))
        d = np.stack([(u - (v['width'] / 2 - 0.5)) / v['f_px'], (vv - (v['height'] / 2 - 0.5)) / v['f_px'],
                      np.ones_like(u)], -1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        out[at['offsets'][k]:at['offsets'][k] + v['height'], :v['width']] = d @ np.asarray(v['optical']).T
    return out


def test_labels_read_the_nearest_view_pixel_and_the_range_runs_along_the_ray(cam):
    """Every output pixel reads the view pixel nearest its ray (within ¾ of a view pixel, never a mix), the blend
    band's pixels the face with the higher weight; a sphere of radius 7 m about the camera gives range 7 m at every
    pixel from the views' planar depths."""
    plan, cg = cam
    at = plan['atlas']
    vr = _view_rays(plan)
    idx = np.arange(at['height'] * at['width']).reshape(at['height'], at['width'])
    got = cg.gather(idx, fill=-1)
    seen = got >= 0
    assert seen.mean() > 0.5 and np.array_equal(seen, cg.ok)
    ray_view = vr.reshape(-1, 3)[got[seen]]
    ang = np.degrees(np.arccos(np.clip(np.sum(ray_view * cg.rays[seen], 1), -1, 1)))
    f_min = min(v['f_px'] for v in plan['views'])
    assert ang.max() < math.degrees(0.75 / f_min)
    b = plan['remap']['blend']
    idx_b, bx, by, bw = (b[k] for k in ('idx', 'x', 'y', 'w'))
    assert np.all(np.argmax(bw, 1) == 0) and cg.blend_moved == 0
    bw2 = bw.copy()
    sel = np.flatnonzero(bw[:, 1] > 0)[::3]
    bw2[sel] = [0.2, 0.8, 0.0]
    cg2 = GT.CameraGT(dict(plan, remap=dict(plan['remap'], blend=dict(b, w=bw2))))
    assert cg2.blend_moved == len(sel)
    got2 = cg2.gather(idx, fill=-1).ravel()[idx_b[sel]]
    want = np.floor(by[sel, 0] + 0.5).astype(int) * at['width'] + np.floor(bx[sel, 0] + 0.5).astype(int)
    assert np.array_equal(got2, want)
    z = np.zeros((at['height'], at['width']), np.float32)
    for k, v in enumerate(plan['views']):
        sl = slice(at['offsets'][k], at['offsets'][k] + v['height'])
        rv = vr[sl, :v['width']] @ np.asarray(v['optical'])        # back into the view's own frame: its z is depth
        z[sl, :v['width']] = 7.0 * rv[..., 2]
    r = cg.range_of(cg.gather(z))
    err = np.abs(r[cg.ok] - 7.0)
    kpx = np.searchsorted(np.asarray(at['offsets']), got[seen] // at['width'], side='right') - 1
    axes = np.array([np.asarray(v['optical'])[:, 2] for v in plan['views']])
    cos_px = np.sum(ray_view * axes[kpx], 1)
    cos_out = np.sum(cg.rays[seen] * axes[kpx], 1)
    assert np.abs(err - 7.0 * np.abs(cos_px / cos_out - 1.0)).max() < 1e-5 and np.all(r[~cg.ok] == 0)
    assert err.max() <= 7.0 * 1.5 * 0.75 / f_min
    assert np.allclose(cg.range_m(z), r)
    z2 = np.full_like(z, 5.0)                                      # a plane facing each view: exact ranges
    r2 = cg.range_of(cg.gather(z2))
    for k, v in enumerate(plan['views']):
        on = cg.ok & ((got // at['width']) >= at['offsets'][k]) & ((got // at['width']) < at['offsets'][k] + v['height'])
        dz = (cg.rays[on] @ np.asarray(v['optical']))[:, 2]
        assert np.allclose(r2[on], 5.0 / dz, rtol=1e-5), v['name']
    png = GT.range_png(r)
    assert np.abs(png[cg.ok] / GT.RANGE_SCALE - r[cg.ok]).max() <= 0.5 / GT.RANGE_SCALE + 1e-6
    assert GT.range_png(np.array([300.0, 0.0, 10.0])).tolist() == [0, 0, 2560]


def test_range_interpolates_inverse_depth_within_a_surface_and_never_across_an_edge(cam):
    plan, cg = cam
    at = plan['atlas']
    vr = _view_rays(plan)
    n = np.array([0.3, -0.5, 0.81])
    n /= np.linalg.norm(n)
    c = 6.0                                                        # the plane n · X = c in the camera's optical frame
    z = np.zeros((at['height'], at['width']))
    for k, v in enumerate(plan['views']):
        sl = slice(at['offsets'][k], at['offsets'][k] + v['height'])
        d = vr[sl, :v['width']]
        t = c / np.maximum(d @ n, 1e-6)
        z[sl, :v['width']] = t * (d @ np.asarray(v['optical']))[..., 2]
    seg = np.zeros(z.shape, np.uint32)
    r = cg.range_interp(z, seg)
    want = c / np.maximum(cg.rays[cg.ok] @ n, 1e-6)
    front = (cg.rays[cg.ok] @ n) > 0.05                            # (the plane in front of the camera)
    assert np.allclose(r[cg.ok][front], want[front], rtol=1e-5)
    z2 = np.where(vr[..., 0] > 0.1, 5.0, 9.0) * 1.0
    seg2 = (vr[..., 0] > 0.1).astype(np.uint32)
    for k, v in enumerate(plan['views']):
        sl = slice(at['offsets'][k], at['offsets'][k] + v['height'])
        z2[sl, :v['width']] *= (vr[sl, :v['width']] @ np.asarray(v['optical']))[..., 2]
    r2 = cg.range_interp(z2, seg2)[cg.ok]
    near5, near9 = np.abs(r2 - 5.0) < 0.05, np.abs(r2 - 9.0) < 0.09
    assert np.all(near5 | near9) and near5.sum() > 100 and near9.sum() > 100
    r3 = cg.range_interp(z, None)[cg.ok]
    nearest = cg.range_of(cg.gather(z))[cg.ok]
    exact = np.isclose(r3, want, rtol=1e-5)
    facing = (cg.rays[cg.ok] @ n) > 0.2                            # (not grazing: the plane not steep across a pixel)
    assert np.all(exact | np.isclose(r3, nearest, rtol=1e-6)) and exact[facing].mean() > 0.9


def _render(cg, pose, boxes):
    """Instance ids, tags and range of boxes in the vehicle frame along the camera's output rays: an independent ray
    cast (groundtruth.ray_box)."""
    R, pos = pose['R'], pose['pos']
    D = cg.rays.reshape(-1, 3) @ (R @ CARLA_TO_OPTICAL.T).T
    best = np.full(len(D), np.inf)
    ids = np.zeros(len(D), np.uint16)
    tags = np.full(len(D), 11, np.uint8)                          # sky
    for rid, tag, c, size, Rb in boxes:
        tin, _ = GT.ray_box(pos, D, c, size, Rb)
        near = tin < best
        best[near] = tin[near]
        ids[near] = rid
        tags[near] = tag
    sh = cg.rays.shape[:2]
    best[~cg.ok.ravel()] = np.inf
    return ids.reshape(sh), tags.reshape(sh), np.where(np.isfinite(best), best, 0.0).reshape(sh).astype(np.float32)


def test_objects_boxes_truncation_and_occlusion(cam):
    plan, cg = cam
    pose = GT.camera_pose(plan['pose'])
    lens = plan['lens']
    car = {'key': ('actor', 7), 'actor_id': 7, 'reg': 1, 'category': 'vehicle.car', 'center': np.array([12.0, 1.0, 0.75]),
           'size': [4.5, 1.9, 1.5], 'R': GT.box_matrix(20.0)}
    occ = (9, 20, np.array([8.0, 0.2, 0.75]), [0.3, 0.8, 1.6], np.eye(3))      # a static thing half in front of it
    ids, tags, rng = _render(cg, pose, [(7, 14, car['center'], car['size'], car['R'])])
    h, w = ids.shape
    found, sem = GT.objects_in_frame(tags, ids, rng, cg.rays, pose, lens, {7: car}, [], ego_id=1, assoc={})
    assert len(found) == 1 and found[0]['class'] == 14 and found[0]['pixels'] == int((ids == 7).sum())
    g = GT.object_geometry(car, pose, lens, w, h, cg.rays, rng, pix=found[0]['pix'])
    assert np.abs(np.array(found[0]['bbox']) - np.array(g['bbox_full'])).max() <= 1.01
    assert g['occluded'] == 0.0 and g['truncated'] == 0.0
    assert g['range_m'] == pytest.approx(np.linalg.norm(car['center'] - pose['pos']), abs=1e-3)
    ids2, tags2, rng2 = _render(cg, pose, [(7, 14, car['center'], car['size'], car['R']), occ])
    found2, _ = GT.objects_in_frame(tags2, ids2, rng2, cg.rays, pose, lens, {7: car}, [], 1, {})
    g2 = GT.object_geometry(car, pose, lens, w, h, cg.rays, rng2, pix=found2[0]['pix'])
    hidden = int(((ids2 == 9) & (ids == 7)).sum())
    assert 0.05 < g2['occluded'] < 0.95
    assert g2['occluded'] == pytest.approx(hidden / (hidden + found2[0]['pixels']), abs=0.01)
    # a box on the image's left border: about half of its surface outside the image
    edge = cg.rays[h // 2, np.flatnonzero(cg.ok[h // 2])[0]]
    centre = pose['pos'] + (pose['R'] @ CARLA_TO_OPTICAL.T) @ edge * 8.0
    side = dict(car, center=centre, size=[1.0, 1.0, 1.0], R=np.eye(3))
    g3 = GT.object_geometry(side, pose, lens, w, h)
    assert 0.2 < g3['truncated'] < 0.8 and g3['occluded'] is None and g3['bbox_full'][0] < 0 < g3['bbox_full'][2]
    ids3 = ids.copy()                                              # the ego's own pixels: class 'ego', never an object
    ids3[ids == 7] = 1
    found3, sem3 = GT.objects_in_frame(tags, ids3, rng, cg.rays, pose, lens, {7: car}, [], 1, {})
    assert not found3 and np.all(sem3[ids == 7] == GT.EGO_CLASS)


def test_level_objects_by_their_boxes_and_lights_claim_their_heads(cam):
    """A parked car of the map (no actor id) is found by its level box; a traffic light's head box claims the static
    pixels inside it whatever CARLA tagged them (here 'pole' and 'other'), each head its own, a car passing in front
    keeps its own."""
    plan, cg = cam
    pose = GT.camera_pose(plan['pose'])
    lens = plan['lens']
    parked = {'key': ('level', 0), 'level_label': 'Car', 'actor_id': None, 'reg': 1000, 'category': 'vehicle.car',
              'center': np.array([10.0, -3.0, 0.75]), 'size': [4.5, 1.9, 1.5], 'R': np.eye(3)}
    head = {'key': ('level', 1), 'level_label': 'TrafficLight', 'actor_id': 31, 'reg': 1001, 'category': 'static.manmade',
            'part_class': 7, 'center': np.array([9.0, 3.0, 1.5]), 'size': [0.4, 0.6, 1.0], 'R': np.eye(3)}
    head2 = dict(head, key=('level', 2), reg=1002, center=np.array([15.0, 4.666, 1.6]))
    ids, tags, rng = _render(cg, pose, [(45000, 14, parked['center'], parked['size'], parked['R']),
                                        (31, 6, head['center'] + [0, 0, 0.25], [0.4, 0.6, 0.5], np.eye(3)),
                                        (31, 22, head['center'] - [0, 0, 0.25], [0.4, 0.6, 0.5], np.eye(3)),
                                        (31, 22, head2['center'], head2['size'], np.eye(3))])
    assoc = {}
    found, sem = GT.objects_in_frame(tags, ids, rng, cg.rays, pose, lens, {}, [parked], 99, assoc,
                                     parts=[head, head2])
    by = {f['obj']['key']: f for f in found}
    assert assoc[45000] == ('level', 0) and by[('level', 0)]['class'] == 14
    hp = by[('level', 1)]
    other = by.get(('level', 2), {'pixels': 0})['pixels']
    assert hp['class'] == 7 and hp['pixels'] > 20 and hp['pixels'] + other == int((ids == 31).sum())
    assert np.all(sem.ravel()[hp['pix']] == 7) and set(np.unique(tags.ravel()[hp['pix']])) == {6, 22}
    assert hp['raw_tags'][6] > 0 and hp['raw_tags'][22] > 0 and hp['raw_tags'][7] == 0
    # a car between the camera and the head keeps its pixels
    cpos = pose['pos'] + (head['center'] - pose['pos']) * 4 / 7
    car = {'key': ('actor', 5), 'actor_id': 5, 'reg': 3, 'category': 'vehicle.car', 'center': cpos,
           'size': [0.3, 0.3, 0.3], 'R': np.eye(3)}
    ids2, tags2, rng2 = _render(cg, pose, [(31, 22, head['center'], head['size'], np.eye(3)),
                                           (5, 14, car['center'], car['size'], np.eye(3))])
    found2, sem2 = GT.objects_in_frame(tags2, ids2, rng2, cg.rays, pose, lens, {5: car}, [], 99, {}, parts=[head])
    by2 = {f['obj']['key']: f for f in found2}
    assert (ids2 == 5).sum() > 20 and by2[('actor', 5)]['pixels'] == int((ids2 == 5).sum())
    assert np.all(sem2[ids2 == 5] == 14) and by2[('level', 1)]['pixels'] > 0


# ======================================================================= end to end: the views in the fake carla
def test_views_rendered_by_carla_give_the_lens_image_its_labels_and_ranges(fake_module):
    """The plan's views spawned as CARLA cameras at their poses (the fake carla renders them as CARLA does): the
    remapped image is the scene seen along each output pixel's ray, the labels read through the remap are the hit's,
    and the range image is the distance along the ray (an independent ray cast of the fake's world)."""
    carla = fake_module
    world = carla.Client('localhost', 2000).get_world()
    st = world.get_settings()
    st.synchronous_mode, st.fixed_delta_seconds = True, 0.05
    world.apply_settings(st)
    pose = {'x': 0.0, 'y': 6.0, 'z': 1.5, 'roll': 0.0, 'pitch': -5.0, 'yaw': 90.0}   # facing the parked cars (+y)
    s = dict(FISHEYE, pose=pose, width=192, height=160, lens=dict(FISHEYE['lens'], f_px=60.0))
    p = L.plan_camera(s)
    lib = world.get_blueprint_library()
    got = {}
    for k, v in enumerate(p['views']):
        q = v['pose']
        tf = carla.Transform(carla.Location(q['x'], q['y'], q['z']), carla.Rotation(q['pitch'], q['yaw'], q['roll']))
        for kind in ('rgb', 'depth', 'instance_segmentation'):
            bp = lib.find('sensor.camera.' + kind)
            for a, val in v['attributes'].items():
                if bp.has_attribute(a):
                    bp.set_attribute(a, str(val))
            world.spawn_actor(bp, tf).listen(lambda d, key=(k, kind): got.__setitem__(key, d))
    world.tick()
    n = len(p['views'])
    img = {kind: [GT.bgra(got[(k, kind)].raw_data, got[(k, kind)].width, got[(k, kind)].height) for k in range(n)]
           for kind in ('rgb', 'depth', 'instance_segmentation')}
    cg = GT.CameraGT(p)
    rng = cg.range_from_packed(L.atlas_of(p, [x.view('<u4')[..., 0] for x in img['depth']]),
                               L.atlas_of(p, [x.view('<u4')[..., 0] for x in img['instance_segmentation']]))
    tag, ids = GT.decode_instance(L.atlas_of(p, img['instance_segmentation']))
    tag_out, ids_out = cg.gather(tag), cg.gather(ids)
    # the truth along the output rays, cast in the fake's world
    cp = GT.camera_pose(pose)
    D = cg.rays.reshape(-1, 3)[cg.ok.ravel()] @ (cp['R'] @ CARLA_TO_OPTICAL.T).T
    boxes = world._boxes()
    t, hid = carla._cast(np.repeat(cp['pos'][None], len(D), 0), D, boxes, 1000.0)
    ttag, tids, _ = world._labels(cp['pos'], D, t, hid, boxes)
    r = rng[cg.ok]
    hit = np.isfinite(t)
    assert hit.mean() > 0.4 and (ids_out[cg.ok] >= 40000).sum() > 50              # the ground and a parked car
    assert np.mean(tag_out[cg.ok] == ttag) > 0.98 and np.mean(ids_out[cg.ok] == tids) > 0.98
    assert np.mean(r[~hit] == 0) > 0.98
    rel = np.abs(r[hit] - t[hit]) / t[hit]
    assert np.median(rel) < 1e-4 and np.mean(rel < 0.01) > 0.97, (np.median(rel), np.mean(rel < 0.01))
    # the image: the scene's colour along each output ray (BGR, as CARLA's raw_data) wherever the scene is uniform
    # around the pixel (at an edge the bilinear read of the views mixes the two sides)
    out = L.remap_image(p, img['rgb']).astype(int)
    h, w = cg.ok.shape
    T = np.zeros((h, w, 3), int)
    T[cg.ok] = carla.render(cp['pos'], D, boxes)[:, ::-1]
    uni = cg.ok.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
        uni &= np.roll(cg.ok, (dy, dx), (0, 1)) & (np.abs(np.roll(T, (dy, dx), (0, 1)) - T).max(2) <= 6)
    uni[[0, -1], :] = uni[:, [0, -1]] = False
    diff = np.abs(out - T).max(2)[uni]
    assert uni.sum() > 0.7 * cg.ok.sum() and np.median(diff) == 0 and np.mean(diff <= 3) > 0.995, np.mean(diff <= 3)


# ======================================================================= the reader
def test_read_and_loaders(tmp_path):
    import cv2
    out = tmp_path
    (out / 'ground_truth' / 'CAM_F').mkdir(parents=True)
    sem = np.random.default_rng(0).integers(0, 29, (6, 8)).astype(np.uint8)
    inst = np.zeros((6, 8), np.uint16)
    inst[1:3, 2:5] = 1003
    rng = np.full((6, 8), 12.5, np.float32)
    for name, a in (('semantic', sem), ('instance', inst), ('range', GT.range_png(rng))):
        cv2.imwrite(str(out / 'ground_truth' / 'CAM_F' / f'x.{name}.png'), a)
    frame = {k: f'ground_truth/CAM_F/x.{k}.png' for k in ('semantic', 'instance', 'range')}
    frame.update(sample_data_token='t', timestamp=1, image=None, objects=[{'id': 1003, 'class': 'car', 'actor_id': 4}])
    (out / 'nuscenes' / 'lidarseg').mkdir(parents=True)
    np.array([17, 24], np.uint8).tofile(out / 'nuscenes' / 'lidarseg' / 'a_lidarseg.bin')
    np.savez_compressed(out / 'nuscenes' / 'lidarseg' / 'a_panoptic.npz', data=np.array([17003, 24000], np.uint16))
    sweep = {'lidarseg': 'nuscenes/lidarseg/a_lidarseg.bin', 'panoptic': 'nuscenes/lidarseg/a_panoptic.npz'}
    doc = {'schema': GT.SCHEMA, 'classes': GT.classes_doc(), 'cameras': {'F': {'channel': 'CAM_F', 'width': 8,
                                                                                'height': 6, 'frames': [frame]}},
           'lidars': {'L1': {'channel': 'LIDAR_L1', 'sweeps': [sweep]}}}
    (out / 'ground_truth' / 'index.json').write_text(json.dumps(doc))
    d = GT.read(str(out))
    assert d is not None and d['_root'] == os.path.abspath(str(out))
    f = GT.frames(d, 'F')[0]
    assert np.array_equal(GT.load_semantic(d, f), sem) and np.array_equal(GT.load_instance(d, f), inst)
    assert np.allclose(GT.load_range(d, f), 12.5)
    assert GT.load_lidarseg(d, sweep).tolist() == [17, 24] and GT.load_panoptic(d, sweep).tolist() == [17003, 24000]
    s = GT.summary(d)
    assert s['objects'] == 1 and s['actors'] == 1 and s['cameras'] == {'F': 1} and s['lidars'] == {'L1': 1}
    (out / 'labels.json').write_text(json.dumps(doc))                         # the index elsewhere: a parameter
    assert GT.read(str(out), index='labels.json')['cameras']['F']['width'] == 8
    (out / 'ground_truth' / 'index.json').write_text(json.dumps(dict(doc, schema='vantage.ground_truth/1')))
    assert GT.read(str(out)) is None and GT.read(str(tmp_path / 'none')) is None


# ======================================================================= viewing
def test_viewer_colours_and_the_range_scale():
    sem = np.zeros((4, 6), np.uint8)
    sem[:, 2:] = GT.CLASS_ID['road']
    sem[0, 5] = GT.CLASS_ID['car']
    sem[1, 0] = GT.EGO_CLASS
    rgb = GT.colour_semantic(sem)
    assert rgb.shape == (4, 6, 3) and rgb.dtype == np.uint8
    assert tuple(rgb[2, 3]) == (128, 64, 128) and tuple(rgb[0, 5]) == (0, 0, 142)
    assert tuple(rgb[1, 0]) == (0, 255, 255)
    cl = GT.classes_in(sem)
    assert [c['name'] for c in cl] == ['road', 'unlabelled', 'car', 'ego']           # a tie: by class id
    assert [c['pixels'] for c in cl] == [15, 7, 1, 1] and math.isclose(sum(c['share'] for c in cl), 1.0)
    ids = np.arange(0, 301)
    cols = GT.object_colours(ids)
    assert (cols[0] == 0).all() and (cols[1:].max(axis=1) > 100).all()
    assert np.array_equal(cols, GT.object_colours(ids)) and np.array_equal(GT.object_colours([7])[0], cols[7])
    d = np.abs(cols[1:-1].astype(int) - cols[2:].astype(int)).sum(axis=1)
    assert d.min() >= 60                                          # neighbouring ids never look alike
    inst = np.array([[0, 5, 5], [1004, 0, 5]], np.uint16)
    ci = GT.colour_instance(inst)
    assert (ci[0, 0] == 0).all() and np.array_equal(ci[0, 1], GT.object_colours([5])[0]) and \
        np.array_equal(ci[1, 0], GT.object_colours([1004])[0])
    rng = np.array([[0.0, 1.0, 10.0, 100.0, 300.0, 0.5]], np.float32)
    t = GT.range_position(rng, 1.0, 100.0)
    assert np.isnan(t[0, 0]) and t[0, 1] == 1.0 and math.isclose(t[0, 2], 0.5, abs_tol=1e-6) and t[0, 3] == 0.0
    assert t[0, 4] == 0.0 and t[0, 5] == 1.0
    r = np.random.default_rng(3).uniform(2.3, 87.0, 10000).astype(np.float32)
    lo, hi = GT.range_limits(np.concatenate([r, np.zeros(500, np.float32)]))
    assert lo in (2.0, 1.0) and hi == 100.0 and GT.range_limits(np.zeros((3, 3))) == (1.0, 100.0)
    assert GT.scale_ticks(0.5, 200.0) == [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    pal = GT.lidarseg_palette()
    assert tuple(pal[GT.NUSC_INDEX['vehicle.car']]) == (0, 0, 142)
    assert tuple(pal[GT.NUSC_INDEX['human.pedestrian.child']]) == (220, 20, 60)
    assert tuple(pal[GT.NUSC_INDEX['vehicle.bicycle']]) == (119, 11, 32)
    assert tuple(pal[GT.NUSC_INDEX['noise']]) == GT.NOISE_RGB and tuple(pal[GT.NUSC_INDEX['vehicle.ego']]) == \
        (0, 255, 255)


def test_lidar_sweep_through_its_calibration_seen_from_above(tmp_path):
    """A sweep's points read from its nuScenes .pcd.bin through its calibrated_sensor (nuScenes' right-handed axes: a
    lidar turned 90° to the left and raised, each point R·p + t) and mirrored into CARLA's vehicle frame (y right),
    the sweep nearest a time, a footprint of a turned box, and the top view (x up, CARLA's y to the right): each point
    at its pixel, the higher point over a lower one, a dot of 2 × 2."""
    V = tmp_path / 'nuscenes' / GT.NUSCENES_VERSION
    V.mkdir(parents=True)
    (tmp_path / 'nuscenes' / 'sweeps' / 'LIDAR_L1').mkdir(parents=True)
    (tmp_path / 'nuscenes' / 'lidarseg' / GT.NUSCENES_VERSION).mkdir(parents=True)
    assert GT.NUSCENES_VERSION == 'v1.0-falcon'
    pts = np.array([[10.0, 0.0, -1.5, 3, 0], [0.0, -4.0, 0.5, 5, 1], [0.0, -4.0, -1.0, 6, 2], [55.0, 0, 0, 1, 1]],
                   np.float32)
    fn = 'sweeps/LIDAR_L1/x__LIDAR_L1__2000.pcd.bin'
    pts.tofile(str(tmp_path / 'nuscenes' / fn))
    lab = np.array([24, 17, 30, 28], np.uint8)
    lab.tofile(str(tmp_path / 'nuscenes' / 'lidarseg' / GT.NUSCENES_VERSION / 'sdB_lidarseg.bin'))
    c, s_ = math.cos(math.radians(45)), math.sin(math.radians(45))             # 90° about z: (w, x, y, z)
    (V / 'sample_data.json').write_text(json.dumps([
        {'token': 'sdA', 'filename': 'sweeps/LIDAR_L1/x__LIDAR_L1__1000.pcd.bin', 'calibrated_sensor_token': 'csA'},
        {'token': 'sdB', 'filename': fn, 'calibrated_sensor_token': 'csA'}]))
    (V / 'calibrated_sensor.json').write_text(json.dumps([{'token': 'csA', 'translation': [1.0, 0.5, 2.0],
                                                           'rotation': [c, 0.0, 0.0, s_]}]))
    rel = f'nuscenes/lidarseg/{GT.NUSCENES_VERSION}'
    doc = {'_root': str(tmp_path), 'lidars': {'L1': {'sweeps': [
        {'sample_data_token': 'sdA', 'timestamp': 1000, 'lidarseg': f'{rel}/sdA_lidarseg.bin'},
        {'sample_data_token': 'sdB', 'timestamp': 2000, 'lidarseg': f'{rel}/sdB_lidarseg.bin'}]}}}
    assert GT.sweep_near(doc, 'L1', 1400)['sample_data_token'] == 'sdA'
    sw = GT.sweep_near(doc, 'L1', 1600)
    assert sw['sample_data_token'] == 'sdB' and GT.sweep_near(doc, 'L9', 0) is None
    P = GT.lidar_points(doc, sw)
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(P, (pts[:, :3] @ R.T + [1.0, 0.5, 2.0]) * [1.0, -1.0, 1.0], atol=1e-5)
    assert np.array_equal(GT.load_lidarseg(doc, sw), lab)
    img = GT.topdown(P, GT.lidarseg_palette()[lab], half_m=20.0, px=400)          # 0.1 m per pixel
    hit = np.argwhere(img[..., 3] > 0)
    assert len(hit) == 2                                   # two points share a pixel; one is beyond the edge
    # point 0 at CARLA (1, −10.5, 0.5): row 200 − 10, col 200 − 105 (to the left); points 1 and 2 meet at (5, −0.5):
    # the higher (car) wins
    assert tuple(img[190, 95, :3]) == tuple(GT.lidarseg_palette()[24])
    assert tuple(img[150, 195, :3]) == tuple(GT.lidarseg_palette()[17])
    big = GT.topdown(P, GT.lidarseg_palette()[lab], half_m=20.0, px=400, dot=2)
    assert (big[..., 3] > 0).sum() == 8 and tuple(big[151, 196, :3]) == tuple(GT.lidarseg_palette()[17])
    fp = GT.box_footprint({'center': [5.0, 2.0, 0.7], 'size': [4.0, 2.0, 1.5], 'yaw_deg': 90.0})
    assert np.allclose(fp, [[4.0, 4.0], [4.0, 0.0], [6.0, 0.0], [6.0, 4.0]], atol=1e-9)
