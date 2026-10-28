"""Faults of known size: each one measured back from what it did."""
import numpy as np
import pytest

from falcon_autonomy.core import faults as F


def _img(h=120, w=160):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.dstack([(xx * 255 // w), (yy * 255 // h), np.full((h, w), 128)]).astype(np.uint8)


def test_noise_has_its_sigma():
    img = np.full((200, 200, 3), 128, np.uint8)
    out = F.noise(img, 5.0, np.random.default_rng(0))
    assert np.std(out.astype(float) - 128) == pytest.approx(5.0, rel=0.03)


@pytest.mark.parametrize('kind', ['dirt', 'rain'])
def test_cover_reaches_its_share(kind):
    img = _img()
    out = getattr(F, kind)(img, 0.2, np.random.default_rng(1))
    changed = np.any(np.abs(out.astype(int) - img.astype(int)) > 6, axis=2).mean()
    assert 0.12 <= changed <= 0.4
    again = getattr(F, kind)(img, 0.2, np.random.default_rng(1))
    assert np.array_equal(out, again)                          # seeded


def test_rolling_shutter_shears_by_the_rotation():
    """Yawing at ω with readout T: the bottom row is shifted against the top by f·ω·T (small angles)."""
    h, w, f = 101, 200, 100.0
    img = np.zeros((h, w, 3), np.uint8)
    img[:, 95:105] = 255                                       # a vertical bar
    K = [[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]
    out = F.rolling_shutter(img, 0.05, K, [0.0, 0.5, 0.0])     # yaw about the optical y (down) axis, 0.5 rad/s
    x = np.arange(w)
    cx_top = np.sum(x * out[0, :, 0]) / np.sum(out[0, :, 0])         # intensity-weighted centroids
    cx_bot = np.sum(x * out[-1, :, 0]) / np.sum(out[-1, :, 0])
    assert abs(cx_bot - cx_top) == pytest.approx(f * 0.5 * 0.05, rel=0.15)


def test_glare_and_compression():
    img = _img()
    g = F.glare(img, 0.5, 0.5, 0.5)
    assert g[60, 80].astype(int).sum() > img[60, 80].astype(int).sum() + 200
    c = F.compression(img, 5)
    assert c.shape == img.shape and np.abs(c.astype(int) - img.astype(int)).mean() > 0.5


def test_poses_clock_and_delivery():
    pose = {'x': 1.0, 'y': 0.0, 'z': 1.5, 'roll': 0.0, 'pitch': -5.0, 'yaw': 0.0}
    fs = [{'kind': 'mount_rotation', 'deg': [0, 1.0, 2.0]}, {'kind': 'mount_translation', 'm': [0.1, 0, 0]},
          {'kind': 'clock_offset', 'ms': 40}, {'kind': 'drop', 'rate': 0.3},
          {'kind': 'freeze', 'from_s': 1.0, 'to_s': 2.0}]
    a = F.actual_pose(pose, fs)
    assert a['pitch'] == -4.0 and a['yaw'] == 2.0 and a['x'] == pytest.approx(1.1)
    assert F.clock_offset_s(fs) == pytest.approx(0.04)
    rng = np.random.default_rng(3)
    d = [F.delivery(fs, i, i * 0.05, rng) for i in range(400)]
    assert 0.2 < d.count('drop') / 400 < 0.4 and 'freeze' in d[20:40]
    assert F.validate(fs) == []
    assert F.validate([{'kind': 'drop', 'rate': 2}, {'kind': 'nope'}, {'kind': 'mount_rotation', 'deg': [90, 0, 0]}])


def test_apply_image_order_and_report():
    img = _img()
    out, done = F.apply_image(img, [{'kind': 'compression', 'quality': 30}, {'kind': 'noise', 'sigma': 2}],
                              np.random.default_rng(0))
    assert done == ['noise', 'compression'] and out.shape == img.shape
