"""Falcon's ByteTrack (core.bytetrack): ids held through a crossing, a track kept alive on low-score boxes while
occluded (the "byte" association), re-found after a full occlusion within the buffer and dropped after it, the
two-stage thresholds, class votes, and the Kalman filter's constant-velocity prediction."""
import numpy as np
import pytest

pytest.importorskip('scipy')

from falcon_autonomy.core import bytetrack as BT  # noqa: E402


def box(cx, cy, w=40, h=30):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def ids_of(out, at):
    """The id of the output box nearest centre x `at`."""
    best = min(out, key=lambda r: abs((r['box'][0] + r['box'][2]) / 2 - at))
    return best['track']


def test_two_objects_crossing_keep_their_ids():
    t = BT.ByteTracker(frame_rate=10)
    a_ids, b_ids = set(), set()
    for k in range(40):
        xa, xb = 100 + 8 * k, 420 - 8 * k                       # they pass each other around k = 20
        ya, yb = 100, 112                                        # slightly offset rows: they overlap while crossing
        out = t.update([box(xa, ya), box(xb, yb)], [0.9, 0.9], ['car', 'motorcycle'])
        assert len(out) == 2
        for r in out:
            (a_ids if r['det'] == 0 else b_ids).add(r['track'])
    assert len(a_ids) == len(b_ids) == 1 and a_ids != b_ids
    kinds = {r['track']: r['kind'] for r in out}
    assert kinds[next(iter(a_ids))] == 'car' and kinds[next(iter(b_ids))] == 'motorcycle'


def test_low_score_boxes_keep_an_occluded_track():
    t = BT.ByteTracker(frame_rate=10)
    first = None
    for k in range(30):
        score = 0.3 if 10 <= k < 18 else 0.9                     # partly hidden: the detector is unsure
        out = t.update([box(100 + 5 * k, 100)], [score], ['car'])
        assert len(out) == 1, k                                   # matched in the second (low-score) association
        first = first or out[0]['track']
        assert out[0]['track'] == first
    # without the byte step the same occlusion loses the track (low boxes below the new-track threshold too)
    t2 = BT.ByteTracker(frame_rate=10, low_thresh=0.6)
    seen = [t2.update([box(100 + 5 * k, 100)], [0.3 if 10 <= k < 18 else 0.9], ['car']) for k in range(30)]
    assert all(not s for s in seen[10:18])


def test_a_full_occlusion_within_the_buffer_and_beyond():
    t = BT.ByteTracker(frame_rate=10, buffer=30)               # 30 at 30 frames/s: 10 frames at 10
    assert t.max_lost == 10
    tid = t.update([box(100, 100)], [0.9], ['car'])[0]['track']
    for k in range(1, 6):
        t.update([box(100 + 4 * k, 100)], [0.9], ['car'])
    for k in range(6, 12):                                     # hidden for 6 frames
        assert t.update([], [], []) == []
    out = t.update([box(100 + 4 * 12, 100)], [0.9], ['car'])   # where constant velocity puts it
    assert out[0]['track'] == tid
    for k in range(13, 13 + 12):                               # hidden for 12 frames: beyond the buffer
        t.update([], [], [])
    out = t.update([box(100 + 4 * 25, 100)], [0.9], ['car'])
    assert out == []                                           # a new track: unconfirmed on its first frame
    out = t.update([box(100 + 4 * 26, 100)], [0.9], ['car'])
    assert out and out[0]['track'] != tid


def test_thresholds_new_tracks_and_unconfirmed():
    t = BT.ByteTracker(frame_rate=30)
    assert t.params()['high_thresh'] == 0.6 and t.params()['low_thresh'] == 0.1 and t.max_lost == 30
    assert t.update([box(50, 50)], [0.65], ['car']) == []      # high but under the new-track threshold 0.7
    assert len(t.tracks) == 0
    assert t.update([box(50, 50)], [0.05], ['car']) == []      # below the low threshold: ignored
    out = t.update([box(50, 50)], [0.8], ['car'])
    assert out == []                                           # born on a later frame: unconfirmed
    out = t.update([box(51, 50)], [0.8], ['car'])
    assert len(out) == 1                                       # confirmed on its second match
    nt = BT.ByteTracker()
    nt.update([], [], [])
    nt.update([box(300, 300)], [0.95], ['car'])
    assert nt.update([box(600, 300)], [0.95], ['car']) == []   # an unconfirmed track not matched: removed
    assert all(tr.start == nt.frame for tr in nt.tracks)


def test_kalman_predicts_constant_velocity_and_iou():
    kf = BT.KalmanBox()
    m, c = kf.initiate(BT.xyxy_to_xyah(box(100, 100)))
    for k in range(1, 8):
        m, c = kf.predict(m, c)
        m, c = kf.update(m, c, BT.xyxy_to_xyah(box(100 + 10 * k, 100)))
    m, c = kf.predict(m, c)
    assert abs(BT.xyah_to_xyxy(m)[0] - box(180, 100)[0]) < 2.0
    iou = BT.iou_matrix([box(0, 0)], [box(0, 0), box(20, 0), box(100, 100)])
    assert iou[0, 0] == pytest.approx(1.0) and iou[0, 1] == pytest.approx(1 / 3) and iou[0, 2] == 0.0
    m_, ur, uc = BT.assign(np.array([[0.1, 0.9], [0.9, 0.95]]), 0.8)
    assert m_ == [(0, 0)] and ur == [1] and uc == [1]
