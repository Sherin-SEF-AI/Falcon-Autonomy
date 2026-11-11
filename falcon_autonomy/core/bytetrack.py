"""Multi-object tracking by ByteTrack — Falcon's own implementation of the algorithm (numpy; scipy for the assignment).

ByteTrack is Zhang et al. 2022, "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV
(arXiv:2110.06864); its reference code (github.com/ifzhang/ByteTrack, now FoundationVision/ByteTrack) is MIT-licensed.
This module re-implements the published algorithm and follows the reference code's choices, credited here; no code
was copied.

The algorithm, per frame:
1. Split the detections by score: high (≥ high_thresh, 0.6 here) and low (low_thresh 0.1 ≤ score < 0.6).
2. Predict every tracked and lost track one step with its Kalman filter (state x, y, a, h — the box centre, aspect
   ratio w/h and height — and their velocities; constant velocity; the noise scaled by the box height as in the
   reference: position σ 1/20 h, velocity σ 1/160 h).
3. First association: tracked + lost tracks against the high detections by IoU distance (1 − IoU, fused with the
   detection's score: 1 − IoU·score, as the reference's fuse_score), solved by scipy's linear_sum_assignment, kept
   when the cost is below match_thresh (0.8). Matched tracks are updated (lost ones re-found).
4. Second association — the "byte": the tracked (not lost) tracks still unmatched against the LOW detections by
   plain IoU distance, kept below 0.5. This keeps an occluded object's track alive on its low-score box.
5. Unconfirmed tracks (born last frame) against the remaining high detections (cost below 0.7); those not matched are
   removed.
6. Tracks still unmatched become lost; a track lost for more than buffer frames (30 at 30 frames/s, scaled by the
   frame rate as the reference does) is removed.
7. Remaining high detections with score ≥ high_thresh + 0.1 (the reference's det_thresh) start new tracks; a new
   track is confirmed on its first frame only in the very first frame, otherwise on its second match.

Classes: association is class-agnostic (as ByteTrack); each track keeps a vote of the classes it was matched with,
and its class is the most voted one (with the scores summed).

Usage: t = ByteTracker(frame_rate=10); for each frame: out = t.update(boxes (N, 4) xyxy, scores (N,), kinds) →
[{'track': id, 'box': [x1, y1, x2, y2] (the matched detection's box), 'score', 'kind', 'det': index of the
detection}] for the confirmed tracks matched in this frame.
"""
from __future__ import annotations

import numpy as np


class KalmanBox:
    """Constant-velocity Kalman filter on (x, y, a, h) — box centre, aspect ratio w/h, height — and velocities."""
    std_pos = 1.0 / 20
    std_vel = 1.0 / 160

    def __init__(self):
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = 1.0
        self.H = np.eye(4, 8)

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self.std_pos * h, 2 * self.std_pos * h, 1e-2, 2 * self.std_pos * h,
               10 * self.std_vel * h, 10 * self.std_vel * h, 1e-5, 10 * self.std_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        q = [self.std_pos * h, self.std_pos * h, 1e-2, self.std_pos * h,
             self.std_vel * h, self.std_vel * h, 1e-5, self.std_vel * h]
        return self.F @ mean, self.F @ cov @ self.F.T + np.diag(np.square(q))

    def update(self, mean, cov, z):
        h = mean[3]
        r = np.diag(np.square([self.std_pos * h, self.std_pos * h, 1e-1, self.std_pos * h]))
        S = self.H @ cov @ self.H.T + r
        K = np.linalg.solve(S, (cov @ self.H.T).T).T
        mean = mean + K @ (z - self.H @ mean)
        cov = cov - K @ S @ K.T
        return mean, cov


def xyxy_to_xyah(b):
    b = np.asarray(b, float)
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h])


def xyah_to_xyxy(m):
    x, y, a, h = m[:4]
    w = a * h
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])


def iou_matrix(a, b) -> np.ndarray:
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def assign(cost: np.ndarray, thresh: float):
    """(matches [(row, col)], unmatched rows, unmatched cols) of the minimum-cost assignment under thresh."""
    from scipy.optimize import linear_sum_assignment
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    c = np.where(cost > thresh, thresh + 1e4, cost)
    r, k = linear_sum_assignment(c)
    m = [(int(i), int(j)) for i, j in zip(r, k) if cost[i, j] <= thresh]
    mr, mc = {i for i, _ in m}, {j for _, j in m}
    return m, [i for i in range(cost.shape[0]) if i not in mr], [j for j in range(cost.shape[1]) if j not in mc]


class Track:
    NEW, TRACKED, LOST, REMOVED = 0, 1, 2, 3

    def __init__(self, box, score, kind, kf: KalmanBox):
        self.kf = kf
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(box))
        self.score = float(score)
        self.state = Track.NEW
        self.confirmed = False
        self.id = 0
        self.start = self.last = 0
        self.votes = {}
        self.vote(kind, score)
        self.box = np.asarray(box, float)

    def vote(self, kind, score):
        if kind is not None:
            self.votes[kind] = self.votes.get(kind, 0.0) + float(score)

    @property
    def kind(self):
        return max(self.votes, key=self.votes.get) if self.votes else None

    def predict(self):
        m = self.mean.copy()
        if self.state != Track.TRACKED:
            m[7] = 0.0                                   # the reference: a lost track's height velocity is zeroed
        self.mean, self.cov = self.kf.predict(m, self.cov)

    def predicted_box(self):
        return xyah_to_xyxy(self.mean)

    def update(self, box, score, kind, frame):
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(box))
        self.state = Track.TRACKED
        self.confirmed = True
        self.score = float(score)
        self.last = frame
        self.box = np.asarray(box, float)
        self.vote(kind, score)


class ByteTracker:
    """ByteTrack (module doc). Thresholds: high 0.6, low 0.1, new-track 0.7, first match 0.8, second 0.5,
    unconfirmed 0.7; buffer 30 frames at 30 frames/s, scaled by frame_rate."""

    def __init__(self, frame_rate: float = 30.0, high_thresh: float = 0.6, low_thresh: float = 0.1,
                 match_thresh: float = 0.8, second_thresh: float = 0.5, unconfirmed_thresh: float = 0.7,
                 buffer: int = 30, fuse_score: bool = True):
        self.high, self.low = float(high_thresh), float(low_thresh)
        self.new_thresh = self.high + 0.1
        self.match, self.second, self.unconf = float(match_thresh), float(second_thresh), float(unconfirmed_thresh)
        self.max_lost = max(1, int(round(frame_rate / 30.0 * buffer)))
        self.fuse = bool(fuse_score)
        self.kf = KalmanBox()
        self.tracks: list[Track] = []
        self.frame = 0
        self.next_id = 1

    def params(self) -> dict:
        return {'high_thresh': self.high, 'low_thresh': self.low, 'new_track_thresh': self.new_thresh,
                'match_thresh': self.match, 'second_thresh': self.second, 'unconfirmed_thresh': self.unconf,
                'max_lost_frames': self.max_lost, 'fuse_score': self.fuse}

    def _cost(self, tracks, boxes, scores, fuse):
        if not tracks or not len(boxes):
            return np.zeros((len(tracks), len(boxes)))
        iou = iou_matrix(np.array([t.predicted_box() for t in tracks]), boxes)
        if fuse:
            iou = iou * np.asarray(scores, float)[None, :]
        return 1.0 - iou

    def update(self, boxes, scores, kinds=None) -> list:
        self.frame += 1
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        scores = np.asarray(scores, float).reshape(-1)
        kinds = list(kinds) if kinds is not None else [None] * len(boxes)
        hi = np.flatnonzero(scores >= self.high)
        lo = np.flatnonzero((scores >= self.low) & (scores < self.high))
        confirmed = [t for t in self.tracks if t.confirmed and t.state in (Track.TRACKED, Track.LOST)]
        unconfirmed = [t for t in self.tracks if not t.confirmed and t.state == Track.NEW]
        for t in confirmed + unconfirmed:
            t.predict()
        out = []

        def hit(t, j):
            t.update(boxes[j], scores[j], kinds[j], self.frame)
            out.append((t, j))
        # 1. tracked and lost against the high detections
        m, um_t, um_d = assign(self._cost(confirmed, boxes[hi], scores[hi], self.fuse), self.match)
        for i, j in m:
            hit(confirmed[i], hi[j])
        rest_hi = [hi[j] for j in um_d]
        # 2. the still tracked ones against the low detections (plain IoU)
        pool = [confirmed[i] for i in um_t if confirmed[i].state == Track.TRACKED]
        m2, um2, _ = assign(self._cost(pool, boxes[lo], scores[lo], False), self.second)
        for i, j in m2:
            hit(pool[i], lo[j])
        matched2 = {id(pool[i]) for i, _ in m2}
        for i in um_t:
            t = confirmed[i]
            if t.state == Track.TRACKED and id(t) not in matched2:
                t.state = Track.LOST
        # 3. unconfirmed against the remaining high detections
        rh = np.array(rest_hi, int)
        m3, um3, um_d3 = assign(self._cost(unconfirmed, boxes[rh] if len(rh) else np.zeros((0, 4)),
                                           scores[rh] if len(rh) else np.zeros(0), self.fuse), self.unconf)
        for i, j in m3:
            hit(unconfirmed[i], rh[j])
            unconfirmed[i].confirmed = True
        for i in um3:
            unconfirmed[i].state = Track.REMOVED
        # 4. new tracks
        for j in (rh[k] for k in um_d3):
            if scores[j] < self.new_thresh:
                continue
            t = Track(boxes[j], scores[j], kinds[j], self.kf)
            t.id, self.next_id = self.next_id, self.next_id + 1
            t.start = t.last = self.frame
            if self.frame == 1:
                t.state, t.confirmed = Track.TRACKED, True
                out.append((t, j))
            self.tracks.append(t)
        # 5. lost too long: removed
        for t in self.tracks:
            if t.state == Track.LOST and self.frame - t.last > self.max_lost:
                t.state = Track.REMOVED
        self.tracks = [t for t in self.tracks if t.state != Track.REMOVED]
        return sorted(({'track': t.id, 'box': [float(v) for v in boxes[j]], 'score': float(scores[j]),
                        'kind': t.kind, 'det': int(j)} for t, j in out if t.confirmed), key=lambda r: r['track'])
