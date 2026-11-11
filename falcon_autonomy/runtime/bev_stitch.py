#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sherin Joseph Roy. Licensed under the Apache License 2.0 (LICENSE in this folder), unlike the
# rest of Vantage (GPL-3.0-or-later): embed it in proprietary and production code freely.
"""Vantage bird's-eye stitcher for live cameras or image files. Needs only numpy and OpenCV.

Loads the remap tables Vantage exports (bev_lut.json + bev_lut.npz, in the parent folder by default) and stitches the
cameras into the bird's-eye view, balancing their exposure from the ground they share. With an OpenCV built with CUDA
(e.g. on a Jetson) --cuda does the remapping on the GPU.

    python3 bev_stitch.py --frames captures/ --out bev.png --balance
    python3 bev_stitch.py --source F=/dev/video0 --source R=/dev/video1 ... --show --balance
    python3 bev_stitch.py --source F="v4l2src device=/dev/video0 ! videoconvert ! appsink" ... --out bev.mp4

A source is anything cv2.VideoCapture opens: a V4L2 device, a GStreamer pipeline (contains '!'), an RTSP URL or a file.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

BODY_RGB = (38, 38, 36)
EMPTY_RGB = (20, 20, 20)
EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pgm', '.ppm')
GAIN_MIN, GAIN_MAX = 0.4, 2.5           # the exposure gains are clipped to this range


class Stitcher:
    def __init__(self, lut_dir, cuda=False):
        import cv2
        self.cv2 = cv2
        with open(os.path.join(lut_dir, 'bev_lut.json'), encoding='utf-8') as fh:
            self.meta = json.load(fh)
        with np.load(os.path.join(lut_dir, 'bev_lut.npz'), allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        self.w, self.h = self.meta['width'], self.meta['height']
        self.tags = [c['tag'] for c in self.meta['cameras']]
        self.sensor = {c['tag']: (c['sensor_width'], c['sensor_height']) for c in self.meta['cameras']}
        key = {c['tag']: os.path.splitext(c['map'])[0][:-4] for c in self.meta['cameras']}   # '<safe>_map.bin'
        self.file_tag = key
        self.map = {t: arrays[f'{key[t]}_map'] for t in self.tags}
        self.weight = {t: arrays[f'{key[t]}_weight'].astype(np.float32) for t in self.tags}
        self.valid = {t: self.map[t][..., 0] >= 0 for t in self.tags}
        self.body = arrays['body_mask'].astype(bool)
        self.cuda = cuda and hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
        self._maps = {}
        self.gains = {t: np.ones(3, np.float32) for t in self.tags}
        self.clipped = {}       # tag -> (3,) solved gains, before clipping, of the cameras the last estimate clipped

    def _maps_for(self, tag, w, h):
        k = (tag, w, h)
        if k not in self._maps:
            cv2 = self.cv2
            sw, sh = self.sensor[tag]
            m = self.map[tag]
            ok = self.valid[tag]
            mx = np.where(ok, m[..., 0] * (w / sw) - 0.5, -1).astype(np.float32)
            my = np.where(ok, m[..., 1] * (h / sh) - 0.5, -1).astype(np.float32)
            if self.cuda:
                self._maps[k] = (cv2.cuda_GpuMat(mx), cv2.cuda_GpuMat(my))
            else:
                self._maps[k] = cv2.convertMaps(mx, my, cv2.CV_16SC2)       # fixed point: same result, faster
        return self._maps[k]

    def warp(self, frames):
        """{tag: RGB uint8 frame} -> {tag: float32 (h, w, 3) on the ground grid}"""
        cv2 = self.cv2
        out = {}
        for t in self.tags:
            f = frames.get(t)
            if f is None:
                continue
            if f.ndim == 2:
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
            h, w = f.shape[:2]
            m1, m2 = self._maps_for(t, w, h)
            if self.cuda:
                g = cv2.cuda.remap(cv2.cuda_GpuMat(f), m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                out[t] = g.download().astype(np.float32)
            else:
                out[t] = cv2.remap(f, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
        return out

    def estimate_gains(self, warped, sigma_n=10.0, sigma_g=0.5, step=2):
        """Brown & Lowe gain compensation over the ground two cameras share (well exposed pixels only). Returns
        {tag: (3,) gains} clipped to GAIN_MIN..GAIN_MAX; self.clipped keeps the solved gains of the cameras clipped."""
        tags = [t for t in self.tags if t in warped]
        idx = {t: i for i, t in enumerate(tags)}
        n = len(tags)
        sl = (slice(None, None, step), slice(None, None, step))
        free = ~self.body[sl]
        pairs = []
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                m = self.valid[a][sl] & self.valid[b][sl] & free
                if m.sum() < 50:
                    continue
                A, B = warped[a][sl][m], warped[b][sl][m]
                la, lb = A.mean(1), B.mean(1)
                ok = (la > 8) & (la < 247) & (lb > 8) & (lb < 247)
                if ok.sum() >= 50:
                    pairs.append((idx[a], idx[b], int(ok.sum()), A[ok].mean(0), B[ok].mean(0)))
        g = np.ones((n, 3))
        for ch in range(3):
            M = np.zeros((n, n))
            r = np.zeros(n)
            for i, j, N, ma, mb in pairs:
                Ia, Ib = ma[ch], mb[ch]
                M[i, i] += N * (Ia * Ia / sigma_n ** 2 + 1 / sigma_g ** 2)
                M[j, j] += N * (Ib * Ib / sigma_n ** 2 + 1 / sigma_g ** 2)
                M[i, j] -= N * Ia * Ib / sigma_n ** 2
                M[j, i] -= N * Ia * Ib / sigma_n ** 2
                r[i] += N / sigma_g ** 2
                r[j] += N / sigma_g ** 2
            for i in range(n):
                if M[i, i] == 0:
                    M[i, i], r[i] = 1.0, 1.0
            g[:, ch] = np.linalg.solve(M, r)
        gc = np.clip(g, GAIN_MIN, GAIN_MAX)
        self.clipped = {t: g[idx[t]].astype(np.float32) for t in tags if np.any(gc[idx[t]] != g[idx[t]])}
        return {t: gc[idx[t]].astype(np.float32) for t in tags}

    def clipped_note(self):
        """'F 2.91 -> 2.50, QR 0.31 -> 0.40': the cameras the last estimate clipped, with the channel clipped most
        (solved -> applied), or '' when none were."""
        out = []
        for t, raw in self.clipped.items():
            raw = np.asarray(raw, float)
            ap = np.clip(raw, GAIN_MIN, GAIN_MAX)
            k = int(np.argmax(np.abs(np.log(np.maximum(raw, 1e-9) / ap))))
            out.append(f'{t} {raw[k]:.2f} -> {ap[k]:.2f}')
        return ', '.join(out)

    def stitch(self, frames, balance=False, estimate=True, smooth=1.0):
        """{tag: frame} -> RGB uint8 bird's-eye view. balance applies the exposure gains, estimate re-measures them
        first; smooth < 1 blends the new gains into the old ones (no flicker on video)."""
        warped = self.warp(frames)
        if balance and estimate:
            new = self.estimate_gains(warped)
            for t, g in new.items():
                self.gains[t] = (1 - smooth) * self.gains[t] + smooth * g
        out = np.zeros((self.h, self.w, 3), np.float32)
        wsum = np.zeros((self.h, self.w), np.float32)
        for t, img in warped.items():
            out += img * self.weight[t][..., None] * (self.gains[t] if balance else 1.0)
            wsum += self.weight[t]
        # normalise over the cameras that delivered a frame, so a dropped frame does not black out its sector
        covered = wsum > 0
        out /= np.where(covered, wsum, 1.0)[..., None]
        out[~covered] = EMPTY_RGB
        out[self.body] = BODY_RGB
        return np.clip(out, 0, 255).astype(np.uint8)


def _read(path):
    import cv2
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f'cannot read {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _frames_from_dir(folder, tags, file_tag=None):
    out = {}
    files = {os.path.splitext(f)[0].lower(): f for f in os.listdir(folder) if f.lower().endswith(EXTS)}
    for t in tags:
        # the camera tag, or the file-safe name the C++ stitcher and the remap files use
        f = files.get(t.lower()) or files.get((file_tag or {}).get(t, t).lower())
        if f:
            out[t] = _read(os.path.join(folder, f))
    return out


def _open(spec):
    import cv2
    if spec.isdigit():
        return cv2.VideoCapture(int(spec))
    if '!' in spec:
        return cv2.VideoCapture(spec, cv2.CAP_GSTREAMER)
    return cv2.VideoCapture(spec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--lut', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'),
                    help='folder with bev_lut.json and bev_lut.npz (default: the parent folder)')
    ap.add_argument('--frames', help='folder of still images named after the camera tags (F.png, QR.jpg ...)')
    ap.add_argument('--source', action='append', default=[], metavar='TAG=SPEC', help='live source for one camera')
    ap.add_argument('--out', help='output image (stills) or video file (live)')
    ap.add_argument('--show', action='store_true', help='show the stitched view in a window')
    ap.add_argument('--balance', action='store_true', help='balance exposure between the cameras')
    ap.add_argument('--every', type=int, default=15, help='live: re-estimate the gains every N frames')
    ap.add_argument('--frames-max', type=int, default=0, help='live: stop after N frames')
    ap.add_argument('--cuda', action='store_true', help='remap on the GPU when OpenCV has CUDA')
    a = ap.parse_args(argv)
    import cv2
    st = Stitcher(a.lut, cuda=a.cuda)
    print(f'{len(st.tags)} cameras ({", ".join(st.tags)}) · {st.w} x {st.h} px · '
          f'{"CUDA" if st.cuda else "CPU"} remap', file=sys.stderr)
    if a.frames:
        frames = _frames_from_dir(a.frames, st.tags, st.file_tag)
        missing = [t for t in st.tags if t not in frames]
        if missing:
            print(f'no image for {", ".join(missing)}', file=sys.stderr)
        t0 = time.perf_counter()
        img = st.stitch(frames, balance=a.balance)
        print(f'stitched in {(time.perf_counter() - t0) * 1000:.1f} ms', file=sys.stderr)
        if a.balance:
            for t, g in st.gains.items():
                print(f'gain {t:6s} {g[0]:.4f} {g[1]:.4f} {g[2]:.4f}')
            if st.clipped:
                print(f'warning: gains clipped to {GAIN_MIN}-{GAIN_MAX}: {st.clipped_note()}', file=sys.stderr)
        if a.out:
            ok, buf = cv2.imencode(os.path.splitext(a.out)[1] or '.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            buf.tofile(a.out)
        if a.show:
            cv2.imshow('Vantage bird\'s-eye', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
        return 0
    if not a.source:
        ap.error('give --frames DIR or one --source TAG=SPEC per camera')
    caps = {}
    for s in a.source:
        tag, _, spec = s.partition('=')
        if tag not in st.tags:
            ap.error(f'{tag}: not a camera in the tables ({", ".join(st.tags)})')
        caps[tag] = _open(spec)
        if not caps[tag].isOpened():
            raise SystemExit(f'cannot open {spec}')
    writer = None
    n = 0
    warned = set()
    t0 = time.perf_counter()
    try:
        while True:
            frames = {}
            for t, c in caps.items():
                ok, f = c.read()
                if ok:
                    frames[t] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            if not frames:
                break
            img = st.stitch(frames, balance=a.balance, estimate=n % max(1, a.every) == 0, smooth=1.0 if n == 0 else 0.3)
            if set(st.clipped) != warned:         # once per change, not on every estimate
                warned = set(st.clipped)
                if warned:
                    print(f'warning: frame {n}: gains clipped to {GAIN_MIN}-{GAIN_MAX}: {st.clipped_note()}',
                          file=sys.stderr)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if a.out:
                if writer is None:
                    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*'mp4v'), 30, (st.w, st.h))
                writer.write(bgr)
            if a.show:
                cv2.imshow('Vantage bird\'s-eye', bgr)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
            n += 1
            if a.frames_max and n >= a.frames_max:
                break
    finally:
        for c in caps.values():
            c.release()
        if writer is not None:
            writer.release()
    dt = time.perf_counter() - t0
    print(f'{n} frames in {dt:.1f} s ({n / max(dt, 1e-9):.1f} fps)', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
