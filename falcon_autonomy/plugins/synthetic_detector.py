"""A known-weak SYNTHETIC detector: it reads the frame's ground truth and detects each object with a probability that
falls for small, occluded and fogged objects, jitters its box, now and then names the wrong class, and adds false
positives — all seeded, so the same frame always gets the same detections.

It is not a model and says nothing about any real detector. It exists to test the harness end to end and to give the
rare-case search a model with a known failure region (its weakness is the documented formula below), before a real
model is registered.

    p(detect) = 1 / (1 + exp(−(h_px − h50) / h_scale)) × (1 − occlusion_weight × occluded) × fog_factor
    fog_factor = exp(−range_m / visibility_m), visibility_m = 7000 / fog_density (CARLA's measured relation)
"""
from __future__ import annotations

import math
import zlib

import numpy as np


class SyntheticDetector:
    name = 'synthetic detector (ground truth with planted weaknesses)'
    synthetic = True

    def __init__(self, h50: float = 24.0, h_scale: float = 6.0, occlusion_weight: float = 0.8,
                 jitter: float = 0.06, confusion: float = 0.03, false_positives: float = 0.2, seed: int = 0):
        self.h50, self.h_scale, self.occ_w = float(h50), float(h_scale), float(occlusion_weight)
        self.jitter, self.confusion, self.fp = float(jitter), float(confusion), float(false_positives)
        self.seed = int(seed)

    def p_detect(self, o: dict, fog_density: float = 0.0) -> float:
        x0, y0, x1, y1 = o['bbox']
        h = max(y1 - y0, 0.0)
        p = 1.0 / (1.0 + math.exp(-(h - self.h50) / self.h_scale))
        p *= 1.0 - self.occ_w * float(o.get('occluded') or 0.0)
        if fog_density > 0:
            p *= math.exp(-float(o.get('range_m') or 0.0) / (7000.0 / fog_density))
        return max(min(p, 1.0), 0.0)

    def predict(self, rgb, meta: dict) -> list:
        objs = meta.get('objects') or []
        fog = float((meta.get('weather') or {}).get('fog_density', 0.0))
        key = f'{self.seed}:{meta.get("run", "")}:{meta.get("step", 0)}'
        rng = np.random.default_rng(zlib.crc32(key.encode()))
        H, W = rgb.shape[:2]
        classes = sorted({o['class'] for o in objs}) or ['car']
        dets = []
        for o in objs:
            if rng.random() >= self.p_detect(o, fog):
                continue
            x0, y0, x1, y1 = o['bbox']
            w, h = x1 - x0, y1 - y0
            j = rng.normal(0.0, self.jitter, 4) * np.array([w, h, w, h])
            b = [max(x0 + j[0], 0.0), max(y0 + j[1], 0.0), min(x1 + j[2], W), min(y1 + j[3], H)]
            cls = o['class'] if rng.random() >= self.confusion else classes[int(rng.integers(len(classes)))]
            dets.append({'bbox': b, 'class': cls, 'score': float(np.clip(0.5 + 0.5 * rng.random(), 0, 1))})
        for _ in range(int(rng.poisson(self.fp))):
            w, h = rng.uniform(10, W / 4), rng.uniform(10, H / 4)
            x, y = rng.uniform(0, W - w), rng.uniform(0, H - h)
            dets.append({'bbox': [x, y, x + w, y + h], 'class': classes[int(rng.integers(len(classes)))],
                         'score': float(rng.uniform(0.3, 0.6))})
        return dets
