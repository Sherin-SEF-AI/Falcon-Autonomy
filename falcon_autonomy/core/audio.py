"""Synthetic directional audio for a recorded drive (schema falcon.audio/1): a multi-channel microphone array on the ego
hearing the run's vehicles, sirens and horns, with per-frame ground truth. numpy only. **SYNTHETIC**: CARLA 0.9.16 has
no audio; every sound here is procedural, from cited source models and labelled heuristics, never a recording.

    derive_audio(run_dir, out_dir=None, options) -> manifest      OUT/audio.wav, ground_truth.jsonl, events.jsonl,
                                                                  audio.json (and doa.json), OUT default RUN/audio
    passby_check(...) -> dict                                     the physics check on a synthetic pass-by

Frames and units (core.rig): CARLA's — x forward, y right, z up, LEFT-handed — metres, seconds, degrees. Azimuth in the
array's frame: 0° straight ahead (+x), positive to the RIGHT (+y), as CARLA's yaw; elevation positive up. Sample values
of the WAV are sound pressure in pascals (float32), so a level is 20·lg(rms / 20 µPa) dB SPL.

Sources (each a point source; "Pa at 1 m" = the free-field pressure 1 m away):
* Vehicles — CNOSSOS-EU road-traffic emission (Commission Directive (EU) 2015/996 Annex, Appendix F, with the
  coefficients of the 2020 amendment, Commission Delegated Directive (EU) 2021/1226; transcribed from NoiseModelling's
  RoadCnossos_2020.json, Université Gustave Eiffel, GPL-3.0): per octave band i (63 Hz–8 kHz) rolling noise
  L_WR = A_R + B_R·lg(v / 70 km/h) and propulsion noise L_WP = A_P + B_P·(v − 70) / 70, summed energetically, a sound
  power level per vehicle; Lp(1 m) = L_W − 10·lg(4π). Categories: 1 light (cars, vans ≤ 3.5 t), 2 medium heavy (two-axle
  buses, the ambulance), 3 heavy (lorries), 4b powered two-wheelers > 50 cc (propulsion only). Propulsion is held at
  20 km/h below the model's 20–130 km/h range (an idling engine is not silent; Falcon's choice). Acceleration:
  IMAGINE's correction ΔL_WP = C_P·a for a ≥ −1 m/s² (C_P·(−1) below), a ≤ a_max = 2 (cat. 1), 1 (cat. 2, 3), 4 (cat. 4)
  m/s² (Peeters & van Blokland 2007, IMAGINE Deliverable D11, §4.3.1); D11 gives C_P per band only as a graph without
  an absolute scale, so Falcon applies C_P = 3.2 dB per m/s² to every band — the slope of D11's figure 18 (category 1,
  microphone 1), labelled. Source height 0.05 m (CNOSSOS-EU), two-wheelers and the autorickshaw 0.30 m (IMAGINE D11
  §2.1.3). Spectrum inside a band: the band levels interpolated in lg f, renormalised to each band's exact power.
  Engine orders (heuristic): half of the propulsion power of the 63–1000 Hz bands is moved into harmonics of the firing
  frequency (rpm/60 × cylinders/2, a four-stroke), the first 24 orders up to 2 kHz, amplitudes ∝ 1/k; rpm from speed
  through a gear model per class (ENGINES). Autorickshaw (no CNOSSOS class): category 4b propulsion (a single-cylinder
  engine) and category 1 rolling × 3/4 (three tyres) — heuristic. Bicycles and walkers are silent. Every motor vehicle has a combustion engine
  (CARLA's electric cars too).
* Sirens on CARLA's emergency vehicles (SIRENS): a band-limited square-ish wave (odd harmonics 1/k) sweeping
  650–1550 Hz, wail at 12 cycles/min and yelp at 200 cycles/min — inside the US federal ambulance specification
  KKK-A-1822 (fundamental 500–2000 Hz, wail 10–18, yelp 150–250 cycles/min); the sweep's ends are Falcon's choice.
  Level 123 dB(A) at 3 m (KKK-A-1822's minimum, on axis), radiated omnidirectionally (no horn directivity: an
  overestimate beside and behind the vehicle). On the box's top centre.
* Horns: CARLA has none. A seeded rule set (HORN_RULES, heuristic, labelled) honks when a vehicle closes on a slower
  vehicle ahead (overtaking or asking the way), has a pedestrian ahead, meets crossing traffic, or waits in a queue; the
  Indian profile honks several times more than the sparse one, two-wheelers and autorickshaws most. Explicit events may
  be given (option horn_events). Tones (HORNS): a pulse-like wave (harmonics 1/k to 6 kHz), one or two notes; levels
  inside AIS-014 §6.1.2 (measured 7 m ahead: 83–112 dB(A) for two- and three-wheelers ≤ 7 kW, 93–112 dB(A) for the
  rest; the chosen values are Falcon's). At the box's front, 0.6 m up.

Propagation, source → each microphone (no reverberation; no ego body):
* Retarded time: the microphone at time t hears what the source emitted at τ with t − τ = |m(t) − p(τ)| / c, solved by
  Newton's method on a 1 ms grid (each source's path a C1 cubic through actors.jsonl's poses, every simulation step)
  and interpolated linearly to every sample; the emitted signal is read at τ with 4-point cubic interpolation. The
  Doppler shift is not applied: it emerges from reading the emission at τ(t) (instantaneous factor dτ/dt =
  (1 − n·v_mic/c) / (1 − n·v_src/c)). c = 343.2·√(T / 293.15 K) m/s (ISO 9613-1).
* Spherical spreading 1/r. The convective amplification of a moving source ((1 − M_r)^−n) is not modelled (≤ 1 dB at
  20 m/s).
* Air absorption: ISO 9613-1:1993 (pure-tone attenuation coefficient α(f, T, RH, p)), applied per bin on the emitted
  signal's 21 ms frames at the frame's distance (the mic separation, < 1 m, changes it by < 0.1 dB at 8 kHz).
* Ground: one specular reflection from an image source mirrored in the flat plane through the ego's ground (the ego's
  origin height) with a real reflection factor (default 0.95: hard, asphalt-like; heuristic — no ground impedance or
  spherical-wave ground wave), its own retarded time and spreading, so its comb filtering and Doppler emerge too.
* Occlusion by the other vehicles' boxes (actors.jsonl, the ego excepted): when the source–array line passes through a
  box, diffraction over its top edge with ISO 9613-2 eq. 14's barrier attenuation D_z = 10·lg(3 + 20·z/λ) (z the path
  difference over the edge, single diffraction, capped at 20 dB); the most attenuating box only. Buildings are not
  modelled (the run does not record their geometry).
* The array hears its microphones' self-noise: uncorrelated white noise per channel (default 29 dB SPL, a typical MEMS
  microphone's; Falcon's default, not a product's).

Ground truth (ground_truth.jsonl, one line per simulation step from t = 0): every source whose received level at the
array is at least `audible_db`: actor id, type, class, kind (engine, siren, horn), azimuth and elevation in the array
frame and distance — of where the sound now arriving left the source (its retarded position) — received level (dB SPL,
and an A-weighted estimate from its emitted spectrum), the instantaneous Doppler factor at the array's centre.

Deterministic: every random draw comes from a generator seeded by (seed, actor id, purpose); the same run, options and
seed give the same samples bit for bit.
"""
from __future__ import annotations

import json
import math
import os
import struct
import time

import numpy as np

SCHEMA = 'falcon.audio/1'
MANIFEST = 'audio.json'
P_REF = 20e-6
OCTAVES = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], float)
V_REF_KMH = 70.0
LP_FROM_LW_DB = 10.0 * math.log10(4.0 * math.pi)        # point source, free field: Lp(1 m) = L_W − 10.99 dB

# CNOSSOS-EU road vehicle coefficients (Directive (EU) 2015/996, Appendix F, Table F-1) as amended in 2020 (Commission
# Delegated Directive (EU) 2021/1226), transcribed by NoiseModelling (RoadCnossos_2020.json): A_R, B_R, A_P, B_P per
# octave band, 63 Hz … 8 kHz
CNOSSOS = {
    '1': {'ar': [83.1, 89.2, 87.7, 93.1, 100.1, 96.7, 86.8, 76.2],
          'br': [30.0, 41.5, 38.9, 25.7, 32.5, 37.2, 39.0, 40.0],
          'ap': [97.9, 92.5, 90.7, 87.2, 84.7, 88.0, 84.4, 77.1],
          'bp': [-1.3, 7.2, 7.7, 8.0, 8.0, 8.0, 8.0, 8.0]},
    '2': {'ar': [88.7, 93.2, 95.7, 100.9, 101.7, 95.1, 87.8, 83.6],
          'br': [30.0, 35.8, 32.6, 23.8, 30.1, 36.2, 38.3, 40.1],
          'ap': [105.5, 100.2, 100.5, 98.7, 101.0, 97.8, 91.2, 85.0],
          'bp': [-1.9, 4.7, 6.4, 6.5, 6.5, 6.5, 6.5, 6.5]},
    '3': {'ar': [91.7, 96.2, 98.2, 104.9, 105.1, 98.5, 91.1, 85.6],
          'br': [30.0, 33.5, 31.3, 25.4, 31.8, 37.1, 38.6, 40.6],
          'ap': [108.8, 104.2, 103.5, 102.9, 102.6, 98.5, 93.8, 87.5],
          'bp': [0.0, 3.0, 4.6, 5.0, 5.0, 5.0, 5.0, 5.0]},
    '4b': {'ar': None, 'br': None,                       # powered two-wheelers > 50 cc: propulsion only
           'ap': [99.9, 101.9, 96.7, 94.4, 95.2, 94.7, 92.1, 88.6],
           'bp': [3.2, 5.9, 11.9, 11.6, 11.5, 12.6, 11.1, 12.0]},
}
ACCEL = {'c_p_db_per_mps2': 3.2, 'a_min': -1.0, 'a_max': {'1': 2.0, '2': 1.0, '3': 1.0, '4b': 4.0}}

# class → emission model. rolling/propulsion: CNOSSOS category; rolling_scale: a power factor (heuristic); h: source
# height (m); engine: ENGINES key; horn: HORNS key
CLASSES = {
    'car': {'propulsion': '1', 'rolling': '1', 'h': 0.05, 'engine': 'petrol4', 'horn': 'car'},
    'van': {'propulsion': '1', 'rolling': '1', 'h': 0.05, 'engine': 'diesel4', 'horn': 'car'},
    'bus': {'propulsion': '2', 'rolling': '2', 'h': 0.05, 'engine': 'diesel6', 'horn': 'air'},
    'truck': {'propulsion': '3', 'rolling': '3', 'h': 0.05, 'engine': 'diesel6', 'horn': 'air'},
    'two_wheeler': {'propulsion': '4b', 'rolling': None, 'h': 0.30, 'engine': 'single_hi', 'horn': 'two_wheeler'},
    'autorickshaw': {'propulsion': '4b', 'rolling': '1', 'rolling_scale': 0.75, 'h': 0.30, 'engine': 'single_lo',
                     'horn': 'autorickshaw'},
    'bicycle': None,
}
# blueprint → (class, CNOSSOS category for both components when it differs from the class's)
TYPES = {
    'vehicle.ford.ambulance': ('van', '2'),               # a cutaway ambulance is over 3.5 t: medium heavy
    'vehicle.carlamotors.firetruck': ('truck', None), 'vehicle.carlamotors.carlacola': ('truck', None),
    'vehicle.carlamotors.european_hgv': ('truck', None), 'vehicle.mitsubishi.fusorosa': ('bus', None),
    'vehicle.tesla.cybertruck': ('car', None), 'vehicle.mercedes.sprinter': ('van', None),
    'vehicle.volkswagen.t2': ('van', None), 'vehicle.volkswagen.t2_2021': ('van', None),
}
# engine heuristics (Falcon's, labelled): cylinders, idle rpm, rpm across each gear, each gear's top speed (km/h)
ENGINES = {
    'petrol4': {'cyl': 4, 'idle': 800, 'rpm': (1300, 2800), 'gears': (20, 40, 60, 85, 115, 170)},
    'diesel4': {'cyl': 4, 'idle': 750, 'rpm': (1200, 2600), 'gears': (18, 35, 55, 80, 110, 150)},
    'diesel6': {'cyl': 6, 'idle': 600, 'rpm': (900, 1900), 'gears': (10, 20, 32, 45, 60, 80, 100)},
    'single_hi': {'cyl': 1, 'idle': 1400, 'rpm': (2500, 6500), 'gears': (15, 30, 45, 60, 80, 120)},
    'single_lo': {'cyl': 1, 'idle': 1300, 'rpm': (2000, 4500), 'gears': (15, 28, 42, 65)},   # a Bajaj RE-like single
}
TONAL_SHARE = 0.5               # heuristic: share of the 63–1000 Hz propulsion power in engine orders
ORDERS = 24                     # firing-frequency harmonics synthesised (1/k amplitudes: 97 % of their power)
SIRENS = {'vehicle.ford.ambulance': 'wail_yelp', 'vehicle.carlamotors.firetruck': 'wail',
          'vehicle.dodge.charger_police': 'yelp', 'vehicle.dodge.charger_police_2020': 'yelp'}
SIREN = {'f_lo': 650.0, 'f_hi': 1550.0, 'wail_cpm': 12.0, 'yelp_cpm': 200.0, 'alternate_s': (8.0, 4.0),
         'dba_at_3m': 123.0, 'harmonics': (1, 3, 5, 7, 9, 11, 13)}
HORNS = {   # notes (Hz), dB(A) 7 m ahead (inside AIS-014 §6.1.2's range for the class; the value is Falcon's)
    'car': {'notes': (420.0, 510.0), 'dba_at_7m': 98.0},
    'air': {'notes': (300.0, 370.0), 'dba_at_7m': 105.0},
    'two_wheeler': {'notes': (400.0,), 'dba_at_7m': 88.0},
    'autorickshaw': {'notes': (380.0,), 'dba_at_7m': 88.0},
}
HORN_RULES = {   # hazard rates (honks per second while the condition holds), heuristic, labelled
    'slow_leader': 0.35, 'pedestrian_ahead': 0.5, 'crossing_traffic': 0.3, 'queue': 0.06,
}
HORN_CLASS_RATE = {'two_wheeler': 1.0, 'autorickshaw': 1.0, 'bus': 0.8, 'truck': 0.8, 'car': 0.5, 'van': 0.5}
HORN_PROFILES = {'indian': 1.0, 'sparse': 0.15, 'none': 0.0}
HORN_TIMING = {'refractory_s': 2.5, 'dur_s': (0.12, 0.45), 'double_p': 0.35, 'gap_s': (0.08, 0.15),
               'second_s': (0.10, 0.30)}

DEFAULTS = {
    'sample_rate': 48000, 'format': 'float32', 'full_scale_pa': 200.0,
    'temperature_c': 20.0, 'humidity_pct': 70.0, 'pressure_kpa': 101.325,
    'air_absorption': True, 'ground': True, 'ground_reflection': 0.95, 'occlusion': True,
    'max_range_m': 200.0, 'audible_db': 30.0, 'mic_self_noise_db': 29.0,
    'horns': 'auto', 'horn_events': [], 'sirens': 'on', 'include_ego': False,
    'array': None, 'seed': None, 'duration_s': None, 'doa': True,
}
STFT_N = 1024
COARSE = 48                     # samples per step of the retarded-time grid (1 ms at 48 kHz)
FADE_S = 0.02                   # fade where a source appears, disappears or jumps
RANGE_FADE_S = 0.3              # fade where a source crosses max_range


# ============================================================================================ physics
def speed_of_sound(temp_c: float) -> float:
    """m/s, ISO 9613-1: c = 343.2·√(T / T0), T0 = 293.15 K."""
    return 343.2 * math.sqrt((temp_c + 273.15) / 293.15)


def air_absorption_db_per_m(f, temp_c=20.0, humidity_pct=70.0, pressure_kpa=101.325):
    """ISO 9613-1:1993 pure-tone atmospheric attenuation coefficient α (dB/m) at frequencies f (Hz)."""
    T, T0, T01, pr = temp_c + 273.15, 293.15, 273.16, 101.325
    pa = pressure_kpa / pr
    h = humidity_pct * 10.0 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151) / pa      # molar concentration of water, %
    fro = pa * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frn = pa * (T / T0) ** -0.5 * (9.0 + 280.0 * h * math.exp(-4.170 * ((T / T0) ** (-1.0 / 3.0) - 1.0)))
    f = np.asarray(f, float)
    return 8.686 * f * f * (1.84e-11 / pa * (T / T0) ** 0.5 + (T / T0) ** -2.5 * (
        0.01275 * math.exp(-2239.1 / T) / (fro + f * f / fro) + 0.1068 * math.exp(-3352.0 / T) / (frn + f * f / frn)))


def a_weighting_db(f):
    """IEC 61672-1 A-weighting (dB) at frequencies f (Hz)."""
    f2 = np.asarray(f, float) ** 2
    ra = 12194.0 ** 2 * f2 * f2 / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) *
                                   (f2 + 12194.0 ** 2))
    with np.errstate(divide='ignore'):
        return 20.0 * np.log10(np.maximum(ra, 1e-30)) + 2.0


def barrier_db(z_m, f, c):
    """ISO 9613-2 eq. 14, single diffraction, K_met = 1: D_z = 10·lg(3 + 20·z/λ), capped at 20 dB; 0 where z ≤ 0."""
    z = np.asarray(z_m, float)[..., None]
    d = 10.0 * np.log10(3.0 + 20.0 * np.maximum(z, 0.0) * np.asarray(f, float) / c)
    return np.where(z > 0, np.minimum(d, 20.0), 0.0)


def cnossos_band_power(cat_p: str, cat_r: str | None, v_mps, a_mps2, rolling_scale=1.0):
    """(rolling, propulsion) octave-band pressure powers at 1 m (Pa², shape (..., 8)) of one vehicle at speeds v (m/s)
    and accelerations a (m/s²): CNOSSOS-EU (module doc) with IMAGINE's acceleration correction."""
    v = np.atleast_1d(np.asarray(v_mps, float)) * 3.6
    a = np.atleast_1d(np.asarray(a_mps2, float))
    cp = CNOSSOS[cat_p]
    vp = np.clip(v, 20.0, 130.0)[..., None]
    acc = ACCEL['c_p_db_per_mps2'] * np.clip(a, ACCEL['a_min'], ACCEL['a_max'][cat_p])[..., None]
    lwp = np.asarray(cp['ap']) + np.asarray(cp['bp']) * (vp - V_REF_KMH) / V_REF_KMH + acc
    prop = P_REF ** 2 * 10.0 ** ((lwp - LP_FROM_LW_DB) / 10.0)
    if cat_r is None or CNOSSOS[cat_r]['ar'] is None:
        return np.zeros_like(prop), prop
    cr = CNOSSOS[cat_r]
    vr = np.clip(v, 0.5, 130.0)[..., None]
    lwr = np.asarray(cr['ar']) + np.asarray(cr['br']) * np.log10(vr / V_REF_KMH)
    roll = rolling_scale * P_REF ** 2 * 10.0 ** ((lwr - LP_FROM_LW_DB) / 10.0)
    return roll, prop


def rpm_of(engine: dict, v_mps):
    """Engine speed (rpm) from road speed through the class's gear model (heuristic, ENGINES)."""
    v = np.asarray(v_mps, float) * 3.6
    tops = np.asarray(engine['gears'], float)
    lo, hi = engine['rpm']
    k = np.minimum(np.searchsorted(tops, v), len(tops) - 1)
    v0 = np.where(k > 0, tops[np.maximum(k - 1, 0)], 0.0)
    r0 = np.where(k > 0, lo, engine['idle'])
    frac = np.clip((v - v0) / (tops[k] - v0), 0.0, 1.0)
    return np.where(v < 1.0, engine['idle'], r0 + (hi - r0) * frac)


def classify(row: dict) -> tuple[str | None, str | None, str | None]:
    """(class, CNOSSOS category override, siren pattern) of an actors.jsonl row; class None: silent."""
    tid, cat, sub = str(row.get('type_id', '')), str(row.get('category', '')), str(row.get('subclass') or '')
    siren = SIRENS.get(tid) or ('wail' if cat.startswith('vehicle.emergency') else None)
    if 'autorickshaw' in sub:
        return 'autorickshaw', None, None
    if 'rickshaw' in sub or not tid.startswith('vehicle.'):
        return None, None, None
    if tid in TYPES:
        c, over = TYPES[tid]
        return c, over, siren
    if cat == 'vehicle.motorcycle':
        return 'two_wheeler', None, siren
    if cat == 'vehicle.bicycle':
        return 'bicycle', None, None
    if cat.startswith('vehicle.bus'):
        return 'bus', None, siren
    if cat == 'vehicle.truck':
        return 'truck', None, siren
    return 'car', None, siren


# ============================================================================================ geometry
class Track:
    """A C1 piecewise cubic through samples (t_k, y_k) (Hermite with Catmull-Rom tangents: central differences, one-
    sided at the ends), extended linearly outside [t_0, t_K]. y may be (K,) or (K, d)."""

    def __init__(self, t, y):
        self.t = np.asarray(t, float)
        y = np.asarray(y, float)
        self.y = y.reshape(len(self.t), -1)
        K = len(self.t)
        m = np.zeros_like(self.y)
        if K > 1:
            m[1:-1] = (self.y[2:] - self.y[:-2]) / (self.t[2:] - self.t[:-2])[:, None]
            m[0] = (self.y[1] - self.y[0]) / (self.t[1] - self.t[0])
            m[-1] = (self.y[-1] - self.y[-2]) / (self.t[-1] - self.t[-2])
        self.m = m

    def __call__(self, tq, deriv: bool = False):
        tq = np.asarray(tq, float)
        t, y, m = self.t, self.y, self.m
        if len(t) == 1:
            out = np.broadcast_to(y[0], tq.shape + (y.shape[1],)).copy()
            return (out, np.zeros_like(out)) if deriv else out
        i = np.clip(np.searchsorted(t, tq, 'right') - 1, 0, len(t) - 2)
        h = (t[i + 1] - t[i])[..., None]
        s = np.clip((tq - t[i])[..., None] / h, 0.0, 1.0)
        s2, s3 = s * s, s * s * s
        p = ((2 * s3 - 3 * s2 + 1) * y[i] + (s3 - 2 * s2 + s) * h * m[i] + (-2 * s3 + 3 * s2) * y[i + 1] +
             (s3 - s2) * h * m[i + 1])
        dp = ((6 * s2 - 6 * s) * y[i] + (3 * s2 - 4 * s + 1) * h * m[i] + (-6 * s2 + 6 * s) * y[i + 1] +
              (3 * s2 - 2 * s) * h * m[i + 1]) / h
        lo, hi = tq < t[0], tq > t[-1]
        if lo.any():
            p[lo] = y[0] + m[0] * (tq[lo] - t[0])[:, None]
            dp[lo] = m[0]
        if hi.any():
            p[hi] = y[-1] + m[-1] * (tq[hi] - t[-1])[:, None]
            dp[hi] = m[-1]
        return (p, dp) if deriv else p


def rot_z(yaw_deg):
    """(..., 3, 3) rotations about z (CARLA yaw, degrees)."""
    a = np.radians(np.asarray(yaw_deg, float))
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1], R[..., 2, 2] = c, -s, s, c, 1.0
    return R


def carla_rotations(roll, pitch, yaw):
    """(..., 3, 3) CARLA rotation matrices (core.rig.carla_rotation, vectorised)."""
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    R = np.empty(np.shape(yaw) + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 0, 2] = cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr
    R[..., 1, 0], R[..., 1, 1], R[..., 1, 2] = sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr
    R[..., 2, 0], R[..., 2, 1], R[..., 2, 2] = sp, -cp * sr, cp * cr
    return R


def array_geometry(sensor: dict | None) -> tuple[dict, np.ndarray]:
    """(pose in the vehicle frame, mic positions (M, 3) in the array's frame) of an audio_array sensor (core.rig)."""
    from .rig import DEFAULTS as RIG_DEFAULTS, POSE0
    s = dict(RIG_DEFAULTS['audio_array'], **(sensor or {}))
    pose = dict(POSE0, **(s.get('pose') or {'z': 1.55}))
    if s.get('mic_positions'):
        mics = np.asarray(s['mic_positions'], float).reshape(-1, 3)
    else:
        n, r = int(s['mics']), float(s['radius'])
        a = 2.0 * np.pi * np.arange(n) / n
        mics = np.stack([r * np.cos(a), r * np.sin(a), np.zeros(n)], axis=1)     # mic 0 ahead, then to the right
    if len(mics) < 2:
        raise ValueError('an audio array needs at least two microphones')
    return pose, mics


# ============================================================================================ signals
class Stft:
    """Short-time Fourier transform with √Hann analysis and synthesis windows at 50 % overlap: unmodified, the inverse
    reconstructs the input exactly; frame i is centred on sample i·hop."""

    def __init__(self, n: int = STFT_N, fs: int = 48000):
        self.n, self.hop, self.fs = n, n // 2, fs
        self.win = np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n))
        self.f = np.fft.rfftfreq(n, 1.0 / fs)

    def frames(self, length: int) -> int:
        return length // self.hop + 1

    def forward(self, x):
        F = self.frames(len(x))
        xp = np.zeros((F + 1) * self.hop)
        xp[self.hop:self.hop + len(x)] = x
        fr = np.lib.stride_tricks.sliding_window_view(xp, self.n)[::self.hop][:F]
        return np.fft.rfft(fr * self.win, axis=1)

    def inverse(self, X, length: int):
        fr = np.fft.irfft(X, self.n, axis=1) * self.win
        F = len(fr)
        y = np.zeros((F + 1) * self.hop)
        y[:F * self.hop] += fr[:, :self.hop].reshape(-1)
        y[self.hop:(F + 1) * self.hop] += fr[:, self.hop:].reshape(-1)
        return y[self.hop:self.hop + length]


def band_shape(stft: Stft) -> dict:
    """Per-bin interpolation of octave-band levels in lg f, and each bin's band (−1 outside 44.7 Hz–11.3 kHz)."""
    f = np.maximum(stft.f, 1.0)
    lf = np.log10(f)
    lc = np.log10(OCTAVES)
    j = np.clip(np.searchsorted(lc, lf) - 1, 0, len(lc) - 2)
    w = np.clip((lf - lc[j]) / (lc[j + 1] - lc[j]), 0.0, 1.0)
    w = np.where(lf < lc[0], 0.0, np.where(lf > lc[-1], 1.0, w))
    edge_lo, edge_hi = OCTAVES / math.sqrt(2.0), OCTAVES * math.sqrt(2.0)
    band = np.full(len(f), -1)
    for b in range(len(OCTAVES)):
        band[(stft.f >= edge_lo[b]) & (stft.f < edge_hi[b])] = b
    roll = np.where(stft.f < edge_lo[0], (np.maximum(stft.f, 1.0) / edge_lo[0]) ** 4, 1.0) * \
        np.where(stft.f >= edge_hi[-1], (edge_hi[-1] / np.maximum(stft.f, 1.0)) ** 4, 1.0)
    return {'j': j, 'w': w, 'band': band, 'bw': edge_hi - edge_lo, 'roll': roll}


def shaped_psd(P_band, shape: dict, df: float):
    """(frames, bins) one-sided PSD (Pa²/Hz) whose power in each octave band equals P_band (frames, 8) exactly, shaped
    by lg-f interpolation of the band densities; tapered steeply outside the eight bands."""
    dens = np.log10(np.maximum(P_band / shape['bw'], 1e-30))
    j, w = shape['j'], shape['w']
    S = 10.0 ** ((1.0 - w) * dens[:, j] + w * dens[:, j + 1]) * shape['roll']
    band = shape['band']
    for b in range(P_band.shape[1]):
        sel = band == b
        have = S[:, sel].sum(axis=1) * df
        S[:, sel] *= (P_band[:, b] / np.maximum(have, 1e-40))[:, None]
    return S


def harmonic_power_a(freqs, amps) -> float:
    """A-weighted mean-square (Pa²) of a sum of sinusoids of the given frequencies and amplitudes."""
    return float(np.sum(0.5 * np.asarray(amps, float) ** 2 * 10.0 ** (a_weighting_db(freqs) / 10.0)))


# ============================================================================================ sources
class Source:
    """One point source: an actor's engine, siren or horn, or a test tone. track: the actor's path (Track of x y z
    yaw), offset: the source's position in the actor's frame (x forward, y right, z up from the actor's origin);
    intervals: [(t0, t1)] when it emits (s)."""

    def __init__(self, actor, type_id, cls, kind, track, offset, intervals, **params):
        self.actor, self.type_id, self.cls, self.kind = actor, type_id, cls, kind
        self.track, self.offset = track, np.asarray(offset, float)
        self.intervals = list(intervals)
        self.params = params

    def pos(self, tau, deriv=False):
        """World position (and velocity, the translation's) at times tau."""
        if deriv:
            p, dp = self.track(tau, deriv=True)
        else:
            p, dp = self.track(tau), None
        off = self.offset
        if np.any(off[:2]):
            R = rot_z(p[..., 3])
            q = p[..., :3] + np.einsum('...ij,j->...i', R, off)
        else:
            q = p[..., :3] + off
        return (q, dp[..., :3]) if deriv else q


def _rng(seed, actor, purpose: int):
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, int(actor) & 0xFFFFFFFF, purpose])


def _fade(n: int, fs: int, fade_in: float, fade_out: float):
    env = np.ones(n)
    for k, sec in ((0, fade_in), (1, fade_out)):
        m = min(int(sec * fs), n // 2)
        if m > 0:
            ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(m) / m)
            if k == 0:
                env[:m] *= ramp
            else:
                env[-m:] *= ramp[::-1]
    return env


def _kinematics(src: Source, tau):
    """Speed (m/s, smoothed) and acceleration (m/s², over 0.5 s) of the source's actor at times tau (uniform)."""
    _, dp = src.track(tau, deriv=True)
    v = np.linalg.norm(dp[:, :3], axis=1)
    dt = tau[1] - tau[0] if len(tau) > 1 else 1.0
    k = max(int(round(0.5 / dt)), 1)
    if len(v) > 2 * k:
        ker = np.ones(k) / k
        vs = np.convolve(np.pad(v, (k, k), mode='edge'), ker, mode='same')[k:-k]
    else:
        vs = v
    return vs, np.gradient(vs, dt) if len(vs) > 1 else np.zeros_like(vs)


def emit(src: Source, t0: float, t1: float, fades, fs: int, stft: Stft, shape, rng):
    """The source's emitted pressure at 1 m on the grid t0 + n / fs, before propagation, as its STFT (so the caller
    filters it once more for propagation): (X (frames × bins), frame centres (s), samples, fade envelope)."""
    n = int(math.ceil((t1 - t0) * fs)) + 1
    tau = t0 + np.arange(n) / fs
    F = stft.frames(n)
    tf = t0 + np.arange(F) * stft.hop / fs
    tonal = np.zeros(n)
    P_noise = None
    p = src.params
    if src.kind == 'engine':
        m = CLASSES[src.cls]
        cat_p = p.get('cnossos') or m['propulsion']
        cat_r = p.get('cnossos') or m['rolling']
        vf, af = _kinematics(src, tf)
        roll, prop = cnossos_band_power(cat_p, cat_r, vf, af, m.get('rolling_scale', 1.0))
        low = OCTAVES <= 1000.0
        Pt = TONAL_SHARE * prop[:, low].sum(axis=1)
        prop[:, low] *= 1.0 - TONAL_SHARE
        P_noise = roll + prop
        eng = ENGINES[m['engine']]
        ctrl = t0 + np.arange(int((t1 - t0) * 100) + 2) * 0.01
        vc, _ = _kinematics(src, ctrl)
        f_fire = np.interp(tau, ctrl, rpm_of(eng, vc) / 60.0 * eng['cyl'] / 2.0)
        z1 = np.exp(2j * np.pi * np.cumsum(f_fire) / fs)            # the firing phase; order k: z1**k, by recursion
        amp = np.sqrt(2.0 * np.interp(tau, tf, Pt))
        ks = np.arange(1, ORDERS + 1)
        w = 1.0 / ks
        w /= np.sqrt(np.sum(w ** 2))
        ph0 = np.exp(1j * rng.uniform(0, 2 * np.pi, len(ks)))
        zk = np.ones(n, complex)
        for k, wk, p0 in zip(ks, w, ph0):
            zk *= z1
            part = (zk * p0).imag
            if k > 1 and k * f_fire.max() > 2000.0:
                live = k * f_fire <= 2000.0
                if not live.any():
                    break
                part = np.where(live, part, 0.0)
            tonal += part * wk
        tonal *= amp
    elif src.kind == 'siren':
        lo, hi = SIREN['f_lo'], SIREN['f_hi']
        pattern = p.get('pattern', 'wail')
        rate = np.full(n, SIREN['wail_cpm'] if pattern != 'yelp' else SIREN['yelp_cpm']) / 60.0
        if pattern == 'wail_yelp':                     # alternating: wail for 8 s, yelp for 4 s
            a, b = SIREN['alternate_s']
            rate = np.where(np.mod(tau - p.get('t_ref', 0.0), a + b) < a, SIREN['wail_cpm'], SIREN['yelp_cpm']) / 60.0
        cyc = np.cumsum(rate) / fs + p.get('cycle0', 0.0)
        f = lo + (hi - lo) * (0.5 - 0.5 * np.cos(2.0 * np.pi * cyc))
        phase = 2.0 * np.pi * np.cumsum(f) / fs
        hs = np.array(SIREN['harmonics'], float)
        amps = 1.0 / hs
        for h, a_h in zip(hs, amps):
            tonal += np.where(h * f < 12000.0, a_h * np.sin(h * phase), 0.0)
        # A-weighted level over a sweep: the mean of the harmonics' A-weighted power across the swept frequencies
        fg = lo + (hi - lo) * (0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, 400, endpoint=False)))
        pa = np.mean([harmonic_power_a(hs * fx, np.where(hs * fx < 12000.0, amps, 0.0)) for fx in fg])
        target = P_REF ** 2 * 10.0 ** ((SIREN['dba_at_3m'] + 20.0 * math.log10(3.0)) / 10.0)
        tonal *= math.sqrt(target / pa)
    elif src.kind == 'horn':
        hp = HORNS[p['horn']]
        on = np.zeros(n)
        for a, b in p['events']:
            i0, i1 = int(max((a - t0) * fs, 0)), int(min((b - t0) * fs, n))
            if i1 > i0:
                on[i0:i1] = _fade(i1 - i0, fs, 0.01, 0.015)
        pa_total = 0.0
        for note in hp['notes']:
            ks = np.arange(1, int(6000.0 // note) + 1)
            amps = 1.0 / ks
            pa_total += harmonic_power_a(ks * note, amps)
            ph = rng.uniform(0, 2 * np.pi, len(ks))
            for k, a_k, p0 in zip(ks, amps, ph):
                tonal += a_k * np.sin(2.0 * np.pi * k * note * (tau - t0) + p0)
        target = P_REF ** 2 * 10.0 ** ((hp['dba_at_7m'] + 20.0 * math.log10(7.0)) / 10.0)
        tonal *= on * math.sqrt(target / pa_total)
    elif src.kind == 'tone':                           # validation: a pure tone of a given level at 1 m
        tonal = math.sqrt(2.0) * P_REF * 10.0 ** (p['db_at_1m'] / 20.0) * np.sin(2.0 * np.pi * p['f'] * (tau - t0))
    else:
        raise ValueError(f'unknown source kind {src.kind!r}')
    X = stft.forward(tonal) if np.any(tonal) else np.zeros((F, len(stft.f)), complex)
    if P_noise is not None:
        S = shaped_psd(P_noise, shape, stft.f[1] - stft.f[0])
        X = X + stft.forward(rng.standard_normal(n)) * np.sqrt(S * fs / 2.0)
    env = _fade(n, fs, *fades)
    return X, tf, n, env


# ============================================================================================ the scene
class Scene:
    """What the array hears: the ego's path and orientation, the array, the sources, the simulation steps."""

    def __init__(self, ego: list, mics_vehicle: np.ndarray, array_R: np.ndarray, sources: list, steps_t: np.ndarray,
                 duration: float, boxes=None, meta=None):
        self.ego = ego                        # [(t_start, Track(x y z), Track(roll pitch yaw unwrapped))]
        self.mics_vehicle = mics_vehicle      # (M, 3) mic positions in the vehicle frame
        self.array_R = array_R                # (3, 3) the array's frame in the vehicle frame
        self.sources = sources
        self.steps_t = np.asarray(steps_t, float)
        self.duration = float(duration)
        self.boxes = boxes                    # {step index: (ids, centres (n, 3), yaws, half extents (n, 3))}
        self.meta = meta or {}

    def ego_pose(self, t):
        """(position (T, 3), rotation (T, 3, 3), segment start (T,)) of the ego at times t."""
        t = np.asarray(t, float)
        P = np.zeros((len(t), 3))
        A = np.zeros((len(t), 3))
        start = np.zeros(len(t))
        starts = [e[0] for e in self.ego]
        seg = np.clip(np.searchsorted(starts, t, 'right') - 1, 0, len(self.ego) - 1)
        for k, (t0, tp, ta) in enumerate(self.ego):
            sel = seg == k
            if sel.any():
                P[sel] = tp(t[sel])
                A[sel] = ta(t[sel])
                start[sel] = t0 if k > 0 else -1e9
        return P, carla_rotations(A[:, 0], A[:, 1], A[:, 2]), start

    def mics_world(self, t):
        P, R, start = self.ego_pose(t)
        mw = P[None] + np.einsum('tij,mj->mti', R, self.mics_vehicle)
        centre = P + np.einsum('tij,j->ti', R, self.mics_vehicle.mean(axis=0))
        return mw, centre, R, start, P[:, 2]


def _segments(n: int, breaks):
    """Split n samples at break indices (a pose jump between k−1 and k): [(i0, i1)], i1 exclusive."""
    cuts = sorted({0, n} | {int(b) for b in breaks if 0 < b < n})
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def scene_from_run(run_dir: str, o: dict) -> Scene:
    """The scene of a recorded run (frames.jsonl, actors.jsonl, drive.json; docs/dataset-format.md)."""
    from .rig import with_defaults
    from .worldmodel import _jsonl
    frames = _jsonl(os.path.join(run_dir, 'frames.jsonl'))
    ap = os.path.join(run_dir, 'actors.jsonl')
    if not os.path.isfile(ap):
        raise ValueError(f'{run_dir}: no actors.jsonl — audio needs every actor\'s path')
    steps = _jsonl(ap)
    if len(frames) < 2 or len(steps) < 2:
        raise ValueError(f'{run_dir}: fewer than two steps')
    drive = {}
    if os.path.isfile(os.path.join(run_dir, 'drive.json')):
        with open(os.path.join(run_dir, 'drive.json'), encoding='utf-8') as fh:
            drive = json.load(fh)
    t_steps = np.array([s['sim_t'] for s in steps], float)
    dt = float(np.median(np.diff(t_steps)))
    c = speed_of_sound(o['temperature_c'])
    pre = o['max_range_m'] / c + 0.05             # sources present at t = 0 emitted before it: extrapolated back
    # the ego: its pose per frame, split where the road-end guard moved it
    ft = np.array([f['sim_t'] for f in frames], float)
    eg = np.array([[f['ego']['x'], f['ego']['y'], f['ego']['z']] for f in frames], float)
    ea = np.unwrap(np.radians([[f['ego'].get('roll', 0.0), f['ego'].get('pitch', 0.0), f['ego']['yaw']]
                               for f in frames]), axis=0)
    ea = np.degrees(ea)
    ebreaks = [k for k, f in enumerate(frames) if f.get('teleport')]
    ego = []
    for a, b in _segments(len(ft), ebreaks):
        ego.append((ft[a], Track(ft[a:b], eg[a:b]), Track(ft[a:b], ea[a:b])))
    # the array (the rig's audio_array sensor, else the default one; option 'array' overrides)
    rig = drive.get('rig') or {}
    sens = next((with_defaults(s) for s in rig.get('sensors', []) if s.get('type') == 'audio_array'), None)
    if o.get('array'):
        sens = dict(sens or {}, **o['array'])
    pose, mics = array_geometry(sens)
    Ra = carla_rotations(pose['roll'], pose['pitch'], pose['yaw'])
    mics_vehicle = np.array([pose['x'], pose['y'], pose['z']]) + mics @ Ra.T
    # the actors: tracks per id, split at pose jumps (respawns, or a step longer than physics allows)
    respawn = {}
    for k, f in enumerate(frames):
        for r in f.get('traffic_respawns') or ():
            if r.get('kind') != 'replaced' and r.get('actor') is not None:
                respawn.setdefault(int(r['actor']), set()).add(k)
    tracks, rows0 = {}, {}
    for k, s in enumerate(steps):
        for a in s['actors']:
            tracks.setdefault(a['id'], []).append((k, a))
            rows0.setdefault(a['id'], a)
    seed = o['seed'] if o.get('seed') is not None else int(drive.get('seed') or 0)
    ego_id = next((a['id'] for s in steps for a in s['actors'] if a.get('role') == 'hero'), None)
    horns = o['horns']
    if horns == 'auto':
        tr = drive.get('traffic') or {}
        horns = 'indian' if tr.get('handedness') == 'left' or 'india' in str(tr.get('id', '')) else 'sparse'
    sources, horn_events, siren_spans = [], [], []
    boxes = {}
    for k, s in enumerate(steps):
        vs = [a for a in s['actors'] if str(a.get('type_id', '')).startswith('vehicle.') and a['id'] != ego_id]
        if vs:
            boxes[k] = (np.array([a['id'] for a in vs]),
                        np.array([[a['x'], a['y'], a['z']] for a in vs]) +
                        np.einsum('nij,nj->ni', rot_z([a['yaw'] for a in vs]),
                                  np.array([a.get('bbox_center') or (0, 0, 0) for a in vs], float)),
                        np.array([a['yaw'] for a in vs], float), np.array([a['extent'] for a in vs], float))
    plan = horn_plan(steps, frames, ego_id, seed, horns) if HORN_PROFILES.get(horns, 0) > 0 else {}
    for ev in o.get('horn_events') or ():
        plan.setdefault(int(ev['actor']), []).append((float(ev['start_s']), float(ev['start_s']) +
                                                       float(ev.get('dur_s', 0.4)), 'explicit'))
    for aid, rows in tracks.items():
        if aid == ego_id and not o['include_ego']:
            continue
        cls, over, siren = classify(rows0[aid])
        if cls is None or CLASSES.get(cls) is None:
            continue
        ks = np.array([k for k, _ in rows])
        tt = t_steps[ks]
        xyz = np.array([[a['x'], a['y'], a['z']] for _, a in rows], float)
        yaw = np.degrees(np.unwrap(np.radians([a['yaw'] for _, a in rows])))
        breaks = [i for i in range(1, len(ks)) if ks[i] != ks[i - 1] + 1 or ks[i] in respawn.get(aid, ())]
        jump = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        vmag = np.array([math.hypot(a['vx'], a['vy']) for _, a in rows])
        allowed = np.maximum(3.0, 3.0 * np.maximum(vmag[1:], vmag[:-1]) * np.diff(tt) + 1.0)
        breaks += [i + 1 for i in np.nonzero(jump > allowed)[0]]
        ext = np.asarray(rows0[aid]['extent'], float)
        ctr = np.asarray(rows0[aid].get('bbox_center') or (0, 0, 0), float)
        for a, b in _segments(len(tt), breaks):
            if b - a < 2:
                continue
            tr = Track(tt[a:b], np.column_stack([xyz[a:b], yaw[a:b]]))
            t0 = tt[a] - (pre if ks[a] == 0 else 0.0)
            t1 = tt[b - 1]
            fi = RANGE_FADE_S if ks[a] == 0 else FADE_S
            spans = _in_range(tr, (t0, t1), ego, o['max_range_m'])
            if not spans:
                continue
            m = CLASSES[cls]
            base = dict(seg_fades=(fi, FADE_S), seg=(t0, t1))
            sources.append(Source(aid, rows0[aid]['type_id'], cls, 'engine', tr, (0.0, 0.0, m['h']), spans,
                                  cnossos=over, **base))
            if siren and o['sirens'] != 'off':
                on = spans
                if o['sirens'] == 'moving':
                    on = _moving_spans(tr, spans)
                if on:
                    sources.append(Source(aid, rows0[aid]['type_id'], cls, 'siren', tr,
                                          (ctr[0], 0.0, ctr[2] + ext[2]), on, pattern=siren,
                                          cycle0=float(_rng(seed, aid, 3).uniform()), t_ref=t0, **base))
                    siren_spans += [{'actor': aid, 'type_id': rows0[aid]['type_id'], 'pattern': siren,
                                     'start_s': round(max(x, 0.0), 4), 'end_s': round(y, 4)} for x, y in on]
            evs = [(x, y, why) for x, y, why in plan.get(aid, ()) if tt[a] <= x < t1]
            if evs:
                sp = [(max(x - 0.05, t0), min(y + 0.05, t1)) for x, y, _ in evs]
                sp = _merge(sp)
                sources.append(Source(aid, rows0[aid]['type_id'], cls, 'horn', tr,
                                      (ctr[0] + ext[0], 0.0, 0.8 if cls == 'two_wheeler' else 0.6), sp,
                                      horn=m['horn'], events=[(x, y) for x, y, _ in evs], **base))
                horn_events += [{'actor': aid, 'type_id': rows0[aid]['type_id'], 'class': cls, 'rule': why,
                                 'start_s': round(x, 4), 'end_s': round(y, 4)} for x, y, why in evs]
    duration = float(t_steps[-1]) if o.get('duration_s') is None else min(float(o['duration_s']), float(t_steps[-1]))
    meta = {'seed': seed, 'horns': horns, 'ego_id': ego_id, 'horn_events': horn_events, 'sirens': siren_spans,
            'array_pose': pose, 'mics_array': mics.tolist(), 'dt': dt,
            'steps': [{'step': s['step'], 'sim_t': s['sim_t']} for s in steps],
            'rows': {aid: rows0[aid] for aid in tracks}}
    return Scene(ego, mics_vehicle, Ra, sources, t_steps, duration, boxes, meta)


def _merge(spans):
    spans = sorted(spans)
    out = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def _in_range(track: Track, seg, ego, max_range):
    """The parts of [t0, t1] (s) when the actor is within max_range (+ 10 m) of the ego, on a 0.1 s grid."""
    t0, t1 = seg
    tg = np.arange(t0, t1 + 0.1, 0.1)
    tg[-1] = min(tg[-1], t1)
    p = track(tg)[:, :3]
    e = np.zeros_like(p)
    for k, (s0, tp, _) in enumerate(ego):
        sel = tg >= (s0 if k > 0 else -1e9)
        e[sel] = tp(tg[sel])
    near = np.linalg.norm(p - e, axis=1) <= max_range + 10.0
    out, start = [], None
    for k, ok in enumerate(near):
        if ok and start is None:
            start = tg[k]
        if not ok and start is not None:
            out.append((start, tg[k]))
            start = None
    if start is not None:
        out.append((start, t1))
    return [(a, b) for a, b in out if b - a > 0.05]


def _moving_spans(track, spans, v_min=1.0):
    out = []
    for a, b in spans:
        tg = np.arange(a, b, 0.1)
        if len(tg) < 2:
            continue
        _, dp = track(tg, deriv=True)
        mv = np.linalg.norm(dp[:, :2], axis=1) > v_min
        start = None
        for k, ok in enumerate(mv):
            if ok and start is None:
                start = tg[k]
            if not ok and start is not None:
                out.append((start, tg[k]))
                start = None
        if start is not None:
            out.append((start, b))
    return [(x, y) for x, y in out if y - x > 0.3]


# ============================================================================================ horns
def horn_plan(steps, frames, ego_id, seed: int, profile: str) -> dict:
    """{actor id: [(start, end, rule)]}: horn events by HORN_RULES (heuristic) at every step, a Poisson draw per vehicle
    from its own seeded generator (adding an actor never changes another's horns)."""
    scale = HORN_PROFILES.get(profile, 0.0)
    out, last, stopped, rngs = {}, {}, {}, {}
    for k, s in enumerate(steps):
        t = float(s['sim_t'])
        dt = float(steps[k + 1]['sim_t'] - t) if k + 1 < len(steps) else float(t - steps[k - 1]['sim_t'])
        acts = s['actors']
        veh = [a for a in acts if str(a.get('type_id', '')).startswith('vehicle.')]
        walk = [a for a in acts if str(a.get('type_id', '')).startswith('walker.')]
        if not veh:
            continue
        P = np.array([[a['x'], a['y']] for a in veh])
        V = np.array([[a['vx'], a['vy']] for a in veh])
        Y = np.radians([a['yaw'] for a in veh])
        W = np.array([[a['x'], a['y']] for a in walk]) if walk else np.zeros((0, 2))
        spd = np.linalg.norm(V, axis=1)
        for i, a in enumerate(veh):
            if a['id'] == ego_id:
                continue
            cls = classify(a)[0]
            rate_c = HORN_CLASS_RATE.get(cls, 0.0)
            if rate_c <= 0:
                continue
            stopped[a['id']] = stopped.get(a['id'], 0.0) + dt if spd[i] < 0.5 else 0.0
            if t - last.get(a['id'], -1e9) < HORN_TIMING['refractory_s']:
                continue
            c, sn = math.cos(Y[i]), math.sin(Y[i])
            d = P - P[i]
            fx, fy = d[:, 0] * c + d[:, 1] * sn, -d[:, 0] * sn + d[:, 1] * c       # others in i's frame (y right)
            vj_along = V[:, 0] * c + V[:, 1] * sn
            dyaw = np.abs((np.degrees(Y - Y[i]) + 180.0) % 360.0 - 180.0)
            others = np.arange(len(veh)) != i
            rules = {}
            if spd[i] > 3.0 and np.any(others & (fx > 3) & (fx < 25) & (np.abs(fy) < 1.8) &
                                       (spd[i] - vj_along > 2.0)):
                rules['slow_leader'] = HORN_RULES['slow_leader']
            if spd[i] > 1.0 and np.any(others & (fx > 3) & (fx < 20) & (np.abs(fy) < 6.0) & (dyaw > 45) &
                                       (dyaw < 135) & (spd > 2.0)):
                rules['crossing_traffic'] = HORN_RULES['crossing_traffic']
            if spd[i] > 2.0 and len(W):
                dw = W - P[i]
                wx, wy = dw[:, 0] * c + dw[:, 1] * sn, -dw[:, 0] * sn + dw[:, 1] * c
                if np.any((wx > 2) & (wx < 20) & (np.abs(wy) < 3.0)):
                    rules['pedestrian_ahead'] = HORN_RULES['pedestrian_ahead']
            if stopped[a['id']] >= 3.0 and np.any(others & (fx > 0) & (fx < 8) & (np.abs(fy) < 1.5)):
                rules['queue'] = HORN_RULES['queue']
            if not rules:
                continue
            lam = sum(rules.values()) * rate_c * scale
            g = rngs.get(a['id'])
            if g is None:
                g = rngs[a['id']] = _rng(seed, a['id'], 11)
            if g.uniform() >= 1.0 - math.exp(-lam * dt):
                continue
            why = max(rules, key=rules.get)
            d1 = g.uniform(*HORN_TIMING['dur_s'])
            ev = [(t, t + d1, why)]
            if g.uniform() < HORN_TIMING['double_p']:
                s2 = t + d1 + g.uniform(*HORN_TIMING['gap_s'])
                ev.append((s2, s2 + g.uniform(*HORN_TIMING['second_s']), why))
            out.setdefault(a['id'], []).extend(ev)
            last[a['id']] = ev[-1][1]
    return out


# ============================================================================================ rendering
def _occlusion_z(scene: Scene, src: Source, tf, centre_fn):
    """Path difference (m) over the top of the most obstructing other vehicle at each frame time tf; 0 unobstructed."""
    z = np.zeros(len(tf))
    if not scene.boxes:
        return z
    st = scene.steps_t
    ks = np.clip(np.searchsorted((st[1:] + st[:-1]) / 2, tf), 0, len(st) - 1)      # the nearest step
    S = src.pos(tf)
    R = centre_fn(tf)
    for u in np.unique(ks):
        if u not in scene.boxes:
            continue
        ids, C, yaw, E = scene.boxes[u]
        keep = ids != src.actor
        if not keep.any():
            continue
        C, yaw, E = C[keep], yaw[keep], E[keep]
        sel = np.nonzero(ks == u)[0]
        a = np.radians(yaw)
        ca, sa = np.cos(a)[None], np.sin(a)[None]
        def local(p):
            d = p[:, None, :] - C[None]
            return np.stack([d[..., 0] * ca + d[..., 1] * sa, -d[..., 0] * sa + d[..., 1] * ca, d[..., 2]], -1)
        p0, p1 = local(S[sel]), local(R[sel])                         # (n, boxes, 3) in each box's frame
        dvec = p1 - p0
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = 1.0 / np.where(np.abs(dvec) < 1e-12, 1e-12, dvec)
            ta, tb = (-E[None] - p0) * inv, (E[None] - p0) * inv
        tmin = np.max(np.minimum(ta, tb), axis=-1)
        tmax = np.min(np.maximum(ta, tb), axis=-1)
        hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin <= 1.0)
        if not hit.any():
            continue
        # over the top edge: the box's top at the point of the path nearest its centre
        dist = np.linalg.norm(R[sel] - S[sel], axis=1)[:, None]
        tc = np.clip(np.clip(-np.sum(p0[..., :2] * dvec[..., :2], -1) /
                             np.maximum(np.sum(dvec[..., :2] ** 2, -1), 1e-9), 0, 1), 0, 1)
        top = (C[None, :, 2] + E[None, :, 2])
        d1, d2 = tc * dist, (1 - tc) * dist
        h1 = top - S[sel][:, 2:3]
        h2 = top - R[sel][:, 2:3]
        zz = np.sqrt(d1 ** 2 + h1 ** 2) + np.sqrt(d2 ** 2 + h2 ** 2) - dist
        z[sel] = np.max(np.where(hit, zz, 0.0), axis=1)
    return z


def _retard(t, m, src: Source, c, mirror_z=None, iters=4):
    """Emission times τ (and distances) for receivers at positions m (n, 3) and times t (n,): Newton on
    t − τ − |m − p(τ)|/c = 0. mirror_z: the image source in the plane z = mirror_z (n,)."""
    p = src.pos(t)
    if mirror_z is not None:
        p[:, 2] = 2 * mirror_z - p[:, 2]
    tau = t - np.linalg.norm(m - p, axis=1) / c
    for _ in range(iters):
        p, v = src.pos(tau, deriv=True)
        if mirror_z is not None:
            p[:, 2] = 2 * mirror_z - p[:, 2]
            v = v * np.array([1.0, 1.0, -1.0])
        d = m - p
        r = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
        g = t - tau - r / c
        gp = -1.0 + np.sum(d * v, axis=1) / r / c
        tau = tau - g / gp
    p, v = src.pos(tau, deriv=True)
    if mirror_z is not None:
        p[:, 2] = 2 * mirror_z - p[:, 2]
    return tau, np.maximum(c * (t - tau), 0.3), p, v


def _cubic(sp, x):
    """4-point Catmull-Rom interpolation at fractional indices x of a signal padded as _pad does (zero outside);
    float32 arithmetic on the samples (the index stays float64)."""
    i = np.floor(x)
    u = (x - i).astype(np.float32)
    j = i.astype(np.int64)
    np.clip(j, -2, len(sp) - 5, out=j)
    j += 2
    y0, y1, y2, y3 = sp[j - 1], sp[j], sp[j + 1], sp[j + 2]
    return y1 + np.float32(0.5) * u * (y2 - y0 + u * (2 * y0 - 5 * y1 + 4 * y2 - y3 + u * (3 * (y1 - y2) + y3 - y0)))


def _pad(sig):
    return np.concatenate([[0.0, 0.0], sig, [0.0, 0.0, 0.0]]).astype(np.float32)


def render(scene: Scene, o: dict, progress=None) -> dict:
    """Every source through propagation to every microphone: {'audio' (M, N) float32 Pa, 'gt': per step, 'stats'}."""
    fs = int(o['sample_rate'])
    c = speed_of_sound(o['temperature_c'])
    stft = Stft(STFT_N, fs)
    shape = band_shape(stft)
    alpha = air_absorption_db_per_m(stft.f, o['temperature_c'], o['humidity_pct'], o['pressure_kpa']) \
        if o['air_absorption'] else np.zeros(len(stft.f))
    aw = 10.0 ** (a_weighting_db(np.maximum(stft.f, 1.0)) / 10.0)
    N = int(round(scene.duration * fs)) + 1
    M = len(scene.mics_vehicle)
    out = np.zeros((M, N), np.float32)
    tc = np.arange(0, N + COARSE, COARSE) / fs                        # the 1 ms grid
    mw, centre, _, egostart, zg = scene.mics_world(tc)
    seed = int(scene.meta.get('seed', 0))

    def centre_fn(t):
        return np.stack([np.interp(t, tc, centre[:, k]) for k in range(3)], axis=1)
    gt_t = scene.steps_t[(scene.steps_t >= 0) & (scene.steps_t <= scene.duration)]
    gt_dt = float(np.median(np.diff(scene.steps_t))) if len(scene.steps_t) > 1 else 0.05
    gt = {k: [] for k in range(len(gt_t))}
    vc = np.gradient(centre, tc, axis=0)
    timing = {'emit_s': 0.0, 'propagate_s': 0.0}
    for si, src in enumerate(scene.sources):
        rng = _rng(seed, src.actor, {'engine': 1, 'siren': 2, 'horn': 4, 'tone': 5}[src.kind])
        seg0, seg1 = src.params.get('seg', (src.intervals[0][0], src.intervals[-1][1]))
        sf_in, sf_out = src.params.get('seg_fades', (FADE_S, FADE_S))
        for a, b in src.intervals:
            t_e = time.monotonic()
            fades = (sf_in if a <= seg0 + 1e-9 else RANGE_FADE_S, sf_out if b >= seg1 - 1e-9 else RANGE_FADE_S)
            if src.kind == 'horn':
                fades = (0.0, 0.0)
            X, tf, n, env = emit(src, a, b, fades, fs, stft, shape, rng)
            # propagation filter on the emitted frames: air absorption at the frame's distance, occlusion
            p_f = src.pos(tf)
            cf = centre_fn(tf)
            r_f = np.linalg.norm(p_f - cf, axis=1)
            cf = centre_fn(tf + r_f / c)
            r_f = np.linalg.norm(p_f - cf, axis=1)
            att = alpha[None] * r_f[:, None]
            if o['occlusion']:
                zf = _occlusion_z(scene, src, tf, lambda t: centre_fn(t + r_f / c))
                if np.any(zf > 0):
                    att = att + barrier_db(zf, stft.f, c)
            G = 10.0 ** (-att / 20.0)
            Y = X * G
            pw = np.abs(Y) ** 2
            frac_a = np.sum(pw * aw, axis=1) / np.maximum(np.sum(pw, axis=1), 1e-40)
            sig = stft.inverse(Y, n) * env
            timing['emit_s'] += time.monotonic() - t_e
            t_p = time.monotonic()
            # receivers: every mic (direct and ground image), and the array's centre for the ground truth
            r_max = o['max_range_m'] + 20.0
            k0 = max(int(np.floor(a * fs / COARSE)), 0)
            k1 = min(int(np.ceil((b + r_max / c) * fs / COARSE)) + 1, len(tc) - 1)
            if k1 <= k0:
                continue
            tk = tc[k0:k1 + 1]
            i0, i1 = k0 * COARSE, min(k1 * COARSE, N)
            ti = np.arange(i0, i1) / fs
            power = np.zeros(i1 - i0, np.float32)
            sp = _pad(sig)
            moved = len(scene.ego) > 1
            paths = [(None, 1.0)] + ([('ground', float(o['ground_reflection']))] if o['ground'] else [])
            for m in range(M):
                for kind, gain in paths:
                    tau, r, _, _ = _retard(tk, mw[m, k0:k1 + 1], src, c, zg[k0:k1 + 1] if kind else None)
                    tau_i = np.interp(ti, tk, tau)
                    rinv = np.interp(ti, tk, 1.0 / r)
                    ok = (tau_i >= a) & (tau_i <= b)
                    if moved:                     # the ego jumped: nothing emitted before it arrived is heard
                        ok &= tau_i >= np.interp(ti, tk, egostart[k0:k1 + 1])
                    val = _cubic(sp, (tau_i - a) * fs) * (rinv * gain * ok).astype(np.float32)
                    out[m, i0:i1] += val
                    if kind is None:
                        acc = val
                    else:
                        acc = acc + val
                power += acc * acc
            power /= M
            timing['propagate_s'] += time.monotonic() - t_p
            # ground truth at the steps this interval reaches
            sel = np.nonzero((gt_t >= tk[0]) & (gt_t <= tk[-1]))[0]
            if not len(sel):
                continue
            tg = gt_t[sel]
            cg = centre_fn(tg)
            tau, r, p, v = _retard(tg, cg, src, c)
            live = (tau >= a) & (tau <= b)
            vm = np.stack([np.interp(tg, tc, vc[:, k]) for k in range(3)], axis=1)
            nvec = (cg - p) / np.linalg.norm(cg - p, axis=1, keepdims=True)
            dop = (1.0 - np.sum(nvec * vm, axis=1) / c) / (1.0 - np.sum(nvec * v, axis=1) / c)
            Rg = scene.ego_pose(tg)[1] @ scene.array_R
            loc = np.einsum('tji,tj->ti', Rg, p - cg)                    # the source in the array's frame
            az = np.degrees(np.arctan2(loc[:, 1], loc[:, 0]))
            el = np.degrees(np.arctan2(loc[:, 2], np.hypot(loc[:, 0], loc[:, 1])))
            w0 = np.clip(((tg - gt_dt / 2) * fs).astype(int) - i0, 0, len(power))
            w1 = np.clip(((tg + gt_dt / 2) * fs).astype(int) - i0, 0, len(power))
            csum = np.concatenate([[0.0], np.cumsum(power)])
            ms = (csum[w1] - csum[w0]) / np.maximum(w1 - w0, 1)
            fa = np.interp(tau, tf, frac_a)
            for j, k in enumerate(sel):
                if not live[j] or ms[j] <= 0:
                    continue
                db = 10.0 * math.log10(ms[j] / P_REF ** 2)
                if db < o['audible_db']:
                    continue
                gt[int(k)].append({'id': int(src.actor), 'type_id': src.type_id, 'class': src.cls, 'kind': src.kind,
                                   'az_deg': round(float(az[j]), 3), 'el_deg': round(float(el[j]), 3),
                                   'distance_m': round(float(r[j]), 3), 'spl_db': round(db, 2),
                                   'spl_dba': round(db + 10.0 * math.log10(max(fa[j], 1e-12)), 2),
                                   'doppler': round(float(dop[j]), 6),
                                   'speed_mps': round(float(np.linalg.norm(v[j])), 3)})
        if progress:
            progress((si + 1) / max(len(scene.sources), 1), f'source {si + 1} of {len(scene.sources)}')
    if o['mic_self_noise_db'] is not None and o['mic_self_noise_db'] > -50:
        g = np.random.default_rng([seed & 0xFFFFFFFF, 0xA0D10])
        sd = P_REF * 10.0 ** (o['mic_self_noise_db'] / 20.0)
        for m in range(M):
            out[m] += (sd * g.standard_normal(N)).astype(np.float32)
    rows = [{'sim_t': round(float(t), 6), 'sources': sorted(gt[k], key=lambda s: -s['spl_db'])}
            for k, t in enumerate(gt_t)]
    return {'audio': out, 'gt': rows, 'gt_t': gt_t, 'timing': timing, 'fs': fs, 'c': c}


# ============================================================================================ WAV
_GUID_TAIL = b'\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71'


def write_wav(path: str, data: np.ndarray, fs: int, fmt: str = 'float32', full_scale: float = 200.0) -> dict:
    """A WAVE_FORMAT_EXTENSIBLE file of data (channels, samples): 'float32' (IEEE float, values as given) or 'pcm24'
    (values / full_scale, clipped). Returns {'clipped': samples clipped}."""
    M, N = data.shape
    if fmt == 'float32':
        width, sub, payload, clipped = 4, b'\x03\x00' + _GUID_TAIL, np.ascontiguousarray(data.T, '<f4'), 0
    elif fmt == 'pcm24':
        q = np.round(np.asarray(data, float).T / full_scale * 8388607.0)
        clipped = int(np.sum(np.abs(q) > 8388607))
        q = np.ascontiguousarray(np.clip(q, -8388608, 8388607).astype('<i4'))
        payload = q.view(np.uint8).reshape(N, M, 4)[:, :, :3]
        width, sub = 3, b'\x01\x00' + _GUID_TAIL
    else:
        raise ValueError(f'unknown WAV format {fmt!r} (float32 or pcm24)')
    size = M * N * width
    if size + 80 > 0xFFFFFFFF:
        raise ValueError(f'{M} channels × {N} samples is over the 4 GB a WAV file holds: use pcm24 or a shorter '
                         'duration_s')
    fmt_chunk = struct.pack('<HHIIHHHHI', 0xFFFE, M, fs, fs * M * width, M * width, width * 8, 22, width * 8, 0) + sub
    fact = b'fact' + struct.pack('<II', 4, N) if fmt == 'float32' else b''
    with open(path, 'wb') as fh:
        fh.write(b'RIFF' + struct.pack('<I', 4 + 8 + len(fmt_chunk) + len(fact) + 8 + size) + b'WAVE')
        fh.write(b'fmt ' + struct.pack('<I', len(fmt_chunk)) + fmt_chunk + fact)
        fh.write(b'data' + struct.pack('<I', size))
        fh.write(np.ascontiguousarray(payload).tobytes())
    return {'clipped': clipped}


def read_wav(path: str, full_scale: float = 200.0) -> tuple[np.ndarray, int]:
    """(data (channels, samples) float, fs) of a WAV written by write_wav (or any PCM 16/24/32 or float32 file)."""
    with open(path, 'rb') as fh:
        b = fh.read()
    if b[:4] != b'RIFF' or b[8:12] != b'WAVE':
        raise ValueError(f'{path}: not a WAV file')
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(b):
        cid, size = b[pos:pos + 4], struct.unpack('<I', b[pos + 4:pos + 8])[0]
        body = b[pos + 8:pos + 8 + size]
        if cid == b'fmt ':
            tag, M, fs, _, _, bits = struct.unpack('<HHIIHH', body[:16])
            if tag == 0xFFFE:
                tag = struct.unpack('<H', body[24:26])[0]
            fmt = (tag, M, fs, bits)
        elif cid == b'data':
            data = body
        pos += 8 + size + (size & 1)
    tag, M, fs, bits = fmt
    if tag == 3 and bits == 32:
        x = np.frombuffer(data, '<f4').astype(float)
    elif tag == 1 and bits == 24:
        u = np.frombuffer(data, np.uint8).reshape(-1, 3).astype(np.int32)
        v = u[:, 0] | (u[:, 1] << 8) | (u[:, 2] << 16)
        x = np.where(v >= 1 << 23, v - (1 << 24), v) / 8388607.0 * full_scale
    elif tag == 1 and bits == 16:
        x = np.frombuffer(data, '<i2') / 32767.0 * full_scale
    else:
        raise ValueError(f'{path}: unsupported WAV encoding (format {tag}, {bits} bits)')
    return x.reshape(-1, M).T, fs


# ============================================================================================ direction of arrival
class SrpPhat:
    """SRP-PHAT (DiBiase, Silverman & Brandstein 2001, "Robust localization in reverberant rooms", in Microphone
    Arrays, Springer): the steered response power of the PHAT-weighted cross-spectra of every microphone pair over a
    far-field grid of azimuths and elevations in the array's frame. mics (M, 3) in the array frame; frames of n
    samples; band (Hz) the frequencies used."""

    def __init__(self, mics, fs: int, c: float, n: int = 2048, az_deg=None, el_deg=(0.0,), band=(200.0, 6000.0)):
        self.az = np.arange(-180.0, 180.0, 1.0) if az_deg is None else np.asarray(az_deg, float)
        self.el = np.asarray(el_deg, float)
        self.n, self.fs = n, fs
        f = np.fft.rfftfreq(n, 1.0 / fs)
        self.fb = (f >= band[0]) & (f <= band[1])
        f = f[self.fb]
        M = len(mics)
        self.pairs = [(i, j) for i in range(M) for j in range(i + 1, M)]
        az, el = np.radians(self.az), np.radians(self.el)
        u = np.stack([np.cos(el)[:, None] * np.cos(az)[None], np.cos(el)[:, None] * np.sin(az)[None],
                      np.sin(el)[:, None] * np.ones_like(az)[None]], axis=-1)       # (el, az, 3) toward the source
        dm = np.stack([np.asarray(mics[i], float) - np.asarray(mics[j], float) for i, j in self.pairs])
        lag = -(u @ dm.T) / c                                    # τ_i − τ_j of a plane wave from u: (el, az, P)
        self.steer = np.exp(2j * np.pi * lag[..., None] * f).reshape(len(el) * len(az), -1).T.astype(np.complex64)
        self.window = np.hanning(n)

    def response(self, frames_x) -> np.ndarray:
        """(frames, el, az) for frames_x (frames, M, n) of raw samples (windowed here)."""
        X = np.fft.rfft(np.asarray(frames_x) * self.window, axis=2)[:, :, self.fb]
        G = np.stack([X[:, i] * np.conj(X[:, j]) for i, j in self.pairs], axis=1)
        G /= np.maximum(np.abs(G), 1e-20)
        resp = np.real(G.reshape(len(G), -1).astype(np.complex64) @ self.steer)
        return resp.reshape(len(G), len(self.el), len(self.az))

    def azimuths(self, frames_x) -> np.ndarray:
        r = self.response(frames_x)
        return self.az[r.reshape(len(r), -1).argmax(axis=1) % len(self.az)]


def _azerr(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def doa_evaluate(audio: np.ndarray, fs: int, mics: np.ndarray, c: float, gt_rows: list, dominance_db=3.0,
                 win: int = 2048, floor_db: float | None = None) -> dict:
    """SRP-PHAT on the array output at every ground-truth step (a window of `win` samples centred on it), compared
    with the ground truth: on steps where the loudest source (spl_db) is at least dominance_db louder than the next,
    the absolute azimuth error to it (overall and by the loudest source's kind); and on every step the error to the
    nearest audible source, next to the same for a uniformly random estimate (the chance level of that measure)."""
    srp = SrpPhat(mics, fs, c, win, el_deg=(-10.0, 0.0, 10.0, 20.0))
    rows = [r for r in gt_rows if r['sources'] and (floor_db is None or r['sources'][0]['spl_db'] >= floor_db)]
    rows = [r for r in rows if 0 <= int(round(r['sim_t'] * fs)) - win // 2 and
            int(round(r['sim_t'] * fs)) + win - win // 2 <= audio.shape[1]]
    if not rows:
        return {'frames': 0}
    est = []
    for k in range(0, len(rows), 256):
        part = rows[k:k + 256]
        fr = np.stack([audio[:, int(round(r['sim_t'] * fs)) - win // 2:int(round(r['sim_t'] * fs)) - win // 2 + win]
                       for r in part])
        est += list(srp.azimuths(fr))
    rnd = np.random.default_rng(0).uniform(-180.0, 180.0, len(rows))
    dom, by_kind, near, chance, picks, n_aud = [], {}, [], [], [], []
    for r, e, z in zip(rows, est, rnd):
        src = r['sources']
        az = np.array([s['az_deg'] for s in src])
        near.append(float(_azerr(e, az).min()))
        chance.append(float(_azerr(z, az).min()))
        n_aud.append(len(src))
        if len(src) == 1 or src[0]['spl_db'] - src[1]['spl_db'] >= dominance_db:
            d = float(_azerr(e, az[0]))
            dom.append(d)
            by_kind.setdefault(src[0]['kind'], []).append(d)
            picks.append({'sim_t': r['sim_t'], 'est_deg': float(e), 'gt_deg': src[0]['az_deg'], 'id': src[0]['id'],
                          'kind': src[0]['kind'], 'spl_db': src[0]['spl_db'], 'distance_m': src[0]['distance_m'],
                          'margin_db': round(src[0]['spl_db'] - (src[1]['spl_db'] if len(src) > 1 else 0.0), 2)})

    def q(a, p):
        return round(float(np.percentile(a, p)), 2) if len(a) else None
    return {'estimator': f'SRP-PHAT, 1° azimuth, elevations −10…20°, 200–6000 Hz, {win}-sample Hann frames',
            'frames': len(rows), 'audible_per_frame_median': q(n_aud, 50), 'dominance_db': dominance_db,
            'dominant': {'frames': len(dom), 'median_deg': q(dom, 50), 'p90_deg': q(dom, 90),
                         'by_kind': {k: {'frames': len(v), 'median_deg': q(v, 50), 'p90_deg': q(v, 90)}
                                     for k, v in sorted(by_kind.items())}},
            'nearest': {'median_deg': q(near, 50), 'p90_deg': q(near, 90),
                        'chance_median_deg': q(chance, 50), 'chance_p90_deg': q(chance, 90)},
            'samples': picks[:3000]}


# ============================================================================================ the product
def settings(options: dict | None) -> dict:
    o = dict(DEFAULTS)
    o.update(options or {})
    bad = sorted(set(o) - set(DEFAULTS))
    if bad:
        raise ValueError(f'unknown options: {", ".join(bad)} (known: {", ".join(sorted(DEFAULTS))})')
    if o['horns'] not in ('auto', *HORN_PROFILES):
        raise ValueError(f'horns: auto, {", ".join(HORN_PROFILES)}')
    if o['sirens'] not in ('on', 'off', 'moving'):
        raise ValueError('sirens: on, off or moving')
    if o['format'] not in ('float32', 'pcm24'):
        raise ValueError('format: float32 or pcm24')
    if int(o['sample_rate']) < 16000:
        raise ValueError('sample_rate: at least 16000 Hz')
    return o


LIMITS = ('synthetic sources from emission models and heuristics, no measured vehicle recordings',
          'point sources, omnidirectional (no siren or horn directivity, no vehicle body directivity)',
          'free field plus one specular ground reflection (real factor): no ground impedance, no reverberation, no '
          'reflections from buildings or vehicles',
          'occlusion only by other vehicles\' boxes (diffraction over the top, ISO 9613-2); buildings and the ego\'s '
          'own body are not modelled',
          'no wind, tyre-road surface or weather effects; no convective amplification of moving sources',
          'horns from a heuristic rule set, not from observed honking')


def derive_audio(run_dir: str, out_dir: str | None = None, options: dict | None = None, progress=None) -> dict:
    """The audio product of a run into out_dir (default RUN/audio); returns the manifest (audio.json). When out_dir is
    inside the run, the run's run.json gets a "derived" → "audio" entry with the parameters."""
    from .provenance import block, read_json, sha256_json, write_json
    o = settings(options)
    out_dir = out_dir or os.path.join(run_dir, 'audio')
    t_all = time.monotonic()
    scene = scene_from_run(run_dir, o)
    t_scene = time.monotonic() - t_all
    res = render(scene, o, progress)
    os.makedirs(out_dir, exist_ok=True)
    wav = write_wav(os.path.join(out_dir, 'audio.wav'), res['audio'], res['fs'], o['format'], o['full_scale_pa'])
    steps = {round(s['sim_t'], 6): s['step'] for s in scene.meta['steps']}
    with open(os.path.join(out_dir, 'ground_truth.jsonl'), 'w', encoding='utf-8') as fh:
        for r in res['gt']:
            fh.write(json.dumps(dict(step=steps.get(r['sim_t']), **r), separators=(',', ':')) + '\n')
    with open(os.path.join(out_dir, 'events.jsonl'), 'w', encoding='utf-8') as fh:
        for e in sorted([dict(kind='horn', **e) for e in scene.meta['horn_events']] +
                        [dict(kind='siren', **e) for e in scene.meta['sirens']], key=lambda e: e['start_s']):
            fh.write(json.dumps(e, separators=(',', ':')) + '\n')
    dur = scene.duration
    total = time.monotonic() - t_all
    doa = None
    if o['doa']:
        t_d = time.monotonic()
        mics_arr = np.asarray(scene.meta['mics_array'])
        doa = doa_evaluate(res['audio'], res['fs'], mics_arr, res['c'], res['gt'],
                           floor_db=o['mic_self_noise_db'] + 20.0 if o['mic_self_noise_db'] is not None else None)
        doa['seconds'] = round(time.monotonic() - t_d, 2)
        write_json(os.path.join(out_dir, 'doa.json'), dict(doa, schema=SCHEMA + '#doa'))
        doa = {k: v for k, v in doa.items() if k != 'samples'}
    kinds = {}
    for s in scene.sources:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
    classes = {}
    for s in scene.sources:
        if s.kind == 'engine':
            classes[s.cls] = classes.get(s.cls, 0) + 1
    ins = {k: os.path.join(run_dir, n) for k, n in (('run', 'run.json'), ('frames', 'frames.jsonl'),
                                                    ('actors', 'actors.jsonl'), ('drive', 'drive.json'))}
    pose = scene.meta['array_pose']
    man = {'schema': SCHEMA, 'synthetic': True, 'run': os.path.abspath(run_dir),
           'label': 'SYNTHETIC audio: procedural sources from emission models and heuristics; not recorded sound',
           'wav': {'file': 'audio.wav', 'sample_rate': res['fs'], 'channels': len(scene.mics_vehicle),
                   'samples': int(res['audio'].shape[1]), 'duration_s': round(dur, 6), 'format': o['format'],
                   'units': 'Pa' if o['format'] == 'float32' else f'full scale = {o["full_scale_pa"]} Pa',
                   'clipped_samples': wav['clipped'], 't0': 'sim_t 0 (the first recorded step)'},
           'array': {'pose_vehicle': pose, 'mics_array_frame': scene.meta['mics_array'],
                     'mics_vehicle_frame': np.round(scene.mics_vehicle, 6).tolist(),
                     'channel_order': 'channel k = microphone k',
                     'frame': 'array frame: the audio_array sensor\'s pose in CARLA\'s vehicle frame (x forward, y '
                              'right, z up; left-handed); azimuth 0° ahead, positive to the right; elevation up'},
           'ground_truth': {'file': 'ground_truth.jsonl', 'steps': len(res['gt']), 'audible_db': o['audible_db'],
                            'fields': ['id', 'type_id', 'class', 'kind', 'az_deg', 'el_deg', 'distance_m', 'spl_db',
                                       'spl_dba', 'doppler', 'speed_mps'],
                            'direction': 'of the source\'s retarded position (where the sound now arriving left it) '
                                         'from the array\'s centre',
                            'spl_db': 'the source\'s received mean-square pressure over the step (centred), mean of '
                                      'the microphones, direct + ground path, dB re 20 µPa',
                            'spl_dba': 'spl_db + the A-weighted share of its emitted, absorbed spectrum (estimate)',
                            'doppler': 'dτ/dt at the array\'s centre: received / emitted frequency'},
           'events': {'file': 'events.jsonl', 'horns': len(scene.meta['horn_events']),
                      'sirens': len(scene.meta['sirens']), 'horn_profile': scene.meta['horns']},
           'sources': {'by_kind': kinds, 'vehicles_by_class': classes},
           'parameters': {**{k: o[k] for k in DEFAULTS if k not in ('horn_events',)},
                          'horn_events': len(o['horn_events'] or ()), 'seed': scene.meta['seed'],
                          'horns': scene.meta['horns'], 'speed_of_sound_mps': round(res['c'], 3)},
           'models': {'vehicles': 'CNOSSOS-EU (Directive (EU) 2015/996, coefficients of Delegated Directive (EU) '
                                  '2021/1226) + IMAGINE acceleration correction (C_P = 3.2 dB per m/s², from D11 '
                                  'fig. 18; heuristic in every band)',
                      'engine_orders': f'heuristic: {TONAL_SHARE:.0%} of the 63–1000 Hz propulsion power, gear model',
                      'sirens': SIREN, 'horns': HORNS, 'horn_rules': HORN_RULES, 'horn_class_rate': HORN_CLASS_RATE,
                      'horn_profiles': HORN_PROFILES, 'air_absorption': 'ISO 9613-1:1993',
                      'occlusion': 'ISO 9613-2 eq. 14 over the top of other vehicles\' boxes (capped 20 dB)',
                      'ground': 'one image source, real reflection factor'},
           'limits': list(LIMITS),
           'cost': {'seconds': round(total, 2), 'scene_s': round(t_scene, 2),
                    'emit_s': round(res['timing']['emit_s'], 2), 'propagate_s': round(res['timing']['propagate_s'], 2),
                    'seconds_per_audio_minute': round(total / max(dur, 1e-9) * 60.0, 1),
                    'sources': len(scene.sources)},
           'doa': doa,
           'provenance': block(ins, scene.meta['seed'], {k: o[k] for k in DEFAULTS})}
    write_json(os.path.join(out_dir, MANIFEST), man)
    if os.path.abspath(out_dir).startswith(os.path.abspath(run_dir) + os.sep):
        rp = os.path.join(run_dir, 'run.json')
        if os.path.isfile(rp):
            run = read_json(rp)
            run.setdefault('derived', {})['audio'] = {
                'path': os.path.relpath(out_dir, run_dir), 'schema': SCHEMA, 'synthetic': True,
                'created': man['provenance']['created'], 'settings_sha256': sha256_json(man['parameters']),
                'parameters': man['parameters'], 'channels': man['wav']['channels'],
                'sample_rate': man['wav']['sample_rate']}
            write_json(rp, run)
    return man


def summary(man: dict) -> str:
    """One line on a made audio product (the CLI's and the GUI job's)."""
    w, d = man['wav'], (man.get('doa') or {}).get('dominant') or {}
    s = (f'synthetic audio: {w["channels"]} channels × {w["duration_s"]:.1f} s at {w["sample_rate"]} Hz, sources '
         f'{json.dumps(man["sources"]["by_kind"])}, {man["events"]["horns"]} horn events, made in '
         f'{man["cost"]["seconds"]:.1f} s ({man["cost"]["seconds_per_audio_minute"]:.0f} s per minute of audio)')
    if d.get('frames'):
        s += (f'; SRP-PHAT azimuth error on {d["frames"]} frames with a dominant source: median {d["median_deg"]}°, '
              f'90th percentile {d["p90_deg"]}°')
    return s


# ============================================================================================ the pass-by check
def passby_check(speed_mps=20.0, offset_m=10.0, f0=1000.0, db_at_1m=100.0, temp_c=20.0, humidity_pct=70.0,
                 air_absorption=True, fs=48000, duration=8.0) -> dict:
    """A pure tone passing a single stationary microphone at a known speed: the measured frequency against
    f0·c/(c − v_r) (v_r the source's speed toward the mic at emission) and the measured level against
    L(1 m) − 20·lg r − α·r, both over the time the source is within 100 m."""
    o = settings({'temperature_c': temp_c, 'humidity_pct': humidity_pct, 'air_absorption': air_absorption,
                  'ground': False, 'occlusion': False, 'mic_self_noise_db': None, 'sample_rate': fs})
    c = speed_of_sound(temp_c)
    t = np.arange(-2.0, duration + 2.0, 0.05)
    x0 = -speed_mps * duration / 2.0
    xyz = np.column_stack([x0 + speed_mps * t, np.full_like(t, offset_m), np.full_like(t, 1.5), np.zeros_like(t)])
    src = Source(1, 'test.tone', 'car', 'tone', Track(t, xyz), (0, 0, 0), [(-1.0, duration + 1.0)],
                 f=f0, db_at_1m=db_at_1m)
    ego = [(0.0, Track([0.0, 1.0], [[0, 0, 0], [0, 0, 0]]), Track([0.0, 1.0], [[0, 0, 0], [0, 0, 0]]))]
    scene = Scene(ego, np.array([[0.0, 0.0, 1.5], [0.0, 0.001, 1.5]]), np.eye(3), [src],
                  np.arange(0, duration + 1e-9, 0.05), duration, None, {'seed': 0})
    res = render(scene, o)
    y = res['audio'][0].astype(float)
    # instantaneous frequency: the analytic signal's phase slope (FFT Hilbert transform), smoothed over 10 ms
    Y = np.fft.fft(y)
    h = np.zeros(len(y))
    h[0] = 1
    h[1:(len(y) + 1) // 2] = 2
    if len(y) % 2 == 0:
        h[len(y) // 2] = 1
    z = np.fft.ifft(Y * h)
    ph = np.unwrap(np.angle(z))
    k = int(0.01 * fs)
    fi = (ph[k:] - ph[:-k]) / (2 * np.pi * k / fs)
    ti = (np.arange(len(fi)) + k / 2) / fs
    tq = np.arange(0.5, duration - 0.5, 0.05)
    f_meas = np.interp(tq, ti, fi)
    # the theory in closed form (straight line, constant speed, equal heights): (x0 + v·τ)² + d² = c²·(t − τ)²
    qa, qb, qc = speed_mps ** 2 - c * c, 2 * (x0 * speed_mps + c * c * tq), x0 ** 2 + offset_m ** 2 - (c * tq) ** 2
    tau = (-qb + np.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)
    tau = np.where(tau < tq, tau, (-qb - np.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa))
    xs = x0 + speed_mps * tau
    r = np.hypot(xs, offset_m)
    v_r = -speed_mps * xs / r                       # the source's speed toward the microphone at emission
    f_theory = f0 * c / (c - v_r)
    near = r <= 100.0
    alpha = float(air_absorption_db_per_m(f0, temp_c, humidity_pct)) if air_absorption else 0.0
    rms = np.sqrt(np.array([np.mean(y[max(int(tt * fs) - 1200, 0):int(tt * fs) + 1200] ** 2) for tt in tq]))
    l_meas = 20 * np.log10(rms / P_REF)
    # the theory at the received frequency (absorption applies at the emitted frequency, within 0.1 dB)
    l_theory = db_at_1m - 20 * np.log10(r) - alpha * r
    df = (f_meas - f_theory)[near]
    dl = (l_meas - l_theory)[near]
    return {'speed_mps': speed_mps, 'offset_m': offset_m, 'f0_hz': f0, 'c_mps': round(c, 3),
            'doppler_span_hz': [round(float(f_theory[near].min()), 2), round(float(f_theory[near].max()), 2)],
            'doppler_max_abs_err_hz': round(float(np.max(np.abs(df))), 4),
            'doppler_median_abs_err_hz': round(float(np.median(np.abs(df))), 4),
            'level_max_abs_err_db': round(float(np.max(np.abs(dl))), 3),
            'level_median_abs_err_db': round(float(np.median(np.abs(dl))), 3),
            'alpha_db_per_m': round(alpha, 6), 'samples': int(near.sum())}
