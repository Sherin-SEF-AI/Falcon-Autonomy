"""Weather and sensor physics CARLA 0.9.16 lacks: documented, seeded post-processing of the lidar, radar and camera
data the workers record and show (numpy at import; OpenCV inside the camera functions — CARLA's Python has both).

CARLA's lidar and radar ignore weather and its camera shows rain only as water on the lens and wet surfaces
(docs/carla-lessons.md). The models here add what a real sensor would suffer, each from a published relation where
one exists and with every other number labelled a heuristic (docs/sensor-physics.md lists parameters, units, sources
and limits). They are NEVER applied silently: a world (or a run's config) must carry

    "sensor_physics": {"enabled": true, "lidar": true, "radar": true, "camera": true, "rain_mm_h": null,
                       "lidar_params": {...}, "radar_params": {...}, "camera_params": {...}}

and a rig sensor can opt out with "weather_physics": false. run.json "sensor_physics" records the resolved
conditions and parameters, the models with their sources, and per sensor how many frames each effect touched; every
frame an effect touched says so in frames.jsonl ("physics": {sensor: [effects]}), as faults do.

Conditions (conditions()): CARLA's weather fields → the fog's meteorological visibility V = 7000 m / fog_density
(measured on 0.9.16, a 5 % contrast threshold, so the visible extinction is −ln 0.05 / V = 3.0 / V), the rain rate
R (mm/h) = RAIN_MM_H_AT_100 × precipitation / 100 unless "rain_mm_h" is given (HEURISTIC: CARLA's precipitation has
no physical unit), wetness and night (the sun below the horizon).

Lidar (apply_lidar(), per slice in the sensor's frame, before the sweep is assembled):
* extinction α = α_fog + α_rain (1/m). Fog: Kim, McArthur & Korevaar (2001), α(λ) = (3.0 / V)(λ / 550 nm)^−q with
  q = 1.6 (V > 50 km), 1.3 (6–50 km), 0.16 V + 0.34 (1–6 km), V − 0.5 (0.5–1 km), 0 (< 0.5 km), V in km (their 3.91 is
  for a 2 % threshold; 3.0 keeps the 5 % one the visibility was measured with). Rain: Carbonneau's empirical
  free-space-optics relation A = 1.076 R^0.67 dB/km (temperate climate; wavelength-independent in the near infrared
  because drops are far larger than the wavelength).
* dropout: a return CARLA gave (a clear-air return) survives the weather with probability
  p = exp(−(e^{2αd} − 1) d² / (κ ρ' R_max²)) — the ratio of detection probabilities with and without the two-way
  transmission e^{−2αd}, for a signal with fully developed speckle (exponentially distributed; Goodman 2007) whose
  mean, relative to the threshold, is κ ρ' (R_max / d)²: ρ' = the surface's reflectivity (a HEURISTIC table per CARLA
  tag, REFLECTIVITY) over the reference reflectivity the lidar's range is rated for (rho_ref), κ = 1 / ln 2 (a
  reference target at the rated range is detected half the time).
* intensity: CARLA's intensity × e^{−2αd} (the two-way transmission).
* backscatter clutter (spurious returns): the medium in front of a target returns light too. Fog: a range cell of
  length Δr = c τ / 2 at range r acts as a target of reflectivity π β Δr, β = α_fog / S with the lidar ratio of water
  droplets S = 18.8 sr at 905 nm (O'Connor, Illingworth & Hogan 2004), seen through the transmitter–receiver overlap
  O(r) = min(1, (r / r_overlap)²) (HEURISTIC near-field geometry). Rain: discrete drops of the Marshall–Palmer size
  distribution N(D) = 8000 e^{−ΛD} m⁻³ mm⁻¹, Λ = 4.1 R^−0.21 mm⁻¹ (Marshall & Palmer 1948), each a specular sphere
  (Fresnel reflectance of water 0.02, Lambertian-equivalent reflectivity 0.02 / 4) filling (D / w(r))² of a beam of
  width w(r) = w0 + θ r. Each cell's detection uses the same speckle statistics; the lidar reports the strongest
  return: a clutter return replaces the target's with probability S_c / (S_c + S_t) (the chance one exponential
  signal exceeds the other), or stands alone when the target's return was lost. Clutter points are tagged
  WEATHER_NOISE_TAG (30, "weather_noise", nuScenes-lidarseg "noise") with actor id 0; real returns keep CARLA's labels.
  Limits: clutter is drawn only along rays that returned a point in CARLA (CARLA reports no ray that hit nothing, so
  rays into open sky get none); multiple scattering, the drops' glory and the pulse shape are not modelled.

Radar (apply_radar(), per measurement):
* rain attenuation: ITU-R P.838-3 specific attenuation γ = k R^α dB/km at 77 GHz, k and α from its table (77 GHz:
  kH 1.1320, αH 0.7177, kV 1.1276, αV 0.7073), combined for an unknown (45°) polarisation and a horizontal path by its
  equations (4) and (5); two-way over the detection's range.
* desensitisation and dropout: Swerling-I detection P_d = P_fa^{1/(1 + SNR)} with SNR = SNR_ref (R_max / d)⁴ for a
  reference cross-section (CARLA's radar reports none), SNR_ref such that P_d = 0.5 at R_max; in rain the signal is
  attenuated and the noise floor raised by the rain clutter in the cell (SNR / (1 + CNR)); a detection survives with
  the ratio of the two P_d.
* rain clutter: the volume reflectivity of rain η = π⁵ |K|² Z / λ⁴ with Z = 200 R^1.6 mm⁶ m⁻³ (Marshall–Palmer Z–R)
  in the Rayleigh approximation (an UPPER bound at 77 GHz, where drops above about 1 mm scatter less than Rayleigh's
  law), over a cell of the radar's resolution; cells where clutter exceeds noise (CNR > 1) produce a spurious
  detection with probability p_spurious each (HEURISTIC: what a CFAR leaves of homogeneous clutter is not a published
  constant), at the radial velocity of drops falling at v(D0) = 9.65 − 10.3 e^{−0.6 D0} m/s (Atlas, Srivastava &
  Sekhon 1973; D0 = 3.67 / Λ, the median volume diameter) seen from the moving radar. Spurious detections are flagged
  in radar/<sensor>/<step>.flags.npy (uint8, 1 = weather clutter), a file written only when physics touched the frame.

Camera (CameraPhysics, per frame; the droplet layer persists from frame to frame):
* lens droplets: drops land at a rate that gives a steady-state cover of cover_max · R / (R + R_half) (HEURISTIC) and
  leave after an exponentially distributed residence time; each drop is a small lens that shows an inverted, minified,
  defocused view of the scene around it (a local displacement with blur inside, a darker rim from total internal
  reflection) — the geometric model of adherent drops of Roser & Geiger (2009) and You et al. (2016), simplified.
* wet-lens contrast loss: a water film scatters a share s of the light into a veil, I' = (1 − s) I + s Ī_blur
  (s = veil_max · wetness of the lens, HEURISTIC).
* glare and bloom: light from bright sources is spread by the lens with a point-spread function falling as 1 / θ²
  (the form of the Stiles–Holladay disability-glare relation, CIE 146:2002), scaled by a veiling-glare index of a few
  percent (ISO 9358; HEURISTIC value) that a wet lens raises. Sources: saturated pixels (the sun, lamps), weighted
  by an over-exposure factor because their true radiance is clipped, and at night the headlights of other vehicles
  facing the camera (a lamp's brightness: the veil it makes 1° away from 10 m, a HEURISTIC 1000 levels on a dry lens)
  — LABEL: HEURISTIC positions (CARLA does not expose its lamps: the front face of the box at
  0.65 m above its bottom, 0.3 m in from each side; one lamp for two-wheelers), inverse-square brightness, and assumed
  on whenever the sun is below the horizon; hidden lamps are dropped when the camera records depth.

Order with faults (core.faults): the weather physics first, then the sensor's faults — the scene's light passes the
atmosphere, then the water on the lens, then the optics (glare), and only then the sensor's own faults (dirt, the
"rain" fault of known size, the glare fault, rolling shutter, noise, compression) in core.faults' order. Mount
faults move the sensor before any of this. The two are recorded apart (frames.jsonl "faults" and "physics").
"""
from __future__ import annotations

import math
import zlib

import numpy as np

KEY = 'sensor_physics'
WEATHER_NOISE_TAG = 30                 # lidar label value of a weather return (groundtruth class 'weather_noise')
RAIN_MM_H_AT_100 = 50.0                # HEURISTIC: CARLA precipitation 100 % read as 50 mm/h (violent rain from there)
VIS_CONTRAST = -math.log(0.05)         # 3.0: the 5 % threshold CARLA's fog visibility was measured with
C_LIGHT = 299_792_458.0

# HEURISTIC diffuse reflectivity at 905 nm per CARLA 0.9.16 tag (typical published ranges, rounded; not measured)
REFLECTIVITY = {0: 0.2, 1: 0.12, 2: 0.25, 3: 0.3, 4: 0.3, 5: 0.3, 6: 0.4, 7: 0.4, 8: 0.8, 9: 0.5, 10: 0.3, 11: 0.2,
                12: 0.3, 13: 0.3, 14: 0.4, 15: 0.4, 16: 0.4, 17: 0.4, 18: 0.35, 19: 0.3, 20: 0.3, 21: 0.3, 22: 0.3,
                23: 0.05, 24: 0.5, 25: 0.2, 26: 0.3, 27: 0.2, 28: 0.5, 29: 0.4}
_REFL_LUT = np.full(256, 0.2)
for _t, _r in REFLECTIVITY.items():
    _REFL_LUT[_t] = _r

LIDAR_DEFAULTS = {
    'wavelength_nm': 905.0,
    'rho_ref': 0.1,                    # the rig's range is read as rated for a 10 % target (a common datasheet basis)
    'kappa': 1.0 / math.log(2.0),      # a reference target at the rated range: detected half the time
    'pulse_ns': 3.0,                   # HEURISTIC typical pulse length: range cell c·τ/2 = 0.45 m
    'r_overlap_m': 5.0,                # HEURISTIC: full transmitter–receiver overlap
    'r_min_m': 0.5,                    # HEURISTIC: near-range blanking
    'lidar_ratio_sr': 18.8,            # water droplets at 905 nm (O'Connor et al. 2004)
    'beam_w0_m': 0.01, 'beam_div_rad': 0.003,     # HEURISTIC typical exit aperture and divergence
    'drop_rho': 0.02 / 4.0,            # Fresnel reflectance of water at normal incidence, as a Lambertian equivalent
    'backscatter': True,
}
RADAR_DEFAULTS = {
    'frequency_ghz': 77.0,
    'p_fa': 1e-6,                      # HEURISTIC design false-alarm probability per cell
    'pd_at_range': 0.5,                # the rig's range: P_d of the reference target
    'k_water': 0.93,                   # |K|² of water (centimetre-wave value; lower at 77 GHz — clutter an upper bound)
    'sigma_ref_m2': 10.0,              # HEURISTIC reference cross-section (a car)
    'range_res_m': 0.5, 'az_res_deg': 2.0, 'el_res_deg': 5.0,     # HEURISTIC resolution cell
    'p_spurious': 0.01,                # HEURISTIC: per clutter-dominated cell
    'clutter': True,
}
CAMERA_DEFAULTS = {
    'droplets': True, 'contrast': True, 'glare': True, 'headlights': True,
    'cover_max': 0.25, 'rain_half_mm_h': 10.0,      # HEURISTIC steady-state drop cover
    'residence_s': 3.0,                # HEURISTIC mean time a drop stays
    'radius_median': 0.025, 'radius_sigma': 0.45,   # HEURISTIC drop radius, share of the image height (lognormal)
    'field': 3.5,                      # HEURISTIC: a drop shows a scene patch this many times its own size, inverted
    'blur': 0.25,                      # HEURISTIC defocus inside a drop, σ as a share of its radius
    'rim_dark': 0.35,                  # HEURISTIC darkening at the drop's rim
    'edge': 0.25,                      # HEURISTIC defocus of a drop's outline, a share of its radius
    'veil_max': 0.25,                  # HEURISTIC contrast veil of a fully wet lens
    'vgi': 0.03,                       # veiling-glare index of a dry lens (ISO 9358: typically 1–4 %)
    'wet_glare_gain': 4.0,             # HEURISTIC: a wet lens scatters this many times more
    'overexposure': 8.0,               # HEURISTIC: true radiance of a clipped pixel over the clip level
    'glare_theta0_deg': 0.3,           # PSF core, keeps 1/θ² finite
    'glare_reach_deg': 20.0,
    'headlight_veil_1deg': 1000.0,     # HEURISTIC: 8-bit levels a headlight 10 m away adds 1° from it (dry lens):
                                       # saturates within about 2°, about 10 levels at 10° (night exposure)
    'headlight_max_m': 150.0,
}
DEFAULTS = {'enabled': False, 'lidar': True, 'radar': True, 'camera': True, 'rain_mm_h': None,
            'lidar_params': {}, 'radar_params': {}, 'camera_params': {}}

# ITU-R P.838-3 (03/2005), table 5: (kH, αH, kV, αV) at these frequencies (GHz)
ITU_P838_3 = {77.0: (1.1320, 0.7177, 1.1276, 0.7073), 80.0: (1.1704, 0.7115, 1.1668, 0.7021)}

SOURCES = {
    'fog_visibility': 'Falcon, measured on CARLA 0.9.16: visibility ≈ 7000 m / fog_density at a 5 % contrast threshold '
                      '(docs/carla-lessons.md)',
    'fog_wavelength': 'I. I. Kim, B. McArthur, E. Korevaar, "Comparison of laser beam propagation at 785 nm and '
                      '1550 nm in fog and haze for optical wireless communications", Proc. SPIE 4214 (2001)',
    'rain_optical': 'Carbonneau\'s empirical rain attenuation for free-space optics (France, temperate climate), '
                    'A = 1.076 R^0.67 dB/km, as tabulated in FSO rain-attenuation studies; a 2009–2012 Czech FSO link '
                    'measured attenuation in agreement with it, tropical (Malaysian) measurements did not',
    'speckle': 'J. W. Goodman, Speckle Phenomena in Optics, Roberts & Co. (2007): fully developed speckle intensity is '
               'exponentially distributed',
    'lidar_ratio': 'E. J. O\'Connor, A. J. Illingworth, R. J. Hogan, "A technique for autocalibration of cloud lidar", '
                   'J. Atmos. Oceanic Technol. 21(5) 777–786 (2004): 18.8 ± 0.8 sr at 905 nm for liquid water clouds',
    'drop_size': 'J. S. Marshall, W. McK. Palmer, "The distribution of raindrops with size", J. Meteorol. 5, 165–166 '
                 '(1948)',
    'lidar_structure': 'the structure of attenuation, dropout and backscatter follows M. Hahner et al., "Fog '
                       'simulation on real LiDAR point clouds for 3D object detection in adverse weather", ICCV 2021, '
                       'and V. Kilic et al., "Lidar light scattering augmentation (LISA)", arXiv:2107.07004 (2021)',
    'radar_rain': 'Recommendation ITU-R P.838-3 (03/2005), "Specific attenuation model for rain for use in prediction '
                  'methods", table 5 and equations (4)–(5)',
    'radar_detection': 'Swerling I target in noise: P_d = P_fa^(1/(1+SNR)) (M. A. Richards, Fundamentals of Radar '
                       'Signal Processing, 2nd ed., 2014)',
    'radar_clutter': 'volume reflectivity of rain in the Rayleigh approximation with the Marshall–Palmer Z = 200 R^1.6',
    'fall_speed': 'D. Atlas, R. C. Srivastava, R. S. Sekhon, "Doppler radar characteristics of precipitation at '
                  'vertical incidence", Rev. Geophys. 11(1) (1973): v(D) = 9.65 − 10.3 e^(−0.6 D) m/s',
    'droplets': 'M. Roser, A. Geiger, "Video-based raindrop detection for improved image registration", ICCV '
                'Workshops 2009; S. You et al., "Adherent raindrop modeling, detection and removal in video", IEEE '
                'TPAMI 38(9) (2016) — the drop as a small lens, simplified to a local inverted displacement',
    'glare_psf': 'CIE 146:2002 / J. J. Vos (2003): disability glare falls as 1/θ² (Stiles–Holladay); ISO 9358:1994 '
                 'veiling glare index',
}


# ================================================================================================ configuration
def settings(cfg_block: dict | None) -> dict:
    """The resolved settings: DEFAULTS with the block on top, each model's parameters filled in."""
    b = dict(DEFAULTS, **(cfg_block or {}))
    b['lidar_params'] = dict(LIDAR_DEFAULTS, **(b.get('lidar_params') or {}))
    b['radar_params'] = dict(RADAR_DEFAULTS, **(b.get('radar_params') or {}))
    b['camera_params'] = dict(CAMERA_DEFAULTS, **(b.get('camera_params') or {}))
    return b


def validate(block: dict | None) -> list:
    out = []
    if not block:
        return out
    for k in block:
        if k not in DEFAULTS:
            out.append(f'sensor_physics: unknown key {k!r}')
    for sect, d in (('lidar_params', LIDAR_DEFAULTS), ('radar_params', RADAR_DEFAULTS),
                    ('camera_params', CAMERA_DEFAULTS)):
        for k in (block.get(sect) or {}):
            if k not in d:
                out.append(f'sensor_physics.{sect}: unknown parameter {k!r}')
    r = block.get('rain_mm_h')
    if r is not None and not 0 <= float(r) <= 300:
        out.append('sensor_physics.rain_mm_h: 0–300 mm/h')
    return out


def enabled_for(block: dict | None, sensor: dict) -> bool:
    """Whether physics applies to this rig sensor: the block enabled, its model on, the sensor not opted out."""
    if not block or not block.get('enabled'):
        return False
    if sensor.get('weather_physics') is False:
        return False
    t = sensor.get('type', '')
    model = 'camera' if t.startswith('camera.') else t
    return model in ('camera', 'lidar', 'radar') and bool(block.get(model, True))


# ================================================================================================ conditions
def fog_visibility_m(fog_density: float) -> float | None:
    return None if fog_density <= 0 else 7000.0 / fog_density


def kim_q(visibility_km: float) -> float:
    """Kim et al. (2001)'s size-distribution exponent q for a visibility in km."""
    v = visibility_km
    if v > 50:
        return 1.6
    if v > 6:
        return 1.3
    if v > 1:
        return 0.16 * v + 0.34
    if v > 0.5:
        return v - 0.5
    return 0.0


def fog_extinction(visibility_m: float | None, wavelength_nm: float = 905.0) -> float:
    """Extinction coefficient (1/m) of fog of this visibility at the wavelength (Kim et al. 2001, 5 % threshold)."""
    if not visibility_m:
        return 0.0
    return VIS_CONTRAST / visibility_m * (wavelength_nm / 550.0) ** (-kim_q(visibility_m / 1000.0))


def rain_extinction_optical(rain_mm_h: float) -> float:
    """Extinction coefficient (1/m) of rain in the near infrared: Carbonneau's 1.076 R^0.67 dB/km."""
    if rain_mm_h <= 0:
        return 0.0
    return 1.076 * rain_mm_h ** 0.67 * math.log(10.0) / 10.0 / 1000.0


def conditions(weather: dict, block: dict | None = None) -> dict:
    """The physical conditions of a CARLA weather (all its fields, core.weather.resolve) for the models."""
    b = settings(block)
    fog = float(weather.get('fog_density', 0.0) or 0.0)
    prec = float(weather.get('precipitation', 0.0) or 0.0)
    rain = float(b['rain_mm_h']) if b.get('rain_mm_h') is not None else RAIN_MM_H_AT_100 * max(prec, 0.0) / 100.0
    vis = fog_visibility_m(fog)
    wl = float(b['lidar_params']['wavelength_nm'])
    a_fog = fog_extinction(vis, wl)
    a_rain = rain_extinction_optical(rain)
    wet = max(float(weather.get('wetness', 0.0) or 0.0) / 100.0, prec / 100.0)
    return {'fog_density': fog, 'visibility_m': vis, 'rain_mm_h': round(rain, 4),
            'rain_source': 'given' if b.get('rain_mm_h') is not None else
            f'heuristic: {RAIN_MM_H_AT_100:g} mm/h × precipitation / 100',
            'alpha_fog_per_m': a_fog, 'alpha_rain_per_m': a_rain, 'alpha_lidar_per_m': a_fog + a_rain,
            'lens_wetness': min(max(prec / 100.0, 0.0), 1.0), 'surface_wetness': min(wet, 1.0),
            'night': float(weather.get('sun_altitude_angle', 45.0)) < 0.0,
            'sun_altitude_deg': float(weather.get('sun_altitude_angle', 45.0))}


def sensor_rng(seed: int, sid: str, salt: str = 'physics'):
    """A generator per sensor, independent of the faults' (seeded by (seed, crc(sensor id), crc(salt)))."""
    return np.random.default_rng([int(seed), zlib.crc32(sid.encode()), zlib.crc32(salt.encode())])


# ================================================================================================ lidar
def reflectivity(tags) -> np.ndarray:
    return _REFL_LUT[np.asarray(tags, np.int64) & 255]


def lidar_survival(d, rho, alpha: float, r_max: float, p: dict) -> np.ndarray:
    """The probability a clear-air return at range d (m) of a surface of reflectivity rho survives extinction alpha."""
    d = np.asarray(d, float)
    if alpha <= 0:
        return np.ones_like(d)
    m0 = p['kappa'] * np.asarray(rho, float) / p['rho_ref'] * (r_max / np.maximum(d, 1e-6)) ** 2
    return np.exp(-np.expm1(2.0 * alpha * d) / m0)


def lidar_signal(d, rho, alpha: float, r_max: float, p: dict) -> np.ndarray:
    """Mean signal over the threshold of a target return in the weather (the speckle mean)."""
    d = np.maximum(np.asarray(d, float), 1e-6)
    return p['kappa'] * np.asarray(rho, float) / p['rho_ref'] * (r_max / d) ** 2 * np.exp(-2.0 * alpha * d)


def clutter_table(cond: dict, r_max: float, p: dict) -> dict:
    """Per range cell (from r_min to the lidar's range): the probability of a detected weather return in the cell
    (fog volume and rain drops), its mean signal, and the cumulative probability of a weather return before it."""
    dr = C_LIGHT * p['pulse_ns'] * 1e-9 / 2.0
    r = np.arange(max(p['r_min_m'], dr / 2), max(r_max, dr), dr) + dr / 2
    alpha = cond['alpha_lidar_per_m']
    T2 = np.exp(-2.0 * alpha * r)
    O = np.minimum(1.0, (r / p['r_overlap_m']) ** 2)
    gain = p['kappa'] / p['rho_ref'] * (r_max / r) ** 2 * O * T2
    # fog: the cell as a target of reflectivity π β Δr
    beta = cond['alpha_fog_per_m'] / p['lidar_ratio_sr']
    s_fog = gain * math.pi * beta * dr
    with np.errstate(divide='ignore', over='ignore'):
        p_fog = np.where(s_fog > 0, np.exp(-1.0 / np.maximum(s_fog, 1e-300)), 0.0)
    # rain: discrete Marshall–Palmer drops filling (D / w)² of the beam
    R = cond['rain_mm_h']
    p_rain = np.zeros_like(r)
    s_rain = np.zeros_like(r)
    if R > 0:
        lam = 4.1 * R ** -0.21
        D = np.linspace(0.1, 6.0, 60)                              # mm
        dD = D[1] - D[0]
        N = 8000.0 * np.exp(-lam * D)                              # m⁻³ mm⁻¹
        w = p['beam_w0_m'] + p['beam_div_rad'] * r                 # m
        f = np.minimum(1.0, (D[None, :] * 1e-3 / w[:, None]) ** 2)
        s = gain[:, None] * p['drop_rho'] * f
        with np.errstate(divide='ignore', over='ignore'):
            pd = np.exp(-1.0 / np.maximum(s, 1e-300))
        dens = N[None, :] * pd * dD                                # detected drops per m³, per D bin
        nu = (math.pi * w ** 2 / 4.0) * dr * dens.sum(1)           # expected detected drops in the cell
        p_rain = -np.expm1(-nu)
        s_rain = np.where(dens.sum(1) > 0, (dens * s).sum(1) / np.maximum(dens.sum(1), 1e-300), 0.0)
    if not p.get('backscatter', True):
        p_fog = np.zeros_like(r)
        p_rain = np.zeros_like(r)
    p_cell = 1.0 - (1.0 - p_fog) * (1.0 - p_rain)
    w_f = p_fog / np.maximum(p_fog + p_rain, 1e-300)
    s_cell = w_f * s_fog + (1.0 - w_f) * s_rain
    surv = np.cumprod(1.0 - p_cell)
    F = 1.0 - surv                                                 # P(a weather return in cells up to and incl. i)
    return {'r': r, 'dr': dr, 'p_cell': p_cell, 's_cell': s_cell, 'F': F, 'p_fog': p_fog, 'p_rain': p_rain}


def _cum_before(tab: dict, d: np.ndarray) -> np.ndarray:
    """P(a weather return in the cells nearer than range d)."""
    k = np.searchsorted(tab['r'] + tab['dr'] / 2, d, side='right')       # cells wholly before d
    F0 = np.r_[0.0, tab['F']]
    return F0[np.minimum(k, len(tab['F']))]


def expected_real_kept(d, rho, cond: dict, r_max: float, p: dict, tab: dict | None = None) -> np.ndarray:
    """The model's expected share of real returns (range d, reflectivity rho) kept: dropout and replacement by a
    stronger weather return in front. apply_lidar draws exactly this (the tests check it)."""
    d = np.asarray(d, float)
    rho = np.broadcast_to(np.asarray(rho, float), d.shape)
    alpha = cond['alpha_lidar_per_m']
    keep = lidar_survival(d, rho, alpha, r_max, p)
    tab = clutter_table(cond, r_max, p) if tab is None else tab
    st = lidar_signal(d, rho, alpha, r_max, p)
    # P(first weather return in cell i and it beats the target), summed over cells nearer than d
    first = np.r_[tab['F'][0], np.diff(tab['F'])]
    n_before = np.searchsorted(tab['r'] + tab['dr'] / 2, d, side='right')
    out = np.empty_like(d)
    sc = tab['s_cell']
    for i in range(len(d)):
        k = int(n_before.flat[i])
        if k <= 0:
            out.flat[i] = keep.flat[i]
            continue
        beat = sc[:k] / np.maximum(sc[:k] + st.flat[i], 1e-300)
        out.flat[i] = keep.flat[i] * (1.0 - float(np.sum(first[:k] * beat)))
    return out


def apply_lidar(xyz, intensity, tag, idx, cond: dict, r_max: float, rng, params: dict | None = None,
                tab: dict | None = None):
    """One slice of a lidar (points in the sensor frame, CARLA's intensity, tags, actor ids) through the weather.
    Returns (xyz, intensity, tag, idx, stats). Draws three uniforms per point (reproducible for a seed)."""
    p = dict(LIDAR_DEFAULTS, **(params or {}))
    xyz = np.asarray(xyz, np.float64)
    n = len(xyz)
    u = rng.random((n, 3)) if n else np.zeros((0, 3))
    stats = {'points_in': n, 'dropped': 0, 'clutter': 0, 'replaced': 0}
    alpha = cond['alpha_lidar_per_m']
    if n == 0 or (alpha <= 0 and cond['rain_mm_h'] <= 0):
        return xyz, np.asarray(intensity), np.asarray(tag), np.asarray(idx), stats
    d = np.linalg.norm(xyz, axis=1)
    rho = reflectivity(tag)
    keep = u[:, 0] < lidar_survival(d, rho, alpha, r_max, p)
    tab = clutter_table(cond, r_max, p) if tab is None else tab
    Fd = _cum_before(tab, d)
    has_c = u[:, 1] < Fd
    cell = np.minimum(np.searchsorted(tab['F'], u[:, 1], side='right'), len(tab['r']) - 1)
    rc = tab['r'][cell]
    sc = tab['s_cell'][cell]
    st = lidar_signal(d, rho, alpha, r_max, p)
    beat = u[:, 2] < sc / np.maximum(sc + st, 1e-300)
    use_c = has_c & (~keep | beat)
    out_keep = keep & ~use_c
    sel = out_keep | use_c
    new_xyz = xyz.copy()
    unit = xyz / np.maximum(d, 1e-9)[:, None]
    new_xyz[use_c] = unit[use_c] * rc[use_c, None]
    inten = np.asarray(intensity, np.float64) * np.exp(-2.0 * alpha * d)
    a_carla = 0.004                                     # CARLA's default atmosphere_attenuation_rate (its convention)
    rho_c = np.clip(sc / (p['kappa'] / p['rho_ref'] * (r_max / np.maximum(rc, 1e-6)) ** 2), 0.0, 1.0)
    inten[use_c] = np.exp(-a_carla * rc[use_c]) * np.exp(-2.0 * alpha * rc[use_c]) * rho_c[use_c]
    t2 = np.asarray(tag).copy()
    i2 = np.asarray(idx).copy()
    t2[use_c] = WEATHER_NOISE_TAG
    i2[use_c] = 0
    stats.update(dropped=int((~sel).sum()), clutter=int(use_c.sum()), replaced=int((use_c & keep).sum()))
    return new_xyz[sel], inten[sel].astype(np.float32), t2[sel], i2[sel], stats


# ================================================================================================ radar
def p838_coefficients(f_ghz: float = 77.0) -> tuple:
    """(k, α) for a horizontal path and 45° (unknown) polarisation, equations (4)–(5) of P.838-3."""
    if f_ghz not in ITU_P838_3:
        raise ValueError(f'ITU-R P.838-3 coefficients are tabulated here only for {sorted(ITU_P838_3)} GHz')
    kH, aH, kV, aV = ITU_P838_3[f_ghz]
    k = (kH + kV) / 2.0
    a = (kH * aH + kV * aV) / (2.0 * k)
    return k, a


def radar_rain_db_per_km(rain_mm_h: float, f_ghz: float = 77.0) -> float:
    if rain_mm_h <= 0:
        return 0.0
    k, a = p838_coefficients(f_ghz)
    return k * rain_mm_h ** a


def swerling1_pd(snr, p_fa: float) -> np.ndarray:
    return np.power(p_fa, 1.0 / (1.0 + np.asarray(snr, float)))


def radar_snr_ref(p: dict) -> float:
    """SNR of the reference target at the rated range: P_d = pd_at_range there."""
    return math.log(p['p_fa']) / math.log(p['pd_at_range']) - 1.0


def radar_cnr(r, rain_mm_h: float, r_max: float, p: dict) -> np.ndarray:
    """Clutter-to-noise ratio of rain in a resolution cell at range r (Rayleigh volume reflectivity: an upper bound)."""
    r = np.maximum(np.asarray(r, float), 0.1)
    if rain_mm_h <= 0:
        return np.zeros_like(r)
    lam = C_LIGHT / (p['frequency_ghz'] * 1e9)
    Z = 200.0 * rain_mm_h ** 1.6 * 1e-18                                   # m⁶ m⁻³
    eta = math.pi ** 5 * p['k_water'] * Z / lam ** 4                       # m² m⁻³
    vol = r ** 2 * math.radians(p['az_res_deg']) * math.radians(p['el_res_deg']) * p['range_res_m']
    L = 10.0 ** (-2.0 * radar_rain_db_per_km(rain_mm_h, p['frequency_ghz']) * r / 1000.0 / 10.0)
    return radar_snr_ref(p) * (eta * vol / p['sigma_ref_m2']) * (r_max / r) ** 4 * L


def radar_survival(d, rain_mm_h: float, r_max: float, p: dict) -> np.ndarray:
    d = np.maximum(np.asarray(d, float), 0.1)
    if rain_mm_h <= 0:
        return np.ones_like(d)
    snr0 = radar_snr_ref(p) * (r_max / d) ** 4
    L = 10.0 ** (-2.0 * radar_rain_db_per_km(rain_mm_h, p['frequency_ghz']) * d / 1000.0 / 10.0)
    snr = snr0 * L / (1.0 + radar_cnr(d, rain_mm_h, r_max, p))
    return swerling1_pd(snr, p['p_fa']) / swerling1_pd(snr0, p['p_fa'])


def fall_speed(d_mm: float) -> float:
    return max(9.65 - 10.3 * math.exp(-0.6 * d_mm), 0.0)


def apply_radar(dets, cond: dict, sensor: dict, ego_speed_mps: float, rng, params: dict | None = None):
    """CARLA radar detections (N × 4 float32: velocity, azimuth, altitude, depth) through the rain. Returns
    (detections, flags uint8 — 1 for weather clutter, stats)."""
    p = dict(RADAR_DEFAULTS, **(params or {}))
    dets = np.asarray(dets, np.float32).reshape(-1, 4)
    R = cond['rain_mm_h']
    r_max = float(sensor.get('range', 100.0))
    stats = {'detections_in': int(len(dets)), 'dropped': 0, 'clutter': 0}
    u = rng.random(len(dets))
    if R <= 0:
        return dets, np.zeros(len(dets), np.uint8), stats
    keep = u < radar_survival(dets[:, 3], R, r_max, p)
    out = dets[keep]
    stats['dropped'] = int((~keep).sum())
    extra = np.zeros((0, 4), np.float32)
    if p.get('clutter', True):
        rr = np.arange(p['range_res_m'] / 2, r_max, p['range_res_m'])
        dom = radar_cnr(rr, R, r_max, p) > 1.0
        n_az = max(int(float(sensor.get('horizontal_fov', 30.0)) / p['az_res_deg']), 1)
        n_cells = int(dom.sum()) * n_az
        n = int(rng.poisson(p['p_spurious'] * n_cells)) if n_cells else 0
        if n:
            ranges = rng.choice(rr[dom], n) + rng.uniform(-0.5, 0.5, n) * p['range_res_m']
            hf, vf = math.radians(float(sensor.get('horizontal_fov', 30.0))), math.radians(
                float(sensor.get('vertical_fov', 10.0)))
            az = rng.uniform(-hf / 2, hf / 2, n)
            alt = rng.uniform(-vf / 2, vf / 2, n)
            lam = 4.1 * R ** -0.21
            vfall = fall_speed(3.67 / lam)
            vel = -ego_speed_mps * np.cos(alt) * np.cos(az) - vfall * np.sin(alt)
            extra = np.c_[vel, az, alt, ranges].astype(np.float32)
            stats['clutter'] = n
    flags = np.r_[np.zeros(len(out), np.uint8), np.ones(len(extra), np.uint8)]
    return np.concatenate([out, extra]).astype(np.float32), flags, stats


# ================================================================================================ camera
class DropletLayer:
    """The drops on one camera's lens, carried from frame to frame (seeded). A drop: (u, v, r) — centre in pixels,
    radius in pixels."""

    def __init__(self, rng, width: int, height: int, params: dict | None = None):
        self.rng, self.w, self.h = rng, int(width), int(height)
        self.p = dict(CAMERA_DEFAULTS, **(params or {}))
        self.drops = np.zeros((0, 3))

    def steady_cover(self, rain_mm_h: float) -> float:
        p = self.p
        return p['cover_max'] * rain_mm_h / (rain_mm_h + p['rain_half_mm_h']) if rain_mm_h > 0 else 0.0

    def advance(self, dt: float, rain_mm_h: float) -> np.ndarray:
        p = self.p
        if len(self.drops):                                       # each drop leaves with rate 1 / residence
            stay = self.rng.random(len(self.drops)) >= -math.expm1(-dt / p['residence_s'])
            self.drops = self.drops[stay]
        cover = self.steady_cover(rain_mm_h)
        if cover > 0:
            mu, s = math.log(p['radius_median'] * self.h), p['radius_sigma']
            mean_area = math.pi * math.exp(2 * mu + 2 * s * s)
            rate = cover * self.w * self.h / (p['residence_s'] * mean_area)       # drops per second
            k = int(self.rng.poisson(rate * dt))
            if k:
                new = np.c_[self.rng.uniform(0, self.w, k), self.rng.uniform(0, self.h, k),
                            np.maximum(self.rng.lognormal(mu, s, k), 2.0)]
                self.drops = np.vstack([self.drops, new])
        return self.drops.copy()

    def warm(self, rain_mm_h: float, seconds: float = None, dt: float = 0.1):
        """Fill the lens to its steady state (as if it had rained for a while before the run)."""
        for _ in range(int((seconds or 3 * self.p['residence_s']) / dt)):
            self.advance(dt, rain_mm_h)


def render_droplets(img: np.ndarray, drops: np.ndarray, params: dict | None = None) -> np.ndarray:
    """Each drop shows an inverted, minified, defocused patch of the scene around it, with a darker rim."""
    import cv2
    p = dict(CAMERA_DEFAULTS, **(params or {}))
    if not len(drops):
        return img
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    rmed = float(np.median(drops[:, 2]))
    blurred = cv2.GaussianBlur(img, (0, 0), max(p['blur'] * rmed, 0.8)).astype(np.float32)
    for cx, cy, r in drops:
        x0, x1 = max(int(cx - r - 1), 0), min(int(cx + r + 2), w)
        y0, y1 = max(int(cy - r - 1), 0), min(int(cy + r + 2), h)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = xx - cx, yy - cy
        rho = np.sqrt(dx * dx + dy * dy) / r
        a = np.clip((1.0 - rho) / max(p['edge'], 1.5 / r), 0.0, 1.0)  # the outline defocused
        if not (a > 0).any():
            continue
        sx = np.clip(cx - dx * p['field'], 0, w - 1).astype(np.float32)
        sy = np.clip(cy - dy * p['field'], 0, h - 1).astype(np.float32)
        patch = cv2.remap(blurred, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        patch *= (1.0 - p['rim_dark'] * np.clip(rho, 0, 1) ** 4)[..., None]
        reg = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = reg * (1 - a[..., None]) + patch * a[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def wet_contrast(img: np.ndarray, s: float) -> np.ndarray:
    import cv2
    if s <= 0:
        return img
    h, w = img.shape[:2]
    f = max(int(w // 160), 1)                      # the veil is smooth: blurred at 1/f size, then enlarged
    small = cv2.resize(img, (max(w // f, 1), max(h // f, 1)), interpolation=cv2.INTER_AREA)
    veil = cv2.resize(cv2.GaussianBlur(small, (0, 0), max(0.05 * w / f, 1.0)), (w, h),
                      interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return np.clip((1.0 - s) * img.astype(np.float32) + s * veil, 0, 255).astype(np.uint8)


def kernel_area_deg2(theta0: float, reach: float) -> float:
    """∫ 2πθ / (θ² + θ0²) dθ over 0..reach: the normaliser of the 1/θ² kernel, in deg²."""
    return math.pi * math.log((reach * reach + theta0 * theta0) / (theta0 * theta0))


def headlight_energy(veil_1deg: float, dist_m: float, deg_per_px: float, p: dict) -> float:
    """A lamp's energy (clipped-pixel units) whose dry-lens veil is veil_1deg levels at 1° from 10 m, inverse
    square with distance: veil(θ) = vgi · 255 · E · a² / (A (θ² + θ0²)), a the pixel's angle, A the kernel's area."""
    A = kernel_area_deg2(p['glare_theta0_deg'], p['glare_reach_deg'])
    e10 = veil_1deg * A * (1.0 + p['glare_theta0_deg'] ** 2) / (p['vgi'] * 255.0 * deg_per_px * deg_per_px)
    return e10 * (10.0 / max(dist_m, 1.0)) ** 2


def glare_kernel(deg_per_px: float, theta0: float, reach: float) -> np.ndarray:
    n = max(int(reach / deg_per_px), 2)
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1].astype(np.float64)
    th = np.hypot(xx, yy) * deg_per_px
    k = 1.0 / (th * th + theta0 * theta0)
    k[th > reach] = 0.0
    return (k / k.sum()).astype(np.float32)


def glare(img: np.ndarray, hfov_deg: float, wet: float, sources=(), params: dict | None = None,
          bright_from: np.ndarray | None = None) -> tuple:
    """Veiling glare and bloom from saturated pixels (of bright_from, the scene before the lens water, else img) and
    extra point sources [(u, v, energy in clipped-pixel units)]. Returns (image, the veil's mean in 8-bit levels)."""
    import cv2
    p = dict(CAMERA_DEFAULTS, **(params or {}))
    h, w = img.shape[:2]
    src = img if bright_from is None else bright_from
    E = (src.max(axis=2) >= 250).astype(np.float32) * p['overexposure']
    f = max(min(w // 160, 8), 1)                  # the veil is smooth: computed at about 160 px wide
    Es = cv2.resize(E, (max(w // f, 1), max(h // f, 1)), interpolation=cv2.INTER_AREA) * (f * f)
    for u, v, e in sources:
        iu, iv = int(u / f), int(v / f)
        if 0 <= iu < Es.shape[1] and 0 <= iv < Es.shape[0]:
            Es[iv, iu] += e
    if Es.sum() <= 0:
        return img, 0.0
    k = glare_kernel(hfov_deg / w * f, p['glare_theta0_deg'], p['glare_reach_deg'])
    veil_s = cv2.filter2D(Es, -1, k, borderType=cv2.BORDER_CONSTANT) / (f * f)
    veil = cv2.resize(veil_s, (w, h), interpolation=cv2.INTER_LINEAR)
    gain = p['vgi'] * (1.0 + p['wet_glare_gain'] * wet) * 255.0
    add = veil * gain
    out = img.astype(np.float32) + add[..., None]
    for u, v, e in sources:                                       # the lamp itself: a clipped core
        r = max(1.0, 0.002 * w)
        cv2.circle(out, (int(round(u)), int(round(v))), int(round(r)), (255, 255, 255), -1, lineType=cv2.LINE_AA)
    return np.clip(out, 0, 255).astype(np.uint8), float(add.mean())


def headlight_points(known: dict, sensor: dict, max_m: float = 150.0) -> list:
    """Candidate headlights of other vehicles facing a pinhole camera: [(u, v, x_forward_m, distance_m)], from the
    scene boxes (camgt.scene_boxes 'known': centre, R, size in the ego's vehicle frame). HEURISTIC lamp positions."""
    from .rig import carla_rotation
    pose = sensor['pose']
    Rb = carla_rotation(pose['roll'], pose['pitch'], pose['yaw'])
    t = np.array([pose['x'], pose['y'], pose['z']])
    W, H, fov = int(sensor['width']), int(sensor['height']), float(sensor.get('fov', 90.0))
    fpx = W / (2.0 * math.tan(math.radians(fov) / 2.0))
    out = []
    for o in (known or {}).values():
        cat = str(o.get('category') or '')
        if not cat.startswith('vehicle.') or cat == 'vehicle.bicycle':
            continue
        c = np.asarray(o['center'], float)
        R = np.asarray(o['R'], float)
        L, Wd, Hh = (float(x) for x in o['size'])
        two = cat == 'vehicle.motorcycle'
        lamps = [(L / 2, 0.0)] if two else [(L / 2, -(Wd / 2 - 0.3)), (L / 2, Wd / 2 - 0.3)]
        for lx, ly in lamps:
            P = c + R @ np.array([lx, ly, -Hh / 2 + 0.65])
            to_cam = t - P
            dist = float(np.linalg.norm(to_cam))
            if dist > max_m or dist < 1.0 or float(R[:, 0] @ to_cam) <= 0:      # its front must face the camera
                continue
            q = Rb.T @ (P - t)                                    # camera body: x forward, y right, z up
            if q[0] <= 0.5:
                continue
            u = fpx * q[1] / q[0] + W / 2.0
            v = -fpx * q[2] / q[0] + H / 2.0
            if 0 <= u < W and 0 <= v < H:
                out.append((float(u), float(v), float(q[0]), dist))
    return out


class CameraPhysics:
    """One camera's weather physics in a run: its droplet layer (advanced in the run's thread, in step order, so the
    drops are the same for a seed whatever the writer threads do) and what it applied."""

    def __init__(self, sensor: dict, cond: dict, seed: int, params: dict | None = None, warm: bool = True):
        self.s, self.cond = sensor, cond
        self.p = dict(CAMERA_DEFAULTS, **(params or {}))
        self.rng = sensor_rng(seed, sensor['id'], 'camera')
        self.layer = DropletLayer(self.rng, sensor['width'], sensor['height'], self.p)
        if warm and self.p['droplets'] and cond['rain_mm_h'] > 0:
            self.layer.warm(cond['rain_mm_h'])
        self.counts: dict = {}

    def plan(self, dt: float, known: dict | None = None) -> dict:
        """This frame's effects (the main thread): the drops now and the headlight candidates. Returns the frame's
        plan for apply() (in a writer thread) — 'effects' lists what will touch the frame."""
        c, p = self.cond, self.p
        eff = []
        drops = self.layer.advance(dt, c['rain_mm_h']) if p['droplets'] else np.zeros((0, 3))
        if len(drops):
            eff.append('droplets')
        veil = p['veil_max'] * c['lens_wetness'] if p['contrast'] else 0.0
        if veil > 0:
            eff.append('wet_contrast')
        lamps = []
        if p['glare'] and p['headlights'] and c['night'] and known and self.s.get('type') == 'camera.pinhole':
            lamps = headlight_points(known, self.s, p['headlight_max_m'])
        if p['glare']:
            eff.append('glare')
        if lamps:
            eff.append('headlight_glare')
        for e in eff:
            self.counts[e] = self.counts.get(e, 0) + 1
        return {'drops': drops, 'veil': veil, 'lamps': lamps, 'effects': eff}

    def apply(self, img: np.ndarray, plan: dict, depth: np.ndarray | None = None) -> tuple:
        """(image, effects) — img H × W × 3 uint8 BGR; depth: the camera's planar depth (m) for hiding lamps."""
        p = self.p
        scene = img
        drops = plan['drops']
        h0, w0 = img.shape[:2]
        if len(drops) and (w0 != self.layer.w or h0 != self.layer.h):      # a view of another size (Live)
            drops = drops * np.array([w0 / self.layer.w, h0 / self.layer.h, h0 / self.layer.h])
        out = render_droplets(img, drops, p) if len(drops) else img
        if plan['veil'] > 0:
            out = wet_contrast(out, plan['veil'])
        eff = list(plan['effects'])
        if p['glare']:
            h, w = img.shape[:2]
            hfov = float(self.s.get('fov', 90.0)) if self.s.get('type') != 'camera.fisheye' else \
                float((self.s.get('lens') or {}).get('fov', 190.0))          # a fisheye: its lens' field, about
            sources = []
            for u, v, x, dist in plan['lamps']:
                if depth is not None:
                    iu, iv = min(int(u), depth.shape[1] - 1), min(int(v), depth.shape[0] - 1)
                    if x > float(depth[iv, iu]) + 1.0:                # something nearer hides the lamp
                        continue
                sources.append((u, v, headlight_energy(p['headlight_veil_1deg'], dist, hfov / w, p)))
            if plan['lamps'] and not sources and 'headlight_glare' in eff:
                eff.remove('headlight_glare')
            out, _ = glare(out, hfov, self.cond['lens_wetness'], sources, p, bright_from=scene)
        return out, eff


# ================================================================================================ a run
class Session:
    """A worker's physics for one run: the conditions, a CameraPhysics per camera, generators per lidar and radar,
    and the record for run.json. None of it exists unless the config's sensor_physics is enabled (from_config)."""

    def __init__(self, cfg: dict):
        self.block = settings(cfg.get(KEY))
        self.cond = conditions(cfg.get('weather') or {}, cfg.get(KEY))
        self.seed = int(cfg.get('seed', 0))
        self.dt = 1.0 / float(cfg.get('fps', 20.0))
        self.sensors = {s['id']: s for s in (cfg.get('rig') or {}).get('sensors', [])
                        if enabled_for(cfg.get(KEY), s)}
        self.cams = {sid: CameraPhysics(s, self.cond, self.seed, self.block['camera_params'])
                     for sid, s in self.sensors.items() if s['type'].startswith('camera.')}
        self.rngs = {sid: sensor_rng(self.seed, sid, s['type']) for sid, s in self.sensors.items()
                     if s['type'] in ('lidar', 'radar')}
        self.tabs = {}
        self.totals: dict = {}

    @classmethod
    def from_config(cls, cfg: dict):
        b = cfg.get(KEY)
        if not b or not b.get('enabled'):
            return None
        bad = validate(b)
        if bad:
            raise ValueError('; '.join(bad))
        return cls(cfg)

    def set_weather(self, weather: dict):
        """A weather change mid-run (a live event): new conditions for every model."""
        self.cond = conditions(weather, self.block)
        for c in self.cams.values():
            c.cond = self.cond
        self.tabs.clear()

    def camera(self, sid: str):
        return self.cams.get(sid)

    def needs_actors(self) -> bool:
        """Whether a camera wants the actors' boxes this run (night headlights)."""
        return self.cond['night'] and any(c.p['glare'] and c.p['headlights'] and c.s.get('type') == 'camera.pinhole'
                                          for c in self.cams.values())

    def camera_frame(self, sid: str, dt: float, boxes: dict | None = None):
        """(CameraPhysics, its plan for this frame) for a camera of the run, or None; boxes: camgt.scene_boxes."""
        c = self.cams.get(sid)
        if c is None:
            return None
        return c, c.plan(dt, (boxes or {}).get('known'))

    def _add(self, sid, stats: dict):
        t = self.totals.setdefault(sid, {})
        for k, v in stats.items():
            t[k] = t.get(k, 0) + int(v)

    def lidar(self, sensor: dict, xyz, intensity, tag, idx):
        sid = sensor['id']
        if sid not in self.rngs:
            return xyz, intensity, tag, idx, None
        r_max = float(sensor.get('range', 100.0))
        tab = self.tabs.get(sid)
        if tab is None:
            tab = self.tabs[sid] = clutter_table(self.cond, r_max, self.block['lidar_params'])
        out = apply_lidar(xyz, intensity, tag, idx, self.cond, r_max, self.rngs[sid], self.block['lidar_params'],
                          tab)
        st = out[4]
        self._add(sid, dict(st, slices=1, slices_touched=int(st['dropped'] + st['clutter'] > 0)))
        return out

    def radar(self, sensor: dict, dets, ego_speed_mps: float):
        sid = sensor['id']
        if sid not in self.rngs:
            return dets, None, None
        out, flags, st = apply_radar(dets, self.cond, sensor, ego_speed_mps, self.rngs[sid],
                                     self.block['radar_params'])
        self._add(sid, dict(st, frames=1, frames_touched=int(st['dropped'] + st['clutter'] > 0)))
        return out, flags, st

    def report(self) -> dict:
        frames = {sid: dict(c.counts) for sid, c in self.cams.items()}
        for sid, t in self.totals.items():
            frames[sid] = dict(t)
        return {'enabled': True, 'conditions': {k: (round(v, 8) if isinstance(v, float) else v)
                                                for k, v in self.cond.items()},
                'settings': {k: v for k, v in self.block.items()}, 'sensors': sorted(self.sensors),
                'models': {'lidar': ['fog and rain extinction', 'dropout (speckle)', 'intensity',
                                     'backscatter clutter (weather_noise, tag 30)'],
                           'radar': ['rain attenuation (ITU-R P.838-3)', 'desensitisation by rain clutter',
                                     'spurious rain detections (heuristic rate)'],
                           'camera': ['lens droplets', 'wet-lens contrast', 'glare and bloom',
                                      'night headlight glare (heuristic positions)']},
                'order': 'weather physics first (atmosphere, lens water, optics), then the sensor\'s faults',
                'frames': frames, 'sources': SOURCES, 'doc': 'docs/sensor-physics.md'}
