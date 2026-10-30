"""Synthetic array audio (core.audio): the physics against theory on synthetic trajectories — retarded time and the
Doppler shift, spherical spreading, ISO 9613-1 air absorption, the ground image, occlusion — the CNOSSOS-EU emission,
SRP-PHAT direction of arrival, determinism, the WAV writer, the rig's audio_array and the product from a run folder."""
import json
import math
import os

import numpy as np
import pytest

from falcon_autonomy.core import audio as A
from falcon_autonomy.core import rig as R
from test_gui import qapp, slot_exceptions, window  # noqa: E402,F401  (the GUI tests' fixtures)


# ------------------------------------------------------------------------------------------- physics
@pytest.mark.parametrize('temp, rh, table', [
    # ISO 9613-2:1996 Table 2 (dB/km at the octave bands' exact midband frequencies, from ISO 9613-1)
    (20.0, 70.0, [0.1, 0.3, 1.1, 2.8, 5.0, 9.0, 22.9, 76.6]),
    (10.0, 70.0, [0.1, 0.4, 1.0, 1.9, 3.7, 9.7, 32.8, 117.0]),
    (30.0, 70.0, [0.1, 0.3, 1.0, 3.1, 7.4, 12.7, 23.1, 59.3]),
    (15.0, 20.0, [0.3, 0.6, 1.2, 2.7, 8.2, 28.2, 88.8, 202.0]),
])
def test_air_absorption_matches_iso_9613(temp, rh, table):
    f = 1000.0 * 10.0 ** (np.arange(-4, 4) * 0.3)             # 63.1 … 7943 Hz: exact midband frequencies (base 10)
    got = A.air_absorption_db_per_m(f, temp, rh) * 1000.0
    for g, t in zip(got, table):
        assert g == pytest.approx(t, abs=0.05 if t < 10 else 0.006 * t)


def test_speed_of_sound_and_a_weighting():
    assert A.speed_of_sound(20.0) == pytest.approx(343.2)
    assert A.speed_of_sound(0.0) == pytest.approx(331.3, abs=0.1)
    aw = A.a_weighting_db([1000.0, 100.0, 63.0, 4000.0])
    assert aw == pytest.approx([0.0, -19.1, -26.2, 1.0], abs=0.06)


@pytest.mark.parametrize('kw', [dict(), dict(speed_mps=30.0, offset_m=4.0, f0=2000.0),
                                dict(speed_mps=12.0, offset_m=25.0, f0=500.0, air_absorption=False)])
def test_passby_doppler_and_level_match_theory(kw):
    """The Doppler shift emerges from the retarded time alone: the measured frequency follows f0·c/(c − v_r) and the
    level L(1 m) − 20·lg r − α·r (closed form, independent of the implementation)."""
    r = A.passby_check(**kw)
    assert r['doppler_max_abs_err_hz'] < 0.1
    assert r['doppler_median_abs_err_hz'] < 0.01
    assert r['level_max_abs_err_db'] < 0.15
    lo, hi = r['doppler_span_hz']
    v, c = r['speed_mps'], r['c_mps']
    assert hi > r['f0_hz'] * c / (c - 0.8 * v) and lo < r['f0_hz'] * c / (c + 0.8 * v)   # the full shift is heard


def test_absorption_is_applied_at_high_frequency():
    r = A.passby_check(f0=8000.0, offset_m=40.0, speed_mps=15.0)
    assert r['alpha_db_per_m'] == pytest.approx(0.0776, abs=0.002)
    assert r['level_max_abs_err_db'] < 0.25


def _static_scene(src, mics=((0.0, 0.0, 1.5), (0.0, 0.2, 1.5)), duration=1.0, boxes=None, seed=0):
    ego = [(0.0, A.Track([0.0, 1.0], [[0, 0, 0], [0, 0, 0]]), A.Track([0.0, 1.0], [[0, 0, 0], [0, 0, 0]]))]
    return A.Scene(ego, np.array(mics, float), np.eye(3), [src], np.arange(0, duration + 1e-9, 0.05), duration,
                   boxes, {'seed': seed})


def _tone(xyz, f=1000.0, db=100.0, t=(-2.0, 3.0)):
    tt = np.array([-3.0, 4.0])
    track = A.Track(tt, np.array([list(xyz) + [0.0]] * 2))
    return A.Source(7, 'test.tone', 'car', 'tone', track, (0, 0, 0), [t], f=f, db_at_1m=db)


def _rms_db(x):
    return 20 * math.log10(float(np.sqrt(np.mean(np.asarray(x, float) ** 2))) / A.P_REF)


def test_spherical_spreading_and_ground_image():
    o = A.settings({'ground': False, 'occlusion': False, 'mic_self_noise_db': None, 'air_absorption': False})
    for d in (5.0, 20.0, 80.0):
        res = A.render(_static_scene(_tone((d, 0.0, 1.5))), o)
        assert _rms_db(res['audio'][0, 24000:]) == pytest.approx(100.0 - 20 * math.log10(d), abs=0.02)
    # one image source in the ground plane: the two paths' phasor sum, r2 from the mirrored source
    o = dict(o, ground=True, ground_reflection=0.8)
    hs, hm, d, f = 0.5, 1.5, 10.0, 700.0
    res = A.render(_static_scene(_tone((d, 0.0, hs), f=f), mics=((0.0, 0.0, hm), (0.0, 0.2, hm))), o)
    c = A.speed_of_sound(20.0)
    r1, r2 = math.hypot(d, hm - hs), math.hypot(d, hm + hs)
    amp = abs(1 / r1 + 0.8 * np.exp(-2j * np.pi * f * (r2 - r1) / c) / r2)
    assert _rms_db(res['audio'][0, 24000:]) == pytest.approx(100.0 + 20 * math.log10(amp), abs=0.03)


def test_retarded_time_satisfies_the_light_cone():
    tt = np.linspace(-2, 4, 61)
    xyz = np.column_stack([np.cos(tt) * 30, np.sin(tt) * 30, 0.5 + 0.1 * tt, tt * 10])     # a curving source
    src = A.Source(1, 't', 'car', 'tone', A.Track(tt, xyz), (1.0, 0.5, 0.0), [(0, 1)], f=1, db_at_1m=0)
    t = np.linspace(0.0, 2.0, 50)
    m = np.column_stack([np.linspace(0, 20, 50), np.zeros(50), np.full(50, 1.5)])
    c = A.speed_of_sound(20.0)
    tau, r, p, _ = A._retard(t, m, src, c)
    assert np.max(np.abs(np.linalg.norm(m - src.pos(tau), axis=1) - c * (t - tau))) < 1e-6
    assert r == pytest.approx(c * (t - tau))


def test_track_is_c1_and_extrapolates_linearly():
    t = np.arange(0.0, 1.0, 0.05)
    y = np.column_stack([np.sin(3 * t), t ** 2])
    tr = A.Track(t, y)
    assert tr(t) == pytest.approx(y, abs=1e-12)
    q = np.linspace(0.0, 0.95, 1000)
    p, dp = tr(q, deriv=True)
    assert np.max(np.abs(np.diff(dp[:, 0]))) < 0.02                # no jumps in velocity between samples
    e, de = tr(np.array([-1.0, 2.0]), deriv=True)
    assert e[0] == pytest.approx(y[0] - de[0])


def test_occlusion_by_a_vehicle_box():
    src = _tone((20.0, 0.0, 0.05), f=1000.0)
    box = {k: (np.array([99]), np.array([[10.0, 0.0, 1.5]]), np.array([0.0]), np.array([[6.0, 1.3, 1.5]]))
           for k in range(21)}
    o = A.settings({'ground': False, 'mic_self_noise_db': None, 'air_absorption': False})
    free = A.render(_static_scene(src, mics=((0.0, 0.0, 1.55), (0.0, 0.2, 1.55))), o)
    hid = A.render(_static_scene(src, mics=((0.0, 0.0, 1.55), (0.0, 0.2, 1.55)), boxes=box), o)
    drop = _rms_db(free['audio'][0, 24000:]) - _rms_db(hid['audio'][0, 24000:])
    # over the 3 m bus: path difference z, ISO 9613-2 D_z = 10 lg(3 + 20 z / λ), capped at 20 dB
    z = math.hypot(10.0, 3.0 - 0.05) + math.hypot(10.0, 3.0 - 1.55) - math.hypot(20.0, 1.5)
    expect = min(10 * math.log10(3 + 20 * z * 1000.0 / A.speed_of_sound(20.0)), 20.0)
    assert drop == pytest.approx(expect, abs=0.6)
    assert A.barrier_db(np.array([-0.1]), [1000.0], 343.2)[0, 0] == 0.0


# ------------------------------------------------------------------------------------------- sources
def test_cnossos_emission_levels():
    roll, prop = A.cnossos_band_power('1', '1', 70 / 3.6, 0.0)
    lw = 10 * np.log10((roll + prop) / A.P_REF ** 2) + A.LP_FROM_LW_DB
    exp = 10 * np.log10(10 ** (np.array(A.CNOSSOS['1']['ar']) / 10) + 10 ** (np.array(A.CNOSSOS['1']['ap']) / 10))
    assert lw[0] == pytest.approx(exp, abs=1e-9)                      # at v_ref both terms are their A coefficients
    _, p2 = A.cnossos_band_power('1', '1', 70 / 3.6, 1.0)
    assert 10 * np.log10(p2 / prop)[0] == pytest.approx(np.full(8, 3.2))          # IMAGINE acceleration, C_P a
    _, p3 = A.cnossos_band_power('1', '1', 70 / 3.6, 5.0)
    assert 10 * np.log10(p3 / prop)[0] == pytest.approx(np.full(8, 6.4))          # a ≤ a_max = 2 m/s² (cat. 1)
    r4, _ = A.cnossos_band_power('4b', None, 10.0, 0.0)
    assert not r4.any()                                                          # two-wheelers: no rolling noise
    slow = A.cnossos_band_power('1', '1', 2.0, 0.0)[1]
    assert slow == pytest.approx(A.cnossos_band_power('1', '1', 20 / 3.6, 0.0)[1])   # propulsion held at 20 km/h


def test_engine_emission_keeps_the_band_powers():
    """The emitted noise + engine orders carry CNOSSOS's octave-band powers (orders moved, not added)."""
    fs = 48000
    tt = np.array([-1.0, 5.0])
    v = 50 / 3.6
    track = A.Track(tt, np.array([[v * t, 0.0, 0.0, 0.0] for t in tt]))
    src = A.Source(3, 'vehicle.audi.a2', 'car', 'engine', track, (0, 0, 0.05), [(0.0, 4.0)])
    stft = A.Stft(1024, fs)
    X, tf, n, env = A.emit(src, 0.0, 4.0, (0.0, 0.0), fs, stft, A.band_shape(stft), np.random.default_rng(1))
    x = stft.inverse(X, n)[fs // 2: -fs // 2]
    P = np.abs(np.fft.rfft(x)) ** 2 * 2 / len(x) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    roll, prop = A.cnossos_band_power('1', '1', v, 0.0)
    want = (roll + prop)[0]
    got = np.array([P[(f >= fc / 2 ** 0.5) & (f < fc * 2 ** 0.5)].sum() for fc in A.OCTAVES])
    lo = A.OCTAVES <= 1000
    assert 10 * math.log10(got[lo].sum() / want[lo].sum()) == pytest.approx(0.0, abs=0.5)
    assert 10 * np.log10(got[~lo] / want[~lo]) == pytest.approx(np.zeros((~lo).sum()), abs=0.5)
    assert A.rpm_of(A.ENGINES['petrol4'], 0.0) == 800
    assert 1300 <= A.rpm_of(A.ENGINES['petrol4'], 50 / 3.6) <= 2800


def test_siren_and_horn_levels():
    fs = 48000
    stft = A.Stft(1024, fs)
    track = A.Track([-1.0, 5.0], [[0, 0, 0, 0], [0, 0, 0, 0]])
    sir = A.Source(5, 'vehicle.ford.ambulance', 'van', 'siren', track, (0, 0, 2), [(0.0, 5.0)], pattern='wail')
    X, tf, n, _ = A.emit(sir, 0.0, 5.0, (0, 0), fs, stft, A.band_shape(stft), np.random.default_rng(0))
    aw = 10 ** (A.a_weighting_db(np.maximum(stft.f, 1)) / 10)
    dba = 10 * math.log10(np.mean(np.sum(np.abs(X) ** 2 * aw, axis=1)) / np.sum(stft.win ** 2) * 2 / stft.n /
                          A.P_REF ** 2)                              # Parseval: mean square from the frames' spectra
    assert dba == pytest.approx(123.0 + 20 * math.log10(3.0), abs=0.5)        # 123 dB(A) at 3 m, KKK-A-1822
    hp = A.HORNS['car']
    horn = A.Source(5, 'vehicle.audi.a2', 'car', 'horn', track, (2, 0, 0.6), [(0.0, 2.0)], horn='car',
                    events=[(0.0, 2.0)])
    X, tf, n, _ = A.emit(horn, 0.0, 2.0, (0, 0), fs, stft, A.band_shape(stft), np.random.default_rng(0))
    dba = 10 * math.log10(np.mean(np.sum(np.abs(X[5:-5]) ** 2 * aw, axis=1)) / np.sum(stft.win ** 2) * 2 / stft.n /
                          A.P_REF ** 2)
    assert dba == pytest.approx(hp['dba_at_7m'] + 20 * math.log10(7.0), abs=0.5)
    assert A.classify({'type_id': 'vehicle.dodge.charger_police_2020', 'category': 'vehicle.emergency.police'}) == \
        ('car', None, 'yelp')
    assert A.classify({'type_id': 'vehicle.micro.microlino', 'subclass': 'autorickshaw (stand-in)'})[0] == \
        'autorickshaw'
    assert A.classify({'type_id': 'walker.pedestrian.0001'})[0] is None
    assert A.classify({'type_id': 'vehicle.yamaha.yzf', 'category': 'vehicle.motorcycle'})[0] == 'two_wheeler'


# ------------------------------------------------------------------------------------------- the array, DOA
def test_srp_phat_finds_a_broadband_source_and_the_azimuth_convention():
    """A broadband source 40° to the RIGHT (+y) of the array: the ground truth says +40 and SRP-PHAT agrees."""
    pose, mics = A.array_geometry(None)
    a = math.radians(40.0)
    d = 30.0
    tt = np.array([-2.0, 4.0])
    xyz = [d * math.cos(a), d * math.sin(a), 0.3]
    track = A.Track(tt, np.array([xyz + [0.0]] * 2))
    src = A.Source(9, 'vehicle.yamaha.yzf', 'two_wheeler', 'engine', track, (0, 0, 0), [(-1.0, 3.0)])
    mv = np.array([pose['x'], pose['y'], pose['z']]) + mics
    o = A.settings({'ground': False, 'mic_self_noise_db': 29.0})
    res = A.render(_static_scene(src, mics=mv, duration=2.0), o)
    row = res['gt'][20]['sources'][0]
    el = math.degrees(math.atan2(0.3 - pose['z'], d))
    assert row['az_deg'] == pytest.approx(40.0, abs=0.01) and row['el_deg'] == pytest.approx(el, abs=0.01)
    assert row['distance_m'] == pytest.approx(math.hypot(d, 0.3 - pose['z']), abs=0.01)
    assert row['doppler'] == pytest.approx(1.0, abs=1e-9)
    ev = A.doa_evaluate(res['audio'], res['fs'], mics, res['c'], res['gt'])
    assert ev['dominant']['frames'] >= 30
    assert ev['dominant']['median_deg'] <= 1.0 and ev['dominant']['p90_deg'] <= 2.0


def test_render_is_deterministic_with_the_seed(tmp_path):
    run = _fixture_run(tmp_path / 'run')
    o = A.settings({'doa': False, 'horn_events': [{'actor': 20, 'start_s': 1.0, 'dur_s': 0.3}]})
    s1 = A.scene_from_run(str(run), o)
    a1 = A.render(s1, o)['audio']
    a2 = A.render(A.scene_from_run(str(run), o), o)['audio']
    assert np.array_equal(a1, a2)
    a3 = A.render(A.scene_from_run(str(run), dict(o, seed=99)), dict(o, seed=99))['audio']
    assert not np.array_equal(a1, a3)


def test_wav_round_trip(tmp_path):
    x = (np.random.default_rng(0).standard_normal((6, 4801)) * 3.0).astype(np.float32)
    A.write_wav(str(tmp_path / 'a.wav'), x, 48000, 'float32')
    y, fs = A.read_wav(str(tmp_path / 'a.wav'))
    assert fs == 48000 and np.array_equal(y.astype(np.float32), x)
    A.write_wav(str(tmp_path / 'b.wav'), x, 48000, 'pcm24', full_scale=200.0)
    y, _ = A.read_wav(str(tmp_path / 'b.wav'), full_scale=200.0)
    assert np.max(np.abs(y - x)) < 200.0 / 8388607 * 0.51 + 1e-9
    import wave
    with pytest.raises(wave.Error):          # the stdlib reads PCM only; the header must still parse as RIFF/WAVE
        wave.open(str(tmp_path / 'a.wav'))
    assert open(tmp_path / 'b.wav', 'rb').read(4) == b'RIFF'


# ------------------------------------------------------------------------------------------- the rig
def test_rig_audio_array_defaults_and_checks():
    s = R.new_sensor('audio_array', 'mics', 'roof mics', {'z': 1.55})
    assert s['mics'] == 8 and s['radius'] == 0.2
    hand = {'id': 'm', 'name': 'm', 'type': 'audio_array', 'pose': dict(R.POSE0, z=1.5)}
    full = R.with_defaults(hand)
    assert full['mics'] == 8 and full['sample_rate'] == 48000
    pose, mics = A.array_geometry(full)
    assert mics.shape == (8, 3) and mics[0] == pytest.approx([0.2, 0.0, 0.0])
    assert mics[2] == pytest.approx([0.0, 0.2, 0.0], abs=1e-12)       # mic 2 to the right (+y)
    _, m3 = A.array_geometry(dict(full, mic_positions=[[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]))
    assert m3.shape == (3, 3)
    rig = {'id': 'r', 'name': 'r', 'vehicle': 'vehicle.lincoln.mkz_2020', 'sensors': [dict(hand, mics=1)]}
    assert any('2–64 microphones' in p for p in R.validate_rig(rig))
    assert R.sensor_cost(full)['mpx'] == 0.0
    from falcon_autonomy.core.project import Project, validate
    p = Project.new('audio')
    p.doc['rigs'][0]['sensors'].append(hand)
    assert not validate(p.doc)


# ------------------------------------------------------------------------------------------- from a run
def _fixture_run(d, seconds=4.0, fps=20.0):
    """A tiny falcon-raw run: the ego driving +x at 8 m/s, an ambulance coming the other way in the next lane, a
    two-wheeler closing on a slow car ahead of it, a walker; an actor that jumps (a respawn) halfway."""
    os.makedirs(d, exist_ok=True)
    n = int(seconds * fps)
    frames, steps = [], []
    for k in range(n):
        t = round(k / fps, 6)
        ego = {'x': 8.0 * t, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        fr = {'step': k, 'frame': 100 + k, 'sim_t': t, 'ego': ego, 'speed_mps': 8.0, 'files': {}, 'recorded': True}
        acts = [dict(id=10, type_id='vehicle.lincoln.mkz_2020', category='vehicle.car', role='hero',
                     extent=[2.45, 1.05, 0.75], bbox_center=[0, 0, 0.75], vx=8.0, vy=0.0, vz=0.0, **ego),
                dict(id=20, type_id='vehicle.ford.ambulance', category='vehicle.emergency.ambulance', role='',
                     extent=[3.2, 1.2, 1.4], bbox_center=[0, 0, 1.4], x=60.0 - 12.0 * t, y=-3.5, z=0.0, roll=0.0,
                     pitch=0.0, yaw=180.0, vx=-12.0, vy=0.0, vz=0.0),
                dict(id=30, type_id='vehicle.yamaha.yzf', category='vehicle.motorcycle', role='',
                     extent=[1.1, 0.4, 0.8], bbox_center=[0, 0, 0.8], x=20.0 + 10.0 * t, y=3.5, z=0.0, roll=0.0,
                     pitch=0.0, yaw=0.0, vx=10.0, vy=0.0, vz=0.0),
                dict(id=40, type_id='vehicle.audi.a2', category='vehicle.car', role='',
                     extent=[1.85, 0.9, 0.77], bbox_center=[0, 0, 0.77], x=30.0 + 3.0 * t, y=3.5, z=0.0, roll=0.0,
                     pitch=0.0, yaw=0.0, vx=3.0, vy=0.0, vz=0.0),
                dict(id=50, type_id='walker.pedestrian.0001', category='human.pedestrian.adult', role='',
                     extent=[0.2, 0.2, 0.9], bbox_center=[0, 0, 0], x=40.0, y=6.0, z=0.9, roll=0.0, pitch=0.0,
                     yaw=90.0, vx=0.0, vy=1.0, vz=0.0),
                dict(id=60, type_id='vehicle.nissan.micra', category='vehicle.car', role='',
                     extent=[1.8, 0.9, 0.77], bbox_center=[0, 0, 0.77], x=(-30.0 if t < seconds / 2 else 90.0) +
                     5.0 * t, y=-7.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, vx=5.0, vy=0.0, vz=0.0)]
        if k == n // 2:
            fr['traffic_respawns'] = [{'sim_t': t, 'kind': 'dead_end', 'actor': 60, 'from': {}, 'to': {}}]
        frames.append(fr)
        steps.append({'step': k, 'frame': 100 + k, 'sim_t': t, 'actors': acts})
    with open(os.path.join(d, 'frames.jsonl'), 'w') as fh:
        fh.writelines(json.dumps(f) + '\n' for f in frames)
    with open(os.path.join(d, 'actors.jsonl'), 'w') as fh:
        fh.writelines(json.dumps(s) + '\n' for s in steps)
    rig = {'id': 'r', 'name': 'r', 'vehicle': 'vehicle.lincoln.mkz_2020',
           'sensors': [{'id': 'mics', 'name': 'mics', 'type': 'audio_array',
                        'pose': dict(R.POSE0, z=1.55), 'mics': 6, 'radius': 0.15}]}
    with open(os.path.join(d, 'drive.json'), 'w') as fh:
        json.dump({'seed': 5, 'fps': fps, 'rig': rig, 'traffic': {'id': 'tp-india-mixed', 'handedness': 'left'}}, fh)
    with open(os.path.join(d, 'run.json'), 'w') as fh:
        json.dump({'schema': 'falcon.run/1', 'status': 'ok', 'provenance': {'seed': 5}}, fh)
    return d


def test_derive_audio_from_a_run(tmp_path):
    run = _fixture_run(tmp_path / 'run')
    man = A.derive_audio(str(run), None, {'horn_events': [{'actor': 40, 'start_s': 2.0, 'dur_s': 0.4}]})
    out = run / 'audio'
    data, fs = A.read_wav(str(out / 'audio.wav'))
    assert data.shape == (6, int(round(3.95 * 48000)) + 1) and fs == 48000
    assert man['synthetic'] and man['wav']['units'] == 'Pa'
    assert man['sources']['by_kind']['siren'] == 1 and man['sources']['vehicles_by_class'] == \
        {'van': 1, 'two_wheeler': 1, 'car': 3}           # ambulance; yzf; a2 and micra (two pieces: it jumped)
    assert man['events']['horn_profile'] == 'indian'
    gt = [json.loads(x) for x in open(out / 'ground_truth.jsonl')]
    assert len(gt) == 80 and gt[0]['step'] == 0 and gt[-1]['step'] == 79
    row = next(s for s in gt[10]['sources'] if s['kind'] == 'siren')
    # the ambulance ahead-left (y −3.5: LEFT is negative azimuth), approaching: Doppler above 1
    assert -10.0 < row['az_deg'] < 0.0 and row['doppler'] > 1.0 and row['class'] == 'van'
    evs = [json.loads(x) for x in open(out / 'events.jsonl')]
    assert any(e['kind'] == 'horn' and e['actor'] == 40 and e['rule'] == 'explicit' for e in evs)
    assert any(e['kind'] == 'siren' and e['actor'] == 20 for e in evs)
    run_doc = json.load(open(run / 'run.json'))
    assert run_doc['derived']['audio']['path'] == 'audio' and run_doc['derived']['audio']['synthetic']
    assert run_doc['status'] == 'ok'                                  # the rest of run.json kept
    assert json.load(open(out / 'audio.json'))['parameters']['seed'] == 5
    assert os.path.isfile(out / 'doa.json')


def test_horn_rules_are_seeded_and_profiled(tmp_path):
    from falcon_autonomy.core.worldmodel import _jsonl
    run = _fixture_run(tmp_path / 'run', seconds=30.0, fps=10.0)
    steps, frames = _jsonl(str(run / 'actors.jsonl')), _jsonl(str(run / 'frames.jsonl'))
    a = A.horn_plan(steps, frames, 10, 5, 'indian')
    assert a == A.horn_plan(steps, frames, 10, 5, 'indian')
    assert A.horn_plan(steps, frames, 10, 5, 'none') == {}
    # the two-wheeler closes on the slow car ahead in its lane: it honks; the ego (hero) never does
    assert 30 in a and all(why in A.HORN_RULES for ev in a.values() for _, _, why in ev)
    assert 10 not in a
    starts = sorted(s for s, _, _ in a[30])
    assert all(b - a_ >= A.HORN_TIMING['refractory_s'] - 1e-9 or b - a_ < 0.8 for a_, b in zip(starts, starts[1:]))


def test_a_jump_splits_the_track(tmp_path):
    run = _fixture_run(tmp_path / 'run')
    scene = A.scene_from_run(str(run), A.settings({'horns': 'none'}))
    micra = [s for s in scene.sources if s.actor == 60]
    assert len(micra) == 2                      # before and after the respawn: no 120 m slide in one step
    assert micra[0].params['seg'][1] < micra[1].params['seg'][0]


def test_cli_derive_audio(tmp_path, capsys):
    from falcon_autonomy.cli import main
    run = _fixture_run(tmp_path / 'run')
    assert main(['derive', 'audio', str(run), '--out', str(tmp_path / 'a'), '--option', 'doa=false',
                 '--option', 'format="pcm24"']) == 0
    assert 'synthetic audio: 6 channels' in capsys.readouterr().out
    assert os.path.isfile(tmp_path / 'a' / 'audio.wav') and not os.path.isfile(tmp_path / 'a' / 'doa.json')
    assert 'derived' not in json.load(open(run / 'run.json'))         # out of the run: run.json untouched
    assert main(['derive', 'audio', str(run), '--option', 'nonsense=1']) == 2
    assert main(['derive', 'audio-check', 'passby']) == 0
    assert '"doppler_max_abs_err_hz"' in capsys.readouterr().out


def test_gui_rig_editor_and_derive_from_the_browser(window, qapp, tmp_path):
    from test_gui import pump
    page = window.go('rigs')
    page.add_sensor('audio_array')
    rig = window.project.doc['rigs'][0]
    assert rig['sensors'][-1]['type'] == 'audio_array' and rig['sensors'][-1]['mics'] == 8
    page.sensors.setCurrentRow(1)
    page.set_field(1, 'mics', 4)
    assert rig['sensors'][-1]['mics'] == 4 and 'synthetic audio' in page.cost.vals['Cost'].text()
    page.view.resize(400, 300)
    assert not page.view.grab().isNull()                          # the array is drawn
    run = _fixture_run(tmp_path / 'run')
    br = window.go('data').browse
    br.run_dir = str(run)
    br.runs.clearSelection()
    br.derive_audio()
    job = window.queue.jobs[-1]
    assert job.spec.kind == 'derive' and job.spec.argv[-3:] == ['derive', 'audio', str(run)]
    assert pump(qapp, lambda: job.state in ('done', 'failed', 'cancelled'), 120), job.log
    assert job.state == 'done', job.log
    assert os.path.isfile(run / 'audio' / 'audio.wav')


def test_a_rig_with_an_audio_array_drives_on_the_fake_and_derives(project, tmp_path):
    """The audio_array is not a CARLA sensor: the drive worker spawns nothing for it, and the run derives audio."""
    import subprocess
    import sys
    from conftest import worker_env
    from falcon_autonomy.core.drive import drive_config
    rig = project.doc['rigs'][0]
    rig['sensors'][0].update(width=160, height=90)
    rig['sensors'].append({'id': 'mics', 'name': 'roof mics', 'type': 'audio_array', 'mics': 4, 'radius': 0.25,
                           'pose': dict(R.POSE0, z=1.55)})
    cfg = drive_config(project, 'world-town10', 'rig-default', str(tmp_path / 'run'), 2310, 8310, seconds=1.0, seed=3)
    os.makedirs(tmp_path / 'run')
    with open(tmp_path / 'run' / 'drive.json', 'w') as fh:          # where the service puts it
        json.dump(cfg, fh)
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', str(tmp_path / 'run' / 'drive.json')],
                       capture_output=True, text=True, env=worker_env(), timeout=120)
    assert r.returncode == 0, r.stderr
    man = A.derive_audio(str(tmp_path / 'run'))
    assert man['wav']['channels'] == 4 and man['array']['mics_vehicle_frame'][0] == pytest.approx([0.25, 0.0, 1.55])
    assert json.load(open(tmp_path / 'run' / 'run.json'))['derived']['audio']['channels'] == 4
