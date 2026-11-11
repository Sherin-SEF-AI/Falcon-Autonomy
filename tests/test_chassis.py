"""The chassis model behind the CAN bridge (core.chassis): the steering's transport delay and first-order lag, its rate
limit and torque mode; the brake's hysteresis loop (its width is the dead band) and its different apply and release
rates; the friction per surface, Horne's hydroplaning speed and the friction ramp above the onset; the wheels' slip
when braking on a waterlogged road against a dry one, and the tone-wheel ticks that show it."""
import math

import numpy as np
import pytest

from falcon_autonomy.core import chassis as CH


def _series(c, seconds, dt=0.001, fn=None):
    t, y = [], []
    for k in range(int(round(seconds / dt))):
        if fn is not None:
            fn(c, k * dt)
        c.step(dt)
        t.append(c.t)
        y.append(c.sw)
    return np.array(t), np.array(y)


def test_steering_step_response_shows_the_delay_then_the_lag():
    c = CH.Chassis({'eps_delay_s': 0.03, 'eps_tau_s': 0.08})
    c.command('angle', 30.0)
    t, y = _series(c, 0.5)
    first = t[np.argmax(y > 0.01 * 30.0)]
    t63 = t[np.argmax(y >= (1 - math.exp(-1)) * 30.0)]
    assert first == pytest.approx(0.03, abs=0.0015)                    # nothing moves before the delay
    assert t63 == pytest.approx(0.03 + 0.08, abs=0.002)                # then a first-order lag of τ
    assert y[-1] == pytest.approx(30.0, abs=0.1)
    assert c.actuation()['road_wheel_deg'] == pytest.approx(c.sw / 15.0)


def test_steering_rate_limit_and_torque_mode():
    c = CH.Chassis({'eps_rate_deg_s': 360.0, 'eps_delay_s': 0.0})
    c.command('angle', 300.0)
    t, y = _series(c, 0.2)
    rate = np.diff(y) / np.diff(t)
    assert rate.max() <= 360.0 + 1e-6 and y[-1] == pytest.approx(72.0, abs=1.0)      # 0.2 s at 360 °/s
    c = CH.Chassis({'eps_torque_gain_deg_per_nm': 60.0})
    c.command('torque', steer_torque_nm=1.5)
    _series(c, 1.0)
    assert c.sw == pytest.approx(90.0, abs=0.5) and c.feedback()['EPS_TORQUE'] == pytest.approx(1.5, abs=0.01)
    c.command('off')
    _series(c, 0.3)
    assert c.sw == pytest.approx(90.0, abs=0.5)                        # off: it stays where it is


def _brake_loop(c, rate_bar_s=10.0, top=100.0):
    """A slow triangle sweep of the request: the pressure where the request crosses each level, up and down."""
    up, dn = {}, {}
    dt = 0.001
    n = int(round(2 * top / rate_bar_s / dt))
    prev = 0.0
    for k in range(n + 1):
        t = k * dt
        r = rate_bar_s * t if t <= top / rate_bar_s else 2 * top - rate_bar_s * t
        c.command(brake_bar=r)
        c.step(dt)
        for lvl in (20.0, 50.0, 80.0):
            if (prev < lvl <= r) and lvl not in up:
                up[lvl] = c.p_brake
            if (prev > lvl >= r) and lvl not in dn:
                dn[lvl] = c.p_brake
        prev = r
    return up, dn


def test_brake_hysteresis_loop_is_the_dead_band_wide():
    c = CH.Chassis({'brake_dead_band_bar': 4.0, 'brake_delay_s': 0.0})
    up, dn = _brake_loop(c)
    for lvl in (20.0, 50.0, 80.0):
        assert dn[lvl] - up[lvl] == pytest.approx(4.0, abs=0.05), (lvl, up, dn)
        assert up[lvl] == pytest.approx(lvl - 2.0, abs=0.05)
    assert c.p_brake == pytest.approx(0.0, abs=1e-9)                   # a request below half the band releases fully
    # a transport delay widens the loop by delay × sweep rate on each side
    c = CH.Chassis({'brake_dead_band_bar': 4.0, 'brake_delay_s': 0.02})
    up, dn = _brake_loop(c)
    assert dn[50.0] - up[50.0] == pytest.approx(4.0 + 2 * 0.02 * 10.0, abs=0.05)


def test_brake_applies_faster_than_it_releases():
    c = CH.Chassis({'brake_apply_bar_s': 300.0, 'brake_release_bar_s': 150.0, 'brake_delay_s': 0.0,
                    'brake_dead_band_bar': 0.0})
    c.command(brake_bar=90.0)
    t_apply = 0.0
    while c.p_brake < 90.0 - 1e-6:
        c.step(0.001)
        t_apply += 0.001
    c.command(brake_bar=0.0)
    t_rel = 0.0
    while c.p_brake > 1e-6:
        c.step(0.001)
        t_rel += 0.001
    assert t_apply == pytest.approx(0.3, abs=0.002) and t_rel == pytest.approx(0.6, abs=0.002)


def test_surfaces_and_hydroplaning():
    assert CH.hydroplaning_speed_kmh(220.0) == pytest.approx(6.35 * math.sqrt(220.0)) == pytest.approx(94.18, abs=0.01)
    assert CH.SURFACES['waterlogged']['range'] == (0.3, 0.5)
    for name, s in CH.SURFACES.items():
        lo, hi = s['range']
        assert lo <= s['mu'] <= hi, name
    w = CH.Chassis({'surface': 'waterlogged', 'mu': 0.45})
    vp = w.v_p
    assert w.mu_eff(0.5 * vp) == 0.45 and w.mu_eff(0.7 * vp) == pytest.approx(0.45)
    assert w.mu_eff(0.85 * vp) == pytest.approx(0.45 + 0.5 * (0.05 - 0.45))
    assert w.mu_eff(vp) == pytest.approx(0.05) and w.mu_eff(2 * vp) == pytest.approx(0.05)
    wet = CH.Chassis({'surface': 'wet_asphalt'})
    assert wet.mu_eff(2 * vp) == 0.6                                   # a thin film: no dynamic hydroplaning
    assert CH.validate({'surface': 'lava'}) and CH.validate({'surface': 'waterlogged', 'mu': 0.8})
    assert CH.validate({'nonsense': 1}) and not CH.validate({'surface': 'ice', 'mu': 0.1})
    with pytest.raises(ValueError):
        CH.Chassis({'surface': 'lava'})
    d = w.describe()
    assert d['hydroplaning_speed_kmh'] == pytest.approx(94.18, abs=0.01) and d['mu'] == 0.45


def _brake_from(surface, kmh, bar, seconds=1.0):
    c = CH.Chassis({'surface': surface})
    c.v = kmh / 3.6
    c.omega[:] = c.v / c.r
    c.command(brake_bar=bar)
    slips, dist = [], 0.0
    t0 = c.ticks().copy()
    for _ in range(int(seconds / 0.01)):
        v0 = c.v
        c.step(0.01)
        dist += 0.5 * (v0 + c.v) * 0.01
        slips.append(c.slip())
    ticks = (c.ticks() - t0).mean()
    return c, np.array(slips), dist, ticks


def test_braking_slips_on_water_and_grips_on_a_dry_road():
    dry, s_dry, d_dry, k_dry = _brake_from('dry_asphalt', 100.0, 40.0)
    wet, s_wet, d_wet, k_wet = _brake_from('waterlogged', 100.0, 40.0)
    assert np.abs(s_dry).max() < 0.1                                   # dry: a few percent of slip
    assert s_wet.min() < -0.9                                          # past the hydroplaning onset: the wheels lock
    assert dry.v < 100 / 3.6 - 3.0 and wet.v > 100 / 3.6 - 1.5        # and the car hardly slows on water
    per_m = 48 / (2 * math.pi * 0.31)
    assert k_dry == pytest.approx(d_dry * per_m, rel=0.05)             # ticks follow the distance on a dry road
    assert k_wet < 0.5 * d_wet * per_m                                 # a locked wheel stops ticking while it slides
    fb = wet.feedback()
    assert fb['HYDROPLANING'] == 1 and fb['MU_EFF'] < 0.3 and fb['SPEED_FL'] < fb['VEHICLE_SPEED'] / 2


def test_drive_ticks_and_yaw_without_slip():
    c = CH.Chassis({'surface': 'dry_asphalt'})
    c.command('angle', 0.0, throttle_pct=20.0)
    dist = 0.0
    for _ in range(300):
        v0 = c.v
        c.step(0.01)
        dist += 0.5 * (v0 + c.v) * 0.01
    assert 3.0 < c.v < 15.0 and abs(c.slip()[:2]).max() < 0.05      # driven front wheels: a little slip
    per_m = 48 / (2 * math.pi * 0.31)
    assert c.ticks()[2:].mean() == pytest.approx(dist * per_m, rel=0.02)     # undriven rear wheels roll freely
    c.command('angle', 90.0, throttle_pct=0.0)
    c.step(0.5)
    expected = math.degrees(c.v * math.tan(math.radians(90.0 / 15.0)) / 2.7)
    assert c.yaw_rate == pytest.approx(expected, rel=0.02) and c.yaw_rate > 0      # left is positive


def test_carla_plant_takes_the_motion_and_the_safe_hold():
    c = CH.Chassis({'surface': 'dry_asphalt'}, plant='carla')
    c.set_motion(10.0, 5.0, 0.0, 0.9)
    c.omega[:] = 10.0 / c.r
    c.step(0.2)
    assert c.v == 10.0 and c.yaw_rate == 5.0                           # the body is CARLA's
    assert abs(c.slip()).max() < 0.01
    c.command('angle', 45.0, brake_bar=5.0)
    c.step(0.1)
    c.safe_hold()
    held = c.sw
    c.step(0.5)
    a = c.actuation()
    assert a['safe'] and a['throttle'] == 0.0 and a['brake_bar'] == pytest.approx(40.0 - 1.0, abs=0.01)
    # the steering returns to where it was when the hold came (what was already in the delay line still arrived)
    c.step(0.5)
    assert c.sw == pytest.approx(held, abs=1e-3)
    with pytest.raises(ValueError):
        CH.Chassis(plant='rocket')
