"""The chassis behind the CAN bridge (docs/hardware.md#the-chassis-model): the actuators a device commands — an
electric power steering (EPS) and a hydraulic brake — and the wheels whose signals it reads back — speeds and tone-wheel
ticks that show slip, on a road surface with its friction and, when waterlogged, hydroplaning. numpy only (it runs in
the session worker, in CARLA's Python, and in `falcon hil can bench`).

    ch = Chassis(params, plant='model' | 'carla')
    ch.command(steer_mode, steer_req_deg, steer_torque_nm, brake_bar, throttle_pct)   # the newest command
    ch.safe_hold()                                         # the watchdog's command (hil.canbridge)
    ch.set_motion(v_mps, yaw_rate_deg_s, ax, ay)           # plant 'carla': the vehicle's motion is CARLA's
    ch.step(dt)                                            # sub-steps of H_S seconds
    ch.feedback()   ch.actuation()

The models, every number a parameter (PARAMS; the defaults are typical of a mid-size car, chosen, not measured):

* Steering: the steering wheel angle θ follows a target with a first-order lag τ (eps_tau_s), rate-limited
  (eps_rate_deg_s), after a pure transport delay (eps_delay_s: a ring buffer of the command, one sample per sub-step):
  θ += (1 − e^(−h/τ))·(u(t − delay) − θ). Angle mode: the target is the requested angle; torque mode: the angle at
  which the requested EPS torque balances a linear self-aligning spring, u = torque × eps_torque_gain_deg_per_nm.
  The road wheel angle is θ / steering_ratio.
* Brake: the line pressure p follows the request r through a dead band of width w (brake_dead_band_bar) — a backlash
  operator: p's target moves only when r leaves [target − w/2, target + w/2], a request below w/2 releases fully —
  then rate limits that differ on apply and release (brake_apply_bar_s > brake_release_bar_s), after a transport delay
  (brake_delay_s). A slow sweep up and down draws a loop w wide (the hysteresis).
* Wheels: per wheel J·dω/dt = T_drive − T_brake − F_x·r, the brake's friction torque never turning the wheel
  backwards (no reverse) and a spinning one capped at a slip of SPIN_CAP (1: twice the road speed, a traction
  control's limit); T_drive is the throttle's share of max_drive_torque_nm on the bench (plant 'model') and, with
  CARLA as the plant, the torque that produced CARLA's measured acceleration (m·a·r, while the throttle is on) — the
  wheels spin only where the surface cannot carry it; the slip ratio κ with a relaxation length σ (SAE J670's
  transient slip):
  σ·dκ/dt + max(|v|, v_low)·κ = ω·r − v (integrated implicitly: stable at standstill); F_x = μ_eff·F_z·MF(κ) with
  Pacejka's magic formula MF(κ) = sin(C·atan(Bκ − E(Bκ − atan Bκ))) (B 10, C 1.9, E 0.97: his textbook's
  longitudinal example), F_z with longitudinal load transfer. Ticks: the wheel's angle × tone_wheel_teeth / 2π,
  counted since the start — a locked wheel stops ticking while the car slides.
* Surface: μ per surface (SURFACES: typical peak tyre–road friction coefficients; waterlogged 0.3–0.5). On a
  waterlogged road above the onset speed the tyre starts to hydroplane: Horne's equation (NASA TN D-2056, 1963) for
  the dynamic hydroplaning speed v_p ≈ 6.35·√p km/h with p the tyre's inflation pressure in kPa (9·√psi knots);
  μ_eff falls linearly from μ at hydroplaning_onset·v_p to mu_hydroplaning at v_p and stays there above it. Horne's
  speed is the measured anchor; the ramp below it is a heuristic (partial hydroplaning has no closed form).
* Body (plant 'model' only): m·dv/dt = ΣF_x − ½ρC_dA·v² − C_rr·m·g; the yaw rate of a kinematic bicycle,
  v·tan(δ)/L, capped where the lateral acceleration would pass μ_eff·g (understeer). With plant 'carla' the vehicle's
  motion is CARLA's (its own tyre model); the wheels here are driven by that motion and the surface's μ — their
  slip and CARLA's deceleration agree only as far as CARLA's tyre friction matches (hil.canbridge scales it).
"""
from __future__ import annotations

import math

import numpy as np

G = 9.81
H_S = 0.0005                     # internal sub-step, s
SPIN_CAP = 1.0                   # the most a driven wheel spins: ω·r ≤ v + SPIN_CAP · max(v, 2 m/s)
WHEELS = ('FL', 'FR', 'RL', 'RR')
SURFACES = {
    'dry_asphalt': {'mu': 0.9, 'range': (0.8, 1.0), 'water': False, 'about': 'dry asphalt or concrete'},
    'wet_asphalt': {'mu': 0.6, 'range': (0.5, 0.7), 'water': False,
                    'about': 'wet asphalt, a thin film (no dynamic hydroplaning)'},
    'waterlogged': {'mu': 0.4, 'range': (0.3, 0.5), 'water': True,
                    'about': 'standing water deeper than the tread (monsoon waterlogging): hydroplaning past the '
                             'onset speed'},
    'gravel': {'mu': 0.55, 'range': (0.4, 0.7), 'water': False, 'about': 'loose gravel'},
    'snow': {'mu': 0.25, 'range': (0.2, 0.3), 'water': False, 'about': 'packed snow'},
    'ice': {'mu': 0.1, 'range': (0.05, 0.15), 'water': False, 'about': 'ice'},
}
PARAMS = {
    # steering (EPS)
    'eps_tau_s': 0.08, 'eps_delay_s': 0.03, 'eps_rate_deg_s': 720.0, 'eps_torque_gain_deg_per_nm': 60.0,
    'steer_max_deg': 540.0, 'steering_ratio': 15.0,
    # brake
    'brake_dead_band_bar': 2.0, 'brake_apply_bar_s': 300.0, 'brake_release_bar_s': 150.0, 'brake_delay_s': 0.02,
    'brake_max_bar': 150.0, 'brake_full_bar': 100.0, 'brake_nm_per_bar_front': 14.0, 'brake_nm_per_bar_rear': 8.0,
    'safe_brake_bar': 40.0,
    # vehicle
    'mass_kg': 1500.0, 'wheelbase_m': 2.7, 'cg_to_front_m': 1.2, 'cg_height_m': 0.55, 'wheel_radius_m': 0.31,
    'wheel_inertia_kgm2': 1.2, 'drive': 'front', 'max_drive_torque_nm': 3000.0, 'drag_kg_per_m': 0.4,
    'rolling_resistance': 0.012,
    # tyres and the road
    'surface': 'dry_asphalt', 'mu': None, 'tyre_pressure_kpa': 220.0, 'hydroplaning_onset': 0.7,
    'mu_hydroplaning': 0.05, 'mf_b': 10.0, 'mf_c': 1.9, 'mf_e': 0.97, 'relaxation_m': 0.3, 'v_low_mps': 0.5,
    'tone_wheel_teeth': 48,
}


def hydroplaning_speed_kmh(pressure_kpa: float) -> float:
    """Horne's dynamic hydroplaning speed: v_p ≈ 6.35·√p km/h, p the tyre pressure in kPa."""
    return 6.35 * math.sqrt(max(float(pressure_kpa), 0.0))


def magic_formula(kappa, b=10.0, c=1.9, e=0.97):
    """Pacejka's magic formula, normalised (peak about 1)."""
    bk = b * np.asarray(kappa, np.float64)
    return np.sin(c * np.arctan(bk - e * (bk - np.arctan(bk))))


def validate(params: dict) -> list:
    """Problems with chassis parameters (an empty list: fine)."""
    p = dict(PARAMS, **(params or {}))
    bad = [f'unknown chassis parameter {k!r}' for k in (params or {}) if k not in PARAMS]
    if p['surface'] not in SURFACES:
        bad.append(f'surface {p["surface"]!r} is not one of {", ".join(SURFACES)}')
    elif p['mu'] is not None:
        lo, hi = SURFACES[p['surface']]['range']
        if not 0.0 < float(p['mu']) <= 1.2:
            bad.append(f'mu {p["mu"]} is outside (0, 1.2]')
        elif not lo <= float(p['mu']) <= hi:
            bad.append(f'mu {p["mu"]} is outside {p["surface"]}\'s typical range {lo}–{hi}')
    for k in ('eps_tau_s', 'mass_kg', 'wheelbase_m', 'wheel_radius_m', 'wheel_inertia_kgm2', 'relaxation_m',
              'brake_apply_bar_s', 'brake_release_bar_s', 'steering_ratio', 'tyre_pressure_kpa'):
        if float(p[k]) <= 0:
            bad.append(f'{k} must be positive')
    for k in ('eps_delay_s', 'brake_delay_s', 'brake_dead_band_bar'):
        if float(p[k]) < 0:
            bad.append(f'{k} must not be negative')
    if not 0.0 < float(p['hydroplaning_onset']) < 1.0:
        bad.append('hydroplaning_onset must be in (0, 1)')
    if p['drive'] not in ('front', 'rear', 'all'):
        bad.append('drive must be front, rear or all')
    return bad


class Delay:
    """A pure transport delay: a ring buffer of one sample per sub-step."""

    def __init__(self, seconds: float, h: float, value: float = 0.0):
        self.n = max(int(round(float(seconds) / h)), 0)
        self.buf = np.full(self.n + 1, float(value))
        self.i = 0

    def push(self, u: float) -> float:
        """Put the newest input in; the one from n sub-steps ago comes out."""
        if self.n == 0:
            return float(u)
        self.buf[self.i] = u
        self.i = (self.i + 1) % (self.n + 1)
        return float(self.buf[self.i])


class Chassis:
    def __init__(self, params: dict | None = None, plant: str = 'model'):
        bad = validate(params or {})
        if bad:
            raise ValueError('; '.join(bad))
        if plant not in ('model', 'carla'):
            raise ValueError(f'plant {plant!r} is not model or carla')
        p = self.p = dict(PARAMS, **(params or {}))
        self.plant = plant
        self.surface = p['surface']
        self.mu = float(p['mu']) if p['mu'] is not None else SURFACES[self.surface]['mu']
        self.v_p = hydroplaning_speed_kmh(p['tyre_pressure_kpa']) / 3.6
        self.t = 0.0
        self.v = 0.0
        self.ax = self.ay = 0.0
        self.yaw_rate = 0.0                                   # deg/s, left positive
        self.sw = 0.0                                         # steering wheel angle, deg
        self.sw_rate = 0.0
        self.p_brake = 0.0                                    # line pressure, bar
        self.brake_target = 0.0                               # after the dead band
        self.omega = np.zeros(4)
        self.kappa = np.zeros(4)
        self.angle = np.zeros(4)                              # wheel rotation, rad (for the ticks)
        self.fx = np.zeros(4)
        self.cmd = {'steer_mode': 'angle', 'steer_req_deg': 0.0, 'steer_torque_nm': 0.0, 'brake_bar': 0.0,
                    'throttle_pct': 0.0, 'safe': False}
        self.steer_delay = Delay(p['eps_delay_s'], H_S)
        self.brake_delay = Delay(p['brake_delay_s'], H_S)
        r = p['wheel_radius_m']
        self.brake_nm = np.array([p['brake_nm_per_bar_front']] * 2 + [p['brake_nm_per_bar_rear']] * 2)
        self.driven = np.array({'front': [1, 1, 0, 0], 'rear': [0, 0, 1, 1], 'all': [1, 1, 1, 1]}[p['drive']], float)
        self.r = r

    # ------------------------------------------------------------------ inputs
    def command(self, steer_mode: str = 'angle', steer_req_deg: float = 0.0, steer_torque_nm: float = 0.0,
                brake_bar: float = 0.0, throttle_pct: float = 0.0):
        p = self.p
        self.cmd = {'steer_mode': steer_mode if steer_mode in ('angle', 'torque') else 'off',
                    'steer_req_deg': float(np.clip(steer_req_deg, -p['steer_max_deg'], p['steer_max_deg'])),
                    'steer_torque_nm': float(steer_torque_nm),
                    'brake_bar': float(np.clip(brake_bar, 0.0, p['brake_max_bar'])),
                    'throttle_pct': float(np.clip(throttle_pct, 0.0, 100.0)), 'safe': False}

    def safe_hold(self):
        """The watchdog's command: no throttle, the steering held where it is, the brake at safe_brake_bar (a held
        stop once stationary)."""
        self.cmd = {'steer_mode': 'angle', 'steer_req_deg': self.sw, 'steer_torque_nm': 0.0,
                    'brake_bar': float(self.p['safe_brake_bar']), 'throttle_pct': 0.0, 'safe': True}

    def set_motion(self, v_mps: float, yaw_rate_deg_s: float = 0.0, ax: float = 0.0, ay: float = 0.0):
        """Plant 'carla': the vehicle's motion this step (its speed over ground, yaw rate left positive, accels)."""
        self.v = max(float(v_mps), 0.0)
        self.yaw_rate, self.ax, self.ay = float(yaw_rate_deg_s), float(ax), float(ay)

    # ------------------------------------------------------------------ the model
    def mu_eff(self, v: float | None = None) -> float:
        """The friction coefficient in effect at speed v (m/s): μ, reduced past the hydroplaning onset on water."""
        v = self.v if v is None else float(v)
        if not SURFACES[self.surface]['water']:
            return self.mu
        on = self.p['hydroplaning_onset'] * self.v_p
        if v <= on:
            return self.mu
        f = min((v - on) / (self.v_p - on), 1.0)
        return self.mu + f * (self.p['mu_hydroplaning'] - self.mu)

    def hydroplaning(self) -> bool:
        return SURFACES[self.surface]['water'] and self.v > self.p['hydroplaning_onset'] * self.v_p

    def _steer(self, h: float):
        p, c = self.p, self.cmd
        if c['steer_mode'] == 'torque':
            u = c['steer_torque_nm'] * p['eps_torque_gain_deg_per_nm']
        elif c['steer_mode'] == 'angle':
            u = c['steer_req_deg']
        else:
            u = self.sw                                      # off: the steering stays where it is
        u = self.steer_delay.push(float(np.clip(u, -p['steer_max_deg'], p['steer_max_deg'])))
        d = (1.0 - math.exp(-h / p['eps_tau_s'])) * (u - self.sw)
        lim = p['eps_rate_deg_s'] * h
        d = min(max(d, -lim), lim)
        self.sw += d
        self.sw_rate = d / h

    def _brake(self, h: float):
        p = self.p
        r = self.brake_delay.push(self.cmd['brake_bar'])
        w = p['brake_dead_band_bar'] / 2.0
        if r <= w:
            self.brake_target = 0.0
        elif r > self.brake_target + w:
            self.brake_target = r - w
        elif r < self.brake_target - w:
            self.brake_target = r + w
        d = self.brake_target - self.p_brake
        lim = (p['brake_apply_bar_s'] if d > 0 else p['brake_release_bar_s']) * h
        self.p_brake += min(max(d, -lim), lim)

    def _wheels(self, h: float):
        p = self.p
        m, L, a, hcg, r = p['mass_kg'], p['wheelbase_m'], p['cg_to_front_m'], p['cg_height_m'], self.r
        b = L - a
        transfer = m * self.ax * hcg / L
        fz_front = max((m * G * b / L - transfer) / 2.0, 0.0)
        fz_rear = max((m * G * a / L + transfer) / 2.0, 0.0)
        fz = np.array([fz_front, fz_front, fz_rear, fz_rear])
        mu = self.mu_eff()
        v = self.v
        n_driven = max(self.driven.sum(), 1.0)
        if self.plant == 'model':
            t_drive = self.driven * p['max_drive_torque_nm'] * self.cmd['throttle_pct'] / 100.0 / n_driven
        else:                                # CARLA's motion: the drive torque that produced its acceleration
            need = m * max(self.ax, 0.0) * r if self.cmd['throttle_pct'] > 0 else 0.0
            t_drive = self.driven * min(need, p['max_drive_torque_nm']) / n_driven
        t_brake = self.brake_nm * self.p_brake
        # slip: implicit in the damping term, with the current wheel speed
        sigma = p['relaxation_m']
        damp = max(abs(v), p['v_low_mps'])
        self.kappa = (self.kappa + h / sigma * (self.omega * r - v)) / (1.0 + h * damp / sigma)
        self.kappa = np.clip(self.kappa, -1.0, 5.0)
        self.fx = mu * fz * magic_formula(self.kappa, p['mf_b'], p['mf_c'], p['mf_e'])
        rest = t_drive - self.fx * r - p['rolling_resistance'] * fz * r * (self.omega > 0)
        moving = self.omega > 1e-9
        # a turning wheel: the brake opposes its rotation; a stopped one turns only if the rest overcomes the brake
        w_new = np.where(moving, self.omega + h / p['wheel_inertia_kgm2'] * (rest - t_brake),
                         self.omega + h / p['wheel_inertia_kgm2'] * np.maximum(rest - t_brake, 0.0))
        # a spinning wheel is capped at a slip of SPIN_CAP (a traction control's limit, and the slip variable's)
        self.omega = np.clip(w_new, 0.0, (v + SPIN_CAP * max(v, 2.0)) / r)
        self.angle += self.omega * h
        if self.plant == 'model':
            drag = p['drag_kg_per_m'] * v * v + (p['rolling_resistance'] * m * G if v > 1e-6 else 0.0)
            f = float(self.fx.sum())
            ax = (f - drag) / m
            if v <= 1e-6 and ax < 0:
                ax = 0.0
            self.v = max(v + ax * h, 0.0)
            self.ax = ax
            delta = math.radians(self.sw / p['steering_ratio'])
            yr = self.v * math.tan(delta) / L
            if abs(yr * self.v) > mu * G and self.v > 0.1:
                yr = math.copysign(mu * G / self.v, yr)
            self.yaw_rate = math.degrees(yr)
            self.ay = yr * self.v

    def step(self, dt: float):
        """Advance dt seconds (in sub-steps of H_S)."""
        n = max(int(round(float(dt) / H_S)), 1)
        for _ in range(n):
            self._steer(H_S)
            self._brake(H_S)
            self._wheels(H_S)
            self.t += H_S

    # ------------------------------------------------------------------ outputs
    def ticks(self) -> np.ndarray:
        """Tone-wheel edges counted since the start (int64; the CAN signal wraps them)."""
        return np.floor(self.angle / (2 * math.pi) * self.p['tone_wheel_teeth']).astype(np.int64)

    def slip(self) -> np.ndarray:
        """The slip ratio of each wheel as its speeds say: (ω·r − v) / v (0 at standstill)."""
        if self.v < 0.1:
            return np.zeros(4)
        return (self.omega * self.r - self.v) / self.v

    def feedback(self) -> dict:
        """The signals the bridge sends (core.canmsgs roles): wheel ticks and speeds, yaw rate, accelerations, the
        steering, the brake pressure, the vehicle speed, the friction in effect."""
        p = self.p
        out = {f'TICKS_{w}': int(t) for w, t in zip(WHEELS, self.ticks())}
        out.update({f'SPEED_{w}': float(o * self.r * 3.6) for w, o in zip(WHEELS, self.omega)})
        out.update(YAW_RATE=self.yaw_rate, LAT_ACCEL=self.ay, LONG_ACCEL=self.ax, STEER_ANGLE=self.sw,
                   STEER_RATE=self.sw_rate, EPS_TORQUE=self.sw / p['eps_torque_gain_deg_per_nm'],
                   BRAKE_PRESSURE=self.p_brake, BRAKE_REQ_ECHO=self.cmd['brake_bar'], VEHICLE_SPEED=self.v * 3.6,
                   MU_EFF=self.mu_eff(), HYDROPLANING=int(self.hydroplaning()),
                   SURFACE=list(SURFACES).index(self.surface), SIM_TIME=self.t)
        return out

    def actuation(self) -> dict:
        """What the actuators do now: the road wheel angle (deg, left positive), the brake pressure and its share of
        brake_full_bar, the throttle (0–1)."""
        p = self.p
        return {'road_wheel_deg': self.sw / p['steering_ratio'], 'brake_bar': self.p_brake,
                'brake': float(min(self.p_brake / p['brake_full_bar'], 1.0)),
                'throttle': self.cmd['throttle_pct'] / 100.0, 'safe': self.cmd['safe']}

    def describe(self) -> dict:
        """The model's parameters as run (for run.json): the surface, μ, Horne's speed."""
        return {'params': dict(self.p), 'surface': self.surface, 'mu': self.mu,
                'surface_about': SURFACES[self.surface]['about'],
                'hydroplaning_speed_kmh': round(self.v_p * 3.6, 2) if SURFACES[self.surface]['water'] else None,
                'plant': self.plant}
