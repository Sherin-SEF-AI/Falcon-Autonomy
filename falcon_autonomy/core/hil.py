"""Hardware-in-the-loop sessions: their configs, the emulated sensor clocks, the clock-offset estimate between the host
and the device, and the latency statistics (standard library and numpy; the worker, the gateway, the CLI and the GUI
import it). docs/hardware.md is the guide; falcon_autonomy.hil holds the processes.

Timestamps. Every stamp is int64 nanoseconds. Host stamps are CLOCK_MONOTONIC (time.monotonic_ns), which the session
worker and the gateway share (one machine). Device stamps are the device's own CLOCK_MONOTONIC, an unrelated clock;
they are put on the host's time line with the offset estimate below, or used only in differences on one clock.
Each sensor message also carries the simulation time of its step and an emulated "hardware" timestamp
(EmulatedClock): hw_t = round(1e9 · (offset_s + (1 + drift_ppm · 1e-6) · sim_t) + N(0, jitter)) ns, seeded — what a
sensor with its own free-running clock would stamp, so a stack can be tested against realistic clocks.

The clock offset (estimate_offset): the gateway pings the device over UDP (NTP's exchange): it sends at t0 (host),
the device stamps receipt t1 and reply t2 (device), the gateway receives at t3 (host). Then
    offset θ = ((t1 − t0) + (t2 − t3)) / 2      (device = host + θ)      round-trip delay δ = (t3 − t0) − (t2 − t1).
If the two directions took equally long, θ is exact; whatever the asymmetry, the truth lies within θ ± δ/2 (each
direction takes at least 0 and at most δ). Queueing only adds delay, so in each window (WINDOW_S, or a sixth of a
shorter session) the sample with the least δ is kept, and a line θ(t) = a + b·(t − t_ref) is fitted through the
windows' minima (b: the drift between the two clocks). The stated uncertainty is half the median of those minimum
delays plus the fit's residual RMS — an error bound under the stated assumption (a delay is never negative) with the
fit's own scatter added, not a confidence interval. One-way stages (the network to the device and back) are shown only when it is below
TRUST_NS; the round trip (host clock only) and the time on the device (device clock only) need no offset and are
always exact.

Latency stages (STAGES), on the camera's path, which is the critical one (the lidar is sent without encoding):
the simulation step, the sensor data reaching the client, the hand-off to the gateway, encoding, the network to the
device, decoding, waiting for the other sensor, the stack, the control leaving the device, the network back, the
control reaching the simulation, and waiting for the next step, where it is applied (actuation). Their sum is the
end-to-end latency, from the start of the step that rendered a frame to the start of the step that applied the
control computed from it: in a real-time session it is quantised to whole steps. A control is on time when it is
applied at the very next step (the deadline is the next step's start); late when at a later one; superseded when a
newer control arrived first; missing when none came back; after_end when it came back after the session's last step
(not counted against the deadline).
"""
from __future__ import annotations

import copy
import math
import os

HIL_SCHEMA = 'falcon.hil_config/1'
WINDOW_S = 2.0
TRUST_NS = 1_000_000                       # one-way stages are trusted when the offset's uncertainty is below 1 ms
DEVICE_DEFAULTS = {'address': '', 'port': 7710, 'ssh_user': '', 'ssh_key': '', 'ssh_port': 22,
                   'remote_dir': '~/falcon-hil', 'python': 'python3', 'agent_pid': None}
CAMERA_TRANSPORTS = ('h264-rtp', 'raw-tcp')
ENCODERS = ('auto', 'nvh264enc', 'x264enc')
LIDAR_TRANSPORTS = ('tcp', 'udp')
DEVICE_MODES = ('local', 'remote')
# a rig without a lidar gets this one for the session (stated in run.json): 32 channels on the roof
DEFAULT_LIDAR = {'id': 'lidar-hil', 'name': 'HIL lidar', 'type': 'lidar',
                 'pose': {'x': 0.0, 'y': 0.0, 'z': 2.4, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
                 'channels': 32, 'range': 100.0, 'points_per_second': 600000, 'rotation_frequency': 10.0,
                 'upper_fov': 10.0, 'lower_fov': -30.0, 'horizontal_fov': 360.0}

# (key, label, start field, end field, clock): host = the host's clock, device = the device's, cross = both
STAGES = (
    ('sim_step', 'Simulation step', 'tick_start', 'tick_end', 'host'),
    ('sensor_read', 'Sensor data to the client', 'tick_end', 'ready', 'host'),
    ('handoff', 'Hand-off to the gateway', 'ready', 'gw_recv', 'host'),
    ('encode', 'Encode and packetise', 'gw_recv', 'cam_sent', 'host'),
    ('uplink', 'Network to the device', 'cam_sent', 'dev_cam_rx', 'cross'),
    ('decode', 'Decode on the device', 'dev_cam_rx', 'dev_dec_end', 'device'),
    ('sync', 'Waiting for the other sensor', 'dev_dec_end', 'dev_stack_start', 'device'),
    ('stack', 'Stack', 'dev_stack_start', 'dev_stack_end', 'device'),
    ('reply', 'Control leaves the device', 'dev_stack_end', 'dev_ctrl_sent', 'device'),
    ('downlink', 'Network back to the host', 'dev_ctrl_sent', 'ctrl_rx', 'cross'),
    ('to_sim', 'Control to the simulation', 'ctrl_rx', 'ctrl_worker', 'host'),
    ('wait_step', 'Waiting for the next step', 'ctrl_worker', 'applied', 'host'),
)
# other measures (key, label, start, end, clock); 'network_rtt' is computed (round trip − time on the device)
MEASURES = (
    ('e2e', 'End to end (sensor to actuation)', 'tick_start', 'applied', 'host'),
    ('processing', 'Step start to control at the simulation', 'tick_start', 'ctrl_worker', 'host'),
    ('round_trip', 'Camera sent to control received', 'cam_sent', 'ctrl_rx', 'host'),
    ('on_device', 'Camera received to control sent (device)', 'dev_cam_rx', 'dev_ctrl_sent', 'device'),
    ('encoder', 'Encoder alone', 'enc_start', 'enc_end', 'host'),
    ('cam_send', 'Camera bytes onto the socket', 'cam_sent', 'cam_sent_end', 'host'),
    ('lidar_send', 'Lidar framed and sent', 'gw_recv', 'lidar_sent_end', 'host'),
    ('lidar_uplink', 'Lidar network to the device', 'lidar_sent', 'dev_lidar_rx', 'cross'),
)


# ============================================================================================ settings and configs
def device_settings(settings) -> dict:
    """The HIL device from the user's settings (never from a project), with defaults."""
    d = dict(DEVICE_DEFAULTS)
    d.update({k: v for k, v in (settings.get('hil_device') or {}).items() if k in DEVICE_DEFAULTS})
    return d


def first_sensor(rig: dict, kind: str):
    return next((s for s in rig.get('sensors') or [] if s['type'] == kind), None)


def hil_config(project, world_id: str, rig_id: str, out_dir: str, port: int, tm_port: int, *, seconds: float = 30.0,
               fps: float = 20.0, seed: int = 0, width: int | None = None, height: int | None = None,
               camera: str = 'h264-rtp', encoder: str = 'auto', bitrate_kbps: int = 8000, lidar: str = 'tcp',
               lidar_encoding: str = 'xyzi', ros2_topic: str = '', lidar_overrides: dict | None = None,
               device: str = 'local', device_port: int = 7710, camera_clock: dict | None = None,
               lidar_clock: dict | None = None, sim_device: dict | None = None, stack: dict | None = None,
               sync_timeout_ms: float | None = None, control_timeout_s: float = 0.5,
               traffic_vehicles: int | None = None, gateway_python: str = '', warmup_steps: int = 10,
               workspace: str = '', host: str = '127.0.0.1', name: str = '') -> dict:
    """A session worker's config (falcon.hil_config/1). The world, its weather and traffic come from the project as
    for a drive (core.drive.drive_config); the rig's first pinhole camera and first lidar are streamed (a rig with
    no lidar gets DEFAULT_LIDAR). seconds 0: until stopped. device 'local' runs the device agent on this machine (a
    simulated device; sim_device: {clock_offset_ms, clock_drift_ppm, decoder}); 'remote' connects to the agent at
    the address in the user's settings, which the config never holds (the worker gets it from its environment)."""
    import sys
    from .drive import drive_config
    if camera not in CAMERA_TRANSPORTS:
        raise ValueError(f'camera transport {camera!r} is not one of {CAMERA_TRANSPORTS}')
    if encoder not in ENCODERS:
        raise ValueError(f'encoder {encoder!r} is not one of {ENCODERS}')
    if lidar not in LIDAR_TRANSPORTS:
        raise ValueError(f'lidar transport {lidar!r} is not one of {LIDAR_TRANSPORTS}')
    if device not in DEVICE_MODES:
        raise ValueError(f'device {device!r} is not one of {DEVICE_MODES}')
    if lidar_encoding not in ('xyzi', 'pointcloud2'):
        raise ValueError(f'lidar encoding {lidar_encoding!r} is not xyzi or pointcloud2')
    d = drive_config(project, world_id, rig_id, out_dir, port, tm_port, seconds=1.0, fps=fps, seed=seed, host=host,
                     name=name or 'HIL session', workspace=workspace)
    rig = d['rig']
    cam = first_sensor(rig, 'camera.pinhole')
    if cam is None:
        raise ValueError(f'rig {rig["name"]!r} has no pinhole camera to stream')
    cam = copy.deepcopy(cam)
    if width:
        cam['width'] = int(width)
    if height:
        cam['height'] = int(height)
    if cam['width'] < 160 or cam['height'] < 64:
        raise ValueError('the streamed camera must be at least 160 × 64 (the smallest frame NVENC encodes)')
    lid = first_sensor(rig, 'lidar')
    notes = []
    if lid is None:
        lid = copy.deepcopy(DEFAULT_LIDAR)
        notes.append('the rig has no lidar: the session adds a 32-channel roof lidar (core.hil.DEFAULT_LIDAR)')
    lid = dict(copy.deepcopy(lid), **(lidar_overrides or {}))
    traffic = dict(d['traffic'])
    if traffic_vehicles is not None:
        traffic['vehicles'] = int(traffic_vehicles)
    traffic['walkers'] = 0
    clk = lambda c: {'offset_ms': 0.0, 'drift_ppm': 0.0, 'jitter_us': 0.0, **(c or {})}   # noqa: E731
    cfg = {
        'schema': HIL_SCHEMA, 'name': d['name'], 'host': host, 'port': int(port), 'tm_port': int(tm_port),
        'town': d['town'], 'xodr': d['xodr'], 'opendrive': d['opendrive'], 'handedness': d['handedness'],
        'weather': d['weather'], 'vehicle': rig['vehicle'], 'traffic': traffic,
        'fps': float(fps), 'seconds': float(seconds), 'warmup_steps': int(warmup_steps), 'seed': int(seed),
        'out': os.path.abspath(out_dir), 'render_budget_mpx': d['render_budget_mpx'],
        'camera': {'sensor': cam, 'transport': camera, 'encoder': encoder, 'bitrate_kbps': int(bitrate_kbps),
                   'gop': max(int(round(fps)), 1)},
        'lidar': {'sensor': lid, 'transport': lidar, 'encoding': 'pointcloud2' if ros2_topic else lidar_encoding,
                  'ros2_topic': ros2_topic},
        'device': {'mode': device, 'port': int(device_port)},
        'sim_device': {'clock_offset_ms': 0.0, 'clock_drift_ppm': 0.0, 'decoder': 'auto', **(sim_device or {})},
        'clocks': {'camera': clk(camera_clock), 'lidar': clk(lidar_clock)},
        'stack': dict(stack or {}),
        'sync_timeout_ms': float(sync_timeout_ms if sync_timeout_ms is not None else 1000.0 / float(fps)),
        'control_timeout_s': float(control_timeout_s),
        'gateway_python': gateway_python or sys.executable,
        'notes': notes,
        'project': dict(d['project']),
    }
    from .oddmonitor import CFG_KEYS
    cfg.update({k: d[k] for k in CFG_KEYS if k in d})     # the workspace's ODD, monitored in the session
    return cfg


def submit_session(svc, cfg: dict, address: str = '', title: str = ''):
    """Write the config into its run folder (hil.json) and queue the session worker. A remote device's address goes
    to the worker in its environment (FALCON_HIL_ADDRESS), never into a file."""
    from .paths import ensure_dir
    from .provenance import write_json
    if cfg['device']['mode'] == 'remote' and not address:
        raise ValueError('no device address: set it on the Hardware page or with `falcon hil device --address`')
    run_dir = ensure_dir(cfg['out'])
    path = os.path.join(run_dir, 'hil.json')
    write_json(path, cfg)
    spec = svc.worker_spec('hil', [path], title or cfg.get('name', 'HIL session'), 'hil', run_dir,
                           meta={'config': path})
    if address:
        spec.env = dict(spec.env or os.environ, FALCON_HIL_ADDRESS=address)
    return svc.queue.submit(spec)


# ============================================================================================ clocks
class EmulatedClock:
    """A sensor's own clock (module doc): hw_t = round(1e9 · (offset_s + (1 + drift_ppm · 1e-6) · sim_t) + N(0,
    jitter)) ns, seeded. offset_ms, drift_ppm, jitter_us."""

    def __init__(self, offset_ms: float = 0.0, drift_ppm: float = 0.0, jitter_us: float = 0.0, seed: int = 0):
        import numpy as np
        self.offset_s = float(offset_ms) / 1e3
        self.drift = float(drift_ppm) * 1e-6
        self.jitter_ns = float(jitter_us) * 1e3
        self.rng = np.random.default_rng(seed)

    def stamp(self, sim_t: float) -> int:
        t = 1e9 * (self.offset_s + (1.0 + self.drift) * float(sim_t))
        if self.jitter_ns:
            t += float(self.rng.normal(0.0, self.jitter_ns))
        return int(round(t))


def fit_clock(sim_t, hw_t) -> dict:
    """What a sensor clock's stamps say about it: offset (ms), drift (ppm) and the residual (jitter, µs RMS) of a
    line through hw_t against sim_t — the emulation's self-check."""
    import numpy as np
    s = np.asarray(sim_t, np.float64)
    h = np.asarray(hw_t, np.float64) / 1e9
    if len(s) < 3:
        return {}
    A = np.c_[np.ones_like(s), s]
    (a, b), *_ = np.linalg.lstsq(A, h, rcond=None)
    res = h - (a + b * s)
    return {'offset_ms': round(a * 1e3, 6), 'drift_ppm': round((b - 1.0) * 1e6, 4),
            'jitter_us': round(float(np.sqrt(np.mean(res ** 2))) * 1e6, 3), 'n': int(len(s))}


def ping_sample(t0: int, t1: int, t2: int, t3: int) -> tuple:
    """(offset θ, round-trip delay δ) of one ping, ns (device = host + θ)."""
    return ((t1 - t0) + (t2 - t3)) / 2.0, float((t3 - t0) - (t2 - t1))


def estimate_offset(samples, window_s: float = WINDOW_S) -> dict:
    """The device clock's offset from the host's (module doc) from pings [(t0, t1, t2, t3)]: offset_ns at t_ref_ns
    (host), drift_ppm, uncertainty_ns, the round-trip delays. {} without samples."""
    import numpy as np
    S = [tuple(int(v) for v in s[:4]) for s in samples if s is not None and len(s) >= 4]
    if not S:
        return {}
    t0 = np.array([s[0] for s in S], np.int64)
    th = np.array([ping_sample(*s)[0] for s in S])
    dl = np.array([ping_sample(*s)[1] for s in S])
    ok = dl >= 0
    if not ok.any():
        return {}
    t0, th, dl = t0[ok], th[ok], dl[ok]
    span_s = float(t0.max() - t0.min()) / 1e9
    window_s = min(float(window_s), max(span_s / 6.0, 0.2))     # at least six windows when the session is short
    w = ((t0 - t0.min()) // int(window_s * 1e9)).astype(np.int64)
    pick = np.array([np.flatnonzero(w == k)[np.argmin(dl[w == k])] for k in np.unique(w)])
    tw, thw, dlw = t0[pick], th[pick], dl[pick]
    t_ref = int(np.median(t0))
    if len(pick) >= 4:
        x = (tw - t_ref).astype(np.float64)
        A = np.c_[np.ones_like(x), x]
        (a, b), *_ = np.linalg.lstsq(A, thw, rcond=None)
        resid = float(np.sqrt(np.mean((thw - (a + b * x)) ** 2)))
    else:
        a, b = float(thw[np.argmin(dlw)]), 0.0
        resid = 0.0
    unc = float(np.median(dlw)) / 2.0 + resid
    return {'offset_ns': float(a), 't_ref_ns': t_ref, 'drift_ppm': float(b) * 1e6, 'uncertainty_ns': unc,
            'fit_residual_ns': resid, 'min_rtt_ns': float(dl.min()), 'median_rtt_ns': float(np.median(dl)),
            'samples': int(len(S)), 'windows': int(len(pick)), 'window_s': float(window_s),
            'trusted': bool(unc <= TRUST_NS),
            'method': 'NTP-style UDP pings; least-delay sample per window; line fit through them; uncertainty = '
                      'half the median least delay + fit residual RMS (a bound if no delay is negative)'}


def offset_at(est: dict, host_ns: float) -> float:
    if not est:
        return 0.0
    return est['offset_ns'] + est['drift_ppm'] * 1e-6 * (host_ns - est['t_ref_ns'])


def to_host(dev_ns, est: dict) -> float:
    """A device stamp on the host's time line: d = h + a + b·(h − t_ref) solved for h."""
    if not est:
        return float(dev_ns)
    b = est['drift_ppm'] * 1e-6
    return (float(dev_ns) - est['offset_ns'] + b * est['t_ref_ns']) / (1.0 + b)


# ============================================================================================ statistics
def describe(values) -> dict:
    """n, median, p95, p99, mean, std (the jitter), min, max of values in ms."""
    import numpy as np
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], np.float64)
    if not len(v):
        return {'n': 0}
    r = lambda x: round(float(x), 3)                          # noqa: E731
    return {'n': int(len(v)), 'median': r(np.median(v)), 'p95': r(np.percentile(v, 95)),
            'p99': r(np.percentile(v, 99)), 'mean': r(v.mean()), 'std': r(v.std()), 'min': r(v.min()),
            'max': r(v.max())}


def _span(rec: dict, a: str, b: str, clock: str, est: dict):
    """b − a in ms on the stage's clock, or None when a stamp is missing."""
    ta, tb = rec.get(a), rec.get(b)
    if ta is None or tb is None:
        return None
    if clock == 'cross':
        if not est:
            return None
        ta = to_host(ta, est) if a.startswith('dev_') else float(ta)
        tb = to_host(tb, est) if b.startswith('dev_') else float(tb)
    return (float(tb) - float(ta)) / 1e6


def stage_values(records, est: dict) -> dict:
    """Per stage and measure, the values (ms) of every record that has both stamps."""
    out = {}
    for key, _label, a, b, clock in STAGES + MEASURES:
        out[key] = [_span(r, a, b, clock, est) for r in records]
    rt, dev = out['round_trip'], out['on_device']
    out['network_rtt'] = [None if x is None or y is None else x - y for x, y in zip(rt, dev)]
    return out


def interarrival_jitter(records) -> dict:
    """RFC 3550's interarrival jitter of the camera at the device: D = (R_i − R_{i−1}) − (S_i − S_{i−1}) with S the
    send time (host) and R the receive time (device) of consecutive received frames; J += (|D| − J) / 16. Each
    clock is only differenced against itself, so no offset is needed."""
    import numpy as np
    pts = sorted((r['frame'], r['cam_sent'], r['dev_cam_rx']) for r in records
                 if r.get('cam_sent') is not None and r.get('dev_cam_rx') is not None)
    if len(pts) < 3:
        return {}
    J = 0.0
    D = []
    for (f0, s0, r0), (f1, s1, r1) in zip(pts, pts[1:]):
        if f1 != f0 + 1:
            continue
        d = ((r1 - r0) - (s1 - s0)) / 1e6
        D.append(abs(d))
        J += (abs(d) - J) / 16.0
    return {'rfc3550_ms': round(J, 4), 'mean_abs_d_ms': round(float(np.mean(D)), 4) if D else None, 'pairs': len(D)}


def summarise(records: list, est: dict, period_ns: int, counts: dict | None = None, device: dict | None = None,
              cfg: dict | None = None) -> dict:
    """The session's latency summary (docs/hardware.md): per stage and measure describe(), the waterfall (median
    cascade), deadline accounting, jitter, the clock estimate and what it allows, the emulated clocks' self-check."""
    import numpy as np
    vals = stage_values(records, est)
    trusted = bool(est) and est.get('trusted', False)
    stages = {}
    for key, label, a, b, clock in STAGES:
        d = describe(vals[key])
        d.update(label=label, clock=clock, start=a, end=b)
        if clock == 'cross':
            d['one_way'] = True
            d['trusted'] = trusted
        stages[key] = d
    measures = {}
    for key, label, a, b, clock in MEASURES:
        measures[key] = dict(describe(vals[key]), label=label, clock=clock)
        if clock == 'cross':
            measures[key]['trusted'] = trusted
    measures['network_rtt'] = dict(describe(vals['network_rtt']), label='Network round trip (exact: round trip − '
                                                                         'time on the device)', clock='host+device')
    cascade, t = [], 0.0
    for key, label, *_ in STAGES:
        m = stages[key].get('median')
        if m is None:
            continue
        cascade.append({'key': key, 'label': label, 'start_ms': round(t, 3), 'median_ms': m,
                        'p95_ms': stages[key].get('p95'), 'clock': stages[key]['clock']})
        t += m
    st = [r.get('status') for r in records]
    n = len(records)
    by = {k: st.count(k) for k in ('on_time', 'late', 'superseded', 'missing', 'after_end')}
    nd = n - by['after_end']                    # a control that came back after the last step had no step to miss
    ticks = np.array(sorted(r['tick_start'] for r in records if r.get('tick_start') is not None), np.int64)
    tick_iv = np.diff(ticks) / 1e6 if len(ticks) > 1 else np.zeros(0)
    deadline = {'frames': nd, **by, 'deadline_misses': nd - by['on_time'],
                'on_time_share': round(by['on_time'] / nd, 4) if nd else None,
                'rule': 'on time: the control computed from step k is applied at step k + 1; controls that came '
                        'back after the session\'s last step (after_end) are not counted'}
    dev_counts = (device or {}).get('counts') or {}
    cam_rx = sum(1 for r in records if r.get('dev_cam_rx') is not None)
    dec = sum(1 for r in records if r.get('dev_dec_end') is not None)
    lid_rx = sum(1 for r in records if r.get('dev_lidar_rx') is not None)
    drops = {'camera_sent': sum(1 for r in records if r.get('cam_sent') is not None),
             'camera_received': cam_rx, 'camera_decoded': dec,
             'lidar_sent': sum(1 for r in records if r.get('lidar_sent') is not None), 'lidar_received': lid_rx,
             'camera_lost': sum(1 for r in records if r.get('cam_sent') is not None and r.get('dev_cam_rx') is None),
             'lidar_lost': sum(1 for r in records if r.get('lidar_sent') is not None and
                               r.get('dev_lidar_rx') is None),
             'device': dev_counts, 'gateway': (counts or {}).get('gateway', {})}
    clocks = {}
    for k, f in (('camera', 'cam_hw_t'), ('lidar', 'lidar_hw_t')):
        rows = [(r['sim_t'], r[f]) for r in records if r.get(f) is not None]
        if rows:
            clocks[k] = {'configured': ((cfg or {}).get('clocks') or {}).get(k), 'recovered': fit_clock(*zip(*rows))}
    out = {'frames': n, 'period_ms': period_ns / 1e6, 'stages': stages, 'measures': measures, 'waterfall': cascade,
           'deadline': deadline, 'drops': drops,
           'jitter': {'step_interval_ms': describe(tick_iv), 'camera_interarrival': interarrival_jitter(records)},
           'clock_offset': est, 'one_way_trusted': trusted, 'sensor_clocks': clocks, 'worker': counts or {}}
    if cfg and cfg.get('device', {}).get('mode') == 'local' and est:
        sd = cfg.get('sim_device') or {}
        # the agent's emulated clock: d = h·(1 + ppm·1e-6) + offset, so θ(t_ref) = offset + ppm·1e-6·t_ref
        truth = float(sd.get('clock_offset_ms', 0.0)) * 1e6 + float(sd.get('clock_drift_ppm', 0.0)) * 1e-6 * \
            est['t_ref_ns']
        out['clock_offset_check'] = {'truth_ns': truth, 'error_ns': est['offset_ns'] - truth,
                                     'within_uncertainty': abs(est['offset_ns'] - truth) <= est['uncertainty_ns'],
                                     'note': 'a simulated device: its clock is this machine\'s, offset and drifted '
                                             'by the emulation, so the truth is known'}
    return out


def live_stats(records: list, est: dict, period_ns: int) -> dict:
    """The compact numbers a running session reports once a second (medians of the recent frames)."""
    s = summarise(records, est, period_ns)
    return {'frames': s['frames'], 'waterfall': s['waterfall'],
            'e2e': {k: s['measures']['e2e'].get(k) for k in ('median', 'p95', 'p99')},
            'round_trip': {k: s['measures']['round_trip'].get(k) for k in ('median', 'p95')},
            'deadline': {k: s['deadline'].get(k) for k in ('on_time', 'late', 'superseded', 'missing')},
            'one_way_trusted': s['one_way_trusted'],
            'offset': None if not est else {'offset_ms': round(est['offset_ns'] / 1e6, 4),
                                            'uncertainty_ms': round(est['uncertainty_ns'] / 1e6, 4)}}


def read_jsonl(path: str) -> list:
    import json
    try:
        with open(path, encoding='utf-8') as fh:
            return [json.loads(x) for x in fh if x.strip()]
    except (OSError, ValueError):
        return []


def summarise_run(run_dir: str) -> dict:
    """A session's summary from its files alone (hil.json, latency.jsonl, clock.jsonl, device.json, session.json):
    what the worker puts in run.json, and what `falcon hil report` recomputes for a session that was cancelled."""
    from .provenance import read_json

    def rd(name):
        try:
            return read_json(os.path.join(run_dir, name))
        except (OSError, ValueError):
            return {}
    cfg = rd('hil.json')
    records = read_jsonl(os.path.join(run_dir, 'latency.jsonl'))
    samples = [(d['t0'], d['t1'], d['t2'], d['t3']) for d in read_jsonl(os.path.join(run_dir, 'clock.jsonl'))]
    device = rd('device.json')
    ses = rd('session.json')
    if not cfg or not records:
        raise ValueError(f'{run_dir} holds no HIL session (hil.json and latency.jsonl)')
    period = int(round(1e9 / float(cfg['fps'])))
    est = estimate_offset(samples)
    counts = dict(ses.get('counts') or {}, gateway=rd('gateway.json').get('counts') or {})
    out = summarise(records, est, period, counts, device.get('stats') or {}, cfg)
    out['session'] = ses.get('session') or {}
    out['performance'] = ses.get('performance') or {}
    return out
