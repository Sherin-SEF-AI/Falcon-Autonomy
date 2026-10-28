"""The runtime ODD monitor of a worker (runs in CARLA's Python; core.oddmonitor does the checking).

A run's config may carry "odd" (a safety-case ODD, or the Loop's, which is mapped) and optionally "odd_reference" (a
reference file from `falcon odd calibrate`), "odd_every" (score every n-th camera frame; 10 by default) and
"odd_workspace" (where the ResNet-50 weights are, for a reference with a feature monitor). Without "odd" nothing is
monitored and nothing is recorded.

Each step the worker calls step(): the conditions are read from the simulator itself — World.get_weather() (so a live
weather event is seen), the ego's speed, the road features at the ego from the map (core.oddmonitor.road_features) —
and the config (place, handedness, background vehicles, the scenario and its actors); every odd_every-th call with a
camera image also scores the image against the reference. A change of state (into or out of the ODD, or its reasons)
is sent to the GUI ({"t": "odd", ...}) and logged; the step's record says {"in", "why", "road"} (and "weather" when it
differs from the config's); report() goes into run.json "odd_monitor" (the timeline and the time outside per reason).
"""
from __future__ import annotations

import math
import os

from ..core import oddmonitor as OM

WEATHER_ATTRS = ('cloudiness', 'precipitation', 'precipitation_deposits', 'wind_intensity', 'sun_azimuth_angle',
                 'sun_altitude_angle', 'fog_density', 'fog_distance', 'fog_falloff', 'wetness', 'scattering_intensity',
                 'mie_scattering_scale', 'rayleigh_scattering_scale', 'dust_storm')


class OddWatch:
    def __init__(self, w, world, cfg: dict):
        self.w, self.world, self.cfg = w, world, cfg
        self.odd = OM.monitor_odd(cfg['odd'])
        self.mon = OM.ConditionMonitor(self.odd)
        self.map = world.get_map()
        self.dm = None
        self.notes = []
        ref = cfg.get('odd_reference')
        if ref:
            try:
                self.dm = OM.DataMonitor(OM.load_reference(ref), cfg.get('odd_workspace') or '')
                if self.dm.unavailable:
                    self.notes.append('the reference\'s feature monitor needs the ResNet-50 weights: appearance only')
            except (OSError, ValueError, OM.OddError) as e:
                self.notes.append(f'the data monitor is off: {e}')
        self.every = max(int(cfg.get('odd_every', 10)), 1)
        self.tl = OM.Timeline(1.0 / float(cfg.get('fps', 20.0)))
        self.n = 0
        self.last_key = None
        self.unmon = set()
        self.cfg_weather = {k: float(v) for k, v in (cfg.get('weather') or {}).items() if isinstance(v, (int, float))}
        st0, _ = OM.static_state(cfg)
        self.static = {k: v for k, v in st0.items() if k in ('place', 'handedness', 'traffic_vehicles', 'actors',
                                                              'scenario')}
        self.last_flag = None
        self.episodes = []

    @classmethod
    def from_config(cls, w, world, cfg: dict):
        if not cfg.get('odd'):
            return None
        return cls(w, world, cfg)

    def wants_frame(self) -> bool:
        """Whether the next step() scores a camera image (a data monitor, every odd_every-th step)."""
        return self.dm is not None and self.n % self.every == 0

    def weather(self) -> dict:
        try:
            wp = self.world.get_weather()
        except RuntimeError:
            return dict(self.cfg_weather)
        return {k: float(getattr(wp, k)) for k in WEATHER_ATTRS if hasattr(wp, k)}

    def step(self, sim_t: float, ego, speed_mps: float | None = None, rgb=None) -> dict:
        wf = self.weather()
        try:
            loc = ego.get_location()
            road = OM.road_features(self.map, loc)
        except RuntimeError:
            road = None
        if speed_mps is None:
            v = ego.get_velocity()
            speed_mps = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        st = OM.state_from(wf, speed_mps, road_type=road)
        st.update(self.static)
        inside, reasons, unmon = self.mon.check(st)
        self.unmon.update(unmon)
        score = None
        scored = self.dm is not None and rgb is not None and self.n % self.every == 0
        if scored:
            r = self.dm.score(rgb)
            self.last_flag, score = bool(r['flag']), r['score']
        self.n += 1
        self.tl.add(sim_t, inside, reasons, self.last_flag, score, scored)    # a flag holds until the next score
        why = [r['text'] for r in reasons] + (['camera data out of distribution'] if self.last_flag else [])
        ok = inside and not self.last_flag
        key = (ok, tuple(sorted(r['dim'] for r in reasons)), bool(self.last_flag))
        if key != self.last_key:
            self.last_key = key
            self.w.emit.send({'t': 'odd', 'inside': ok, 'sim_t': round(sim_t, 3), 'why': why})
            self.w.emit.log(f'ODD: {"inside" if ok else "OUTSIDE — " + "; ".join(why)} at {sim_t:.1f} s')
        rec = {'in': ok, 'road': road}
        if why:
            rec['why'] = why
        diff = {k: v for k, v in wf.items() if abs(v - self.cfg_weather.get(k, v)) > 1e-3}
        if diff:
            rec['weather'] = diff
        if score is not None:
            rec['score'] = round(float(score), 4)
        return rec

    def end_episode(self) -> dict:
        """An episode ended (gym): its timeline's summary; the next episode starts a fresh one."""
        s = self.tl.summary()
        self.episodes.append((s['time_s'], s['time_outside_s']))
        self.tl = OM.Timeline(self.tl.dt)
        self.n, self.last_key, self.last_flag = 0, None, None
        return s

    def report(self) -> dict:
        if self.episodes:
            return {'odd': self.odd.get('name'), 'episodes': len(self.episodes),
                    'time_s': round(sum(a for a, _ in self.episodes), 4),
                    'time_outside_s': round(sum(b for _, b in self.episodes), 4),
                    'monitored': sorted(set(self.odd['dimensions']) - self.unmon), 'not_monitored': sorted(self.unmon),
                    'data_monitor': self.dm is not None, 'notes': self.notes}
        out = {'odd': self.odd.get('name'), 'monitored': sorted(set(self.odd['dimensions']) - self.unmon),
               'not_monitored': sorted(self.unmon), 'data_monitor': self.dm is not None, 'every': self.every,
               'notes': self.notes, **self.tl.summary()}
        return out


def write(w, path_dir: str, watch) -> None:
    """odd-monitor.json beside the run (the same document `falcon odd monitor` writes offline)."""
    if watch is None:
        return
    from ..core.provenance import write_json
    doc = dict(watch.report(), schema=OM.TIMELINE_SCHEMA, run=os.path.abspath(path_dir), source='runtime')
    write_json(os.path.join(path_dir, OM.MONITOR_FILE), doc)
