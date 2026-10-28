"""handedness: which side the traffic actually drives on in a world — verified, not assumed.

    python -m falcon_autonomy.workers handedness CONFIG.json

CONFIG.json is a drive config (the world, the seed, the traffic); the worker loads the world, spawns the traffic on
the traffic manager's autopilot, steps `seconds` of simulation, and once a second samples every vehicle: the lane it
is in (Map.get_waypoint), whether that lane runs along +s or −s of its road (does s grow along Waypoint.next?), and
whether the vehicle moves with the lane. In OpenDRIVE, positive lanes lie left of the reference line, so a vehicle
keeps left when it travels along +s in a positive lane or along −s in a negative lane, and keeps right otherwise.

run.json "handedness": samples, the share keeping left and the share moving with their lane, the verdict (LHT when
≥ 95 % keep left, RHT when ≥ 95 % keep right, else mixed), and the share of lanes the map itself says run along +s on
their left side (the map's rule, from its waypoints).
"""
from __future__ import annotations

import argparse
import math

from ..core import protocol as P
from ..core.provenance import read_json
from . import scene
from .common import Failed, Worker

VERDICT_SHARE = 0.95


def lane_travels_plus_s(wp) -> bool | None:
    """Whether the lane of wp runs along +s: s grows along next() on the same road (None when it cannot be told)."""
    nxt = wp.next(1.0)
    for n in nxt:
        if n.road_id == wp.road_id and abs(n.s - wp.s) > 1e-3:
            return n.s > wp.s
    prv = wp.previous(1.0)
    for n in prv:
        if n.road_id == wp.road_id and abs(n.s - wp.s) > 1e-3:
            return n.s < wp.s
    return None


class Handedness(Worker):
    name = 'handedness'

    def __init__(self, cfg, argv, cfg_path=''):
        super().__init__(argv, cfg['out'], cfg.get('timeout', 20.0), cfg.get('seed', 0), cfg,
                         {'config': cfg_path, 'xodr': cfg.get('xodr')} if cfg_path else None)
        self.cfg = cfg

    def body(self):
        cfg = self.cfg
        carla = self.import_carla()
        import numpy as np
        self.connect(cfg.get('host', '127.0.0.1'), cfg['port'], cfg.get('connect_wait_s', 60.0))
        world = scene.load_map(self, cfg)
        rng = np.random.default_rng(int(cfg.get('seed', 0)))
        fps = float(cfg.get('fps', 20.0))
        scene.synchronous(self, world, fps, int(cfg.get('seed', 0)), cfg.get('tm_port', 8310))
        world.tick()
        lib = world.get_blueprint_library()
        cmap = world.get_map()
        spawns = list(cmap.get_spawn_points())
        if not spawns:
            raise Failed('the map has no spawn points')
        prof = dict(cfg.get('traffic') or {}, walkers=0)
        prof['vehicles'] = int(cfg.get('vehicles', prof.get('vehicles', 30)) or 30)
        order = list(rng.permutation(len(spawns)))
        placed = scene.spawn_traffic(self, world, lib, spawns, order, rng, prof)
        if not placed:
            raise Failed('no vehicle could be placed')
        # the map's own lanes: a sample of driving waypoints
        lanes_left_plus = lanes_known = 0
        for wp in cmap.generate_waypoints(10.0)[:4000]:
            if wp.lane_type != carla.LaneType.Driving:
                continue
            d = lane_travels_plus_s(wp)
            if d is None:
                continue
            lanes_known += 1
            lanes_left_plus += int((wp.lane_id > 0) == d)
        self.phase('run')
        seconds = float(cfg.get('seconds', 30.0))
        steps = int(seconds * fps)
        left = right = aligned = moving = 0
        for i in range(steps):
            self.check()
            world.tick()
            self.alive()
            if (i + 1) % int(fps) or i < fps * 3:           # after 3 s, once a second
                continue
            for a, _kind in placed:
                v = a.get_velocity()
                speed = math.hypot(v.x, v.y)
                if speed < 1.0:
                    continue
                wp = cmap.get_waypoint(a.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp is None or wp.is_junction:
                    continue
                d = lane_travels_plus_s(wp)
                if d is None:
                    continue
                moving += 1
                yaw = math.radians(wp.transform.rotation.yaw)
                if (v.x * math.cos(yaw) + v.y * math.sin(yaw)) / speed > 0.7:
                    aligned += 1
                if (wp.lane_id > 0) == d:
                    left += 1
                else:
                    right += 1
            self.emit.progress((i + 1) / steps, f'{moving} samples')
        if moving == 0:
            raise Failed('no vehicle moved on a road outside junctions: nothing to verify')
        share_left = left / moving
        verdict = 'LHT' if share_left >= VERDICT_SHARE else 'RHT' if share_left <= 1 - VERDICT_SHARE else 'mixed'
        rep = {'samples': moving, 'vehicles': len(placed), 'keep_left_share': round(share_left, 4),
               'with_lane_share': round(aligned / moving, 4), 'verdict': verdict,
               'map_lanes_sampled': lanes_known,
               'map_left_lanes_along_plus_s_share': round(lanes_left_plus / max(lanes_known, 1), 4),
               'expected': cfg.get('handedness'), 'seconds': seconds,
               'method': 'traffic-manager vehicles sampled once a second after 3 s, outside junctions, moving > 1 m/s'}
        self.emit.log(f'{verdict}: {100 * share_left:.1f} % of {moving} samples keep left, '
                      f'{100 * aligned / moving:.1f} % move with their lane')
        return {'handedness': rep, 'summary': {'verdict': verdict, 'keep_left_share': rep['keep_left_share'],
                                                'samples': moving}, 'name': f'handedness of {cfg.get("name", "")}'}


def main(argv) -> int:
    ap = argparse.ArgumentParser(prog='falcon worker handedness')
    ap.add_argument('config')
    a = ap.parse_args(argv)
    try:
        cfg = read_json(a.config)
    except (OSError, ValueError) as e:
        print(f'failed: cannot read {a.config}: {e}', flush=True)
        return P.FAILED
    return Handedness(cfg, argv, a.config).execute()
