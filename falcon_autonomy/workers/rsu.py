"""Roadside units in a run (CARLA's Python): their sensors standing in the world, recorded with the ego's labels, and
their collective perception messages (core.cpm) built from what those sensors actually saw. docs/v2x.md.

CARLA lets a client spawn a sensor attached to nothing (World.spawn_actor without attach_to): it stays at its world
transform. A station spawns each of its RSU's sensors that way at the pose core.v2x.to_world gives (the station turns
only by yaw, so the composition is exact). Measured on CARLA 0.9.16 (docs/v2x.md): such cameras and lidars deliver
every synchronous tick like attached ones, and the instance camera and the semantic lidar label what they see.

Recording (drives). The station's sensors are recorded like the ego's, under ids "<rsu id>.<sensor id>": a camera by
workers.camgt.CameraRecorder (RGB, semantic, instance, range, objects with box3d in the STATION frame — its foot,
x along its yaw), a lidar by workers.lidar.LidarRecorder (points in the lidar's frame, per-point tag and actor id).
The anchor standing in for the ego is a fixed transform (Anchor): the label code reads its transform as it reads the
ego's, and no pixel is the "ego" class (the ego vehicle is an ordinary car to a roadside camera, with its track id).

Perception (every CPM). Visible: a road user (a vehicle or walker the world holds) is perceived by a camera when at
least min_pixels pixels of the camera's instance image carry its actor id with a road-user tag (12–19), by a lidar
when at least min_points points of its latest full rotation hit it (the semantic lidar's object index) — so an
occluded object, which no pixel or point reaches, is never reported. Its reported state is the ground truth at the
tick (the box centre, velocity, yaw, yaw rate, size, class) in the station's East-North-Up frame, with Gaussian noise
when the RSU's cpm.noise asks (seeded by the run's seed, the station and the step), the 95 % confidences following.
Free space: each sensor's field of view in `sectors` slices, each cut at the nearest STATIC occluder the sensor saw
in it (buildings, walls, fences, poles, lights, signs, vegetation, static and dynamic props, bridges, guard rails, and
vehicles that are no actor: the map's parked cars) — camera pixels with their depth, lidar points more than 0.3 m above
the ground — as a PerceptionRegion (RadialShapes) with shadowingApplies true: what is behind a reported road user is
not claimed free; the declared range where nothing was hit.
"""
from __future__ import annotations

import collections
import json
import math
import os

import numpy as np

from ..core import cpm as C
from ..core import v2x as V
from .common import ROLE_SENSOR

ROAD_USER_TAGS = np.arange(12, 20)
STATIC_OCCLUDER_TAGS = np.array([3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 26, 28])
GROUND_CLEARANCE_M = 0.3
QUALITY_FULL = {'pixels': 400, 'points': 40}     # evidence giving objectPerceptionQuality 15 (heuristic)
TRACK_GAP_S = 1.0                                # an object unseen this long starts a new age


class Anchor:
    """A fixed station frame standing in for the ego actor: get_transform() like an actor's; compose() puts a transform
    given in the station frame into the world (yaw-only station: exact)."""
    is_static_anchor = True
    id = -1

    def __init__(self, carla, x: float, y: float, z: float, yaw: float):
        self.carla = carla
        self.x, self.y, self.z, self.yaw = float(x), float(y), float(z), float(yaw)
        self.tf = carla.Transform(carla.Location(x=self.x, y=self.y, z=self.z), carla.Rotation(yaw=self.yaw))

    def get_transform(self):
        return self.tf

    def compose(self, tf):
        a = math.radians(self.yaw)
        c, s = math.cos(a), math.sin(a)
        lx, ly, lz = tf.location.x, tf.location.y, tf.location.z
        carla = self.carla
        return carla.Transform(carla.Location(x=self.x + c * lx - s * ly, y=self.y + s * lx + c * ly, z=self.z + lz),
                               carla.Rotation(pitch=tf.rotation.pitch, yaw=tf.rotation.yaw + self.yaw,
                                              roll=tf.rotation.roll))


class StaticCameras:
    """scene.Cameras for a station's sensors: spawn() puts the camera at its world transform, attached to nothing;
    everything else (queues, the writer pool, get) is the run's own Cameras."""

    def __init__(self, cameras, anchor: Anchor):
        self._c, self._a = cameras, anchor

    def spawn(self, world, lib, bp_id, width, height, fov, tf, parent=None, attachment=None, attrs=None):
        return self._c.spawn(world, lib, bp_id, width, height, fov, self._a.compose(tf), None, attrs=attrs)

    def __getattr__(self, k):
        return getattr(self._c, k)


def _view_rays(W: int, H: int, fov: float, stride: int):
    """Directions (x forward, y right, z up: CARLA's camera axes) of every stride-th pixel centre, and the pixel
    indices: a planar depth d puts the point at d × direction."""
    f = W / (2.0 * math.tan(math.radians(fov) / 2.0))
    us = np.arange(stride // 2, W, stride)
    vs = np.arange(stride // 2, H, stride)
    u, v = np.meshgrid(us, vs)
    d = np.stack([np.ones(u.shape), (u + 0.5 - W / 2.0) / f, -(v + 0.5 - H / 2.0) / f], -1).reshape(-1, 3)
    return d, (v * W + u).ravel()


def _matrix(p: dict) -> np.ndarray:
    from .lidar import mount_matrix
    return mount_matrix(p)


class Station:
    """One RSU in the run. record: its sensors are recorded (drives); else only the label sensors the CPM needs are
    spawned (stack runs: instance and depth cameras, semantic lidars)."""

    def __init__(self, w, world, lib, rsu: dict, cameras, raster=None, record=True, fps=20.0, seed=0):
        carla = w.carla
        self.w, self.world, self.rsu, self.record, self.fps = w, world, rsu, record, float(fps)
        self.raster = raster
        self.id = rsu['id']
        self.cfg = V.cpm_settings(rsu)
        self.station_id = V.station_id(rsu)
        if rsu.get('z') is None:
            try:
                wp = world.get_map().get_waypoint(carla.Location(x=float(rsu['x']), y=float(rsu['y']), z=0.0))
                self.ground_z = float(wp.transform.location.z) if wp is not None else 0.0
            except RuntimeError:
                self.ground_z = 0.0
        else:
            self.ground_z = float(rsu['z'])
        x0, y0, z0 = V.station_xyz(rsu, self.ground_z)
        self.anchor = Anchor(carla, x0, y0, z0, float(rsu.get('yaw', 0.0)))
        self.T = np.asarray(_matrix({'x': x0, 'y': y0, 'z': z0, 'roll': 0.0, 'pitch': 0.0,
                                     'yaw': float(rsu.get('yaw', 0.0))}))
        try:
            g = world.get_map().transform_to_geolocation(carla.Location(x=x0, y=y0, z=z0))
            self.geo = (float(g.latitude), float(g.longitude), float(g.altitude))
        except (RuntimeError, AttributeError):
            self.geo = (0.0, 0.0, z0)
            w.note(f'{self.id}: the map has no geo-reference; its CPMs give 0° N 0° E as the reference position')
        self.cams, self.lids = [], []
        static = StaticCameras(cameras, self.anchor)
        self.cameras = cameras
        from ..core import lens as LENS
        k = 0
        for s in rsu['sensors']:
            s = V.sensor_with_defaults(s)
            s2 = dict(s, id=f'{self.id}.{s["id"]}', pose=V.mount_pose(rsu, s), station=self.id)
            k += 1
            if s['type'].startswith('camera.'):
                if record:
                    from .camgt import CameraRecorder
                    r = CameraRecorder(w, world, lib, self.anchor, s2, static, s2['pose'])
                    plan, views = r.plan, r.views
                else:
                    r = None
                    plan = LENS.plan_camera(s2)
                    views = []
                    for v in plan['views']:
                        vp = v['pose']
                        tf = carla.Transform(carla.Location(x=vp['x'], y=vp['y'], z=vp['z']),
                                             carla.Rotation(pitch=vp['pitch'], yaw=vp['yaw'], roll=vp['roll']))
                        attrs = LENS.LABEL_VIEW_ATTRIBUTES if s['type'] == 'camera.fisheye' else None
                        views.append({kind: static.spawn(world, lib, bp, v['width'], v['height'], v['fov'], tf,
                                                         attrs=attrs)
                                      for kind, bp in (('instance', 'sensor.camera.instance_segmentation'),
                                                       ('depth', 'sensor.camera.depth'))})
                geo = []
                for v in plan['views']:
                    d, pix = _view_rays(int(v['width']), int(v['height']), float(v['fov']), 4)
                    M = _matrix(v['pose'])
                    geo.append({'dirs_station': d @ M[:3, :3].T, 'origin': M[:3, 3], 'pix': pix})
                hfov = float(s.get('fov', 90.0)) if s['type'] == 'camera.pinhole' else \
                    float((s.get('lens') or {}).get('fov', 190.0))
                vfov = 2.0 * math.degrees(math.atan(math.tan(math.radians(min(hfov, 179.0)) / 2.0) *
                                                    s['height'] / s['width'])) if s['type'] == 'camera.pinhole' else hfov
                self.cams.append({'s': s2, 'sid': k, 'recorder': r, 'views': views, 'geo': geo, 'hfov': hfov,
                                  'vfov': vfov, 'range': float(self.cfg['camera_range_m'])})
            elif s['type'] == 'lidar':
                from .lidar import LidarRecorder, lidar_blueprints
                if record:
                    L = LidarRecorder(w, world, lib, self.anchor, s2, cameras, ROLE_SENSOR, raster)
                    L.set_rate(fps)
                    sem = None
                    per = L.steps_per_sweep
                else:
                    L = None
                    bp = lidar_blueprints(lib, s2, ROLE_SENSOR)[1]
                    tf = carla.Transform(carla.Location(x=s2['pose']['x'], y=s2['pose']['y'], z=s2['pose']['z']),
                                         carla.Rotation(pitch=s2['pose']['pitch'], yaw=s2['pose']['yaw'],
                                                        roll=s2['pose']['roll']))
                    sem = w.own(world.spawn_actor(bp, self.anchor.compose(tf)))
                    import queue
                    q = queue.Queue()
                    cameras.queues[sem.id] = q
                    sem.listen(q.put)
                    per = max(int(round(fps / float(s2['rotation_frequency']))), 1)
                self.lids.append({'s': s2, 'sid': k, 'recorder': L, 'sem': sem, 'window': collections.deque(maxlen=per),
                                  'M': self.T @ _matrix(s2['pose']), 'range': float(s2['range'])})
        self.every = max(int(round(self.fps / max(float(self.cfg['rate_hz']), 1e-6))), 1)
        self.first_seen: dict = {}
        self.last_seen: dict = {}
        self.sent = 0
        self.bytes = []
        self.objects_sent = 0
        self.rng_seed = int(seed)
        self.t0_ms = C.its_time_ms(self.cfg['epoch'])
        w.alive()

    # ------------------------------------------------------------------ a step
    @property
    def recorders(self):
        return [c['recorder'] for c in self.cams if c['recorder'] is not None]

    @property
    def lidar_recorders(self):
        return [x['recorder'] for x in self.lids if x['recorder'] is not None]

    def step(self, i: int, frame: int, sim_t: float, rec: dict, out_dir: str = '', record_frame: bool = True,
             boxes_fn=None, director=None) -> dict | None:
        """Collect the station's sensors for this frame; record them (drives, on recorded steps); every `every`
        steps build the CPM. Returns the CPM line (cpm.jsonl) or None."""
        imgs = {}
        for c in self.cams:
            if c['recorder'] is not None:
                imgs[c['sid']] = c['recorder'].collect(frame)
            else:
                imgs[c['sid']] = [{k: self.cameras.get(a.id, frame) for k, a in v.items()} for v in c['views']]
        if self.record and record_frame and self.recorders:
            boxes = boxes_fn(self.anchor) if boxes_fn is not None else None
            if boxes is not None:
                boxes = dict(boxes, ego_id=None)        # no "ego" pixels: the ego is a car to a roadside camera
            for c in self.cams:
                r = c['recorder']
                kinds = r.kinds()
                for kind in kinds:
                    ext = {'rgb': 'jpg', 'objects': 'json', 'flow': 'npy'}.get(kind, 'png')
                    rec['files'][f'{r.s["id"]}/{kind}'] = os.path.join(kind, r.s['id'], f'{i:06d}.{ext}')
                self.cameras.pending.append(self.cameras.pool.submit(r.write, imgs[c['sid']], i, out_dir, boxes,
                                                                     self.raster))
        for x in self.lids:
            if x['recorder'] is not None:
                x['recorder'].step(frame, sim_t, out_dir, rec)
                xyz, tag, idx = x['recorder'].last_slice
            else:
                m = self.cameras.get(x['sem'].id, frame)
                S = np.frombuffer(m.raw_data, C_SEM)
                xyz = np.c_[S['x'], S['y'], S['z']].astype(np.float64)
                tag, idx = S['tag'].astype(np.uint8), S['idx'].astype(np.uint32)
            x['window'].append((xyz, tag, idx))
        if i % self.every:
            return None
        return self.perceive(i, frame, sim_t, imgs, director)

    # ------------------------------------------------------------------ perception
    def perceive(self, i, frame, sim_t, imgs, director=None) -> dict:
        cfg = self.cfg
        actors = {}
        for a in self.world.get_actors():
            if a.type_id.startswith(V.ROAD_USER):
                actors[a.id] = a
        by16 = {}
        for aid in actors:
            by16.setdefault(aid & 0xFFFF, []).append(aid)
        evidence: dict = {}                 # actor -> {sensor id: pixels or points}
        sectors_of = {}
        for c in self.cams:
            occ = []
            for v, g in zip(imgs[c['sid']], c['geo']):
                ins = v['instance']
                a = np.frombuffer(ins.raw_data, np.uint8).reshape(ins.height, ins.width, 4)
                tags = a[:, :, 2]
                ids = a[:, :, 1].astype(np.int64) | (a[:, :, 0].astype(np.int64) << 8)
                road = np.isin(tags, ROAD_USER_TAGS)
                cnt = np.bincount(ids[road].ravel(), minlength=1)
                for rid in np.flatnonzero(cnt >= int(cfg['min_pixels'])):
                    for aid in by16.get(int(rid), ()):
                        evidence.setdefault(aid, {})[c['sid']] = evidence.get(aid, {}).get(c['sid'], 0) + int(cnt[rid])
                if cfg.get('free_space', True):
                    d = v['depth']
                    b = np.frombuffer(d.raw_data, np.uint8).reshape(d.height, d.width, 4).astype(np.float64)
                    depth = 1000.0 * (b[:, :, 2] + 256.0 * b[:, :, 1] + 65536.0 * b[:, :, 0]) / (2 ** 24 - 1)
                    t = tags.ravel()[g['pix']]
                    rid = ids.ravel()[g['pix']]
                    stat = np.isin(t, STATIC_OCCLUDER_TAGS) | (np.isin(t, ROAD_USER_TAGS) &
                                                              ~np.isin(rid, list(by16)))
                    dd = depth.ravel()[g['pix']]
                    stat &= dd < c['range'] * 2.0
                    if stat.any():
                        P = g['origin'] + g['dirs_station'][stat] * dd[stat, None]
                        occ.append(P)
            sectors_of[c['sid']] = self._sectors(c, np.concatenate(occ) if occ else np.zeros((0, 3)))
        for x in self.lids:
            if not x['window']:
                continue
            idx = np.concatenate([w_[2] for w_ in x['window']])
            u, n = np.unique(idx[idx > 0], return_counts=True)
            for aid, k in zip(u.tolist(), n.tolist()):
                if k >= int(cfg['min_points']) and aid in actors:
                    evidence.setdefault(aid, {})[x['sid']] = evidence.get(aid, {}).get(x['sid'], 0) + int(k)
            if cfg.get('free_space', True):
                xyz = np.concatenate([w_[0] for w_ in x['window']])
                tag = np.concatenate([w_[1] for w_ in x['window']])
                P = xyz @ x['M'][:3, :3].T + x['M'][:3, 3]                  # world
                stat = np.isin(tag, STATIC_OCCLUDER_TAGS) | (np.isin(tag, ROAD_USER_TAGS) & (idx == 0))
                stat &= P[:, 2] > self.anchor.z + GROUND_CLEARANCE_M
                Ps = (P[stat] - self.T[:3, 3]) @ self.T[:3, :3]             # station frame
                sectors_of[x['sid']] = self._sectors(x, Ps)
        return self.message(i, frame, sim_t, actors, evidence, sectors_of, director)

    def _sectors(self, sen: dict, P_station: np.ndarray) -> list:
        """[(range_m, start_wgs_deg, end_wgs_deg)] of a sensor's field of view in cfg sectors, each cut at its nearest
        occluder point (station frame) — the declared range when none."""
        s = sen['s']
        p = s['pose']
        hfov = sen.get('hfov') or float(s.get('horizontal_fov', 360.0))
        n = max(int(self.cfg['sectors']), 1)
        hdg = C.wgs84_heading(float(self.rsu.get('yaw', 0.0)) + float(p.get('yaw', 0.0)))
        start = hdg - hfov / 2.0 if hfov < 359.9 else 0.0
        width = min(hfov, 360.0) / n
        rng = np.full(n, sen['range'])
        if len(P_station):
            rel = P_station[:, :2] - np.array([p['x'], p['y']])
            a = math.radians(float(self.rsu.get('yaw', 0.0)))
            c_, s_ = math.cos(a), math.sin(a)
            east = c_ * rel[:, 0] - s_ * rel[:, 1]
            south = s_ * rel[:, 0] + c_ * rel[:, 1]
            bearing = np.degrees(np.arctan2(east, -south)) % 360.0      # clockwise from North
            dist = np.hypot(east, south)
            k = np.floor(((bearing - start) % 360.0) / width).astype(int)
            ok = (k >= 0) & (k < n) & (dist <= sen['range'])
            if ok.any():
                np.minimum.at(rng, k[ok], dist[ok])
        return [(float(rng[j]), start + j * width, start + (j + 1) * width) for j in range(n)]

    def message(self, i, frame, sim_t, actors, evidence, sectors_of, director=None) -> dict:
        cfg = self.cfg
        from ..core.groundtruth import category_of
        snap = self.world.get_snapshot()
        noise = cfg['noise']
        sp, sv = float(noise.get('position_m', 0.0)), float(noise.get('velocity_mps', 0.0))
        rng = np.random.default_rng([self.rng_seed & 0xFFFFFFFF, self.station_id, i])
        rows = []
        for aid in sorted(evidence):
            a = actors[aid]
            sa = snap.find(aid)
            t = sa.get_transform() if sa is not None else a.get_transform()
            v = sa.get_velocity() if sa is not None else a.get_velocity()
            if director is not None and getattr(director, 'kinematic', False) and director.k_vel(a) is not None:
                v = director.k_vel(a)
            wz = (sa.get_angular_velocity() if sa is not None else a.get_angular_velocity()).z
            bb = a.bounding_box
            M = _matrix({'x': t.location.x, 'y': t.location.y, 'z': t.location.z, 'roll': t.rotation.roll,
                         'pitch': t.rotation.pitch, 'yaw': t.rotation.yaw})
            ctr = M[:3, :3] @ np.array([bb.location.x, bb.location.y, bb.location.z]) + M[:3, 3]
            pos = V.to_enu(self.rsu, ctr, self.ground_z)[0]
            vel = np.array([v.x, -v.y])
            if sp > 0:
                pos = pos + rng.normal(0.0, sp, 3)
            if sv > 0:
                vel = vel + rng.normal(0.0, sv, 2)
            dist = float(math.hypot(pos[0], pos[1]))
            prev = self.last_seen.get(aid)
            if prev is None or sim_t - prev > TRACK_GAP_S:
                self.first_seen[aid] = sim_t
            self.last_seen[aid] = sim_t
            ev = evidence[aid]
            px = sum(n for s, n in ev.items() if s in {c['sid'] for c in self.cams})
            pts = sum(n for s, n in ev.items() if s in {x['sid'] for x in self.lids})
            q = min(1.0, max(px / QUALITY_FULL['pixels'], pts / QUALITY_FULL['points']))
            cat = category_of(a.type_id, dict(getattr(a, 'attributes', {}) or {}))
            rows.append({'object_id': aid, 'pos': tuple(pos), 'vel': tuple(vel), 'yaw_enu_deg': C.enu_angle(t.rotation.yaw),
                         'yaw_rate_dps': -float(wz), 'size': (2 * bb.extent.x, 2 * bb.extent.y, 2 * bb.extent.z),
                         'age_ms': int(round((sim_t - self.first_seen[aid]) * 1000)), 'quality': int(round(15 * q)),
                         'sensor_ids': sorted(ev), 'category': cat, 'sigma_pos_m': sp, 'sigma_vel_mps': sv,
                         '_dist': dist, '_type': a.type_id, '_ev': ev})
        rows.sort(key=lambda r: r['_dist'])
        rows = rows[:int(cfg['max_objects'])]
        objs = [C.perceived_object(r) for r in rows]
        sensors, regions = [], []
        for sen in self.cams + self.lids:
            s = sen['s']
            p = s['pose']
            off = V.to_enu(self.rsu, self._world_xyz(p), self.ground_z)[0]
            hdg = C.wgs84_heading(float(self.rsu.get('yaw', 0.0)) + float(p.get('yaw', 0.0)))
            if s['type'].startswith('camera.'):
                sensors.append(C.sensor_information(sen['sid'], 'camera', hdg, sen['hfov'], sen['range'],
                                                    float(p.get('pitch', 0.0)), sen['vfov'], off))
            else:
                sensors.append(C.sensor_information(sen['sid'], 'lidar', hdg, float(s.get('horizontal_fov', 360.0)),
                                                    sen['range'], 0.0, float(s['upper_fov']) - float(s['lower_fov']),
                                                    off))
                sensors[-1]['perceptionRegionShape']['radial']['verticalOpeningAngleStart'] = \
                    C.deci_deg(float(p.get('pitch', 0.0)) + float(s['lower_fov']))
                sensors[-1]['perceptionRegionShape']['radial']['verticalOpeningAngleEnd'] = \
                    C.deci_deg(float(p.get('pitch', 0.0)) + float(s['upper_fov']))
            if sen['sid'] in sectors_of:
                inside = [r['object_id'] for r in rows if sen['sid'] in r['_ev']]
                regions.append(C.perception_region(sen['sid'], off, sectors_of[sen['sid']], inside))
        ref_ms = self.t0_ms + int(round(sim_t * 1000.0))
        msg = C.cpm_message(self.station_id, ref_ms, self.geo[0], self.geo[1], self.geo[2], sensors, regions, objs,
                            rate=float(cfg['rate_hz']))
        data = C.encode(msg)
        self.sent += 1
        self.bytes.append(len(data))
        self.objects_sent += len(objs)
        return {'step': i, 'frame': int(frame), 'sim_t': sim_t, 'station': self.id, 'station_id': self.station_id,
                'cpm': msg, 'uper': C.b64(data), 'bytes': len(data),
                'objects': [{'objectId': o['objectId'], 'actor_id': r['object_id'], 'type_id': r['_type'],
                             'category': r['category'], 'distance_m': round(r['_dist'], 3),
                             'evidence': {str(k): n for k, n in sorted(r['_ev'].items())}}
                            for o, r in zip(objs, rows)]}

    def _world_xyz(self, p):
        q = V.to_world(self.rsu, p, self.ground_z)
        return np.array([q['x'], q['y'], q['z']])

    # ------------------------------------------------------------------ documents
    def doc(self) -> dict:
        """The station in dataset.json and run.json: its pose, geo-reference, sensors (station-frame poses)."""
        return {'id': self.id, 'name': self.rsu.get('name', self.id), 'station_id': self.station_id,
                'x': self.anchor.x, 'y': self.anchor.y, 'z': self.anchor.z, 'yaw': self.anchor.yaw,
                'height_m': float(self.rsu.get('height_m', V.HEIGHT_M)),
                'reference_position': {'latitude': self.geo[0], 'longitude': self.geo[1], 'altitude': self.geo[2]},
                'cpm': self.cfg,
                'sensors': [{'id': c['s']['id'], 'cpm_sensor_id': c['sid'], 'type': c['s']['type'],
                             'pose_station': c['s']['pose']} for c in self.cams + self.lids]}

    def report(self) -> dict:
        b = np.asarray(self.bytes, float)
        return {'id': self.id, 'station_id': self.station_id, 'cpms': self.sent, 'objects_reported': self.objects_sent,
                'bytes': {'mean': round(float(b.mean()), 1), 'max': int(b.max())} if len(b) else None,
                'every_steps': self.every}


C_SEM = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('cos', '<f4'), ('idx', '<u4'), ('tag', '<u4')])


def stations(w, world, lib, cfg: dict, cameras, raster=None, record=True) -> list:
    """The config's RSUs (core.v2x.attach) as stations, or [] when it has none."""
    inf = cfg.get(V.KEY) or {}
    out = [Station(w, world, lib, r, cameras, raster, record, float(cfg.get('fps', 20.0)), int(cfg.get('seed', 0)))
           for r in inf.get('rsus') or []]
    if out:
        w.emit.log(f'{len(out)} roadside unit{"s" if len(out) != 1 else ""}: '
                   + '; '.join(f'{s.id} ({len(s.cams)} camera{"s" if len(s.cams) != 1 else ""}, {len(s.lids)} '
                               f'lidar{"s" if len(s.lids) != 1 else ""}, a CPM every {s.every} steps)' for s in out))
    return out


class CpmLog:
    """cpm.jsonl: one line per CPM sent (core.v2x.benefit and falcon v2x decode read it)."""

    def __init__(self, path: str):
        self.fh = open(path, 'w', encoding='utf-8')

    def write(self, line: dict):
        self.fh.write(json.dumps(line, separators=(',', ':')) + '\n')

    def close(self):
        self.fh.close()
