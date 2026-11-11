"""probe: what server answers at host:port — versions, the open map, the maps it has, its mode, and what its backend
has (`falcon backend probe` keeps it: sim.backends): blueprint ids, CityObjectLabel names and values, weather fields,
camera attributes, the client module's API capabilities. Changes nothing.

    python -m falcon_autonomy.workers probe --port 2310 [--wait 60]

Prints one result line {"t": "result", "status": "ok", "server": {...}}; exits OK, NO_SERVER or NO_CARLA.
"""
from __future__ import annotations

import argparse

from .common import Worker, add_server_args


class Probe(Worker):
    name = 'probe'

    def __init__(self, a, argv):
        super().__init__(argv, '', a.timeout)
        self.a = a

    def body(self):
        _client, world = self.connect(self.a.host, self.a.port, self.a.wait)
        c = self.client
        st = world.get_settings()
        info = {
            'server_version': c.get_server_version(), 'client_version': c.get_client_version(),
            'map': world.get_map().name.rpartition('/')[2],
            'maps': sorted({m.rpartition('/')[2] for m in c.get_available_maps()}),
            'synchronous_mode': bool(st.synchronous_mode), 'fixed_delta_seconds': st.fixed_delta_seconds,
            'actors': len(world.get_actors()), 'blueprints': len(world.get_blueprint_library()),
        }
        info.update(self.backend_info(world))
        self.alive()
        self.world = None                    # a probe spawned nothing and must not change the server's mode
        self.emit.send({'t': 'server', **info})
        return {'summary': info}

    def backend_info(self, world) -> dict:
        from ..core.groundtruth import labels_of
        from ..sim.backends import CAPS
        carla = self.carla
        lib = world.get_blueprint_library()
        ids = sorted(b.id for b in lib)
        cam = {}
        for k in ('sensor.camera.rgb', 'sensor.camera.instance_segmentation', 'sensor.lidar.ray_cast_semantic'):
            try:
                cam[k] = sorted(a.id for a in lib.find(k))
            except (IndexError, RuntimeError):
                cam[k] = None
        wp = carla.WeatherParameters()
        weather = sorted(k for k in dir(wp) if not k.startswith('_') and isinstance(getattr(wp, k, None), float))
        caps = {c: self.compat.has(c) for c in CAPS if c != 'opendrive_world'}
        return {'backend': self.compat.backend_id, 'blueprint_count': len(ids), 'blueprints': ids,
                'labels': labels_of(getattr(carla, 'CityObjectLabel', None)), 'weather_fields': weather,
                'sensor_attributes': cam, 'caps': caps}


def main(argv) -> int:
    ap = argparse.ArgumentParser(prog='falcon worker probe')
    add_server_args(ap)
    ap.add_argument('--wait', type=float, default=10.0, help='wait this long for a server that is starting, s')
    a = ap.parse_args(argv)
    return Probe(a, argv).execute()
