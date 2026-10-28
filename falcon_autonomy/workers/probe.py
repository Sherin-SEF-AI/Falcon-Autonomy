"""probe: what server answers at host:port — versions, the open map, the maps it has, its mode. Changes nothing.

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
        self.alive()
        self.world = None                    # a probe spawned nothing and must not change the server's mode
        self.emit.send({'t': 'server', **info})
        return {'summary': info}


def main(argv) -> int:
    ap = argparse.ArgumentParser(prog='falcon worker probe')
    add_server_args(ap)
    ap.add_argument('--wait', type=float, default=10.0, help='wait this long for a server that is starting, s')
    a = ap.parse_args(argv)
    return Probe(a, argv).execute()
