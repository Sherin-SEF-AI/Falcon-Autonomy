"""tidy: put a server back after a worker was killed (SIGKILL, its memory cap) and could not tidy up itself.

    python -m falcon_autonomy.workers tidy --port 2310

Stops and destroys what a Falcon worker spawned and nothing else — sensors named falcon_sensor, vehicles, walkers and
props whose role name starts with falcon_, a 'hero' only when it carries a Falcon sensor, the AI controllers of those
walkers — and puts the world and traffic manager back in asynchronous mode (variable step).
"""
from __future__ import annotations

import argparse

from .common import ROLE_HERO, ROLE_PREFIX, ROLE_SENSOR, Worker, add_server_args, role


def tidy_server(client, world) -> dict:
    # a client knows the actors from the snapshots it is sent, and a synchronous server nobody ticks sends none:
    # measured on 0.9.16, a fresh client listed 0 of the 51 actors a killed run left. One step first.
    if world.get_settings().synchronous_mode:
        world.tick()
    else:
        world.wait_for_tick()
    acts = list(world.get_actors())

    def quiet(fn, *args):
        try:
            return fn(*args)
        except Exception:                                  # noqa: BLE001
            return None
    sensors = [a for a in acts if a.type_id.startswith('sensor.') and role(a) == ROLE_SENSOR]
    egos = {a.parent.id for a in sensors if getattr(a, 'parent', None) is not None and role(a.parent) == ROLE_HERO}
    mine = [a for a in acts if not a.type_id.startswith(('sensor.', 'controller.')) and
            (role(a).startswith(ROLE_PREFIX) or a.id in egos)]
    walkers = {a.id for a in mine if a.type_id.startswith('walker.')}
    ctrls = [a for a in acts if a.type_id.startswith('controller.') and getattr(a, 'parent', None) is not None and
             a.parent.id in walkers]
    sync = bool(world.get_settings().synchronous_mode)
    for c in ctrls:
        quiet(c.stop)
    for a in mine:
        if a.type_id.startswith('vehicle.'):
            quiet(a.set_autopilot, False)
    for s in sensors:
        quiet(s.stop)
    if sync:
        quiet(world.tick)
    gone = sum(1 for a in ctrls + sensors if quiet(a.destroy))
    import carla                                       # vehicles by id (see common.Worker.tidy)
    res = quiet(client.apply_batch_sync, [carla.command.DestroyActor(a.id) for a in mine]) or []
    gone += sum(1 for r in res if not r.error)
    if sync:
        quiet(world.tick)
    st = world.get_settings()
    st.synchronous_mode = False
    st.fixed_delta_seconds = None
    if getattr(st, 'no_rendering_mode', False):
        st.no_rendering_mode = False                   # only the gym worker's no_rendering option switches it on
    world.apply_settings(st)
    return {'destroyed': gone, 'sensors': len(sensors), 'controllers': len(ctrls),
            'vehicles': sum(1 for a in mine if a.type_id.startswith('vehicle.')), 'walkers': len(walkers),
            'props': sum(1 for a in mine if a.type_id.startswith('static.')), 'was_synchronous': sync}


class Tidy(Worker):
    name = 'tidy'

    def __init__(self, a, argv):
        super().__init__(argv, '', a.timeout)
        self.a = a

    def body(self):
        _c, world = self.connect(self.a.host, self.a.port, 5.0)
        self.phase('tidy')
        res = tidy_server(self.client, world)
        # no traffic manager is touched: one lives in the process that created it (the killed worker, gone with it),
        # and asking for one on a port where none runs would start a new one in this process
        self.world = None                                  # done here; the base class has nothing more to tidy
        self.emit.log(f'tidied: {res}')
        return {'summary': res}


def main(argv) -> int:
    ap = argparse.ArgumentParser(prog='falcon worker tidy')
    add_server_args(ap)
    a = ap.parse_args(argv)
    return Tidy(a, argv).execute()
