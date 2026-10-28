"""What every worker shares: the watchdog, cancel handling, run.json, connecting to CARLA, tidying up.

Lessons applied (CARLA 0.9.16):

* Signals: SIGINT / SIGTERM are blocked before numpy is imported (__main__) and taken only by the watchdog thread
  (sigtimedwait), which sets the cancel flag even while the main thread is blocked in a call into CARLA's client. The
  main thread then tidies up; if it does not finish within SIGNAL_GRACE_S (a second signal: at once), the watchdog
  writes run.json and ends the process with CANCELLED.
* A server that crashed leaves CARLA's client hanging forever in its call. The main thread reports a sign of life
  after every call that returned (alive()); while it talks to the server, no sign for longer than the expected limit
  (twice the client timeout, or what expect() set for a slow call such as loading a map) makes the watchdog write
  run.json saying why and exit with SERVER_LOST.
* Tidy-up in all cases: stop and destroy every actor this worker spawned (sensors first), switch the traffic manager
  and the world back to asynchronous mode (variable step).

Role names: everything a worker spawns carries a role name starting with ROLE_PREFIX (the ego: CARLA's 'hero', with
Falcon sensors on it), so `workers tidy` can remove exactly that after a worker was killed.
"""
from __future__ import annotations

import os
import shutil
import signal
import sys
import threading
import time
import traceback

from ..core import protocol as P
from ..core.provenance import new_run, utc_now, write_json

SIGNAL_GRACE_S = 15.0
ROLE_PREFIX = 'falcon_'
ROLE_SENSOR, ROLE_TRAFFIC, ROLE_WALKER, ROLE_PROP = 'falcon_sensor', 'falcon_traffic', 'falcon_walker', 'falcon_prop'
ROLE_HERO = 'hero'


class Cancelled(Exception):
    pass


class Failed(Exception):
    """An expected failure: the message goes to run.json "error", the exit code is FAILED (or code)."""

    def __init__(self, msg, code=P.FAILED):
        super().__init__(msg)
        self.code = code


class Worker:
    """Base of a worker run. Subclasses implement body(); execute() runs it with the watchdog and maps outcomes to
    exit codes and run.json."""

    name = 'worker'

    def __init__(self, argv: list, out_dir: str = '', timeout: float = 20.0, seed=None, settings=None, inputs=None):
        self.argv = list(argv)
        self.emit = P.Emitter()
        self.timeout = float(timeout)
        self.out_dir = os.path.abspath(out_dir) if out_dir else ''
        self.cancel = threading.Event()
        self.cancel_at = None
        self.life = {'phase': 'setup', 't': time.monotonic(), 'limit': 2.0 * max(self.timeout, 1.0)}
        self.run = new_run(self.name, argv, seed, settings, inputs)
        self.notes: list = []
        self.spawned: list = []            # actors to destroy at tidy-up, in spawn order
        self.client = self.world = self.tm = None
        self.carla = None
        self.remove_outputs_on_cancel = False
        self._done = False

    # ---------------------------------------------------------------- life and cancel
    def alive(self):
        self.life['t'] = time.monotonic()

    def expect(self, seconds: float):
        """The next sign of life may take this long (a slow call: loading a map); alive() resets nothing else."""
        self.life['limit'] = max(float(seconds), 1.0) * 1.5
        self.life['t'] = time.monotonic()

    def phase(self, name: str):
        self.life['phase'] = name
        self.life['limit'] = 2.0 * max(self.timeout, 1.0)
        self.alive()

    def check(self):
        if self.cancel.is_set():
            raise Cancelled()

    def note(self, text: str):
        self.notes.append(text)
        self.emit.log(text, 'warning')

    def _on_command(self, cmd: dict):
        if cmd.get('cmd') == 'cancel':
            self._request_cancel()
        else:
            self.on_command(cmd)

    def on_command(self, cmd: dict):
        """Other stdin commands (live sessions override this)."""

    def _request_cancel(self):
        now = time.monotonic()
        if self.cancel.is_set():
            self.cancel_at = now - SIGNAL_GRACE_S - 1.0     # a second cancel: no more waiting
        else:
            self.cancel_at = now
            self.cancel.set()

    def _watchdog(self):
        sigs = {signal.SIGINT, signal.SIGTERM}
        waited = hasattr(signal, 'sigtimedwait')
        while not self._done:
            got = signal.sigtimedwait(sigs, 0.5) if waited else time.sleep(0.5)
            if got is not None:
                self._request_cancel()
            now = time.monotonic()
            why = None
            if self.cancel_at is not None and now - self.cancel_at > SIGNAL_GRACE_S:
                why = (P.CANCELLED, 'cancelled; the CARLA server did not answer while tidying up, so it may be left in '
                                    'synchronous mode with this run\'s actors (falcon tidy --port P)')
            elif self.life['phase'] in ('setup', 'run', 'tidy') and now - self.life['t'] > self.life['limit']:
                why = (P.SERVER_LOST, f'the CARLA server stopped answering (no reply for {self.life["limit"]:.0f} s '
                                      f'during {self.life["phase"]}): the run was aborted; the server may be left in '
                                      'synchronous mode with its actors (falcon tidy --port P)')
            if why is None:
                continue
            code, msg = why
            try:
                if code == P.CANCELLED and self.remove_outputs_on_cancel:
                    self.remove_outputs()
                self.finish_run(code, msg)
                self.emit.result(P.status_of(code), error=msg, run_json=self.run_json_path)
            except Exception:                              # noqa: BLE001 (ending regardless)
                pass
            print(f'{P.status_of(code)}: {msg}', file=sys.stderr, flush=True)
            sys.stdout.flush()
            os._exit(code)

    # ---------------------------------------------------------------- run.json
    @property
    def run_json_path(self) -> str:
        return os.path.join(self.out_dir, 'run.json') if self.out_dir else ''

    def finish_run(self, code: int, error=None, **extra):
        self.run.update(status=P.status_of(code), exit_code=code, finished=utc_now(), error=error,
                        notes=self.notes, **extra)
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            write_json(self.run_json_path, self.run)

    def remove_outputs(self):
        """A cancelled or failed run leaves only its run.json (never half a dataset)."""
        if not self.out_dir or not os.path.isdir(self.out_dir):
            return
        for f in os.listdir(self.out_dir):
            if f == 'run.json':
                continue
            p = os.path.join(self.out_dir, f)
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)

    # ---------------------------------------------------------------- CARLA
    def import_carla(self):
        try:
            import carla
        except Exception as e:                             # noqa: BLE001
            raise Failed(f'could not import carla in {sys.executable}: {e}', P.NO_CARLA) from e
        self.carla = carla
        return carla

    def connect(self, host: str, port: int, wait_s: float = 60.0):
        """Client and world of the server at host:port; waits up to wait_s for a server that is still starting.
        Raises Failed(NO_SERVER) when none answers, Failed(NO_CARLA) when the client's version is not the
        server's."""
        carla = self.carla or self.import_carla()
        t0 = time.monotonic()
        last = None
        self.expect(wait_s + self.timeout)
        while time.monotonic() - t0 < wait_s:
            self.check()
            try:
                c = carla.Client(host, int(port))
                c.set_timeout(min(self.timeout, 10.0))
                sv = c.get_server_version()
                c.set_timeout(self.timeout)
                world = c.get_world()
                self.alive()
                cv = c.get_client_version()
                if sv.split('-')[0] != cv.split('-')[0]:
                    raise Failed(f'CARLA\'s client is {cv} but the server is {sv}: install the matching carla wheel '
                                 'in CARLA\'s Python', P.NO_CARLA)
                self.client, self.world = c, world
                self.run['provenance']['host'].update(carla_server=sv, carla_client=cv)
                self.phase(self.life['phase'])
                return c, world
            except Failed:
                raise
            except RuntimeError as e:                     # time-out: not up yet
                last = e
                time.sleep(1.0)
        raise Failed(f'no CARLA server answered at {host}:{port} within {wait_s:.0f} s ({last})', P.NO_SERVER)

    def own(self, actor):
        if actor is not None:
            self.spawned.append(actor)
        return actor

    def tidy(self):
        """Stop and destroy what this worker spawned (sensors first), traffic manager and world back to asynchronous
        mode. Every step is tried even when an earlier one fails."""
        if self.world is None:
            return
        self.phase('tidy')

        def quiet(fn, *a):
            try:
                r = fn(*a)
                self.alive()
                return r
            except Exception as e:                         # noqa: BLE001
                return e
        sensors = [a for a in self.spawned if a.type_id.startswith('sensor.')]
        walkers_ai = [a for a in self.spawned if a.type_id.startswith('controller.')]
        rest = [a for a in self.spawned if a not in sensors and a not in walkers_ai]
        for a in walkers_ai:
            quiet(a.stop)
        for a in rest:
            if a.type_id.startswith('vehicle.'):
                quiet(a.set_autopilot, False)
        for s in sensors:
            quiet(s.stop)
        st = quiet(self.world.get_settings)
        sync = not isinstance(st, Exception) and bool(st.synchronous_mode)
        if sync:
            # one step so that the traffic manager takes its vehicles off the autopilot and no sensor data is in
            # flight; a step *after* the vehicles are destroyed makes the traffic manager (its thread, in this
            # process) operate on destroyed actors and libcarla std::terminate the process (measured on 0.9.16)
            quiet(self.world.tick)
        # children before parents (destroying the ego before the sensors on it fails the same way); the sensors by
        # Actor.destroy(), everything else by id in a batch: Actor.destroy() marks the client's actor object destroyed
        # while the traffic manager's thread still holds it, and its next update std::terminates the process
        for a in sensors:
            quiet(a.destroy)
        cmd = self.carla.command
        res = quiet(self.client.apply_batch_sync, [cmd.DestroyActor(a.id) for a in walkers_ai + list(reversed(rest))])
        if isinstance(res, Exception):
            self.note(f'destroying this run\'s actors failed: {res}')
        self.spawned.clear()
        if self.tm is not None:
            quiet(self.tm.set_synchronous_mode, False)
        if not isinstance(st, Exception):
            st.synchronous_mode = False
            st.fixed_delta_seconds = None
            quiet(self.world.apply_settings, st)

    # ---------------------------------------------------------------- running
    def body(self) -> dict:
        raise NotImplementedError

    def execute(self) -> int:
        self.emit.hello(self.name)
        P.CommandReader(self._on_command).start()
        threading.Thread(target=self._watchdog, daemon=True, name='falcon-watchdog').start()
        code, err, extra = P.OK, None, {}
        try:
            extra = self.body() or {}
        except Cancelled:
            code, err = P.CANCELLED, 'cancelled by the user'
        except Failed as e:
            code, err = e.code, str(e)
        except RuntimeError as e:
            # CARLA's client raises RuntimeError on a time-out: after a successful connect it means the server stopped
            # answering in time
            code = P.SERVER_LOST if self.client is not None and 'time-out' in str(e) else P.CRASHED
            err = f'{type(e).__name__}: {e}'
            if code == P.CRASHED:
                self.run['traceback'] = traceback.format_exc()
        except Exception as e:                             # noqa: BLE001 (a bug: reported in run.json)
            code, err = P.CRASHED, f'{type(e).__name__}: {e}'
            self.run['traceback'] = traceback.format_exc()
        if code != P.SERVER_LOST:
            try:
                self.tidy()
            except Exception as e:                         # noqa: BLE001
                self.note(f'tidy-up failed: {type(e).__name__}: {e}')
        if code != P.OK and self.remove_outputs_on_cancel:
            self.remove_outputs()
        self.finish_run(code, err, **extra)
        self.life['phase'] = 'done'
        self._done = True
        self.emit.result(P.status_of(code), error=err, run_json=self.run_json_path, **{
            k: v for k, v in extra.items() if k in ('summary',)})
        if err:
            print(f'{P.status_of(code)}: {err}', file=sys.stderr, flush=True)
        return code


def role(actor) -> str:
    try:
        return str((getattr(actor, 'attributes', None) or {}).get('role_name', ''))
    except Exception:                                      # noqa: BLE001
        return ''


def add_server_args(ap, port_default=None):
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, required=port_default is None, default=port_default,
                    help='the CARLA server\'s RPC port')
    ap.add_argument('--timeout', type=float, default=20.0, help='CARLA client time-out per call, s')
