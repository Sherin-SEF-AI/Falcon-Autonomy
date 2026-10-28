"""Short background calls (a check, reading a run's files) off the GUI thread, delivered back safely.

run(owner, fn, on_done, on_error) runs fn() on Qt's thread pool and calls on_done(result) on the GUI thread — but only
while owner (a QObject) is alive: an analysis thread emitting into a deleted window segfaults. Results travel through
a carrier QObject that lives on the GUI thread (never the owner itself), and every task is tracked so that wait_all()
can block until all have finished (windows call it before they close; tests call it before teardown).
Long actions are jobs (falcon_autonomy.core.jobs), not tasks.
"""
from __future__ import annotations

import threading
import traceback

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

_lock = threading.Lock()
_running: set = set()
_idle = threading.Condition(_lock)


class _Carrier(QObject):
    done = pyqtSignal(object, object)          # (result, error)


class _Task(QRunnable):
    def __init__(self, fn, carrier):
        super().__init__()
        self.fn, self.carrier = fn, carrier
        self.setAutoDelete(True)

    def run(self):
        res = err = None
        try:
            res = self.fn()
        except Exception as e:                  # noqa: BLE001 (delivered to on_error)
            err = (e, traceback.format_exc())
        try:
            self.carrier.done.emit(res, err)
        finally:
            with _lock:
                _running.discard(self)
                _idle.notify_all()


def _alive(obj) -> bool:
    try:
        from PyQt6 import sip
        return obj is not None and not sip.isdeleted(obj)
    except ImportError:
        return obj is not None


def run(owner: QObject, fn, on_done=None, on_error=None):
    carrier = _Carrier()                       # created here, on the GUI thread: its slots run there

    def deliver(res, err):
        carrier.deleteLater()
        if not _alive(owner):
            return
        if err is not None:
            if on_error is not None:
                on_error(err[0])
            else:
                print(err[1])
        elif on_done is not None:
            on_done(res)
    carrier.done.connect(deliver)
    t = _Task(fn, carrier)
    with _lock:
        _running.add(t)
    QThreadPool.globalInstance().start(t)
    return t


def pending() -> int:
    with _lock:
        return len(_running)


def wait_all(timeout: float = 30.0) -> bool:
    """Block until every task has finished (their results are then delivered by the event loop). False on timeout."""
    with _idle:
        return _idle.wait_for(lambda: not _running, timeout)
