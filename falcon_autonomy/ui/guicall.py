"""Run a function on the GUI thread from any thread and wait for its result (the assistant's read tools that look at the
window run off the GUI thread; widgets may only be touched on it).

call(fn, timeout) on the GUI thread just calls fn. From another thread it hands fn to a small QObject living on the
GUI thread through a queued signal and waits for the result (or the exception, re-raised here). The helper object is
made on first use in the calling thread and moved to the application's thread (moveToThread may push an object from
its own thread), so the window needs no setup.
"""
from __future__ import annotations

import threading

from PyQt6.QtCore import QCoreApplication, QObject, QThread, pyqtSignal, pyqtSlot

_LOCK = threading.Lock()
_RUNNER = None


class _Runner(QObject):
    run = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.run.connect(self._exec)

    @pyqtSlot(object)
    def _exec(self, box):
        try:
            box['result'] = box['fn']()
        except BaseException as e:            # noqa: BLE001 (re-raised in the waiting thread)
            box['error'] = e
        finally:
            box['done'].set()


def _runner() -> _Runner:
    global _RUNNER
    with _LOCK:
        if _RUNNER is None:
            r = _Runner()
            app = QCoreApplication.instance()
            if app is not None and r.thread() is not app.thread():
                r.moveToThread(app.thread())
            _RUNNER = r
        return _RUNNER


def call(fn, timeout: float = 10.0):
    app = QCoreApplication.instance()
    if app is None or QThread.currentThread() is app.thread():
        return fn()
    box = {'fn': fn, 'done': threading.Event()}
    _runner().run.emit(box)
    if not box['done'].wait(timeout):
        raise TimeoutError('the window did not answer in time (busy)')
    if 'error' in box:
        raise box['error']
    return box.get('result')
