"""The GUI's side of the job queue: QProcess launches workers on the event loop (no threads), and JobsBridge turns
queue events into Qt signals. The queue, specs, worker commands and protocol are the CLI's (core.jobs)."""
from __future__ import annotations

import os
import signal

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from ..core import protocol as P
from ..core.jobs import JobQueue, Launcher


class _QHandle:
    def __init__(self, proc: QProcess):
        self.proc = proc

    def send(self, cmd: dict) -> bool:
        import json
        if self.proc.state() != QProcess.ProcessState.Running:
            return False
        return self.proc.write((json.dumps(cmd) + '\n').encode()) > 0

    def cancel(self):
        if self.proc.state() == QProcess.ProcessState.NotRunning:
            return
        self.proc.write(b'{"cmd": "cancel"}\n')
        self.proc.terminate()                    # SIGTERM: the worker's watchdog takes it

    def kill(self):
        self.proc.kill()                         # SIGKILL (systemd-run --scope execs into the worker: same pid)


class QProcessLauncher(Launcher):
    def __init__(self, parent: QObject):
        self.parent = parent
        self.procs: dict = {}

    def start(self, job, queue: JobQueue):
        s = job.spec
        proc = QProcess(self.parent)
        env = QProcessEnvironment()
        for k, v in (s.env if s.env is not None else os.environ).items():
            env.insert(k, v)
        proc.setProcessEnvironment(env)
        if s.cwd:
            proc.setWorkingDirectory(s.cwd)
        buf = {'out': b'', 'err': b''}

        def read(which):
            data = proc.readAllStandardOutput() if which == 'out' else proc.readAllStandardError()
            buf[which] += bytes(data)
            *lines, buf[which] = buf[which].split(b'\n')
            for raw in lines:
                text = raw.decode('utf-8', 'replace')
                if which == 'out':
                    queue.on_message(job, P.parse_line(text))
                elif text.strip():
                    queue.on_message(job, {'t': 'log', 'level': 'stderr', 'text': text})

        def finished(code, status):
            read('out')
            read('err')
            for which in ('out', 'err'):
                if buf[which].strip():
                    rest = buf[which].decode('utf-8', 'replace')
                    queue.on_message(job, P.parse_line(rest) if which == 'out' else
                                     {'t': 'log', 'level': 'stderr', 'text': rest})
            if status == QProcess.ExitStatus.CrashExit:
                code = -signal.SIGKILL if job.cancel_requested else -abs(code or signal.SIGABRT)
            self.procs.pop(job.id, None)
            queue.on_exit(job, code)
            proc.deleteLater()

        def failed(err):
            if err == QProcess.ProcessError.FailedToStart:
                job.log.append(('error', f'could not start {s.argv[0]}: {proc.errorString()}'))
                self.procs.pop(job.id, None)
                queue.on_exit(job, P.FAILED)
        proc.readyReadStandardOutput.connect(lambda: read('out'))
        proc.readyReadStandardError.connect(lambda: read('err'))
        proc.finished.connect(finished)
        proc.errorOccurred.connect(failed)
        proc.start(s.argv[0], s.argv[1:])
        self.procs[job.id] = proc
        return _QHandle(proc)

    def running(self) -> int:
        return sum(1 for p in self.procs.values() if p.state() != QProcess.ProcessState.NotRunning)


class JobsBridge(QObject):
    """Qt signals for the queue's events, and the timer that polls it (memory and server waits clear over time)."""
    added = pyqtSignal(object)
    changed = pyqtSignal(object)
    finished = pyqtSignal(object)
    message = pyqtSignal(object)

    def __init__(self, queue: JobQueue, parent=None, interval_ms=1000):
        super().__init__(parent)
        self.queue = queue
        queue.subscribe(self._on)
        self.timer = QTimer(self)
        self.timer.timeout.connect(queue.poll)
        self.timer.start(interval_ms)

    def _on(self, ev, job):
        {'added': self.added, 'changed': self.changed, 'finished': self.finished,
         'message': self.message}[ev].emit(job)
