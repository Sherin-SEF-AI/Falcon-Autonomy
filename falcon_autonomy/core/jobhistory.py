"""The job history: every finished job of a workspace, kept across restarts (standard library only).

A job is recorded when it ends (done, failed or cancelled) as one line of WORKSPACE/jobs/jobs.jsonl:

    {"schema": "falcon.job-history/1", "uid", "kind", "title", "submitted", "started", "finished" (UTC ISO),
     "elapsed_s", "status", "exit_code", "exit" (what the exit code means), "result" (the run folder or result file,
     when the job names one), "page" (the page it was started from), "error" (its last error line, when it failed),
     "log" (the per-job log file, relative to the jobs folder)}

and its whole log (every line the worker reported, with its level) is written to WORKSPACE/jobs/logs/<uid>.log. The
history holds what the job queue knew; nothing is invented for jobs that ran before it existed. Reading tolerates a
damaged line (it is skipped) and the file is trimmed to the newest KEEP entries when it grows past twice that.
"""
from __future__ import annotations

import json
import os
import time

from . import protocol as P

SCHEMA = 'falcon.job-history/1'
KEEP = 2000


def _iso(t):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t)) if t else None


def result_path(job) -> str:
    """Where a job's result is: its run folder, or a path its result message names."""
    s = job.spec
    if s.run_dir:
        return s.run_dir
    r = job.result or {}
    for k in ('path', 'out', 'dir', 'run_dir', 'report'):
        v = r.get(k)
        if isinstance(v, str) and v:
            return v
    return ''


class JobHistory:
    def __init__(self, workspace: str):
        self.dir = os.path.join(workspace, 'jobs') if workspace else ''
        self.path = os.path.join(self.dir, 'jobs.jsonl') if self.dir else ''

    def record(self, job, page: str = '') -> dict | None:
        """Append a finished job (and write its log). Returns the record, or None without a workspace."""
        if not self.dir:
            return None
        stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime(job.started or job.submitted))
        uid = f'{stamp}-{os.getpid()}-{job.id}'
        log_rel = os.path.join('logs', uid + '.log')
        err = next((t for lvl, t in reversed(job.log) if lvl in ('error', 'stderr') and t), '')
        rec = {'schema': SCHEMA, 'uid': uid, 'kind': job.spec.kind, 'title': job.spec.title,
               'submitted': _iso(job.submitted), 'started': _iso(job.started), 'finished': _iso(job.finished),
               'elapsed_s': round(job.elapsed, 2), 'status': job.state, 'exit_code': job.exit_code,
               'exit': P.describe_exit(job.exit_code) if job.exit_code is not None else '',
               'result': result_path(job), 'page': page or '',
               'error': err[:500] if job.state == 'failed' else '', 'log': log_rel}
        os.makedirs(os.path.join(self.dir, 'logs'), exist_ok=True)
        with open(os.path.join(self.dir, log_rel), 'w', encoding='utf-8') as fh:
            fh.write(f'# {job.spec.title} — {job.spec.kind}, {job.state} '
                     f'({rec["exit"] or "no exit code"}) after {job.elapsed:.1f} s\n')
            for lvl, text in job.log:
                fh.write(f'{lvl}: {text}\n' if lvl != 'info' else f'{text}\n')
        with open(self.path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec) + '\n')
        self._trim()
        return rec

    def read(self, limit: int = 500) -> list:
        """The newest `limit` records, newest first (a damaged line is skipped)."""
        if not self.path or not os.path.isfile(self.path):
            return []
        out = []
        with open(self.path, encoding='utf-8') as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get('schema') == SCHEMA:
                    out.append(r)
        return out[::-1][:limit]

    def log_text(self, rec: dict) -> str:
        p = os.path.join(self.dir, rec.get('log') or '')
        if not rec.get('log') or not os.path.isfile(p):
            return '(no log kept for this job)'
        with open(p, encoding='utf-8', errors='replace') as fh:
            return fh.read()

    def _trim(self):
        try:
            with open(self.path, encoding='utf-8') as fh:
                lines = fh.readlines()
        except OSError:
            return
        if len(lines) <= 2 * KEEP:
            return
        keep, drop = lines[-KEEP:], lines[:-KEEP]
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            fh.writelines(keep)
        os.replace(tmp, self.path)
        for line in drop:                                # the dropped entries' logs go with them
            try:
                p = os.path.join(self.dir, json.loads(line).get('log') or '')
                if p != self.dir and os.path.isfile(p):
                    os.remove(p)
            except (ValueError, OSError, AttributeError):
                pass
