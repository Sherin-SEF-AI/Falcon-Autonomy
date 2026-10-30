"""A project's conversation with the assistant and its audit log, kept in the workspace (standard library only).

WORKSPACE/assistant/conversation.json — the conversation, resumable after a restart: the messages in the wire format
(user, assistant with its tool calls, tool results), with Falcon's own notes on them under keys starting with '_'
(never sent: how long a tool took, how it was approved, the numbers of an answer no tool returned, a job event); the
plan checklist; the tools the user approved for the whole conversation; the tool groups loaded; the activity log
(what it changed, and how to undo it); the tokens it used. Never the key — nothing here ever holds it.

WORKSPACE/assistant/audit.jsonl — one line per request to the model server: the time (UTC), what it was for, the model,
the HTTP status, the token counts, the tools the answer called and the SHA-256 of the exact payload sent (not the
payload). Kept so the user can see every time Falcon spoke to the server.
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time

from ..core.paths import atomic_write

VERSION = 1


def folder(workspace: str) -> str:
    return os.path.join(workspace, 'assistant')


def utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def wire(messages: list) -> list:
    """The messages as sent: Falcon's own '_' notes taken off."""
    return [{k: v for k, v in m.items() if not k.startswith('_')} for m in messages]


class Session:
    def __init__(self, workspace: str):
        self.workspace = workspace
        self.path = os.path.join(folder(workspace), 'conversation.json') if workspace else ''
        self.lock = threading.RLock()
        self.messages: list = []
        self.plan: list = []                 # [{'text', 'done'}]
        self.allowed: list = []              # tool names approved for the whole conversation
        self.loaded: list = []               # tool groups whose schemas are sent (tools_load, or the user's words)
        self.activity: list = []             # [{'id', 'time', 'tool', 'kind', 'text', ...undo data}]
        self.usage = {'prompt': 0, 'completion': 0, 'cost': 0.0, 'requests': 0}
        self.created = utc()
        self.notes: list = []                # what loading said (a damaged file kept aside)
        self.load()

    # ------------------------------------------------------------------ persistence
    def load(self):
        if not self.path or not os.path.isfile(self.path):
            return
        try:
            with open(self.path, encoding='utf-8') as fh:
                d = json.load(fh)
            if not isinstance(d, dict) or not isinstance(d.get('messages'), list):
                raise ValueError('not a conversation')
        except (OSError, ValueError) as e:
            aside = self.path + '.damaged'
            try:
                os.replace(self.path, aside)
                self.notes.append(f'the saved conversation could not be read ({e}); kept as {os.path.basename(aside)}')
            except OSError:
                pass
            return
        self.messages = [m for m in d['messages'] if isinstance(m, dict) and m.get('role')]
        self.plan = [p for p in d.get('plan') or [] if isinstance(p, dict)]
        self.allowed = [str(x) for x in d.get('allowed') or []]
        self.loaded = [str(x) for x in d.get('loaded') or []]
        self.activity = [a for a in d.get('activity') or [] if isinstance(a, dict)]
        self.usage.update({k: v for k, v in (d.get('usage') or {}).items() if k in self.usage})
        self.created = d.get('created') or self.created

    def to_dict(self) -> dict:
        with self.lock:
            return {'version': VERSION, 'created': self.created, 'saved': utc(), 'messages': copy.deepcopy(self.messages),
                    'plan': copy.deepcopy(self.plan), 'allowed': list(self.allowed),
                    'loaded': list(self.loaded),
                    'activity': copy.deepcopy(self.activity), 'usage': dict(self.usage)}

    def save(self):
        if not self.path:
            return
        atomic_write(self.path, json.dumps(self.to_dict(), indent=1, ensure_ascii=False) + '\n')

    def clear(self):
        with self.lock:
            self.messages, self.plan, self.allowed, self.activity, self.loaded = [], [], [], [], []
            self.usage = {'prompt': 0, 'completion': 0, 'cost': 0.0, 'requests': 0}
            self.created = utc()
        self.save()

    # ------------------------------------------------------------------ editing
    def add(self, msg: dict) -> int:
        with self.lock:
            self.messages.append(msg)
            return len(self.messages) - 1

    def snapshot(self) -> list:
        with self.lock:
            return copy.deepcopy(self.messages)

    def add_activity(self, entry: dict) -> dict:
        with self.lock:
            entry = dict(entry)
            entry.setdefault('id', f'a{len(self.activity) + 1}-{int(time.time() * 1000) % 100000}')
            entry.setdefault('time', utc())
            self.activity.append(entry)
            return entry

    def update_activity(self, ident: str, **kw):
        with self.lock:
            for a in self.activity:
                if a.get('id') == ident:
                    a.update(kw)
                    return a
        return None


class Audit:
    """The audit log: appended from any thread, one JSON line per request (see the module doc)."""

    def __init__(self, workspace: str):
        self.path = os.path.join(folder(workspace), 'audit.jsonl') if workspace else ''
        self._lock = threading.Lock()

    def write(self, record: dict):
        if not self.path:
            return
        rec = {'time': utc()}
        rec.update({k: v for k, v in record.items() if k in ('purpose', 'model', 'status', 'attempts', 'bytes',
                                                             'prompt_tokens', 'completion_tokens', 'tools',
                                                             'payload_sha256', 'base_url')})
        with self._lock:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + '\n')

    def read(self, n: int = 200) -> list:
        try:
            with open(self.path, encoding='utf-8') as fh:
                lines = fh.readlines()[-n:]
        except OSError:
            return []
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
        return out
