"""A shared team workspace on a plain shared folder or NAS (standard library only): who you are, a lock that works on
network filesystems, who recorded each run, comments, and a triage queue. docs/team.md is the manual.

There is no server. Several people's Falcon instances open the same project on the same folder (NFS, SMB/CIFS, a
synced or local disk) and coordinate through files alone, with three rules:

1. **Append-only, one file per fact.** A comment, a triage event, a run's owner, a member's last visit is one small
   JSON file with a unique name (UTC time, user, a random suffix), written to a temporary name in the same folder and
   renamed into place, never rewritten. Two people writing at once write two files; nothing is merged and nothing is
   lost. (Appending lines to one shared JSONL is not safe here: O_APPEND is not atomic over NFS.)
2. **State is a fold of events.** A triage item is its creation record plus its events; its state, assignee and links
   are the events applied in time order. A transition that is not allowed from the state the earlier events reached
   (two people marking the same item fixed, one verifying what another reopened) is kept in the history as ignored,
   with why — the item never ends in a state nobody chose.
3. **Rewritten files are locked.** The few files that are rewritten whole (the project file, persisted failure
   clusters) are written under FileLock, and the project file only when it has not changed since this instance read
   it (save_project).

FileLock: a lock file beside what it guards, created atomically with its owner, host, pid, a random token and the
time. Creation tries a hard link of a fully written temporary file onto the lock name — atomic on local disks and on
NFS v2/v3/v4 (the method open(2) documents for NFS; a lost reply is checked by the temporary file's link count) — and
falls back to O_CREAT|O_EXCL, which SMB/CIFS servers perform atomically (FILE_CREATE) where hard links are refused.
fcntl/flock locks are not used: over NFS they need a working lock daemon (lockd/NLM, or NFSv4 state) and over SMB
they map to byte-range locks with other semantics, and both are lost silently when a client loses its lease. The
holder refreshes the lock file's modification time (its heartbeat) at least every HEARTBEAT_S; a lock whose
heartbeat is older than STALE_S by the file server's own clock (read from a file touched in the same folder, so
clocks of different machines are not compared), or whose holder is a process on this host that no longer exists, is
stale and is taken over by renaming it aside — rename is atomic, so of two takers one wins, and a taker that finds it
renamed a lock refreshed meanwhile puts it back. Safe on: local filesystems, NFSv3/v4 (any mount options; close-to-open
consistency is enough), SMB2/3 (Samba, Windows, NAS appliances). Not safe on: file-sync services that copy files
between machines (Dropbox, Syncthing, OneDrive, Google Drive): there is no shared file to be atomic on, and two
machines can each hold "the" lock — use them for append-only files only (rules 1 and 2 hold there too).

The job queue's machine-wide CARLA and GPU locks stay in the user's config folder (core.jobs): they guard this
machine's server and GPU, not shared files. The catalogue (SQLite) is never kept on the shared folder: SQLite's
locking is unreliable over NFS and SMB (sqlite.org/faq.html#q5); a team workspace's catalogue is a per-user local file
rebuilt from the run folders, which are the truth (core.catalog.catalog_path).

Identity: settings `user_name` (Settings ▸ Team), else $FALCON_USER, else the login name; with the host name. It is a
label, not an account: anyone with write access to the folder can write as anyone.
"""
from __future__ import annotations

import errno
import getpass
import json
import os
import random
import re
import socket
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

TEAM_DIR = 'team'
SCHEMA = 'falcon.team/1'
COMMENT_SCHEMA = 'falcon.comment/1'
ITEM_SCHEMA = 'falcon.triage_item/1'
EVENT_SCHEMA = 'falcon.triage_event/1'
HEARTBEAT_S = 15.0            # the holder touches its lock at least this often
STALE_S = 120.0               # a heartbeat older than this (server clock) makes a lock stale
POLL_S = 0.2
TEXT_MAX = 4000               # characters of a comment or a title
STATES = ('open', 'assigned', 'investigating', 'fixed', 'verified')
# state -> the states an event may move it to (a reassignment keeps the state; open -> assigned by assigning)
TRANSITIONS = {'open': ('assigned', 'investigating'), 'assigned': ('investigating', 'open'),
               'investigating': ('fixed', 'assigned', 'open'), 'fixed': ('verified', 'investigating'),
               'verified': ('investigating',)}
LINK_KINDS = ('run', 'scenario', 'case', 'suite', 'ci', 'cluster', 'bisect', 'commit', 'study', 'triage')
_SAFE = re.compile(r'[^A-Za-z0-9._-]+')


class TeamError(Exception):
    """A refusal to show the user (a bad id, a transition not allowed, a lock busy, a conflicting save)."""


class LockBusy(TeamError):
    pass


class SaveConflict(TeamError):
    pass


# ================================================================================================= identity, time
def identity(settings=None) -> dict:
    """{'user', 'host'}: settings user_name, else $FALCON_USER, else the login name."""
    name = ''
    if settings is not None:
        try:
            name = str(settings.get('user_name') or '')
        except (AttributeError, KeyError):
            name = ''
    name = (name or os.environ.get('FALCON_USER') or _login()).strip()
    return {'user': name[:64] or 'unknown', 'host': socket.gethostname()[:64]}


def _login() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return 'unknown'


def utc_stamp(t: float | None = None) -> str:
    """UTC with microseconds: what orders events (2026-11-02T10:11:12.123456Z)."""
    d = datetime.fromtimestamp(time.time() if t is None else t, tz=timezone.utc)
    return d.strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def safe(name: str) -> str:
    return _SAFE.sub('_', str(name)).strip('._')[:120] or '_'


def _unique(ident: dict) -> str:
    """A file name that sorts by time and never collides: <compact UTC>-<user>-<random>."""
    t = utc_stamp().replace('-', '').replace(':', '').replace('.', '')
    return f'{t}-{safe(ident["user"])[:24]}-{uuid.uuid4().hex[:8]}'


# ================================================================================================= files
def write_new(path: str, doc: dict) -> bool:
    """Write a JSON file that must not exist yet: a temporary file in the same folder, fsynced, linked (or, where links
    are refused, renamed) into place. False when the name exists already (someone else wrote it)."""
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f'.{os.path.basename(path)}.{uuid.uuid4().hex[:12]}.tmp')
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False, default=str)
        fh.write('\n')
        fh.flush()
        os.fsync(fh.fileno())
    try:
        return _link_exclusive(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _link_exclusive(tmp: str, path: str) -> bool:
    """Put tmp's content at path only if path does not exist: link(2) (atomic, NFS-safe), else O_EXCL create + copy
    (SMB). True when this call made it."""
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    except OSError as e:
        if e.errno in (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV, errno.ENOSYS, errno.EMLINK,
                       errno.EACCES):
            pass                                  # no hard links here (SMB, FAT): exclusive create instead
        else:
            # NFS: the reply to a link that succeeded can be lost and the retry fail; the link count says
            try:
                if os.stat(tmp).st_nlink == 2:
                    return True
            except OSError:
                pass
            raise
    with open(tmp, 'rb') as fh:
        data = fh.read()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, 'wb') as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    return True


def read_json(path: str, default=None):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _docs(folder: str) -> list:
    """Every JSON document in a folder, by file name (time order); unreadable or half-written ones skipped."""
    out = []
    try:
        names = sorted(n for n in os.listdir(folder) if n.endswith('.json') and not n.startswith('.'))
    except OSError:
        return out
    for n in names:
        d = read_json(os.path.join(folder, n))
        if isinstance(d, dict):
            d.setdefault('id', n[:-5])
            out.append(d)
    return out


# ================================================================================================= the lock
def _server_now(folder: str) -> float:
    """The file server's clock: the modification time of a file touched in that folder just now (this machine's
    clock when the folder cannot be written)."""
    p = os.path.join(folder, f'.clock-{safe(socket.gethostname())}-{os.getpid()}')
    try:
        with open(p, 'a', encoding='utf-8'):
            pass
        os.utime(p, None)
        t = os.stat(p).st_mtime
        os.unlink(p)
        return t
    except OSError:
        return time.time()


def _alive_here(info: dict) -> bool | None:
    """Whether the holder of a lock is alive, when it is on this host (None: another host, cannot tell)."""
    if info.get('host') != socket.gethostname():
        return None
    try:
        pid = int(info.get('pid'))
    except (TypeError, ValueError):
        return None
    if pid == os.getpid():
        return True
    if os.path.isdir('/proc'):
        return os.path.exists(f'/proc/{pid}')
    return None


class FileLock:
    """An exclusive lock on a shared folder (module doc): `with FileLock(path_to_guard + '.lock', ident): …`.

    acquire(timeout) waits up to timeout seconds (0: one try) and raises LockBusy naming the holder. While held, a
    daemon thread refreshes the heartbeat every HEARTBEAT_S (heartbeat=False: the caller calls beat()). lost is True
    when the lock was found taken over (the holder stalled past STALE_S); release() then leaves the new holder's
    lock alone."""

    def __init__(self, path: str, ident: dict | None = None, stale_s: float = STALE_S,
                 heartbeat_s: float = HEARTBEAT_S, heartbeat: bool = True):
        self.path = os.path.abspath(path)
        self.ident = ident or identity()
        self.stale_s, self.heartbeat_s, self.auto = float(stale_s), float(heartbeat_s), heartbeat
        self.token = ''
        self.lost = False
        self._stop = threading.Event()
        self._thread = None

    # -- what is in a lock file
    def holder(self) -> dict | None:
        """The current holder's record (None: free), with 'age_s' by the server's clock and 'stale'."""
        info = read_json(self.path)
        if not isinstance(info, dict):
            if not os.path.exists(self.path):
                return None
            info = {}
        try:
            mt = os.stat(self.path).st_mtime
        except OSError:
            return None
        info['age_s'] = round(_server_now(os.path.dirname(self.path)) - mt, 3)
        alive = _alive_here(info)
        info['stale'] = bool(alive is False or info['age_s'] > self.stale_s)
        return info

    def _record(self) -> dict:
        return {'schema': 'falcon.lock/1', 'user': self.ident['user'], 'host': self.ident.get('host') or
                socket.gethostname(), 'pid': os.getpid(), 'token': self.token, 'created': utc_stamp(),
                'guards': os.path.basename(self.path)[:-5] if self.path.endswith('.lock') else ''}

    # -- taking and giving back
    def try_acquire(self) -> bool:
        if self.token:
            return True
        self.token = uuid.uuid4().hex
        if write_new(self.path, self._record()):
            self.lost = False
            self._start()
            return True
        self.token = ''
        h = self.holder()
        if h is not None and h.get('stale'):
            self._take_over(h)
            self.token = uuid.uuid4().hex
            if write_new(self.path, self._record()):
                self.lost = False
                self._start()
                return True
            self.token = ''
        return False

    def _take_over(self, seen: dict):
        """Move a stale lock aside (atomic rename: one taker wins). If what was moved is not the stale lock that was
        seen (its holder refreshed it, or another taker already replaced it), put it back."""
        aside = f'{self.path}.stale.{uuid.uuid4().hex[:8]}'
        try:
            os.rename(self.path, aside)
        except OSError:
            return
        moved = read_json(aside) or {}
        if moved.get('token') != seen.get('token'):
            try:
                os.link(aside, self.path)          # fails when a new holder is already in place: theirs stands
            except OSError:
                pass
        try:
            os.unlink(aside)
        except OSError:
            pass

    def acquire(self, timeout: float = 30.0):
        end = time.monotonic() + max(float(timeout), 0.0)
        while True:
            if self.try_acquire():
                return self
            if time.monotonic() >= end:
                h = self.holder() or {}
                raise LockBusy(f'{os.path.basename(self.path)[:-5] or self.path} is being written by '
                               f'{h.get("user", "someone")} on {h.get("host", "another machine")} (since '
                               f'{h.get("created", "?")}; heartbeat {h.get("age_s", "?")} s ago): try again shortly')
            time.sleep(POLL_S * (0.5 + random.random()))

    def held(self) -> bool:
        """Whether this lock is still ours (the file holds our token)."""
        return bool(self.token) and (read_json(self.path) or {}).get('token') == self.token

    def beat(self) -> bool:
        """Refresh the heartbeat; False (and lost) when the lock is no longer ours."""
        if not self.held():
            self.lost = bool(self.token)
            return False
        try:
            os.utime(self.path, None)
        except OSError:
            return False
        return True

    def _start(self):
        if not self.auto:
            return
        self._stop.clear()

        def run():
            while not self._stop.wait(self.heartbeat_s):
                if not self.beat():
                    return
        self._thread = threading.Thread(target=run, name='falcon-lock-heartbeat', daemon=True)
        self._thread.start()

    def release(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.token and self.held():
            aside = f'{self.path}.release.{uuid.uuid4().hex[:8]}'
            try:
                os.rename(self.path, aside)        # then check: a takeover between read and rename is put back
                if (read_json(aside) or {}).get('token') == self.token:
                    os.unlink(aside)
                else:
                    try:
                        os.link(aside, self.path)
                    except OSError:
                        pass
                    os.unlink(aside)
            except OSError:
                pass
        elif self.token:
            self.lost = True
        self.token = ''

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


# ================================================================================================= the workspace
def team_dir(ws: str) -> str:
    return os.path.join(ws, TEAM_DIR)


def is_team(ws: str) -> bool:
    return bool(ws) and os.path.isfile(os.path.join(team_dir(ws), 'team.json'))


def init(ws: str, name: str = '', ident: dict | None = None) -> dict:
    """Make a workspace a team workspace (team/team.json); the folders its files go in. Idempotent."""
    ident = ident or identity()
    td = team_dir(ws)
    for sub in ('comments', 'triage', 'owners', 'members'):
        os.makedirs(os.path.join(td, sub), exist_ok=True)
    doc = {'schema': SCHEMA, 'name': name or os.path.basename(os.path.normpath(ws)), 'created': utc_stamp(),
           'created_by': ident}
    write_new(os.path.join(td, 'team.json'), doc)
    seen(ws, ident)
    return info(ws)


def seen(ws: str, ident: dict, settings_note: str = ''):
    """Record a member's visit: team/members/<user>@<host>.json, rewritten only by that member."""
    from .paths import atomic_write
    p = os.path.join(team_dir(ws), 'members', f'{safe(ident["user"])}@{safe(ident.get("host", ""))}.json')
    try:
        atomic_write(p, json.dumps({'user': ident['user'], 'host': ident.get('host'), 'last_seen': utc_stamp()}))
    except OSError:
        pass


def info(ws: str) -> dict:
    """The team workspace's record, its members (last seen) and counts; {'team': False} for a solo workspace."""
    if not is_team(ws):
        return {'team': False, 'workspace': ws}
    doc = read_json(os.path.join(team_dir(ws), 'team.json'), {}) or {}
    members = sorted(_docs(os.path.join(team_dir(ws), 'members')), key=lambda m: m.get('last_seen') or '',
                     reverse=True)
    return {'team': True, 'workspace': ws, 'name': doc.get('name'), 'created': doc.get('created'),
            'created_by': doc.get('created_by'), 'members': [{k: m.get(k) for k in ('user', 'host', 'last_seen')}
                                                              for m in members],
            'items': len(item_ids(ws)), 'open_items': sum(1 for i in items(ws) if i['state'] != 'verified')}


# ================================================================================================= run owners
def run_key(ws: str, run_dir: str) -> str:
    """A run's id in the workspace (its folder relative to it: runs/<name>, studies/<id>/runs/<name>)."""
    return os.path.relpath(os.path.realpath(run_dir), os.path.realpath(ws))


def record_owner(ws: str, run_dir: str, ident: dict, what: str = '') -> bool:
    """Who started a run: team/owners/<run id>.json, written once (the first writer stands). Only in a team
    workspace; False otherwise or when it was recorded already."""
    if not is_team(ws):
        return False
    key = run_key(ws, run_dir)
    if key.startswith('..'):
        return False
    doc = {'run': key, 'user': ident['user'], 'host': ident.get('host'), 'time': utc_stamp(), 'what': what}
    try:
        return write_new(os.path.join(team_dir(ws), 'owners', safe(key.replace(os.sep, '__')) + '.json'), doc)
    except OSError:
        return False


def owner_of(ws: str, run_dir_or_key: str) -> dict | None:
    key = run_dir_or_key if not os.path.isabs(run_dir_or_key) else run_key(ws, run_dir_or_key)
    return read_json(os.path.join(team_dir(ws), 'owners', safe(key.replace(os.sep, '__')) + '.json'))


def owners(ws: str) -> dict:
    """run id -> owner record, for every recorded owner."""
    return {d.get('run'): d for d in _docs(os.path.join(team_dir(ws), 'owners')) if d.get('run')}


# ================================================================================================= comments
def check_ref(ref: str) -> str:
    """An object reference 'kind:id' (LINK_KINDS); TeamError otherwise."""
    kind, sep, rest = str(ref).partition(':')
    if not sep or kind not in LINK_KINDS or not rest.strip() or '\x00' in rest or '..' in rest.split('/'):
        raise TeamError(f'{ref!r} is not a reference: use kind:id with kind one of {", ".join(LINK_KINDS)} '
                        '(e.g. run:runs/2026-11-02-drive, scenario:lead_brake, case:search-lead_brake#4, triage:T-3)')
    return f'{kind}:{rest.strip()}'


def _comment_dir(ws: str, ref: str) -> str:
    return os.path.join(team_dir(ws), 'comments', safe(ref.replace('/', '__')))


def add_comment(ws: str, ref: str, text: str, ident: dict, reply_to: str = '') -> dict:
    """A comment on an object: one new file, never rewritten (module doc). Returns it."""
    ref = check_ref(ref)
    text = str(text or '').strip()
    if not text:
        raise TeamError('an empty comment')
    if len(text) > TEXT_MAX:
        raise TeamError(f'a comment is at most {TEXT_MAX} characters')
    cid = _unique(ident)
    doc = {'schema': COMMENT_SCHEMA, 'id': cid, 'object': ref, 'user': ident['user'], 'host': ident.get('host'),
           'time': utc_stamp(), 'text': text}
    if reply_to:
        doc['reply_to'] = str(reply_to)
    if ident.get('via'):
        doc['via'] = str(ident['via'])          # written for the user by the assistant, said so
    write_new(os.path.join(_comment_dir(ws, ref), cid + '.json'), doc)
    return doc


def comments(ws: str, ref: str) -> list:
    """An object's comments, oldest first."""
    ref = check_ref(ref)
    return sorted(_docs(_comment_dir(ws, ref)), key=lambda c: (c.get('time') or '', c.get('id') or ''))


def commented(ws: str) -> dict:
    """object ref -> number of comments, over the workspace."""
    root = os.path.join(team_dir(ws), 'comments')
    out = {}
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for n in names:
        cs = _docs(os.path.join(root, n))
        if cs:
            out[cs[0].get('object') or n] = len(cs)
    return out


# ================================================================================================= triage
def _triage_root(ws: str) -> str:
    return os.path.join(team_dir(ws), 'triage')


def item_ids(ws: str) -> list:
    try:
        names = os.listdir(_triage_root(ws))
    except OSError:
        return []
    ids = [n for n in names if re.fullmatch(r'T-\d+', n) and os.path.isfile(os.path.join(_triage_root(ws), n,
                                                                                         'item.json'))]
    return sorted(ids, key=lambda n: int(n[2:]))


def norm_item(item: str) -> str:
    s = str(item).strip()
    if s.lower().startswith('triage:'):
        s = s[7:]
    if re.fullmatch(r'\d+', s):
        s = 'T-' + s
    s = s.upper()
    if not re.fullmatch(r'T-\d+', s):
        raise TeamError(f'{item!r} is not a triage item id (T-12, or 12)')
    return s


def create_item(ws: str, title: str, ident: dict, links=(), kind: str = 'failure', assignee: str = '',
                note: str = '') -> dict:
    """A new triage item T-<n>: the number claimed by an atomic mkdir (two people creating at once get two numbers),
    then item.json written once. assignee: assigned at once (an 'assign' event)."""
    title = ' '.join(str(title or '').split())
    if not title:
        raise TeamError('a triage item needs a title')
    if len(title) > 200:
        raise TeamError('a title is at most 200 characters')
    if kind not in ('failure', 'issue'):
        raise TeamError('kind is failure or issue')
    links = [check_ref(r) for r in links or ()]
    root = _triage_root(ws)
    os.makedirs(root, exist_ok=True)
    nums = [int(n[2:]) for n in os.listdir(root) if re.fullmatch(r'T-\d+', n)]
    n = max(nums, default=0) + 1
    for _ in range(1000):
        try:
            os.mkdir(os.path.join(root, f'T-{n}'))
            break
        except FileExistsError:
            n += 1
    else:
        raise TeamError('could not claim a triage number')
    iid = f'T-{n}'
    doc = {'schema': ITEM_SCHEMA, 'id': iid, 'title': title, 'kind': kind, 'created': utc_stamp(),
           'created_by': {k: ident.get(k) for k in ('user', 'host', 'via') if ident.get(k)},
           'links': list(dict.fromkeys(links)), 'note': str(note or '')[:TEXT_MAX]}
    write_new(os.path.join(root, iid, 'item.json'), doc)
    if assignee:
        add_event(ws, iid, {'type': 'assign', 'to': assignee}, ident)
    return get_item(ws, iid)


def add_event(ws: str, item: str, event: dict, ident: dict) -> dict:
    """Append an event to an item (one new file). Checked against the item's state now; a concurrent event that the
    fold later finds out of order is kept as ignored (module doc). Returns the item after it."""
    iid = norm_item(item)
    cur = get_item(ws, iid)
    ev = dict(event)
    typ = ev.get('type')
    if typ == 'assign':
        to = ' '.join(str(ev.get('to') or '').split())[:64]
        if not to:
            raise TeamError('assign to whom?')
        ev['to'] = to
    elif typ == 'state':
        to = ev.get('to')
        if to not in STATES:
            raise TeamError(f'state is one of {", ".join(STATES)}')
        why = _transition_problem(cur['state'], to, cur['assignee'])
        if why:
            raise TeamError(f'{iid}: {why}')
    elif typ == 'link':
        ev['add'] = [check_ref(r) for r in ev.get('add') or ()]
        ev['remove'] = [check_ref(r) for r in ev.get('remove') or ()]
        if not ev['add'] and not ev['remove']:
            raise TeamError('link: nothing to add or remove')
    elif typ == 'title':
        t = ' '.join(str(ev.get('to') or '').split())
        if not t or len(t) > 200:
            raise TeamError('a title of 1–200 characters')
        ev['to'] = t
    else:
        raise TeamError(f'unknown event {typ!r} (assign, state, link, title)')
    eid = _unique(ident)
    ev.update(schema=EVENT_SCHEMA, id=eid, user=ident['user'], host=ident.get('host'), time=utc_stamp())
    if ident.get('via'):
        ev['via'] = str(ident['via'])
    write_new(os.path.join(_triage_root(ws), iid, 'events', eid + '.json'), ev)
    return get_item(ws, iid)


def _transition_problem(state: str, to: str, assignee: str) -> str:
    if to == state:
        return f'it is {state} already'
    if to not in TRANSITIONS.get(state, ()):
        allowed = ', '.join(TRANSITIONS.get(state, ())) or 'nothing'
        return f'{state} → {to} is not a step of the queue (from {state}: {allowed})'
    if to in ('investigating', 'fixed') and not assignee:
        return f'{to} needs an assignee: assign it first'
    return ''


def fold(item: dict, events: list) -> dict:
    """The item's state, assignee, links, title and history from its events in time order (ties by id)."""
    st = {'state': 'open', 'assignee': '', 'links': list(item.get('links') or []), 'title': item.get('title'),
          'history': [], 'fixed_by': '', 'verified_by': ''}
    st['history'].append({'time': item.get('created'), 'user': (item.get('created_by') or {}).get('user'),
                          'what': 'created', 'ok': True})
    for ev in sorted(events, key=lambda e: (e.get('time') or '', e.get('id') or '')):
        typ, who = ev.get('type'), ev.get('user')
        h = {'time': ev.get('time'), 'user': who, 'id': ev.get('id'), 'ok': True}
        if ev.get('via'):
            h['via'] = ev['via']
        if typ == 'assign':
            st['assignee'] = ev.get('to') or ''
            h['what'] = f'assigned to {st["assignee"]}'
            if st['state'] == 'open':
                st['state'] = 'assigned'
        elif typ == 'state':
            why = _transition_problem(st['state'], ev.get('to'), st['assignee'])
            if why:
                h.update(ok=False, what=f'{ev.get("to")} ignored: {why} (an event written at the same time by '
                                        'another instance, applied in time order)')
            else:
                h['what'] = f'{st["state"]} → {ev["to"]}'
                st['state'] = ev['to']
                if ev['to'] == 'fixed':
                    st['fixed_by'] = who or ''
                if ev['to'] == 'verified':
                    st['verified_by'] = who or ''
                    if who and who == st['fixed_by']:
                        h['note'] = 'verified by whoever marked it fixed'
        elif typ == 'link':
            for r in ev.get('add') or ():
                if r not in st['links']:
                    st['links'].append(r)
            st['links'] = [r for r in st['links'] if r not in set(ev.get('remove') or ())]
            h['what'] = 'links ' + ', '.join([f'+{r}' for r in ev.get('add') or ()] +
                                             [f'−{r}' for r in ev.get('remove') or ()])
        elif typ == 'title':
            st['title'] = ev.get('to') or st['title']
            h['what'] = f'title: {st["title"]}'
        else:
            h.update(ok=False, what=f'unknown event {typ!r} ignored')
        if ev.get('note'):
            h['note'] = str(ev['note'])[:400]
        st['history'].append(h)
    return st


def get_item(ws: str, item: str) -> dict:
    iid = norm_item(item)
    d = os.path.join(_triage_root(ws), iid)
    doc = read_json(os.path.join(d, 'item.json'))
    if not isinstance(doc, dict):
        raise TeamError(f'no triage item {iid} (falcon triage list shows them)')
    events = _docs(os.path.join(d, 'events'))
    st = fold(doc, events)
    return {'id': iid, 'kind': doc.get('kind'), 'title': st['title'], 'state': st['state'],
            'assignee': st['assignee'], 'links': st['links'], 'created': doc.get('created'),
            'created_by': (doc.get('created_by') or {}).get('user'), 'note': doc.get('note') or '',
            'history': st['history'], 'fixed_by': st['fixed_by'], 'verified_by': st['verified_by'],
            'updated': st['history'][-1]['time'] if st['history'] else doc.get('created'),
            'conflicts': sum(1 for h in st['history'] if not h.get('ok')), 'path': d}


def items(ws: str, state: str = '', assignee: str = '', link: str = '') -> list:
    """Every triage item (newest first), optionally only one state, one assignee or those linking one object."""
    out = []
    for iid in item_ids(ws):
        try:
            it = get_item(ws, iid)
        except TeamError:
            continue
        if state and it['state'] != state:
            continue
        if assignee and it['assignee'] != assignee:
            continue
        if link and link not in it['links']:
            continue
        out.append(it)
    return sorted(out, key=lambda i: -int(i['id'][2:]))


def assign(ws, item, to, ident):
    return add_event(ws, item, {'type': 'assign', 'to': to}, ident)


def set_state(ws, item, to, ident, note: str = ''):
    ev = {'type': 'state', 'to': to}
    if note:
        ev['note'] = str(note)[:400]
    return add_event(ws, item, ev, ident)


def link(ws, item, add=(), remove=(), ident=None):
    return add_event(ws, item, {'type': 'link', 'add': list(add), 'remove': list(remove)}, ident or identity())


# ================================================================================================= the project file
def file_sha(path: str) -> str:
    import hashlib
    try:
        with open(path, 'rb') as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return ''


def remember(project):
    """Note the project file as this instance read it (save_project compares with it)."""
    project._team_base = file_sha(project.path) if getattr(project, 'path', '') else ''


def save_project(project, ident: dict, force: bool = False, timeout: float = 10.0):
    """Save a project. In a team workspace: under the project's lock, and only when the file is still what this
    instance read (else SaveConflict naming who saved it last — reopen to take their changes, or force). Elsewhere a
    plain save."""
    ws = project.workspace
    if not ws or not is_team(ws) or not project.path:
        project.save()
        remember(project)
        return
    lock = FileLock(project.path + '.lock', ident).acquire(timeout)
    try:
        base = getattr(project, '_team_base', None)
        now = file_sha(project.path)
        if not force and base is not None and now and now != base:
            last = read_json(os.path.join(team_dir(ws), 'last-save.json'), {}) or {}
            raise SaveConflict(f'{os.path.basename(project.path)} was saved by {last.get("user", "someone else")} '
                               f'on {last.get("host", "another machine")} at {last.get("time", "?")} after you opened '
                               'it: saving now would discard their changes. Reopen it (File ▸ Open) and redo yours, '
                               'or save anyway to keep only yours.')
        project.save()
        remember(project)
        from .paths import atomic_write
        atomic_write(os.path.join(team_dir(ws), 'last-save.json'),
                     json.dumps({'user': ident['user'], 'host': ident.get('host'), 'time': utc_stamp(),
                                 'sha256': project._team_base}))
    finally:
        lock.release()


# ================================================================================================= the CLI
def _ws_of(path: str) -> str:
    from .paths import PROJECT_SUFFIX, workspace_for
    p = os.path.abspath(path)
    if p.endswith(PROJECT_SUFFIX) or os.path.isfile(p):
        return workspace_for(p)
    return p


def _ident():
    from .settings import Settings
    return identity(Settings())


def _print_item(it: dict, full: bool = False):
    print(f'{it["id"]:6s} {it["state"]:13s} {(it["assignee"] or "—")[:16]:16s} {it["title"]}')
    if full:
        print(f'  created {it["created"]} by {it["created_by"]}; {it["kind"]}')
        for r in it['links']:
            print(f'  link  {r}')
        for h in it['history']:
            print(f'  {h["time"]}  {h.get("user") or "?":12s} {"" if h.get("ok") else "!! "}{h["what"]}'
                  + (f' ({h["note"]})' if h.get('note') else ''))


def main_team(argv) -> int:
    """`falcon team init|info|owner …` (docs/team.md). 0 ok, 2 failed."""
    import argparse
    ap = argparse.ArgumentParser(prog='falcon team', description='a shared team workspace (docs/team.md)')
    sub = ap.add_subparsers(dest='what', required=True)
    i = sub.add_parser('init', help='make a project\'s workspace a team workspace')
    i.add_argument('project', help='the project file (or its workspace folder)')
    i.add_argument('--name', default='')
    f = sub.add_parser('info', help='the team workspace: members, items, who you are')
    f.add_argument('project')
    f.add_argument('--json', action='store_true')
    o = sub.add_parser('owner', help='who recorded a run')
    o.add_argument('project')
    o.add_argument('run', help='the run id (runs/<folder>) or folder')
    lk = sub.add_parser('lock', help='who holds the lock on a shared file (or take a stale one: --clear-stale)')
    lk.add_argument('file')
    lk.add_argument('--clear-stale', action='store_true')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    try:
        ident = _ident()
        if a.what == 'init':
            d = init(_ws_of(a.project), a.name, ident)
            print(f'team workspace {d["name"]} at {d["workspace"]} (you: {ident["user"]} on {ident["host"]})')
            return 0
        if a.what == 'info':
            d = info(_ws_of(a.project))
            d['you'] = ident
            if a.json:
                print(json.dumps(d, indent=2))
                return 0
            if not d['team']:
                print(f'not a team workspace: {d["workspace"]} (falcon team init PROJECT makes it one)')
                return 0
            print(f'{d["name"]}: {d["items"]} triage items, {d["open_items"]} not verified; you are {ident["user"]} '
                  f'on {ident["host"]}')
            for m in d['members']:
                print(f'  {m["user"]:16s} {m["host"]:20s} last seen {m["last_seen"]}')
            return 0
        if a.what == 'owner':
            ws = _ws_of(a.project)
            rd = a.run if os.path.isabs(a.run) else os.path.join(ws, a.run)
            o = owner_of(ws, rd)
            print(f'{o["user"]} on {o["host"]} at {o["time"]}' if o else 'not recorded (runs are owned from when the '
                                                                          'workspace became a team workspace)')
            return 0
        if a.what == 'lock':
            lk = FileLock(a.file if a.file.endswith('.lock') else a.file + '.lock', ident)
            h = lk.holder()
            if h is None:
                print('free')
                return 0
            print(f'held by {h.get("user")} on {h.get("host")} (pid {h.get("pid")}) since {h.get("created")}; '
                  f'heartbeat {h.get("age_s")} s ago{" — STALE" if h.get("stale") else ""}')
            if a.clear_stale and h.get('stale'):
                lk._take_over(h)
                print('the stale lock was moved aside')
            return 0
    except (TeamError, OSError) as e:
        print(f'failed: {e}', file=sys.stderr, flush=True)
        return 2
    return 2


def main_triage(argv) -> int:
    """`falcon triage list|show|add|assign|state|link …` (docs/team.md). 0 ok, 2 failed."""
    import argparse
    ap = argparse.ArgumentParser(prog='falcon triage', description='the triage queue of a workspace (docs/team.md)')
    sub = ap.add_subparsers(dest='what', required=True)
    ls = sub.add_parser('list', help='the items, newest first')
    ls.add_argument('project')
    ls.add_argument('--state', choices=STATES, default='')
    ls.add_argument('--assignee', default='')
    ls.add_argument('--mine', action='store_true')
    ls.add_argument('--json', action='store_true')
    sh = sub.add_parser('show', help='one item with its history and comments')
    sh.add_argument('project')
    sh.add_argument('item')
    sh.add_argument('--json', action='store_true')
    ad = sub.add_parser('add', help='a new item')
    ad.add_argument('project')
    ad.add_argument('title')
    ad.add_argument('--link', action='append', default=[], help='kind:id (run:, scenario:, case:, ci:, cluster:, …)')
    ad.add_argument('--assign', default='')
    ad.add_argument('--kind', choices=('failure', 'issue'), default='failure')
    ad.add_argument('--note', default='')
    asg = sub.add_parser('assign', help='assign an item')
    asg.add_argument('project')
    asg.add_argument('item')
    asg.add_argument('to', nargs='?', default='', help='who (default: you)')
    st = sub.add_parser('state', help='move an item: assigned, investigating, fixed, verified (or back)')
    st.add_argument('project')
    st.add_argument('item')
    st.add_argument('to', choices=STATES)
    st.add_argument('--note', default='')
    lk = sub.add_parser('link', help='add or remove links')
    lk.add_argument('project')
    lk.add_argument('item')
    lk.add_argument('--add', action='append', default=[])
    lk.add_argument('--remove', action='append', default=[])
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    try:
        ws = _ws_of(a.project)
        ident = _ident()
        if a.what == 'list':
            rows = items(ws, a.state, ident['user'] if a.mine else a.assignee)
            if a.json:
                print(json.dumps([{k: v for k, v in r.items() if k not in ('history', 'path')} for r in rows],
                                 indent=2))
                return 0
            if not rows:
                print('no triage items' + (' with those filters' if a.state or a.assignee or a.mine else ''))
            for r in rows:
                _print_item(r)
            return 0
        if a.what == 'show':
            it = get_item(ws, a.item)
            cs = comments(ws, f'triage:{it["id"]}')
            if a.json:
                print(json.dumps(dict({k: v for k, v in it.items() if k != 'path'}, comments=cs), indent=2))
                return 0
            _print_item(it, full=True)
            for c in cs:
                print(f'  comment {c["time"]} {c["user"]}: {c["text"]}')
            return 0
        if a.what == 'add':
            it = create_item(ws, a.title, ident, a.link, a.kind, a.assign, a.note)
        elif a.what == 'assign':
            it = assign(ws, a.item, a.to or ident['user'], ident)
        elif a.what == 'state':
            it = set_state(ws, a.item, a.to, ident, a.note)
        else:
            it = link(ws, a.item, a.add, a.remove, ident)
        _print_item(it)
        return 0
    except (TeamError, OSError) as e:
        print(f'failed: {e}', file=sys.stderr, flush=True)
        return 2


def main_comment(argv) -> int:
    """`falcon comment PROJECT REF [TEXT]`: list an object's comments, or add one (docs/team.md). 0 ok, 2 failed."""
    import argparse
    ap = argparse.ArgumentParser(prog='falcon comment', description='comments on runs, failures and triage items '
                                                                    '(docs/team.md)')
    ap.add_argument('project')
    ap.add_argument('ref', help='kind:id — run:runs/<folder>, case:<study>#<row>, scenario:<id>, triage:T-3, '
                                'cluster:<id>, ci:<name>')
    ap.add_argument('text', nargs='?', default='', help='the comment (none: list the comments)')
    ap.add_argument('--reply-to', default='')
    ap.add_argument('--json', action='store_true')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    try:
        ws = _ws_of(a.project)
        if a.text:
            c = add_comment(ws, a.ref, a.text, _ident(), a.reply_to)
            print(f'{c["id"]}: {c["user"]} on {c["object"]}')
            return 0
        cs = comments(ws, a.ref)
        if a.json:
            print(json.dumps(cs, indent=2))
            return 0
        if not cs:
            print(f'no comments on {check_ref(a.ref)}')
        for c in cs:
            print(f'{c["time"]}  {c["user"]}: {c["text"]}')
        return 0
    except (TeamError, OSError) as e:
        print(f'failed: {e}', file=sys.stderr, flush=True)
        return 2
