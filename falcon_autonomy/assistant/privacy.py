"""What the assistant may show a language model (standard library only).

redact(obj): every string in a result, recursively — the user's home directory shortened to ~ (paths say who the user
is), and anything shaped like a credential replaced by [secret] (a Groq key gsk_…, an OpenAI-style sk-… key, a bearer
token, a password= or token= value). Applied to every tool result (tool_api.ToolContext.redact) and to every message
before it leaves the machine.

private_reason(path): why a file or folder must not reach the model, or None. The user's dashcam footage is private: it
shows where and when they drove. Refused: anything under ~/Videos; a run, reconstruction, image set, translator or world
whose own records say it came from dashcam footage — a drive along a dashcam route (dashcam_route in its settings, or in
its provenance.settings, where run.json keeps the config), a reconstruction posed from a dashcam segment (dashcam.json),
frames extracted from clips (worker dashcam-extract), a real image set or translator of kind dashcam, an OSM import
whose source is a dashcam route, an incident made a scenario (worker dashcam-incident) and any drive, stack run or study
of such a scenario (its scenario's source {"format": "dashcam"}, in drive.json, stack.json, study.json or run.json's
provenance), and a folder of clips ingested for near-miss mining (core.dashcam_ingest): the folder, every clip in it and
everything under its falcon-ingest/ (perception, rankings, event folders, cut clips) — found at any depth by the
ingest's own record, falcon-ingest/ingest.json. A reconstruction of a dashcam segment or near-miss (dashcam.json; its
posed/tracks.json of the ingest's road users with it) and everything rendered from it — re-rendered runs, hybrid runs,
closed-loop drives (their records carry the source marker {"format": "dashcam"}, and closedloop.json is read too) —
are private; a record that names the reconstruction it was rendered from (its 'reconstruction', under 'neural' or
'hybrid' too), or a reconstruction's run.json naming its source, is private when that is (followed two levels), so
older outputs without the marker are covered while their reconstruction exists. The markers are read from the
folder's own run.json / import.json / model json (and its parent's, for a file in a run); a folder without any record
is not private by this test — the tools only hand over Falcon's own outputs.

guard(path): raise ToolError(private_reason) when it is private. Tools call it on every path they read before
returning anything from it.
"""
from __future__ import annotations

import json
import os
import re

from .tool_api import ToolError

HOME = os.path.expanduser('~')
SECRET_PATTERNS = (
    re.compile(r'gsk_[A-Za-z0-9]{16,}'),                       # Groq
    re.compile(r'sk-[A-Za-z0-9_\-]{16,}'),                     # OpenAI-style
    re.compile(r'(?i)bearer\s+[A-Za-z0-9._\-]{16,}'),
    re.compile(r'(?i)\b(password|passwd|token|api[_-]?key|secret)\s*[=:]\s*\S+'),
)
RECORDS = ('run.json', 'import.json', 'dashcam.json', 'translator.json', 'model.json', 'set.json', 'drive.json',
           'stack.json', 'study.json', 'closedloop.json')


def redact_text(s: str) -> str:
    if HOME and HOME != '/':
        s = s.replace(HOME + os.sep, '~' + os.sep)
        if s == HOME:
            s = '~'
    for p in SECRET_PATTERNS:
        s = p.sub('[secret]', s)
    return s


def redact(obj):
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        return {redact_text(k) if isinstance(k, str) else k: redact(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(x) for x in obj]
    return obj


def _says_dashcam(doc) -> str | None:
    """A record's own word that it came from dashcam footage (the markers in the module doc)."""
    if not isinstance(doc, dict):
        return None
    st = doc.get('settings') or {}
    if isinstance(st, dict) and st.get('dashcam_route'):
        return 'a drive along a route taken from the dashcam footage'
    pst = (doc.get('provenance') or {}).get('settings') if isinstance(doc.get('provenance'), dict) else None
    if isinstance(pst, dict) and pst.get('dashcam_route'):     # where a run.json keeps its config (new_run)
        return 'a drive along a route taken from the dashcam footage'
    if doc.get('dashcam_route'):
        return 'a drive along a route taken from the dashcam footage'
    for d in (doc, st, pst):                                  # a scenario made from a dashcam event (core.incident)
        scn = d.get('scenario') if isinstance(d, dict) else None
        src = scn.get('source') if isinstance(scn, dict) else None
        if isinstance(src, dict) and src.get('format') == 'dashcam':
            return 'a scenario made from an event in the dashcam footage'
    worker = str(doc.get('worker') or (doc.get('provenance') or {}).get('worker') or '')
    if worker.startswith('dashcam'):
        return 'made from the dashcam footage'
    for key in ('source', 'real', 'data'):
        v = doc.get(key)
        vals = v if isinstance(v, list) else [v]
        for x in vals:
            if isinstance(x, dict) and ('dashcam' in (str(x.get('format', '')), str(x.get('kind', '')))):
                return 'its source is the dashcam footage'
            if isinstance(x, str) and x.split(':', 1)[0] == 'dashcam':
                return 'its source is the dashcam footage'
    return None


def _made_from(doc) -> list:
    """The reconstructions (and a reconstruction's source) a record says it was made from: a re-rendered run's, a
    hybrid run's or a closed-loop drive's 'reconstruction' (top level, or under 'neural' / 'hybrid'), and a
    reconstruction's own source folder (run.json 'source' of a recon.* worker) — so what is rendered from a private
    reconstruction, or reconstructed from a private drive, is private too even without its own marker."""
    out = []
    if not isinstance(doc, dict):
        return out
    for d in (doc, doc.get('neural'), doc.get('hybrid')):
        r = d.get('reconstruction') if isinstance(d, dict) else None
        if isinstance(r, dict):
            r = r.get('reconstruction')
        if isinstance(r, str) and r:
            out.append(r)
    src = doc.get('source')
    if str(doc.get('worker') or '').startswith('recon.') and isinstance(src, dict) and isinstance(src.get('path'),
                                                                                                 str):
        out.append(src['path'])
    return out


INGEST_DIR = 'falcon-ingest'               # core.dashcam_ingest.INGEST_DIR (not imported: this module stays light)


def ingest_reason(p: str) -> str | None:
    """A folder ingested for near-miss mining, anything in it, or anything under its falcon-ingest/ (every ancestor
    is looked at: event folders are deep)."""
    d = p if os.path.isdir(p) else os.path.dirname(p)
    while True:
        if os.path.basename(d) == INGEST_DIR and os.path.isfile(os.path.join(d, 'ingest.json')):
            return 'made from dashcam footage (near-miss mining)'
        if os.path.isfile(os.path.join(d, INGEST_DIR, 'ingest.json')):
            return 'a folder of dashcam footage ingested for near-miss mining'
        up = os.path.dirname(d)
        if up == d:
            return None
        d = up


def private_reason(path: str, _depth: int = 0) -> str | None:
    if not path:
        return None
    p = os.path.realpath(os.path.expanduser(str(path)))
    videos = os.path.realpath(os.path.join(HOME, 'Videos'))
    if p == videos or p.startswith(videos + os.sep):
        return 'the user\'s own footage folder is private'
    why = ingest_reason(p)
    if why:
        return why
    folders = [p] if os.path.isdir(p) else []
    d = os.path.dirname(p)
    for _ in range(3):                       # a file inside a run, or a run inside a set: its owners' records too
        folders.append(d)
        d = os.path.dirname(d)
    for f in folders:
        if os.path.isfile(os.path.join(f, 'dashcam.json')):
            return 'a reconstruction posed from the dashcam footage'
        for name in RECORDS:
            fp = os.path.join(f, name)
            if not os.path.isfile(fp):
                continue
            try:
                with open(fp, encoding='utf-8') as fh:
                    doc = json.load(fh)
            except (OSError, ValueError):
                continue
            why = _says_dashcam(doc)
            if why:
                return why
            if _depth < 2:                   # rendered from a private reconstruction, or of a private source
                for src in _made_from(doc):
                    if src and os.path.realpath(os.path.expanduser(src)) != p and private_reason(src, _depth + 1):
                        return 'made from data derived from the dashcam footage (a reconstruction, or the drive it ' \
                               'reconstructs)'
    return None


def guard(path: str):
    why = private_reason(path)
    if why:
        raise ToolError(f'private: {why} — the assistant never reads it (docs/assistant.md, "Privacy")')
