"""The run index: one summary per recorded run of a workspace, a small query language over them, tags, and drive sets.

Every run folder with a run.json is indexed — drives, scenario runs, a study's search runs, replays (dashcam routes
among them) and live sessions — from what the run itself wrote: run.json (the worker, its status, its config in
provenance.settings, the scenario's criticality, the ODD monitor's summary) and frames.jsonl (the ego's speed and
path, teleports). Nothing is estimated: a fact a run did not record is None and shows as "not recorded".

The summaries are cached in <workspace>/runindex.json, keyed by the run's path and the sizes and modification times of
the files read, so a large workspace is read once; the cache can be deleted at any time (the files are the truth).

Tags are the user's, not the run's: they live in the project (run_tags, keyed by the run's path relative to the
workspace), so a recorded run is never modified. Drive sets (the project's drive_sets) are named lists of runs or saved
queries, re-evaluated when used.

The query language (docs/runs.md):

    world:w-blr weather:rain status:ok events:collision dur>20 after:2026-10-28 tag:night "lead brake"

Terms are ANDed. ``key:value`` matches a field (case-insensitive substring; ``events:`` and ``tag:`` membership),
``key>n``, ``key<n``, ``key>=n``, ``key<=n`` and ``key=n`` compare numbers, ``after:``/``before:`` compare the start
date (an ISO date or date-time, UTC), ``-term`` negates a term, and anything else is free text matched against the
run's name, folder, world, map, weather, scenario and tags. Quotes keep spaces in a value.
"""
from __future__ import annotations

import json
import math
import os
import re
import shlex

from .provenance import utc_now

SCHEMA = 'falcon.runindex/1'
CACHE_FILE = 'runindex.json'
FILES = ('run.json', 'frames.jsonl', 'odd-monitor.json')

# time of day from the sun's altitude, as core.weather.describe names it (degrees)
TOD_BOUNDS = ((-6.0, 'night'), (6.0, 'twilight'), (20.0, 'low sun'))
TODS = ('day', 'low sun', 'twilight', 'night')
WEATHER_KINDS = ('clear', 'cloudy', 'wet', 'rain')

# fields the query language knows: name -> (entry key, kind)
FIELDS = {
    'name': ('name', 'text'), 'world': ('world', 'text'), 'map': ('map', 'text'), 'weather': ('weather', 'text'),
    'tod': ('tod', 'text'), 'kind': ('kind', 'text'), 'worker': ('worker', 'text'), 'status': ('status', 'text'),
    'scenario': ('scenario', 'text'), 'rig': ('rig', 'text'), 'folder': ('rel', 'text'),
    'events': ('events', 'set'), 'event': ('events', 'set'), 'tag': ('tags', 'set'), 'tags': ('tags', 'set'),
    'dur': ('duration_s', 'num'), 'duration': ('duration_s', 'num'), 'frames': ('frames', 'num'),
    'seed': ('seed', 'num'), 'fps': ('fps', 'num'), 'speed': ('speed_max_kmh', 'num'),
    'mean_speed': ('speed_mean_kmh', 'num'), 'dist': ('distance_m', 'num'), 'distance': ('distance_m', 'num'),
    'ttc': ('min_ttc_s', 'num'), 'collisions': ('collisions', 'num'), 'rain': ('rain', 'num'),
    'fog': ('fog', 'num'), 'sun': ('sun_altitude', 'num'),
}
DATE_KEYS = ('after', 'before')


# ================================================================================================ one run
def tod_of(sun_altitude: float | None) -> str | None:
    if sun_altitude is None:
        return None
    for hi, name in TOD_BOUNDS:
        if sun_altitude < hi:
            return name
    return 'day'


def weather_kind(w: dict | None) -> str | None:
    """clear, cloudy, wet (a wet road without rain) or rain, from CARLA's weather parameters (heuristic thresholds:
    any rain; wetness or puddles above 0; cloud cover of 50 % or more). Fog is a separate number (fog_density)."""
    if not w:
        return None
    if float(w.get('precipitation') or 0) > 0:
        return 'rain'
    if float(w.get('wetness') or 0) > 0 or float(w.get('precipitation_deposits') or 0) > 0:
        return 'wet'
    if float(w.get('cloudiness') or 0) >= 50:
        return 'cloudy'
    return 'clear'


def _stat(paths) -> list:
    out = []
    for p in paths:
        try:
            st = os.stat(p)
            out.append([os.path.basename(p), st.st_size, st.st_mtime_ns])
        except OSError:
            out.append([os.path.basename(p), None, None])
    return out


def _kind(run: dict, settings: dict, run_dir: str) -> str:
    worker = run.get('worker') or ''
    if settings.get('dashcam_route'):
        return 'route'
    parts = os.path.normpath(run_dir).split(os.sep)
    if len(parts) >= 3 and parts[-2] == 'runs' and 'studies' in parts[:-2]:
        return 'search'
    if settings.get('scenario'):
        return 'scenario'
    return worker or 'run'


def _scan_frames(path: str) -> dict:
    """Speed, path length, duration and teleports from frames.jsonl (one pass, nothing kept per frame)."""
    n = 0
    sp_sum = 0.0
    sp_max = None
    dist = 0.0
    prev = None
    t0 = t1 = None
    teleports = []
    odd_out = 0
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                f = json.loads(line)
            except ValueError:
                continue
            n += 1
            t = f.get('sim_t')
            if t is not None:
                t0 = t if t0 is None else t0
                t1 = t
            s = f.get('speed_mps')
            if s is not None:
                sp_sum += float(s)
                sp_max = float(s) if sp_max is None else max(sp_max, float(s))
            e = f.get('ego') or {}
            if 'x' in e and 'y' in e:
                if f.get('teleport') is not None:            # seconds from the run's first frame
                    teleports.append(round(float(t or 0.0) - float(t0 or 0.0), 3))
                elif prev is not None:
                    dist += math.hypot(e['x'] - prev[0], e['y'] - prev[1])
                prev = (float(e['x']), float(e['y']))
            o = f.get('odd')
            if isinstance(o, dict) and o.get('in') is False:
                odd_out += 1
    return {'n': n, 'speed_mean': (sp_sum / n) if n and sp_max is not None else None, 'speed_max': sp_max,
            'distance_m': dist if prev is not None else None, 'span_s': (t1 - t0) if t0 is not None else None,
            'teleports': teleports, 'odd_out_frames': odd_out}


def summarise(run_dir: str, workspace: str = '') -> dict | None:
    """One run's summary (None without a readable run.json)."""
    rp = os.path.join(run_dir, 'run.json')
    try:
        with open(rp, encoding='utf-8') as fh:
            run = json.load(fh)
    except (OSError, ValueError):
        return None
    prov = run.get('provenance') or {}
    st = prov.get('settings') or {}
    proj = st.get('project') or {}
    perf = run.get('performance') or {}
    w = st.get('weather') or {}
    wsrc = st.get('weather_source') or {}
    scn = st.get('scenario') or {}
    xodr = st.get('xodr') or ''
    fps = st.get('fps')
    e = {
        'path': os.path.abspath(run_dir), 'id': os.path.basename(os.path.normpath(run_dir)),
        'rel': _rel(run_dir, workspace), 'name': run.get('name') or st.get('name') or os.path.basename(run_dir),
        'worker': run.get('worker'), 'status': run.get('status'), 'error': run.get('error'),
        'started': run.get('started'), 'finished': run.get('finished'),
        'kind': _kind(run, st, run_dir),
        'world': proj.get('world'), 'map': st.get('town') or (os.path.basename(os.path.dirname(xodr)) if xodr else None),
        'rig': proj.get('rig') or (st.get('rig') or {}).get('id'),
        'weather': wsrc.get('preset') or ('custom' if w else None),
        'weather_kind': weather_kind(w), 'rain': w.get('precipitation'), 'fog': w.get('fog_density'),
        'sun_altitude': w.get('sun_altitude_angle'), 'tod': tod_of(w.get('sun_altitude_angle')),
        'scenario': scn.get('id') or scn.get('name'), 'scenario_name': scn.get('name'),
        'values': scn.get('values') or {}, 'seed': st.get('seed', prov.get('seed')), 'fps': fps,
        'frames': perf.get('steps'), 'duration_s': perf.get('sim_s'),
        'speed_mean_kmh': None, 'speed_max_kmh': None, 'distance_m': None,
        'collisions': None, 'min_ttc_s': None, 'passed': None, 'teleports': [], 'odd_outside_s': None,
        'events': [], 'app_version': prov.get('app_version'), 'config_sha256': run.get('config_sha256'),
        'dashcam_route': bool(st.get('dashcam_route')), 'notes': len(run.get('notes') or []),
        'stat': _stat([os.path.join(run_dir, f) for f in FILES]),
    }
    fp = os.path.join(run_dir, 'frames.jsonl')
    if os.path.isfile(fp):
        fr = _scan_frames(fp)
        e['frames'] = fr['n'] if e['frames'] is None else e['frames']
        if e['duration_s'] is None and fr['n'] and fps:
            e['duration_s'] = round(fr['n'] / float(fps), 3)
        if fr['speed_mean'] is not None:
            e['speed_mean_kmh'] = round(fr['speed_mean'] * 3.6, 2)
            e['speed_max_kmh'] = round(fr['speed_max'] * 3.6, 2)
        if fr['distance_m'] is not None:
            e['distance_m'] = round(fr['distance_m'], 2)
        e['teleports'] = fr['teleports']
    sres = run.get('scenario') or {}
    crit = sres.get('criticality') or {}
    if crit:
        hit = [k for k, a in (crit.get('actors') or {}).items() if a.get('collision')]
        e['collisions'] = len(hit) + len(crit.get('other_collisions') or [])
        e['min_ttc_s'] = (crit.get('overall') or {}).get('min_ttc_s')
    if 'passed' in sres:
        e['passed'] = bool(sres['passed'])
    odd = run.get('odd_monitor') or {}
    if odd.get('time_outside_s') is not None:
        e['odd_outside_s'] = odd['time_outside_s']
    ev = []
    if e['collisions']:
        ev.append('collision')
    if e['teleports']:
        ev.append('teleport')
    if e['odd_outside_s']:
        ev.append('odd_exit')
    if e['passed'] is False:
        ev.append('failed_check')
    if run.get('faults_applied'):
        ev.append('fault')
    if any((run.get('trigger_rules') or {}).get('fired', {}).values()):
        ev.append('trigger')                      # a trigger rule fired (core.triggers)
    if e['status'] not in ('ok', None):
        ev.append(str(e['status']))
    e['events'] = ev
    return e


def _rel(path: str, workspace: str) -> str:
    p = os.path.abspath(path)
    if workspace:
        ws = os.path.abspath(workspace)
        if p == ws or p.startswith(ws + os.sep):
            return os.path.relpath(p, ws)
    return p


def key_of(path: str, workspace: str) -> str:
    """How the project refers to a run (tags, drive sets): relative to the workspace when inside it."""
    return _rel(path, workspace)


def path_of(key: str, workspace: str) -> str:
    return key if os.path.isabs(key) or not workspace else os.path.join(workspace, key)


# ================================================================================================ the index
def run_dirs(workspace: str, extra=()) -> list:
    """Every run folder of a workspace: runs/*, studies/*/runs/*, and any others given (the catalogue's)."""
    out, seen = [], set()

    def add(d):
        d = os.path.abspath(d)
        if d not in seen and os.path.isfile(os.path.join(d, 'run.json')):
            seen.add(d)
            out.append(d)
    if workspace:
        for sub in (os.path.join(workspace, 'runs'),):
            if os.path.isdir(sub):
                for n in sorted(os.listdir(sub)):
                    add(os.path.join(sub, n))
        st = os.path.join(workspace, 'studies')
        if os.path.isdir(st):
            for s in sorted(os.listdir(st)):
                r = os.path.join(st, s, 'runs')
                if os.path.isdir(r):
                    for n in sorted(os.listdir(r)):
                        add(os.path.join(r, n))
    for d in extra:
        add(d)
    return out


def _load_cache(workspace: str) -> dict:
    if not workspace:
        return {}
    try:
        with open(os.path.join(workspace, CACHE_FILE), encoding='utf-8') as fh:
            doc = json.load(fh)
        if doc.get('schema') == SCHEMA:
            return doc.get('runs') or {}
    except (OSError, ValueError):
        pass
    return {}


def _save_cache(workspace: str, runs: dict):
    if not workspace or not os.path.isdir(workspace):
        return
    from .paths import atomic_write
    try:
        atomic_write(os.path.join(workspace, CACHE_FILE),
                     json.dumps({'schema': SCHEMA, 'written': utc_now(), 'runs': runs}, separators=(',', ':')))
    except OSError:
        pass


def build(workspace: str, extra=(), cache: bool = True) -> list:
    """Every run's summary, newest first; unchanged runs come from the cache. Saved rule checks
    (<workspace>/rules/*/rules.json) add their violation counts."""
    old = _load_cache(workspace) if cache else {}
    new, out = {}, []
    for d in run_dirs(workspace, extra):
        stat = _stat([os.path.join(d, f) for f in FILES])
        c = old.get(d)
        e = c if c is not None and c.get('stat') == stat else summarise(d, workspace)
        if e is None:
            continue
        new[d] = e
        out.append(dict(e))
    if cache and new != old:
        _save_cache(workspace, new)
    vio = saved_rule_checks(workspace)
    for e in out:
        e['rule_violations'] = vio.get(e['path'])
        if e['rule_violations']:
            e['events'] = e['events'] + ['rule_violation']
    out.sort(key=lambda e: e.get('started') or '', reverse=True)
    return out


def saved_rule_checks(workspace: str) -> dict:
    """{run path: violations} from the rule checks saved under <workspace>/rules (the newest per run)."""
    out, when = {}, {}
    root = os.path.join(workspace, 'rules') if workspace else ''
    if not root or not os.path.isdir(root):
        return out
    for n in os.listdir(root):
        p = os.path.join(root, n, 'rules.json')
        try:
            mt = os.path.getmtime(p)
            with open(p, encoding='utf-8') as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        for r in doc.get('results') or ():
            run = r.get('run')
            if run and mt >= when.get(run, -1):
                when[run] = mt
                out[os.path.abspath(run)] = (r.get('totals') or {}).get('violations')
    return out


def with_tags(entries: list, project_doc: dict, workspace: str) -> list:
    tags = (project_doc or {}).get('run_tags') or {}
    for e in entries:
        e['tags'] = sorted(tags.get(key_of(e['path'], workspace)) or ())
    return entries


# ================================================================================================ queries
class QueryError(ValueError):
    pass


_CMP = re.compile(r'^(-?)([a-z_]+)(>=|<=|>|<|=|:)(.*)$')


def parse(text: str) -> list:
    """The terms of a query: [(negate, field or None, op, value)]. Raises QueryError on a malformed term."""
    try:
        toks = shlex.split(text or '')
    except ValueError as e:
        raise QueryError(f'unbalanced quotes: {e}') from None
    out = []
    for t in toks:
        m = _CMP.match(t)
        if m and (m.group(2) in FIELDS or m.group(2) in DATE_KEYS):
            neg, key, op, val = m.groups()
            if key in DATE_KEYS:
                if op != ':':
                    raise QueryError(f'{key} takes a date: {key}:2026-10-28')
                if not re.match(r'^\d{4}-\d{2}-\d{2}', val):
                    raise QueryError(f'{key}: {val!r} is not a date (YYYY-MM-DD)')
            elif FIELDS[key][1] == 'num' and op != ':':
                try:
                    val = float(val)
                except ValueError:
                    raise QueryError(f'{key}{op}{val}: not a number') from None
            elif FIELDS[key][1] != 'num' and op != ':':
                raise QueryError(f'{key} is not a number field: use {key}:{val}')
            out.append((bool(neg), key, op, val))
        else:
            neg = t.startswith('-') and len(t) > 1
            out.append((neg, None, ':', t[1:] if neg else t))
    return out


def _text(e: dict) -> str:
    return ' '.join(str(e.get(k) or '') for k in ('name', 'rel', 'world', 'map', 'weather', 'weather_kind', 'tod',
                                                  'scenario', 'scenario_name', 'kind', 'status')).lower() + ' ' + \
        ' '.join(e.get('tags') or ()).lower()


def _term(e: dict, key, op, val) -> bool:
    if key is None:
        return str(val).lower() in _text(e)
    if key in DATE_KEYS:
        s = (e.get('started') or '')[:len(val)]
        if not s:
            return False
        return s >= val if key == 'after' else s < val
    field, kind = FIELDS[key]
    v = e.get(field)
    if kind == 'set':
        want = str(val).lower()
        return any(want == str(x).lower() or (want and want in str(x).lower()) for x in (v or ()))
    if kind == 'num' and op != ':':
        if v is None:
            return False
        v = float(v)
        return {'>': v > val, '<': v < val, '>=': v >= val, '<=': v <= val, '=': abs(v - val) < 1e-9}[op]
    if kind == 'num':
        try:
            return v is not None and abs(float(v) - float(val)) < 1e-9
        except ValueError:
            return False
    s = str(v or '').lower()
    if key == 'weather':                   # a preset or the kind (rain, cloudy …)
        s += ' ' + str(e.get('weather_kind') or '')
    return str(val).lower() in s


def match(e: dict, terms: list) -> bool:
    return all(_term(e, k, op, v) != neg for neg, k, op, v in terms)


def search(entries: list, text: str) -> list:
    terms = parse(text)
    return [e for e in entries if match(e, terms)]


SORT_KEYS = {'date': 'started', 'name': 'name', 'world': 'world', 'weather': 'weather', 'duration': 'duration_s',
             'status': 'status', 'frames': 'frames'}


def sort(entries: list, by: str = 'date', reverse: bool | None = None) -> list:
    k = SORT_KEYS.get(by, by)
    rev = (by == 'date') if reverse is None else reverse

    have = [e for e in entries if e.get(k) is not None]
    have.sort(key=lambda e: e[k].lower() if isinstance(e[k], str) else e[k], reverse=rev)
    return have + [e for e in entries if e.get(k) is None]           # not recorded: last either way


# ================================================================================================ drive sets
def resolve_set(ds: dict, entries: list, workspace: str) -> list:
    """The run paths of a drive set: a fixed list (those still on disk) or a saved query over the index."""
    if ds.get('kind') == 'query':
        return [e['path'] for e in search(entries, ds.get('query') or '')]
    have = {e['path'] for e in entries}
    out = []
    for k in ds.get('runs') or ():
        p = os.path.abspath(path_of(k, workspace))
        if p in have or os.path.isfile(os.path.join(p, 'run.json')):
            out.append(p)
    return out


def describe_set(ds: dict) -> str:
    if ds.get('kind') == 'query':
        return f'query: {ds.get("query") or "(everything)"}'
    return f'{len(ds.get("runs") or ())} runs'


def main_runs(argv) -> int:
    """falcon runs search PROJECT|WORKSPACE [QUERY…] [--sort KEY] [--json]: search the runs."""
    import argparse
    from .paths import PROJECT_SUFFIX, workspace_for
    ap = argparse.ArgumentParser(prog='falcon runs search', description='Search the runs of a project\'s workspace '
                                                                         '(docs/runs.md).')
    ap.add_argument('workspace', help='a project file (*.falcon.json) or its workspace folder')
    ap.add_argument('query', nargs='*')
    ap.add_argument('--sort', default='date', choices=sorted(SORT_KEYS))
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    ws = workspace_for(a.workspace) if a.workspace.endswith(PROJECT_SUFFIX) else a.workspace
    try:
        rows = sort(search(build(ws), ' '.join(a.query)), a.sort)
    except QueryError as e:
        print(f'falcon runs search: {e}')
        return 2
    if a.json:
        print(json.dumps(rows, indent=1, default=str))
        return 0
    for e in rows:
        print(f'{(e.get("started") or "")[:16]:16}  {e["status"] or "?":9}  {e.get("world") or e.get("map") or "?":14} '
              f'{e.get("weather") or "?":14} {e.get("duration_s") or 0:7.1f} s  {e["rel"]}')
    print(f'{len(rows)} run(s)')
    return 0
