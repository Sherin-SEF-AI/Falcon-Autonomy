"""`falcon assistant tools [--json]` and `falcon assistant call NAME [JSON] [--project P] [--allow-act]`: the
assistant's tools from the command line, for scripts and checks — the same registry, validation, privacy and
redaction the agent goes through (tool_api.call).

From the command line a read tool always runs; an act tool only with --allow-act (a project edit is then saved to
the project file; a job is run to its end in this process, its progress on stderr, Ctrl+C cancelling it); a trash tool
never (it always asks the user, in the window); a window tool cannot run. The result is printed as JSON on stdout.
Exit codes: 0 ok, 2 an error result or a usage error, 3 a queued job did not finish done.
"""
from __future__ import annotations

import argparse
import json
import sys


def _table(rows) -> str:
    w = max((len(r['name']) for r in rows), default=4)
    out = []
    for r in rows:
        first = r['description'].split('. ')[0]
        out.append(f'{r["name"]:{w}}  {r["group"]:9} {r["tier"]:5}{" gui" if r["gui"] else "    "}  {first[:90]}')
    return '\n'.join(out)


DOC_HEAD = """# Assistant tools

<!-- Generated from the tool registry by `falcon assistant tools --doc > docs/assistant-tools.md`; a test fails when
this file is stale. Do not edit by hand. -->

Every action Falcon's assistant can take is one of these tools, on the code paths the GUI pages and the command line
use. **Tier** is what the permission modes key on: *read* looks only; *act* changes something reversibly — an undoable
project edit, a cancellable job, a file in the workspace, the window's view; *trash* moves runs into the workspace's
trash folder (restorable) and always asks. Tools marked *window* need Falcon's window. What no tool can do, in any
mode: delete runs or projects, change settings or ports, start or stop a server outside the job queue, run a shell
command, push or post anywhere, download a model, or send images, video, lidar, audio or anything derived from the
user's dashcam footage to the model.

From the command line: `falcon assistant tools [--json]` lists them; `falcon assistant call NAME '{"arg": …}'
--project P.falcon.json` runs a read tool (`--allow-act` lets an act tool run; trash tools never run from there).
"""
GROUP_TITLES = {'project': 'Project', 'runs': 'Runs', 'jobs': 'Jobs', 'analysis': 'Analysis', 'lab': 'Lab',
                'ui': 'Window', 'knowledge': 'Knowledge', 'system': 'System'}


def doc_markdown() -> str:
    """docs/assistant-tools.md, generated from the registry: the model's schema size per group, then per group a table
    of name, tier, arguments (with their full descriptions) and the full description of what the tool does."""
    from .tool_api import GROUPS
    rs = rows()
    out = [DOC_HEAD]
    counts = {g: sum(1 for r in rs if r['group'] == g) for g in GROUPS}
    out.append(f'{len(rs)} tools: ' + ', '.join(f'{GROUP_TITLES[g].lower()} {counts[g]}' for g in GROUPS) + '.\n')
    toks = schema_tokens()
    out.append('What the model is sent is each tool\'s one-line description and a compact schema (tools/_compact.py); '
               'the texts below are the full ones. Schema size, estimated as characters / 4: ' +
               ', '.join(f'{GROUP_TITLES[g].lower()} {toks[g]}' for g in GROUPS) + f'; all {toks["all"]} tokens.\n')
    for g in GROUPS:
        out.append(f'## {GROUP_TITLES[g]}\n')
        out.append('| tool | tier | arguments | what it does |')
        out.append('|---|---|---|---|')
        for r in rs:
            if r['group'] != g:
                continue
            props = r['details_parameters'].get('properties') or {}
            req = set(r['details_parameters'].get('required') or ())
            args = '<br>'.join(f'`{k}`' + ('' if k in req else '?') + (f' {v["description"]}' if v.get('description')
                                                                         else '') for k, v in props.items()) or '—'
            args = args.replace('|', '\\|')
            tier = r['tier'] + (', window' if r['gui'] else '')
            desc = r['details'].replace('|', '\\|').replace('\n', ' ')
            out.append(f'| `{r["name"]}` | {tier} | {args} | {desc} |')
        out.append('')
    return '\n'.join(out)


def schema_tokens() -> dict:
    """The schemas the model is sent, in tokens estimated as compact-JSON characters / 4: per group and all."""
    from . import tools  # noqa: F401 (registers them)
    from .tool_api import GROUPS, REGISTRY, schemas
    size = {g: 0 for g in GROUPS}
    for t in REGISTRY.values():
        size[t.group] += len(json.dumps(schemas([t.name]), separators=(',', ':'), ensure_ascii=False))
    out = {g: v // 4 for g, v in size.items()}
    out['all'] = len(json.dumps(schemas(), separators=(',', ':'), ensure_ascii=False)) // 4
    return out


def rows() -> list:
    """Every tool: what the model is sent (description, parameters) and the full texts (details, details_parameters:
    tools._compact.DETAILS)."""
    from . import tools  # noqa: F401 (registers them)
    from .tool_api import REGISTRY
    from .tools._compact import DETAILS
    return [{'name': t.name, 'group': t.group, 'tier': t.tier, 'gui': t.gui, 'description': t.description,
             'parameters': t.params, 'details': DETAILS.get(t.name, {}).get('description', t.description),
             'details_parameters': DETAILS.get(t.name, {}).get('params', t.params)} for t in REGISTRY.values()]


def _wait_jobs(ctx, before: set) -> int:
    """Run the jobs this call queued to their end (the CLI's own loop: poll the queue)."""
    import signal
    import time
    from ..core.jobs import FINAL
    q = ctx.queue
    mine = [j for j in q.jobs if j.id not in before]
    if not mine:
        return 0
    hits = {'n': 0}

    def on_int(*_):
        hits['n'] += 1
        for j in mine:
            if j.state not in FINAL:
                q.cancel(j) if hits['n'] == 1 else (j.handle.kill() if j.handle is not None else None)
    old = signal.signal(signal.SIGINT, on_int)
    last = {}
    try:
        while any(j.state not in FINAL for j in mine):
            q.poll()
            for j in mine:
                line = j.summary()
                if last.get(j.id) != line:
                    print(f'[{j.spec.title}] {line}', file=sys.stderr)
                    last[j.id] = line
            time.sleep(0.25)
    finally:
        signal.signal(signal.SIGINT, old)
    for j in mine:
        print(f'[{j.spec.title}] {j.state}', file=sys.stderr)
    return 0 if all(j.state == 'done' for j in mine) else 3


def main_assistant(argv) -> int:
    ap = argparse.ArgumentParser(prog='falcon assistant',
                                 description='The assistant\'s tools (docs/assistant-tools.md).')
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('tools', help='list the tools: name, group, tier, description')
    t.add_argument('--json', action='store_true')
    t.add_argument('--doc', action='store_true', help='print docs/assistant-tools.md (generated from the registry)')
    c = sub.add_parser('call', help='call one tool with JSON arguments')
    c.add_argument('name')
    c.add_argument('args', nargs='?', default='{}', help='the arguments as a JSON object (default {})')
    c.add_argument('--project', default='', help='the project file (*.falcon.json)')
    c.add_argument('--allow-act', action='store_true', help='let an act tool run (edits are saved, jobs run)')
    c.add_argument('--fake-carla', default='', help=argparse.SUPPRESS)
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    rs = rows()
    if a.cmd == 'tools':
        if a.doc:
            sys.stdout.write(doc_markdown())
        elif a.json:
            print(json.dumps(rs, indent=1))
        else:
            print(_table(rs))
            print(f'{len(rs)} tools')
        return 0
    from .tool_api import NEVER, REGISTRY, call
    tl = REGISTRY.get(a.name)
    if a.name in NEVER or tl is None:
        print(json.dumps(call(None, a.name, {})))
        return 2
    if tl.tier == 'trash':
        print(json.dumps({'error': f'{a.name} moves runs to the trash: only from the window, where the user is asked'}))
        return 2
    if tl.tier == 'act' and not a.allow_act:
        print(json.dumps({'error': f'{a.name} changes something: pass --allow-act to let it'}))
        return 2
    try:
        args = json.loads(a.args)
    except ValueError as e:
        print(json.dumps({'error': f'the arguments are not JSON: {e}'}))
        return 2
    from . import tools
    from ..core.project import ProjectError
    try:
        ctx = tools.context(a.project, a.fake_carla)
    except (OSError, ProjectError) as e:
        print(json.dumps({'error': f'cannot open the project: {e}'}))
        return 2
    before = {j.id for j in ctx.queue.jobs}
    code = 0
    try:
        res = call(ctx, a.name, args)
        print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
        if 'error' in res:
            code = 2
        elif tl.tier == 'act':
            code = _wait_jobs(ctx, before)
    finally:
        ctx.service.shutdown()
    return code
