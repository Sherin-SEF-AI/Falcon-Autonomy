"""Every command's --help works under the system python3 (3.14's argparse checks each help string when an argument is
added and refuses a bare %: one such string took a whole command line down, 7f4e27f), and no help= string literal in
the package has a bare % (a lint that runs under any Python).

The --help walk runs in one subprocess of `python3` (this file run as a script): `falcon CMD --help` for every
top-level subcommand of cli.parser() (and the routes main() dispatches before argparse: dashcam route|incident|
ingest|nearmiss, runs search), then `falcon CMD SUB --help` for every first-level choice a command's own parser
declares (a subparser, or a positional with choices); every ArgumentParser constructed on the way — nested subparsers
included — is formatted (help and usage). A 4.0 command whose module has not landed must say "not in this build" and
exit 2; anything else must exit 0.
"""
from __future__ import annotations

import ast
import glob
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA_ROUTES = (['dashcam', 'route'], ['dashcam', 'incident'], ['dashcam', 'ingest'], ['dashcam', 'nearmiss'],
                ['runs', 'search'])
SKIP_SECOND = {'gui', 'project'}          # project: its choices are positional words, covered by its own --help


def _walk() -> dict:
    """Run in the subprocess: {'calls': [{argv, code, err}], 'broken': [{parser, error}]}."""
    import argparse
    import contextlib
    import io
    sys.path.insert(0, ROOT)
    from falcon_autonomy import cli

    made: list = []
    init = argparse.ArgumentParser.__init__

    def record(self, *a, **k):
        init(self, *a, **k)
        made.append(self)
    argparse.ArgumentParser.__init__ = record
    calls, broken, seen = [], [], set()

    def run(argv):
        out, err = io.StringIO(), io.StringIO()
        start = len(made)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = cli.main(list(argv))
        except SystemExit as e:
            code = e.code
        except Exception as e:                      # noqa: BLE001 (reported: a badly formed help string, …)
            code = f'{type(e).__name__}: {e}'
        calls.append({'argv': list(argv), 'code': 0 if code is None else code, 'err': err.getvalue()[-400:]})
        new = made[start:]
        for p in new:
            if id(p) in seen:
                continue
            seen.add(id(p))
            try:
                p.format_help()
                p.format_usage()
            except Exception as e:                  # noqa: BLE001
                broken.append({'parser': p.prog, 'error': f'{type(e).__name__}: {e}'})
        return new

    top = [a for a in cli.parser()._actions if isinstance(a, argparse._SubParsersAction)][0]
    seconds = []
    for cmd in top.choices:
        new = run([cmd, '--help'])
        if cmd in SKIP_SECOND or not new or new[0].prog == 'falcon':
            continue                                 # parsed by cli.parser() itself: walked as a whole
        for act in new[0]._actions:                  # the command's own root parser: its first positional
            if act.option_strings:
                continue
            seconds += [[cmd, str(choice)] for choice in (act.choices or ())]
            break
    for argv in list(EXTRA_ROUTES) + seconds:
        run(argv + ['--help'])
    return {'calls': calls, 'broken': broken}


def test_every_command_help_under_python3(tmp_path):
    py = shutil.which('python3')
    if py is None:
        pytest.skip('no python3 on PATH')
    env = dict(os.environ, XDG_CONFIG_HOME=str(tmp_path / 'config'), QT_QPA_PLATFORM='offscreen',
               PYTHONPATH=ROOT + os.pathsep + os.environ.get('PYTHONPATH', ''))
    r = subprocess.run([py, '-W', 'error::SyntaxWarning', os.path.abspath(__file__), 'walk'], cwd=str(tmp_path),
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    doc = json.loads(r.stdout.strip().splitlines()[-1])
    version = doc.pop('python')
    from falcon_autonomy import cli
    lazy = set(cli.LAZY) | {f'dashcam {x}' for x in cli.DASHCAM_LAZY}
    bad = []
    for c in doc['calls']:
        name = ' '.join(c['argv'][:2]) if c['argv'][0] == 'dashcam' else c['argv'][0]
        if c['code'] == 0:
            continue
        if c['code'] == 2 and name in lazy and 'is not in this build' in c['err']:
            continue
        bad.append(c)
    assert not bad, json.dumps(bad, indent=1)[:4000]
    assert not doc['broken'], doc['broken']
    assert len(doc['calls']) > 60, len(doc['calls'])
    assert version >= [3, 12], version


# ------------------------------------------------------------------------------------------- the % lint
def _bare_percent(text: str) -> bool:
    """A % that is neither %% nor the start of %(name)s."""
    i = 0
    while True:
        i = text.find('%', i)
        if i < 0:
            return False
        nxt = text[i + 1:i + 2]
        if nxt == '%':
            i += 2
            continue
        if nxt != '(':
            return True
        i += 1


def _help_strings(node):
    """The string literals a help= value holds (a constant, an f-string's literal parts, a concatenation)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value
    elif isinstance(node, ast.JoinedStr):
        for v in node.values:
            yield from _help_strings(v)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        yield from _help_strings(node.left)
        yield from _help_strings(node.right)


def test_no_bare_percent_in_help_strings():
    bad = []
    for path in sorted(glob.glob(os.path.join(ROOT, 'falcon_autonomy', '**', '*.py'), recursive=True)):
        with open(path, encoding='utf-8') as fh:
            tree = ast.parse(fh.read(), path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != 'help':
                    continue
                for s in _help_strings(kw.value):
                    if _bare_percent(s):
                        bad.append(f'{os.path.relpath(path, ROOT)}:{kw.value.lineno}: {s!r}')
    assert not bad, '\n'.join(bad)


def test_the_lint_itself():
    assert _bare_percent('a 95 % interval') and _bare_percent('100%')
    assert not _bare_percent('a 95 %% interval') and not _bare_percent('default %(default)s')
    tree = ast.parse("f(help=f'{x} 5 %' 'y')")
    call = tree.body[0].value
    assert any(_bare_percent(s) for s in _help_strings(call.keywords[0].value))


if __name__ == '__main__' and sys.argv[1:] == ['walk']:
    res = _walk()
    res['python'] = list(sys.version_info[:2])
    sys.stdout.flush()
    print(json.dumps(res))
