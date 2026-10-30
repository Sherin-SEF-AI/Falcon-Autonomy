"""Knowledge tools (read): the built-in manual searched offline (core.manual: BM25 over its sections), the command
line's commands (docs/cli.md's table), what a setting does (core.settings.EXPLAIN), and the assistant's own tools. No
network."""
from __future__ import annotations

import re

from ..tool_api import ToolError, tool
from . import _common as C


@tool('knowledge_manual', 'knowledge', 'read',
      'Search Falcon\'s built-in manual (docs/*.md, offline, BM25 over sections): the best sections with their file, '
      'heading trail, link (file#anchor) and text (trimmed). Use the manual\'s own words; read a section in full '
      'with section_chars up to 6000.',
      C.obj({'query': C.s('what to look for'), 'limit': C.i('sections (default 3)', minimum=1, maximum=8),
             'section_chars': C.i('characters of text per section (default 1200)', minimum=200, maximum=6000)},
            ['query']))
def knowledge_manual(ctx, query, limit=3, section_chars=1200):
    from ...core import manual as M
    if not M.docs_dir():
        raise ToolError('the manual is not bundled with this build')
    hits = M.search(query, int(limit))
    return {'query': query, 'sections': [{'file': s['file'], 'heading': ' ▸ '.join(s['trail']),
                                          'link': f'docs/{s["file"]}' + (f'#{s["anchor"]}' if s['anchor'] else ''),
                                          'score': round(sc, 2), 'text': C.clip(s['text'], int(section_chars))}
                                         for sc, s in hits],
            'found': bool(hits)}


def _cli_rows() -> list:
    from ...core import manual as M
    import os
    p = os.path.join(M.docs_dir(), 'cli.md')
    rows = []
    try:
        with open(p, encoding='utf-8') as fh:
            for line in fh:
                m = re.match(r'^\|\s*`(falcon[^`]*)`(.*?)\|\s*(.+?)\s*\|\s*$', line)
                if m:
                    cmd = m.group(1) + re.sub(r'`', '', m.group(2))
                    rows.append({'command': re.sub(r'\s+', ' ', cmd).strip(),
                                 'does': re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', m.group(3))})
    except OSError:
        pass
    return rows


@tool('knowledge_cli', 'knowledge', 'read',
      'Falcon\'s command-line commands from the manual\'s CLI reference (docs/cli.md): each command with its options '
      'and what it does, optionally filtered by a word. For telling the user what to run — the assistant itself '
      'never runs a shell.',
      C.obj({'filter': C.s('only commands or descriptions containing this')}))
def knowledge_cli(ctx, filter=''):
    rows = [r for r in _cli_rows() if filter.lower() in (r['command'] + ' ' + r['does']).lower()]
    return C.capped([{'command': C.clip(r['command'], 300), 'does': C.clip(r['does'], 400)} for r in rows], 30,
                    'narrow with filter')


@tool('knowledge_setting', 'knowledge', 'read',
      'What a Falcon setting does and where it is set (Settings page), with its current value — or every setting '
      'when none is named. Private settings (recent projects, the hardware device, a cloud model) are explained '
      'without their values. The assistant never changes settings.',
      C.obj({'key': C.s('a setting key (empty: all)')}))
def knowledge_setting(ctx, key=''):
    from ...core.settings import DEFAULTS, EXPLAIN, PRIVATE_SETTINGS
    st = ctx.settings if ctx.win is None else ctx.win.settings
    keys = [key] if key else list(DEFAULTS)
    out = []
    for k in keys:
        if k not in EXPLAIN:
            raise ToolError(f'no setting {k!r} (keys: {", ".join(DEFAULTS)})')
        where, what = EXPLAIN[k]
        row = {'key': k, 'where': where, 'what': what, 'default': DEFAULTS[k] if k not in PRIVATE_SETTINGS else None}
        if k in PRIVATE_SETTINGS:
            row['value'] = '(private: not shown)'
        elif st is not None:
            v = st.get(k)
            row['value'] = ({'source': v.get('source'), 'measured': bool(v.get('points'))} if k == 'measured' and
                            isinstance(v, dict) else v)
        out.append(row)
    return {'settings': out}


@tool('knowledge_tools', 'knowledge', 'read',
      'The assistant\'s own tools by group and tier (read: looks only; act: changes something reversibly — an '
      'undoable edit, a cancellable job, the view; trash: moves runs to the trash, always asks), with their one-line '
      'purpose. What is never possible: deleting runs or projects, changing settings or ports, starting or stopping '
      'servers outside the job queue, running a shell, pushing or posting, downloading models, seeing images or '
      'dashcam-derived data.',
      C.obj({'group': C.s('one group', enum=['project', 'runs', 'jobs', 'analysis', 'lab', 'ui', 'knowledge', 'system'])}))
def knowledge_tools(ctx, group=''):
    from ..tool_api import NEVER, REGISTRY
    rows = [{'name': t.name, 'group': t.group, 'tier': t.tier, 'gui': t.gui,
             'purpose': t.description.split('. ')[0][:200]} for t in REGISTRY.values() if not group or t.group == group]
    return {'tools': rows, 'never': list(NEVER)}
