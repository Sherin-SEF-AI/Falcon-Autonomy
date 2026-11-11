"""The MCP server's resources: the project's state by URI, read through the same tools (and so the same privacy guard
and redaction) as a tool call. Ported from Vantage mcp/catalog.py (GPL).

    falcon://status       where the calls go (window or file), the steps made from here, jobs
    falcon://overview     the project at a glance (project_overview)
    falcon://runs         the recorded runs, newest first (runs_search)
    falcon://queue        the job queue: counts, why jobs wait, who holds the locks (system_queue)
    falcon://run/{run}    one run compactly (runs_get), e.g. falcon://run/runs/drive-0003
"""
from __future__ import annotations

import json
import urllib.parse


class NotFound(Exception):
    pass


RESOURCES = [
    {'uri': 'falcon://status', 'name': 'status', 'title': 'Where calls go',
     'description': 'Falcon\'s window or a project file, the steps made from here, unsaved changes, jobs.',
     'mimeType': 'application/json'},
    {'uri': 'falcon://overview', 'name': 'overview', 'title': 'Project overview',
     'description': 'Worlds, rigs, scenarios, profiles, rules; runs, evaluations and studies made.',
     'mimeType': 'application/json'},
    {'uri': 'falcon://runs', 'name': 'runs', 'title': 'Recorded runs',
     'description': 'The project\'s recorded runs, newest first (runs_search narrows them).',
     'mimeType': 'application/json'},
    {'uri': 'falcon://queue', 'name': 'queue', 'title': 'Job queue',
     'description': 'Jobs by state, why jobs wait, and who holds the CARLA and GPU locks.',
     'mimeType': 'application/json'},
]
TEMPLATES = [
    {'uriTemplate': 'falcon://run/{run}', 'name': 'run', 'title': 'One run',
     'description': 'One run compactly: settings, performance, summary, scenario result (a run id from '
                    'falcon://runs, e.g. runs/drive-0003).', 'mimeType': 'application/json'},
]
_FIXED = {'falcon://status': ('falcon_status', {}), 'falcon://overview': ('project_overview', {}),
          'falcon://runs': ('runs_search', {'sort': 'date', 'limit': 25}), 'falcon://queue': ('system_queue', {})}


def resolve(uri: str) -> tuple:
    """(tool, arguments) a resource is read with; NotFound for an unknown URI."""
    uri = str(uri or '')
    if uri in _FIXED:
        name, args = _FIXED[uri]
        return name, dict(args)
    if uri.startswith('falcon://run/'):
        run = urllib.parse.unquote(uri[len('falcon://run/'):]).strip()
        if run:
            return 'runs_get', {'run': run}
    raise NotFound(f'no resource {uri!r} (resources/list and resources/templates/list list them)')


def contents(uri: str, result: dict) -> list:
    """resources/read's contents from a tool's MCP result: its JSON as text."""
    st = result.get('structuredContent')
    body = json.dumps(st, ensure_ascii=False, default=str) if st is not None else next(
        (c.get('text', '') for c in result.get('content', []) if c.get('type') == 'text'), '')
    return [{'uri': uri, 'mimeType': 'application/json', 'text': body}]
