"""The assistant's tool API: what a tool is, the registry every tool is in, and how a call is checked and run (standard
library + jsonschema; no Qt, no network).

A tool is one thing the assistant can do in Falcon, on the code paths the GUI and the CLI use: read the project or a
run, edit the project (through its undoable edits), queue a job, drive the window. Each tool declares a JSON Schema for
its arguments (validated before it runs), a tier, and for the acting tiers what an approval card shows: an estimate of
the cost and a preview of the change.

Tiers (what the permission modes key on):
    read    looks, never changes anything (the project, the workspace, the window's state apart from what is shown)
    act     changes something, reversibly: a project edit (undoable), a job queued (cancellable), a file written into
            the workspace (an export, a report), the window navigated
    trash   moves runs into the workspace's trash folder (restorable from it); always asks, in every mode

Never, in any mode (NEVER): deleting runs or projects outright, changing Falcon's settings (its ports, its server, the
key), starting or stopping a server outside the job queue or touching a server the user started, running a shell
command, pushing or posting anywhere, downloading a model without the user accepting its licence, and sending images,
video, lidar, audio or anything derived from the user's dashcam footage to the model. No tool exists for these; call()
refuses a name in NEVER even if one were registered.

A tool's function takes a ToolContext and the validated arguments and returns a compact JSON-serialisable dict. It
raises ToolError for a refusal the model should read (a missing run, an invalid scenario); call() turns that into
{'error': ...}. Anything else that goes wrong is reported as {'error': ..., 'internal': True} with the exception's
type — never its traceback, which can carry paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

TIERS = ('read', 'act', 'trash')
GROUPS = ('project', 'runs', 'jobs', 'analysis', 'ui', 'knowledge', 'system')
NEVER = ('delete_run', 'delete_project', 'change_settings', 'set_port', 'start_server', 'stop_server', 'shell',
         'git_push', 'download_model', 'send_image')


class ToolError(Exception):
    """A refusal the model reads and can act on (a missing run, a bad scenario id, a job that is not running)."""


@dataclass
class ToolContext:
    """What a tool runs against. win is the MainWindow in the GUI and None from the command line; ui-group tools need
    it (Tool.gui). emit(kind, payload) reports progress or app events to whoever drives the agent (the dock, a log).
    redact(obj) is applied to every result before the model sees it (core of the privacy rules, assistant.privacy)."""
    project: object = None
    settings: object = None
    queue: object = None
    service: object = None
    win: object = None
    workspace: str = ''
    emit: Callable = field(default=lambda kind, payload: None)
    redact: Callable = field(default=lambda obj: obj)


@dataclass(frozen=True)
class Tool:
    name: str                               # unique, snake_case, '<group>_<what>' (e.g. 'runs_search')
    group: str                              # GROUPS
    tier: str                               # TIERS
    description: str                        # what the model reads: what it does, what it returns, its limits
    params: dict                            # JSON Schema of the arguments (type object)
    fn: Callable                            # fn(ctx, **args) -> dict
    estimate: Callable | None = None        # estimate(ctx, **args) -> {'seconds': float | None, 'what': str}
    preview: Callable | None = None         # preview(ctx, **args) -> str: the command, or the change as a diff
    gui: bool = False                       # needs the window (ctx.win)


REGISTRY: dict = {}


def register(t: Tool) -> Tool:
    if t.tier not in TIERS:
        raise ValueError(f'tool {t.name}: tier {t.tier!r} is not one of {TIERS}')
    if t.group not in GROUPS:
        raise ValueError(f'tool {t.name}: group {t.group!r} is not one of {GROUPS}')
    if t.name in NEVER:
        raise ValueError(f'tool {t.name}: that is never allowed (tool_api.NEVER)')
    if t.name in REGISTRY:
        raise ValueError(f'tool {t.name} is registered twice')
    if (t.params or {}).get('type') != 'object':
        raise ValueError(f'tool {t.name}: params must be a JSON Schema of type object')
    REGISTRY[t.name] = t
    return t


def tool(name: str, group: str, tier: str, description: str, params: dict | None = None, **kw):
    """Decorator: register the function as a tool. params defaults to no arguments."""
    def deco(fn):
        register(Tool(name, group, tier, description, params or {'type': 'object', 'properties': {}}, fn, **kw))
        return fn
    return deco


def schemas(names=None, gui: bool = True) -> list:
    """The tools in OpenAI's function-calling form (Groq's API takes the same), every registered one or those named;
    gui=False leaves out the tools that need the window."""
    out = []
    for t in REGISTRY.values():
        if (names is not None and t.name not in names) or (t.gui and not gui):
            continue
        out.append({'type': 'function', 'function': {'name': t.name, 'description': t.description,
                                                      'parameters': t.params}})
    return out


def check_args(t: Tool, args) -> list:
    """The arguments' problems against the tool's schema, each as 'path: message' (empty: valid)."""
    import jsonschema
    if not isinstance(args, dict):
        return ['the arguments must be a JSON object']
    v = jsonschema.Draft202012Validator(t.params)
    return [f'{"/".join(str(p) for p in e.absolute_path) or "(arguments)"}: {e.message}'
            for e in sorted(v.iter_errors(args), key=lambda e: list(e.absolute_path))]


def call(ctx: ToolContext, name: str, args) -> dict:
    """Run a tool: refuse what is never allowed or unknown, validate the arguments, run it, redact its result. The
    caller (the agent loop) has already applied the permission mode — this does not ask anyone."""
    if name in NEVER:
        return {'error': f'{name} is never allowed (docs/assistant.md, "What the assistant never does")'}
    t = REGISTRY.get(name)
    if t is None:
        return {'error': f'no tool named {name!r}'}
    if t.gui and ctx.win is None:
        return {'error': f'{name} needs Falcon\'s window (not available from the command line)'}
    errs = check_args(t, args if args is not None else {})
    if errs:
        return {'error': 'invalid arguments', 'problems': errs}
    try:
        res = t.fn(ctx, **(args or {}))
    except ToolError as e:
        return {'error': str(e)}
    except Exception as e:                    # noqa: BLE001 (reported to the model, without a traceback)
        return {'error': f'{type(e).__name__} in {name}', 'internal': True}
    return ctx.redact(res if isinstance(res, dict) else {'result': res})
