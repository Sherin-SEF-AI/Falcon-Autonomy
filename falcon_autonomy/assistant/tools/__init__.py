"""The assistant's tools: importing this package registers every one in tool_api.REGISTRY (docs/assistant-tools.md is
generated from it). One module per group — project, runs, jobs, analysis, ui, knowledge, system — each calling the code
the GUI pages and the CLI call; _common holds what they share (ids to workspace paths, privacy checks, edits, jobs,
compact results); _compact holds each tool's one-line description for the model (the full texts are the doc).

context(project_path, fake_carla) builds a ToolContext for the command line (`falcon assistant call`): the project
loaded, the user's settings, a job queue with the subprocess launcher and a CarlaService on the project's workspace.
"""
from __future__ import annotations

from ..tool_api import REGISTRY
from . import analysis, jobs, knowledge, project, runs, system, ui  # noqa: F401 (registers the tools)
from ._compact import compact_registry

compact_registry(REGISTRY)          # what the model is sent: one-line descriptions, short argument texts (_compact)


def context(project_path: str = '', fake_carla: str = '', emit=None):
    """A ToolContext for the command line (no window)."""
    from ...core.jobs import JobQueue, SubprocessLauncher
    from ...core.project import Project
    from ...core.settings import MEMORY_RESERVE_GB, Settings
    from ...sim.service import CarlaService
    from ..privacy import redact
    from ..tool_api import ToolContext
    st = Settings()
    proj = Project.load(project_path) if project_path else None
    q = JobQueue(SubprocessLauncher(), memory_reserve_gb=MEMORY_RESERVE_GB)
    ws = proj.workspace if proj is not None else ''
    svc = CarlaService(st, q, ws, fake_carla=fake_carla)
    return ToolContext(project=proj, settings=st, queue=q, service=svc, win=None, workspace=ws,
                       emit=emit or (lambda kind, payload: None), redact=redact)
