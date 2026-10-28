"""Live session configs (standard library only): a drive config without a step count, plus the live link's socket,
the view, the driving mode and where recordings go. The GUI builds them; the live worker runs them."""
from __future__ import annotations

from .drive import drive_config

LIVE_SCHEMA = 'falcon.live_config/1'
MODES = ('manual', 'autopilot', 'policy')
VIEWS = ('chase', 'free', 'sensor')


def live_config(project, world_id: str, rig_id: str, out_dir: str, port: int, tm_port: int, socket_path: str,
                recordings_dir: str, fps: float = 20.0, seed: int = 0, mode: str = 'autopilot',
                view: dict | None = None, policy: str = '', workspace: str = '', host: str = '127.0.0.1') -> dict:
    cfg = drive_config(project, world_id, rig_id, out_dir, port, tm_port, seconds=1.0, fps=fps, seed=seed,
                       host=host, name='live session', workspace=workspace)
    cfg.pop('steps', None)
    cfg.pop('max_wall_s', None)
    if mode not in MODES:
        raise ValueError(f'mode {mode!r} is not one of {MODES}')
    cfg.update(schema=LIVE_SCHEMA, socket=socket_path, recordings_dir=recordings_dir, mode=mode,
               view=dict(view or {'kind': 'chase', 'width': 1280, 'height': 720, 'fov': 90.0}),
               policy=policy, warmup_steps=5)
    return cfg
