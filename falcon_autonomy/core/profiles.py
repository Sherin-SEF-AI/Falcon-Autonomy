"""Performance profiles: standard, and low_vram for a GPU with 8 GB of video memory (standard library only).

A profile is a named set of settings values. `falcon profile set low_vram` (or Settings ▸ Performance) writes its
values into the settings, keeping the ones it replaced in profile-saved.json beside settings.json; `set standard` puts
those back. Every part of Falcon then reads the settings as it always did — the server's quality, memory cap and
texture pool when it starts, the render budget a run is checked against, the precision a model evaluation runs at —
and a value changed by hand afterwards stays changed (`falcon profile show` lists it as a difference).

low_vram:

    server_quality        Low      CARLA's Low quality level (-quality-level=Low)
    server_memory_gb      10       the server's memory cap (its scope; RAM, not video memory)
    render_budget_mpx     96       runs rendering more megapixels per step are refused before they start
    texture_pool_mb       512      Unreal's texture streaming pool (r.Streaming.PoolSize; UE4's default is 1000 MB),
                                   passed as -ini:Engine:[/Script/Engine.RendererSettings]:r.Streaming.PoolSize=512
                                   and -ExecCmds=r.Streaming.PoolSize 512 — which of the two CARLA 0.9.16's packaged
                                   server honours has not been measured
    inference_precision   auto     a model evaluation runs FP16 when its AP stays within 0.5 points of FP32 on the
                                   evaluated frames, else FP32 (core.models.fp16_gate)

and caps for 3D reconstruction (RECON_CAPS: fewer Gaussians, a narrower training image, a smaller share of the GPU),
applied where a reconstruction's settings are resolved (cap_recon()).

None of these numbers is measured on an 8 GB card: the acceptance target is a measured peak of at most 7.5 GB of
video memory over an 8-step loop on the 16 GB reference card (`falcon profile measure -- COMMAND` samples it), since a
smaller card cannot be enforced on a larger one. `doctor` suggests low_vram on a GPU with 9 GB or less.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PROFILES = {
    'standard': {'about': 'the settings as you set them (CARLA Epic quality, a 12 GB server cap, a 200 Mpx render '
                          'budget, FP32 inference)', 'values': {}},
    'low_vram': {'about': 'for a GPU with about 8 GB: Low quality, a 512 MB texture pool, a 96 Mpx render budget, '
                          'FP16 inference where accuracy holds, a 10 GB server cap, smaller reconstructions',
                 'values': {'server_quality': 'Low', 'server_memory_gb': 10.0, 'render_budget_mpx': 96.0,
                            'texture_pool_mb': 512, 'inference_precision': 'auto'}},
}
KEYS = ('server_quality', 'server_memory_gb', 'render_budget_mpx', 'texture_pool_mb', 'inference_precision')
RECON_CAPS = {'max_gaussians': 300_000, 'width': 480, 'max_init_points': 150_000, 'gpu_memory_fraction': 0.5}
SUGGEST_AT_GB = 9.0                 # doctor suggests low_vram on a GPU with this much video memory or less
TARGET_PEAK_GB = 7.5                # the acceptance target on the reference card (see the module doc)
PRECISIONS = ('fp32', 'fp16', 'auto')


class ProfileError(ValueError):
    pass


def saved_path(settings) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(settings.path)), 'profile-saved.json')


def current(settings) -> str:
    p = settings.get('profile', 'standard') or 'standard'
    return p if p in PROFILES else 'standard'


def set_profile(settings, name: str, save: bool = True) -> dict:
    """Switch to a profile: its values written (the ones they replace kept in profile-saved.json), or, back to
    standard, the kept ones restored; the settings saved unless save is False. {key: (old, new)} of what changed."""
    if name not in PROFILES:
        raise ProfileError(f'unknown profile {name!r} (one of {", ".join(PROFILES)})')
    from .paths import atomic_write
    from .settings import DEFAULTS
    was = current(settings)
    changed = {}
    if name == 'standard':
        try:
            with open(saved_path(settings), encoding='utf-8') as fh:
                back = json.load(fh)
        except (OSError, ValueError):
            back = {k: DEFAULTS.get(k) for k in PROFILES[was]['values']}
        for k, v in back.items():
            if k in KEYS and settings.get(k) != v:
                changed[k] = (settings.get(k), v)
                settings.set(k, v, save=False)
        try:
            os.remove(saved_path(settings))
        except OSError:
            pass
    else:
        vals = PROFILES[name]['values']
        if was == 'standard':
            atomic_write(saved_path(settings), json.dumps({k: settings.get(k) for k in vals}, indent=1) + '\n')
        for k, v in vals.items():
            if settings.get(k) != v:
                changed[k] = (settings.get(k), v)
                settings.set(k, v, save=False)
    settings.set('profile', name, save=False)
    if save:
        settings.save()
    return changed


def effective(settings) -> dict:
    """The profile in force and what it means now: {'profile', 'about', 'values': {key: the setting's value},
    'differs': {key: {'profile', 'setting'}} (changed by hand since), 'recon': caps or {}, 'server_args': the
    texture pool's server arguments}."""
    from ..sim.backends import POOL_EXEC, POOL_INI
    name = current(settings)
    prof = PROFILES[name]
    values = {k: settings.get(k) for k in KEYS}
    differs = {k: {'profile': v, 'setting': values[k]} for k, v in prof['values'].items() if values[k] != v}
    mb = int(values.get('texture_pool_mb') or 0)
    return {'profile': name, 'about': prof['about'], 'values': values, 'differs': differs,
            'recon': dict(RECON_CAPS) if name == 'low_vram' else {},
            'server_args': [POOL_INI.format(mb=mb), POOL_EXEC.format(mb=mb)] if mb > 0 else []}


def cap_recon(recon_settings: dict | None, settings=None) -> dict:
    """A reconstruction's settings with the profile's caps applied (low_vram: never above RECON_CAPS; standard:
    unchanged). The keys are core.recon.train's."""
    out = dict(recon_settings or {})
    if settings is None:
        from .settings import Settings
        settings = Settings()
    if current(settings) != 'low_vram':
        return out
    for k, cap in RECON_CAPS.items():         # every cap is below core.recon.train's default for its key
        v = out.get(k)
        out[k] = cap if v is None else min(cap, type(cap)(v))
    return out


def suggestion(gpus: list, profile: str) -> tuple:
    """(ok, detail) for doctor: low_vram suggested when the smallest NVIDIA GPU has SUGGEST_AT_GB or less."""
    if not gpus:
        return True, f'{profile} (no NVIDIA GPU found to size it by)'
    gb = min(g['memory_total_mb'] for g in gpus) / 1024.0
    if gb <= SUGGEST_AT_GB and profile != 'low_vram':
        return True, (f'{profile} — this GPU has {gb:.1f} GB of video memory: `falcon profile set low_vram` is '
                      f'suggested ({SUGGEST_AT_GB:g} GB or less)')
    return True, f'{profile} ({gb:.1f} GB of video memory)'


# ================================================================================================ measuring
def measure(cmd: list | None = None, seconds: float = 10.0, interval: float = 0.5, sampler=None) -> dict:
    """Video memory while a command runs (or for `seconds`): {'baseline_mb', 'peak_mb', 'peak_delta_mb',
    'samples', 'seconds', 'exit_code', 'gpu'}. Used memory of the whole GPU (other programs' too), sampled with
    nvidia-smi every `interval` s: a peak shorter than that can be missed."""
    from .sysinfo import VramSampler
    s = sampler or VramSampler(interval)
    base = s.sample()
    if base is None:
        raise ProfileError('no NVIDIA GPU answered (nvidia-smi): nothing to measure')
    s.start()
    t0 = time.monotonic()
    code = None
    try:
        if cmd:
            code = subprocess.call(cmd)
        else:
            time.sleep(max(float(seconds), 0.0))
    finally:
        s.stop()
    peak = max([base] + s.values)
    return {'baseline_mb': base, 'peak_mb': peak, 'peak_delta_mb': peak - base, 'samples': len(s.values) + 1,
            'seconds': round(time.monotonic() - t0, 2), 'exit_code': code, 'gpu': s.gpu_name}


def main_profile(argv) -> int:
    """`falcon profile show [--json]`, `falcon profile set standard|low_vram`, `falcon profile measure [--seconds S]
    [--interval S] [--limit-gb G] [-- COMMAND …]`: 0 ok, 2 failed (measure: over the limit, or the command's code)."""
    import argparse
    from . import protocol as P
    from .settings import Settings
    ap = argparse.ArgumentParser(prog='falcon profile', description='performance profiles (standard, low_vram)')
    sub = ap.add_subparsers(dest='action', required=True)
    sp = sub.add_parser('show', help='the profile in force, its values and what differs from it')
    sp.add_argument('--json', action='store_true')
    st_ = sub.add_parser('set', help='switch profile (low_vram keeps the values it replaces; standard restores them)')
    st_.add_argument('name', choices=tuple(PROFILES))
    mp = sub.add_parser('measure', help='the GPU\'s peak video memory while a command runs (after --), or for '
                                        '--seconds')
    mp.add_argument('--seconds', type=float, default=10.0)
    mp.add_argument('--interval', type=float, default=0.5, help='sampling interval, s (nvidia-smi)')
    mp.add_argument('--limit-gb', type=float, default=TARGET_PEAK_GB,
                    help=f'fail when the peak is above this (default {TARGET_PEAK_GB:g} GB, low_vram\'s target)')
    mp.add_argument('--json', action='store_true')
    mp.add_argument('cmd', nargs=argparse.REMAINDER, help='-- COMMAND …')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    st = Settings()
    if a.action == 'show':
        eff = effective(st)
        if a.json:
            print(json.dumps(eff, indent=1))
            return P.OK
        print(f'{eff["profile"]}: {eff["about"]}')
        for k, v in eff['values'].items():
            d = eff['differs'].get(k)
            print(f'  {k:20s} {v}' + (f'   (the profile\'s: {d["profile"]})' if d else ''))
        if eff['recon']:
            print('  reconstruction caps  ' + ', '.join(f'{k} {v}' for k, v in eff['recon'].items()))
        return P.OK
    if a.action == 'set':
        try:
            changed = set_profile(st, a.name)
        except (ProfileError, KeyError, OSError) as e:
            print(f'failed: {e}', file=sys.stderr)
            return P.FAILED
        print(f'profile {a.name}' + (': ' + ', '.join(f'{k} {o} → {n}' for k, (o, n) in changed.items())
                                    if changed else ' (nothing changed)'))
        return P.OK
    cmd = [c for c in a.cmd if c != '--'] if a.cmd and a.cmd[0] == '--' else list(a.cmd or [])
    try:
        res = measure(cmd or None, a.seconds, a.interval)
    except (ProfileError, OSError) as e:
        print(f'failed: {e}', file=sys.stderr)
        return P.FAILED
    res['limit_mb'] = a.limit_gb * 1024.0
    res['within'] = res['peak_mb'] <= res['limit_mb']
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print(f'{res["gpu"]}: peak {res["peak_mb"] / 1024:.2f} GB used (baseline {res["baseline_mb"] / 1024:.2f} GB, '
              f'+{res["peak_delta_mb"] / 1024:.2f} GB) over {res["seconds"]:.1f} s, {res["samples"]} samples — '
              + ('within' if res['within'] else 'over') + f' {a.limit_gb:g} GB (the whole GPU, other programs too)')
    if res['exit_code']:
        return int(res['exit_code'])
    return P.OK if res['within'] else P.FAILED
