"""Cloud-burst tools (the cluster group): a priced estimate and an approved burst on RunPod or Lambda, under the caps in
Settings ▸ Compute (falcon_autonomy.cluster.cloud; docs/cloud.md).

    cluster_spot_estimate  read   price a study on a cloud GPU now (the provider's price list: one read of its API):
                                  low, expected and high dollars, the TTL after which the machine is terminated, the
                                  most it can cost, whether the caps allow it, and the estimate id to approve
    cluster_spot_burst     act    run an estimate: asks the user on a card in EVERY mode (no "allow all": each burst
                                  is its own approval, Tool.approval); the job re-prices it first and launches nothing
                                  when the price or the plan changed (a new id), the estimate was used already, or the
                                  caps refuse it. Its preview and estimate come from the saved estimate (no network).

Both names start with cluster_ (the group). Neither ever sees a key: the providers' keys are in the keyring.
A module adds its tools' one-line descriptions to _compact.SHORT itself (tools/__init__ explains why)."""
from __future__ import annotations

import argparse

from ..tool_api import ToolError, tool
from . import _common as C
from . import _compact

MINE = {
    'cluster_spot_estimate': 'Price a study on a cloud GPU: $ low/expected/high, TTL, caps; returns the estimate id.',
    'cluster_spot_burst': 'Run an approved cloud estimate by its id (always asks the user; caps apply).',
}
assert not set(MINE) & set(_compact.SHORT), set(MINE) & set(_compact.SHORT)
_compact.SHORT.update(MINE)


def _settings(ctx):
    return ctx.settings if ctx.win is None else ctx.win.settings


def _summary(est: dict) -> dict:
    return {'estimate_id': est['estimate_id'], 'provider': est['provider'], 'gpu': est['gpu'],
            'usd_per_hour': est['usd_per_hour'], 'on_demand': True, 'usd': est['usd'], 'hours': est['hours'],
            'ttl_h': est['ttl_h'], 'max_usd': est['max_usd'], 'evaluations': est['plan']['evaluations'],
            'seconds_per_eval': est['plan']['seconds_per_eval'], 'throughput_source': C.clip(est['plan']['source'],
                                                                                            160),
            'no_render': bool(est['bundle'].get('no_render')), 'study': est['bundle'].get('name'),
            'caps': est['caps'], 'caps_refuse': est['cap_check'] or None}


@tool('cluster_spot_estimate', 'cluster', 'read',
      'Price a study on a cloud GPU (RunPod or Lambda, on demand: neither offers spot through its current API): '
      'the study (failrate, sweep or search of a scenario on a CARLA town; criticality only, run without rendering '
      'unless render) is sized from this machine\'s measured throughput, priced at the provider\'s price now, and '
      'given a TTL (expected hours × the ttl_factor setting) after which the machine is terminated whatever happens: '
      'max_usd is the most it can cost. Returns the estimate id (cluster_spot_burst runs it once the user approves) '
      'and whether the caps (per burst, per month) refuse it. Needs the provider\'s key in Settings ▸ Compute.',
      C.obj({'kind': C.s('the study', enum=['failrate', 'sweep', 'search']),
             'scenario': C.s('scenario key or id'), 'provider': C.s('the provider', enum=['runpod', 'lambda']),
             'gpu': C.s('the provider\'s GPU type id (its catalogue name)'),
             'samples': C.i('sweep: runs', minimum=2, maximum=100000),
             'budget': C.i('search: runs', minimum=4, maximum=100000),
             'stage_size': C.i('failrate: runs per stage', minimum=10, maximum=100000),
             'final': C.i('failrate: final stage runs', minimum=10, maximum=100000),
             'max_stages': C.i('failrate: stages at most', minimum=1, maximum=20),
             'seed': C.i('seed'), 'world': C.s('world id'), 'rig': C.s('rig id'), 'name': C.s('study name'),
             'render': C.b('render on the machine (default false)')},
            ['kind', 'scenario', 'gpu']))
def cluster_spot_estimate(ctx, kind, scenario, gpu, provider='', samples=None, budget=None, stage_size=None,
                          final=None, max_stages=None, seed=None, world='', rig='', name='', render=False):
    from ...cluster import cloud as CL
    from ...cluster.bundle import BundleError
    from ...cluster.cloudcli import build_estimate
    p = C.project(ctx)
    if not p.path:
        raise ToolError('the project has no file yet: save it before pricing a study')
    a = argparse.Namespace(project=p.path, kind=kind, scenario=scenario, name=name, world=world, rig=rig,
                           provider=provider, gpu=gpu, image='', render=render, samples=samples, budget=budget,
                           stage_size=stage_size, final=final, max_stages=max_stages, seed=seed, grid=None, rho=None,
                           defensive=None, batch=None, threshold=None, search_seed=None, fps=None, judge=None,
                           method=None, count=None, no_feasibility=False)
    try:
        est = build_estimate(a, _settings(ctx))
    except (CL.CloudError, BundleError, ValueError, OSError) as e:
        raise ToolError(CL.scrub(e)) from None
    return _summary(est)


def _load(estimate_id: str) -> dict:
    from ...cluster import cloud as CL
    try:
        return CL.load_estimate(estimate_id)
    except CL.CloudError as e:
        raise ToolError(str(e)) from None


def _burst_preview(ctx, estimate_id: str) -> str:
    est = _load(estimate_id)
    u = est['usd']
    lines = [f'Rent a {est["gpu"]} on {est["provider"]} (on demand, ${est["usd_per_hour"]:.2f}/h) to run '
             f'{est["plan"]["command"]} study {est["bundle"].get("name")!r}:',
             f'  ${u["low"]:.2f} low, ${u["expected"]:.2f} expected, ${u["high"]:.2f} high; terminated after '
             f'{est["ttl_h"]:.2f} h whatever happens — ${est["max_usd"]:.2f} at most.',
             f'  Caps: ${est["caps"]["max_usd_per_burst"]:.2f} per burst, ${est["caps"]["max_usd_month"]:.2f} a month.',
             '  Re-priced before launch: a changed price launches nothing.']
    if est.get('cap_check'):
        lines.append(f'  The caps refuse it now: {est["cap_check"]}')
    return '\n'.join(lines)


def _burst_estimate(ctx, estimate_id: str) -> dict:
    est = _load(estimate_id)
    return {'seconds': round(est['hours']['expected'] * 3600.0, 1),
            'what': f'{est["hours"]["expected"]:.2f} h expected on {est["provider"]} {est["gpu"]}, '
                    f'${est["usd"]["expected"]:.2f} (at most ${est["max_usd"]:.2f})'}


def _gate(ctx, estimate_id: str = '', **_a):
    """Tool.approval: a card in every mode, no allow-all — each burst spends the user's money on its own approval."""
    return {'ask': True, 'allow_all': False,
            'why': 'it rents a cloud machine with your money: every burst asks you, in every mode'}


@tool('cluster_spot_burst', 'cluster', 'act',
      'Run a cloud estimate (cluster_spot_estimate\'s id) as a burst: the user approves it on a card in every mode. '
      'The queued job prices it again and launches nothing when the price or plan changed (a new estimate id), the '
      'estimate was used before, it is more than a day old, or the caps refuse it; otherwise it launches the machine '
      'with a key of its own, runs the study, brings its results into studies/<name>, terminates the machine and '
      'confirms it (an independent watchdog terminates it at the TTL, after 10 silent minutes or over a cap).',
      C.obj({'estimate_id': C.s('the estimate id (16 hex digits)', pattern='^[0-9a-f]{16}$')}, ['estimate_id']),
      estimate=_burst_estimate, preview=_burst_preview, approval=_gate)
def cluster_spot_burst(ctx, estimate_id: str):
    est = _load(estimate_id)
    if est.get('cap_check'):
        raise ToolError(f'refused by the caps: {est["cap_check"]}')
    from ...core.jobs import JobSpec
    argv = ['cloud', 'burst', '--approve', estimate_id]
    spec = JobSpec('cloud_burst', f'Cloud burst {est["bundle"].get("name")} on {est["provider"]}', C.cli_argv(argv),
                   memory_gb=0.5)
    return C.cli_job(ctx, 'cloud_burst', spec.title, argv, spec=spec)
