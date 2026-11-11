"""`falcon cloud …`: priced, approved cloud bursts (docs/cloud.md).

    falcon cloud estimate PROJECT --kind failrate|sweep|search --scenario S --provider runpod|lambda --gpu G
                          [--name N] [--world W] [--rig R] [--image I] [--render] [study options] [--json]
    falcon cloud estimate --provider P --list-gpus           the provider's GPUs and prices now
    falcon cloud burst --approve ESTIMATE_ID                 run it (re-priced: a changed estimate launches nothing)
    falcon cloud status [--json]                             bursts, this month's spending, an alert if any
    falcon cloud terminate BURST                             terminate a burst's machine now
    falcon cloud watchdog                                    one watchdog pass (what its systemd timer runs)
    falcon cloud ledger [--json]                             the append-only ledger
    falcon cloud orphans [--terminate] [--json]              Falcon-named machines at the providers

Exit codes: 0 ok; 2 failed or refused (the message says why); a burst whose study failed exits with the study's code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from ..core.project import ProjectError
from . import cloud as C
from .bundle import BundleError


def _study_opts(ap):
    for name, typ in (('stage-size', int), ('final', int), ('rho', float), ('defensive', float), ('max-stages', int),
                      ('batch', int), ('threshold', float), ('seed', int), ('search-seed', int), ('grid', int),
                      ('samples', int), ('budget', int), ('fps', float)):
        ap.add_argument('--' + name, type=typ, default=None)
    ap.add_argument('--judge', default=None)
    ap.add_argument('--method', default=None)
    ap.add_argument('--count', default=None, choices=('avoidable', 'all'))
    ap.add_argument('--no-feasibility', action='store_true')


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='falcon cloud', description='priced, approved cloud bursts (docs/cloud.md)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('estimate', help='price a study on a cloud GPU: low, expected and high $ and its TTL')
    e.add_argument('project', nargs='?', default='')
    e.add_argument('--kind', default='failrate', choices=('failrate', 'sweep', 'search'))
    e.add_argument('--scenario', default='')
    e.add_argument('--name', default='', help='the study (default cloud-<kind>-<scenario>)')
    e.add_argument('--world', default='')
    e.add_argument('--rig', default='')
    e.add_argument('--provider', default='', help='runpod or lambda (default: Settings ▸ Compute)')
    e.add_argument('--gpu', default='', help='the GPU type (falcon cloud estimate --list-gpus)')
    e.add_argument('--image', default='', help='the Falcon worker image (default: the setting cloud.image)')
    e.add_argument('--render', action='store_true', help='render on the machine (default: no rendering — the '
                                                         'criticality needs no images, and a container may lack '
                                                         'Vulkan)')
    e.add_argument('--list-gpus', action='store_true')
    e.add_argument('--json', action='store_true')
    _study_opts(e)
    b = sub.add_parser('burst', help='run an approved estimate (re-priced first; the caps still apply)')
    b.add_argument('--approve', required=True, metavar='ESTIMATE_ID')
    s = sub.add_parser('status', help='bursts, this month\'s spending and any alert')
    s.add_argument('--json', action='store_true')
    t = sub.add_parser('terminate', help='terminate a burst\'s machine now')
    t.add_argument('burst')
    sub.add_parser('watchdog', help='one watchdog pass: terminate what is past its TTL, silent, done or over a cap')
    lg = sub.add_parser('ledger', help='the append-only ledger of every burst')
    lg.add_argument('--json', action='store_true')
    o = sub.add_parser('orphans', help='Falcon-named machines at the providers, and which are orphans')
    o.add_argument('--terminate', action='store_true', help='terminate the orphans')
    o.add_argument('--json', action='store_true')
    return ap


def _red(text: str) -> str:
    return f'\033[31m{text}\033[0m' if sys.stdout.isatty() else text


def build_estimate(a, settings) -> dict:
    """The estimate of `falcon cloud estimate` (also the assistant's cluster_spot_estimate)."""
    from ..core import scenarios as SC
    from ..core.project import Project
    from . import bundle as B
    cs = C.cloud_settings(settings)
    pname = a.provider or cs.get('provider') or ''
    if not pname:
        raise C.CloudError('no provider: --provider runpod|lambda, or choose one in Settings ▸ Compute')
    if not a.gpu:
        raise C.CloudError('--gpu: a GPU type of the provider (falcon cloud estimate --list-gpus)')
    proj = Project.load(a.project)
    scn = proj.item('scenarios', a.scenario) or (SC.builtin(a.scenario) if a.scenario in SC.LIBRARY else None)
    if scn is None and a.scenario.endswith('.json') and os.path.isfile(a.scenario):
        with open(a.scenario, encoding='utf-8') as fh:
            scn = json.load(fh)
    if scn is None:
        raise C.CloudError(f'no scenario {a.scenario!r}')
    world = a.world or proj.doc['worlds'][0]['id']
    rig = a.rig or proj.doc['rigs'][0]['id']
    opts = {}
    for k in B.STUDY_OPTIONS[a.kind]:
        v = getattr(a, k, None)
        if v not in (None, False):
            opts[k] = v
    name = a.name or f'cloud-{a.kind}-{scn.get("id", "scenario")}'
    b = B.make_study(proj, a.kind, scn, world, rig, name, opts, no_render=not a.render)
    rigd = next(r for r in b['project']['rigs'])
    plan = C.plan_of(proj, a.kind, scn, opts, rigd, settings)
    plan['workspace'] = proj.workspace
    image = a.image or str(cs.get('image') or '')
    if not image:
        raise C.CloudError('--image: the Falcon worker image in a registry the provider can pull '
                           '(packaging/cluster/Dockerfile), or the setting cloud.image')
    prov = C.provider(pname, settings=settings)
    return C.make_estimate(plan, b, prov, a.gpu, settings, image)


def _show_estimate(est: dict):
    u, h = est['usd'], est['hours']
    print(f'estimate {est["estimate_id"]}: {est["plan"]["command"]} on {est["provider"]} {est["gpu"]} at '
          f'${est["usd_per_hour"]:.2f}/h (on demand)')
    print(f'  evaluations {est["plan"]["evaluations"]["low"]}–{est["plan"]["evaluations"]["high"]} '
          f'(expected {est["plan"]["evaluations"]["expected"]}), {est["plan"]["seconds_per_eval"]:.0f} s each '
          f'({est["plan"]["source"]})')
    print(f'  cost: ${u["low"]:.2f} low, ${u["expected"]:.2f} expected, ${u["high"]:.2f} high '
          f'({h["low"]:.2f}–{h["high"]:.2f} h with {C.BOOT_H:g} h to boot)')
    print(f'  TTL {est["ttl_h"]:.2f} h: terminated then whatever happens — ${est["max_usd"]:.2f} at most')
    if est['cap_check']:
        print(_red(f'  the caps refuse it: {est["cap_check"]}'))
    else:
        print(f'  to run it: falcon cloud burst --approve {est["estimate_id"]}')


def main_cloud(argv) -> int:
    from ..core.settings import Settings
    a = parser().parse_args(argv)
    st = Settings()
    try:
        if a.cmd == 'estimate':
            if a.list_gpus:
                pname = a.provider or C.cloud_settings(st).get('provider') or ''
                rows = C.provider(pname, settings=st).prices()
                if a.json:
                    print(json.dumps(rows, indent=1))
                for r in rows if not a.json else ():
                    price = f'${r["usd_per_hour"]:.2f}/h' if r['usd_per_hour'] is not None else 'no price'
                    print(f'{r["gpu"]:40s} {price:>10s}  {"available" if r["available"] else "no capacity"}')
                return 0
            if not a.project or not a.scenario:
                print('failed: estimate needs PROJECT and --scenario', file=sys.stderr)
                return 2
            est = build_estimate(a, st)
            if a.json:
                print(json.dumps({k: v for k, v in est.items() if k != 'bundle'}, indent=1))
            else:
                _show_estimate(est)
            return 0
        if a.cmd == 'burst':
            est = C.load_estimate(a.approve)
            out = C.burst(a.approve, est['plan'].get('workspace') or '', st)
            print(f'burst {out["burst"]}: {out["state"]}' + (f', cost ${out.get("cost_usd", 0):.2f}'
                                                             if out.get('cost_usd') is not None else ''))
            if out.get('error'):
                print(f'failed: {out["error"]}', file=sys.stderr)
            if out['state'] == 'terminate_failed':
                print(_red('THE MACHINE IS NOT TERMINATED: the watchdog keeps trying every minute; check the '
                           'provider\'s console'), file=sys.stderr)
                return 2
            return 0 if out.get('study_exit') == 0 else (out.get('study_exit') or 2)
        if a.cmd == 'status':
            rows = C.states()
            month = C.Ledger().month_spend()
            al = C.alert()
            if a.json:
                print(json.dumps({'bursts': rows, 'month': {k: v for k, v in month.items() if k != 'bursts'},
                                  'alert': al}, indent=1))
                return 0
            if al:
                print(_red(al['message']))
            cs = C.cloud_settings(st)
            print(f'{month["month"]}: ${month["spent"]:.2f} spent, ${month["reserved"]:.2f} reserved by running '
                  f'bursts, cap ${float(cs["max_usd_month"]):.2f}')
            for r in rows:
                print(f'{r["burst"]}  {r["state"]:16s} {r.get("provider", "")} {r.get("gpu", "")}  '
                      + (f'${r["cost_usd"]:.2f}' if r.get('cost_usd') is not None else f'≤ ${r.get("max_usd", 0):.2f}'))
            return 0
        if a.cmd == 'terminate':
            s = C.load_state(a.burst)
            if s is None:
                print(f'failed: no burst {a.burst}', file=sys.stderr)
                return 2
            ok = C.terminate(C.provider(s['provider'], settings=st), s, 'user', C.Ledger())
            print('terminated' if ok else _red('NOT terminated: see falcon cloud status'))
            return 0 if ok else 2
        if a.cmd == 'watchdog':
            for bid, why, ok in C.watchdog_tick(st, echo=print):
                print(f'{bid}: {"; ".join(why)} — {"terminated" if ok else "NOT terminated"}')
            return 0
        if a.cmd == 'ledger':
            recs = C.Ledger().records()
            if a.json:
                print(json.dumps(recs, indent=1))
            for r in recs if not a.json else ():
                rest = {k: v for k, v in r.items() if k not in ('t', 'event', 'burst')}
                print(f'{r["t"]} {r["event"]:16s} {r["burst"]} {json.dumps(rest, sort_keys=True)}')
            return 0
        if a.cmd == 'orphans':
            rows = C.orphans(st, terminate_them=a.terminate)
            if a.json:
                print(json.dumps(rows, indent=1))
            for r in rows if not a.json else ():
                if r.get('error'):
                    print(f'{r["provider"]}: {r["error"]}')
                    continue
                tag = 'ORPHAN' if r['orphan'] else 'ok'
                print(f'{r["provider"]} {r["instance_id"]} {r["name"]}: {tag} ({r["why"]})'
                      + (f' — {"terminated" if r.get("terminated") else "NOT terminated"}' if 'terminated' in r else ''))
            return 0
    except (C.CloudError, BundleError, ProjectError, OSError, ValueError) as e:
        print(f'failed: {C.scrub(e)}', file=sys.stderr)
        return 2
    return 2


if __name__ == '__main__':
    sys.exit(main_cloud(sys.argv[1:]))
