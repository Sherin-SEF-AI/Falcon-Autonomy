"""`falcon scenario from-text|check`: a scenario written from a description in words, run once and checked against
the description's claims (core.scenario_text, core.scenario_check; docs/scenario-text.md).

    falcon scenario from-text PROJECT "DESCRIPTION" [--world W] [--rig R] [--seed N] [--rounds 2] [--offline]
        [--model M] [--apply] [--run] [--fake-carla DIR] [--keep-server] [--json]
    falcon scenario check RUN ["DESCRIPTION"] [--claims FILE] [--model] [--json] [--no-write]

from-text asks the assistant's main model (Groq; the user's key and consent, Settings ▸ Assistant — refused with
nothing sent otherwise) unless --offline, which drafts from the phrases Falcon recognises. --apply adds the scenario to
the project (saved); --run also runs it once on Falcon's server (the scenario-run path of Scenarios ▸ Run) and checks
the run against the description. check exits 0 when no claim is unmet, 1 when one is, 2 on an error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from ..core import protocol as P


def _assistant():
    from ..assistant.service import AssistantService
    from ..core.settings import Settings
    return AssistantService(Settings())


def run_scenario(svc, project, scn: dict, world: str, rig: str, seed: int = 0, quiet: bool = False) -> tuple:
    """(exit code, run folder): the scenario run once through the queue (cli._run_job's wait)."""
    from ..cli import _run_job
    from ..core.drive import run_dir_name, scenario_config
    out = os.path.join(svc.runs_dir(), run_dir_name(f'scenario-{scn["id"]}'))
    cfg = scenario_config(project, scn, world, rig, out, svc.port, svc.tm_port, seed=seed,
                          render_budget_mpx=float(svc.settings.get('render_budget_mpx')),
                          workspace=project.workspace, trigger_rules='project')
    job = svc.submit_drive(cfg, title=f'{scn["name"]} (seed {seed})')
    return _run_job(svc, job, quiet=quiet), out


def _print_record(rec: dict):
    print(f'{rec["description"]!r}: {rec["status"].replace("_", " ")} ({rec["mode"]}'
          + (f', {rec.get("model")}' if rec.get('model') else '') + f', {len(rec["attempts"])} attempt'
          + ('s' if len(rec['attempts']) != 1 else '') + ')')
    for c in rec['claims']:
        from ..core.scenario_check import claim_text
        print(f'  claim {c["id"]}: {claim_text(c)}  ({c["source"]}: {c.get("phrase", "")!r})')
    if rec.get('unrecognised'):
        print(f'  not recognised: {", ".join(rec["unrecognised"])}')
    for a in rec['attempts']:
        if a['errors'] or a['mismatches']:
            print(f'  attempt {a["round"] + 1}: ' + '; '.join(a['errors'][:4] + a['mismatches'][:4]))
    for m in rec.get('mismatches') or ():
        print(f'  mismatch: {m}')
    for n in rec.get('notes') or ():
        print(f'  note: {n}')
    if rec['sent']['requests']:
        print(f'  sent: {rec["sent"]["requests"]} request(s), {rec["sent"]["characters"]} characters of text — '
              + '; '.join(rec['sent']['what']))


def cmd_from_text(a) -> int:
    from ..core import scenario_check as CK
    from ..core import scenario_text as T
    from ..core.project import Edit, Project, ProjectError
    try:
        proj = Project.load(a.project)
    except (OSError, ProjectError, ValueError) as e:
        print(f'failed: {e}', file=sys.stderr)
        return 2
    world = a.world or proj.doc['worlds'][0]['id']
    rig = a.rig or proj.doc['rigs'][0]['id']
    if proj.item('worlds', world) is None or proj.item('rigs', rig) is None:
        print(f'failed: no world {world!r} or rig {rig!r} in the project', file=sys.stderr)
        return 2
    hand = T.world_handedness(proj, world)
    complete = None
    if not a.offline:
        from ..assistant.provider import ProviderError
        try:
            complete = _assistant().scenario_writer(a.description, model=a.model)
        except ProviderError as e:
            print(f'nothing was sent. {e.message} (--offline drafts the scenario from the phrases Falcon recognises)',
                  file=sys.stderr)
            return 2
    try:
        rec = T.write(a.description, complete, hand, a.rounds)
    except T.TextError as e:
        print(f'failed: {e}', file=sys.stderr)
        return 2
    except Exception as e:                    # noqa: BLE001 (the provider's errors are sentences)
        print(f'failed: {getattr(e, "message", None) or e}', file=sys.stderr)
        return 2
    if rec.get('scenario'):
        rec['notes'] = T.plan_notes(rec['scenario'])
    path = T.save_record(proj.workspace, rec) if proj.workspace else ''
    if a.json:
        print(json.dumps(rec, indent=2, default=str))
    else:
        _print_record(rec)
        if rec.get('scenario'):
            print(json.dumps(rec['scenario'], indent=2))
        if path:
            print(f'record: {path}')
    scn = rec.get('scenario')
    if scn is None:
        return 2
    if not (a.apply or a.run):
        return 0
    if proj.item('scenarios', scn['id']) is None:
        proj.apply(Edit('insert', ['scenarios'], scn, text=f'Add the scenario {scn["name"]} (from a description)'))
        proj.save()
    print(f'added to the project: {scn["id"]}', file=sys.stderr)
    if not a.run:
        return 0
    from ..cli import _service
    svc, _st = _service(proj.path, a.fake_carla)
    try:
        code, out = run_scenario(svc, proj, scn, world, rig, a.seed)
    except (ProjectError, RuntimeError, ValueError) as e:
        print(f'failed: {e}', file=sys.stderr)
        return P.FAILED
    finally:
        if not a.keep_server:
            svc.shutdown()
    if code != P.OK:
        return code
    print(out)
    try:
        rep = CK.check_run(out, rec['description'], model_claims=[c for c in rec['claims']
                                                                    if c.get('source') == 'model'])
    except CK.CheckError as e:
        print(f'failed: {e}', file=sys.stderr)
        return 2
    print(json.dumps(rep, indent=2, default=str) if a.json else CK.table(rep))
    return 0


def cmd_check(a) -> int:
    from ..core import scenario_check as CK
    claims = None
    if a.claims:
        try:
            with open(a.claims, encoding='utf-8') as fh:
                doc = json.load(fh)
            claims = doc['claims'] if isinstance(doc, dict) else doc
        except (OSError, ValueError, KeyError) as e:
            print(f'failed: {a.claims}: {e}', file=sys.stderr)
            return 2
    model_claims = []
    if a.model:
        if not a.description:
            print('failed: --model needs the description', file=sys.stderr)
            return 2
        from ..assistant.provider import ProviderError
        try:
            complete = _assistant().scenario_writer(a.description)
            model_claims, _bad = CK.parse_model_claims(complete(CK.claims_prompt(a.description)))
        except (ProviderError, ValueError) as e:
            print(f'nothing was sent or the reply was unusable: {getattr(e, "message", None) or e}', file=sys.stderr)
            return 2
    try:
        rep = CK.check_run(a.run, a.description or None, claims, write=not a.no_write, model_claims=model_claims)
    except CK.CheckError as e:
        print(f'failed: {e}', file=sys.stderr)
        return 2
    print(json.dumps(rep, indent=2, default=str) if a.json else CK.table(rep))
    return 1 if rep['counts']['not_met'] else 0


def main_scenario(argv) -> int:
    from ..core import scenario_text as T
    ap = argparse.ArgumentParser(prog='falcon scenario', description='a scenario from a description in words, and '
                                 'a run checked against its description (docs/scenario-text.md)')
    sub = ap.add_subparsers(dest='action', required=True)
    p = sub.add_parser('from-text', help='write a scenario from a description (the assistant\'s model, or --offline)')
    p.add_argument('project')
    p.add_argument('description')
    p.add_argument('--world', default='')
    p.add_argument('--rig', default='')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rounds', type=int, default=T.ROUNDS, help=f'repair rounds (0–{T.MAX_ROUNDS})')
    p.add_argument('--offline', action='store_true', help='no model: Falcon\'s draft from the phrases it recognises')
    p.add_argument('--model', default='', help='the model (default: Settings ▸ Assistant\'s main model)')
    p.add_argument('--apply', action='store_true', help='add the scenario to the project (saved)')
    p.add_argument('--run', action='store_true', help='then run it once and check the run (implies --apply)')
    p.add_argument('--fake-carla', default='')
    p.add_argument('--keep-server', action='store_true')
    p.add_argument('--json', action='store_true')
    c = sub.add_parser('check', help='check a recorded run against a description')
    c.add_argument('run')
    c.add_argument('description', nargs='?', default='',
                   help='default: the description the run\'s scenario was written from')
    c.add_argument('--claims', default='', help='a JSON file of claims instead of the description\'s')
    c.add_argument('--model', action='store_true', help='also ask the assistant\'s model for claims (the text only)')
    c.add_argument('--json', action='store_true')
    c.add_argument('--no-write', action='store_true', help='do not write RUN/scenario-check.json')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    return cmd_from_text(a) if a.action == 'from-text' else cmd_check(a)
