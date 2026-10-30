"""What the model is sent: each tool's one-line description and its argument schema with short argument texts.

Groq's base tier for gpt-oss allows about 8,000 tokens a minute, so the schemas must be small. Each tool keeps its full
description (what it returns, its limits) in its module — that text, and every argument's full description, is
docs/assistant-tools.md (DETAILS) — while the registry the model sees holds SHORT[name] (one or two sentences) and the
same schema with each argument's description cut before its first parenthesis or semicolon and to ARG_CHARS (or left
out when it only repeats the name) and without bounds (minimum, maximum, item counts, patterns, additionalProperties);
types, enums and required stay. The full schema is still enforced: a wrapper (_strict) checks every call against it
before the tool runs, so validation is the same.
"""
from __future__ import annotations

import copy
import dataclasses
import re

ARG_CHARS = 48
DETAILS: dict = {}          # name -> {'description': the full text, 'params': the full schema}

SHORT = {
    # analysis
    'analysis_outputs': 'List outputs of one kind (studies, suites, CI runs, …) by id; with id, a lab output\'s '
                        'numbers.',
    'analysis_study': 'A sweep or search study: ranked failures with RSS labels, parameter values and run ids.',
    'analysis_failure_map': 'Failure-region map of a study over two parameters: runs, failure share, mean per cell.',
    'analysis_criticality': 'A scenario run\'s recorded criticality (TTC, PET, distance, deceleration) and RSS label.',
    'analysis_rootcause': 'A root-cause answer: which factor\'s removal fixes the failure, with caveats.',
    'analysis_evaluation': 'A model evaluation\'s scores with 95 % intervals, or a comparison\'s changes.',
    'analysis_gap': 'A sim-to-real gap result: appearance statistics and feature distances against their floor.',
    'analysis_suite': 'A stack suite\'s pass rates with Wilson intervals, failures, and safety-case risks.',
    'analysis_v2x_benefit': 'A V2X benefit result: road users seen with and without roadside units.',
    'analysis_rules': 'Check a run against the Indian traffic rules: violations, per km, lane margins.',
    'analysis_report_sources': 'What can be reported on (per report kind) and the reports already written.',
    # jobs
    'jobs_drive': 'Queue a recorded drive. Returns the job and run ids.',
    'jobs_scenario_run': 'Queue one run of a scenario, sampled with a seed or given parameter values.',
    'jobs_sweep': 'Queue a parameter sweep of a scenario (a resumable study).',
    'jobs_search': 'Queue a rare-case search over a scenario\'s parameters (criticality or perception).',
    'jobs_replay_failure': 'Queue a replay of one study row as a scenario run with its exact values.',
    'jobs_reproduce': 'Queue a re-run of a drive from its config, compared with the original.',
    'jobs_train_detector': 'Queue fine-tuning a detector on drives (GPU job).',
    'jobs_train_policy': 'Queue training a driving policy: behaviour cloning (bc) or PPO.',
    'jobs_train_translator': 'Queue training a sim-to-real translator against a real image set (GPU job).',
    'jobs_translate_apply': 'Queue translating drives with a trained translator into new labelled runs.',
    'jobs_train_worldmodel': 'Queue training a latent world model on drives.',
    'jobs_evaluate': 'Queue scoring a registered model on drives (AP/AR or mIoU with intervals).',
    'jobs_compare_evaluations': 'Queue a paired comparison of two evaluations (B − A with intervals).',
    'jobs_gap': 'Queue measuring the sim-to-real gap of drives against a real image set.',
    'jobs_export': 'Queue exporting a run in a dataset or log format.',
    'jobs_derive': 'Queue deriving synthetic audio, world-model or language data from a drive.',
    'jobs_reconstruct': 'Queue a neural (Gaussian splatting) reconstruction of a drive (GPU job).',
    'jobs_hybrid': 'Queue a hybrid world: a drive\'s CARLA actors composited over a reconstruction.',
    'jobs_stack_suite': 'Queue a stack test suite: scenarios × seeds, judged pass/fail.',
    'jobs_safety_case': 'Queue the safety-case report of a finished suite against the workspace ODD.',
    'jobs_v2x_benefit': 'Queue measuring what roadside units add, over runs that recorded CPMs.',
    'jobs_report': 'Queue an HTML report of a source, with provenance.',
    'jobs_why': 'Queue "why did it fail?" counterfactual re-runs of a failing scenario run.',
    'jobs_loop': 'Queue the closed loop\'s next step, a run to step N, or a new iteration.',
    'jobs_apply_map_edits': 'Queue re-converting an imported world with its map edits, or a re-import when it has '
                            'none.',
    'jobs_list': 'This session\'s jobs (or the workspace\'s job history) with state and progress.',
    'jobs_get': 'One job: state, progress, step, ETA, exit, error and log tail.',
    'jobs_cancel': 'Cancel a queued or running job (it tidies up first).',
    'jobs_wait': 'Wait until any of the jobs is final or the timeout (≤ 600 s) passes.',
    # lab
    'lab_hil': 'Queue a hardware-in-the-loop session on the simulated device.',
    'lab_deploy': 'Queue deploying an ONNX detector as TensorRT engines, profiled and gated.',
    'lab_ci': 'Queue CI mode on a stack suite, with its gates.',
    'lab_replay': 'Log replay: import a log, replay it in CARLA, or compare two replays.',
    'lab_adversary': 'Queue training or evaluating adversarial traffic.',
    'lab_experiment': 'Queue a mixed sim/real or active-learning experiment.',
    'lab_vlm': 'Queue scoring a VLM\'s driving explanations.',
    # knowledge
    'knowledge_manual': 'Search the built-in manual offline: best sections with heading, link and text.',
    'knowledge_cli': 'Falcon\'s command-line commands with what they do, optionally filtered.',
    'knowledge_setting': 'What a setting does, where it is set, and its value (private ones hidden).',
    'knowledge_tools': 'The assistant\'s tools by group and tier, and what is never possible.',
    # project
    'project_overview': 'The project at a glance: ids and names of worlds, rigs, scenarios, profiles, rules, units.',
    'project_get': 'One project item in full by kind and id (also the loop or safety-case ODD).',
    'project_scenarios': 'Every runnable scenario: built-ins and the project\'s own, with parameters.',
    'project_options': 'Valid choices for edits: towns, weather, sensor types, actions, rule syntax, presets.',
    'project_world_add': 'Add a CARLA town as a world. Returns its id.',
    'project_world_set': 'Change the given settings of a world.',
    'project_world_import_osm': 'Queue importing a place from an OSM file in the workspace\'s osm/ folder.',
    'project_world_add_imported': 'Add a finished OSM import (or a finished map-edit re-conversion) to the project.',
    'project_map_edit': 'Add, replace or drop one map edit of an imported world (checked against its OSM data).',
    'project_rig_add': 'Add a rig (default, or a copy of one). Returns its id.',
    'project_rig_set': 'Rename a rig or change its vehicle.',
    'project_sensor_add': 'Add a sensor of a given type to a rig.',
    'project_sensor_set': 'Change a sensor: name, pose, fields (resolution, fov, labels, faults, …).',
    'project_scenario_create': 'Create an editable scenario as a copy of a built-in or project scenario.',
    'project_scenario_set': 'Edit the given parts of a project scenario.',
    'project_traffic_add': 'Add a traffic profile (preset or copy). Returns its id.',
    'project_traffic_set': 'Change the given settings of a traffic profile.',
    'project_trigger_add': 'Add a trigger rule "when <condition> -> <action>" (parsed and checked).',
    'project_trigger_set': 'Change, enable or disable a trigger rule (checked).',
    'project_trigger_check': 'Check the project\'s trigger rules, or one rule text, with the rule parser.',
    'project_rsu_junctions': 'An imported world\'s junctions as roadside-unit places, signalised first.',
    'project_rsu_place': 'Place a roadside unit at a junction corner or a CARLA position. Returns its id.',
    'project_rsu_set': 'Change a roadside unit: position, yaw, height, CPM settings.',
    'project_odd_set': 'Set the loop\'s ODD or model, or write the safety-case ODD (odd.json).',
    'project_remove': 'Remove one project item by kind and id.',
    # runs
    'runs_search': 'Search runs with the run query language (e.g. "weather:rain dur>20 events:collision").',
    'runs_get': 'One run compactly: settings, performance, summary, scenario result, streams held.',
    'runs_frames': 'A window of a run\'s frames as numbers: time, speed, pose, controls, object counts.',
    'runs_events': 'A run\'s events with times: collisions, min TTC, triggers, teleports, ODD exits.',
    'runs_compare': 'Compare 2–4 runs side by side with differences from the first.',
    'runs_health': 'Dataset health over runs: class balance, empty frames, label problems, duplicates, coverage.',
    'runs_tag': 'Set runs\' tags (empty list removes them).',
    'runs_drive_set_add': 'Save a drive set: a list of runs or a saved query. Returns its id.',
    'runs_trash': 'Move runs into the workspace trash (restorable; nothing is deleted).',
    'runs_trash_list': 'The trash\'s items, for runs_restore.',
    'runs_restore': 'Put a trashed run back where it was.',
    # system
    'system_server': 'Falcon\'s CARLA server state, port and last error.',
    'system_machine': 'Memory, GPUs and the machine checks, measured now.',
    'system_queue': 'Job counts by state, why jobs wait, and who holds the CARLA and GPU locks.',
    'system_versions': 'Falcon, project format, Python and CARLA versions, and the open project.',
    # ui
    'ui_current': 'The open page and tab, every page\'s tabs, and the run shown in Data ▸ Browse.',
    'ui_open_page': 'Go to a page and optionally one of its tabs.',
    'ui_show_run': 'Open a run in Data ▸ Browse, optionally at a step or time.',
    'ui_show_frame': 'Show one frame (by index) of the run in Data ▸ Browse.',
    'ui_compare_runs': 'Open Data ▸ Compare with 2–4 runs.',
    'ui_open_job': 'Go to where a job\'s result is shown.',
    'ui_controls': 'A page\'s named controls, for ui_highlight.',
    'ui_highlight': '"Show me": outline a control on a page for a few seconds (nothing is clicked).',
}


def _short_arg(text: str) -> str:
    t = re.split(r' \(|; ', text, maxsplit=1)[0].strip()
    return t if len(t) <= ARG_CHARS else t[:ARG_CHARS - 1].rstrip() + '…'


FILLER = {'id', 'ids', 'its', 'the', 'a', 'an', 'of', 'to', 'default', 'run', 'runs', 'drives', 'rows', 'new',
          'another', 'short'}
BOOL_FILLER = {'default', 'true', 'false', 'on', 'off'}


def _words(text: str) -> set:
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def _says_nothing(name: str, text: str, enum=(), kind: str = '') -> bool:
    """A description that only repeats its argument's name ('seed', 'world id', 'its name', 'a new name'), a boolean's
    that only gives its default ('default true'), or one beside an enum that is short ('search method') or only lists
    the enum's values ('once, every or cooldown')."""
    words = _words(text)
    own = _words(name.replace('_', ' ')) | FILLER
    if words <= own or (kind == 'boolean' and words <= own | BOOL_FILLER):
        return True
    if enum:
        return len(words) <= 3 or words <= set().union(*(_words(str(e)) for e in enum)) | {'or', 'and'}
    return False


def minify(schema):
    """The schema with every description shortened (_short_arg) or dropped when it only repeats the argument's name
    (or describes an array's items), and without BOUNDS (a dict-valued additionalProperties, a map's value type, stays);
    types, enums and required as they were."""
    s = copy.deepcopy(schema)

    def walk(x, name='', top=False, item=False):
        if isinstance(x, dict):
            for k in BOUNDS:
                if not (k == 'additionalProperties' and isinstance(x.get(k), dict)):
                    x.pop(k, None)
            d = x.get('description')
            if isinstance(d, str) and not top:
                d = _short_arg(d)
                if item or _says_nothing(name, d, x.get('enum') or (x.get('items') or {}).get('enum') or (),
                                         x.get('type', '')):
                    x.pop('description')
                else:
                    x['description'] = d
            for k, v in x.items():
                if k == 'properties':
                    for pn, p in v.items():
                        walk(p, pn)
                elif k in ('items', 'additionalProperties') and isinstance(v, dict):
                    walk(v, name, item=True)
        return x
    return walk(s, top=True)


BOUNDS = ('minimum', 'maximum', 'minItems', 'maxItems', 'pattern', 'additionalProperties')


def _strict(fn, full: dict):
    """fn behind the full schema: the arguments are checked against it (bounds, item counts, patterns, no unknown
    arguments) before fn runs, so the compact schema the model sees can leave those out."""
    import jsonschema
    v = jsonschema.Draft202012Validator(full)

    def wrapped(ctx, **args):
        from ..tool_api import ToolError
        errs = sorted(v.iter_errors(args), key=lambda e: list(e.absolute_path))
        if errs:
            raise ToolError('invalid arguments: ' + '; '.join(
                f'{"/".join(str(p) for p in e.absolute_path) or "(arguments)"}: {e.message}' for e in errs[:5]))
        return fn(ctx, **args)
    wrapped.__wrapped__ = fn
    return wrapped


def compact_registry(registry: dict):
    """Swap every tool for its compact form (SHORT, minified schema; the full schema checked by _strict), keeping the
    full texts in DETAILS."""
    for name, t in list(registry.items()):
        if name in DETAILS:
            continue
        DETAILS[name] = {'description': t.description, 'params': t.params}
        registry[name] = dataclasses.replace(t, description=SHORT.get(name, t.description), params=minify(t.params),
                                             fn=_strict(t.fn, t.params))
