"""How real a result is: the simulator (or model) that produced a run, a study or a verdict, as one word that travels with
the result so two results of different fidelity are never compared unknowingly.

Levels (LEVELS): `carla` (a real CARLA 0.9.16 server), `carla-ue5` (a real CARLA UE5-line server), `fake` (the fake
`carla` module the tests and demos use — a test double, not a simulator), `kinematic` (Falcon's cheap kinematic
screen, core.fastsim), `neural` (a neural reconstruction rendered in closed loop, core.recon), `recorded` (an offline
analysis of a recorded run or log — its own fidelity is the source run's), `mixed` (a result built from inputs of
different levels) and `unknown` (a run written before 4.2, which did not record it).

A worker records `provenance.fidelity` in run.json (workers.common: the fake modules set `FALCON_FAKE = True`). A study
summary records `fidelity` as combine() over its runs. The UI shows it as a chip next to every run, chart and verdict
(ui.widgets.fidelity_chip)."""
from __future__ import annotations

import json
import os

LEVELS = {
    #  id           short label            sentence                                                         tone
    'carla':     ('CARLA',               'Real CARLA 0.9.16 server.',                                      ''),
    'carla-ue5': ('CARLA UE5',           'Real CARLA UE5-line server.',                                    ''),
    'fake':      ('Fake simulator',      'The fake carla module (a test double): numbers show the tool works, '
                                         'not how a driver behaves.',                                      'warning'),
    'kinematic': ('Kinematic',           'Falcon\'s cheap kinematic screen: no sensors, no rendering.',    'warning'),
    'neural':    ('Neural render',       'A neural reconstruction driven in closed loop: trusted only near '
                                         'the logged path.',                                               'accent'),
    'recorded':  ('Recorded',            'An offline analysis of a recorded run or log.',                  ''),
    'mixed':     ('Mixed fidelity',      'Built from results of different fidelity: compare with care.',   'critical'),
    'unknown':   ('Fidelity unknown',    'Written before Falcon recorded fidelity (4.2).',                 ''),
}
REAL = ('carla', 'carla-ue5')


def label(level: str) -> str:
    return LEVELS.get(level or 'unknown', LEVELS['unknown'])[0]


def sentence(level: str) -> str:
    return LEVELS.get(level or 'unknown', LEVELS['unknown'])[1]


def tone(level: str) -> str:
    return LEVELS.get(level or 'unknown', LEVELS['unknown'])[2]


def of_module(carla_module) -> str:
    """The level of the `carla` module a worker imported: fake when it says so, else by backend."""
    if getattr(carla_module, 'FALCON_FAKE', False):
        return 'fake'
    ver = str(getattr(carla_module, '__version__', '') or '')
    return 'carla-ue5' if ver.startswith('0.10') else 'carla'


def of_doc(doc: dict | None) -> str:
    """The level a run.json or a summary records: provenance.fidelity, a top-level fidelity, else unknown."""
    if not isinstance(doc, dict):
        return 'unknown'
    lv = doc.get('fidelity') or (doc.get('provenance') or {}).get('fidelity')
    return lv if lv in LEVELS else 'unknown'


def of_run(path: str) -> str:
    """The level of a run folder (or its run.json, or a summary file); unknown when nothing says."""
    p = os.path.join(path, 'run.json') if os.path.isdir(path) else path
    try:
        with open(p, encoding='utf-8') as f:
            return of_doc(json.load(f))
    except (OSError, ValueError):
        return 'unknown'


def combine(levels) -> str:
    """One level for a result built from several: the level when they agree, `mixed` when they do not. Unknown
    inputs are ignored when at least one level is known (old runs beside new ones); nothing known gives unknown."""
    known = {lv for lv in levels if lv and lv != 'unknown'}
    if not known:
        return 'unknown'
    if len(known) == 1:
        return known.pop()
    return 'mixed'


def comparable(a: str, b: str) -> tuple[bool, str]:
    """Whether two results may be compared as like for like, and the reason when not. Unknown is comparable with
    anything (with the reason said); a fake or kinematic result is never comparable with a real one."""
    a, b = a or 'unknown', b or 'unknown'
    if a == b and a != 'mixed':
        return True, ''
    if 'unknown' in (a, b):
        return True, f'one side does not record its fidelity ({label(a)} vs {label(b)})'
    return False, f'{label(a)} vs {label(b)}: different fidelity, not like for like'
