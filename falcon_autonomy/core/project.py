"""The project file: JSON with a published schema (falcon_autonomy/schemas/project.schema.json), a version, migrations.

A project holds the user's definitions — worlds, traffic profiles, rigs, scenarios, dataset specs, model
registrations, experiments and the loop's settings — never large data: runs, datasets and models live in the
workspace folder beside it (paths.workspace_for) and are indexed by the catalogue. Machine-specific things (where
CARLA is, ports, a Jetson's address and credentials) are user settings, never in a project.

Every change goes through an edit operation (Edit) so that it can be undone: the GUI wraps them in a QUndoStack, the CLI
applies them directly. Paths into the document are lists of keys and indices, e.g. ['rigs', 0, 'sensors', 1, 'fov'].

docs/project-format.md documents the format; migrations are listed in MIGRATIONS, from version n to n + 1.
"""
from __future__ import annotations

import copy
import json
import os
import uuid

from .. import __version__
from .paths import PROJECT_SUFFIX, atomic_write, workspace_for
from .provenance import utc_now

FORMAT = 'falcon.project'
VERSION = 1
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'schemas',
                           'project.schema.json')
SECTIONS = ('worlds', 'traffic_profiles', 'rigs', 'scenarios', 'dataset_specs', 'models', 'experiments')


class ProjectError(ValueError):
    pass


def new_id(prefix: str) -> str:
    return f'{prefix}-{uuid.uuid4().hex[:8]}'


def default_rig() -> dict:
    """The rig a new project starts with: one forward pinhole camera at 1280 × 720 with semantic labels, on CARLA's
    Lincoln MKZ (the car CARLA's own examples use)."""
    return {
        'id': 'rig-default', 'name': 'Front camera', 'vehicle': 'vehicle.lincoln.mkz_2020',
        'sensors': [{
            'id': 'cam-front', 'name': 'front', 'type': 'camera.pinhole',
            'pose': {'x': 1.5, 'y': 0.0, 'z': 1.6, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
            'width': 1280, 'height': 720, 'fov': 90.0, 'labels': ['semantic'],
        }],
    }


def default_traffic_profile() -> dict:
    return {
        'id': 'tp-default', 'name': 'CARLA defaults', 'handedness': 'map',
        'vehicles': 30, 'walkers': 0,
        'fleet': {'car': 1.0},
        'behaviour': {'distance_to_leading_m': 2.5, 'speed_difference_pct': 0.0, 'ignore_lights_pct': 0.0,
                      'ignore_signs_pct': 0.0, 'auto_lane_change': True, 'walker_crossing_factor': 0.0},
    }


def new_project(name: str) -> dict:
    now = utc_now()
    return {
        'format': FORMAT, 'version': VERSION, 'name': name, 'created': now, 'modified': now,
        'app_version': __version__,
        'worlds': [{'id': 'world-town10', 'name': 'Town 10', 'kind': 'town', 'town': 'Town10HD_Opt',
                    'weather': {'preset': 'ClearNoon'}, 'traffic_profile': 'tp-default'}],
        'traffic_profiles': [default_traffic_profile()],
        'rigs': [default_rig()],
        'scenarios': [], 'dataset_specs': [], 'models': [], 'experiments': [],
        'loop': {'model': None, 'odd': None, 'iterations': []},
    }


# --------------------------------------------------------------------------------------------- schema and migrations
_SCHEMA = None


def schema() -> dict:
    global _SCHEMA
    if _SCHEMA is None:
        with open(SCHEMA_PATH, encoding='utf-8') as fh:
            _SCHEMA = json.load(fh)
    return _SCHEMA


def validate(doc: dict) -> list:
    """Schema errors as readable strings ([] when valid). Uses jsonschema (a dependency of the GUI and CLI)."""
    import jsonschema
    v = jsonschema.Draft202012Validator(schema())
    out = []
    for e in sorted(v.iter_errors(doc), key=lambda e: list(e.absolute_path)):
        where = '/'.join(str(p) for p in e.absolute_path) or '(root)'
        out.append(f'{where}: {e.message}')
    return out


def _migrate_0_to_1(doc: dict) -> dict:
    """Version 0 (the pre-release format of 0.0.x builds) kept rigs' sensors under 'cameras' and had no traffic
    profiles; version 1 has typed 'sensors' and named traffic profiles."""
    doc = copy.deepcopy(doc)
    for rig in doc.get('rigs', []):
        if 'cameras' in rig and 'sensors' not in rig:
            rig['sensors'] = [dict(c, type=c.get('type', 'camera.pinhole')) for c in rig.pop('cameras')]
    doc.setdefault('traffic_profiles', [default_traffic_profile()])
    for s in SECTIONS:
        doc.setdefault(s, [])
    doc.setdefault('loop', {'model': None, 'odd': None, 'iterations': []})
    doc['format'] = FORMAT
    doc['version'] = 1
    return doc


MIGRATIONS = {0: _migrate_0_to_1}


def migrate(doc: dict) -> tuple[dict, list]:
    """doc brought to VERSION: (document, [what was migrated]). A newer version than this build knows is refused."""
    if not isinstance(doc, dict) or doc.get('format', FORMAT) != FORMAT:
        raise ProjectError('Not a Falcon Autonomy project (its "format" is not "falcon.project").')
    v = doc.get('version', 0)
    if not isinstance(v, int) or v < 0:
        raise ProjectError(f'The project\'s version {v!r} is not a whole number.')
    if v > VERSION:
        raise ProjectError(f'The project is format version {v}; this build reads up to {VERSION}. Update Falcon '
                           'Autonomy to open it.')
    notes = []
    while v < VERSION:
        doc = MIGRATIONS[v](doc)
        notes.append(f'migrated from format version {v} to {v + 1}')
        v += 1
    return doc, notes


# --------------------------------------------------------------------------------------------- edits (undoable)
def get_at(doc, path):
    cur = doc
    for k in path:
        cur = cur[k]
    return cur


class Edit:
    """One reversible change. kind: 'set' (path -> value), 'insert' (list at path, index, value), 'remove' (list at
    path, index). apply() returns the inverse Edit."""

    def __init__(self, kind: str, path: list, value=None, index: int | None = None, text: str = ''):
        if kind not in ('set', 'insert', 'remove'):
            raise ValueError(kind)
        self.kind, self.path, self.value, self.index = kind, list(path), value, index
        self.text = text or f'{kind} {"/".join(map(str, path))}'

    def apply(self, doc: dict) -> 'Edit':
        if self.kind == 'set':
            parent = get_at(doc, self.path[:-1])
            key = self.path[-1]
            had = (key in parent) if isinstance(parent, dict) else True
            old = copy.deepcopy(parent[key]) if had else _MISSING
            parent[key] = copy.deepcopy(self.value)
            if old is _MISSING:
                return _Unset(self.path, self.text)
            return Edit('set', self.path, old, text=self.text)
        lst = get_at(doc, self.path)
        if not isinstance(lst, list):
            raise ProjectError(f'{"/".join(map(str, self.path))} is not a list')
        if self.kind == 'insert':
            i = len(lst) if self.index is None else int(self.index)
            lst.insert(i, copy.deepcopy(self.value))
            return Edit('remove', self.path, index=i, text=self.text)
        i = int(self.index)
        old = lst.pop(i)
        return Edit('insert', self.path, old, index=i, text=self.text)

    def __repr__(self):
        return f'Edit({self.kind}, {self.path}, index={self.index})'


_MISSING = object()


class _Unset(Edit):
    """The inverse of a 'set' that created a key: removes it again."""

    def __init__(self, path, text):
        super().__init__('set', path, None, text=text)

    def apply(self, doc):
        parent = get_at(doc, self.path[:-1])
        old = parent.pop(self.path[-1])
        return Edit('set', self.path, old, text=self.text)


class Project:
    """A project document with its file path, dirty flag and an edit history (undo / redo for the CLI; the GUI
    keeps its own QUndoStack of the same Edits)."""

    def __init__(self, doc: dict, path: str = ''):
        self.doc = doc
        self.path = path
        self.dirty = False
        self.notes: list = []
        self._undo: list = []
        self._redo: list = []

    # -- files
    @classmethod
    def new(cls, name: str, path: str = '') -> 'Project':
        return cls(new_project(name), path)

    @classmethod
    def load(cls, path: str) -> 'Project':
        try:
            with open(path, encoding='utf-8') as fh:
                doc = json.load(fh)
        except json.JSONDecodeError as e:
            raise ProjectError(f'The project file is not valid JSON: {e}') from e
        doc, notes = migrate(doc)
        errs = validate(doc)
        if errs:
            raise ProjectError('The project does not match its schema:\n' + '\n'.join(errs[:20]))
        p = cls(doc, path)
        p.notes = notes
        p.dirty = bool(notes)
        return p

    def save(self, path: str | None = None):
        path = path or self.path
        if not path:
            raise ProjectError('The project has no file yet.')
        if not path.endswith(PROJECT_SUFFIX):
            path += PROJECT_SUFFIX
        self.doc['modified'] = utc_now()
        self.doc['app_version'] = __version__
        errs = validate(self.doc)
        if errs:
            raise ProjectError('Refusing to save a project that does not match its schema:\n' + '\n'.join(errs[:20]))
        atomic_write(path, json.dumps(self.doc, indent=2, ensure_ascii=False) + '\n')
        self.path = path
        self.dirty = False

    @property
    def workspace(self) -> str:
        return workspace_for(self.path) if self.path else ''

    # -- edits
    def apply(self, edit: Edit) -> Edit:
        inv = edit.apply(self.doc)
        self._undo.append((edit, inv))
        self._redo.clear()
        self.dirty = True
        return inv

    def undo(self) -> bool:
        if not self._undo:
            return False
        edit, inv = self._undo.pop()
        inv.apply(self.doc)
        self._redo.append(edit)
        self.dirty = True
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        edit = self._redo.pop()
        inv = edit.apply(self.doc)
        self._undo.append((edit, inv))
        self.dirty = True
        return True

    # -- lookups
    def find(self, section: str, ident: str) -> tuple[int, dict] | tuple[None, None]:
        for i, item in enumerate(self.doc.get(section, [])):
            if item.get('id') == ident:
                return i, item
        return None, None

    def item(self, section: str, ident: str) -> dict | None:
        return self.find(section, ident)[1]
