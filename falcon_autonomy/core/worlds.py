"""World imports (standard library only): the configs the OSM import worker runs, and the world entries they add.

An imported world's files live in the workspace under worlds/<world id>/ (area.osm, raw.xodr, world.xodr, graph.json,
import.json, run.json); the project's world entry names world.xodr relative to the workspace and keeps the import's
source and settings in its "osm" block, so that the import can be re-run.
"""
from __future__ import annotations

import os

IMPORT_SCHEMA = 'falcon.osm_import/1'
DEFAULT_SETTINGS = {'lane_width': 3.35, 'traffic_lights': True, 'all_junction_lights': False, 'way_types': None,
                    'excluded_light_way_types': None, 'speed_breakers': 'osm', 'osm_bump': 'osm_bump'}
MESH_DEFAULTS = {'vertex_distance': 2.0, 'max_road_length': 500.0, 'wall_height': 0.0, 'additional_width': 0.6,
                 'smooth_junctions': True, 'enable_mesh_visibility': True, 'enable_pedestrian_navigation': True}


def world_dir(workspace: str, world_id: str) -> str:
    return os.path.join(workspace, 'worlds', world_id)


def import_config(source: str, out_dir: str, bbox=None, lht: bool = True, settings: dict | None = None,
                  centre=None, seed: int = 0, lane_check_sample: int = 50) -> dict:
    return {'schema': IMPORT_SCHEMA, 'source': os.path.abspath(source), 'bbox': list(bbox) if bbox else None,
            'centre': list(centre) if centre else None, 'lht': bool(lht),
            'settings': dict(DEFAULT_SETTINGS, **(settings or {})), 'out': os.path.abspath(out_dir), 'seed': seed,
            'lane_check_sample': lane_check_sample}


def world_entry(world_id: str, name: str, cfg: dict, report: dict, mesh: dict | None = None,
                traffic_profile: str = 'tp-default') -> dict:
    return {'id': world_id, 'name': name, 'kind': 'osm', 'xodr': os.path.join('worlds', world_id, 'world.xodr'),
            'osm': {'source': cfg['source'], 'bbox': cfg['bbox'], 'centre': report.get('centre'), 'lht': cfg['lht'],
                    'settings': cfg['settings'],
                    'mesh': _mesh(dict(MESH_DEFAULTS, **(mesh or {})), report),
                    'handedness': report.get('handedness', 'RHT')},
            'weather': {'preset': 'ClearNoon'}, 'traffic_profile': traffic_profile}


def _mesh(mesh: dict, report: dict) -> dict:
    """A world with road features is generated at the vertex distance that renders them (core.roadfeatures)."""
    from .roadfeatures import mesh_for
    return mesh_for(mesh, bool(((report or {}).get('road_features') or {}).get('placed')))
