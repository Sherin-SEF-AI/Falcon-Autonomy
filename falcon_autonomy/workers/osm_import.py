"""osm_import: an OpenStreetMap area to an OpenDRIVE world CARLA can load (runs in CARLA's Python; no server needed).

    python -m falcon_autonomy.workers osm_import CONFIG.json

CONFIG.json (schema falcon.osm_import/1, core.worlds.import_config): source (.osm / .osm.pbf), bbox (south, west,
north, east) or none, the converter's settings, lht (left-hand traffic), out (the world's folder in the workspace).
The import:

1. reads the source, cropped to the bbox (core.osm; ways crossing the box are kept whole), and writes it (area.osm);
2. converts it with carla.Osm2Odr (SUMO's netconvert inside) projected with a transverse Mercator centred on the area
   (core.osm.proj_string, center_map off), so OpenDRIVE x, y are metres east and north of the centre and the
   geoReference is the area's — CARLA's default projection, centred on 0°E, stretches a place far from it (central
   Bengaluru by about 3× east-west and 4× north-south); objects and signals a few millimetres past their road's end
   (SUMO writes some) are clamped onto it, since CARLA's world generation crashes on them (core.xodr.sanitize);
3. with lht, converts the right-hand output to left-hand traffic (core.xodr.to_lht; see docs/worlds.md);
4. writes world.xodr, the road graph for the preview (graph.json), and import.json: what was read, converted and
   dropped, the lane check against OSM on a seeded sample (core.xodr.lane_check), the handedness, what an OpenDRIVE
   world does not contain (buildings, vegetation, textures), timings.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from ..core import osm as O
from ..core import protocol as P
from ..core import xodr as X
from ..core.provenance import read_json, write_json
from .common import Failed, Worker

IMPORT_SCHEMA = 'falcon.osm_import/1'


class OsmImport(Worker):
    name = 'osm_import'

    def __init__(self, cfg, argv, cfg_path=''):
        super().__init__(argv, cfg['out'], 20.0, cfg.get('seed', 0), cfg,
                         {'config': cfg_path, 'source': cfg.get('source')} if cfg_path else None)
        self.cfg = cfg
        self.life['phase'] = 'local'           # no server: the watchdog only takes signals

    def body(self):
        cfg = self.cfg
        if cfg.get('schema') != IMPORT_SCHEMA:
            raise Failed(f'not an OSM import config (schema {cfg.get("schema")!r})')
        carla = self.import_carla()
        t = {}
        t0 = time.monotonic()
        bbox = tuple(cfg['bbox']) if cfg.get('bbox') else None
        self.emit.progress(0.02, 'reading OpenStreetMap', force=True)
        data = O.read_osm(cfg['source'], bbox=bbox,
                          progress=lambda f, text: self.emit.progress(0.02 + 0.3 * f, text))
        self.check()
        hw = [w for w in data.ways if w.get('kind') == 'highway']
        if not hw:
            raise Failed('no roads (highway=* ways) in the chosen area')
        summ = O.summary(data)
        lat0, lon0 = cfg.get('centre') or summ['centre']
        os.makedirs(self.out_dir, exist_ok=True)
        area = os.path.join(self.out_dir, 'area.osm')
        O.write_osm(data, area)
        t['read_s'] = round(time.monotonic() - t0, 2)
        self.emit.progress(0.35, f'converting {len(hw)} roads (carla.Osm2Odr)', force=True)
        s = cfg.get('settings') or {}
        st = carla.Osm2OdrSettings()
        st.proj_string = O.proj_string(lat0, lon0)
        st.center_map = False
        st.use_offsets = False
        st.default_lane_width = float(s.get('lane_width', 3.35))
        st.generate_traffic_lights = bool(s.get('traffic_lights', True))
        st.all_junctions_with_traffic_lights = bool(s.get('all_junction_lights', False))
        if s.get('way_types'):
            st.set_osm_way_types(list(s['way_types']))
        if s.get('excluded_light_way_types'):
            st.set_traffic_light_excluded_way_types(list(s['excluded_light_way_types']))
        t1 = time.monotonic()
        with open(area, encoding='utf-8') as fh:
            raw = carla.Osm2Odr.convert(fh.read(), st)
        t['convert_s'] = round(time.monotonic() - t1, 2)
        if not raw or '<road ' not in raw:
            raise Failed('the converter produced no roads (are the way types in the area among those converted?)')
        with open(os.path.join(self.out_dir, 'raw.xodr'), 'w', encoding='utf-8') as fh:
            fh.write(raw)
        self.check()
        notes = list(data.notes)
        text, fixes = X.sanitize(raw)
        notes += fixes
        if cfg.get('lht'):
            self.emit.progress(0.6, 'converting to left-hand traffic', force=True)
            text, lht_notes = X.to_lht(raw)
            notes += lht_notes
        with open(os.path.join(self.out_dir, 'world.xodr'), 'w', encoding='utf-8') as fh:
            fh.write(text)
        self.emit.progress(0.7, 'road graph and lane check', force=True)
        od = X.parse(text)
        graph = X.road_graph(od)
        write_json(os.path.join(self.out_dir, 'graph.json'), graph)
        proj = O.local_projection(lat0, lon0)
        check = X.lane_check(od, data, proj, sample=int(cfg.get('lane_check_sample', 50)),
                             seed=int(cfg.get('seed', 0)))
        offset = od.header.get('offset')
        if offset:
            notes.append(f'the converter shifted coordinates by {offset}: the lane check may not line up')
        t['total_s'] = round(time.monotonic() - t0, 2)
        rep = {'source': cfg['source'], 'bbox': bbox, 'centre': [lat0, lon0], 'proj': O.proj_string(lat0, lon0),
               'osm': summ, 'counts': data.counts, 'graph_totals': graph.get('totals'),
               'handedness': X.handedness(od), 'lane_check': check, 'settings': s, 'notes': notes,
               'not_represented': {
                   'buildings': summ.get('buildings', 0),
                   'text': 'an OpenDRIVE world is roads, lanes, sidewalks and junctions only: no buildings, '
                           'vegetation, textures or props (CARLA cannot spawn arbitrary meshes at runtime); '
                           'docs/worlds.md describes the full-fidelity path'},
               'timings': t}
        write_json(os.path.join(self.out_dir, 'import.json'), rep)
        self.emit.progress(1.0, f'{graph["totals"].get("roads", len(graph.get("roads", [])))} roads, lane check '
                                f'{check.get("matched")}/{check.get("comparable")}', force=True)
        return {'import': {k: rep[k] for k in ('handedness', 'timings', 'bbox', 'centre')},
                'summary': {'handedness': rep['handedness'], 'lane_agreement': check.get('agreement'),
                            'roads_km': (graph.get('totals') or {}).get('road_km')},
                'name': f'import {os.path.basename(cfg["source"])}'}


def main(argv) -> int:
    ap = argparse.ArgumentParser(prog='falcon worker osm_import')
    ap.add_argument('config')
    a = ap.parse_args(argv)
    try:
        cfg = read_json(a.config)
    except (OSError, ValueError) as e:
        print(f'failed: cannot read {a.config}: {e}', flush=True)
        return P.FAILED
    return OsmImport(cfg, argv, a.config).execute()
