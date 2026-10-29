"""osm_import: an OpenStreetMap area to an OpenDRIVE world CARLA can load (runs in CARLA's Python; no server needed).

    python -m falcon_autonomy.workers osm_import CONFIG.json

CONFIG.json (schema falcon.osm_import/1, core.worlds.import_config): source (.osm / .osm.pbf), bbox (south, west,
north, east) or none, the converter's settings, lht (left-hand traffic), out (the world's folder in the workspace).
The import:

1. reads the source, cropped to the bbox (core.osm; ways crossing the box are kept whole), applies the config's map
   edits (core.mapedit; the area as read is kept as area.source.osm) and writes it (area.osm);
2. converts it with carla.Osm2Odr (SUMO's netconvert inside) projected with a transverse Mercator centred on the area
   (core.osm.proj_string, center_map off), so OpenDRIVE x, y are metres east and north of the centre and the
   geoReference is the area's — CARLA's default projection, centred on 0°E, stretches a place far from it (central
   Bengaluru by about 3× east-west and 4× north-south); objects and signals a few millimetres past their road's end
   (SUMO writes some) are clamped onto it, since CARLA's world generation crashes on them (core.xodr.sanitize);
3. with lht, converts the right-hand output to left-hand traffic (core.xodr.to_lht with the OSM ways, so each one-way
   road is mirrored about the line its way runs along; see docs/worlds.md);
4. places speed breakers where OSM tags traffic_calming (bump, hump, table) as elevation profiles that CARLA's road
   mesh renders (core.roadfeatures; world.base.xodr keeps the file without them, features.json says what was placed
   and skipped) unless settings speed_breakers is "off";
5. writes world.xodr, the road graph for the preview (graph.json), and import.json: what was read, converted and
   dropped, the lane check against OSM on a seeded sample (core.xodr.lane_check: lane counts and positions) and on
   every edited way, the lane links that do not meet, the map edits applied and skipped, the handedness, what an
   OpenDRIVE world does not contain (buildings, vegetation, textures), timings.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

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
        edits, edit_rep = cfg.get('edits') or [], None
        types = tuple((cfg.get('settings') or {}).get('way_types') or O.CARLA_HIGHWAYS)
        if edits:                                # map edits (core.mapedit) change what the converter reads
            from ..core import mapedit as ME
            O.write_osm(data, os.path.join(self.out_dir, 'area.source.osm'))     # the area as read, unedited
            data, edit_rep = ME.apply_edits(data, edits, types)
            if not [w for w in data.ways if w.get('kind') == 'highway']:
                raise Failed('the map edits leave no roads')
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
        if edit_rep is not None:
            notes.append(f'{len(edit_rep["applied"])} of {edit_rep["count"]} map edits applied'
                         + (f', {len(edit_rep["skipped"])} skipped (map_edits in import.json says why)'
                            if edit_rep['skipped'] else ''))
        text, fixes = X.sanitize(raw)
        notes += fixes
        if cfg.get('lht'):
            self.emit.progress(0.6, 'converting to left-hand traffic', force=True)
            text, lht_notes = X.to_lht(text, ways=X.osm_lines(data, O.local_projection(lat0, lon0)))
            notes += lht_notes
        if X.sanitize(text)[1]:                  # the file CARLA loads must have nothing past a road's end
            raise Failed('positions past a road\'s end remain after the conversion (a bug): world.xodr not written')
        with open(os.path.join(self.out_dir, 'world.xodr'), 'w', encoding='utf-8') as fh:
            fh.write(text)
        features = None
        if s.get('speed_breakers', 'osm') != 'off':     # traffic_calming as elevation profiles (core.roadfeatures)
            from ..core import roadfeatures as RF
            base = os.path.join(self.out_dir, 'world.base.xodr')
            if os.path.isfile(base):
                os.remove(base)                         # a re-import: the new conversion is the base
            opts = dict(cfg.get('road_features') or {'osm': True, 'osm_bump': s.get('osm_bump', 'osm_bump')})
            doc = RF.apply_to_world(self.out_dir, opts)
            features = {'placed': len(doc['placed']), 'counts': doc['counts'], 'skipped': len(doc['skipped']),
                        'not_represented': doc['not_represented'], 'notes': doc['notes'] + [doc['mesh_note']]}
            if doc['placed']:
                with open(os.path.join(self.out_dir, 'world.xodr'), encoding='utf-8') as fh:
                    text = fh.read()
                notes.append(f'{len(doc["placed"])} speed breakers placed from OpenStreetMap\'s traffic_calming '
                             f'tags as elevation profiles ({", ".join(f"{v} {k}" for k, v in doc["counts"].items())}; '
                             'features.json)')
        self.emit.progress(0.7, 'road graph and lane check', force=True)
        od = X.parse(text)
        graph = X.road_graph(od)
        write_json(os.path.join(self.out_dir, 'graph.json'), graph)
        proj = O.local_projection(lat0, lon0)
        check = X.lane_check(od, data, proj, sample=int(cfg.get('lane_check_sample', 50)),
                             seed=int(cfg.get('seed', 0)))
        pos = check.get('position') or {}
        if pos.get('checked'):
            notes.append(f'lane positions against OSM: median {pos["median_m"]} m, 95th percentile {pos["p95_m"]} m '
                         f'off; {pos["misplaced"]} of {pos["checked"]} checked ways misplaced by more than 0.75 of a '
                         'lane width (the converter\'s own placement included)')
        if edit_rep is not None:                 # the edited ways themselves, not a sample
            edit_rep['lane_check'] = X.lane_check(od, data, proj, ways=edit_rep['touched_ways'])
            edit_rep['signals'] = graph['totals'].get('signals')
        gaps = X.lane_link_gaps(od)
        links = {'links': int(len(gaps)), 'over_0_5_m': int((gaps > 0.5).sum()) if len(gaps) else 0,
                 'p95_m': round(float(np.percentile(gaps, 95)), 3) if len(gaps) else None,
                 'method': 'every lane link between roads: the distance between the two lanes\' centres where they '
                           'meet (core.xodr.lane_link_gaps)'}
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
               'timings': t, 'road_features': features, 'lane_links': links, 'map_edits': edit_rep,
               'signal_lanes': X.signal_lanes(text)}
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
