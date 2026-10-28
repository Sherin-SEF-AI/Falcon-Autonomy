"""Indian road assets within what the packaged CARLA 0.9.16 allows (standard library only): the measured inventory of
what the server can spawn, the Indian classes Falcon approximates with it — each a labelled stand-in or proxy chosen by
measured size closeness — and their cited sizes and kinematic limits.

    falcon assets inventory [--out FILE]   measure the server's blueprints (Falcon's own server; workers.assets)
    falcon assets list                     the Indian classes, their stand-ins and limits
    falcon assets show ID                  one class: its sources, the ranked candidates by size
    falcon assets verify [--out DIR]       the road-feature and asset checks on the server (docs/indian-assets.md)

What CARLA 0.9.16's packaged release allows (measured: falcon_autonomy/resources/carla/inventory-0.9.16.json and
docs/indian-assets.md): spawning the blueprints of its library — 41 vehicles, 52 walkers, 98 static props and
static.prop.mesh, which spawns any mesh cooked into the package by its path, with a uniform `scale`; it cannot import
a new mesh (that needs a source build with the Unreal editor: docs/indian-assets.md). No animal, three-wheeler,
rickshaw or handcart is among them, so:

* a stand-in (vehicles) drives as a CARLA vehicle of the nearest measured size, with the real class's kinematic limits
  (its top speed as the traffic manager's desired speed per vehicle, its acceleration by scaling the stand-in's engine
  torque, both measured on the server);
* a proxy (obstacles) is a prop — or, when it stands still, a cooked mesh spawned through static.prop.mesh at the
  uniform scale that best matches — of the real object's measured or cited size, placed as the object would stand,
  moved kinematically (set_transform) when the object moves. Measured on 0.9.16: a static.prop.* actor moves so; a
  static.prop.mesh actor does not move (set_transform and set_location leave it, set_transform resets its scale to 1)
  and is not drawn when attached to an empty actor; so a moving proxy is an unscaled prop.

Labels: the semantic and instance classes stay CARLA's (a stand-in autorickshaw is a car mesh, tagged car); objects
JSON and actors.jsonl rows carry Falcon's `subclass` ("autorickshaw (stand-in)") — docs/dataset-format.md says what it
means for training.

Size closeness: the root mean square of the relative differences of length, width and height from the real class's
(size_score); a prop may be turned 90° (its width along the road) and uniformly scaled within SCALE_RANGE (the scale
that minimises the score, in closed form).
"""
from __future__ import annotations

import json
import math
import os

INVENTORY_SCHEMA = 'falcon.carla_inventory/1'
RESOURCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources', 'carla',
                        'inventory-0.9.16.json')
SCALE_RANGE = (0.5, 2.0)

# sources (cited in docs/indian-assets.md)
BAJAJ_RE = ('Bajaj Auto, "RE" three-wheeler specifications, bajajauto.com/three-wheelers/re/specifications '
            '(read 2026-10-29): overall length 2635 mm, width 1300 mm, height 1700 mm, wheelbase 2000 mm, kerb '
            'weight 362–398 kg, 6.2–7.6 kW')
BAJAJ_RE_SPEED = ('max speed 65 km/h for the Bajaj RE in a dealer listing (trucks.tractorjunction.com, Bajaj RE '
                  'specifications; read 2026-10-29); Bajaj\'s own specification page gives none')
THREE_WHEELER_ACCEL = ('Pathivada BK, Vedagiri P (2017), "Modeling acceleration characteristics of motorized '
                       'three-wheeler", ICAHETS 2017, University of Moratuwa: GPS performance-box data near free flow, '
                       'maximum acceleration 1.88–2.08 m/s², higher at low speeds')
IRC_11 = ('IRC:11-2015 (Indian Roads Congress), "Recommended practice for the design and layout of cycle tracks", '
          'Table 1: passenger rickshaw 2200 mm long, 1000 mm wide with rider, 1200 mm high; goods rickshaw 2400 × '
          '1220 × 1200 mm; §5.2: non-motorised traffic averages 5–15 km/h, rarely 20 km/h')
CATTLE_SRC = ('Thalkar MG, Bhagat DJ et al. (2016), "Morphological and morphometric features of non-descript cattle '
              'in Raigad district of Maharashtra", Journal of Livestock Biodiversity 6(2), ICAR ePubs: 360 animals, '
              'mean body length 95.69 cm, height at withers 85.10 cm, chest girth 139.55 cm, face length 39.96 cm')

CLASSES = {
    'autorickshaw': {
        'label': 'autorickshaw (stand-in)', 'kind': 'vehicle', 'size_m': (2.635, 1.300, 1.700),
        'size_source': BAJAJ_RE, 'base_types': None,
        'top_kmh': 65.0, 'top_source': BAJAJ_RE_SPEED,
        'accel_mps2': 2.0, 'accel_source': THREE_WHEELER_ACCEL + ': 2.0 m/s² (their range\'s middle) from rest',
        # measured on 0.9.16 (falcon assets verify, stage accel; flat road, full throttle): the Microlino reaches
        # 30 km/h in 2.90 s as it is, 4.22 s with its torque × 0.7 (a mean 1.97 m/s²; 2.0 m/s² takes 4.17 s); its
        # peak over 0.5 s at launch stays 4.5 m/s² (CARLA's first gear), its top speed 79 km/h (the cap is 65)
        'torque_scale': 0.7,
        'about': 'A motorised three-wheeler (Bajaj RE class). CARLA has none: the car nearest in measured size '
                 'drives in its place with the three-wheeler\'s top speed and acceleration.'},
    'cycle_rickshaw': {
        'label': 'cycle-rickshaw (stand-in)', 'kind': 'vehicle', 'size_m': (2.200, 1.000, 1.200),
        'size_source': IRC_11 + ' (the passenger rickshaw)', 'base_types': ('bicycle',),
        'top_kmh': 15.0, 'top_source': IRC_11 + ': the upper average',
        'accel_mps2': None, 'accel_source': 'no measured figure found: the stand-in\'s own (a pedalled bicycle)',
        'about': 'A pedalled three-wheeler. Chosen among CARLA\'s bicycles (a pedalling rider; semantic bicycle and '
                 'rider), the nearest in measured size; the nearest of all vehicles is a motorcycle, whose rider and '
                 'class would mislabel it.'},
    'cattle': {
        'label': 'cattle (proxy obstacle)', 'kind': 'prop', 'size_m': (1.61, 0.44, 0.85),
        'size_source': CATTLE_SRC + '. Length on the road = body length + face length + 0.25 m for the neck, width = '
                                    'chest girth / π (both Falcon\'s, heuristic); height = height at withers',
        'top_kmh': 0.0, 'top_source': 'stray cattle standing or lying in the lane',
        'about': 'A stationary obstacle of a non-descript cow\'s size in the lane. It does not look like a cow: '
                 'appearance-based perception is not tested by it, only that something of that size is there.'},
    'handcart': {
        'label': 'handcart (proxy)', 'kind': 'prop', 'size_m': (2.400, 1.220, 1.200),
        'size_source': IRC_11 + ' (the goods rickshaw: the NMT goods design vehicle; IRC:11 names vendors\' '
                                'four-wheeled trolleys without giving their size)',
        'top_kmh': 6.0, 'top_source': 'walking pace (pushed): Falcon\'s range 3–6 km/h (heuristic)', 'moves': True,
        'about': 'A pushed goods cart moving at walking pace near the kerb: a prop of the goods rickshaw\'s size, '
                 'moved kinematically, with a pedestrian walking behind it.'},
    'street_vendor': {
        'label': 'street vendor (proxy)', 'kind': 'prop', 'size_m': None,
        'size_source': 'CARLA\'s food cart prop as it is (a cart with an umbrella); its size is CARLA\'s',
        'top_kmh': 0.0, 'top_source': 'stationary',
        'fixed': 'static.prop.foodcart',
        'about': 'A vendor\'s cart standing at the kerb with a pedestrian beside it: CARLA\'s food cart, a European '
                 'street-food cart, not an Indian thela.'},
}


# ------------------------------------------------------------------------------------------------ inventory
def _clean(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def save_inventory(doc: dict, path: str = RESOURCE, package_json: str = '') -> dict:
    """Write an inventory (the assets worker's inventory.json) as the repo's data: NaN and ±inf (an empty mesh's box)
    as null, each prop's mesh path from CARLA's Default.Package.json when given, one entry per line."""
    doc = _clean(doc)
    if package_json and os.path.isfile(package_json):
        with open(package_json, encoding='utf-8') as fh:
            props = json.load(fh).get('props') or []
        paths = {'static.prop.' + p['name'].lower(): p['path'] for p in props}
        for row in doc.get('props', []):
            if row['id'] in paths:
                row['mesh_path'] = paths[row['id']]
        doc['package_json'] = os.path.basename(package_json)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ['{']
    keys = list(doc)
    for i, k in enumerate(keys):
        v = doc[k]
        end = ',' if i < len(keys) - 1 else ''
        if isinstance(v, list):
            lines.append(f'  {json.dumps(k)}: [')
            lines += [f'    {json.dumps(x, separators=(",", ":"))}{"," if j < len(v) - 1 else ""}'
                      for j, x in enumerate(v)]
            lines.append(f'  ]{end}')
        else:
            lines.append(f'  {json.dumps(k)}: {json.dumps(v, separators=(",", ":"))}{end}')
    lines.append('}')
    with open(path + '.tmp', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    os.replace(path + '.tmp', path)
    return doc


def load_inventory(path: str = RESOURCE) -> dict:
    with open(path, encoding='utf-8') as fh:
        doc = json.load(fh)
    if doc.get('schema') != INVENTORY_SCHEMA:
        raise ValueError(f'{path}: not a CARLA inventory ({doc.get("schema")!r})')
    return doc


def sizes(inv: dict, family: str) -> dict:
    """{blueprint id: (length, width, height)} of the spawned, non-empty measured actors of a family (vehicles,
    walkers, props)."""
    out = {}
    for r in inv.get(family) or []:
        b = r.get('box') or {}
        s = b.get('size_m')
        if r.get('spawned') and s and all(isinstance(x, (int, float)) and x > 0 for x in s):
            out[r['id']] = tuple(float(x) for x in s)
    return out


# ------------------------------------------------------------------------------------------------ choosing
def size_score(size, target) -> float:
    """The RMS of the relative differences of (length, width, height)."""
    return math.sqrt(sum(((a - t) / t) ** 2 for a, t in zip(size, target)) / 3.0)


def best_scale(size, target, lo: float = SCALE_RANGE[0], hi: float = SCALE_RANGE[1]) -> float:
    """The uniform scale s minimising size_score(s · size, target): Σ(r_i) / Σ(r_i²) with r_i = size_i / target_i
    (the score is a quadratic in s), clipped to [lo, hi]."""
    r = [a / t for a, t in zip(size, target)]
    s = sum(r) / sum(x * x for x in r)
    return min(max(s, lo), hi)


def rank_vehicles(cls: str, inv: dict, base_types=None, top: int = 5) -> list:
    """Vehicles by size closeness to the class: [{'id', 'size_m', 'score', 'base_type'}], best first. base_types
    (default the class's) restricts the candidates."""
    spec = CLASSES[cls]
    want = tuple(spec['size_m'])
    bt = spec.get('base_types') if base_types is None else base_types
    rows = []
    meta = {r['id']: r for r in inv.get('vehicles') or []}
    for bid, s in sizes(inv, 'vehicles').items():
        b = str(meta[bid].get('base_type') or '').lower()
        if bt and b not in bt:
            continue
        rows.append({'id': bid, 'size_m': list(s), 'score': round(size_score(s, want), 4), 'base_type': b})
    rows.sort(key=lambda r: (r['score'], r['id']))
    return rows[:top] if top else rows


def rank_props(cls: str, inv: dict, top: int = 5, scaled: bool = True) -> list:
    """Props by size closeness to the class, each possibly turned 90° and (with a mesh path) uniformly scaled:
    [{'id', 'mesh_path', 'scale', 'yaw', 'size_m' (as placed), 'score'}], best first."""
    want = tuple(CLASSES[cls]['size_m'])
    rows = []
    meta = {r['id']: r for r in inv.get('props') or []}
    for pid, s in sizes(inv, 'props').items():
        mesh = meta[pid].get('mesh_path')
        for yaw, (l, w, h) in ((0.0, s), (90.0, (s[1], s[0], s[2]))):
            sc = best_scale((l, w, h), want) if scaled and mesh else 1.0
            placed = (l * sc, w * sc, h * sc)
            rows.append({'id': pid, 'mesh_path': mesh if sc != 1.0 else None, 'scale': round(sc, 4), 'yaw': yaw,
                         'size_m': [round(x, 3) for x in placed], 'score': round(size_score(placed, want), 4)})
    rows.sort(key=lambda r: (r['score'], r['id'], r['yaw']))
    return rows[:top] if top else rows


def choose(cls: str, inv: dict | None = None) -> dict:
    """The class's stand-in or proxy from the inventory (the repo's when None): {'class', 'label', 'kind', 'blueprint'
    or prop fields, 'size_m' (as spawned), 'target_m', 'score', 'ranked'}."""
    inv = inv if inv is not None else load_inventory()
    spec = CLASSES[cls]
    out = {'class': cls, 'label': spec['label'], 'kind': spec['kind'], 'target_m': spec['size_m'],
           'top_kmh': spec.get('top_kmh'), 'accel_mps2': spec.get('accel_mps2')}
    if spec.get('fixed'):
        s = sizes(inv, 'props').get(spec['fixed'])
        out.update(id=spec['fixed'], blueprint=spec['fixed'], mesh_path=None, scale=1.0, yaw=0.0,
                   size_m=list(s) if s else None, score=None, ranked=[])
        return out
    # a moving proxy is an unscaled prop: a scaled static.prop.mesh cannot be moved (measured on 0.9.16)
    ranked = rank_vehicles(cls, inv) if spec['kind'] == 'vehicle' else rank_props(cls, inv,
                                                                                  scaled=not spec.get('moves'))
    if not ranked:
        raise ValueError(f'no candidate in the inventory for {cls}')
    best = ranked[0]
    out.update(best, ranked=ranked)
    out['blueprint'] = 'static.prop.mesh' if best.get('mesh_path') else best['id']
    if spec['kind'] == 'vehicle':
        out['nearest_any'] = rank_vehicles(cls, inv, base_types=(), top=3)
    return out


def placement_z(inv: dict, pid: str, scale: float = 1.0) -> float:
    """How high to spawn a prop's origin so its box rests on the ground (its box's bottom, centre z − extent z, is
    measured from the origin; scaled with the prop)."""
    row = next((r for r in inv.get('props') or [] if r['id'] == pid), None)
    if not row or not row.get('box'):
        return 0.0
    c, e = row['box']['center'][2], row['box']['extent'][2]
    return round(-(c - e) * scale, 4)


# ------------------------------------------------------------------------------------------------ runtime
def subclass_of(type_id: str, role: str = '', table: dict | None = None, actor_id=None) -> str | None:
    """The Falcon subclass of an actor: a run's own table {actor id: label} first, else None (a plain CARLA actor —
    a Microlino in a fleet that does not name it a stand-in is a car)."""
    if table and actor_id is not None:
        return table.get(actor_id) or table.get(str(actor_id))
    return None


def desired_speed_kmh(limit_kmh: float, diff_pct: float, top_kmh: float | None) -> float:
    """A stand-in's traffic-manager speed: what the traffic manager would drive (the road's limit less the vehicle's
    percentage difference), capped at the class's top speed. CARLA's set_desired_speed replaces the limit-relative
    target, so the limit is applied here (re-read as the vehicle moves)."""
    v = float(limit_kmh) * (1.0 - float(diff_pct) / 100.0)
    return max(0.0, min(v, float(top_kmh))) if top_kmh else max(0.0, v)


def fleet_classes(profile: dict) -> dict:
    """{fleet key: class id} of a traffic profile's stand-ins ('fleet_classes'), with the Indian profile's default."""
    return dict((profile or {}).get('fleet_classes') or {})


# ------------------------------------------------------------------------------------------------ CLI
def _table(rows, cols) -> str:
    w = {c: max(len(c), *(len(str(r.get(c, ''))) for r in rows)) for c in cols}
    out = ['  '.join(c.ljust(w[c]) for c in cols)]
    out += ['  '.join(str(r.get(c, '')).ljust(w[c]) for c in cols) for r in rows]
    return '\n'.join(out)


def main_assets(argv) -> int:
    """`falcon assets inventory|list|show ID|verify`. 0 ok, 2 failed."""
    import argparse
    ap = argparse.ArgumentParser(prog='falcon assets', description='Indian road assets within the packaged CARLA')
    sub = ap.add_subparsers(dest='cmd', required=True)
    i = sub.add_parser('inventory', help='measure the server\'s blueprints (Falcon\'s own server)')
    i.add_argument('--out', default='', help='write here instead of the repo\'s inventory')
    i.add_argument('--run-dir', default='', help='the worker\'s run folder (default: a temporary one)')
    sub.add_parser('list', help='the Indian classes, their stand-ins and limits')
    s = sub.add_parser('show', help='one class: its sources and ranked candidates')
    s.add_argument('cls', choices=sorted(CLASSES))
    v = sub.add_parser('verify', help='the road-feature and asset checks on Falcon\'s own server')
    v.add_argument('--out', required=True)
    v.add_argument('--stages', default='', help='comma-separated (default: all)')
    for p in (i, v):
        p.add_argument('--keep-server', action='store_true')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    if a.cmd == 'list':
        try:
            inv = load_inventory()
        except (OSError, ValueError) as e:
            print(f'failed: {e}', flush=True)
            return 2
        rows = []
        for k in CLASSES:
            c = choose(k, inv)
            mesh = f' {c["mesh_path"].rsplit("/", 1)[-1]} ×{c["scale"]:g}' if c.get('mesh_path') else ''
            rows.append({'class': k, 'label': c['label'], 'as': c['blueprint'] + mesh,
                         'size (m)': ' × '.join(f'{x:.2f}' for x in c['size_m']) if c.get('size_m') else '',
                         'real (m)': ' × '.join(f'{x:.2f}' for x in c['target_m']) if c.get('target_m') else '',
                         'score': c.get('score') if c.get('score') is not None else '',
                         'top km/h': c.get('top_kmh') if c.get('top_kmh') is not None else ''})
        print(_table(rows, ['class', 'label', 'as', 'size (m)', 'real (m)', 'score', 'top km/h']))
        print(f'\ninventory: {inv.get("server")} ({inv.get("map")}), {len(inv.get("vehicles", []))} vehicles, '
              f'{len(inv.get("walkers", []))} walkers, {len(inv.get("props", []))} props; mesh paths: '
              f'{(inv.get("mesh_spawning") or {}).get("verdict")}')
        return 0
    if a.cmd == 'show':
        inv = load_inventory()
        spec = CLASSES[a.cls]
        c = choose(a.cls, inv)
        print(f'{a.cls}: {spec["label"]} — {spec["about"]}')
        print(f'size: {spec["size_source"]}')
        if spec.get('top_kmh') is not None:
            print(f'top speed {spec["top_kmh"]:g} km/h: {spec["top_source"]}')
        if spec.get('accel_mps2'):
            print(f'acceleration {spec["accel_mps2"]:g} m/s²: {spec["accel_source"]}')
        if c['ranked']:
            print(_table([dict(r, size_m=' × '.join(f'{x:.2f}' for x in r['size_m'])) for r in c['ranked']],
                         [k for k in ('id', 'base_type', 'scale', 'yaw', 'size_m', 'score') if k in c['ranked'][0]]))
        return 0
    return _run_on_server(a)


def _run_on_server(a) -> int:
    """inventory / verify: the assets worker on Falcon's own server (started under its memory cap, stopped after)."""
    import tempfile
    from .jobs import FINAL, JobQueue, SubprocessLauncher
    from .settings import Settings
    from ..sim.service import CarlaService
    st = Settings()
    svc = CarlaService(st, JobQueue(SubprocessLauncher()), '')
    run_dir = (getattr(a, 'run_dir', '') or (a.out if a.cmd == 'verify' else '')) or \
        tempfile.mkdtemp(prefix='falcon-inv-')
    os.makedirs(run_dir, exist_ok=True)
    cfg = {'schema': 'falcon.assets/1', 'mode': a.cmd, 'out': os.path.abspath(run_dir), 'port': svc.port,
           'tm_port': svc.tm_port}
    if a.cmd == 'verify' and a.stages:
        cfg['stages'] = [x.strip() for x in a.stages.split(',') if x.strip()]
    cp = os.path.join(run_dir, 'assets.json')
    with open(cp, 'w', encoding='utf-8') as fh:
        json.dump(cfg, fh, indent=1)
    import time
    code = 2
    try:
        job = svc.queue.submit(svc.worker_spec('assets', [cp], f'assets {a.cmd}', 'assets', run_dir, memory_gb=1.0))
        last = ''
        while job.state not in FINAL:
            svc.queue.poll()
            line = job.summary()
            if line != last:
                print(f'[assets {a.cmd}] {line}', flush=True)
                last = line
            time.sleep(0.5)
        code = 0 if job.state == 'done' else 2
        if code:
            for lvl, text in job.log[-20:]:
                if lvl in ('warning', 'error', 'stderr'):
                    print(f'  {lvl}: {text}', flush=True)
    finally:
        if not a.keep_server:
            svc.shutdown()
    if code == 0 and a.cmd == 'inventory':
        from ..sim.server import candidate_roots
        root = st.get('carla_root') or next(iter(candidate_roots()), '')
        pj = os.path.join(root, 'CarlaUE4', 'Content', 'Carla', 'Config', 'Default.Package.json') if root else ''
        with open(os.path.join(run_dir, 'inventory.json'), encoding='utf-8') as fh:
            doc = json.load(fh)
        out = a.out or RESOURCE
        save_inventory(doc, out, pj)
        print(f'wrote {out}', flush=True)
    return code
