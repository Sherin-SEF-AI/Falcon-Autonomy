"""OpenSCENARIO 1.x: export a concrete Falcon scenario to .xosc, and import that subset back (standard library).

ScenarioRunner (CARLA's OpenSCENARIO interpreter) is not part of a CARLA 0.9.16 install and is not on this machine;
Falcon runs scenarios with its own director (workers.director) and reads and writes this documented subset:

* FileHeader; ParameterDeclarations (the concrete values; a distribution's range is kept as a <ValueConstraintGroup>
  on its parameter — OpenSCENARIO 1.2's constraints — with the distribution named in a falcon: comment);
* Entities: one ScenarioObject per actor and the ego ("hero"), each a Vehicle (vehicleCategory car, truck, bus, van,
  motorbike, bicycle) or a Pedestrian, with its bounding box;
* Init: TeleportAction to a RelativeLanePosition from the ego (dLane: 0 own lane, ±1 the neighbour lanes, measured
  in the lane's own direction; ds metres along it; offset metres sideways) and an AbsoluteTargetSpeed;
* Story → Act → ManeuverGroup → Maneuver → Event per timeline entry, its StartTrigger a SimulationTimeCondition or a
  RelativeDistanceCondition (freespace false, rule lessThan) — Falcon's "ahead" trigger is a RelativeDistanceCondition
  with rule greaterThan and relativeDistanceType longitudinal;
* actions: SpeedAction (AbsoluteTargetSpeed, linear dynamics at the given rate), LaneChangeAction (RelativeTargetLane,
  time dynamics), CustomCommandAction for what OpenSCENARIO 1.x has no standard action for (type "falcon:open_door",
  "falcon:walk", "falcon:go", "falcon:stop");
* StopTrigger: a SimulationTimeCondition at the scenario's duration.

Recorded events (core.incident): the ego's start as a TeleportAction to a WorldPosition and its route as an
AssignRouteAction through WorldPosition waypoints (every 5 m); a recorded track as a FollowTrajectoryAction (a
Trajectory of a Polyline of timed WorldPositions, absolute timing, followingMode position) in an event at time 0 of
its own maneuver group ("tracks"), the actor's Init a WorldPosition at its first pose. WorldPositions are in
OpenDRIVE's frame (y north, h counter-clockwise in radians): CARLA's (x, −y, −yaw).

Lane names relative to the handedness ("kerb", "sidewalk", "adjacent", "line") are kept in each entity's
<Properties> (falcon:lane), since OpenSCENARIO's relative lanes depend on the road's lane numbering. Anything outside
this subset is refused on import, saying what.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from .. import __version__

CATEGORY = {'car': 'car', 'truck': 'truck', 'bus': 'bus', 'van': 'van', 'motorcycle': 'motorbike',
            'bicycle': 'bicycle'}
SIZES = {'bus': (10.3, 3.9, 3.8), 'truck': (6.0, 2.6, 3.0), 'van': (5.5, 2.0, 2.3), 'motorcycle': (2.2, 0.9, 1.5),
         'bicycle': (1.7, 0.6, 1.6), 'pedestrian': (0.5, 0.5, 1.8), 'child': (0.4, 0.4, 1.2)}
# the Indian kinds (core.assets): the real class's size; exported as a Vehicle (OpenSCENARIO has no rickshaw; a proxy
# obstacle is a Vehicle of category car here, its falcon:kind property says what it is — a known limit of the subset)
EXPORT_CATEGORY = {'autorickshaw': 'car', 'cycle_rickshaw': 'bicycle'}      # export only (import reads falcon:kind)
SIZES.update({'autorickshaw': (2.635, 1.3, 1.7), 'cycle_rickshaw': (2.2, 1.0, 1.2), 'cattle': (1.61, 0.44, 0.85),
              'handcart': (2.4, 1.22, 1.2), 'vendor': (2.3, 1.5, 3.0)})
DLANE = {'ego': 0, 'line': 0, 'kerb': 0, 'sidewalk': 0, 'adjacent': 1, 'opposite': -1, 'far_sidewalk': -1}


def _sub(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def export_xosc(scn: dict) -> str:
    """A concrete scenario (core.scenarios.sample / concrete) as OpenSCENARIO 1.2 XML."""
    root = ET.Element('OpenSCENARIO')
    _sub(root, 'FileHeader', revMajor=1, revMinor=2, date='2026-01-01T00:00:00', author=f'Falcon Autonomy {__version__}',
         description=f'{scn.get("name", "")}: {scn.get("about", "")}')
    pd = _sub(root, 'ParameterDeclarations')
    for name, value in sorted((scn.get('values') or {}).items()):
        p = scn.get('params', {}).get(name, {})
        typ = 'string' if isinstance(value, str) else 'double'
        e = _sub(pd, 'ParameterDeclaration', name=name, parameterType=typ, value=value)
        if p.get('dist') == 'uniform':
            e.append(ET.Comment(f' falcon:distribution uniform {p["lo"]} {p["hi"]} {p.get("unit", "")} '))
            cg = _sub(e, 'ConstraintGroup')
            _sub(cg, 'ValueConstraint', rule='greaterOrEqual', value=p['lo'])
            _sub(cg, 'ValueConstraint', rule='lessOrEqual', value=p['hi'])
        elif p.get('dist') == 'choice':
            e.append(ET.Comment(f' falcon:distribution choice {"|".join(map(str, p["choices"]))} '))
    _sub(root, 'CatalogLocations')
    rn = _sub(root, 'RoadNetwork')
    _sub(rn, 'LogicFile', filepath='')
    ents = _sub(root, 'Entities')

    def entity(name, kind, lane=''):
        so = _sub(ents, 'ScenarioObject', name=name)
        L, W, H = SIZES.get(kind, (4.6, 2.0, 1.5))
        if kind in ('pedestrian', 'child'):
            e = _sub(so, 'Pedestrian', name=name, mass=60 if kind == 'pedestrian' else 30,
                     pedestrianCategory='pedestrian', model=kind)
        else:
            e = _sub(so, 'Vehicle', name=name, vehicleCategory=CATEGORY.get(kind) or EXPORT_CATEGORY.get(kind, 'car'))
            _sub(e, 'Performance', maxSpeed=70, maxAcceleration=10, maxDeceleration=10)
        _sub(e, 'ParameterDeclarations')
        bb = _sub(e, 'BoundingBox')
        _sub(bb, 'Center', x=0, y=0, z=H / 2)
        _sub(bb, 'Dimensions', width=W, length=L, height=H)
        if kind not in ('pedestrian', 'child'):
            ax = _sub(e, 'Axles')
            _sub(ax, 'FrontAxle', maxSteering=0.5, wheelDiameter=0.6, trackWidth=W * 0.85, positionX=L * 0.3,
                 positionZ=0.3)
            _sub(ax, 'RearAxle', maxSteering=0, wheelDiameter=0.6, trackWidth=W * 0.85, positionX=-L * 0.3,
                 positionZ=0.3)
        props = _sub(e, 'Properties')
        _sub(props, 'Property', name='falcon:kind', value=kind)
        if lane:
            _sub(props, 'Property', name='falcon:lane', value=lane)
        return e
    ego = scn.get('ego', {})
    entity('hero', 'car')
    for a in scn.get('actors', []):
        entity(a['id'], a['kind'], a.get('place', {}).get('lane', 'ego'))
    sb = _sub(root, 'Storyboard')
    init = _sub(_sub(sb, 'Init'), 'Actions')

    def private(entity_ref, actions):
        pr = _sub(init, 'Private', entityRef=entity_ref)
        for a in actions:
            pr.append(a)
    hero = [_speed_action(float(ego.get('speed_kmh', 30)) / 3.6, None)]
    if scn.get('start'):
        st = scn['start']
        hero.insert(0, _teleport_world(st['x'], st['y'], st.get('yaw', 0.0)))
    if scn.get('route_xy'):
        hero.append(_route_action(scn['route_xy']))
    private('hero', hero)
    if ego.get('driver'):
        init.findall('Private')[-1].append(ET.Comment(f' falcon:driver {ego["driver"]} '))
    for a in scn.get('actors', []):
        pl = a.get('place', {})
        if a.get('track'):
            tr = a['track']
            pa = _teleport_world(tr['x'][0], tr['y'][0], tr['yaw'][0])
            pa.append(ET.Comment(f' falcon:place {pl.get("lane", "ego")} {float(pl.get("ahead_m", 0.0))} '
                                 f'{float(pl.get("lateral_m", 0.0))} '))
        else:
            pa = ET.Element('PrivateAction')
            tp = _sub(_sub(pa, 'TeleportAction'), 'Position')
            _sub(tp, 'RelativeLanePosition', entityRef='hero', dLane=DLANE.get(pl.get('lane', 'ego'), 0),
                 ds=float(pl.get('ahead_m', 0.0)), offset=float(pl.get('lateral_m', 0.0)))
        moving = a.get('start', 'moving') == 'moving'
        private(a['id'], [pa, _speed_action(float(a.get('speed_kmh', 0)) / 3.6 if moving else 0.0, None)])
        pr = init.findall('Private')[-1]
        pr.append(ET.Comment(f' falcon:heading {a.get("heading", "lane")} start {a.get("start", "moving")} '))
    story = _sub(sb, 'Story', name='story')
    act = _sub(story, 'Act', name='act')
    for k, ev in enumerate(scn.get('timeline', [])):
        mg = _sub(act, 'ManeuverGroup', maximumExecutionCount=1, name=f'mg{k}')
        actors = _sub(mg, 'Actors', selectTriggeringEntities='false')
        for a in {x['actor'] for x in ev.get('actions', [])}:
            _sub(actors, 'EntityRef', entityRef=a)
        man = _sub(mg, 'Maneuver', name=f'm{k}')
        event = _sub(man, 'Event', name=f'e{k}', priority='override', maximumExecutionCount=1)
        for i, a in enumerate(ev.get('actions', [])):
            act_el = _sub(event, 'Action', name=f'a{k}_{i}')
            act_el.append(ET.Comment(f' falcon:actor {a["actor"]} '))
            act_el.append(_action(a))
        event.append(_trigger(ev.get('trigger', {})))
    tracked = ([('hero', ego['track'])] if ego.get('track') else []) + [(a['id'], a['track'])
                                                                        for a in scn.get('actors', []) if a.get('track')]
    if tracked:
        mg = _sub(act, 'ManeuverGroup', maximumExecutionCount=1, name='tracks')
        actors = _sub(mg, 'Actors', selectTriggeringEntities='false')
        for ref, _ in tracked:
            _sub(actors, 'EntityRef', entityRef=ref)
        man = _sub(mg, 'Maneuver', name='tracks')
        for ref, tr in tracked:
            event = _sub(man, 'Event', name=f'track_{ref}', priority='parallel', maximumExecutionCount=1)
            act_el = _sub(event, 'Action', name=f'follow_{ref}')
            act_el.append(ET.Comment(f' falcon:actor {ref} '))
            act_el.append(_follow_trajectory(ref, tr))
            event.append(_trigger({'type': 'time', 't_s': 0.0}))
    st = _sub(act, 'StartTrigger')
    cg = _sub(st, 'ConditionGroup')
    c = _sub(cg, 'Condition', name='start', delay=0, conditionEdge='rising')
    _sub(_sub(c, 'ByValueCondition'), 'SimulationTimeCondition', value=0, rule='greaterThan')
    stop = _sub(sb, 'StopTrigger')
    c = _sub(_sub(stop, 'ConditionGroup'), 'Condition', name='end', delay=0, conditionEdge='rising')
    _sub(_sub(c, 'ByValueCondition'), 'SimulationTimeCondition', value=float(scn.get('duration_s', 12)),
         rule='greaterThan')
    ET.indent(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding='unicode') + '\n'


def _world(parent, x, y, yaw_deg):
    """A WorldPosition from CARLA's frame (module doc)."""
    return _sub(parent, 'WorldPosition', x=round(float(x), 4), y=round(-float(y), 4), z=0,
                h=round(-math.radians(float(yaw_deg)), 6))


def _teleport_world(x, y, yaw_deg):
    pa = ET.Element('PrivateAction')
    _world(_sub(_sub(pa, 'TeleportAction'), 'Position'), x, y, yaw_deg)
    return pa


def _route_action(pts, step=5.0):
    """AssignRouteAction > Route > Waypoint+ (OpenSCENARIO 1.2) through the route's points every `step` m."""
    pa = ET.Element('PrivateAction')
    route = _sub(_sub(_sub(pa, 'RoutingAction'), 'AssignRouteAction'), 'Route', name='route', closed='false')
    keep, last = [], None
    for k, (x, y) in enumerate(pts):
        if last is None or k == len(pts) - 1 or math.hypot(x - last[0], y - last[1]) >= step:
            keep.append((float(x), float(y)))
            last = (x, y)
    for k, (x, y) in enumerate(keep):
        a, b = (keep[k], keep[k + 1]) if k + 1 < len(keep) else (keep[max(k - 1, 0)], keep[k])
        yaw = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) if a != b else 0.0
        _world(_sub(_sub(route, 'Waypoint', routeStrategy='shortest'), 'Position'), x, y, yaw)
    return pa


def _follow_trajectory(ref, tr):
    pa = ET.Element('PrivateAction')
    fta = _sub(_sub(pa, 'RoutingAction'), 'FollowTrajectoryAction')
    traj = _sub(_sub(fta, 'TrajectoryRef'), 'Trajectory', name=f'track_{ref}', closed='false')
    _sub(traj, 'ParameterDeclarations')
    poly = _sub(_sub(traj, 'Shape'), 'Polyline')
    for t, x, y, yaw in zip(tr['t'], tr['x'], tr['y'], tr['yaw']):
        _world(_sub(_sub(poly, 'Vertex', time=round(float(t), 4)), 'Position'), x, y, yaw)
    _sub(_sub(fta, 'TimeReference'), 'Timing', domainAbsoluteRelative='absolute', scale=1, offset=0)
    _sub(fta, 'TrajectoryFollowingMode', followingMode='position')
    return pa


def _carla(wpos) -> tuple:
    return (float(wpos.get('x', 0)), -float(wpos.get('y', 0)), -math.degrees(float(wpos.get('h', 0))))


def _speed_action(v, rate):
    pa = ET.Element('PrivateAction')
    sa = _sub(_sub(pa, 'LongitudinalAction'), 'SpeedAction')
    if rate is None:
        _sub(sa, 'SpeedActionDynamics', dynamicsShape='step', value=0, dynamicsDimension='time')
    else:
        _sub(sa, 'SpeedActionDynamics', dynamicsShape='linear', value=rate, dynamicsDimension='rate')
    _sub(_sub(sa, 'SpeedActionTarget'), 'AbsoluteTargetSpeed', value=v)
    return pa


def _action(a):
    do = a['do']
    if do == 'speed':
        return _speed_action(float(a['kmh']) / 3.6, float(a.get('accel', 3.0)))
    if do == 'brake':
        return _speed_action(0.0, float(a.get('decel', 6.0)))
    if do == 'lane_change':
        pa = ET.Element('PrivateAction')
        lc = _sub(_sub(pa, 'LateralAction'), 'LaneChangeAction')
        _sub(lc, 'LaneChangeActionDynamics', dynamicsShape='linear', value=float(a.get('duration_s', 2.0)),
             dynamicsDimension='time')
        _sub(_sub(lc, 'LaneChangeTarget'), 'RelativeTargetLane', entityRef='hero', value=0)
        lc.append(ET.Comment(f' falcon:to {a.get("to", "ego")} '))
        return pa
    ua = ET.Element('UserDefinedAction')
    extra = {k: v for k, v in a.items() if k not in ('actor', 'do')}
    cc = _sub(ua, 'CustomCommandAction', type=f'falcon:{do}')
    cc.text = ' '.join(f'{k}={v}' for k, v in sorted(extra.items()))
    return ua


def _trigger(t):
    st = ET.Element('StartTrigger')
    c = _sub(_sub(st, 'ConditionGroup'), 'Condition', name='trigger', delay=0, conditionEdge='rising')
    if t.get('type') == 'time':
        _sub(_sub(c, 'ByValueCondition'), 'SimulationTimeCondition', value=float(t['t_s']), rule='greaterThan')
    else:
        bec = _sub(c, 'ByEntityCondition')
        te = _sub(bec, 'TriggeringEntities', triggeringEntitiesRule='any')
        _sub(te, 'EntityRef', entityRef='hero')
        ahead = t.get('type') == 'ahead'
        _sub(_sub(bec, 'EntityCondition'), 'RelativeDistanceCondition', entityRef=t['actor'], freespace='false',
             relativeDistanceType='longitudinal' if ahead else 'cartesianDistance',
             value=float(t['gt_m'] if ahead else t['lt_m']), rule='greaterThan' if ahead else 'lessThan',
             coordinateSystem='entity')
    return st


def import_xosc(text: str) -> dict:
    """A concrete Falcon scenario from an .xosc of this subset (fixed parameter values)."""
    # Falcon's notes are XML comments: keep them (ElementTree drops comments by default)
    root = ET.fromstring(text, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    if root.tag != 'OpenSCENARIO':
        raise ValueError('not an OpenSCENARIO file')
    hdr = root.find('FileHeader')
    values = {}
    for p in root.findall('ParameterDeclarations/ParameterDeclaration'):
        v = p.get('value')
        values[p.get('name')] = float(v) if p.get('parameterType') in ('double', 'int', 'integer') else v
    kinds, lanes = {}, {}
    for so in root.findall('Entities/ScenarioObject'):
        e = so.find('Vehicle')
        if e is None:
            e = so.find('Pedestrian')
        if e is None:
            raise ValueError(f'entity {so.get("name")}: only Vehicle and Pedestrian are in Falcon\'s subset')
        props = {p.get('name'): p.get('value') for p in e.findall('Properties/Property')}
        cat = e.get('vehicleCategory', 'pedestrian')
        inv = {v: k for k, v in CATEGORY.items()}
        kinds[so.get('name')] = props.get('falcon:kind') or inv.get(cat, 'pedestrian' if e.tag == 'Pedestrian' else 'car')
        lanes[so.get('name')] = props.get('falcon:lane')
    scn = {'name': (hdr.get('description', '') if hdr is not None else '').split(':')[0] or 'imported',
           'about': 'imported from OpenSCENARIO', 'needs': [], 'actors': [], 'timeline': [], 'values': values,
           'params': {}, 'ego': {'driver': 'follower', 'route': 'straight'}}
    inv_dl = {0: 'ego', 1: 'adjacent', -1: 'opposite'}
    for pr in root.findall('Storyboard/Init/Actions/Private'):
        ref = pr.get('entityRef')
        speed = None
        pos = world = None
        for pa in pr.findall('PrivateAction'):
            ts = pa.find('LongitudinalAction/SpeedAction/SpeedActionTarget/AbsoluteTargetSpeed')
            if ts is not None:
                speed = float(ts.get('value'))
            rl = pa.find('TeleportAction/Position/RelativeLanePosition')
            wl = pa.find('TeleportAction/Position/WorldPosition')
            route = pa.find('RoutingAction/AssignRouteAction/Route')
            if rl is not None:
                pos = rl
            elif wl is not None:
                world = wl
            elif pa.find('TeleportAction') is not None:
                raise ValueError(f'{ref}: only RelativeLanePosition and WorldPosition teleports are in Falcon\'s '
                                 f'subset')
            if route is not None and ref == 'hero':
                scn['route_xy'] = [list(_carla(w)[:2]) for w in route.findall('Waypoint/Position/WorldPosition')]
        comment = ' '.join(c.text for c in pr.iter() if c.tag is ET.Comment)
        heading = _field(comment, 'heading', 'lane')
        start = _field(comment, 'start', 'moving')
        if ref == 'hero':
            scn['ego']['speed_kmh'] = 3.6 * (speed or 0.0)
            scn['ego']['driver'] = _field(comment, 'driver', 'follower')
            if world is not None:
                x, y, yaw = _carla(world)
                scn['start'] = {'x': x, 'y': y, 'yaw': yaw}
            continue
        if pos is not None:
            lane = lanes.get(ref) or inv_dl.get(int(float(pos.get('dLane', 0))), 'ego')
            place = {'lane': lane, 'ahead_m': float(pos.get('ds', 0)), 'lateral_m': float(pos.get('offset', 0))}
        else:
            parts = comment.split()
            k = parts.index('falcon:place') if 'falcon:place' in parts else -1
            place = ({'lane': parts[k + 1], 'ahead_m': float(parts[k + 2]), 'lateral_m': float(parts[k + 3])}
                     if k >= 0 else {'lane': lanes.get(ref) or 'ego', 'ahead_m': 0.0, 'lateral_m': 0.0})
        scn['actors'].append({'id': ref, 'kind': kinds.get(ref, 'car'), 'heading': heading, 'start': start,
                              'speed_kmh': 3.6 * (speed or 0.0), 'place': place})
    by_id = {a['id']: a for a in scn['actors']}
    for ev in root.iter('Event'):
        actions = []
        tracks = 0
        for act in ev.findall('Action'):
            comment = ' '.join(c.text for c in act.iter() if c.tag is ET.Comment)
            actor = _field(comment, 'actor', None)
            ft = act.find('PrivateAction/RoutingAction/FollowTrajectoryAction')
            if ft is not None:
                vs = ft.findall('.//Polyline/Vertex')
                pts = [(float(v.get('time')),) + _carla(v.find('Position/WorldPosition')) for v in vs]
                tr = {'t': [p[0] for p in pts], 'x': [p[1] for p in pts], 'y': [p[2] for p in pts],
                      'yaw': [p[3] for p in pts]}
                if actor == 'hero':
                    scn['ego']['track'] = tr
                elif actor in by_id:
                    by_id[actor]['track'] = tr
                else:
                    raise ValueError(f'event {ev.get("name")}: a trajectory for an unknown entity {actor!r}')
                tracks += 1
                continue
            sa = act.find('PrivateAction/LongitudinalAction/SpeedAction')
            lc = act.find('PrivateAction/LateralAction/LaneChangeAction')
            cc = act.find('UserDefinedAction/CustomCommandAction')
            if sa is not None:
                v = float(sa.find('SpeedActionTarget/AbsoluteTargetSpeed').get('value'))
                rate = float(sa.find('SpeedActionDynamics').get('value'))
                actions.append({'actor': actor, 'do': 'brake', 'decel': rate} if v == 0 else
                               {'actor': actor, 'do': 'speed', 'kmh': v * 3.6, 'accel': rate})
            elif lc is not None:
                dur = float(lc.find('LaneChangeActionDynamics').get('value'))
                actions.append({'actor': actor, 'do': 'lane_change', 'duration_s': dur,
                                'to': _field(' '.join(c.text for c in lc.iter() if c.tag is ET.Comment), 'to', 'ego')})
            elif cc is not None and (cc.get('type') or '').startswith('falcon:'):
                a = {'actor': actor, 'do': cc.get('type')[7:]}
                for kv in (cc.text or '').split():
                    k, _, v = kv.partition('=')
                    try:
                        a[k] = float(v)
                    except ValueError:
                        a[k] = v
                actions.append(a)
            else:
                raise ValueError(f'event {ev.get("name")}: an action outside Falcon\'s OpenSCENARIO subset')
        if tracks and not actions:
            continue
        cond = ev.find('StartTrigger/ConditionGroup/Condition')
        stc = cond.find('ByValueCondition/SimulationTimeCondition')
        rdc = cond.find('ByEntityCondition/EntityCondition/RelativeDistanceCondition')
        if stc is not None:
            trig = {'type': 'time', 't_s': float(stc.get('value'))}
        elif rdc is not None:
            if rdc.get('rule') == 'greaterThan':
                trig = {'type': 'ahead', 'actor': rdc.get('entityRef'), 'gt_m': float(rdc.get('value'))}
            else:
                trig = {'type': 'distance', 'actor': rdc.get('entityRef'), 'lt_m': float(rdc.get('value'))}
        else:
            raise ValueError(f'event {ev.get("name")}: a trigger outside Falcon\'s OpenSCENARIO subset')
        scn['timeline'].append({'trigger': trig, 'actions': actions})
    stop = root.find('Storyboard/StopTrigger/ConditionGroup/Condition/ByValueCondition/SimulationTimeCondition')
    scn['duration_s'] = float(stop.get('value')) if stop is not None else 12.0
    return scn


def _field(text, key, default):
    parts = (text or '').split()
    for k in range(len(parts) - 1):
        if parts[k] == f'falcon:{key}' or (parts[k] == key and k > 0):
            return parts[k + 1]
    return default
