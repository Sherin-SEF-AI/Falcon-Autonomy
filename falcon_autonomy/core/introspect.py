"""Introspection of driving models (numpy at import; torch inside functions): what a token policy (core.tokenpolicy)
or a vision-language model (core.vlm_models, Qwen2-VL with eager attention) attends to, where that is in the world,
and what changes when a token, a head or an activation is intervened on. core.biasaudit runs the closed-loop audits;
docs/introspection.md is the guide.

    falcon introspect attention --policy DIR (--frames FRAMES.npz | PROJECT --scenario K [--seed S] [--steps N])
                                --out DIR
    falcon introspect attention --vlm hf:FOLDER --image IMG --prompt TEXT --out DIR [--depth DEPTH.npy]
    falcon introspect intervene --policy DIR --frames FRAMES.npz --op JSON [--op JSON]... --out DIR
    falcon introspect audit PROJECT --policy DIR|follower --out DIR [--arms ...] [--seeds N]   (core.biasaudit)

Hooks. Every attention module is hooked with a forward hook that keeps its weights (B, heads, queries, keys): the
token policy's four encoder layers and its plan query (core.tokenpolicy.Attention returns its weights), and the
language model's self-attention layers of a Hugging Face VLM loaded with attn_implementation='eager' (only eager
attention materialises the weights: SDPA and flash attention never form them). The VLM's hook points are the
modules that hold a LoRA adapter's wrapped projections (core.vlm_lora.lora_modules — their parents, the attention
modules) when an adapter is loaded, else every attention module with q_proj and o_proj outside the vision tower.
A head is zeroed by a forward pre-hook on the attention's output projection (o / o_proj) that zeroes that head's
slice of its input — the same mechanism on both models.

Attention rollout (Abnar & Zuidema 2020, "Quantifying Attention Flow in Transformers", ACL): per layer the
head-averaged weights with the residual added, Â = ½ (A + I), rows renormalised, multiplied through the layers; the
token policy's relevance of each token is its plan query's (head-averaged) attention times the rollout. Attention is
not explanation (Jain & Wallace 2019; Wiegreffe & Pinter 2019): the interventions below are the causal test.

In the world. A patch token's cell centre (u, v) is a ray through core.lens (the camera's intrinsics, unproject), its
point at the planar depth of the aligned depth camera (the median over the cell; the environment's camera depth
option), then the camera pose (core.lens.optical_to_vehicle) and the ego's pose: world points coloured by relevance,
which ui/view3d.py draws; patches whose depth is beyond FAR_M (sky) are dropped. An object token is its road user's
box (position, heading and size from the token itself).

Interventions (intervene; ops are dicts, applied to every frame of a recorded sequence, actions compared with the
unchanged ones — steering and acceleration deltas per frame):
    {"op": "mask_object", "id": N}            hide one road user's token from every query
    {"op": "mask_class", "class": "autorickshaw"}   hide every token of a class (core.tokenfeatures.OBJ_CLASSES)
    {"op": "mask_tokens", "type": "patch"|"object"|"route"|"ego", "index": [k, …]}   (all of the type without index)
    {"op": "zero_head", "layer": L | "plan", "head": h}
    {"op": "patch", "layer": L, "tokens": "object"|"patch"|"all"|[k, …], "frames": FRAMES.npz | "shift": n}
        activation patching (Vig et al. 2020; Meng et al. 2022): the residual stream after encoder layer L at those
        tokens is replaced by its value on the counterfactual observation (the same frame of another recording, or
        the frame n steps away)
    {"op": "flip", "feature": "brake_light", "id": N | "class": C | "all": true}    a binary object feature flipped
    {"op": "set", "feature": F, "id": N, "value": V}           an object feature set (core.tokenfeatures.OBJ_FEATURES)
    {"op": "speed_history", "offset": m/s}                     the ego's speed history shifted (the scene unchanged)

Frames files (record; FRAMES.npz): every array of the observation stacked over the steps (state, camera, depth,
objects, object_ids, ego_history), the ego's pose per step, the camera spec, the policy's actions, the episode's
seed and options — so attention and interventions are reproducible offline.

Limits: a patch's position is its cell's centre (a ResNet cell sees more than its cell); rollout assumes residual
mixing of ½; the VLM path reads the language model's attention to the image tokens of the last prompt position
(merged 2 × 2 patches in Qwen2-VL); patching replaces whole token vectors at one layer.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from . import tokenfeatures as TF

FAR_M = 80.0
SCHEMA = 'falcon.introspect/1'


# ============================================================================================ hooks
def token_hook_points(net) -> list:
    """[(name, attention module)] of a token policy: layer0 … layerN−1 and plan."""
    return [(f'layer{i}', b.attn) for i, b in enumerate(net.blocks)] + [('plan', net.plan)]


def vlm_hook_points(model) -> list:
    """[(name, attention module)] of a VLM (module doc): the parents of a LoRA adapter's wrapped q projections when
    it has one (core.vlm_lora.lora_modules), else the attention modules with q_proj and o_proj outside the vision
    tower."""
    from .vlm_lora import lora_modules
    mods = dict(model.named_modules())
    lora = lora_modules(model)
    if lora:
        names = sorted({n.rpartition('.')[0] for n in lora if n.endswith('q_proj')}, key=_layer_key)
        return [(n, mods[n]) for n in names if n in mods]
    out = [(n, m) for n, m in mods.items() if hasattr(m, 'q_proj') and hasattr(m, 'o_proj') and
           'vision' not in n.lower() and 'visual' not in n.lower()]
    return sorted(out, key=lambda t: _layer_key(t[0]))


def _layer_key(name: str):
    nums = [int(x) for x in name.split('.') if x.isdigit()]
    return (nums, name)


class AttentionRecorder:
    """Forward hooks that keep each hooked module's attention weights (module doc): with AttentionRecorder(points)
    as rec: model(...); rec.weights[name] → (B, heads, Tq, Tk) tensor (detached)."""

    def __init__(self, points: list):
        self.points = points
        self.weights: dict = {}
        self.handles = []

    def __enter__(self):
        for name, mod in self.points:
            self.handles.append(mod.register_forward_hook(self._hook(name)))
        return self

    def _hook(self, name):
        def fn(_mod, _inp, out):
            w = out[1] if isinstance(out, tuple) and len(out) > 1 else None
            if w is not None:
                self.weights[name] = w.detach()
        return fn

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


def zero_head_hook(attn, head: int, n_heads: int):
    """A pre-hook on the attention's output projection (o or o_proj) zeroing one head's slice; returns its handle."""
    proj = getattr(attn, 'o', None) or getattr(attn, 'o_proj')
    if not 0 <= int(head) < int(n_heads):
        raise ValueError(f'head {head} is not one of 0…{n_heads - 1}')

    def pre(_mod, args):
        x = args[0].clone()
        d = x.shape[-1] // n_heads
        x[..., int(head) * d:(int(head) + 1) * d] = 0.0
        return (x,) + tuple(args[1:])
    return proj.register_forward_pre_hook(pre)


# ============================================================================================ rollout
def rollout(weights: list, residual: float = 0.5):
    """Attention rollout over a list of (B, h, T, T) weights (module doc): (B, T, T) numpy."""
    out = None
    for w in weights:
        a = np.asarray(w.float().cpu().numpy() if hasattr(w, 'cpu') else w, np.float64).mean(1)
        T = a.shape[-1]
        a = residual * a + (1.0 - residual) * np.eye(T)[None]
        a = a / a.sum(-1, keepdims=True)
        out = a if out is None else a @ out
    return out


# ============================================================================================ token policy
def _batch(pol, obs_list: list, names, edits=None):
    from .tokenpolicy import inputs_np, to_torch
    image = pol.doc['net'].get('image', True)
    arr = []
    for obs in obs_list:
        d = inputs_np(obs, pol.norm, pol.features, names, image)
        if edits:
            d = edits(d, obs)
        arr.append(d)
    return to_torch(arr, pol.device)


def token_layout(pol, obs) -> dict:
    """The token layout of a policy on an observation (core.tokenpolicy.TokenNet.layout after a forward)."""
    import torch
    from .drivefeatures import FEATURE_NAMES
    inp = _batch(pol, [obs], FEATURE_NAMES)
    with torch.no_grad():
        pol.net(inp)
    return pol.net.layout(int(np.asarray(obs['objects']).shape[0]))


def describe_tokens(layout: dict, obs) -> list:
    """One row per token: {'index', 'type', and patch row/col, or object id/class/brake light}."""
    rows = []
    gh, gw = layout['grid']
    p0, p1 = layout['patch']
    for k in range(p0, p1):
        r, c = divmod(k - p0, max(gw, 1))
        rows.append({'index': k, 'type': 'patch', 'row': r, 'col': c})
    o0, o1 = layout['object']
    objs, ids = np.asarray(obs['objects']), np.asarray(obs['object_ids'])
    for k in range(o0, o1):
        f = objs[k - o0]
        if f[TF.OI['present']] < 0.5:
            rows.append({'index': k, 'type': 'object', 'id': None})
            continue
        cls = TF.OBJ_CLASSES[int(np.argmax(f[1:1 + len(TF.OBJ_CLASSES)]))]
        rows.append({'index': k, 'type': 'object', 'id': int(ids[k - o0]), 'class': cls,
                     'dx_m': round(float(f[TF.OI['dx_m']]), 2), 'dy_m': round(float(f[TF.OI['dy_m']]), 2),
                     'brake_light': bool(f[TF.OI['brake_light']] > 0.5)})
    rows.append({'index': layout['route'], 'type': 'route'})
    rows.append({'index': layout['ego'], 'type': 'ego'})
    return rows


def patch_points(grid, camera: dict, depth, ego: dict | None = None, relevance=None) -> list:
    """World (or vehicle-frame, ego None) points of the patch cells (module doc): [{'row', 'col', 'u', 'v', 'depth_m',
    'x', 'y', 'z', 'weight'}] for the cells whose depth is within FAR_M."""
    from . import lens
    gh, gw = grid
    H, W = int(camera['height']), int(camera['width'])
    intr = lens.intrinsics({'type': 'camera.pinhole', 'width': W, 'height': H, 'fov': float(camera.get('fov', 90.0))})
    pose = dict({'x': 1.5, 'y': 0.0, 'z': 1.6, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}, **(camera.get('pose') or {}))
    Rv = lens.optical_to_vehicle(pose)
    depth = np.asarray(depth, float)
    out = []
    for r in range(gh):
        for c in range(gw):
            v0, v1 = int(r * H / gh), int((r + 1) * H / gh)
            u0, u1 = int(c * W / gw), int((c + 1) * W / gw)
            cell = depth[v0:max(v1, v0 + 1), u0:max(u1, u0 + 1)]
            z = float(np.median(cell)) if cell.size else math.inf
            if not math.isfinite(z) or z > FAR_M or z <= 0:
                continue
            u, v = (c + 0.5) * W / gw, (r + 0.5) * H / gh
            d, _th = intr.unproject(u, v)
            d = np.asarray(d, float).reshape(3)
            p_opt = d / max(d[2], 1e-9) * z                     # planar depth: the optical z is the depth
            p = Rv @ p_opt + np.asarray([pose['x'], pose['y'], pose['z']])
            if ego is not None:
                yaw = math.radians(float(ego.get('yaw', 0.0)))
                cy, sy = math.cos(yaw), math.sin(yaw)
                p = np.asarray([ego['x'] + cy * p[0] - sy * p[1], ego['y'] + sy * p[0] + cy * p[1],
                                float(ego.get('z', 0.0)) + p[2]])
            w = float(relevance[r * gw + c]) if relevance is not None else None
            out.append({'row': r, 'col': c, 'u': round(u, 2), 'v': round(v, 2), 'depth_m': round(z, 3),
                        'x': round(float(p[0]), 3), 'y': round(float(p[1]), 3), 'z': round(float(p[2]), 3),
                        'weight': w})
    return out


def object_boxes(obs, relevance=None, ego: dict | None = None) -> list:
    """The object tokens as boxes (vehicle frame; world with ego) with their relevance."""
    objs, ids = np.asarray(obs['objects']), np.asarray(obs['object_ids'])
    out = []
    for k, f in enumerate(objs):
        if f[TF.OI['present']] < 0.5:
            continue
        x, y = float(f[TF.OI['dx_m']]), float(f[TF.OI['dy_m']])
        yaw = math.degrees(math.atan2(float(f[TF.OI['sin_rel_yaw']]), float(f[TF.OI['cos_rel_yaw']])))
        if ego is not None:
            r = math.radians(float(ego.get('yaw', 0.0)))
            x, y = ego['x'] + math.cos(r) * x - math.sin(r) * y, ego['y'] + math.sin(r) * x + math.cos(r) * y
            yaw += float(ego.get('yaw', 0.0))
        out.append({'id': int(ids[k]), 'class': TF.OBJ_CLASSES[int(np.argmax(f[1:1 + len(TF.OBJ_CLASSES)]))],
                    'x': round(x, 3), 'y': round(y, 3), 'yaw': round(yaw, 2),
                    'length': round(float(f[TF.OI['length_m']]), 3), 'width': round(float(f[TF.OI['width_m']]), 3),
                    'brake_light': bool(f[TF.OI['brake_light']] > 0.5),
                    'weight': float(relevance[k]) if relevance is not None else None})
    return out


def attention(pol, obs: dict, names=None, ego: dict | None = None) -> dict:
    """Attention of a token policy on one observation (module doc): per-layer head-averaged weights of the plan
    query, the rollout relevance of every token, its token table, patch points and object boxes in the vehicle frame
    (patch points when the observation has depth) — and in the world (points_world, objects_world) when the ego's
    pose is given — and the action."""
    import torch
    from .drivefeatures import FEATURE_NAMES
    names = names or FEATURE_NAMES
    net = pol.net
    inp = _batch(pol, [obs], names)
    with AttentionRecorder(token_hook_points(net)) as rec, torch.no_grad():
        y = net(inp).cpu().numpy()[0]
    lay = net.layout(int(np.asarray(obs['objects']).shape[0]))
    layers = [rec.weights[f'layer{i}'] for i in range(len(net.blocks))]
    R = rollout(layers)[0]
    plan = rec.weights['plan'][0].float().cpu().numpy()            # (h, 1, T)
    rel = plan.mean(0)[0] @ R
    rel = rel / max(rel.sum(), 1e-12)
    rows = describe_tokens(lay, obs)
    for r in rows:
        r['relevance'] = round(float(rel[r['index']]), 6)
        r['plan_attention'] = round(float(plan.mean(0)[0][r['index']]), 6)
    out = {'action': [round(float(v), 6) for v in y], 'tokens': rows, 'layout': {k: v for k, v in lay.items()},
           'plan_heads': plan[:, 0, :].round(6).tolist(),
           'layers': {k: v[0].mean(0).float().cpu().numpy().round(5).tolist() for k, v in rec.weights.items()
                      if k != 'plan'},
           'by_type': {t: round(float(sum(r['relevance'] for r in rows if r['type'] == t)), 6)
                       for t in ('patch', 'object', 'route', 'ego')}}
    o0, o1 = lay['object']
    p0, p1 = lay['patch']
    out['objects'] = object_boxes(obs, rel[o0:o1])                        # the vehicle frame (ui/view3d.py) …
    if p1 > p0 and obs.get('depth') is not None and pol.doc.get('camera'):
        out['points'] = patch_points(lay['grid'], pol.doc['camera'], obs['depth'], None, rel[p0:p1])
    if ego is not None:                                                    # … and the world, with the ego's pose
        out['ego'] = {k: ego.get(k) for k in ('x', 'y', 'z', 'yaw')}
        out['objects_world'] = object_boxes(obs, rel[o0:o1], ego)
        if 'points' in out:
            out['points_world'] = patch_points(lay['grid'], pol.doc['camera'], obs['depth'], ego, rel[p0:p1])
    return out


# ============================================================================================ interventions
OPS = ('mask_object', 'mask_class', 'mask_tokens', 'zero_head', 'patch', 'flip', 'set', 'speed_history')


def check_ops(ops: list):
    for o in ops:
        if not isinstance(o, dict) or o.get('op') not in OPS:
            raise ValueError(f'an intervention is {{"op": one of {", ".join(OPS)}, …}}, not {o!r}')
        if o['op'] == 'mask_class' and o.get('class') not in TF.OBJ_CLASSES:
            raise ValueError(f'mask_class: class must be one of {", ".join(TF.OBJ_CLASSES)}')
        if o['op'] in ('flip', 'set') and o.get('feature') not in TF.OBJ_FEATURE_NAMES:
            raise ValueError(f'{o["op"]}: feature must be one of {", ".join(TF.OBJ_FEATURE_NAMES)}')
        if o['op'] == 'mask_tokens' and o.get('type') not in ('patch', 'object', 'route', 'ego'):
            raise ValueError('mask_tokens: type must be patch, object, route or ego')


def _select(objs, ids, o) -> np.ndarray:
    """The object rows an op names (id, class or all)."""
    present = objs[:, TF.OI['present']] > 0.5
    if o.get('all'):
        return present
    if 'id' in o:
        return present & (ids == int(o['id']))
    if 'class' in o:
        return present & (objs[:, TF.OI[f'class_{o["class"]}']] > 0.5)
    raise ValueError(f'{o["op"]}: name the object by id, class or all')


def edit_observation(obs: dict, ops: list) -> dict:
    """The observation with the input-side ops applied (flip, set, speed_history); the rest is untouched."""
    o2 = dict(obs)
    objs = np.array(obs['objects'], np.float32, copy=True)
    ids = np.asarray(obs['object_ids'])
    for o in ops:
        if o['op'] == 'flip':
            m = _select(objs, ids, o)
            objs[m, TF.OI[o['feature']]] = 1.0 - (objs[m, TF.OI[o['feature']]] > 0.5)
        elif o['op'] == 'set':
            m = _select(objs, ids, o)
            objs[m, TF.OI[o['feature']]] = float(o['value'])
        elif o['op'] == 'speed_history':
            o2['ego_history'] = np.maximum(np.asarray(obs['ego_history'], np.float32) + float(o['offset']), 0.0)
    o2['objects'] = objs
    return o2


def token_mask(lay: dict, obs: dict, ops: list) -> np.ndarray | None:
    """(T,) bool of the tokens the mask ops hide, None when none does."""
    m = np.zeros(lay['n'], bool)
    objs, ids = np.asarray(obs['objects']), np.asarray(obs['object_ids'])
    o0, o1 = lay['object']
    any_ = False
    for o in ops:
        if o['op'] in ('mask_object', 'mask_class'):
            sel = _select(objs, ids, o)
            m[o0:o1] |= sel
            any_ = True
        elif o['op'] == 'mask_tokens':
            lo, hi = {'patch': lay['patch'], 'object': lay['object'], 'route': (lay['route'], lay['route'] + 1),
                      'ego': (lay['ego'], lay['ego'] + 1)}[o['type']]
            idx = range(lo, hi) if o.get('index') is None else [lo + int(k) for k in o['index'] if lo + int(k) < hi]
            m[list(idx)] = True
            any_ = True
    return m if any_ else None


def act_with(pol, obs: dict, ops: list, names=None, counterfactual: dict | None = None):
    """The token policy's action on one observation under ops (module doc) — (action, the number of tokens hidden)."""
    import torch
    from .drivefeatures import FEATURE_NAMES
    names = names or FEATURE_NAMES
    net = pol.net
    lay = token_layout(pol, obs)
    obs2 = edit_observation(obs, ops)
    m = token_mask(lay, obs, ops)
    hidden = int(m.sum()) if m is not None else 0

    def edits(d, _o):
        if m is not None:
            d = dict(d, token_mask=m.copy())
        return d
    handles = []
    try:
        for o in ops:
            if o['op'] == 'zero_head':
                pts = dict(token_hook_points(net))
                key = 'plan' if o.get('layer') == 'plan' else f'layer{int(o["layer"])}'
                if key not in pts:
                    raise ValueError(f'zero_head: no layer {o.get("layer")!r} (layers 0…{len(net.blocks) - 1}, plan)')
                handles.append(zero_head_hook(pts[key], int(o['head']), net.spec['heads']))
            elif o['op'] == 'patch':
                if counterfactual is None:
                    raise ValueError('patch needs a counterfactual observation (frames or shift)')
                L = int(o['layer'])
                if not 0 <= L < len(net.blocks):
                    raise ValueError(f'patch: layer {L} is not one of 0…{len(net.blocks) - 1}')
                store = {}
                h = net.blocks[L].register_forward_hook(lambda _m, _i, out: store.__setitem__('x', out.detach()))
                with torch.no_grad():
                    net(_batch(pol, [counterfactual], names))
                h.remove()
                tok = o.get('tokens', 'all')
                if isinstance(tok, str):
                    rng_ = {'all': (0, lay['n']), 'object': lay['object'], 'patch': lay['patch']}[tok]
                    idx = list(range(*rng_))
                else:
                    idx = [int(k) for k in tok]
                src = store['x']

                def repl(_m, _i, out, src=src, idx=idx):
                    out = out.clone()
                    out[:, idx] = src[:, idx]
                    return out
                handles.append(net.blocks[L].register_forward_hook(repl))
        with torch.no_grad():
            y = net(_batch(pol, [obs2], names, edits)).cpu().numpy()[0]
    finally:
        for h in handles:
            h.remove()
    return np.clip(y, -1.0, 1.0), hidden


def intervene(pol, frames: list, ops: list, names=None, counterfactuals: list | None = None, shift: int = 0) -> dict:
    """Steering and acceleration deltas of ops over a sequence of observations (module doc): per frame the action
    without and with them, and a summary (mean and max |Δ|, the mean signed Δ)."""
    check_ops(ops)
    rows = []
    for k, obs in enumerate(frames):
        cf = None
        if any(o['op'] == 'patch' for o in ops):
            if counterfactuals is not None:
                cf = counterfactuals[min(k, len(counterfactuals) - 1)]
            else:
                cf = frames[min(max(k + int(shift), 0), len(frames) - 1)]
        base, _ = act_with(pol, obs, [], names)
        new, hidden = act_with(pol, obs, ops, names, cf)
        rows.append({'frame': k, 'base': [round(float(v), 6) for v in base], 'intervened': [round(float(v), 6)
                                                                                            for v in new],
                     'delta': [round(float(new[0] - base[0]), 6), round(float(new[1] - base[1]), 6)],
                     'tokens_hidden': hidden})
    D = np.asarray([r['delta'] for r in rows], float).reshape(-1, 2)
    summ = {'frames': len(rows)}
    for i, key in enumerate(('steer', 'accel')):
        summ[key] = {'mean_delta': float(D[:, i].mean()) if len(D) else None,
                     'mean_abs_delta': float(np.abs(D[:, i]).mean()) if len(D) else None,
                     'max_abs_delta': float(np.abs(D[:, i]).max()) if len(D) else None}
    return {'schema': SCHEMA + '#intervene', 'ops': ops, 'frames': rows, 'summary': summ}


# ============================================================================================ VLM
def vlm_attention(model, enc: dict, image_token_id: int, grid_thw=None, merge: int = 2, layers=None) -> dict:
    """Attention of a VLM's language model to the image tokens, from the last prompt position (module doc): per hooked
    layer the head-averaged weights over the image tokens, their rollout over the whole sequence, and the rollout's
    image-token relevance as a (h, w) heat map when grid_thw (Qwen2-VL's image_grid_thw: t, h, w in patches) is
    given (merged merge × merge). Needs eager attention."""
    import torch
    impl = getattr(getattr(model, 'config', None), '_attn_implementation', None)
    text_impl = getattr(getattr(getattr(model, 'config', None), 'text_config', None), '_attn_implementation', impl)
    if 'eager' not in (impl, text_impl):
        raise ValueError(f'attention weights need attn_implementation=eager (the model uses {impl!r}): load it with '
                         'hf:…,attn_implementation=eager')
    pts = vlm_hook_points(model)
    if layers is not None:
        pts = [p for p in pts if p[0] in set(layers)]
    if not pts:
        raise ValueError('no attention module to hook in this model')
    ids = enc['input_ids'][0]
    img = (ids == int(image_token_id)).nonzero().flatten().cpu().numpy()
    with AttentionRecorder(pts) as rec, torch.no_grad():
        out = model(**enc)
    W = [rec.weights[n] for n, _ in pts if n in rec.weights]
    if not W:
        raise ValueError('the hooked modules returned no attention weights (eager attention is needed)')
    R = rollout(W)[0]
    last = R[-1]
    per_layer = {n: rec.weights[n][0, :, -1, :].float().mean(0).cpu().numpy()[img].round(6).tolist()
                 for n, _ in pts if n in rec.weights}
    rel = last[img] / max(last[img].sum(), 1e-12) if len(img) else np.zeros(0)
    doc = {'layers': [n for n, _ in pts], 'image_tokens': int(len(img)), 'image_relevance': rel.round(6).tolist(),
           'image_share': float(last[img].sum()) if len(img) else 0.0, 'per_layer_last': per_layer,
           'logits_last': out.logits[0, -1].float().cpu().numpy()}
    if grid_thw is not None and len(img):
        t, h, w = (int(v) for v in grid_thw)
        hh, ww = h // merge, w // merge
        if t * hh * ww == len(img):
            doc['heatmap'] = rel.reshape(t, hh, ww).mean(0).round(6).tolist()
    return doc


def vlm_zero_head(model, enc: dict, layer: str, head: int) -> dict:
    """The change of the next-token distribution when one head of a hooked layer is zeroed: KL(base ‖ zeroed) and
    the top token before and after."""
    import torch
    pts = dict(vlm_hook_points(model))
    if layer not in pts:
        raise ValueError(f'no hooked layer {layer!r}')
    att = pts[layer]
    n_heads = int(getattr(att, 'num_heads', 0) or getattr(getattr(att, 'config', None), 'num_attention_heads', 0))
    with torch.no_grad():
        base = model(**enc).logits[0, -1].float()
        h = zero_head_hook(att, head, n_heads)
        try:
            new = model(**enc).logits[0, -1].float()
        finally:
            h.remove()
    p, q = torch.log_softmax(base, -1), torch.log_softmax(new, -1)
    return {'layer': layer, 'head': int(head), 'kl': float((p.exp() * (p - q)).sum()),
            'top_before': int(base.argmax()), 'top_after': int(new.argmax())}


# ============================================================================================ frames
FRAME_KEYS = ('state', 'camera', 'depth', 'objects', 'object_ids', 'ego_history')


def record(pol, env, seed: int, options: dict | None = None, steps: int = 0, out_path: str = '') -> dict:
    """Drive one episode with the policy and keep every observation (module doc); with out_path, a frames file."""
    obs, info = env.reset(seed=int(seed), options=options)
    frames, egos, actions = [obs], [info.get('ego')], []
    done, k = False, 0
    while not done and (not steps or k < steps):
        a = pol.act(obs, env.feature_names)
        obs, _r, term, trunc, info = env.step(a)
        actions.append([float(v) for v in np.asarray(a).reshape(-1)])
        frames.append(obs)
        egos.append(info.get('ego'))
        done = term or trunc
        k += 1
    doc = {'frames': frames, 'egos': egos, 'actions': actions, 'seed': int(seed), 'options': options or {},
           'camera': (env.cfg.get('camera') or None), 'features': list(env.feature_names)}
    if out_path:
        save_frames(out_path, doc)
    return doc


def save_frames(path: str, doc: dict):
    arrays = {}
    for k in FRAME_KEYS:
        if all(k in f and f[k] is not None for f in doc['frames']):
            arrays[k] = np.stack([np.asarray(f[k]) for f in doc['frames']])
    meta = {kk: doc.get(kk) for kk in ('egos', 'actions', 'seed', 'options', 'camera', 'features')}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, meta=np.asarray(json.dumps(meta, default=float)), **arrays)


def load_frames(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['meta']))
        keys = [k for k in FRAME_KEYS if k in z.files]
        n = len(z[keys[0]]) if keys else 0
        frames = [{k: z[k][i] for k in keys} for i in range(n)]
    meta['frames'] = frames
    return meta


# ============================================================================================ the CLI
def _write(path, doc):
    from .provenance import write_json
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_json(path, doc)


def main_introspect(argv) -> int:
    """`falcon introspect attention|intervene|audit` (module doc). 0 ok, 2 failed or a usage error."""
    import argparse
    from . import protocol as P
    usage = ('usage: falcon introspect attention --policy DIR (--frames FRAMES.npz | PROJECT --scenario K) --out DIR\n'
             '       falcon introspect attention --vlm hf:FOLDER --image IMG --prompt TEXT --out DIR\n'
             '       falcon introspect intervene --policy DIR --frames FRAMES.npz --op JSON... --out DIR\n'
             '       falcon introspect audit PROJECT --policy DIR|follower --out DIR [--arms A,B] [--seeds N]')
    if not argv or argv[0] not in ('attention', 'intervene', 'audit'):
        print(usage, flush=True)
        return P.FAILED if argv and argv[0] not in ('-h', '--help') else 0
    if argv[0] == 'audit':
        from .biasaudit import main_audit
        return main_audit(argv[1:])
    what = argv[0]
    ap = argparse.ArgumentParser(prog=f'falcon introspect {what}')
    ap.add_argument('project', nargs='?', default='', help='a project (to record an episode)')
    ap.add_argument('--policy', default='', help='a token policy folder (core.tokenpolicy)')
    ap.add_argument('--frames', default='', help='a frames file (falcon introspect attention writes frames.npz)')
    ap.add_argument('--out', '-o', required=True)
    ap.add_argument('--device', default='cpu')
    if what == 'attention':
        ap.add_argument('--scenario', default='lead_brake', help='the scenario to record (with PROJECT)')
        ap.add_argument('--seed', type=int, default=0)
        ap.add_argument('--steps', type=int, default=0, help='record at most this many steps (0: the episode)')
        ap.add_argument('--every', type=int, default=5, help='attention every n-th frame')
        ap.add_argument('--world', default='')
        ap.add_argument('--fake-carla', default='')
        ap.add_argument('--vlm', default='', help='a VLM runner spec (hf:FOLDER or a candidate): its attention')
        ap.add_argument('--image', action='append', default=[], help='--vlm: an image (repeat for several)')
        ap.add_argument('--prompt', default='Describe the driving scene.')
        ap.add_argument('--depth', default='', help='--vlm: a .npy planar depth image aligned with the image')
    else:
        ap.add_argument('--op', action='append', required=True, help='an intervention as JSON (module doc)')
        ap.add_argument('--counterfactual', default='', help='a frames file of the counterfactual (op patch)')
        ap.add_argument('--shift', type=int, default=0, help='op patch: the counterfactual is n frames away')
    try:
        a = ap.parse_args(argv[1:])
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    try:
        if what == 'attention' and a.vlm:
            return _vlm_cli(a)
        from .policy import load_policy
        if not a.policy:
            raise ValueError('--policy is needed (a token policy folder)')
        pol = load_policy(a.policy, a.device)
        if not getattr(pol, 'uses_tokens', False):
            raise ValueError(f'{a.policy} is not a token policy (net kind {pol.doc["net"]["kind"]}): its attention is '
                             'not defined; falcon introspect audit runs the simulation-side arms on any ego')
        if a.frames:
            rec = load_frames(a.frames)
        elif what == 'attention' and a.project:
            rec = _record_cli(a, pol)
        else:
            raise ValueError('give --frames FRAMES.npz, or PROJECT to record an episode')
        frames = rec['frames']
        if what == 'attention':
            rows = []
            for k in range(0, len(frames), max(int(a.every), 1)):
                d = attention(pol, frames[k], rec.get('features'), (rec.get('egos') or [None] * len(frames))[k])
                d.pop('layers', None)
                rows.append(dict(d, frame=k))
            doc = {'schema': SCHEMA, 'policy': pol.path, 'frames': rows, 'seed': rec.get('seed'),
                   'options': rec.get('options'), 'camera': rec.get('camera'),
                   'note': 'attention is not explanation: intervene measures what the policy relies on'}
            _write(os.path.join(a.out, 'attention.json'), doc)
            mean = {t: float(np.mean([r['by_type'][t] for r in rows])) for t in ('patch', 'object', 'route', 'ego')}
            print(json.dumps({'frames': len(rows), 'mean_relevance_by_type': mean,
                              'out': os.path.join(a.out, 'attention.json')}, indent=2), flush=True)
            return P.OK
        ops = [json.loads(o) for o in a.op]
        cf = load_frames(a.counterfactual)['frames'] if a.counterfactual else None
        doc = intervene(pol, frames, ops, rec.get('features'), cf, a.shift)
        doc['policy'] = pol.path
        _write(os.path.join(a.out, 'intervene.json'), doc)
        print(json.dumps(doc['summary'], indent=2), flush=True)
        return P.OK
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED


def _record_cli(a, pol) -> dict:
    from .gymenv import FalconDriveEnv, gym_config, make_service
    from .project import Project
    proj = Project.load(a.project)
    svc = make_service(fake_carla=a.fake_carla, workspace=proj.workspace)
    env = None
    try:
        cam = dict(pol.doc.get('camera') or {'width': 128, 'height': 72, 'fov': 90.0}, depth=True)
        cfg = gym_config(proj, a.world or proj.doc['worlds'][0]['id'], os.path.join(a.out, 'env'), svc.port,
                         svc.tm_port, seed=a.seed, scenario=a.scenario or None, camera=cam,
                         tokens=pol.doc.get('tokens') or True, workspace=proj.workspace)
        env = FalconDriveEnv(cfg, svc)
        return record(pol, env, a.seed, None, a.steps, os.path.join(a.out, 'frames.npz'))
    finally:
        if env is not None:
            env.close()
        svc.shutdown()


def _vlm_cli(a) -> int:
    from . import protocol as P
    from .vlm_models import make_runner
    spec = a.vlm if a.vlm.startswith('hf:') else 'hf:' + a.vlm
    if 'attn_implementation' not in spec:
        spec += ',attn_implementation=eager'                  # only eager attention materialises the weights
    if not a.image:
        raise ValueError('--vlm needs --image')
    runner = make_runner(spec)
    try:
        enc, _text = runner.inputs(a.image, a.prompt)
        enc = {k: (v.to(runner.device) if hasattr(v, 'to') else v) for k, v in enc.items()}
        tok = runner.model.config.image_token_id
        doc = vlm_attention(runner.model, enc, tok, enc.get('image_grid_thw')[0].tolist()
                            if enc.get('image_grid_thw') is not None else None)
        doc.pop('logits_last', None)
        doc.update(schema=SCHEMA + '#vlm', model=runner.name, prompt=a.prompt, images=a.image)
        if a.depth and doc.get('heatmap'):
            depth = np.load(a.depth)
            hm = np.asarray(doc['heatmap'])
            cam = {'width': depth.shape[1], 'height': depth.shape[0], 'fov': 90.0}
            doc['points'] = patch_points(hm.shape, cam, depth, None, hm.ravel())
        _write(os.path.join(a.out, 'vlm-attention.json'), doc)
        print(json.dumps({'image_tokens': doc['image_tokens'], 'image_share': doc['image_share'],
                          'layers': len(doc['layers'])}, indent=2), flush=True)
        return P.OK
    finally:
        runner.close()
