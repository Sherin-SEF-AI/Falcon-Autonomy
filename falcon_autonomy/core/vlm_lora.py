"""LoRA fine-tuning of a vision-language model on Falcon's grounded questions and answers (standard library and numpy at
import; torch inside the functions).

LoRA (Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models", arXiv:2106.09685): a frozen linear layer
W gets a trainable low-rank update, y = W x + (alpha / r) · B A x, with A (r × in) initialised as Kaiming-uniform
(a = √5, as the authors' loralib does) and B (out × r) as zeros, so training starts from the base model exactly.
Dropout on the update's input (p = dropout). Written here in plain torch: Falcon does not depend on peft.

    apply_lora(model, targets, r, alpha, dropout, scope) -> [names]   wrap the matching nn.Linear modules, freeze
                                                                      the rest
    merge_lora(model)                                                 fold B A into W (inference without the wrapper)
    save_adapter(model, folder, meta) / load_adapter(model, folder)   adapter.pt + adapter.json
    finetune(settings, out_dir, ...) -> summary                       train an adapter (needs the base weights)

Targets: by default the attention projections q_proj, k_proj, v_proj, o_proj of the LANGUAGE model (scope "text":
modules whose qualified name holds "vision" are left alone, so the image encoder stays as published); scope "all"
adapts the vision encoder too, and targets may add the MLP projections (gate_proj, up_proj, down_proj).

Fine-tuning (finetune). Examples are (the frame's images, a question, core.language's template answer) from language
folders (core.language): every n-th record, core.vlm.questions(). TRAIN AND VALIDATION ARE SPLIT BY DRIVE (frames of
one drive are not independent): the drives are put in a seeded order and the last ⌈val_share · n⌉ (at least one)
validate; fewer than two drives is refused. The loss is the language model's cross-entropy on the answer's tokens
only (the prompt's labels are -100). AdamW (weight decay 0) on the LoRA parameters, a linear warm-up then cosine,
gradient accumulation, clipping at a global norm of 1. After every epoch: the validation loss, and (val_generate > 0)
core.vlm's scores of the model's own greedy answers on that many validation questions; a checkpoint (adapter,
optimiser, epoch, the random states) from which a cancelled run resumes. The kept adapter is the epoch with the
lowest validation loss. Seeded (the order of an epoch is a function of seed and epoch; torch is re-seeded per step).

It runs only once a base model's weights are on disk (core.vlm_models: falcon vlm fetch NAME, asked for and
confirmed); nothing is downloaded here. An adapter is a derivative of its base model and carries the base model's
licence (adapter.json "licence"); the training data are Falcon's own simulated drives.
"""
from __future__ import annotations

import json
import math
import os
import time

DEFAULT_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
DEFAULTS = {'model': '', 'lang': [], 'workspace': '', 'r': 8, 'alpha': 16.0, 'dropout': 0.05,
            'targets': list(DEFAULT_TARGETS), 'scope': 'text', 'epochs': 2, 'lr': 2e-4, 'warmup_steps': 10,
            'accumulate': 8, 'clip_norm': 1.0, 'val_share': 0.25, 'every': 5, 'max_objects': 2, 'types': None,
            'clip': 1, 'clip_step': 5, 'camera': None, 'seed': 0, 'device': 'auto', 'max_side': 768,
            'val_generate': 32, 'max_new_tokens': 48, 'max_examples': 0}
VOLATILE = ('val_generate', 'max_new_tokens', 'device', 'workspace')


class LoRAError(Exception):
    pass


def _lora_linear_class():
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        """y = base(x) + (alpha / r) · B A dropout(x); base frozen (module doc)."""

        def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
            super().__init__()
            if r < 1:
                raise LoRAError('the LoRA rank r must be at least 1')
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.r, self.alpha = int(r), float(alpha)
            self.scale = self.alpha / self.r
            dev, dt = base.weight.device, base.weight.dtype
            self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features, device=dev, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r, device=dev, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self._dt = dt

        @property
        def weight(self):                    # code that reads .weight (dtype, device) sees the base layer's
            return self.base.weight

        @property
        def bias(self):
            return self.base.bias

        @property
        def in_features(self):
            return self.base.in_features

        @property
        def out_features(self):
            return self.base.out_features

        def forward(self, x):
            y = self.base(x)
            u = self.drop(x).to(self.lora_A.dtype) @ self.lora_A.t() @ self.lora_B.t()
            return y + (self.scale * u).to(y.dtype)

        def delta(self):
            return (self.lora_B @ self.lora_A) * self.scale

    return LoRALinear


_CLS = None


def lora_class():
    global _CLS
    if _CLS is None:
        _CLS = _lora_linear_class()
    return _CLS


def _matches(name: str, targets, scope: str) -> bool:
    if scope == 'text' and 'vision' in name.lower():
        return False
    return any(name == t or name.endswith('.' + t) for t in targets)


def apply_lora(model, targets=DEFAULT_TARGETS, r: int = 8, alpha: float = 16.0, dropout: float = 0.05,
               scope: str = 'text') -> list:
    """Wrap every nn.Linear whose qualified name ends with a target (module doc) and freeze every other parameter.
    Returns the wrapped names; raises when none matched."""
    from torch import nn
    if scope not in ('text', 'all'):
        raise LoRAError("scope is 'text' (the language model only) or 'all'")
    L = lora_class()
    for p in model.parameters():
        p.requires_grad_(False)
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and _matches(n, targets, scope)]
    if not names:
        raise LoRAError(f'no linear layer matches {list(targets)} (scope {scope})')
    for n in names:
        parent, _, child = n.rpartition('.')
        holder = model.get_submodule(parent) if parent else model
        setattr(holder, child, L(getattr(holder, child), r, alpha, dropout))
    return names


def lora_modules(model) -> dict:
    L = lora_class()
    return {n: m for n, m in model.named_modules() if isinstance(m, L)}


def lora_parameters(model) -> list:
    return [p for m in lora_modules(model).values() for p in (m.lora_A, m.lora_B)]


def lora_state_dict(model) -> dict:
    return {f'{n}.{k}': getattr(m, k).detach().cpu().clone() for n, m in lora_modules(model).items()
            for k in ('lora_A', 'lora_B')}


def merge_lora(model) -> int:
    """Fold each LoRA update into its base weight and put the plain nn.Linear back; returns how many were merged."""
    import torch
    mods = lora_modules(model)
    with torch.no_grad():
        for n, m in mods.items():
            m.base.weight += m.delta().to(m.base.weight.dtype)
            parent, _, child = n.rpartition('.')
            holder = model.get_submodule(parent) if parent else model
            setattr(holder, child, m.base)
    return len(mods)


def save_adapter(model, folder: str, meta: dict) -> dict:
    import torch
    os.makedirs(folder, exist_ok=True)
    mods = lora_modules(model)
    if not mods:
        raise LoRAError('the model has no LoRA layers')
    one = next(iter(mods.values()))
    cfg = dict(meta, r=one.r, alpha=one.alpha, modules=sorted(mods), schema='falcon.vlm.lora/1')
    tmp = os.path.join(folder, 'adapter.pt.part')
    torch.save(lora_state_dict(model), tmp)
    os.replace(tmp, os.path.join(folder, 'adapter.pt'))
    from .provenance import sha256_file, write_json
    cfg['sha256'] = sha256_file(os.path.join(folder, 'adapter.pt'))
    write_json(os.path.join(folder, 'adapter.json'), cfg)
    return cfg


def load_adapter(model, folder: str) -> dict:
    """Wrap model as adapter.json says and load adapter.pt (its SHA-256 checked)."""
    import torch
    from .provenance import read_json, sha256_file
    cfg = read_json(os.path.join(folder, 'adapter.json'))
    p = os.path.join(folder, 'adapter.pt')
    if cfg.get('sha256') and sha256_file(p) != cfg['sha256']:
        raise LoRAError(f'{p}: SHA-256 differs from adapter.json')
    names = apply_lora(model, cfg['targets'], cfg['r'], cfg['alpha'], 0.0, cfg.get('scope', 'text'))
    if sorted(names) != sorted(cfg['modules']):
        raise LoRAError('the adapter was trained on another architecture (its layers do not match this model)')
    state = torch.load(p, map_location='cpu', weights_only=True)
    mods = lora_modules(model)
    with torch.no_grad():
        for k, v in state.items():
            n, _, which = k.rpartition('.')
            getattr(mods[n], which).copy_(v)
    return cfg


# ================================================================================================ data
def split_drives(drives: list, val_share: float, seed: int) -> tuple:
    """(train, val) drive names: a seeded order, the last ⌈val_share · n⌉ (≥ 1) validate (module doc)."""
    from . import vlm as V
    ds = sorted(set(drives))
    if len(ds) < 2:
        raise LoRAError(f'{len(ds)} drive(s): train and validation are split by drive, so at least two are needed')
    rng = V._rng('split', seed)
    rng.shuffle(ds)
    k = max(1, math.ceil(val_share * len(ds)))
    if k >= len(ds):
        k = len(ds) - 1
    return sorted(ds[:-k]), sorted(ds[-k:])


def examples(lang_dirs: list, s: dict) -> dict:
    """{drive: [example]}: (images, prompt, answer = core.language's template, question, context) per question."""
    from . import vlm as V
    from .vlm_eval import load_language, images_for
    out = {}
    for d in lang_dirs:
        L = load_language(d)
        rows = []
        for k, rec in enumerate(L['records']):
            if k % max(int(s['every']), 1):
                continue
            ctx = V.context_of(rec, L['manifest'], L['name'])
            imgs = images_for(L, rec['step'], s['camera'], s['clip'], s['clip_step'])
            if not imgs:
                continue
            opts = {'max_objects': s['max_objects']}
            if s.get('types'):
                opts['types'] = s['types']
            for q in V.questions(rec, ctx, opts):
                if not q['template']:
                    continue
                rows.append({'images': imgs, 'prompt': f'{q["text"]} {V.PROMPT_SUFFIX}', 'answer': q['template'],
                             'question': q, 'context': ctx, 'drive': L['name'], 'sim_t': rec.get('sim_t')})
        if s.get('max_examples'):
            rows = rows[:int(s['max_examples'])]
        if L['name'] in out:
            raise LoRAError(f'two language folders name the same drive {L["name"]!r}')
        out[L['name']] = rows
    return out


def encode(runner, ex: dict):
    """(inputs, labels) for one example: the answer's tokens are the targets, the prompt's labels -100."""
    full, _ = runner.inputs(ex['images'], ex['prompt'], answer=ex['answer'] + _eos(runner))
    prompt, _ = runner.inputs(ex['images'], ex['prompt'])
    ids = full['input_ids']
    n = prompt['input_ids'].shape[1]
    if ids.shape[1] <= n or not (ids[0, :n] == prompt['input_ids'][0]).all():
        raise LoRAError('the tokenised prompt is not a prefix of the tokenised prompt + answer: this processor '
                        'cannot be fine-tuned here')
    labels = ids.clone()
    labels[:, :n] = -100
    if 'attention_mask' in full:
        labels[full['attention_mask'] == 0] = -100
    return full, labels


def _eos(runner) -> str:
    tok = getattr(runner.processor, 'tokenizer', runner.processor)
    for name in ('eos_token',):
        t = getattr(tok, name, None)
        if t:
            return t
    return ''


# ================================================================================================ training
def _versions() -> dict:
    import torch
    v = {'torch': torch.__version__, 'cuda': torch.version.cuda}
    try:
        import transformers
        v['transformers'] = transformers.__version__
    except ImportError:
        v['transformers'] = None
    return v


def settings_key(s: dict) -> str:
    from .provenance import sha256_json
    return sha256_json({k: v for k, v in s.items() if k not in VOLATILE})


def finetune(settings: dict, out_dir: str, emitter=None, cancel=None, argv=None, runner=None) -> dict:
    """Train a LoRA adapter (module doc) into out_dir: adapter/ (the best epoch), checkpoint.pt, metrics.jsonl,
    split.json, run.json. runner: an HFRunner already loaded (tests); else one is loaded from settings['model'].
    Resumes a cancelled run with the same settings; a finished one is returned as it is."""
    import torch
    from . import provenance as PV
    from . import vlm as V
    from .training import Cancelled, _Quiet
    from .vlm_models import CANDIDATES, HFRunner
    s = dict(DEFAULTS, **settings)
    unknown = sorted(set(s) - set(DEFAULTS))
    if unknown:
        raise LoRAError(f'unknown settings: {", ".join(unknown)}')
    em = emitter or _Quiet()
    cancel = cancel or (lambda: False)
    os.makedirs(out_dir, exist_ok=True)
    key = settings_key(s)
    run_p = os.path.join(out_dir, 'run.json')
    prev = PV.read_json(run_p) if os.path.isfile(run_p) else None
    if prev is not None:
        if prev.get('settings_key') != key:
            raise LoRAError(f'{out_dir} holds a fine-tuning with other settings; use another folder')
        if prev.get('status') == 'ok':
            return prev['summary']
    if not s['lang']:
        raise LoRAError('no language folders (lang): derive them with falcon derive language RUN --out DIR')
    t_start = time.time()
    if runner is None:
        if not s['model']:
            raise LoRAError('no base model: a downloaded candidate (falcon vlm list) or a model folder')
        runner = HFRunner(s['model'], s['workspace'], device=s['device'], max_new_tokens=s['max_new_tokens'],
                          max_side=s['max_side'], seed=s['seed'])
    model, dev = runner.model, runner.device
    data = examples(s['lang'], s)
    train_d, val_d = split_drives(list(data), s['val_share'], s['seed'])
    train = [e for d in train_d for e in data[d]]
    val = [e for d in val_d for e in data[d]]
    if not train or not val:
        raise LoRAError('no training or no validation examples (do the drives have camera images?)')
    PV.write_json(os.path.join(out_dir, 'split.json'), {'train': train_d, 'val': val_d, 'n_train': len(train),
                                                        'n_val': len(val), 'rule': 'by drive (core.vlm_lora)'})
    torch.manual_seed(s['seed'])
    names = apply_lora(model, s['targets'], s['r'], s['alpha'], s['dropout'], s['scope'])
    params = lora_parameters(model)
    n_train_params = sum(p.numel() for p in params)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(params, lr=s['lr'], weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / s['accumulate'])
    total = steps_per_epoch * s['epochs']
    ck_p = os.path.join(out_dir, 'checkpoint.pt')
    start_epoch, start_i, best, metrics = 0, 0, None, []
    if os.path.isfile(ck_p):
        ck = torch.load(ck_p, map_location='cpu', weights_only=False)
        mods = lora_modules(model)
        with torch.no_grad():
            for k, v in ck['lora'].items():
                n, _, which = k.rpartition('.')
                getattr(mods[n], which).copy_(v)
        opt.load_state_dict(ck['opt'])
        start_epoch, start_i, best, metrics = ck['epoch'], ck['i'], ck['best'], ck['metrics']
        em.log(f'resuming at epoch {start_epoch + 1}, example {start_i}')
    lic = runner.licence or (CANDIDATES.get(s['model']) or {}).get('licence')
    run = PV.new_run('vlm-finetune', argv or [], s['seed'], {k: v for k, v in s.items()},
                     {f'lang:{os.path.basename(os.path.normpath(d))}': os.path.join(d, 'language.jsonl')
                      for d in s['lang']})
    run.update(settings_key=key, model={'source': s['model'], 'folder': getattr(runner, 'folder', None),
                                        'licence': lic, 'name': runner.name},
               lora={'targets': s['targets'], 'scope': s['scope'], 'r': s['r'], 'alpha': s['alpha'],
                     'modules': len(names), 'trainable_parameters': n_train_params, 'parameters': n_params},
               versions=_versions(), device=str(dev))

    def lr_at(step):
        if step < s['warmup_steps']:
            return s['lr'] * (step + 1) / max(s['warmup_steps'], 1)
        f = (step - s['warmup_steps']) / max(total - s['warmup_steps'], 1)
        return s['lr'] * 0.5 * (1 + math.cos(math.pi * min(f, 1.0)))

    def save_ck(epoch, i):
        tmp = ck_p + '.part'
        torch.save({'lora': lora_state_dict(model), 'opt': opt.state_dict(), 'epoch': epoch, 'i': i, 'best': best,
                    'metrics': metrics}, tmp)
        os.replace(tmp, ck_p)

    def loss_of(ex):
        enc, labels = encode(runner, ex)
        enc = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in enc.items()}
        return model(**enc, labels=labels.to(dev)).loss

    try:
        for epoch in range(start_epoch, s['epochs']):
            order = list(range(len(train)))
            V._rng('order', s['seed'], epoch).shuffle(order)
            model.train()
            t0, seen, run_loss = time.time(), 0, 0.0
            opt.zero_grad(set_to_none=True)
            for i in range(start_i, len(order)):
                if cancel():                   # back to the last optimiser step: the partial sum is redone on resume
                    save_ck(epoch, i - i % s['accumulate'])
                    raise Cancelled()
                torch.manual_seed(V._rng('step', s['seed'], epoch, i).getrandbits(31))
                loss = loss_of(train[order[i]])
                (loss / s['accumulate']).backward()
                run_loss += float(loss.detach())
                seen += 1
                if (i + 1) % s['accumulate'] == 0 or i == len(order) - 1:
                    step = epoch * steps_per_epoch + i // s['accumulate']
                    for g in opt.param_groups:
                        g['lr'] = lr_at(step)
                    torch.nn.utils.clip_grad_norm_(params, s['clip_norm'])
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                em.progress((epoch + (i + 1) / len(order)) / s['epochs'], f'epoch {epoch + 1}/{s["epochs"]}')
            start_i = 0
            model.eval()
            with torch.no_grad():
                vl = [float(loss_of(e)) for e in val]
            row = {'epoch': epoch + 1, 'train_loss': run_loss / max(seen, 1), 'val_loss': sum(vl) / len(vl),
                   'train_s': time.time() - t0, 'examples_per_s': seen / max(time.time() - t0, 1e-9)}
            if s['val_generate']:
                k = min(int(s['val_generate']), len(val))
                pick = list(range(len(val)))
                V._rng('valgen', s['seed']).shuffle(pick)
                scored = []
                for j in sorted(pick[:k]):
                    e = val[j]
                    a = runner.answer(e['images'], e['prompt'], {})
                    sc = V.score_answer(e['question'], a, e['context'])
                    scored.append(sc['correct'])
                row['val_generated'] = k
                row['val_accuracy'] = sum(scored) / k if k else None
            metrics.append(row)
            em.metric('val_loss', row['val_loss'], source='measured', epoch=epoch + 1)
            em.log(f'epoch {epoch + 1}: train loss {row["train_loss"]:.4f}, validation loss {row["val_loss"]:.4f}')
            if best is None or row['val_loss'] < best['val_loss']:
                best = dict(row)
                save_adapter(model, os.path.join(out_dir, 'adapter'), {
                    'base': s['model'] or getattr(runner, 'folder', ''), 'licence': lic, 'targets': list(s['targets']),
                    'scope': s['scope'], 'epoch': epoch + 1, 'val_loss': row['val_loss'],
                    'note': 'a derivative of its base model, under the base model\'s licence'})
            save_ck(epoch + 1, 0)
            with open(os.path.join(out_dir, 'metrics.jsonl'), 'w', encoding='utf-8') as fh:
                fh.writelines(json.dumps(r) + '\n' for r in metrics)
    except Cancelled:
        run.update(status='cancelled', exit_code=3, finished=PV.utc_now())
        PV.write_json(run_p, run)
        raise
    summary = {'best_epoch': best['epoch'], 'best_val_loss': best['val_loss'], 'epochs': len(metrics),
               'train_drives': train_d, 'val_drives': val_d, 'n_train': len(train), 'n_val': len(val),
               'trainable_parameters': n_train_params, 'adapter': os.path.join(out_dir, 'adapter'),
               'val_accuracy': best.get('val_accuracy')}
    run.update(status='ok', exit_code=0, finished=PV.utc_now(), summary=summary, metrics=metrics,
               timings={'total_s': time.time() - t_start}, outputs=PV.hash_tree(out_dir, skip=('run.json',
                                                                                              'checkpoint.pt')))
    PV.write_json(run_p, run)
    return summary
