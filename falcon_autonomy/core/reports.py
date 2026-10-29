"""Reports: self-contained HTML pages of Falcon's results, with their provenance (standard library only; the Reports
page and `falcon report KIND SRC --out FILE`).

A report is one .html file: inline CSS (light and dark through prefers-color-scheme), charts as inline SVG (each mark
has a <title>, so a browser shows its value on hover, and every chart has a table of the same numbers beside it), no
script, font or image fetched from anywhere. It never computes a new number: it shows what the source files say,
with the source's own provenance, and records the SHA-256 of every file it read (REPORT_SCHEMA meta, embedded as JSON
in <script type="application/json" id="falcon-report">, which read_meta() reads back).

Kinds (KINDS) and their source:
- iteration    a loop iteration's folder (workspace/loop/NNN) or its iteration.json: the model and ODD, each step
               with its cost estimated and measured and its decisions, the weak cells, what was searched and
               generated, the retraining, and the before and after with intervals
- evaluation   an evaluation folder (evaluation.json): the numbers with their intervals, per class and range, latency
- dataset      a dataset spec's folder (workspace/datasets/SPEC, with plan.json) or a dataset version file: coverage
               against the targets, per class, range bin and cell
- study        a scenario study's folder (workspace/studies/NAME): the search, its curve and the failing cases
- leaderboard  a Leaderboard evaluation's folder (leaderboard.json): driving score, route completion and infraction
               penalty with intervals, per scenario and per km; the traffic-rule violations beside them when checked
- rootcause    a root-cause attribution's folder (rootcause.json; core.rootcause, `falcon why`): a failing case's
               factors with whether removing each fixes it (necessary) and whether it alone fails (sufficient), the
               re-runs; or an evaluation's correlates of misses with intervals; or a policy failure by its inputs
- rules        a traffic-rule check (core.rules): a folder with rules.json (`falcon rules check … --out DIR`, the
               Data page's check, or a stack run's rules.json): the rule set and its formulas, per run each rule's
               robustness, violations and counts per km, a timeline of the violations and the unknown stretches, and
               the aggregate over runs with intervals
- vlm          a vision-language evaluation's folder (scores.json; core.vlm_eval, `falcon vlm eval`): accuracy per
               question type with intervals, grounding, hallucination, how it is scored, the runner and its licence
- deploy       a deployment folder (deploy.json; core.deploy, `falcon deploy`): TensorRT engines per precision on the
               Jetson (or the labelled simulated device), latency, throughput, power and estimated energy, accuracy
               against the host's ONNX model with paired intervals, the release gate's verdicts and evidence
- ci           a CI run's output folder (ci-results.json; core.ci, `falcon ci run`): the verdict, each gate with what
               it measured and the rule that judged it, pass rates with Wilson intervals, the paired comparison with
               the baseline, the runs, the latency, the commit under test

Colours: before / first series blue (#2a78d6 light, #3987e5 dark), after / second orange (#eb6834, #d95926) — checked
for colour-vision deficiency (ΔE ≥ 24 protan) — and the two also differ by mark shape (circle, square) and are named
in a legend, so colour is never the only cue.
"""
from __future__ import annotations

import html
import json
import math
import os

from . import provenance as PV
from .paths import tilde

REPORT_SCHEMA = 'falcon.report/1'
KINDS = ('iteration', 'evaluation', 'dataset', 'study', 'leaderboard', 'rules', 'rootcause', 'vlm', 'deploy', 'ci')

CSS = """
:root{--bg:#fcfcfb;--panel:#f3f2ef;--ink:#0b0b0b;--ink2:#52514e;--muted:#6f6e69;--hair:#e1e0d9;--grid:#dcdad3;
--b:#2a78d6;--a:#eb6834;--good:#0a7f0a;--warn:#9a6a00;--crit:#b8312f}
@media (prefers-color-scheme:dark){:root{--bg:#121211;--panel:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a9993;
--hair:#2c2c2a;--grid:#383835;--b:#3987e5;--a:#d95926;--good:#3dbb3d;--warn:#fab219;--crit:#e66767}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 "IBM Plex Sans",system-ui,
-apple-system,"Segoe UI",sans-serif}main{max-width:1080px;margin:0 auto;padding:24px 20px 48px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px;padding-bottom:4px;border-bottom:1px solid
var(--hair)}h3{font-size:14px;margin:16px 0 6px}p,li{color:var(--ink2)}.muted{color:var(--muted)}
.note{background:var(--panel);border-left:3px solid var(--warn);padding:8px 12px;margin:10px 0;color:var(--ink2)}
table{border-collapse:collapse;width:100%;margin:6px 0 12px;font-variant-numeric:tabular-nums}th,td{text-align:left;
padding:4px 8px;border-bottom:1px solid var(--hair);vertical-align:top}th{color:var(--muted);font-weight:500}
td.n{text-align:right;font-family:"IBM Plex Mono",ui-monospace,monospace}code,.mono{font-family:"IBM Plex Mono",
ui-monospace,monospace;font-size:12px;word-break:break-all}dl{display:grid;grid-template-columns:max-content 1fr;
gap:2px 16px;margin:6px 0}dt{color:var(--muted)}dd{margin:0}svg{display:block;max-width:100%;height:auto}
svg text{fill:var(--ink2);font-size:11px}svg .ax{stroke:var(--grid)}svg .b{fill:var(--b);stroke:var(--b)}
svg .a{fill:var(--a);stroke:var(--a)}svg .ln{fill:none;stroke-width:2}svg .wb{stroke:var(--b);stroke-width:2}
svg .wa{stroke:var(--a);stroke-width:2}svg .bar{fill:var(--b)}svg .tgt{stroke:var(--ink);stroke-width:2}
svg .thr{stroke:var(--crit);stroke-dasharray:4 3}.legend span{display:inline-flex;align-items:center;gap:6px;
margin-right:16px;color:var(--ink2)}.sw{width:10px;height:10px;display:inline-block}.sw.b{background:var(--b);
border-radius:50%}.sw.a{background:var(--a)}.good{color:var(--good)}.crit{color:var(--crit)}.warn{color:var(--warn)}
footer{margin-top:32px;color:var(--muted);font-size:12px}
"""


class ReportError(Exception):
    pass


def esc(x) -> str:
    return html.escape('' if x is None else str(x))


def num(v, dp=3, signed=False) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '—'
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return f'{v:+d}' if signed else str(v)
    return f'{v:+.{dp}f}' if signed else f'{v:.{dp}f}'


def secs(v) -> str:
    if v is None:
        return '—'
    from .budget import fmt_duration
    return fmt_duration(v)


def table(head: list, rows: list, numeric=()) -> str:
    h = ''.join(f'<th>{esc(x)}</th>' for x in head)
    body = []
    for r in rows:
        body.append('<tr>' + ''.join(f'<td class="n">{c}</td>' if i in numeric else f'<td>{c}</td>'
                                     for i, c in enumerate(r)) + '</tr>')
    return f'<table><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def kv(pairs) -> str:
    return '<dl>' + ''.join(f'<dt>{esc(k)}</dt><dd>{v}</dd>' for k, v in pairs) + '</dl>'


def ci(v: dict | None) -> str:
    v = v or {}
    if v.get('value') is None:
        return '—'
    if v.get('ci_low') is None:
        return num(v['value'])
    return f'{num(v["value"])} [{num(v["ci_low"])}, {num(v["ci_high"])}]'


# ------------------------------------------------------------------------------------------------ charts (SVG)
def interval_chart(rows: list, lo: float = 0.0, hi: float = 1.0, title: str = '', names=('before', 'after')) -> str:
    """Dots with 95 % whiskers on one axis: rows [{'label', 'before': {value, ci_low, ci_high}, 'after': {...}}]
    (after optional). Before: blue circles; after: orange squares."""
    if not rows:
        return '<p class="muted">Nothing to chart.</p>'
    lw, w, rh = 260, 660, 30
    h = 30 + rh * len(rows)

    def X(v):
        return lw + (w - lw - 16) * (min(max(v, lo), hi) - lo) / ((hi - lo) or 1)
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        out.append(f'<line class="ax" x1="{X(v):.1f}" y1="6" x2="{X(v):.1f}" y2="{h - 18}"/>'
                   f'<text x="{X(v):.1f}" y="{h - 4}" text-anchor="middle">{v:g}</text>')
    for i, r in enumerate(rows):
        y = 18 + i * rh
        lab = str(r['label'])
        out.append(f'<text x="{lw - 8}" y="{y + 4}" text-anchor="end">{esc(lab[:46])}<title>{esc(lab)}</title>'
                   '</text>')
        for key, cls, dy in (('before', 'b', -5), ('after', 'a', 5)):
            v = r.get(key) or {}
            if v.get('value') is None:
                continue
            yy = y + (dy if r.get('after') else 0)
            tip = f'{lab} — {names[0] if key == "before" else names[1]}: {ci(v)}'
            if v.get('ci_low') is not None and v.get('ci_high') is not None:
                out.append(f'<line class="w{cls}" x1="{X(v["ci_low"]):.1f}" y1="{yy}" x2="{X(v["ci_high"]):.1f}" '
                           f'y2="{yy}"><title>{esc(tip)}</title></line>')
            if cls == 'b':
                out.append(f'<circle class="b" cx="{X(v["value"]):.1f}" cy="{yy}" r="4.5"><title>{esc(tip)}</title>'
                           '</circle>')
            else:
                out.append(f'<rect class="a" x="{X(v["value"]) - 4.5:.1f}" y="{yy - 4.5}" width="9" height="9" '
                           f'rx="1"><title>{esc(tip)}</title></rect>')
    out.append('</svg>')
    return ''.join(out)


def legend(before='before', after='after') -> str:
    return (f'<p class="legend"><span><i class="sw b"></i>{esc(before)} (circle)</span>'
            + (f'<span><i class="sw a"></i>{esc(after)} (square)</span>' if after else '')
            + '<span class="muted">lines: 95 % intervals</span></p>')


def bar_chart(rows: list, title: str = '') -> str:
    """Horizontal bars of what is there, with each target as a dark tick: rows [(label, have, target or None)]."""
    if not rows:
        return '<p class="muted">Nothing to chart.</p>'
    lw, w, rh = 160, 660, 24
    h = 26 + rh * len(rows)
    top = max([max(r[1], r[2] or 0) for r in rows] + [1])
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">']

    def X(v):
        return lw + (w - lw - 20) * v / top
    for i, (lab, have, tgt) in enumerate(rows):
        y = 6 + i * rh
        tip = f'{lab}: {have}' + (f' of {tgt}' if tgt else '')
        out.append(f'<text x="{lw - 8}" y="{y + 13}" text-anchor="end">{esc(lab)}</text>'
                   f'<rect class="bar" x="{lw}" y="{y + 3}" width="{max(X(have) - lw, 0):.1f}" height="14" rx="2">'
                   f'<title>{esc(tip)}</title></rect>')
        if tgt:
            out.append(f'<line class="tgt" x1="{X(tgt):.1f}" y1="{y}" x2="{X(tgt):.1f}" y2="{y + 20}">'
                       f'<title>{esc(f"target {tgt}")}</title></line>')
    out.append(f'<text x="{lw}" y="{h - 2}">0</text><text x="{w - 20}" y="{h - 2}" text-anchor="end">{top:g}</text>')
    out.append('</svg>')
    return ''.join(out)


def curve_chart(values: list, threshold: float | None, title: str = '') -> str:
    """The best value so far against evaluations (0–1), with the threshold dashed."""
    if not values:
        return '<p class="muted">No evaluations.</p>'
    w, h, x0, y0 = 660, 200, 40, 10
    pw, ph = w - x0 - 16, h - 40
    best, pts = 0.0, []
    for i, v in enumerate(values):
        best = max(best, v)
        pts.append((x0 + pw * i / max(len(values) - 1, 1), y0 + ph * (1 - min(max(best, 0), 1))))
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">',
           f'<rect class="ax" x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none"/>']
    if threshold is not None:
        ty = y0 + ph * (1 - threshold)
        out.append(f'<line class="thr" x1="{x0}" y1="{ty:.1f}" x2="{x0 + pw}" y2="{ty:.1f}"><title>threshold '
                   f'{threshold:g}</title></line>')
    out.append('<polyline class="b ln" fill="none" points="' + ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts) + '"/>')
    for i, (x, y) in enumerate(pts):
        out.append(f'<circle class="b" cx="{x:.1f}" cy="{y:.1f}" r="3"><title>evaluation {i + 1}: '
                   f'{values[i]:.3f} (best so far {1 - (y - y0) / ph:.3f})</title></circle>')
    out.append(f'<text x="{x0}" y="{h - 8}">1</text><text x="{x0 + pw}" y="{h - 8}" text-anchor="end">'
               f'{len(values)} evaluations</text><text x="{x0 - 6}" y="{y0 + 10}" text-anchor="end">1</text>'
               f'<text x="{x0 - 6}" y="{y0 + ph}" text-anchor="end">0</text>')
    out.append('</svg>')
    return ''.join(out)


# ------------------------------------------------------------------------------------------------ the page
class Page:
    def __init__(self, kind: str, title: str, source: str):
        self.kind, self.title, self.source = kind, title, os.path.abspath(source)
        self.parts: list = []
        self.read: dict = {}

    def json(self, path: str) -> dict:
        try:
            d = PV.read_json(path)
        except (OSError, ValueError) as e:
            raise ReportError(f'cannot read {path}: {e}') from None
        self.read[os.path.abspath(path)] = PV.sha256_file(path)
        return d

    def jsonl(self, path: str) -> list:
        if not os.path.isfile(path):
            return []
        self.read[os.path.abspath(path)] = PV.sha256_file(path)
        with open(path, encoding='utf-8') as fh:
            return [json.loads(x) for x in fh if x.strip()]

    def add(self, *parts):
        self.parts.extend(parts)

    def h2(self, t):
        self.add(f'<h2>{esc(t)}</h2>')

    def note(self, t):
        if t:
            self.add(f'<p class="note">{esc(t)}</p>')

    def provenance(self, prov: dict | None, what: str = 'the source'):
        self.h2('Provenance')
        if prov:
            ins = prov.get('inputs') or {}
            self.add(kv([('made by', esc(f'{prov.get("app")} {prov.get("app_version")}')),
                         ('created', esc(prov.get('created'))), ('seed', esc(prov.get('seed'))),
                         ('host', esc(', '.join(f'{k} {v}' for k, v in (prov.get('host') or {}).items())))]))
            if ins:
                self.add(f'<h3>Inputs of {esc(what)}</h3>', table(['role', 'path', 'SHA-256'], [
                    [esc(k), f'<span class="mono">{esc(v.get("path"))}</span>',
                     f'<span class="mono">{esc(v.get("sha256") or ("missing" if v.get("missing") else "—"))}</span>']
                    for k, v in ins.items()]))
        self.add('<h3>Files this report read</h3>', table(['file', 'SHA-256'], [
            [f'<span class="mono">{esc(p)}</span>', f'<span class="mono">{esc(h)}</span>']
            for p, h in sorted(self.read.items())]))

    def meta(self) -> dict:
        from .. import __version__
        return {'schema': REPORT_SCHEMA, 'kind': self.kind, 'title': self.title, 'source': self.source,
                'created': PV.utc_now(), 'app_version': __version__, 'inputs': dict(sorted(self.read.items()))}

    def render(self) -> tuple:
        meta = self.meta()
        blob = json.dumps(meta, ensure_ascii=False).replace('</', '<\\/')
        doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width,initial-scale=1">'
               f'<meta name="color-scheme" content="light dark"><title>{esc(self.title)}</title>'
               f'<style>{CSS}</style>'
               f'<script type="application/json" id="falcon-report">{blob}</script></head><body><main>'
               f'<h1>{esc(self.title)}</h1><p class="muted">{esc(self.kind)} report · source '
               f'<span class="mono">{esc(tilde(self.source))}</span> · '
               f'{esc(meta["created"])} · Falcon Autonomy {esc(meta["app_version"])}</p>'
               + ''.join(self.parts) +
               '<footer>Every number here was read from the files listed under Provenance, as Falcon wrote them; the '
               'report computes nothing new. Intervals are 95 % bootstrap intervals unless said otherwise.</footer>'
               '</main></body></html>\n')
        return doc, meta


# ------------------------------------------------------------------------------------------------ kinds
def _evaluation_body(pg: Page, d: str, heading: str = ''):
    ev = pg.json(os.path.join(d, 'evaluation.json'))
    m = ev.get('model') or {}
    tab = {r['metric']: r for r in ev['scores']['table']}
    if heading:
        pg.add(f'<h3>{esc(heading)}</h3>')
    pg.add(kv([('model', esc(f'{m.get("name")} ({m.get("runtime")}, {m.get("kind")})')),
               ('licence', esc(m.get('licence'))), ('drives', esc(len(ev.get('runs') or []))),
               ('frames', esc(ev.get('n_frames'))), ('camera', esc(', '.join(ev.get('cameras') or []))),
               ('interval', esc(f'bootstrap over {ev["scores"]["unit"]}s ({ev["scores"]["n_units"]}), '
                                f'{ev["scores"]["n_boot"]} resamples, seed {ev["scores"]["seed"]}')),
               ('ignored objects', esc(f'fewer than {ev["method"]["min_pixels"]} visible pixels or more than '
                                       f'{ev["method"]["max_occluded"]:.0%} hidden')),
               ('latency', esc(f'median {num(ev["latency_ms"]["median"], 2)} ms, p90 '
                               f'{num(ev["latency_ms"]["p90"], 2)} ms (measured on this machine; '
                               f'{", ".join(ev.get("providers") or [])})'))]))
    if ev.get('synthetic_model'):
        pg.note('SYNTHETIC: this model reads the ground truth (a test model with planted weaknesses); its numbers say '
                'nothing about any real detector.')
    pg.note(ev['scores'].get('note'))
    head = [r for k, r in tab.items() if k in ('AP', 'AP50', 'AP75', 'AR100')]
    pg.add(table(['metric', 'value', '95 % interval', 'units'], [[esc(r['metric']), num(r['value']),
                  f'[{num(r["ci_low"])}, {num(r["ci_high"])}]', esc(r.get('n'))] for r in head], numeric=(1, 2, 3)))
    for prefix, what in (('AP.range_m', 'AP by range (m)'), ('AP.', 'AP by class')):
        rows = [r for k, r in tab.items() if k.startswith(prefix) and (prefix != 'AP.' or k.count('[') == 0)]
        if not rows:
            continue
        pg.add(f'<h3>{esc(what)}</h3>', legend('estimate', None),
               interval_chart([{'label': r['metric'], 'before': r} for r in rows], title=what,
                              names=('estimate', None)),
               table(['metric', 'AP', '95 % interval'], [[esc(r['metric']), num(r['value']),
                     f'[{num(r["ci_low"])}, {num(r["ci_high"])}]'] for r in rows], numeric=(1, 2)))
    return ev


def report_evaluation(src: str) -> Page:
    d = src if os.path.isdir(src) else os.path.dirname(src)
    pg = Page('evaluation', 'Model evaluation', d)
    ev = _evaluation_body(pg, d)
    pg.title = f'Evaluation of {(ev.get("model") or {}).get("name")}'
    if ev.get('gallery'):
        pg.h2('Worst frames')
        pg.add(table(['frame', 'errors', 'missed', 'false positives'], [
            [f'<span class="mono">{esc(g["image_id"])}</span>', esc(g['n_errors']), esc(len(g['missed'])),
             esc(len(g['false_positives']))] for g in ev['gallery'][:12]], numeric=(1, 2, 3)))
    pg.provenance(ev.get('provenance'), 'the evaluation')
    return pg


def report_iteration(src: str) -> Page:
    from . import loop as L
    path = src if src.endswith('.json') else os.path.join(src, 'iteration.json')
    d = os.path.dirname(os.path.abspath(path))
    pg = Page('iteration', 'Loop iteration', d)
    it = pg.json(path)
    odd = it['odd']
    pg.title = f'Loop iteration {it["n"]}: {it["model"].get("name")} on {odd["name"]}'
    pg.add(kv([('status', esc(it['status']) + (f' — {esc(it.get("stopped"))}' if it.get('stopped') else '')),
               ('model', esc(f'{it["model"].get("name")} ({it.get("model_why")})')),
               ('model SHA-256', f'<span class="mono">{esc(it["model_sha256"])}</span>'),
               ('ODD', esc(f'{odd["name"]} ({odd["id"]})')),
               ('ODD SHA-256', f'<span class="mono">{esc(it["odd_sha256"])}</span>'),
               ('cells', esc(', '.join(c['label'] for c in L.odd_cells(odd)))),
               ('classes', esc(', '.join(odd['classes']))),
               ('range bins (m)', esc(', '.join(f'{b:g}' for b in odd['range_bins']) + ', open')),
               ('evaluation set', f'<span class="mono">{esc(it.get("evalset"))}</span> (held out: never trained on)'),
               ('created', esc(it['created']))]))
    summ = it.get('summary') or {}
    if summ.get('rows'):
        pg.h2('Before and after')
        pg.note(summ.get('note'))
        pg.add(legend('before', 'after'), interval_chart(summ['rows'], title='before and after'))
        pg.add(table(['', 'before', 'after', 'difference (paired)', 'verdict'], [
            [esc(r['label']), ci(r['before']), ci(r['after']),
             f'{num(r["diff"]["value"], signed=True)} [{num(r["diff"]["ci_low"], signed=True)}, '
             f'{num(r["diff"]["ci_high"], signed=True)}]',
             f'<span class="{ {"improvement": "good", "regression": "crit"}.get(r["verdict"], "")}">'
             f'{esc(r["verdict"])}</span>'] for r in summ['rows']], numeric=(1, 2, 3)))
        pg.add(f'<p class="muted">The same {esc(summ.get("n_frames"))} frames of the evaluation set before and after; '
               f'the difference is a paired bootstrap over {esc(summ.get("unit"))}s ({esc(summ.get("n_units"))}). '
               'A verdict needs the whole interval on one side of zero; otherwise there is no detectable change at '
               '95 %, which is not the same as no change.</p>')
    pg.h2('Steps and their cost')
    rows = []
    for k, s in enumerate(L.STEPS, 1):
        rec = it['steps'][s]
        est = rec.get('estimate') or {}
        meas = (rec.get('measured') or {}).get('wall_s')
        rows.append([esc(f'{k}. {L.STEP_TITLES[s]}'), esc(rec.get('state')),
                     esc(f'{secs(est.get("seconds"))} ({est.get("label", "—")})'), secs(meas),
                     '<br>'.join(esc(x) for x in (rec.get('decisions') or [])) + (
                         f'<br><span class="crit">{esc(rec["error"])}</span>' if rec.get('error') else '')])
    pg.add(table(['step', 'state', 'estimated before', 'measured', 'decisions'], rows, numeric=(2, 3)))
    pg.add('<p class="muted">Estimates are made before a step runs, from this machine\'s measured throughput where '
           'there is one (labelled measured) or the reference machine\'s (typical); "not estimated" says nothing was '
           'measured to go on.</p>')
    for s in L.STEPS:
        src_note = (it['steps'][s].get('estimate') or {}).get('source')
        if src_note:
            pg.add(f'<p class="muted">Step {L.STEPS.index(s) + 1}: {esc(src_note)}</p>')
    wcp = os.path.join(d, 'weak_cells.json')
    if os.path.isfile(wcp):
        wc = pg.json(wcp)
        pg.h2('Weak cells')
        pg.add(f'<p>{esc(wc["rule"])}. Overall AP {ci(wc["overall_AP"])}; at least {esc(wc["min_objects"])} objects.'
               '</p>')
        if wc.get('fallback_used'):
            pg.note('No cell was weak by the rule; the ODD asked for the fallback, so the lowest cells were targeted.')
        chosen = wc['chosen']
        pg.add(legend('AP of the cell', None), interval_chart([{'label': L._cell_name(c), 'before': {
            'value': c['AP'], 'ci_low': c['ci_low'], 'ci_high': c['ci_high']}} for c in chosen], title='weak cells',
            names=('AP of the cell', None)))
        lowest = sorted([r for r in wc['rows'] if r['ci_high'] is not None and r['n_objects']],
                        key=lambda r: (r['ci_high'], -r['n_objects']))[:15]
        pg.add(table(['cell', 'AP', '95 % interval', 'objects', 'weak', 'chosen'], [
            [esc(L._cell_name(r)), num(r['AP']), f'[{num(r["ci_low"])}, {num(r["ci_high"])}]', esc(r['n_objects']),
             'yes' if r.get('weak') else '', esc(r.get('rank') or '')] for r in lowest], numeric=(1, 2, 3)))
    sp = os.path.join(d, 'search.json')
    if os.path.isfile(sp):
        sd = pg.json(sp)
        pg.h2('Rare-case search')
        pg.note(f'Objective: {sd["objective"]}.')
        pg.add(table(['cell', 'scenario', 'conditions', 'runs', 'failing', 'best criticality', 'study'], [
            [esc(c['cell']), esc(c.get('scenario') or (c.get('note') or 'none')), esc(c['condition']),
             esc(c.get('evaluated', 0)), esc(len(c.get('failures') or [])), num(c.get('best')),
             f'<span class="mono">{esc(c.get("study") or "")}</span>'] for c in sd['cells']], numeric=(3, 4, 5)))
        pg.add('<p class="muted">Scenario for a class: ' + esc('; '.join(f'{k} → {", ".join(v)}' for k, v in
                                                                         sd['mapping'].items())) + '.</p>')
    gp = os.path.join(d, 'generate.json')
    if os.path.isfile(gp):
        g = pg.json(gp)
        pg.h2('Generated data')
        sp_ = g['spec']
        pg.add(kv([('dataset spec', esc(f'{sp_["name"]} ({sp_["id"]})')),
                   ('targets', esc(', '.join(f'{v} {c}' for c, v in sp_['targets'].items()) or 'none')),
                   ('weathers', esc(', '.join(w['name'] for w in sp_['weathers']))),
                   ('drives', esc(len(g['drives']))), ('failing cases re-run with labels', esc(len(g['reruns']))),
                   ('camera frames', esc(g['frames'])), ('yield', esc(g['plan'].get('yield_source')))]))
    rt = it['steps'].get('retrain') or {}
    pg.h2('Retraining')
    if rt.get('state') == 'stopped':
        pg.note(f'Stopped: {rt.get("error")}')
    elif rt.get('result'):
        r = rt['result']
        meas = rt.get('measured') or {}
        pg.add(kv([('training', f'<span class="mono">{esc(r["training"])}</span>'),
                   ('model', esc(r['registration'].get('name'))),
                   ('best validation AP', num(r.get('best_val_AP'))), ('ONNX verified', esc(r.get('onnx_verified'))),
                   ('measured', esc(f'{num(meas.get("images_per_s"), 1)} images/s on {meas.get("device")}'
                                    + (f', peak {num(meas.get("peak_gpu_mib"), 0)} MiB' if meas.get('peak_gpu_mib')
                                       else '')))]))
    else:
        pg.add('<p class="muted">Not run yet.</p>')
    pg.provenance(it.get('provenance'), 'the iteration')
    outs = [(s, p, h) for s in L.STEPS for p, h in ((it['steps'][s].get('outputs') or {}).items())]
    if outs:
        pg.add('<h3>Outputs of each step</h3>', table(['step', 'file', 'SHA-256'], [
            [esc(L.STEPS.index(s) + 1), f'<span class="mono">{esc(p)}</span>', f'<span class="mono">{esc(h)}</span>']
            for s, p, h in outs]))
    return pg


def report_dataset(src: str) -> Page:
    from . import datasets as D
    if src.endswith('.json'):
        pg = Page('dataset', 'Dataset version', src)
        v = pg.json(src)
        ws = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(src)))))
        spec = v['spec_snapshot']
        runs = [os.path.join(ws, e['run']) for e in v['runs']]
        pg.title = f'Dataset {v["name"]}, version {v["version"]}'
        pg.add(kv([('split by', esc(v['split_by'])), ('runs', esc(len(runs))),
                   ('SHA-256', f'<span class="mono">{esc(v.get("sha256"))}</span>')]))
        pg.add(table(['split', 'units'], [[esc(k), esc(len(x))] for k, x in v['splits'].items()], numeric=(1,)))
        prov = v.get('provenance')
    else:
        pg = Page('dataset', 'Dataset coverage', src)
        plan = pg.json(os.path.join(src, 'plan.json'))
        spec = plan['spec']
        from ..sim.engine import finished_runs
        ws = os.path.dirname(os.path.dirname(os.path.abspath(src)))
        runs = finished_runs(ws, spec['id'])
        pg.title = f'Dataset {spec["name"]}: coverage'
        prov = None
    for r in runs:
        rj = os.path.join(r, 'run.json')
        if os.path.isfile(rj):
            pg.read[os.path.abspath(rj)] = PV.sha256_file(rj)
    rows = [D.coverage_of_run(r, spec.get('bins')) for r in runs]
    cov = D.coverage(spec, rows)
    pg.add(kv([('spec', esc(f'{spec["name"]} ({spec["id"]})')), ('drives', esc(cov['drives'])),
               ('camera frames', esc(cov['frames'])),
               ('worlds', esc(', '.join(spec.get('worlds') or []))),
               ('weathers', esc(', '.join(w if isinstance(w, str) else w.get('name') or w.get('preset', '?')
                                          for w in spec.get('weathers') or [])))]))
    pg.h2('Objects against the targets')
    classes = sorted(set(cov['have']) | set(cov['targets']))
    pg.add('<p class="legend"><span><i class="sw b"></i>objects recorded</span><span>| target</span></p>',
           bar_chart([(c, cov['have'].get(c, 0), cov['targets'].get(c)) for c in classes], 'coverage'),
           table(['class', 'objects', 'target', 'filled'], [
               [esc(c), esc(cov['have'].get(c, 0)), esc(cov['targets'].get(c, '—')),
                f'{cov["fill"][c]:.0%}' if c in cov['fill'] else '—'] for c in classes], numeric=(1, 2, 3)))
    for b, by in cov['bins'].items():
        edges = (spec.get('bins') or D.DEFAULT_BINS).get(b) or []
        if not by or not edges:
            continue
        pg.add(f'<h3>By {esc(b)}</h3>', table(['class'] + [f'{a:g}–{c:g}' for a, c in zip(edges, edges[1:])], [
            [esc(c)] + [esc(n) for n in counts] for c, counts in sorted(by.items())],
            numeric=tuple(range(1, len(edges)))))
    if cov['cells']:
        pg.h2('Per cell')
        pg.add(table(['cell', 'frames', 'objects'], [[esc(k), esc(v['frames']), esc(', '.join(
            f'{n} {c}' for c, n in sorted(v['objects'].items())))] for k, v in sorted(cov['cells'].items())],
            numeric=(1,)))
    pg.provenance(prov, 'the dataset version')
    return pg


def report_study(src: str) -> Page:
    pg = Page('study', 'Rare-case study', src)
    st = pg.json(os.path.join(src, 'study.json'))
    results = pg.jsonl(os.path.join(src, 'results.jsonl'))
    scn = st['scenario']
    pg.title = f'Rare cases: {scn.get("name")} ({st["name"]})'
    pg.add(kv([('scenario', esc(f'{scn.get("name")} ({scn.get("id")})')), ('world', esc(st.get('world'))),
               ('rig', esc(st.get('rig'))), ('seed', esc(st.get('seed'))),
               ('weather', esc(json.dumps(st['weather'])) if st.get('weather') else 'the world\'s'),
               ('evaluations', esc(len(results)))]))
    obj = st.get('objective') or {}
    pg.note(f'The objective is {obj["about"]}.' if obj.get('about') else
            'The objective is the run\'s criticality (core.criticality; 1 for a collision): where driving the '
            'scenario becomes critical, not where a perception model fails.')
    searches = sorted(f for f in os.listdir(src) if f.startswith('search-') and f.endswith('.json'))
    for f in searches:
        s = pg.json(os.path.join(src, f))
        pg.h2(f'Search: {s["method"]}')
        pg.add(kv([('budget', esc(s['budget'])), ('evaluated', esc(len(s['history']))),
                   ('failure threshold', esc(s['threshold'])), ('failures', esc(len(s['failures'])))]))
        vals = [h['value'] for h in sorted(s['history'], key=lambda h: h['i'])]
        pg.add('<p class="legend"><span><i class="sw b"></i>best criticality so far</span><span>dashed: the '
               'failure threshold</span></p>', curve_chart(vals, s['threshold'], 'search curve'))
        pg.add(table(['rank', 'criticality', 'parameters'], [
            [esc(k + 1), num(h['value']), f'<span class="mono">{esc(json.dumps(h["params"]))}</span>']
            for k, h in enumerate(s['failures'][:20])], numeric=(0, 1)))
    if not searches and results:
        pg.h2('Evaluations')
        pg.add(curve_chart([r['objective'] for r in results], None, 'evaluations'))
    pg.h2('Runs')
    pg.add(table(['parameters', 'criticality', 'status', 'run'], [
        [f'<span class="mono">{esc(json.dumps(r["values"]))}</span>', num(r.get('objective')), esc(r.get('status')),
         f'<span class="mono">{esc(os.path.basename(r.get("run") or ""))}</span>']
        for r in sorted(results, key=lambda r: -(r.get('objective') or 0))[:30]], numeric=(1,)))
    pg.provenance(None)
    return pg


def report_leaderboard(src: str) -> Page:
    d = src if os.path.isdir(src) else os.path.dirname(src)
    pg = Page('leaderboard', 'Leaderboard-style evaluation', d)
    lb = pg.json(os.path.join(d, 'leaderboard.json'))
    pg.title = f'Leaderboard-style driving evaluation ({lb["routes"]} routes)'
    pg.add(kv([('policy', f'<span class="mono">{esc(lb.get("policy"))}</span>'), ('routes', esc(lb['routes'])),
               ('distance', esc(f'{lb["distance_km"]:.2f} km')), ('method', esc(lb.get('method')))]))
    pg.note('Falcon\'s routes follow the CARLA Leaderboard\'s scoring but are not Leaderboard routes: these numbers '
            'are not comparable with its public results. Not scored: ' + str(lb.get('not_scored')))
    mets = lb['metrics']
    pg.h2('Scores')
    pg.add(legend('estimate', None), interval_chart([{'label': k.replace('_', ' '), 'before': v}
                                                     for k, v in mets.items()], title='scores',
                                                    names=('estimate', None)),
           table(['metric', 'value', '95 % interval'], [[esc(k.replace('_', ' ')), num(v['value']),
                 f'[{num(v["ci_low"])}, {num(v["ci_high"])}]'] for k, v in mets.items()], numeric=(1, 2)))
    if lb.get('by_scenario'):
        pg.h2('Driving score by scenario')
        pg.add(table(['scenario', 'mean driving score'], [[esc(k), num(v)] for k, v in lb['by_scenario'].items()],
                     numeric=(1,)))
    if lb.get('infractions_per_km'):
        pg.h2('Infractions per km')
        pg.add(table(['kind', 'per km', 'penalty'], [[esc(k), num(v), num((lb.get('penalties') or {}).get(k))]
                                                     for k, v in lb['infractions_per_km'].items()], numeric=(1, 2)))
    if lb.get('rules'):
        pg.h2('Traffic rules (beside the score, not in it)')
        _rules_aggregate(pg, lb['rules'])
    if lb.get('failed'):
        pg.h2('Routes that could not run')
        pg.add(table(['scenario', 'error'], [[esc(f['scenario']), esc(f['error'])] for f in lb['failed']]))
    pg.provenance(lb.get('provenance'), 'the evaluation')
    return pg


STATE_CLASS = {'violated': 'crit', 'satisfied': 'good', 'not evaluable': 'muted'}


def timeline_chart(result: dict, width: int = 900) -> str:
    """One row per rule over the run's time: violations as red bars, unknown stretches as grey ones (each with a
    <title>); the table beside it lists the same intervals."""
    rows = result['rules']
    T = max(float(result.get('duration_s') or 0.0), 1e-6)
    x0, rh, pw = 190, 18, width - 200
    h = 26 + rh * len(rows)
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="rule violations over time">']
    for k, r in enumerate(rows):
        y = 6 + k * rh
        out.append(f'<text x="{x0 - 6}" y="{y + 12}" text-anchor="end">{esc(r["id"])}</text>'
                   f'<line class="ax" x1="{x0}" y1="{y + rh - 2}" x2="{x0 + pw}" y2="{y + rh - 2}"/>')
        for a, b in r.get('unknown') or ():
            out.append(f'<rect x="{x0 + pw * a / T:.1f}" y="{y + 3}" width="{max(pw * (b - a) / T, 1):.1f}" '
                       f'height="{rh - 6}" style="fill:var(--grid)"><title>{esc(r["id"])}: unknown '
                       f'{a:.2f}–{b:.2f} s</title></rect>')
        for v in r['violations']:
            out.append(f'<rect x="{x0 + pw * v["start"] / T:.1f}" y="{y + 2}" '
                       f'width="{max(pw * (v["end"] - v["start"]) / T, 2):.1f}" height="{rh - 4}" '
                       f'style="fill:var(--crit)"><title>{esc(r["id"])}: violated {v["start"]:.2f}–{v["end"]:.2f} s, '
                       f'worst robustness {v["min_robustness"]:.3f} at {v["at"]:.2f} s</title></rect>')
    yb = 10 + rh * len(rows)
    out.append(f'<text x="{x0}" y="{yb + 12}">0 s</text><text x="{x0 + pw}" y="{yb + 12}" text-anchor="end">'
               f'{T:.1f} s</text></svg>')
    return ''.join(out)


def _rules_result(pg: Page, r: dict):
    pg.add(kv([('samples', esc(f'{r["samples"]} every {r.get("step_s")} s ({r["duration_s"]} s)')),
               ('distance', esc(f'{r["distance_km"]} km')), ('handedness of the run', esc(r.get('handedness'))),
               ('sources', esc('; '.join(f'{k}: {v}' for k, v in (r.get('sources') or {}).items())))]))
    for w in r.get('warnings') or ():
        pg.note(w)
    pg.add('<p class="legend"><span><i class="sw" style="background:var(--crit)"></i>violated</span><span><i '
           'class="sw" style="background:var(--grid)"></i>unknown (not recorded or not measurable)</span></p>',
           timeline_chart(r))
    pg.add(table(['rule', 'state', 'robustness', 'violations', 'per km', 'evaluated', 'intervals (s)'], [
        [f'{esc(x["title"])} <span class="mono">{esc(x["id"])}</span>',
         f'<span class="{STATE_CLASS.get(x["state"], "muted")}">{esc(x["state"])}</span>',
         '∞' if x.get('vacuous') else num(x['robustness']), esc(x['count']),
         num(x['per_km'], 2), esc(f'{100 * (x["evaluated_share"] or 0):.0f} %'),
         esc(', '.join(f'{v["start"]:.2f}–{v["end"]:.2f}' for v in x['violations'][:8])
             + (' …' if len(x['violations']) > 8 else '')
             + (f' (unknown: {", ".join(x["missing_signals"])})' if x.get('missing_signals') else ''))]
        for x in r['rules']], numeric=(2, 3, 4, 5)))
    for n in r.get('notes') or ():
        pg.add(f'<p class="muted">{esc(n)}</p>')


def _rules_aggregate(pg: Page, agg: dict):
    pg.add(kv([('runs', esc(agg.get('runs'))), ('distance', esc(f'{agg.get("distance_km", 0):.3f} km')),
               ('violations', esc(agg.get('violations'))), ('per km', num(agg.get('per_km'), 2)),
               ('rule set', esc((agg.get('ruleset') or {}).get('name'))), ('intervals', esc(agg.get('interval')))]))
    pg.add(table(['rule', 'runs evaluated', 'violations', 'per km [95 %]', 'runs violating [95 %]',
                  'lowest robustness'], [
        [esc(r['title']), esc(r['runs_evaluated']), esc(r['violations']),
         num(r['per_km'], 2) + (f' [{num(r["per_km_ci95"][0], 2)}, {num(r["per_km_ci95"][1], 2)}]'
                                if r.get('per_km_ci95') else ''),
         esc(f'{r["runs_violating"]}') + (f' [{num(r["share_ci95"][0], 2)}, {num(r["share_ci95"][1], 2)}]'
                                          if r.get('share_ci95') else ''), num(r['min_robustness'])]
        for r in agg.get('rules') or ()], numeric=(1, 2, 3, 4, 5)))


def report_rules(src: str) -> Page:
    p = src if src.endswith('.json') else os.path.join(src, 'rules.json')
    pg = Page('rules', 'Traffic-rule check', p)
    doc = pg.json(p)
    results = doc['results'] if doc.get('schema') == 'falcon.rules_results/1' else [doc]
    if not results or results[0].get('schema') != 'falcon.rules_check/1':
        raise ReportError(f'{p} is not a traffic-rule check (falcon.rules_check/1)')
    sets = {}
    for r in results:
        sets.setdefault(r['ruleset']['id'], r)
    pg.title = (f'Traffic rules: {", ".join(r["ruleset"].get("name") or k for k, r in sets.items())} on '
                f'{len(results)} run(s)')
    pg.note('Rules marked heuristic use thresholds that are Falcon\'s choices, not the law\'s. Not recorded, so not '
            'checked: ' + '; '.join(results[0].get('not_recorded') or ()))
    for k, first in sets.items():
        rs = first['ruleset']
        pg.h2(f'Rule set {rs.get("name") or k}')
        pg.add(kv([('handedness', esc(rs.get('handedness'))),
                   ('rule file', f'<span class="mono">{esc(rs.get("source"))}</span>'),
                   ('its SHA-256', f'<span class="mono">{esc(rs.get("sha256"))}</span>'),
                   ('method', esc(first.get('method')))]))
        pg.add(table(['rule', 'tags', 'formula'], [[esc(f'{x["title"]} ({x["id"]})'), esc(' '.join(x['tags'])),
                                                   f'<span class="mono">{esc(x["formula"])}</span>']
                                                  for x in first['rules']]))
        agg = (doc.get('aggregates') or {}).get(k)
        if agg:
            pg.add(f'<h3>Over its {agg["runs"]} runs</h3>')
            _rules_aggregate(pg, agg)
    for r in results:
        pg.h2(f'Run {os.path.basename(str(r.get("run") or r.get("source") or ""))}'
              + (f' episode {r["episode"]}' if r.get('episode') is not None else ''))
        _rules_result(pg, r)
    pg.provenance(doc.get('provenance'), 'the check')
    return pg


BUILDERS = {'iteration': report_iteration, 'evaluation': report_evaluation, 'dataset': report_dataset,
            'study': report_study, 'leaderboard': report_leaderboard, 'rules': report_rules,
            'rootcause': lambda src: _rootcause(src), 'vlm': lambda src: _vlm(src),
            'deploy': lambda src: _deploy(src), 'ci': lambda src: _ci(src)}


def _rootcause(src: str) -> Page:
    from .rootcause import report
    return report(src)


def _vlm(src: str) -> Page:
    from .vlm_eval import report
    return report(src)


def _deploy(src: str) -> Page:
    from .deploy import report
    return report(src)


def _ci(src: str) -> Page:
    from .ci import report
    return report(src)


def build(kind: str, src: str) -> Page:
    if kind not in BUILDERS:
        raise ReportError(f'unknown kind {kind!r}: one of {", ".join(KINDS)}')
    if not os.path.exists(src):
        raise ReportError(f'{src} does not exist')
    try:
        return BUILDERS[kind](src)
    except (KeyError, TypeError, IndexError) as e:
        raise ReportError(f'{src} is not a {kind} Falcon can report on ({type(e).__name__}: {e})') from None


def write_report(kind: str, src: str, out: str) -> dict:
    """Write the report of src as out (.html); returns its meta."""
    from .paths import atomic_write
    doc, meta = build(kind, src).render()
    meta['path'] = os.path.abspath(out)
    atomic_write(out, doc)
    return meta


def read_meta(path: str) -> dict | None:
    """The meta a report embeds (REPORT_SCHEMA), or None for a file that is not a Falcon report."""
    try:
        with open(path, encoding='utf-8') as fh:
            head = fh.read(1 << 20)
    except OSError:
        return None
    tag = '<script type="application/json" id="falcon-report">'
    i = head.find(tag)
    if i < 0:
        return None
    j = head.find('</script>', i)
    try:
        meta = json.loads(head[i + len(tag):j].replace('<\\/', '</'))
    except ValueError:
        return None
    if meta.get('schema') != REPORT_SCHEMA:
        return None
    meta['path'] = os.path.abspath(path)
    return meta


def reports_dir(ws: str) -> str:
    return os.path.join(ws, 'reports')


def list_reports(folder: str) -> list:
    """The reports in a folder, newest first."""
    out = []
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            if f.endswith('.html'):
                m = read_meta(os.path.join(folder, f))
                if m is not None:
                    out.append(m)
    return sorted(out, key=lambda m: (m.get('created') or '', m['path']), reverse=True)


def sources(ws: str) -> dict:
    """What can be reported on in a workspace: {kind: [(label, path)]}, newest first where names carry time."""
    out = {k: [] for k in KINDS}
    for it in sorted(_listdir(os.path.join(ws, 'loop')), reverse=True):
        if it.isdigit() and os.path.isfile(os.path.join(ws, 'loop', it, 'iteration.json')):
            out['iteration'].append((f'iteration {int(it)}', os.path.join(ws, 'loop', it)))
    for base in ('evaluations',):
        for e in sorted(_listdir(os.path.join(ws, base)), reverse=True):
            if os.path.isfile(os.path.join(ws, base, e, 'evaluation.json')):
                out['evaluation'].append((e, os.path.join(ws, base, e)))
    for it in sorted(_listdir(os.path.join(ws, 'loop')), reverse=True):
        for sub in ('before', 'after'):
            p = os.path.join(ws, 'loop', it, sub)
            if os.path.isfile(os.path.join(p, 'evaluation.json')):
                out['evaluation'].append((f'loop {it} {sub}', p))
    for spec in sorted(_listdir(os.path.join(ws, 'datasets'))):
        sd = os.path.join(ws, 'datasets', spec)
        if os.path.isfile(os.path.join(sd, 'plan.json')):
            out['dataset'].append((f'{spec} (coverage)', sd))
        for v in sorted(_listdir(os.path.join(sd, 'versions'))):
            if v.endswith('.json'):
                out['dataset'].append((f'{spec} {v[:-5]}', os.path.join(sd, 'versions', v)))
    for s in sorted(_listdir(os.path.join(ws, 'studies'))):
        if os.path.isfile(os.path.join(ws, 'studies', s, 'study.json')):
            out['study'].append((s, os.path.join(ws, 'studies', s)))
    for d in sorted(_listdir(os.path.join(ws, 'leaderboard')), reverse=True):      # the Training page's folders
        if os.path.isfile(os.path.join(ws, 'leaderboard', d, 'leaderboard.json')):
            out['leaderboard'].append((d, os.path.join(ws, 'leaderboard', d)))
    for d in sorted(_listdir(os.path.join(ws, 'rootcause')), reverse=True):        # "Why did this fail?"
        if os.path.isfile(os.path.join(ws, 'rootcause', d, 'rootcause.json')):
            out['rootcause'].append((d, os.path.join(ws, 'rootcause', d)))
    for d in sorted(_listdir(os.path.join(ws, 'rules')), reverse=True):            # the Data page's rule checks
        if os.path.isfile(os.path.join(ws, 'rules', d, 'rules.json')):
            out['rules'].append((d, os.path.join(ws, 'rules', d)))
    for d in sorted(_listdir(os.path.join(ws, 'vlm', 'evaluations')), reverse=True):  # the Models page's Language tab
        if os.path.isfile(os.path.join(ws, 'vlm', 'evaluations', d, 'scores.json')):
            out['vlm'].append((d, os.path.join(ws, 'vlm', 'evaluations', d)))
    for d in sorted(_listdir(os.path.join(ws, 'deploy')), reverse=True):           # the Models page's Deploy tab
        if os.path.isfile(os.path.join(ws, 'deploy', d, 'deploy.json')):
            out['deploy'].append((d, os.path.join(ws, 'deploy', d)))
    for d in sorted(_listdir(os.path.join(ws, 'ci')), reverse=True):               # `falcon ci run --out <ws>/ci/…`
        if os.path.isfile(os.path.join(ws, 'ci', d, 'ci-results.json')):
            out['ci'].append((d, os.path.join(ws, 'ci', d)))
    return out


def _listdir(d):
    try:
        return os.listdir(d)
    except OSError:
        return []


def export(paths: list, folder: str) -> list:
    """Copy reports into a folder (they are self-contained); returns the new paths."""
    import shutil
    os.makedirs(folder, exist_ok=True)
    out = []
    for p in paths:
        q = os.path.join(folder, os.path.basename(p))
        shutil.copyfile(p, q)
        out.append(q)
    return out


def default_name(kind: str, src: str) -> str:
    import re
    import time
    base = re.sub(r'[^A-Za-z0-9_.-]+', '-', os.path.basename(os.path.normpath(src)).replace('.json', ''))
    return f'{time.strftime("%Y%m%d-%H%M%S")}-{kind}-{base}.html'


def main_report(argv) -> int:
    """`falcon report KIND SRC --out FILE` (KIND: iteration, evaluation, dataset, study, leaderboard, rules,
    rootcause, vlm, deploy, ci): 0 ok, 2 failed."""
    import argparse
    from . import protocol as P
    ap = argparse.ArgumentParser(prog='falcon report', description='a self-contained HTML report (docs/reports.md)')
    ap.add_argument('kind', choices=KINDS)
    ap.add_argument('src')
    ap.add_argument('--out', '-o', required=True)
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    try:
        meta = write_report(a.kind, a.src, a.out)
    except (ReportError, OSError, ValueError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    P.Emitter().result('ok', path=meta['path'], title=meta['title'], inputs=len(meta['inputs']))
    return P.OK
