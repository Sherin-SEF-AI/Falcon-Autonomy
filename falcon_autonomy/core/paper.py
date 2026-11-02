"""Outputs ready for papers (standard library + numpy; OpenCV for PNG): any comparison as a LaTeX table (booktabs),
CSV and figures (PDF, PNG, SVG) with its confidence intervals, and a data card for each exported dataset.
docs/experiments.md ("Outputs for papers") is the manual; `falcon paper KIND SOURCE --out DIR` the command.

Sources (export(kind, source, out_dir)):
- experiment   an experiment folder with results.json: an ablation (core.ablation: each contrast's difference with
               its nominal and simultaneous intervals, effect size dz with its interval, adjusted p-value and verdict;
               each cell's mean with its interval) or a mixed sim/real experiment (core.experiments: each arm's AP
               with its interval and seed spread, the paired difference from real only)
- evaluation   an evaluation folder (evaluation.json; core.harness): every number with its bootstrap interval
- comparison   a comparison folder (comparison.json; `falcon evaluate --compare`): A, B and B − A with its paired
               interval and P(B > A)
- runs         two to four run folders (core.runcompare's table: what each run recorded, the difference from the
               first) — single runs carry no interval, so this kind has tables and no figure, and says so
- datacard     an export folder (falcon_export.json; core.exporters): the data card (below)

Each writes into out_dir: <name>.tex and <name>.csv per table, <name>.pdf / .png / .svg per figure, and paper.json
(PAPER_SCHEMA: the source's files with their SHA-256, every file written with its SHA-256). DETERMINISTIC: the same
source gives byte-identical files — numbers are formatted at fixed decimals, nothing records the time or the machine,
the PDF has no creation date, the PNG is drawn by OpenCV without metadata — so outputs can be diffed and kept in a
paper's repository. A LaTeX table needs \\usepackage{booktabs}; its header comment names the source and its hash.

Figures are dot-and-whisker plots: one row per number, the estimate as a dot, its interval as a thick whisker (and
the simultaneous interval of an ablation as a thin one), a dashed line at 0 for differences; up to four series told
apart by colour AND shape (blue circles, orange squares, green diamonds, purple triangles; blue and orange as in
core.reports, checked for colour-vision deficiency), named in a legend. Text is Helvetica (the PDF's standard font,
no embedding); the PNG uses OpenCV's Hershey font at 3 px per point. Glyphs outside Latin-1 (−, Δ, →) are written as
their ASCII stand-ins in the PDF and PNG.

The data card (datacard(), written by every export as DATACARD.md and datacard.json) follows Datasheets for Datasets
(Gebru et al. 2021, Communications of the ACM 64(12)), the sections that apply to a rendered set: composition
(runs, frames per split, objects per class), the collection process (CARLA build, Falcon version, rig and sensors,
rate, seeds, traffic), the labels, the conditions covered (worlds × weather, time of day, rain, fog), known gaps
(conditions and classes absent or thin, what the export left out, what is heuristic), the licence, the provenance
(every run's hashes). Runs that come from the user's dashcam footage (assistant.privacy's rule) are listed without
their names, places, routes, dates or folders.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os

from . import provenance as PV

PAPER_SCHEMA = 'falcon.paper/1'
DATACARD_SCHEMA = 'falcon.datacard/1'
KINDS = ('experiment', 'evaluation', 'comparison', 'runs', 'datacard')
FIGURE_FORMATS = ('pdf', 'png', 'svg')
SERIES = (('#2a78d6', 'circle'), ('#eb6834', 'square'), ('#1b9e77', 'diamond'), ('#7570b3', 'triangle'))
INK, MUTED, GRID = '#0b0b0b', '#6f6e69', '#dcdad3'
FEW_OBJECTS = 100


class PaperError(Exception):
    pass


# ================================================================================================ numbers
def fmt(v, digits: int = 3, signed: bool = False) -> str:
    """A number at fixed decimals ('' for None), never '-0.000'."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ''
    v = float(v)
    if round(v, digits) == 0:
        v = 0.0
    return f'{v:+.{digits}f}' if signed and v != 0 else f'{v:.{digits}f}'


def fmt_p(p) -> str:
    if p is None:
        return ''
    return '<0.001' if p < 0.001 else f'{p:.3f}'


# ================================================================================================ tables
def column(key: str, title: str, kind: str = 'num', digits: int = 3, signed: bool = False) -> dict:
    """kind: text, int, num, ci (row value (v, lo, hi)), p (a p-value)."""
    return {'key': key, 'title': title, 'kind': kind, 'digits': digits, 'signed': signed}


def table(name: str, caption: str, columns: list, rows: list, notes=(), label: str = '') -> dict:
    return {'name': name, 'caption': caption, 'columns': columns, 'rows': rows, 'notes': list(notes),
            'label': label or f'tab:{name}'}


_TEX = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{',
        '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}', '−': '$-$', 'Δ': r'$\Delta$',
        '×': r'$\times$', '→': r'$\rightarrow$', '≥': r'$\geq$', '≤': r'$\leq$', '±': r'$\pm$', '·': r'$\cdot$',
        '…': r'\ldots{}', '–': '--', '—': '---', '√': r'$\surd$', 'α': r'$\alpha$', '|': r'\textbar{}',
        '<': r'\textless{}', '>': r'\textgreater{}'}


def tex(s) -> str:
    """Text escaped for LaTeX (and the few non-ASCII signs Falcon writes, as math)."""
    return ''.join(_TEX.get(ch, ch) for ch in str(s))


def _tex_num(s: str) -> str:
    return s.replace('-', '$-$').replace('<', '$<$')


def _cell_tex(c: dict, v) -> str:
    k = c['kind']
    if k == 'text':
        return tex('' if v is None else v)
    if k == 'int':
        return '' if v is None else str(int(v))
    if k == 'p':
        return _tex_num(fmt_p(v))
    if k == 'ci':
        x, lo, hi = v if v is not None else (None, None, None)
        if x is None:
            return ''
        s = fmt(x, c['digits'], c['signed'])
        if lo is not None and hi is not None:
            s += f' [{fmt(lo, c["digits"], c["signed"])}, {fmt(hi, c["digits"], c["signed"])}]'
        return _tex_num(s)
    return _tex_num(fmt(v, c['digits'], c['signed']))


def to_latex(t: dict, source: str = '') -> str:
    """A booktabs table environment (needs \\usepackage{booktabs})."""
    from .. import __version__
    align = ''.join('l' if c['kind'] == 'text' else 'r' for c in t['columns'])
    out = [f'% {t["name"]}: written by Falcon Autonomy {__version__}' + (f' from {source}' if source else ''),
           '% needs \\usepackage{booktabs}', '\\begin{table}[tbp]', '  \\centering',
           f'  \\caption{{{tex(t["caption"])}}}', f'  \\label{{{t["label"]}}}', f'  \\begin{{tabular}}{{{align}}}',
           '    \\toprule', '    ' + ' & '.join(tex(c['title']) for c in t['columns']) + ' \\\\', '    \\midrule']
    for r in t['rows']:
        out.append('    ' + ' & '.join(_cell_tex(c, r.get(c['key'])) for c in t['columns']) + ' \\\\')
    out += ['    \\bottomrule', '  \\end{tabular}']
    if t['notes']:
        txt = ' '.join(n if n.rstrip().endswith('.') else n.rstrip() + '.' for n in t['notes'])
        out.append('  \\par\\smallskip{\\footnotesize ' + tex(txt) + '}')
    out.append('\\end{table}')
    return '\n'.join(out) + '\n'


def to_csv(t: dict) -> str:
    """CSV with a header; an interval column becomes three (KEY, KEY_ci_low, KEY_ci_high); numbers at 6 decimals."""
    head = []
    for c in t['columns']:
        head += [c['key'], f'{c["key"]}_ci_low', f'{c["key"]}_ci_high'] if c['kind'] == 'ci' else [c['key']]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(head)
    for r in t['rows']:
        row = []
        for c in t['columns']:
            v = r.get(c['key'])
            if c['kind'] == 'ci':
                row += [fmt(x, 6) for x in (v if v is not None else (None, None, None))]
            elif c['kind'] == 'text':
                row.append('' if v is None else str(v))
            elif c['kind'] == 'int':
                row.append('' if v is None else str(int(v)))
            else:
                row.append(fmt(v, 6))
        w.writerow(row)
    return buf.getvalue()


# ================================================================================================ figures
# Helvetica's advance widths (Adobe's AFM, 1/1000 em) for ASCII 32–126: text layout and right-aligned labels
_HELV = [278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556, 556, 556, 556,
         556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556, 1015, 667, 667, 722, 722, 667, 611, 778, 722, 278,
         500, 667, 556, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469,
         556, 333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500,
         278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584]
_ASCII = {'−': '-', 'Δ': 'Delta', '→': '->', '×': 'x', '…': '...', '≥': '>=', '≤': '<=', '±': '+/-', '·': '.',
          '–': '-', '—': '-', 'α': 'a', '√': 'sqrt', '’': "'", '‘': "'", '“': '"', '”': '"'}


def ascii_text(s: str) -> str:
    return ''.join(_ASCII.get(ch, ch if 32 <= ord(ch) < 127 else '?') for ch in str(s))


def text_width(s: str, size: float) -> float:
    return sum(_HELV[ord(ch) - 32] for ch in ascii_text(s)) * size / 1000.0


def _clip_text(s: str, size: float, width: float) -> str:
    if text_width(s, size) <= width:
        return s
    while s and text_width(s + '…', size) > width:
        s = s[:-1]
    return s + '…'


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.ceil(lo / step - 1e-9) * step
    out, k = [], 0
    while start + k * step <= hi + 1e-9 * step:
        out.append(round(start + k * step, 12))
        k += 1
    return out


def _tick_text(v: float, ticks: list) -> str:
    """A tick at the decimals its step needs (0.25 steps: two decimals; 0.5 steps: one)."""
    step = ticks[1] - ticks[0] if len(ticks) > 1 else 1.0
    for d in range(0, 7):
        if abs(round(step, d) - step) < 1e-9 * max(step, 1e-12):
            return fmt(v, d)
    return fmt(v, 6)


def figure(name: str, rows: list, xlabel: str, series: list | None = None, zero: bool = False,
           caption: str = '') -> dict:
    """A dot-and-whisker figure: rows [{'label', 'points': [{'value', 'lo', 'hi', 'lo2', 'hi2'} | None per
    series]}]; series names (one per point position; None: one unnamed series)."""
    return {'name': name, 'rows': rows, 'xlabel': xlabel, 'series': series, 'zero': zero, 'caption': caption}


def layout(fig: dict, width: float = 480.0) -> dict:
    """The figure as drawing primitives in points (y down): {'w', 'h', 'items': [...]}. Every backend draws these."""
    rows = fig['rows']
    if not rows:
        raise PaperError(f'figure {fig["name"]}: nothing to draw')
    ns = max(len(r['points']) for r in rows)
    if ns > len(SERIES):
        raise PaperError(f'figure {fig["name"]}: at most {len(SERIES)} series')
    fs, row_h = 8.0, max(16.0, 7.0 * ns + 6.0)
    lab_w = min(max(text_width(r['label'], fs) for r in rows), width * 0.45)
    x0, x1 = lab_w + 14.0, width - 14.0
    legend = bool(fig.get('series')) and any(fig['series'])
    y0 = 10.0 + (16.0 if legend else 0.0)
    y1 = y0 + row_h * len(rows)
    h = y1 + 38.0
    vals = []
    for r in rows:
        for p in r['points']:
            if p and p.get('value') is not None:
                vals += [p[k] for k in ('value', 'lo', 'hi', 'lo2', 'hi2') if p.get(k) is not None]
    if fig.get('zero'):
        vals.append(0.0)
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    pad = (hi - lo) * 0.06 or max(abs(hi), 1.0) * 0.1
    ticks = _nice_ticks(lo - pad, hi + pad)
    vlo, vhi = min(ticks[0], lo - pad), max(ticks[-1], hi + pad)

    def X(v):
        return x0 + (v - vlo) / (vhi - vlo) * (x1 - x0)
    it = []
    for t in ticks:
        it.append(('line', X(t), y0, X(t), y1, GRID, 0.5, None))
        it.append(('text', X(t), y1 + 11.0, _tick_text(t, ticks), 7.0, 'middle', MUTED))
    it.append(('line', x0, y1, x1, y1, MUTED, 0.6, None))
    if fig.get('zero') and vlo <= 0 <= vhi:
        it.append(('line', X(0.0), y0 - 2.0, X(0.0), y1, INK, 0.7, (3.0, 2.0)))
    it.append(('text', (x0 + x1) / 2, y1 + 26.0, fig['xlabel'], 8.5, 'middle', INK))
    if legend:
        lx = x0
        for j, nm in enumerate(fig['series']):
            if nm is None:
                continue
            col, shape = SERIES[j]
            it.append((shape, lx + 4.0, 12.0, 3.2, col, ''))
            it.append(('text', lx + 11.0, 14.8, nm, 8.0, 'start', INK))
            lx += 22.0 + text_width(nm, 8.0)
    for i, r in enumerate(rows):
        yc = y0 + row_h * (i + 0.5)
        it.append(('text', x0 - 6.0, yc + 2.8, _clip_text(r['label'], fs, lab_w), fs, 'end', INK))
        k = len(r['points'])
        drawn = False
        for j, p in enumerate(r['points']):
            if not p or p.get('value') is None:
                continue
            drawn = True
            col, shape = SERIES[j]
            y = yc + (j - (k - 1) / 2.0) * 7.0
            if p.get('lo2') is not None and p.get('hi2') is not None:
                it.append(('line', X(p['lo2']), y, X(p['hi2']), y, col, 0.6, None))
            if p.get('lo') is not None and p.get('hi') is not None:
                it.append(('line', X(p['lo']), y, X(p['hi']), y, col, 1.8, None))
            tip = fmt(p['value'], 4) + (f' [{fmt(p["lo"], 4)}, {fmt(p["hi"], 4)}]' if p.get('lo') is not None
                                        else '')
            it.append((shape, X(p['value']), y, 3.0, col, f'{r["label"]}: {tip}'))
        if not drawn:
            it.append(('text', x0 + 4.0, yc + 2.8, 'not measured', 7.0, 'start', MUTED))
    return {'w': width, 'h': h, 'items': it}


def _rgb(c: str) -> tuple:
    return tuple(int(c[k:k + 2], 16) for k in (1, 3, 5))


def _esc_xml(s: str) -> str:
    return (str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;'))


def render_svg(lay: dict) -> str:
    w, h = lay['w'], lay['h']
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.2f}pt" height="{h:.2f}pt" '
           f'viewBox="0 0 {w:.2f} {h:.2f}" font-family="Helvetica, Arial, sans-serif">',
           f'<rect x="0" y="0" width="{w:.2f}" height="{h:.2f}" fill="#ffffff"/>']
    for item in lay['items']:
        kind = item[0]
        if kind == 'line':
            _, xa, ya, xb, yb, col, lw, dash = item
            d = f' stroke-dasharray="{dash[0]:.2f} {dash[1]:.2f}"' if dash else ''
            out.append(f'<line x1="{xa:.2f}" y1="{ya:.2f}" x2="{xb:.2f}" y2="{yb:.2f}" stroke="{col}" '
                       f'stroke-width="{lw:.2f}"{d}/>')
        elif kind == 'text':
            _, x, y, s, size, anchor, col = item
            out.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size:.2f}" text-anchor="{anchor}" '
                       f'fill="{col}">{_esc_xml(s)}</text>')
        else:
            _, x, y, r, col, tip = item
            t = f'<title>{_esc_xml(tip)}</title>' if tip else ''
            if kind == 'circle':
                out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{col}">{t}</circle>')
            else:
                pts = _shape_points(kind, x, y, r)
                out.append('<polygon points="' + ' '.join(f'{a:.2f},{b:.2f}' for a, b in pts) +
                           f'" fill="{col}">{t}</polygon>')
    out.append('</svg>')
    return '\n'.join(out) + '\n'


def _shape_points(kind: str, x: float, y: float, r: float) -> list:
    if kind == 'square':
        q = r * 0.9
        return [(x - q, y - q), (x + q, y - q), (x + q, y + q), (x - q, y + q)]
    if kind == 'diamond':
        q = r * 1.25
        return [(x, y - q), (x + q, y), (x, y + q), (x - q, y)]
    q = r * 1.2                                                    # triangle
    return [(x, y - q), (x + q, y + q * 0.75), (x - q, y + q * 0.75)]


def _pdf_str(s: str) -> str:
    b = ascii_text(s)
    return b.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')


def render_pdf(lay: dict) -> bytes:
    """A one-page PDF 1.4 (vector; Helvetica, not embedded; no creation date: the same figure gives the same bytes)."""
    w, h = lay['w'], lay['h']
    ops = ['1 1 1 rg', f'0 0 {w:.2f} {h:.2f} re f']
    for item in lay['items']:
        kind = item[0]
        if kind == 'line':
            _, xa, ya, xb, yb, col, lw, dash = item
            r, g, b = (c / 255.0 for c in _rgb(col))
            d = f'[{dash[0]:.2f} {dash[1]:.2f}] 0 d' if dash else '[] 0 d'
            ops.append(f'{r:.3f} {g:.3f} {b:.3f} RG {lw:.2f} w {d} {xa:.2f} {h - ya:.2f} m {xb:.2f} {h - yb:.2f} l S')
        elif kind == 'text':
            _, x, y, s, size, anchor, col = item
            r, g, b = (c / 255.0 for c in _rgb(col))
            tw = text_width(s, size)
            xx = x - tw if anchor == 'end' else x - tw / 2 if anchor == 'middle' else x
            ops.append(f'BT /F1 {size:.2f} Tf {r:.3f} {g:.3f} {b:.3f} rg {xx:.2f} {h - y:.2f} Td '
                       f'({_pdf_str(s)}) Tj ET')
        else:
            _, x, y, rad, col, _tip = item
            r, g, b = (c / 255.0 for c in _rgb(col))
            if kind == 'circle':
                k = 0.5523 * rad
                cx, cy = x, h - y
                ops.append(f'{r:.3f} {g:.3f} {b:.3f} rg {cx + rad:.2f} {cy:.2f} m '
                           f'{cx + rad:.2f} {cy + k:.2f} {cx + k:.2f} {cy + rad:.2f} {cx:.2f} {cy + rad:.2f} c '
                           f'{cx - k:.2f} {cy + rad:.2f} {cx - rad:.2f} {cy + k:.2f} {cx - rad:.2f} {cy:.2f} c '
                           f'{cx - rad:.2f} {cy - k:.2f} {cx - k:.2f} {cy - rad:.2f} {cx:.2f} {cy - rad:.2f} c '
                           f'{cx + k:.2f} {cy - rad:.2f} {cx + rad:.2f} {cy - k:.2f} {cx + rad:.2f} {cy:.2f} c f')
            else:
                pts = _shape_points(kind, x, y, rad)
                path = ' '.join(f'{a:.2f} {h - c:.2f} {"m" if i == 0 else "l"}' for i, (a, c) in enumerate(pts))
                ops.append(f'{r:.3f} {g:.3f} {b:.3f} rg {path} h f')
    content = '\n'.join(ops).encode('latin-1')
    objs = [b'<< /Type /Catalog /Pages 2 0 R >>', b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
            (f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w:.2f} {h:.2f}] /Contents 4 0 R '
             f'/Resources << /Font << /F1 5 0 R >> >> >>').encode('ascii'),
            b'<< /Length ' + str(len(content)).encode('ascii') + b' >>\nstream\n' + content + b'\nendstream',
            b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>']
    out = bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f'{i} 0 obj\n'.encode('ascii') + o + b'\nendobj\n'
    xref = len(out)
    out += f'xref\n0 {len(objs) + 1}\n0000000000 65535 f \n'.encode('ascii')
    for o in offs:
        out += f'{o:010d} 00000 n \n'.encode('ascii')
    out += f'trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode('ascii')
    return bytes(out)


def render_png(lay: dict, scale: float = 3.0) -> bytes:
    """The figure rasterised by OpenCV at scale px per point (anti-aliased lines, Hershey text)."""
    import cv2
    import numpy as np
    W, H = int(round(lay['w'] * scale)), int(round(lay['h'] * scale))
    img = np.full((H, W, 3), 255, np.uint8)

    def bgr(c):
        r, g, b = _rgb(c)
        return (b, g, r)

    def P(x, y):
        return int(round(x * scale)), int(round(y * scale))
    for item in lay['items']:
        kind = item[0]
        if kind == 'line':
            _, xa, ya, xb, yb, col, lw, dash = item
            th = max(1, int(round(lw * scale)))
            if dash:
                L = math.hypot(xb - xa, yb - ya)
                on, off = dash
                t = 0.0
                while t < L:
                    t1 = min(t + on, L)
                    cv2.line(img, P(xa + (xb - xa) * t / L, ya + (yb - ya) * t / L),
                             P(xa + (xb - xa) * t1 / L, ya + (yb - ya) * t1 / L), bgr(col), th, cv2.LINE_AA)
                    t = t1 + off
            else:
                cv2.line(img, P(xa, ya), P(xb, yb), bgr(col), th, cv2.LINE_AA)
        elif kind == 'text':
            _, x, y, s, size, anchor, col = item
            want = text_width(s, size) * scale                  # the width the layout gave it (Helvetica's)
            s = ascii_text(s)
            fsc = 0.72 * size * scale / 22.0
            th = max(1, int(round(size * scale / 14.0)))
            (tw, _th), _b = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fsc, th)
            if tw > want > 0:                                    # Hershey runs wider: fit it
                fsc *= want / tw
                (tw, _th), _b = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fsc, th)
            px, py = P(x, y)
            px = px - tw if anchor == 'end' else px - tw // 2 if anchor == 'middle' else px
            cv2.putText(img, s, (px, py), cv2.FONT_HERSHEY_SIMPLEX, fsc, bgr(col), th, cv2.LINE_AA)
        else:
            _, x, y, r, col, _tip = item
            if kind == 'circle':
                cv2.circle(img, P(x, y), max(1, int(round(r * scale))), bgr(col), -1, cv2.LINE_AA)
            else:
                pts = np.array([P(a, b) for a, b in _shape_points(kind, x, y, r)], np.int32)
                cv2.fillPoly(img, [pts], bgr(col), cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise PaperError('the PNG could not be encoded')
    return buf.tobytes()


# ================================================================================================ sources
def _ci(v, lo, hi):
    return (v, lo, hi)


def experiment_outputs(exp_dir: str) -> tuple:
    """(tables, figures, inputs) of an experiment's results.json: an ablation or a mixed sim/real experiment."""
    p = os.path.join(exp_dir, 'results.json')
    if not os.path.isfile(p):
        raise PaperError(f'{os.path.basename(exp_dir)} has no results.json yet (analyse it first)')
    res = PV.read_json(p)
    if res.get('schema') == 'falcon.ablation_results/1':
        return _ablation(res) + ({'results.json': p},)
    if res.get('schema') == 'falcon.mixed_results/1':
        return _mixed(res) + ({'results.json': p},)
    raise PaperError(f'{res.get("schema")!r} results have no paper outputs (ablations and mixed experiments do)')


def _ablation(res: dict) -> tuple:
    h = res['hypothesis']
    m = len(res['contrasts'])
    d = res['design']
    ci = 95
    sim_note = ('simultaneous intervals at the Bonferroni level over the ' + str(m) + ' contrasts' if m > 1 else '')
    corr = res['method']['correction']
    cap = (f'{h["text"]}: {res["verdict"]}. {res["metric"]}, {res["replicates"]} replicates per cell; paired '
           f'bootstrap over units (expanded percentile, {ci} %)' + (f'; {sim_note}' if sim_note else '') +
           f'; p-values two-sided, {corr}-adjusted.')
    notes = list(res.get('warnings') or ())
    rows = []
    for c in res['contrasts']:
        rows.append({'label': c['label'], 'n': c['n_pairs'], 'a': c['mean_a'], 'b': c['mean_b'],
                     'diff': _ci(c['diff'], c['ci_low'], c['ci_high']),
                     'sim': _ci(c['diff'], c['sim_ci_low'], c['sim_ci_high']),
                     'dz': _ci(c['dz'], c['dz_ci_low'], c['dz_ci_high']), 'p': c['p_adjusted'],
                     'verdict': c['verdict']})
    cols = [column('label', 'Contrast', 'text'), column('n', 'n', 'int'), column('a', 'A'), column('b', 'B'),
            column('diff', f'Δ [{ci} % CI]', 'ci', signed=True)]
    if m > 1:
        cols.append(column('sim', 'Δ [simultaneous CI]', 'ci', signed=True))
    cols += [column('dz', 'dz [CI]', 'ci', 2, True), column('p', f'p ({corr})', 'p'),
             column('verdict', 'Verdict', 'text')]
    t1 = table('contrasts', cap, cols, rows, notes, f'tab:{res["experiment"]}-contrasts')
    names = [f['name'] for f in res['factors']]
    crow = []
    for c in res['cells']:
        r = {n: c['cell'][n] for n in names}
        r.update(n=c['n'], mean=_ci(c['mean'], c['ci_low'], c['ci_high']), sd=c['sd'])
        crow.append(r)
    t2 = table('cells', f'{res["metric"]} per cell: the mean over replicates with its {ci} % interval (expanded '
                        'percentile bootstrap over replicates).',
               [column(n, n, 'text') for n in names] + [column('n', 'n', 'int'),
                                                       column('mean', f'{res["metric"]} [{ci} % CI]', 'ci'),
                                                       column('sd', 'sd')], crow, (),
               f'tab:{res["experiment"]}-cells')
    f1 = figure('contrasts', [{'label': c['label'], 'points': [{'value': c['diff'], 'lo': c['ci_low'],
                                                                'hi': c['ci_high'], 'lo2': c['sim_ci_low'] if m > 1
                                                                else None, 'hi2': c['sim_ci_high'] if m > 1
                                                                else None}]} for c in res['contrasts']],
                f'Δ {res["metric"]} (B − A)', None, True,
                f'Each contrast\'s difference with its {ci} % interval (thick)' +
                (' and the simultaneous interval (thin)' if m > 1 else '') + '.')
    sf = res['contrasts'][0]['factor'] if res['contrasts'] else None
    fac = {f['name']: f for f in res['factors']}
    if sf and len(fac[sf]['levels']) <= len(SERIES):
        others = [n for n in names if n != sf]
        groups = {}
        for c in res['cells']:
            key = tuple(c['cell'][n] for n in others)
            groups.setdefault(key, {})[c['cell'][sf]] = c
        frows = [{'label': ', '.join(f'{n}={v}' for n, v in zip(others, key)) or res['metric'],
                  'points': [({'value': g[lv]['mean'], 'lo': g[lv]['ci_low'], 'hi': g[lv]['ci_high']}
                              if lv in g else None) for lv in fac[sf]['levels']]} for key, g in groups.items()]
        f2 = figure('cells', frows, res['metric'], [f'{sf}={lv}' for lv in fac[sf]['levels']], False,
                    f'{res["metric"]} per cell, by {sf}: mean with its {ci} % interval.')
    else:
        f2 = figure('cells', [{'label': ', '.join(f'{k}={v}' for k, v in c['cell'].items()),
                               'points': [{'value': c['mean'], 'lo': c['ci_low'], 'hi': c['ci_high']}]}
                              for c in res['cells']], res['metric'], None, False,
                    f'{res["metric"]} per cell: mean with its {ci} % interval.')
    if d.get('dz') is not None:
        t1['notes'].append(f'Planned: dz {fmt(d["dz"], 2)} at alpha {fmt(d["alpha_per_contrast"], 4)} per contrast '
                           f'and power {fmt(d["target_power"], 2)}, {d["replicates"]} replicates.')
    return [t1, t2], [f1, f2]


def _mixed(res: dict) -> tuple:
    rows = []
    for a in res['arms']:
        v = a.get('vs_real_only') or {}
        ap = v.get('AP') or {}
        rows.append({'arm': a['arm'], 'real': a['n_real'], 'synthetic': a['n_synthetic'],
                     'ap': _ci(a['AP']['value'], a['AP']['ci_low'], a['AP']['ci_high']),
                     'sd': a['seed_spread']['AP']['sd'],
                     'diff': _ci(ap.get('diff'), ap.get('ci_low'), ap.get('ci_high')) if ap else None,
                     'verdict': ap.get('verdict', 'the base arm' if not v else '')})
    t = res['test']
    cap = (f'Real only against real + synthetic at {res["budget_images"]} training images per run, {len(res["seeds"])} '
           f'seeds; AP on {t["images"]} real test images, 95 % bootstrap intervals over test {t["unit"]}s; the '
           f'difference from real only is paired.')
    t1 = table('arms', cap, [column('arm', 'Synthetic share', 'text'), column('real', 'Real', 'int'),
                             column('synthetic', 'Synthetic', 'int'), column('ap', 'AP [95 % CI]', 'ci'),
                             column('sd', 'Seed sd'), column('diff', 'Δ vs real only [95 % CI]', 'ci', signed=True),
                             column('verdict', 'Verdict', 'text')], rows, res.get('warnings') or (),
               f'tab:{res["experiment"]}-arms')
    f1 = figure('arms', [{'label': a['arm'], 'points': [{'value': a['AP']['value'], 'lo': a['AP']['ci_low'],
                                                         'hi': a['AP']['ci_high']}]} for a in res['arms']], 'AP',
                None, False, 'AP per synthetic share with its 95 % interval.')
    return [t1], [f1]


def evaluation_outputs(ev_dir: str) -> tuple:
    p = os.path.join(ev_dir, 'evaluation.json')
    if not os.path.isfile(p):
        raise PaperError(f'{os.path.basename(ev_dir)} has no evaluation.json')
    ev = PV.read_json(p)
    sc = ev['scores']
    rows = [{'metric': r['metric'], 'value': _ci(r['value'], r['ci_low'], r['ci_high']), 'n': r.get('n')}
            for r in sc['table']]
    name = (ev.get('model') or {}).get('name') or (ev.get('model') or {}).get('id') or 'the model'
    unit = 'drives' if sc.get('unit') == 'drive' else 'frames'
    cap = (f'{name} on {ev.get("n_frames")} frames of {len(ev.get("runs") or ())} drive(s): 95 % bootstrap intervals '
           f'over {unit} ({sc.get("n_boot")} resamples).')
    notes = [sc['note']] if sc.get('note') else []
    if ev.get('synthetic_model'):
        notes.append('The synthetic detector is not a model: these numbers test the pipeline only.')
    t1 = table('scores', cap, [column('metric', 'Metric', 'text'), column('value', 'Value [95 % CI]', 'ci'),
                               column('n', f'n ({unit})', 'int')], rows, notes, 'tab:evaluation')
    head = [r for r in sc['table'] if '[' not in r['metric']]
    f1 = figure('scores', [{'label': r['metric'], 'points': [{'value': r['value'], 'lo': r['ci_low'],
                                                              'hi': r['ci_high']}]} for r in head], 'value', None,
                False, 'Headline and per-class numbers with their 95 % intervals.')
    return [t1], [f1], {'evaluation.json': p}


def comparison_outputs(cmp_dir: str) -> tuple:
    p = os.path.join(cmp_dir, 'comparison.json')
    if not os.path.isfile(p):
        raise PaperError(f'{os.path.basename(cmp_dir)} has no comparison.json')
    c = PV.read_json(p)
    an = ((c.get('a') or {}).get('model') or {}).get('name') or 'A'
    bn = ((c.get('b') or {}).get('model') or {}).get('name') or 'B'
    rows = [{'metric': r['metric'], 'a': r['a'], 'b': r['b'], 'diff': _ci(r['diff'], r['ci_low'], r['ci_high']),
             'p': r.get('p_b_gt_a')} for r in c['rows']]
    cap = (f'{bn} (B) against {an} (A) on the same {c.get("n_frames")} frames: paired bootstrap over '
           f'{c.get("unit")}s ({c.get("n_units")} units, {c.get("n_boot")} resamples), 95 % intervals of B − A.')
    t1 = table('comparison', cap, [column('metric', 'Metric', 'text'), column('a', 'A'), column('b', 'B'),
                                   column('diff', 'B − A [95 % CI]', 'ci', signed=True),
                                   column('p', 'P(B > A)', 'num', 3)], rows, (), 'tab:comparison')
    head = [r for r in c['rows'] if '[' not in r['metric']]
    f1 = figure('comparison', [{'label': r['metric'], 'points': [{'value': r['diff'], 'lo': r['ci_low'],
                                                                  'hi': r['ci_high']}]} for r in head], 'B − A',
                None, True, 'Each number\'s paired difference with its 95 % interval.')
    return [t1], [f1], {'comparison.json': p}


def _private(path: str) -> bool:
    from ..assistant.privacy import private_reason
    return bool(private_reason(path))


def runs_outputs(run_dirs: list) -> tuple:
    """core.runcompare's table of two to four runs (no interval: single runs). A run from the dashcam footage is
    labelled 'run k', never by its name."""
    from . import runcompare as RC
    from . import runindex as RI
    if not 2 <= len(run_dirs) <= 4:
        raise PaperError('give two to four runs')
    entries, labels, inputs = [], [], {}
    for k, d in enumerate(run_dirs, 1):
        e = RI.summarise(d)
        if e is None:
            raise PaperError(f'{os.path.basename(os.path.normpath(d))} has no readable run.json')
        entries.append(e)
        labels.append(f'run {k}' if _private(d) else str(e.get('name') or e.get('id')))
        inputs[f'run{k}/run.json'] = os.path.join(d, 'run.json')
    rows = RC.table(entries)
    keys = [f'r{k}' for k in range(len(entries))]
    out = []
    for r in rows:
        row = {'label': r['label'] + (f' ({r["unit"]})' if r['unit'] else '')}
        nd = next(x[3] for x in RC.ROWS if x[0] == r['key'])
        for key, c in zip(keys, r['cells']):
            row[key] = c['value']
        out.append((row, nd))
    cols = [column('label', 'Quantity', 'text')] + [column(k, lab, 'num', 3) for k, lab in zip(keys, labels)]
    t1 = table('runs', f'{len(entries)} runs side by side, as each recorded them (single runs: no interval).',
               cols, [r for r, _ in out], ('An empty cell: the run did not record it.',), 'tab:runs')
    return [t1], [], inputs


# ================================================================================================ the data card
def _conditions(e: dict) -> dict:
    return {'world': e.get('world') or e.get('map'), 'weather': e.get('weather'), 'weather_kind': e.get('weather_kind'),
            'time_of_day': e.get('tod'), 'rain': e.get('rain'), 'fog': e.get('fog')}


def datacard(export_dir: str) -> dict:
    """The data card of an export folder (falcon_export.json): DATACARD_SCHEMA (the module docstring)."""
    from . import datasets as DS
    from . import runindex as RI
    mp = os.path.join(export_dir, 'falcon_export.json')
    if not os.path.isfile(mp):
        raise PaperError(f'{os.path.basename(os.path.normpath(export_dir))} is not an export (no falcon_export.json)')
    man = PV.read_json(mp)
    runs, objects, cond, private_n, carla, apps, rigs, workers = [], {}, {}, 0, set(), set(), {}, {}
    alias = {}
    for k, r in enumerate(man.get('runs') or (), 1):
        d = r.get('dir') or ''
        priv = bool(d) and os.path.isdir(d) and _private(d)
        alias[r.get('key')] = f'run {k}' if priv else r.get('key')
        private_n += priv
        e = RI.summarise(d) if d and os.path.isdir(d) else None
        cov = DS.coverage_of_run(d) if d and os.path.isdir(d) else {'objects': {}, 'frames': 0}
        for c, n in (cov.get('objects') or {}).items():
            objects[c] = objects.get(c, 0) + n
        rj = {}
        try:
            rj = PV.read_json(os.path.join(d, 'run.json'))
        except (OSError, ValueError):
            pass
        host = (rj.get('provenance') or {}).get('host') or {}
        if host.get('carla_server'):
            carla.add(host['carla_server'])
        if (rj.get('provenance') or {}).get('app_version'):
            apps.add(rj['provenance']['app_version'])
        workers[rj.get('worker') or 'unknown'] = workers.get(rj.get('worker') or 'unknown', 0) + 1
        st = (rj.get('provenance') or {}).get('settings') or {}
        rig = st.get('rig') or {}
        if rig:
            rigs[PV.sha256_json(rig)[:12]] = _rig_text(rig)
        c = _conditions(e or {})
        if priv:
            c['world'] = 'withheld (dashcam footage)'
        key = (c['world'] or '?', c['weather'] or '?', c['time_of_day'] or '?')
        cell = cond.setdefault(key, {'runs': 0, 'frames': 0})
        cell['runs'] += 1
        cell['frames'] += int(r.get('steps') or 0)
        runs.append({'run': f'run {k}' if priv else r.get('key'), 'split': r.get('split'), 'steps': r.get('steps'),
                     'fps': r.get('fps'), 'seed': r.get('seed'),
                     'world': c['world'], 'weather': c['weather'], 'time_of_day': c['time_of_day'],
                     'run_json_sha256': r.get('run_json_sha256'), 'config_sha256': r.get('config_sha256'),
                     'private': priv, 'worker': rj.get('worker'),
                     'faults': bool(r.get('faults_applied'))})
    tods = {k[2] for k in cond}
    kinds = {r['weather'] for r in runs}
    gaps = []
    for t in ('day', 'twilight', 'night'):
        if t not in tods:
            gaps.append(f'no {t} frames')
    if not any((r['weather'] or '').lower().find('rain') >= 0 for r in runs):
        gaps.append('no rain preset among the runs')
    thin = sorted(c for c, n in objects.items() if n < FEW_OBJECTS)
    if thin:
        gaps.append(f'fewer than {FEW_OBJECTS} objects of: {", ".join(thin)}')
    gaps += [f'left out of the export: {x["what"]} — {x["why"]}' for x in man.get('left_out') or ()]
    gaps += [f'heuristic: {x["what"]} — {x["how"]}' for x in man.get('heuristic') or ()]
    if len(runs) < 5:
        gaps.append(f'{len(runs)} run(s): intervals over drives computed on this set rest on few units')
    lic = ['Rendered with CARLA (its code MIT-licensed, its own assets CC-BY, as CARLA\'s licence states); Falcon '
           '(GPL-3.0-or-later) claims nothing over the data it writes. The licence of this set is its publisher\'s '
           'to state here.']
    if workers.get('translate'):
        lic.append('Some runs were translated by a learned translator: they inherit the terms of the real data it '
                   'was trained on (docs/translation.md).')
    card = {'schema': DATACARD_SCHEMA, 'format': man.get('format'), 'export_created': None if private_n else
            man.get('created'),
            'composition': {'runs': len(runs), 'splits': {s: [alias.get(x, x) for x in keys] for s, keys in
                                                          (man.get('splits') or {}).items()},
                            'frames_written': man.get('frames_written'),
                            'counts': man.get('counts'), 'objects_per_class': dict(sorted(objects.items())),
                            'objects_note': 'counted in the runs\' recorded object files (every camera and step), '
                                            'which can be more than the frames the export kept (its every and '
                                            'cameras options)'},
            'collection': {'simulator': sorted(carla) or ['CARLA (version not recorded)'],
                           'falcon_versions': sorted(apps), 'workers': dict(sorted(workers.items())),
                           'rigs': [rigs[k] for k in sorted(rigs)],
                           'process': 'drives recorded by Falcon\'s drive worker in synchronous mode: every sensor '
                                      'and label rendered for the same simulation step (docs/dataset-format.md)'},
            'labels': {'source': 'the simulator\'s ground truth (semantic and instance cameras, actors\' boxes), not '
                                 'human annotation', 'format': man.get('format'),
                       'load': man.get('load'), 'coordinates': man.get('coordinates')},
            'conditions': [{'world': k[0], 'weather': k[1], 'time_of_day': k[2], **v}
                           for k, v in sorted(cond.items())],
            'weather_presets': sorted(x for x in kinds if x),
            'known_gaps': gaps + ['synthetic images: a sim-to-real gap is expected (measure it: docs/sim-to-real.md)'],
            'licence': lic,
            'provenance': {'export_manifest_sha256': PV.sha256_file(mp), 'runs': runs},
            'privacy': (f'{private_n} run(s) come from the user\'s own dashcam footage: their names, places, routes, '
                        'dates and folders are withheld') if private_n else None}
    return card


def _rig_text(rig: dict) -> str:
    parts = []
    for s in rig.get('sensors') or ():
        t = s.get('type', '')
        if t.startswith('camera'):
            parts.append(f'{t} {s.get("width")}×{s.get("height")}, fov {s.get("fov")}°'
                         + (f', labels {"/".join(s.get("labels") or ())}' if s.get('labels') else ''))
        else:
            parts.append(t)
    return f'{rig.get("vehicle") or "vehicle"}: ' + '; '.join(parts)


def datacard_markdown(card: dict) -> str:
    c, col, lab = card['composition'], card['collection'], card['labels']
    L = [f'# Data card: {card["format"]} export of {c["runs"]} run(s)', '',
         'Written by Falcon Autonomy (core.paper.datacard) after Datasheets for Datasets (Gebru et al. 2021): the '
         'sections that apply to a rendered set. Every number is read from the export and its runs.', '']
    if card.get('privacy'):
        L += [f'> {card["privacy"]}.', '']
    L += ['## Composition', '', f'- runs: {c["runs"]}; splits: ' + ', '.join(
        f'{k} {len(v)}' for k, v in sorted((c['splits'] or {}).items())),
          '- frames written: ' + ', '.join(f'{k} {v}' for k, v in sorted((c['frames_written'] or {}).items())),
          '- export counts: ' + (', '.join(f'{k} {v}' for k, v in sorted((c['counts'] or {}).items())) or '—'), '',
          '| class | objects |', '|---|---:|']
    L += [f'| {k} | {v} |' for k, v in c['objects_per_class'].items()] or ['| — | 0 |']
    L += ['', f'Objects are {c["objects_note"]}.', '', '## Collection process', '',
          f'- simulator: {", ".join(col["simulator"])}',
          f'- Falcon versions: {", ".join(col["falcon_versions"]) or "—"}',
          '- workers: ' + ', '.join(f'{k} ({v})' for k, v in col['workers'].items()),
          f'- process: {col["process"]}']
    L += [f'- rig: {r}' for r in col['rigs']]
    L += ['', '## Labels', '', f'- source: {lab["source"]}', f'- format: {lab["format"]}', '',
          '## Conditions covered', '', '| world | weather | time of day | runs | frames |', '|---|---|---|---:|---:|']
    L += [f'| {x["world"]} | {x["weather"]} | {x["time_of_day"]} | {x["runs"]} | {x["frames"]} |'
          for x in card['conditions']]
    L += ['', '## Known gaps', ''] + [f'- {g}' for g in card['known_gaps']]
    L += ['', '## Licence', ''] + [f'- {x}' for x in card['licence']]
    L += ['', '## Provenance', '', f'- export manifest SHA-256: `{card["provenance"]["export_manifest_sha256"]}`', '',
          '| run | split | steps | seed | run.json SHA-256 |', '|---|---|---:|---:|---|']
    L += [f'| {r["run"]} | {r["split"] or "all"} | {r["steps"]} | {r["seed"]} | '
          f'`{(r["run_json_sha256"] or "")[:16]}…` |' for r in card['provenance']['runs']]
    return '\n'.join(L) + '\n'


def write_datacard(export_dir: str) -> dict:
    """DATACARD.md and datacard.json into the export folder; returns the card."""
    card = datacard(export_dir)
    with open(os.path.join(export_dir, 'DATACARD.md'), 'w', encoding='utf-8') as fh:
        fh.write(datacard_markdown(card))
    with open(os.path.join(export_dir, 'datacard.json'), 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(card, indent=1, ensure_ascii=False, sort_keys=True) + '\n')
    return card


# ================================================================================================ export
def sources(ws: str) -> dict:
    """What a workspace holds that has paper outputs: {kind: [(label, path)]} (runs are chosen on Data ▸ Compare)."""
    from .reports import sources as report_sources

    def ls(*parts):
        d = os.path.join(ws, *parts)
        return sorted(os.listdir(d), reverse=True) if os.path.isdir(d) else []
    out = {k: [] for k in KINDS if k != 'runs'}
    for e in ls('experiments'):
        p = os.path.join(ws, 'experiments', e, 'results.json')
        try:
            if os.path.isfile(p) and PV.read_json(p).get('schema') in ('falcon.ablation_results/1',
                                                                       'falcon.mixed_results/1'):
                out['experiment'].append((e, os.path.dirname(p)))
        except (OSError, ValueError):
            continue
    out['evaluation'] = list(report_sources(ws).get('evaluation') or ())
    out['comparison'] = [(c, os.path.join(ws, 'comparisons', c)) for c in ls('comparisons')
                         if os.path.isfile(os.path.join(ws, 'comparisons', c, 'comparison.json'))]
    out['datacard'] = [(x, os.path.join(ws, 'exports', x)) for x in ls('exports')
                       if os.path.isfile(os.path.join(ws, 'exports', x, 'falcon_export.json'))]
    return out


def export(kind: str, source, out_dir: str, formats=FIGURE_FORMATS) -> dict:
    """Write the outputs of a source (KINDS) into out_dir; returns paper.json's document."""
    if kind not in KINDS:
        raise PaperError(f'kind is one of {", ".join(KINDS)}')
    bad = [f for f in formats if f not in FIGURE_FORMATS]
    if bad:
        raise PaperError(f'figure formats are {", ".join(FIGURE_FORMATS)}')
    if kind == 'datacard':
        src = source if isinstance(source, str) else source[0]
        card = datacard(src)
        os.makedirs(out_dir, exist_ok=True)
        files = {'DATACARD.md': datacard_markdown(card).encode('utf-8'),
                 'datacard.json': (json.dumps(card, indent=1, ensure_ascii=False, sort_keys=True) + '\n').encode()}
        inputs = {'falcon_export.json': os.path.join(src, 'falcon_export.json')}
    else:
        if kind == 'experiment':
            tables, figs, inputs = experiment_outputs(source)
        elif kind == 'evaluation':
            tables, figs, inputs = evaluation_outputs(source)
        elif kind == 'comparison':
            tables, figs, inputs = comparison_outputs(source)
        else:
            tables, figs, inputs = runs_outputs(list(source))
        sha = PV.sha256_json({k: PV.sha256_file(p) for k, p in sorted(inputs.items())})
        label = os.path.basename(os.path.normpath(source)) if isinstance(source, str) else 'runs'
        if kind == 'runs' or (isinstance(source, str) and _private(source)):
            label = kind
        files = {}
        for t in tables:
            files[f'{t["name"]}.tex'] = to_latex(t, f'{label} (inputs SHA-256 {sha[:16]})').encode('utf-8')
            files[f'{t["name"]}.csv'] = to_csv(t).encode('utf-8')
        for f in figs:
            lay = layout(f)
            if 'svg' in formats:
                files[f'{f["name"]}.svg'] = render_svg(lay).encode('utf-8')
            if 'pdf' in formats:
                files[f'{f["name"]}.pdf'] = render_pdf(lay)
            if 'png' in formats:
                files[f'{f["name"]}.png'] = render_png(lay)
        os.makedirs(out_dir, exist_ok=True)
    for name, data in files.items():
        with open(os.path.join(out_dir, name), 'wb') as fh:
            fh.write(data)
    doc = {'schema': PAPER_SCHEMA, 'kind': kind,
           'inputs': {k: PV.sha256_file(p) for k, p in sorted(inputs.items())},
           'files': {k: PV.sha256_bytes(v) for k, v in sorted(files.items())},
           'captions': {} if kind == 'datacard' else {**{t['name']: t['caption'] for t in tables},
                                                      **{f['name']: f['caption'] for f in figs}},
           'no_figure': 'single runs carry no interval: tables only' if kind == 'runs' else None}
    with open(os.path.join(out_dir, 'paper.json'), 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True) + '\n')
    return doc


def main_paper(argv) -> int:
    """`falcon paper experiment|evaluation|comparison|datacard SOURCE --out DIR` or `falcon paper runs RUN RUN… --out
    DIR`: 0 written, 2 failed."""
    import argparse
    ap = argparse.ArgumentParser(prog='falcon paper', description='outputs ready for papers: LaTeX (booktabs) and CSV '
                                                                  'tables, PDF/PNG/SVG figures with their intervals, '
                                                                  'data cards (docs/experiments.md)')
    ap.add_argument('kind', choices=KINDS)
    ap.add_argument('source', nargs='+', help='the experiment, evaluation, comparison or export folder; runs: 2–4 '
                                              'run folders')
    ap.add_argument('--out', '-o', default='', help='the output folder (datacard: default the export folder)')
    ap.add_argument('--formats', default='pdf,png,svg', help='figure formats (default pdf,png,svg)')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    try:
        if a.kind != 'runs' and len(a.source) != 1:
            raise PaperError(f'{a.kind} takes one source folder')
        out = a.out or (a.source[0] if a.kind == 'datacard' else '')
        if not out:
            raise PaperError('--out DIR is needed')
        fm = tuple(x.strip() for x in a.formats.split(',') if x.strip())
        doc = export(a.kind, a.source if a.kind == 'runs' else a.source[0], out, fm)
    except (PaperError, OSError, ValueError, KeyError) as e:
        print(f'failed: {e}', flush=True)
        return 2
    for name, sha in doc['files'].items():
        print(f'{os.path.join(out, name)}  {sha[:16]}', flush=True)
    if doc.get('no_figure'):
        print(f'note: {doc["no_figure"]}', flush=True)
    return 0
