"""A HIL session's report: report.html (self-contained: the latency waterfall as inline SVG, the stage table, the
per-frame latency, the clock offset, drops and deadlines, what ran where) and waterfall.png (OpenCV when importable).
Standard library, numpy; cv2 optional. Run by the session worker at its end and by `falcon hil report RUN`.

The waterfall is a cascade: one row per stage, in the order a frame goes through them; each bar starts where the
previous stage's median ended and is as long as its own median, so the rows add up to the median frame, and a thin
whisker reaches its 95th percentile. The last row is the end-to-end latency itself, with ticks at its median, p95 and
p99 (the sum of stage medians is not the median of the sum; both are shown). Bars are coloured by where the stage runs
— this machine, the network, the device — and each row is labelled, so colour is never the only key. One-way network
stages need the clock-offset estimate: when it is not trusted (core.hil.TRUST_NS) they are hatched and labelled so.
"""
from __future__ import annotations

import html
import json
import os

PLACE = {'host': ('This machine', '--host'), 'cross': ('Network (one-way, needs the clock offset)', '--net'),
         'device': ('The device', '--dev')}
PNG_RGB = {'host': (214, 120, 42), 'cross': (52, 104, 235), 'device': (122, 175, 27)}     # BGR of the light theme


def load_summary(run_dir: str) -> tuple:
    """(summary, run.json) of a session: run.json's when the worker wrote one, else recomputed from the files."""
    from ..core.hil import summarise_run
    from ..core.provenance import read_json
    run = {}
    try:
        run = read_json(os.path.join(run_dir, 'run.json'))
    except (OSError, ValueError):
        pass
    s = run.get('hil')
    if not s or 'waterfall' not in s:
        s = summarise_run(run_dir)
    return s, run


def _f(x, nd=2, unit=' ms'):
    return '—' if x is None else f'{x:.{nd}f}{unit}'


def waterfall_svg(s: dict, width: int = 860) -> str:
    rows = s.get('waterfall') or []
    e2e = (s.get('measures') or {}).get('e2e') or {}
    trusted = s.get('one_way_trusted')
    ends = [r['start_ms'] + max(r['p95_ms'] or 0, r['median_ms']) for r in rows] + [e2e.get('p99') or 0]
    span = max(ends + [1.0]) * 1.05
    lw, rw, rh, top = 230, 150, 26, 28
    pw = width - lw - rw
    h = top + rh * (len(rows) + 1) + 44
    X = lambda ms: lw + pw * ms / span                    # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img" aria-label="Latency waterfall: the median frame '
           f'stage by stage" xmlns="http://www.w3.org/2000/svg">',
           '<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
           '<rect width="6" height="6" fill="var(--net)" opacity="0.35"/><line x1="0" y1="0" x2="0" y2="6" '
           'stroke="var(--net)" stroke-width="2"/></pattern></defs>']
    step = 5 if span < 60 else 10 if span < 150 else 25 if span < 400 else 50
    t = 0
    while t <= span:
        x = X(t)
        out.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{h - 30}" class="grid"/>'
                   f'<text x="{x:.1f}" y="{h - 14}" class="tick" text-anchor="middle">{t:g}</text>')
        t += step
    out.append(f'<text x="{lw + pw / 2:.0f}" y="{h - 1}" class="tick" text-anchor="middle">milliseconds from the '
               'start of the step</text>')
    period = s.get('period_ms')
    if period:
        for k in range(1, int(span // period) + 1):
            x = X(k * period)
            out.append(f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{h - 30}" class="period"/>'
                       f'<text x="{x + 3:.1f}" y="{top - 12}" class="tick">step {k}</text>')
    for i, r in enumerate(rows):
        y = top + i * rh
        x0, x1 = X(r['start_ms']), X(r['start_ms'] + r['median_ms'])
        oneway = r['clock'] == 'cross'
        fill = 'url(#hatch)' if oneway and not trusted else f'var({PLACE[r["clock"]][1]})'
        tip = (f'{r["label"]}: median {r["median_ms"]:.3f} ms, p95 {_f(r["p95_ms"], 3)}'
               + (' (one-way: needs the clock offset; not trusted)' if oneway and not trusted else ''))
        out.append(f'<text x="{lw - 10}" y="{y + rh / 2 + 4:.0f}" class="lab" text-anchor="end">'
                   f'{html.escape(r["label"])}</text>')
        out.append(f'<g><title>{html.escape(tip)}</title>'
                   f'<rect x="{x0:.1f}" y="{y + 5}" width="{max(x1 - x0, 2.0):.1f}" height="{rh - 10}" rx="3" '
                   f'fill="{fill}"/></g>')
        if r.get('p95_ms') and r['p95_ms'] > r['median_ms']:
            xp = X(r['start_ms'] + r['p95_ms'])
            out.append(f'<line x1="{x1:.1f}" y1="{y + rh / 2}" x2="{xp:.1f}" y2="{y + rh / 2}" class="whisk"/>'
                       f'<line x1="{xp:.1f}" y1="{y + 8}" x2="{xp:.1f}" y2="{y + rh - 8}" class="whisk"/>')
        out.append(f'<text x="{width - rw + 8}" y="{y + rh / 2 + 4:.0f}" class="val">{r["median_ms"]:.2f}'
                   f' · p95 {_f(r["p95_ms"], 2, "")} ms</text>')
    y = top + len(rows) * rh
    if e2e.get('median') is not None:
        out.append(f'<text x="{lw - 10}" y="{y + rh / 2 + 4:.0f}" class="lab strong" text-anchor="end">'
                   'End to end (sensor to actuation)</text>')
        out.append(f'<g><title>End to end: median {e2e["median"]:.2f} ms, p95 {e2e["p95"]:.2f} ms, p99 '
                   f'{e2e["p99"]:.2f} ms over {e2e["n"]} frames</title><rect x="{lw}" y="{y + 5}" '
                   f'width="{max(X(e2e["median"]) - lw, 2):.1f}" height="{rh - 10}" rx="3" class="total"/></g>')
        x95, x99 = X(e2e['p95']), X(e2e['p99'])
        for xk in (x95, x99):
            out.append(f'<line x1="{xk:.1f}" y1="{y + 3}" x2="{xk:.1f}" y2="{y + rh - 3}" class="mark"/>')
        if x99 - x95 < 30:                                # too close for two labels: one under both
            out.append(f'<text x="{x99 + 3:.1f}" y="{y + rh + 9}" class="tick">p95, p99</text>')
        else:
            out += [f'<text x="{xk + 3:.1f}" y="{y + rh + 9}" class="tick">{k}</text>'
                    for xk, k in ((x95, 'p95'), (x99, 'p99'))]
        out.append(f'<text x="{width - rw + 8}" y="{y + rh / 2 + 4:.0f}" class="val strong">{e2e["median"]:.2f}'
                   f' · p95 {e2e["p95"]:.2f} ms</text>')
    out.append('</svg>')
    return ''.join(out)


def series_svg(records: list, period_ms: float, width: int = 860, height: int = 200) -> str:
    """Per frame: step start to control at the simulation (continuous), with the step period as a reference."""
    pts = [(r['frame'], (r['ctrl_worker'] - r['tick_start']) / 1e6) for r in records
           if r.get('ctrl_worker') is not None and r.get('tick_start') is not None]
    if len(pts) < 2:
        return '<p class="muted">No frame had its control back: nothing to plot.</p>'
    f0, f1 = pts[0][0], pts[-1][0]
    ymax = max(max(p[1] for p in pts), period_ms or 0) * 1.1
    l, r_, t, b = 50, 12, 10, 26
    X = lambda f: l + (width - l - r_) * (f - f0) / max(f1 - f0, 1)      # noqa: E731
    Y = lambda v: t + (height - t - b) * (1 - v / ymax)                  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Latency per frame" '
           'xmlns="http://www.w3.org/2000/svg">']
    for k in range(5):
        v = ymax * k / 4
        out.append(f'<line x1="{l}" x2="{width - r_}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                   f'<text x="{l - 6}" y="{Y(v) + 4:.1f}" class="tick" text-anchor="end">{v:.0f}</text>')
    if period_ms:
        out.append(f'<line x1="{l}" x2="{width - r_}" y1="{Y(period_ms):.1f}" y2="{Y(period_ms):.1f}" class="period"/>'
                   f'<text x="{width - r_}" y="{Y(period_ms) - 4:.1f}" class="tick" text-anchor="end">deadline: the '
                   f'next step ({period_ms:g} ms)</text>')
    path = ' '.join(f'{"M" if i == 0 else "L"}{X(f):.1f},{Y(v):.1f}' for i, (f, v) in enumerate(pts))
    out.append(f'<path d="{path}" class="line"/>')
    out.append(f'<text x="{l}" y="{height - 6}" class="tick">frame {f0}</text><text x="{width - r_}" '
               f'y="{height - 6}" class="tick" text-anchor="end">frame {f1}</text></svg>')
    return ''.join(out)


CSS = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--muted:#8a8984;--grid:#e6e5e1;--host:#2a78d6;--net:#eb6834;
--dev:#1baf7a;--total:#52514e;--line:#2a78d6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;
--muted:#8f8e86;--grid:#33332f;--host:#3987e5;--net:#d95926;--dev:#199e70;--total:#c3c2b7;--line:#3987e5}}
:root[data-theme="dark"]{--bg:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--muted:#8f8e86;--grid:#33332f;--host:#3987e5;
--net:#d95926;--dev:#199e70;--total:#c3c2b7;--line:#3987e5}
body{background:var(--bg);color:var(--fg);font:14px/1.45 "IBM Plex Sans",system-ui,sans-serif;margin:0 auto;
max-width:960px;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}.muted{color:var(--fg2)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--grid)}td.n,th.n{text-align:right}
.tiles{display:flex;flex-wrap:wrap;gap:12px}.tile{border:1px solid var(--grid);border-radius:8px;padding:10px 14px;
min-width:150px}.tile b{display:block;font-size:22px}.tile span{color:var(--fg2);font-size:12px}
svg text{fill:var(--fg2);font-size:12px}svg .lab{fill:var(--fg)}svg .strong{font-weight:600;fill:var(--fg)}
svg .tick{fill:var(--muted);font-size:11px}svg .grid{stroke:var(--grid);stroke-width:1}
svg .period{stroke:var(--muted);stroke-dasharray:4 4;stroke-width:1}svg .whisk{stroke:var(--fg2);stroke-width:1.5}
svg .mark{stroke:var(--fg);stroke-width:2}svg .total{fill:var(--total)}svg .line{fill:none;stroke:var(--line);
stroke-width:2}.key{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--fg2);margin:6px 0}
.key i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-2px}
code{font-family:"IBM Plex Mono",monospace;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}
td code{overflow-wrap:anywhere}
"""


def load_note(ses: dict) -> str:
    ld = ses.get('host_load') or {}
    if not ld.get('end'):
        return ''
    st = ld.get('start') or ['?']
    return (f'This machine\'s load average (1 min) was {st[0]} at the start and {ld["end"][0]} at the end, on '
            f'{ld.get("cpus")} CPUs: every number here shares the machine with whatever else ran.')


def report_html(s: dict, run: dict, records: list) -> str:
    ses = s.get('session') or {}
    dev = ses.get('device') or {}
    env = dev.get('env') or {}
    cam, lid = ses.get('camera') or {}, ses.get('lidar') or {}
    m = s.get('measures') or {}
    dl = s.get('deadline') or {}
    est = s.get('clock_offset') or {}
    perf = s.get('performance') or run.get('performance') or {}
    local = dev.get('mode') == 'local'
    where = ('a <b>simulated device</b>: the device agent running on this machine (127.0.0.1), so the network '
             'stages are loopback, not a LAN' if local else
             f'<b>{html.escape(env.get("model") or "the remote device")}</b> on the network')
    e = lambda k: m.get(k) or {}                          # noqa: E731
    tiles = [('End to end, median', _f(e('e2e').get('median'))), ('End to end, p95', _f(e('e2e').get('p95'))),
             ('Step start to control, median', _f(e('processing').get('median'))),
             ('End to end, p99', _f(e('e2e').get('p99'))), ('Round trip, median', _f(e('round_trip').get('median'))),
             ('On time', f'{dl.get("on_time", 0)} of {dl.get("frames", 0)}'),
             ('Real-time factor', _f(perf.get('realtime_factor'), 3, ''))]
    rows = []
    for k, st in (s.get('stages') or {}).items():
        rows.append((st['label'], PLACE[st['clock']][0], st))
    mrows = [(v['label'], v.get('clock', ''), v) for v in m.values()]

    def tr(label, place, d, extra=''):
        cells = ''.join(f'<td class="n">{_f(d.get(k), 3, "")}</td>' for k in ('median', 'p95', 'p99', 'mean', 'std',
                                                                                  'max'))
        return f'<tr><td>{html.escape(label)}{extra}</td><td>{html.escape(place)}</td>{cells}<td class="n">' \
               f'{d.get("n", 0)}</td></tr>'
    head = ('<tr><th>stage</th><th>runs on</th><th class="n">median</th><th class="n">p95</th><th class="n">p99</th>'
            '<th class="n">mean</th><th class="n">std (jitter)</th><th class="n">max</th>'
            '<th class="n">frames</th></tr>')
    trust = s.get('one_way_trusted')
    off = ''
    if est:
        off = (f'<p>Offset of the device\'s clock: <b>{est["offset_ns"] / 1e6:.4f} ms</b> ± '
               f'{est["uncertainty_ns"] / 1e6:.4f}'
               f' ms, drift {est["drift_ppm"]:.2f} ppm, from {est["samples"]} pings (least round trip '
               f'{est["min_rtt_ns"] / 1e6:.3f} ms, median {est["median_rtt_ns"] / 1e6:.3f} ms). One-way stages are '
               f'<b>{"trusted" if trust else "not trusted"}</b> (the uncertainty must be below 1 ms). '
               f'{html.escape(est.get("method", ""))}.</p>')
        chk = s.get('clock_offset_check')
        if chk:
            off += (f'<p>Simulated device, known truth: {chk["truth_ns"] / 1e6:.4f} ms; the estimate is off by '
                    f'{chk["error_ns"] / 1e3:.2f} µs — {"within" if chk["within_uncertainty"] else "OUTSIDE"} its '
                    'stated uncertainty.</p>')
    else:
        off = '<p>No clock pings were answered: one-way stages are not available.</p>'
    drops = s.get('drops') or {}
    jit = s.get('jitter') or {}
    sc = s.get('sensor_clocks') or {}
    clk_rows = ''.join(
        f'<tr><td>{k}</td><td>{html.escape(json.dumps(v.get("configured")))}</td>'
        f'<td>{html.escape(json.dumps(v.get("recovered")))}</td></tr>' for k, v in sc.items())
    prov = run.get('provenance') or {}
    ia = jit.get('camera_interarrival') or {}
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>HIL latency report</title>
<style>{CSS}</style></head><body>
<h1>Hardware-in-the-loop latency</h1>
<p class="muted">{html.escape(run.get('name') or 'HIL session')} · {html.escape(str(run.get('started') or ''))} ·
Falcon Autonomy {html.escape(str(prov.get('app_version', '')))} · seed {html.escape(str(ses.get('seed', '')))}</p>
<p>Measured with {where}. Camera {cam.get('width')} × {cam.get('height')} at {ses.get('fps')} Hz as
<code>{html.escape(str(cam.get('transport')))}</code> (encoder
<code>{html.escape(str(cam.get('encoder_used')))}</code>),
lidar {lid.get('channels')} channels at {lid.get('points_per_second')} points/s over
<code>{html.escape(str(lid.get('transport')))}</code> ({html.escape(str(lid.get('encoding')))}); device: Python
{html.escape(str(env.get('python')))}, {html.escape(str(env.get('platform')))}, decoder
<code>{html.escape(str(dev.get('decoder')))}</code>, stack <code>{html.escape(str(dev.get('stack')))}</code>.
{load_note(ses)}</p>
<div class="tiles">{''.join(f'<div class="tile"><span>{a}</span><b>{b}</b></div>' for a, b in tiles)}</div>
<h2>Latency waterfall (the median frame)</h2>
<div class="key"><span><i style="background:var(--host)"></i>this machine</span><span><i style="background:var(--net)">
</i>network (one-way)</span><span><i style="background:var(--dev)"></i>the device</span><span>whisker: p95 of the stage
</span></div>
{waterfall_svg(s)}
<p class="muted">Each bar starts where the previous stage's median ended. The end-to-end latency runs from the start of
the step that rendered the frame to the start of the step that applied its control; in a real-time session it is
quantised to whole steps (dashed lines).</p>
<h2>Stages</h2><table>{head}{''.join(tr(*r) for r in rows)}</table>
<h2>Other measures</h2><table>{head}{''.join(tr(*r) for r in mrows)}</table>
<h2>Step start to control at the simulation, per frame</h2>{series_svg(records, s.get('period_ms') or 0)}
<h2>Deadlines and drops</h2>
<table><tr><th>frames streamed</th><th class="n">{dl.get('frames')}</th></tr>
<tr><td>control on time (applied at the next step)</td><td class="n">{dl.get('on_time')}</td></tr>
<tr><td>control late (applied at a later step)</td><td class="n">{dl.get('late')}</td></tr>
<tr><td>control superseded (a newer one came first)</td><td class="n">{dl.get('superseded')}</td></tr>
<tr><td>no control came back</td><td class="n">{dl.get('missing')}</td></tr>
<tr><td>camera frames received / decoded by the device</td><td class="n">{drops.get('camera_received')} /
{drops.get('camera_decoded')}</td></tr>
<tr><td>camera frames lost</td><td class="n">{drops.get('camera_lost')}</td></tr>
<tr><td>lidar messages received / lost</td><td class="n">{drops.get('lidar_received')} /
{drops.get('lidar_lost')}</td></tr>
<tr><td>dropped by the gateway (camera, lidar)</td><td class="n">{(drops.get('gateway') or {}).get('camera_dropped')},
{(drops.get('gateway') or {}).get('lidar_dropped')}</td></tr>
<tr><td>device counters</td><td class="n"><code>{html.escape(json.dumps(drops.get('device') or {}))}</code></td></tr>
<tr><td>steps that ran late (pacing restarted)</td><td class="n">{(s.get('worker') or {}).get('steps_late')}</td></tr>
<tr><td>steps braked for lack of control (time-out)</td><td class="n">
{(s.get('worker') or {}).get('control_timeouts')}</td></tr>
</table>
<h2>Jitter</h2><p>Step interval: median {_f((jit.get('step_interval_ms') or {}).get('median'), 3)}, std
{_f((jit.get('step_interval_ms') or {}).get('std'), 3)}. Camera interarrival at the device (RFC 3550):
{_f(ia.get('rfc3550_ms'), 4)} (mean |D| {_f(ia.get('mean_abs_d_ms'), 4)} over {ia.get('pairs', 0)} pairs).</p>
<h2>Clocks</h2>{off}
<table><tr><th>sensor</th><th>emulated clock (configured)</th><th>recovered from its stamps</th></tr>{clk_rows}</table>
<h2>Provenance</h2><pre><code>{html.escape(json.dumps({'run': run.get('name'), 'status': run.get('status'),
'host': prov.get('host'), 'inputs': prov.get('inputs'), 'device_env': env,
'gateway': (ses.get('gateway') or {}).get('python')}, indent=1, default=str))}</code></pre>
</body></html>"""
    return body


def waterfall_png(s: dict, path: str) -> bool:
    """The waterfall as a PNG (OpenCV; light theme). False when cv2 is not importable."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False
    rows = s.get('waterfall') or []
    e2e = (s.get('measures') or {}).get('e2e') or {}
    W, lw, rw, rh, top = 1400, 380, 260, 40, 50
    H = top + rh * (len(rows) + 1) + 60
    img = np.full((H, W, 3), 251, np.uint8)
    span = max([r['start_ms'] + max(r['p95_ms'] or 0, r['median_ms']) for r in rows] + [e2e.get('p99') or 0, 1]) * 1.05
    pw = W - lw - rw
    X = lambda ms: int(lw + pw * ms / span)              # noqa: E731
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'HIL latency waterfall (median frame; whisker: p95)', (20, 30), font, 0.7, (11, 11, 11), 2,
                cv2.LINE_AA)
    kx = W - 560
    for name, key in (('this machine', 'host'), ('network (one-way)', 'cross'), ('the device', 'device')):
        cv2.rectangle(img, (kx, 18), (kx + 14, 32), PNG_RGB[key], -1)
        cv2.putText(img, name, (kx + 20, 30), font, 0.5, (82, 81, 78), 1, cv2.LINE_AA)
        kx += 40 + cv2.getTextSize(name, font, 0.5, 1)[0][0]
    period = s.get('period_ms') or 0
    for k in range(1, int(span // period) + 1 if period else 0):
        x = X(k * period)
        for yy in range(top, H - 50, 10):
            cv2.line(img, (x, yy), (x, yy + 5), (160, 160, 160), 1)
        cv2.putText(img, f'step {k}', (x + 4, top - 6), font, 0.45, (110, 110, 110), 1, cv2.LINE_AA)
    for i, r in enumerate(rows + [None]):
        y = top + i * rh
        if r is None:
            if e2e.get('median') is None:
                break
            label, x0, x1, col = 'End to end', lw, X(e2e['median']), (78, 81, 82)
            val = f'{e2e["median"]:.2f} | p95 {e2e["p95"]:.2f} ms'
        else:
            label, x0, x1 = r['label'], X(r['start_ms']), X(r['start_ms'] + r['median_ms'])
            if r['clock'] == 'cross' and not s.get('one_way_trusted'):
                label += ' (offset not trusted)'
            col = PNG_RGB[r['clock']]
            val = f'{r["median_ms"]:.2f} | p95 {_f(r["p95_ms"], 2, "")} ms'
        (tw, _), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.putText(img, label, (lw - 12 - tw, y + 26), font, 0.5, (11, 11, 11), 1, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y + 8), (max(x1, x0 + 2), y + rh - 8), col, -1)
        if r is not None and r.get('p95_ms') and r['p95_ms'] > r['median_ms']:
            xp = X(r['start_ms'] + r['p95_ms'])
            cv2.line(img, (x1, y + rh // 2), (xp, y + rh // 2), (82, 81, 78), 1)
            cv2.line(img, (xp, y + 12), (xp, y + rh - 12), (82, 81, 78), 1)
        cv2.putText(img, val, (W - rw + 10, y + 26), font, 0.5, (78, 81, 82), 1, cv2.LINE_AA)
    cv2.putText(img, f'0 .. {span:.0f} ms from the start of the step', (lw, H - 20), font, 0.5, (110, 110, 110), 1,
                cv2.LINE_AA)
    return bool(cv2.imwrite(path, img))


def write_report(run_dir: str, out_dir: str = '') -> dict:
    """report.html and waterfall.png for a session (in out_dir, default the run folder); their paths."""
    from ..core.hil import read_jsonl
    s, run = load_summary(run_dir)
    records = read_jsonl(os.path.join(run_dir, 'latency.jsonl'))
    out_dir = out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    p = os.path.join(out_dir, 'report.html')
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write(report_html(s, run, records))
    files['html'] = p
    png = os.path.join(out_dir, 'waterfall.png')
    if waterfall_png(s, png):
        files['png'] = png
    return files
