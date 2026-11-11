"""PTP (IEEE 1588) and gPTP (IEEE 802.1AS) for hardware in the loop (docs/hardware.md#clocks-ptp-and-the-ntp-style-
fallback): what linuxptp says, the configuration files for the user to install, and the session's clock source —
PTP when both machines are synchronised to one grandmaster, else the NTP-style ping estimate (core.hil.estimate_offset),
both recorded with their disagreement. Standard library and numpy. Falcon never starts ptp4l, phc2sys or anything
with privileges: `falcon hil setup` prints the commands.

Parsers:
    parse_pmc(text)          `pmc -u -b 0 'GET TIME_STATUS_NP' 'GET PORT_DATA_SET' 'GET CURRENT_DATA_SET'` → {ID: {...}}
    parse_ptp4l_line(line)   one line of `ptp4l -m` / `phc2sys -m` (offsets with servo state, rms summaries, port
                             state changes, best-master selection)
    summarise_log(lines)     their statistics: offsets in the locked state (s2), the last port state, locked or not
    parse_ethtool_t(text)    `ethtool -T IF`: software / hardware timestamping, the PTP hardware clock

Configurations (ptp4l_conf): 'software' — the default (E2E, UDPv4) profile with `time_stamping software`: what a NIC
without a hardware clock (this machine's, the user's) can do; ptp4l then steers CLOCK_REALTIME itself, no phc2sys.
Expect tens of microseconds: the stamps are taken in the kernel's network stack, after interrupt and softirq latency.
'gptp' — linuxptp's gPTP profile (transportSpecific 0x1, Layer 2 to 01:80:C2:00:00:0E, peer-to-peer delay, two-step,
follow_up_info); 802.1AS assumes hardware timestamping (a neighbour's propagation delay must stay under 800 ns to be
"asCapable"), so with software stamps the threshold is raised and the result is no longer conformant (the file says
so). Hardware timestamping needs a NIC with a PHC (ethtool -T: hardware-transmit, hardware-receive, a PTP Hardware
Clock) and phc2sys to bring CLOCK_REALTIME to it.

The clock source. Session stamps are each machine's CLOCK_MONOTONIC; PTP disciplines CLOCK_REALTIME. The device
reports pairs (its CLOCK_REALTIME, its monotonic stamp clock) read back to back; the gateway notes this machine's
CLOCK_REALTIME − CLOCK_MONOTONIC (Δ_h) when each arrives. If both realtime clocks agree (PTP), the device's stamp
clock is this machine's monotonic plus θ = Δ_h − Δ_d (Δ_d = the device's realtime − its stamp clock), at host time
t = realtime − Δ_h; a line θ(t) = a + b·(t − t_ref) is fitted through them as for the pings. Its uncertainty: the fit's
residual RMS + the worst |master offset| either machine's pmc reported during the session + READ_GAP_NS (two clock
reads are not simultaneous). A simulated device shares this machine's CLOCK_REALTIME by construction ("shared
realtime"), which is what PTP provides between two machines, so its estimate exercises the same path. PTP counts
only when both pmc report a grandmaster present, the same grandmaster, a SLAVE or MASTER port, and |offset| below
PTP_MAX_OFFSET_NS; otherwise its reasons are recorded and the NTP-style estimate is used.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

SOURCES = ('auto', 'ptp', 'ntp-style')
READ_GAP_NS = 2000.0
PTP_MAX_OFFSET_NS = 1_000_000
TRUST_NS = 1_000_000


# ============================================================================================ parsers
def _value(v: str):
    v = v.strip()
    if v in ('true', 'false'):
        return v == 'true'
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


_RESP = re.compile(r'^\s*(\S+)\s+seq\s+\d+\s+RESPONSE\s+MANAGEMENT\s+(\w+)')


def parse_pmc(text: str) -> dict:
    """pmc's answers: {management ID: {field: value, '_port': responding port identity}} (the last of each ID)."""
    out, cur = {}, None
    for line in (text or '').splitlines():
        m = _RESP.match(line)
        if m:
            cur = {'_port': m.group(1)}
            out[m.group(2)] = cur
            continue
        if line.startswith('sending:') or not line.strip():
            continue
        if cur is not None and line.startswith(('\t\t', '        ')):
            parts = line.split(None, 1)
            if len(parts) == 2:
                cur[parts[0]] = _value(parts[1])
            elif len(parts) == 1:
                cur[parts[0]] = ''
    return out


def pmc_summary(parsed: dict) -> dict:
    """The fields that matter from parse_pmc: grandmaster present and its identity, the offset from the master
    (ns), the mean path delay, the port's state, the steps removed."""
    ts = parsed.get('TIME_STATUS_NP') or {}
    cds = parsed.get('CURRENT_DATA_SET') or {}
    pds = parsed.get('PORT_DATA_SET') or {}
    off = ts.get('master_offset', cds.get('offsetFromMaster'))
    return {'gm_present': ts.get('gmPresent'), 'gm_identity': ts.get('gmIdentity'),
            'offset_ns': None if off is None else float(off), 'mean_path_delay_ns': cds.get('meanPathDelay'),
            'steps_removed': cds.get('stepsRemoved'), 'port_state': pds.get('portState'),
            'clock_identity': str(ts.get('_port') or pds.get('_port') or '').split('-')[0] or None}


_PFX = r'^(?:(?P<prog>ptp4l|phc2sys)\[(?P<t>[\d.]+)\]:\s*)?'
_OFFSET = re.compile(_PFX + r'(?:(?P<clk>\S+)\s+phc\s+offset|master\s+offset)\s+(?P<off>-?\d+)\s+(?P<st>s\d)\s+'
                            r'freq\s+(?P<freq>[+-]?\d+)(?:\s+(?:path\s+)?delay\s+(?P<delay>-?\d+))?')
_RMS = re.compile(_PFX + r'rms\s+(?P<rms>\d+)\s+max\s+(?P<max>\d+)\s+freq\s+(?P<freq>[+-]?\d+)\s+\+/-\s+(?P<fsd>\d+)'
                         r'(?:\s+delay\s+(?P<delay>\d+)\s+\+/-\s+(?P<dsd>\d+))?')
_PORT = re.compile(_PFX + r'port\s+(?P<port>\d+)(?:\s+\([^)]*\))?:\s+(?P<from>[A-Z_]+)\s+to\s+(?P<to>[A-Z_]+)'
                          r'(?:\s+on\s+(?P<ev>\S+))?')
_BMC = re.compile(_PFX + r'selected\s+best\s+master\s+clock\s+(?P<id>\S+)')
_GM = re.compile(_PFX + r'assuming\s+the\s+grand\s+master\s+role')


def parse_ptp4l_line(line: str) -> dict | None:
    """One log line of ptp4l -m or phc2sys -m, or None when it is none of the kinds read."""
    line = line.strip()
    for kind, rx in (('offset', _OFFSET), ('rms', _RMS), ('port', _PORT), ('best_master', _BMC),
                     ('grandmaster', _GM)):
        m = rx.match(line)
        if not m:
            continue
        d = {'kind': kind, 'program': m.group('prog') or 'ptp4l',
             't': float(m.group('t')) if m.group('t') else None}
        g = m.groupdict()
        if kind == 'offset':
            d.update(offset_ns=int(g['off']), state=g['st'], freq_ppb=int(g['freq']),
                     delay_ns=int(g['delay']) if g['delay'] else None, clock=g['clk'] or 'phc')
        elif kind == 'rms':
            d.update(rms_ns=int(g['rms']), max_ns=int(g['max']), freq_ppb=int(g['freq']), freq_sd_ppb=int(g['fsd']),
                     delay_ns=int(g['delay']) if g['delay'] else None)
        elif kind == 'port':
            d.update(port=int(g['port']), frm=g['from'], to=g['to'], event=g['ev'])
        elif kind == 'best_master':
            d.update(identity=g['id'])
        return d
    return None


def summarise_log(lines) -> dict:
    """What a ptp4l / phc2sys log says: offsets while locked (servo state s2), rms summaries, the last port state,
    the grandmaster chosen, and whether it is locked now."""
    import numpy as np
    offs, rms, state, gm, last_servo = [], [], None, None, None
    for line in lines:
        d = parse_ptp4l_line(line)
        if d is None:
            continue
        if d['kind'] == 'offset':
            last_servo = d['state']
            if d['state'] == 's2':
                offs.append(d['offset_ns'])
        elif d['kind'] == 'rms':
            rms.append(d['rms_ns'])
            last_servo = 's2'
        elif d['kind'] == 'port':
            state = d['to']
        elif d['kind'] == 'best_master':
            gm = d['identity']
        elif d['kind'] == 'grandmaster':
            gm, state = 'self', state or 'MASTER'
    a = np.abs(np.asarray(offs, np.float64))
    return {'offsets': len(offs), 'offset_abs_median_ns': float(np.median(a)) if len(a) else None,
            'offset_abs_max_ns': float(a.max()) if len(a) else None, 'rms_last_ns': rms[-1] if rms else None,
            'port_state': state, 'grandmaster': gm, 'servo': last_servo,
            'locked': last_servo == 's2' and state in ('SLAVE', 'MASTER', None)}


def parse_ethtool_t(text: str) -> dict:
    """`ethtool -T IF`: {capabilities, phc (index or None), hardware, software, tx_modes, rx_filters}. Reads both the
    older "PTP Hardware Clock:" and the newer "Hardware timestamp provider index:" lines."""
    caps, tx, rx, phc, section = [], [], [], None, None
    for line in (text or '').splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith('capabilities'):
            section = 'caps'
            continue
        if low.startswith(('ptp hardware clock', 'hardware timestamp provider index')):
            v = s.split(':', 1)[1].strip()
            phc = int(v) if v.lstrip('-').isdigit() and int(v) >= 0 else None
            section = None
            continue
        if low.startswith('hardware transmit timestamp modes'):
            section = 'tx'
            rest = s.split(':', 1)[1].strip()
            if rest and rest != 'none':
                tx.append(rest)
            continue
        if low.startswith('hardware receive filter modes'):
            section = 'rx'
            rest = s.split(':', 1)[1].strip()
            if rest and rest != 'none':
                rx.append(rest)
            continue
        if low.startswith('time stamping parameters'):
            continue
        if line.startswith(('\t', ' ')) and section:
            name = s.split()[0]
            {'caps': caps, 'tx': tx, 'rx': rx}[section].append(name)
        elif ':' in s:
            section = None
    hw = 'hardware-transmit' in caps and 'hardware-receive' in caps and phc is not None
    sw = 'software-transmit' in caps and 'software-receive' in caps
    return {'capabilities': caps, 'phc': phc, 'hardware': hw, 'software': sw, 'tx_modes': tx, 'rx_filters': rx}


# ============================================================================================ configurations
def ptp4l_conf(profile: str = 'software', interface: str = 'eth0', domain: int = 0, slave_only: bool = False,
               timestamping: str | None = None) -> str:
    """A ptp4l.conf for the user to install (module doc). profile 'software' | 'gptp'; timestamping 'software' |
    'hardware' (default: software for 'software', hardware for 'gptp')."""
    if profile not in ('software', 'gptp'):
        raise ValueError(f'profile {profile!r} is not software or gptp')
    if not re.match(r'^[A-Za-z0-9_.:-]{1,15}$', interface or ''):
        raise ValueError(f'{interface!r} is not an interface name')
    ts = timestamping or ('software' if profile == 'software' else 'hardware')
    if ts not in ('software', 'hardware'):
        raise ValueError('timestamping is software or hardware')
    head = ['# ptp4l configuration written by Falcon Autonomy (falcon hil ptp conf); Falcon does not install or run',
            f'# it. Install: sudo cp THIS /etc/linuxptp/falcon-{profile}.conf; run: sudo ptp4l -f THAT -m']
    if profile == 'software':
        lines = head + ['# Default (E2E, UDPv4) profile with software timestamping: ptp4l steers CLOCK_REALTIME',
                        '# directly (no phc2sys). Expect tens of microseconds of error: the stamps are taken in the',
                        '# kernel\'s network stack, after interrupt latency.',
                        '[global]',
                        f'time_stamping           {ts}',
                        f'domainNumber            {int(domain)}',
                        f'slaveOnly               {1 if slave_only else 0}',
                        'network_transport       UDPv4',
                        'delay_mechanism         E2E',
                        'clock_servo             pi',
                        'step_threshold          0.001',
                        'first_step_threshold    0.00002',
                        'summary_interval        0',
                        'logging_level           6',
                        'tx_timestamp_timeout    10']
    else:
        lines = head + ['# gPTP (IEEE 802.1AS) profile, as linuxptp\'s configs/gPTP.cfg: Layer 2, peer-to-peer delay.']
        if ts == 'software':
            lines += ['# SOFTWARE TIMESTAMPING: 802.1AS needs a neighbour delay under 800 ns to be asCapable, which',
                      '# software stamps cannot measure; the threshold below is raised to 20 ms so that the link',
                      '# synchronises at all. This is NOT a conformant gPTP setup: use it to test the plumbing only.']
        else:
            lines += ['# Hardware timestamping: the NIC needs a PTP hardware clock (ethtool -T: hardware-transmit,',
                      f'# hardware-receive). Bring CLOCK_REALTIME to it: sudo phc2sys -s {interface} -c CLOCK_REALTIME '
                      '-w -m']
        lines += ['[global]',
                  f'time_stamping           {ts}',
                  f'domainNumber            {int(domain)}',
                  f'slaveOnly               {1 if slave_only else 0}',
                  'gmCapable               1',
                  'priority1               248',
                  'priority2               248',
                  'logAnnounceInterval     0',
                  'logSyncInterval         -3',
                  'syncReceiptTimeout      3',
                  f'neighborPropDelayThresh {800 if ts == "hardware" else 20000000}',
                  'min_neighbor_prop_delay -20000000',
                  'assume_two_step         1',
                  'path_trace_enabled      1',
                  'follow_up_info          1',
                  'transportSpecific       0x1',
                  'ptp_dst_mac             01:80:C2:00:00:0E',
                  'network_transport       L2',
                  'delay_mechanism         P2P',
                  'summary_interval        0',
                  'logging_level           6']
    lines += ['', f'[{interface}]', '']
    return '\n'.join(lines)


def commands(interface: str = 'eth0', profile: str = 'software', conf_path: str = '') -> list:
    """The commands to run PTP with the generated file (printed by `falcon hil setup`, never run by Falcon)."""
    conf = conf_path or f'/etc/linuxptp/falcon-{profile}.conf'
    out = ['sudo apt install linuxptp                       # ptp4l, phc2sys, pmc',
           f'falcon hil ptp conf --profile {profile} --interface {interface} > falcon-{profile}.conf',
           f'sudo install -D -m 644 falcon-{profile}.conf {conf}',
           f'sudo ptp4l -f {conf} -m                     # on both machines; one becomes the grandmaster']
    if profile == 'gptp':
        out.append(f'sudo phc2sys -s {interface} -c CLOCK_REALTIME -w -m   # hardware timestamping only')
    out.append('sudo pmc -u -b 0 "GET TIME_STATUS_NP"        # check: gmPresent true, master_offset small')
    return out


# ============================================================================================ this machine
def _running(comm: str) -> bool:
    try:
        pids = [p for p in os.listdir('/proc') if p.isdigit()]
    except OSError:
        return False
    for p in pids:
        try:
            with open(f'/proc/{p}/comm', encoding='utf-8') as fh:
                if fh.read().strip() == comm:
                    return True
        except OSError:
            continue
    return False


def query_pmc(timeout: float = 3.0) -> dict:
    """This machine's pmc answers: {available, text, code, error, parsed, summary} (pmc talks to ptp4l's socket;
    without the right to it, the error says so — nothing is elevated)."""
    pmc = shutil.which('pmc')
    out = {'pmc': pmc, 'ptp4l_running': _running('ptp4l'), 'phc2sys_running': _running('phc2sys'), 'text': None,
           'code': None, 'error': None}
    if not pmc:
        out['error'] = 'pmc is not installed (linuxptp)'
        return dict(out, parsed={}, summary={})
    try:
        r = subprocess.run([pmc, '-u', '-b', '0', 'GET TIME_STATUS_NP', 'GET PORT_DATA_SET', 'GET CURRENT_DATA_SET'],
                           capture_output=True, text=True, timeout=timeout)
        out.update(text=r.stdout[-8000:], code=r.returncode, error=r.stderr.strip()[-400:] or None)
    except (OSError, subprocess.TimeoutExpired) as e:
        out['error'] = f'{type(e).__name__}: {e}'
    return with_parsed(out)


def with_parsed(status: dict) -> dict:
    """A pmc status (this machine's, or one the device agent reported raw) with its parse and summary."""
    st = dict(status or {})
    parsed = parse_pmc(st.get('text') or '')
    st['parsed'] = parsed
    st['summary'] = pmc_summary(parsed) if parsed else {}
    return st


def ethtool_t(interface: str) -> dict:
    """`ethtool -T` of an interface, parsed ({} when ethtool is missing or the interface is unknown)."""
    if not interface or not re.match(r'^[A-Za-z0-9_.:-]{1,15}$', interface):
        return {}
    exe = shutil.which('ethtool') or ('/usr/sbin/ethtool' if os.path.exists('/usr/sbin/ethtool') else None)
    if not exe:
        return {}
    try:
        r = subprocess.run([exe, '-T', interface], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if r.returncode != 0:
        return {'error': (r.stderr or r.stdout).strip()[:200]}
    return parse_ethtool_t(r.stdout)


def default_interface() -> str:
    """The interface of the default route (/proc/net/route), '' when none."""
    try:
        with open('/proc/net/route', encoding='ascii') as fh:
            for line in fh.readlines()[1:]:
                f = line.split()
                if len(f) > 2 and f[1] == '00000000':
                    return f[0]
    except OSError:
        pass
    return ''


def doctor_rows(settings=None) -> list:
    """(name, ok, detail) for `falcon doctor` (informational: PTP is optional)."""
    cfg = (settings.get('hil_ptp') if settings is not None else None) or {}
    iface = cfg.get('interface') or default_interface()
    rows = []
    for exe in ('ptp4l', 'phc2sys', 'pmc'):
        p = shutil.which(exe) or (f'/usr/sbin/{exe}' if os.path.exists(f'/usr/sbin/{exe}') else None)
        rows.append((exe, bool(p), p or 'not installed (optional: sudo apt install linuxptp)'))
    if iface:
        t = ethtool_t(iface)
        if t.get('error') or not t:
            rows.append((f'timestamping {iface}', False, t.get('error') or 'ethtool not available'))
        else:
            kind = ('hardware (PHC %d) and software' % t['phc']) if t['hardware'] else \
                'software only: no PTP hardware clock (expect tens of µs)' if t['software'] else \
                f'not enough for ptp4l ({", ".join(t["capabilities"]) or "nothing"}: it needs transmit and receive ' \
                'timestamps — a wired interface usually has them)'
            rows.append((f'timestamping {iface}', t['software'] or t['hardware'], kind))
    run = _running('ptp4l')
    rows.append(('ptp4l running', run, 'yes' if run else 'no (the NTP-style estimate is used)'))
    return rows


# ============================================================================================ the clock source
def ptp_basis(local: bool, host: dict | None, devices: list) -> dict:
    """Whether a realtime-based offset may be used: {ok, basis, why, bound_ns}. host: this machine's with_parsed
    status; devices: the device's statuses during the session (with_parsed)."""
    if local:
        return {'ok': True, 'basis': 'shared realtime',
                'why': 'a simulated device: it reads this machine\'s CLOCK_REALTIME', 'bound_ns': 0.0}
    why = []
    hs = (host or {}).get('summary') or {}
    dss = [d.get('summary') or {} for d in devices if d]
    if not hs:
        why.append('this machine\'s pmc gave no answer' + (f' ({host.get("error")})' if (host or {}).get('error')
                                                           else ''))
    if not dss:
        why.append('the device reported no pmc status')
    for name, s in [('this machine', hs)] + [('the device', d) for d in dss[-1:]]:
        if not s:
            continue
        if not s.get('gm_present') and s.get('port_state') != 'MASTER':
            why.append(f'{name}: no grandmaster present')
        if s.get('port_state') not in ('SLAVE', 'MASTER', 'GRAND_MASTER', None):
            why.append(f'{name}: port {s.get("port_state")}')
        if s.get('offset_ns') is not None and abs(s['offset_ns']) > PTP_MAX_OFFSET_NS:
            why.append(f'{name}: offset {s["offset_ns"]:.0f} ns')
    if hs and dss:
        gh, gd = hs.get('gm_identity'), dss[-1].get('gm_identity')
        if gh and gd and gh != gd:
            why.append(f'different grandmasters ({gh}, {gd})')
    offs = [abs(s['offset_ns']) for s in [hs] + dss if s and s.get('offset_ns') is not None]
    return {'ok': not why, 'basis': 'ptp', 'why': '; '.join(why) or 'both synchronised to one grandmaster',
            'bound_ns': float(max(offs)) if offs else 0.0}


def realtime_estimate(rows: list, bound_ns: float = 0.0, basis: str = 'ptp') -> dict:
    """The device clock's offset from realtime pairs (module doc). rows: [{h_rt, h_mono, d_rt, d_ns}] — the host's
    realtime and monotonic read when the device's pair arrived, the device's realtime and stamp clock read together.
    The same keys as core.hil.estimate_offset."""
    import numpy as np
    R = [r for r in rows if all(r.get(k) is not None for k in ('h_rt', 'h_mono', 'd_rt', 'd_ns'))]
    if not R:
        return {}
    dh = np.array([r['h_rt'] - r['h_mono'] for r in R], np.float64)
    dd = np.array([r['d_rt'] - r['d_ns'] for r in R], np.float64)
    th = dh - dd
    t = np.array([r['d_rt'] for r in R], np.float64) - dh          # host monotonic at the device's reading
    t_ref = float(np.median(t))
    if len(R) >= 3 and np.ptp(t) > 0:
        x = t - t_ref
        A = np.c_[np.ones_like(x), x]
        (a, b), *_ = np.linalg.lstsq(A, th, rcond=None)
        resid = float(np.sqrt(np.mean((th - (a + b * x)) ** 2)))
    else:
        a, b, resid = float(np.median(th)), 0.0, float(np.std(th))
    unc = resid + float(bound_ns) + READ_GAP_NS
    return {'offset_ns': float(a), 't_ref_ns': int(t_ref), 'drift_ppm': float(b) * 1e6, 'uncertainty_ns': unc,
            'fit_residual_ns': resid, 'samples': len(R), 'trusted': bool(unc <= TRUST_NS), 'basis': basis,
            'bound_ns': float(bound_ns),
            'method': f'realtime pairs ({basis}): θ = (host realtime − monotonic) − (device realtime − stamp clock); '
                      'line fit; uncertainty = fit residual + worst |master offset| + two clock reads'}


def theta_at(est: dict, host_ns: float) -> float:
    return est['offset_ns'] + est['drift_ppm'] * 1e-6 * (host_ns - est['t_ref_ns'])


def choose(preference: str, ptp_est: dict, ntp_est: dict, basis: dict | None = None) -> dict:
    """The session's clock source: 'auto' takes PTP when it may be used, 'ptp' insists (falls back, saying so),
    'ntp-style' never takes it. Both estimates and their disagreement are kept: θ_ptp − θ_ntp at the NTP-style
    estimate's reference time, consistent when within the sum of their uncertainties."""
    pref = preference if preference in SOURCES else 'auto'
    basis = basis or {}
    usable = bool(ptp_est) and basis.get('ok', bool(ptp_est))
    used = 'ptp' if usable and pref in ('auto', 'ptp') else 'ntp-style'
    if used == 'ntp-style' and not ntp_est and usable:
        used = 'ptp'
    out = {'preference': pref, 'used': used, 'ptp': ptp_est or None, 'ntp_style': ntp_est or None,
           'ptp_basis': basis or None,
           'why': (basis.get('why') if used == 'ptp' else
                   'the NTP-style estimate was asked for' if pref == 'ntp-style' else
                   f'PTP not usable: {basis.get("why") or "no realtime pairs"}')}
    if ptp_est and ntp_est:
        t = ntp_est['t_ref_ns']
        d = theta_at(ptp_est, t) - theta_at(ntp_est, t)
        out['disagreement_ns'] = d
        out['consistent'] = abs(d) <= ptp_est['uncertainty_ns'] + ntp_est['uncertainty_ns']
    out['estimate'] = (ptp_est if used == 'ptp' else ntp_est) or {}
    return out
