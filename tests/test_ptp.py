"""PTP for hardware in the loop (core.ptp): linuxptp's pmc answers, ptp4l / phc2sys log lines and ethtool -T parsed;
the ptp4l.conf files (software timestamping, the gPTP profile) for the user to install; a stand-in pmc and ptp4l on
PATH; the clock source — PTP's realtime pairs recovering a known offset and drift, when PTP may be used and when not
(and why), the choice between it and the NTP-style estimate, and their disagreement."""
import os
import stat

import numpy as np
import pytest

from falcon_autonomy.core import hil as H
from falcon_autonomy.core import ptp

PMC = """sending: GET TIME_STATUS_NP
\t48b02d.fffe.5a1c22-1 seq 0 RESPONSE MANAGEMENT TIME_STATUS_NP
\t\tmaster_offset              -27
\t\tingress_time               1794422044714916883
\t\tcumulativeScaledRateOffset +0.000000000
\t\tscaledLastGmPhaseChange    0
\t\tgmTimeBaseIndicator        0
\t\tlastGmPhaseChange          0x0000'0000000000000000.0000
\t\tgmPresent                  true
\t\tgmIdentity                 001b21.fffe.7c3d10
sending: GET PORT_DATA_SET
\t48b02d.fffe.5a1c22-1 seq 1 RESPONSE MANAGEMENT PORT_DATA_SET
\t\tportIdentity            48b02d.fffe.5a1c22-1
\t\tportState               SLAVE
\t\tlogMinDelayReqInterval  0
\t\tpeerMeanPathDelay       0
\t\tdelayMechanism          1
sending: GET CURRENT_DATA_SET
\t48b02d.fffe.5a1c22-1 seq 2 RESPONSE MANAGEMENT CURRENT_DATA_SET
\t\tstepsRemoved     1
\t\toffsetFromMaster -27.0
\t\tmeanPathDelay    7345.0
"""

LOG = """ptp4l[5178.110]: selected /dev/ptp0 as PTP clock
ptp4l[5178.110]: port 1 (enp1s0): INITIALIZING to LISTENING on INIT_COMPLETE
ptp4l[5179.640]: port 1 (enp1s0): LISTENING to UNCALIBRATED on RS_SLAVE
ptp4l[5179.641]: selected best master clock 001b21.fffe.7c3d10
ptp4l[5180.012]: master offset      -4123 s0 freq   +2000 path delay      7312
ptp4l[5181.012]: master offset       -812 s1 freq    -511 path delay      7350
ptp4l[5181.700]: port 1: UNCALIBRATED to SLAVE on MASTER_CLOCK_SELECTED
ptp4l[5182.012]: master offset        -31 s2 freq    -540 path delay      7345
ptp4l[5183.012]: master offset         18 s2 freq    -522 path delay      7340
ptp4l[5184.012]: master offset         -9 s2 freq    -530 path delay      7343
phc2sys[5185.000]: CLOCK_REALTIME phc offset       -15 s2 freq   +2345 delay    512
ptp4l[5190.000]: rms   14 max   31 freq   -531 +/-   6 delay  7344 +/-   3
something else entirely
"""

ETHTOOL_HW = """Time stamping parameters for enp1s0:
Capabilities:
\thardware-transmit
\tsoftware-transmit
\thardware-receive
\tsoftware-receive
\tsoftware-system-clock
\thardware-raw-clock
PTP Hardware Clock: 0
Hardware Transmit Timestamp Modes:
\toff
\ton
Hardware Receive Filter Modes:
\tnone
\tall
"""
ETHTOOL_NEW = ETHTOOL_HW.replace('PTP Hardware Clock: 0', 'Hardware timestamp provider index: 2')
ETHTOOL_SW = """Time stamping parameters for enp14s0:
Capabilities:
\tsoftware-transmit
\tsoftware-receive
\tsoftware-system-clock
PTP Hardware Clock: none
Hardware Transmit Timestamp Modes: none
Hardware Receive Filter Modes: none
"""


def test_parse_pmc():
    p = ptp.parse_pmc(PMC)
    assert set(p) == {'TIME_STATUS_NP', 'PORT_DATA_SET', 'CURRENT_DATA_SET'}
    ts = p['TIME_STATUS_NP']
    assert ts['master_offset'] == -27 and ts['gmPresent'] is True and ts['gmIdentity'] == '001b21.fffe.7c3d10'
    assert ts['ingress_time'] == 1794422044714916883 and ts['_port'] == '48b02d.fffe.5a1c22-1'
    s = ptp.pmc_summary(p)
    assert s == {'gm_present': True, 'gm_identity': '001b21.fffe.7c3d10', 'offset_ns': -27.0,
                 'mean_path_delay_ns': 7345.0, 'steps_removed': 1, 'port_state': 'SLAVE',
                 'clock_identity': '48b02d.fffe.5a1c22'}
    assert ptp.parse_pmc('') == {} and ptp.pmc_summary({}) == {k: None for k in s}


def test_parse_ptp4l_and_phc2sys_lines():
    d = ptp.parse_ptp4l_line('ptp4l[5182.012]: master offset        -31 s2 freq    -540 path delay      7345')
    assert d == {'kind': 'offset', 'program': 'ptp4l', 't': 5182.012, 'offset_ns': -31, 'state': 's2',
                 'freq_ppb': -540, 'delay_ns': 7345, 'clock': 'phc'}
    d = ptp.parse_ptp4l_line('phc2sys[5185.000]: CLOCK_REALTIME phc offset       -15 s2 freq   +2345 delay    512')
    assert d['program'] == 'phc2sys' and d['clock'] == 'CLOCK_REALTIME' and d['offset_ns'] == -15 and \
        d['freq_ppb'] == 2345
    d = ptp.parse_ptp4l_line('ptp4l[5190.000]: rms   14 max   31 freq   -531 +/-   6 delay  7344 +/-   3')
    assert d['kind'] == 'rms' and d['rms_ns'] == 14 and d['max_ns'] == 31 and d['delay_ns'] == 7344
    d = ptp.parse_ptp4l_line('ptp4l[5181.700]: port 1 (enp1s0): UNCALIBRATED to SLAVE on MASTER_CLOCK_SELECTED')
    assert d['kind'] == 'port' and d['frm'] == 'UNCALIBRATED' and d['to'] == 'SLAVE'
    assert ptp.parse_ptp4l_line('master offset 5 s2 freq +1 path delay 9')['t'] is None     # without the prefix
    assert ptp.parse_ptp4l_line('ptp4l[1.0]: assuming the grand master role')['kind'] == 'grandmaster'
    assert ptp.parse_ptp4l_line('hello') is None
    s = ptp.summarise_log(LOG.splitlines())
    assert s['offsets'] == 4 and s['offset_abs_max_ns'] == 31 and s['port_state'] == 'SLAVE' and s['locked']
    assert s['grandmaster'] == '001b21.fffe.7c3d10' and s['rms_last_ns'] == 14
    s = ptp.summarise_log(LOG.splitlines()[:6])
    assert not s['locked'] and s['servo'] == 's1'


def test_parse_ethtool():
    hw = ptp.parse_ethtool_t(ETHTOOL_HW)
    assert hw['hardware'] and hw['software'] and hw['phc'] == 0 and hw['tx_modes'] == ['off', 'on']
    assert hw['rx_filters'] == ['none', 'all'] and 'hardware-raw-clock' in hw['capabilities']
    assert ptp.parse_ethtool_t(ETHTOOL_NEW)['phc'] == 2
    sw = ptp.parse_ethtool_t(ETHTOOL_SW)
    assert sw['software'] and not sw['hardware'] and sw['phc'] is None and sw['tx_modes'] == []
    wifi = ptp.parse_ethtool_t(ETHTOOL_SW.replace('\tsoftware-transmit\n', ''))
    assert not wifi['software'] and not wifi['hardware']


def test_ptp4l_configs():
    sw = ptp.ptp4l_conf('software', 'enp14s0', domain=3)
    assert 'time_stamping           software' in sw and 'domainNumber            3' in sw
    assert 'network_transport       UDPv4' in sw and sw.rstrip().endswith('[enp14s0]')
    g = ptp.ptp4l_conf('gptp', 'enp1s0')
    for line in ('transportSpecific       0x1', 'network_transport       L2', 'delay_mechanism         P2P',
                 'ptp_dst_mac             01:80:C2:00:00:0E', 'time_stamping           hardware',
                 'neighborPropDelayThresh 800', 'follow_up_info          1', 'phc2sys -s enp1s0'):
        assert line in g, line
    gs = ptp.ptp4l_conf('gptp', 'enp1s0', timestamping='software')
    assert 'neighborPropDelayThresh 20000000' in gs and 'NOT a conformant gPTP' in gs
    for bad in (('ptpv3', 'eth0'), ('software', 'eth0; rm -rf'), ('software', '')):
        with pytest.raises(ValueError):
            ptp.ptp4l_conf(*bad)
    cmds = ptp.commands('enp1s0', 'gptp')
    assert any('ptp4l -f /etc/linuxptp/falcon-gptp.conf' in c for c in cmds) and any('phc2sys' in c for c in cmds)


@pytest.fixture
def fake_linuxptp(tmp_path, monkeypatch):
    """pmc and ethtool stand-ins on PATH: pmc prints the answers above (or what PMC_TEXT holds)."""
    bin_ = tmp_path / 'bin'
    bin_.mkdir()
    text = tmp_path / 'pmc.txt'
    text.write_text(PMC)
    (bin_ / 'pmc').write_text(f'#!/bin/sh\necho "$@" >> {tmp_path}/pmc.args\ncat {text}\n')
    (bin_ / 'ethtool').write_text(f'#!/bin/sh\ncat <<EOF\n{ETHTOOL_HW}EOF\n')
    (bin_ / 'ptp4l').write_text('#!/bin/sh\nexit 0\n')
    for f in ('pmc', 'ethtool', 'ptp4l'):
        os.chmod(bin_ / f, stat.S_IRWXU)
    monkeypatch.setenv('PATH', f'{bin_}{os.pathsep}{os.environ["PATH"]}')
    return tmp_path, text


def test_query_pmc_and_doctor_with_stand_ins(fake_linuxptp, settings):
    tmp, _ = fake_linuxptp
    q = ptp.query_pmc()
    assert q['code'] == 0 and q['summary']['gm_present'] and q['summary']['offset_ns'] == -27.0
    args = (tmp / 'pmc.args').read_text()
    assert '-u -b 0 GET TIME_STATUS_NP GET PORT_DATA_SET GET CURRENT_DATA_SET' in args
    settings.set('hil_ptp', {'interface': 'enp1s0', 'clock_source': 'auto'})
    rows = {r[0]: r for r in ptp.doctor_rows(settings)}
    assert rows['pmc'][1] and rows['ptp4l'][1]
    assert rows['timestamping enp1s0'][1] and 'hardware (PHC 0)' in rows['timestamping enp1s0'][2]
    # the device's raw status is parsed the same way
    st = ptp.with_parsed({'text': PMC})
    assert st['summary']['port_state'] == 'SLAVE'


# ============================================================================================ the clock source
def _pairs(offset_ns, drift_ppm, n=30, jitter_ns=0.0, seed=0):
    """Realtime pairs from a device whose stamp clock is host monotonic × (1 + drift) + offset, both realtime clocks
    equal (PTP) and the host's realtime − monotonic constant."""
    rng = np.random.default_rng(seed)
    d_host = 1_700_000_000_000_000_000
    rows = []
    for k in range(n):
        h_mono = 5_000_000_000 + k * 1_000_000_000
        rt = h_mono + d_host
        d_ns = h_mono * (1 + drift_ppm * 1e-6) + offset_ns + rng.normal(0, jitter_ns)
        rows.append({'t': 'pair', 'h_rt': rt + 300_000, 'h_mono': h_mono + 300_000, 'd_rt': rt, 'd_ns': int(d_ns)})
    return rows


@pytest.mark.parametrize('offset_ns,drift_ppm', [(1.5e9, 20.0), (-3.2e12, -15.0), (0.0, 0.0)])
def test_realtime_pairs_recover_offset_and_drift(offset_ns, drift_ppm):
    est = ptp.realtime_estimate(_pairs(offset_ns, drift_ppm, jitter_ns=200.0), bound_ns=500.0)
    truth = offset_ns + drift_ppm * 1e-6 * est['t_ref_ns']
    assert abs(est['offset_ns'] - truth) < est['uncertainty_ns'] and abs(est['offset_ns'] - truth) < 1000
    assert est['drift_ppm'] == pytest.approx(drift_ppm, abs=0.01)
    assert est['uncertainty_ns'] >= 500.0 + ptp.READ_GAP_NS and est['trusted']
    # a device stamp maps back onto the host's line the same way as the pings' estimate
    h = 12_345_678_901
    d = h * (1 + drift_ppm * 1e-6) + offset_ns
    assert H.to_host(d, est) == pytest.approx(h, abs=1000)
    assert ptp.realtime_estimate([]) == {}


def _status(gm='001b21.fffe.7c3d10', offset=-27, state='SLAVE', present='true'):
    text = PMC.replace('001b21.fffe.7c3d10', gm).replace('-27\n', f'{offset}\n').replace('SLAVE', state) \
        .replace('gmPresent                  true', f'gmPresent                  {present}')
    return ptp.with_parsed({'text': text})


def test_when_ptp_may_be_used():
    assert ptp.ptp_basis(True, None, [])['ok']                         # a simulated device: shared realtime
    ok = ptp.ptp_basis(False, _status(), [_status(offset=40)])
    assert ok['ok'] and ok['bound_ns'] == 40.0 and ok['basis'] == 'ptp'
    for host, dev, why in ((None, [_status()], 'gave no answer'),
                           (_status(), [], 'no pmc status'),
                           (_status(), [_status(gm='aaaaaa.fffe.bbbbbb')], 'different grandmasters'),
                           (_status(present='false'), [_status()], 'no grandmaster present'),
                           (_status(state='LISTENING'), [_status()], 'port LISTENING'),
                           (_status(), [_status(offset=5_000_000)], 'offset 5000000')):
        b = ptp.ptp_basis(False, host, dev)
        assert not b['ok'] and why in b['why'], (why, b['why'])


def test_choosing_the_clock_source_and_their_disagreement():
    ntp = {'offset_ns': 1000.0, 't_ref_ns': 10_000, 'drift_ppm': 0.0, 'uncertainty_ns': 30_000.0, 'samples': 50,
           'min_rtt_ns': 1.0, 'median_rtt_ns': 2.0, 'trusted': True}
    rt = {'offset_ns': 6000.0, 't_ref_ns': 10_000, 'drift_ppm': 0.0, 'uncertainty_ns': 3000.0, 'samples': 9,
          'trusted': True, 'basis': 'ptp'}
    ok = {'ok': True, 'why': 'both synchronised to one grandmaster'}
    c = ptp.choose('auto', rt, ntp, ok)
    assert c['used'] == 'ptp' and c['estimate'] is rt and c['disagreement_ns'] == 5000.0 and c['consistent']
    assert ptp.choose('ntp-style', rt, ntp, ok)['used'] == 'ntp-style'
    c = ptp.choose('ptp', {}, ntp, {'ok': False, 'why': 'different grandmasters (a, b)'})
    assert c['used'] == 'ntp-style' and 'different grandmasters' in c['why'] and 'disagreement_ns' not in c
    far = dict(rt, offset_ns=100_000.0)
    assert not ptp.choose('auto', far, ntp, ok)['consistent']          # 99 µs apart, ±33 µs between them
    assert ptp.choose('auto', rt, {}, ok)['used'] == 'ptp'
    assert ptp.choose('bogus', {}, ntp, {})['preference'] == 'auto'


def test_session_files_give_both_sources(tmp_path):
    """core.hil.clock_sources from a run folder's ptp.jsonl (realtime pairs, pmc statuses) and clock.jsonl."""
    import json
    rows = _pairs(2.5e6, 10.0, n=12)
    rows.append({'t': 'host_pmc', 'h_mono': 1, 'status': {'text': PMC}})
    rows.append({'t': 'device_pmc', 'h_mono': 2, 'status': {'text': PMC.replace('-27\n', '55\n')}})
    with open(tmp_path / 'ptp.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    pings = []
    for k in range(60):                                                # the same clock seen through pings
        t0 = 5_000_000_000 + k * 200_000_000
        dev = lambda h: h * (1 + 10e-6) + 2.5e6                        # noqa: E731
        pings.append((t0, int(dev(t0 + 40_000)), int(dev(t0 + 50_000)), t0 + 90_000))
    cfg = {'device': {'mode': 'remote'}, 'clock_source': 'auto'}
    cs = H.clock_sources(str(tmp_path), cfg, pings)
    assert cs['used'] == 'ptp' and cs['ptp_basis']['bound_ns'] == 55.0 and cs['consistent'], cs
    assert cs['estimate'] is cs['ptp'] and cs['ntp_style']['samples'] == 60
    cfg['clock_source'] = 'ntp-style'
    assert H.clock_sources(str(tmp_path), cfg, pings)['used'] == 'ntp-style'
    # a device on another grandmaster: PTP is not used, its would-be estimate kept for the record
    rows[-1] = {'t': 'device_pmc', 'h_mono': 2, 'status': {'text': PMC.replace('001b21.fffe.7c3d10',
                                                                               'aaaaaa.fffe.bbbbbb')}}
    with open(tmp_path / 'ptp.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    cs = H.clock_sources(str(tmp_path), dict(cfg, clock_source='auto'), pings)
    assert cs['used'] == 'ntp-style' and cs['ptp'] is None and cs['ptp_basis']['estimate_if_used']
