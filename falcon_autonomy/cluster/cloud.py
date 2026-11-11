"""Cloud bursts: one study run on a rented GPU machine, priced first, approved by its estimate id, capped, and
terminated by an independent watchdog (standard library; docs/cloud.md).

Estimate. A study (failrate, sweep or search of a CARLA town, docs/cloud.md says why only these) becomes a study bundle
(bundle.make_study: kilobytes; dashcam-derived data refused). Its evaluations (low, expected, high) times this
machine's measured seconds per evaluation (core.budget; a stated prior until a drive was measured) give hours; the
provider's current price per hour gives dollars. The TTL is the expected hours times ttl_factor (Settings ▸ Compute,
1.5) plus the boot allowance; the most a burst can cost is the TTL at that price — the watchdog terminates it there.
The estimate is saved under an id: a hash of everything it depends on (the plan, the bundle, the provider, the GPU,
the price, the TTL), so any change — a new price included — is a different id.

Approval. `falcon cloud burst --approve ID` (or the assistant's cluster_spot_burst, which asks the user on a card in
every mode) re-prices the plan: when the id it gets is not ID, nothing is launched. The caps then refuse even an
approved burst: its worst case over max_usd_per_burst, or the month's spending (finished bursts' costs plus every
active burst's worst case) plus it over max_usd_month. Approvals older than APPROVAL_MAX_S are stale.

Lifecycle. ledger "intent" (before any launch) → an ephemeral Ed25519 key (this burst's only) → launch, ledger
"launched" → the watchdog timer started → wait for the machine → the bundle put on it → the study run there (no_render
for a criticality study: actor states need no GPU rendering, and a container may have no Vulkan) → its summary pulled
back into the workspace (study files, the ten most critical runs' drive.json, run.json and recorder log) → terminate
and confirm (the provider says terminated, or no longer knows the machine), ledger "terminated" with the cost. Any
error on the way still terminates. A terminate that fails is retried with backoff; when it keeps failing the burst
is marked terminate_failed, a red banner shows in Falcon (cloud/alert.json) and a desktop notification is sent; the
watchdog keeps trying every minute.

Watchdog (`falcon cloud watchdog`, run every 60 s by a systemd user timer — `systemd-run --user`, independent of
Falcon's process — started with the first burst and stopped when none is left): terminates a burst past its TTL,
silent for 10 minutes (its driver stopped updating it: Falcon closed, the laptop asleep), done, or over its approved
cost or the monthly cap. Orphans: every instance named falcon-burst-<id>-<deadline> at a provider whose key Falcon
has; one this machine's ledger does not know is terminated only once past the deadline in its name (another
machine's burst is never cut short); `falcon cloud orphans` lists them and Falcon's start sweeps them.

Keys. The providers' API keys are in the keyring (assistant.keystore names runpod_api_key, lambda_api_key), sent only
in the Authorization header, and scrubbed from every message, log line and file Falcon writes (scrub()).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from . import bundle as B

APPROVAL_MAX_S = 24 * 3600.0
BOOT_H = 0.25                         # pulling the image, starting CARLA: a quarter of an hour, counted every time
SILENT_S = 600.0
TIMER_UNIT = 'falcon-cloud-watchdog'
NAME_PREFIX = 'falcon-burst-'
ACTIVE = ('intent', 'launching', 'running', 'collecting', 'done', 'failed', 'terminating', 'terminate_failed')
FINAL = ('terminated', 'never_launched')
TERMINATE_TRIES = 4
PRIOR_S_PER_EVAL = 60.0               # seconds per scenario evaluation before this machine measured a drive


class CloudError(Exception):
    pass


# ================================================================================================ settings, keys
def cloud_settings(settings) -> dict:
    from ..core.settings import CLOUD_DEFAULTS
    d = dict(CLOUD_DEFAULTS)
    d.update(settings.get('cloud') or {})
    return d


KEY_NAMES = {'runpod': 'runpod_api_key', 'lambda': 'lambda_api_key'}


def provider_key(name: str) -> str:
    from ..assistant.keystore import KeyStore
    return KeyStore(name=KEY_NAMES[name]).get() or ''


def _known_keys() -> list:
    out = []
    for n in KEY_NAMES:
        try:
            k = provider_key(n)
        except Exception:                          # noqa: BLE001 (a locked keyring: nothing to scrub from it)
            k = ''
        if k:
            out.append(k)
    return out


_EXTRA_SECRETS: list = []                          # keys handed in directly (a provider made with one)


def scrub(text) -> str:
    """text with every provider key Falcon knows replaced by [secret] (and the assistant's credential patterns)."""
    s = str(text)
    for k in set(_known_keys() + _EXTRA_SECRETS):
        if k and len(k) >= 8:
            s = s.replace(k, '[secret]')
    from ..assistant.privacy import SECRET_PATTERNS
    for p in SECRET_PATTERNS:
        s = p.sub('[secret]', s)
    return s


# ================================================================================================ HTTP
class Provider:
    """A GPU cloud's API (runpod.RunPod, lambda_.Lambda). Every method raises CloudError with a scrubbed message."""
    name = ''
    ssh_user = 'root'
    base = ''

    def __init__(self, key: str, base: str = '', timeout: float = 20.0):
        if not key:
            raise CloudError(f'no {self.name} API key: add it in Settings ▸ Compute (kept in the keyring)')
        self._key = key
        _EXTRA_SECRETS.append(key)
        self.base = (base or self.base).rstrip('/')
        self.timeout = float(timeout)

    def _headers(self) -> dict:
        return {'Authorization': f'Bearer {self._key}', 'Content-Type': 'application/json',
                'Accept': 'application/json', 'User-Agent': 'falcon-autonomy-cloud'}

    def request(self, method: str, path: str, body=None) -> tuple:
        """(HTTP status, parsed JSON or None). Network failures raise CloudError; HTTP errors are returned."""
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
                status = r.status
        except urllib.error.HTTPError as e:
            raw, status = e.read() or b'', e.code
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise CloudError(scrub(f'{self.name}: cannot reach the API: {getattr(e, "reason", e)}')) from None
        try:
            doc = json.loads(raw) if raw.strip() else None
        except ValueError:
            doc = None
        return status, doc

    def error(self, what: str, status: int, doc) -> CloudError:
        return CloudError(scrub(f'{self.name}: {what} failed (HTTP {status}): {self.error_text(doc)}'))

    def error_text(self, doc) -> str:
        return json.dumps(doc)[:300] if doc is not None else 'no details'

    # implemented by each provider
    def prices(self) -> list:
        raise NotImplementedError

    def launch(self, name: str, gpu: str, public_key: str, image: str) -> dict:
        raise NotImplementedError

    def status(self, instance_id: str) -> dict:
        raise NotImplementedError

    def terminate(self, instance_id: str):
        raise NotImplementedError

    def instances(self) -> list:
        raise NotImplementedError

    def cleanup(self, launched: dict):
        """What the launch left besides the machine (Lambda: the burst's SSH key)."""

    def remote_command(self, args: str, image: str) -> str:
        return f'falcon-worker-daemon {args}'

    def price(self, gpu: str) -> dict:
        for p in self.prices():
            if p['gpu'] == gpu:
                if p.get('usd_per_hour') is None:
                    raise CloudError(f'{self.name}: no price for {gpu}')
                return p
        raise CloudError(f'{self.name} has no GPU {gpu!r} (falcon cloud estimate --list-gpus)')


def provider(name: str, key: str | None = None, settings=None) -> Provider:
    from . import lambda_, runpod
    cls = {'runpod': runpod.RunPod, 'lambda': lambda_.Lambda}.get(name)
    if cls is None:
        raise CloudError(f'no cloud provider {name!r} (runpod or lambda)')
    cs = cloud_settings(settings) if settings is not None else {}
    p = cls(provider_key(name) if key is None else key, base=str(cs.get(f'{name}_base_url') or ''))
    if name == 'runpod' and str(cs.get('runpod_cloud') or '').upper() in ('SECURE', 'COMMUNITY'):
        p.cloud = str(cs['runpod_cloud']).upper()
    return p


# ================================================================================================ storage
def cloud_dir() -> str:
    from ..core.paths import config_dir
    d = os.path.join(config_dir(), 'cloud')
    os.makedirs(d, mode=0o700, exist_ok=True)
    return d


def _write(path: str, doc: dict):
    from ..core.paths import atomic_write
    atomic_write(path, scrub(json.dumps(doc, indent=1, sort_keys=True)) + '\n')


def _read(path: str) -> dict | None:
    try:
        with open(path, encoding='utf-8') as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


class Ledger:
    """Append-only: one JSON line per event, written (and fsynced) before what it records happens when it is an
    intent. Never rewritten; a month's spending is read from it."""

    def __init__(self, path: str = ''):
        self.path = path or os.path.join(cloud_dir(), 'ledger.jsonl')
        self.lock = threading.Lock()

    def append(self, event: str, burst: str, now: float | None = None, **fields):
        rec = {'t': _iso(time.time() if now is None else now), 'event': event, 'burst': burst, **fields}
        line = scrub(json.dumps(rec, sort_keys=True)) + '\n'
        with self.lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, line.encode())
                os.fsync(fd)
            finally:
                os.close(fd)
        return rec

    def records(self) -> list:
        try:
            with open(self.path, encoding='utf-8') as fh:
                return [json.loads(x) for x in fh if x.strip()]
        except (OSError, ValueError):
            return []

    def month_spend(self, now: float | None = None) -> dict:
        """{'spent': finished bursts' costs this month, 'reserved': active bursts' worst cases, 'bursts': {id: …}}."""
        month = _iso(time.time() if now is None else now)[:7]
        by = {}
        for r in self.records():
            b = by.setdefault(r['burst'], {'month': r['t'][:7]})
            if r['event'] == 'intent':
                b.update(month=r['t'][:7], max_usd=float(r.get('max_usd') or 0.0), state='intent')
            elif r['event'] == 'launched':
                b['state'] = 'launched'
            elif r['event'] == 'terminated':
                b.update(state='terminated', cost=float(r.get('cost_usd') or 0.0))
            elif r['event'] == 'never_launched':
                b.update(state='never_launched', cost=0.0)
        spent = sum(b.get('cost', 0.0) for b in by.values() if b['month'] == month and 'cost' in b)
        reserved = sum(b.get('max_usd', 0.0) for b in by.values()
                       if b['month'] == month and b.get('state') in ('intent', 'launched'))
        return {'month': month, 'spent': round(spent, 4), 'reserved': round(reserved, 4), 'bursts': by}


def _iso(t: float) -> str:
    return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def burst_dir(bid: str) -> str:
    d = os.path.join(cloud_dir(), 'bursts', bid)
    os.makedirs(d, mode=0o700, exist_ok=True)
    return d


def load_state(bid: str) -> dict | None:
    return _read(os.path.join(cloud_dir(), 'bursts', bid, 'state.json'))


def save_state(st: dict):
    _write(os.path.join(burst_dir(st['burst']), 'state.json'), st)


def states() -> list:
    root = os.path.join(cloud_dir(), 'bursts')
    out = []
    for b in sorted(os.listdir(root)) if os.path.isdir(root) else ():
        s = load_state(b)
        if s:
            out.append(s)
    return out


def log(bid: str, text: str, echo=None):
    line = scrub(f'{_iso(time.time())} {text}')
    with open(os.path.join(burst_dir(bid), 'log.txt'), 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')
    if echo:
        echo(line)


# ================================================================================================ estimates
def plan_of(project, command: str, scenario: dict, options: dict, rig: dict, settings) -> dict:
    """The evaluations a study makes (low, expected, high) and this machine's measured seconds for one."""
    from ..core import budget
    from ..core.budget import Throughput
    from ..core.failrate import DEFAULTS as FD
    o = options
    if command == 'failrate':
        st, fin, ms = int(o.get('stage_size') or FD['stage_size']), int(o.get('final') or FD['final_size']), \
            int(o.get('max_stages') or FD['max_stages'])
        ev = {'low': st + fin, 'expected': st * min(3, ms) + fin, 'high': st * ms + fin}
    elif command == 'sweep':
        if o.get('grid'):
            from ..core import scenarios as SC
            n = 1
            for p in SC.search_space(scenario):
                n *= len(p['choices']) if 'choices' in p else int(o['grid'])
        else:
            n = int(o.get('samples') or 10)
        ev = {'low': n, 'expected': n, 'high': n}
    else:
        n = int(o.get('budget') or 40)
        ev = {'low': n, 'expected': n, 'high': n}
    fps = float(o.get('fps') or 20.0)
    steps = int(float(scenario.get('duration_s', 12)) * fps)
    t = Throughput.from_settings(settings.get('measured'))
    try:
        est = t.estimate(steps, budget.rig_mpx(rig), 20.0)
        sec, src = float(est['seconds']), str(est.get('source') or est.get('label') or 'measured')
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        sec, src = PRIOR_S_PER_EVAL, f'a prior of {PRIOR_S_PER_EVAL:g} s per evaluation (no drive measured yet)'
    return {'command': command, 'evaluations': ev, 'seconds_per_eval': round(max(sec, 1.0), 2), 'source': src,
            'steps': steps}


def make_estimate(plan: dict, bundle: dict, prov: Provider, gpu: str, settings, image: str,
                  now: float | None = None, save: bool = True) -> dict:
    cs = cloud_settings(settings)
    p = prov.price(gpu)
    rate = float(p['usd_per_hour'])
    hours = {k: round(v * plan['seconds_per_eval'] / 3600.0 + BOOT_H, 4) for k, v in plan['evaluations'].items()}
    usd = {k: round(h * rate, 2) for k, h in hours.items()}
    factor = max(float(cs.get('ttl_factor') or 1.5), 1.0)
    ttl_h = round(max(hours['expected'] * factor, hours['expected'] + BOOT_H), 3)
    body = {'plan': plan, 'bundle_sha256': B.sha(bundle), 'provider': prov.name, 'gpu': gpu,
            'usd_per_hour': rate, 'ttl_h': ttl_h, 'image': image}
    eid = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    est = dict(body, estimate_id=eid, usd=usd, hours=hours, max_usd=round(ttl_h * rate, 2),
               created=time.time() if now is None else now, spot=False, gpu_info=p,
               caps={'max_usd_per_burst': float(cs['max_usd_per_burst']), 'max_usd_month': float(cs['max_usd_month'])},
               bundle=bundle)
    est['cap_check'] = cap_check(est, settings, now)
    if save:
        os.makedirs(os.path.join(cloud_dir(), 'estimates'), mode=0o700, exist_ok=True)
        _write(os.path.join(cloud_dir(), 'estimates', f'{eid}.json'), est)
    return est


def load_estimate(eid: str) -> dict:
    if not eid or not all(c in '0123456789abcdef' for c in eid):
        raise CloudError('an estimate id is 16 hex digits (falcon cloud estimate prints it)')
    est = _read(os.path.join(cloud_dir(), 'estimates', f'{eid}.json'))
    if est is None:
        raise CloudError(f'no estimate {eid}: make one with falcon cloud estimate')
    return est


def cap_check(est: dict, settings, now: float | None = None) -> str:
    """'' when the caps allow the burst, else why not."""
    cs = cloud_settings(settings)
    per, month = float(cs['max_usd_per_burst']), float(cs['max_usd_month'])
    if est['max_usd'] > per:
        return (f'its worst case ${est["max_usd"]:.2f} (the TTL of {est["ttl_h"]:.2f} h at ${est["usd_per_hour"]:.2f}/h) '
                f'is over the cap of ${per:.2f} per burst (Settings ▸ Compute)')
    m = Ledger().month_spend(now)
    if m['spent'] + m['reserved'] + est['max_usd'] > month:
        return (f'this month ${m["spent"]:.2f} was spent and ${m["reserved"]:.2f} is reserved by running bursts: '
                f'${est["max_usd"]:.2f} more is over the monthly cap of ${month:.2f}')
    return ''


def approve(eid: str, settings, prov: Provider, now: float | None = None) -> dict:
    """The estimate, priced again: CloudError when it changed (another id), is stale or the caps refuse it."""
    old = load_estimate(eid)
    now = time.time() if now is None else now
    if now - float(old.get('created') or 0) > APPROVAL_MAX_S:
        raise CloudError(f'estimate {eid} is more than a day old: make a new one (prices change)')
    if prov.name != old.get('provider'):
        raise CloudError(f'estimate {eid} is for {old.get("provider")}, not {prov.name}')
    used = [r['burst'] for r in Ledger().records() if r.get('event') == 'intent' and r.get('estimate_id') == eid]
    if used:
        raise CloudError(f'estimate {eid} was already used by burst {used[0]}: an approval runs one burst; make a '
                         'new estimate')
    new = make_estimate(old['plan'], old['bundle'], prov, old['gpu'], settings, old['image'], now, save=False)
    if new['estimate_id'] != eid:
        raise CloudError(f'the price or the plan changed since estimate {eid}: nothing was launched. The new '
                         f'estimate is {new["estimate_id"]} (${new["usd"]["expected"]:.2f} expected, '
                         f'${new["max_usd"]:.2f} at most): approve that one')
    why = cap_check(new, settings, now)
    if why:
        raise CloudError(f'refused by the caps: {why}')
    return new


# ================================================================================================ keys
def ephemeral_key(d: str) -> tuple:
    """An Ed25519 key pair for one burst (OpenSSH format, the private half 0600 in the burst's folder):
    (private key path, public key line)."""
    priv = os.path.join(d, 'id_ed25519')
    try:
        from cryptography.hazmat.primitives import serialization as S
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        k = Ed25519PrivateKey.generate()
        pem = k.private_bytes(S.Encoding.PEM, S.PrivateFormat.OpenSSH, S.NoEncryption())
        pub = k.public_key().public_bytes(S.Encoding.OpenSSH, S.PublicFormat.OpenSSH).decode()
        fd = os.open(priv, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'wb') as fh:
            fh.write(pem)
    except ImportError:
        kg = shutil.which('ssh-keygen')
        if not kg:
            raise CloudError('an SSH key for the burst needs the cryptography package or ssh-keygen') from None
        subprocess.run([kg, '-q', '-t', 'ed25519', '-N', '', '-C', 'falcon-burst', '-f', priv], check=True,
                       capture_output=True)
        with open(priv + '.pub', encoding='ascii') as fh:
            pub = fh.read().strip()
    pub = pub.split()[0] + ' ' + pub.split()[1] + ' falcon-burst'
    with open(priv + '.pub', 'w', encoding='ascii') as fh:
        fh.write(pub + '\n')
    return priv, pub


# ================================================================================================ terminate
def terminate(prov: Provider, st: dict, by: str, ledger: Ledger, sleep=time.sleep, now=None,
              confirm_s: float = 120.0, notify=None) -> bool:
    """Terminate the burst's machine and confirm it (provider says terminated, or no longer knows it), retrying with
    backoff (2, 4, 8 s …); the ledger and the state say what happened. False when it is still not terminated: the
    burst is then terminate_failed, the alert file is written and a desktop notification sent."""
    iid = st.get('instance_id')
    bid = st['burst']
    if not iid:                    # never launched (a machine the provider made anyway is swept by its name: orphans)
        st['state'] = 'never_launched'
        save_state(st)
        ledger.append('never_launched', bid, now)
        _alerts()
        return True
    err = ''
    for attempt in range(TERMINATE_TRIES):
        try:
            prov.terminate(iid)
            t0 = time.monotonic()
            while True:
                s = prov.status(iid)
                if s['state'] in ('terminated', 'missing'):
                    break
                if time.monotonic() - t0 > confirm_s:
                    raise CloudError(f'still {s["state"]} {confirm_s:.0f} s after the terminate call')
                sleep(5.0)
            t_end = time.time() if now is None else now
            hours = max(t_end - float(st.get('launched_at') or t_end), 0.0) / 3600.0
            cost = round(hours * float(st.get('usd_per_hour') or 0.0), 4)
            st.update(state='terminated', terminated_by=by, cost_usd=cost, terminated_at=t_end)
            save_state(st)
            ledger.append('terminated', bid, now, instance_id=iid, cost_usd=cost, by=by, attempts=attempt + 1)
            try:
                prov.cleanup(st.get('launch') or {})
            except CloudError as e:
                log(bid, f'cleanup after terminate: {e}')
            _alerts()
            return True
        except CloudError as e:
            err = str(e)
            log(bid, f'terminate attempt {attempt + 1} failed: {err}')
            ledger.append('terminate_failed', bid, now, attempt=attempt + 1, error=err[:300], by=by)
            if attempt + 1 < TERMINATE_TRIES:
                sleep(2.0 * 2 ** attempt)
    st.update(state='terminate_failed', terminate_error=err[:300], terminate_failures=int(st.get(
        'terminate_failures') or 0) + 1)
    save_state(st)
    _alerts()
    (notify or desktop_notify)('Falcon: a cloud machine is NOT terminated',
                               f'burst {bid} on {st.get("provider")}: {err[:160]} — Falcon keeps trying every '
                               'minute; terminate it in the provider\'s console if this stays.')
    return False


def _alerts():
    """cloud/alert.json lists the bursts whose machine could not be terminated (the GUI's red banner reads it)."""
    bad = [s for s in states() if s.get('state') == 'terminate_failed']
    p = os.path.join(cloud_dir(), 'alert.json')
    if bad:
        _write(p, {'bursts': [{k: s.get(k) for k in ('burst', 'provider', 'instance_id', 'terminate_error')}
                              for s in bad],
                   'message': f'{len(bad)} cloud machine{"s" * (len(bad) > 1)} could not be terminated: Falcon '
                              'keeps trying every minute — check the provider\'s console'})
    else:
        try:
            os.remove(p)
        except OSError:
            pass


def alert() -> dict | None:
    return _read(os.path.join(cloud_dir(), 'alert.json'))


def desktop_notify(title: str, body: str):
    ns = shutil.which('notify-send')
    if ns and os.environ.get('QT_QPA_PLATFORM') != 'offscreen':
        try:
            subprocess.run([ns, '-u', 'critical', '-a', 'Falcon Autonomy', title, scrub(body)], timeout=10,
                           capture_output=True)
        except (OSError, subprocess.TimeoutExpired):
            pass


# ================================================================================================ the watchdog
def systemd_timer(action: str) -> bool:
    """Start ('start') or stop ('stop') the watchdog's systemd user timer (every 60 s, independent of Falcon)."""
    if action == 'stop':
        try:
            subprocess.run(['systemctl', '--user', 'stop', f'{TIMER_UNIT}.timer'], capture_output=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return True
    sr = shutil.which('systemd-run')
    if not sr:
        return False
    try:
        r = subprocess.run(['systemctl', '--user', 'is-active', f'{TIMER_UNIT}.timer'], capture_output=True,
                           text=True, timeout=20)
        if r.stdout.strip() == 'active':
            return True
    except (OSError, subprocess.TimeoutExpired):
        pass
    from ..sim.server import falcon_source
    env = [f'--setenv=PYTHONPATH={falcon_source()}']
    for k in ('XDG_CONFIG_HOME', 'XDG_CACHE_HOME'):
        if os.environ.get(k):
            env.append(f'--setenv={k}={os.environ[k]}')
    cmd = [sr, '--user', f'--unit={TIMER_UNIT}', '--on-active=60', '--on-unit-active=60',
           '--timer-property=AccuracySec=5s', '-p', 'MemoryMax=512M'] + env + \
          ['--', sys.executable, '-m', 'falcon_autonomy.cluster.cloudcli', 'watchdog']
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def reasons(st: dict, now: float, settings, month: dict | None = None) -> list:
    """Why the watchdog terminates this burst now ([] to leave it)."""
    if st.get('state') in FINAL:
        return []
    out = []
    if st.get('state') == 'terminate_failed':
        out.append('its terminate failed before')
    if st.get('deadline') and now > float(st['deadline']):
        out.append('past its TTL')
    if st.get('last_seen') and now - float(st['last_seen']) > SILENT_S:
        out.append(f'silent for {(now - float(st["last_seen"])) / 60:.0f} min')
    if st.get('state') in ('done', 'failed'):
        out.append(f'its study is {st["state"]}')
    if st.get('launched_at') and st.get('usd_per_hour'):
        cost = (now - float(st['launched_at'])) / 3600.0 * float(st['usd_per_hour'])
        if cost > float(st.get('max_usd') or 0) + 1e-9:
            out.append(f'over its approved cost (${cost:.2f} of ${float(st.get("max_usd") or 0):.2f})')
        cs = cloud_settings(settings)
        if month is not None and month['spent'] + cost > float(cs['max_usd_month']):
            out.append('over the monthly cap')
    return out


def watchdog_tick(settings=None, now: float | None = None, providers=None, ledger: Ledger | None = None,
                  sleep=time.sleep, notify=None, timer=systemd_timer, echo=None) -> list:
    """One pass: terminate what must be ([(burst, reasons, terminated?)]); the timer stops when nothing is left.
    providers: {name: Provider} (made from the keyring when not given)."""
    from ..core.settings import Settings
    settings = settings if settings is not None else Settings()
    now = time.time() if now is None else now
    ledger = ledger or Ledger()
    month = ledger.month_spend(now)
    out = []
    active = [s for s in states() if s.get('state') not in FINAL]
    for st in active:
        why = reasons(st, now, settings, month)
        if not why:
            continue
        try:
            prov = (providers or {}).get(st['provider']) or provider(st['provider'], settings=settings)
        except CloudError as e:
            log(st['burst'], f'watchdog: {e}')
            out.append((st['burst'], why, False))
            continue
        log(st['burst'], f'watchdog: terminating ({"; ".join(why)})', echo)
        ok = terminate(prov, st, 'watchdog', ledger, sleep=sleep, now=now, notify=notify)
        out.append((st['burst'], why, ok))
    if not [s for s in states() if s.get('state') not in FINAL] and timer is not None:
        timer('stop')
    return out


# ================================================================================================ orphans
def instance_name(bid: str, deadline: float) -> str:
    return f'{NAME_PREFIX}{bid}-{int(deadline)}'


def parse_name(name: str) -> tuple:
    """(burst id, deadline) from falcon-burst-<id>-<deadline>, or (None, None)."""
    if not str(name).startswith(NAME_PREFIX):
        return None, None
    rest = name[len(NAME_PREFIX):]
    bid, _, dl = rest.rpartition('-')
    return (bid, float(dl)) if bid and dl.isdigit() else (None, None)


ORPHAN_GRACE_S = 600.0


def orphans(settings=None, providers=None, now: float | None = None, terminate_them=False, ledger=None,
            sleep=time.sleep, notify=None) -> list:
    """Every Falcon-named instance at the providers whose key Falcon has, with whether it is an orphan: this machine's
    burst that is finished but still there, or past its TTL; another machine's (unknown here) only once past the
    deadline in its name plus ORPHAN_GRACE_S. terminate_them: terminate the orphans."""
    from ..core.settings import Settings
    settings = settings if settings is not None else Settings()
    now = time.time() if now is None else now
    ledger = ledger or Ledger()
    mine = {s['burst']: s for s in states()}
    out = []
    provs = providers if providers is not None else {}
    if providers is None:
        for n in KEY_NAMES:
            try:
                if provider_key(n):
                    provs[n] = provider(n, settings=settings)
            except CloudError:
                continue
    for pname, prov in provs.items():
        try:
            insts = prov.instances()
        except CloudError as e:
            out.append({'provider': pname, 'error': str(e)})
            continue
        for inst in insts:
            bid, deadline = parse_name(inst.get('name') or '')
            if bid is None or inst.get('state') in ('terminated', 'missing'):
                continue
            st = mine.get(bid)
            if st is not None:
                orphan = st.get('state') in FINAL or now > float(st.get('deadline') or deadline)
                why = 'finished here but still running' if st.get('state') in FINAL else 'past its TTL'
            else:
                orphan = now > deadline + ORPHAN_GRACE_S
                why = 'not this machine\'s burst, past the deadline in its name'
            row = {'provider': pname, 'instance_id': inst['id'], 'name': inst.get('name'), 'burst': bid,
                   'orphan': orphan, 'why': why if orphan else 'running within its deadline'}
            if orphan and terminate_them:
                s = st if st is not None and st.get('instance_id') == inst['id'] else {
                    'burst': bid, 'provider': pname, 'instance_id': inst['id'], 'state': 'running',
                    'launched_at': None, 'usd_per_hour': 0.0}
                if st is None:
                    ledger.append('orphan', bid, now, provider=pname, instance_id=inst['id'])
                row['terminated'] = terminate(prov, s, 'orphans', ledger, sleep=sleep, now=now, notify=notify)
            out.append(row)
    return out


def startup_check(settings=None, background=True):
    """At Falcon's start: the watchdog's pass and the orphan sweep, in a background thread, only when this machine
    has an active burst or a provider key (no network otherwise)."""
    def run():
        try:
            if any(s.get('state') not in FINAL for s in states()):
                watchdog_tick(settings)
            if any(_safe_key(n) for n in KEY_NAMES):
                orphans(settings, terminate_them=True)
        except Exception:                          # noqa: BLE001 (never in the way of Falcon starting)
            pass
    if background:
        threading.Thread(target=run, daemon=True, name='falcon-cloud-sweep').start()
    else:
        run()


def _safe_key(n) -> bool:
    try:
        return bool(provider_key(n))
    except Exception:                              # noqa: BLE001
        return False


# ================================================================================================ the burst
class SSHRunner:
    """Runs the study on the machine over SSH with the burst's own key: put the bundle, run oneshot (its lines go to
    on_line), pull the summary back as a tar. (host, port, user) come from the provider's status."""

    def __init__(self, prov: Provider, image: str):
        self.prov, self.image = prov, image

    def __call__(self, st: dict, bundle_bytes: bytes, on_line, key_path: str) -> tuple:
        from .transport import Connection, TransportError
        node = {'name': st['burst'], 'host': st['ip'], 'ssh_port': int(st.get('ssh_port') or 22),
                'ssh_user': self.prov.ssh_user, 'ssh_key': key_path,
                'known_hosts': os.path.join(burst_dir(st['burst']), 'known_hosts')}
        conn = Connection(node, timeout=20)
        t0 = time.monotonic()
        while True:
            try:
                conn.connect()
                break
            except TransportError:
                if time.monotonic() - t0 > 900:
                    raise CloudError('the machine never accepted SSH within 15 min') from None
                on_line('{"t":"log","level":"info","text":"waiting for SSH on the machine"}')
                time.sleep(10)
        try:
            conn.put(bundle_bytes, 'falcon-burst/bundle')
            cmd = self.prov.remote_command('oneshot --bundle falcon-burst/bundle --out falcon-burst/out', self.image)
            code = conn.stream(cmd, on_line, timeout=None)
            rc, tar, err = conn.exec('tar -C falcon-burst/out/summary -cf - .', timeout=900)
            if rc != 0:
                raise CloudError(f'the summary could not be read from the machine (study exit {code})')
            return code, tar
        finally:
            conn.close()


def burst(eid: str, workspace: str, settings=None, prov: Provider | None = None, runner=None, timer=systemd_timer,
          sleep=time.sleep, notify=None, echo=print, wait_ready_s: float = 1800.0, poll_s: float = 10.0) -> dict:
    """An approved burst, start to end (module doc). Returns its final state; CloudError before launch (refused)."""
    from ..core import protocol as CP
    from ..core.settings import Settings
    settings = settings if settings is not None else Settings()
    old = load_estimate(eid)
    prov = prov or provider(old['provider'], settings=settings)
    est = approve(eid, settings, prov)                       # raises: a changed price, a stale approval, the caps
    ledger = Ledger()
    bid = time.strftime('%Y%m%d%H%M%S') + '-' + hashlib.sha256(os.urandom(16)).hexdigest()[:6]
    now = time.time()
    st = {'burst': bid, 'estimate_id': eid, 'provider': prov.name, 'gpu': est['gpu'], 'image': est['image'],
          'usd_per_hour': est['usd_per_hour'], 'max_usd': est['max_usd'], 'ttl_h': est['ttl_h'], 'state': 'intent',
          'created': now, 'last_seen': now, 'study': est['bundle'].get('name'), 'workspace': workspace,
          'deadline': now + est['ttl_h'] * 3600.0}
    save_state(st)
    ledger.append('intent', bid, provider=prov.name, gpu=est['gpu'], estimate_id=eid, max_usd=est['max_usd'],
                  ttl_h=est['ttl_h'], usd_per_hour=est['usd_per_hour'])
    log(bid, f'approved estimate {eid}: ${est["usd"]["expected"]:.2f} expected, ${est["max_usd"]:.2f} at most, '
             f'TTL {est["ttl_h"]:.2f} h', echo)
    d = burst_dir(bid)
    try:
        key_path, pub = ephemeral_key(d)
        bundle_bytes = B.encode(est['bundle'])
        if len(bundle_bytes) > B.CLOUD_MAX:
            raise CloudError('the study bundle grew over its cap')
        st['state'] = 'launching'
        save_state(st)
        launched = prov.launch(instance_name(bid, st['deadline']), est['gpu'], pub, est['image'])
        st.update(instance_id=launched['id'], launch=launched, launched_at=time.time(), state='running',
                  last_seen=time.time())
        st['deadline'] = st['launched_at'] + est['ttl_h'] * 3600.0
        save_state(st)
        ledger.append('launched', bid, instance_id=launched['id'], usd_per_hour=est['usd_per_hour'])
        log(bid, f'launched {launched["id"]} on {prov.name} ({est["gpu"]})', echo)
        if timer is not None and not timer('start'):
            log(bid, 'the watchdog timer could not be started (systemd-run): the burst still ends itself, and '
                     'Falcon\'s next start sweeps it', echo)
        t0 = time.monotonic()
        while True:
            s = prov.status(launched['id'])
            st['last_seen'] = time.time()
            save_state(st)
            if s['state'] == 'running' and s.get('ip'):
                st.update(ip=s['ip'], ssh_port=s.get('ssh_port') or 22)
                break
            if s['state'] in ('terminated', 'missing', 'error'):
                raise CloudError(f'the machine ended before it was ready ({s["state"]})')
            if time.monotonic() - t0 > wait_ready_s:
                raise CloudError('the machine was not ready in time')
            sleep(poll_s)
        save_state(st)

        def on_line(line):
            st['last_seen'] = time.time()
            save_state(st)
            m = CP.parse_line(line)
            if m.get('t') == 'progress':
                log(bid, f'{100 * float(m.get("f") or 0):.0f} % {m.get("text", "")}', echo)
            elif m.get('t') == 'log':
                log(bid, m.get('text', ''), echo)
        run = runner or SSHRunner(prov, est['image'])
        code, tar = run(st, bundle_bytes, on_line, key_path)
        st['state'] = 'collecting'
        save_state(st)
        files = _import_summary(tar, workspace, est['bundle'], {'provider': prov.name, 'burst': bid, 'gpu': est['gpu'],
                                                                 'estimate_id': eid,
                                                                 'instance_id': launched['id']})
        st.update(state='done' if code == 0 else 'failed', study_exit=code, files=len(files))
        save_state(st)
        log(bid, f'study {"finished" if code == 0 else f"ended with exit {code}"}; {len(files)} files back in '
                 f'the workspace', echo)
    except Exception as e:                                   # noqa: BLE001 (whatever happened, terminate below)
        st['state'] = 'failed' if st.get('instance_id') else st['state']
        st['error'] = scrub(str(e))[:400]
        save_state(st)
        log(bid, f'burst failed: {scrub(e)}', echo)
    finally:
        terminate(prov, st, 'burst', ledger, sleep=sleep, notify=notify)
        if timer is not None and not [s for s in states() if s.get('state') not in FINAL]:
            timer('stop')
    return st


def _import_summary(tar: bytes, workspace: str, b: dict, origin: dict | None = None) -> list:
    """The pulled summary into WORKSPACE/studies/<name>/ (the study's files; runs/<key>/ of its top failures), the
    machine's workspace path replaced by this one; cloud.json says where it ran (origin: the provider, the burst's
    id, the GPU, the approved estimate's id and the instance) — what a GSN Solution bound to it records
    (core.gsn.where_ran)."""
    from . import protocol as W
    dest = os.path.join(workspace, 'studies', str(b['name']))
    names = W.unpack_tar(tar, dest, max_bytes=512 << 20)
    meta = _read(os.path.join(dest, 'workspace.json')) or {}
    if meta.get('workspace'):
        W.translate_paths(dest, names, meta['workspace'], os.path.abspath(workspace))
    with open(os.path.join(dest, 'cloud.json'), 'w', encoding='utf-8') as fh:
        json.dump(dict({'note': 'run on a cloud burst: the study\'s files and its ten most critical runs\' records '
                                'came back (not their frames)', 'no_render': bool(b.get('no_render'))},
                       **(origin or {})), fh, indent=1)
    return names
