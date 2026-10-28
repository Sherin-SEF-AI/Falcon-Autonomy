"""Putting the device agent on the device and starting it, over SSH with the user's key (standard library; the ssh and
scp programs of OpenSSH). The device's address, SSH user, key path and port live only in the user's settings
(core.hil.device_settings); no password is stored or asked for — Falcon uses keys only (`ssh-copy-id` once), with
BatchMode so that nothing ever waits for a prompt, and accepts a new host key on first contact (accept-new), never a
changed one. Messages name neither the address nor the user.

    copy(dev)          mkdir the remote folder, copy device_agent.py into it
    check(dev)         run `python3 device_agent.py --check` there: the device's Python, numpy, GStreamer, decoder,
                       model and L4T release (JSON)
    start(dev, once)   start the agent in the background (nohup), its log beside it; returns its pid
    stop(dev, pid)     stop it by its pid
    deploy(dev, ...)   copy, check and optionally start: what `falcon hil deploy` and the Hardware page call
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess

from . import device_agent as A

AGENT = os.path.abspath(A.__file__)
_HOST = re.compile(r'^[A-Za-z0-9][A-Za-z0-9.:_-]*$')       # a host name, IPv4 or IPv6 address (no option-like text)
_USER = re.compile(r'^[A-Za-z_][A-Za-z0-9._-]*$')


class DeployError(RuntimeError):
    pass


def validate(dev: dict, need_ssh: bool = True):
    if not dev.get('address') or not _HOST.match(str(dev['address'])):
        raise DeployError('the device\'s address is not set (or not a host name or IP address)')
    if not need_ssh:
        return
    if not dev.get('ssh_user') or not _USER.match(str(dev['ssh_user'])):
        raise DeployError('the device\'s SSH user is not set (or not a valid user name)')
    key = os.path.expanduser(str(dev.get('ssh_key') or ''))
    if not key or not os.path.isfile(key):
        raise DeployError('the SSH key file is not set or does not exist (Falcon uses keys only: create one with '
                          'ssh-keygen and install it with ssh-copy-id)')
    for tool in ('ssh', 'scp'):
        if shutil.which(tool) is None:
            raise DeployError(f'{tool} (OpenSSH) is not installed')


def _opts(dev: dict) -> list:
    return ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'IdentitiesOnly=yes', '-i', os.path.expanduser(str(dev['ssh_key']))]


def ssh_command(dev: dict, remote: str) -> list:
    return [shutil.which('ssh') or 'ssh', '-p', str(int(dev.get('ssh_port') or 22))] + _opts(dev) + \
        [f'{dev["ssh_user"]}@{dev["address"]}', remote]


def remote_dir(dev: dict) -> str:
    """The agent's folder as a shell word: ~/x becomes "$HOME"/x (quoted, still expanded)."""
    d = str(dev.get('remote_dir') or '~/falcon-hil')
    if d == '~':
        return '"$HOME"'
    if d.startswith('~/'):
        return '"$HOME"/' + shlex.quote(d[2:])
    return shlex.quote(d)


def run(cmd: list, timeout: float = 60.0) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as e:
        raise DeployError(f'the device did not answer within {timeout:.0f} s') from e
    except OSError as e:
        raise DeployError(f'cannot run {os.path.basename(cmd[0])}: {e}') from e


def _clean(text: str, dev: dict) -> str:
    """ssh's message without the address or the user."""
    for k in ('address', 'ssh_user'):
        if dev.get(k):
            text = text.replace(str(dev[k]), f'<{k}>')
    return text.strip()[-600:]


def ssh(dev: dict, remote: str, timeout: float = 60.0) -> str:
    r = run(ssh_command(dev, remote), timeout)
    if r.returncode != 0:
        why = 'SSH could not log in (is the key installed? ssh-copy-id)' if r.returncode == 255 else \
            f'the command failed on the device (exit {r.returncode})'
        raise DeployError(f'{why}: {_clean(r.stderr or r.stdout, dev)}')
    return r.stdout


def copy(dev: dict):
    validate(dev)
    d = remote_dir(dev)
    ssh(dev, f'mkdir -p {d}')
    rd = str(dev.get('remote_dir') or '~/falcon-hil')
    target = rd[2:] if rd.startswith('~/') else rd          # scp's paths are relative to the home folder
    cmd = [shutil.which('scp') or 'scp', '-P', str(int(dev.get('ssh_port') or 22))] + _opts(dev) + \
        ['-q', AGENT, f'{dev["ssh_user"]}@{dev["address"]}:{target.rstrip("/")}/device_agent.py']
    r = run(cmd, 120.0)
    if r.returncode != 0:
        raise DeployError(f'copying the agent failed: {_clean(r.stderr or r.stdout, dev)}')


def check(dev: dict) -> dict:
    """What the device has (device_agent.environment), by running the copied agent with --check."""
    validate(dev)
    py = shlex.quote(str(dev.get('python') or 'python3'))
    out = ssh(dev, f'{py} {remote_dir(dev)}/device_agent.py --check', 120.0)
    try:
        return json.loads(out[out.index('{'):])
    except ValueError as e:
        raise DeployError(f'the agent\'s check printed no JSON: {_clean(out, dev)}') from e


def start(dev: dict, once: bool = True, extra: list | None = None) -> int:
    """Start the agent in the background on the device; its pid. once: it exits after one session."""
    validate(dev)
    d = remote_dir(dev)
    py = shlex.quote(str(dev.get('python') or 'python3'))
    args = ['--port', str(int(dev.get('port') or A.DEFAULT_PORT))] + (['--once'] if once else []) + \
        [shlex.quote(str(x)) for x in extra or []]
    # not `cd && nohup … &`: that backgrounds the whole list, whose shell then holds ssh's output open
    out = ssh(dev, f'cd {d} || exit 3; nohup {py} device_agent.py {" ".join(args)} > agent.log 2>&1 < /dev/null & '
                   'echo $!')
    try:
        return int(out.strip().splitlines()[-1])
    except (ValueError, IndexError) as e:
        raise DeployError(f'the agent did not report its pid: {_clean(out, dev)}') from e


def stop(dev: dict, pid: int) -> bool:
    """Stop the agent by its pid (only that process); False when it had already ended."""
    validate(dev)
    r = run(ssh_command(dev, f'kill {int(pid)} 2>/dev/null && echo stopped || echo gone'), 30.0)
    return 'stopped' in (r.stdout or '')


def log_tail(dev: dict, lines: int = 40) -> str:
    validate(dev)
    return ssh(dev, f'tail -n {int(lines)} {remote_dir(dev)}/agent.log 2>/dev/null || true')


def deploy(dev: dict, start_agent: bool = False, once: bool = True) -> dict:
    """Copy and check (and start): {'env': ..., 'pid': ...}. Raises DeployError saying what failed."""
    copy(dev)
    env = check(dev)
    out = {'env': env, 'pid': None}
    problems = []
    if not env.get('numpy'):
        problems.append('numpy is missing (sudo apt install python3-numpy)')
    if not env.get('gstreamer'):
        problems.append('GStreamer\'s Python bindings are missing (sudo apt install python3-gi gir1.2-gstreamer-1.0): '
                        'only the raw camera transport will work')
    elif not env.get('decoder'):
        problems.append('no H.264 decoder element was found')
    out['problems'] = problems
    if start_agent:
        out['pid'] = start(dev, once)
    return out
