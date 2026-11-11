"""Where the user's API key lives: the OS keyring when the `keyring` package is importable and has a usable backend,
else the user's secrets file ($XDG_CONFIG_HOME/FalconAutonomy/secrets.json, mode 0600; core.settings.Secrets). Never
in settings.json, a project, a run, a log, a report, the job history or the assistant's own files; the key is read
only when a request is made and is never shown — Settings shows where it is stored, not what it is.

KeyStore(name=...) keeps another secret the same way (NAMES: the cloud providers' API keys for cluster bursts); the
default is the assistant's key. set() refuses an empty value; clear() forgets the secret wherever it is stored."""
from __future__ import annotations

SERVICE = 'FalconAutonomy'
NAME = 'assistant_api_key'
NAMES = (NAME, 'runpod_api_key', 'lambda_api_key')        # the secrets Falcon keeps (one keyring entry each)


def _keyring():
    try:
        import keyring
        from keyring.backends import fail
    except Exception:                          # noqa: BLE001 (not installed, or a broken backend import)
        return None
    try:
        if isinstance(keyring.get_keyring(), fail.Keyring):
            return None
    except Exception:                          # noqa: BLE001
        return None
    return keyring


class KeyStore:
    def __init__(self, secrets=None, use_keyring: bool = True, name: str = NAME):
        if name not in NAMES:
            raise ValueError(f'unknown secret {name!r}: one of {", ".join(NAMES)}')
        self.name = name
        if secrets is None:
            from ..core.settings import Secrets
            secrets = Secrets()
        self.secrets = secrets
        self.kr = _keyring() if use_keyring else None

    def where(self) -> str:
        return 'the system keyring' if self.kr is not None else 'your secrets file (readable only by you)'

    def get(self) -> str:
        if self.kr is not None:
            try:
                v = self.kr.get_password(SERVICE, self.name)
                if v:
                    return v
            except Exception:                  # noqa: BLE001 (a locked keyring: fall back to the file)
                pass
        return self.secrets.get(self.name) or ''

    def has(self) -> bool:
        return bool(self.get())

    def set(self, key: str):
        key = (key or '').strip()
        if not key:
            raise ValueError('the key is empty')
        if any(c.isspace() for c in key):
            raise ValueError('a key has no spaces in it')
        if self.kr is not None:
            try:
                self.kr.set_password(SERVICE, self.name, key)
                if self.secrets.has(self.name):     # one place only
                    self.secrets.set(self.name, None)
                return
            except Exception:                  # noqa: BLE001 (no keyring after all: the 0600 file)
                pass
        self.secrets.set(self.name, key)

    def remove(self):
        if self.kr is not None:
            try:
                self.kr.delete_password(SERVICE, self.name)
            except Exception:                  # noqa: BLE001 (nothing stored there)
                pass
        if self.secrets.has(self.name):
            self.secrets.set(self.name, None)

    def clear(self):
        """Forget the secret wherever it is stored (the only way to empty it: set() refuses an empty value)."""
        self.remove()
