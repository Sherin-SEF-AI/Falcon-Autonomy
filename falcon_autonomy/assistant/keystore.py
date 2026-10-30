"""Where the user's API key lives: the OS keyring when the `keyring` package is importable and has a usable backend,
else the user's secrets file ($XDG_CONFIG_HOME/FalconAutonomy/secrets.json, mode 0600; core.settings.Secrets). Never
in settings.json, a project, a run, a log, a report, the job history or the assistant's own files; the key is read
only when a request is made and is never shown — Settings shows where it is stored, not what it is."""
from __future__ import annotations

SERVICE = 'FalconAutonomy'
NAME = 'assistant_api_key'


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
    def __init__(self, secrets=None, use_keyring: bool = True):
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
                v = self.kr.get_password(SERVICE, NAME)
                if v:
                    return v
            except Exception:                  # noqa: BLE001 (a locked keyring: fall back to the file)
                pass
        return self.secrets.get(NAME) or ''

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
                self.kr.set_password(SERVICE, NAME, key)
                if self.secrets.has(NAME):     # one place only
                    self.secrets.set(NAME, None)
                return
            except Exception:                  # noqa: BLE001 (no keyring after all: the 0600 file)
                pass
        self.secrets.set(NAME, key)

    def remove(self):
        if self.kr is not None:
            try:
                self.kr.delete_password(SERVICE, NAME)
            except Exception:                  # noqa: BLE001 (nothing stored there)
                pass
        if self.secrets.has(NAME):
            self.secrets.set(NAME, None)
