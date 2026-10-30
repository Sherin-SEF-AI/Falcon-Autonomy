"""The assistant as the rest of Falcon sees it (Qt-free): its configuration, the key, the consent gate, the two models,
usage and cost, and the hooks the next features call (docs/assistant.md).

Two models: the main model (cfg main_model) runs the agent — planning, tool calls, reasoning over results; the fast
model (cfg fast_model, Qwen by default) answers one-shot helpers through quick(): a summary, "explain this error"
(assistant.explain), wording suggestions (assistant.suggest). Qwen is a Preview model on Groq: when the server's model
list lacks it, or a call says it is gone, the fast lane falls back to FAST_FALLBACK and says so once
(notice_listeners).

Tokens per minute: the responses of the last 60 s are counted (tpm_used) against the user's limit (cfg tpm_limit);
tpm_wait says how long a request must wait to fit.

ask() opens the Assistant panel with a request; propose() puts an approval card for a suggested action in it. The
panel registers itself as the handler of both (ask_handler, propose_handler).

The gate: no request leaves the machine — not a chat, not a model list, not a connection test — until the user has
turned the assistant on and accepted the notice (NOTICE, version NOTICE_VERSION). Every request is audited
(session.Audit) in the current workspace.
"""
from __future__ import annotations

import copy
import threading
import time

from ..core.settings import ASSISTANT_DEFAULTS
from .groq import DEFAULT_BASE_URL, GroqClient
from .keystore import KeyStore
from .provider import CancelToken, ProviderError, Usage, cost, model_params
from .session import Audit

NOTICE_VERSION = 2                        # 2: suggestions and Explain this error added to what is sent
FAST_FALLBACK = 'openai/gpt-oss-20b'      # production on Groq: used while the (Preview) fast model is not offered
NOTICE = """Falcon's assistant sends text to the model server you configure — Groq (api.groq.com) unless you set another \
— using your own API key. Groq's terms and privacy policy apply to what it receives.

What is sent, as text only:
• your messages, and text files you attach yourself;
• the conversation so far, including the results of the tools the assistant used: names, ids, settings and measured \
numbers of your project, runs, jobs, scenarios, models and evaluations, and passages of Falcon's manual;
• the descriptions of the tools it can use, and which page of Falcon is open;
• with suggestions on, the next steps Falcon found in your project — their titles and the facts behind them (names, \
ids and numbers of your runs, studies, suites, worlds and jobs) — for the fast model to rank and word;
• when you press Explain on a failed job: its title, what its exit means and the tail of its log, with passages of \
Falcon's manual.
Before anything is sent, your home folder is shortened to ~ and anything shaped like a password, token or key is \
replaced by [secret].

What is never sent: images, video, lidar, radar, audio, or anything from your dashcam footage — the tools refuse those \
sources, and Falcon refuses a message or an attachment that names them. Your key goes only to the server, in the \
request header.

Every request is listed in the workspace's audit log (time, model, token counts, tool names and a fingerprint of the \
payload — not the payload). You can turn the assistant off at any time in Settings ▸ Assistant."""


class AssistantService:
    def __init__(self, settings, keystore: KeyStore | None = None, workspace_fn=None):
        self.settings = settings
        self.keys = keystore or KeyStore()
        self.workspace_fn = workspace_fn or (lambda: '')
        self.ask_handler = None             # ask_handler(text, context): the panel
        self.propose_handler = None         # propose_handler(action): the panel
        self.usage_listeners: list = []     # fn(totals): the status bar
        self.notice_listeners: list = []    # fn(text): something the user should know once (a model fallen back)
        self._lock = threading.Lock()
        self.totals = {'prompt': 0, 'completion': 0, 'cost': 0.0, 'requests': 0, 'unpriced': 0, 'model': ''}
        self._window: list = []             # (monotonic time, tokens) of this minute's responses: the TPM in use
        self._gone: set = set()             # models the server said it does not have (this session)
        self._told: set = set()

    # ------------------------------------------------------------------ configuration
    def cfg(self) -> dict:
        c = copy.deepcopy(ASSISTANT_DEFAULTS)
        got = self.settings.get('assistant') or {}
        if isinstance(got, dict):
            c.update({k: v for k, v in got.items() if k in ASSISTANT_DEFAULTS})
        return c

    def set_cfg(self, save: bool = True, **kw):
        c = self.cfg()
        for k, v in kw.items():
            if k not in ASSISTANT_DEFAULTS:
                raise KeyError(f'unknown assistant setting {k!r}')
            c[k] = v
        self.settings.set('assistant', c, save=save)

    def consented(self) -> bool:
        return int(self.cfg().get('consent') or 0) >= NOTICE_VERSION

    def accept_notice(self):
        self.set_cfg(consent=NOTICE_VERSION, consent_time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))

    def ready(self) -> tuple:
        """(True, '') when a request may go; else (False, what is missing — a sentence for the empty state)."""
        c = self.cfg()
        if not c['enabled']:
            return False, 'The assistant is off. Turn it on in Settings ▸ Assistant.'
        if not self.consented():
            return False, 'Read and accept what the assistant sends before it sends anything.'
        if not self.keys.has():
            return False, 'No API key yet: get one at console.groq.com and paste it in Settings ▸ Assistant.'
        return True, ''

    def gate(self):
        c = self.cfg()
        if not c['enabled'] or not self.consented():
            raise ProviderError('consent', 'Nothing was sent: turn the assistant on and accept its notice in '
                                'Settings ▸ Assistant first.')

    # ------------------------------------------------------------------ the server
    def client(self, purpose: str = 'agent', key: str | None = None, base_url: str | None = None) -> GroqClient:
        c = self.cfg()
        k = key if key else self.keys.get()
        if not k and self.consented() and c['enabled']:
            raise ProviderError('no_key', 'No API key: paste yours in Settings ▸ Assistant.')
        return GroqClient(base_url or c['base_url'] or DEFAULT_BASE_URL, k, gate=self.gate,
                          on_request=lambda rec: self._audit(rec, base_url or c['base_url']))

    def _audit(self, rec: dict, base_url: str):
        ws = self.workspace_fn()
        if ws:
            import urllib.parse
            Audit(ws).write(dict(rec, base_url=urllib.parse.urlsplit(base_url or '').netloc))

    def models(self, key: str | None = None, base_url: str | None = None, cancel: CancelToken | None = None) -> list:
        """The server's live model list (and remember it for the pickers and context windows)."""
        cl = self.client('models', key, base_url)
        ms = cl.models(cancel)
        self.set_cfg(models=[{'id': m['id'], 'context_window': m.get('context_window')} for m in ms
                             if m.get('active', True) is not False],
                     models_checked=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        return ms

    def test_connection(self, key: str | None = None, base_url: str | None = None) -> str:
        ms = self.models(key, base_url)
        ids = {m['id'] for m in ms}
        c = self.cfg()
        parts = [f'Connected: the server offers {len(ids)} models.']
        for role, m in (('main', c['main_model']), ('fast', c['fast_model'])):
            parts.append(f'The {role} model {m} is ' + (
                'available.' if m in ids else f'NOT offered — Falcon uses {FAST_FALLBACK} instead.'
                if role == 'fast' and FAST_FALLBACK in ids else 'NOT offered — choose another.'))
        return ' '.join(parts)

    # ------------------------------------------------------------------ the fast model, and its fallback
    def listed(self) -> set:
        return {m['id'] for m in self.cfg().get('models') or [] if isinstance(m, dict) and m.get('id')}

    def fast_model(self) -> str:
        """The fast model to use: the chosen one, or FAST_FALLBACK when the server's last list (Refresh, Test
        connection) does not have it or a call this session said it is gone — told once (notice)."""
        m = self.cfg()['fast_model']
        if m == FAST_FALLBACK:
            return m
        listed = self.listed()
        if m in self._gone or (listed and m not in listed):
            self.notice(f'fallback:{m}', f'The fast model {m} is not offered by the server now (Groq lists it as '
                        f'Preview: it can go at short notice); Falcon uses {FAST_FALLBACK} instead. Settings ▸ '
                        'Assistant chooses another.')
            return FAST_FALLBACK
        return m

    def model_gone(self, model: str):
        self._gone.add(model)

    def notice(self, key: str, text: str):
        """Tell the user something once per session (the panel shows it, the log keeps it)."""
        with self._lock:
            if key in self._told:
                return
            self._told.add(key)
        for fn in list(self.notice_listeners):
            try:
                fn(text)
            except Exception:                  # noqa: BLE001
                pass

    def context_window(self, model: str):
        for m in self.cfg().get('models') or []:
            if isinstance(m, dict) and m.get('id') == model:
                return m.get('context_window')
        return None

    # ------------------------------------------------------------------ usage and cost
    def tpm_used(self, now: float | None = None) -> int:
        """Tokens of the responses in the last 60 s (prompt and completion, as the server counted them)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            self._window = [(t, n) for t, n in self._window if now - t < 60.0]
            return sum(n for _t, n in self._window)

    def tpm_wait(self, tokens: int, now: float | None = None) -> float:
        """Seconds until a request of about `tokens` fits the user's per-minute limit (0: now). A request larger than
        the limit itself never fits: the agent trims it first."""
        limit = int(self.cfg().get('tpm_limit') or 0)
        now = time.monotonic() if now is None else now
        if limit <= 0 or self.tpm_used(now) + tokens <= limit:
            return 0.0
        with self._lock:
            win = sorted(self._window)
        used = sum(n for _t, n in win)
        for t, n in win:                       # the oldest responses leave the window first
            used -= n
            if used + tokens <= limit:
                return max(0.0, 60.0 - (now - t)) + 0.05
        return 0.0

    def add_usage(self, model: str, usage: Usage):
        c = cost(self.cfg().get('prices') or {}, model, usage)
        with self._lock:
            self._window.append((time.monotonic(), usage.prompt + usage.completion))
            t = self.totals
            t['prompt'] += usage.prompt
            t['completion'] += usage.completion
            t['requests'] += 1
            t['model'] = model
            if c is None:
                t['unpriced'] += 1
            else:
                t['cost'] = round(t['cost'] + c, 6)
            snap = dict(t)
        snap['tpm_used'], snap['tpm_limit'] = self.tpm_used(), int(self.cfg().get('tpm_limit') or 0)
        for fn in list(self.usage_listeners):
            try:
                fn(snap)
            except Exception:                  # noqa: BLE001 (a listener's bug must not stop an answer)
                pass
        return c

    # ------------------------------------------------------------------ the hooks the next phase uses
    def quick(self, prompt: str, context=None, system: str = '', max_tokens: int = 700,
              cancel: CancelToken | None = None) -> str:
        """A one-shot answer from the fast model (blocking: call it off the GUI thread). context: text or a
        JSON-serialisable value the answer is about (an error, a run's summary) — redacted like everything sent."""
        import json
        model = self.fast_model()
        msgs = [{'role': 'system', 'content': system or (
            'You are Falcon\'s assistant (Falcon Autonomy: self-driving AI development on CARLA). Answer briefly and '
            'precisely from the context given; never invent numbers; say when the context does not hold the '
            'answer.')}]
        if context is not None:
            msgs.append({'role': 'user', 'content': 'Context:\n' + (context if isinstance(context, str) else
                                                                    json.dumps(context, default=str)[:12000])})
        msgs.append({'role': 'user', 'content': prompt})
        while True:
            try:
                res = self.client('quick').chat(msgs, model, stream=False, temperature=0.2,
                                                max_completion_tokens=max_tokens, extra=model_params(model, quick=True),
                                                cancel=cancel, purpose='quick')
                break
            except ProviderError as e:
                if e.kind != 'model' or model == FAST_FALLBACK:
                    raise
                self.model_gone(model)          # retired mid-session: the fallback, said once
                model = self.fast_model()
        self.add_usage(model, res.usage)
        return (res.content or '').strip()

    def explain_error(self, error_text: str, where: str = '') -> str:
        return self.quick('Explain this error in two or three sentences for the user, and the most likely fix.',
                          {'where': where, 'error': error_text[-4000:]})

    def ask(self, text: str, context=None):
        """Open the Assistant panel with a request (sent at once when the assistant is ready, else left in the box)."""
        if self.ask_handler is None:
            raise RuntimeError('the Assistant panel is not open')
        return self.ask_handler(text, context)

    def propose(self, action: dict):
        """An approval card for a suggested action: {'tool': name, 'args': {...}, 'why': 'one sentence'}."""
        if self.propose_handler is None:
            raise RuntimeError('the Assistant panel is not open')
        if not isinstance(action, dict) or not action.get('tool'):
            raise ValueError('an action is {"tool": name, "args": {...}, "why": text}')
        return self.propose_handler(action)
