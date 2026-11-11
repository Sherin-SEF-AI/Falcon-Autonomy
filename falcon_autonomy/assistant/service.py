"""The assistant as the rest of Falcon sees it (Qt-free): its configuration, the key, the consent gate, the provider and
its two models, usage and cost, and the hooks the next features call (docs/assistant.md).

Providers (cfg provider): an OpenAI-compatible server — Groq (the default) or a server of the user's own: vLLM,
Ollama, SGLang or any other (provider.PROFILES), all through groq.GroqClient — or the Claude API (anthropic_client
.AnthropicClient, models claude-opus-5-5 and claude-sonnet-5-5 only). A server on this machine or the local network
may be used without a key (provider.local_address); anything else needs one. The Groq-compatible key lives in the key
store under the assistant's name, the Claude key under anthropic_api_key (keystore.NAMES).

Configuration: cfg() is the effective configuration — with the Claude provider, main_model, fast_model, the token
limit, the model list and the server are Claude's (kept apart under cfg['anthropic'], so switching back finds the
Groq or local choices as they were); set_cfg() routes those keys to wherever the active provider keeps them. Claude's
own settings: effort ('' follows Falcon's reasoning effort; low … max), fallback (server-side refusal fallback, on by
default). Its prices are Anthropic's published ones (cache reads and writes included) unless the user's price table
names the model.

Two models: the main model (cfg main_model) runs the agent — planning, tool calls, reasoning over results; the fast
model (cfg fast_model, Qwen on Groq, Sonnet 5.5 on Claude) answers one-shot helpers through quick(): a summary,
"explain this error" (assistant.explain), wording suggestions (assistant.suggest). When the server's model list lacks
the fast model, or a call says it is gone, the fast lane falls back (Groq: FAST_FALLBACK; Claude: Opus 5.5; another
server: its main model) and says so once (notice_listeners).

Tokens per minute: the responses of the last 60 s are counted (tpm_used) against the user's limit (cfg tpm_limit; 0:
no pacing at all); tpm_wait says how long a request must wait to fit. Claude's cache reads do not count.

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
import urllib.parse

from ..core.settings import ASSISTANT_DEFAULTS
from .anthropic_client import ALLOWED as CLAUDE_MODELS
from .anthropic_client import DEFAULT_BASE_URL as CLAUDE_URL
from .anthropic_client import FAST_MODEL as CLAUDE_FAST
from .anthropic_client import MAIN_MODEL as CLAUDE_MAIN
from .anthropic_client import PRICES as CLAUDE_PRICES
from .anthropic_client import AnthropicClient, effort_of
from .groq import DEFAULT_BASE_URL, GroqClient
from .keystore import KeyStore
from .provider import (PROFILES, CancelToken, ProviderError, Usage, cost, local_address, model_params,
                       parallel_tools, profile)
from .session import Audit

NOTICE_VERSION = 2                        # 2: suggestions and Explain this error added to what is sent
FAST_FALLBACK = 'openai/gpt-oss-20b'      # production on Groq: used while the (Preview) fast model is not offered
CLAUDE_KEY = 'anthropic_api_key'
PROVIDERS = tuple(PROFILES) + ('anthropic',)
PROVIDER_TITLES = dict({k: p.title for k, p in PROFILES.items()}, anthropic='Claude (Anthropic)')
ANTHROPIC_DEFAULTS = {
    'main_model': CLAUDE_MAIN,
    'fast_model': CLAUDE_FAST,
    'effort': '',                         # '' : Falcon's reasoning effort; else low | medium | high | xhigh | max
    'fallback': True,                     # server-side refusal fallback (fallbacks "default"); off: a refusal stops
    'tpm_limit': 0,                       # no pacing unless the user sets their tier's limit
    'base_url': CLAUDE_URL,
    'models': [],
    'models_checked': '',
}
EXTRA_DEFAULTS = {'provider': 'groq', 'anthropic': ANTHROPIC_DEFAULTS,   # beside core.settings.ASSISTANT_DEFAULTS
                  'mcp_bridge': True}     # the window lets a local MCP client (Claude Code) in (ui.mcp_bridge)
SCOPED = ('main_model', 'fast_model', 'tpm_limit', 'models', 'models_checked', 'base_url')  # Claude keeps its own
NOTICE = """Falcon's assistant sends text to the model server you configure — Groq (api.groq.com) unless you choose \
another: Anthropic's Claude API (api.anthropic.com) or a server of your own (vLLM, Ollama, SGLang, any \
OpenAI-compatible one) — using your own API key. The terms and privacy policy of that server's provider apply to what \
it receives.

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


def claude_keys_for(keystore) -> object:
    """The Claude key's store beside a given store: the same secrets file and keyring choice, name anthropic_api_key."""
    secrets = getattr(keystore, 'secrets', None)
    if secrets is None:
        return KeyStore(name=CLAUDE_KEY)
    return KeyStore(secrets, use_keyring=getattr(keystore, 'kr', None) is not None, name=CLAUDE_KEY)


class AssistantService:
    def __init__(self, settings, keystore: KeyStore | None = None, workspace_fn=None, claude_keys=None):
        self.settings = settings
        self.keys = keystore or KeyStore()
        self.claude_keys = claude_keys if claude_keys is not None else claude_keys_for(self.keys)
        self.workspace_fn = workspace_fn or (lambda: '')
        self.ask_handler = None             # ask_handler(text, context): the panel
        self.propose_handler = None         # propose_handler(action): the panel
        self.usage_listeners: list = []     # fn(totals): the status bar
        self.notice_listeners: list = []    # fn(text): something the user should know once (a model fallen back)
        self._lock = threading.Lock()
        self.totals = {'prompt': 0, 'completion': 0, 'cost': 0.0, 'requests': 0, 'unpriced': 0, 'model': ''}
        self._window: list = []             # (monotonic time, tokens) of this minute's responses: the TPM in use
        self._gone: set = set()             # models the server said it does not have (this session)
        self.account_limits: dict = {}      # {'tokens', 'requests'} per minute, as the server last stated them
        self._told: set = set()

    # ------------------------------------------------------------------ configuration
    def stored(self) -> dict:
        """The configuration as kept: Falcon's defaults, the extra keys of the providers, the user's values."""
        c = copy.deepcopy(ASSISTANT_DEFAULTS)
        for k, v in EXTRA_DEFAULTS.items():
            c.setdefault(k, copy.deepcopy(v))
        got = self.settings.get('assistant') or {}
        if isinstance(got, dict):
            c.update({k: copy.deepcopy(v) for k, v in got.items() if k in c})
        a = copy.deepcopy(ANTHROPIC_DEFAULTS)
        if isinstance(c.get('anthropic'), dict):
            a.update({k: v for k, v in c['anthropic'].items() if k in ANTHROPIC_DEFAULTS})
        c['anthropic'] = a
        if c.get('provider') not in PROVIDERS:
            c['provider'] = 'groq'
        return c

    def cfg(self) -> dict:
        """The effective configuration: with the Claude provider, its models, limit, model list and server in the
        common keys."""
        c = self.stored()
        if c['provider'] == 'anthropic':
            c.update({k: copy.deepcopy(c['anthropic'][k]) for k in SCOPED})
        return c

    def set_cfg(self, save: bool = True, **kw):
        c = self.stored()
        claude = c['provider'] == 'anthropic' and kw.get('provider', 'anthropic') == 'anthropic'
        for k, v in kw.items():
            if k not in c:
                raise KeyError(f'unknown assistant setting {k!r}')
            if k == 'provider' and v not in PROVIDERS:
                raise ValueError(f'provider: one of {", ".join(PROVIDERS)}')
            if claude and k in SCOPED:
                if k == 'base_url' and (not v or v == DEFAULT_BASE_URL):
                    v = CLAUDE_URL               # the Groq default typed into the shared field means Anthropic's
                c['anthropic'][k] = v
            elif k == 'anthropic' and isinstance(v, dict):
                c['anthropic'].update({kk: vv for kk, vv in v.items() if kk in ANTHROPIC_DEFAULTS})
            else:
                c[k] = v
        self.settings.set('assistant', c, save=save)

    def provider(self) -> str:
        return self.cfg()['provider']

    def provider_title(self) -> str:
        return PROVIDER_TITLES.get(self.provider(), self.provider())

    def consented(self) -> bool:
        return int(self.cfg().get('consent') or 0) >= NOTICE_VERSION

    def accept_notice(self):
        self.set_cfg(consent=NOTICE_VERSION, consent_time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))

    def key_store(self, provider: str | None = None):
        """Where the key of provider (default: the active one) is kept."""
        return self.claude_keys if (provider or self.provider()) == 'anthropic' else self.keys

    def needs_key(self, c: dict | None = None) -> bool:
        """Whether the active provider needs a key: Claude and Groq always; a local profile's server only when it is
        not on this machine or the local network."""
        c = c or self.cfg()
        if c['provider'] == 'anthropic':
            return True
        return profile(c['provider']).needs_key or not local_address(c['base_url'] or '')

    def ready(self) -> tuple:
        """(True, '') when a request may go; else (False, what is missing — a sentence for the empty state)."""
        c = self.cfg()
        if not c['enabled']:
            return False, 'The assistant is off. Turn it on in Settings ▸ Assistant.'
        if not self.consented():
            return False, 'Read and accept what the assistant sends before it sends anything.'
        if c['provider'] == 'anthropic':
            if not self.claude_keys.has():
                return False, ('No Claude API key yet: create one at console.anthropic.com (API keys) and paste it in '
                               'Settings ▸ Assistant ▸ Provider.')
            try:
                import anthropic  # noqa: F401
            except ImportError:
                return False, ('The Claude provider needs Anthropic\'s Python package: pip install '
                               '"falcon-autonomy[claude]".')
            return True, ''
        if not self.keys.has():
            if c['provider'] == 'groq':
                return False, 'No API key yet: get one at console.groq.com and paste it in Settings ▸ Assistant.'
            if self.needs_key(c):
                host = urllib.parse.urlsplit(c['base_url'] or '').netloc or 'that server'
                return False, (f'No API key: {host} is not on this machine or your local network, and a server '
                               'elsewhere needs a key (Settings ▸ Assistant).')
        return True, ''

    def gate(self):
        c = self.cfg()
        if not c['enabled'] or not self.consented():
            raise ProviderError('consent', 'Nothing was sent: turn the assistant on and accept its notice in '
                                'Settings ▸ Assistant first.')

    # ------------------------------------------------------------------ the server
    def client(self, purpose: str = 'agent', key: str | None = None, base_url: str | None = None):
        """The active provider's client: AnthropicClient for Claude, GroqClient (with the server's profile) else."""
        c = self.cfg()
        url = base_url or c['base_url']
        if c['provider'] == 'anthropic':
            k = key if key else self.claude_keys.get()
            if not k and self.consented() and c['enabled']:
                raise ProviderError('no_key', 'No Claude API key: paste yours in Settings ▸ Assistant ▸ Provider.')
            a = c['anthropic']
            return AnthropicClient(k, url or CLAUDE_URL, gate=self.gate,
                                   on_request=lambda rec: self._audit(rec, url or CLAUDE_URL),
                                   fallback=bool(a.get('fallback', True)),
                                   effort=effort_of(str(a.get('effort') or ''), c.get('reasoning_effort', 'medium')))
        k = key if key else self.keys.get()
        require = profile(c['provider']).needs_key or not local_address(url or DEFAULT_BASE_URL)
        if not k and require and self.consented() and c['enabled']:
            raise ProviderError('no_key', 'No API key: paste yours in Settings ▸ Assistant.')
        cl = GroqClient(url or DEFAULT_BASE_URL, k, gate=self.gate,
                        on_request=lambda rec: self._audit(rec, url), profile_id=c['provider'], require_key=require)
        cl.on_limits = self.account_limits.update      # what the server says this account may use per minute
        return cl

    def params(self, model: str, quick: bool = False) -> dict:
        """The reasoning parameters sent with model on the active provider: Claude's effort (low for the fast lane);
        Groq's documented parameters per model family; on another server only the standard reasoning_effort of a
        gpt-oss model (anything a server does not know is dropped and retried once)."""
        c = self.cfg()
        if c['provider'] == 'anthropic':
            return {'effort': 'low' if quick else effort_of(str(c['anthropic'].get('effort') or ''),
                                                           c.get('reasoning_effort', 'medium'))}
        effort = c.get('reasoning_effort', 'medium')
        if profile(c['provider']).known_families:
            return model_params(model, effort, quick)
        if 'gpt-oss' in (model or '').lower():
            return {'reasoning_effort': 'low' if quick else effort}
        return {}

    def parallel(self, model: str):
        """parallel_tool_calls for model: Groq's known families only (another server, and Claude, decide alone)."""
        c = self.cfg()
        if c['provider'] != 'anthropic' and profile(c['provider']).known_families:
            return parallel_tools(model)
        return None

    def _audit(self, rec: dict, base_url: str):
        ws = self.workspace_fn()
        if ws:
            Audit(ws).write(dict(rec, base_url=urllib.parse.urlsplit(base_url or '').netloc))

    def models(self, key: str | None = None, base_url: str | None = None, cancel: CancelToken | None = None) -> list:
        """The server's live model list (and remember it for the pickers and context windows); Claude's filtered to
        the two Falcon uses."""
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
        claude = c['provider'] == 'anthropic'
        parts = [f'Connected: Anthropic offers {len(ids)} of the models Falcon uses ({", ".join(CLAUDE_MODELS)}).'
                 if claude else f'Connected: the server offers {len(ids)} models.']
        fb = self.fast_fallback()
        for role, m in (('main', c['main_model']), ('fast', c['fast_model'])):
            parts.append(f'The {role} model {m} is ' + (
                'available.' if m in ids else f'NOT offered — Falcon uses {fb} instead.'
                if role == 'fast' and fb in ids else 'NOT offered — choose another.'))
        return ' '.join(parts)

    # ------------------------------------------------------------------ the fast model, and its fallback
    def listed(self) -> set:
        return {m['id'] for m in self.cfg().get('models') or [] if isinstance(m, dict) and m.get('id')}

    def fast_fallback(self) -> str:
        c = self.cfg()
        if c['provider'] == 'anthropic':
            return CLAUDE_MAIN
        if c['provider'] == 'groq':
            return FAST_FALLBACK
        return c['main_model']

    def fast_model(self) -> str:
        """The fast model to use: the chosen one, or the provider's fallback when the server's last list (Refresh,
        Test connection) does not have it or a call this session said it is gone — told once (notice)."""
        c = self.cfg()
        m, fb = c['fast_model'], self.fast_fallback()
        if m == fb:
            return m
        listed = self.listed()
        if m in self._gone or (listed and m not in listed):
            if c['provider'] == 'groq':
                text = (f'The fast model {m} is not offered by the server now (Groq lists it as Preview: it can go at '
                        f'short notice); Falcon uses {fb} instead. Settings ▸ Assistant chooses another.')
            else:
                text = (f'The fast model {m} is not offered by the server now; Falcon uses {fb} instead. Settings ▸ '
                        'Assistant chooses another.')
            self.notice(f'fallback:{m}', text)
            return fb
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
        c = self.cfg()
        for m in c.get('models') or []:
            if isinstance(m, dict) and m.get('id') == model:
                return m.get('context_window')
        if c['provider'] not in ('groq', 'anthropic'):
            return profile(c['provider']).default_context
        return None

    # ------------------------------------------------------------------ usage and cost
    def tpm_used(self, now: float | None = None) -> int:
        """Tokens of the responses in the last 60 s (prompt and completion, as the server counted them; Claude's
        cache reads left out)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            self._window = [(t, n) for t, n in self._window if now - t < 60.0]
            return sum(n for _t, n in self._window)

    def tpm_wait(self, tokens: int, now: float | None = None) -> float:
        """Seconds until a request of about `tokens` fits the user's per-minute limit (0: now; a limit of 0 never
        waits). A request larger than the limit itself never fits: the agent trims it first."""
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

    def cost_of(self, model: str, usage: Usage, iterations=()):
        """$ for one response: per attempt when a fallback ran (each at its model's price), else usage at model's;
        None when a model has no price."""
        prices = dict(CLAUDE_PRICES, **(self.cfg().get('prices') or {}))     # the user's table first
        if iterations:
            parts = [cost(prices, m, u) for m, u in iterations]
            return None if any(p is None for p in parts) else sum(parts)
        return cost(prices, model, usage)

    def add_usage(self, model: str, usage: Usage, iterations=()):
        c = self.cost_of(model, usage, iterations)
        with self._lock:
            self._window.append((time.monotonic(), usage.paced))
            t = self.totals
            t['prompt'] += usage.prompt
            t['completion'] += usage.completion
            if usage.cache_read or usage.cache_write:
                t['cache_read'] = t.get('cache_read', 0) + usage.cache_read
                t['cache_write'] = t.get('cache_write', 0) + usage.cache_write
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
                                                max_completion_tokens=max_tokens,
                                                extra=self.params(model, quick=True), cancel=cancel, purpose='quick')
                break
            except ProviderError as e:
                if e.kind != 'model' or model == self.fast_fallback():
                    raise
                self.model_gone(model)          # retired mid-session: the fallback, said once
                model = self.fast_model()
        self.add_usage(res.model or model, res.usage, res.iterations)
        if res.finish_reason == 'refusal':
            raise ProviderError('refusal', refusal_text(res))
        return (res.content or '').strip()

    def complete(self, messages: list, model: str = '', max_tokens: int = 6000,
                 cancel: CancelToken | None = None, purpose: str = 'scenario') -> str:
        """One answer from the main model (or `model`) to messages Falcon composed itself — text only, e.g. a
        scenario from a description (core.scenario_text): blocking, off the GUI thread; usage counted."""
        model = model or self.cfg()['main_model']
        for m in messages:
            if not isinstance(m.get('content'), str):
                raise ProviderError('refused', 'Only text is ever sent to the model.')
        res = self.client(purpose).chat(messages, model, stream=False, temperature=0.2,
                                        max_completion_tokens=max_tokens, extra=self.params(model), cancel=cancel,
                                        purpose=purpose)
        self.add_usage(res.model or model, res.usage, res.iterations)
        if res.finish_reason == 'refusal':
            raise ProviderError('refusal', refusal_text(res))
        return (res.content or '').strip()

    def scenario_writer(self, description: str, cancel: CancelToken | None = None, model: str = ''):
        """complete(messages) -> str for core.scenario_text.write, with .model; the description is refused first when
        it names a private location (assistant.agent.compose: the dashcam footage is never read)."""
        from .agent import compose
        _text, why = compose(description)
        if why:
            raise ProviderError('refused', why)
        ok, why = self.ready()
        if not ok:
            raise ProviderError('consent', why)
        m = model or self.cfg()['main_model']

        def complete(messages):
            return self.complete(messages, m, cancel=cancel)
        complete.model = m
        return complete

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


def refusal_text(res) -> str:
    """A refused Claude answer as a sentence: the category and Anthropic's explanation, when given."""
    sd = res.stop_details or {}
    cat = sd.get('category')
    why = (sd.get('explanation') or '').strip()
    out = 'Claude declined this request' + (f' (category: {cat})' if cat else '') + '.'
    if why:
        out += f' Anthropic says: {why[:300]}'
    if res.request_id:
        out += f' (request {res.request_id})'
    return out
