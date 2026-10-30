"""What a chat-completions provider returns and how it fails, Qt-free and standard library only (docs/assistant.md).

ChatResult: one completed model turn — its text, its tool calls (assembled from streamed deltas), why it stopped, and
the token usage. ProviderError: every failure as a kind and a sentence a user can act on (a bad key, a retired model, a
conversation too long, the network down) — never a traceback, never the key. Usage costs come from a price table the
user keeps in Settings ▸ Assistant (their configuration, not a claim of what the server charges).

model_params(model, effort, quick): the parameters each model family takes, as Groq documents them (verified
2026-10-30, docs/assistant.md "What was verified"): gpt-oss takes reasoning_effort low/medium/high and include_reasoning;
qwen3-32b takes reasoning_effort none/default; other Qwen 3 models take reasoning_format (parsed or hidden with tools,
never raw). Anything a server rejects is dropped and the request retried once (groq.GroqClient), so a local
OpenAI-compatible server that knows none of them still works.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

# kinds of failure: what the user is told, by kind (the server's own message is added where it helps)
KINDS = ('consent', 'no_key', 'auth', 'forbidden', 'model', 'context', 'too_large', 'no_tools', 'tool_use_failed',
         'bad_request', 'rate', 'server', 'network', 'cancelled', 'protocol')


class ProviderError(Exception):
    def __init__(self, kind: str, message: str, status: int | None = None, retry_after: float | None = None,
                 detail: str = ''):
        super().__init__(message)
        self.kind, self.message, self.status, self.retry_after, self.detail = kind, message, status, retry_after, detail

    def __str__(self):
        return self.message


class Cancelled(ProviderError):
    def __init__(self):
        super().__init__('cancelled', 'Stopped.')


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str = ''                     # the raw JSON text the model wrote (may be malformed)

    def wire(self) -> dict:
        return {'id': self.id, 'type': 'function', 'function': {'name': self.name, 'arguments': self.arguments}}


@dataclass
class Usage:
    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    def add(self, other: 'Usage'):
        self.prompt += other.prompt
        self.completion += other.completion

    @classmethod
    def of(cls, d) -> 'Usage':
        d = d if isinstance(d, dict) else {}
        return cls(int(d.get('prompt_tokens') or 0), int(d.get('completion_tokens') or 0))


@dataclass
class ChatResult:
    content: str = ''
    reasoning: str = ''
    tool_calls: list = field(default_factory=list)       # ToolCall
    finish_reason: str = ''
    usage: Usage = field(default_factory=Usage)
    model: str = ''
    attempts: int = 1

    def message(self) -> dict:
        """The assistant message as it goes back into the conversation."""
        m = {'role': 'assistant', 'content': self.content or None}
        if self.tool_calls:
            m['tool_calls'] = [t.wire() for t in self.tool_calls]
        return m


class CancelToken:
    """Stop for one turn: set() wakes every wait and closes the connection in flight (the client registers it)."""

    def __init__(self):
        self._ev = threading.Event()
        self._closers: list = []
        self._lock = threading.Lock()

    def set(self):
        self._ev.set()
        with self._lock:
            closers, self._closers = list(self._closers), []
        for c in closers:
            try:
                c()
            except Exception:              # noqa: BLE001 (a socket already gone)
                pass

    def is_set(self) -> bool:
        return self._ev.is_set()

    def wait(self, seconds: float) -> bool:
        return self._ev.wait(seconds)

    def on_cancel(self, closer):
        with self._lock:
            if not self._ev.is_set():
                self._closers.append(closer)
                return
        closer()

    def off(self, closer):
        with self._lock:
            if closer in self._closers:
                self._closers.remove(closer)


# --------------------------------------------------------------------------------------------- model families
REASONING_EFFORTS = ('low', 'medium', 'high')


def model_params(model: str, effort: str = 'medium', quick: bool = False) -> dict:
    """The reasoning parameters model takes (empty for a model Falcon knows nothing about). quick: the fast lane —
    the least reasoning the model offers, none shown."""
    m = (model or '').lower()
    if m.startswith('openai/gpt-oss'):
        return {'reasoning_effort': 'low' if quick else (effort if effort in REASONING_EFFORTS else 'medium'),
                'include_reasoning': False}
    if m == 'qwen/qwen3-32b':
        return {'reasoning_effort': 'none'} if quick else {'reasoning_format': 'hidden'}
    if m.startswith('qwen/qwen3'):             # qwen3.8-27b: none (its default: no reasoning tokens) … high
        return {'reasoning_effort': 'none', 'reasoning_format': 'hidden'} if quick else \
            {'reasoning_format': 'hidden', 'reasoning_effort': effort if effort in REASONING_EFFORTS else 'medium'}
    return {}


# parallel tool use, as Groq's tool-use page lists it (2026-10-30): gpt-oss takes tools one call at a time
PARALLEL_TOOLS = {'openai/gpt-oss': False, 'qwen/': True, 'llama-3': True, 'meta-llama/': True, 'minimaxai/': True,
                  'meta-models/': True}


def parallel_tools(model: str):
    """parallel_tool_calls for model: False for the models without parallel tool use (sent, so a call never fans out),
    True for those with it, None (left out) for a model Falcon knows nothing about."""
    m = (model or '').lower()
    for prefix, on in PARALLEL_TOOLS.items():
        if m.startswith(prefix):
            return on
    return None


OPTIONAL_PARAMS = ('reasoning_effort', 'include_reasoning', 'reasoning_format', 'stream_options',
                   'parallel_tool_calls')


def rejected_params(message: str, payload: dict) -> list:
    """The optional parameters a 400's message names (a server that does not know them)."""
    low = (message or '').lower()
    return [p for p in OPTIONAL_PARAMS if p in payload and (p in low or p.replace('_', ' ') in low)]


# --------------------------------------------------------------------------------------------- errors, by status
def classify(status: int, message: str, model: str, code: str = '') -> ProviderError:
    """An HTTP failure as a sentence the user can act on (message: the server's own, already scrubbed)."""
    low = (message or '').lower()
    said = f' The server said: {message.strip()[:300]}' if message and message.strip() else ''
    if status == 401:
        return ProviderError('auth', 'The server refused the API key (401). Check it in Settings ▸ Assistant — '
                             'a key from console.groq.com starts with gsk_.', status)
    if status == 403:
        return ProviderError('forbidden', 'The server refused this request for this key (403): the key\'s '
                             'organisation may not have access to ' + (model or 'this model') + '.' + said, status)
    if status == 404 or code == 'model_not_found' or 'model_not_found' in low or \
            ('model' in low and ('does not exist' in low or 'decommissioned' in low or 'not found' in low)):
        return ProviderError('model', f'The model {model or "?"} is not available (it may have been retired). '
                             'Choose another in Settings ▸ Assistant — Refresh lists what the server offers.' + said,
                             status)
    if status == 413:                          # a 429 over the TPM limit is a wait (retry-after), not this
        if 'tokens per minute' in low or 'tpm' in low or 'rate limit' in low:
            return ProviderError('too_large', 'This request is larger than your plan\'s tokens-per-minute limit on '
                                 'the server. Clear the conversation or wait a minute; the limit is on your account '
                                 'page at console.groq.com.' + said, status)
        return ProviderError('context', f'The conversation is too long for {model or "the model"}\'s context window. '
                             'Clear it (Clear in the Assistant panel) and ask again.' + said, status)
    if status == 400 and code == 'tool_use_failed' or 'tool_use_failed' in low or 'failed_generation' in low:
        return ProviderError('tool_use_failed', 'The model wrote a tool call the server could not parse.' + said,
                             status)
    if status == 400 and ('context' in low and ('length' in low or 'window' in low) or 'too long' in low or
                          'maximum context' in low or 'reduce the length' in low or 'context_length' in low):
        return ProviderError('context', f'The conversation is too long for {model or "the model"}\'s context window. '
                             'Clear it (Clear in the Assistant panel) and ask again.' + said, status)
    if status == 400 and 'tool' in low and ('not support' in low or 'unsupported' in low or 'does not support' in low):
        return ProviderError('no_tools', f'{model or "This model"} does not take tools, so it cannot use Falcon. '
                             'Choose a main model that does in Settings ▸ Assistant.' + said, status)
    if status == 429:
        return ProviderError('rate', 'The server\'s rate limit for this key was reached and retries did not clear '
                             'it. Wait a minute and ask again; your plan\'s limits are at console.groq.com.' + said,
                             status)
    if status in (498, 500, 502, 503, 504):
        return ProviderError('server', f'The model server is failing ({status}) and retries did not help. Try again '
                             'in a few minutes.' + said, status)
    return ProviderError('bad_request', f'The server rejected the request ({status}).' + said, status)


RETRY_STATUS = (429, 498, 500, 502, 503, 504)


def scrubber(*secrets):
    """A function that removes these exact strings (the key) from any text — on top of privacy.redact's patterns."""
    vals = [s for s in secrets if s and len(s) >= 6]

    def scrub(text: str) -> str:
        text = str(text)
        for s in vals:
            text = text.replace(s, '[secret]')
        return re.sub(r'gsk_[A-Za-z0-9]{8,}', '[secret]', text)
    return scrub


# --------------------------------------------------------------------------------------------- cost
def cost(prices: dict, model: str, usage: Usage) -> float | None:
    """US dollars for usage from the user's price table ({model: [input, output] per million tokens}); None when the
    table has no price for model."""
    p = (prices or {}).get(model)
    if not isinstance(p, (list, tuple)) or len(p) != 2:
        return None
    try:
        return usage.prompt / 1e6 * float(p[0]) + usage.completion / 1e6 * float(p[1])
    except (TypeError, ValueError):
        return None


def parse_prices(text: str) -> dict:
    """The price table as the user writes it: one 'model input output' per line ($ per million tokens), # comments."""
    out = {}
    for line in (text or '').splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        parts = line.replace(',', ' ').split()
        if len(parts) != 3:
            raise ValueError(f'"{line}": write a model and two prices (input, output per million tokens)')
        try:
            out[parts[0]] = [float(parts[1]), float(parts[2])]
        except ValueError:
            raise ValueError(f'"{line}": the prices must be numbers') from None
    return out


def format_prices(prices: dict) -> str:
    return '\n'.join(f'{m}  {p[0]:g}  {p[1]:g}' for m, p in sorted((prices or {}).items()))


def estimate_tokens(*objs) -> int:
    """A request's tokens before it is sent: its JSON's characters / 4 (an estimate; the server's count comes back)."""
    import json
    return sum(len(json.dumps(o, ensure_ascii=False)) for o in objs if o) // 4


def fmt_tokens(n: int) -> str:
    return f'{n / 1000:.1f}k' if n >= 1000 else str(n)
