"""Settings ▸ Assistant ▸ Provider (docs/assistant.md, "Providers"; docs/mcp.md): which model server the assistant
uses — Groq, a local vLLM, Ollama or SGLang server or any OpenAI-compatible one, or Anthropic's Claude API — and the
Claude provider's own settings: its key (kept in the key store as anthropic_api_key, never shown), the two models it
may use (claude-opus-5-5 for the agent, claude-sonnet-5-5 for the fast lane), its effort, and the server-side refusal
fallback. Also whether this window lets a local MCP client (Claude Code) in.

A lazily built section of the Settings page (ui.pages.settings.SECTIONS): section(page) builds it; its apply(settings)
is called by the page's Apply ('' : applied). Choosing a provider takes effect at once (the Assistant section above
then shows that provider's server, models and limit); the Claude settings are kept by Apply.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLineEdit, QMessageBox

from ..assistant.anthropic_client import ALLOWED, EFFORTS, FAST_MODEL, MAIN_MODEL
from ..assistant.provider import PROFILES
from ..assistant.service import PROVIDER_TITLES, PROVIDERS
from . import tasks
from .widgets import Form, Section, button, label

FALLBACK_TEXT = ('When Claude\'s safety classifiers decline a request, Anthropic runs it again on the model it '
                 'recommends for that kind of request (beta server-side-fallback-2026-07-01), and Falcon says which '
                 'model answered. Off: a declined request just stops with Anthropic\'s reason.')
NOTES = dict({k: p.note for k, p in PROFILES.items()},
             anthropic='Anthropic\'s Claude API: claude-opus-5-5 runs the agent, claude-sonnet-5-5 the fast lane. '
                       'A key '
                       'from console.anthropic.com ▸ API keys; prices are Anthropic\'s ($4 / $20 per million tokens '
                       'for Opus 5.5, $2 / $10 for Sonnet 5.5; cache reads $0.20 / $0.10).')


class ProviderSection(Section):
    def __init__(self, page):
        super().__init__('Provider', 'The model server the assistant uses: Groq, a server of your own on this machine '
                                     'or network (vLLM, Ollama, SGLang — no key needed there), or Anthropic\'s Claude '
                                     'API. What is sent is the same text-only conversation in every case.')
        self.page = page
        self.win = page.win
        svc = self.win.assistant
        c = svc.stored()
        f = Form()
        self.provider = QComboBox()
        for p in PROVIDERS:
            self.provider.addItem(PROVIDER_TITLES[p], p)
        self.provider.setCurrentIndex(PROVIDERS.index(c['provider']))
        self.provider.setAccessibleName('Provider')
        self.provider.currentIndexChanged.connect(self._provider_changed)
        f.row('Provider', self.provider)
        self.note = label('', muted=True, wrap=True)
        f.row('', self.note)
        a = c['anthropic']
        kr = QHBoxLayout()
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setAccessibleName('Claude API key')
        self.key.setProperty('mono', True)
        kr.addWidget(self.key, 1)
        self.test = button('Test connection', self.test_connection, tip='List the Claude models this key may use')
        kr.addWidget(self.test)
        self.remove = button('Remove', self.remove_key, tip='Delete the stored Claude key')
        kr.addWidget(self.remove)
        f.row('Claude API key', kr, 'saved by Apply; never shown')
        self.main = QComboBox()
        self.fast = QComboBox()
        for cb, cur in ((self.main, a['main_model']), (self.fast, a['fast_model'])):
            cb.addItems(list(ALLOWED))
            cb.setCurrentText(cur if cur in ALLOWED else (MAIN_MODEL if cb is self.main else FAST_MODEL))
        f.row('Claude main model', self.main, 'the agent (Opus 5.5 by default)')
        f.row('Claude fast model', self.fast, 'summaries, explaining an error, suggestions (Sonnet 5.5 by default)')
        self.effort = QComboBox()
        self.effort.addItem('as Reasoning effort above', '')
        for e in EFFORTS:
            self.effort.addItem(e, e)
        self.effort.setCurrentIndex(max(0, self.effort.findData(a.get('effort') or '')))
        f.row('Claude effort', self.effort, 'how much Claude thinks (output_config.effort, always sent; the fast '
              'lane uses low); Claude takes no temperature')
        self.fallback = QCheckBox('Refusal fallback')
        self.fallback.setChecked(bool(a.get('fallback', True)))
        self.fallback.setToolTip(FALLBACK_TEXT)
        f.row('Declined requests', self.fallback, FALLBACK_TEXT)
        self.mcp = QCheckBox('Let MCP clients work in this window')
        self.mcp.setChecked(bool(c.get('mcp_bridge', True)))
        self.mcp.setToolTip('Claude Code (or another MCP client) started with `falcon --mcp` works in this window: '
                            'each change one undo step "Claude · …". Local only (127.0.0.1), with a token only you can '
                            'read. docs/mcp.md')
        f.row('MCP', self.mcp, 'claude mcp add --scope user falcon -- python3 -m falcon_autonomy --mcp')
        self.status = label('', muted=True, wrap=True)
        f.row('', self.status)
        self.add(f)
        self._claude_rows = [self.key, self.test, self.remove, self.main, self.fast, self.effort, self.fallback]
        self._refresh()
        self._sync_page()

    # ------------------------------------------------------------------ state
    def current(self) -> str:
        return self.provider.currentData()

    def _refresh(self):
        svc = self.win.assistant
        claude = self.current() == 'anthropic'
        for w in self._claude_rows:
            w.setEnabled(claude)
        has = svc.claude_keys.has()
        self.key.setPlaceholderText(f'stored in {svc.claude_keys.where()} — type to replace' if has else
                                    'paste your Claude key (sk-ant-…)')
        self.remove.setEnabled(claude and has)
        self.note.setText(NOTES.get(self.current(), ''))

    def _provider_changed(self, _i=None):
        """A provider chosen takes effect at once: the Assistant section shows its server, models and limit."""
        svc = self.win.assistant
        new = self.current()
        old = svc.stored()
        url = old['base_url']
        defaults = {p.default_url for p in PROFILES.values() if p.default_url}
        svc.set_cfg(provider=new)
        prof = PROFILES.get(new)
        if prof is not None and prof.default_url and (not url or url in defaults):
            svc.set_cfg(base_url=prof.default_url)        # a known server's address, unless the user typed their own
        self._refresh()
        self._sync_page()
        if hasattr(self.page, '_assistant_changed'):
            self.page._assistant_changed()

    def _sync_page(self):
        """The Assistant section's fields for the provider now in effect (its Apply routes them there)."""
        p, c = self.page, self.win.assistant.cfg()
        claude = c['provider'] == 'anthropic'
        if hasattr(p, 'as_url'):
            p.as_url.setText(c['base_url'])
        for name, key in (('as_main', 'main_model'), ('as_fast', 'fast_model')):
            cb = getattr(p, name, None)
            if cb is None:
                continue
            ids = list(ALLOWED) if claude else [m['id'] for m in c.get('models') or [] if isinstance(m, dict)
                                                and m.get('id')]
            cb.clear()
            cb.addItems(ids if c[key] in ids else [c[key]] + ids)
            cb.setCurrentText(c[key])
        if hasattr(p, 'as_tpm'):
            p.as_tpm.setValue(int(c.get('tpm_limit') or 0))
        if hasattr(p, 'as_temp'):
            p.as_temp.setEnabled(not claude)
            p.as_temp.setToolTip('Claude takes no temperature' if claude else '')
        if hasattr(p, 'as_key'):
            p.as_key.setEnabled(not claude)
            if claude:
                p.as_key.clear()
                p.as_key.setPlaceholderText('the Claude key is set under Provider below')

    # ------------------------------------------------------------------ actions
    def _save_key(self) -> str:
        k = self.key.text().strip()
        if not k:
            return ''
        try:
            self.win.assistant.claude_keys.set(k)
        except ValueError as e:
            return f'The Claude key was not saved: {e}.'
        self.key.clear()
        self.win.log.append_line(f'assistant: Claude key saved in {self.win.assistant.claude_keys.where()}')
        return ''

    def test_connection(self):
        from ..assistant.provider import ProviderError
        from .assistant_dock import ensure_consent
        svc = self.win.assistant
        why = self._save_key()
        if why:
            self.status.setText('⚠ ' + why)
            return
        if not ensure_consent(self, svc):
            self.status.setText('Nothing was sent: the notice was not accepted.')
            return
        if not svc.claude_keys.has():
            self.status.setText('Paste your Claude key first.')
            return
        self.status.setText('asking Anthropic…')
        self._refresh()

        def err(e):
            self.status.setText(('⚠ ' + e.message) if isinstance(e, ProviderError) else f'⚠ {type(e).__name__}')
        tasks.run(self, svc.test_connection, lambda msg: (self.status.setText('✓ ' + msg), self._sync_page()), err)

    def remove_key(self):
        svc = self.win.assistant
        if QMessageBox.question(self, 'Remove the Claude key',
                                f'Delete the Claude API key stored in {svc.claude_keys.where()}?') \
                != QMessageBox.StandardButton.Yes:
            return
        svc.claude_keys.remove()
        self.status.setText('Claude key removed.')
        self._refresh()

    def apply(self, settings) -> str:
        """The page's Apply: the Claude settings, the key typed, the MCP bridge on or off ('' : done)."""
        del settings                           # the service writes through the same Settings, unsaved; Apply saves
        why = self._save_key()
        if why:
            return why
        svc = self.win.assistant
        svc.set_cfg(save=False, mcp_bridge=self.mcp.isChecked(),
                    anthropic={'main_model': self.main.currentText(), 'fast_model': self.fast.currentText(),
                               'effort': self.effort.currentData() or '', 'fallback': self.fallback.isChecked()})
        try:
            from .mcp_bridge import set_enabled
            set_enabled(self.win, self.mcp.isChecked())
        except Exception as e:                 # noqa: BLE001 (the setting is kept; the window says why)
            self.win.log.append_line(f'MCP bridge: {type(e).__name__}: {e}')
        self._refresh()
        return ''


def section(page):
    return ProviderSection(page)

