"""Settings: where CARLA is, Falcon's ports, the server's memory cap and quality, the render budget, the theme.

Machine settings, kept in the user's config folder ($XDG_CONFIG_HOME/FalconAutonomy), never in a project. Secrets
(a Jetson's credentials) are set where they are used and stored apart, never shown."""
from __future__ import annotations

from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLineEdit,
                             QMessageBox, QSpinBox)

from ...core.paths import tilde
from ...core.settings import RESERVED_PORTS
from ...sim import server as S
from .. import tasks
from ..widgets import Form, Page, Section, button, label


class SettingsPage(Page):
    key, title, icon = 'settings', 'Settings', 'gear'
    about = 'Machine settings: kept in your config folder, never in a project.'

    def __init__(self, win):
        super().__init__(win)
        st = win.settings
        self.fields: list = []           # (label, note, field) of every row: the command palette searches them
        v = self.page_layout(note=f'{self.about} ({tilde(st.path)})')

        carla = Section('CARLA')
        f = Form()
        self.root = QLineEdit(st.get('carla_root'))
        self.root.setPlaceholderText(tilde(win.service.carla_root) + ' (found)' if win.service.carla_root else
                                     'not found')
        rb = QHBoxLayout()
        rb.addWidget(self.root, 1)
        rb.addWidget(button('Browse…', self.browse_root))
        self._row(f, 'CARLA folder', rb, 'holds CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping')
        self.py = QLineEdit(st.get('client_python'))
        self.py.setPlaceholderText(tilde(S.client_python(win.service.carla_root)) or 'a venv beside the CARLA folder')
        pb = QHBoxLayout()
        pb.addWidget(self.py, 1)
        pb.addWidget(button('Check', self.check_python))
        self._row(f, 'CARLA\'s Python', pb, 'the venv with carla==0.9.16, numpy and OpenCV (workers run in it)')
        self.py_status = label('', muted=True, wrap=True)
        self._row(f, '', self.py_status)
        carla.add(f)
        v.addWidget(carla)

        srv = Section('Falcon\'s server', 'Falcon starts its own headless server on its own ports, capped in memory with '
                                         'no swap. Ports 2000–2010 and 8100 (CARLA\'s defaults, where your own server '
                                         'usually listens) are used only if you choose them here — Falcon then '
                                         'connects to your server and never starts or stops it.')
        g = Form()
        self.port = QSpinBox()
        self.port.setRange(1024, 65533)
        self.port.setValue(int(st.get('rpc_port')))
        self._row(g, 'RPC port', self.port, 'CARLA also listens on the next two ports')
        self.tm_port = QSpinBox()
        self.tm_port.setRange(1024, 65535)
        self.tm_port.setValue(int(st.get('tm_port')))
        self._row(g, 'Traffic manager port', self.tm_port)
        self.mem = QDoubleSpinBox()
        self.mem.setRange(4, 64)
        self.mem.setSuffix(' GB')
        self.mem.setValue(float(st.get('server_memory_gb')))
        self._row(g, 'Memory cap', self.mem, 'a 0.9.16 server capped at 12 GB ran 77 Mpx per step and crashed at 306')
        self.quality = QComboBox()
        self.quality.addItems(['Epic', 'Low'])
        self.quality.setCurrentText(st.get('server_quality'))
        self._row(g, 'Quality', self.quality)
        self.free_gpu = QCheckBox('Stop the idle server before a training or other GPU job')
        self.free_gpu.setChecked(bool(st.get('free_gpu_for_jobs')))
        self._row(g, 'GPU jobs', self.free_gpu, 'an idle 0.9.16 server holds about 6 GB of video memory')
        self.budget = QDoubleSpinBox()
        self.budget.setRange(1, 2000)
        self.budget.setSuffix(' Mpx')
        self.budget.setValue(float(st.get('render_budget_mpx')))
        self._row(g, 'Render budget', self.budget, 'runs rendering more per step are refused before they start')
        srv.add(g)
        v.addWidget(srv)

        look = Section('Appearance, privacy and notifications')
        h = Form()
        self.theme = QComboBox()
        self.theme.addItems(['dark', 'light'])
        self.theme.setCurrentText(st.get('theme'))
        self.theme.activated.connect(lambda _: win.set_theme(self.theme.currentText()))
        self._row(h, 'Theme', self.theme)
        self.geo = QCheckBox('Allow place-name search over the network (OpenStreetMap Nominatim)')
        self.geo.setChecked(bool(st.get('geocoding_allowed')))
        self._row(h, 'Geocoding', self.geo, 'off: imports work from files and bounding boxes only')
        self.notify = QCheckBox('Notify me when a job of 20 s or more ends while Falcon is in the background')
        self.notify.setChecked(bool(st.get('notify_jobs')))
        self._row(h, 'Job notifications', self.notify, 'through the desktop\'s notifications; nothing leaves this '
                  'machine')
        look.add(h)
        v.addWidget(look)
        self._assistant_section(v)
        b = QHBoxLayout()
        b.addWidget(button('Apply', self.apply, role='primary'))
        b.addStretch(1)
        v.addLayout(b)
        v.addStretch(1)

    # ------------------------------------------------------------------ the assistant (docs/assistant.md)
    def _assistant_section(self, v):
        from PyQt6.QtWidgets import QPlainTextEdit, QToolButton
        from ...assistant.agent import MODE_TEXT, MODES
        from ...assistant.provider import REASONING_EFFORTS, format_prices
        from .. import icons
        svc = self.win.assistant
        c = svc.cfg()
        sec = Section('Assistant', 'An agent that uses Falcon through its own tools, on a model server with your own '
                                   'API key (Groq by default: get a key at console.groq.com). It sends text only — '
                                   'never images, video, lidar, audio or anything from your dashcam footage — and '
                                   'nothing at all before you turn it on and accept its notice. The key is kept in '
                                   'the system keyring or your secrets file, never in a project, a log or a report.')
        self.assistant_section = sec
        f = Form()
        self.as_on = QCheckBox('Use the assistant')
        self.as_on.setChecked(bool(c['enabled']))
        self._row(f, 'Assistant', self.as_on, 'off: nothing is ever sent')
        kr = QHBoxLayout()
        self.as_key = QLineEdit()
        self.as_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.as_key.setAccessibleName('API key')
        self.as_key.setProperty('mono', True)
        self.as_eye = QToolButton()
        self.as_eye.setCheckable(True)
        self.as_eye.setIcon(icons.icon('eye', 14))
        self.as_eye.setToolTip('Show or hide what you type (a stored key is never shown)')
        self.as_eye.toggled.connect(lambda on: (self.as_key.setEchoMode(
            QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password),
            self.as_eye.setIcon(icons.icon('eye-off' if on else 'eye', 14))))
        kr.addWidget(self.as_key, 1)
        kr.addWidget(self.as_eye)
        kr.addWidget(button('Test connection', self.assistant_test))
        self.as_remove = button('Remove', self.assistant_remove_key, tip='Delete the stored key')
        kr.addWidget(self.as_remove)
        self._row(f, 'API key', kr, 'saved by Apply')
        self.as_url = QLineEdit(c['base_url'])
        self.as_url.setPlaceholderText('https://api.groq.com/openai/v1')
        self._row(f, 'Server', self.as_url, 'any OpenAI-compatible chat-completions API')
        mr = QHBoxLayout()
        self.as_main = QComboBox()
        self.as_fast = QComboBox()
        for cb, cur in ((self.as_main, c['main_model']), (self.as_fast, c['fast_model'])):
            cb.setEditable(True)
            cb.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            ids = [m['id'] for m in c.get('models') or [] if isinstance(m, dict) and m.get('id')]
            cb.addItems(ids if cur in ids else [cur] + ids)
            cb.setCurrentText(cur)
        mr.addWidget(self.as_main, 1)
        mr.addWidget(button('Refresh', self.assistant_refresh, tip='List the models the server offers now'))
        self._row(f, 'Main model', mr, 'plans, calls Falcon\'s tools, reasons over results')
        self._row(f, 'Fast model', self.as_fast, 'short answers: summaries, explaining an error; a Qwen model '
                  '(Preview on Groq) falls back to openai/gpt-oss-20b while the server does not offer it')
        self.as_tpm = QSpinBox()
        self.as_tpm.setRange(0, 10_000_000)
        self.as_tpm.setSingleStep(1000)
        self.as_tpm.setSuffix(' tokens/min')
        self.as_tpm.setSpecialValueText('no limit')
        self.as_tpm.setValue(int(c.get('tpm_limit') or 0))
        self._row(f, 'Token limit', self.as_tpm, 'your plan\'s tokens per minute (Groq\'s base developer plan: '
                  '8000) — requests are trimmed and paced to it; your account\'s limits page at console.groq.com '
                  'is authoritative')
        self.as_temp = QDoubleSpinBox()
        self.as_temp.setRange(0.0, 2.0)
        self.as_temp.setSingleStep(0.1)
        self.as_temp.setDecimals(1)
        self.as_temp.setValue(float(c['temperature']))
        self._row(f, 'Temperature', self.as_temp, 'lower is steadier')
        self.as_effort = QComboBox()
        self.as_effort.addItems(list(REASONING_EFFORTS))
        self.as_effort.setCurrentText(c['reasoning_effort'])
        self._row(f, 'Reasoning effort', self.as_effort, 'for models that take it (gpt-oss, Qwen 3)')
        self.as_mode = QComboBox()
        for m in MODES:
            self.as_mode.addItem(MODE_TEXT[m], m)
        self.as_mode.setCurrentIndex(MODES.index(c['mode']) if c['mode'] in MODES else 0)
        self._row(f, 'Permission mode', self.as_mode, 'Ask: every change waits for you; trash always asks')
        self.as_suggest = QCheckBox('Suggest next steps')
        self.as_suggest.setChecked(bool(c.get('suggestions', True)))
        self.as_suggest.setToolTip('One-click chips on the pages and the next best action on Home, found from the '
                                   'project\'s own state and files; worded by the fast model when the assistant is on '
                                   '(every number checked), else in Falcon\'s own words — they work offline')
        self._row(f, 'Suggestions', self.as_suggest, 'at most three per page; dismissed ones stay dismissed for the '
                  'project')
        self.as_prices = QPlainTextEdit(format_prices(c.get('prices') or {}))
        self.as_prices.setFixedHeight(64)
        self.as_prices.setToolTip('One line per model: the model, then $ per million input and output tokens. Your '
                                  'own table: the cost shown is estimated from it.')
        self._row(f, 'Prices', self.as_prices, '$ per million tokens, input and output — your table; costs shown '
                  'are estimates from it')
        nr = QHBoxLayout()
        self.as_notice = label('', muted=True, wrap=True)
        nr.addWidget(self.as_notice, 1)
        nr.addWidget(button('Read the notice…', self.assistant_notice))
        self._row(f, 'What is sent', nr)
        self.as_status = label('', muted=True, wrap=True)
        self._row(f, '', self.as_status)
        sec.add(f)
        v.addWidget(sec)
        self._assistant_refresh_state()

    def _assistant_refresh_state(self):
        svc = self.win.assistant
        has = svc.keys.has()
        self.as_key.setPlaceholderText(f'stored in {svc.keys.where()} — type to replace' if has else
                                       'paste your key (gsk_…)')
        self.as_remove.setEnabled(has)
        c = svc.cfg()
        self.as_notice.setText(f'accepted {c["consent_time"].replace("T", " ").rstrip("Z")} UTC' if svc.consented()
                               else 'not accepted yet: nothing is sent until you do')

    def show_assistant(self):
        """Scroll to the Assistant section (from the Assistant panel) and put the keyboard where setup continues."""
        from PyQt6.QtWidgets import QScrollArea
        p = self.assistant_section.parentWidget()
        while p is not None and not isinstance(p, QScrollArea):
            p = p.parentWidget()
        if p is not None:
            p.ensureWidgetVisible(self.assistant_section, 0, 40)
        (self.as_key if self.as_on.isChecked() and not self.win.assistant.keys.has() else self.as_on).setFocus()

    def _assistant_gate(self) -> bool:
        from ..assistant_dock import ensure_consent
        if ensure_consent(self, self.win.assistant):
            self.as_on.setChecked(True)
            self._assistant_refresh_state()
            return True
        self.as_status.setText('Nothing was sent: the notice was not accepted.')
        return False

    def _assistant_call(self, fn, done):
        from ...assistant.provider import ProviderError
        key = self.as_key.text().strip() or None
        url = self.as_url.text().strip() or None
        if not self._assistant_gate():
            return
        if key is None and not self.win.assistant.keys.has():
            self.as_status.setText('Paste your key first.')
            return
        self.as_status.setText('asking the server…')

        def err(e):
            self.as_status.setText(('⚠ ' + e.message) if isinstance(e, ProviderError) else
                                   f'⚠ {type(e).__name__}')
        tasks.run(self, lambda: fn(key, url), done, err)

    def assistant_test(self):
        self._assistant_call(lambda k, u: self.win.assistant.test_connection(k, u),
                             lambda msg: (self.as_status.setText('✓ ' + msg), self._fill_models()))

    def assistant_refresh(self):
        self._assistant_call(lambda k, u: self.win.assistant.models(k, u),
                             lambda ms: (self._fill_models(), self.as_status.setText(
                                 f'✓ {len(ms)} models listed by the server')))

    def _fill_models(self):
        c = self.win.assistant.cfg()
        ids = [m['id'] for m in c.get('models') or [] if isinstance(m, dict) and m.get('id')]
        for cb in (self.as_main, self.as_fast):
            cur = cb.currentText()
            cb.clear()
            cb.addItems(ids if cur in ids else [cur] + ids)
            cb.setCurrentText(cur)
            cb.setToolTip('' if cur in ids else f'{cur} is not among the models the server lists')

    def assistant_remove_key(self):
        svc = self.win.assistant
        if QMessageBox.question(self, 'Remove the key', f'Delete the API key stored in {svc.keys.where()}?') \
                != QMessageBox.StandardButton.Yes:
            return
        svc.keys.remove()
        self.as_status.setText('Key removed.')
        self._assistant_refresh_state()
        self._assistant_changed()

    def assistant_notice(self):
        from ..assistant_dock import ConsentDialog
        svc = self.win.assistant
        d = ConsentDialog(self)
        if svc.consented():
            d.ok.setText('Keep it on')
        if d.exec() == QDialog.DialogCode.Accepted:
            if not svc.consented():
                svc.accept_notice()
            svc.set_cfg(enabled=True)
            self.as_on.setChecked(True)
            self._assistant_refresh_state()
            self._assistant_changed()

    def _assistant_changed(self):
        if hasattr(self.win, 'assistant_panel'):
            self.win.assistant_panel.refresh_state()
            self.win.refresh_assistant_badge()
        if getattr(self.win, 'suggestions', None) is not None:
            self.win.suggestions.refresh(forced=True)

    def _assistant_values(self):
        """The section's values, checked: (dict, None) or (None, why)."""
        from ...assistant.provider import parse_prices
        try:
            prices = parse_prices(self.as_prices.toPlainText())
        except ValueError as e:
            return None, f'Prices: {e}'
        url = self.as_url.text().strip() or 'https://api.groq.com/openai/v1'
        if not url.startswith(('https://', 'http://')):
            return None, 'Server: an http(s) address, e.g. https://api.groq.com/openai/v1'
        main, fast = self.as_main.currentText().strip(), self.as_fast.currentText().strip()
        if not main or not fast:
            return None, 'Choose a main and a fast model.'
        return {'base_url': url, 'main_model': main, 'fast_model': fast, 'temperature': self.as_temp.value(),
                'reasoning_effort': self.as_effort.currentText(), 'mode': self.as_mode.currentData(),
                'prices': prices, 'tpm_limit': self.as_tpm.value(),
                'suggestions': self.as_suggest.isChecked()}, None

    def _assistant_apply(self, vals) -> bool:
        svc = self.win.assistant
        on = self.as_on.isChecked()
        if on and not svc.consented():
            from ..assistant_dock import ConsentDialog
            if ConsentDialog(self).exec() != QDialog.DialogCode.Accepted:
                on = False
                self.as_on.setChecked(False)
            else:
                svc.accept_notice()
        key = self.as_key.text().strip()
        if key:
            try:
                svc.keys.set(key)
            except ValueError as e:
                QMessageBox.warning(self, 'API key', f'The key was not saved: {e}.')
                return False
            self.as_key.clear()
            self.as_eye.setChecked(False)
            self.win.log.append_line(f'assistant: key saved in {svc.keys.where()}')
        svc.set_cfg(save=False, enabled=on, **vals)
        self._assistant_refresh_state()
        return True

    def _row(self, form, text, field, note=''):
        self.fields.append((text, note, field))
        return form.row(text, field, note)

    def focus_field(self, text) -> bool:
        """Put the keyboard focus on the field of the row labelled `text` (the command palette's settings)."""
        from PyQt6.QtWidgets import QWidget
        for t, _note, field in self.fields:
            if t == text:
                w = field if isinstance(field, QWidget) else next(
                    (field.itemAt(k).widget() for k in range(field.count()) if field.itemAt(k).widget()), None)
                if w is not None:
                    w.setFocus()
                    return True
        return False

    def browse_root(self):
        d = QFileDialog.getExistingDirectory(self, 'CARLA folder', self.root.text() or self.win.service.carla_root)
        if d:
            self.root.setText(d)

    def check_python(self):
        py = self.py.text().strip() or S.client_python(self.root.text().strip() or self.win.service.carla_root)
        self.py_status.setText('checking…')
        tasks.run(self, lambda: S.check_client(py), lambda r: self.py_status.setText(
            ('✓ ' if r['ok'] else '⚠ ') + r['text']))

    def apply(self):
        st = self.win.settings
        avals, why = self._assistant_values()
        if why:
            QMessageBox.warning(self, 'Settings', why)
            return
        root = self.root.text().strip()
        if root and not S.is_carla_root(root):
            QMessageBox.warning(self, 'Settings', f'{root} is not a CARLA folder ({S.BINARY} is missing).')
            return
        port, tm = self.port.value(), self.tm_port.value()
        allowed = set(st.get('allowed_ports') or [])
        for p in (port, tm):
            if p in RESERVED_PORTS and p not in allowed:
                r = QMessageBox.question(self, 'A default CARLA port', f'Port {p} is one of CARLA\'s defaults, where a '
                                         'server you started may listen. Use it? Falcon will then connect to the '
                                         'server there and never start or stop it.')
                if r != QMessageBox.StandardButton.Yes:
                    return
                allowed.add(p)
        allowed &= {port, tm}                  # a choice holds only while that port is chosen
        if self.win.service.server_state() in ('ready', 'starting') and port != self.win.service.port:
            QMessageBox.information(self, 'Settings', 'Stop Falcon\'s server before changing its port.')
            return
        st.set('carla_root', root, save=False)
        st.set('client_python', self.py.text().strip(), save=False)
        st.set('rpc_port', port, save=False)
        st.set('tm_port', tm, save=False)
        st.set('allowed_ports', sorted(allowed), save=False)
        st.set('server_memory_gb', self.mem.value(), save=False)
        st.set('server_quality', self.quality.currentText(), save=False)
        st.set('free_gpu_for_jobs', self.free_gpu.isChecked(), save=False)
        st.set('render_budget_mpx', self.budget.value(), save=False)
        st.set('geocoding_allowed', self.geo.isChecked(), save=False)
        st.set('notify_jobs', self.notify.isChecked(), save=False)
        if not self._assistant_apply(avals):
            return
        st.save()
        self._assistant_changed()
        self.win.log.append_line(f'settings saved ({tilde(st.path)})')
        for p in self.win.pages:
            if p is not self:
                p.refresh()
