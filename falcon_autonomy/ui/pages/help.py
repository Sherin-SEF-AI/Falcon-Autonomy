"""Help: the manual, searchable. The pages are the docs folder's Markdown files (bundled with the app)."""
from __future__ import annotations

import os
import re

from PyQt6.QtWidgets import QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QTextBrowser, QVBoxLayout

from ..widgets import Page, label

DOCS_DIRS = (os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))), 'docs'), os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                         'docs'))


def docs_dir() -> str:
    return next((d for d in DOCS_DIRS if os.path.isdir(d)), '')


def load_docs() -> list:
    """[(title, filename, markdown)] in the manual's order (index.md first, then by name)."""
    d = docs_dir()
    out = []
    if not d:
        return out
    for fn in sorted(os.listdir(d), key=lambda f: (f != 'index.md', f)):
        if not fn.endswith('.md'):
            continue
        with open(os.path.join(d, fn), encoding='utf-8') as fh:
            text = fh.read()
        m = re.search(r'^#\s+(.+)$', text, re.M)
        out.append((m.group(1).strip() if m else fn, fn, text))
    return out


class HelpPage(Page):
    key, title, icon = 'help', 'Help', 'help'
    about = 'The manual: every page, the project format, the CLI, the methods behind every number.'

    def __init__(self, win):
        super().__init__(win)
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 12, 16, 12)
        v.addWidget(label(self.title, page=True))
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search the manual…')
        v.addWidget(self.search)
        h = QHBoxLayout()
        self.list = QListWidget()
        self.list.setFixedWidth(240)
        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(False)
        self.view.anchorClicked.connect(self.follow)
        h.addWidget(self.list)
        h.addWidget(self.view, 1)
        v.addLayout(h, 1)
        self.docs = load_docs()
        self.search.textChanged.connect(self.filter)
        self.list.currentItemChanged.connect(lambda it, _p: self.show(it))
        self.filter('')

    def filter(self, text):
        q = text.strip().lower()
        self.list.clear()
        for title, fn, md in self.docs:
            n = md.lower().count(q) if q else 0
            if q and n == 0 and q not in title.lower():
                continue
            it = QListWidgetItem(f'{title}' + (f'  ({n})' if q else ''))
            it.setData(256, fn)
            self.list.addItem(it)
        if self.list.count():
            self.list.setCurrentRow(0)
        else:
            self.view.setMarkdown('Nothing in the manual matches.' if self.docs else
                                  'The manual (the docs folder) was not found.')

    def show(self, it):
        if it is None:
            return
        fn = it.data(256)
        md = next(m for _, f, m in self.docs if f == fn)
        self.view.setMarkdown(md)
        q = self.search.text().strip()
        if q:
            self.view.find(q)

    def follow(self, url):
        target = url.toString().split('#')[0]
        for i in range(self.list.count()):
            if self.list.item(i).data(256) == os.path.basename(target):
                self.list.setCurrentRow(i)
                return
