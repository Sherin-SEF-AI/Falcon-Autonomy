"""Triage: the team's queue of failures (core.team), the workspace's failure modes (core.failclusters) and regression
bisects of a driving stack (core.regbisect) — what failed, who is on it, which kind of failure it is, and which commit
broke it."""
from __future__ import annotations

import json
import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QInputDialog, QLineEdit, QMessageBox,
                             QPlainTextEdit, QSpinBox, QSplitter, QTreeWidgetItem, QVBoxLayout, QWidget)

from ...core import failclusters as FC
from ...core import team as TM
from ...core.jobs import FINAL, JobSpec
from ...core.paths import tilde
from .. import tasks
from ..team import CommentsPanel, ident
from ..widgets import Form, KeyValues, Page, PlaceholderList, PlaceholderTree, Section, button, label, set_enabled

STATE_TITLES = {'open': 'Open', 'assigned': 'Assigned', 'investigating': 'Investigating', 'fixed': 'Fixed',
                'verified': 'Verified'}
STATE_TONE = {'open': 'warning', 'assigned': 'warning', 'investigating': 'accent', 'fixed': 'good',
              'verified': 'good'}


class TriagePage(Page):
    key, title, icon = 'triage', 'Triage', 'warning'
    about = ('The team\'s queue of failures — each item assigned, investigated, fixed and verified, with links to its '
             'runs, scenarios, cases and clusters, its history and comments — the workspace\'s failures clustered '
             'into named failure modes, and regression bisects that find the commit of a stack that broke a scenario. '
             'In a team workspace on a shared folder every member sees the same queue (docs/team.md).')

    def __init__(self, win):
        super().__init__(win)
        self.jobs: dict = {}
        self.clusters: dict | None = None
        self.labels: dict = {}                      # cluster id -> the assistant's grounded name
        v = self.page_layout()
        # ---------------------------------------------------------------- the team workspace
        tsec = Section('Team workspace', 'A project on a shared folder or NAS that several people open at once: '
                                         'comments, triage events and run owners are one file each (never merged), '
                                         'the project file is saved under a lock and never over another member\'s '
                                         'save, and the catalogue stays on each machine (docs/team.md).')
        self.team_info = KeyValues(('Workspace', 'You', 'Members'))
        tsec.add(self.team_info)
        row = QHBoxLayout()
        self.make_team = button('Make this a team workspace', self.init_team)
        row.addWidget(self.make_team)
        row.addWidget(button('Your name…', lambda: self.win.go('settings')))
        row.addStretch(1)
        tsec.add(row)
        v.addWidget(tsec)
        # ---------------------------------------------------------------- the queue
        qsec = Section('Triage queue', 'States: open → assigned → investigating → fixed → verified (a fix that does '
                                       'not hold goes back to investigating). Every change is an event with who and '
                                       'when; two changes made at once on two machines are applied in time order, '
                                       'and one that no longer fits is kept in the history as ignored.')
        bar = QHBoxLayout()
        self.state = QComboBox()
        self.state.addItem('All states', '')
        for s in TM.STATES:
            self.state.addItem(STATE_TITLES[s], s)
        self.state.currentIndexChanged.connect(lambda _i: self.fill_items())
        self.mine = QCheckBox('Only mine')
        self.mine.toggled.connect(lambda _b: self.fill_items())
        bar.addWidget(self.state)
        bar.addWidget(self.mine)
        bar.addStretch(1)
        bar.addWidget(button('New item…', self.new_item, role='primary'))
        bar.addWidget(button('Refresh', self.refresh))
        qsec.add(bar)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.items = PlaceholderTree('No triage items yet: New item…, or Triage this cluster below.')
        self.items.setHeaderLabels(['Item', 'State', 'Assignee', 'Title', 'Updated'])
        self.items.setRootIsDecorated(False)
        self.items.setMinimumHeight(200)
        self.items.setColumnWidth(0, 60)
        self.items.setColumnWidth(1, 110)
        self.items.setColumnWidth(3, 300)
        self.items.currentItemChanged.connect(lambda it, _p: self.show_item(it.data(0, 256) if it else None))
        split.addWidget(self.items)
        detail = QWidget()
        dv = QVBoxLayout(detail)
        dv.setContentsMargins(0, 0, 0, 0)
        self.item_info = KeyValues(('Title', 'State', 'Assignee', 'Links', 'Created'))
        dv.addWidget(self.item_info)
        acts = QHBoxLayout()
        self.b_me = button('Assign to me', lambda: self._event('assign'))
        self.b_assign = button('Assign…', lambda: self._event('assign_to'))
        self.b_inv = button('Investigate', lambda: self._event('investigating'))
        self.b_fixed = button('Fixed', lambda: self._event('fixed'))
        self.b_verified = button('Verified', lambda: self._event('verified'))
        self.b_reopen = button('Reopen', lambda: self._event('reopen'))
        for b in (self.b_me, self.b_assign, self.b_inv, self.b_fixed, self.b_verified, self.b_reopen):
            acts.addWidget(b)
        acts.addStretch(1)
        dv.addLayout(acts)
        dv.addWidget(label('History', muted=True))
        self.history = PlaceholderList('')
        self.history.setMaximumHeight(140)
        dv.addWidget(self.history)
        dv.addWidget(label('Comments', muted=True))
        self.comments = CommentsPanel(win)
        dv.addWidget(self.comments)
        split.addWidget(detail)
        split.setSizes([520, 520])
        qsec.add(split)
        v.addWidget(qsec)
        # ---------------------------------------------------------------- failure modes
        csec = Section('Failure modes', 'Every failure of the workspace — studies\' failing cases, suites\' and CI '
                                        'runs\' failed runs — clustered by a Gower distance over scenario, '
                                        'parameters, conditions, outcome and root-cause factors (average linkage, '
                                        'cut 0.3), each named from what sets it apart (docs/failure-clusters.md). A '
                                        'CI run flags failures that fall in none of them as new failure modes.')
        crow = QHBoxLayout()
        self.b_cluster = button('Cluster the failures', self.build_clusters, role='primary')
        crow.addWidget(self.b_cluster)
        self.b_name = button('Name them with the assistant', self.name_clusters,
                             tip='The fast model rewords the names; a name is kept only when its numbers are the '
                                 'cluster\'s own')
        crow.addWidget(self.b_name)
        self.b_triage_cluster = button('Triage this cluster', self.triage_cluster)
        crow.addWidget(self.b_triage_cluster)
        crow.addStretch(1)
        self.cluster_status = label('', muted=True)
        crow.addWidget(self.cluster_status)
        csec.add(crow)
        csplit = QSplitter(Qt.Orientation.Horizontal)
        self.cl_tree = PlaceholderTree('No clusters yet: Cluster the failures (it reads the studies, suites and CI '
                                       'runs of the workspace; nothing is run).')
        self.cl_tree.setHeaderLabels(['Cluster', 'Failures', 'Name', 'Sources'])
        self.cl_tree.setRootIsDecorated(False)
        self.cl_tree.setMinimumHeight(180)
        self.cl_tree.setColumnWidth(2, 340)
        self.cl_tree.currentItemChanged.connect(lambda it, _p: self.show_cluster(it.data(0, 256) if it else None))
        csplit.addWidget(self.cl_tree)
        self.cl_detail = PlaceholderTree('')
        self.cl_detail.setHeaderLabels(['Feature or member', 'Detail'])
        self.cl_detail.setRootIsDecorated(True)
        self.cl_detail.itemDoubleClicked.connect(self._open_member)
        csplit.addWidget(self.cl_detail)
        csplit.setSizes([560, 480])
        csec.add(csplit)
        v.addWidget(csec)
        # ---------------------------------------------------------------- bisect
        bsec = Section('Regression bisect', 'Runs a stack\'s CI config at a good and a bad git ref, finds the '
                                            'scenarios that got worse (paired runs when deterministic, else '
                                            'Fisher\'s exact test over repeats), then bisects the history between '
                                            'them through CI mode to the first commit that broke each one; commits '
                                            'that do not check out or build are skipped (docs/bisect.md).')
        f = Form()
        self.repo = QLineEdit()
        self.repo.setPlaceholderText('the stack\'s git repository')
        rrow = QHBoxLayout()
        rrow.addWidget(self.repo, 1)
        rrow.addWidget(button('Browse…', self._browse_repo))
        f.row('Repository', rrow)
        self.config = QLineEdit('falcon-ci.yml')
        f.row('CI config', self.config, 'in the repository; read from each commit')
        self.good = QLineEdit()
        self.good.setPlaceholderText('e.g. v1.4 or a commit')
        f.row('Good ref', self.good, 'older: the stack passed')
        self.bad = QLineEdit('HEAD')
        f.row('Bad ref', self.bad, 'newer: it fails')
        self.repeats = QSpinBox()
        self.repeats.setRange(0, 20)
        self.repeats.setSpecialValueText('auto')
        f.row('Repeats', self.repeats, 'per commit when the mode is not deterministic (auto: 5; deterministic: 1)')
        self.build_cmd = QLineEdit()
        self.build_cmd.setPlaceholderText('optional, e.g. docker build -t my-stack .')
        f.row('Build command', self.build_cmd, 'run in each checkout first; a failure skips the commit')
        bsec.add(f)
        brow = QHBoxLayout()
        self.b_bisect = button('Start the bisect', self.start_bisect, role='primary')
        brow.addWidget(self.b_bisect)
        brow.addStretch(1)
        bsec.add(brow)
        self.bisects = PlaceholderTree('No bisects yet.')
        self.bisects.setHeaderLabels(['Bisect', 'Regressed', 'First bad commits'])
        self.bisects.setRootIsDecorated(False)
        self.bisects.setMinimumHeight(110)
        self.bisects.setColumnWidth(0, 260)
        self.bisects.currentItemChanged.connect(lambda it, _p: self.show_bisect(it.data(0, 256) if it else None))
        bsec.add(self.bisects)
        self.bisect_text = QPlainTextEdit()
        self.bisect_text.setReadOnly(True)
        self.bisect_text.setMaximumHeight(180)
        bsec.add(self.bisect_text)
        v.addWidget(bsec)
        v.addStretch(1)
        win.bridge.finished.connect(self._finished)
        self.refresh()

    # ------------------------------------------------------------------ refresh
    def ws(self) -> str:
        return self.win.workspace()

    def showEvent(self, ev):
        super().showEvent(ev)
        self.refresh()

    def refresh(self):
        ws = self.ws()
        info = TM.info(ws) if self.win.project.path else {'team': False}
        me = ident(self.win)
        self.team_info.set('Workspace', (f'team workspace "{info.get("name")}" — {tilde(ws)}' if info['team'] else
                                         f'{tilde(ws)} (this machine only)') if self.win.project.path else
                           'not saved yet')
        self.team_info.set('You', f'{me["user"]} on {me["host"]}')
        self.team_info.set('Members', ', '.join(f'{m["user"]}@{m["host"]}' for m in info.get('members') or [])
                           or '—')
        set_enabled(self.make_team, bool(self.win.project.path) and not info['team'],
                    'Save the project first' if not self.win.project.path else 'It is a team workspace already')
        self.fill_items()
        self.fill_bisects()
        if self.clusters is None:
            saved = FC.load(ws) if self.win.project.path else None
            if saved:
                self._show_clusters(saved)
        self._enable()

    def _enable(self):
        it = self._item()
        st = it['state'] if it else ''
        for b, ok in ((self.b_me, it), (self.b_assign, it), (self.b_inv, st in ('assigned', 'open')),
                      (self.b_fixed, st == 'investigating'), (self.b_verified, st == 'fixed'),
                      (self.b_reopen, st in ('fixed', 'verified'))):
            set_enabled(b, bool(ok), 'Choose an item' if not it else f'Not a step from {st}')
        set_enabled(self.b_name, bool(self.clusters and self.clusters.get('clusters')) and
                    self.win.assistant.ready()[0], 'Cluster first, with the assistant on (Settings ▸ Assistant)')
        set_enabled(self.b_triage_cluster, self.cl_tree.currentItem() is not None, 'Choose a cluster')
        saved = bool(self.win.project.path)
        set_enabled(self.b_bisect, saved, 'Save the project first: the bisect is written into its workspace')
        set_enabled(self.b_cluster, saved, 'Save the project first')

    # ------------------------------------------------------------------ the queue
    def fill_items(self):
        cur = self._item()
        self.items.clear()
        if not self.win.project.path:
            self.show_item(None)
            return
        me = ident(self.win)['user']
        rows = TM.items(self.ws(), self.state.currentData() or '', me if self.mine.isChecked() else '')
        from ..theme import THEME
        for r in rows:
            t = QTreeWidgetItem([r['id'], STATE_TITLES[r['state']], r['assignee'] or '—', r['title'],
                                 str(r['updated'] or '')[:16].replace('T', ' ')])
            t.setData(0, 256, r['id'])
            t.setForeground(1, THEME.color(STATE_TONE[r['state']]))
            if r['conflicts']:
                t.setToolTip(1, f'{r["conflicts"]} change(s) made at the same time elsewhere were ignored: see the '
                                'history')
            self.items.addTopLevelItem(t)
            if cur and r['id'] == cur['id']:
                self.items.setCurrentItem(t)
        if self.items.currentItem() is None:
            self.show_item(None)

    def _item(self):
        it = self.items.currentItem() if hasattr(self, 'items') else None
        if it is None:
            return None
        try:
            return TM.get_item(self.ws(), it.data(0, 256))
        except TM.TeamError:
            return None

    def show_item(self, iid):
        self.history.clear()
        if not iid:
            for k in ('Title', 'State', 'Assignee', 'Links', 'Created'):
                self.item_info.set(k, '—')
            self.comments.set_ref('')
            self._enable()
            return
        try:
            it = TM.get_item(self.ws(), iid)
        except TM.TeamError:
            return
        self.item_info.set('Title', it['title'])
        self.item_info.set('State', STATE_TITLES[it['state']] + (f' (fixed by {it["fixed_by"]})' if it['fixed_by'] and
                                                                  it['state'] in ('fixed', 'verified') else ''))
        self.item_info.set('Assignee', it['assignee'] or '—')
        self.item_info.set('Links', ', '.join(it['links']) or '—')
        self.item_info.set('Created', f'{str(it["created"])[:16].replace("T", " ")} by {it["created_by"]}')
        for h in it['history']:
            self.history.addItem(f'{str(h.get("time") or "")[:16].replace("T", " ")}  {h.get("user") or "?"}'
                                 f'{" (via the assistant)" if h.get("via") else ""}: '
                                 f'{"" if h.get("ok") else "ignored — "}{h["what"]}'
                                 + (f' ({h["note"]})' if h.get('note') else ''))
        self.comments.set_ref(f'triage:{it["id"]}')
        self._enable()

    def new_item(self, links=(), title=''):
        if not self.win.project.path:
            QMessageBox.information(self, 'Triage', 'Save the project first: the queue lives in its workspace.')
            return None
        if not title:
            title, ok = QInputDialog.getText(self, 'New triage item', 'What failed?')
            if not ok or not title.strip():
                return None
        try:
            it = TM.create_item(self.ws(), title, ident(self.win), links)
        except (TM.TeamError, OSError) as e:
            QMessageBox.warning(self, 'Triage', str(e))
            return None
        self.fill_items()
        for i in range(self.items.topLevelItemCount()):
            if self.items.topLevelItem(i).data(0, 256) == it['id']:
                self.items.setCurrentItem(self.items.topLevelItem(i))
        return it

    def _event(self, what, to: str = ''):
        it = self._item()
        if it is None:
            return None
        me = ident(self.win)
        try:
            if what == 'assign':
                r = TM.assign(self.ws(), it['id'], me['user'], me)
            elif what == 'assign_to':
                to = to or QInputDialog.getText(self, 'Assign', 'Assign to (a name as your team writes it):')[0]
                if not to.strip():
                    return None
                r = TM.assign(self.ws(), it['id'], to.strip(), me)
            elif what == 'reopen':
                r = TM.set_state(self.ws(), it['id'], 'investigating', me, 'reopened')
            else:
                r = TM.set_state(self.ws(), it['id'], what, me)
        except (TM.TeamError, OSError) as e:
            QMessageBox.warning(self, 'Triage', str(e))
            return None
        self.fill_items()
        self.show_item(r['id'])
        return r

    def init_team(self):
        try:
            TM.init(self.ws(), self.win.project.doc.get('name', ''), ident(self.win))
        except OSError as e:
            QMessageBox.warning(self, 'Team workspace', str(e))
            return
        self.win.service.set_workspace(self.ws())            # its catalogue moves to this machine (core.catalog)
        self.win.log.append_line(f'team workspace: {tilde(self.ws())}')
        self.refresh()

    # ------------------------------------------------------------------ failure modes
    def build_clusters(self):
        ws = self.ws()
        self.cluster_status.setText('clustering…')
        set_enabled(self.b_cluster, False, 'Clustering…')
        me = ident(self.win)

        def done(doc):
            self.labels = {}
            self._show_clusters(doc)
            self._enable()

        def fail(e):
            self.cluster_status.setText(f'not clustered: {e}')
            self._enable()
        return tasks.run(self, lambda: FC.build(ws, ident=me), done, fail)

    def _show_clusters(self, doc):
        self.clusters = doc
        self.cl_tree.clear()
        for c in doc.get('clusters') or []:
            name = self.labels.get(c['id']) or c['name']
            t = QTreeWidgetItem([c['id'], str(c['size']), name,
                                 ', '.join(f'{k} {v}' for k, v in sorted(c['sources'].items()))])
            t.setData(0, 256, c['id'])
            t.setToolTip(2, c['name'] + ('\n(worded by the assistant; the name from the features: ' + c['name'] + ')'
                                         if c['id'] in self.labels else ''))
            t.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.cl_tree.addTopLevelItem(t)
        self.cluster_status.setText(f'{doc.get("failures", 0)} failures in {len(doc.get("clusters") or [])} clusters '
                                    f'({doc.get("singletons", 0)} single) · {str(doc.get("built", ""))[:16]}')

    def _cluster(self, cid):
        return next((c for c in (self.clusters or {}).get('clusters') or [] if c['id'] == cid), None)

    def show_cluster(self, cid):
        self.cl_detail.clear()
        c = self._cluster(cid)
        self._enable()
        if c is None:
            return
        feats = QTreeWidgetItem(['What sets it apart', ''])
        for f in c['features'][:8]:
            detail = (f'{round(100 * f["share_in"])} % of it, {round(100 * (f.get("share_out") or 0))} % of the rest'
                      if 'share_in' in f else f'median {f.get("median_in")} against {f.get("median_out")}')
            feats.addChild(QTreeWidgetItem([f['text'], detail]))
        self.cl_detail.addTopLevelItem(feats)
        mem = QTreeWidgetItem([f'Members ({c["size"]})', f'cohesion {c["cohesion"]:.3f} (mean distance within)'])
        idx = (self.clusters or {}).get('index') or {}
        for k in c['members'][:200]:
            f = idx.get(k) or {}
            o = f.get('outcome') or {}
            t = QTreeWidgetItem([k, f'{f.get("scenario")} seed {f.get("seed")}'
                                 + (' · collision' if o.get('collision') else '')
                                 + (f' · {", ".join(o.get("checks") or [])}' if o.get('checks') else '')])
            t.setData(0, 256, f.get('run'))
            mem.addChild(t)
        self.cl_detail.addTopLevelItem(mem)
        feats.setExpanded(True)
        mem.setExpanded(True)

    def _open_member(self, it, _c):
        run = it.data(0, 256)
        if run and os.path.isfile(os.path.join(run, 'run.json')):
            self.win.go('data').show_run(run)

    def triage_cluster(self):
        it = self.cl_tree.currentItem()
        c = self._cluster(it.data(0, 256)) if it else None
        if c is None:
            return None
        idx = (self.clusters or {}).get('index') or {}
        ws = os.path.realpath(self.ws())
        runs = []
        for k in c['members'][:5]:
            r = (idx.get(k) or {}).get('run')
            if r and os.path.realpath(r).startswith(ws + os.sep):
                runs.append('run:' + os.path.relpath(os.path.realpath(r), ws))
        name = self.labels.get(c['id']) or c['name']
        return self.new_item([f'cluster:{c["id"]}'] + runs, f'{name} ({c["size"]} failures)'[:200])

    def name_clusters(self):
        from ...assistant import suggest as S
        from ...assistant.privacy import private_reason
        svc = self.win.assistant
        idx = (self.clusters or {}).get('index') or {}
        # a cluster holding a run from the dashcam footage is never described to the model (assistant.privacy)
        cl = [c for c in (self.clusters or {}).get('clusters') or []
              if not any((idx.get(k) or {}).get('run') and private_reason(idx[k]['run']) for k in c['members'])]
        if not cl or not svc.ready()[0]:
            return None
        self.cluster_status.setText('asking the assistant for names…')

        def done(res):
            self.labels.update(res.get('names') or {})
            self._show_clusters(self.clusters)
            n_rej = len(res.get('rejected') or {})
            if n_rej:
                self.win.log.append_line(f'assistant: {n_rej} cluster name(s) kept their own wording (a number not '
                                         'in the cluster\'s features)')

        def fail(e):
            self.cluster_status.setText(f'names kept: {getattr(e, "message", None) or type(e).__name__}')
        return tasks.run(self, lambda: S.phrase_clusters(svc, cl), done, fail)

    # ------------------------------------------------------------------ bisect
    def _browse_repo(self):
        d = QFileDialog.getExistingDirectory(self, 'The stack\'s git repository', self.repo.text() or
                                             os.path.expanduser('~'))
        if d:
            self.repo.setText(d)

    def bisect_argv(self, name: str) -> list:
        repo = os.path.abspath(os.path.expanduser(self.repo.text().strip()))
        out = os.path.join(self.ws(), 'bisect', name)
        argv = [sys.executable, '-m', 'falcon_autonomy', 'bisect', 'run', repo, '--config',
                self.config.text().strip(), '--good', self.good.text().strip(), '--bad', self.bad.text().strip(),
                '--out', out]
        if self.repeats.value():
            argv += ['--repeats', str(self.repeats.value())]
        if self.build_cmd.text().strip():
            argv += ['--build', self.build_cmd.text().strip()]
        if getattr(self.win.service, 'fake_carla', ''):
            argv += ['--fake-carla', self.win.service.fake_carla]
        return argv

    def start_bisect(self):
        repo = os.path.expanduser(self.repo.text().strip())
        why = ('Choose the stack\'s git repository' if not repo or not os.path.isdir(os.path.join(repo, '.git')) else
               'Give a good and a bad ref' if not self.good.text().strip() or not self.bad.text().strip() else
               f'No {self.config.text().strip()} in the repository' if not os.path.isfile(
                   os.path.join(repo, self.config.text().strip())) else '')
        if why:
            QMessageBox.information(self, 'Regression bisect', why)
            return None
        import re
        import time
        name = f'{time.strftime("%Y%m%d-%H%M%S")}-' + re.sub(r'[^A-Za-z0-9._-]+', '_',
                                                             f'{self.good.text().strip()}-{self.bad.text().strip()}')
        os.makedirs(os.path.join(self.ws(), 'bisect'), exist_ok=True)
        job = self.win.queue.submit(JobSpec('bisect', f'Bisect {self.good.text().strip()}..{self.bad.text().strip()}',
                                            self.bisect_argv(name), memory_gb=0.5,
                                            run_dir=os.path.join(self.ws(), 'bisect', name)))
        self.jobs[job.id] = job
        self.win.jobs_dock.raise_()
        return job

    def fill_bisects(self):
        cur = self.bisects.currentItem().data(0, 256) if self.bisects.currentItem() else None
        self.bisects.clear()
        root = os.path.join(self.ws(), 'bisect')
        if not self.win.project.path or not os.path.isdir(root):
            return
        for n in sorted(os.listdir(root), reverse=True):
            p = os.path.join(root, n, 'bisect.json')
            try:
                with open(p, encoding='utf-8') as fh:
                    doc = json.load(fh)
            except (OSError, ValueError):
                continue
            found = [f'{s}: {(r.get("first_bad") or {}).get("short") or r.get("status")}'
                     for s, r in (doc.get('bisect') or {}).items()]
            t = QTreeWidgetItem([n, ', '.join(doc.get('regressed') or []) or 'none', '; '.join(found) or '—'])
            t.setData(0, 256, os.path.dirname(p))
            self.bisects.addTopLevelItem(t)
            if cur == os.path.dirname(p):
                self.bisects.setCurrentItem(t)

    def show_bisect(self, d):
        if not d:
            self.bisect_text.setPlainText('')
            return
        try:
            with open(os.path.join(d, 'summary.md'), encoding='utf-8') as fh:
                self.bisect_text.setPlainText(fh.read())
        except OSError as e:
            self.bisect_text.setPlainText(str(e))

    def _finished(self, job):
        if self.jobs.pop(job.id, None) is not None:
            self.fill_bisects()
            if job.state == 'failed':
                self.win.log.append_line(f'bisect: {job.summary()}')

    def busy(self) -> bool:
        return any(j.state not in FINAL for j in self.jobs.values())
