"""Training: fine-tune detectors on drives and real data (core.training), run the mixed sim/real experiment
(core.experiments), learn driving policies by behaviour cloning or PPO (core.imitation, core.rl), and score a policy
in the style of the CARLA Leaderboard with a replay of any infraction (core.leaderboard). Every action is a job; the
heavy work runs in its own process."""
from __future__ import annotations

import glob
import json
import os
import sys
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                             QPlainTextEdit, QSlider, QSpinBox, QSplitter, QTabWidget, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget)

from ...core import leaderboard as LB
from ...core import scenarios as SC
from ...core import training as T
from ...core.jobs import FINAL, JobSpec
from ...core.paths import tilde
from ...core.project import Edit
from ...core.provenance import read_json
from ..theme import STATUS
from ..widgets import Form, KeyValues, Page, Section, button, label, scroll


def _fmt(v, nd=3):
    return '—' if v is None else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


def _jsonl(p):
    try:
        with open(p, encoding='utf-8') as fh:
            return [json.loads(x) for x in fh if x.strip()]
    except (OSError, ValueError):
        return []


class Flipbook(QWidget):
    """A replay's frames with a slider."""

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        self.view = QLabel('Replay an infraction to see it here')
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumHeight(220)
        v.addWidget(self.view, 1)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.show)
        v.addWidget(self.slider)
        self.caption = label('', muted=True)
        v.addWidget(self.caption)
        self.frames, self.doc = [], None

    def load(self, doc):
        self.doc = doc
        self.frames = [p for p in (doc or {}).get('frames') or () if os.path.isfile(p)]
        self.slider.setRange(0, max(len(self.frames) - 1, 0))
        self.slider.setValue(len(self.frames) - 1 if self.frames else 0)
        self.show(self.slider.value())

    def show(self, k=0):
        if not self.frames:
            self.view.setPixmap(QPixmap())
            self.view.setText('No frames' if self.doc else 'Replay an infraction to see it here')
            self.caption.setText('')
            return
        self.view.setPixmap(QPixmap(self.frames[k]).scaled(self.view.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                           Qt.TransformationMode.SmoothTransformation))
        inf = (self.doc or {}).get('infraction') or {}
        step = int(os.path.splitext(os.path.basename(self.frames[k]))[0])
        fps = float((self.doc or {}).get('fps') or 20.0)
        rel = (step - inf['step']) / fps if inf else 0.0
        self.caption.setText(f'step {step} ({rel:+.2f} s from the {inf.get("kind", "infraction")})' if inf else
                             f'step {step}')


class TrainingPage(Page):
    key, title, icon = 'training', 'Training', 'sparkle'
    about = ('Fine-tune detectors on drives and real data, test how much synthetic data helps, learn driving '
             'policies, and score them as the CARLA Leaderboard does — replaying any infraction.')

    def __init__(self, win):
        super().__init__(win)
        self.jobs = {}
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._perception_tab(), 'Detectors')
        self.tabs.addTab(self._experiment_tab(), 'Sim + real experiment')
        self.tabs.addTab(self._policy_tab(), 'Driving policies')
        self.tabs.addTab(self._leaderboard_tab(), 'Leaderboard')
        v.addWidget(self.tabs)
        self.state = KeyValues(('Job',))
        v.addWidget(self.state)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(1500)
        win.bridge.finished.connect(self.finished)
        self.refresh()

    def showEvent(self, ev):
        super().showEvent(ev)
        self.refresh()                    # drives, trainings and results made elsewhere since it was last shown

    # ------------------------------------------------------------------ shared
    def ws(self, *parts):
        return os.path.join(self.win.workspace(), *parts)

    def new_dir(self, sub, name):
        base = self.ws(sub, name)
        p, k = base, 2
        while os.path.exists(p):
            p, k = f'{base}-{k}', k + 1
        os.makedirs(p)
        return p

    def _runs_list(self):
        w = QListWidget()
        w.setMinimumHeight(110)
        return w

    def _fill_runs(self, w):
        checked = {w.item(i).data(256) for i in range(w.count()) if w.item(i).checkState() == Qt.CheckState.Checked}
        w.clear()
        cat = self.win.service.catalog
        for r in (cat.list('run', 500) if cat is not None else []):
            if r['summary'].get('worker') not in (None, 'drive') or r['status'] != 'ok':
                continue
            it = QListWidgetItem(r['name'])
            it.setData(256, r['path'])
            it.setToolTip(tilde(r['path']))
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if r['path'] in checked else Qt.CheckState.Unchecked)
            w.addItem(it)

    @staticmethod
    def _checked(w):
        return [w.item(i).data(256) for i in range(w.count()) if w.item(i).checkState() == Qt.CheckState.Checked]

    def _ready(self) -> bool:
        if not self.win.project.path or self.win.project.dirty:
            return self.win.save()
        return True

    def _submit(self, spec):
        job = self.win.queue.submit(spec)
        self.jobs[job.id] = job
        self.win.jobs_dock.raise_()
        return job

    def _cli(self, kind, title, argv, needs_gpu=False, memory_gb=1.0, gpu_held=False):
        env = dict(os.environ, FALCON_GPU_LOCK_HELD='1') if gpu_held else None
        return self._submit(JobSpec(kind, title, [sys.executable, '-m', 'falcon_autonomy'] + argv, env=env,
                                    needs_gpu=needs_gpu, memory_gb=memory_gb))

    def _carla_args(self):
        return ['--fake-carla', self.win.fake_carla] if self.win.fake_carla else []

    def poll(self):
        live = [j for j in self.jobs.values() if j.state not in FINAL]
        if live:
            self.state.set('Job', ' · '.join(j.summary() for j in live))

    def finished(self, job):
        if job.id not in self.jobs:
            return
        self.jobs.pop(job.id)
        self.state.set('Job', job.summary())
        self.refresh()

    def refresh(self):
        for w in (self.det_runs, self.exp_synth, self.bc_runs):
            self._fill_runs(w)
        self.load_trainings()
        self.load_experiments()
        self.load_policies()
        self.load_leaderboards()

    @staticmethod
    def _combo_dirs(combo, root, name):
        cur = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        if os.path.isdir(root):
            for d in sorted(os.listdir(root), reverse=True):
                if os.path.isfile(os.path.join(root, d, name)):
                    combo.addItem(d, os.path.join(root, d))
        k = combo.findData(cur)
        combo.setCurrentIndex(k if k >= 0 else 0)
        combo.blockSignals(False)

    # ================================================================== detectors
    def _perception_tab(self):
        split = QSplitter()
        left = QWidget()
        lv = QVBoxLayout(left)
        sec = Section('Fine-tune a detector', 'torchvision detectors (BSD-3-Clause). The checked drives train it; '
                                             'the validation drive is held out and chooses the best epoch. A '
                                             'training waits while CARLA holds the GPU.')
        f = Form()
        self.arch = QComboBox()
        for a in T.ARCHS:
            self.arch.addItem(a, a)
        self.init = QComboBox()
        for k in ('coco', 'imagenet', 'scratch'):
            self.init.addItem(k, k)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 500)
        self.epochs.setValue(12)
        self.batch = QSpinBox()
        self.batch.setRange(1, 64)
        self.batch.setValue(8)
        self.size = QLineEdit('640x384')
        self.det_seed = QSpinBox()
        self.det_seed.setRange(0, 2 ** 31 - 1)
        self.val_run = QComboBox()
        self.real_src = QLineEdit()
        self.real_src.setPlaceholderText('optional real data, KIND:PATH (kitti_object, coco, bdd100k, nuscenes)')
        for text, w in (('Model', self.arch), ('Start from', self.init), ('Epochs', self.epochs),
                        ('Batch', self.batch), ('Input size', self.size), ('Seed', self.det_seed),
                        ('Validation drive', self.val_run), ('Real data', self.real_src)):
            f.row(text, w)
        sec.add(f)
        sec.add(label('Training drives', h=True))
        self.det_runs = self._runs_list()
        sec.add(self.det_runs)
        b = QHBoxLayout()
        b.addWidget(button('Train', self.train, role='primary'))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        sec = Section('Pretrained weights', 'Downloaded only when you ask, from download.pytorch.org; the SHA-256 '
                                            'is checked before every use.')
        self.weights = QTreeWidget()
        self.weights.setHeaderLabels(['Weights', 'MB', 'Here', 'Terms'])
        self.weights.setRootIsDecorated(False)
        self.weights.setMinimumHeight(140)
        sec.add(self.weights)
        b = QHBoxLayout()
        b.addWidget(button('Download the selected…', self.fetch_weights))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        lv.addStretch(1)
        split.addWidget(scroll(left))
        right = QWidget()
        rv = QVBoxLayout(right)
        self.trainings = QComboBox()
        self.trainings.currentIndexChanged.connect(lambda _: self.show_training())
        rv.addWidget(self.trainings)
        self.tr_info = KeyValues(('Model', 'Device', 'Best epoch', 'Export check', 'Peak GPU memory', 'Note'))
        rv.addWidget(self.tr_info)
        self.tr_table = QTreeWidget()
        self.tr_table.setHeaderLabels(['Epoch', 'Loss', 'Validation AP', 'AP50', 'Images/s'])
        self.tr_table.setRootIsDecorated(False)
        rv.addWidget(self.tr_table, 1)
        b = QHBoxLayout()
        b.addWidget(button('Register as a model', self.register_trained))
        b.addStretch(1)
        rv.addLayout(b)
        split.addWidget(right)
        split.setSizes([460, 800])
        return split

    def load_trainings(self):
        self._combo_dirs(self.trainings, self.ws('training'), 'run.json')
        self.val_run.clear()
        self.val_run.addItem('—', '')
        for i in range(self.det_runs.count()):
            self.val_run.addItem(self.det_runs.item(i).text(), self.det_runs.item(i).data(256))
        self.weights.clear()
        dirs = T.weights_dirs(self.win.workspace())
        for name, w in T.WEIGHTS.items():
            fn = os.path.basename(w['url'])
            here = any(os.path.isfile(os.path.join(d, fn)) for d in dirs)
            it = QTreeWidgetItem([name, f'{w.get("bytes", 0) / 1e6:.0f}', 'yes' if here else '—', w['licence']])
            it.setData(0, 256, name)
            it.setToolTip(3, w['licence'])
            self.weights.addTopLevelItem(it)
        self.weights.resizeColumnToContents(0)
        self.show_training()

    def train(self):
        runs = self._checked(self.det_runs)
        val = self.val_run.currentData()
        if not runs:
            QMessageBox.information(self, 'Train', 'Check at least one drive to train on.')
            return
        if val and val in runs:
            QMessageBox.information(self, 'Train', 'The validation drive must not also train the model.')
            return
        if not self._ready():
            return
        tid = f'{time.strftime("%Y%m%d-%H%M%S")}-{self.arch.currentData()}'
        argv = ['--workspace', self.win.workspace(), '--id', tid, '--arch', self.arch.currentData(), '--init',
                self.init.currentData(), '--epochs', str(self.epochs.value()), '--batch', str(self.batch.value()),
                '--size', self.size.text().strip() or '640x384', '--seed', str(self.det_seed.value())]
        for r in runs:
            argv += ['--train', f'falcon:{r}']
        if self.real_src.text().strip():
            argv += ['--train', self.real_src.text().strip()]
        if val:
            argv += ['--val', f'falcon:{val}']
        self._submit(T.job_spec(argv, f'Train {self.arch.currentData()} on {len(runs)} drive(s)'))

    def fetch_weights(self):
        it = self.weights.currentItem()
        if not it:
            return
        name = it.data(0, 256)
        w = T.WEIGHTS[name]
        if QMessageBox.question(self, 'Download the weights?', f'{name} ({w.get("bytes", 0) / 1e6:.0f} MB)\n\nFrom: '
                                f'{w["url"]}\nTerms: {w["licence"]}\nSHA-256: {w["sha256"]}') \
                != QMessageBox.StandardButton.Yes:
            return
        self._cli('fetch', f'Download {name}', ['train', '--fetch', name, '--workspace', self.win.workspace()],
                  memory_gb=0.3)

    def show_training(self):
        self.tr_table.clear()
        d = self.trainings.currentData()
        if not d:
            self.tr_info.clear()
            return
        try:
            run = read_json(os.path.join(d, 'run.json'))
        except (OSError, ValueError):
            return
        st = (run.get('provenance') or {}).get('settings') or {}
        summ = run.get('summary') or {}
        dev = run.get('device') or {}
        self.tr_info.set('Model', f'{st.get("arch", "?")} from {st.get("init", "?")}')
        self.tr_info.set('Device', f'{dev.get("type", "?")}{" " + str(dev["gpu"]) if dev.get("gpu") else ""}'
                                   f'{", AMP" if dev.get("amp") else ""}' if dev else '—')
        self.tr_info.set('Best epoch', f'{summ.get("best_epoch")} (validation AP {_fmt(summ.get("best_val_AP"))})'
                         if summ.get('best_epoch') else '—')
        ver = summ.get('verification')
        self.tr_info.set('Export check', '—' if not ver else ('ONNX matches torch' if ver.get('ok') else
                                                              f'failed: {ver.get("error") or ver}'))
        peak = (run.get('timings') or {}).get('peak_gpu_mib')
        self.tr_info.set('Peak GPU memory', f'{peak:.0f} MiB' if isinstance(peak, (int, float)) else '—')
        self.tr_info.set('Note', f'{run.get("status", "")}' + (f' — {run["error"]}' if run.get('error') else ''))
        for r in _jsonl(os.path.join(d, 'metrics.jsonl')):
            val = r.get('val') or {}
            self.tr_table.addTopLevelItem(QTreeWidgetItem([
                str(r.get('epoch')), _fmt(r.get('train_loss')), _fmt(val.get('AP')), _fmt(val.get('AP50')),
                _fmt(r.get('images_per_s'), 1)]))

    def register_trained(self):
        d = self.trainings.currentData()
        p = os.path.join(d or '', 'model.json')
        if not d or not os.path.isfile(p):
            QMessageBox.information(self, 'Register', 'This training has no exported model yet.')
            return
        m = read_json(p)
        if any(x['id'] == m['id'] for x in self.win.project.doc['models']):
            QMessageBox.information(self, 'Register', 'It is registered already (Models page).')
            return
        self.win.edit(Edit('insert', ['models'], m, text=f'Register {m["name"]}'))
        QMessageBox.information(self, 'Register', f'{m["name"]} is on the Models page, ready to evaluate.')

    # ================================================================== the mixed experiment
    def _experiment_tab(self):
        split = QSplitter()
        left = QWidget()
        lv = QVBoxLayout(left)
        sec = Section('Real only against real + synthetic', 'Every arm trains on the same number of images for the '
                                                           'same number of steps; only the synthetic share changes. '
                                                           'Each arm is scored on the same real test set, against '
                                                           'the real-only arm, with paired intervals and the spread '
                                                           'over seeds.')
        f = Form()
        self.exp_id = QLineEdit('mixed')
        self.real_train = QLineEdit()
        self.real_train.setPlaceholderText('KIND:PATH, e.g. kitti_object:/data/kitti/training')
        self.real_test = QLineEdit()
        self.real_test.setPlaceholderText('KIND:PATH (held out; never trained on)')
        self.shares = QLineEdit('0,25,50,100')
        self.exp_seeds = QSpinBox()
        self.exp_seeds.setRange(1, 20)
        self.exp_seeds.setValue(3)
        self.exp_epochs = QSpinBox()
        self.exp_epochs.setRange(1, 200)
        self.exp_epochs.setValue(6)
        for text, w in (('Experiment', self.exp_id), ('Real training data', self.real_train),
                        ('Real test data', self.real_test), ('Synthetic shares %', self.shares),
                        ('Seeds per arm', self.exp_seeds), ('Epochs', self.exp_epochs)):
            f.row(text, w)
        sec.add(f)
        sec.add(label('Synthetic drives', h=True))
        self.exp_synth = self._runs_list()
        sec.add(self.exp_synth)
        b = QHBoxLayout()
        b.addWidget(button('Run the experiment', self.run_experiment, role='primary'))
        b.addWidget(button('Export for MLflow', self.export_mlflow))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        lv.addStretch(1)
        split.addWidget(scroll(left))
        right = QWidget()
        rv = QVBoxLayout(right)
        self.exps = QComboBox()
        self.exps.currentIndexChanged.connect(lambda _: self.show_experiment())
        rv.addWidget(self.exps)
        self.exp_text = QPlainTextEdit()
        self.exp_text.setReadOnly(True)
        self.exp_text.setObjectName('mono')
        rv.addWidget(self.exp_text, 1)
        split.addWidget(right)
        split.setSizes([460, 800])
        return split

    def load_experiments(self):
        self._combo_dirs(self.exps, self.ws('experiments'), 'experiment.json')
        self.show_experiment()

    def run_experiment(self):
        synth = self._checked(self.exp_synth)
        if not synth or not self.real_train.text().strip() or not self.real_test.text().strip():
            QMessageBox.information(self, 'Experiment', 'Check the synthetic drives and give the real training and '
                                                        'test data.')
            return
        if not self._ready():
            return
        argv = ['experiment', 'mixed', '--workspace', self.win.workspace(), '--id', self.exp_id.text().strip(),
                '--real-train', self.real_train.text().strip(), '--real-test', self.real_test.text().strip(),
                '--shares', self.shares.text().strip(), '--seeds', str(self.exp_seeds.value()), '--epochs',
                str(self.exp_epochs.value())]
        for r in synth:
            argv += ['--synthetic', f'falcon:{r}']
        self._cli('experiment', f'Mixed experiment {self.exp_id.text().strip()}', argv, needs_gpu=True,
                  memory_gb=6.0, gpu_held=True)

    def export_mlflow(self):
        d = self.exps.currentData()
        if not d:
            return
        self._cli('export', 'Export for MLflow', ['experiment', 'export', d, '--mlflow', self.ws('mlruns')],
                  memory_gb=0.3)

    def show_experiment(self):
        from ...core.experiments import table
        d = self.exps.currentData()
        if not d:
            self.exp_text.setPlainText('No experiment yet.')
            return
        p = os.path.join(d, 'results.json')
        if os.path.isfile(p):
            try:
                self.exp_text.setPlainText(table(read_json(p)))
                return
            except (OSError, ValueError, KeyError, TypeError) as e:
                self.exp_text.setPlainText(f'results.json could not be read: {e}')
                return
        n = len(glob.glob(os.path.join(d, 'runs', '*', 'run.json')))
        self.exp_text.setPlainText(f'Running or stopped: {n} run(s) so far; the analysis is written when every arm '
                                   'and seed has finished.')

    # ================================================================== driving policies
    def _policy_tab(self):
        split = QSplitter()
        left = QWidget()
        lv = QVBoxLayout(left)
        sec = Section('Behaviour cloning', 'Learns the recorded control of the checked drives (autopilot or your '
                                           'own driving) from the same features the environment gives a policy. '
                                           'It needs two drives at least (it validates on whole drives), recorded '
                                           'with a labelled camera (they then carry the map\'s lanes).')
        f = Form()
        self.bc_epochs = QSpinBox()
        self.bc_epochs.setRange(1, 1000)
        self.bc_epochs.setValue(30)
        f.row('Epochs', self.bc_epochs)
        sec.add(f)
        self.bc_runs = self._runs_list()
        sec.add(self.bc_runs)
        b = QHBoxLayout()
        b.addWidget(button('Clone', self.bc, role='primary'))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        sec = Section('Reinforcement learning (PPO)', 'Learns in CARLA through the environment, along the default '
                                                      'curriculum of scenarios. Minutes of training do not make a '
                                                      'good driver: read the episodes, not the hopes.')
        f = Form()
        self.ppo_updates = QSpinBox()
        self.ppo_updates.setRange(1, 100000)
        self.ppo_updates.setValue(50)
        self.ppo_steps = QSpinBox()
        self.ppo_steps.setRange(64, 65536)
        self.ppo_steps.setValue(2048)
        f.row('Updates', self.ppo_updates)
        f.row('Steps per update', self.ppo_steps)
        sec.add(f)
        b = QHBoxLayout()
        b.addWidget(button('Train', self.ppo, role='primary'))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        lv.addStretch(1)
        split.addWidget(scroll(left))
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(label('Policies', h=True))
        self.policies = QTreeWidget()
        self.policies.setHeaderLabels(['Policy', 'Method', 'Features', 'Action', 'Result'])
        self.policies.setRootIsDecorated(False)
        rv.addWidget(self.policies, 1)
        split.addWidget(right)
        split.setSizes([460, 800])
        return split

    def _policy_dirs(self):
        out = []
        for p in sorted(glob.glob(self.ws('policies', '*', 'policy.json')) +
                        glob.glob(self.ws('policies', '*', 'policy', 'policy.json')), reverse=True):
            out.append(os.path.dirname(p))
        return out

    def load_policies(self):
        self.policies.clear()
        for d in self._policy_dirs():
            try:
                doc = read_json(os.path.join(d, 'policy.json'))
            except (OSError, ValueError):
                continue
            met = {}
            try:
                met = read_json(os.path.join(d, 'metrics.json'))
            except (OSError, ValueError):
                pass
            res = ', '.join(f'{k} {_fmt(v)}' for k, v in list(met.items())[:3] if isinstance(v, (int, float)))
            name = os.path.relpath(d, self.ws('policies'))
            it = QTreeWidgetItem([name, str(doc.get('trained_by', '')), str(len(doc.get('features') or ())),
                                  str((doc.get('action') or {}).get('kind', '')), res])
            it.setData(0, 256, d)
            self.policies.addTopLevelItem(it)
        self.lb_policy.blockSignals(True)
        cur = self.lb_policy.currentData()
        self.lb_policy.clear()
        for d in self._policy_dirs():
            self.lb_policy.addItem(os.path.relpath(d, self.ws('policies')), d)
        k = self.lb_policy.findData(cur)
        self.lb_policy.setCurrentIndex(k if k >= 0 else 0)
        self.lb_policy.blockSignals(False)

    def bc(self):
        runs = self._checked(self.bc_runs)
        if not runs:
            QMessageBox.information(self, 'Behaviour cloning', 'Check at least one drive.')
            return
        out = self.new_dir('policies', f'bc-{time.strftime("%Y%m%d-%H%M%S")}')
        argv = ['policy', 'bc', '--out', out, '--epochs', str(self.bc_epochs.value())]
        for r in runs:
            argv += ['--run', r]
        self._cli('policy', f'Behaviour cloning on {len(runs)} drive(s)', argv, memory_gb=2.0)

    def ppo(self):
        if not self._ready():
            return
        out = self.new_dir('policies', f'ppo-{time.strftime("%Y%m%d-%H%M%S")}')
        argv = ['policy', 'ppo', self.win.project.path, '--out', out, '--curriculum', 'default', '--updates',
                str(self.ppo_updates.value()), '--steps', str(self.ppo_steps.value())] + self._carla_args()
        self._cli('policy', f'PPO, {self.ppo_updates.value()} updates', argv, memory_gb=2.0)

    # ================================================================== the leaderboard
    def _leaderboard_tab(self):
        split = QSplitter()
        left = QWidget()
        lv = QVBoxLayout(left)
        sec = Section('Score a policy', 'Each route scored as the CARLA Leaderboard scores one: route completion × '
                                        'the infraction penalty (pedestrian 0.50, vehicle 0.60, static 0.65, red '
                                        'light 0.70, stop sign 0.80). Falcon\'s routes are not the Leaderboard\'s: '
                                        'compare policies here, not with its public results.')
        f = Form()
        self.lb_policy = QComboBox()
        self.lb_seeds = QSpinBox()
        self.lb_seeds.setRange(1, 100)
        self.lb_seeds.setValue(3)
        self.lb_free = QSpinBox()
        self.lb_free.setRange(0, 100)
        self.lb_free.setValue(3)
        f.row('Policy', self.lb_policy)
        f.row('Seeds per scenario', self.lb_seeds)
        f.row('Free routes', self.lb_free)
        sec.add(f)
        self.lb_scn = QListWidget()
        self.lb_scn.setMinimumHeight(150)
        for k in SC.LIBRARY:
            it = QListWidgetItem(SC.builtin(k)['name'])
            it.setData(256, k)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Unchecked)
            self.lb_scn.addItem(it)
        sec.add(self.lb_scn)
        b = QHBoxLayout()
        b.addWidget(button('Score', self.score_policy, role='primary'))
        b.addStretch(1)
        sec.add(b)
        lv.addWidget(sec)
        lv.addStretch(1)
        split.addWidget(scroll(left))
        right = QSplitter(Qt.Orientation.Vertical)
        top = QWidget()
        tv = QVBoxLayout(top)
        self.lbs = QComboBox()
        self.lbs.currentIndexChanged.connect(lambda _: self.show_leaderboard())
        tv.addWidget(self.lbs)
        self.lb_info = KeyValues(('Driving score', 'Route completion', 'Infraction penalty', 'Routes',
                                  'Infractions per km', 'Not scored'))
        tv.addWidget(self.lb_info)
        self.infr = QTreeWidget()
        self.infr.setHeaderLabels(['Route', 'Scenario', 'DS', 'Infraction', 'Step', 'Time'])
        self.infr.setRootIsDecorated(False)
        tv.addWidget(self.infr, 1)
        b = QHBoxLayout()
        b.addWidget(button('Replay the selected infraction', self.replay))
        b.addStretch(1)
        tv.addLayout(b)
        right.addWidget(top)
        self.flip = Flipbook()
        right.addWidget(self.flip)
        split.addWidget(right)
        split.setSizes([420, 860])
        return split

    def load_leaderboards(self):
        self._combo_dirs(self.lbs, self.ws('leaderboard'), 'leaderboard.json')
        self.show_leaderboard()

    def score_policy(self):
        pol = self.lb_policy.currentData()
        if not pol:
            QMessageBox.information(self, 'Leaderboard', 'Train or clone a policy first.')
            return
        if not self._ready():
            return
        scn = [self.lb_scn.item(i).data(256) for i in range(self.lb_scn.count())
               if self.lb_scn.item(i).checkState() == Qt.CheckState.Checked]
        out = self.new_dir('leaderboard', f'{time.strftime("%Y%m%d-%H%M%S")}-{os.path.basename(pol)}')
        argv = ['policy', 'leaderboard', self.win.project.path, '--policy', pol, '--out', out, '--seeds-per',
                str(self.lb_seeds.value()), '--free', str(self.lb_free.value())] + self._carla_args()
        for k in scn:
            argv += ['--scenario', k]
        self._cli('leaderboard', f'Leaderboard: {os.path.basename(pol)}', argv, memory_gb=2.0)

    def show_leaderboard(self):
        self.infr.clear()
        d = self.lbs.currentData()
        if not d:
            self.lb_info.clear()
            return
        try:
            doc = read_json(os.path.join(d, 'leaderboard.json'))
        except (OSError, ValueError):
            return
        m = doc['metrics']
        for key, name in (('driving_score', 'Driving score'), ('route_completion', 'Route completion'),
                          ('infraction_penalty', 'Infraction penalty')):
            x = m[key]
            self.lb_info.set(name, f'{x["value"]:.3f} [{_fmt(x["ci_low"])}, {_fmt(x["ci_high"])}]')
        self.lb_info.set('Routes', f'{doc["routes"]} ({doc["distance_km"]:.2f} km)' +
                         (f'; {len(doc["failed"])} scenario(s) failed to run' if doc.get('failed') else ''))
        self.lb_info.set('Infractions per km', ', '.join(f'{k} {_fmt(v, 2)}' for k, v in
                                                         doc['infractions_per_km'].items()) or 'none')
        self.lb_info.set('Not scored', doc.get('not_scored', ''))
        eps = _jsonl(os.path.join(d, 'episodes.jsonl'))
        per = doc.get('per_route') or []
        for i, e in enumerate(eps):
            ds = per[i]['driving_score'] if i < len(per) else None
            scn = (e.get('scenario') or {}).get('id') or 'free'
            infs = e.get('infractions') or []
            if not infs:
                it = QTreeWidgetItem([str(i), scn, _fmt(ds), 'none', '', ''])
                self.infr.addTopLevelItem(it)
                continue
            for k, inf in enumerate(infs):
                it = QTreeWidgetItem([str(i), scn, _fmt(ds), inf['kind'], str(inf['step']),
                                      f'{inf.get("sim_t", 0):.2f} s'])
                it.setData(0, 256, (i, k))
                if inf['kind'] in LB.PENALTY:
                    it.setForeground(3, QColor(STATUS['critical']))
                self.infr.addTopLevelItem(it)
        for c in range(4):
            self.infr.resizeColumnToContents(c)
        rp = sorted(glob.glob(os.path.join(d, 'replays', '*', 'replay.json')))
        self.flip.load(read_json(rp[-1]) if rp else None)

    def replay(self):
        d = self.lbs.currentData()
        it = self.infr.currentItem()
        if not d or not it or it.data(0, 256) is None:
            QMessageBox.information(self, 'Replay', 'Select an infraction.')
            return
        i, k = it.data(0, 256)
        eps = _jsonl(os.path.join(d, 'episodes.jsonl'))
        why = LB.replayable(eps[i]) if i < len(eps) else 'the episode is missing'
        if why:
            QMessageBox.information(self, 'Replay', f'This episode cannot be replayed exactly: {why}.')
            return
        out = os.path.join(d, 'replays', f'{i:03d}-{k:02d}')
        argv = ['policy', 'replay', self.win.project.path, '--episodes', os.path.join(d, 'episodes.jsonl'),
                '--episode', str(i), '--infraction', str(k), '--out', out] + self._carla_args()
        self._cli('replay', f'Replay route {i}: {eps[i]["infractions"][k]["kind"]}', argv, memory_gb=1.0)
