"""Un côté (Gauche ou Droite) : signal, lecture, figure, chaîne de traitements."""
from __future__ import annotations

import csv
import html

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QMenu,
                               QPushButton, QSplitter, QStackedWidget, QTabBar, QToolButton, QVBoxLayout,
                               QWidget)

from .. import audio, dsp
from . import plotting
from .chain_widget import ChainWidget

DEBOUNCE_MS = 150
BRANCH_COLORS = ["#ff7f0e", "#2ca02c", "#9467bd", "#17becf", "#8c564b", "#e377c2", "#bcbd22"]


class Branch:
    """Une branche = une chaîne de traitements appliquée à l'original, tracée dans sa couleur."""

    def __init__(self, name, color, chain):
        self.name = name
        self.color = color
        self.chain = chain
        self.y = None


class SidePanel(QWidget):
    def __init__(self, name, owner):
        super().__init__()
        self.name = name
        self.owner = owner          # fenêtre principale (settings, player, infos, status)
        self.x = None
        self.info = None
        self.fs = 1.0
        self.cache = {}

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(DEBOUNCE_MS)
        self.timer.timeout.connect(self.recompute)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        title = QLabel(f"<h2>{name}</h2>")
        root.addWidget(title)

        row = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.setMinimumWidth(300)
        for info in owner.infos:
            self.combo.addItem(info.label, info.id)
        self.btn_orig = QPushButton("▶ Jouer original")
        self.btn_proc = QPushButton("▶ Jouer traité")
        self.btn_stop = QPushButton("■ Stop")
        self.btn_orig.setToolTip("Écouter le signal original (EMG/cinématique : rendu sonore accéléré)")
        self.btn_proc.setToolTip("Écouter le signal après la chaîne de traitements")
        self.btn_export = QToolButton()
        self.btn_export.setText("Exporter ▾")
        self.btn_export.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.btn_export)
        menu.addAction("Signal traité en WAV…", self.export_wav_dialog)
        menu.addAction("Signal traité en CSV…", self.export_csv_dialog)
        menu.addAction("Figure en PNG…", self.export_png_dialog)
        self.btn_export.setMenu(menu)
        row.addWidget(self.combo, 1)
        row.addWidget(self.btn_orig)
        row.addWidget(self.btn_proc)
        row.addWidget(self.btn_stop)
        row.addWidget(self.btn_export)
        root.addLayout(row)

        self.desc = QLabel()
        self.desc.setWordWrap(True)
        self.desc.setObjectName("desc")
        root.addWidget(self.desc)

        split = QSplitter(Qt.Vertical)
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(layout="constrained", facecolor="white")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(220)
        self.toolbar = NavToolbar(self.canvas, self)
        tl.addWidget(self.toolbar)
        tl.addWidget(self.canvas, 1)
        win = QHBoxLayout()
        win.addWidget(QLabel("Fenêtre temporelle : début"))
        self.sp_t0 = QDoubleSpinBox()
        self.sp_dur = QDoubleSpinBox()
        for sp, suf in ((self.sp_t0, " s"), (self.sp_dur, " s")):
            sp.setDecimals(3)
            sp.setSuffix(suf)
            sp.setKeyboardTracking(False)
            sp.setRange(0.0, 3600.0)
        self.sp_dur.setMinimum(0.001)
        self.sp_t0.setSingleStep(0.01)
        self.sp_dur.setSingleStep(0.01)
        self.sp_t0.setToolTip("Début de la fenêtre affichée en mode Temps")
        self.sp_dur.setToolTip("Durée de la fenêtre affichée en mode Temps (zoom)")
        self.btn_all = QPushButton("Tout")
        self.btn_all.setToolTip("Afficher tout le signal")
        win.addWidget(self.sp_t0)
        win.addWidget(QLabel("durée"))
        win.addWidget(self.sp_dur)
        win.addWidget(self.btn_all)
        win.addStretch(1)
        tl.addLayout(win)
        split.addWidget(top)

        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        brow = QHBoxLayout()
        self.tabs = QTabBar()
        self.tabs.setExpanding(False)
        self.tabs.setTabsClosable(False)
        self.tabs.setToolTip("Chaque branche applique sa propre chaîne de traitements à l'original")
        self.btn_new = QPushButton("＋ Nouvelle branche")
        self.btn_new.setToolTip("Nouvelle branche vide, repartant du signal original")
        self.btn_dup = QPushButton("⑂ Dupliquer")
        self.btn_dup.setToolTip("Copie la branche active pour en dériver une variante")
        brow.addWidget(self.tabs, 1)
        brow.addWidget(self.btn_new)
        brow.addWidget(self.btn_dup)
        bl.addLayout(brow)
        self.stack = QStackedWidget()
        bl.addWidget(self.stack, 1)
        self.branches = []
        self._counter = 0
        split.addWidget(bottom)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([520, 330])
        root.addWidget(split, 1)

        self.combo.currentIndexChanged.connect(self._on_signal_changed)
        self.tabs.currentChanged.connect(self._on_tab)
        self.tabs.tabCloseRequested.connect(self.remove_branch)
        self.btn_new.clicked.connect(lambda: self.add_branch())
        self.btn_dup.clicked.connect(self.duplicate_branch)
        self.add_branch()
        self.sp_t0.valueChanged.connect(self.redraw)
        self.sp_dur.valueChanged.connect(self.redraw)
        self.btn_all.clicked.connect(self.show_all)
        self.btn_orig.clicked.connect(lambda: self.play(False))
        self.btn_proc.clicked.connect(lambda: self.play(True))
        self.btn_stop.clicked.connect(self.owner.player.stop)

    # --------------------------------------------------------------- branches
    @property
    def active(self):
        return self.branches[max(self.tabs.currentIndex(), 0)]

    @property
    def chain(self):
        """ChainWidget de la branche active."""
        return self.active.chain

    @property
    def y(self):
        """Signal traité de la branche active (None si sa chaîne est vide)."""
        return self.active.y

    def add_branch(self, chain=None):
        self._counter += 1
        color = BRANCH_COLORS[(self._counter - 1) % len(BRANCH_COLORS)]
        cw = ChainWidget()
        if self.info is not None:
            cw.set_kind(self.info.kind)
        cw.changed.connect(self.schedule)
        br = Branch(f"Branche {self._counter}", color, cw)
        self.branches.append(br)
        self.stack.addWidget(cw)
        i = self.tabs.addTab(br.name)
        self.tabs.setTabTextColor(i, QColor(color))
        self.tabs.setTabsClosable(len(self.branches) > 1)
        self.tabs.setCurrentIndex(i)
        if chain:
            cw.set_chain(chain)
        self.schedule()
        return br

    def duplicate_branch(self):
        self.add_branch(self.chain.chain())

    def remove_branch(self, i):
        if len(self.branches) <= 1 or not 0 <= i < len(self.branches):
            return
        br = self.branches.pop(i)
        self.tabs.removeTab(i)
        self.stack.removeWidget(br.chain)
        br.chain.deleteLater()
        self.tabs.setTabsClosable(len(self.branches) > 1)
        self.schedule()

    def _on_tab(self, i):
        if 0 <= i < len(self.branches):
            self.stack.setCurrentWidget(self.branches[i].chain)
            self.redraw()

    def active_branches(self):
        return [(b.name, b.color, b.y) for b in self.branches if b.y is not None]

    # ------------------------------------------------------------------ signal
    def set_signal(self, sig_id):
        i = self.combo.findData(sig_id)
        if i >= 0:
            if self.combo.currentIndex() == i:
                self._on_signal_changed()
            else:
                self.combo.setCurrentIndex(i)

    def _on_signal_changed(self, *_):
        sig_id = self.combo.currentData()
        if sig_id is None:
            return
        self.x, self.info = self.owner.load(sig_id)
        self.fs = float(self.info.fs)
        for b in self.branches:
            b.chain.set_kind(self.info.kind)
        text = html.escape(self.info.description)
        if self.info.source:
            text += f"  <i>[{html.escape(self.info.source)}]</i>"
        self.desc.setText(text)
        total = len(self.x) / self.fs
        if self.info.kind == "audio":
            dur = min(total, 8.0 / self.info.f0 if self.info.f0 else 0.05)
            dur = max(dur, 0.01)
        else:
            dur = total
        for sp in (self.sp_t0, self.sp_dur):
            sp.blockSignals(True)
        self.sp_t0.setMaximum(max(total - 0.001, 0.0))
        self.sp_dur.setMaximum(max(total, 0.001))
        self.sp_t0.setValue(0.0)
        self.sp_dur.setValue(dur)
        for sp in (self.sp_t0, self.sp_dur):
            sp.blockSignals(False)
        self.recompute()

    def show_all(self):
        if self.x is None:
            return
        self.sp_t0.blockSignals(True)
        self.sp_t0.setValue(0.0)
        self.sp_t0.blockSignals(False)
        self.sp_dur.setValue(len(self.x) / self.fs)

    # ---------------------------------------------------------------- calcul
    def schedule(self):
        self.timer.start()

    def recompute(self):
        self.timer.stop()
        if self.x is None:
            return
        for b in self.branches:
            chain = b.chain.chain()
            b.y = None
            if chain:
                try:
                    y = dsp.apply_chain(self.x, self.fs, chain)
                    b.y = np.nan_to_num(np.asarray(y, dtype=float))
                except Exception as e:  # ne jamais planter l'interface
                    self.owner.status(f"{self.name} : erreur de traitement ({b.name} : {e})")
        self.cache.clear()
        self.redraw()

    def processed(self):
        """Signal traité de la branche active (ou original si sa chaîne est vide)."""
        return self.y if self.y is not None else self.x

    # ----------------------------------------------------------------- tracé
    def redraw(self, *_):
        if self.x is None:
            return
        s = self.owner.settings
        br = self.active_branches()
        marker = next((k for k, b in enumerate(br) if b[2] is self.active.y), -1)
        try:
            if s.mode == "time":
                plotting.plot_time(self.fig, self.info, self.x, br, self.fs,
                                   self.sp_t0.value(), self.sp_dur.value(), s.overlay)
            elif s.mode == "freq":
                plotting.plot_freq(self.fig, self.info, self.x, br, self.fs, s.fmax,
                                   s.log_f, s.db, s.overlay, self.cache, marker)
            else:
                plotting.plot_spectrogram(self.fig, self.info, self.x, br, self.fs, s.fmax, s.log_f)
        except Exception as e:
            self.owner.status(f"{self.name} : erreur de tracé ({e})")
        self.canvas.draw_idle()
        self.owner.sync_axes()
        n = sum(len(b.chain.chain()) for b in self.branches)
        self.owner.status(f"{self.name} : {self.info.label} – fs = {self.fs:g} Hz – "
                          f"{len(self.branches)} branche(s), {n} traitement(s) actif(s)")

    # ----------------------------------------------------------------- audio
    def play(self, processed):
        if self.x is None:
            return
        sig = self.processed() if processed else self.x
        pcm, fs = audio.to_playable(sig, self.fs, self.info.audify_speed)
        what = f"traité ({self.active.name})" if processed else "original"
        if self.owner.player.play(pcm, fs):
            self.owner.status(f"Lecture : {self.name} – {what} ({len(pcm) / fs:.1f} s)")
        else:
            self.owner.status("Lecture impossible : aucune sortie audio disponible")

    # --------------------------------------------------------------- export
    def export_wav(self, path):
        pcm, fs = audio.to_playable(self.processed(), self.fs, self.info.audify_speed)
        audio.write_wav(path, pcm, fs)

    def export_csv(self, path):
        t = np.arange(len(self.x)) / self.fs
        y = self.processed()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["# signal", self.info.id, "fs_Hz", self.fs])
            outs = [b for b in self.branches if b.y is not None]
            cols = [b.y for b in outs] or [y]
            w.writerow(["t_s", "original"] + ([b.name for b in outs] or ["traite"]))
            for row in zip(t, self.x, *cols):
                w.writerow([f"{v:.9g}" for v in row])

    def export_png(self, path):
        self.fig.savefig(path, dpi=150)

    def _ask(self, title, filt, default):
        path, _ = QFileDialog.getSaveFileName(self, title, default, filt)
        return path

    def export_wav_dialog(self):
        p = self._ask("Exporter en WAV", "WAV (*.wav)", f"{self.info.id}_traite.wav")
        if p:
            self.export_wav(p)
            self.owner.status(f"Exporté : {p}")

    def export_csv_dialog(self):
        p = self._ask("Exporter en CSV", "CSV (*.csv)", f"{self.info.id}_traite.csv")
        if p:
            self.export_csv(p)
            self.owner.status(f"Exporté : {p}")

    def export_png_dialog(self):
        p = self._ask("Enregistrer la figure", "PNG (*.png)", f"{self.info.id}.png")
        if p:
            self.export_png(p)
            self.owner.status(f"Exporté : {p}")
