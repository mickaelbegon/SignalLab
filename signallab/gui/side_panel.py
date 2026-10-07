"""Un côté (Gauche ou Droite) : signal, lecture, figure, chaîne de traitements."""
from __future__ import annotations

import csv
import html

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QMenu,
                               QPushButton, QSplitter, QToolButton, QVBoxLayout, QWidget)

from .. import audio, dsp
from . import plotting
from .chain_widget import ChainWidget

DEBOUNCE_MS = 150


class SidePanel(QWidget):
    def __init__(self, name, owner):
        super().__init__()
        self.name = name
        self.owner = owner          # fenêtre principale (settings, player, infos, status)
        self.x = None
        self.y = None
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

        self.chain = ChainWidget()
        split.addWidget(self.chain)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([520, 330])
        root.addWidget(split, 1)

        self.combo.currentIndexChanged.connect(self._on_signal_changed)
        self.chain.changed.connect(self.schedule)
        self.sp_t0.valueChanged.connect(self.redraw)
        self.sp_dur.valueChanged.connect(self.redraw)
        self.btn_all.clicked.connect(self.show_all)
        self.btn_orig.clicked.connect(lambda: self.play(False))
        self.btn_proc.clicked.connect(lambda: self.play(True))
        self.btn_stop.clicked.connect(self.owner.player.stop)

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
        self.chain.set_kind(self.info.kind)
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
        chain = self.chain.chain()
        self.y = None
        if chain:
            try:
                y = dsp.apply_chain(self.x, self.fs, chain)
                self.y = np.nan_to_num(np.asarray(y, dtype=float))
            except Exception as e:  # ne jamais planter l'interface
                self.owner.status(f"{self.name} : erreur de traitement ({e})")
        self.cache.clear()
        self.redraw()

    def processed(self):
        """Signal traité (ou original si la chaîne est vide)."""
        return self.y if self.y is not None else self.x

    # ----------------------------------------------------------------- tracé
    def redraw(self, *_):
        if self.x is None:
            return
        s = self.owner.settings
        try:
            if s.mode == "time":
                plotting.plot_time(self.fig, self.info, self.x, self.y, self.fs,
                                   self.sp_t0.value(), self.sp_dur.value(), s.overlay)
            elif s.mode == "freq":
                plotting.plot_freq(self.fig, self.info, self.x, self.y, self.fs, s.fmax,
                                   s.log_f, s.db, s.overlay, self.cache)
            else:
                plotting.plot_spectrogram(self.fig, self.info, self.x, self.y, self.fs, s.fmax, s.log_f)
        except Exception as e:
            self.owner.status(f"{self.name} : erreur de tracé ({e})")
        self.canvas.draw_idle()
        self.owner.sync_axes()
        n = len(self.chain.chain())
        self.owner.status(f"{self.name} : {self.info.label} – fs = {self.fs:g} Hz – {n} traitement(s) actif(s)")

    # ----------------------------------------------------------------- audio
    def play(self, processed):
        if self.x is None:
            return
        sig = self.processed() if processed else self.x
        pcm, fs = audio.to_playable(sig, self.fs, self.info.audify_speed)
        what = "traité" if processed else "original"
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
            w.writerow(["t_s", "original", "traite"])
            for row in zip(t, self.x, y):
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
