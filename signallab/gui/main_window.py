"""Fenêtre principale de SignalLab."""
from __future__ import annotations

import sys
from dataclasses import dataclass

import matplotlib
import numpy as np

matplotlib.use("QtAgg")

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout,
                               QLabel, QMainWindow, QMenu, QPushButton, QToolButton, QVBoxLayout, QWidget)

from .. import audio, signals
from .side_panel import SidePanel

STYLE = """
* { font-size: 13pt; }
QMainWindow, QWidget#central { background: #f6f7f9; }
QLabel#desc { color: #333; background: #eef2f7; border-radius: 4px; padding: 4px 6px; font-size: 11pt; }
QFrame#procItem { background: white; border: 1px solid #c8ced6; border-radius: 6px; }
QPushButton, QToolButton { padding: 4px 10px; }
QPushButton:checked { background: #1f77b4; color: white; }
QPushButton#mode { min-width: 120px; }
QPushButton#mode:checked { background: #1f77b4; color: white; font-weight: bold; }
QToolTip { font-size: 12pt; padding: 4px; }
QStatusBar { font-size: 11pt; }
"""

# (libellé, type de signal requis, préférences d'identifiant, chaîne)
PRESETS = [
    ("Retirer le 50 Hz de l'EMG", "emg", ["raw", ""],
     [("notch", {"f0": 50.0, "width": 4.0})]),
    ("Enveloppe EMG", "emg", ["raw", ""],
     [("bandpass", {"f_low": 20.0, "f_high": 450.0}), ("remove_mean", {}),
      ("envelope_lowpass", {"fc": 6.0})]),
    ("Aliasing", "audio", ["sine", "flute", ""],
     [("downsample", {"factor": 4})]),
    ("Voix téléphone (passe-bande 300–3400)", "audio", ["piano", ""],
     [("bandpass", {"f_low": 300.0, "f_high": 3400.0})]),
]


@dataclass
class PlotSettings:
    mode: str = "time"       # "time" | "freq" | "spec"
    log_f: bool = False
    db: bool = False
    fmax: float = 5000.0
    overlay: bool = True
    link: bool = False


class MainWindow(QMainWindow):
    def __init__(self, player=None, left_id=None, right_id=None):
        super().__init__()
        self.setWindowTitle("SignalLab – voir et entendre le traitement du signal")
        self.resize(1700, 1000)
        self.settings = PlotSettings()
        self.player = player if player is not None else audio.Player(self)
        self.infos = signals.list_signals()
        self._cache = {}

        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.addLayout(self._build_toolbar())

        cols = QHBoxLayout()
        self.left = SidePanel("Gauche", self)
        self.right = SidePanel("Droite", self)
        cols.addWidget(self.left, 1)
        cols.addWidget(self.right, 1)
        root.addLayout(cols, 1)
        self.panels = [self.left, self.right]
        self.statusBar().showMessage("Prêt")
        self.player.stateChanged.connect(self._on_play_state)

        ids = [i.id for i in self.infos]
        lid = left_id or (ids[0] if ids else None)
        rid = right_id or (next((i.id for i in self.infos if i.kind == "emg"), None) or (ids[-1] if ids else None))
        if lid:
            self.left.set_signal(lid)
        if rid:
            self.right.set_signal(rid)

    # --------------------------------------------------------------- services
    def load(self, sig_id):
        if sig_id not in self._cache:
            r = signals.load_signal(sig_id)
            if isinstance(r, tuple):
                x, info = r
            else:  # tolérance : chargeur renvoyant seulement le tableau
                x, info = r, next(i for i in self.infos if i.id == sig_id)
            self._cache[sig_id] = (np.asarray(x, dtype=float), info)
        return self._cache[sig_id]

    def status(self, msg):
        self.statusBar().showMessage(msg)

    def _on_play_state(self, playing):
        if not playing:
            self.status("Lecture terminée")

    # ---------------------------------------------------------------- toolbar
    def _build_toolbar(self):
        lay = QHBoxLayout()
        lay.addWidget(QLabel("<b>Affichage :</b>"))
        self.mode_group = QButtonGroup(self)
        self.mode_btns = {}
        for key, label, tip in (("time", "Temps", "Signal en fonction du temps"),
                                ("freq", "Fréquence", "Spectre d'amplitude (FFT) avec fondamentale et harmoniques"),
                                ("spec", "Spectrogramme", "Contenu fréquentiel en fonction du temps")):
            b = QPushButton(label)
            b.setObjectName("mode")
            b.setCheckable(True)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: self.set_mode(k))
            self.mode_group.addButton(b)
            self.mode_btns[key] = b
            lay.addWidget(b)
        self.mode_btns["time"].setChecked(True)
        self.btn_toggle = QPushButton("Temps ⇄ Fréquence")
        self.btn_toggle.setToolTip("Basculer entre l'affichage temporel et fréquentiel")
        self.btn_toggle.clicked.connect(self.toggle_mode)
        lay.addWidget(self.btn_toggle)

        lay.addSpacing(16)
        lay.addWidget(QLabel("Axe f :"))
        self.cb_logf = QComboBox()
        self.cb_logf.addItems(["Linéaire", "Log"])
        self.cb_logf.currentIndexChanged.connect(lambda i: self._set("log_f", bool(i)))
        lay.addWidget(self.cb_logf)
        lay.addWidget(QLabel("Amplitude :"))
        self.cb_db = QComboBox()
        self.cb_db.addItems(["Linéaire", "dB"])
        self.cb_db.currentIndexChanged.connect(lambda i: self._set("db", bool(i)))
        lay.addWidget(self.cb_db)
        lay.addWidget(QLabel("f max :"))
        self.sp_fmax = QDoubleSpinBox()
        self.sp_fmax.setRange(1, 24000)
        self.sp_fmax.setDecimals(0)
        self.sp_fmax.setSingleStep(100)
        self.sp_fmax.setSuffix(" Hz")
        self.sp_fmax.setValue(self.settings.fmax)
        self.sp_fmax.setKeyboardTracking(False)
        self.sp_fmax.setToolTip("Zoom fréquentiel : fréquence maximale affichée")
        self.sp_fmax.valueChanged.connect(lambda v: self._set("fmax", float(v)))
        lay.addWidget(self.sp_fmax)

        self.ck_overlay = QCheckBox("Superposer l'original")
        self.ck_overlay.setChecked(True)
        self.ck_overlay.toggled.connect(lambda v: self._set("overlay", v))
        self.ck_link = QCheckBox("Axes liés G/D")
        self.ck_link.setToolTip("Utiliser les mêmes limites d'axes des deux côtés pour comparer")
        self.ck_link.toggled.connect(lambda v: self._set("link", v))
        lay.addWidget(self.ck_overlay)
        lay.addWidget(self.ck_link)
        lay.addStretch(1)

        self.btn_presets = QToolButton()
        self.btn_presets.setText("Presets ▾")
        self.btn_presets.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.btn_presets)
        for side_name, idx in (("Gauche", 0), ("Droite", 1)):
            sub = menu.addMenu(f"Appliquer à {side_name}")
            for p in range(len(PRESETS)):
                sub.addAction(PRESETS[p][0], lambda i=idx, p=p: self.apply_preset(i, p))
        self.btn_presets.setMenu(menu)
        lay.addWidget(self.btn_presets)
        return lay

    def _set(self, name, value):
        setattr(self.settings, name, value)
        self.redraw_all()

    def set_mode(self, mode):
        self.settings.mode = mode
        self.mode_btns[mode].setChecked(True)
        self.redraw_all()

    def toggle_mode(self):
        self.set_mode("freq" if self.settings.mode != "freq" else "time")

    def redraw_all(self):
        for p in getattr(self, "panels", []):
            p.redraw()
        self.sync_axes()

    def sync_axes(self):
        """Axes liés : mêmes limites x/y des deux côtés (union)."""
        if not self.settings.link or self.settings.mode == "spec" or len(getattr(self, "panels", [])) < 2:
            return
        axes = [p.fig.axes[0] for p in self.panels if p.fig.axes]
        if len(axes) < 2 or axes[0].get_xscale() != axes[1].get_xscale():
            return
        xl = (min(a.get_xlim()[0] for a in axes), max(a.get_xlim()[1] for a in axes))
        yl = (min(a.get_ylim()[0] for a in axes), max(a.get_ylim()[1] for a in axes))
        for a in axes:
            a.set_xlim(xl)
            a.set_ylim(yl)
        for p in self.panels:
            p.canvas.draw_idle()

    # ---------------------------------------------------------------- presets
    def apply_preset(self, side_idx, preset_idx):
        label, kind, prefs, chain = PRESETS[preset_idx]
        panel = self.panels[side_idx]
        if panel.info is None or panel.info.kind != kind:
            cands = [i for i in self.infos if i.kind == kind]
            if not cands:
                self.status("Preset indisponible : aucun signal de ce type")
                return
            pick = cands[0]
            for pref in prefs:
                m = next((i for i in cands if pref in i.id), None)
                if m:
                    pick = m
                    break
            panel.set_signal(pick.id)
        panel.chain.set_chain(chain)
        panel.recompute()
        self.status(f"Preset « {label} » appliqué ({panel.name})")

    def closeEvent(self, e):
        self.player.stop()
        super().closeEvent(e)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    f = QFont(app.font())
    f.setPointSize(13)
    app.setFont(f)
    matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                                "legend.fontsize": 12, "lines.linewidth": 1.5})
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
