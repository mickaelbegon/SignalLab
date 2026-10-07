"""Chaîne de traitements : widgets de paramètres générés depuis ParamSpec."""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
                               QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget)

from .. import dsp

SLIDER_STEPS = 1000


def _decimals(step):
    if step >= 1:
        return 0
    return min(4, max(1, int(math.ceil(-math.log10(step)))))


class ParamWidget(QWidget):
    """Slider + spinbox synchronisés pour un ParamSpec."""

    changed = Signal()

    def __init__(self, ps, value):
        super().__init__()
        self.ps = ps
        self.is_int = isinstance(ps.default, int) and not isinstance(ps.default, bool)
        self._log = bool(ps.log) and ps.min > 0
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(ps.label)
        self.label.setMinimumWidth(190)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(0 if self.is_int else _decimals(ps.step))
        self.spin.setRange(ps.min, ps.max)
        self.spin.setSingleStep(ps.step)
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(120)
        if ps.unit:
            self.spin.setSuffix(" " + ps.unit)
        lay.addWidget(self.label)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self.set_value(value)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _to_pos(self, v):
        lo, hi = self.ps.min, self.ps.max
        v = min(max(v, lo), hi)
        if hi <= lo:
            return 0
        if self._log:
            r = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
        else:
            r = (v - lo) / (hi - lo)
        return int(round(r * SLIDER_STEPS))

    def _from_pos(self, pos):
        lo, hi = self.ps.min, self.ps.max
        r = pos / SLIDER_STEPS
        v = math.exp(math.log(lo) + r * (math.log(hi) - math.log(lo))) if self._log else lo + r * (hi - lo)
        if self.is_int:
            return float(round(v))
        if self._log:  # arrondi à 3 chiffres significatifs
            return float(f"{v:.3g}")
        step = self.ps.step
        return round(round(v / step) * step, 6) if step > 0 else v

    def _from_slider(self, pos):
        self.spin.blockSignals(True)
        self.spin.setValue(self._from_pos(pos))
        self.spin.blockSignals(False)
        self.changed.emit()

    def _from_spin(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_pos(v))
        self.slider.blockSignals(False)
        self.changed.emit()

    def set_value(self, v):
        self.spin.blockSignals(True)
        self.slider.blockSignals(True)
        self.spin.setValue(v)
        self.slider.setValue(self._to_pos(self.spin.value()))
        self.spin.blockSignals(False)
        self.slider.blockSignals(False)

    def value(self):
        v = self.spin.value()
        return int(round(v)) if self.is_int else float(v)


class ProcessorItem(QFrame):
    changed = Signal()
    move = Signal(object, int)
    remove = Signal(object)

    def __init__(self, spec, values):
        super().__init__()
        self.spec = spec
        self.setObjectName("procItem")
        self.setFrameShape(QFrame.StyledPanel)
        self.setToolTip(spec.help)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(2)
        head = QHBoxLayout()
        self.title = QLabel(f"<b>{spec.label}</b>")
        self.title.setToolTip(spec.help)
        head.addWidget(self.title, 1)
        self.bypass = QPushButton("Bypass")
        self.bypass.setCheckable(True)
        self.bypass.setToolTip("Désactiver temporairement ce traitement (comparer avant/après)")
        self.btn_up = QPushButton("▲")
        self.btn_down = QPushButton("▼")
        self.btn_del = QPushButton("✕")
        self.btn_up.setToolTip("Monter dans la chaîne")
        self.btn_down.setToolTip("Descendre dans la chaîne")
        self.btn_del.setToolTip("Supprimer ce traitement")
        for b in (self.btn_up, self.btn_down, self.btn_del):
            b.setFixedWidth(40)
        for b in (self.bypass, self.btn_up, self.btn_down, self.btn_del):
            head.addWidget(b)
        root.addLayout(head)
        self.params = {}
        for ps in spec.params:
            pw = ParamWidget(ps, values.get(ps.key, ps.default))
            pw.setToolTip(spec.help)
            pw.changed.connect(self.changed)
            self.params[ps.key] = pw
            root.addWidget(pw)
        self.bypass.toggled.connect(self._on_bypass)
        self.btn_up.clicked.connect(lambda: self.move.emit(self, -1))
        self.btn_down.clicked.connect(lambda: self.move.emit(self, +1))
        self.btn_del.clicked.connect(lambda: self.remove.emit(self))

    def _on_bypass(self, on):
        self.title.setEnabled(not on)
        for pw in self.params.values():
            pw.setEnabled(not on)
        self.changed.emit()

    def values(self):
        return {k: pw.value() for k, pw in self.params.items()}

    def active(self):
        return not self.bypass.isChecked()


class ChainWidget(QWidget):
    """Liste ordonnée de traitements pour un côté."""

    changed = Signal()

    def __init__(self):
        super().__init__()
        self.kind = "audio"
        self.items: list[ProcessorItem] = []
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.setMinimumWidth(320)
        self.combo.activated.connect(self._on_combo)
        self.btn_clear = QPushButton("Tout effacer")
        self.btn_clear.clicked.connect(lambda: self.clear())
        top.addWidget(self.combo, 1)
        top.addWidget(self.btn_clear)
        root.addLayout(top)
        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        self.inner = QWidget()
        self.vbox = QVBoxLayout(self.inner)
        self.vbox.setAlignment(Qt.AlignTop)
        self.vbox.setSpacing(4)
        self.area.setWidget(self.inner)
        root.addWidget(self.area, 1)
        self.set_kind("audio")

    def _fill_combo(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("+ Ajouter un traitement…", None)
        for i, spec in enumerate(dsp.processors_for(self.kind), start=1):
            self.combo.addItem(spec.label, spec.key)
            self.combo.setItemData(i, spec.help, Qt.ToolTipRole)
        self.combo.blockSignals(False)

    def set_kind(self, kind):
        self.kind = kind
        self._fill_combo()
        ok = {s.key for s in dsp.processors_for(kind)}
        removed = False
        for it in list(self.items):
            if it.spec.key not in ok:
                self._drop(it)
                removed = True
        if removed:
            self.changed.emit()

    def _on_combo(self, idx):
        key = self.combo.itemData(idx)
        self.combo.setCurrentIndex(0)
        if key:
            self.add(key)

    def add(self, key, values=None, bypass=False, emit=True):
        spec = dsp.REGISTRY[key]
        vals = dsp.default_params(key, self.kind)
        vals.update(values or {})
        item = ProcessorItem(spec, vals)
        item.bypass.setChecked(bypass)
        item.changed.connect(self.changed)
        item.move.connect(self._move)
        item.remove.connect(self._remove)
        self.items.append(item)
        self.vbox.addWidget(item)
        if emit:
            self.changed.emit()
        return item

    def _drop(self, item):
        self.items.remove(item)
        self.vbox.removeWidget(item)
        item.hide()
        item.deleteLater()

    def _remove(self, item):
        self._drop(item)
        self.changed.emit()

    def _move(self, item, d):
        i = self.items.index(item)
        j = i + d
        if 0 <= j < len(self.items):
            self.items[i], self.items[j] = self.items[j], self.items[i]
            for it in self.items:
                self.vbox.removeWidget(it)
            for it in self.items:
                self.vbox.addWidget(it)
            self.changed.emit()

    def clear(self, emit=True):
        for it in list(self.items):
            self._drop(it)
        if emit is not False:
            self.changed.emit()

    def chain(self):
        return [(it.spec.key, it.values()) for it in self.items if it.active()]

    def set_chain(self, chain):
        self.clear(emit=False)
        for key, vals in chain:
            self.add(key, vals, emit=False)
        self.changed.emit()
