"""Test smoke de la GUI en mode offscreen (aucune lecture audio réelle)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.path.isdir(r"C:\Windows\Fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from signallab import audio, signals
from signallab.gui import main_window as mw


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(mw.STYLE)
    return a


@pytest.fixture()
def win(app):
    w = mw.MainWindow(player=audio.Player(null=True))
    w.show()
    yield w
    w.close()


def test_to_playable():
    fs = 8000
    x = np.sin(2 * np.pi * 440 * np.arange(fs) / fs)
    pcm, out_fs = audio.to_playable(x, fs)
    assert pcm.dtype == np.int16 and out_fs == 44100
    assert abs(len(pcm) - 44100) <= 2
    assert abs(int(pcm[0])) < 100 and abs(int(pcm[-1])) < 100        # fondu
    assert np.abs(pcm).max() > 20000                                   # normalisé
    pcm2, _ = audio.to_playable(np.random.randn(2000), 2000)           # EMG
    assert len(pcm2) == 44100
    pcm3, _ = audio.to_playable(np.random.randn(1000), 100, 50)        # cin accélérée x50
    assert abs(len(pcm3) - 8820) <= 3
    assert len(audio.to_playable(np.zeros(0), 100)[0]) == 0
    assert np.all(audio.to_playable(np.zeros(500), 1000)[0] == 0)


def test_window_flow(win):
    import tempfile
    tmp_path = Path(tempfile.mkdtemp(prefix='signallab_'))
    infos = signals.list_signals()
    audio_id = next(i.id for i in infos if i.kind == "audio")
    other = next((i.id for i in infos if i.kind != "audio"), audio_id)
    win.left.set_signal(audio_id)
    win.right.set_signal(other)
    win.left.chain.add("lowpass")
    win.left.recompute()
    assert win.left.y is not None and len(win.left.y) == len(win.left.x)
    for mode in ("freq", "spec", "time"):
        win.set_mode(mode)
    win.cb_db.setCurrentIndex(1)
    win.cb_logf.setCurrentIndex(1)
    win.set_mode("freq")
    win.ck_link.setChecked(True)
    win.ck_overlay.setChecked(False)
    win.toggle_mode()
    for i in range(len(mw.PRESETS)):
        win.apply_preset(i % 2, i)
        assert win.panels[i % 2].y is not None
    # chaîne : bypass / monter / supprimer
    ch = win.left.chain
    ch.add("highpass")
    ch.items[0].bypass.setChecked(True)
    assert all(k != ch.items[0].spec.key for k, _ in ch.chain())
    ch._move(ch.items[1], -1)
    ch._remove(ch.items[0])
    ch.clear()
    win.left.recompute()
    assert win.left.y is None
    # lecture simulée
    win.left.play(False)
    assert win.player.playing
    win.right.play(True)
    win.player.stop()
    assert not win.player.playing
    # exports
    win.left.export_wav(tmp_path / "a.wav")
    win.left.export_csv(tmp_path / "a.csv")
    win.left.export_png(tmp_path / "a.png")
    assert all((tmp_path / n).stat().st_size > 0 for n in ("a.wav", "a.csv", "a.png"))
    # mode de tous les types de signaux
    for i in infos:
        win.right.set_signal(i.id)


def test_screenshot(win):
    infos = signals.list_signals()
    ids = {i.id for i in infos}
    win.left.set_signal("piano_A4" if "piano_A4" in ids else infos[0].id)
    emg = next((i.id for i in infos if i.kind == "emg"), None)
    if emg:
        win.right.set_signal(emg)
    win.left.chain.add("lowpass")
    win.left.recompute()
    win.right.chain.add("notch")
    win.right.recompute()
    win.set_mode("freq")
    out = Path(__file__).resolve().parents[1] / "docs"
    out.mkdir(exist_ok=True)
    win.resize(1700, 1000)
    for _ in range(3):
        QApplication.processEvents()
    win.grab().save(str(out / "screenshot.png"))
