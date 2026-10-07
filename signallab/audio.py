"""Conversion en son jouable (PCM16 44,1 kHz mono) et lecture via QtMultimedia."""
from __future__ import annotations

import wave
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

PLAY_FS = 44100
FADE_MS = 10.0


def to_playable(x, fs, audify_speed=1.0):
    """Prépare un signal pour l'écoute.

    - accélère le signal de ``audify_speed`` (la fréquence d'échantillonnage effective devient fs*vitesse),
    - rééchantillonne à 44100 Hz (resample_poly), retire la moyenne, normalise,
    - applique un fondu d'entrée/sortie de 10 ms.

    Retourne (tableau np.int16, 44100).
    """
    x = np.nan_to_num(np.asarray(x, dtype=np.float64).ravel())
    if x.size == 0:
        return np.zeros(0, dtype=np.int16), PLAY_FS
    fs_eff = float(fs) * max(float(audify_speed or 1.0), 1e-6)
    ratio = Fraction(PLAY_FS / fs_eff).limit_denominator(2000)
    up, down = ratio.numerator, ratio.denominator
    if up != down:
        x = resample_poly(x, up, down)
    x = x - np.mean(x)
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak > 0:
        x = x / peak * 0.9
    n_fade = min(int(PLAY_FS * FADE_MS / 1000.0), x.size // 2)
    if n_fade > 0:
        ramp = np.linspace(0.0, 1.0, n_fade)
        x[:n_fade] *= ramp
        x[-n_fade:] *= ramp[::-1]
    return np.round(np.clip(x, -1, 1) * 32767).astype(np.int16), PLAY_FS


def write_wav(path, pcm, fs=PLAY_FS):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes(np.asarray(pcm, dtype="<i2").tobytes())


try:
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QObject, QTimer, Signal
    from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices
except Exception:  # pragma: no cover - PySide6 absent
    QObject = object


class Player(QObject):
    """Lecteur audio : un seul son à la fois. ``null=True`` simule la lecture (tests)."""

    stateChanged = Signal(bool)  # True = lecture en cours

    def __init__(self, parent=None, null=False):
        super().__init__(parent)
        self.null = null
        self._sink = None
        self._buf = None
        self._data = None
        self._playing = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.stop)

    @property
    def playing(self):
        return self._playing

    def play(self, pcm, fs=PLAY_FS) -> bool:
        self.stop()
        if len(pcm) == 0:
            return False
        duration_ms = int(1000 * len(pcm) / fs) + 250
        if not self.null:
            dev = QMediaDevices.defaultAudioOutput()
            if dev.isNull():
                return False
            fmt = QAudioFormat()
            fmt.setSampleRate(int(fs))
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            self._data = QByteArray(np.asarray(pcm, dtype="<i2").tobytes())
            self._buf = QBuffer(self._data, self)
            self._buf.open(QIODevice.OpenModeFlag.ReadOnly)
            self._sink = QAudioSink(dev, fmt, self)
            self._sink.start(self._buf)
        self._playing = True
        self._timer.start(duration_ms)
        self.stateChanged.emit(True)
        return True

    def stop(self):
        self._timer.stop()
        if self._sink is not None:
            self._sink.stop()
            self._sink.deleteLater()
            self._sink = None
        if self._buf is not None:
            self._buf.close()
            self._buf.deleteLater()
            self._buf = None
        if self._playing:
            self._playing = False
            self.stateChanged.emit(False)
