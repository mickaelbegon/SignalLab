"""Génère les 3 sons SYNTHÉTISÉS de SignalLab (sinusoïde, onde carrée, guitare Karplus-Strong).

Les 7 autres sons sont de vrais enregistrements : voir tools/fetch_samples.py, qui appelle ce module
et écrit aussi catalog_audio.json.
Usage : python tools/make_audio.py   (régénère seulement les 3 WAV synthétiques)
"""
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

FS = 44100
DUR = 2.0
N = int(FS * DUR)
T = np.arange(N) / FS
ROOT = Path(__file__).resolve().parents[1] / "signallab" / "data"
A4 = 440.0
SRC = "Synthétisé (tools/make_audio.py, numpy) – CC0 / domaine public"


def rng():
    return np.random.default_rng(12345)


def adsr(a, d, s, r, dur=DUR):
    """Enveloppe attaque/déclin/maintien/relâchement (secondes, s = niveau)."""
    return np.interp(T, [0, a, a + d, dur - r, dur], [0, 1, s, s, 0])


def additive(f0, amps, vib=(0, 0), decays=None, inharm=0.0):
    """Somme de partiels k*f0 d'amplitudes amps[k-1]; vibrato (Hz, profondeur relative)."""
    vf, vd = vib
    f_inst = f0 * (1 + vd * np.sin(2 * np.pi * vf * T)) if vf else f0 * np.ones(N)
    phi = 2 * np.pi * np.cumsum(f_inst) / FS
    y = np.zeros(N)
    for k, a in enumerate(amps, start=1):
        if a == 0:
            continue
        fk = k * np.sqrt(1 + inharm * k * k)
        if fk * f0 >= FS / 2 * 0.98:
            break
        part = a * np.sin(fk * phi)
        if decays is not None:
            part = part * np.exp(-T * decays(k))
        y += part
    return y


def bandnoise(lo, hi, g):
    b, a = butter(2, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return lfilter(b, a, rng().standard_normal(N)) * g


def sine(f0):
    return np.sin(2 * np.pi * f0 * T) * adsr(0.02, 0.01, 1.0, 0.05)


def sine(f0):
    return np.sin(2 * np.pi * f0 * T) * adsr(0.02, 0.01, 1.0, 0.05)


def square(f0):
    amps = [1.0 / k if k % 2 else 0 for k in range(1, 60)]
    return additive(f0, amps) * adsr(0.01, 0.01, 1.0, 0.03)


def guitar(f0):
    """Karplus-Strong : période = L + 0.5 échantillon (filtre de moyenne)."""
    L = int(round(FS / f0 - 0.5))
    y = np.zeros(N)
    buf = rng().uniform(-1, 1, L)
    y[:L] = buf - buf.mean()
    for n in range(L, N):
        y[n] = 0.9993 * (0.5 * (y[n - L] + y[n - L - 1]) if n > L else y[n - L])
    return y * np.interp(T, [0, DUR - 0.15, DUR], [1, 1, 0])



SOUNDS = [
    ("sine_A4", "Sinusoïde pure – La4 (440 Hz)", sine, A4, "A4",
     "Signal synthétisé de référence : une seule raie dans le spectre à 440 Hz, aucun harmonique. Sert de comparaison pour le timbre des instruments réels."),
    ("square_A4", "Onde carrée – La4 (440 Hz)", square, A4, "A4",
     "Signal synthétisé : seuls les harmoniques impairs (440, 1320, 2200 Hz...) existent, d'amplitude 1/k : exemple de série de Fourier."),
    ("guitar_A4", "Guitare (corde pincée, synthèse) – La4 (440 Hz)", guitar, A4, "A4",
     "Synthèse Karplus-Strong (non enregistrée) : corde pincée, harmoniques aigus qui disparaissent plus vite que les graves (effet passe-bas)."),
]


def make_synth(out):
    """Écrit les WAV synthétiques dans out ; renvoie les entrées de catalogue."""
    out.mkdir(parents=True, exist_ok=True)
    cat = {}
    for sid, label, fn, f0, note, desc in SOUNDS:
        x = np.asarray(fn(f0), float)
        x -= x.mean()
        x = 0.89 * x / np.max(np.abs(x))
        wavfile.write(out / f"{sid}.wav", FS, np.round(x * 32767).astype(np.int16))
        cat[sid] = dict(id=sid, label=label, kind="audio", fs=FS, f0=f0, note=note, description=desc,
                        source=SRC, origin="synthétisé", audify_speed=1)
    return cat


if __name__ == "__main__":
    make_synth(ROOT / "audio")
    print("3 sons synthétiques écrits (catalogue : python tools/fetch_samples.py)")
