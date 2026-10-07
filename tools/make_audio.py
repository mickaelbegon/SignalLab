"""Génère les 10 sons pédagogiques de SignalLab (synthèse reproductible, graine fixe).

Usage : python tools/make_audio.py
Écrit signallab/data/audio/*.wav et signallab/data/catalog_audio.json.
Synthèse additive (harmoniques + enveloppes propres à l'instrument), Karplus-Strong pour la guitare.
"""
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

FS = 44100
DUR = 2.0
N = int(FS * DUR)
T = np.arange(N) / FS
ROOT = Path(__file__).resolve().parents[1] / "signallab" / "data"
A4, C4 = 440.0, 261.63
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


def piano(f0):
    amps = [1.0, 0.7, 0.45, 0.35, 0.2, 0.18, 0.1, 0.08, 0.05, 0.04, 0.03, 0.02]
    y = additive(f0, amps, decays=lambda k: 1.2 + 0.9 * k, inharm=0.0004)
    y += 0.4 * additive(f0 * 1.0007, amps[:6], decays=lambda k: 0.5 + 0.5 * k)  # corde double désaccordée
    env = np.interp(T, [0, 0.004, 0.05, DUR - 0.1, DUR], [0, 1, 0.8, 0.45, 0])
    return y * env


def cello(f0):
    amps = [1.0 / k ** 0.9 for k in range(1, 25)]
    amps[1] *= 1.3
    amps[2] *= 1.2
    y = additive(f0, amps, vib=(5.2, 0.004))
    return (y + bandnoise(1500, 6000, 0.03)) * adsr(0.12, 0.1, 0.85, 0.25)


def violin(f0):
    amps = [1.0 / k ** 0.7 for k in range(1, 30)]
    for k in (3, 4, 5):
        amps[k - 1] *= 1.4
    y = additive(f0, amps, vib=(6.0, 0.005))
    return (y + bandnoise(2000, 8000, 0.02)) * adsr(0.08, 0.1, 0.9, 0.2)


def flute(f0):
    amps = [1.0, 0.35, 0.12, 0.05, 0.03, 0.015]
    y = additive(f0, amps, vib=(5.0, 0.003))
    return (y + bandnoise(2500, 7000, 0.06)) * adsr(0.09, 0.05, 0.9, 0.2)


def clarinet(f0):
    amps = [0.04] * 19
    for k, a in zip((1, 3, 5, 7, 9, 11, 13), (1.0, 0.75, 0.5, 0.3, 0.18, 0.1, 0.06)):
        amps[k - 1] = a
    y = additive(f0, amps, vib=(4.5, 0.001))
    return (y + bandnoise(2000, 6000, 0.01)) * adsr(0.05, 0.05, 0.9, 0.15)


def oboe(f0):
    amps = [0.6, 1.0, 0.9, 0.7, 0.5, 0.4, 0.25, 0.15, 0.1, 0.06, 0.04, 0.03]
    y = additive(f0, amps, vib=(5.5, 0.003))
    return (y + bandnoise(1500, 5000, 0.02)) * adsr(0.06, 0.08, 0.9, 0.18)


def trumpet(f0):
    amps = [0.5, 1.0, 0.9, 0.75, 0.6, 0.45, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03]
    y = np.zeros(N)
    for k, a in enumerate(amps, 1):  # les harmoniques aigus entrent progressivement
        y += additive(f0, [0] * (k - 1) + [a]) * np.clip(T / (0.02 + 0.012 * k), 0, 1)
    return y * adsr(0.05, 0.08, 0.9, 0.15)


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
     "Son sans harmoniques : une seule raie dans le spectre à 440 Hz. Référence pour comparer le timbre des autres instruments."),
    ("piano_A4", "Piano – La4 (440 Hz)", piano, A4, "A4",
     "Attaque brève puis décroissance ; harmoniques quasi entiers qui s'éteignent plus vite aux hautes fréquences."),
    ("cello_A4", "Violoncelle – La4 (440 Hz)", cello, A4, "A4",
     "Archet : beaucoup d'harmoniques (raies décroissantes) et vibrato léger ; son entretenu."),
    ("flute_A4", "Flûte – La4 (440 Hz)", flute, A4, "A4",
     "Fondamentale dominante, très peu d'harmoniques (2 et 3 faibles) et bruit de souffle à haute fréquence."),
    ("violin_A4", "Violon – La4 (440 Hz)", violin, A4, "A4",
     "Spectre très riche en harmoniques (proche d'une dent de scie) avec un renforcement vers 1,3–2,2 kHz ; vibrato."),
    ("square_A4", "Onde carrée – La4 (440 Hz)", square, A4, "A4",
     "Seuls les harmoniques impairs (440, 1320, 2200 Hz...) existent, d'amplitude 1/k : exemple de série de Fourier."),
    ("clarinet_C4", "Clarinette – Do4 (261,6 Hz)", clarinet, C4, "C4",
     "Tube cylindrique fermé : harmoniques impairs dominants (3f0, 5f0...), harmoniques pairs très faibles."),
    ("oboe_C4", "Hautbois – Do4 (261,6 Hz)", oboe, C4, "C4",
     "Anche double : timbre nasillard, harmoniques 2 à 5 plus forts que la fondamentale (formant vers 1 kHz)."),
    ("trumpet_C4", "Trompette – Do4 (261,6 Hz)", trumpet, C4, "C4",
     "Timbre cuivré : harmoniques forts jusqu'à ~3 kHz ; les aigus apparaissent progressivement pendant l'attaque."),
    ("guitar_C4", "Guitare (corde pincée) – Do4 (261,6 Hz)", guitar, C4, "C4",
     "Modèle Karplus-Strong : corde pincée, harmoniques aigus qui disparaissent plus vite que les graves (effet passe-bas)."),
]


def main():
    out = ROOT / "audio"
    out.mkdir(parents=True, exist_ok=True)
    catalog = []
    for sid, label, fn, f0, note, desc in SOUNDS:
        x = np.asarray(fn(f0), float)
        x -= x.mean()
        x = 0.89 * x / np.max(np.abs(x))
        wavfile.write(out / f"{sid}.wav", FS, np.round(x * 32767).astype(np.int16))
        catalog.append(dict(id=sid, label=label, kind="audio", fs=FS, f0=f0, note=note,
                            description=desc, source=SRC, audify_speed=1))
    (ROOT / "catalog_audio.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'id':14s}{'f0 decl.':>10s}{'f0 FFT':>10s}{'err %':>8s}")
    ok = True
    for c in catalog:
        fs, x = wavfile.read(out / f"{c['id']}.wav")
        x = x / 32768.0
        X = np.abs(np.fft.rfft(x * np.hanning(len(x)), 8 * len(x)))
        fr = np.fft.rfftfreq(8 * len(x), 1 / fs)
        thr = 0.3 * X.max()  # plus bas pic >= 30 % du max
        idx = next(i for i in range(1, len(X) - 1) if X[i] >= thr and X[i] >= X[i - 1] and X[i] >= X[i + 1])
        err = 100 * (fr[idx] - c["f0"]) / c["f0"]
        ok &= abs(err) <= 1
        print(f"{c['id']:14s}{c['f0']:10.2f}{fr[idx]:10.2f}{err:8.2f}")
    sz = sum(p.stat().st_size for p in out.glob("*.wav")) / 1e6
    print(f"Taille totale WAV : {sz:.2f} Mo ; f0 OK : {ok}")


if __name__ == "__main__":
    main()
