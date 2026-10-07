"""Signaux biomécaniques synthétiques (EMG de biceps, cinématique du coude / genou).

Tout est généré avec des graines fixes (reproductible) et stocké dans
data/biomech/*.npz (compressé). Unités physiques : EMG en mV, angles en degrés.
Si un .npz manque, le signal est régénéré à la volée.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal as sps

from ._types import SignalInfo

DATA_DIR = Path(__file__).parent / "data" / "biomech"
DURATION = 10.0
FS_EMG = 2000.0
FS_KIN = 100.0
N_REPS = 8
F_MOVE = N_REPS / DURATION  # 0.8 Hz : une flexion + une extension = 1.25 s
SOURCE = "Synthétique (signallab/biomech.py, graine fixe) – MIT"

# audify_speed cinématique : mouvement 0.8 Hz x 20 = 16 Hz (limite basse de
# l'audible) ; le bruit de mesure et les harmoniques montent vers ~100+ Hz.
# Genou à la marche (1 Hz) x 20 = 20 Hz. L'audio.py rééchantillonne en conséquence.
KIN_SPEED = 20.0


def _min_jerk_profile(t: np.ndarray, f: float) -> np.ndarray:
    """Position normalisée [0,1] : flexion puis extension, chacune en minimum-jerk."""
    ph = (t * f) % 1.0
    s = np.where(ph < 0.5, ph * 2.0, (1.0 - ph) * 2.0)  # triangle 0->1->0
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def elbow_angle_clean(fs: float = FS_KIN) -> np.ndarray:
    t = np.arange(int(DURATION * fs)) / fs
    return 15.0 + 125.0 * _min_jerk_profile(t, F_MOVE)  # 15° (étendu) -> 140° (fléchi)


def _bandlimited_noise(rng, n, fs, lo, hi, order=4):
    sos = sps.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    x = sps.sosfiltfilt(sos, rng.standard_normal(n + 2000))[1000:1000 + n]
    return x / x.std()


def emg_clean() -> np.ndarray:
    rng = np.random.default_rng(1234)
    fs = FS_EMG
    n = int(DURATION * fs)
    t = np.arange(n) / fs
    u = (elbow_angle_clean(FS_KIN) - 15.0) / 125.0
    u = np.interp(t, np.arange(len(u)) / FS_KIN, u)
    act = u**2                                   # biceps actif quand le coude est fléchi
    sos = sps.butter(2, 8.0, fs=fs, output="sos")
    env = sps.sosfiltfilt(sos, act)              # enveloppe lissée
    env = 0.08 + 0.92 * np.clip(env, 0, None)
    gain = 1.0 + 0.12 * _bandlimited_noise(rng, n, fs, 0.2, 1.0, 2)  # variabilité ~12 %
    env = env * gain
    carrier = _bandlimited_noise(rng, n, fs, 20.0, 450.0)
    return carrier * env * 0.33                  # pics ~ +-1 mV


def emg_raw() -> np.ndarray:
    rng = np.random.default_rng(4321)
    fs = FS_EMG
    n = int(DURATION * fs)
    t = np.arange(n) / fs
    x = emg_clean()
    base = 0.012 * rng.standard_normal(n)                        # bruit de base (~12 uV rms)
    motion = 0.12 * _bandlimited_noise(rng, n, fs, 0.3, 5.0, 2)  # artefact de mouvement < 5 Hz
    motion += 0.10 * np.sin(2 * np.pi * F_MOVE * t + 0.6)
    mains = 0.06 * np.sin(2 * np.pi * 50.0 * t + 0.3)            # ronflement secteur
    return x + base + motion + mains


def elbow_angle_measured() -> np.ndarray:
    rng = np.random.default_rng(99)
    a = elbow_angle_clean()
    return a + 0.4 * rng.standard_normal(a.size)


def knee_gait_measured() -> np.ndarray:
    """Angle de flexion du genou à la marche, 1 Hz (10 foulées)."""
    rng = np.random.default_rng(7)
    t = np.arange(int(DURATION * FS_KIN)) / FS_KIN
    w = 2 * np.pi * 1.0 * t
    a = 28 - 8 * np.cos(w + 0.3) + 22 * np.cos(2 * w - 2.6) - 6 * np.cos(3 * w - 0.5)
    a = a - a.min()
    return a + 0.4 * rng.standard_normal(a.size)


_GENERATORS = {
    "emg_biceps_raw": emg_raw,
    "emg_biceps_clean": emg_clean,
    "kin_elbow_angle": elbow_angle_measured,
    "kin_knee_gait": knee_gait_measured,
}

_INFOS = [
    SignalInfo(
        "emg_biceps_raw", "EMG biceps – brut (8 flexions du coude)", "emg", FS_EMG, None, None,
        "EMG de surface du biceps pendant 8 flexions/extensions du coude en 10 s, en mV. Il contient "
        "du bruit de base, un artefact de mouvement < 5 Hz et un ronflement secteur à 50 Hz : "
        "essayez un passe-haut ~20 Hz, un notch 50 Hz, puis une enveloppe.",
        SOURCE, 1.0),
    SignalInfo(
        "emg_biceps_clean", "EMG biceps – propre (8 flexions du coude)", "emg", FS_EMG, None, None,
        "Même EMG sans artefact ni 50 Hz (bande 20-450 Hz modulée par les bouffées d'activation). "
        "Sert de référence pour comparer avec la version brute après filtrage.",
        SOURCE, 1.0),
    SignalInfo(
        "kin_elbow_angle", "Angle du coude – 8 flexions (synchrone EMG)", "kin", FS_KIN, F_MOVE, None,
        "Angle du coude en degrés (15° à 140°), trajectoire minimum-jerk à 0,8 Hz, synchrone avec "
        "l'EMG, avec un petit bruit de mesure haute fréquence. Filtrez (passe-bas ~6 Hz) puis dérivez "
        "pour obtenir vitesse et accélération.",
        SOURCE, KIN_SPEED),
    SignalInfo(
        "kin_knee_gait", "Angle du genou – marche (1 Hz)", "kin", FS_KIN, 1.0, None,
        "Angle de flexion du genou en degrés pendant la marche (une foulée par seconde), avec bruit "
        "de mesure. Plusieurs harmoniques du pas sont présentes dans le spectre.",
        SOURCE, KIN_SPEED),
]


def biomech_catalog() -> list[SignalInfo]:
    return list(_INFOS)


def write_all(directory: Path = DATA_DIR) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    out = []
    for info in _INFOS:
        x = _GENERATORS[info.id]()
        p = directory / f"{info.id}.npz"
        np.savez_compressed(p, x=x.astype(np.float32), fs=info.fs)
        out.append(p)
    return out


def load_biomech(id: str) -> np.ndarray:
    """Signal en unités physiques (mV ou °), float64 1D."""
    if id not in _GENERATORS:
        raise KeyError(f"Signal biomécanique inconnu : {id}")
    p = DATA_DIR / f"{id}.npz"
    if p.exists():
        with np.load(p) as d:
            return d["x"].astype(np.float64)
    return _GENERATORS[id]().astype(np.float64)
