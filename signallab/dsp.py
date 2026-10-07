"""Traitement du signal pour SignalLab (numpy / scipy uniquement).

Tous les processeurs ont la signature ``func(x, fs, **params) -> ndarray`` et
renvoient un tableau float64 de meme longueur que ``x``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import signal

NYQ_FRACTION = 0.99          # les frequences sont bornees a ]0, 0.99 * Nyquist[
NOISE_SEED = 12345           # graine fixe du bruit blanc (reproductible)


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------
@dataclass
class ParamSpec:
    key: str
    label: str
    default: float
    min: float
    max: float
    step: float
    unit: str = ""
    log: bool = False


@dataclass
class ProcessorSpec:
    key: str
    label: str
    params: list = field(default_factory=list)
    applies_to: set = field(default_factory=set)
    help: str = ""
    func: Callable = None


# ---------------------------------------------------------------------------
# Utilitaires internes
# ---------------------------------------------------------------------------
def _as_float(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).copy()


def _clamp_freq(f, fs) -> float:
    """Borne f dans ]0, 0.99 * Nyquist[ sans jamais lever d'exception."""
    nyq = 0.5 * float(fs)
    try:
        f = float(f)
    except (TypeError, ValueError):
        f = 0.0
    if not np.isfinite(f):
        f = 0.0
    return float(min(max(f, 1e-3 * nyq), NYQ_FRACTION * nyq))


def _order(order) -> int:
    try:
        return int(min(max(round(float(order)), 1), 8))
    except (TypeError, ValueError):
        return 4


def _sosfiltfilt(sos, x: np.ndarray) -> np.ndarray:
    n = len(x)
    if n < 4:
        return x.copy()
    default_pad = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
    padlen = int(min(default_pad, n - 1))
    return signal.sosfiltfilt(sos, x, padlen=padlen)


def _butter(x, fs, btype, wn, order):
    x = _as_float(x)
    sos = signal.butter(_order(order), wn, btype=btype, fs=fs, output="sos")
    return _sosfiltfilt(sos, x)


def _win_samples(ms, fs, n_max) -> int:
    n = int(round(max(float(ms), 0.0) * 1e-3 * fs))
    n = max(n, 1)
    return min(n, max(n_max, 1))


def _moving_average(v: np.ndarray, n: int) -> np.ndarray:
    """Moyenne glissante centree (convolution 'same'), normalisee aux bords."""
    if n <= 1 or len(v) == 0:
        return v.copy()
    k = np.ones(n)
    num = signal.fftconvolve(v, k, mode="same")
    den = signal.fftconvolve(np.ones(len(v)), k, mode="same")
    return num / den


# ---------------------------------------------------------------------------
# Processeurs
# ---------------------------------------------------------------------------
def lowpass(x, fs, fc=1000.0, order=4):
    return _butter(x, fs, "lowpass", _clamp_freq(fc, fs), order)


def highpass(x, fs, fc=20.0, order=4):
    return _butter(x, fs, "highpass", _clamp_freq(fc, fs), order)


def bandpass(x, fs, f_low=300.0, f_high=3400.0, order=4):
    lo, hi = _clamp_freq(f_low, fs), _clamp_freq(f_high, fs)
    if lo > hi:
        lo, hi = hi, lo
    if hi <= lo * 1.0001:               # bande degeneree : on l'elargit un peu
        lo = max(lo / 1.05, 1e-3 * 0.5 * fs)
        hi = min(hi * 1.05, NYQ_FRACTION * 0.5 * fs)
    return _butter(x, fs, "bandpass", [lo, hi], order)


def notch(x, fs, f0=50.0, width=4.0, order=2):
    f0 = _clamp_freq(f0, fs)
    try:
        width = max(float(width), 1e-6)
    except (TypeError, ValueError):
        width = 4.0
    lo = max(f0 - width / 2, 1e-3 * 0.5 * fs)
    hi = min(f0 + width / 2, NYQ_FRACTION * 0.5 * fs)
    if hi <= lo * 1.0001:
        lo, hi = f0 / 1.02, min(f0 * 1.02, NYQ_FRACTION * 0.5 * fs)
    return _butter(x, fs, "bandstop", [lo, hi], order)


def _rms(x) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def add_sine_noise(x, fs, freq=50.0, amp=0.5):
    x = _as_float(x)
    if len(x) == 0:
        return x
    freq = _clamp_freq(freq, fs)
    ref = _rms(x)
    if ref <= 0:
        ref = 1.0
    t = np.arange(len(x)) / fs
    return x + float(amp) * ref * np.sin(2 * np.pi * freq * t)


def add_white_noise(x, fs, snr_db=20.0):
    x = _as_float(x)
    if len(x) == 0:
        return x
    p_sig = float(np.mean(x ** 2))
    if p_sig <= 0:
        return x
    p_noise = p_sig / (10.0 ** (float(snr_db) / 10.0))
    rng = np.random.default_rng(NOISE_SEED)
    return x + rng.standard_normal(len(x)) * np.sqrt(p_noise)


def rectify(x, fs):
    return np.abs(_as_float(x))


def moving_rms(x, fs, window_ms=50.0):
    x = _as_float(x)
    if len(x) == 0:
        return x
    n = _win_samples(window_ms, fs, len(x))
    return np.sqrt(np.clip(_moving_average(x ** 2, n), 0.0, None))


def envelope_lowpass(x, fs, fc=6.0, order=4):
    r = np.abs(_as_float(x))
    return np.clip(lowpass(r, fs, fc=fc, order=order), 0.0, None)


def envelope_movavg(x, fs, window_ms=100.0):
    r = np.abs(_as_float(x))
    if len(r) == 0:
        return r
    return _moving_average(r, _win_samples(window_ms, fs, len(r)))


def derivative(x, fs):
    x = _as_float(x)
    if len(x) < 2:
        return np.zeros_like(x)
    return np.gradient(x) * fs


def second_derivative(x, fs):
    return derivative(derivative(x, fs), fs)


def remove_mean(x, fs):
    x = _as_float(x)
    return x - np.mean(x) if len(x) else x


def downsample(x, fs, factor=4):
    x = _as_float(x)
    n = len(x)
    if n == 0:
        return x
    factor = int(min(max(round(float(factor)), 1), 256))
    y = x[::factor]                      # decimation SANS anti-repliement
    return np.repeat(y, factor)[:n]      # maintien a la fs d'origine


def bit_crush(x, fs, bits=6):
    x = _as_float(x)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak <= 0:
        return x
    bits = int(min(max(round(float(bits)), 1), 24))
    levels = 2.0 ** (bits - 1)
    return np.round(x / peak * levels) / levels * peak


def clip(x, fs, threshold_pct=50.0):
    x = _as_float(x)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak <= 0:
        return x
    thr = peak * min(max(float(threshold_pct), 1.0), 100.0) / 100.0
    return np.clip(x, -thr, thr)


def gain(x, fs, gain_db=6.0):
    return _as_float(x) * 10.0 ** (float(gain_db) / 20.0)


def echo(x, fs, delay_ms=200.0, mix=0.5, repeats=3):
    x = _as_float(x)
    n = len(x)
    y = x.copy()
    if n == 0:
        return y
    d = int(round(max(float(delay_ms), 1.0) * 1e-3 * fs))
    mix = min(max(float(mix), 0.0), 1.0)
    for k in range(1, int(min(max(round(float(repeats)), 1), 20)) + 1):
        s = k * d
        if s >= n:
            break
        y[s:] += (mix ** k) * x[: n - s]
    return y


# ---------------------------------------------------------------------------
# Registre
# ---------------------------------------------------------------------------
_ALL = {"audio", "emg", "kin"}
_ORDER = ParamSpec("order", "Ordre du filtre", 4, 1, 8, 1, "")


def _P(*a, **k):
    return ParamSpec(*a, **k)


_SPECS = [
    ProcessorSpec(
        "lowpass", "Passe-bas", [_P("fc", "Fréquence de coupure", 1000, 0.1, 20000, 1, "Hz", True), _ORDER],
        _ALL,
        "Laisse passer les basses fréquences et atténue tout ce qui est au-dessus de la coupure. "
        "Filtre de Butterworth appliqué dans les deux sens (sans retard).", lowpass),
    ProcessorSpec(
        "highpass", "Passe-haut", [_P("fc", "Fréquence de coupure", 20, 0.1, 20000, 1, "Hz", True), _ORDER],
        _ALL,
        "Laisse passer les hautes fréquences et supprime les basses (dérive, composante continue, "
        "artefacts de mouvement).", highpass),
    ProcessorSpec(
        "bandpass", "Passe-bande",
        [_P("f_low", "Coupure basse", 300, 0.1, 20000, 1, "Hz", True),
         _P("f_high", "Coupure haute", 3400, 0.1, 20000, 1, "Hz", True), _ORDER],
        _ALL,
        "Ne garde que les fréquences comprises entre les deux coupures (ex. 300–3400 Hz : voix au téléphone).",
        bandpass),
    ProcessorSpec(
        "notch", "Coupe-bande (notch)",
        [_P("f0", "Fréquence à retirer", 50, 0.1, 20000, 0.5, "Hz", True),
         _P("width", "Largeur de bande", 4, 0.1, 500, 0.1, "Hz", True),
         _P("order", "Ordre du filtre", 2, 1, 8, 1, "")],
        _ALL,
        "Supprime une bande étroite autour d'une fréquence, typiquement le 50/60 Hz du secteur "
        "qui contamine l'EMG.", notch),
    ProcessorSpec(
        "add_sine_noise", "Ajouter un bruit sinusoïdal",
        [_P("freq", "Fréquence", 50, 0.1, 20000, 0.5, "Hz", True),
         _P("amp", "Amplitude relative au RMS", 0.5, 0.0, 10.0, 0.05, "×RMS")],
        _ALL,
        "Ajoute une sinusoïde parasite (ex. secteur 50 Hz). Son amplitude est relative à l'amplitude RMS du signal.",
        add_sine_noise),
    ProcessorSpec(
        "add_white_noise", "Ajouter un bruit blanc",
        [_P("snr_db", "Rapport signal/bruit", 20, -10, 60, 1, "dB")],
        _ALL,
        "Ajoute un bruit aléatoire de toutes les fréquences. Plus le rapport signal/bruit (SNR) est faible, "
        "plus le bruit est fort. Le bruit est toujours le même (reproductible).", add_white_noise),
    ProcessorSpec(
        "rectify", "Redressement", [], {"emg", "kin"},
        "Prend la valeur absolue du signal : toutes les valeurs deviennent positives. "
        "Première étape pour obtenir l'enveloppe d'un EMG.", rectify),
    ProcessorSpec(
        "moving_rms", "RMS glissant", [_P("window_ms", "Fenêtre", 50, 1, 2000, 1, "ms", True)], {"emg", "kin"},
        "Valeur efficace (RMS) calculée sur une fenêtre glissante centrée : "
        "mesure de l'amplitude locale du signal.", moving_rms),
    ProcessorSpec(
        "envelope_lowpass", "Enveloppe (redressement + passe-bas)",
        [_P("fc", "Fréquence de coupure", 6, 0.5, 100, 0.5, "Hz", True), _ORDER], {"emg", "kin"},
        "Redresse le signal puis le filtre par un passe-bas : donne l'enveloppe lisse de l'activité musculaire.",
        envelope_lowpass),
    ProcessorSpec(
        "envelope_movavg", "Enveloppe (redressement + moyenne glissante)",
        [_P("window_ms", "Fenêtre", 100, 1, 2000, 1, "ms", True)], {"emg", "kin"},
        "Redresse le signal puis calcule une moyenne glissante centrée. "
        "Plus la fenêtre est longue, plus l'enveloppe est lisse.", envelope_movavg),
    ProcessorSpec(
        "derivative", "Dérivée (vitesse)", [], {"kin"},
        "Dérive le signal par rapport au temps : la position devient une vitesse. Amplifie le bruit haute fréquence.",
        derivative),
    ProcessorSpec(
        "second_derivative", "Dérivée seconde (accélération)", [], {"kin"},
        "Dérive deux fois : la position devient une accélération. Très sensible au bruit, filtrer avant !",
        second_derivative),
    ProcessorSpec(
        "remove_mean", "Retirer la moyenne", [], _ALL,
        "Soustrait la valeur moyenne du signal (supprime la composante continue ou le décalage).", remove_mean),
    ProcessorSpec(
        "downsample", "Sous-échantillonnage (aliasing)",
        [_P("factor", "Facteur de décimation", 4, 1, 64, 1, "×")], {"audio"},
        "Garde un échantillon sur N sans filtre anti-repliement. Quand la nouvelle fréquence "
        "d'échantillonnage passe sous 2× la fréquence du signal, il y a repliement (aliasing) : "
        "de fausses fréquences apparaissent.", downsample),
    ProcessorSpec(
        "bit_crush", "Quantification (bits)", [_P("bits", "Nombre de bits", 6, 1, 16, 1, "bits")], {"audio"},
        "Réduit le nombre de niveaux d'amplitude possibles. Peu de bits = bruit de quantification audible.",
        bit_crush),
    ProcessorSpec(
        "clip", "Saturation (écrêtage)", [_P("threshold_pct", "Seuil (% du pic)", 50, 1, 100, 1, "%")], {"audio"},
        "Coupe les amplitudes au-delà du seuil : crée des harmoniques supplémentaires (distorsion).", clip),
    ProcessorSpec(
        "gain", "Gain", [_P("gain_db", "Gain", 6, -60, 40, 0.5, "dB")], {"audio"},
        "Amplifie (+) ou atténue (−) le signal en décibels. +6 dB ≈ amplitude ×2.", gain),
    ProcessorSpec(
        "echo", "Écho",
        [_P("delay_ms", "Retard", 200, 1, 2000, 1, "ms", True),
         _P("mix", "Niveau de l'écho", 0.5, 0.0, 1.0, 0.05, ""),
         _P("repeats", "Répétitions", 3, 1, 10, 1, "")], {"audio"},
        "Ajoute des copies retardées et atténuées du signal.", echo),
]

REGISTRY: dict = {s.key: s for s in _SPECS}


def processors_for(kind: str) -> list:
    return [s for s in REGISTRY.values() if kind in s.applies_to]


def default_params(key: str, kind: str = "audio") -> dict:
    """Paramètres par défaut adaptés au type de signal ('audio', 'emg', 'kin')."""
    spec = REGISTRY[key]
    p = {ps.key: ps.default for ps in spec.params}
    if key == "lowpass":
        p["fc"] = {"audio": 1000, "emg": 450, "kin": 20}.get(kind, 1000)
    elif key == "highpass":
        p["fc"] = {"audio": 100, "emg": 20, "kin": 0.5}.get(kind, 20)
    elif key == "bandpass":
        p["f_low"], p["f_high"] = {"audio": (300, 3400), "emg": (20, 450), "kin": (0.5, 20)}.get(kind, (300, 3400))
    elif key == "notch":
        p["f0"] = {"audio": 440, "emg": 50, "kin": 50}.get(kind, 50)
        p["width"] = {"audio": 20, "emg": 4, "kin": 2}.get(kind, 4)
    elif key == "add_sine_noise":
        p["freq"] = {"audio": 1000, "emg": 50, "kin": 8}.get(kind, 50)
        p["amp"] = 0.5
    elif key == "moving_rms":
        p["window_ms"] = {"emg": 50, "kin": 200, "audio": 20}.get(kind, 50)
    elif key == "envelope_lowpass":
        p["fc"] = {"emg": 6, "kin": 2}.get(kind, 6)
    elif key == "envelope_movavg":
        p["window_ms"] = {"emg": 100, "kin": 200}.get(kind, 100)
    return p


def apply_chain(x, fs, chain) -> np.ndarray:
    """Applique successivement [(key, params), ...]. Clés inconnues ignorées."""
    y = _as_float(x)
    for item in chain or []:
        key, params = (item[0], item[1] if len(item) > 1 else {})
        spec = REGISTRY.get(key)
        if spec is None:
            continue
        allowed = {ps.key for ps in spec.params}
        kw = {k: v for k, v in (params or {}).items() if k in allowed}
        y = np.asarray(spec.func(y, fs, **kw), dtype=np.float64)
    return y


# ---------------------------------------------------------------------------
# Analyse spectrale
# ---------------------------------------------------------------------------
def spectrum(x, fs, window="hann", n_fft=None):
    """(freqs Hz, amplitude linéaire) ; un sinus d'amplitude A donne un pic ≈ A."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    if window in (None, "rect", "rectangular", "boxcar"):
        w = np.ones(n)
    else:
        w = signal.get_window(window, n)
    nfft = int(n_fft) if n_fft else n
    nfft = max(nfft, n)
    X = np.fft.rfft(x * w, nfft)
    amp = np.abs(X) / np.sum(w)
    amp[1:] *= 2.0
    if nfft % 2 == 0:
        amp[-1] /= 2.0                  # Nyquist : pas de doublement
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return freqs, amp


def _refine_peak(freqs, amp, i):
    """Interpolation parabolique (sur log-amplitude) autour de l'indice i."""
    if i <= 0 or i >= len(amp) - 1:
        return float(freqs[i]), float(amp[i])
    a, b, c = (np.log(max(amp[j], 1e-300)) for j in (i - 1, i, i + 1))
    den = a - 2 * b + c
    if den >= 0:
        return float(freqs[i]), float(amp[i])
    p = 0.5 * (a - c) / den
    p = min(max(p, -0.5), 0.5)
    df = freqs[1] - freqs[0]
    return float(freqs[i] + p * df), float(np.exp(b - 0.25 * (a - c) * p))


def _local_peak(freqs, amp, f_lo, f_hi):
    """Indice du maximum local le plus fort dans [f_lo, f_hi], ou None."""
    idx = np.where((freqs >= f_lo) & (freqs <= f_hi))[0]
    if len(idx) == 0:
        return None
    i = int(idx[np.argmax(amp[idx])])
    if i == 0 or i >= len(amp) - 1:
        return None
    if amp[i] < amp[i - 1] or amp[i] < amp[i + 1] or amp[i] <= 0:
        return None
    return i


def find_peaks_harmonics(freqs, amp, f0=None, n_harm=10) -> dict:
    freqs = np.asarray(freqs, dtype=np.float64)
    amp = np.asarray(amp, dtype=np.float64)
    out = {"f0": None, "harmonics": []}
    if len(freqs) < 4 or len(amp) != len(freqs) or not np.any(amp > 0):
        return out
    df = freqs[1] - freqs[0]
    fmax = freqs[-1]

    if f0 is not None and f0 > 0:
        i0 = _local_peak(freqs, amp, 0.95 * f0, 1.05 * f0)
        if i0 is None:
            return out
    else:
        fmin = max(3 * df, 1.0)
        cand = np.where(freqs >= fmin)[0]
        if len(cand) < 3:
            return out
        sub = amp[cand]
        i0 = None
        order = np.argsort(sub)[::-1]
        for j in order[:50]:                 # plus fort maximum local
            i = int(cand[j])
            if 0 < i < len(amp) - 1 and amp[i] >= amp[i - 1] and amp[i] >= amp[i + 1]:
                i0 = i
                break
        if i0 is None:
            return out
        # garde anti-harmonique 2 : si un pic notable existe à f/2, c'est la fondamentale
        for _ in range(3):
            f = freqs[i0]
            if f / 2 < fmin:
                break
            j = _local_peak(freqs, amp, 0.97 * f / 2, 1.03 * f / 2)
            if j is not None and amp[j] >= 0.2 * amp[i0] and freqs[j] >= fmin:
                i0 = j
            else:
                break

    f_est, a_est = _refine_peak(freqs, amp, i0)
    out["f0"] = f_est
    floor = 1e-3 * a_est
    harms = [(f_est, a_est)]
    for k in range(2, int(n_harm) + 1):
        fk = k * f_est
        if fk >= fmax:
            break
        tol = max(0.03 * fk, 2 * df)
        i = _local_peak(freqs, amp, fk - tol, fk + tol)
        if i is None or amp[i] < floor:
            continue
        harms.append(_refine_peak(freqs, amp, i))
    out["harmonics"] = harms
    return out


_NOTES = ["Do", "Do#", "Ré", "Ré#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]


def note_name(f) -> str:
    """Nom de note français avec écart en cents, ex. 'La4 (+3 cents)'."""
    try:
        f = float(f)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(f) or f <= 0:
        return "—"
    midi = 69.0 + 12.0 * np.log2(f / 440.0)
    n = int(np.round(midi))
    cents = int(np.round((midi - n) * 100.0))
    return f"{_NOTES[n % 12]}{n // 12 - 1} ({cents:+d} cents)"
