"""Tracés matplotlib (temps, fréquence, spectrogramme) indépendants de Qt."""
from __future__ import annotations

import numpy as np
from scipy.signal import spectrogram

from .. import dsp

C_ORIG = "#1f77b4"   # bleu
C_PROC = "#ff7f0e"   # orange
C_MARK = "#c0392b"
MAX_POINTS = 30000


def _unit(info):
    return {"audio": "amplitude (u.a.)", "emg": "EMG (u.a.)", "kin": "position (u.a.)"}.get(info.kind, "amplitude")


def _slice(x, fs, t0, dur):
    i0 = max(0, int(round(t0 * fs)))
    i1 = min(len(x), max(i0 + 2, int(round((t0 + dur) * fs))))
    t = np.arange(i0, i1) / fs
    seg = x[i0:i1]
    step = max(1, len(seg) // MAX_POINTS)
    return t[::step], seg[::step]


def plot_time(fig, info, x, branches, fs, t0, dur, show_orig=True):
    """branches : liste de (nom, couleur, signal traité)."""
    fig.clear()
    ax = fig.add_subplot(111)
    if not branches or show_orig:
        t, s = _slice(x, fs, t0, dur)
        ax.plot(t, s, color=C_ORIG, lw=1.4, label="Original")
    for name, color, y in branches:
        t, s = _slice(y, fs, t0, dur)
        ax.plot(t, s, color=color, lw=1.4, label=name)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel(_unit(info))
    ax.set_title(info.label, fontsize="medium")
    ax.grid(alpha=0.3)
    ax.margins(x=0)
    ax.legend(loc="upper right", fontsize="small", framealpha=0.85)
    return ax


def _spec(cache, name, x, fs):
    if name not in cache:
        cache[name] = dsp.spectrum(x, fs)
    return cache[name]


def plot_freq(fig, info, x, branches, fs, fmax, log_f=False, db=False, show_orig=True, cache=None, marker=-1):
    cache = {} if cache is None else cache
    fig.clear()
    ax = fig.add_subplot(111)
    fo, ao = _spec(cache, "orig", x, fs)
    ref = float(np.max(ao[1:])) if len(ao) > 1 else 1.0
    ref = ref if ref > 0 else 1.0

    def conv(a):
        return 20 * np.log10(np.maximum(a / ref, 1e-6)) if db else a

    fmax = min(fmax, fs / 2)
    curves = []
    if not branches or show_orig:
        ax.plot(fo, conv(ao), color=C_ORIG, lw=1.4, label="Original")
        curves.append((fo, ao))
    marked = None
    for k, (name, color, y) in enumerate(branches):
        fp, ap = _spec(cache, f"b{k}", y, fs)
        ax.plot(fp, conv(ap), color=color, lw=1.4, label=name)
        curves.append((fp, ap))
        if k == marker:
            marked = (fp, ap)
    # marqueurs sur la branche active, sinon la dernière courbe tracée
    fm, am = marked or curves[-1]
    try:
        if info.kind == "emg" and not info.f0:
            raise ValueError
        pk = dsp.find_peaks_harmonics(fm, am, f0=info.f0, n_harm=10)
        harm = [(f, a) for f, a in pk.get("harmonics", []) if 0 < f <= fmax]
        for n, (f, a) in enumerate(harm, start=1):
            v = float(conv(np.array([a]))[0])
            ax.plot([f], [v], "v", color=C_MARK, ms=9, zorder=5)
            if n == 1 and info.kind == "audio":
                lab = f"{f:.1f} Hz\n{dsp.note_name(f)}"
            elif n == 1:
                lab = f"{f:.1f} Hz"
            else:
                lab = f"×{n}" if abs(f / harm[0][0] - n) < 0.2 * n else f"{f:.0f}"
            ax.annotate(lab, (f, v), xytext=(0, 9), textcoords="offset points", ha="center",
                        va="bottom", fontsize="small", color=C_MARK, clip_on=True)
    except Exception:
        pass
    if log_f:
        ax.set_xscale("log")
        lo = max(float(fo[1]) if len(fo) > 1 else 1.0, 20.0 if info.kind == "audio" else 0.1)
        ax.set_xlim(lo, max(fmax, lo * 2))
    else:
        ax.set_xlim(0, fmax)
    vis = [a[(f <= fmax) & (f > 0)] for f, a in curves]
    vmax = max((float(np.max(v)) for v in vis if v.size), default=1.0)
    if db:
        top = float(20 * np.log10(max(vmax / ref, 1e-6))) + 12
        ax.set_ylim(top - 100, top)
        ax.set_ylabel("Amplitude (dB, réf. pic de l'original)")
    else:
        ax.set_ylim(0, vmax * 1.5)
        ax.set_ylabel("Amplitude (u.a.)")
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_title(info.label, fontsize="medium")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize="small", framealpha=0.85)
    return ax


def plot_spectrogram(fig, info, x, branches, fs, fmax, log_f=False):
    fig.clear()
    sigs = [("Original", x)] + [(name, y) for name, _c, y in branches]
    axes = fig.subplots(len(sigs), 1, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    nfft = int(2 ** np.round(np.log2(max(fs * 0.04, 64))))
    nfft = max(32, min(nfft, len(x)))
    ref = None
    mesh = None
    fmax = min(fmax, fs / 2)
    for ax, (name, s) in zip(axes, sigs):
        f, t, S = spectrogram(s, fs, nperseg=nfft, noverlap=nfft * 3 // 4, mode="magnitude")
        D = 20 * np.log10(np.maximum(S, 1e-9))
        if ref is None:
            ref = float(D.max())
        mesh = ax.pcolormesh(t, f, D, shading="auto", cmap="magma", vmin=ref - 80, vmax=ref + 3, rasterized=True)
        ax.set_ylim(0, fmax)
        ax.set_ylabel(f"{name}\nFréquence (Hz)")
        ax.set_title(info.label if name == "Original" else "", fontsize="medium")
    axes[-1].set_xlabel("Temps (s)")
    fig.colorbar(mesh, ax=list(axes), label="dB", pad=0.01)
    return axes[0]
