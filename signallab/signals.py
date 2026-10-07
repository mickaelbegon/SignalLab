"""Catalogue et chargement des signaux (audio ici ; EMG/cinématique via biomech.py si présent)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class SignalInfo:
    id: str
    label: str
    kind: str                 # "audio" | "emg" | "kin"
    fs: float
    f0: float | None
    note: str | None
    description: str
    source: str
    audify_speed: float = 1.0


def _audio_catalog() -> list[SignalInfo]:
    path = DATA_DIR / "catalog_audio.json"
    if not path.exists():
        return []
    items = json.loads(path.read_text(encoding="utf-8"))
    return [SignalInfo(**{k: d.get(k) for k in SignalInfo.__dataclass_fields__ if k in d}) for d in items]


def _biomech_module():
    try:
        from . import biomech
        if hasattr(biomech, "biomech_catalog") and hasattr(biomech, "load_biomech"):
            return biomech
    except ImportError:
        pass
    return None


def list_signals() -> list[SignalInfo]:
    sigs = _audio_catalog()
    bm = _biomech_module()
    if bm is not None:
        sigs = sigs + list(bm.biomech_catalog())
    return sigs


def load_signal(id: str) -> tuple[np.ndarray, SignalInfo]:
    """Retourne (signal float64 mono, SignalInfo). Audio normalisé |x| <= 1."""
    for info in _audio_catalog():
        if info.id == id:
            fs, x = wavfile.read(DATA_DIR / "audio" / f"{id}.wav")
            x = np.asarray(x, dtype=np.float64)
            if x.ndim > 1:
                x = x.mean(axis=1)
            x = x / 32768.0
            peak = np.max(np.abs(x))
            if peak > 0:
                x = x / peak
            info.fs = float(fs)
            return x, info
    bm = _biomech_module()
    if bm is not None:
        for info in bm.biomech_catalog():
            if info.id == id:
                return np.asarray(bm.load_biomech(id), dtype=np.float64), info
    raise KeyError(f"Signal inconnu : {id}")
