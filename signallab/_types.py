"""Types partagés (contrat SPEC.md)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SignalInfo:
    id: str              # ex "piano_A4"
    label: str           # ex "Piano – La4 (440 Hz)"
    kind: str            # "audio" | "emg" | "kin"
    fs: float            # Hz
    f0: float | None     # fréquence fondamentale attendue (Hz) ou None
    note: str | None     # "A4"
    description: str     # 1-2 phrases pour les étudiants
    source: str          # origine / licence
    audify_speed: float  # facteur d'accélération pour l'écoute (1 pour audio, >1 pour kin)
