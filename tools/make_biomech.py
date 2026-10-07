"""Régénère signallab/data/biomech/*.npz (graines fixes)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signallab.biomech import write_all  # noqa: E402

if __name__ == "__main__":
    for p in write_all():
        print(p, p.stat().st_size, "octets")
