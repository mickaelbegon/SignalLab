"""Télécharge de VRAIS enregistrements (University of Iowa Musical Instrument Samples), découpe une note
La4 de ~2 s par instrument et reconstruit signallab/data/audio/*.wav + catalog_audio.json.

Usage : python tools/fetch_samples.py           (~35 Mo téléchargés dans un dossier temporaire, supprimés ensuite)

Source : https://theremin.music.uiowa.edu/MIS.html  (Lawrence Fritts, Univ. of Iowa Electronic Music Studios)
Conditions : "freely available ... may be downloaded and used for any projects, without restrictions".
Fichiers pre-2012 : 16 bits / 44,1 kHz mono, gammes chromatiques jouées note par note, nuance mf
(le piano est un fichier par note, stéréo -> moyenné en mono). Les fichiers bruts ne sont PAS versionnés.

Découpe : on détecte les attaques (enveloppe RMS 10 ms > 4 % du max, silence >= 0,3 s avant), on estime
f0 de chaque note par autocorrélation (1 s après l'attaque) et on retient la note la plus proche de 440 Hz.
Segment = de 5 ms avant l'attaque à 2,0 s après, retrait de la composante continue, mono, 44,1 kHz,
normalisation à 0,89 en crête, fondu de fin en cosinus de 100 ms. f0 est ensuite MESURÉ par FFT du WAV final
(pic FFT autour de l'estimation par autocorrélation, interpolation parabolique) : c'est cette valeur qui est écrite au catalogue.
Les 3 signaux synthétiques (sinusoïde, carrée, guitare Karplus-Strong) viennent de tools/make_audio.py.
"""
import json
import shutil
import struct
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_audio as ma  # noqa: E402

FS = 44100
DUR = 2.0
BASE = "https://theremin.music.uiowa.edu/sound files/MIS/"
PAGE = "https://theremin.music.uiowa.edu/MIS.html"
LICENCE = "University of Iowa Musical Instrument Samples – usage libre sans restriction (« freely available … used for any projects, without restrictions »)"

# (id, instrument FR, chemin relatif, nuance/jeu, description FR)
SAMPLES = [
    ("piano_A4", "Piano", "Piano_Other/piano/Piano.mf.A4.aiff", "mf",
     "Attaque brève (percussion de la corde) puis décroissance lente ; harmoniques quasi entiers (légère inharmonicité) qui s'éteignent plus vite aux hautes fréquences."),
    ("cello_A4", "Violoncelle", "Strings/cello/Cello.arco.mf.sulA.C4C5.aiff", "arco, corde de La, mf",
     "Archet : de nombreux harmoniques (raies décroissantes) et un son entretenu ; l'attaque est plus lente que celle du piano."),
    ("flute_A4", "Flûte", "Woodwinds/flute/Flute.nonvib.mf.B3B4.aiff", "sans vibrato, mf",
     "Fondamentale dominante et peu d'harmoniques (2 et 3 plus faibles), avec un bruit de souffle à haute fréquence : timbre presque sinusoïdal."),
    ("violin_A4", "Violon", "Strings/violin/Violin.arco.mf.sulA.A4B4.aiff", "arco, corde de La, mf",
     "Spectre riche en harmoniques (proche d'une dent de scie) avec des renforcements dus à la caisse de résonance."),
    ("clarinet_A4", "Clarinette (Si bémol)", "Woodwinds/Bbclarinet/BbClar.mf.C4B4.aiff", "mf",
     "Tube cylindrique fermé à un bout : harmoniques impairs (3f0, 5f0...) dominants, harmoniques pairs plus faibles."),
    ("oboe_A4", "Hautbois", "Woodwinds/oboe/Oboe.mf.C4B4.aiff", "mf",
     "Anche double : timbre nasillard, harmoniques de rang 2 à 5 très présents (formant vers 1 kHz), parfois plus forts que la fondamentale."),
    ("trumpet_A4", "Trompette (Si bémol)", "Brass/Bbtrumpet/Trumpet.novib.mf.C4B4.aiff", "sans vibrato, mf",
     "Timbre cuivré : harmoniques forts jusqu'à ~3 kHz ; les aigus apparaissent progressivement pendant l'attaque. Accordage pouvant s'écarter de 440 Hz de quelques dixièmes de %."),
]
LABELS = {"piano_A4": "Piano", "cello_A4": "Violoncelle", "flute_A4": "Flûte", "violin_A4": "Violon",
          "clarinet_A4": "Clarinette", "oboe_A4": "Hautbois", "trumpet_A4": "Trompette"}
ORDER = ["sine_A4", "piano_A4", "cello_A4", "flute_A4", "violin_A4", "square_A4",
         "clarinet_A4", "oboe_A4", "trumpet_A4", "guitar_A4"]


def read_aiff(path):
    """Lecteur AIFF PCM 16 bits (module aifc supprimé en Python 3.13) -> (fs, mono float)."""
    d = Path(path).read_bytes()
    assert d[:4] == b"FORM" and d[8:12] == b"AIFF", "pas un AIFF"
    p, ch, bits, fs, pcm = 12, None, None, None, None
    while p + 8 <= len(d):
        cid, sz = d[p:p + 4], struct.unpack(">I", d[p + 4:p + 8])[0]
        b = d[p + 8:p + 8 + sz]
        if cid == b"COMM":
            ch, _, bits = struct.unpack(">hIh", b[:8])
            e = struct.unpack(">H", b[8:10])[0]
            m = struct.unpack(">Q", b[10:18])[0]
            fs = (m / 2 ** 63) * 2 ** ((e & 0x7FFF) - 16383)
        elif cid == b"SSND":
            pcm = b[8:]
        p += 8 + sz + (sz & 1)
    assert bits == 16 and pcm is not None, "AIFF 16 bits attendu"
    x = np.frombuffer(pcm, ">i2").astype(float) / 32768
    return int(round(fs)), x.reshape(-1, ch).mean(1)


def onsets(x, fs, rel=0.04, gap=0.3):
    h = int(0.01 * fs)
    n = len(x) // h
    e = np.sqrt((x[:n * h].reshape(n, h) ** 2).mean(1))
    on = e > rel * e.max()
    k = int(gap / 0.01)
    return [i * 0.01 for i in range(n) if on[i] and not on[max(0, i - k):i].any()]


def f0_autocorr(seg, fs):
    seg = seg - seg.mean()
    n = len(seg)
    r = np.fft.irfft(np.abs(np.fft.rfft(seg, 2 * n)) ** 2)[:n]
    r /= r[0]
    lo, hi = int(fs / 1000), int(fs / 100)
    m = lo + np.argmax(r[lo:hi])
    j = next(j for j in range(lo + 1, hi) if r[j] >= r[j - 1] and r[j] >= r[j + 1] and r[j] >= 0.9 * r[m])
    a, b, c = r[j - 1], r[j], r[j + 1]
    return fs / (j + 0.5 * (a - c) / (a - 2 * b + c))


def cut_note(x, fs, target=440.0):
    """Segment de 2 s de la note la plus proche de `target` dans un fichier (gamme ou note unique)."""
    best = min(onsets(x, fs), key=lambda t: abs(np.log2(
        f0_autocorr(x[int((t + 0.3) * fs):int((t + 1.3) * fs)], fs) / target)))
    a = max(0, int((best - 0.005) * fs))
    return x[a:a + int(DUR * fs)], best


def measure_f0(x, fs):
    """f0 par FFT (Hann, zero-padding x8, interpolation parabolique) : pic le plus fort dans +-3 % de
    l'estimation par autocorrélation (évite de prendre un harmonique quand la fondamentale est faible)."""
    fa = f0_autocorr(x, fs)
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)), 8 * len(x)))
    fr = np.fft.rfftfreq(8 * len(x), 1 / fs)
    m = np.where((fr > 0.97 * fa) & (fr < 1.03 * fa))[0]
    i = m[np.argmax(X[m])]
    a, b, c = X[i - 1], X[i], X[i + 1]
    return float(fr[i] + 0.5 * (a - c) / (a - 2 * b + c) * (fr[1] - fr[0]))


def cents(f, ref=440.0):
    return 1200 * np.log2(f / ref)


def main():
    root = Path(ma.ROOT)
    out = root / "audio"
    catalog = ma.make_synth(out)
    tmp = Path(tempfile.mkdtemp(prefix="iowa_mis_"))
    try:
        for sid, name, rel, play, desc in SAMPLES:
            url = BASE + rel
            path = tmp / Path(rel).name
            print("téléchargement", urllib.parse.quote(url, safe=":/"))
            urllib.request.urlretrieve(urllib.parse.quote(url, safe=":/"), path)
            fs, x = read_aiff(path)
            assert fs == FS, f"{path.name}: fs={fs}"
            seg, t0 = cut_note(x, fs)
            seg = seg - seg.mean()
            seg = 0.89 * seg / np.max(np.abs(seg))
            nf = int(0.1 * fs)
            seg[-nf:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, nf)))
            wavfile.write(out / f"{sid}.wav", FS, np.round(seg * 32767).astype(np.int16))
            f0 = measure_f0(seg, FS)
            cc = cents(f0)
            tag = f"{cc:+.0f} cents par rapport à 440 Hz"
            catalog[sid] = dict(
                id=sid, label=f"{name} – La4 (≈{f0:.0f} Hz)", kind="audio", fs=FS, f0=round(f0, 1), note="A4",
                description=f"Enregistrement réel ({play}) ; f0 mesurée {f0:.1f} Hz ({tag}). {desc}",
                source=f"{LICENCE}. Fichier : {urllib.parse.quote(url, safe=':/')} (note découpée à t = {t0:.2f} s, "
                       f"2 s, mono, normalisée). Page : {PAGE}",
                origin="réel (enregistrement)", audify_speed=1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for i in ("sine_A4", "square_A4", "guitar_A4"):  # f0 mesurée aussi pour les sons synthétiques
        fs, x = wavfile.read(out / f"{i}.wav")
        catalog[i]["f0"] = round(measure_f0(x / 32768.0, fs), 1)
    (root / "catalog_audio.json").write_text(
        json.dumps([catalog[i] for i in ORDER], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'id':14s}{'origine':24s}{'f0 FFT':>9s}{'cents':>7s}")
    for i in ORDER:
        fs, x = wavfile.read(out / f"{i}.wav")
        f = measure_f0(x / 32768.0, fs)
        print(f"{i:14s}{catalog[i]['origin']:24s}{f:9.2f}{cents(f):7.1f}")
    print(f"Taille totale WAV : {sum(p.stat().st_size for p in out.glob('*.wav')) / 1e6:.2f} Mo")


if __name__ == "__main__":
    main()
