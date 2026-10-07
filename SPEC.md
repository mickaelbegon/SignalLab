# SignalLab – contrat d'architecture (partagé entre les agents)

But : GUI pédagogique (PySide6 + matplotlib + numpy/scipy, Python 3.11, Windows) pour que des étudiants
"voient et entendent" l'effet du traitement du signal sur des sons d'instruments, un EMG et un signal cinématique.
Interface en **français**. Pas de dépendance autre que : numpy, scipy, matplotlib, PySide6 (lecture audio via QtMultimedia).

## Arborescence
```
SignalLab/
  run.py                      # python run.py -> lance la GUI
  requirements.txt, README.md, .gitignore, LICENSE (MIT)
  signallab/
    __init__.py
    signals.py                # catalogue + chargement   (agent SONS + BIOMECA)
    biomech.py                # génération EMG / cinématique (agent BIOMECA)
    dsp.py                    # filtres, bruits, spectre   (agent DSP)
    audio.py                  # to_playable(), lecture Qt  (agent GUI)
    gui/                      # fenêtre, panneaux, canvas  (agent GUI)
    data/audio/*.wav          # 10 sons (7 enregistrements Iowa + 3 synthèses)
    data/catalog_audio.json   # métadonnées des sons (agent SONS)
    data/biomech/*.npz        # EMG + cinématique (agent BIOMECA)
  tests/
  tools/                      # scripts de génération des données
```

## API signals.py
```python
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
def list_signals() -> list[SignalInfo]
def load_signal(id: str) -> tuple[np.ndarray, SignalInfo]   # float64 mono, normalisé |x|<=1 (EMG: en mV réalistes ok mais normalisé pour l'écoute par audio.py)
```

## API dsp.py
```python
@dataclass
class ParamSpec: key; label; default; min; max; step; unit; log=False
@dataclass
class ProcessorSpec: key; label; params: list[ParamSpec]; applies_to: set[str]  # sous-ensemble de {"audio","emg","kin"}
                     help: str   # courte explication pédagogique en français
                     func: Callable  # func(x, fs, **params) -> ndarray même longueur que x
REGISTRY: dict[str, ProcessorSpec]
def processors_for(kind: str) -> list[ProcessorSpec]
def apply_chain(x, fs, chain: list[tuple[str, dict]]) -> np.ndarray
def spectrum(x, fs, window="hann", n_fft=None) -> tuple[np.ndarray, np.ndarray]  # (freqs Hz, amplitude linéaire, normalisée pour qu'un sinus d'amplitude A donne un pic ≈ A)
def find_peaks_harmonics(freqs, amp, f0=None, n_harm=10) -> dict  # {"f0":..., "harmonics":[(f,a),...]}
def note_name(f) -> str   # "A4 (+3 cents)"
```
Processeurs obligatoires (key) :
- lowpass, highpass, bandpass (Butterworth sosfiltfilt, ordre réglable), notch (coupe-bande, ex. 50/60 Hz)
- add_sine_noise (fréquence, amplitude relative), add_white_noise (SNR dB, graine fixe reproductible)
- EMG/cin : rectify, moving_rms (fenêtre ms), envelope_lowpass (redresse + passe-bas, fc), envelope_movavg (redresse + moyenne glissante, fenêtre ms)
- cin : derivative (vitesse), second_derivative (accélération) ; tous : remove_mean (détendance)
- pédagogiques "audio" : downsample (démo repliement/aliasing : décimation SANS filtre anti-repliement, signal rééchantillonné à fs initiale par maintien), bit_crush (quantification n bits), clip (saturation), gain (dB), echo
Tous doivent gérer fréquences hors-Nyquist en les bornant, sans lever d'exception.

## audio.py (agent GUI)
`to_playable(x, fs, audify_speed=1) -> (np.int16 array, 44100)` : rééchantillonne (resample_poly), normalise, fondu in/out 10 ms,
applique l'accélération pour cin/EMG (EMG : 2000 Hz joué tel quel après suréchantillonnage = grondement ; cin : accélérée ×audify_speed).
Lecture : QAudioSink/QMediaPlayer sur WAV temporaire, bouton stop.

## GUI (agent GUI)
- Fenêtre principale : **deux colonnes (Gauche / Droite)** ; chacune a : liste déroulante du signal, boutons ▶ Jouer original, ▶ Jouer traité, ■ Stop, description du signal.
- **Haut** : une figure matplotlib par côté ; un bouton bascule global "Temps ⇄ Fréquence" (+ 3e mode "Spectrogramme"). Original = bleu, traité = orange.
  Mode fréquence : marqueurs de la fondamentale et des harmoniques (détectés), nom de note + cents, axe fréquentiel lin/log, amplitude lin/dB, zoom fmax.
  Mode temps : fenêtre de zoom (début/durée) réglable. Option pour superposer ou non l'original.
  Option : fixer les axes des deux côtés pour comparer (axes liés).
- **Bas** : par côté, chaîne de traitements : combo "Ajouter un traitement" (filtré selon le type de signal), liste ordonnée des traitements avec leurs paramètres (sliders/spinbox), boutons monter/descendre/supprimer/bypass, "Tout effacer". Recalcul en direct (avec debounce).
- Infobulle `help` sur chaque traitement. Barre d'état. Thème clair lisible en projection (grandes polices).
- Presets rapides : "Retirer le 50 Hz de l'EMG", "Enveloppe EMG", "Aliasing", "Voix téléphone (passe-bande 300–3400)".
- Exporter le signal traité en WAV / CSV, sauvegarde PNG de la figure.
