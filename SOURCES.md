# Sources et attributions des sons

## Enregistrements réels (7 sons)

University of Iowa Musical Instrument Samples (MIS), Lawrence Fritts, Electronic Music Studios,
University of Iowa. Page : <https://theremin.music.uiowa.edu/MIS.html>.
Conditions (selon la page) : enregistrements « freely available », téléchargeables et utilisables
« for any projects, without restrictions ». Merci de citer la source.

Chaque note est découpée (2 s depuis l'attaque), mono, 16 bits / 44,1 kHz, normalisée, avec fondu de fin
(voir `tools/fetch_samples.py`, qui télécharge les fichiers d'origine et reconstruit les WAV).

| Son | Fichier d'origine (sous `https://theremin.music.uiowa.edu/sound%20files/MIS/`) |
|---|---|
| piano_A4 | `Piano_Other/piano/Piano.mf.A4.aiff` |
| cello_A4 | `Strings/cello/Cello.arco.mf.sulA.C4C5.aiff` (La4 de la gamme) |
| flute_A4 | `Woodwinds/flute/Flute.nonvib.mf.B3B4.aiff` |
| violin_A4 | `Strings/violin/Violin.arco.mf.sulA.A4B4.aiff` |
| clarinet_A4 | `Woodwinds/Bbclarinet/BbClar.mf.C4B4.aiff` |
| oboe_A4 | `Woodwinds/oboe/Oboe.mf.C4B4.aiff` |
| trumpet_A4 | `Brass/Bbtrumpet/Trumpet.novib.mf.C4B4.aiff` |

## Sons synthétisés (3 sons, CC0 / domaine public)

`sine_A4`, `square_A4`, `guitar_A4` (Karplus-Strong) : générés par `tools/make_audio.py` (numpy).
