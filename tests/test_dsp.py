import numpy as np
import pytest

from signallab import dsp


def sine(f, fs, dur=1.0, a=1.0):
    t = np.arange(int(fs * dur)) / fs
    return a * np.sin(2 * np.pi * f * t)


def peak_amp(x, fs, f):
    fr, a = dsp.spectrum(x, fs)
    return a[np.argmin(np.abs(fr - f))]


def test_lowpass_removes_2000():
    fs = 44100
    x = sine(440, fs) + sine(2000, fs)
    y = dsp.apply_chain(x, fs, [("lowpass", {"fc": 1000, "order": 4})])
    assert peak_amp(y, fs, 440) == pytest.approx(1.0, abs=0.05)
    assert peak_amp(y, fs, 2000) < 0.02


def test_highpass_removes_low():
    fs = 2000
    x = sine(5, fs) + sine(100, fs)
    y = dsp.highpass(x, fs, fc=40)
    assert peak_amp(y, fs, 5) < 0.01
    assert peak_amp(y, fs, 100) == pytest.approx(1.0, abs=0.05)


def test_bandpass():
    fs = 44100
    x = sine(100, fs) + sine(1000, fs) + sine(10000, fs)
    y = dsp.bandpass(x, fs, f_low=300, f_high=3400)
    assert peak_amp(y, fs, 1000) == pytest.approx(1.0, abs=0.05)
    assert peak_amp(y, fs, 100) < 0.02
    assert peak_amp(y, fs, 10000) < 0.02


def test_spectrum_amplitude():
    fs = 8000
    x = 0.7 * sine(440, fs)
    fr, a = dsp.spectrum(x, fs)
    assert a.max() == pytest.approx(0.7, rel=0.02)
    assert fr[np.argmax(a)] == pytest.approx(440, abs=1.0)
    fr2, a2 = dsp.spectrum(x, fs, n_fft=4 * len(x))
    assert a2.max() == pytest.approx(0.7, rel=0.01)
    assert len(fr2) == len(a2)


def test_notch_50hz():
    fs = 2000
    x = sine(50, fs, a=1.0) + sine(120, fs, a=1.0)
    y = dsp.notch(x, fs, f0=50, width=4)
    assert peak_amp(y, fs, 50) < 0.02
    assert peak_amp(y, fs, 120) == pytest.approx(1.0, abs=0.05)


def test_rectify_nonnegative():
    x = np.random.default_rng(0).standard_normal(1000)
    assert dsp.rectify(x, 1000).min() >= 0


def test_rms_of_sine():
    fs = 2000
    x = 2.0 * sine(37, fs, dur=2.0)
    y = dsp.moving_rms(x, fs, window_ms=100)  # 200 ms = multiple de periodes ? non, moyenne ~ ok
    mid = y[len(y) // 4: 3 * len(y) // 4]
    assert np.mean(mid) == pytest.approx(2.0 / np.sqrt(2), rel=0.03)


def test_moving_rms_no_shift():
    fs = 1000
    x = np.zeros(2000)
    x[1000 - 50:1000 + 50] = 1.0
    y = dsp.moving_rms(x, fs, 40)
    assert abs(int(np.argmax(y)) - 1000) <= 50
    assert abs(np.sum(y * np.arange(len(y))) / np.sum(y) - 1000) < 2


def test_envelope_movavg_and_lowpass():
    fs = 2000
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4 * fs)
    for key in ("envelope_lowpass", "envelope_movavg"):
        y = dsp.REGISTRY[key].func(x, fs)
        assert y.min() >= 0 and len(y) == len(x)
        assert y.std() < np.abs(x).std()


def test_derivative():
    fs = 100
    t = np.arange(500) / fs
    v = dsp.derivative(t ** 2, fs)
    assert v[10:-10] == pytest.approx(2 * t[10:-10], abs=1e-6)
    a = dsp.second_derivative(t ** 2, fs)
    assert a[10:-10] == pytest.approx(2.0, abs=1e-6)


def test_white_noise_snr_and_reproducible():
    fs = 8000
    x = sine(300, fs, dur=4.0)
    y1 = dsp.add_white_noise(x, fs, snr_db=10)
    y2 = dsp.add_white_noise(x, fs, snr_db=10)
    assert np.array_equal(y1, y2)
    snr = 10 * np.log10(np.mean(x ** 2) / np.mean((y1 - x) ** 2))
    assert snr == pytest.approx(10, abs=0.3)


def test_sine_noise_relative_to_rms():
    fs = 2000
    x = 3.0 * sine(10, fs)
    n = dsp.add_sine_noise(x, fs, freq=50, amp=0.5) - x
    assert np.sqrt(np.mean(n ** 2)) == pytest.approx(0.5 * 3 / np.sqrt(2) / np.sqrt(2), rel=0.02)
    assert peak_amp(n, fs, 50) == pytest.approx(0.5 * 3 / np.sqrt(2), rel=0.02)


def test_remove_mean():
    x = np.ones(100) * 5 + sine(10, 1000, 0.1)
    assert abs(dsp.remove_mean(x, 1000).mean()) < 1e-12


def test_downsample_aliasing():
    fs = 8000
    x = sine(3000, fs)
    y = dsp.downsample(x, fs, factor=4)   # fs eff. = 2000 -> 3000 Hz se replie vers 1000 Hz
    assert len(y) == len(x)
    fr, a = dsp.spectrum(y, fs)
    assert fr[np.argmax(a)] == pytest.approx(1000, abs=5)


def test_bitcrush_clip_gain_echo():
    fs = 8000
    x = sine(200, fs)
    assert len(np.unique(dsp.bit_crush(x, fs, 3))) <= 2 ** 3 + 1
    assert np.max(np.abs(dsp.clip(x, fs, 50))) == pytest.approx(0.5, abs=1e-9)
    assert np.max(np.abs(dsp.gain(x, fs, 6.0206))) == pytest.approx(2.0, rel=1e-3)
    e = dsp.echo(x, fs, delay_ms=100, mix=0.5, repeats=2)
    assert len(e) == len(x) and np.max(np.abs(e)) > 1.0


@pytest.mark.parametrize("key", list(dsp.REGISTRY))
@pytest.mark.parametrize("fs,n", [(100, 500), (2000, 3000), (44100, 5000)])
def test_all_same_length_float64(key, fs, n):
    x = np.random.default_rng(2).standard_normal(n)
    y = dsp.REGISTRY[key].func(x, fs)
    assert y.shape == x.shape and y.dtype == np.float64
    assert np.all(np.isfinite(y))


@pytest.mark.parametrize("key", ["lowpass", "highpass", "bandpass", "notch", "add_sine_noise"])
def test_out_of_nyquist_no_exception(key):
    fs = 100
    x = np.random.default_rng(3).standard_normal(400)
    spec = dsp.REGISTRY[key]
    for bad in (0.0, -5.0, 50.0, 1e6, float("nan")):
        kw = {p.key: bad for p in spec.params if p.key in ("fc", "f0", "f_low", "f_high", "freq")}
        y = spec.func(x, fs, **kw)
        assert len(y) == len(x) and np.all(np.isfinite(y))


def test_short_signals():
    for key in dsp.REGISTRY:
        for n in (0, 1, 3, 10):
            y = dsp.REGISTRY[key].func(np.ones(n), 1000)
            assert len(y) == n


def test_empty_chain_and_chain_order():
    x = np.arange(10.0)
    y = dsp.apply_chain(x, 100, [])
    assert np.array_equal(x, y) and y is not x
    z = dsp.apply_chain(x, 100, [("gain", {"gain_db": 0}), ("remove_mean", {})])
    assert abs(z.mean()) < 1e-12
    assert np.array_equal(dsp.apply_chain(x, 100, [("inconnu", {})]), x)


def test_processors_for_and_defaults():
    assert all("emg" in s.applies_to for s in dsp.processors_for("emg"))
    assert "derivative" in {s.key for s in dsp.processors_for("kin")}
    assert "derivative" not in {s.key for s in dsp.processors_for("audio")}
    assert "downsample" in {s.key for s in dsp.processors_for("audio")}
    for k, s in dsp.REGISTRY.items():
        assert s.help and s.label and s.key == k
        for kind in ("audio", "emg", "kin"):
            p = dsp.default_params(k, kind)
            assert set(p) == {ps.key for ps in s.params}
            dsp.REGISTRY[k].func(np.random.default_rng(0).standard_normal(3000), 2000, **p)
    assert dsp.default_params("lowpass", "emg")["fc"] == 450
    assert dsp.default_params("notch", "audio")["f0"] == 440
    assert dsp.default_params("highpass", "emg")["fc"] == 20


def test_note_name():
    assert dsp.note_name(440).startswith("La4")
    assert dsp.note_name(261.6256).startswith("Do4")
    assert dsp.note_name(440 * 2 ** (3 / 1200)) == "La4 (+3 cents)"
    assert dsp.note_name(466.1638).startswith("La#4")
    assert dsp.note_name(0) == "—"


def test_find_peaks_harmonics():
    fs = 44100
    f0 = 220.0
    x = sum(sine(f0 * k + 0.7, fs, a=1.0 / k) for k in range(1, 6))
    fr, a = dsp.spectrum(x, fs)
    r = dsp.find_peaks_harmonics(fr, a)
    assert r["f0"] == pytest.approx(f0 + 0.7, abs=0.3)
    assert len(r["harmonics"]) >= 5
    assert r["harmonics"][1][0] == pytest.approx(2 * f0 + 0.7, abs=0.3)
    r2 = dsp.find_peaks_harmonics(fr, a, f0=f0)
    assert r2["f0"] == pytest.approx(f0 + 0.7, abs=0.3)


def test_find_f0_when_harmonic_dominates():
    fs = 44100
    x = 0.5 * sine(220, fs) + 1.0 * sine(440, fs)
    fr, a = dsp.spectrum(x, fs)
    assert dsp.find_peaks_harmonics(fr, a)["f0"] == pytest.approx(220, abs=0.5)


def test_find_peaks_degenerate():
    assert dsp.find_peaks_harmonics(np.arange(10.0), np.zeros(10))["f0"] is None
