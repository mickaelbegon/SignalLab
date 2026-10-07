import numpy as np
from scipy import signal as sps

from signallab.biomech import biomech_catalog, load_biomech


def _info(i):
    return next(s for s in biomech_catalog() if s.id == i)


def test_catalog_shapes():
    cat = biomech_catalog()
    assert {s.kind for s in cat} == {"emg", "kin"}
    for s in cat:
        x = load_biomech(s.id)
        assert x.ndim == 1 and np.all(np.isfinite(x))
        assert len(x) == int(round(10 * s.fs))
        assert s.audify_speed >= 1 and s.description
    assert _info("emg_biceps_raw").fs == 2000 and _info("kin_elbow_angle").fs == 100


def test_emg_amplitude_and_50hz():
    raw, clean = load_biomech("emg_biceps_raw"), load_biomech("emg_biceps_clean")
    assert 0.5 < np.abs(clean).max() < 2.5
    f, p = sps.welch(raw, 2000, nperseg=4096)
    _, p2 = sps.welch(clean, 2000, nperseg=4096)
    i50 = np.argmin(abs(f - 50))
    near = (abs(f - 50) > 5) & (abs(f - 50) < 15)
    assert p[i50] > 20 * p[near].mean()
    assert p[i50] > 20 * p2[i50]


def test_emg_angle_correlation():
    emg = load_biomech("emg_biceps_clean")
    ang = load_biomech("kin_elbow_angle")
    env = np.sqrt(np.convolve(emg**2, np.ones(200) / 200, mode="same"))[::20]  # 100 Hz
    env = np.convolve(env, np.ones(5) / 5, mode="same")
    assert len(env) == len(ang)
    assert np.corrcoef(env, ang)[0, 1] > 0.7
    f, p = sps.welch(ang - ang.mean(), 100, nperseg=1000)
    assert abs(f[np.argmax(p)] - 0.8) < 0.15


def test_reproducible():
    assert np.allclose(load_biomech("kin_knee_gait"), load_biomech("kin_knee_gait"))
