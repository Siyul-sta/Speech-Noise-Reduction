"""Regression tests for selectable v4 DSP inference and its fixed config."""
import numpy as np

from src.engine_runner import enhance
from src.engines.tsnr_hrnr_v3 import Config, enhance as enhance_with_config
from src.engines.tsnr_hrnr_v4 import V4_CONFIG, enhance as enhance_v4


def test_v4_config_does_not_change_v3_defaults():
    assert Config().harmonic_weight == V4_CONFIG.harmonic_weight == 2.0
    assert Config().adapt == 0.0
    assert Config().dry_mix == 0.0
    assert V4_CONFIG.adapt == 0.01
    assert V4_CONFIG.dry_mix == 0.05


def test_v4_wrapper_matches_explicit_v3_configuration():
    sr = 16000
    rng = np.random.default_rng(7)
    x = 0.1 * np.sin(2 * np.pi * 300 * np.arange(sr) / sr) + 0.01 * rng.normal(size=sr)
    expected = enhance_with_config(x, sr, V4_CONFIG)
    actual = enhance_v4(x, sr)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(enhance(x, sr, "v4"), expected, atol=1e-12)
    assert len(actual) == len(x)
    assert np.isfinite(actual).all()


def test_v4_stereo_retains_shape():
    x = np.zeros((2048, 2), dtype=np.float64)
    y = enhance(x, 16000, "v4")
    assert y.shape == x.shape
