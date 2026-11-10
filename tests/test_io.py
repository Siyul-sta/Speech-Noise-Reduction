"""Structural tests that do not require the actual engine implementations."""
import numpy as np
from src import engine_runner


def test_shape_and_channel_preservation(monkeypatch):
    monkeypatch.setattr(engine_runner, "enhance_a", lambda x, sr: x * 0.7)
    x = np.arange(20, dtype=float).reshape(10, 2) / 100
    out = engine_runner.enhance(x, 16000, "a")
    assert out.shape == x.shape
    np.testing.assert_allclose(out, x * 0.7)


def test_reject_unsupported_method():
    try:
        engine_runner.enhance(np.zeros(10), 16000, "auto")
    except ValueError:
        pass
    else:
        raise AssertionError("Unvalidated automatic routing must be rejected")
