"""V4 candidate: adapt noise PSD gently and preserve a little dry speech.

No clean/reference signal is accepted during inference. This module reuses the
validated v3 DSP implementation with an explicit fixed configuration to ensure
the tested parameters are the exact parameters used for submission.
NOTE: Official hidden-test PESQ-WB has NOT been measured for v4.
"""
from .tsnr_hrnr_v3 import Config, enhance as enhance_with_config

V4_CONFIG = Config(
    harmonic_weight=2.0,  # preserve v3 HRNR; reducing it hurt proxy metrics
    adapt=0.01,            # slow, speech-gated noise PSD update
    dry_mix=0.05,          # gently preserve speech components
)


def enhance(x, sr):
    """Denoise mono noisy waveform, returning same-length waveform."""
    return enhance_with_config(x, sr, V4_CONFIG)
