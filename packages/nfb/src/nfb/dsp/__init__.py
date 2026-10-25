"""Signal processing: blocks in, readings out.

Pure functions over arrays plus stateful streaming wrappers around them, so every measurement
can be checked offline against a known signal and then run live without a second implementation.
Nothing here knows about devices or feedback.
"""

from .artifact import ArtifactGate, ArtifactGates, ArtifactPolicy, Judgement, REJECT_MODES
from .buffer import AlignedWindower, ChannelBuffer, Window, Windower
from .filters import (
    ALPHA, BETA, DELTA, GAMMA, SMR, THETA,
    Band, ELLIPTIC_RIPPLE_DB, FILTER_KINDS, OnePole, StreamFilter, bandpass, bandstop,
    highpass, lowpass, notch,
)
from .synchrony import (
    DEFAULT_WIDTH_HZ, MEASURES, ChannelAligner, Lockin, LockinBandPower, PairAligner, Reward,
    SynchronyMetric, bandwidth_tau,
)
from .product import (
    DEFAULT_BANDWIDTH_HZ, DEFAULT_ORDER as PRODUCT_ORDER, DEFAULT_POST_SECONDS,
    FORK_MARGIN_HZ, THETA_NOTCH, ProductSynchrony, edges, is_lowpass,
)
from .coupling import (
    DEFAULT_WINDOW_SECONDS as COUPLING_WINDOW_SECONDS, PHASE_BINS, SURROGATE_SHIFTS,
    Coupling, CouplingMetric, Locking, LockingMetric, modulation_index,
)
from .criticality import (
    BALANCED, DFA_FLOOR, DFA_MIN_SECONDS, DFA_NOISE, FEI_OVERLAP, FEI_WINDOW_SECONDS, dfa,
    functional_ei,
)
from .scp import SCP_LOWPASS_HZ, SCP_ORDER, SlowShift, shift_survival
from .wave import (
    DETREND_POLES, DETREND_RATIO, MAX_HZ as WAVE_MAX_HZ, SignedWave, detrend_tau, wave_edges,
)
from .envelope import (
    CFIRBandPower, CFIREnvelope, DRIFT_CORNER_MAX_HZ, band_centre_gain, design_cfir,
    true_envelope,
)
from .metrics import (
    BandPower, Metric, PairMetric, Reading, WindowMetric,
    band_amplitude, band_power, band_ratio, coherence, lempel_ziv,
    phase_locking_value, sample_entropy,
)
from .pipeline import Pipeline

__all__ = [
    "ArtifactGate", "ArtifactGates", "ArtifactPolicy", "Judgement", "REJECT_MODES",
    "DEFAULT_WIDTH_HZ", "MEASURES", "ChannelAligner", "Lockin", "LockinBandPower", "PairAligner",
    "Reward", "SynchronyMetric", "bandwidth_tau",
    "DEFAULT_BANDWIDTH_HZ", "DEFAULT_POST_SECONDS", "PRODUCT_ORDER", "FORK_MARGIN_HZ",
    "THETA_NOTCH", "ProductSynchrony", "edges", "is_lowpass",
    "AlignedWindower", "ChannelBuffer", "Window", "Windower",
    "Band", "ELLIPTIC_RIPPLE_DB", "FILTER_KINDS", "OnePole", "StreamFilter", "bandpass", "bandstop",
    "highpass", "lowpass",
    "notch",
    "DELTA", "THETA", "ALPHA", "SMR", "BETA", "GAMMA",
    "BandPower", "Metric", "PairMetric", "Reading", "WindowMetric",
    "band_amplitude", "band_power", "band_ratio", "coherence", "lempel_ziv",
    "phase_locking_value", "sample_entropy",
    "CFIRBandPower", "CFIREnvelope", "DRIFT_CORNER_MAX_HZ", "band_centre_gain", "design_cfir",
    "true_envelope",
    "DETREND_POLES", "DETREND_RATIO", "WAVE_MAX_HZ", "SignedWave", "detrend_tau", "wave_edges",
    "COUPLING_WINDOW_SECONDS", "PHASE_BINS", "SURROGATE_SHIFTS",
    "Coupling", "CouplingMetric", "Locking", "LockingMetric", "modulation_index",
    "BALANCED", "DFA_FLOOR", "DFA_MIN_SECONDS", "DFA_NOISE", "FEI_OVERLAP", "FEI_WINDOW_SECONDS", "dfa",
    "functional_ei",
    "SCP_LOWPASS_HZ", "SCP_ORDER", "SlowShift", "shift_survival",
    "Pipeline",
]
