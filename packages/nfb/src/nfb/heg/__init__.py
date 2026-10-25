"""HEG: the Mendi fNIRS headband, from wire protocol to trainable metrics.

A sibling of `nfb.hrv` for the same reason that is a sibling of the EEG path: frames from a headband
are not samples from an amplifier, and pretending otherwise forces a fake sample-rate contract onto a
device that has its own. The seams follow this repo's patterns: synchronous iterators, evicting queues,
liveness and per-attach resets.
"""

from .metrics import DEFAULT_HEG_METRIC, HEG_METRICS
from .processor import (FrameProcessor, Processed, QUALITY_GOOD, QUALITY_MOTION,
                        QUALITY_SETTLING)
from .signals import HegState, SignalProcessor
from .source import (Channel, Frame, FrameSource, HEG_RATE_HZ, LiveMendi, SimulatedMendi,
                     frame_source, problem)

__all__ = [
    "Channel", "DEFAULT_HEG_METRIC", "Frame", "FrameProcessor", "FrameSource", "HEG_METRICS",
    "HEG_RATE_HZ", "HegState", "LiveMendi", "Processed", "QUALITY_GOOD", "QUALITY_MOTION",
    "QUALITY_SETTLING", "SignalProcessor", "SimulatedMendi", "frame_source", "problem",
]
