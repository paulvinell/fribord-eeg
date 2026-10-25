"""Pupil size: where the diameter comes from, and which readings can be trusted.

A sibling of `nfb.heg` and `nfb.hrv`, for the same reason: a pupil diameter is not an amplifier sample,
and it arrives with its own rate and its own failures — a blink removes it, and a lid edge shortens it.
Synchronous iterators, a processor that owns no thread, and per-attach resets, as there.
"""

from .processor import PupilProcessor, PupilState
from .source import PUPIL_RATE_HZ, PupilSample, PupilSource, SimulatedPupil, problem, pupil_source

__all__ = [
    "PUPIL_RATE_HZ", "PupilProcessor", "PupilSample", "PupilSource", "PupilState", "SimulatedPupil",
    "problem", "pupil_source",
]
