"""Neurofeedback acquisition and processing.

Three boundaries, deliberately: acquisition (`sources`, `decode`, `stream`) owns devices and
turns frames into absolute-indexed sample blocks; processing works on blocks and knows nothing
about devices; feedback maps values to output and is the only place a reward mapping lives.

`record` spans them — sessions are stored as raw device frames so that processing can be
developed and regression-tested against real data with no hardware attached.
"""

from .stream import Block, ChannelSpec, Gap, PhaseTracker, SampleSource
from .record import SessionHeader, SessionReader, SessionWriter
from .decode import FrameDecoder
from .device import Device, EpochEnd, Mode, Settings, Status, Subscription
from .reader import Arrival, FrameReader
from .recorder import Recorder
from .edf import EDFWriter
from .sources import OptimaSource, ReplaySource, SampleFileSource, WaitingSource

__all__ = [
    "Block", "ChannelSpec", "Gap", "PhaseTracker", "SampleSource",
    "SessionHeader", "SessionReader", "SessionWriter",
    "FrameDecoder", "Arrival", "FrameReader", "OptimaSource", "ReplaySource",
    "SampleFileSource", "WaitingSource",
    "Device", "Mode", "Settings", "Status", "Subscription", "EpochEnd",
]
