"""Acquisition-boundary tests: phase unwrapping, indexing, and replay fidelity.

The wrap tests exist because the earlier hardware verification ran for 90 s, which is under the
262 s wrap period, so it could not have caught a wrap bug. These make the failure mode cheap to
check.
"""

from __future__ import annotations

import struct
import time

import optima
import pytest

from conftest import waitfor

from nfb.decode import FrameDecoder
from nfb.record import SessionHeader, SessionReader, SessionWriter
from nfb.sources import ReplaySource
from nfb.stream import PHASE_MODULUS, Gap, PhaseTracker, first_due_group, sample_index

GROUPS_PER_FRAME = 8


def eeg_config(rates=(8, 8, 8, 8)):
    cfg = optima.build_eeg_config()
    for ch, exp in enumerate(rates):
        cfg = optima.set_field(cfg, optima.CHANNEL_FIELDS["sample_rate"], exp, ch)
    assert optima.validate_config(cfg) == []
    return cfg


def frame(phase, plan, groups=GROUPS_PER_FRAME, state=0,
          value=lambda ch, g: ch * 1000 + g % 1000):   # keep inside int16
    """Build a NOT_DATA payload following the device's own interleave rule."""
    vals = []
    for step in range(groups):
        g = (phase + step) % PHASE_MODULUS
        for c in plan:
            if c.enabled and not (c.mask & (g + 1)):
                vals.append(value(c.index, g))
    return struct.pack("<HH", phase, state) + struct.pack(f"<{len(vals)}h", *vals)


# --- phase unwrapping ---------------------------------------------------------------------

def test_tracker_is_monotonic_across_the_wrap():
    t = PhaseTracker()
    phase = PHASE_MODULUS - 4 * GROUPS_PER_FRAME
    starts = []
    for _ in range(8):
        start, gap = t.feed(phase, GROUPS_PER_FRAME)
        assert gap is None
        starts.append(start)
        phase = (phase + GROUPS_PER_FRAME) % PHASE_MODULUS
    assert starts == sorted(starts), "absolute index went backwards at the wrap"
    assert starts[-1] - starts[0] == 7 * GROUPS_PER_FRAME
    # the wrap must not be mistaken for a discontinuity
    assert t.gaps == []


def test_tracker_reports_a_real_gap():
    t = PhaseTracker()
    t.feed(0, GROUPS_PER_FRAME)
    _, gap = t.feed(GROUPS_PER_FRAME * 3, GROUPS_PER_FRAME)   # skipped two frames
    assert isinstance(gap, Gap)
    assert gap.groups_lost == GROUPS_PER_FRAME * 2


def test_tracker_reports_a_gap_that_straddles_the_wrap():
    t = PhaseTracker()
    t.feed(PHASE_MODULUS - GROUPS_PER_FRAME, GROUPS_PER_FRAME)
    _, gap = t.feed(GROUPS_PER_FRAME, GROUPS_PER_FRAME)       # expected 0, lost one frame
    assert gap is not None and gap.groups_lost == GROUPS_PER_FRAME


def test_index_arithmetic_matches_the_device_rule():
    for mask in (0, 1, 3, 7):
        due = [g for g in range(64) if mask & (g + 1) == 0]
        assert [sample_index(g, mask) for g in due] == list(range(len(due)))
        assert first_due_group(0, 64, mask) == due[0]
    assert first_due_group(0, 1, 3) is None      # nothing due in this window


# --- decoding -----------------------------------------------------------------------------

def test_blocks_carry_absolute_indices_and_survive_the_wrap():
    cfg = eeg_config()
    d = FrameDecoder(cfg)
    plan = d.plan
    start_phase = PHASE_MODULUS - 2 * GROUPS_PER_FRAME
    seen: dict[int, list[int]] = {}
    for i in range(4):
        phase = (start_phase + i * GROUPS_PER_FRAME) % PHASE_MODULUS
        for item in d.feed(optima.NOT_DATA, frame(phase, plan)):
            if isinstance(item, Gap):
                pytest.fail("spurious gap across the wrap")
            seen.setdefault(item.channel, []).append(item.start_index)
    for ch, idx in seen.items():
        assert idx == sorted(idx), f"ch{ch} indices not monotonic: {idx}"
        assert idx[-1] - idx[0] == 3 * GROUPS_PER_FRAME


def test_mixed_rates_give_proportional_sample_counts_and_indices():
    cfg = eeg_config(rates=(8, 7, 6, 6))     # 250, 125, 62.5, 62.5
    d = FrameDecoder(cfg)
    counts: dict[int, int] = {}
    for i in range(8):
        for item in d.feed(optima.NOT_DATA, frame(i * GROUPS_PER_FRAME, d.plan)):
            assert not isinstance(item, Gap)
            counts[item.channel] = counts.get(item.channel, 0) + len(item.samples)
    assert counts[0] == 4 * counts[2], counts
    assert counts[1] == 2 * counts[2], counts
    specs = d.channels()
    assert specs[0].rate_hz == 250 and specs[2].rate_hz == 62.5
    assert specs[0].decimation == 1 and specs[2].decimation == 4


def test_channel_specs_describe_scaling():
    d = FrameDecoder(eeg_config())
    specs = d.channels()
    assert specs[0].unit == "uV"
    assert specs[0].scale == pytest.approx(800 / 32768)
    # a full-scale count must read as full-scale microvolts
    block = next(b for b in d.feed(optima.NOT_DATA,
                                  frame(0, d.plan, value=lambda ch, g: -32768)))
    assert block.values[0] == pytest.approx(-800.0)


def test_events_are_collected_not_decoded():
    d = FrameDecoder(eeg_config())
    assert d.feed(optima.NOT_EVENT, bytes([optima.EVENT_BATTERY, 3])) == []
    assert len(d.events) == 1 and d.events[0].payload[0] == optima.EVENT_BATTERY


# --- recording round trip -----------------------------------------------------------------

def test_replay_reproduces_live_decoding_exactly(tmp_path):
    cfg = eeg_config(rates=(8, 7, 6, 6))
    plan = optima.plan_channels(cfg)
    frames = [frame(i * GROUPS_PER_FRAME, plan) for i in range(12)]

    path = tmp_path / "s.jsonl.gz"
    with SessionWriter(path, SessionHeader(config=cfg, device="test", serial="1")) as w:
        for i, payload in enumerate(frames):
            w.write(optima.NOT_DATA, payload, elapsed=i * 0.032)

    live = FrameDecoder(cfg)
    live_out = [b for payload in frames for b in live.feed(optima.NOT_DATA, payload)]
    replay_out = list(ReplaySource(path).blocks())

    assert len(replay_out) == len(live_out) > 0
    for a, b in zip(live_out, replay_out):
        assert (a.channel, a.start_index, a.samples) == (b.channel, b.start_index, b.samples)
        assert a.spec == b.spec


def test_session_header_carries_what_decoding_needs(tmp_path):
    cfg = eeg_config()
    path = tmp_path / "s.jsonl"
    with SessionWriter(path, SessionHeader(config=cfg, device="Optima+ 4", serial="00000000",
                                          firmware="2.9.5", note="hello")) as w:
        w.write(optima.NOT_DATA, frame(0, optima.plan_channels(cfg)))
    h = SessionReader(path).header
    assert h.config == cfg and h.serial == "00000000" and h.note == "hello"


def test_truncated_recording_reads_up_to_the_cut(tmp_path):
    cfg = eeg_config()
    path = tmp_path / "s.jsonl"
    with SessionWriter(path, SessionHeader(config=cfg)) as w:
        for i in range(4):
            w.write(optima.NOT_DATA, frame(i * GROUPS_PER_FRAME, optima.plan_channels(cfg)))
    with open(path, "a") as fh:
        fh.write('{"c":193,"p":"00')          # interrupted mid-write
    blocks = list(ReplaySource(path).blocks())
    assert len(blocks) == 4 * 4               # four frames, four channels, nothing raised


def test_unknown_format_is_rejected(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text('{"format": "nfb-session/99", "config": "00"}\n')
    with pytest.raises(ValueError, match="unsupported session format"):
        SessionReader(path)


# --- reader thread ------------------------------------------------------------------------

class FakeDevice:
    """Stands in for optima.Optima: hands out prepared frames, then blocks."""

    def __init__(self, frames, stall=0.0):
        self._frames = list(frames)
        self._stall = stall
        self.reads = 0

    def read_message(self):
        import time as _t
        if self._stall:
            _t.sleep(self._stall)
        if not self._frames:
            raise TimeoutError("no more frames")
        self.reads += 1
        return self._frames.pop(0)


def test_reader_thread_delivers_every_frame_in_order():
    from nfb.reader import FrameReader
    cfg = eeg_config()
    plan = optima.plan_channels(cfg)
    msgs = [optima.Message(optima.NOT_DATA, frame(i * GROUPS_PER_FRAME, plan)) for i in range(20)]
    r = FrameReader(FakeDevice(msgs))
    r.start()
    got = []
    try:
        for a in r.arrivals(timeout=2.0):
            got.append(a)
            if len(got) == 20:
                break
    finally:
        r.stop()
    assert len(got) == 20
    assert [a.payload for a in got] == [m.payload for m in msgs]
    assert all(b.at >= a.at for a, b in zip(got, got[1:])), "arrival times not monotonic"
    assert r.dropped == 0


def test_reader_thread_surfaces_device_errors_to_the_consumer():
    from nfb.reader import FrameReader

    class Exploding:
        def read_message(self):
            raise optima.ProtocolError("stream out of sync")

    r = FrameReader(Exploding())
    r.start()
    try:
        with pytest.raises(optima.ProtocolError, match="out of sync"):
            next(iter(r.arrivals(timeout=2.0)))
    finally:
        r.stop()


def test_reader_counts_drops_rather_than_blocking_when_the_consumer_never_reads():
    """A full queue must not stall the reader — losing a frame is better than losing the link."""
    from nfb.reader import FrameReader
    cfg = eeg_config()
    plan = optima.plan_channels(cfg)
    msgs = [optima.Message(optima.NOT_DATA, frame(i * GROUPS_PER_FRAME, plan)) for i in range(50)]
    r = FrameReader(FakeDevice(msgs), queue_size=8)
    r.start()
    waitfor(lambda: r.dropped >= 1, what="the reader never dropped a frame")   # never consume
    r.stop()
    assert r.depth == 8, r.depth
    assert r.dropped >= 1, "reader should have dropped rather than blocked"


def test_reference_mode_is_selectable():
    """Linked by default — one shared reference, joined inside the device — but switchable."""
    F = optima.CHANNEL_FIELDS["common_reference"]
    expect = {"linked": 1, "bipolar": 0, "external": 2}
    for mode, code in expect.items():
        cfg = optima.build_eeg_config(reference=mode)
        assert optima.validate_config(cfg) == []
        assert [optima.get_field(cfg, F, ch) for ch in range(4)] == [code] * 4, mode
    assert optima.get_field(optima.build_eeg_config(), F, 0) == 1, "default must be linked"
    with pytest.raises(ValueError, match="reference must be one of"):
        optima.build_eeg_config(reference="linked-ears")
