"""Sample-stream decoding: channel interleave plans and NOT_DATA frames."""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .protocol import ProtocolError

from .tables import TEST_RATE_EXPONENT, rate_exponent

from .config import CFG_CHANNEL_STRIDE, CFG_HEADER, CHANNEL_FIELDS, get_field


@dataclass
class ChannelPlan:
    """What the sample decoder needs to know about one channel, read out of the config
    blob's per-channel record."""

    index: int
    enabled: bool
    rate_exponent: int
    mask: int = 0  # decimation mask, filled in by plan_channels()


def plan_channels(cfg: bytes, test_mode: bool = False) -> list[ChannelPlan]:
    """Derive the channel interleave plan from a config blob.

    Mirrors the driver's own setup at 0x10007044: the largest rate exponent among enabled
    channels (floor 4) sets the group rate, and each channel's decimation mask is
    (1 << (max - own)) - 1.

    In test_mode the driver leaves every mask at 0 and excludes channels whose function is
    not < 4, so all participating channels report once per phase group at TEST_RATE_HZ.
    """
    if len(cfg) < CFG_HEADER:
        raise ValueError(f"config blob too short: {len(cfg)} bytes")
    nchan = (len(cfg) - CFG_HEADER) // CFG_CHANNEL_STRIDE
    chans = []
    for i in range(nchan):
        rec = cfg[CFG_HEADER + i * CFG_CHANNEL_STRIDE :][:CFG_CHANNEL_STRIDE]
        chans.append(
            ChannelPlan(
                index=i,
                enabled=bool(get_field(rec, CHANNEL_FIELDS["enable"])),
                rate_exponent=get_field(rec, CHANNEL_FIELDS["sample_rate"]),
            )
        )
    if test_mode:
        for c in chans:
            func = get_field(cfg, CHANNEL_FIELDS["channel_function"], c.index)
            c.enabled = c.enabled and func < 4
            c.rate_exponent = TEST_RATE_EXPONENT
            c.mask = 0
        return chans
    top = max([c.rate_exponent for c in chans if c.enabled] + [4])
    for c in chans:
        c.mask = (1 << (top - c.rate_exponent)) - 1 if c.enabled else 0
    return chans


@dataclass
class DataFrame:
    """A decoded NOT_DATA payload."""

    phase: int
    sum_state: int
    channel_states: dict[int, int]
    samples: dict[int, list[int]]  # channel index -> raw int16 samples
    next_phase: int


def decode_data(payload: bytes, plan: list[ChannelPlan], phase: int | None = None) -> DataFrame:
    """Decode a NOT_DATA payload into per-channel raw sample lists.

    Reimplements the decoder at 0x100071d4. Samples are raw signed 16-bit counts; no unit
    conversion is applied, because the counts-to-volts factor depends on the measurement
    range and has not been recovered yet.

    `phase` overrides the frame's own phase counter for interleave purposes. Frames can end
    mid-group, so a continuous stream should thread `next_phase` from one call to the next
    rather than trusting each frame's header to start a group.
    """
    if len(payload) < 4:
        raise ProtocolError(f"NOT_DATA payload is {len(payload)} bytes, needs 4")
    frame_phase, state = struct.unpack_from("<HH", payload, 0)
    cur = frame_phase if phase is None else phase

    sum_state = state & 0x03
    channel_states = {c.index: (state >> (2 * (c.index + 1))) & 0x03 for c in plan if c.enabled}

    body = payload[4:]
    body = body[: len(body) & ~1]  # the driver truncates to a whole number of int16s
    values = struct.unpack(f"<{len(body) // 2}h", body)

    samples: dict[int, list[int]] = {c.index: [] for c in plan if c.enabled}
    i = 0
    while i < len(values):
        for c in plan:
            if not c.enabled:
                continue
            if c.mask & (cur + 1):
                continue  # not due at this phase
            if i >= len(values):
                break
            samples[c.index].append(values[i])
            i += 1
        cur = (cur + 1) & 0xFFFF
    return DataFrame(frame_phase, sum_state, channel_states, samples, cur)
