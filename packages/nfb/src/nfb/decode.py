"""Frames to blocks. The one implementation both the live device and replay use.

Keeping this separate from either source is what makes replay trustworthy: a recording decodes
through exactly the same path as the hardware did, so an offline result and a live result differ
only if the frames differ.
"""

from __future__ import annotations

import optima

from .stream import Block, ChannelSpec, Gap, PhaseTracker, first_due_group, sample_index


class FrameDecoder:
    """Turns NOT_DATA frames into per-channel blocks with absolute sample indices.

    Built from the config blob, because nothing else knows each channel's rate, scale or
    interleave mask. `test_mode` switches to the impedance test's plan: no decimation, only
    channels whose function is analog, and a fixed 3.90625 Hz.
    """

    def __init__(self, config: bytes, test_mode: bool = False):
        self.config = config
        self.test_mode = test_mode
        self.plan = optima.plan_channels(config, test_mode=test_mode)
        self.tracker = PhaseTracker()
        self.specs = self._build_specs()
        self.events: list[optima.Message] = []

    def _build_specs(self) -> dict[int, ChannelSpec]:
        specs = {}
        for c in self.plan:
            if not c.enabled:
                continue
            func = optima.get_field(self.config, optima.CHANNEL_FIELDS["channel_function"], c.index)
            if self.test_mode:
                scale, unit = optima.test_scale(), optima.TEST_UNIT
            else:
                scale, unit = optima.channel_scale(self.config, c.index)
            specs[c.index] = ChannelSpec(
                index=c.index,
                rate_hz=optima.rate_hz(c.rate_exponent),
                unit=unit,
                scale=scale,
                function=optima.FUNC_NAME.get(func, str(func)),
                decimation=c.mask + 1,
            )
        return specs

    def channels(self) -> dict[int, ChannelSpec]:
        return dict(self.specs)

    def rebase(self) -> None:
        """The next frame continues the stream, whatever its phase counter says. See
        `PhaseTracker.rebase` — this passes the plan's decimation grid so mixed-rate channels keep
        their sample alignment across the seam."""
        self.tracker.rebase(max((c.mask + 1 for c in self.plan if c.enabled), default=1))

    def feed(self, code: int, payload: bytes) -> list[Block | Gap]:
        """Decode one frame. Non-data notifications are collected into `events`."""
        if code != optima.NOT_DATA:
            if code & 0xC0 == optima.CLASS_NOT:
                self.events.append(optima.Message(code, payload))
            return []

        frame = optima.decode_data(payload, self.plan)
        group_count = (frame.next_phase - frame.phase) % 0x10000
        start_group, gap = self.tracker.feed(frame.phase, group_count)

        out: list[Block | Gap] = []
        if gap is not None:
            out.append(gap)
        for c in self.plan:
            if not c.enabled:
                continue
            samples = frame.samples.get(c.index)
            if not samples:
                continue
            first = first_due_group(start_group, group_count, c.mask)
            if first is None:
                continue
            out.append(Block(
                channel=c.index,
                start_index=sample_index(first, c.mask),
                samples=tuple(samples),
                spec=self.specs[c.index],
                state=frame.channel_states.get(c.index, 0),
            ))
        return out
