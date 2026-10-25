"""The acquisition boundary: device frames in, timestamped sample blocks out.

Everything above this layer works in absolute sample indices and physical units, and knows
nothing about devices, framing or wire encodings. Everything below it deals in frames.

The one job that cannot be done anywhere else is turning the device's 16-bit phase counter into
a monotonic index. The counter wraps every 65536 sample groups — measured at 262.3 s on a
250 sps channel, 33 s at 2000 sps — so a raw phase must never escape upward into a timestamp, a
recording index, or cross-channel alignment. PhaseTracker does the unwrapping; nothing above
sees a 16-bit value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol

PHASE_MODULUS = 1 << 16

# The two signal states that mean the device thinks something is wrong. Named because the numbers were
# wrong for a long time in the other direction — see `Block.state`. 0 is a usable channel; 1 has never been
# observed on this hardware.
OK = 0
NO_SIGNAL = 2               # an open or disconnected input, verified with nothing attached
OVER_RANGE = 3


@dataclass(frozen=True)
class ChannelSpec:
    """Static description of one channel, fixed for a session."""

    index: int
    rate_hz: float
    unit: str
    scale: float              # physical units per raw count
    function: str = "voltage"
    decimation: int = 1       # sample groups between this channel's samples


@dataclass(frozen=True)
class Block:
    """A run of consecutive samples from one channel.

    `start_index` is an absolute, monotonic, per-channel sample index: sample 0 is the first
    sample of the session and it never wraps or resets. Samples are raw counts; `values`
    applies the channel's scale.
    """

    channel: int
    start_index: int
    samples: tuple[int, ...]
    spec: ChannelSpec
    state: int = 0
    """Device signal state for this channel over this frame, two bits from every data frame.

    **The usable value is 0, not 1, and getting that wrong made every live session look broken.** The
    protocol notes only ever established one value from hardware — 2 with no electrodes attached, so 2 is an
    open or disconnected input — and "1 ok" beside it was a transcription of the vendor manual's *prose
    ordering* rather than a wire value. Measured on this device with a working electrode at C4: **19960
    samples of real EEG arrived with state 0** and one frame with state 2, while three channels with nothing
    attached reported 2 throughout.

    So: **0 usable, 2 no signal, 3 over range**, and 1 is unobserved — no channel here has ever emitted it.
    `valid` therefore asks whether the state is one of the two the device uses to say something is *wrong*,
    which is robust to whatever 1 turns out to mean.
    """

    @property
    def valid(self) -> bool:
        """Whether the device considered this channel's signal usable.

        Nothing in the repo caught the old `state == 1` because nothing exercised a live device: the test
        fake was written with the same assumption (`0b01_01_01_01_01`) and the file sources synthesise their
        own states. Every session from a recording was therefore valid and every session from the amplifier
        was not — which is exactly the shape of the symptom, a screen that works from a file and shows a
        dead grey trace from the device.
        """
        return self.state not in (NO_SIGNAL, OVER_RANGE)

    @property
    def values(self) -> list[float]:
        return [s * self.spec.scale for s in self.samples]

    @property
    def end_index(self) -> int:
        return self.start_index + len(self.samples)

    @property
    def start_seconds(self) -> float:
        return self.start_index / self.spec.rate_hz


@dataclass
class Gap:
    """A detected discontinuity: the device's phase counter skipped."""

    expected_group: int
    actual_group: int

    @property
    def groups_lost(self) -> int:
        return (self.actual_group - self.expected_group) % PHASE_MODULUS


class SampleSource(Protocol):
    """What a live device and a recording both look like from above."""

    def channels(self) -> dict[int, ChannelSpec]: ...
    def blocks(self) -> Iterator[Block | Gap]: ...
    def close(self) -> None: ...


@dataclass
class PhaseTracker:
    """Unwraps the device's 16-bit phase counter into a monotonic group index.

    Feed it each frame's phase in arrival order; it returns the absolute group index of that
    frame's first group. Gaps are reported rather than silently absorbed, because a gap means
    lost samples and any metric computed across it is invalid.
    """

    groups_seen: int = 0
    absolute: int = 0
    _expected: int | None = None
    _last_wrapped: int | None = None
    _last_count: int = 0
    _rebase: bool = False
    _rebase_modulus: int = 1
    gaps: list[Gap] = field(default_factory=list)

    def rebase(self, modulus: int = 1) -> None:
        """Declare the next frame contiguous with the last, whatever its phase counter says.

        For loopers: a replay that starts its file over is not missing samples — the seam is a splice,
        not a hole — and without this the wrapped counter restarting reads as a ~63k-group jump. Every
        filter downstream then resets and re-runs its settling time once per loop, which on a short
        recording under a slow protocol means the screen never settles at all.

        `modulus` is the decimation grid: sample layout *inside* a frame follows the wrapped counter, so
        the absolute counter must agree with it modulo the largest `mask + 1` in the plan, or a decimated
        channel's samples would be attributed to the wrong indices. With every channel at full rate it is
        1 and the seam is perfectly seamless; on a mixed-rate plan up to `modulus - 1` groups are skipped
        to stay on the grid — a hole of a few samples once per loop, against 63k without it.
        """
        self._rebase = True
        self._rebase_modulus = max(1, int(modulus))

    def feed(self, wrapped_phase: int, group_count: int) -> tuple[int, Gap | None]:
        gap = None
        if self._expected is None:
            self.absolute = wrapped_phase          # anchor on the first frame
        elif self._rebase:
            cont = self.absolute + self._last_count
            cont += (wrapped_phase - cont) % self._rebase_modulus
            self.absolute = cont                   # contiguous by decree: no gap reported
        else:
            step = (wrapped_phase - self._last_wrapped) % PHASE_MODULUS
            if wrapped_phase != self._expected:
                gap = Gap(self._expected, wrapped_phase)
                self.gaps.append(gap)
            self.absolute += step
        self._rebase = False
        self._last_wrapped = wrapped_phase
        self._last_count = group_count
        self._expected = (wrapped_phase + group_count) % PHASE_MODULUS
        start = self.absolute
        self.groups_seen += group_count
        return start, gap


def first_due_group(start_group: int, group_count: int, mask: int) -> int | None:
    """First group at or after `start_group` at which a channel with this mask is sampled.

    Mirrors the device's rule: a channel is due when `mask & (group + 1) == 0`. Walks rather
    than solving in closed form because group_count is single digits and being obviously
    correct matters more here than being clever.
    """
    for g in range(start_group, start_group + group_count):
        if mask & (g + 1) == 0:
            return g
    return None


def sample_index(group: int, mask: int) -> int:
    """Per-channel sample index of the sample delivered at absolute `group`.

    A channel with mask 2**k - 1 is sampled at groups k-aligned to the end of each block of
    2**k, so its n-th sample sits at group (n + 1) * 2**k - 1.
    """
    stride = mask + 1
    return (group + 1) // stride - 1
