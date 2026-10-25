"""Frame-level processing: low-pass, motion gate, settling, and the projection to HbO/HbR.

The projection lives here too, so the whole raw-frame-to-two-numbers story is in one file.

**The projection is not Beer-Lambert, and saying so is part of the design.** A real fNIRS pipeline
converts intensities to hemoglobin concentrations through molar extinction coefficients and a path
length. This does something simpler: bilateral L/R averaging with common-mode removal against the P
channel, which sits between the two and was empirically observed to move far less — the signature of a
short-separation/reference pickup. The absolute scale is therefore arbitrary; every strategy
downstream normalises, so what matters is only that the number is *monotonic* in cortical
hemodynamics. Sign convention: more arterial HbO absorbs more IR → lower IR intensity → lower `hbo`;
the direction row on the screen owns polarity, as it does for every metric in this repo.

Quality semantics on every processed frame:
    1.0  good — filtered, clean, past settling
    0.5  settling — the filters are still warming up; usable for warming a scale, not for training
    0.0  motion — the optical channels are held flat at the last good value

**The filters always advance, including through motion and settling.** Freezing them for a 2 s motion
cooldown would leave them behind the signal level and snap when they resume — the same argument, at a
smaller time constant, as the artifact stand-ins on the EEG side. What the gate changes is what is
*emitted*: held-flat optical channels, so downstream sees a boring stretch rather than a corrupted one.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from .source import Channel, Frame, HEG_RATE_HZ

QUALITY_GOOD = 1.0
QUALITY_SETTLING = 0.5
QUALITY_MOTION = 0.0

LOWPASS_HZ = 0.5
LOWPASS_ORDER = 3
"""0.5 Hz keeps the hemodynamic response (~0.01–0.2 Hz) and removes the ~1 Hz cardiac ripple that
dominates the raw stream. Order 3: sharper than 1 without ringing on transients."""

SETTLING_SECONDS = 11.0
"""The time the filters need to settle. The settling clock resets with the
filters — a settled flag that outlives what it describes is the waveform screen's warmup bug."""

MOTION_WINDOW_MS = 200.0
MOTION_COOLDOWN_MS = 2000.0
"""The gate watches accelerometer-*magnitude* standard deviation: magnitude for orientation
invariance (Z carries gravity, tilt moves it between axes), stddev for DC invariance (a steadily
tilted head is not motion). The cooldown is sticky because the optical channels lag a jolt by a
hundred-ish milliseconds of mechanical settling on the skin."""

MOTION_SENSITIVITY = 4.0
"""How far above the wearer's *own* stillness counts as movement.

**This replaced a fixed threshold of 80, and the fixed threshold was never anybody's measurement.**
It was a config default carried in with the rest of the chain, and the one thing
this repo does know about the accelerometer is that its counts have no established scale;
the IMU is recorded as raw counts. A
threshold in units nobody can convert cannot be right except by luck, and it was not lucky: on the
bench headband it held half of all frames while somebody sat still, which reports a fidgeting session
for a still one and silently drops half the training data.

So the comparison is to the person rather than to a constant. The gate keeps the recent history of its
own 200 ms standard deviations and holds when one is `sensitivity` times the median of them — the
median because it is what the wearer's noise floor looks like even when a fifth of the window is
movement. Scale-free by construction, so it survives a different headband, a different full-scale
setting on the IMU, and a firmware that renumbers the axes.

4 is the default because it is roughly where a normal-ish noise floor stops producing excursions on
its own; it is a dial on the screen, and off is one of its positions."""

MOTION_FLOOR_SAMPLES = 90
"""Stillness measured over three seconds at ~30 Hz before the gate is entitled to hold anything.

Without it the first standard deviation is compared against a median of one — itself — and everything
either always gates or never does, decided by whichever frames happened to arrive first."""


class ChannelLowpass:
    """Causal SOS Butterworth low-pass for one scalar series, seeded lazily from its first sample —
    without the seeding, a 3rd-order 0.5 Hz filter at 30 Hz spends 5–10 s ringing up from zero to the
    signal's DC level."""

    def __init__(self, cutoff_hz: float | None = LOWPASS_HZ, fs_hz: float = HEG_RATE_HZ,
                 order: int = LOWPASS_ORDER):
        if cutoff_hz is None:
            self._sos = None
            self._zi = None
            return
        nyq = 0.5 * fs_hz
        if not 0.0 < cutoff_hz < nyq:
            raise ValueError(f"cutoff {cutoff_hz} Hz needs (0, {nyq}) at {fs_hz} sps")
        self._sos = butter(order, cutoff_hz, btype="low", fs=fs_hz, output="sos")
        self._zi = None

    def push(self, x: float) -> float:
        if self._sos is None:
            return float(x)
        if self._zi is None:
            self._zi = sosfilt_zi(self._sos) * float(x)
        y, self._zi = sosfilt(self._sos, [float(x)], zi=self._zi)
        return float(y[0])


class _MotionGate:
    """Movement judged against the wearer's own stillness — see `MOTION_SENSITIVITY`."""

    FLOOR_SECONDS = 120.0
    """How much stillness the noise floor is taken over. Long, because it is meant to describe the
    person rather than the last few seconds, and a floor measured over a short window rises while
    somebody is moving and then licenses more movement."""

    def __init__(self, sensitivity: float = MOTION_SENSITIVITY,
                 window_ms: float = MOTION_WINDOW_MS, cooldown_ms: float = MOTION_COOLDOWN_MS,
                 fs_hz: float = HEG_RATE_HZ):
        self._capacity = max(2, int(round(window_ms / 1000.0 * fs_hz)))
        self._buf: deque[float] = deque(maxlen=self._capacity)
        self.sensitivity = float(sensitivity)
        self._floor: deque[float] = deque(maxlen=max(MOTION_FLOOR_SAMPLES,
                                                     int(round(self.FLOOR_SECONDS * fs_hz))))
        self._cooldown_s = cooldown_ms / 1000.0
        self._gated_until = 0.0

    @property
    def stillness(self) -> float | None:
        """The noise floor in the accelerometer's own counts, once there is enough of it to be one."""
        if len(self._floor) < MOTION_FLOOR_SAMPLES:
            return None
        return float(np.median(np.fromiter(self._floor, dtype=float, count=len(self._floor))))

    def push(self, t: float, acc: tuple[float, float, float]) -> bool:
        # The window always advances — gated samples still inform the noise floor, or one event
        # would make the gate permanently sticky.
        self._buf.append(math.sqrt(acc[0] ** 2 + acc[1] ** 2 + acc[2] ** 2))
        if len(self._buf) < self._capacity:
            return t < self._gated_until
        n = len(self._buf)
        mean = sum(self._buf) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in self._buf) / n)
        # Movement contributes to the floor as well: the median wants the whole distribution, and a
        # floor fed only what it already called still would be measuring its own decisions.
        self._floor.append(std)
        if t < self._gated_until:
            return True
        floor = self.stillness
        if floor is None:
            return False        # not yet entitled to an opinion; see MOTION_FLOOR_SAMPLES
        # `std > 0` matters only for a sensor whose floor is exactly zero — a simulator, or a
        # perfectly quantised stretch. There, any excursion at all is movement, and without the
        # second half of this every frame would clear a threshold of zero.
        if std > self.sensitivity * floor and std > 0.0:
            self._gated_until = t + self._cooldown_s
            return True
        return False


@dataclass(frozen=True)
class Processed:
    """One frame after the chain: the projection, plus what a screen needs to say about it."""

    t: float
    hbo: float                  # mean(L.ir, R.ir) − P.ir, filtered
    hbr: float                  # mean(L.red, R.red) − P.red, filtered
    raw_ir: float               # bilateral IR mean, unsubtracted — the display's raw trace
    quality: float              # QUALITY_GOOD / QUALITY_SETTLING / QUALITY_MOTION
    temp_c: float


class FrameProcessor:
    """Six low-pass filters (L/R/P × IR/red — ambient passes through), the motion gate, the settling
    clock, and the held-last-good buffer. One instance per source; `reset()` starts a fresh record."""

    def __init__(self, fs_hz: float = HEG_RATE_HZ, lowpass_hz: float | None = LOWPASS_HZ,
                 settling_seconds: float = SETTLING_SECONDS, motion_gate: bool = True,
                 motion_sensitivity: float = MOTION_SENSITIVITY):
        self.fs_hz = fs_hz
        self.lowpass_hz = lowpass_hz
        self.settling_seconds = settling_seconds
        # **Both of these are live, not construction-time.** The gate is always built and `motion_gate`
        # is read per frame, so switching the hold off mid-session takes effect on the next frame rather
        # than at the next session — which is the only way somebody can tell whether it was the gate
        # spoiling their run. Rebuilding the processor instead would reset the filters and the settling
        # clock, costing eleven seconds to answer that question.
        self.motion_gate = motion_gate
        self.gated = 0
        self.seen = 0
        self._build()
        self.motion_sensitivity = motion_sensitivity

    @property
    def motion_sensitivity(self) -> float:
        return self._gate.sensitivity

    @motion_sensitivity.setter
    def motion_sensitivity(self, value: float) -> None:
        self._gate.sensitivity = max(1.0, float(value))

    @property
    def stillness(self) -> float | None:
        """What the gate is comparing against, in the accelerometer's own counts. For the screen."""
        return self._gate.stillness

    def _build(self) -> None:
        self._lp = {(ch, wl): ChannelLowpass(self.lowpass_hz, self.fs_hz)
                    for ch in "lrp" for wl in ("ir", "red")}
        keep = getattr(self, "_gate", None)
        self._gate = _MotionGate(fs_hz=self.fs_hz)
        if keep is not None:
            self._gate.sensitivity = keep.sensitivity     # a reset is not a change of setting
        self._start_t: float | None = None
        self._held: tuple[float, float] | None = None    # (hbo, hbr) at the last good frame

    def reset(self) -> None:
        self.gated = 0
        self.seen = 0
        self._build()

    @property
    def gate_fraction(self) -> float:
        """Share of frames held for motion — the screen's honesty figure, like the EEG gate's."""
        return self.gated / self.seen if self.seen else 0.0

    def _filtered(self, name: str, ch: Channel) -> Channel:
        return Channel(ir=self._lp[(name, "ir")].push(ch.ir),
                       red=self._lp[(name, "red")].push(ch.red), amb=ch.amb)

    def process(self, frame: Frame) -> Processed:
        self.seen += 1
        l = self._filtered("l", frame.l)
        r = self._filtered("r", frame.r)
        p = self._filtered("p", frame.p)
        bilateral_ir = 0.5 * (l.ir + r.ir)
        bilateral_red = 0.5 * (l.red + r.red)
        hbo = float(bilateral_ir - p.ir)
        hbr = float(bilateral_red - p.red)

        if self._start_t is None:
            self._start_t = frame.t
        settled = (frame.t - self._start_t) >= self.settling_seconds
        # Pushed even with the hold switched off, so the noise floor keeps being measured and switching
        # it back on mid-session judges against the whole session rather than starting again.
        moved = self._gate.push(frame.t, tuple(map(float, frame.acc)))
        gated = moved and self.motion_gate

        if gated:
            self.gated += 1
            quality = QUALITY_MOTION
            if self._held is not None:
                hbo, hbr = self._held      # optical channels hold flat; time and quality advance
        elif not settled:
            quality = QUALITY_SETTLING
            self._held = (hbo, hbr)
        else:
            quality = QUALITY_GOOD
            self._held = (hbo, hbr)

        return Processed(t=frame.t, hbo=hbo, hbr=hbr, raw_ir=float(bilateral_ir),
                         quality=quality, temp_c=frame.temp_c)
