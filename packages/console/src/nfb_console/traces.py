"""Display-resolution trace buffers.

A trace is drawn to a few hundred pixel columns, so there is no point carrying 1000 samples per
second into the UI. Each column keeps the min and max of the samples that fall in it, which is what
preserves the visual shape of a waveform under heavy decimation — plotting every Nth sample instead
would alias spikes away entirely.

Four of them, and the last two exist because the infra-slow screens cannot use the first two: a window in
seconds cannot serve both 0.05 Hz and 40 Hz, and a spectrum in 1 Hz bins has nothing to say about a 0.05 Hz
wave. `WaveRing` measures its window in *cycles* and `SlowRing` decimates before it transforms.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy import signal as sps

WAVE_CYCLES = 12
"""Cycles of the trained frequency across a `WaveRing`. Enough to read the shape of a slow wave without the
window taking so long to fill that the picture is empty for the first minutes."""

WAVE_POINTS_PER_CYCLE = 48


class TraceBuffer:
    """Min/max envelope per pixel column, for one channel."""

    def __init__(self, columns: int, seconds: float, rate_hz: float):
        self.columns = columns
        self.per_column = max(1, int(round(seconds * rate_hz / columns)))
        self._cols: deque[tuple[float, float]] = deque([(0.0, 0.0)] * columns, maxlen=columns)
        self._marks: deque[bool] = deque([False] * columns, maxlen=columns)
        self._lo = float("inf")
        self._hi = float("-inf")
        self._count = 0
        self._marked = False
        self.valid = True

    def push(self, values: np.ndarray, valid: bool = True, accepted: bool = True) -> None:
        """`accepted` is the artifact gate's verdict on this block, kept per column so the trace can
        show *which* part of the window was thrown away. A column is marked if any block feeding it
        was rejected — a blink is shorter than a column, and the excursion is the thing to point at."""
        self.valid = valid
        self._marked = self._marked or not accepted
        for v in values:
            if v < self._lo:
                self._lo = v
            if v > self._hi:
                self._hi = v
            self._count += 1
            if self._count >= self.per_column:
                self._cols.append((self._lo, self._hi))
                self._marks.append(self._marked)
                self._lo, self._hi, self._count, self._marked = (
                    float("inf"), float("-inf"), 0, False)

    def marks(self) -> np.ndarray:
        """Per column: was anything feeding it rejected. Same length as the envelope."""
        return np.fromiter(self._marks, dtype=bool, count=len(self._marks))

    def envelope(self) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(self._cols, dtype=np.float64)
        return arr[:, 0], arr[:, 1]

    def scale(self) -> float:
        """Symmetric half-range for drawing, from recent content."""
        lo, hi = self.envelope()
        peak = float(max(np.abs(lo).max(), np.abs(hi).max()))
        return peak if peak > 1e-9 else 1.0


class RawRing:
    """Raw samples, for a spectrum. The trace buffer keeps min/max envelopes, which cannot be
    transformed.

    Shared by the amplitude screen (one channel at a time) and the CFC screen (the *sum* of a channel
    pair, which is what its cascading spectral display shows). The band edges are arguments rather than
    module constants for exactly that: CFC reads 1–40 Hz because "this is where most of the clinically
    relevant changes take place during a session", while the amplitude screen goes to 60 to show mains.
    """

    def __init__(self, seconds: float, rate_hz: float, lo_hz: float = 1.0, hi_hz: float = 60.0,
                 bin_hz: float = 1.0):
        self.rate_hz = rate_hz
        self.lo_hz = lo_hz
        self.hi_hz = hi_hz
        self.bin_hz = bin_hz
        self.buf = np.zeros(int(seconds * rate_hz))
        self.filled = 0

    def push(self, values: np.ndarray) -> None:
        n = len(values)
        if n >= len(self.buf):
            self.buf[:] = values[-len(self.buf):]
        else:
            self.buf = np.roll(self.buf, -n)
            self.buf[-n:] = values
        self.filled = min(self.filled + n, len(self.buf))

    def psd(self):
        if self.filled < self.rate_hz:
            return None
        nperseg = int(min(self.filled, self.rate_hz))
        return sps.welch(self.buf[-self.filled:], fs=self.rate_hz, nperseg=nperseg)

    @property
    def top_hz(self) -> float:
        """The spectrum stops at Nyquist, whatever the display asks for. A 62.5 sps channel cannot show
        60 Hz and drawing empty bins there would imply it had looked and found nothing."""
        return min(self.hi_hz, self.rate_hz / 2)

    def centres(self) -> np.ndarray:
        """One bin *centred on* each frequency, not spanning upward from it.

        The rfft bin *k* is exactly *k* Hz, and getting it wrong is visible: binning `[f, f+1)` and then
        drawing the bar starting at f's
        gridline puts every feature half a bin to the right, so a notch at 50 Hz appears to begin at 50
        and extend past it.
        """
        return np.arange(self.lo_hz, self.top_hz + self.bin_hz / 2, self.bin_hz)

    def spectrum(self, relative: bool = False) -> np.ndarray:
        got = self.psd()
        if got is None:
            return np.zeros(0)
        freqs, psd = got
        half = self.bin_hz / 2
        centres = self.centres()
        out = np.zeros(len(centres))
        for i, centre in enumerate(centres):
            sel = (freqs >= centre - half) & (freqs < centre + half)
            if sel.any():
                out[i] = float(np.sqrt(psd[sel].mean()))
        if relative and out.sum() > 0:
            out = 100.0 * out / out.sum()
        return out

    def band_values(self, bands, relative: bool = False) -> dict[str, float]:
        got = self.psd()
        if got is None:
            return {}
        freqs, psd = got
        raw, total = {}, 0.0
        for band in bands:
            sel = (freqs >= band.low) & (freqs < band.high)
            amp = float(np.sqrt(psd[sel].mean())) if sel.any() else 0.0
            raw[band.label] = amp
            total += amp
        if relative and total > 0:
            return {k: 100.0 * v / total for k, v in raw.items()}
        return raw

    def band_peaks(self, bands) -> dict[str, float]:
        """Where inside each band the energy actually sits, in Hz. The band's *dominant frequency*.

        A band amplitude is a mean over the band, so it says how much and never says where — and where is
        what a protocol asks for constantly. The coupling screen is explicit about it: its reward frequency
        is found as "the frequency in the spectral display showing the highest amplitude in a particular
        band", and until now that meant reading it off the bars by eye.

        **Interpolated, because the answer is finer than the bins are.** The resolution of a Welch estimate
        over a one-second segment is 1 Hz, and a peak that falls between two bins reports as whichever is
        nominally larger — so a 10.4 Hz rhythm reads as "10" and a 10.6 Hz one as "11", which is a whole bin
        of movement for a tenth of one in the signal. Fitting a parabola through the three points around the
        maximum recovers the sub-bin position, which is the standard estimator for exactly this and costs
        three multiplies. The correction is clamped to half a bin: a parabola through noise can suggest an
        offset of several bins, and a peak that jumped outside its own neighbourhood would be worse than a
        quantised one.

        Reported per band rather than for the whole spectrum, since the whole spectrum's peak is nearly
        always the slowest thing present and says nothing about where alpha is.
        """
        got = self.psd()
        if got is None:
            return {}
        freqs, psd = got
        out: dict[str, float] = {}
        for band in bands:
            idx = np.flatnonzero((freqs >= band.low) & (freqs < band.high))
            if idx.size == 0 or not float(psd[idx].max()) > 0:
                continue
            top = int(idx[int(np.argmax(psd[idx]))])
            peak = float(freqs[top])
            if 0 < top < len(psd) - 1:
                a, b, c = (float(psd[top - 1]), float(psd[top]), float(psd[top + 1]))
                denom = a - 2 * b + c
                if denom != 0:
                    step = float(freqs[top + 1] - freqs[top])
                    offset = max(-0.5, min(0.5, 0.5 * (a - c) / denom))
                    peak += offset * step
            # An interpolated peak can land a hair outside the band it was found in; the band is the claim
            # being made, so it is the band that wins.
            out[band.label] = max(band.low, min(band.high, peak))
        return out



class SummedRing(RawRing):
    """The channels added, as a spectrum source — **derived from the per-channel rings, never fed.**

    Every screen keeps one `RawRing` per channel and the blocks arrive per channel in lockstep, so the
    rings' tails are the same stretch of time and their element-wise sum is the summed signal. That makes a
    sum spectrum available on every screen with two or more channels at one rate, without each screen
    building and feeding a ring of its own — which is how three screens came to offer no sum at all while
    four did, each with hand-written plumbing. A screen that *measures* on a sum keeps its own fed ring,
    since there the sum is the instrument and the alignment is the aligner's business.
    """

    def __init__(self, rings):
        self.rings = list(rings)
        first = self.rings[0]
        self.rate_hz, self.lo_hz, self.hi_hz, self.bin_hz = (first.rate_hz, first.lo_hz, first.hi_hz,
                                                            first.bin_hz)

    @property
    def filled(self) -> int:
        return min(r.filled for r in self.rings)

    @property
    def buf(self) -> np.ndarray:
        n = min(len(r.buf) for r in self.rings)
        return sum(r.buf[-n:] for r in self.rings)

    def push(self, values: np.ndarray) -> None:
        raise TypeError("a summed ring is derived from its channels and is not fed")

    def psd(self):
        n = self.filled
        if n < self.rate_hz:
            return None
        total = sum(r.buf[-n:] for r in self.rings)
        return sps.welch(total, fs=self.rate_hz, nperseg=int(min(n, self.rate_hz)))

class WaveRing:
    """Filtered waveforms, decimated to a fixed number of points per cycle of the trained frequency.

    **Two constraints, and only satisfying both gives a picture that is not lying.**

    - The window is `cycles` cycles of the *centre* frequency, because a window in seconds cannot serve
      both 0.05 Hz and 40 Hz. So the point rate has to be at least `points_per_cycle × centre`.
    - The band's *upper edge* sets a Nyquist floor. At a 0.05 Hz centre the filter is a 0–0.1 Hz lowpass, so
      the waveform holds content up to 0.1 Hz — and on a wider band, sampling at 48 points per slow cycle
      aliases the faster content into a shape it does not have. So the point rate has to be at least
      `4 × high` as well.

    Taking the maximum of the two is what makes a slow waveform and a Lissajous figure honest at both ends.
    The column count then follows the *achievable* stride rather than the reverse — the stride clamps at 1,
    and taking columns as fixed silently changed what the window was: measured, a 10 Hz ring held 23 cycles
    under a caption that said 12.

    Decimation is on the absolute sample index, not once per call, which is the discipline every estimator
    here follows: per-call would make the rate a function of the block size and the channel count. So `feed`
    takes the index the waveform starts at, which is what `ProductSynchrony.wave_start` and
    `nfb.dsp.wave` both publish for this.

    Shared by the synchrony screen, which draws two channels and a Lissajous figure off the tail, and the
    waveform screen, which draws one. It was the first of those and moved here when the second needed it —
    with `WAVE_CYCLES` and the two cycle counts becoming arguments, since a screen with no Lissajous has no
    use for a tail and the number of cycles worth showing is a screen's judgement.
    """

    def __init__(self, centre_hz: float, high_hz: float, rate_hz: float, channels,
                 cycles: int = WAVE_CYCLES, points_per_cycle: int = WAVE_POINTS_PER_CYCLE,
                 tail_cycles: float = 2.0):
        self.channels = tuple(channels)
        self.rate_hz = rate_hz
        self.centre_hz = centre_hz
        self.cycles = cycles
        self.tail_cycles = tail_cycles
        wanted = max(4.0 * high_hz, points_per_cycle * centre_hz)
        self.stride = max(1, int(round(rate_hz / max(wanted, 1e-9))))
        self.points_per_cycle = rate_hz / max(centre_hz * self.stride, 1e-9)
        self.columns = max(24, min(4000, int(round(cycles * self.points_per_cycle))))
        self.seconds = self.columns * self.stride / rate_hz
        self._series = {c: deque([0.0] * self.columns, maxlen=self.columns) for c in self.channels}
        self._next: int | None = None
        self.pushed = 0

    def feed(self, start: int | None, waves: dict) -> None:
        """`waves` is per channel, all starting at absolute index `start`."""
        if start is None:
            return
        lengths = [len(waves.get(c, ())) for c in self.channels]
        length = min(lengths) if lengths else 0
        if not length:
            return
        if self._next is None or self._next < start:
            self._next = start
        while self._next < start + length:
            offset = self._next - start
            for c in self.channels:
                self._series[c].append(float(waves[c][offset]))
            self.pushed += 1
            self._next += self.stride

    def resize(self, cycles: int) -> None:
        """Change how many cycles the window shows, keeping every point that still fits.

        Possible because the stride is set by the centre and the band's upper edge, never by the cycle
        count — the points in a 4-cycle ring and a 24-cycle ring are the same points at the same rate, so
        the tail carries over. Rebuilding instead threw the history away: at 0.05 Hz, pressing a width
        pill destroyed up to eight minutes of a wave that takes eight minutes to re-collect, for a change
        that was only ever about how much of it to *show*.
        """
        real = min(self.pushed, self.columns)
        tails = {c: list(s)[len(s) - real:] for c, s in self._series.items()}
        self.cycles = int(cycles)
        self.columns = max(24, min(4000, int(round(self.cycles * self.points_per_cycle))))
        self.seconds = self.columns * self.stride / max(self.rate_hz, 1e-9)
        kept = min(real, self.columns)
        self._series = {c: deque([0.0] * (self.columns - kept) + tails[c][real - kept:],
                                 maxlen=self.columns) for c in self.channels}
        self.pushed = kept

    def series(self) -> dict[int, np.ndarray]:
        return {c: np.asarray(s, dtype=float) for c, s in self._series.items()}

    def peak(self) -> float:
        return max((float(np.max(np.abs(s))) for s in self.series().values()), default=0.0)

    @property
    def tail_points(self) -> int:
        return max(2, min(self.columns, int(round(self.tail_cycles * self.points_per_cycle))))

    @property
    def tail_seconds(self) -> float:
        return self.tail_points * self.stride / max(self.rate_hz, 1e-9)

    def tail(self) -> tuple[np.ndarray, np.ndarray]:
        """The last `tail_cycles` of the first two channels — what a Lissajous figure is drawn from. A
        figure over the whole window is drawn on top of itself."""
        n = min(self.pushed, self.tail_points)
        if n < 2 or len(self.channels) < 2:
            return np.zeros(0), np.zeros(0)
        series = self.series()
        return series[self.channels[0]][-n:], series[self.channels[1]][-n:]


class SlowRing:
    """Raw samples decimated to a few per second, for a spectrum **below** 1 Hz.

    `RawRing` cannot do this and the arithmetic says why: resolving 0.01 Hz needs a window of a hundred
    seconds, which at 250 sps is 25 000 samples per transform, and its bins are a hertz wide where the whole
    range of interest here is one hertz. So this averages the signal down to `target_hz` first — an
    anti-aliased decimation by block mean, which is the right pre-filter for a band that stops far below the
    new Nyquist — and transforms that.

    The window is minutes, and it has to be: the screen's own frequency box reaches 0.01 Hz, and a spectrum
    that cannot see the frequency being trained is decoration. `filled_seconds` is public so a caption can
    say how much of it is real, because a spectrum drawn from a tenth of its window is not the spectrum it
    claims to be.
    """

    def __init__(self, seconds: float = 300.0, rate_hz: float = 250.0, lo_hz: float = 0.005,
                 hi_hz: float = 1.0, target_hz: float = 4.0):
        self.rate_hz = rate_hz
        self.lo_hz = lo_hz
        self.hi_hz = hi_hz
        self.stride = max(1, int(round(rate_hz / target_hz)))
        self.decimated_hz = rate_hz / self.stride
        self.buf = np.zeros(max(16, int(seconds * self.decimated_hz)))
        self.filled = 0
        self._acc = 0.0
        self._count = 0

    def push(self, values) -> None:
        for v in np.asarray(values, dtype=float):
            self._acc += v
            self._count += 1
            if self._count >= self.stride:
                self.buf = np.roll(self.buf, -1)
                self.buf[-1] = self._acc / self._count
                self.filled = min(self.filled + 1, len(self.buf))
                self._acc, self._count = 0.0, 0

    @property
    def filled_seconds(self) -> float:
        return self.filled / max(self.decimated_hz, 1e-9)

    @property
    def window_seconds(self) -> float:
        return len(self.buf) / max(self.decimated_hz, 1e-9)

    @property
    def resolution_hz(self) -> float:
        """The bin width the current fill supports — half the window, since Welch takes two segments."""
        return 2.0 / max(self.filled_seconds, 1e-9)

    MIN_SAMPLES = 32
    """Enough for a transform at all — about eight seconds at the decimated rate.

    **It used to be two cycles of `lo_hz`, which is 6.7 minutes of an empty pane.** The reasoning was that a
    spectrum which cannot see the lowest frequency asked for is not the spectrum it claims to be, and that
    is true of the *lowest bin* and of nothing else: at thirty seconds the window resolves 0.07 Hz upward
    perfectly well, and there is no honest reason to withhold those bins while waiting for one at the far
    end. A gate may say a value cannot be computed; it may not say a value is not accurate enough yet.

    The bins below what the current fill resolves come back empty on their own — no `freqs` falls inside
    them — so the picture fills in from the right as the window grows, and the caption states the resolution
    it currently has. Nothing is claimed that is not there.

    **And the old gate was unreachable, so the pane was empty forever.** `filled_seconds` saturates at
    `window_seconds`, which on the waveform screen is 300 s, while the gate wanted 2 / 0.005 = 400 — a
    condition the ring could not satisfy however long a session ran. The caption underneath said "needs
    6.7 min before it can resolve 0.005 Hz", which was a promise it could never keep. The lowest bin here
    genuinely needs a longer window than this ring holds; what that justifies is *that bin* staying empty,
    which it now does on its own, and not the other twenty-three being withheld with it."""

    def psd(self):
        """None only while there is too little to transform at all — see `MIN_SAMPLES`."""
        if self.filled < self.MIN_SAMPLES:
            return None
        segment = max(16, self.filled // 2)
        return sps.welch(self.buf[-self.filled:], fs=self.decimated_hz, nperseg=segment,
                         detrend="linear")

    def bins(self, count: int = 24) -> tuple[np.ndarray, np.ndarray]:
        """Amplitude density on **log-spaced** centres, which is the only spacing that works over two
        decades: linear bins from 0.005 to 1 Hz would put twenty of twenty-four of them above 0.2 Hz, where
        nothing this screen trains lives.

        Returns `(centres, amplitudes)`, both empty until `psd` has something.
        """
        got = self.psd()
        if got is None:
            return np.zeros(0), np.zeros(0)
        freqs, psd = got
        centres = np.geomspace(self.lo_hz, self.hi_hz, count)
        ratio = (self.hi_hz / self.lo_hz) ** (0.5 / max(1, count - 1))
        out = np.zeros(count)
        for i, centre in enumerate(centres):
            sel = (freqs >= centre / ratio) & (freqs < centre * ratio)
            if sel.any():
                out[i] = float(np.sqrt(psd[sel].mean()))
        return centres, out
