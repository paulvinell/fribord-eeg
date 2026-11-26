"""The rings the traces and the spectrum read: where a band's energy sits, how far up the spectrum goes, and the sum of two rings."""

from __future__ import annotations

import pytest

from nfb_console.spectrum_pane import TILE_BANDS

def test_spectrum_bins_are_centred_on_their_frequency():
    """The spectrum takes frq/resol samples so rfft bin k *is* k Hz. Binning [f, f+1) instead and
    drawing the bar from f's gridline rightward puts every feature half a bin late — a mains notch at
    50 Hz then looks like it starts at 50 and continues past it."""
    import numpy as np
    from nfb_console.amplitude_view import SPECTRUM_BIN_HZ, SPECTRUM_LO, RawRing

    for tone in (10.0, 25.0, 50.0):
        ring = RawRing(2.0, 256.0)
        times = np.arange(512) / 256.0
        ring.push(10 * np.sin(2 * np.pi * tone * times))
        values = ring.spectrum(relative=False)
        peak_hz = SPECTRUM_LO + int(np.argmax(values)) * SPECTRUM_BIN_HZ
        assert peak_hz == tone, f"{tone} Hz tone landed in the bin labelled {peak_hz}"


def test_each_band_reports_where_its_energy_sits_not_only_how_much():
    """The dominant frequency per band, which is what a reward frequency gets picked from.

    The instruction for the coupling screen is to put the reward at the frequency in the spectral
    display showing the highest amplitude in a particular band, so the number needed is per band and
    not the peak of the whole spectrum — which is nearly always the slowest thing present and says nothing
    about where alpha is.
    """
    import numpy as np

    from nfb.dsp import ALPHA, THETA
    from nfb_console.amplitude_view import RawRing

    ring = RawRing(2.0, 256.0)
    times = np.arange(512) / 256.0
    # a strong slow component and a weaker alpha one, so the whole-spectrum peak is *not* the alpha answer
    ring.push(40 * np.sin(2 * np.pi * 5.0 * times) + 8 * np.sin(2 * np.pi * 10.0 * times))
    peaks = ring.band_peaks((THETA, ALPHA))
    assert peaks[THETA.label] == pytest.approx(5.0, abs=0.3)
    assert peaks[ALPHA.label] == pytest.approx(10.0, abs=0.3), "found inside its own band, not globally"


def test_a_band_peak_resolves_finer_than_the_bins_it_is_drawn_in():
    """1 Hz bins would report a 10.4 Hz rhythm as 10 and a 10.6 Hz one as 11 — a whole bin of apparent
    movement for a tenth of one in the signal. A parabola through the three points around the maximum
    recovers the position between bins, which is the standard estimator for it."""
    import numpy as np

    from nfb.dsp import ALPHA
    from nfb_console.amplitude_view import RawRing

    for tone in (9.4, 10.0, 10.6, 11.5):
        ring = RawRing(2.0, 256.0)
        times = np.arange(512) / 256.0
        ring.push(20 * np.sin(2 * np.pi * tone * times))
        peak = ring.band_peaks((ALPHA,))[ALPHA.label]
        assert peak == pytest.approx(tone, abs=0.25), f"{tone} Hz reported as {peak}"


def test_a_band_peak_never_lands_outside_its_own_band():
    """The band is the claim being made, so an interpolated peak is clamped to it: a tile headed "alpha"
    reporting 12.3 Hz would be describing something it is not measuring."""
    import numpy as np

    from nfb.dsp import ALPHA
    from nfb_console.amplitude_view import RawRing

    ring = RawRing(2.0, 256.0)
    times = np.arange(512) / 256.0
    # energy piled against the top edge of alpha, where the interpolation has every reason to overshoot
    ring.push(20 * np.sin(2 * np.pi * 11.9 * times) + 20 * np.sin(2 * np.pi * 13.0 * times))
    peak = ring.band_peaks((ALPHA,))[ALPHA.label]
    assert ALPHA.low <= peak <= ALPHA.high, f"alpha's peak reported at {peak} Hz"


def test_the_spectrum_stops_at_nyquist():
    from nfb_console.amplitude_view import RawRing
    assert RawRing(2.0, 62.5).top_hz == 31.25, "a 62.5 sps channel cannot show 60 Hz"
    assert RawRing(2.0, 1000.0).top_hz == 60.0, "and nothing shows more than the display asks for"


def test_a_summed_ring_reads_as_the_ring_of_the_sum():
    """Derived from the channel rings rather than fed, and byte-for-byte the spectrum a fed ring of the
    same sum would give — which is what lets every screen offer it without plumbing."""
    import numpy as np

    from nfb_console.traces import RawRing, SummedRing

    rate = 250.0
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal(500), rng.standard_normal(500)
    ra, rb, fed = (RawRing(2.0, rate, 1.0, 40.0, 1.0) for _ in range(3))
    for i in range(0, 500, 250):
        ra.push(a[i:i + 250]); rb.push(b[i:i + 250]); fed.push(a[i:i + 250] + b[i:i + 250])
    derived = SummedRing([ra, rb])
    assert derived.filled == 500 and derived.top_hz == fed.top_hz
    assert np.allclose(derived.spectrum(), fed.spectrum())
    assert derived.band_peaks(TILE_BANDS) == fed.band_peaks(TILE_BANDS)
