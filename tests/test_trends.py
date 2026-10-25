"""The Trends pane: band history, averages over a chosen span, per-band peak frequencies.

The tests here pin three properties rather than the drawing: one adaptable span governing every number, averages
rather than instantaneous values, and collection that runs whether or not the tab is open.
"""

from __future__ import annotations

import numpy as np
import pytest

from nfb_console.spectrum_pane import SpectrumSource, TILE_BANDS
from nfb_console.traces import RawRing
from nfb_console.trends_pane import UPDATE_SECONDS, BandTrends


def ring_with(alpha_uv: float, rate: float = 250.0, beta_hz: float = 17.0,
              beta_uv: float = 3.0) -> RawRing:
    """Two seconds of a two-rhythm signal: alpha at 10 Hz at the stated size, a smaller 17 Hz beside it."""
    ring = RawRing(seconds=2.0, rate_hz=rate)
    times = np.arange(int(2.0 * rate)) / rate
    ring.push(alpha_uv * np.sin(2 * np.pi * 10.0 * times)
              + beta_uv * np.sin(2 * np.pi * beta_hz * times))
    return ring


def ticked(trends: BandTrends, source: SpectrumSource, count: int, start: float = 0.0) -> float:
    """Advance signal time through `count` samples, at exactly the collector's own cadence."""
    now = start
    for _ in range(count):
        trends.tick([source], now)
        now += UPDATE_SECONDS
    return now


def test_the_averages_answer_over_the_span_and_the_peak_is_a_frequency():
    trends = BandTrends()
    source = SpectrumSource("ch0", ring_with(10.0))
    ticked(trends, source, 10)

    averages = trends.averages("ch0", 0.0, relative=False)
    assert averages["alpha"] > averages["delta"], "the rhythm is where it was put"
    assert averages["alpha"] > averages["beta2"]

    relative = trends.averages("ch0", 0.0, relative=True)
    assert sum(relative.values()) == pytest.approx(100.0, abs=0.5), "shares total the whole"
    assert relative["alpha"] > relative["delta"]

    peaks = trends.peak_averages("ch0", 0.0)
    assert peaks["alpha"] == pytest.approx(10.0, abs=0.4), "the dominant frequency, not the band centre"
    assert peaks["beta1"] == pytest.approx(17.0, abs=0.5)


def test_the_span_selects_the_most_recent_minutes_and_zero_is_the_session():
    """One typed span governs the lines, the
    averages and the peaks — so 'last 2 minutes' after a change reads the change, and 0 reads it all."""
    trends = BandTrends()
    quiet = SpectrumSource("ch0", ring_with(5.0))
    loud = SpectrumSource("ch0", ring_with(15.0))

    now = ticked(trends, quiet, 60)                     # two minutes of quiet alpha...
    ticked(trends, loud, 60, start=now)                 # ...then two minutes of three times as much

    recent = trends.averages("ch0", 120.0, relative=False)["alpha"]
    whole = trends.averages("ch0", 0.0, relative=False)["alpha"]
    early = trends.averages("ch0", 0.0, relative=False)
    assert recent > 1.8 * (2 * whole - recent), "the last two minutes see only the loud half"
    assert whole == pytest.approx((recent + (2 * whole - recent)) / 2), "and 0 averages both halves"
    assert len(trends.window("ch0", 120.0)) == pytest.approx(60, abs=1)
    assert len(trends.window("ch0", 0.0)) == 120
    assert early["alpha"] < recent


def test_relative_shares_survive_an_overall_amplitude_change_that_microvolts_do_not():
    """The reason the % switch exists at all, held to on the averages: gel drying scales everything, and
    a share is the number that stays comparable across it."""
    trends = BandTrends()
    now = ticked(trends, SpectrumSource("ch0", ring_with(5.0, beta_uv=1.5)), 30)
    ticked(trends, SpectrumSource("ch0", ring_with(15.0, beta_uv=4.5)), 30, start=now)

    first = trends.averages("ch0", 0.0, relative=True)["alpha"]
    recent = trends.averages("ch0", 60.0, relative=True)["alpha"]
    assert recent == pytest.approx(first, rel=0.05), "same proportions, three times the microvolts"

    absolute_whole = trends.averages("ch0", 0.0, relative=False)["alpha"]
    absolute_recent = trends.averages("ch0", 60.0, relative=False)["alpha"]
    assert absolute_recent > 2 * absolute_whole - absolute_recent, "while the µV number moved"


def test_the_peak_frequency_is_a_trend_you_can_see_not_only_a_number():
    """Half of the pane's reason to exist: "did the alpha peak drift" is a question about a line. The
    average alone answers it the way a mean temperature answers whether it rained."""
    trends = BandTrends()
    rate = 250.0

    def alpha_at(hz: float) -> SpectrumSource:
        ring = RawRing(seconds=2.0, rate_hz=rate)
        times = np.arange(int(2.0 * rate)) / rate
        ring.push(10.0 * np.sin(2 * np.pi * hz * times))
        return SpectrumSource("ch0", ring)

    now = ticked(trends, alpha_at(9.0), 30)             # a minute at 9 Hz...
    ticked(trends, alpha_at(11.0), 30, start=now)       # ...then a minute at 11

    line = trends.peak_series("ch0", 0.0)["alpha"]
    assert len(line) == 60
    early = [hz for _t, hz in line[:25]]
    late = [hz for _t, hz in line[-25:]]
    assert np.mean(early) == pytest.approx(9.0, abs=0.4)
    assert np.mean(late) == pytest.approx(11.0, abs=0.4), "the drift is in the line itself"

    # And the same span control governs it: the last minute averages near 11, the whole session between.
    assert trends.peak_averages("ch0", 60.0)["alpha"] == pytest.approx(11.0, abs=0.4)
    assert 9.5 < trends.peak_averages("ch0", 0.0)["alpha"] < 10.5


def test_collection_is_rate_limited_and_needs_a_second_of_signal():
    trends = BandTrends()
    source = SpectrumSource("ch0", ring_with(10.0))
    trends.tick([source], 0.0)
    trends.tick([source], 0.5)                          # inside the cadence: not a second sample
    assert len(trends.window("ch0", 0.0)) == 1

    empty = RawRing(seconds=2.0, rate_hz=250.0)
    starved = BandTrends()
    starved.tick([SpectrumSource("ch0", empty)], 0.0)
    assert starved.window("ch0", 0.0) == [], "no spectral window yet means no sample, not a zero"


def test_the_pane_collects_whether_or_not_its_tab_was_ever_opened():
    """The session average must not depend on when the operator first clicked Trends, so the sampling is
    driven from the screen's draw loop, not from the pane being visible."""
    from test_screens import StubPage, replay

    from nfb_console import amplitude_view as av
    from nfb_console import trends_pane
    from nfb_console.protocols import BY_CODE

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=1)
    try:
        for item in session.source.blocks():
            session.feed_block(item)
        pane = trends_pane.TrendsPane(StubPage(), session, lambda reserve=0: 600.0)
        assert not pane.control.visible, "built hidden, like every non-default tab"
        pane.tick(session.elapsed if session.elapsed > 0 else 10.0)
        assert pane.trends.labels(), "the history exists before the tab is ever shown"
        pane.refresh()
        assert pane.caption.value == "", "a hidden pane paints nothing — that cost starved clicks"
        pane.control.visible = True             # the tab is opened; the stored history draws at once
        pane.refresh()
        shown = [pane.tile_values[b.label].value for b in pane.trends.bands]
        assert any(v not in ("", "—") for v in shown), "and the tiles carry numbers from it"
        assert "µV" not in pane.caption.value or "%" not in pane.caption.value

        # The third pill: the graph redraws as each band's dominant frequency, on its own Hz axis.
        pane.tick(1000.0)                    # a second sample, so the peak lines have two points
        pane._show_peaks()
        pane.refresh()
        assert pane.graph_shows == "peak"
        assert "Hz ·" in pane.caption.value and "energy sits" in pane.caption.value
        pane._set_relative(False)
        assert pane.graph_shows == "amplitude", "choosing a unit returns the graph to amplitudes"
    finally:
        session.stop()


def test_the_history_survives_the_view_being_rebuilt_around_it():
    """Going to the device screen to check contact and coming back rebuilds every control against the
    same session — deliberately, and `Console.resume` says so. The band history was on the pane, so the
    trip emptied it: an hour of session, then a look at an electrode, then a graph starting from zero.
    """
    from test_screens import StubPage, replay, toolbar

    from nfb_console import amplitude_view as av
    from nfb_console.protocols import BY_CODE

    page = StubPage()
    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        av.view(page, session, toolbar(), lambda: None)
        collector = session.band_trends
        assert collector is not None, "the session holds it, so a fresh view can find it again"
        source = SpectrumSource("ch0", ring_with(10.0))
        ticked(collector, source, 30)
        before = len(collector.window("ch0", 0.0))
        assert before == 30

        av.view(page, session, toolbar(), lambda: None)      # what coming back does
        assert session.band_trends is collector
        assert len(session.band_trends.window("ch0", 0.0)) == before, \
            "half an hour of trend, thrown away by a look at the electrodes"
    finally:
        session.stop()


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
