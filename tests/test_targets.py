"""Thresholds: the two modes, and the calibration that a fixed one needs.

Calibration is the part that is easy to get subtly wrong, because it is a *phase* rather than a mode:
a fixed threshold has to come from somewhere, so a session that opens with one spends its first 30 s
deriving it. Getting that wrong shows up as a threshold that never settles, or one that a hand
adjustment silently loses.
"""

from __future__ import annotations

import numpy as np
import pytest

from nfb.dsp import ALPHA
from nfb_console.feedback import FeedbackSettings
from nfb_console.targets import (CALIBRATION_SECONDS, DEFAULT_MODES, HIT_RATES, MODES,
                                 RECENT_WINDOW, Target)

RATE = 20.0        # the metric output rate


def feed(tgt: Target, seconds: float, mean: float = 10.0, sd: float = 2.0, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(int(seconds * RATE)):
        tgt.observe(mean + rng.normal(0, sd), RATE)


def test_there_are_two_modes_and_baseline_is_not_one_of_them():
    """A 30 s baseline is what happens when a fixed threshold starts, not a third thing to pick."""
    assert MODES == ("fixed", "automatic")


def test_a_fixed_threshold_calibrates_for_thirty_seconds_then_settles():
    tgt = Target(0, ALPHA, "increase", mode="fixed")
    assert tgt.calibrating
    assert tgt.calibration_remaining == pytest.approx(CALIBRATION_SECONDS)

    feed(tgt, 10.0)
    assert tgt.calibrating, "still collecting"
    assert tgt.calibration_remaining == pytest.approx(CALIBRATION_SECONDS - 10, abs=0.1)

    feed(tgt, 21.0)
    assert not tgt.calibrating, "30 s in, the threshold is derived and the label goes away"
    assert tgt.calibration_remaining == 0.0
    assert 6.0 < tgt.threshold < 12.0, "somewhere in the distribution it just saw"


def test_calibration_gives_a_usable_threshold_from_the_first_second():
    """The whole point of the phase: it feeds back *while* it settles rather than after.

    A fixed target used to derive nothing until the 30 s were up, so it spent that time thresholded at
    whatever it was constructed with — the class default of 10 µV for anything that did not pass one. On a
    quantity whose real amplitude is nowhere near that, the reward simply never fired, and the operator's
    first half-minute of every session was a screen that could not be told apart from a broken one.
    """
    tgt = Target(0, ALPHA, "increase", mode="fixed")
    assert tgt.threshold == 10.0, "the constructed placeholder, which is nothing to do with this signal"

    feed(tgt, 2.0, mean=100.0, sd=10.0)
    assert tgt.calibrating, "still a phase in progress — the label and the countdown stay"
    assert tgt.threshold > 40.0, "and yet a threshold somewhere near the signal, not the placeholder"
    early = tgt.threshold

    # It tracks what arrives, rather than latching the first thing it saw.
    feed(tgt, 10.0, mean=400.0, sd=10.0, seed=2)
    assert tgt.threshold > early, "provisional means it keeps up as the window fills"
    assert tgt.calibrating, "and it is still provisional: nothing has settled yet"

    feed(tgt, 20.0, mean=400.0, sd=10.0, seed=3)
    assert not tgt.calibrating, "30 s in, the same derivation as before lands and the label goes"


def test_a_provisional_threshold_never_counts_as_settled():
    """`_settled` is what the rest of the app reads, and a moving number must not claim to be final.

    The countdown, the card's note, and `nudge`'s promise that a hand-set number survives a calibration
    finishing are all keyed off this flag, so a provisional derivation setting it early would silently
    convert "still working it out" into "this is your threshold".
    """
    tgt = Target(0, ALPHA, "increase", mode="fixed")
    feed(tgt, 5.0, mean=50.0)
    assert not tgt._settled and tgt.calibrating
    assert tgt.calibration_remaining == pytest.approx(CALIBRATION_SECONDS - 5, abs=0.2)

    by_hand = 123.0
    tgt.threshold = by_hand
    tgt.nudge(1.0)
    feed(tgt, 40.0, mean=50.0, seed=4)
    assert tgt.threshold == by_hand, "a hand-set number outranks a provisional one and a final one"


def test_a_settled_fixed_threshold_stops_moving():
    tgt = Target(0, ALPHA, "increase", mode="fixed")
    feed(tgt, 31.0, mean=10.0)
    settled = tgt.threshold
    feed(tgt, 60.0, mean=40.0, seed=1)      # amplitude quadruples
    assert tgt.threshold == settled, "fixed means fixed; that is the point of the mode"


def test_automatic_keeps_tracking_and_needs_no_calibration():
    tgt = Target(0, ALPHA, "increase", mode="automatic")
    assert not tgt.calibrating
    feed(tgt, 40.0, mean=10.0)
    low = tgt.threshold
    feed(tgt, 90.0, mean=40.0, seed=2)
    assert tgt.threshold > low * 2, "the tracker follows the signal up"


def test_redefine_is_immediate_when_the_history_is_there():
    tgt = Target(0, ALPHA, "increase", mode="automatic")
    feed(tgt, 40.0, mean=10.0)
    feed(tgt, 40.0, mean=40.0, seed=3)
    tgt.redefine()
    assert tgt.mode == "fixed", "redefining is how you get a fixed threshold from what just happened"
    assert not tgt.calibrating, "no waiting: the last 30 s already exist"
    assert 30.0 < tgt.threshold < 50.0, "from the recent loud stretch, not the quiet one"


def test_redefine_with_no_history_falls_back_to_calibrating():
    tgt = Target(0, ALPHA, "increase", mode="automatic")
    tgt.redefine()
    assert tgt.mode == "fixed" and tgt.calibrating


def test_redefine_waits_for_the_estimator_the_way_the_opening_calibration_does():
    """**A slow estimator's ramp is not a signal, and redefine was deriving from it.**

    It used to accept any twenty readings — one second at the metric rate — and mark the result final. The
    opening calibration is careful about exactly this: `settle_seconds` says how long this target's
    estimator takes to mean anything, and the calibration runs that long *before* its window. Measured on a
    0.1 Hz synchrony reward, pressing redefine two seconds after retuning to it set the threshold from the
    ramp — 19 µV where the settled signal ran 110–228 — and CFC's thresholds are fixed, so that stood for
    the session.

    Now the immediate path asks the same question, and anything short of it falls through to calibrating,
    which counts down on the card and re-derives as the data arrives.
    """
    tgt = Target(0, ALPHA, "increase", mode="automatic", settle_seconds=15.0)

    feed(tgt, 20.0, mean=10.0)              # past a full window, but not past settle + a window
    placeholder = tgt.threshold
    tgt.redefine()
    assert tgt.calibrating, "the estimator has not been running long enough to be believed"
    assert tgt.threshold == placeholder, "and nothing was frozen from the ramp"

    feed(tgt, 30.0, mean=10.0, seed=5)      # now 50 s in: 15 s of settling and a full window behind it
    tgt.redefine()
    assert not tgt.calibrating and 6.0 < tgt.threshold < 14.0

    # A target with no settling to wait for is unchanged: this costs the fast bands nothing.
    fast = Target(0, ALPHA, "increase", mode="automatic")
    feed(fast, 31.0, mean=10.0)
    fast.redefine()
    assert not fast.calibrating


def test_redefine_is_not_offered_for_automatic():
    """It would be a no-op with a misleading label: automatic is already recomputing continuously."""
    assert Target(0, ALPHA, "increase", mode="fixed").can_redefine
    assert not Target(0, ALPHA, "increase", mode="automatic").can_redefine


def test_a_hand_set_threshold_survives_everything():
    tgt = Target(0, ALPHA, "increase", mode="automatic")
    feed(tgt, 40.0, mean=10.0)
    tgt.nudge(1.5)
    by_hand = tgt.threshold
    assert tgt.mode == "fixed" and not tgt.calibrating
    feed(tgt, 60.0, mean=40.0, seed=4)
    assert tgt.threshold == by_hand, "neither the tracker nor a calibration may overwrite it"


def test_switching_to_fixed_derives_a_threshold_from_what_just_happened():
    tgt = Target(0, ALPHA, "increase", mode="automatic")
    feed(tgt, 40.0, mean=25.0)
    tgt.set_mode("fixed")
    assert not tgt.calibrating and 15.0 < tgt.threshold < 35.0


def test_the_threshold_lands_where_the_role_hit_rate_says_it_should():
    """A reward and an inhibit are aimed at different rates, and neither is 50%.

    **The aim comes from the role and the tail from the direction**, so the four combinations are four
    real arrangements: a reward is earned 90% of the time whether it is trained up or down, and an
    inhibit is kept clear 75% of the time either way."""
    for role, rate in HIT_RATES.items():
        for direction in ("increase", "decrease"):
            tgt = Target(0, ALPHA, direction, role=role, mode="fixed")
            feed(tgt, 60.0, mean=10.0, sd=3.0, seed=5)
            tgt.hits = tgt.total = 0
            feed(tgt, 60.0, mean=10.0, sd=3.0, seed=6)
            assert abs(tgt.percent - rate * 100) < 12, \
                f"{role} {direction} aimed at {rate:.0%}, got {tgt.percent:.0f}%"


def test_rewards_and_inhibits_open_in_different_modes_and_at_different_rates():
    """A reward tracks, so it keeps rewarding as amplitude drifts; an inhibit holds still, so crossing
    the line means the same thing at minute 20 as at minute 1."""
    # The mode follows **which side of the line the target is met on** — a tracker follows the trainee
    # down and pins the percentage where it started, which erases the change the training exists to
    # produce. The aim follows the **role**.
    assert DEFAULT_MODES == {True: "automatic", False: "fixed"}
    assert Target(0, ALPHA).mode == "automatic", "a reward on the way up tracks"
    assert Target(0, ALPHA, role="inhibit").mode == "fixed", "and an ordinary inhibit holds still"
    assert Target(0, ALPHA).hit_rate == 0.90
    assert Target(0, ALPHA, role="inhibit").hit_rate == 0.75
    # An alpha-down reward is held below a line like an inhibit, so it opens fixed — at a reward's aim.
    down_reward = Target(0, ALPHA, "decrease", role="reward")
    assert (down_reward.mode, down_reward.hit_rate) == ("fixed", 0.90)
    assert Target(0, ALPHA, "increase", mode="fixed").mode == "fixed", "still overridable"


def test_on_target_is_reported_over_the_recent_window_not_the_session():
    tgt = Target(0, ALPHA, "increase", mode="fixed", threshold=10.0)
    assert tgt.recent_percent() is None, "no claim until there is a window's worth"
    feed(tgt, RECENT_WINDOW - 1, mean=30.0)
    assert tgt.recent_percent() is None

    feed(tgt, 2.0, mean=30.0)
    # 90%, not 100%, and that is the point of `rescore`. Calibration finished during this window and moved
    # the threshold from the initial 10 to ~28.7, aimed at being met 90% of the time. The hit record is
    # re-judged against the threshold **actually in force**, so the number agrees with the line drawn on the
    # card. It used to read 100% because the record was kept against the threshold that had been replaced —
    # a percentage of a comparison nobody was making any more.
    assert tgt.recent_percent() == pytest.approx(tgt.hit_rate * 100, abs=3.0)
    assert tgt.threshold > 25.0, "the sanity check on that: calibration did move it"

    feed(tgt, RECENT_WINDOW, mean=0.0, sd=0.1)          # collapses
    assert tgt.recent_percent() == pytest.approx(0.0), "the recent window forgets the good stretch"
    assert tgt.percent > 20, "the session figure still remembers it, which is why it is not the headline"


def test_moving_a_threshold_does_not_blank_the_card():
    """Nudging a threshold changes the *comparison*, not the measurement, so every value already collected
    can be re-judged against the new number exactly. Clearing the history instead made the card read
    "waiting" for a full thirty seconds after every nudge — and cleared the session counters too, so a hand
    adjustment twenty minutes in threw the whole session's summary away."""
    tgt = Target(0, ALPHA, "increase", mode="fixed", threshold=10.0)
    tgt._settled = True                             # already calibrated: this test is about the nudge alone
    feed(tgt, RECENT_WINDOW + 5, mean=12.0, sd=0.5)
    before, total_before = tgt.recent_percent(), tgt.total
    assert before is not None and before > 95.0

    tgt.nudge(2.0)                                  # threshold to ~24: nothing collected now qualifies
    assert tgt.recent_percent() is not None, "the card must not go blank"
    assert tgt.recent_percent() == pytest.approx(0.0), "and it must answer for the new threshold"
    assert tgt.total == total_before, "a nudge is not a new session"

    tgt.nudge(0.5)                                  # back to ~12: it qualifies again, immediately
    assert tgt.recent_percent() > 95.0

    # a band edit keeps the history too — the commonest one is a 0.1 Hz nudge while hunting a frequency
    tgt.shift_band(0.1)
    assert tgt.recent_percent() is not None, "0.1 Hz is not a new session"

    # a *source* change is the genuine restart: the values came from a different electrode
    tgt.reset_stats()
    assert tgt.recent_percent() is None and tgt.total == 0


def test_a_band_takes_its_colour_from_its_centre_frequency():
    """The point is that one band is one colour everywhere."""
    from nfb.dsp import Band
    alpha = Target(0, Band(8, 12, ""), "increase")
    beta = Target(0, Band(20, 30, ""), "decrease")
    assert alpha.colour != beta.colour
    assert Target(0, Band(8, 12, ""), "decrease").colour == alpha.colour, \
        "the direction must not change the colour; the band is what is being named"


def test_the_moving_average_smooths_and_keeps_its_length():
    tgt = Target(0, ALPHA, "increase", mode="fixed")
    tgt.trend.extend([0.0, 10.0] * 30)
    average = tgt.moving_average(10)
    assert len(average) == len(tgt.trend), "one point per plotted point, or the lines misalign"
    assert average[0] == 0.0, "the first point averages only itself"
    assert 4.0 < average[-1] < 6.0, "settles on the mean of the alternating series"
    assert max(average[5:]) - min(average[5:]) < 3.0, "much flatter than the 0–10 it came from"


def test_an_empty_trend_has_no_moving_average():
    assert Target(0, ALPHA, "increase").moving_average(10) == []


def test_asking_for_the_drawn_window_gives_the_same_numbers_as_averaging_everything():
    """**The amplitude graph got heavier the longer a session ran**, and this is the half of the fix that
    could have changed an answer. It averaged the whole trend and the caller kept a tenth of it — 446 µs per
    target per frame at a full `TREND_POINTS`, in a Python loop, for points nobody sees.

    Taking the tail first is only sound if the lead-in comes with it: the first point returned still needs
    its whole window behind it. So the test is exactness against the old behaviour rather than closeness,
    across window lengths shorter, equal and longer than the trend itself.
    """
    rng = np.random.default_rng(0)
    for total in (5, 60, 300, 1200):
        tgt = Target(0, ALPHA, "increase", mode="fixed")
        tgt.trend.extend(rng.random(total))
        whole = tgt.moving_average(60)
        for span in (1, 10, 60, 100, 1200):
            assert tgt.moving_average(60, span=span) == pytest.approx(whole[-span:]), (total, span)


def test_a_target_is_observed_once_per_reading_of_its_own_metric():
    """The rate a target is told has to be the rate it gets, or every window is wrong by that factor.

    Observing once per arriving block ran at blocks × channels per second — 128 Hz on a 256 sps
    four-channel file against the 20 Hz declared — so a "30 s" calibration completed in five and
    "last 30 s on target" covered five.
    """
    from nfb.dsp import Band
    from nfb_console.amplitude_view import METRIC_HZ, AmplitudeSession
    from nfb_console.protocols import BY_CODE
    from nfb_console.targets import PAIR

    session = AmplitudeSession.__new__(AmplitudeSession)
    session.protocol = BY_CODE["AMP"]
    session.specs = {0: object(), 1: object(), 2: object()}
    session.feedback = FeedbackSettings()      # the envelope constant is the session's, so `filter_for` needs it

    on_ch0 = Target(0, Band(8, 12, ""), "increase")
    on_ch1 = Target(1, Band(8, 12, ""), "increase")
    on_sum = Target(PAIR, Band(8, 12, ""), "increase", pair=(0, 1, 2))
    down = Target(0, Band(8, 12, ""), "decrease", filter_kind="elliptic", filter_order=8)
    # the metric key carries the filter, so the same band measured two ways is two estimators
    up_key = ("8-12", session.filter_for(on_ch0))
    down_key = ("8-12", session.filter_for(down))
    assert up_key != down_key

    assert session._due(on_ch0, {(0, *up_key)})
    assert not session._due(on_ch0, {(1, *up_key)}), "another channel's reading is not this target's"
    assert not session._due(on_ch0, {(0, "4-8", up_key[1])}), "another band's reading is not either"
    assert not session._due(on_ch0, {(0, *down_key)}), "nor the same band through a different filter"
    assert session._due(down, {(0, *down_key)})
    assert session._due(on_ch1, {(1, *up_key)})

    # A summed target has **one** estimator, on a key that is no channel's, so it advances on that and on
    # nothing else. A contributing channel's own reading is an input to the sum rather than a reading of it,
    # and counting those would advance the target once per electrode — every window it has too short by that
    # factor, which is the bug `_due` exists for.
    from nfb_console.amplitude_view import SUM_KEY
    assert session._due(on_sum, {(SUM_KEY, *up_key)})
    assert not session._due(on_sum, {(0, *up_key)})

    assert METRIC_HZ == 20.0


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


def test_a_number_field_clamps_and_survives_its_own_rebuild():
    """Two failures this covers. Typing a value past a bound used to be rejected outright, and the
    field's trailing `update()` raised out of the blur handler once `on_set` had rebuilt the panel the
    field lived in — "Control must be added to the page first"."""
    from nfb_console.amplitude_view import num_box

    seen: list[float] = []
    # This field is never added to a page, which is exactly the state a rebuilt panel leaves the old
    # field in — so every commit below exercises the guard that used to raise from `field.update()`.
    row = num_box(5.0, seen.append, step=0.1, low=1.0, high=10.0)
    field, stepper = row.controls[0], row.controls[1]
    up, down = stepper.controls

    assert field.value == "5.0", "formatted to the step's precision"
    up.on_click(None)
    assert seen[-1] == pytest.approx(5.1)
    up.on_click(None)
    assert seen[-1] == pytest.approx(5.2), "the arrows step from what is shown, not from the initial"
    down.on_click(None)
    assert seen[-1] == pytest.approx(5.1)

    field.value = "99"
    field.on_blur(None)
    assert seen[-1] == 10.0, "clamped to the bound rather than refused"
    field.value = "-4"
    field.on_blur(None)
    assert seen[-1] == 1.0

    assert field.value == "1.0", "the box shows the value that was accepted, not what was typed"

    field.value = "not a number"
    field.on_blur(None)
    assert seen[-1] == 1.0, "unparseable input snaps back to the last accepted value"
    assert field.value == "1.0"


# --- where a percentile stops being estimable -------------------------------------------------

def test_the_derivation_follows_the_bandwidth_not_the_protocol():
    """A percentile is a tail estimate and needs values in that tail. A band's envelope cannot change
    faster than its own bandwidth, so 30 s of a 4 Hz band holds ~120 independent values and 30 s of a
    0.2 Hz band holds ~6 — and at six, the sample minimum is already the 17th percentile, so a 2.5th
    percentile is an extrapolation rather than an estimate.

    The switch is statistical, which is why the amplitude screen is untouched by it.
    """
    from nfb.dsp import Band
    from nfb_console.targets import Target

    alpha_up = Target(0, Band(8, 12, ""), "increase")
    slow_down = Target(0, Band(2, 6, ""), "decrease")
    assert alpha_up.derivation(30) == "percentile"
    assert slow_down.derivation(30) == "percentile"
    assert "percentile" in alpha_up.describe_derivation(30)

    reward = Target(0, Band(4.9, 5.1, ""), "increase", hit_rate_override=0.975)
    assert reward.independent_samples(30) == pytest.approx(6.0)
    assert reward.derivation(30) == "median"
    assert "too few" in reward.describe_derivation(30)

    # the same target with a longer window crosses back over on its own
    assert reward.derivation(30) == "median"
    assert reward.independent_samples(3000) == pytest.approx(600.0)
    assert reward.derivation(3000) == "percentile"


def test_a_measured_spread_beats_an_assumed_one_where_the_shape_does_not_fit():
    """**The Rayleigh's one parameter is an assumption, and sub-hertz readings break it.**

    That shape says how spread out the readings are as soon as you know their middle: its 5th percentile
    sits at 0.27 of its median, always. Measured on a real 0.1 Hz synchrony reward, the readings ran
    110–228 µV around a median of 175 — a 5th percentile at 0.70 of the median, a third as spread out as
    the shape predicts, because down there the reading is a drifting level and not a fluctuating envelope.
    So the threshold landed at a fifth of the median, below anything the signal reached, and the reward was
    met 100% of the session.

    Where a screen declares the shape inapplicable, the spread is measured from the body of the data
    instead. The old fallback there was a percentile — the estimator already established as unusable on a
    thin window, so the worst case had the worst tool.
    """
    import numpy as np

    from nfb.dsp import Band
    from nfb_console.targets import SPREAD_SAMPLES, Target

    def threshold_for(narrowband: bool, samples) -> float:
        tgt = Target(0, Band(0.075, 0.125, ""), "increase", hit_rate_override=0.975,
                     narrowband_envelope=narrowband, calibration_seconds=240.0)
        tgt._apply_threshold(list(samples), 240.0)
        return tgt.threshold

    # A drifting level: log-normal, a fifth as spread out in the log as a Rayleigh is.
    rng = np.random.default_rng(7)
    drifting = np.exp(rng.normal(np.log(175.0), 0.2, 4000))
    truth = float(np.percentile(drifting, 2.5))

    assumed = threshold_for(True, drifting)
    measured = threshold_for(False, drifting)
    assert measured == pytest.approx(truth, rel=0.1), "the aim, from the data's own spread"
    assert assumed < truth / 2, f"the assumed shape puts it far below anything: {assumed:g} vs {truth:g}"

    # And where the shape *does* fit, nothing changed: that route is still the one-parameter fit.
    fast = Target(0, Band(4.9, 5.1, ""), "increase", hit_rate_override=0.975)
    assert fast.derivation(30) == "median"
    assert Target(0, Band(0.075, 0.125, ""), "increase", hit_rate_override=0.975,
                  narrowband_envelope=False, calibration_seconds=240.0).derivation(240) == "spread"

    # Below a handful of independent values there is no spread to measure and it falls back — too strict
    # rather than too loose, which is the right way round.
    thin = Target(0, Band(0.075, 0.125, ""), "increase", hit_rate_override=0.975,
                  narrowband_envelope=False)
    assert thin.independent_samples(30) < SPREAD_SAMPLES
    assert thin.derivation(30) == "percentile"


def test_a_sustained_rhythm_is_not_a_rayleigh_and_redefine_must_not_assume_it_is():
    """Reported from a session: `redefine` on a 11.7–14.7 Hz reward with a 90% aim left the card reading
    **100% of the last 30 s**, immediately and every time.

    The arithmetic behind it: 3 Hz over 30 s is about 90 independent values, and 90 × 10% is 9 expected
    values in the tail — one short of what a percentile is trusted with — so the derivation fell through to
    the one-parameter Rayleigh fit. A Rayleigh is the envelope of narrowband *noise*, and a reward band
    holding a sustained rhythm is a steady component plus noise, which is much less spread out: the fit
    puts the threshold at 0.39 of the median where the signal's own tenth percentile is at 0.70, so
    everything clears it.

    The fix is an ordering rather than a new estimator — with ninety independent values the spread is
    something to *measure*, and only below `MEASURE_SPREAD_OVER_SHAPE` is assuming a shape worth more than
    measuring one. Simulated at the top of that constant; here it is the reported case end to end.
    """
    import numpy as np

    from nfb.dsp import Band
    from nfb_console.targets import Target

    rng = np.random.default_rng(5)
    # A steady 12 µV rhythm plus noise — Rician, which is what a band being trained looks like. Its
    # log-spread is a third of a Rayleigh's, which is the property the shape fit gets wrong.
    steady = np.hypot(rng.normal(12.0, 2.0, 20000), rng.normal(0.0, 2.0, 20000))

    reward = Target(0, Band(11.7, 14.7, ""), "increase", hit_rate_override=0.9)
    assert reward.derivation(30) == "spread", "ninety independent values is enough to measure"
    reward._apply_threshold(list(steady[:600]), 30.0)
    achieved = float((steady[600:] >= reward.threshold).mean())
    assert 0.85 < achieved < 0.94, f"aimed at 90%, delivered {achieved:.0%}"

    assumed = Target(0, Band(11.7, 14.7, ""), "increase", hit_rate_override=0.9)
    assumed.narrowband_envelope = True
    assumed.threshold = 0.0
    assumed._apply_threshold(list(steady[:600]), 30.0)      # via "spread" now; the shape fit is below
    scale = float(np.median(steady[:600])) / 1.1774100225154747
    rayleigh = scale * float(np.sqrt(-2.0 * np.log(0.9)))
    assert float((steady[600:] >= rayleigh).mean()) > 0.99, "which is what was on the screen"

    # A band too slow to measure a spread from still gets the shape, which is what it is for.
    slow = Target(0, Band(0.4, 0.6, ""), "increase", hit_rate_override=0.9)
    assert slow.independent_samples(30) < 20
    assert slow.derivation(30) == "median"


def test_the_median_fit_hits_the_aim_where_a_percentile_cannot():
    """Measured, not asserted: at six independent values a percentile aiming at 97.5% delivered anywhere
    from 59% to 98%, and a one-parameter Rayleigh fit delivered 94.6-99.1%."""
    import numpy as np

    from nfb.dsp import Band
    from nfb_console.targets import Target

    rng = np.random.default_rng(11)
    fresh = rng.rayleigh(1.0, 100000)
    achieved = []
    for _ in range(300):
        tgt = Target(0, Band(4.9, 5.1, ""), "increase", hit_rate_override=0.975)
        tgt._apply_threshold(list(rng.rayleigh(1.0, 6)), 30.0)
        achieved.append((fresh >= tgt.threshold).mean())
    lo, hi = np.percentile(achieved, [5, 95])
    assert 0.90 < lo and hi < 1.0, (lo, hi)
    assert hi - lo < 0.12, "a percentile on the same data spans about 0.39"


def test_a_target_in_the_body_of_the_distribution_is_unchanged():
    """The two routes are equivalent where the percentile already works, so nothing about the amplitude
    screen's thresholds moves."""
    import numpy as np

    from nfb.dsp import Band
    from nfb_console.targets import Target

    rng = np.random.default_rng(3)
    samples = list(rng.rayleigh(1.0, 400))
    tgt = Target(0, Band(8, 12, ""), "increase")
    tgt._apply_threshold(samples, 30.0)
    assert tgt.threshold == pytest.approx(float(np.percentile(samples, 10)), abs=0.01)


def test_the_signal_floor_reaches_the_percentage_as_well_as_the_verdict():
    """**`wanted` applied the floor and `rescore` did not**, so a target with a signal floor answered "am I on
    target" one way and "what fraction of the last 30 s" another.

    Measured on ILF Synchrony against the no-electrode bench recording: both channels rail to a constant, a
    bandpass of a constant is zero, the product is zero, and a threshold derived from 30 s of zeros is `0.0`.
    `wanted` correctly said no because of the floor; `recent_percent` said **100%**, because `rescore`
    compared `0 >= 0`. One rule now, `Target.meets`, called from both.
    """
    from nfb_console.targets import Target
    from nfb.dsp import Band

    tgt = Target(0, Band(8.0, 12.0, ""), "increase", threshold=0.0, mode="fixed",
                 signal_floor=0.5)
    for _ in range(100):
        tgt.observe(0.0, 20.0)
    assert not tgt.wanted, "zero is under the floor"
    tgt.rescore()
    assert sum(tgt._hits) == 0, "and the hit record has to agree with the verdict"
    assert tgt.recent_percent(1.0) == 0.0
    assert not tgt.meets(0.4) and tgt.meets(0.6)

    # The floor does NOT bite on an inhibit: below it is the most inhibited state there is, and
    # failing it made alpha-theta's reward tone cut out the moment high beta went properly quiet —
    # the inhibits gate the reward, so the floor inverted the protocol at its destination. The dead
    # channel an inhibit might otherwise reward trivially is `target_is_live`'s case, not this one's.
    down = Target(0, Band(8.0, 12.0, ""), role="inhibit", threshold=10.0, signal_floor=0.5)
    assert down.meets(0.0), "quieter than the floor is still (especially) inhibited"
    assert down.meets(1.0)
    assert not down.meets(11.0)


def test_role_and_direction_are_two_axes_and_not_one():
    """**Direction names the movement; the role says what is done about it.**

    For a long time `direction` answered both "which way is this trained" and "is this a reward or an
    inhibit", so a band trained downward could only ever be an inhibit — and an arrangement whose
    *subject* is trained down had its subject in the group that shares 20% of the level with the artifact
    bands. Six combinations exist now, and the four that matter are the four sentences below.
    """
    from nfb_console.targets import ROLES

    assert ROLES == ("reward", "inhibit", "monitor")
    assert (Target(0, ALPHA).direction, Target(0, ALPHA).role) == ("increase", "reward")

    def met_at(role, direction, value):
        tgt = Target(0, ALPHA, direction, role=role, threshold=10.0)
        tgt.value = value
        return tgt.wanted

    # reward increase alpha  → earned as alpha rises   → expect alpha to rise
    assert met_at("reward", "increase", 15.0) and not met_at("reward", "increase", 5.0)
    # reward decrease alpha  → earned as alpha falls   → expect alpha to fall
    assert met_at("reward", "decrease", 5.0) and not met_at("reward", "decrease", 15.0)
    # inhibit increase beta  → rises are suppressed    → expect beta to fall
    assert met_at("inhibit", "increase", 5.0) and not met_at("inhibit", "increase", 15.0)
    # inhibit decrease beta  → falls are suppressed    → expect beta to rise
    assert met_at("inhibit", "decrease", 15.0) and not met_at("inhibit", "decrease", 5.0)

    # `wants_above` is the one place the two are resolved, and nothing else may ask.
    assert Target(0, ALPHA, "increase", role="inhibit").wants_above is False
    assert Target(0, ALPHA, "decrease", role="reward").wants_above is False
    assert Target(0, ALPHA, "decrease", role="inhibit").wants_above is True

    # The score has to agree with the verdict, or the level and the click disagree about the same target.
    beta = Target(0, ALPHA, "increase", role="inhibit", mode="fixed", threshold=10.0)
    beta.nudge(1.0)
    beta.value = 5.0
    assert beta.score() > 0.5, "quiet beta is a good score for an inhibit on rises"
    beta.value = 20.0
    assert beta.score() < 0.5

    # A monitor is a role, and it keeps everything a role does not decide.
    watched = Target(0, ALPHA, "decrease", role="monitor")
    assert watched.monitor and watched.verb == "DECREASE"
    assert watched.hit_rate and watched.mode and watched.threshold


def test_a_down_trained_reward_still_cannot_earn_on_a_dead_channel():
    """**The hazard the split creates**, and it is the old one wearing the other sign.

    The signal floor exists to stop a reward calibrating onto noise and then paying out on it all
    session. It used to apply to `increase` targets, which was the same set as the rewards; an inhibit is
    exempt because below the floor is the most inhibited state there is. Read off the direction, an
    alpha-down reward would have been exempted silently — and a dead electrode reads near zero, satisfies
    it perfectly, and earns for twenty minutes.
    """
    down = Target(0, ALPHA, "decrease", role="reward", threshold=10.0, signal_floor=1.0)
    down.value = 5.0
    assert down.wanted, "quiet is what this protocol is asking for"
    down.value = 0.2
    assert not down.wanted, "and silence is a dead electrode, not a perfect result"

    inhibit = Target(0, ALPHA, "increase", role="inhibit", threshold=10.0, signal_floor=1.0)
    inhibit.value = 0.2
    assert inhibit.wanted, "an inhibit is still exempt — below the floor is the most inhibited state"
