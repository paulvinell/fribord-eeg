"""Where the HRV level comes from: a metric, and a scaling.

Two layers: a metric and a scaling. A third — a set of "channels" rendering the result — was built
and deleted, because it reimplemented the mechanisms this codebase already shares across every protocol: a
second `DisplayShadow`, a noise mask that is `music` on noise plus `sound.invert`, and no master volume at all.
The mechanisms are the general, calibrated, persisted machinery; what was missing was the layer *underneath*
them, which on an EEG screen is a threshold and here cannot be.
"""

from __future__ import annotations

import importlib
import time

import numpy as np
import pytest

from nfb.hrv import Beat, DEFAULT_METRIC, HRV_METRICS, HRVProcessor, metric_value
from nfb_console import normalise


def coherent_state(seconds: float = 200.0, depth: float = 0.09, jitter: float = 0.0):
    """A `BeatState` from a tachogram modulated at 0.1 Hz — what this protocol is training toward."""
    proc = HRVProcessor()
    rng = np.random.default_rng(5)
    t, state = 1000.0, None
    while t < 1000.0 + seconds:
        rr = 1.0 + depth * np.sin(2 * np.pi * 0.1 * t) + jitter * rng.standard_normal()
        t += rr
        state = proc.push(Beat(t, 60.0 / rr, [rr]))
    return state


# --- layer one: what to feed back ------------------------------------------------------------

def test_every_metric_extracts_a_real_number_from_a_real_state():
    """A catalogue whose entries throw or return None on live data is a menu of dead options."""
    state = coherent_state()
    for metric_id, spec in HRV_METRICS.items():
        value = metric_value(spec, state)
        assert value is not None, f"{metric_id} produced nothing"
        assert np.isfinite(value), f"{metric_id} produced {value}"
        assert spec.label and spec.description and spec.detail, f"{metric_id} is undocumented"
        assert spec.strategies, f"{metric_id} recommends no scaling"
        for suggested in spec.strategies:
            assert suggested in normalise.STRATEGIES, f"{metric_id} names a strategy that is gone"


def test_a_metric_reports_nothing_rather_than_zero_before_it_can_speak():
    """Zero is a reading and 'no reading yet' is not. A channel handed 0.0 drives itself to its floor and
    reads as a bad result, which is the opposite of the truth."""
    proc = HRVProcessor()
    early = proc.push(Beat(1000.0, 60.0, [1.0]))          # one beat: no spectrum
    assert metric_value(HRV_METRICS["sharpness"], early) is None
    assert metric_value(HRV_METRICS["coherence"], early) is None
    # The time-domain metrics used to claim readiness here and return a literal 0.0 — which is not
    # nothing, it is a reading, and it anchored the bottom of the rolling scale for the whole window.
    assert metric_value(HRV_METRICS["rsa_amplitude"], early) is None
    assert metric_value(HRV_METRICS["rmssd"], early) is None
    # the rate itself works from the start, which is what makes it the wiring-check metric
    assert metric_value(HRV_METRICS["hr_smoothed"], early) is not None
    assert metric_value(HRV_METRICS["sharpness"], None) is None

    state = early
    for i in range(2, 15):
        state = proc.push(Beat(1000.0 + float(i), 60.0, [1.0]))
    assert metric_value(HRV_METRICS["rsa_amplitude"], state) is not None
    assert metric_value(HRV_METRICS["rmssd"], state) is not None, \
        "and once the beats exist, the number is a number"


def test_the_metrics_that_are_better_low_say_so():
    """`invert` is not a preference. A channel that got brighter as the spectrum drifted further from ideal
    would be reinforcing the wrong thing, so it is a property of the metric."""
    assert HRV_METRICS["spectrum_distance"].invert, "0 is the ideal"
    assert HRV_METRICS["lf_hf_ratio"].invert
    assert HRV_METRICS["hr_smoothed"].invert, "a falling rate is what relaxation looks like"
    for good_high in ("coherence", "resonance_ratio", "sharpness", "rsa_amplitude", "rmssd"):
        assert not HRV_METRICS[good_high].invert


def test_the_metrics_disagree_which_is_why_it_is_a_choice():
    """If they ranked the same states the same way there would be no reason to offer eleven.

    A tight peak slightly off the target is the case that separates them: sharpness punishes it hard
    because it integrates a band around the target, and coherence does not because it locates the peak.
    That is the whole argument for the target being settable — and for coherence being the metric to
    train on until you know where your own resonance is.
    """
    on_target = coherent_state(depth=0.09)
    # the same modulation depth at 0.13 Hz — coherent breathing, at the wrong rate
    proc = HRVProcessor()
    t, off_target = 1000.0, None
    while t < 1200.0:
        rr = 1.0 + 0.09 * np.sin(2 * np.pi * 0.13 * t)
        t += rr
        off_target = proc.push(Beat(t, 60.0 / rr, [rr]))

    sharp_on = metric_value(HRV_METRICS["sharpness"], on_target)
    sharp_off = metric_value(HRV_METRICS["sharpness"], off_target)
    coh_on = metric_value(HRV_METRICS["coherence"], on_target)
    coh_off = metric_value(HRV_METRICS["coherence"], off_target)
    assert sharp_off < sharp_on / 2, "sharpness is pinned to the target and punishes 0.13 hard"
    assert coh_off > coh_on * 0.5, "coherence locates the peak, so it barely notices"


def test_the_target_metrics_follow_the_target_and_the_published_ones_do_not():
    """**Which numbers a personal target may move, and which it must not.**

    Sharpness, peak proximity and the distance from ideal are scored against a chosen frequency, so
    moving the target to where somebody actually resonates has to move them — that is the point. The
    coherence measures and LF/HF are computed over the standard published bands, and re-cutting those
    around a personal target would produce numbers that look like the conventional ones and are not.
    """
    def state_at(target_hz):
        proc = HRVProcessor(target_hz=target_hz)
        t, state = 1000.0, None
        while t < 1200.0:
            rr = 1.0 + 0.09 * np.sin(2 * np.pi * 0.065 * t)
            t += rr
            state = proc.push(Beat(t, 60.0 / rr, [rr]))
        return state

    default, moved = state_at(0.1), state_at(0.065)

    for follows in ("sharpness", "peak_proximity", "spectrum_distance"):
        assert (metric_value(HRV_METRICS[follows], default)
                != metric_value(HRV_METRICS[follows], moved)), f"{follows} must follow the target"
    assert metric_value(HRV_METRICS["sharpness"], moved) > \
        metric_value(HRV_METRICS["sharpness"], default) * 5
    assert metric_value(HRV_METRICS["peak_proximity"], moved) > 0.95, \
        "a peak exactly on the target scores 1.0 wherever the target is"
    assert metric_value(HRV_METRICS["spectrum_distance"], moved) < \
        metric_value(HRV_METRICS["spectrum_distance"], default)

    for fixed in ("coherence", "resonance_ratio", "lf_hf_ratio"):
        assert (metric_value(HRV_METRICS[fixed], default)
                == pytest.approx(metric_value(HRV_METRICS[fixed], moved))), \
            f"{fixed} is quoted against published bands and must not move"


# --- layer two: how to scale it ---------------------------------------------------------------

def test_every_strategy_maps_into_the_unit_interval_and_is_documented():
    for strategy_id, spec in normalise.STRATEGIES.items():
        assert spec.label and spec.description and spec.detail
        strategy = normalise.build(strategy_id)
        for t, value in enumerate([-1e6, -1.0, 0.0, 0.5, 1.0, 1e6]):
            out = strategy.push(value, float(t))
            assert 0.0 <= out <= 1.0, f"{strategy_id} produced {out}"
        assert strategy.describe()


def test_an_unknown_strategy_raises_rather_than_falling_back():
    """A silent fallback would normalise by a rule nobody chose, and everything downstream would look fine."""
    with pytest.raises(ValueError):
        normalise.build("vibes")


def test_a_rolling_strategy_reports_the_middle_and_says_it_is_not_ready():
    """It has to return *something*, and 0.5 is the least wrong something — but a screen that showed that as
    a real reading would be showing 'average' for a signal it knows nothing about. Hence `warmup`."""
    strategy = normalise.build("rolling_minmax", window_s=60.0)
    assert strategy.warmup == 0.0
    assert strategy.push(5.0, 0.0) == 0.5, "one value cannot have a range"
    assert strategy.warmup < 1.0
    for i in range(1, 61):
        strategy.push(float(i), float(i))
    assert strategy.warmup == 1.0
    assert strategy.push(1000.0, 61.0) == 1.0, "a new maximum is the top of the range"

    # the stateless ones are ready immediately, which is the whole reason to prefer them
    assert normalise.build("fixed_range").warmup == 1.0
    assert normalise.build("tanh").warmup == 1.0


def test_a_scaling_only_scales_and_the_direction_belongs_to_the_metric():
    """**Every strategy rises.** Which end is good is the metric's property — `MetricSpec.invert` — applied
    after normalisation, and a scaling that also flipped the sign would be a second way to say the same thing.
    One of them was: `1/(1 + value/scale)`, offered in the "scaled by" row under the label "small is good",
    where it read as a claim about the metric rather than about the scale. It also let a metric that already
    inverts be inverted twice, which lands back where it started with nothing on screen saying so.
    """
    for sid in normalise.STRATEGIES:
        strategy = normalise.build(sid)
        seen = [strategy.push(v, float(i)) for i, v in enumerate(np.linspace(0.0, 4.0, 40))]
        assert all(b >= a - 1e-9 for a, b in zip(seen, seen[1:])), f"{sid} falls as the metric rises"

    # and no label or description claims a direction, since the pill sits under "scaled by"
    for spec in normalise.STRATEGIES.values():
        words = f"{spec.label} {spec.description}".lower()
        for claim in ("small is good", "large is good", "high is better", "zero is the goal"):
            assert claim not in words, f"{spec.id} describes a direction, not a scale"


def test_the_strategies_are_not_interchangeable():
    """Each one exists because it answers a different question about the number's shape."""
    fixed = normalise.build("fixed_range", low=0.0, high=1.0)
    assert fixed.push(0.25, 0.0) == pytest.approx(0.25), "a straight map, comparable across sessions"
    assert fixed.push(50.0, 1.0) == 1.0, "and pinned for a metric outside its range"

    open_ended = normalise.build("saturating", scale=2.0)
    assert open_ended.push(0.0, 0.0) == pytest.approx(0.0)
    assert open_ended.push(2.0, 1.0) == pytest.approx(0.5), "the knob is where it crosses the middle"
    assert open_ended.push(100.0, 2.0) > 0.95, "and it approaches the top without reaching it"
    assert open_ended.push(1e12, 3.0) < 1.0

    centred = normalise.build("tanh", scale=1.0)
    assert centred.push(0.0, 0.0) == pytest.approx(0.5), "zero is neutral"
    assert centred.push(1.0, 1.0) == pytest.approx(0.88, abs=0.01)
    assert centred.push(-1.0, 2.0) == pytest.approx(0.12, abs=0.01)
    # It saturates gracefully rather than hitting a wall: a clamped strategy gives the same answer for
    # everything past its limit, where this keeps its ordering over any range a real metric reaches.
    # (Far enough out, float itself saturates — tanh(1e9) is exactly 1.0. That is arithmetic, not the shape.)
    assert centred.push(5.0, 3.0) < centred.push(8.0, 4.0) < 1.0
    assert 0.0 < centred.push(-8.0, 5.0) < centred.push(-5.0, 6.0)
    clamped = normalise.build("zscore", window_s=60.0, clamp_k=2.0)
    for i in range(60):
        clamped.push(float(i % 3), float(i))            # a small, stable spread
    assert clamped.push(1e3, 61.0) == clamped.push(1e6, 62.0) == 1.0, "a clamp stops distinguishing"


# --- and what carries it: the shared mechanisms, not a parallel set ----------------------------

def test_the_bounded_pitch_survived_as_music_and_the_rest_did_not():
    """Of the five rendering modes that were built, one carries something no mechanism could: a **bounded
    pitch**, which conveys a position with no reference point needed — you hear which way it moved without
    remembering where it was, and a volume cannot do that. It is a `music` source now, beside the unbounded
    Shepard tone, which is a different instrument rather than a setting of it.

    The others were the mechanisms reimplemented. `system_brightness` is `shadow`. `noise_mask` is `music` on
    noise with `sound.invert` — the Quiet Reward arrangement, which already existed and is measured elsewhere.
    """
    from nfb_console.feedback import MUSIC_SOURCES, Music, Sound

    assert "tone" in MUSIC_SOURCES and "shepard" in MUSIC_SOURCES
    settings = Music()
    assert (settings.tone_base_hz, settings.tone_top_hz) == (220.0, 660.0)

    # the mask, expressed with what was already there: loudest at 0, silent at 1
    quiet = Sound(enabled=True, low=0.0, high=60.0, invert=True)
    assert quiet.map(0.0) > quiet.map(1.0)
    assert quiet.map(1.0) == pytest.approx(0.0)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("nfb_console.passive")


def test_the_pitch_source_gets_its_own_rows_and_not_the_file_browser():
    """Its two ends are a real choice, because unlike the Shepard wheel it is bounded — and a source with no
    branch of its own falls through to the file browser and offers to play an mp3, which is what the Shepard
    music did before it got one."""
    import flet as ft
    import sys
    sys.path.insert(0, "tests")
    from test_screens import StubPage

    from nfb_console import signal_pane

    session = _session()
    try:
        pane = signal_pane.SignalPane(StubPage(), session)
        pane._select(session.offers.index("music"))
        pane.control.visible = True

        def visible(x, d: int = 0) -> str:
            if x is None or d > 30 or getattr(x, "visible", True) is False:
                return ""
            if isinstance(x, ft.Text):
                return x.value or ""
            out = [visible(c, d + 1) for c in (getattr(x, "controls", None) or [])]
            content = getattr(x, "content", None)
            if content is not None:
                out.append(visible(content, d + 1))
            return " ".join(v for v in out if v)

        shown = visible(pane.control)
        assert "pitch" in shown and "glide" in shown
        assert "file or folder" not in shown and "browse" not in shown
    finally:
        session.stop()


def test_the_pitch_source_tracks_the_level_between_its_two_frequencies():
    from nfb_console import music
    from nfb_console.feedback import Music

    out = music.MusicOutput()
    settings = Music(enabled=True, source="tone")
    try:
        for level in (0.0, 0.5, 1.0):
            out.apply(settings, level=level)
            out.tone._block(2048)
        # the glide means it approaches rather than jumps, so the check is direction and bounds
        assert settings.tone_base_hz <= out.tone.pitch_hz <= settings.tone_top_hz
        out.apply(settings, level=0.0)
        for _ in range(80):
            out.tone._block(2048)
        low = out.tone.pitch_hz
        out.apply(settings, level=1.0)
        for _ in range(80):
            out.tone._block(2048)
        assert out.tone.pitch_hz > low + 100.0, "the pitch has to follow the signal"
    finally:
        out.release()


# --- end to end on the screen -----------------------------------------------------------------

def _session():
    from nfb_console.hrv_view import HRVSession
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    return HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))


def _feed(session, seconds: float = 250.0, depth: float = 0.09):
    """Run a coherent tachogram through the session with its clock advancing.

    The clock has to advance or the rolling strategies see every value at one instant, never warm up, and
    report the middle for ever — which is how a first check of this looked like it was working and was not.
    """
    t, clock = 1000.0, 0.0
    session._resumed_at = None
    while clock < seconds:
        rr = 1.0 + depth * np.sin(2 * np.pi * 0.1 * t)
        t += rr
        clock += rr
        session._accumulated = clock
        session.heart.state = session.heart.processor.push(Beat(t, 60.0 / rr, [rr]))
        session.observe()


def test_the_metric_and_scaling_become_the_level_the_mechanisms_follow():
    """The whole composition: metric, scaling, level, then the same four mechanisms every screen has."""
    session = _session()
    try:
        session.toggle_pause()
        session.feedback.sound.enabled = True
        session.feedback.shadow.enabled = True
        assert session.signal is None, "nothing before the metric can speak"
        assert session.active_targets == []
        _feed(session)
        assert session.raw_metric is not None
        assert 0.0 <= session.signal <= 1.0
        assert session.level > 0.5, "a clean 0.1 Hz peak should read high"
        assert session.strategy.warmup > 0.9
        outputs = session.feedback.outputs_for(session.level, True)
        assert outputs["sound"] > session.feedback.sound.low
        assert outputs["shadow"] > session.feedback.shadow.low
    finally:
        session.stop()


def test_the_level_comes_from_the_scaling_and_not_from_a_threshold():
    """The one target on this screen exists for the pane's contribution row and for the click to ask; it is
    fed the *already normalised* signal with its threshold at the middle of that range, so nothing is
    calibrated twice and the level is the scaling's own output."""
    from nfb_console.hrv_view import CONDITION_LEVEL

    session = _session()
    try:
        session.toggle_pause()
        _feed(session)
        assert len(session.targets) == 1
        target = session.targets[0]
        assert target.band is None, "it measures no band, so it gets no card and no filter gear"
        assert target.threshold == CONDITION_LEVEL == 0.5
        assert not target.calibrating, "the scaling already did that job"
        assert target.value == pytest.approx(session.signal)
        # the level is the signal, smoothed — not a logistic of it
        assert session.level == pytest.approx(session.signal, abs=0.35)
    finally:
        session.stop()


def test_changing_the_metric_rebuilds_the_scaling_and_refills_it_from_the_beats_in_hand():
    """A rolling window holds values in the *old* metric's units. Carrying it across would normalise
    milliseconds against a history of dimensionless ratios, and the signal would sit pinned at an end until
    the window aged out — so it is rebuilt.

    **Rebuilt is not the same as empty.** An empty rolling scale is gated until a quarter of its window
    exists, which meant changing the metric stopped the feedback for thirty seconds; and there was never
    anything to wait for, since the beats are still in hand and a metric is a pure function of a state.
    The new scale is handed the values it would have had if it had been running all along.
    """
    session = _session()
    try:
        session.toggle_pause()
        _feed(session)
        assert session.strategy.warmup > 0.9
        before = session.strategy
        session.set_metric("rmssd")
        assert session.strategy is not before, "the old window was in the old metric's units"
        assert session.strategy.warmup > 0.9, "and the new one arrives full, from the same beats"

        # The values in it are the new metric's, not the old one's: RMSSD is in milliseconds and runs to
        # tens, a coherence score does not.
        held = [v for _t, v in session.strategy._buf]
        assert held and max(held) > 5.0, f"the window holds {max(held):.3g}, which is not milliseconds"
        session.observe()
        assert session.signal is not None, "so the feedback continues rather than stopping"
        # and the strategy follows the metric's own recommendation when the current one does not suit it
        session.set_strategy("fixed_range", low=0.0, high=1.0)
        session.set_metric("coherence")
        assert session.strategy_id in HRV_METRICS["coherence"].strategies
    finally:
        session.stop()


def test_the_direction_is_the_operators_and_every_metric_arrives_with_one():
    """**Which way you are training it is a setting, not a property of the number.**

    It was `MetricSpec.invert` and nothing else — unreachable from the screen, so the most obvious thing an
    operator wants to say about a quantity could not be said. LF/HF is the case that makes it plain: inverted
    in the catalogue because sympathetic drive is the usual concern, and exactly backwards for somebody
    training a flat, over-parasympathetic profile. The metric supplies the default, the way a protocol's bands
    arrive as increase or decrease on the amplitude screen and can be cycled there.
    """
    session = _session()
    try:
        # every metric ships an aim, so there is never a direction nobody chose
        for spec in HRV_METRICS.values():
            assert spec.default_direction in ("increase", "decrease")
        assert HRV_METRICS["coherence"].default_direction == "increase"
        assert HRV_METRICS["lf_hf_ratio"].default_direction == "decrease"

        session.toggle_pause()
        assert session.direction == "increase" and not session.inverts
        _feed(session)
        up = session.signal
        assert up > 0.5

        # flipping it inverts the level and nothing else: the scaling keeps its window and its warm-up
        window = session.strategy
        session.set_direction("decrease")
        assert session.strategy is window, "the direction is not a rescaling"
        assert session.signal is None, "and the level it already reported was under the other aim"

        # the same coherent tachogram, trained the other way, reads low
        other = _session()
        try:
            other.toggle_pause()
            other.set_direction("decrease")
            _feed(other)
            assert other.inverts
            assert other.signal == pytest.approx(1.0 - up, abs=0.05)
            assert other.signal < 0.5
        finally:
            other.stop()

        # changing the metric puts the aim back to that metric's own — carrying "decrease" onto RMSSD would
        # train somebody toward *less* variability with only the metric's name having changed
        session.set_metric("rmssd")
        assert session.direction == "increase"
        session.set_metric("spectrum_distance")
        assert session.direction == "decrease", "and it takes the inverted default where there is one"

        # it survives a stop, like the metric and the scaling
        session.set_direction("increase")
        second = _session()
        try:
            second.adopt(session)
            assert second.direction == "increase"
            assert second.metric == "spectrum_distance"
        finally:
            second.stop()
    finally:
        session.stop()


def test_the_screen_puts_the_direction_in_its_own_row_and_says_what_is_usual():
    import flet as ft
    import sys
    sys.path.insert(0, "tests")
    from test_screens import StubPage

    from nfb_console import hrv_view

    def visible(x, d: int = 0) -> str:
        if x is None or d > 30 or getattr(x, "visible", True) is False:
            return ""
        if isinstance(x, ft.Text):
            return x.value or ""
        out = [visible(c, d + 1) for c in (getattr(x, "controls", None) or [])]
        content = getattr(x, "content", None)
        if content is not None:
            out.append(visible(content, d + 1))
        return " ".join(v for v in out if v)

    session = _session()
    try:
        source = hrv_view.SignalSource(StubPage(), session)
        shown = visible(source.control)
        assert "aim to" in shown and "increase" in shown and "decrease" in shown
        assert "the usual aim for coherence" in shown
        # and it is not in the scaling's row: a scaling describes a shape, not an end
        assert "small is good" not in shown

        session.set_direction("decrease")
        source.rebuild()
        assert "coherence is usually trained to increase" in visible(source.control), \
            "off the usual aim, the row says so rather than looking like the default"
    finally:
        session.stop()


def test_an_inverted_metric_inverts_the_level_and_not_the_scaling():
    """The inversion is applied after normalisation, so a rolling *minimum* stays a minimum and the
    description on the pane keeps telling the truth about what the window holds."""
    session = _session()
    try:
        session.toggle_pause()
        session.set_metric("spectrum_distance")         # arrives set to decrease
        session.set_strategy("fixed_range", low=0.0, high=1.0)
        _feed(session, seconds=200.0)
        raw, signal = session.raw_metric, session.signal
        assert raw is not None and raw < 0.5, "a coherent spectrum is close to the ideal"
        assert signal == pytest.approx(1.0 - raw, abs=1e-6), "inverted, so close-to-ideal reads high"
    finally:
        session.stop()


def test_the_screen_has_one_feedback_tab_carrying_the_shared_mechanisms():
    import sys
    sys.path.insert(0, "tests")
    from test_screens import StubPage, toolbar

    import flet as ft

    from nfb_console import hrv_view

    def visible(x, d: int = 0) -> str:
        if x is None or d > 20 or getattr(x, "visible", True) is False:
            return ""
        if isinstance(x, ft.Text):
            return x.value or ""
        out = [visible(c, d + 1) for c in (getattr(x, "controls", None) or [])]
        content = getattr(x, "content", None)
        if content is not None:
            out.append(visible(content, d + 1))
        return " ".join(v for v in out if v)

    page = StubPage()
    session = _session()
    try:
        control = hrv_view.view(page, session, toolbar(), lambda: None)
        session.toggle_pause()
        _feed(session)
        session.redraw()
        panes = control.controls[-1].controls
        assert len(panes) == 3, "Train, Feedback, Session"
        for i, pane in enumerate(panes):
            pane.visible = (i == 1)
        shown = visible(control)
        # the two rows that are ours
        assert "SIGNAL SOURCE" in shown
        assert "feed back" in shown and "scaled by" in shown
        # and the shared mechanisms below them
        assert "OUTPUT" in shown, "the pane every other protocol uses"
        for mechanism in ("sound", "shadow", "click", "music"):
            assert mechanism in shown.lower(), f"{mechanism} is configurable here"
        # and none of the panels or rows that belong to a level built by scoring bands: absent rather than
        # emptied, which is what `conditions_signal` and `scores_bands` are for
        for gone in ("ARTIFACTS", "rejection", "band averaging", "2 · scale from", "how far counts",
                     "3 · level from", "4 · condition"):
            assert gone not in shown, f"{gone!r} has nothing to act on here"
        assert "output smoothing" in shown, "the one step that applies to any level"
        # and no second set of channels
        assert "ambient tint" not in shown and "the room" not in shown
    finally:
        session.stop()


def test_the_toolbar_lifecycle_exists_and_pausing_hands_the_room_back():
    """The device chip on the shared toolbar calls suspend()/suspended/rearm() on whatever session is
    up; this session not having them was an AttributeError inside a Flet event handler — a click that
    silently did nothing. And toggle_pause only flipped a flag, which stops the outputs being *updated*
    rather than being *held*: the display stayed dimmed and the music sang on, the exact bug
    `TrainingSession.toggle_pause` documents having been fixed on every other screen."""
    session = _session()
    try:
        released = []
        for name in ("display", "output_volume", "music"):
            obj = getattr(session, name)
            setattr(obj, "release", (lambda n=name: lambda: released.append(n))())

        assert session.paused, "a session opens paused"
        session.toggle_pause()
        assert not session.paused and released == [], "resuming releases nothing"
        session.toggle_pause()
        assert session.paused
        assert released == ["display", "output_volume", "music"], "pausing hands all three back"

        released.clear()
        session.suspend()                    # already paused: still releases, never raises
        assert released == ["display", "output_volume", "music"]
        assert session.suspended
        session.rearm()                      # and coming back is a no-op, not a crash
    finally:
        session.stop()


def test_a_finished_plan_releases_the_outputs_too():
    session = _session()
    try:
        released = []
        for name in ("display", "output_volume", "music"):
            obj = getattr(session, name)
            setattr(obj, "release", (lambda n=name: lambda: released.append(n))())
        session.plan.length_seconds = 0.01
        session.toggle_pause()               # start the clock
        import time as _time

        _time.sleep(0.02)
        session.tick()
        assert session.finished and session.paused
        assert released == ["display", "output_volume", "music"], \
            "a timed session that ends must not leave the room held"
    finally:
        session.stop()


def test_a_dead_or_silent_beat_source_stops_the_feedback_and_says_so():
    """Two flavours, both of which used to run the room on a frozen snapshot: a source that raises sets
    an error but froze `state` with no gate downstream, and a source whose generator simply *ends* — how
    BleBeats reports its worker dying — set no error at all. Now an unrequested end is a fault, and
    observe() refuses to feed outputs from an errored or stale runner."""
    import time as _time

    from nfb_console.heart import HeartRunner
    from nfb_console.inputs import HeartChoice

    class Ends:
        def beats(self):
            yield Beat(1000.0, 60.0, [1.0])
            yield Beat(1001.0, 60.0, [1.0])

        def close(self):
            pass

        def describe(self):
            return "test source"

    session = _session()
    try:
        runner = session.heart
        runner.source = Ends()
        runner._stop.clear()
        # One attempt, not the retry loop the reader thread runs: what is under test is that an ending
        # nobody asked for is reported as a fault rather than passing in silence.
        runner.error = runner._read_once()   # the generator ends on its own
        assert runner.error, "an end nobody asked for is a fault, not a silence"

        applies = []
        for name in ("display", "output_volume", "music"):
            obj = getattr(session, name)
            setattr(obj, "apply", (lambda n=name: lambda *a, **kw: applies.append(n))())
            setattr(obj, "release", lambda: None)
        session.toggle_pause()
        session.observe()
        assert session.signal is None and applies == [], \
            "an errored source must not keep driving the outputs"
    finally:
        session.stop()


def test_the_invented_simulator_is_handed_back_switched_off():
    """The simulator default is this screen's, not the app's: the runner is shared with CFC, and one
    visit to HRV used to leave the app-wide choice pointing at the simulator — CFC then drew invented
    beats, and could raise the split warning on them, beside a real EEG. Stopping now restores the
    runner to off with the simulated state cleared. A choice the operator actually made is kept."""
    from nfb_console.inputs import HeartChoice

    session = _session()                     # _session passes an explicit simulator choice
    assert not session._invented, "an explicit choice is the operator's and is kept"

    from nfb_console.hrv_view import HRVSession
    from nfb_console.protocols import BY_CODE

    invented = HRVSession(BY_CODE["HRV"], choice=None)
    try:
        assert invented._invented and invented.choice.kind == "simulator"
        invented.heart.state = object()      # pretend the simulator ran
        invented.heart.error = "leftover"
    finally:
        invented.stop()
    assert invented.heart.choice.kind == "off", "the shared runner goes back the way it was found"
    assert invented.heart.state is None and invented.heart.error is None


def test_a_fresh_source_starts_a_fresh_record(monkeypatch):
    """attach() resets the processor: a session started within the buffer's three minutes of the last
    used to inherit its beats, its pre-satisfied warm-up, and a pending beat from another session — or
    another device, since attach is also how simulator switches to strap.

    The reset is `attach`'s own work, done before it starts the reader, so it is checked the moment it
    returns rather than by sleeping until beats have arrived through a thread."""
    from nfb_console.heart import HeartRunner
    from nfb_console.inputs import HeartChoice

    # A source that ends at once, so the reader thread neither sleeps between beats nor has to be waited
    # for: what is under test is the reset, and a real simulator's 0.9 s beat interval was the cost of
    # releasing it afterwards.
    class Empty:
        def beats(self):
            return iter(())

        def close(self):
            pass

        def describe(self):
            return "nothing"

    monkeypatch.setattr("nfb.hrv.beat_source", lambda kind, **kw: Empty())
    runner = HeartRunner(HeartChoice(kind="simulator"))
    runner.processor.push(Beat(1000.0, 60.0, [1.0]))
    runner.processor.push(Beat(1001.0, 60.0, [1.0]))
    runner.state = object()
    assert runner.processor._live, "old beats are in the buffers"
    assert runner.attach()
    try:
        # Nothing from before the attach is in the buffers — true the instant it returns, and true
        # afterwards however many real beats the reader has added by the time this runs. The old version
        # slept a fifth of a second first so that there would be real beats to compare against, which
        # made the scheduler part of the assertion.
        assert all(bt > 1e6 for bt, _ in runner.processor._live), "old beats survived the attach"
        assert all(bt > 1e6 for bt, _ in runner.processor._beats)
    finally:
        runner.release()


def test_the_pacer_sounds_each_turn_once_and_the_two_are_different_pitches():
    """A paced breath you have to watch owns your eyes for the whole session, on a screen whose whole
    point is the instruments. So the turn is audible: a higher tone in, a lower one out, and the pitch goes the way the breath goes."""
    from nfb_console import hrv_view

    session = _session()
    session.pacer_locked = False              # the metronome, so the turn is on the clock
    session.set_pacer_period(10.0)
    played: list[str] = []
    session.cue = lambda name, *a, **kw: played.append(name)
    try:
        session.toggle_pause()                # running
        session._resumed_at = time.monotonic() - 1.0     # 1 s in: inhaling
        session.tick()
        session.tick()                        # nothing changed, so nothing sounds
        assert len(played) == 1, "one sound per turn, not one per draw"

        session._resumed_at = time.monotonic() - 6.0     # 6 s in: exhaling
        session.tick()
        assert len(played) == 2
        assert played[0] != played[1], "in and out are told apart by pitch"
        assert str(int(hrv_view.PACER_IN_HZ)) in played[0]
        assert str(int(hrv_view.PACER_OUT_HZ)) in played[1]

        session.pacer_sound = False
        session._resumed_at = time.monotonic() - 11.0    # back to inhaling
        session.tick()
        assert len(played) == 2, "and it can be switched off"
    finally:
        session.stop()


def test_the_metronome_has_a_phase_to_breathe_with_and_the_heart_pacer_does_not():
    """The ring grows through the whole inhale and shrinks through the whole exhale, which is the
    instruction and the pace in one object. The heart-locked pacer turns on an event — the heart starting
    to slow — and has no idea how far through a phase it is, so it holds still rather than animating a
    guess."""
    session = _session()
    session.pacer_locked = False
    session.set_pacer_period(10.0)
    try:
        session.toggle_pause()
        session._resumed_at = time.monotonic() - 0.05
        assert session.breathe_in() and session.breath_phase() < 0.1, "the inhale starts small"
        session._resumed_at = time.monotonic() - 4.9
        assert session.breathe_in() and session.breath_phase() > 0.9, "and ends full"
        session._resumed_at = time.monotonic() - 5.1
        assert not session.breathe_in() and session.breath_phase() < 0.1, \
            "the exhale starts full and shrinks, which is the same phase read the other way"

        session.pacer_locked = True
        session.heart.state = type("S", (), {"accelerating": True, "beats": 9})()
        assert session.breath_phase() is None, "nothing to animate when the turn is an event"
    finally:
        session.stop()


def test_the_room_does_something_from_the_first_beats_rather_than_after_two_minutes():
    """**The opening minute fed back nothing at all**, in three successive versions of this screen: the
    spectrum refused to exist below eighty seconds, then twenty; the spectral metrics waited four cycles
    of the target; and the rolling scale wanted a quarter of its window on top. Every one of them bought
    accuracy with a minute in which somebody breathes and nothing answers.

    Now nothing waits. Fourteen beats is a real spectrum — coarse, and labelled coarse by the span and
    bin width under the curve — so the chosen metric has a value and the room runs on it."""
    from nfb.hrv import HRVProcessor
    from nfb.hrv.beats import TARGET_HZ
    from nfb_console.hrv_view import STANDIN_METRIC

    session = _session()
    try:
        session.toggle_pause()
        proc = HRVProcessor()
        t0 = 1000.0
        state = None
        for i in range(14):                    # fourteen beats: no spectrum, plenty for a swing
            rr = 1.0 + 0.12 * np.sin(2 * np.pi * 0.1 * (t0 + i))
            t0 += rr
            state = proc.push(Beat(t0, 60.0 / rr, [rr]))
        session.heart.state = state
        session.heart.seen_at = time.monotonic()
        # Fourteen beats cannot *resolve* 0.1 Hz — four cycles of it is forty seconds — and that is no
        # longer a reason to withhold the number. The spectrum exists, so the metric has a value.
        assert not state.spectrum.resolves(TARGET_HZ), "fourteen beats is a coarse window, and says so"
        assert state.spectrum.ready, "but it is a spectrum, and the transform is defined on it"

        session.observe()
        assert session.signal is not None, "and the room has a level to run on"
        assert 0.0 <= session.signal <= 1.0
        assert not session.standing_in, \
            f"the chosen metric speaks from the first spectrum, so {STANDIN_METRIC} is not needed"
        assert session.raw_metric == metric_value(session.metric_spec, state)
    finally:
        session.stop()


def test_nothing_between_the_first_beats_and_the_feedback_waits_for_a_number_to_settle():
    """The rule, stated once against the objects that used to break it. A gate may say "this cannot be
    computed"; none may say "this is not accurate enough yet". Accuracy is reported — the span and the
    bin width under the curve, the warm-up bar beside the readout — never withheld."""
    from nfb.hrv import HRVProcessor
    from nfb_console.hrv_view import WARMUP_GATE

    assert WARMUP_GATE == 0.0, "the level does not wait for a rolling scale to fill"

    proc = HRVProcessor()
    t, state = 1000.0, None
    for i in range(4):                      # four beats: the arithmetic floor, nothing above it
        rr = 1.0 + 0.1 * np.sin(2 * np.pi * 0.1 * t)
        t += rr
        state = proc.push(Beat(t, 60.0 / rr, [rr]))

    assert state.spectrum.ready, "three beats make a spectrum"
    for metric_id in HRV_METRICS:
        assert metric_value(HRV_METRICS[metric_id], state) is not None, \
            f"{metric_id} has everything it arithmetically needs and must produce a number"


def test_the_stand_in_covers_a_metric_with_nothing_to_say_and_not_a_scale_that_is_still_settling():
    """**The stand-in's job shrank to the honest one.** It used to hold the room until the chosen
    metric's scale was a quarter full, which is a settling time, and the same trade this screen made
    three times over — an opening minute spent on accuracy nobody asked for. The chosen metric now
    drives from its first value, however rough, and the warm-up bar says it is still firming up.

    What remains is real: a metric with *no* value — the first seconds, before enough beats exist for a
    difference between beats — where the swing of the heart rate is the first thing that can be measured.
    """
    session = _session()
    try:
        session.toggle_pause()
        session.heart.state = coherent_state()          # a full spectrum: coherence has a value
        session.heart.seen_at = time.monotonic()

        session.observe()
        assert not session.standing_in, "one value is a value; the scale reports the middle and widens"
        assert session.signal is not None
        assert len(session.strategy._buf) == 1, "and the scale is being fed the whole time"
        assert session.raw_metric == metric_value(session.metric_spec, session.heart.state)

        # A window's worth of readings later, nothing about who is driving has changed.
        _feed(session)
        assert not session.standing_in
        assert session.raw_metric == metric_value(session.metric_spec, session.heart.state)
        assert session.signal is not None, "and the level never stopped on the way"
    finally:
        session.stop()


def test_the_stand_in_still_covers_the_seconds_before_any_metric_has_a_value():
    """The one case it is for: two beats, so there is no spectrum and no successive difference. The
    chosen metric is `None` and the level would otherwise be missing at the moment somebody presses
    play."""
    from nfb.hrv import HRVProcessor
    from nfb_console.hrv_view import STANDIN_METRIC

    session = _session()
    try:
        session.toggle_pause()
        proc = HRVProcessor()
        t, state = 1000.0, None
        for i in range(6):
            rr = 1.0 + 0.12 * np.sin(2 * np.pi * 0.1 * t)
            t += rr
            state = proc.push(Beat(t, 60.0 / rr, [rr]))
        session.set_metric("rmssd")
        session.heart.state = state
        session.heart.seen_at = time.monotonic()
        session.observe()
        assert session.raw_metric is not None and session.signal is not None

        # and with a metric that genuinely has nothing, the swing carries it
        session.heart.state = proc.push(Beat(t + 1.0, 60.0, [1.0]))
        session.set_metric(STANDIN_METRIC)
        session.observe()
        assert session.signal is not None
    finally:
        session.stop()


def test_the_pacer_asks_about_the_cycle_only_when_the_cycle_is_what_is_running():
    """A control that does nothing in the mode you are in is a question about a mode you did not pick.
    The cycle length belongs to the metronome; "follow the heart" takes its timing from the tachogram."""
    import flet as ft

    from nfb_console import chrome, hrv_view

    class Page:
        width = 1240

        def __init__(self):
            self.controls, self.services, self.tasks, self.dialogs = [], [], [], []
            self.window = type("W", (), {})()

        def add(self, control):
            self.controls.append(control)

        def update(self, *a):
            pass

        def run_task(self, fn, *a):
            self.tasks.append((fn, a))

        def show_dialog(self, dialog):
            self.dialogs.append(dialog)

    def labels(control):
        if isinstance(control, ft.Text):
            return [control.value or ""]
        out = []
        for child in getattr(control, "controls", None) or []:
            out += labels(child)
        if (inner := getattr(control, "content", None)) is not None:
            out += labels(inner)
        return out

    class StubDevice:
        from nfb import Mode as _Mode
        status = type("S", (), {"mode": _Mode.DISCONNECTED, "connected": False, "failure": None,
                                "remedy": None, "model": "", "serial": "", "firmware": "",
                                "port": "", "epoch": 0,
                                "settings": type("T", (), {"hz": 250.0, "reference": "linked",
                                                           "mains_hz": 50, "lower_limit_hz": 0.5})()})()

    session = _session()
    try:
        page = Page()
        bar = chrome.Toolbar(StubDevice(), lambda: None, lambda: None)
        control = hrv_view.view(page, session, bar, lambda: None)

        assert session.pacer_locked, "follow the heart is the default"
        shown = labels(control)
        assert "follow the heart" in shown
        assert "cycle" not in shown, "the metronome's period has no meaning while following the heart"

        # switching to the metronome brings it back, since that is the mode it belongs to
        for ctrl in _walk(control):
            if isinstance(ctrl, ft.Container) and isinstance(ctrl.content, ft.Text) \
                    and ctrl.content.value == "metronome" and ctrl.on_click is not None:
                ctrl.on_click(None)
                break
        assert "cycle" in labels(control), "and it is there in the mode it belongs to"

        # And the outer ring says what it is — it was drawn with nothing anywhere naming it. Written on
        # each frame, like every other reading on the screen, so the test has to draw one.
        session.redraw()
        assert any("the outer ring is the level" in x for x in labels(control))
    finally:
        session.stop()


def _walk(control):
    import flet as ft

    yield control
    for child in getattr(control, "controls", None) or []:
        yield from _walk(child)
    if (inner := getattr(control, "content", None)) is not None and isinstance(inner, ft.Control):
        yield from _walk(inner)


def test_the_level_keeps_running_through_a_metric_change_and_is_warm_when_you_press_play():
    """Two delays that were never necessary, both from a scale that starts empty.

    Changing the metric rebuilds the normaliser — it has to, the units differ — and the rebuilt one was
    empty, so the feedback stopped for a quarter of its window. And the scale was only fed while the
    session ran, so pressing play began a further thirty seconds of silence. Both are answered from the
    beats already in hand: the metric is a pure function of a state, and the states can be recomputed.
    """
    session = _session()
    try:
        session.toggle_pause()
        _feed(session)
        session.toggle_pause()                       # paused again, beats still arriving
        assert session.strategy.warmup > 0.9

        before = session.signal
        session.set_metric("rmssd")
        session.observe()
        assert session.signal is not None and before is not None, "the level did not stop"

        # And while paused the scale keeps learning, so play starts on a warm one rather than on nothing.
        session.set_strategy("rolling_minmax")
        assert session.strategy.warmup > 0.9, "refilled from history the moment it was built"
        assert not session.running
        session.observe()
        assert session.raw_metric is not None, "measured while paused"
    finally:
        session.stop()


def test_measuring_while_paused_never_touches_the_room():
    """The reason the measurement was gated on running in the first place, and it is still the rule —
    just applied to the one place that drives anything instead of to the whole path."""
    session = _session()
    try:
        _feed(session, seconds=60.0)
        applied: list[str] = []
        for name in ("display", "output_volume", "music"):
            obj = getattr(session, name)
            setattr(obj, "apply", (lambda n=name: lambda *a, **kw: applied.append(n))())
            setattr(obj, "release", lambda: None)
        assert not session.running
        for _ in range(5):
            session.observe()
        assert applied == [], "a paused session measures and drives nothing"
        assert session.raw_metric is not None
    finally:
        session.stop()


def test_a_missing_level_says_why_instead_of_blaming_a_channel_that_does_not_exist():
    """**"no channel is delivering signal", in red, while the beats were arriving and visible.** The
    shared pane's only vocabulary for an unavailable level was about amplifier channels, and on this
    screen there are none.

    The situation it used to describe — a scale that had not filled a quarter of its window — is gone,
    because nothing waits for that any more. What is left is the honest case: not enough beats yet for a
    difference between beats to exist. The wording is checked here; the counting is checked below."""
    from nfb.hrv import Beat, HRVProcessor
    from nfb_console import signal_pane

    session = _session()
    try:
        session.toggle_pause()
        # Three beats: the source is plainly alive, and the swing still needs a fourth to exist.
        proc = HRVProcessor()
        state = None
        for i in range(3):
            state = proc.push(Beat(1000.0 + i, 60.0, [1.0]))
        session.set_metric("rsa_amplitude")
        session.heart.state = state
        session.heart.seen_at = time.monotonic()
        session.observe()

        assert session.dead_channels() == [], "the source is fine and must not be blamed"
        holding = signal_pane._holding(session)
        assert holding is not None
        headline, note = holding
        assert "counting beats" in headline
        assert "channel" not in headline and "channel" not in note

        # and a barely-started rolling scale is *not* one of these situations any more
        _feed(session, seconds=20.0)
        session.set_strategy("rolling_minmax", window_s=600.0)
        session._strategy.reset()                      # as if freshly built with nothing behind it
        session.observe()
        assert signal_pane._holding(session) is None, \
            "an empty scale reports the middle and widens; it does not withhold the level"
        assert session.signal is not None
    finally:
        session.stop()


def test_the_opening_seconds_say_what_they_are_counting():
    """"waiting for data", on a screen whose chip reads "reading the pulse waveform" two inches above,
    is a screen contradicting itself. Variability cannot be measured without beats — four of them, which
    is the arithmetic floor for a swing and not a settling time — but the wait can at least be counted
    out."""
    from nfb.hrv import Beat, HRVProcessor
    from nfb.hrv.metrics import RSA_MIN_BEATS
    from nfb_console import signal_pane

    session = _session()
    try:
        session.toggle_pause()
        proc = HRVProcessor()
        t = 1000.0
        for i in range(3):                       # three beats: not yet a swing
            t += 1.0
            session.heart.state = proc.push(Beat(t, 60.0, [1.0]))
        session.heart.seen_at = time.monotonic()
        session.observe()

        assert session.signal is None, "three beats is not a measurement of variability"
        holding = signal_pane._holding(session)
        assert holding is not None
        assert f"of {RSA_MIN_BEATS}" in holding[0], holding
        assert "waiting for data" not in holding[1]

        for i in range(4):                       # five in total, and the level starts
            t += 1.0
            session.heart.state = proc.push(Beat(t, 60.0 + 4.0 * (i % 2), [1.0]))
        session.observe()
        assert session.signal is not None, "the level runs from the first beats that can carry it"
    finally:
        session.stop()
