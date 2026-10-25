"""The waveform screen and the metric under it: training a signed wave's position.

Two halves. The first checks `nfb.dsp.wave` against signals whose answer is known — a slow sine buried under a
ramp of electrode drift, and a fast band whose amplitude swings slowly — because every claim the screen makes
rests on those two being recovered. The second checks the screen: that nothing is fed back before the filters
have settled and the range has been measured, that the range is then held, and that the one control which is
genuinely arbitrary (which way is up on a bipolar pair) belongs to the operator.

Signals are fed as `Block`s directly rather than through a recording, for the reason `test_ilf.py` does the
same: what these need is a *known* waveform, and the bench recording is a constant at the rail.
"""

from __future__ import annotations


import numpy as np

from conftest import counts
import pytest

from nfb import ReplaySource, Settings, WaitingSource
from nfb.dsp import OnePole, SignedWave, wave_edges
from nfb.dsp.wave import DETREND_POLES, DETREND_RATIO, MAX_HZ, detrend_tau
from nfb.stream import Block, ChannelSpec
from nfb_console import wave_view
from nfb_console.protocols import BY_CODE

from conftest import BENCH_OK

RATE = 250.0
SPEC = ChannelSpec(0, RATE, "µV", 800.0 / 32767)


BLOCK = 250
"""Samples per test block, and deliberately not the device's eight.

The metric emits on a fixed sample hop rather than per block, so the readings and their timestamps are
identical whatever this is — while the cost is not: a `sosfilt` call on eight samples spends 18.7 of its
18.8 µs validating its arguments, so a small block pays that overhead thirty times over for the same
answer. Blocking is the subject in a few tests here (a gap, an index discontinuity); those pass
`step=8`."""


def _feed(metric, x: np.ndarray, spec: ChannelSpec = SPEC, start: int = 0,
          step: int = BLOCK) -> np.ndarray:
    """Push µV through a metric in blocks. Returns (t, value)."""
    ints = counts(x, spec.scale, truncate=True)
    out: list[tuple[float, float]] = []
    for i in range(0, len(x) - (step - 1), step):
        block = Block(spec.index, start + i, tuple(ints[i:i + step].tolist()), spec)
        for r in metric.feed(block):
            out.append((r.t, r.value))
    return np.asarray(out) if out else np.zeros((0, 2))


def _best_lag_corr(ts, values, hz: float) -> tuple[float, float]:
    """Correlation against a sine of `hz`, over the lag that fits best, and that lag.

    Lag-aware on purpose: every filter here delays, and at 0.05 Hz a couple of seconds is a tenth of a cycle,
    so a plain correlation would report a perfectly recovered wave as unrelated — which is exactly what it did
    while `EnvelopeWave` was averaging over a quarter of a period.
    """
    period = 1.0 / hz
    best = (-2.0, 0.0)
    for lag in np.arange(0.0, period, period / 60):
        c = float(np.corrcoef(values, np.sin(2 * np.pi * hz * (ts - lag)))[0, 1])
        if c > best[0]:
            best = (c, float(lag))
    return best


# --- the metric ------------------------------------------------------------------------------

def test_the_band_is_a_lowpass_at_twice_the_centre_and_stops_at_one_hertz():
    """Their `Out3 = centre * 2`, and the refusal above `MAX_HZ`.

    The cap is the load-bearing half. Above about 1 Hz a cycle is under a second, so there is no position
    anybody can follow — and the tempting thing to feed back instead is a band's *amplitude*, whose swing is
    slow enough to pass for the same quantity while being a size rather than a position. The metric refuses
    the frequency rather than quietly measuring something else, which is what an earlier version did.
    """
    assert wave_edges(0.05) == (0.0, 0.1), "constant-Q, and no lower edge at all"
    assert wave_edges(1.0) == (0.0, 2.0)
    assert MAX_HZ == 1.0

    assert SignedWave(SPEC, MAX_HZ).high == 2.0
    for hz in (1.01, 10.0, 40.0):
        with pytest.raises(ValueError) as raised:
            SignedWave(SPEC, hz)
        assert "follow" in str(raised.value), "say why, not just that"
    with pytest.raises(ValueError):
        SignedWave(SPEC, 0.0)


def test_the_drift_leaves_the_signal_and_the_wave_does_not():
    """A 20 µV wave at 0.05 Hz under 300 µV of ramping drift — the measured case on this hardware.

    Both halves matter. The wave has to survive the de-trender (its corner is an eighth of the frequency, so
    99% of it should), and the *ramp* has to go, which is the part a single pole cannot do.
    """
    t = np.arange(int(600 * RATE)) / RATE
    wave = 20.0 * np.sin(2 * np.pi * 0.05 * t)
    metric = SignedWave(SPEC, 0.05)
    got = _feed(metric, wave + 300.0 * t / t[-1])
    ts, values = got[:, 0], got[:, 1]
    settled = ts > metric.settle_seconds
    assert settled.sum() > 100, "the run has to outlast the settling for this to say anything"
    assert abs(values[settled].mean()) < 1.0, "a standing bias is drift that was not removed"
    assert values[settled].std() * np.sqrt(2) == pytest.approx(20.0, rel=0.15)
    corr, lag = _best_lag_corr(ts[settled], values[settled], 0.05)
    assert corr > 0.99, f"the wave itself should come back, got {corr:.2f}"
    assert lag < 4.0, f"{lag:.1f} s of lag on a 20 s cycle is more than a filter's worth"
    assert metric.trend > 200.0, "and it should say how much it took out"


def test_one_pole_of_de_trending_would_have_left_a_standing_bias():
    """Why `DETREND_POLES` is two, in the arithmetic rather than in a docstring alone.

    A first-order high-pass removes a constant completely and leaves `τ × slope` of a ramp. That is not a
    small residue at these settings: an eighth-of-the-frequency corner at 0.05 Hz is a 25 s time constant, and
    on the drift this hardware actually produces it stands 19 µV off zero — half of a calibrated ±40 µV range.
    """
    assert DETREND_POLES >= 2
    tau = detrend_tau(0.05)
    assert tau == pytest.approx(DETREND_RATIO / (2 * np.pi * 0.05), rel=1e-6)

    slope = 300.0 / 400.0          # µV per second, the measured drift over a session
    t = np.arange(int(400 * RATE)) / RATE
    ramp = slope * t
    one_pole = OnePole(tau, RATE)
    residue = ramp - one_pole.process(ramp)
    assert residue[-1] == pytest.approx(tau * slope, rel=0.05)
    assert residue[-1] > 15.0, "the bias one pole leaves is not a rounding error"


def test_nothing_claims_to_be_settled_before_the_de_trender_is():
    """Settling is the band *and* the drift follower, and the second one dominates at the slow end.

    A screen that waited only for the band would feed back the drift it has not finished measuring — which is
    what `warmup` exists to prevent, so it has to include both terms.
    """
    metric = SignedWave(SPEC, 0.05)
    assert metric.warmup == 0.0
    assert metric.settle_seconds > 3 * detrend_tau(0.05), "the de-trender is in the figure"
    _feed(metric, np.zeros(int(30 * RATE)))
    assert 0.0 < metric.warmup < 1.0, "thirty seconds is not settled at 0.05 Hz"
    # The second feed starts past a deliberate gap, which resets the filters *and* the settling clock,
    # so the full settling time is needed again from there — plus a block for the rounding. Device-sized
    # blocks here on purpose: the offsets and that spare block are counted in them.
    _feed(metric, np.zeros(int(metric.settle_seconds * RATE) + 16), start=int(30 * RATE) + 8, step=8)
    assert metric.warmup == 1.0


def test_a_gap_resets_the_filters_rather_than_smearing_it_forward():
    """A hole in the stream is not a step in the signal, and the drift follower is where that matters most:
    carrying its state across missing samples would subtract a drift measured before an unknown interval.

    The settling clock restarts with the filters. It used not to — `samples_seen` kept counting across the
    reset, so `warmup` went on reading 1.0 while the output was pure transient, and the screen said
    "settled" over a reading that meant nothing. Note this only ever concerns *device* gaps: the artifact
    gate feeds a stand-in block for rejections precisely so a blink never lands in this branch.
    """
    metric = SignedWave(SPEC, 0.2)
    _feed(metric, 10.0 * np.ones(int(60 * RATE)))
    before = metric.samples_seen
    got = _feed(metric, 10.0 * np.ones(int(20 * RATE)), start=10 ** 6)
    assert metric.samples_seen < before, "the settling clock restarted with the filters"
    assert metric.warmup < 1.0, "twenty seconds after a reset is not settled at 0.2 Hz"
    assert np.all(np.abs(got[:, 1]) < 200.0), "no ring, no runaway"


# --- the screen ------------------------------------------------------------------------------

def _page():
    from test_screens import StubPage

    return StubPage()


def _toolbar():
    from test_screens import toolbar

    return toolbar()


def _session(channels: int = 2, source=None, **kwargs) -> wave_view.WaveSession:
    return wave_view.WaveSession(BY_CODE["WAVE"],
                                 source=source or ReplaySource(str(BENCH_OK), realtime=False,
                                                               loop=True),
                                 channels=channels, **kwargs)


def _pump(session, until=None, blocks: int = 200_000) -> None:
    """Drive the session's feed on this thread until `until` is true.

    No thread and no clock: `feed_block` is the body of the feed loop, so a test can be the loop. The
    thread-plus-sleep version measured the scheduler and paid 50 ms of granularity per check.
    """
    session.toggle_pause()
    for n, item in enumerate(session.source.blocks()):
        session.feed_block(item)
        if (until is not None and until()) or n >= blocks:
            break
    session._stop = True


def _texts(control, depth: int = 0):
    from test_screens import _texts as walk

    return walk(control, depth)


def _numbers(control):
    """Every typeable value on a control tree. A number you can set is a `TextField`, not a `Text`, which is
    the whole point of it being settable — see `test_screens._numbers`."""
    from test_screens import _numbers as walk

    return walk(control)


def test_the_screen_draws_before_any_data_and_says_what_it_is_waiting_for():
    """A blank pane is what a broken screen looks like, so the empty state is not optional.

    **It is now only the empty state.** It used to cover the settling and the calibration as well — minutes
    of a caption across the plot on a screen that was working — and both of those are states the wave is
    drawn during now, with the card's note carrying how settled the range is.
    """
    page, session = _page(), _session()
    try:
        control = wave_view.view(page, session, _toolbar(), lambda: None)
        session.redraw()
        assert session.error is None
        joined = " | ".join(c.value for c in _texts(control) if c.value)
        assert "waiting for the first reading" in joined
        assert "full scale" in joined and "across" in joined, "every graph states its window and scale"
    finally:
        session.stop()


def test_every_tab_draws_with_real_data_through_it():
    """The one thing object tests cannot do: produce a frame. Panes are held in a `Stack` with `visible`
    toggled, so the draw loop keeps writing to the hidden ones and a stale name in any of them throws."""
    page, session = _page(), _session()
    try:
        control = wave_view.view(page, session, _toolbar(), lambda: None)
        _pump(session, until=lambda: session.signal is not None)
        panes = control.controls[2].controls
        for i in range(len(panes)):
            for j, pane in enumerate(panes):
                pane.visible = i == j
            session.redraw()
        assert session.error is None, session.error
        joined = " | ".join(c.value for c in _texts(control) if c.value)
        assert "drift removed" in joined, "the caption has to say what is being taken out"
        assert "settling" not in joined, "it settled; the empty state should be gone"
    finally:
        session.stop()


def test_every_control_on_the_screen_can_be_pressed():
    """Press everything and draw afterwards.

    Every control here is a lambda closing over a session, and a wrong argument name or a stale attribute in
    one of them raises **out of a Flet event handler** — which surfaces as a traceback in the terminal and
    nothing at all on screen. That is the failure mode this repo has hit twice (`retune_filter` on a lock-in
    screen, and a rebuilt control list), and pressing them is the only way an object test can see it.
    """
    from nfb_console import normalise

    page, session = _page(), _session()
    try:
        control = wave_view.view(page, session, _toolbar(), lambda: None)
        handlers = _clickables(control)
        assert len(handlers) > 30, "the screen is supposed to have controls on it"
        for label, handler in handlers:
            if label in (None, "?", "browse…", "▶", "▲", "▼", "×"):
                continue          # dialogs, steppers and file pickers: not this test's business
            handler(None)
        session.redraw()
        assert session.error is None
        assert isinstance(session.strategy, normalise.Strategy), "still a working scaling"
    finally:
        session.stop()


def _clickables(control, depth: int = 0, out=None):
    import flet as ft

    out = [] if out is None else out
    if control is None or depth > 16:
        return out
    handler = getattr(control, "on_click", None)
    if handler is not None:
        content = getattr(control, "content", None)
        label = content.value if isinstance(content, ft.Text) else None
        out.append((label, handler))
    for attr in ("content", "controls", "actions"):
        value = getattr(control, attr, None)
        if isinstance(value, list):
            for item in value:
                _clickables(item, depth + 1, out)
        elif value is not None:
            _clickables(value, depth + 1, out)
    return out


def test_the_level_arrives_with_the_first_reading_and_not_after_the_calibration():
    """**Four and a half minutes of progress bars where a protocol should be.**

    The level was gated on the filters settling *and* the range being measured, which at 0.05 Hz is 2.9 min
    of "settling" followed by 90 s of "calibrating the range" with nothing fed back through either. The
    reading exists from the first block; what did not exist was a range somebody had blessed. So the range
    is computed from what has arrived and refined as more does, and the feedback runs the whole time.

    The gate that stays is `valid`, and it is a different kind: a railed reading is not a rough estimate of
    a wave, it is a reference electrode that has come off.
    """
    session = _session()
    try:
        assert session.signal is None and session.outputs == {}
        assert not session.settled and not session.calibrated
        _pump(session, until=lambda: session.signal is not None)
        assert session.signal is not None
        assert session.raw_wave is not None
        assert not session.settled, \
            "the point of the change: a level long before the filters are done"
        assert not session.calibrated, "and long before the range stops moving"
        assert session.outputs, "the mechanisms are being driven, not merely the number computed"
    finally:
        session.stop()


def test_the_range_is_in_use_while_it_is_still_being_refined():
    """The other half of the same rule: a provisional number that improves, not a placeholder.

    It is refined from a **rolling** window, so it tracks the estimate as the estimate gets better — an
    early range taken partly off the drift follower's ramp is replaced rather than averaged with.
    """
    session = _session()
    spec = session.specs[session.channel]
    try:
        first = np.arange(int(30 * RATE)) / RATE
        _feed_session(session, 5.0 * np.sin(2 * np.pi * 0.05 * first), spec)
        assert not session.calibrated, "30 s is nowhere near the window"
        early = session.range_uv
        assert early > 0.0, "a range is in use from the first reading"

        rest = np.arange(int(400 * RATE)) / RATE
        _feed_session(session, 40.0 * np.sin(2 * np.pi * 0.05 * rest), spec,
                      start=int(30 * RATE))
        assert session.range_uv > early, "the window rolled; the range followed the signal into it"
    finally:
        session.stop()


def test_the_range_is_calibrated_from_the_signal_and_then_holds_still():
    """A calibration is a phase, not an auto-range — which is the one place this differs from the reference.

    The range comes out at `RANGE_SIGMAS` times the RMS actually measured, and then it stops moving: a scale
    that kept following the signal would report the trainee's variance rather than their state, which is the
    argument `music.ShepardPlayer` is built on.
    """
    session = _session()
    try:
        spec = session.specs[session.channel]
        t = np.arange(int(400 * RATE)) / RATE
        _feed_session(session, 20.0 * np.sin(2 * np.pi * 0.05 * t), spec)
        assert session.calibrated, "400 s of signal is past the settling and the calibration"
        # a 20 µV sine is 14.1 µV RMS, so two sigmas is about 28
        assert session.range_uv == pytest.approx(2.0 * 20.0 / np.sqrt(2), rel=0.25)
        held = session.range_uv
        _feed_session(session, 60.0 * np.sin(2 * np.pi * 0.05 * t), spec,
                      start=int(400 * RATE))
        assert session.range_uv == held, "tripling the signal must not move the scale"
    finally:
        session.stop()


def _feed_session(session, x: np.ndarray, spec, start: int = 0) -> None:
    """Push a known waveform through the session's own `consume`, as the feed thread would."""
    ints = counts(x, spec.scale, truncate=True)
    for i in range(0, len(x) - 7, 8):
        block = Block(spec.index, start + i, tuple(ints[i:i + 8].tolist()), spec)
        session.consume(block)


def test_the_level_follows_the_wave_and_the_condition_is_the_crossing():
    """The claim the whole screen rests on, driven through the same path the feed thread takes.

    Position, not size: the level should be near the top when the wave is at its peak, near the bottom at its
    trough, and the condition — which every discrete output follows — should be exactly the crossing of the
    wave's own baseline.
    """
    from nfb_console.feedback import CONDITION

    session = _session()
    try:
        spec = session.specs[session.channel]
        session.state[session.channel] = True          # the bench channels report no signal
        session.toggle_pause()
        session.feedback.smoothing_seconds = 0.0       # read the level, not its smoother
        t = np.arange(int(500 * RATE)) / RATE
        _drive_session(session, 20.0 * np.sin(2 * np.pi * 0.05 * t), spec)
        assert session.calibrated

        seen: list[tuple[float, float, bool]] = []
        base = int(500 * RATE)               # block-aligned: contiguous with the drive, no gap
        for cycle in range(2):
            start = base + cycle * int(20 * RATE)
            wave = 20.0 * np.sin(2 * np.pi * 0.05 * (np.arange(int(20 * RATE)) / RATE))
            ints = counts(wave, spec.scale, truncate=True)
            for i in range(0, len(wave) - 7, 8):
                block = Block(spec.index, start + i, tuple(ints[i:i + 8].tolist()), spec)
                fresh = session.consume(block)
                for tgt in session.targets:
                    if session._due(tgt, fresh):
                        tgt.observe(session.value_for(tgt), 20.0)
                        session._advance_level()
                if session.raw_wave is not None:
                    seen.append((session.raw_wave, session.level,
                                 session.feedback.on_target(session.channel_targets(CONDITION))))
        values = np.array([v for v, _lvl, _met in seen])
        levels = np.array([lvl for _v, lvl, _met in seen])
        met = np.array([m for _v, _lvl, m in seen])
        assert float(np.corrcoef(values, levels)[0, 1]) > 0.95, "the level is the position"
        assert levels.max() > 0.8 and levels.min() < 0.2, "and it uses the calibrated range"
        # The condition is the crossing, so it holds on the positive half and not on the negative one.
        assert met[values > 5.0].all()
        assert not met[values < -5.0].any()
    finally:
        session.stop()


def _drive_session(session, x: np.ndarray, spec, start: int = 0) -> None:
    """`_feed_session`, plus the target observation and level advance `TrainingSession.run` does."""
    ints = counts(x, spec.scale, truncate=True)
    for i in range(0, len(x) - 7, 8):
        block = Block(spec.index, start + i, tuple(ints[i:i + 8].tolist()), spec)
        fresh = session.consume(block)
        for tgt in session.targets:
            if session._due(tgt, fresh):
                tgt.observe(session.value_for(tgt), 20.0)
                session._advance_level()


def test_the_aim_belongs_to_the_operator_because_polarity_is_arbitrary():
    """On a signed wave "increase" is not a clinical aim: in a bipolar pair which way is up is decided by
    which electrode went in which socket. So it is a control, and flipping it mirrors the level."""
    session = _session()
    try:
        spec = session.specs[session.channel]
        t = np.arange(int(400 * RATE)) / RATE
        _feed_session(session, 20.0 * np.sin(2 * np.pi * 0.05 * t), spec)
        up = session.signal
        assert up is not None
        session.set_direction("decrease")
        assert session.inverts and session.signal is None, "the old signal meant the opposite"
        _feed_session(session, np.full(int(4 * RATE), 0.0), spec, start=int(400 * RATE))
        down = session.signal
        assert down is not None
        assert "below its own baseline" in session.level_source[0]
    finally:
        session.stop()


def test_changing_what_is_measured_restarts_the_settling_and_the_calibration():
    """Every one of these changes the quantity, so a range calibrated against the old one would be a scale
    for a different signal — the same reason the synchrony screen rebuilds its reward rather than editing it."""
    session = _session()
    try:
        spec = session.specs[session.channel]
        t = np.arange(int(400 * RATE)) / RATE
        _feed_session(session, 20.0 * np.sin(2 * np.pi * 0.05 * t), spec)
        assert session.calibrated
        for change in ({"hz": 0.2}, {"order": 1}, {"hz": 0.5}):
            _feed_session(session, 20.0 * np.sin(2 * np.pi * 0.05 * t), spec)
            assert session.set_frequency(**change), change
            assert not session.calibrated and session.signal is None, change
            assert not session.settled, f"{change} rebuilt the filters, so they are unsettled"
    finally:
        session.stop()


def test_the_slow_branch_says_when_the_devices_corner_is_the_only_lower_edge():
    """In the lowpass branch there is no high-pass at all, so the amplifier's corner becomes the band's lower
    edge — and at 0.5 Hz it sits *above* a band that stops at 0.1 Hz, so there is no band left.

    First order here, unlike the synchrony screen's product, which loses it squared: this is one channel.
    """
    bad = _session(source=WaitingSource(Settings(channels=2, lower_limit_hz=0.5).build()))
    try:
        problem = bad.wiring_problem()
        assert problem and "0.01 Hz" in problem and "corner" in problem
        assert "%" in problem, "say how much survives, not merely that something is wrong"
        assert len(problem) < 220, "it sits in a narrow column; the arithmetic is behind the ?"
    finally:
        bad.stop()
    good = _session(source=WaitingSource(Settings(channels=2, lower_limit_hz=0.01).build()))
    try:
        assert good.wiring_problem() is None
    finally:
        good.stop()


def test_the_plots_scale_is_not_the_feedbacks():
    """They were one number, and that made a display choice change what the mechanisms mapped over.

    Three scales exist on this screen and each has its own owner: the plot's height (pills, and `fit`), the
    feedback range (calibrated, edited beside the scaling), and the raw trace's gain. Moving any one of them
    must leave the others where they were.
    """
    session = _session()
    try:
        spec = session.specs[session.channel]
        t = np.arange(int(400 * RATE)) / RATE
        _feed_session(session, 20.0 * np.sin(2 * np.pi * 0.05 * t), spec)
        assert session.calibrated
        calibrated = session.feedback_range_uv
        assert session.wave_uv != calibrated, "the calibration must not have moved the plot"

        session.wave_uv = 100.0
        assert session.feedback_range_uv == calibrated, "a plot preset is not a feedback range"
        assert session.strategy.push(calibrated, 0.0) == pytest.approx(1.0), \
            "and the level still maps over the calibrated ends"

        session.set_range(calibrated * 3)
        assert session.wave_uv == 100.0, "and a feedback range is not a plot preset"
        assert session.gain_uv == 50.0, "nor the raw trace's gain"
    finally:
        session.stop()


def test_the_signal_pane_is_a_card_and_its_bar_is_signed():
    """It borrows `target_card`'s layout and not its class, and the bar is where that shows.

    A signed value on the shared bar is a negative width, so this one is centre-zero and fills either way.
    The threshold row still exists, because a level crossing has one — it is the baseline, stated rather than
    offered, since the de-trender has already put it at zero and nudging zero by 10% leaves it at zero.
    """
    page, session = _page(), _session()
    try:
        wave_view.view(page, session, _toolbar(), lambda: None)
        card = wave_view.WaveCard(page, session, lambda: None)
        card.refresh()          # the scale line is written per frame, like every other card's
        joined = " | ".join(c.value for c in _texts(card.control) if c.value)
        assert "threshold" in joined and "the baseline" in joined
        # The row says what the number *does* rather than naming the quantity from the inside.
        assert "full at ±" in joined and "µV" in joined
        assert "0% at" in joined and "50% at the baseline" in joined, "the bar states the mapping"
        middle = wave_view.BAR_W / 2

        spec = session.specs[session.channel]
        t = np.arange(int(400 * RATE)) / RATE
        _feed_session(session, 20.0 * np.sin(2 * np.pi * 0.05 * t), spec)
        session.raw_wave = session.feedback_range_uv / 2
        card.refresh()
        assert card.fill.left == middle and card.fill.width > 0, "positive fills to the right"
        session.raw_wave = -session.feedback_range_uv / 2
        card.refresh()
        assert card.fill.left < middle and card.fill.width > 0, "negative fills to the left"
        assert card.fill.left + card.fill.width == pytest.approx(middle)
        session.raw_wave = session.feedback_range_uv * 10
        card.refresh()
        assert card.fill.width <= middle, "and an excursion past the range cannot run off the bar"
    finally:
        session.stop()


def test_the_card_fits_its_column_and_the_range_is_on_it():
    """Three faults from one screenshot, and each of them is checkable.

    The identity row had four channel pills and two order pills either side of the frequency box, in a 380 px
    column: the order ran off the right-hand edge, where a control that is present and invisible is worse than
    one that is absent. A channel is a menu now and the order is behind the gear — the same convention every
    other screen's per-target filter settings follow — which leaves the row short enough to fit.

    And the range went back onto the card. It had been moved to the Feedback tab's scaling to avoid two
    editors for one number, which achieved that and left the operator with no way to reach it from the tab
    they were looking at the wave on. It is here and stated there instead.
    """
    page, session = _page(), _session(channels=4)
    try:
        card = wave_view.WaveCard(page, session, lambda: None)
        card.refresh()
        assert len(card.identity.controls) <= 6, "a long identity row is one that runs off the card"
        labels = [c.value for c in _texts(card.control) if c.value]
        assert not any(x.startswith("order") for x in labels), "the order is behind the gear"
        assert "⚙" in labels

        # Two typeable numbers on the card: the frequency and the range.
        typeable = _numbers(card.control)
        assert f"{session.frequency_hz:g}" in typeable
        assert f"{session.feedback_range_uv:g}" in typeable, "the range is editable here"
        card._set_range(35.0)
        assert session.feedback_range_uv == 35.0
        assert session.strategy.push(35.0, 0.0) == pytest.approx(1.0), "and it reaches the scaling"

        # The gear carries the order and what choosing it costs.
        card.identity.controls[-1].on_click(None)
        dialog = " | ".join(c.value for c in _texts(page.dialogs[-1].content) if c.value)
        assert "order" in dialog and "reacts in" in dialog and "drift follower" in dialog
    finally:
        session.stop()


def test_the_two_microvolt_scales_say_which_they_are():
    """Both are ±µV on one screen — the plot's height and the feedback's range — so the words have to
    distinguish them. They were the same number once, and the operator's objection to that is why they are
    not; nothing in the pixels says so, only the captions."""
    page, session = _page(), _session()
    try:
        control = wave_view.view(page, session, _toolbar(), lambda: None)
        session.redraw()
        captions = [c.value for c in _texts(control) if c.value]
        assert any("full scale, this plot's own" in x for x in captions)
        assert any("full at" in x for x in captions)
        assert any(x == "full at ±" for x in captions), "and the card's row says what it does"
        # And the plot says *which* signal it is drawing, as the synchrony screen's does.
        assert any("Hz wave" in x for x in captions)
    finally:
        session.stop()


def test_the_catalogue_offers_it_and_a_screen_handles_it():
    """The launcher lists what can actually be run. A protocol with no screen behind it is a card that fails
    after you click it, which is what `Protocol.available` exists to prevent."""
    from nfb_console.app import Console

    protocol = BY_CODE["WAVE"]
    assert protocol.available and protocol.name == "Waveform"
    assert Console.screen_for(protocol) is wave_view


def test_the_calibration_survives_a_slow_excursion_the_gate_cannot_see():
    """The artifact gate judges on 0.5-50 Hz; this protocol trains below 0.1. An in-band excursion — a
    movement, an electrode pop, tens of seconds long — therefore passes the gate untouched, and a range
    calibrated as a *mean* was owned by it: one 300 µV / 20 s event measured the range at 5.3× its true
    width, after which the feedback was nearly motionless for the whole session. The median refuses to be
    owned by any minority of the window, and for the sinusoid this metric measures it equals the mean."""
    session = _session()
    try:
        spec = session.specs[session.channel]
        t = np.arange(int(400 * RATE)) / RATE
        wave = 20.0 * np.sin(2 * np.pi * 0.05 * t)
        dirty = wave.copy()
        at = int(300 * RATE)                          # inside the calibration window, past settling
        dirty[at:at + int(20 * RATE)] += 300.0
        _feed_session(session, dirty, spec)
        assert session.calibrated
        assert session.range_uv == pytest.approx(2.0 * 20.0 / np.sqrt(2), rel=0.3), \
            f"the excursion owned the range: ±{session.range_uv:.0f} µV"
    finally:
        session.stop()


def test_a_railed_channel_neither_calibrates_nor_steers_the_level():
    """Device-invalid blocks bypass the gate by design (`pass_through`), so their readings arrive here
    marked invalid — and used to be accumulated anyway. A reference electrode lifting during the
    measurement window set the range from the ±800 µV rail; lifting later steered the level with it."""
    session = _session()
    try:
        spec = session.specs[session.channel]
        t = np.arange(int(400 * RATE)) / RATE
        wave = 20.0 * np.sin(2 * np.pi * 0.05 * t)
        ints = counts(wave, spec.scale, truncate=True)
        rail_from = int(250 * RATE)                   # rails mid-calibration
        for i in range(0, len(wave) - 7, 8):
            railed = i >= rail_from
            samples = (32767,) * 8 if railed else tuple(ints[i:i + 8].tolist())
            session.consume(Block(spec.index, i, samples, spec, state=2 if railed else 0))
        if session.calibrated:
            assert session.range_uv < 100.0, f"calibrated on the rail: ±{session.range_uv:.0f} µV"
    finally:
        session.stop()


def test_the_slow_spectrum_fills_in_from_the_right_instead_of_showing_nothing():
    """**The old gate was unreachable, so this pane was empty for the whole session, every session.**

    `psd` returned None until `filled_seconds` reached two cycles of the lowest frequency asked for — 400 s
    at 0.005 Hz — while `filled_seconds` saturates at `window_seconds`, which this screen sets to 300. The
    condition could not be met however long anyone sat there, and the caption underneath promised 6.7 min.

    Underneath that is the rule the whole screen now follows: a gate may say a value cannot be computed, and
    may not say a value is not accurate enough yet. The lowest bin really does need a longer window than
    this ring holds — so that bin stays empty, on its own, because no frequency falls inside it. The other
    twenty-three arrive as soon as the window resolves them.
    """
    from nfb_console.traces import SlowRing
    from nfb_console.wave_view import SLOW_BINS, SLOW_HI_HZ, SLOW_LO_HZ, SLOW_WINDOW_SECONDS

    assert SLOW_WINDOW_SECONDS < 2.0 / SLOW_LO_HZ, \
        "the ring cannot hold two cycles of its own lowest bin — which is what made the old gate forever"

    ring = SlowRing(SLOW_WINDOW_SECONDS, RATE, SLOW_LO_HZ, SLOW_HI_HZ)
    t = np.arange(int(600 * RATE)) / RATE
    x = 20.0 * np.sin(2 * np.pi * 0.05 * t) + 5.0 * np.sin(2 * np.pi * 0.3 * t)

    drawn = {}
    for i in range(0, len(x), int(RATE)):
        ring.push(x[i:i + int(RATE)])
        at = (i + int(RATE)) / RATE
        for mark in (10, 60, 300):
            if mark not in drawn and at >= mark:
                drawn[mark] = int((ring.bins(SLOW_BINS)[1] > 0).sum())

    assert drawn[10] > 0, "ten seconds resolves the top of the range perfectly well"
    assert drawn[60] > drawn[10], "and the picture fills in as the window grows"
    assert drawn[300] > drawn[60]
    assert drawn[300] < SLOW_BINS, "the bins it genuinely cannot resolve stay empty rather than being faked"
