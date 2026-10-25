"""ILF Synchrony: the measure, the frequency ladder, the bands, and the Shepard tone.

Every number pinned here is a design decision, and each test says why — because the value of pinning
them is that a later change has to argue with the reason rather than with a magic number. Three of them
are choices made on this project's own measurements: the 1 Hz crossover between the two bandwidth rules,
the four-cycle floor under the auto-threshold window, and the Shepard tone's envelope width.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from nfb import ReplaySource, Settings, WaitingSource
from nfb.dsp import ProductSynchrony, SynchronyMetric, Reward, is_lowpass
from nfb.stream import Block, ChannelSpec
from nfb_console import ilf_view, traces, widgets
from nfb_console.feedback import LEVEL, STATE, Music
from nfb_console.plots import nice_ceiling
from nfb_console.protocols import BY_CODE

from conftest import BENCH
from test_screens import StubPage, _numbers, toolbar

RATE = 250.0
RECORDING = str(BENCH)      # see conftest: a generated bench recording, not sessions/


SCALE = 0.0244140625
"""The device's real µV-per-count on its 800 µV range. These specs carried `scale=1.0` for a long time,
which made `samples` and `values` the same numbers — and so hid a bug where the scale was applied twice
and every ILF reading came out 41× small (the product, a square, 1677×). Blocks here are built in counts,
as they arrive from the device, so the scaling path is what the assertions exercise."""


def specs():
    return {0: ChannelSpec(index=0, rate_hz=RATE, unit="uV", scale=SCALE),
            1: ChannelSpec(index=1, rate_hz=RATE, unit="uV", scale=SCALE)}


BLOCK = 250
"""How many samples a test block carries, and it is **not** the device's eight.

The metrics emit readings on a fixed sample hop, not per block: 60 s of signal gives 1250 readings at a
block of 8, 250 or 1000, with identical timestamps and identical values to five figures. So the block
size is an input-chunking detail here, and choosing the device's eight cost a factor of nineteen — a
`sosfilt` call on eight samples is 18.8 µs of which 18.7 is scipy validating its arguments, and the
0.05 Hz case paid that 150,000 times to reach an answer it also reaches in 300.

Where blocking *is* the subject — a gap, a held block, an index discontinuity — pass `step=8` and mean
it."""


def feed(metric, a, b, chans, step: int = BLOCK):
    # Divided once for the whole array rather than once per sample. Note these stay *floats*, as this
    # file's blocks always have: the scaling path is what the assertions exercise here.
    ca, cb = np.asarray(a) / SCALE, np.asarray(b) / SCALE
    out = []
    for i in range(0, len(a) - (step - 1), step):
        out += metric.feed(Block(0, i, tuple(ca[i:i + step].tolist()), chans[0]))
        out += metric.feed(Block(1, i, tuple(cb[i:i + step].tolist()), chans[1]))
    return out


def settled(readings, n: int = 100, name: str = "product") -> float:
    """The mean of the last `n` readings of one kind. The metric emits **two** per stride — the product and
    the summed amplitude — so a bare tail average would interleave microvolts squared with microvolts."""
    picked = [r.value for r in readings if r.name.startswith(name)]
    return float(np.mean(picked[-n:])) if picked else 0.0


# --- the measure ----------------------------------------------------------------------------

@pytest.mark.parametrize("hz,seconds,tail", [(10.0, 120, 100), (0.05, 300, 1200)])
def test_the_product_is_zero_in_antiphase_and_maximal_in_phase(hz, seconds, tail):
    """`(In1*In2)>0?(In1*In2):0`, off the synchrony dialog's own `pos(x * y)` element.

    Two 20 µV sinusoids in phase multiply to a mean of ½·20·20 = 200 µV²; in antiphase every sample of the
    product is negative and the rectifier takes it to exactly zero. That is the property the measure exists
    for, and it is the one a summed reward does not have — `sum` also reads zero in antiphase, but only
    because the two amplitudes happen to be equal, where this reads zero at any amplitudes.
    """
    # **The tail has to span several of the product's own 2f ripple periods**, and at 0.05 Hz that is 10 s
    # each — the 100 ms post filter removes none of it down there, so a five-second average lands anywhere
    # between 0 and twice the mean. Measured at a 5 s tail: 312 against a mean of 188.
    chans = specs()
    n = int(seconds * RATE)
    tt = np.arange(n) / RATE
    wave = 20 * np.sin(2 * np.pi * hz * tt)
    in_phase = feed(ProductSynchrony(chans, (0, 1), hz), wave, wave, chans)
    anti = feed(ProductSynchrony(chans, (0, 1), hz), wave, -wave, chans)
    assert settled(in_phase, tail) == pytest.approx(200.0, rel=0.15)
    assert settled(anti, tail) == pytest.approx(0.0, abs=0.5)


def test_antiphase_reads_zero_even_when_the_two_channels_are_unbalanced():
    """The reason to prefer a product over a sum, stated as a measurement.

    `sum` is `|z_a + z_b|`, so two antiphase channels at 40 and 10 µV leave 30 µV standing and the reward
    reports it. The product is negative at every sample whatever the amplitudes, so it rectifies to zero.
    """
    chans = specs()
    n = int(60 * RATE)
    tt = np.arange(n) / RATE
    a = 40 * np.sin(2 * np.pi * 10 * tt)
    b = 10 * np.sin(2 * np.pi * 10 * tt + np.pi)
    product = settled(feed(ProductSynchrony(chans, (0, 1), 10.0), a, b, chans))
    summed = settled(feed(SynchronyMetric(chans, (0, 1), Reward(10.0, 5.0, "sum")), a, b, chans),
                     name="sync")
    assert product == pytest.approx(0.0, abs=0.5)
    assert summed == pytest.approx(30.0, rel=0.1), "the sum still sees the imbalance as signal"


def test_the_product_reports_its_settling_and_it_is_the_transient_not_the_delay():
    """**The step transient outlives the group delay by an order of magnitude, and the plot needs the
    former.** At the ¼–2× default on 0.05 Hz the cascade's group delay is 3.84 s while the measured step
    transient lasts 66 s (see `SLOW_BANDS`) — and that transient, in µV² against a signal in the ones,
    pinned the synchrony plot's auto scale at 50000 µV² for the ten minutes it took to scroll out.
    `settle_seconds` is what lets the screen keep it off the plot; `warmup` is how it asks.
    """
    chans = specs()
    slow = ProductSynchrony(chans, (0, 1), 0.05)
    assert slow.settle_seconds > 10 * slow.delay_seconds, \
        "settling is set by the corners' time constants, not by the delay at the centre"
    assert 60.0 < slow.settle_seconds < 100.0, "the conservative side of the measured 66 s"
    assert slow.warmup == 0.0
    n = int(slow.settle_seconds * RATE) + RATE
    tt = np.arange(n) / RATE
    wave = 8 * np.sin(2 * np.pi * 0.05 * tt)
    feed(slow, wave, wave, chans)
    assert slow.warmup == 1.0
    # The fast branch settles on its bandwidth: under a second for a 3 Hz band, so the plot never waits.
    assert ProductSynchrony(chans, (0, 1), 10.0).settle_seconds < 1.0


def test_a_product_reward_reports_squared_microvolts():
    """A product of two voltages is not an amplitude, and a card printing µV over it would be lying. Every
    threshold here is derived from data so nothing breaks — but the unit has to say which quantity it is."""
    assert ProductSynchrony(specs(), (0, 1), 10.0).unit == "µV²"


def test_the_filter_becomes_a_bare_lowpass_when_the_lower_edge_would_fall_past_zero():
    """**The fork, and it is the whole protocol at infra-slow frequencies.**
    The filter is a lowpass when the lower edge would fall past zero, with a fixed bandwidth of 3 Hz. So a 0.05 Hz centre is a
    0–1.55 Hz lowpass and the reward is the product of two slow *waveforms* — ILF on the raw waveform, not a
    narrow band at 0.05 Hz.

    Everything that looks like a problem under the narrow-band reading — three independent values a minute
    behind a threshold, an epoch holding half a cycle, a bandwidth reaching below DC — is an artefact of that
    reading. Pinned because getting it wrong produced a whole layer of machinery to work around a situation
    that does not arise, and the lowpass version is also **48× faster** at 0.05 Hz.
    """
    # the fork carries a 0.3 Hz margin, so with a 3 Hz bandwidth it trips at 1.8 Hz and not at 1.5
    # 1.8 exactly is not asserted: `1.8 - 1.5 - 0.3` is 1.1e-16 in float64, so the boundary itself is decided
    # by rounding rather than by the design.
    assert is_lowpass(0.05, 3.0) and is_lowpass(1.7, 3.0)
    assert not is_lowpass(1.9, 3.0) and not is_lowpass(10.0, 3.0)
    # The `0–2×` band, still available and still exactly what it was: no lower edge, and the lowpass
    # corner is **twice the centre**, not centre + bandwidth/2.
    slow = ProductSynchrony(specs(), (0, 1), 0.05, slow_band="0–2×")
    assert slow.lowpass_only and (slow.low, slow.high) == (0.0, 0.1)
    assert "no high-pass" in slow.describe() and "twice the centre" in slow.describe()
    alpha = ProductSynchrony(specs(), (0, 1), 10.0)
    assert (alpha.low, alpha.high) == (8.5, 11.5), "a 3 Hz bandwidth at the default centre"
    assert not alpha.lowpass_only


# --- the frequency ladder -------------------------------------------------------------------

def test_the_frequency_grid_is_finer_at_the_top():
    """`round to 0.05 increments`: free below 0.01 Hz, hundredths below 0.1, twentieths below 1, quarters
    above. One function serves the arrows and the snapping so the two cannot disagree.

    **The top rung is 0.1 Hz because a quarter of a hertz cannot place a reward on an individual alpha or
    gamma peak.** `COARSE_GRID_HZ` carries the reasoning; the three slower rungs are untouched, so
    infra-slow work is unaffected.
    """
    assert ilf_view.grid_hz(0.005) == 0.001
    assert ilf_view.grid_hz(0.05) == 0.01
    assert ilf_view.grid_hz(0.5) == 0.05, "twentieths below 1 Hz"
    assert ilf_view.grid_hz(10.0) == widgets.COARSE_GRID_HZ == 0.1

    assert ilf_view.snap_hz(9.87) == 9.9, "tenths above 1 Hz"
    assert ilf_view.snap_hz(0.037) == 0.04, "hundredths below 0.1 Hz"
    assert ilf_view.snap_hz(0.0001) == 0.0001, "left alone below 0.01 Hz"

    # The arrows and the snap have to come from the same rung, or a step moves the box and springs back.
    for hz in (0.005, 0.05, 0.5, 10.0, 40.0):
        stepped = ilf_view.snap_hz(hz + ilf_view.grid_hz(hz))
        assert stepped != pytest.approx(ilf_view.snap_hz(hz)), f"a step at {hz} Hz has to move it"


# --- the bands and the aims -----------------------------------------------------------------

def test_the_inhibit_ladder_is_contiguous_and_covers_the_whole_range():
    """Nine bands. Two of the nine are filled in by contiguity — which only works if the other seven
    establish it, so contiguity is the assertion that licenses the construction."""
    assert len(ilf_view.INHIBITS) == 9
    for (_, high), (low, _) in zip(ilf_view.INHIBITS, ilf_view.INHIBITS[1:]):
        assert high == low, "no gaps and no overlaps"
    assert ilf_view.INHIBITS[-1][1] == 40.0
    assert ilf_view.INHIBITS[0][0] > 0, "a band reaching DC is not a band"


def test_the_reward_aim_is_ten_percent_and_the_inhibit_aim_ninety_five():
    """The reward aim is 10 and the inhibit aim is 95. The 10 would look like a mistake without a reason:
    the reward is deliberately rare, and a rarer reward is a stronger event."""
    assert ilf_view.REWARD_AIM == 0.10
    assert ilf_view.INHIBIT_AIM == 0.95


def test_a_rectified_product_is_not_fitted_with_a_rayleigh():
    """About half a rectified product's mass is an atom at exactly zero, so the one-parameter envelope fit
    that stands in for a percentile on narrow bands describes the wrong distribution. It must not be
    reached, whatever the sample count says — see `Target.narrowband_envelope`."""
    from nfb_console.targets import Target

    from nfb.dsp.filters import Band

    narrow = Band(0.025, 0.075, "")
    product = Target(-2, narrow, "increase", hit_rate_override=ilf_view.REWARD_AIM,
                     narrowband_envelope=False)
    envelope = Target(-2, narrow, "increase", hit_rate_override=ilf_view.REWARD_AIM)
    window = ilf_view.EPOCH_SECONDS
    assert product.derivation(window) == "percentile", "the fit may not stand in for one here"
    assert envelope.derivation(window) == "median", "where it is a real envelope, it may"
    # And with the faithful lowpass band there are plenty of values anyway, so nothing has to be excused.
    wide = Band(0.0001, 1.55, "")
    assert Target(-2, wide, "increase", hit_rate_override=ilf_view.REWARD_AIM,
                  narrowband_envelope=False).tail_events(window) > 4.0


def test_the_ratchet_mode_is_gone_and_stays_gone():
    """It was built on a misreading — `Reset, if transition from 100%` takes the *success-rate setting* as
    its input, not the signal, so it is an operator recalibrate and not a signal-driven ratchet. It also
    deadlocked: a target met on its first sample cleared its window every sample and never derived a
    threshold at all, so three rewards sat at the 10.0 default while reading 188, 3637 and 31 794.

    Pinned as a test because the idea is attractive and the evidence for it was published in two docs.
    """
    from nfb_console import targets

    assert targets.MODES == ("fixed", "automatic")
    assert not hasattr(targets, "RATCHET")
    assert not any(t.mode == "ratchet" for t in _session().targets)


# --- the session ----------------------------------------------------------------------------

def _stub_device(reference: str = "bipolar", lower_limit_hz: float = 0.01):
    """Just enough device for `wiring_problem`, which is the only thing on this screen that reads one."""
    settings = type("S", (), {"reference": reference, "lower_limit_hz": lower_limit_hz, "hz": RATE})()
    status = type("St", (), {"settings": settings, "connected": True})()
    # `stop` because `TrainingSession.stop` calls it on the way out, and a stub that only satisfies the
    # method under test makes the teardown the failure instead of the assertion.
    return type("Dev", (), {"status": status, "stop": lambda self: None})()


def _session(channels: int = 2, loop: bool = False):
    return ilf_view.ILFSession(BY_CODE["ILF"],
                               source=ReplaySource(RECORDING, realtime=False, loop=loop),
                               channels=channels)


def _session_at_corner(corner_hz: float, channels: int = 2):
    """A session whose high-pass corner is **stated rather than inherited**.

    `_session` above reads `sessions/latest.jsonl.gz`, which is gitignored and is whatever the operator
    recorded last — so once `corner_hz` started reading the corner from the source (correctly), a capture
    made at 0.01 Hz silently changed what the corner tests were testing. A config built here carries the
    corner the test is about and nothing else can move it.
    """
    config = Settings(channels=channels, hz=RATE, reference="bipolar",
                      lower_limit_hz=corner_hz).build()
    return ilf_view.ILFSession(BY_CODE["ILF"], source=WaitingSource(config), channels=channels)



def _all_text(x, d: int = 0) -> list:
    import flet as ft
    if x is None or d > 25 or getattr(x, "visible", True) is False:
        return []
    if isinstance(x, ft.Text):
        return [x.value or ""]
    out = []
    for c in (getattr(x, "controls", None) or []):
        out += _all_text(c, d + 1)
    if getattr(x, "content", None) is not None:
        out += _all_text(x.content, d + 1)
    return out


def _find_trace_row(x, label: str, d: int = 0):
    """The clickable container holding a channel's dot and label in the RAW pane."""
    import flet as ft
    if x is None or d > 25:
        return None
    if isinstance(x, ft.Container) and getattr(x, "on_click", None) is not None:
        if any(v.startswith(label) for v in _all_text(x)):
            return x
    for c in (getattr(x, "controls", None) or []):
        found = _find_trace_row(c, label, d + 1)
        if found is not None:
            return found
    if getattr(x, "content", None) is not None:
        return _find_trace_row(x.content, label, d + 1)
    return None


def test_the_session_builds_two_rewards_and_nine_bands_on_the_sum():
    """**One instrument, not three.** The measure takes a single reward frequency, so the three named frequencies
    are settings of one reward rather than three rewards. An earlier version instantiated three concurrently."""
    session = _session()
    try:
        # **Two** rewards: the product, and the band amplitude on the summed channels. The amplitude
        # shares the module's filters and is a reward in its own right, ANDed with the same inhibits.
        assert session.reward is not None and session.amplitude is not None
        assert session.reward.unit == "µV²" and session.amplitude.unit == "µV"
        assert session.reward.hit_rate == ilf_view.REWARD_AIM == 0.10
        assert session.amplitude.hit_rate == ilf_view.AMPLITUDE_AIM == 0.60, "the amplitude reward aim"
        assert session.frequency_hz == ilf_view.DEFAULT_HZ == 10.0, "the default frequency"
        # the product keeps the module's own 30 s window; the amplitude has its own epoch
        assert session.reward.window == ilf_view.EPOCH_SECONDS == 30.0
        assert session.amplitude.window == ilf_view.amplitude_epoch(10.0) == 10.0

        # **Nine bands, not eighteen** — the inhibits are per mode rather than per channel,
        # and they read the summed signal.
        assert len(session.inhibit_targets) == 9
        assert all(x.pair == session.pair for x in session.inhibit_targets), "measured on the sum"
        assert all(session.latency_for(x) is not None for x in session.targets)

        # and it retunes rather than adding
        assert session.set_frequency(0.05)
        assert session.frequency_hz == 0.05
        assert len([x for x in session.targets if x.role == "reward"]) == 2
        assert session.lowpass_only, "the filter has no high-pass down there"
    finally:
        session.stop()


def test_the_reward_is_gated_by_every_inhibit_band():
    """The arrangement is an AND, which is what `on_target` already is — and it is why there is no alert
    on this screen. Eighteen bands aimed at 95% put something over threshold most of the time."""
    session = _session()
    try:
        assert "alert" not in session.offers
        assert session.feedback.alert.enabled is False
        channel = session.channel_targets("condition")
        assert all(x in channel for x in session.inhibit_targets), \
            "an inhibit that reaches no output is measured, drawn and connected to nothing"
    finally:
        session.stop()


def test_the_reward_does_not_fire_all_session_on_a_dead_channel():
    """**The end-to-end check, and it found a real defect rather than confirming one.**

    The bench recording is a no-electrode capture: both channels rail to a constant ±800 µV, a bandpass of a
    constant is zero, so the product is zero. A threshold derived from 30 s of zeros is `0.0` — and then
    `value >= threshold` is `0 >= 0`, so the reward fired **100% of the session on no signal at all**, with a
    threshold that looked derived because it was not the `Target` default any more.

    `SIGNAL_FLOOR_UV2` closes it, and the number is a 1.0 µV per-channel gate converted into this
    measure's unit. It also checks the chain is wired at all, which is what caught the deadlocked ratchet.
    """
    # The first line of defence has since moved upstream: a channel the device reports as unusable is now
    # excluded from *observation* entirely — its readings must not even enter the threshold windows, or an
    # automatic threshold tracks the rail and a fixed one calibrates on it. So on this recording the reward
    # is never fed at all, and the floor remains as the second line for the case the device cannot flag: a
    # connected channel reading almost nothing.
    session = _session(loop=True)
    try:
        session.toggle_pause()          # a session opens paused, and a paused feed observes nothing
        thread = threading.Thread(target=session.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if session.amplitudes:      # readings are flowing, whatever the targets make of them
                break
            time.sleep(0.05)
        session._stop = True
        thread.join(timeout=2.0)
        assert session.error is None, session.error
        tgt = session.reward
        assert session.amplitudes, "the metrics still ran — only the judging was withheld"
        assert tgt.total == 0, f"{tgt.label} was fed {tgt.total} readings from a dead channel"
        assert not tgt.wanted, "a reward met while reading nothing is a reward for having no electrode on"
        assert tgt not in session.active_targets
        assert tgt.signal_floor == ilf_view.SIGNAL_FLOOR_UV2, "the second line is still in place"
        floor_only = ilf_view.SIGNAL_FLOOR_UV2
        assert not tgt.meets(floor_only / 2), "below the floor never earns a reward"
    finally:
        session.stop()


def test_the_wave_ring_holds_whole_cycles_and_still_resolves_the_bands_top_edge():
    """Two constraints, and only both give an honest picture.

    The window is `traces.WAVE_CYCLES` cycles of the *centre*, because seconds cannot serve 0.05 Hz and 40 Hz. But at
    0.05 Hz the filter is a 0–1.55 Hz lowpass, so the waveform holds content up to 1.55 Hz — and 48 points per
    0.05 Hz cycle is one point every 0.42 s, which aliases it into a shape it does not have. So the point rate
    is the larger of the two requirements.

    And the column count follows the achievable stride rather than the reverse: the stride clamps at 1, and
    taking columns as fixed silently made a 10 Hz window hold 23 cycles under a caption saying 12.
    """
    session = _session()
    try:
        for hz in (0.05, 10.0, 40.0):
            assert session.set_frequency(hz)
            ring, metric = session.wave, session.metric
            cycles = ring.seconds * hz
            assert cycles == pytest.approx(traces.WAVE_CYCLES, rel=0.1), \
                f"{hz} Hz ring holds {cycles:.1f} cycles"
            assert ring.tail_seconds * hz == pytest.approx(ilf_view.LISSAJOUS_CYCLES, rel=0.2)
            sampled_hz = ring.rate_hz / ring.stride
            assert sampled_hz >= 2 * metric.high, \
                f"at {hz} Hz the ring samples at {sampled_hz:.2f} Hz for a band reaching {metric.high} Hz"
    finally:
        session.stop()


def test_changing_the_wave_window_keeps_the_history_already_collected():
    """**The cycle pills change how much is shown, not what was measured, so pressing one must not erase
    the wave.** The ring was rebuilt on every width change, which at 0.05 Hz threw away up to eight
    minutes of a wave that takes eight minutes to re-collect — for a display choice. The stride is set by
    the centre and the band's top edge, never by the cycle count, so the points are the same points at any
    width and the tail that fits carries over: widening shows everything already gathered, narrowing keeps
    the most recent stretch, and the fill fraction stays honest either way.
    """
    session = _session()
    try:
        assert session.set_frequency(0.05)
        ring = session.wave
        # Half fill the 4-cycle window with a recognisable ramp on both channels.
        session.set_view(wave_cycles=4)
        ring = session.wave
        n = ring.columns // 2
        values = np.arange(float(n))
        ring.feed(0, {c: np.repeat(values, ring.stride) for c in ring.channels})
        assert ring.pushed == n
        tail_before = ring.series()[ring.channels[0]][-n:]

        session.set_view(wave_cycles=24)                # widen: everything survives
        wide = session.wave
        assert wide is ring, "resized in place, not rebuilt"
        assert wide.pushed == n
        assert np.array_equal(wide.series()[wide.channels[0]][-n:], tail_before)

        session.set_view(wave_cycles=4)                 # narrow again: the most recent points remain
        assert ring.pushed == n
        assert np.array_equal(ring.series()[ring.channels[0]][-n:], tail_before)
    finally:
        session.stop()


def test_the_condition_is_the_product_and_the_inhibits_with_the_amplitude_outside_it():
    """**The two rewards reach the game on separate pipes and are never combined** — the synchrony
    threshold comes from the module and a combined threshold from the amplitude's own AND with the
    negated inhibits. ANDing the amplitude into our one click let a background state aimed at 60% veto a
    synchrony event designed to be rare at 10%. The amplitude keeps its card and its threshold; its
    discrete channel simply is not the click."""
    session = _session()
    try:
        from nfb_console.feedback import CONDITION, LEVEL
        condition = session.channel_targets(CONDITION)
        assert session.reward in condition
        assert session.amplitude not in condition, "the 60% background state must not veto the 10% event"
        assert all(x.role == "inhibit" for x in condition if x is not session.reward), \
            "everything else in the condition is an inhibit"
        assert session.channel_targets(LEVEL) == [session.reward]
    finally:
        session.stop()


def test_the_summed_wave_is_the_state_and_it_is_the_signed_wave_not_an_envelope():
    """Infra-low feedback follows the waveform's position — the Shepard tone hangs off the signed slow
    signal, never an envelope. `sum_wave` is that value for the summed pair, `state_balance` maps it against
    the slow plot's own scale so the picture and the pitch cannot disagree, and outside the lowpass branch
    there is no slow wave to be a state at all."""
    session = _session()
    try:
        assert session.set_frequency(0.05)
        metric = session.metric
        # The exposed value is the filtered signed sum: by linearity it equals the two channel waveforms
        # added, and it must go negative — an envelope cannot.
        n = int(120 * RATE)
        tt = np.arange(n) / RATE
        wave = 8 * np.sin(2 * np.pi * 0.05 * tt)
        feed(metric, wave, wave, session.specs)
        a, b = (metric.waveforms[c] for c in session.pair)
        assert metric.sum_wave == pytest.approx(float(a[-1] + b[-1]), rel=1e-6)

        session.wave_scale_uv = 20.0
        metric.samples_seen = int(metric.settle_seconds * RATE) + 1     # settled, by construction
        session.wave.feed(0, {c: np.zeros(4) for c in session.pair})
        metric.sum_wave = 0.0
        assert session.state_balance == pytest.approx(0.5), "a wave at zero sits mid-range"
        metric.sum_wave = 20.0
        assert session.state_balance == pytest.approx(1.0), "the top of the plot is the top of the range"
        metric.sum_wave = -60.0
        assert session.state_balance == 0.0, "clamped, not wrapped"

        metric.samples_seen = 0
        assert session.state_balance is None, "a settling wave is mostly transient and is not a state"
        assert session.set_frequency(10.0)
        assert session.state_balance is None, "a 10 Hz oscillation is not a state"
        assert session.feedback.music.follows == STATE, "the Shepard follows the wave by default here"
    finally:
        session.stop()


def test_the_settling_transient_does_not_set_the_states_gain():
    """**The measured failure this mapping exists to fix.** The state used to divide the summed wave by the
    slow plot's auto scale, which is `nice_ceiling` of a *maximum* over twelve cycles — and the largest
    thing that ever enters that ring is the cascade's own settling transient. On a real 5 min capture
    (bipolar, 0.01 Hz corner, 0.05 Hz) a 387 µV transient held the scale at 500 µV against a 20 µV rms
    wave for the ring's whole residency: the state ran p5 0.448 to p95 0.555, **11% of its range**, and the
    Shepard tone following it moved half an octave a minute.

    Reproduced here in the same shape and a tenth of the wall clock: a standing offset whose step at t=0
    rings through the cascade, with a small wave riding on it. What the state must do is measure the wave,
    not the ring's memory of the transient.
    """
    session = _session_at_corner(0.01)
    try:
        assert session.set_frequency(0.2)               # 0.05 Hz behaves identically and takes 5× as long
        specs = session.specs
        n = int(90 * RATE)
        tt = np.arange(n) / RATE
        signal = 200.0 + 8.0 * np.sin(2 * np.pi * 0.2 * tt)
        seen = []
        for i in range(0, n - 7, 8):
            for ch in (0, 1):
                session.consume(Block(ch, i, tuple(v / SCALE for v in signal[i:i + 8]), specs[ch]))
            if session.state_balance is not None:
                seen.append((session.wave.peak(), session.metric.sum_wave, session.state_balance))

        assert max(p for p, _, _ in seen) > 5 * 16.0, "the transient dominates the ring, as it does live"
        # The stretch under test is the one where the ring still *remembers* the transient while the signal
        # itself is back to being the wave — which is the whole failure, and on a real session lasts for the
        # ring's length after every artifact. Both mappings are computed over exactly those samples.
        window = [(p, w, s) for p, w, s in seen if p > 5 * 16.0 and abs(w) < 2 * 16.0]
        assert len(window) > 100, "the ring outlives the transient that filled it — that is the case here"
        was = [min(1.0, max(0.0, 0.5 + w / (2.0 * nice_ceiling(max(p, 1.0))))) for p, w, _ in window]
        now = [s for _, _, s in window]
        old_span = float(np.subtract(*np.percentile(was, [95, 5])))
        new_span = float(np.subtract(*np.percentile(now, [95, 5])))
        assert old_span < 0.2, f"the mapping this replaces is flat here, and was {old_span:.3f}"
        assert new_span > 0.5, f"and this one is not: {new_span:.3f} of the range, against {old_span:.3f}"
        assert min(now) > 0.001 and max(now) < 0.999, "reaching neither end, so both still mean something"
        # Two channels of an 8 µV wave sum to 16 µV peak, whose rms is 16/√2.
        assert session.state_scale_uv == pytest.approx(16.0 / np.sqrt(2), rel=0.25), \
            "the scale is the summed wave's own spread, not the ring's peak"
    finally:
        session.stop()


def test_a_pinned_plot_scale_maps_the_state_absolutely_and_says_so():
    """The escape hatch, and the only mapping that is comparable between sessions: pinning the slow plot
    pins the feedback with it, linearly, so the wave at the top of the plot is the top of the range. The
    caption has to distinguish the two, because on auto the gain is a σ no axis shows."""
    session = _session_at_corner(0.01)
    try:
        assert session.set_frequency(0.05)
        metric = session.metric
        metric.samples_seen = int(metric.settle_seconds * RATE) + 1     # settled, by construction
        session.wave_scale_uv = 20.0
        metric.sum_wave = 10.0
        assert session.state_balance == pytest.approx(0.75), "half way up a pinned ±20 µV"
        assert "feedback ±20 µV" in ilf_view._state_scale_note(session)

        session.wave_scale_uv = 0.0
        assert session.state_balance is None, "no spread measured yet, so the mechanisms follow the level"
        assert "settling" in ilf_view._state_scale_note(session)
        session._state_var, session._state_weight, session._state_seconds = 100.0, 1.0, 1e6
        assert session.state_balance == pytest.approx(0.5 + 0.5 * np.tanh(10.0 / 15.0))
        assert "±15 µV" in ilf_view._state_scale_note(session), "the derived gain is printed, not implied"
    finally:
        session.stop()


def test_sensitivity_is_the_one_dial_for_how_hard_the_state_pushes():
    """**The knob a session that feels dead has to have**, and there is exactly one of it.

    An earlier version had none: the state was normalised against a measurement, and "how much movement do
    I want out of it" was left to each mechanism's own range. That does not answer the question — a range
    says where an output starts and stops, not how much of it a wave of this size asks for — and it left an
    operator watching a shadow with nothing to turn. It multiplies the movement in both mappings, so the
    number means the same thing whether the plot is pinned or not, and the caption prints the microvolts.
    """
    session = _session_at_corner(0.01)
    try:
        assert session.set_frequency(0.05)
        metric = session.metric
        metric.samples_seen = int(metric.settle_seconds * RATE) + 1
        session._state_var, session._state_weight, session._state_seconds = 100.0, 1.0, 1e6
        metric.sum_wave = 10.0

        assert session.state_sensitivity == 1.0, "unchanged behaviour until somebody turns it"
        gentle = session.state_balance
        session.set_view(state_sensitivity=3.0)
        assert session.state_full_uv == pytest.approx(15.0 / 3), "three times the sensitivity, a third the µV"
        assert session.state_balance > gentle + 0.15, "and the same wave sits much further up the range"
        assert "×3" in ilf_view._state_scale_note(session), "and the caption says it is turned up"

        # The pinned mapping takes it too, so the dial does not change meaning when the plot is pinned.
        session.wave_scale_uv = 20.0
        session.set_view(state_sensitivity=2.0)
        assert session.state_balance == pytest.approx(0.5 + 2.0 * 10.0 / (2 * 20.0))
    finally:
        session.stop()


def test_the_sensitivity_row_is_under_the_shadow_and_only_where_a_state_is_measured():
    """**Under the control it acts on.** It was first put on the plot panel that draws the state, which is
    where the mapping is *explained* and not where the mechanism it drives is set — an operator watching a
    flat screen is in the shadow pane, and from there the dial did not exist.

    Shown only with the shadow following the state, and only where the protocol offers the choices. A state
    that arrives bounded 0…1 by construction — HEG's normalised signal, the heart screen's coherence —
    already uses its range, so a gain there would be a second way of saying what the brightness range says.
    """
    from nfb_console import signal_pane

    session = _session_at_corner(0.01)
    try:
        session.feedback.shadow.follows = LEVEL
        pane = signal_pane.SignalPane(StubPage(), session)
        assert "sensitivity" not in " ".join(_all_text(pane.shadow_rows)), \
            "nothing to scale when the brightness is following the score"

        session.feedback.shadow.follows = STATE
        pane = signal_pane.SignalPane(StubPage(), session)
        shown = " ".join(_all_text(pane.shadow_rows))
        assert "sensitivity" in shown
        # **A box, not a ladder of pills**: this is a continuous quantity, and 1.4 is a legitimate answer
        # to "a bit more than now" that no ladder short enough to fit the row contains.
        assert "1.00" in _numbers(pane.shadow_rows), "the value is typeable, so it is a field"
        assert "measuring the wave's size" in shown, "and it says what it is waiting for"

        # With a spread measured, the row prints what the setting costs in wave rather than a bare ×.
        session.metric.samples_seen = int(session.metric.settle_seconds * RATE) + 1
        session._state_var, session._state_weight, session._state_seconds = 100.0, 1.0, 1e6
        pane = signal_pane.SignalPane(StubPage(), session)
        assert "±15 µV of wave" in " ".join(_all_text(pane.shadow_rows))
        session.set_state_sensitivity(2.0)
        pane = signal_pane.SignalPane(StubPage(), session)
        assert "±8 µV of wave" in " ".join(_all_text(pane.shadow_rows)), "turning it up costs less wave"
        assert session.set_state_sensitivity(9.0) is None and session.state_sensitivity == 3.0, \
            "clamped to the protocol's own bounds, where the curve stops buying movement"
    finally:
        session.stop()

    from nfb_console.heg_view import HEGSession

    heg = HEGSession(BY_CODE["HEG"], choice=None)
    try:
        assert getattr(heg, "state_sensitivity_range", None) is None, \
            "a state already bounded 0-1 has nothing to scale"
        heg.feedback.shadow.follows = STATE
        pane = signal_pane.SignalPane(StubPage(), heg)
        assert "sensitivity" not in " ".join(_all_text(pane.shadow_rows))
    finally:
        heg.stop()


def test_stops_while_failing_means_an_inhibit_is_violated_not_the_rare_reward():
    """A game output receives the inhibits as a channel of their own, independent of every reward, and
    the video model pauses on them. Gating the playback on the
    whole condition instead would silence it 90% of the session by construction, since the product is met
    10% of the time by design. Quiet Reward keeps the whole condition: there, doing well earns the silence
    and doing well is the rare event."""
    session = _session()
    try:
        inhibit = next(t for t in session.targets if t.role == "inhibit")
        inhibit.threshold = 10.0
        assert session.feedback.music.stops_when == "failing", \
            "the default, so a violation has a consequence without anyone finding a pill"

        inhibit.observe(5.0, 20.0)                      # clear: under its threshold
        assert not session.music_hold(met=False), "the rare reward failing must not pause the playback"
        inhibit.observe(50.0, 20.0)                     # violated
        assert session.music_hold(met=True), "an inhibit violation pauses it whatever the reward says"

        session.feedback.music.stops_when = "met"       # Quiet Reward keeps the whole condition
        assert session.music_hold(met=True)
        assert not session.music_hold(met=False)
    finally:
        session.stop()


def test_the_shepard_pitch_can_follow_the_state_while_the_volume_follows_the_level():
    """`Music.follows` is the shadow's LEVEL/STATE fork applied to the pitch: each signal keeps its own
    perceptual dimension — synchrony is how loud, the summed wave is how high — instead of two quantities
    with different dynamics blended into one number."""
    session = _session()
    try:
        heard: dict = {}
        session.music.apply = lambda *a, **kw: heard.update(kw)

        session.feedback.music.follows = STATE
        session._apply_outputs(level=0.3, met=True, shadow_in=0.3, balance=0.8)
        assert heard["level"] == pytest.approx(0.8), "the pitch rides the state"

        session._apply_outputs(level=0.3, met=True, shadow_in=0.3, balance=None)
        assert heard["level"] == pytest.approx(0.3), "no state yet, so it falls back to the level"

        session.feedback.music.follows = LEVEL
        session._apply_outputs(level=0.3, met=True, shadow_in=0.3, balance=0.8)
        assert heard["level"] == pytest.approx(0.3), "and LEVEL is still choosable"
    finally:
        session.stop()


def test_a_shared_reference_is_named_as_a_problem_rather_than_trained():
    """With one reference in both channels the reference's own signal multiplies with itself, so the reward
    reports its noise as synchrony between two sites that could be disconnected."""
    session = _session()
    try:
        session.device = _stub_device(reference="linked")
        trouble = session.wiring_problem()
        assert trouble and "bipolar" in trouble
    finally:
        session.stop()


def test_the_hardware_corner_becoming_the_missing_high_pass_is_named():
    """The honest version of the high-pass warning, and it only makes sense once the lowpass fork is
    understood: with no high-pass in the filter, the *device's* corner becomes one — so a 0.05 Hz setting
    against a 0.5 Hz corner trains 0.5–1.55 Hz and calls it infra-slow. At alpha there is nothing to say."""
    session = _session_at_corner(0.5)
    try:
        assert session.set_frequency(10.0)
        assert session.wiring_problem() is None, "a bandpass at alpha has its own lower edge"
        assert session.set_frequency(0.05)
        trouble = session.wiring_problem()
        # The wording follows whether *our* band has a lower edge — see `SLOW_BANDS`. Under the default it
        # does, so the complaint is that the device's corner sits above the whole band rather than that
        # there is no high-pass at all; saying the latter would send the operator after the wrong thing.
        assert trouble and "0.5" in trouble and "above this band's top edge" in trouble
        assert "no high-pass" not in trouble, "there is one now, in software"

        session.set_frequency(0.05, slow_band="0–2×")
        bare = session.wiring_problem()
        assert bare and "no high-pass" in bare, "the bare lowpass path really has none"
    finally:
        session.stop()

    # **The corner is read from what is streaming, not from a device**, so a recording captured at 0.01 Hz
    # replays without the warning. It used to warn against a file already recorded at the right corner,
    # because `Settings` only exists live and the fallback was 0.5.
    session = _session_at_corner(0.01)
    try:
        assert session.corner_hz == 0.01, "read out of the config the source is decoding with"
        assert session.set_frequency(0.05)
        assert session.wiring_problem() is None, "at the best corner there is nothing to warn about"
    finally:
        session.stop()


def test_the_settings_survive_a_stop_and_the_ladder_does_not():
    """`adopt_targets`: where the instrument points is what an operator sets, so it comes across; the
    nine-band ladder at a fixed aim is the protocol, so it is rebuilt. And a reward carried across without its
    estimator would read zero for ever, which is why this rebuilds rather than assigning."""
    first = _session()
    try:
        first.set_frequency(40.0, bandwidth_hz=1.0, order=1, post_seconds=0.25)
        second = _session()
        try:
            second.adopt(first)
            assert second.frequency_hz == 40.0
            assert second.bandwidth_hz == 1.0 and second.order == 1
            assert second.post_seconds == pytest.approx(0.25)
            assert second.metric is not None and (second.metric.low, second.metric.high) == (39.5, 40.5)
            assert len(second.inhibit_targets) == 9, "the ladder is rebuilt, not adopted"
        finally:
            second.stop()
    finally:
        first.stop()


# --- the Shepard tone ------------------------------------------------------------------------

def test_the_shepard_wheel_is_periodic_in_position_so_the_rise_never_ends():
    """The property that makes it a Shepard tone: sliding the comb by one octave returns the *same set* of
    frequencies and the same set of amplitudes, so there is no top to reach. Twelve
    subtones from 1 Hz make the wheel exactly 12 octaves and land the last partial at 4096 Hz."""
    from nfb_console import music

    player = music.ShepardPlayer()
    f0, a0 = player.partials(0.0)
    f1, a1 = player.partials(1.0)
    assert np.allclose(np.sort(f0), np.sort(f1))
    assert np.allclose(np.sort(a0), np.sort(a1))
    assert music.SHEPARD_SUBTONES == 12 and music.SHEPARD_MIN_HZ == 1.0
    assert f0.max() == pytest.approx(2048.0), "twelve octaves from 1 Hz, so the wheel spans to 4096"


def test_the_partial_that_wraps_is_almost_silent_when_it_does():
    """Why the wrap is inaudible, as a number rather than as a hope: the envelope's value at both ends of the
    wheel is the same and it is far down, so what crosses the seam is a partial nobody can hear. A flat
    envelope would click once per octave of travel."""
    from nfb_console import music

    player = music.ShepardPlayer()
    _f, amps = player.partials(0.0)
    edge = float(amps[0])
    assert edge < 0.1, f"the end partial is at {edge:.3f} of centre amplitude"
    _f2, amps2 = player.partials(11.999)
    assert float(amps2.min()) == pytest.approx(edge, abs=0.02), "and the same on the way out"


def test_a_hold_fades_the_music_out_instead_of_cutting_it():
    """**A gate that flickers must not click.** `held` used to return zeros immediately, truncating the
    waveform mid-cycle — a step, which is a click — and under Quiet Reward the gate follows a condition
    that sits near its threshold and flickers. The entry now rides the same one-block ramp the resume
    always did: the seam at the hold onset is no larger than the waveform's own slope, the fade reaches
    zero, and silence follows."""
    from nfb_console import music

    players = [(music.ShepardPlayer(), lambda b: b.configure(100.0, 0.5)),
            (music.NoisePlayer(), lambda b: b.configure("pink", 100.0)),
            (music.TonePlayer(), lambda b: b.configure(100.0, 0.5))]
    for player, configure in players:
        configure(player)
        for _ in range(10):
            player._block(2048)
        last = player._block(2048)[:, 0]
        player.held = True
        fade = player._block(2048)[:, 0]
        name = type(player).__name__
        seam = abs(float(last[-1] - fade[0]))
        slope = float(np.abs(np.diff(last)).max())
        assert seam <= max(slope, 1e-6) * 1.5, f"{name}: seam {seam:.5f} against slope {slope:.5f}"
        assert fade[-1] == pytest.approx(0.0, abs=1e-5), f"{name}: the fade must reach silence"
        assert not player._block(2048).any(), f"{name}: and stay there"


def test_the_shepard_tone_keeps_its_energy_where_a_speaker_can_reproduce_it():
    """**The comb's amplitude bell centres on 64 Hz, so most of the sound was cone excursion.**

    Measured before the floor: 60.8% of the rendered power below 70 Hz and 20.4% below 20 — worse than the
    pink noise (28% below 80 Hz) whose crackle forced `HIGHPASS_HZ`. A laptop driver cannot move air down
    there; each low partial is heard as the cone crackling at its travel limit instead, which is why it
    reads as crackling notes. `SHEPARD_FLOOR_HZ` is the same fix arriving at the Shepard tone, smooth in frequency
    so a partial sliding through the corner fades rather than steps.
    """
    from nfb_console import music

    for level in (0.0, 0.5, 1.0):
        player = music.ShepardPlayer()
        player.configure(100.0, level, glide_seconds=0.01)
        x = np.concatenate([player._block(2048)[:, 0] for _ in range(60)])
        spec = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), 1.0 / music.SAMPLE_RATE)
        below = float(spec[freqs < music.SHEPARD_FLOOR_HZ].sum() / spec.sum())
        assert below < 0.15, f"{below:.1%} of the power below {music.SHEPARD_FLOOR_HZ:g} Hz at level {level}"


def test_the_shepard_tone_is_continuous_across_block_boundaries():
    """Each partial's frequency changes as the comb slides, so `sin(2πft)` with a shared clock would step in
    phase every block — twelve clicks per 43 ms, on music whose whole job is to be continuous. Phase has to
    be integrated per partial, and the test is that the seam is no larger than the waveform's own slope."""
    from nfb_console import music

    player = music.ShepardPlayer()
    player.configure(60.0, 0.5)
    blocks = [player._block(2048)[:, 0] for _ in range(30)]
    inside = max(float(np.abs(np.diff(b)).max()) for b in blocks)
    seams = max(abs(float(blocks[i][-1] - blocks[i + 1][0])) for i in range(len(blocks) - 1))
    assert seams <= inside * 1.5, f"seam {seams:.5f} against {inside:.5f} inside a block"


def test_every_generated_sound_is_as_loud_as_the_others_at_the_same_setting():
    """`Music.volume` is persisted per protocol and shared across sources, so two sounds at different
    loudnesses for the same number would make switching source an unmarked volume change.

    All three of the sounds we *generate* are calibrated to the pink noise. The two we do not — a supplied file
    and the EEG sonifier — are out of it by their nature: a track's loudness is whatever was mastered into it.
    """
    from nfb_console import music

    def rendered(player) -> float:
        return float(np.sqrt((np.concatenate([player._block(2048)[:, 0] for _ in range(40)]) ** 2).mean()))

    noise = music.NoisePlayer(); noise.configure("pink", 100.0)
    reference = rendered(noise)
    shepard = music.ShepardPlayer(); shepard.configure(100.0, 0.5)
    assert rendered(shepard) / reference == pytest.approx(1.0, abs=0.2)
    tone = music.TonePlayer(); tone.configure(100.0, 0.5)
    # Deliberately *over* by RMS: loudness sums across critical bands, so a lone sine at the broadband
    # sounds' RMS is heard at about half their loudness — reported from sessions twice before the
    # direction of the correction was fixed. See `music.TONE_PEAK` for the ear-side calibration.
    assert 2.0 < rendered(tone) / reference < 3.2

    for colour in music.NOISE_COLOURS:
        other = music.NoisePlayer(); other.configure(colour, 100.0)
        assert rendered(other) / reference == pytest.approx(1.0, abs=0.15), colour


def test_the_generated_sounds_are_quiet_enough_to_be_a_background():
    """**The absolute loudness, because nothing else in this suite listens.**

    Reported from a session: the noise at the default volume of 60 was about three times louder than a
    background should be. The fix is `music.HEADROOM`, the pad on the generated source — the alternative is
    telling the operator to run the control at 20, which is the same sound with less room left underneath it.
    The default volume stays at 60.

    Pinned as a number so the next change to a peak constant has to say what it does to the loudness, and so
    that a screen deriving its own volume from the pad — `at_view.MUSIC_VOLUME` — cannot be left behind.
    """
    from nfb_console import music
    from nfb_console.feedback import Music

    assert Music().volume == 60.0, "the default the report was made against"
    player = music.NoisePlayer()
    player.configure("pink", Music().volume)
    rms = float(np.sqrt((np.concatenate([player._block(2048)[:, 0] for _ in range(40)]) ** 2).mean()))
    assert rms == pytest.approx(0.0093, rel=0.15), f"{20 * np.log10(rms):.1f} dBFS at volume 60"


def test_the_level_moves_the_shepard_tone_upward_and_nothing_else_does():
    """It follows the *score*, where a crossfade follows a named *state* — which is why `MusicOutput.apply`
    takes them as two arguments. Collapsing them would undo the distinction `Shadow.follows` draws."""
    from nfb_console import music
    from nfb_console.feedback import Music

    out = music.MusicOutput()
    settings = Music(enabled=True, source="shepard")
    out.apply(settings, level=0.0)
    low = out.shepard.level
    out.apply(settings, level=1.0)
    assert out.shepard.level == 1.0 and low == 0.0
    # and the glide is a glide: one block does not jump the whole span
    out.shepard._block(2048)
    assert 0.0 < out.shepard.octaves < music.SHEPARD_OCTAVES
    out.release()


def test_the_glide_arrives_inside_the_hold_and_the_level_is_not_smoothed_twice():
    """Two sources of lag this screen was adding to a chain that already carries seconds of filter delay.

    The Shepard glide's 1.5 s default was chosen against a level updating twenty times a second, where it is
    what stops the tone reading as vibrato. This screen's level *steps*, once per `SAMPLE_HOLD_SECONDS`, so a
    1.5 s glide covers only 74% of each step before the next lands — the pitch never arrives and the whole
    time constant is lag. And `smoothing_seconds` has no job at all here: the level is a
    straight line from the output window, handed onward with nothing after it.
    """
    import math

    from nfb_console import music
    from nfb_console.ilf_view import GLIDE_SECONDS, SAMPLE_HOLD_SECONDS

    session = _session()
    try:
        assert session.feedback.smoothing_seconds == 0.0, "a 0.3 s pole on a 2 s staircase is pure lag"
        assert session.feedback.music.glide_seconds == GLIDE_SECONDS
    finally:
        session.stop()

    # The criterion, as arithmetic rather than as taste: reach the step before the next one arrives.
    arrives = 1 - math.exp(-SAMPLE_HOLD_SECONDS / GLIDE_SECONDS)
    default = 1 - math.exp(-SAMPLE_HOLD_SECONDS / music.SHEPARD_GLIDE_SECONDS)
    assert arrives > 0.98, f"only {arrives:.0%} of a step is covered before the next"
    assert default < 0.80, f"the default covered {default:.0%}, which is why it is overridden here"

    # And it is still a glide: a step must not arrive as a jump, or a Shepard comb reads as a different
    # tone rather than as movement, which is the whole percept.
    player = music.ShepardPlayer()
    player.configure(60.0, 1.0, GLIDE_SECONDS)
    player._block(2048)
    assert 0.0 < player.octaves < music.SHEPARD_OCTAVES

    # Left alone, the music keeps the default — the waveform screen's level flows at 20 Hz and wants it.
    other = music.ShepardPlayer()
    other.configure(60.0, 1.0)
    assert other.glide_seconds == music.SHEPARD_GLIDE_SECONDS


def test_how_far_the_tone_travels_is_this_mechanisms_own_range_and_is_set_here():
    """**Where "I want more movement" belongs**, and the reason the state's mapping upstream is not a
    sensitivity control: every continuous mechanism owns how far its output travels — the shadow its
    brightness span, the bounded `tone` its two frequencies — and this one had that welded shut as a module
    constant while carrying an infra-slow state. Two gains for one effect is a pane where a feedback that
    feels wrong has two knobs and no way to tell which.

    Capped at half the wheel, which is not taste: past six octaves of a twelve-octave wheel the top of the
    range starts sounding like the bottom, which is the failure a design with no ceiling has instead of
    saturation.
    """
    from nfb_console import music

    player = music.ShepardPlayer()
    player.configure(60.0, 1.0, 0.01, travel_octaves=2.0)
    for _ in range(60):
        player._block(2048)
    assert player.octaves == pytest.approx(2.0, abs=0.05), "the whole range is two octaves when asked"

    wide = music.ShepardPlayer()
    wide.configure(60.0, 1.0, 0.01, travel_octaves=0.0)          # 0 takes the module default
    assert wide.travel_octaves == music.SHEPARD_OCTAVES
    wide.configure(60.0, 1.0, 0.01, travel_octaves=99.0)
    assert wide.travel_octaves == music.SHEPARD_OCTAVES, "half the wheel, whatever is asked for"

    settings = Music(source="shepard", travel_octaves=3.0)
    assert settings.travel_octaves == 3.0, "and it is a protocol setting, like the glide beside it"


def test_the_shepard_tone_is_available_but_is_not_this_instruments_feedback():
    """**It belongs to the infra-low reward, not to synchrony.** The synchrony measure produces a mark, a
    percentage above threshold, a continuous output, a bar, a threshold and the two traces — numbers for
    a game and a lamp, and no sound of its own. An earlier version enabled the Shepard tone by default
    here, which presented another instrument's feedback as this one's.

    Offered rather than removed, because off-by-default and unavailable are different claims and only the
    first is true: the level exists on this screen, so the music works. It is simply not the default.
    """
    session = _session()
    try:
        assert "shepard" in session.music_sources
        assert not session.feedback.music.enabled, "synchrony makes no sound by default"
        assert session.feedback.click.enabled, "the mark is the discrete reward"
        assert session.feedback.sound.enabled, "the volume is the continuous output"
        assert not session.feedback.alert.enabled and "alert" not in session.offers
    finally:
        session.stop()




def test_the_sum_spectrum_sees_what_the_artifact_gate_rejected():
    """Every other display on every screen draws rejected data, because a rejection you cannot see is
    indistinguishable from a dead electrode. The sum spectrum broke that when the summed signal moved into
    `consume`, which only runs on accepted blocks — so it needs its own aligner in `observe_raw`.

    Two aligners rather than one because an aligner hands each run over exactly once; sharing it would starve
    whichever consumer asked second.
    """
    session = _session()
    try:
        assert session.inhibit_aligner is not session.display_aligner
        # the display path is fed from observe_raw, which the base class calls before the gate's verdict
        import inspect
        source = inspect.getsource(ilf_view.ILFSession.observe_raw)
        assert "display_aligner" in source and "sum_ring" in source
        assert "sum_ring" not in inspect.getsource(ilf_view.ILFSession._consume_inhibits)
    finally:
        session.stop()


def test_the_level_rule_is_left_at_its_default_because_it_cannot_matter_here():
    """With the inhibits on the condition only, `scaled`, `weighted` and `weakest` all reduce to the mean of
    the rewards — so the `combine = "scaled"` this screen had inherited from CFC was a line that could not
    change the output while reading as a decision somebody made."""
    session = _session()
    try:
        assert session.feedback.combine == "weighted", "the default, not a copied rule"
        assert not any(x.direction != "increase" for x in session.channel_targets("level"))
    finally:
        session.stop()


def test_the_theta_notch_never_removes_the_band_being_trained():
    """The notch deletes 4–6 Hz before the two channels are compared, and the frequency control reaches into
    it. At a 5 Hz centre the band is 3.5–6.5 Hz, which the notch takes out almost entirely — measured, a 5 Hz
    tone comes through a 4–6 Hz order-2 bandstop at 1.0% of its amplitude, and the reward is a product, so
    0.01% of its scale. The reward would read zero with nothing on screen saying why.

    Suppressed rather than warned about: there is no reading of "notch out what you are measuring" that
    anybody wants. `describe` says which case is running.
    """
    chans = specs()
    inside = ProductSynchrony(chans, (0, 1), 5.0)
    assert not inside.theta_notch and inside.notch_suppressed
    assert "notch off" in inside.describe()

    clear = ProductSynchrony(chans, (0, 1), 10.0)
    assert clear.theta_notch and not clear.notch_suppressed
    assert "4–6 Hz notched" in clear.describe()

    # and it really would have gutted the reward: the notched version reads far below the clean one
    n = int(120 * RATE)
    tt = np.arange(n) / RATE
    wave = 20 * np.sin(2 * np.pi * 5.0 * tt)
    kept = settled(feed(ProductSynchrony(chans, (0, 1), 5.0), wave, wave, chans))
    notched = settled(feed(ProductSynchrony(chans, (0, 1), 5.0, theta_notch=True), wave, wave, chans))
    assert kept > 100.0, "with the notch suppressed the reward sees the signal"
    assert notched == pytest.approx(kept, rel=0.01), "and asking for it changes nothing, because it is off"


def test_the_slow_signal_panel_is_only_there_when_a_slow_signal_is_being_trained():
    """At 10 Hz it was a 1.2 s oscilloscope of a bandpassed sine under a heading claiming otherwise, which the
    raw traces already show. The test is the filter's own fork — below 1.8 Hz it degrades to a lowpass, which
    is exactly when there is a slow waveform to read."""
    import sys
    sys.path.insert(0, "tests")
    from test_screens import StubPage, _numbers, toolbar

    import flet as ft

    def visible_text(control, depth: int = 0) -> str:
        """Text the operator can actually see. `test_screens._texts` walks hidden branches too — which is
        right there, since it reads the inactive tabs on purpose — and so cannot answer this question."""
        if control is None or depth > 16 or getattr(control, "visible", True) is False:
            return ""
        if isinstance(control, ft.Text):
            return control.value or ""
        out = []
        for child in getattr(control, "controls", None) or []:
            out.append(visible_text(child, depth + 1))
        content = getattr(control, "content", None)
        if content is not None:
            out.append(visible_text(content, depth + 1))
        return " ".join(x for x in out if x)

    page = StubPage()
    session = _session()
    try:
        control = ilf_view.view(page, session, toolbar(), lambda: None)
        session.set_frequency(10.0)
        session.redraw()
        assert not session.lowpass_only
        assert "SLOW SIGNAL" not in visible_text(control)
        assert "LISSAJOUS" in visible_text(control), "the alignment picture is meaningful at every frequency"

        session.set_frequency(0.05)
        session.redraw()
        assert session.lowpass_only
        assert "SLOW SIGNAL" in visible_text(control)
    finally:
        session.stop()


def test_the_trace_is_where_you_overrule_the_device_about_a_channel():
    """**The evidence is on the trace, so the control is too.**

    The amplifier's below-range detector asserts no-signal on a working infra-low channel — 800 µV is the
    narrowest range a voltage channel offers, so 20 µV of EEG sits at 2.5% of full scale — and every target
    here is on the *pair*, so one condemned channel takes the whole reward down while its waveform draws
    healthily beside the dot. Nothing in software adjudicates that better than the person looking at it.

    Three states rather than two, because "the device's word" has to remain reachable: a switch that only
    forces a channel on would make the operator's first click permanent.
    """
    page = StubPage()
    session = _session()
    try:
        control = ilf_view.view(page, session, toolbar(), lambda: None)
        row = _find_trace_row(control, "ch0")
        assert row is not None and row.on_click is not None, "the channel label has to be clickable"

        assert session.trusted.get(0) is None
        row.on_click(None)
        assert session.trusted[0] is True, "first click: I can see it, trust it"
        row.on_click(None)
        assert session.trusted[0] is False, "second: no, ignore that channel"
        row.on_click(None)
        assert session.trusted.get(0) is None, "third: hand it back to the device"

        # And it says so, in the label and in the caption, or the operator cannot tell their own decision
        # from the amplifier's.
        row.on_click(None)
        session.redraw()
        shown = " ".join(_all_text(control))
        assert "ch0 ✓" in shown and "ch0 forced on" in shown
    finally:
        session.stop()


def test_the_continuous_output_is_a_straight_window_and_not_an_average():
    """**The output is a position in a window below the threshold.**

        output = (value*100 - threshold*size + position*threshold) / (position*threshold)
        mark   = value >= threshold

    The value is compared with the threshold (30 s window, 10% aim), with a range size of **35** and a
    range position of **100**. So the window is 65% to 100% of the threshold and the output is the position
    in it — a straight line that is full *at* the threshold, which is where the mark fires too.

    An earlier version had this running from the threshold up to 1.5x. The size and position say otherwise.
    """
    assert ilf_view.RANGE_POSITION == 100.0 and ilf_view.RANGE_SIZE == 35.0

    out = ilf_view.reward_output
    assert out(0.50 * 100, 100) == 0.0, "below the window"
    assert out(0.65 * 100, 100) == 0.0, "the bottom of the window is 0.65x the threshold"
    assert out(0.80 * 100, 100) == pytest.approx((0.80 - 0.65) / 0.35, abs=1e-6), "linear, not a logistic"
    assert out(1.00 * 100, 100) == 1.0, "full **at** the threshold, not above it"
    assert out(2.50 * 100, 100) == 1.0, "and it stays full; the window ends there"
    # No soft ramp anywhere, so equal steps in value are equal steps out — the property a logistic lacks.
    lo, mid, hi = out(70, 100), out(80, 100), out(90, 100)
    assert mid - lo == pytest.approx(hi - mid, abs=1e-9)


def test_the_level_reads_the_product_alone_and_saturates_at_the_reward():
    """The amplitude reward keeps its card and its place in the condition; it does not colour the sound.

    The amplitude has its own bar and is ANDed into the same reward event, which is
    exactly what it keeps here. What it loses is a share of the continuous output — averaging two rewards aimed at 10% and 60% produces a number that mostly reports the 60% one.
    """
    session = _session()
    try:
        assert session.scores_bands is False, "the SIGNAL panel's ramp steps have nothing to act on here"
        assert "threshold" in session.level_source[0]

        # No signal yet: nothing to report rather than half of something.
        assert session.raw_level == 0.0

        reward = session.reward
        assert reward is not None
        reward.threshold = 100.0
        for value, expected in ((50.0, 0.0), (82.5, 0.5), (100.0, 1.0), (400.0, 1.0)):
            session.amplitudes[("product",)] = value
            assert session.raw_level == pytest.approx(expected, abs=0.01), value

        # The continuous output is full exactly when the discrete reward fires — the output reaching the top of
        # the window and the mark going true are the same instant.
        session.amplitudes[("product",)] = 99.0
        assert session.raw_level < 1.0
        session.amplitudes[("product",)] = 100.0
        assert session.raw_level == 1.0
    finally:
        session.stop()


def test_the_output_window_is_settable_and_is_the_sensitivity_control():
    """**A range size and position, not a generic ramp.** Dropping `scores_bands` took the SIGNAL panel's ramp
    controls off this screen, which was right — the output has no logistic to widen — and left nothing to
    adjust, which was not. The adjustment is the range size and the range position,
    both 0–200 in steps of 1, set to 35 and 100.

    `Range Size` is the sensitivity proper. Narrower means the output covers the approach to the threshold
    over a shorter distance, so the same change in synchrony moves the sound more.
    """
    session = _session()
    try:
        session.reward.threshold = 100.0
        assert (session.range_size, session.range_position) == (35.0, 100.0), "the default values"

        def at(value: float) -> float:
            session.amplitudes[("product",)] = value
            return session.raw_level

        # The screen sets the *ends* of the window — where the output is silent and where it is full —
        # because "35% below 100% of threshold" describes a size and position pair and not the behaviour, and was
        # reported as unreadable. `output_bottom` is the derived end; `range_size` stays the stored width.
        session.range_position, session.output_bottom = 100.0, 65.0
        assert session.range_size == 35.0, "the ends and the stored width stay consistent"
        session.output_bottom = 90.0
        assert session.range_size == 10.0

        session.range_size = 35.0
        wide = at(80.0)
        session.range_size = 10.0
        narrow = at(80.0)
        assert wide > narrow, "a narrower window reaches 0 later, so 80% of threshold scores lower"
        session.range_size = 100.0
        widest = at(80.0)
        assert widest == pytest.approx(0.8, abs=0.01), "a full-width window is just value/threshold"

        # Both ends move with `Range Position`, which is the top of the window.
        session.range_size, session.range_position = 35.0, 200.0
        assert at(100.0) < 1.0, "the top is now twice the threshold, so being at it is no longer full"
        assert "2× the threshold" in session.level_source[1] or "full at 2×" in session.level_source[1]

        # And they are clamped to the sliders' own range rather than being free numbers.
        assert ilf_view.RANGE_LIMITS == (0.0, 200.0)
    finally:
        session.stop()


def test_the_bandwidth_box_is_absent_where_the_bandwidth_does_nothing():
    """**A "3 Hz" box beside a 0.05 Hz centre is the thing that makes this screen look deranged.**

    The bandwidth is one fixed width in hertz spanning the whole frequency range, and below the fork at
    1.8 Hz the filter abandons it for a lowpass at twice the centre. So at the infra-slow preset every value
    the slider can take — 1 through 3 — produces the identical 0–0.1 Hz band. Measured across the range, not
    argued from the formula.

    A fixed width is a control that does not generalise, plus a fork to survive it. An *editable box*
    that cannot affect anything is not acceptable, and it invites the reasonable conclusion that the software is nonsense. The band is stated there
    instead. Order still reaches the lowpass, so it stays a control.
    """
    from nfb.dsp import ProductSynchrony

    chans = specs()
    bands = {ProductSynchrony(chans, (0, 1), 0.05, bandwidth_hz=bw).high
             for bw in (1.0, 1.5, 2.0, 2.5, 3.0)}
    assert bands == {0.1}, "every bandwidth the slider offers gives the same band at 0.05 Hz"

    for hz, editable in ((0.05, False), (1.0, False), (10.0, True), (40.0, True)):
        session = _session()
        try:
            session.set_frequency(hz)
            assert session.lowpass_only is (not editable)
        finally:
            session.stop()


def test_the_two_second_sample_and_hold_is_the_default_and_is_choosable():
    """**A sample and hold of 2.0 seconds, keeping the last value.**

    It feeds *both* the threshold and the output window, so the threshold and the value and the continuous
    output all move once every two seconds. That is not
    a smoothing detail: at a 10% aim a 30 s window holds about **fifteen** values where a 20 Hz feed puts
    six hundred in it, and a threshold taken from fifteen sits near the highest one or two of the last half
    minute and steps visibly.

    The shorter feed stays available, because a 2 s staircase drives a game's speed well and may read as sluggish on a
    screen with no game. The 2 s hold is the default.
    """
    assert ilf_view.SAMPLE_HOLD_SECONDS == 2.0
    assert set(ilf_view.HOLD_CHOICES) == {2.0, 0.0}

    session = _session()
    try:
        assert session.hold_seconds == 2.0, "2 s by default"
        assert session.hold_choices is ilf_view.HOLD_CHOICES

        spec = session.specs[0]
        # Feed a value that changes on every reading, and count how often what lands in `amplitudes` moves.
        def run(hold: float) -> int:
            session.hold_seconds = hold
            session._held_at.clear()
            session._held_value.clear()
            seen = []
            for i in range(500):
                block = Block(0, i * 8, (i,) * 8, spec, state=0)
                seen.append(session._held(("product",), float(i), block))
            return sum(1 for a, b in zip(seen, seen[1:]) if a != b)

        held = run(2.0)
        free = run(0.0)
        seconds = 500 * 8 / spec.rate_hz
        assert free == 499, "with no hold every reading is its own value"
        assert held == pytest.approx(seconds / 2.0, abs=1), f"one change every 2 s over {seconds:g} s"
    finally:
        session.stop()


def test_a_fresh_session_opens_on_a_preset_the_amplifier_can_reach():
    """**Sub-hertz where the corner allows it, alpha otherwise, and always one of them.**

    The default is 10 Hz, but it assumes a machine set up for whatever was last done. The
    corner is in the config the source is decoding with, so this screen can tell whether the infra-slow preset
    is *reachable*: at 0.5 Hz a 0.05 Hz reward measures the filter's roll-off and `wiring_problem` says so,
    which is a poor way to open a screen.
    """
    for corner, hz, name in ((0.5, 10.0, "alpha"), (0.01, 0.05, "infra-slow")):
        session = _session_at_corner(corner)
        try:
            assert session.corner_hz == corner
            assert session.frequency_hz == pytest.approx(hz)
            assert ilf_view.preset_name(session.frequency_hz) == name
            assert session.wiring_problem() is None, "and it must not open on a setting it warns about"
        finally:
            session.stop()


def test_pausing_lets_go_of_the_outputs():
    """**A paused session is not feeding anything back, so it must not hold the machine.**

    The feed loop skips the outputs while paused, which stopped them being *updated* rather than being
    *held*: the Shepard tone sang on at the pitch it had reached, the display stayed dimmed, the system volume
    stayed where the signal left it. Leaving for the device screen did release them, because `suspend` always
    has — which is exactly why pausing looked broken beside it.
    """
    session = _session()
    try:
        released = []
        for name in ("display", "output_volume", "music"):
            obj = getattr(session, name)
            setattr(obj, "release", (lambda n=name: lambda: released.append(n))())

        assert session.paused, "a session starts paused"
        session.toggle_pause()
        assert not session.paused and released == [], "resuming releases nothing"

        session.toggle_pause()
        assert session.paused
        assert released == ["display", "output_volume", "music"], "pausing hands all three back"

        # Suspending an already-paused session still releases, since it cannot rely on the toggle running.
        released.clear()
        session.suspend()
        assert released == ["display", "output_volume", "music"]
    finally:
        session.stop()


def test_the_slow_band_is_a_choice_and_the_old_one_is_still_there():
    """**A bare lowpass assumes the hardware supplies the lower edge, and ours cannot.**

    The `0–2×` band has no software high-pass, so the band reaches DC, exactly as it did at first.

    The Optima offers 0.5 Hz, which annihilates a 0.05 Hz band, and 0.01 Hz, which passes electrode drift
    whole. Nothing in between, and in between is where this band's edge belongs. Measured on a real capture:
    99% of the reward's variance was offset-driven and the product went negative 0% of the time where the two
    waves were on opposite sides of their baselines 16% of it.

    The `½` options take a low cut of `centre − bandwidth/2` with bandwidth = centre, chosen for these
    frequencies. `½–1½×` is that band entire.

    **The default is `¼–2×`**, and it is chosen on group delay: see `SLOW_BANDS`. At
    `centre/2` the edge carried as much delay as the whole lowpass.
    """
    from nfb.dsp import ProductSynchrony
    from nfb.dsp.product import DEFAULT_SLOW_BAND, SLOW_BANDS

    assert set(SLOW_BANDS) == {"0–2×", "¼–2×", "½–2×", "½–1½×"}
    assert SLOW_BANDS["0–2×"] == (0.0, 2.0), "the bare lowpass, untouched"
    assert SLOW_BANDS["½–1½×"] == (0.5, 1.5), "the low cut and high cut at half the centre"
    assert DEFAULT_SLOW_BAND == "¼–2×", "a band reaching DC is not what the product measure is for"

    # Proportional, so the shape is the same wherever the frequency is pointed.
    for hz in (0.05, 0.02, 0.01):
        for name, (lo, hi) in SLOW_BANDS.items():
            m = ProductSynchrony(specs(), (0, 1), hz, slow_band=name)
            assert m.lowpass_only
            assert (m.low, m.high) == pytest.approx((hz * lo, hz * hi)), (hz, name)

    # Above the fork nothing changes: the bandpass has its own edges and the choice does not apply.
    for name in SLOW_BANDS:
        alpha = ProductSynchrony(specs(), (0, 1), 10.0, slow_band=name)
        assert (alpha.low, alpha.high) == (8.5, 11.5)


def test_the_slow_band_keeps_the_wave_and_attenuates_the_drift():
    """What the default buys, as numbers rather than as an argument, **and what it gives up.**

    The edge exists so the two waveforms cross zero — a product rectified about zero is the whole measure, and
    two channels sitting on a standing offset of the same sign make it positive almost always. So the wave has
    to survive and the drift has to be pushed down, and the two pull against each other: rejection needs a
    corner close to the band, and a corner close to the band is delay.

    `¼–2×` is the chosen point and the trade is deliberate. Against the old `½–2×` it keeps *more* of the wave
    (99.8% against 97%) and carries **1.20 s** of group delay instead of 2.68 s — which on a 20 s cycle is a
    quarter of the reward's phase — and it pays for that with about 4× less rejection of drift a decade below.
    Both halves are asserted, because a later change that quietly gives up more rejection should have to argue
    with a number.
    """
    import numpy as np

    from nfb.dsp import ProductSynchrony
    from nfb.dsp.product import DEFAULT_SLOW_BAND

    chans = specs()
    for hz in (0.05, 0.01):
        seconds = 400 / hz                       # long enough to settle at either frequency
        tt = np.arange(int(seconds * RATE)) / RATE
        def through(sig, band=DEFAULT_SLOW_BAND):
            filt = ProductSynchrony(chans, (0, 1), hz, slow_band=band).filters[0]
            out = sig
            for stage in filt:
                out = stage.process(out)
            return out[len(out) // 2:]           # discard the settling half
        wave = through(np.sin(2 * np.pi * hz * tt))
        drift = through(np.sin(2 * np.pi * (hz / 10) * tt))
        assert wave.std() > 0.68, f"the trained wave survives at {hz} Hz ({wave.std():.2f})"
        assert drift.std() < 0.13, f"drift a decade below is pushed down ({drift.std():.2f})"

        # The wave is kept *better* than under the old default, which is the half of the trade that is free:
        # a corner further below the band is flatter inside it.
        was = through(np.sin(2 * np.pi * hz * tt), band="½–2×")
        assert wave.std() > was.std(), "a lower corner is flatter in the band, not merely faster"


def test_every_plot_can_be_given_a_window_and_a_scale():
    """**A constant is a guess about a session you are not in**, and three of them were wrong at once.

    The slow band's high-pass costs settling, and the settling transient is the largest thing in the window
    for minutes: at 0.05 Hz it pinned the slow plot's auto-scale at ±10 µV while the wave under it was below
    ±0.5, so the picture was empty and stayed empty until the transient scrolled out four minutes later. The
    synchrony plot auto-scales the same way. And the raw trace's ±100 µV is wrong in both directions at this
    corner — too small for drift, far too large for 8 µV of EEG.

    Auto stays on the ladder and stays the default, so nothing moves until somebody chooses a number.
    """
    session = _session()
    try:
        assert (session.wave_scale_uv, session.reward_scale) == (0.0, 0.0), "auto by default"
        assert session.wave_cycles == 12 and session.trace_seconds == 5.0

        # The wave window resizes in place, keeping its history — its stride does not depend on the
        # cycle count, so the same points serve any width. See the resize test for the full contract.
        before = session.wave
        session.set_view(wave_cycles=4)
        assert session.wave is before and session.wave.cycles == 4

        columns = session.reward_plot.columns
        session.set_view(reward_seconds=120.0)
        assert session.reward_plot.columns != columns

        session.set_view(trace_seconds=30.0)
        assert session.traces[0].per_column * 320 / RATE == pytest.approx(30.0, rel=0.05)

        # Scales are read at draw time and cost nothing, so they do not rebuild anything.
        wave = session.wave
        session.set_view(wave_scale_uv=2.0, reward_scale=5.0)
        assert session.wave is wave
        assert (session.wave_scale_uv, session.reward_scale) == (2.0, 5.0)

        session.set_view(wave_scale_uv=0.0)
        assert session.wave_scale_uv == 0.0, "and auto is reachable again"
    finally:
        session.stop()
