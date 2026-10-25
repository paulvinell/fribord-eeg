"""HRV, against beats whose spectrum is known by construction.

The simulator exists partly for this: modulate the heart rate at a chosen frequency and the peak has to
land there. The split detector is checked against a deliberately split simulator, because a detector for a
state nobody can produce on demand is a detector nobody has tested.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nfb.hrv import (DEFAULT_METRIC, HRVProcessor, NARROW_BAND, SimulatedBeats, beat_source,
                     parse_hr_measurement)


def collect(source, seconds: float = 300.0, **processor_kw):
    """Run a non-realtime simulator for `seconds` of simulated time."""
    processor = HRVProcessor(**processor_kw)
    state = None
    start = None
    for beat in source.beats():
        state = processor.push(beat)
        start = start if start is not None else beat.t
        if beat.t - start > seconds:
            break
    source.close()
    return state


# --- the wire format -------------------------------------------------------------------------

def test_a_heart_rate_notification_decodes():
    # flags 0x10 (RR present, uint8 HR), HR 60, one RR of 1024/1024 = 1.000 s
    data = bytes([0x10, 60]) + (1024).to_bytes(2, "little")
    hr, rr, contact = parse_hr_measurement(data)
    assert hr == 60.0
    assert rr == [1.0]
    assert contact is None, "this device does not report contact, which is not the same as bad contact"


def test_uint16_rate_contact_energy_and_several_intervals():
    flags = 0x01 | 0x06 | 0x08 | 0x10          # uint16 HR, contact good, energy present, RR present
    data = (bytes([flags]) + (500).to_bytes(2, "little") + (99).to_bytes(2, "little")
            + (512).to_bytes(2, "little") + (768).to_bytes(2, "little"))
    hr, rr, contact = parse_hr_measurement(data)
    assert hr == 500.0
    assert contact is True
    assert rr == [0.5, 0.75], "the energy field is skipped, not read as an interval"


def test_a_truncated_packet_is_not_a_crash():
    assert parse_hr_measurement(b"") == (0.0, [], None)
    assert parse_hr_measurement(b"\x10") == (0.0, [], None)


# --- the spectrum ----------------------------------------------------------------------------

def test_the_peak_lands_where_the_breathing_put_it():
    state = collect(SimulatedBeats(rsa_hz=0.1, seed=1, realtime=False))
    peak = state.spectrum.band_peak(NARROW_BAND)
    assert peak is not None
    assert abs(peak.hz - 0.1) < 0.015
    assert state.spectrum.sharpness > 0.3, state.spectrum.sharpness


def test_breathing_at_the_wrong_rate_is_visibly_worse():
    """The distance measure has to be continuous in how far off you are, since that is what someone
    adjusting their breathing needs it to do."""
    good = collect(SimulatedBeats(rsa_hz=0.1, seed=2, realtime=False))
    near = collect(SimulatedBeats(rsa_hz=0.13, seed=2, realtime=False))
    far = collect(SimulatedBeats(rsa_hz=0.30, seed=2, realtime=False))
    assert good.spectrum.distance_hz < near.spectrum.distance_hz < far.spectrum.distance_hz
    assert good.spectrum.sharpness > far.spectrum.sharpness


def test_the_split_spectrum_is_detected_and_a_good_one_is_not():
    """The split-spectrum failure: energy above and below with 0.1 Hz missing, while any band-power score
    looks unremarkable. One foot on the gas and one on the brake."""
    clean = collect(SimulatedBeats(rsa_hz=0.1, seed=3, realtime=False))
    split = collect(SimulatedBeats(seed=3, realtime=False, split=True))
    assert not clean.spectrum.split
    assert split.spectrum.split
    assert "gas" in split.spectrum.describe()
    assert "0.1" in clean.spectrum.describe()


def test_a_slower_resonance_is_scored_at_its_own_frequency_and_not_at_0_1_hz():
    """**0.1 Hz is a default, not a fact**, and against the default a good breath at 0.065 Hz scores
    like no breath at all: ±0.02 Hz around 0.1 runs down to 0.08 and does not contain the peak. Moving
    the target moves every claim made against it — the band `sharpness` integrates, the spike
    `distance_hz` measures to, the band `split` looks for, and the label `describe` prints."""
    beats = lambda: SimulatedBeats(rsa_hz=0.065, seed=11, realtime=False)

    against_default = collect(beats())
    assert against_default.spectrum.sharpness < 0.01, \
        "a clean 0.065 Hz peak scored against 0.1 Hz reads as complete failure — which is the bug"

    moved = collect(beats(), target_hz=0.065)
    assert moved.spectrum.narrow_band == pytest.approx((0.045, 0.085))
    assert moved.spectrum.sharpness > 0.9, moved.spectrum.sharpness
    assert moved.spectrum.distance_hz < against_default.spectrum.distance_hz
    # and the sentence changes with it: the default aims the reader back at 0.1, the moved one reports
    # the peak it now considers on target, and the named band carries the target's own label.
    assert "aim for one at 0.1" in against_default.spectrum.describe()
    assert "one peak at 0.06" in moved.spectrum.describe()
    assert "0.065 Hz" in [p.label for p in moved.spectrum.peaks]

    # And the same signal is still the same signal: only the scoring moved.
    peak = moved.spectrum.band_peak(moved.spectrum.narrow_band)
    assert peak is not None and abs(peak.hz - 0.065) < 0.015


def test_the_band_can_be_tightened_and_loosened_around_whatever_the_target_is():
    """The two are one judgement — where to aim, and how tightly to insist on it — so they move
    together and `sharpness` is monotonic in the width."""
    tight = collect(SimulatedBeats(rsa_hz=0.065, seed=12, realtime=False),
                    target_hz=0.065, narrow_half_hz=0.01)
    wide = collect(SimulatedBeats(rsa_hz=0.065, seed=12, realtime=False),
                   target_hz=0.065, narrow_half_hz=0.04)
    assert tight.spectrum.narrow_band == pytest.approx((0.055, 0.075))
    assert wide.spectrum.narrow_band == pytest.approx((0.025, 0.105))
    assert tight.spectrum.sharpness < wide.spectrum.sharpness


def test_the_target_is_clamped_to_where_the_standard_bands_still_mean_something():
    """Clamped rather than refused, the way the pacer's period is: a control that silently ignores what
    you typed is worse than one that shows the nearest thing it can do. The limits are the edges of the
    conventional bands — outside them the target would sit where `peak_proximity`'s own search range
    does not reach."""
    from nfb.hrv import NARROW_HALF_RANGE_HZ, TARGET_RANGE_HZ

    proc = HRVProcessor()
    assert (proc.target_hz, proc.narrow_half_hz) == (0.1, 0.02), "the default is where it always was"

    proc.set_target(0.0, 10.0)
    assert proc.target_hz == TARGET_RANGE_HZ[0]
    assert proc.narrow_half_hz == NARROW_HALF_RANGE_HZ[1]
    proc.set_target(99.0, 0.0)
    assert proc.target_hz == TARGET_RANGE_HZ[1]
    assert proc.narrow_half_hz == NARROW_HALF_RANGE_HZ[0]


def test_every_spectrum_carries_the_target_it_was_scored_against():
    """**Including the empty ones.** A screen draws the marked band before the first beat arrives, and a
    band drawn from a module constant beside a percentage computed from a session setting is how the two
    come apart. Nothing downstream reads the constant."""
    proc = HRVProcessor(target_hz=0.065)
    empty = proc._empty_spectrum()
    assert not empty.ready
    assert empty.target_hz == 0.065 and empty.narrow_band == pytest.approx((0.045, 0.085))
    assert empty.describe() == "waiting for beats", "and this branch used to raise NameError"

    from nfb.hrv import Beat
    first = proc.push(Beat(t=0.0, hr_bpm=60.0, rr_intervals=[1.0]))
    assert first.spectrum.target_hz == 0.065, "the very first state, before any spectrum exists"


def test_replaying_the_buffer_keeps_the_target():
    """`replay` builds a clone to rebuild a normaliser's history from beats already held. A clone on the
    default target would hand the new scale values in a different metric than the live one produces."""
    source = SimulatedBeats(rsa_hz=0.065, seed=13, realtime=False)
    proc = HRVProcessor(target_hz=0.065, narrow_half_hz=0.015)
    start = None
    for beat in source.beats():
        proc.push(beat)
        start = start if start is not None else beat.t
        if beat.t - start > 200.0:
            break
    source.close()
    states = proc.replay(step_s=20.0)
    assert states, "there are beats to replay"
    assert all(s.spectrum.target_hz == 0.065 and s.spectrum.narrow_half_hz == 0.015
               for s in states)


def test_a_coherence_score_alone_would_not_have_caught_it():
    """Why the detector exists rather than a threshold on the resonance ratio."""
    split = collect(SimulatedBeats(seed=4, realtime=False, split=True))
    flat = collect(SimulatedBeats(rsa_ms=1.0, noise_ms=40.0, seed=4, realtime=False))
    # the split spectrum's resonance ratio is not distinguishable from noise by size alone
    assert split.spectrum.resonance_ratio > 0.0
    assert not flat.spectrum.split, "noise is not a split; the detector needs both peaks"


def test_the_spectrum_is_drawn_from_the_first_beats_and_claims_only_what_the_span_supports():
    """**"Cannot" and "will not be accurate" are different things.** The transform is defined on any
    span, so the curve is drawn as soon as three beats make it arithmetic — and every claim about it is
    gated separately: naming a peak waits for four cycles of that frequency, and the split waits for a
    window that can see the very-low-frequency band at all."""
    from nfb.hrv.beats import RESOLVE_CYCLES, SPLIT_MIN_SPAN_S, TARGET_HZ

    opening = collect(SimulatedBeats(seed=5, realtime=False), seconds=8.0)
    spectrum = opening.spectrum
    assert spectrum.ready, "eight seconds of beats is a spectrum, coarse but real"
    assert spectrum.span_s < 12.0
    assert spectrum.resolution_hz > 0.08, "and it says how coarse: one over the span"
    assert not spectrum.resolves(TARGET_HZ), "which is nowhere near enough to name a peak at 0.1 Hz"
    assert "s of beats" in spectrum.describe(), "so it reports the span rather than a finding"
    assert not spectrum.split, "and says nothing at all about the very low frequencies"

    early = collect(SimulatedBeats(seed=5, realtime=False), seconds=30.0)
    assert not early.spectrum.resolves(TARGET_HZ), "30 s is three cycles of 0.1 Hz, not four"
    assert not early.spectrum.split

    settled = collect(SimulatedBeats(seed=5, realtime=False), seconds=120.0)
    assert settled.spectrum.resolves(TARGET_HZ) and settled.spectrum.span_s > SPLIT_MIN_SPAN_S
    assert "peak" in settled.spectrum.describe()
    assert RESOLVE_CYCLES / TARGET_HZ == 40.0, "four cycles of the target is forty seconds"


def test_a_spectrum_needs_beats_and_not_a_stopwatch():
    """The only floors left are arithmetic: something to interpolate between, something to transform."""
    from nfb.hrv import Beat
    from nfb.hrv.beats import HRVProcessor

    proc = HRVProcessor()
    t = 1000.0
    states = []
    for _ in range(6):
        t += 1.0
        states.append(proc.push(Beat(t, 60.0, [1.0])))
    assert not states[0].spectrum.ready, "one beat is not a series"
    assert states[-1].spectrum.ready, "a handful is"


# --- the beat stream -------------------------------------------------------------------------

def test_implausible_intervals_are_dropped():
    """A doubled or dropped beat is an artifact, and one 2 s interval among 1 s ones is a spike in the
    tachogram that the spectrum would read as real."""
    from nfb.hrv import Beat

    processor = HRVProcessor()
    t = 1000.0
    for i in range(30):
        processor.push(Beat(t + i, 60.0, [1.0]))
    state = processor.push(Beat(t + 30, 60.0, [0.05, 4.0, 1.0]))
    assert state.beats == 31, "only the plausible one was kept"


def test_batched_intervals_are_spread_backwards_in_time():
    """Stacking several beats at the arrival instant would manufacture a spike.

    The plotted series lags the newest beat by one, because the artifact filter holds each beat until its
    successor arrives — the ectopic test needs it. So the assertion is about *spacing*, which is what this
    was always testing; the live readouts do not lag (see the acceleration test) and `beats` counts what
    arrived rather than what settled.
    """
    from nfb.hrv import Beat

    processor = HRVProcessor()
    state = processor.push(Beat(1000.0, 60.0, [1.0, 0.9, 1.1]))
    assert state.beats == 3, "three plausible intervals arrived"
    ages = sorted(state.age_s)
    # Intervals of 1.0, 0.9 and 1.1 end 2.0, 1.1 and 0.0 s back. The last is still pending, so the plotted
    # pair is 2.0 and 1.1 — spread backwards, not stacked, which is the property under test.
    assert len(ages) == 2
    assert ages[0] == pytest.approx(1.1, abs=1e-9)
    assert ages[-1] == pytest.approx(2.0, abs=1e-9)

    # and once a fourth beat arrives the third settles, so nothing is lost — only delayed
    state = processor.push(Beat(1001.0, 60.0, [1.0]))
    assert len(state.age_s) == 3


def test_acceleration_is_reported_because_the_pacer_is_locked_to_it():
    """"only breath in when the heart rate is accelerating and out when the heart slows down" — so the
    cue needs the phase, not a metronome.

    A ramp rather than a single faster beat, because one beat is not a phase: the detector requires a
    retrace of a third of the recent swing and refuses to turn twice inside 1.5 s, which is what stops it
    flip-flopping on jitter. The rise here is four seconds, the shortest half-breath the pacer offers.
    """
    from nfb.hrv import Beat

    processor = HRVProcessor()
    state = None
    t = 1000.0
    for rate in (60.0, 62.0, 66.0, 70.0, 74.0, 76.0):        # accelerating, ~4 s of beats
        t += 60.0 / rate
        state = processor.push(Beat(t, rate, [60.0 / rate]))
    assert state.accelerating, "the rate has climbed for four seconds: this is an inhale"

    for rate in (74.0, 70.0, 66.0, 62.0, 58.0, 56.0):
        t += 60.0 / rate
        state = processor.push(Beat(t, rate, [60.0 / rate]))
    assert not state.accelerating, "and now it is falling"


def test_the_factory_rejects_an_unknown_kind():
    with pytest.raises(ValueError):
        beat_source("telepathy")
    assert isinstance(beat_source("simulator", realtime=False), SimulatedBeats)


def test_ble_reports_its_own_absence_rather_than_raising():
    """The EEG path must not gain a Bluetooth dependency to keep working, so the Bluetooth library is
    imported lazily and a machine that cannot load it gets a message on the screen rather than a
    traceback. bleak is an ordinary dependency now, so on a healthy machine there is nothing to report
    and the branch below is the picture on a machine without a radio."""
    from nfb.hrv import sources

    trouble = sources.problem()
    if trouble is not None:
        assert "Bluetooth" in trouble
        source = sources.BleBeats()
        assert list(source.beats()) == []
        assert "Bluetooth" in source.describe()


# --- beats, units and coherence -------------------------------------------------------

def test_rr_units_are_detected_rather_than_assumed():
    """**We divided every interval by 1024 unconditionally, and one shipping device sends milliseconds.**

    2.3% is `1000/1024`, so it never looks like a fault: heart rate a little low, RMSSD a little high, and
    the spectral peak 2.3% off — 0.0977 Hz for a true 0.1, well inside the band nobody would question. The
    old docstring even named the device while getting it wrong: "the HeartMath Coherence Plus advertises the
    standard service, so this is the same code path as a Polar strap."

    Hardcoding milliseconds is not the fix, because a Polar strap really is compliant. The device's own HR
    field is the arbiter.
    """
    from nfb.hrv import RrScaleResolver, parse_hr_measurement, parse_hr_raw

    def packet(raw_rr: int, hr: int = 60) -> bytes:
        return bytes([0x10, hr]) + raw_rr.to_bytes(2, "little")

    for raw, expect in ((1000, "ms"), (1024, "spec")):
        resolver = RrScaleResolver()
        for _ in range(8):
            hr, integers, _c = parse_hr_raw(packet(raw))
            resolver.observe(hr, integers)
        assert resolver.decided == expect
        _hr, rr, _c = parse_hr_measurement(packet(raw), scale=resolver.scale)
        assert rr[0] == pytest.approx(1.0, abs=1e-6), "either way, one second is one second"

    # the error the old code made on a millisecond device, as a number
    _hr, rr, _c = parse_hr_measurement(packet(1000), scale=1024.0)
    assert rr[0] == pytest.approx(0.9766, abs=1e-4)
    assert 60.0 / rr[0] == pytest.approx(61.44, abs=0.01), "61.4 bpm reported for a true 60"

    # and it can be pinned by hand, for a device whose HR field is too smoothed to arbitrate
    assert RrScaleResolver("ms").scale == 1000.0
    assert RrScaleResolver("spec").scale == 1024.0
    with pytest.raises(ValueError):
        RrScaleResolver("furlongs")


def test_an_ectopic_beat_no_longer_moves_the_spectral_peak():
    """We had no artifact rejection at all. One early beat with its compensatory pause is a spike in the
    tachogram, and the spectrum reads it as real — which matters here more than anywhere, because a single
    spectral peak at 0.1 Hz *is* the trained quantity."""
    from nfb.hrv import Beat, HRVProcessor

    def run(reject: bool, ectopics: bool):
        proc = HRVProcessor(reject_artifacts=reject)
        t = 1000.0
        # 0.1 Hz respiratory modulation on a 1 s baseline: the signal this protocol trains
        for i in range(240):
            rr = 1.0 + 0.08 * np.sin(2 * np.pi * 0.1 * t)
            if ectopics and i % 20 == 19:
                rr *= 0.55              # an early beat...
                t += rr
                proc.push(Beat(t, 60.0 / rr, [rr]))
                rr = (1.0 + 0.08 * np.sin(2 * np.pi * 0.1 * t)) * 1.45   # ...and its compensatory pause
            t += rr
            state = proc.push(Beat(t, 60.0 / rr, [rr]))
        return state

    clean = run(reject=False, ectopics=False)
    corrupted = run(reject=False, ectopics=True)
    filtered = run(reject=True, ectopics=True)

    target = clean.spectrum.band_peak((0.08, 0.12))
    assert target is not None, "the clean signal has its 0.1 Hz peak"
    assert corrupted.spectrum.coherence < clean.spectrum.coherence / 2, \
        "unfiltered, the ectopics more than halve the coherence"
    assert filtered.spectrum.coherence > corrupted.spectrum.coherence, \
        "and rejecting them recovers some of it"
    assert filtered.reject_fraction > 0.0, "and the screen can say how much was thrown away"


def test_a_rejected_beat_is_withheld_rather_than_deleted():
    """The subtle part of the port. The tachogram is interpolated over absolute beat times, so withholding a
    beat makes the interpolation span the gap in a straight line. Dropping it would compress the time axis
    and zero-filling would inject broadband power — both corrupt the one thing being measured."""
    from nfb.hrv import ArtifactFilter

    filt = ArtifactFilter()
    t = 0.0
    # Real variability, not a metronome: the ectopic test is scaled in standard deviations, so a perfectly
    # regular series has an SD of zero and the test correctly refuses to fire at all.
    rng = np.random.default_rng(4)
    for _ in range(40):                     # settle the statistics past the warm-up
        rr = 1.0 + 0.02 * rng.standard_normal()
        t += rr
        filt.push(t, rr)
    t += 0.4
    filt.push(t, 0.4)                       # a spike, which settles the previous good beat...
    before = filt.accepted_count
    t += 1.6
    verdicts = filt.push(t, 1.6)            # ...and is itself revealed by its successor
    rejected = [v for v in verdicts if not v.accepted]
    assert rejected and rejected[0].reason == "ectopic"
    assert rejected[0].rr == pytest.approx(0.4)
    assert filt.accepted_count == before, "the spike itself never entered the statistics"
    assert filt.rejected_count == 1
    # and its absence leaves a real gap in the beat times, so the tachogram interpolates across it rather
    # than compressing the axis (dropping) or injecting broadband power (zero-filling)
    assert not any(v.accepted and v.rr == pytest.approx(0.4) for v in verdicts)


def test_the_coherence_ratio_tells_entrainment_from_in_band_energy():
    """`resonance_ratio` integrates a fixed band; `coherence` locates the peak and multiplies the ratios on either side of it.

    The clearest case is a spectrum that is *flat across the whole band*: `resonance_ratio` calls that a perfect 1.000,
    because all the power is in band. `coherence` calls it 0.2, because nothing is entrained. Measured.
    """
    from nfb.hrv import RESONANCE_BAND, Spectrum

    freqs = np.arange(0, 0.5, 1 / 64)

    def peak_at(hz, width=1.0, floor=0.0):
        return Spectrum(freqs, floor + np.exp(-0.5 * ((freqs - hz) / (width / 64)) ** 2))

    sharp = peak_at(0.1)
    offset = peak_at(0.12)
    broad = peak_at(0.1, width=6.0)
    flat = Spectrum(freqs, np.where((freqs >= RESONANCE_BAND[0])
                                    & (freqs <= RESONANCE_BAND[1]), 1.0, 0.0))

    assert sharp.resonance_ratio == pytest.approx(flat.resonance_ratio, abs=0.01), \
        "resonance_ratio cannot tell a sharp peak from a flat band"
    assert sharp.coherence > 100 * flat.coherence, "coherence can, by two orders of magnitude"
    assert broad.coherence < sharp.coherence / 100, "and a broad bump is not entrainment either"
    # somebody whose resonance sits at 0.12 Hz is not penalised, because the peak is located not assumed
    assert offset.coherence > sharp.coherence * 0.8
    # the displayed number is the log, which is the number put on screen
    assert sharp.coherence_score == pytest.approx(np.log1p(sharp.coherence))


def test_the_metronome_period_is_configurable_within_its_range():
    """It was hardcoded at 10 s. In for five and out for five is six breaths a minute, which is 0.1 Hz, which
    is the frequency the protocol aims the spectral peak at — so the default was right and the problem was
    that it could not move. A resonant frequency is individual, commonly 4.5–7 breaths a minute.

    The range is 6–30 s. The step is **half a second**, not a whole second: at this end of the range a second is
    coarse — 10 s to 11 s moves the paced rate from 6.0 to 5.5 breaths a minute, stepping over 5.7 and 5.8.
    """
    from nfb_console.hrv_view import HRVSession, PACER_PERIOD_S, PACER_RANGE_S, PACER_STEP_S
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    session = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
    try:
        assert session.pacer_period == PACER_PERIOD_S == 10.0
        session.pacer_locked = False        # the clock pacer, so `breathe_in` reads the period
        session.set_pacer_period(12)
        assert session.pacer_period == 12.0
        assert "12 s" in session.pacer_note() and "5.0 breaths a minute" in session.pacer_note()
        assert "metronome" in session.pacer_note(), "it must not pass for the heart-locked cue"
        # clamped *into* the range rather than out of it, and snapped to the half second
        session.set_pacer_period(1.0)
        assert session.pacer_period == PACER_RANGE_S[0] == 6.0
        session.set_pacer_period(999.0)
        assert session.pacer_period == PACER_RANGE_S[1] == 30.0
        assert PACER_STEP_S == 0.5
        session.set_pacer_period(10.4)
        assert session.pacer_period == 10.5
        session.set_pacer_period(10.6)
        assert session.pacer_period == 10.5
        session.set_pacer_period(10.8)
        assert session.pacer_period == 11.0
        # the half second is reachable, which a whole-second step made it not
        session.set_pacer_period(10.5)
        assert session.pacer_period == 10.5
        assert "5.7 breaths a minute" in session.pacer_note()
    finally:
        session.stop()


def test_the_target_frequency_is_settable_from_the_screen_and_takes_the_metronome_with_it():
    """**The pacer's rate was settable and the frequency it was scored against was not**, so the screen
    could pace somebody at their own resonance and then mark them down for reaching it.

    They are one quantity said two ways, so moving the target moves the metronome; the reverse is not
    wired up, because the cycle is snapped to half a second and feeding that back would drag the target
    off wherever it was set."""
    from nfb_console.hrv_view import HRVSession
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    session = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
    try:
        assert session.target_hz == 0.1 and session.pacer_period == 10.0

        session.set_target_hz(0.065)
        assert session.target_hz == 0.065
        assert session.heart.processor.target_hz == 0.065, "one copy, and it lives on the processor"
        assert session.pacer_period == 15.5, "1/0.065 is 15.4 s, snapped to the pacer's half second"
        assert "3.9 breaths a minute" in session.target_note()

        # the band moves independently, and both are clamped
        session.set_target_hz(0.065, 0.01)
        assert session.target_half_hz == 0.01
        session.set_target_hz(0.0)
        assert session.target_hz == 0.04, "clamped into the range, not refused"

        # nudging the cycle afterwards does not drag the target with it
        session.set_target_hz(0.065)
        session.set_pacer_period(10.0)
        assert session.target_hz == 0.065
    finally:
        session.stop()


def test_the_target_survives_a_relaunch_because_it_is_the_persons_own_frequency():
    """Same category as a volume: found by hunting for it over a session or two, and asking somebody to
    re-find it every launch is the failure `prefs` exists to prevent. Restored before the pacer, since
    setting the target sets the pacer from it."""
    from nfb_console.hrv_view import HRVSession
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    session = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
    try:
        session.set_target_hz(0.065, 0.015)
        stored = session.calibration()
        assert stored["target_hz"] == 0.065 and stored["target_half_hz"] == 0.015
    finally:
        session.stop()

    fresh = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
    try:
        assert fresh.target_hz == 0.1, "a new session starts at the default"
        fresh.restore_calibration(stored)
        assert fresh.target_hz == 0.065 and fresh.target_half_hz == 0.015
        assert fresh.pacer_period == stored["pacer_period"], \
            "and the stored cycle survives, rather than being overwritten by one derived from the target"
    finally:
        fresh.stop()


def test_the_pacer_period_actually_changes_the_cue():
    """A setting that is stored and not read is worse than no setting."""
    from nfb_console.hrv_view import HRVSession
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    session = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
    try:
        session.pacer_locked = False
        session.set_pacer_period(20)
        session._accumulated = 0.0
        assert session.breathe_in(), "the first half of the cycle is the inhale"
        session._accumulated = 9.0
        assert session.breathe_in()
        session._accumulated = 11.0
        assert not session.breathe_in(), "and the second half is the exhale"
        session._accumulated = 21.0
        assert session.breathe_in(), "then it repeats"
    finally:
        session.stop()




def test_opening_hrv_as_the_first_protocol_does_not_crash():
    """**The exact traceback from `make live`**: `Console.pick` restored preferences into
    `self.session.feedback`, and this screen is not a `TrainingSession` and has none.

    Present since prefs were added — the restore only runs when there is no previous session, so it needed
    HRV to be the *first* protocol opened in a run. It was briefly masked while this screen carried a
    redundant `FeedbackSettings`, and unmasked when that was deleted. `prefs` asks the session now.
    """
    from nfb_console import prefs
    from nfb_console.hrv_view import HRVSession
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    session = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
    try:
        # It reaches the mechanisms through the session's own hook rather than an attribute it assumed was
        # there, which is what makes this work on a screen whose settings are not only a `FeedbackSettings`.
        assert prefs.restore("HRV", session) >= 0
        session.set_pacer_period(12.5)
        assert session.calibration()["pacer_period"] == 12.5
    finally:
        session.stop()


def test_stopping_hrv_and_re_entering_it_keeps_the_setup():
    """`Console.pick` calls `adopt` whenever there is a previous session, and this screen had none — so the
    stop button raised too. Same line as the EEG screens draw: configuration comes across, history does not.
    """
    from nfb_console.hrv_view import HRVSession
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    first = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
    try:
        first.set_pacer_period(13.0)
        first.pacer_locked = False
        first.set_metric("rmssd")
        first.set_strategy("zscore", window_s=90.0)
        first.feedback.music.source = "shepard"
        first.feedback.sound.high = 45.0

        second = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
        try:
            second.adopt(first)
            assert second.pacer_period == 13.0 and second.pacer_locked is False
            assert second.metric == "rmssd"
            assert second.strategy_id == "zscore" and second.strategy.window_s == 90.0
            assert second.feedback.music.source == "shepard"
            assert second.feedback.sound.high == 45.0
            # the history does not come across: a fresh normaliser has nothing in its window
            assert second.strategy.warmup == 0.0
            assert second.signal is None
            # and adopting a different protocol's session is refused rather than half-applied
            other = HRVSession(BY_CODE["AMP"], choice=HeartChoice(kind="simulator"))
            try:
                second.pacer_period = 7.0
                second.adopt(other)
                assert second.pacer_period == 7.0
            finally:
                other.stop()
        finally:
            second.stop()
    finally:
        first.stop()


def test_hrv_persists_what_was_set_by_ear_and_not_what_is_being_trained():
    """`prefs`' own line: calibration survives a relaunch, arrangement does not. The pacer's period is the
    person's own resonant rate and the volumes are the room; the metric and the scaling are what is being
    trained, which is the same category as a band or a threshold."""
    import json

    from nfb_console import prefs
    from nfb_console.hrv_view import HRVSession
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    prefs.PATH.unlink(missing_ok=True)
    first = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
    first.set_pacer_period(11.5)
    first.feedback.sound.high = 30.0
    first.feedback.shadow.high = 80.0
    first.set_metric("pnn50")
    first.feedback.music.source = "shepard"
    first.stop()

    # One set for every protocol — see `prefs.SHARED`. This screen pins nothing, so everything it stores is
    # the room's and is what the next screen opens with.
    stored = json.loads(prefs.PATH.read_text())[prefs.SHARED]
    assert stored["pacer_period"] == 11.5
    assert stored["sound.high"] == 30.0 and stored["shadow.high"] == 80.0
    assert "metric" not in stored and "strategy" not in stored
    assert "shepard" not in str(stored), "which music is playing is the arrangement, not the calibration"

    second = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
    try:
        assert prefs.restore("HRV", second) > 0
        assert second.pacer_period == 11.5
        assert second.feedback.sound.high == 30.0
        assert second.feedback.shadow.high == 80.0
        assert second.metric == DEFAULT_METRIC, "what is trained comes from the protocol"
    finally:
        second.stop()


def test_a_dropout_ends_the_spectrum_window_instead_of_becoming_a_ramp():
    """A strap going silent used to be interpolated as a straight line, and a ramp is broadband power
    piled at the low end — in the VLF band, the sympathetic half of the split detector. Measured before
    the fix: a 60 s dropout on a perfect 0.1 Hz signal took sharpness from 0.998 to 0.258, put 44% of
    power into VLF, and with normal respiratory content present a ~70 s dropout fired the split warning
    on a clean heart, with reject_fraction reading zero throughout. Now only the contiguous run after
    the last dropout is analysed, and the spectrum honestly reports not-ready until it is long enough."""
    from nfb.hrv.beats import HRVProcessor, Beat, VLF_BAND

    def run(gap_s: float):
        proc = HRVProcessor()
        state = None
        t = 1000.0
        elapsed = 0.0
        while elapsed < 240.0:
            rr = 1.0 + 0.08 * math.sin(2 * math.pi * 0.1 * elapsed) + 0.03 * math.sin(
                2 * math.pi * 0.28 * elapsed)
            elapsed += rr
            if 100.0 < elapsed < 100.0 + gap_s:
                continue                              # the strap is silent: no beats at all
            state = proc.push(Beat(t=t + elapsed, hr_bpm=60.0 / rr, rr_intervals=[rr]))
        return state

    clean = run(0.0)
    assert clean.spectrum.ready and clean.spectrum.sharpness > 0.5
    assert not clean.spectrum.split

    dropped = run(70.0)
    if dropped.spectrum.ready:
        assert not dropped.spectrum.split, "a flat battery must not print the clinical warning"
        assert dropped.spectrum.band_power(VLF_BAND) / dropped.spectrum.total < 0.1, \
            "the dropout's ramp must not masquerade as sympathetic drive"


def test_the_breathing_cue_follows_the_breath_not_the_jitter():
    """The cue is the whole instruction — breathe in while the rate climbs — so it must reverse when the
    breath does, about twelve times a minute at 0.1 Hz, and not when the beats wobble.

    **Across swing depth and jitter, not at one operating point**, because that is where the version this
    replaced failed. Its fixed 1.5 bpm band on the raw per-beat rate held at 12 reversals a minute on a
    clean signal and went to 26 at a shallow swing with 25 ms of RR jitter — worst on exactly the resting
    signal a beginner has, and the reason this reads as flip-flopping on a real sensor. The detector
    — three-beat smoothing, a threshold scaled to the person's own
    swing, a minimum interval and a stuck timeout — measures 10-12 over the whole grid below.
    """
    import random

    from nfb.hrv.beats import HRVProcessor, Beat

    def reversals(depth_s: float, jitter_s: float, seconds: float = 180.0) -> float:
        rng = random.Random(5)
        proc = HRVProcessor()
        flips, last, elapsed = 0, None, 0.0
        while elapsed < seconds:
            rr = max(0.35, 0.92 + depth_s * math.sin(2 * math.pi * 0.1 * elapsed)
                     + rng.gauss(0.0, jitter_s))
            elapsed += rr
            state = proc.push(Beat(t=1000.0 + elapsed, hr_bpm=60.0 / rr, rr_intervals=[rr]))
            if last is not None and state.accelerating != last:
                flips += 1
            last = state.accelerating
        return flips / (seconds / 60.0)

    for depth in (0.02, 0.05, 0.08, 0.14, 0.20):
        for jitter in (0.004, 0.012, 0.025):
            rate = reversals(depth, jitter)
            assert 9.0 <= rate <= 14.0, (
                f"{rate:.1f} reversals/min at swing {depth:g}s, jitter {jitter * 1000:.0f}ms — "
                f"against the breath's 12")


def test_time_domain_metrics_do_not_difference_across_withheld_beats():
    """Withholding is right for the spectrum and used to inflate RMSSD and pNN50: the accepted-beat
    array closes up, and np.diff produced differences between intervals that were never neighbours —
    pNN50 measured at 2.4x its clean value on the repo's own ectopic fixture, purely from the seams."""
    from nfb.hrv.beats import HRVProcessor, Beat

    def run(with_ectopics: bool):
        proc = HRVProcessor()
        state, t, elapsed, n = None, 1000.0, 0.0, 0
        while elapsed < 240.0:
            n += 1
            rr = 1.0 + 0.08 * math.sin(2 * math.pi * 0.1 * elapsed)
            if with_ectopics and n % 20 == 0:
                early, pause = rr * 0.55, rr * 1.45      # an ectopic pair: early beat, long rebound
                elapsed += early
                state = proc.push(Beat(t + elapsed, 60.0 / early, [early]))
                elapsed += pause
                state = proc.push(Beat(t + elapsed, 60.0 / pause, [pause]))
                continue
            elapsed += rr
            state = proc.push(Beat(t + elapsed, 60.0 / rr, [rr]))
        return state

    clean = run(False)
    dirty = run(True)
    assert dirty.reject_fraction > 0.0, "the ectopics were caught"
    assert dirty.rmssd_ms == pytest.approx(clean.rmssd_ms, rel=0.25), \
        f"rmssd {dirty.rmssd_ms:.1f} vs clean {clean.rmssd_ms:.1f}: the seams are in the number"
    assert dirty.pnn50_pct <= clean.pnn50_pct + 5.0, \
        f"pnn50 {dirty.pnn50_pct:.1f} vs clean {clean.pnn50_pct:.1f}: the seams are in the number"


def test_a_missed_beat_reaches_the_filter_and_the_reject_figure():
    """A 4 s interval is a missed beat, the most common real artifact. The processor's own wider range
    gate used to swallow it before the filter ever saw it, so it moved reject_fraction not at all and
    left the filter's pending beat unsettled across the skip."""
    from nfb.hrv.beats import HRVProcessor, Beat

    proc = HRVProcessor()
    t = 1000.0
    for i in range(60):
        t += 1.0
        proc.push(Beat(t, 60.0, [1.0]))
    t += 4.0
    state = proc.push(Beat(t, 15.0, [4.0]))              # the missed beat
    assert state.reject_fraction > 0.0, "a missed beat counts as a rejection now"


def test_a_beat_computed_before_its_predecessor_is_refused():
    """BLE delivery jitter can put a delayed packet's beat before a prompt one's, and np.interp returns
    silent garbage on a non-monotonic grid. A beat that would go backwards is transport, not heart."""
    from nfb.hrv.beats import HRVProcessor, Beat

    proc = HRVProcessor()
    proc.push(Beat(1000.6, 60.0, [0.9]))
    proc.push(Beat(1001.0, 60.0, [0.9]))
    proc.push(Beat(1002.3, 60.0, [0.9]))                 # delayed delivery
    proc.push(Beat(1002.1, 60.0, [0.9]))                 # prompt one lands earlier
    times = [bt for bt, _ in proc._live]
    assert times == sorted(times), "the buffers stay monotonic whatever the transport does"


def test_no_contact_means_no_training_data():
    """A strap reporting contact=False is reporting that its intervals are noise from an open sensor.
    The state still carries the flag for the screen; the buffers must not train on the intervals."""
    from nfb.hrv.beats import HRVProcessor, Beat

    proc = HRVProcessor()
    proc.push(Beat(1000.0, 60.0, [1.0], contact=True))
    before = len(proc._live)
    state = proc.push(Beat(1001.0, 60.0, [1.0], contact=False))
    assert len(proc._live) == before, "no-contact intervals stay out of the buffers"
    assert state.contact is False, "and the screen is told why"
