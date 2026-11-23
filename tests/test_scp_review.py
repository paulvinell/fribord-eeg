"""The review of the SCP data-quality branch: one test per kept defect, S1 to S25, one per mutant that
survived the mutation run, and one for the replay of an extra channel at the input limit.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from nfb.stream import Block, ChannelSpec, Gap
from nfb_console import scp_view
from nfb_console.trials import ACTIVE, BASELINE, LOST, POSITIVE, REJECTED, RESULT, TrialRun

from conftest import counts, wandering
from test_scp_view import (BLOCK, RATE, _feed, _joined, _page, _ring_view, _run, _screen, _session,
                            _until_done)


def _limited_spec(index: int = 0) -> ChannelSpec:
    return ChannelSpec(index=index, rate_hz=RATE, unit="µV", scale=800.0 / 32767, input_range=800.0)


def _block(spec: ChannelSpec, start: int, uv) -> Block:
    return Block(spec.index, start, tuple(counts(uv, spec.scale).tolist()), spec)


def _held(run, size: float) -> float:
    """A shift of `size` toward the cue that rises over the first second of the effort and falls over the
    first second of the verdict, so the gate never sees a step."""
    if run.required is None or run.phase not in (ACTIVE, RESULT):
        return 0.0
    sign = 1.0 if run.required == POSITIVE else -1.0
    if run.phase == ACTIVE:
        part = min(1.0, (run.active_s - run.phase_remaining()) / 1.0)
    else:
        part = max(0.0, 1.0 - (run.result_s - run.phase_remaining()) / 1.0)
    return sign * size * part


# --- S1 -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fed, pressed", [(0, 1), (1, 0)], ids=["training-block", "extra-block"])
def test_s1_a_channel_press_while_a_block_is_in_flight_does_not_stop_the_session(fed, pressed):
    """A press between `observe_raw` and `consume`: on a block of the training channel that the press makes
    an extra, and on an extra's block while the press rebuilds the channels."""
    session = _session(channels=2)
    try:
        session.toggle_pause()
        original = session.observe_raw

        def press(block, judged):
            original(block, judged)
            session.set_channel(pressed)

        session.observe_raw = press
        spec = session.specs[fed]
        try:
            session.feed_block(_block(spec, 0, wandering(1.5, 4.0, RATE, 1, 2.0)[:BLOCK]))
        except KeyError as exc:
            pytest.fail(f"the feed raises KeyError {exc}, and the session stops")
    finally:
        session.stop()


# --- S2 -------------------------------------------------------------------------------------------

def test_s2_a_success_said_good_during_its_trial_and_a_short_shift_never_did():
    """"Good" is what the click and the music ask: the condition target, met past the frozen limit."""
    session = _session()
    try:
        session.toggle_pause()
        session.set_success_at(30.0, typed=True)
        session.set_full_at(20.0, typed=True)
        run = _run(session, n_trials=2, transfer=0.0, seed=21)
        size = {1: 48.0, 2: 24.0}
        good: dict[int, bool] = {1: False, 2: False}

        def watch():
            if run.phase == ACTIVE and session.active_targets and session.signal_target.wanted:
                good[run.trial_number] = True

        x = wandering(1.5, 60.0, RATE, 51, 2.0)
        _feed(session, {0: x}, block=25, watch=watch, until=_until_done(session),
              add=lambda s, _ch: _held(s.trials, size[s.trials.trial_number]))
        first, second = run.results
        assert first.valid and first.success, first
        assert second.valid and not second.success, second
        assert not good[2], "a shift short of succeeds at said good"
        assert good[1], "a trial that succeeds never said good while it ran"
    finally:
        session.stop()


# --- S3 -------------------------------------------------------------------------------------------

def test_s3_the_share_on_a_void_line_is_the_share_of_the_named_cause():
    session = _session(channels=1)
    try:
        run = TrialRun(n_trials=1, transfer_fraction=0.0, seed=1, metric_hz=10.0)
        k, t = 0, 0.0
        skipped = False
        while not run.results:
            in_trial = run.phase in (BASELINE, ACTIVE)
            if run.phase == ACTIVE and run.phase_remaining() < 3.0 and not skipped:
                t += 0.1                                # one reading never arrives
                skipped = True
            run.feed(t, 5.0, valid=not (in_trial and k % 4 == 0), cause=REJECTED)
            k += 1
            t = round(t + 0.1, 6)
        session.trials = run
        trial = run.results[0]
        assert trial.cause == REJECTED, "the count with the most readings is named"
        what, _action = session.void_words(trial)
        expected = run.baseline_s * 10 + run.active_s * 10
        rejected = trial.invalid_share - 1 / expected          # all but the one that did not arrive
        shares = [int(p) for p in re.findall(r"(\d+)%", what)]
        assert shares, what
        assert abs(shares[0] - 100 * rejected) <= 1, f"{what!r}: the rejected share alone, not the lost one"
    finally:
        session.stop()


# --- S4 -------------------------------------------------------------------------------------------

def test_s4_an_extra_channel_of_a_file_with_a_standing_offset_has_a_signal(tmp_path):
    from nfb.sources import SampleFileSource

    n = int(10 * RATE)
    rng = np.random.default_rng(4)
    ch0 = wandering(1.5, 10.0, RATE, 61, 2.0)
    ch1 = 100.0 + 3.0 * rng.standard_normal(n)
    path = tmp_path / "export.txt"
    path.write_text("Sampling rate: 250\n" + "".join(f"{a:.5f}\t{b:.5f}\n" for a, b in zip(ch0, ch1)))
    source = SampleFileSource(path, realtime=False, loop=False)
    session = _session(channels=2, source=source)
    try:
        session.set_extra(1, True)
        for block in source.blocks():
            session.feed_block(block)
        line, _problem = session.extra_status(1)
        assert "no signal" not in line, line
        assert session.headroom_uv(1) is None and not session.at_input_limit(1), "an export has no input limit"
    finally:
        session.stop()


# --- S5 -------------------------------------------------------------------------------------------

def test_s5_the_signal_pane_does_not_say_nothing_is_rejected_while_a_block_is_held():
    from nfb.dsp.artifact import ArtifactGates, ArtifactPolicy

    gates = ArtifactGates(policy=ArtifactPolicy(enabled=False))
    spec = _limited_spec()
    uv = np.zeros(8)
    uv[3] = -800.0
    judged = gates.feed(_block(spec, 0, uv))
    assert not judged.accepted, "the block at the input limit is held"
    assert "holding" in gates.describe(), f"the pane says {gates.describe()!r}"
    assert gates.percent > 0, "the held block is counted where the pane counts"


# --- S6 -------------------------------------------------------------------------------------------

def test_s6_a_trial_across_the_seam_of_a_looped_file_has_no_result(tmp_path):
    from nfb.sources import SampleFileSource

    seconds = 30.0
    tt = np.arange(int(seconds * RATE)) / RATE
    rise = 0.5 - 0.5 * np.cos(np.pi * np.clip((tt - 15.0) / 10.0, 0.0, 1.0))
    x = -300.0 + 200.0 * rise + wandering(1.5, seconds, RATE, 71, 2.0)
    path = tmp_path / "export.txt"
    path.write_text("Sampling rate: 250\n" + "".join(f"{v:.5f}\n" for v in x))
    source = SampleFileSource(path, realtime=False, loop=True)
    session = _session(channels=1, source=source)
    try:
        session.toggle_pause()
        run = _run(session, n_trials=12, transfer=0.0, seed=22)
        across: set[int] = set()
        for block in source.blocks():
            if block.seam and run.phase in (BASELINE, ACTIVE):
                across.add(run.trial_number)
            session.feed_block(block)
            if run.done or (across and len(run.results) >= max(across)):
                break
        assert across, "no trial ran across a seam"
        scored = [run.results[i - 1] for i in sorted(across)]
        assert not any(trial.valid for trial in scored), [(t.index, t.score_uv) for t in scored]
        assert all(trial.cause == "seam" for trial in scored), [t.cause for t in scored]
    finally:
        session.stop()


# --- S7 ---

def test_s7_the_slow_level_is_offered_no_mode_that_feeds_it_zeros():
    """Clamp sets a rejected sample to zero, a step of the whole standing offset on a slow level. SCP does
    not offer it, with the reason stated on the session; the Feedback tab draws only what is offered."""
    session, control = _screen(channels=1)
    try:
        assert session.reject_modes == ("hold",)
        joined = _joined(session, control)
        assert "hold" in joined.split(" | ") and "clamp" not in joined.split(" | "), joined
    finally:
        session.stop()


# --- S8 -------------------------------------------------------------------------------------------

def test_s8_a_stall_whose_frames_arrive_later_loses_no_reading():
    from nfb.heg import HegState
    from nfb.heg.processor import QUALITY_GOOD

    from test_heg_trials import _session as heg_session

    session = heg_session(n_trials=1)
    try:
        frame = 0
        while not session.trials.results:
            made = frame / 30.0
            phase = made % 2.0
            arrival = made if phase < 1.6 else made - phase + 1.998 + 0.0025 * (phase - 1.6)
            state = HegState(t=arrival, smoothed_hbo=10.0 + 0.3 * math.sin(made), smoothed_hbr=0.0,
                             quality=QUALITY_GOOD, samples_seen=1000, windowed_ready=True)
            session.heg.state = state
            session.heg.readings.append(state)
            if frame % 3 == 0:
                session.observe()
            frame += 1
            assert made < 400.0, "no trial came"
        trial = session.trials.results[0]
        assert trial.valid, trial
        assert trial.invalid_share == pytest.approx(0.0), f"{trial.invalid_share:.0%} of the readings lost"
    finally:
        session.stop()


def test_s8_frames_that_never_arrive_are_still_lost_readings():
    """The control: a stall whose frames never come is a hole, and its readings count as lost."""
    from nfb.heg import HegState
    from nfb.heg.processor import QUALITY_GOOD

    from test_heg_trials import _session as heg_session

    session = heg_session(n_trials=1)
    try:
        frame = 0
        while not session.trials.results:
            made = frame / 30.0
            frame += 1
            if made % 2.0 >= 1.6:
                continue                                # this frame is lost on the way
            state = HegState(t=made, smoothed_hbo=10.0 + 0.3 * math.sin(made), smoothed_hbr=0.0,
                             quality=QUALITY_GOOD, samples_seen=1000, windowed_ready=True)
            session.heg.state = state
            session.heg.readings.append(state)
            if frame % 3 == 0:
                session.observe()
            assert made < 400.0, "no trial came"
        assert session.trials.results[0].invalid_share > 0.1
    finally:
        session.stop()


# --- S9 -------------------------------------------------------------------------------------------

def _write_recording(path, start_group: int, seconds: float, hz: float = 250.0) -> None:
    """A recording that starts `start_group` sample groups into a session: frames of eight groups whose
    phase is the wrapped counter, as the device sends them, with a wandering level on every channel."""
    import struct

    import optima
    from nfb.record import SessionHeader, SessionWriter
    from nfb.stream import PHASE_MODULUS

    cfg = optima.build_eeg_config(4, hz=hz)
    plan = [c for c in optima.plan_channels(cfg) if c.enabled]
    n = int(seconds * hz)
    waves = [counts(wandering(1.5, seconds, hz, 80 + p, 2.0), 800.0 / 32767) for p in range(len(plan))]
    header = SessionHeader(config=cfg, device="Neurobit Optima+ 4 USB", serial="00000000",
                           firmware="2.9.5", note="synthetic recording started mid-session")
    with SessionWriter(path, header) as writer:
        for frame in range(n // 8):
            phase = (start_group + frame * 8) % PHASE_MODULUS
            values = [int(waves[p][frame * 8 + step]) for step in range(8) for p in range(len(plan))]
            payload = struct.pack("<HH", phase, 0) + struct.pack(f"<{len(values)}h", *values)
            writer.write(optima.NOT_DATA, payload, elapsed=frame * 8 / hz)


def test_s9_a_marks_sample_index_is_the_live_streams_and_its_wall_time_places_it(tmp_path):
    """The claim that a mark's index is a sample of any recording is gone: a recording that starts
    mid-session replays from its wrapped phase. The index is the live stream's, and the mark keeps the
    wall time that places it in a recording."""
    from nfb import ReplaySource

    live_start = 70000
    path = tmp_path / "mid-session.jsonl.gz"
    _write_recording(path, live_start, 12.0)
    replayed = [b for b in ReplaySource(str(path), realtime=False, loop=False).blocks()
                if isinstance(b, Block)]
    offset = live_start - min(b.start_index for b in replayed)
    live = set()
    for b in replayed:
        if b.channel == 0:
            live.update(range(b.start_index + offset, b.end_index + offset))
    session = _session(channels=1, source=ReplaySource(str(path), realtime=False, loop=False))
    try:
        session.toggle_pause()
        _run(session, n_trials=1, transfer=0.0, seed=24)
        for b in replayed:
            if b.channel in session.specs:
                session.feed_block(Block(b.channel, b.start_index + offset, b.samples, b.spec, b.state))
        cues = [(at, note) for at, note in session.marks() if " · cue · " in note]
        assert cues, session.marks()
        at, note = cues[0]
        assert int(re.search(r"ch0 sample (\d+)", note).group(1)) in live
        assert at > 1e9, "a wall time, which places the mark in a recording"
        assert "recording" not in (scp_view.SCPSession.marks.__doc__ or "").split(".")[0]
    finally:
        session.stop()


# --- S10 ------------------------------------------------------------------------------------------

def test_s10_a_trial_before_a_rearm_is_not_checked_on_signal_from_after_it():
    session, _control = _screen()
    try:
        run = _run(session, n_trials=14, transfer=0.0, seed=25)
        flat = np.zeros(int(200 * RATE))
        _feed(session, {0: wandering(1.5, 200.0, RATE, 90, 2.0), 1: flat},
              until=lambda _s: len(run.results) >= 5)
        before = len(run.results)
        session.feed_block(Gap(0, 0))
        _feed(session, {0: wandering(1.5, 200.0, RATE, 91, 2.0), 1: wandering(1.5, 200.0, RATE, 92, 2.0)},
              until=_until_done(session))
        assert run.done
        session.set_extra(1, True)
        checks = session.extra_checks()[1][:before]
        assert checks == [None] * before, f"ch1 was flat before the rearm, and its scores are {checks}"
    finally:
        session.stop()


# --- S11 ------------------------------------------------------------------------------------------

def test_s11_the_level_line_names_the_full_at_of_the_trial_in_flight():
    from test_heg_trials import _push
    from test_heg_trials import _session as heg_session
    from test_pupil_view import _eye, _to_feedback
    from test_pupil_view import _session as pupil_session

    pupil = pupil_session("strong", trial_direction="increase", n_trials=3)
    try:
        _to_feedback(pupil, _eye(pupil))
        in_use = pupil.scale_in_use()[0]
        pupil.set_full_at(0.5, typed=True)
        assert f"±{pupil.scale_in_use()[0]:g} mm" in pupil.level_source[1], \
            (pupil.level_source[1], in_use)
    finally:
        pupil.stop()
    heg = heg_session(n_trials=2)
    try:
        t = 0.0
        while heg.trials.phase != ACTIVE:
            t = _push(heg, 1.0, t)
            assert t < 300.0, "no feedback window came"
        heg.set_full_at(5.0, typed=True)
        assert f"±{heg.scale_in_use()[0]:g} sd" in heg.level_source[1], heg.level_source[1]
    finally:
        heg.stop()


# --- S12 ------------------------------------------------------------------------------------------

def test_s12_after_a_seam_the_level_shows_nothing_until_a_sample_is_accepted():
    import dataclasses

    from nfb.dsp.scp import SlowShift

    spec = _limited_spec()
    metric = SlowShift(spec, output_hz=20.0)
    metric.feed(_block(spec, 0, np.full(BLOCK, -300.0)), accepted=True)
    assert math.isfinite(metric.level)
    seam = dataclasses.replace(_block(spec, BLOCK, np.full(BLOCK, -100.0)), seam=True)
    metric.feed(seam, accepted=False)
    assert math.isnan(metric.level), f"the level after the seam is {metric.level:.0f} µV, the old signal"


# --- S13 ------------------------------------------------------------------------------------------

def test_s13_a_trial_lost_to_a_channel_forced_off_does_not_ask_the_person_to_keep_still():
    session = _session()
    try:
        ui = _ring_view(session)
        session.toggle_pause()
        run = _run(session, n_trials=1, transfer=0.0, seed=26)
        x = wandering(1.5, 40.0, RATE, 93, 2.0)
        at = _feed(session, {0: x}, until=lambda _s: run.phase == ACTIVE)
        session.trust_channel(0, False)
        _feed(session, {0: x}, start=at, until=lambda _s: bool(run.results))
        trial = run.results[0]
        assert not trial.valid
        ui.draw_fixation("")
        assert "keep still" not in ui.verdict_detail.value, (trial.cause, ui.verdict_detail.value)
    finally:
        session.stop()


# --- S14 ------------------------------------------------------------------------------------------

def test_s14_the_no_signal_count_leaves_out_trials_cut_by_a_pause():
    session, control = _screen()
    try:
        session.set_extra(1, True)
        run = _run(session, n_trials=9, transfer=0.0, seed=27)
        n = int(300 * RATE)
        ch0, clean = wandering(1.5, 300.0, RATE, 94, 2.0), wandering(1.5, 300.0, RATE, 95, 2.0)
        cut = (1, 2, 3)
        at = 0
        while not run.done and at + BLOCK <= n:
            number = run.trial_number
            ch1 = clean if number in cut and len(run.results) < number else np.zeros(n)
            at = _feed(session, {0: ch0[:at + BLOCK], 1: ch1[:at + BLOCK]}, start=at)
            if number in cut and run.phase == ACTIVE and len(run.results) == number - 1:
                session.toggle_pause()
                at = _feed(session, {0: ch0[:at + int(1.5 * RATE)], 1: clean[:at + int(1.5 * RATE)]},
                           start=at)
                session.toggle_pause()
        assert run.done and [r.cause for r in run.results[:3]] == ["paused"] * 3
        text = _joined(session, control)
        said = re.search(r"ch1 had no signal in (\d+) of (\d+) trials", text)
        assert said, text
        windowed = sum(1 for r in run.results if r.cause != "paused")
        assert int(said.group(1)) <= windowed, f"{said.group(0)}; 3 of them were cut by a pause, with a " \
                                               f"clean signal on ch1"
    finally:
        session.stop()


# --- S15 ------------------------------------------------------------------------------------------

def test_s15_a_short_pause_in_the_effort_voids_the_trial_in_flight():
    session = _session()
    try:
        session.toggle_pause()
        run = _run(session, n_trials=1, transfer=0.0, seed=28)
        x = wandering(1.5, 40.0, RATE, 96, 2.0)
        at = _feed(session, {0: x}, until=lambda _s: run.phase == ACTIVE and run.phase_remaining() < 4.0)
        session.toggle_pause()
        at = _feed(session, {0: x[:at + int(0.8 * RATE)]}, start=at)
        session.toggle_pause()
        _feed(session, {0: x}, start=at, until=lambda _s: bool(run.results))
        trial = run.results[0]
        assert not trial.valid and trial.cause == "paused", trial
    finally:
        session.stop()


def test_s15_a_short_pause_voids_the_pupil_trial_in_flight_too():
    from test_pupil_view import _eye, _to_feedback
    from test_pupil_view import _session as pupil_session

    pupil = pupil_session("strong", trial_direction="increase", n_trials=2)
    try:
        samples = _eye(pupil)
        _to_feedback(pupil, samples)
        pupil.toggle_pause()
        for _ in range(30):                             # half a second of the eye while paused
            pupil.eye.push(next(samples)[1])
        pupil.toggle_pause()
        for i, sample in samples:
            pupil.eye.push(sample)
            if not i % 6:
                pupil.observe()
            if pupil.trials.results:
                break
        trial = pupil.trials.results[0]
        assert not trial.valid and trial.cause == "paused", trial
    finally:
        pupil.stop()


# --- S17 ------------------------------------------------------------------------------------------

def test_s17_the_extra_channel_is_checked_in_the_windows_of_the_training_trial():
    from nfb_console.training import true_metric_hz

    session, _control = _screen()
    try:
        session.set_extra(1, True)
        run = _run(session, n_trials=1, transfer=0.0, seed=29)
        x = {0: wandering(1.5, 60.0, RATE, 97, 2.0), 1: wandering(1.5, 60.0, RATE, 98, 2.0)}
        at = _feed(session, x, block=25, until=lambda _s: run.phase == BASELINE
                   and run.phase_remaining() < 0.3)
        at += int(0.6 * RATE)                           # a hole of 0.6 s across the feedback start
        _feed(session, x, start=at, block=25, until=lambda _s: bool(run.results))
        assert run.results[0].valid
        (_index, b0, a0, _a1), = session._windows
        step = 1.0 / true_metric_hz(RATE)
        assert abs(a0 - (b0 + run.baseline_s)) <= 1.5 * step, \
            f"the extra window starts {a0 - b0 - run.baseline_s:.2f} s after the feedback start"
    finally:
        session.stop()


# --- S16 ------------------------------------------------------------------------------------------

def test_s16_an_emptied_box_changes_neither_the_mode_nor_the_run():
    session = _session()
    try:
        card = scp_view.SCPCard(_page(), session, lambda: None)
        session.chance_spread = lambda: (1.2, 30.0)
        card.refresh()
        field = card._scale_boxes["success_at"].controls[0]
        assert field.value == "1.2", field.value
        run = session.trials
        field.value = ""
        field.on_blur(None)
        assert session.success_at_mode == "automatic", (session.success_at_mode, session.success_at)
        assert session.trials is run, "the run started again"
        assert field.value == "1.2", "the box shows its last value again"
        field.value = "abc"
        field.on_submit(None)
        assert (session.success_at_mode, field.value) == ("automatic", "1.2") and session.trials is run
    finally:
        session.stop()


def test_the_automatic_pill_acts_on_each_press():
    """A press on the lit pill fixes the value on screen; a second press goes back to automatic."""
    session = _session()
    try:
        card = scp_view.SCPCard(_page(), session, lambda: None)
        session.chance_spread = lambda: (1.23, 30.0)
        card.refresh()
        for name, spreads in (("success_at", 1.0), ("full_at", 2.0)):
            press = card._scale_boxes[f"{name}_pill"].on_click
            press(None)
            assert getattr(session, f"{name}_mode") == "fixed", name
            assert getattr(session, name) == pytest.approx(round(spreads * 1.23, 1)), "the value on screen"
            press(None)
            assert getattr(session, f"{name}_mode") == "automatic", name
    finally:
        session.stop()


# --- S20 ------------------------------------------------------------------------------------------

def test_s20_pupil_and_heg_help_explains_the_words_of_the_run_verdict():
    from nfb_console import heg_view, pupil_view

    for name, text in (("Pupil", pupil_view.MODE_HELP), ("HEG", heg_view.MODE_HELP)):
        assert "chance range" in text, f"the {name} help does not explain the chance range"


# --- S22 ------------------------------------------------------------------------------------------

def test_s22_a_blink_after_a_partly_railed_first_block_is_still_rejected():
    from nfb.dsp.artifact import ArtifactGate

    spec = _limited_spec()
    tt = np.arange(int(3 * RATE)) / RATE
    x = 8.0 * np.sin(2 * np.pi * 10.0 * tt) + wandering(1.5, 4.0, RATE, 99, 2.0)[:len(tt)]
    blink = slice(int(1.0 * RATE), int(1.0 * RATE) + 8)
    x[blink] += 120.0
    x[:4] = -800.0                                      # the first block is half on the input limit
    gate = ArtifactGate(spec)
    judged = [gate.feed(_block(spec, i, x[i:i + 8])) for i in range(0, len(x) - 7, 8)]
    assert not judged[0].accepted
    assert not judged[blink.start // 8].accepted, \
        f"a 120 µV blink is accepted at a threshold of {judged[blink.start // 8].threshold:.0f} µV"


# --- S18 and S19: the tests that were missing --------------------------------------------------------

def test_s18_the_output_level_is_above_the_middle_toward_the_cue_and_below_it_away():
    """A shift toward the cue drives every output above the middle, and a shift away drives it below."""
    for toward in (True, False):
        session = _session()
        try:
            session.toggle_pause()
            run = _run(session, n_trials=2, transfer=0.0, seed=30)
            seen: list[float] = []

            def watch():
                if run.phase == ACTIVE and run.active_s - run.phase_remaining() > 2.0 \
                        and session.signal is not None:
                    seen.append(session.signal)

            _feed(session, {0: wandering(1.5, 40.0, RATE, 100, 1.0)}, block=25, watch=watch,
                  until=_until_done(session), add=lambda s, _ch: (1 if toward else -1) * _held(s.trials, 12.0))
            assert seen
            assert all((level > 0.5) == toward for level in seen), (toward, min(seen), max(seen))
        finally:
            session.stop()


def test_s19_with_rejection_off_a_sample_at_the_input_limit_still_trains_nothing():
    from nfb.dsp.artifact import ArtifactGate, ArtifactPolicy

    spec = _limited_spec()
    gate = ArtifactGate(spec, ArtifactPolicy(enabled=False))
    uv = np.full(8, 5.0)
    assert gate.feed(_block(spec, 0, uv)).accepted
    uv[5] = 800.0
    judged = gate.feed(_block(spec, 8, uv))
    assert not judged.accepted
    assert max(abs(v) for v in judged.fed.values) < 0.98 * spec.input_range


def test_s19_in_clamp_mode_a_sample_at_the_input_limit_holds_the_block():
    from nfb.dsp.artifact import ArtifactGate, ArtifactPolicy

    spec = _limited_spec()
    gate = ArtifactGate(spec, ArtifactPolicy(mode="clamp"))
    tt = np.arange(int(2 * RATE)) / RATE
    x = 8.0 * np.sin(2 * np.pi * 10.0 * tt)
    for i in range(0, len(x) - 7, 8):
        gate.feed(_block(spec, i, x[i:i + 8]))
    uv = np.full(8, 5.0)
    uv[5] = -800.0
    judged = gate.feed(_block(spec, len(x), uv))
    assert not judged.accepted, "a zero in place of the input limit is no signal either"


def test_s19_an_extra_channel_is_not_checked_for_5_s_after_it_touches_the_input_limit():
    session, _control = _screen()
    try:
        session.set_extra(1, True)
        run = _run(session, n_trials=1, transfer=0.0, seed=31)
        x = {0: wandering(1.5, 40.0, RATE, 101, 2.0), 1: wandering(1.5, 40.0, RATE, 102, 2.0)}
        at = _feed(session, x, until=lambda _s: run.phase == BASELINE)
        x[1][at + 10] = 800.0                           # one sample at the limit as the baseline starts
        _feed(session, x, start=at, until=lambda _s: bool(run.results))
        assert run.results[0].valid
        assert session.extra_checks()[1] == [None], "5 s of the window came after a touch of the limit"
    finally:
        session.stop()


# --- S21 ------------------------------------------------------------------------------------------

def test_s21_a_done_run_is_judged_once_and_again_only_when_its_checks_change(monkeypatch):
    from nfb_console import trial_training

    from test_trial_engine import _run_of, _summary

    run, checks = _run_of(2, skill=2.0, extra=(2.0, 0.0))
    calls: list[int] = []
    original = trial_training._run_verdict
    monkeypatch.setattr(trial_training, "_run_verdict", lambda *a, **k: calls.append(1) or original(*a, **k))
    first = _summary(run, checks)
    assert _summary(run, checks) == first and len(calls) == 1, "drawn every frame, judged once"
    assert _summary(run, {}) != first and len(calls) == 2, "a chip pressed after the end is judged again"


# --- S24, and the surviving mutant V6 -------------------------------------------------------------

def test_v6_an_extra_with_a_thin_baseline_vouches_for_nothing():
    """The extra channel is flat for the first 1.2 s of the baseline: 15% of the trial's readings, under the
    20% rule, but less than half of its baseline. By the trial engine's own baseline rule it checks nothing."""
    session, _control = _screen()
    try:
        session.set_extra(1, True)
        run = _run(session, n_trials=1, transfer=0.0, seed=33)
        x = {0: wandering(1.5, 40.0, RATE, 103, 2.0), 1: wandering(1.5, 40.0, RATE, 104, 2.0)}
        at = _feed(session, x, until=lambda _s: run.phase == BASELINE)
        x[1][at:at + int(1.2 * RATE)] = 5.0                 # flat: no signal there
        _feed(session, x, start=at, until=lambda _s: bool(run.results))
        assert run.results[0].valid
        assert session.extra_checks()[1] == [None], "a thin baseline on the extra vouched for the trial"
    finally:
        session.stop()


# --- The surviving mutants of the mutation run ------------------------------------------------------

def _scripted_run(groups: dict[bool, tuple[list[float], list[float]]]) -> TrialRun:
    """A done run with these scores: per transfer flag, the up scores and the down scores."""
    from nfb_console.trials import NEGATIVE, Trial

    trials = [(POSITIVE, transfer, s) for transfer, (up, down) in groups.items() for s in up]
    trials += [(NEGATIVE, transfer, s) for transfer, (up, down) in groups.items() for s in down]
    run = TrialRun(n_trials=len(trials))
    run.results.extend(Trial(index=i + 1, required=d, transfer=tr, score_uv=s, valid=True, success=False)
                       for i, (d, tr, s) in enumerate(trials))
    return run


def test_u7_the_chance_level_is_split_over_the_groups_judged():
    """A feedback group whose difference passes the 5% range and not the 2.5% one, beside a group without
    feedback: with two groups, each is judged at 2.5%, so the run does not show control."""
    from nfb_console.trial_training import trial_summary

    spread = [-1.0, 0.0, 1.0, -0.5, 0.5]
    run = _scripted_run({False: ([1.05 + e for e in spread], spread),
                         True: ([0.1, -0.1, 0.2], [0.0, 0.1, -0.2])})
    assert run.done
    text = trial_summary(run, {POSITIVE: "up", "negative": "down"}, "µV")
    assert "inside the chance range" in text and "past the chance range" not in text, text


def test_u10_keep_this_strategy_is_said_once_when_an_extra_channel_vouches():
    from test_trial_engine import _run_of, _summary

    said = []
    for seed in range(10):
        run, checks = _run_of(seed, skill=2.0, extra=(2.0, 0.0))
        text = _summary(run, checks)
        if "did not follow the cues" in text and "past the chance range" in text:
            said.append(text.count("keep this strategy"))
    assert said and all(n == 1 for n in said), said


def test_u6_the_verdict_line_of_a_down_trial_compares_the_shift_toward_the_cue():
    from nfb_console.trial_training import AUTOMATIC, TrialScale, verdict_line
    from nfb_console.trials import NEGATIVE, Trial

    session = _session()
    try:
        scale = TrialScale(20.0, 5.0, AUTOMATIC, AUTOMATIC, 5.0, 30.0)
        down = Trial(index=1, required=NEGATIVE, transfer=False, score_uv=-8.0, valid=True, success=True,
                     scale=scale)
        assert verdict_line(down, session)[1].startswith("more than the 5.0 µV of succeeds at"), \
            verdict_line(down, session)
        short = Trial(index=2, required=NEGATIVE, transfer=False, score_uv=8.0, valid=True, success=False,
                      scale=scale)
        assert verdict_line(short, session)[1].startswith("less than"), verdict_line(short, session)
    finally:
        session.stop()


def test_u4_at_a_verdict_every_reader_still_takes_the_trials_record():
    session = _session()
    try:
        session.toggle_pause()
        session.set_full_at(20.0, typed=True)
        session.set_success_at(4.0, typed=True)
        run = _run(session, n_trials=2, transfer=0.0, seed=34)
        _feed(session, {0: wandering(1.5, 40.0, RATE, 105, 2.0)}, until=lambda _s: run.phase == RESULT)
        session.set_full_at(50.0, typed=True)
        session.success_at_setting = session.success_at_setting._replace(value=9.0)
        assert session.scale_in_use() == (20.0, 4.0), session.scale_in_use()
    finally:
        session.stop()


def test_u9_a_fixed_box_keeps_the_value_typed_for_the_next_trial():
    """Inside a trial the readers take the record, but a box on *fixed* shows what was typed, which acts
    from the next trial: the frame does not write the record over it."""
    session = _session()
    try:
        card = scp_view.SCPCard(_page(), session, lambda: None)
        session.toggle_pause()
        session.set_full_at(20.0, typed=True)
        card.rebuild()
        run = _run(session, n_trials=2, transfer=0.0, seed=35)
        _feed(session, {0: wandering(1.5, 40.0, RATE, 106, 2.0)}, until=lambda _s: run.phase == ACTIVE)
        field = card._scale_boxes["full_at"].controls[0]
        field.value = "30"
        field.on_submit(None)
        card.refresh()
        shown = card._scale_boxes["full_at"].controls[0].value
        assert session.scale_in_use()[0] == 20.0 and shown == "30.0", shown
    finally:
        session.stop()


def test_v9_at_a_verdict_the_spread_is_demeaned_without_the_trial_it_judges():
    from nfb_console.trials import Trial

    session = _session()
    try:
        run = TrialRun(n_trials=5)
        run.results.extend(Trial(index=i + 1, required=POSITIVE, transfer=False, score_uv=s, valid=True,
                                 success=False) for i, s in enumerate((2.0, 4.0, 30.0)))
        assert session._own_shift(run, (RESULT, POSITIVE, 3)) == pytest.approx(3.0)
        assert session._own_shift(run, (ACTIVE, POSITIVE, 3)) == pytest.approx(12.0)
    finally:
        session.stop()


def test_t1_the_input_limit_clears_five_seconds_of_samples_after_the_last_one_there():
    session = _session()
    try:
        spec = session.specs[0]
        uv = np.full(8, 5.0)
        uv[0] = 800.0                                   # the last sample at the limit opens the block
        session.feed_block(_block(spec, 0, uv))
        left = round(session.LIMIT_CLEAR_SECONDS * spec.rate_hz) - 7
        session.feed_block(_block(spec, 8, np.full(left - 1, 5.0)))
        assert session.at_input_limit(0)
        session.feed_block(_block(spec, 8 + left - 1, np.full(1, 5.0)))
        assert not session.at_input_limit(0), "5 s of samples after the last one at the limit"
    finally:
        session.stop()


def test_w2_a_box_with_an_edit_in_progress_keeps_the_edit():
    from nfb_console.widgets import num_box

    box = num_box(1.0, lambda _v: None, step=0.1)
    field, show = box.controls[0], box.data
    show(2.0)
    assert field.value == "2.0", "a value that moves on its own reaches an untouched box"
    field.value = "7"
    show(3.0)
    assert field.value == "7", "the next frame does not write over a value being typed"


# --- The replay: an extra channel at the input limit for most of a run ------------------------------

def test_an_extra_that_spent_the_run_at_the_input_limit_says_why_it_checked_nothing():
    session, control = _screen()
    try:
        session.set_extra(1, True)
        run = _run(session, n_trials=3, transfer=0.0, seed=36)
        x = {0: wandering(1.5, 80.0, RATE, 107, 2.0), 1: np.full(int(80 * RATE), -800.0)}
        at = _feed(session, x, until=lambda _s: run.done)
        x[1][at:] = wandering(1.5, 80.0, RATE, 108, 2.0)[at:]
        _feed(session, x, start=at, until=lambda s: s.stream_seconds * RATE > at + 6 * RATE)
        assert not session.at_input_limit(1), "the channel has left the input limit"
        line, problem = session.extra_status(1)
        assert line == "ch1 · checked in 0 of 3 · 3 at the input limit" and problem, (line, problem)
        assert line in _joined(session, control)
    finally:
        session.stop()


# --- S25 ------------------------------------------------------------------------------------------

def test_s25_each_run_is_marked_and_a_void_trial_mark_has_no_score():
    session = _session()
    try:
        session.toggle_pause()
        run = _run(session, n_trials=1, transfer=0.0, seed=37)
        x = wandering(1.5, 40.0, RATE, 109, 2.0)
        at = _feed(session, {0: x}, until=lambda _s: run.phase == ACTIVE)
        session.trust_channel(0, False)
        _feed(session, {0: x}, start=at, until=lambda _s: bool(run.results))
        session.restart_trials()
        notes = [note for _at, note in session.marks()]
        verdict = next(n for n in notes if " · verdict · " in n)
        assert "no result · no signal" in verdict and "µV" not in verdict, verdict
        assert notes[-1].startswith("run starts · "), notes
    finally:
        session.stop()
