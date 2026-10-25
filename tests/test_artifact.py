"""Artifact rejection, against signals where the right answer is known by construction.

The point of every test here is that rejection must be *selective*. A gate that rejects nothing is the
state we were already in; a gate that rejects a third of the data looks like a working session with a
quiet subject, which is worse, because nothing on screen contradicts it.
"""

from __future__ import annotations

import numpy as np

from conftest import counts

from nfb.dsp import ArtifactGate, ArtifactGates, ArtifactPolicy
from nfb.dsp.artifact import JUDGE_FLOOR_HZ
from nfb.stream import Block, ChannelSpec

RATE = 250.0
SPEC = ChannelSpec(index=0, rate_hz=RATE, unit="uV", scale=0.001)


def blocks(x, spec=SPEC, size=8, state=1):
    """Chop µV into Blocks. Scale is 1 nV per count, as the file sources use, so µV survive
    round-tripping through the integral sample contract."""
    ints = counts(x, spec.scale)
    return [Block(spec.index, i, tuple(ints[i:i + size].tolist()), spec, state=state)
            for i in range(0, len(x) - size + 1, size)]


def sine(freq, seconds, amp, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    return amp * np.sin(2 * np.pi * freq * t)


def run(gate, x):
    return [gate.feed(b) for b in blocks(x)]


def test_a_clean_signal_is_not_rejected():
    # 40 µV peak alpha is a large but entirely ordinary EEG amplitude
    judged = run(ArtifactGate(SPEC), sine(10.0, 20.0, 40.0))
    assert all(j.accepted for j in judged)
    assert sum(j.rejected for j in judged) == 0


def test_a_blink_sized_excursion_is_rejected_and_its_neighbours_are_not():
    x = sine(10.0, 20.0, 30.0)
    at = int(15.0 * RATE)
    x[at:at + 25] += 400.0                       # a blink: ~100 ms, an order of magnitude too big
    gate = ArtifactGate(SPEC)
    judged = run(gate, x)
    rejected = sum(j.rejected for j in judged)
    assert 10 <= rejected <= 60, rejected        # the excursion, not the session
    held = [i for i, j in enumerate(judged) if not j.accepted]
    assert held, "the blink should have held at least one block out"
    # Every held block is at the excursion, not scattered through the clean signal. The window has to allow
    # the judged band's own settling time on either side, because band-limiting spreads a step in time and
    # rejecting that tail is right rather than sloppy. The **lower** edge is what sizes it: a single pole at
    # `JUDGE_FLOOR_HZ` decays with a time constant of 1/(2π·0.5) = 318 ms, so a 400 µV step leaves something
    # above threshold for most of that. Derived rather than a magic number, since the two move together.
    tail = int(RATE / (2 * np.pi * JUDGE_FLOOR_HZ))
    for i in held:
        assert at - 16 <= i * 8 <= at + 25 + tail


def test_disabled_rejects_nothing():
    policy = ArtifactPolicy(enabled=False)
    x = sine(10.0, 10.0, 30.0)
    x[500:520] += 900.0
    judged = run(ArtifactGate(SPEC, policy), x)
    assert all(j.accepted and j.clean for j in judged)


def test_the_floor_protects_a_quiet_channel():
    """3 µV of noise with a 12 µV blip is not an artifact — it is a small signal.

    Without a floor the adaptive threshold tracks down to a few µV and starts rejecting the signal
    itself, which is the failure the µV clamps on thresholds exist to prevent as well.
    """
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1.0, int(20 * RATE))
    x[2000:2010] += 12.0
    judged = run(ArtifactGate(SPEC, ArtifactPolicy(floor_uv=30.0)), x)
    assert sum(j.rejected for j in judged) == 0


def test_the_ceiling_rejects_an_enormous_excursion_however_loud_the_channel_has_been():
    """What the ceiling is for: an in-band excursion past any real signal, whatever the running level says.

    It used to be written against a **constant** ±800 µV rail, and that stopped being the ceiling's business
    when the judged band gained a lower edge — a high-pass of a constant is zero, so a DC rail is now
    correctly no excursion at all. Which is the right division and not a hole: a rail is over-range, the
    device reports it in the state word, and `pass_through` keeps it out of the gate entirely so the traces
    do not stop. `test_a_channel_the_device_already_calls_unusable_is_not_gated` is that half.
    """
    x = sine(3.0, 5.0, 800.0)
    judged = run(ArtifactGate(SPEC, ArtifactPolicy(ceiling_uv=250.0)), x)
    assert judged[-1].threshold == 250.0, "pinned at the ceiling, not chasing the excursion"
    held = [j for j in judged if not j.accepted]
    assert len(held) > len(judged) * 0.5, "most of it is over the line"


def test_a_direct_current_offset_is_not_an_excursion():
    """The reason the judged band needs a lower edge, as a test rather than as an argument.

    At the device's 0.01 Hz corner an electrode's standing offset reaches **hundreds of µV** — measured 305
    on a real capture, with 7-8 µV of actual EEG on top. Judged on a band with no lower edge that offset is
    an excursion on every sample and the gate holds 90% of the session; judged on 0.5-50 Hz it is not there
    at all, and the EEG riding on it is what sets the threshold.
    """
    x = 305.0 + sine(10.0, 60.0, 8.0)
    gate = ArtifactGate(SPEC)
    judged = run(gate, x)
    # The offset arrives as a *step* here, and the high-pass starts from rest, so the first couple of time
    # constants are genuinely an excursion as far as the gate can tell. That settling is the whole of the
    # 0.3% seen on the real capture, so it is worth naming rather than tuning away: two time constants of a
    # single pole at `JUDGE_FLOOR_HZ`, and nothing after it.
    settling = int(2 * RATE / (2 * np.pi * JUDGE_FLOOR_HZ) / 8) + 1
    assert sum(j.rejected for j in judged[settling:]) == 0, "the offset is not an artifact; the EEG is signal"
    assert sum(j.rejected for j in judged) < 0.01 * len(judged) * 8, "and the settling is a rounding error"
    # The level is *primed* on the first block, so that step sets it to ~305 and it decays from there with
    # `tau_seconds`. That leaves the threshold pinned at its ceiling for the first half-minute or so — less
    # sensitive at the start of a session, never more, so it fails in the safe direction and is not worth
    # machinery to avoid. What matters is where it settles: an 8 µV sine has a mean absolute of 8·2/π = 5.1.
    assert gate.level < 10.0, "the running level settles on the EEG's, not the offset's"


def test_the_threshold_does_not_chase_a_sustained_artifact():
    """The whole point of adapting from accepted samples only.

    A long excursion — a settling electrode, someone scratching — would otherwise raise the threshold
    until it is inside it, and rejection switches itself off exactly when it is needed.
    """
    x = np.concatenate([sine(10.0, 10.0, 30.0), sine(3.0, 10.0, 400.0)])
    gate = ArtifactGate(SPEC)
    judged = run(gate, x)
    before = judged[len(judged) // 3].threshold
    after = judged[-1].threshold
    assert after < before * 1.5, (before, after)
    # and it is still rejecting at the end of the artifact, not tolerating it
    assert not judged[-1].accepted


def test_clamp_keeps_the_block_but_zeroes_the_sample():
    x = sine(10.0, 10.0, 30.0)
    x[1000:1004] += 500.0
    gate = ArtifactGate(SPEC, ArtifactPolicy(mode="clamp"))
    judged = run(gate, x)
    dirty = [j for j in judged if not j.clean]
    assert dirty, "the excursion should have been detected"
    for j in dirty:
        assert j.accepted, "clamp carries on rather than holding"
        assert len(j.fed.samples) == 8
        assert 0 in j.fed.samples
        assert 0 not in j.block.samples, "the trace still sees the excursion, not the rewrite"


def test_rate_and_percent_are_reported_separately():
    """The live rate answers "is it rejecting now"; the session percent cannot."""
    x = sine(10.0, 60.0, 30.0)
    x[500:800] += 600.0                     # 1.2 s of mess early on, then nothing
    gate = ArtifactGate(SPEC)
    run(gate, x)
    assert 0.0 < gate.percent < 0.1
    assert gate.rate < 0.01, "long over by the end of the run"


def test_there_is_no_threshold_log_because_a_replay_reproduces_it():
    """A log of the live threshold would only be needed with two implementations of the gate. We have one
    implementation, so replaying re-derives the same
    thresholds and the same rejections — a log would be a second copy of something reproducible, and the
    failure mode of those is that they disagree."""
    assert not hasattr(ArtifactGate(SPEC), "log")

    x = sine(10.0, 20.0, 30.0)
    x[2000:2100] += 400.0
    runs = []
    for _ in range(2):
        gate = ArtifactGate(SPEC)
        judged = run(gate, x)
        runs.append((gate.rejected, gate.seen, [round(j.threshold, 6) for j in judged]))
    assert runs[0] == runs[1]
    assert runs[0][0] > 0, "and it did actually reject something"


def test_gates_share_one_policy_across_channels():
    specs = {i: ChannelSpec(index=i, rate_hz=RATE, unit="uV", scale=0.001) for i in (0, 1)}
    gates = ArtifactGates()
    x = sine(10.0, 10.0, 30.0)
    dirty = x.copy()
    dirty[1000:1010] += 600.0
    for b in blocks(x, specs[0]):
        gates.feed(b)
    for b in blocks(dirty, specs[1]):
        gates.feed(b)
    assert gates.gates[0].percent == 0.0
    assert gates.gates[1].percent > 0.0
    assert gates.percent > 0.0
    assert "ch1" in gates.describe()
    gates.policy.enabled = False
    assert "not rejecting" in gates.describe()


def test_a_low_rate_channel_skips_the_band_limit_rather_than_failing():
    """62.5 sps is what the device forces on non-voltage functions; 0-50 Hz is already true there."""
    slow = ChannelSpec(index=3, rate_hz=62.5, unit="uV", scale=0.001)
    gate = ArtifactGate(slow)
    judged = [gate.feed(b) for b in blocks(sine(5.0, 20.0, 30.0, rate=62.5), slow)]
    assert all(j.accepted for j in judged)


def test_a_channel_the_device_already_calls_unusable_is_not_gated():
    """Measured, and it mattered: the no-electrode bench recording rails to ±800 µV, so every block
    failed the ceiling, the whole measurement was held, and the screen went dead — hiding exactly what an
    operator needs to see. The device flags gross excursions in hardware (state 2 and 3); the gate is for
    what happens inside range."""
    gates = ArtifactGates()
    railed = blocks(np.full(int(2 * RATE), 800.0), state=2)
    for b in railed:
        judged = gates.pass_through(b)
        assert judged.accepted and judged.clean
    # and it has not moved the running level, which is the whole point of adapting from accepted samples
    assert gates.for_channel(SPEC).level == 0.0
    assert gates.percent == 0.0


def test_a_held_block_keeps_the_stream_contiguous_for_the_measurement():
    """A rejection is not an acquisition gap, and for a long time the two were the same code path.

    Skipping a held block leaves a hole in the sample indices, every estimator reads that as a device gap
    and resets its filters, and each blink re-runs the whole settling time. Where that bites is the slow
    protocols — the waveform chain settles over minutes — so the comparison here is a 20 µV slow wave with
    a blink every 12 s, scored against the same metric fed the clean signal. `hold` now feeds the
    measurement a stand-in block at the last accepted value: same indices, nothing to train on, nothing to
    reset.
    """
    from nfb.dsp.wave import SignedWave

    seconds = 300
    tt = np.arange(int(seconds * RATE)) / RATE
    wave = 20.0 * np.sin(2 * np.pi * 0.2 * tt)
    x = wave.copy()
    for blink_at in np.arange(30.0, seconds - 1, 12.0):
        i = int(blink_at * RATE)
        x[i:i + 25] += 150.0

    reference = {}                               # the metric on the clean signal, no gate
    clean = SignedWave(SPEC, 0.2)
    for b in blocks(wave):
        for r in clean.feed(b):
            reference[r.t] = r.value

    def rms_error(readings):
        errs = [abs(v - reference[rt]) for rt, v in readings if rt > 100.0 and rt in reference]
        return float(np.sqrt(np.mean(np.square(errs))))

    gate = ArtifactGate(SPEC)
    held_metric = SignedWave(SPEC, 0.2)
    skip_metric = SignedWave(SPEC, 0.2)
    held_readings, skip_readings = [], []
    held = 0
    last_end = None
    for b in blocks(x):
        j = gate.feed(b)
        if not j.accepted:
            held += 1
            assert j.fed.start_index == j.block.start_index
            assert len(set(j.fed.samples)) == 1, "filled with the one held value"
        if last_end is not None:
            assert j.fed.start_index == last_end, "no hole in the fed stream, ever"
        last_end = j.fed.end_index
        held_readings += [(r.t, r.value) for r in held_metric.feed(j.fed)]
        if j.accepted:                           # the old behaviour, kept as the comparison
            skip_readings += [(r.t, r.value) for r in skip_metric.feed(j.block)]
    assert held > 10, "the blinks should have held blocks out"

    # Measured when this was fixed: 0.15 µV against 3.1 — a twentyfold difference on a 20 µV wave,
    # and the skip path also emits a fraction of the readings because every reset re-anchors its grid.
    assert rms_error(held_readings) < 0.5, "holding rides the rejections out"
    assert rms_error(skip_readings) > 2.0, "skipping provably re-runs the settling per blink"


def test_a_railed_stretch_does_not_leave_its_step_in_a_slow_filter():
    """`pass_through` hands the metrics a stand-in too. The raw rail used to go through — readings were
    invalid and every screen ignored them, but the filters had still eaten an 800 µV step, and the
    waveform chain's drift follower (25 s per pole at 0.05 Hz) carried it long after the device called
    the channel good again, under a warmup that said settled (the indices never broke, so no reset
    fired). Measured here: ten seconds of lifted reference leaves 230 µV RMS of transient on a 20 µV
    wave in the first half-minute after recovery, and still 8 µV — 40% of the signal — two minutes on.
    The stand-in idles the filters at the last accepted value, so recovery costs only the signal's own
    step: 3.9 µV falling to 0.1."""
    import dataclasses

    from nfb.dsp.wave import SignedWave

    seconds, hz = 480, 0.05
    rail = int(round(800.0 / SPEC.scale))                     # what an open input reads, in counts
    tt = np.arange(int(seconds * RATE)) / RATE
    wave = 20.0 * np.sin(2 * np.pi * hz * tt)
    rail_from, rail_to = int(300 * RATE), int(310 * RATE)     # ten seconds of lifted reference

    reference = {}
    clean = SignedWave(SPEC, hz)
    for b in blocks(wave):
        for r in clean.feed(b):
            reference[r.t] = r.value

    def rms_after_recovery(stand_in: bool) -> float:
        gates = ArtifactGates()
        metric = SignedWave(SPEC, hz)
        out = []
        for b in blocks(wave):
            railed = rail_from <= b.start_index < rail_to
            judged = (gates.pass_through(dataclasses.replace(b, samples=(rail,) * 8, state=2))
                      if railed else gates.feed(b))
            fed = judged.fed if stand_in else judged.block
            out += [(r.t, r.value) for r in metric.feed(fed)]
        lo = rail_to / RATE + 30                              # after the device is long back
        errs = [abs(v - reference[rt]) for rt, v in out if lo < rt < lo + 60 and rt in reference]
        return float(np.sqrt(np.mean(np.square(errs))))

    assert rms_after_recovery(stand_in=False) > 10.0, \
        "the raw rail provably poisons the filters long past recovery"
    assert rms_after_recovery(stand_in=True) < 1.0, "the stand-in idles them through the outage"
