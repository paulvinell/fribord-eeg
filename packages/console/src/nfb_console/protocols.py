"""The protocol catalogue.

`available` reflects whether the signal a protocol needs can actually be computed today, not whether
the protocol is finished. Greying out what does not work yet is more useful than a launcher full of
buttons that fail after you click them — and the HEG card follows this rule.

**But a greyed card is a promise, and a card that only says "no screen for it yet" is a note excusing
an absence.** Coherence sat here greyed for that reason — `nfb.dsp.coherence` exists and nothing
declares `HANDLES = ("COH",)` — and the honest launcher is the one that lists what can be run. It comes
back when it has a screen.

SCP — slow cortical potentials — was the one entry here that needed something no protocol had, and it is
the protocol that built it. The acquisition side was always there: the device's high-pass reaches 0.01 Hz
(`Settings.lower_limit_hz`), which is what a sub-0.1 Hz shift needs and what the ultra-low-frequency work
uses. What was missing was above it — **trial structure**. SCP training is a sequence of trials with a
baseline and a required direction per trial, scored per trial, where every other protocol here is one
undifferentiated run against a threshold. That was the session-structure gap; it
is closed by `nfb_console.trials`, a state machine owning no signal and no screen, and this entry is live.

SSVEP was in this catalogue and was **removed rather than left greyed out**, which is a judgement worth
recording: it is a stimulus-driven BCI paradigm rather than a training one. The occipital response to a
flicker is largely automatic, the therapeutic literature on training it is thin to nonexistent, and the
flicker itself brings a photosensitive-seizure consideration nothing else here has. A greyed card promises
the thing will exist one day; this one should not, so the card is gone.

**Waveform is not SCP, and the difference is exactly that**: it trains the same slow signal continuously,
against a calibrated range rather than trial by trial, so it needs nothing that does not exist. Both entries
stay, because a continuous slow-wave protocol and a trialled one are different trainings and not two names
for one.

**A band arrangement is not a protocol, and this catalogue is the line.** Ros et al.'s alpha-down —
8–12 Hz trained downward at Pz — is the amplitude screen with three different starting bands and one
target's role changed: same estimators, same cards, same everything. It briefly had a card here, and by
that logic every arrangement somebody uses twice gets one and the launcher becomes a menu of favourites.
What it did need was for the arrangement to be *expressible*, which is `targets.ROLES`; it is then two
clicks on a card.

**Two cards name a coupling and a third measures one, so the names have to do the work the codes cannot.**
`CFC` is a vendor's method: two *between-channel* narrowband synchrony rewards at two chosen frequencies,
which computes no phase-amplitude coupling anywhere. `PAC` is phase-amplitude coupling in the sense the
literature means it — one rhythm's phase deciding when another is loud. Called "Coupling" for a day, `PAC`
sat two cards from "Cross-frequency coupling" with no way to tell which did what, and a reader guessing
from the names would have guessed wrong. Both are named for their measure now, and the blurbs separate
them: two sites in step, against fast bursts nested in a slow rhythm.

**A blurb is one line that does not wrap**, so it has a width budget — see
`splash.protocol_card`, which keeps every text slot a fixed height on purpose. Four of these were over it
and clipped mid-word on screen; `tests/test_screens` now measures them.

HEG and HRV are here as protocols, not only as tabs on the device screen, because from the operator's
side they are things you sit down to *do* — the fact that they need a different amplifier is a detail of
how, not of what. Both were unavailable for the same reason and both now have their source: they speak
BLE rather than the Optima's serial protocol, so each needed one of its own, and neither deals in a
sample stream — HRV works on beat-to-beat intervals and HEG on 30 Hz optical frames.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Protocol:
    code: str
    name: str
    blurb: str
    available: bool = True
    note: str = ""          # shown when unavailable: why, not just that


CATALOGUE = (
    Protocol(
        "AMP", "Amplitude",
        "Train band amplitude",
    ),
    Protocol(
        "SCP", "Slow cortical potentials",
        "Slow shifts, trial by trial",
    ),
    Protocol(
        "A-T", "Alpha-Theta",
        "Alpha against theta",
    ),
    Protocol(
        "CFC", "Cross-frequency coupling",
        "Two sites in step, two frequencies",
    ),
    Protocol(
        "PAC", "Phase-amplitude coupling",
        "Fast bursts nested in a slow rhythm",
    ),
    Protocol(
        "ILF", "ILF Synchrony",
        "Two sites in phase, from 0.05 Hz",
    ),
    Protocol(
        "WAVE", "Waveform",
        "The slow wave itself, from 0.01 Hz",
    ),
    Protocol(
        "SLEEP", "Sleep",
        "Record the night, alarm on failure",
    ),
    Protocol(
        "HEG", "HEG",
        "Prefrontal blood flow",
    ),
    Protocol(
        "HRV", "HRV",
        "One spectral peak, at 0.1 Hz",
    ),
    Protocol(
        "PUPIL", "Pupil",
        "Pupil size, larger or smaller",
    ),
)

BY_CODE = {p.code: p for p in CATALOGUE}
