"""The Feedback tab, shared by every protocol screen.

Nothing on it depends on what is being trained. It takes a `FeedbackSettings` and a list of targets, and
those are the same objects whichever protocol produced them — which is why this was the largest single
block to lift out of `amplitude_view.py` and the one that would have been most damaging to copy.

Four things, in the order the reader needs them:

1. **OUTPUT** — what the person in the chair is perceiving, right now. Leads with the *outputs*
   ("volume 93 · 85% bright"), not the 0…1 signal that produced them: that number means "how far between
   each mechanism's floor and its ceiling", which nobody experiences, and leading with it made the panel
   unreadable. When no mechanism is on it says **"nothing is being fed back"** rather than showing a
   healthy-looking percentage — a signal with no output is the one state that looks fine and is useless.
2. **ARTIFACTS** — the gate, before anything downstream. It is here rather than on a device screen
   because it belongs to the *signal*: it is step 0 of turning microvolts into a number, and it changes
   what every target measures. And it is here rather than nowhere because a gate that silently ate 40% of
   the data would look exactly like a working session with a quiet subject.
3. **SIGNAL** — the steps of one calculation, numbered: band averaging, `scale from`, `level from`,
   smoothing. They were previously two unrelated-looking menus, which is what made them unintelligible.
   It ends by **naming the two channels the chain forks into** — the level and the condition — with each
   one's live value and who is following it. Without that the third row read as though it governed every
   output, and it governs the level alone; the click and the alert ask the targets directly.
4. **One pane per mechanism**, nested inline, because each has its own range, units and meaning, grouped
   by which channel it reads. Above them, the arrangement presets — the three named models, which
   are settings and not a fourth kind of thing.

A control lives in exactly **one** place in the Flet tree, so `strip()` is a factory: anything appearing
on two tabs is built twice and registers both instances for the draw loop. The failure when it is not is
silent — one copy renders, the other never updates.
"""

from __future__ import annotations

import flet as ft

from nfb.dsp import REJECT_MODES
from nfb.dsp import artifact as dsp_artifact

from . import chrome, cues, music, quality_pane, shadow, sysvolume
from . import theme as t
from .feedback import (CONDITION, CONDITIONS, CONDITION_DETAIL, HOLD_FRACTION,
                       LEVEL, STATE,
                       SENSITIVITY_RANGE,
                       COMBINE_DETAIL, COMBINE_HELP, COMBINE_SHARES, COMBINES,
                       FEEDBACK_MODELS,
                       FEEDBACK_MODELS_HELP, MUSIC_SOURCES, NORMALISATIONS, NORMALISATION_HELP,
                       SHADOW_MECHANISMS, STOPS_WHEN, STOPS_WHEN_DETAIL, SUPPORTED_SHADOW)
from .widgets import help_button, labelled, num_box, panel, pill, range_row, reads_chip, \
    test_button

SIGNAL_BAR_W = 300

OUTPUT_HELP = (
    "This shows what you hear and see right now: the loudness, the brightness, and whether a click "
    "sounds.\n\n"
    "Two values drive the outputs. **The level** is one number from 0 to 1, a blend of every reward and "
    "inhibit on every channel. Sound and shadow follow the level, mapped between each output's own floor "
    "and ceiling. **The condition** is a plain yes or no: every target on its wanted side. The click, the "
    "alert and the music's stop follow the condition instead. A click can be silent while the level is "
    "high, or sound while the level drops. They answer different questions.\n\n"
    "The level reads 0.5 when every target sits exactly on its threshold. What counts as above that "
    "depends on the rule set on the SIGNAL panel below, named in the caption. The percentage shows how "
    "far the level sits between each output's own ends. This is why it appears as a bar and a caption, "
    "instead of the headline.\n\n"
    "Three headlines are not a level at all: waiting for data, no channel delivering signal, and nothing "
    "being fed back. The last one can look like a working session. A signal with no output turned on "
    "gives no feedback."
)

CLICK_HELP = (
    "A click sounds at a fixed interval while every target is met. It is silent otherwise. The click "
    "carries one piece of information: earned, or not earned.\n\n"
    "The interval sets how often you hear it. It does not speed up or slow down with the signal, so a "
    "click always means the same thing: the last interval was earned.\n\n"
    "The click checks the targets directly, instead of reading the combined level. The level can sit at "
    "60% with one target failing, which fits a continuous display but not an event that means earned. "
    "The **volume** row sets how loud the click is. It is separate from the music volume."
)

TONE_HELP = (
    "A tone sounds at a fixed interval while the condition holds. It is silent otherwise. The tone "
    "carries one piece of information: earned, or not earned.\n\n"
    "**Its pitch says which reward earned it**: low for a slow band, high for a fast one. This matters "
    "when the rewards are alternatives instead of conditions you meet together. The pitch is the only "
    "way to tell them apart with your eyes closed.\n\n"
    "This mechanism plays a tone instead of a click because a short click cannot carry a pitch. The "
    "alert sound says where trouble is. This tone says which reward is earning.\n\n"
    "**On a tempo, not on transitions.** It sounds the instant the condition is met, then no faster than "
    "the interval. The interval sets how often you hear it.\n\n"
    "The volume is for calibration, not information. Balance it against the music once, then let Sound "
    "move the master volume over both. Press ▶ beside a reward to hear its tone."
)

# **The OUTPUT readout is height-reserved, and that is not cosmetic.** Its two lines are rewritten every
# frame from live values, so as soon as one of them grew long enough to wrap, the panel grew a row — and
# every pane below it on the Feedback tab jumped by that much, then jumped back on the next frame. A readout
# that changes the layout of the screen it is on is unusable however correct its numbers are; this is the
# same rule as the chronospectrogram not changing shape while it fills.
#
# Reserving means the box is sized for the worst case and the text is capped at that many lines, so the
# content can vary freely and the geometry cannot. 24 px and 11 px type at ~1.3 line spacing.
HEADLINE_H = 2 * 31
NOTE_H = 2 * 15



def _flicker_left(tau_seconds: float, hz: float = 3.0) -> float:
    """What fraction of a `hz` component survives a one-pole with this time constant.

    3 Hz because that is the measured flicker rate of a multi-target condition — see
    `feedback.HOLD_FRACTION` — so the number answers the question the operator actually has, which is not
    "how many seconds" but "will I still see it".
    """
    import math

    return 1.0 / math.sqrt(1.0 + (2.0 * math.pi * hz * max(0.0, tau_seconds)) ** 2)


def _holding(session) -> tuple[str, str] | None:
    """Why this session has no level right now, in its own words, if it knows.

    Only the single-instrument screens implement it — on an amplifier screen "no level" really is about
    channels, which the branch below names. Returns (headline, note) or None.
    """
    say = getattr(session, "level_note", None)
    return say() if say is not None else None


class SignalPane:
    """Build once, mutate on Flet's loop. `rebuild()` after a setting changes, `refresh()` per frame."""

    def __init__(self, page: ft.Page, session, inset: bool = True):
        """`inset` is False when this pane is not a tab of its own.

        It carries a tab's padding and a tab's scroll region because on the EEG screens it *is* one of the
        tabbed panes. The HRV screen puts it inside a tab underneath two rows of its own, where both of those
        land on top of the enclosing tab's: every panel here sat 28 px from the edge while the rows above sat
        at 14, and the mismatch is visible down the left-hand side. The scroll doubled too — a scrolling
        region inside a scrolling region, each with its own idea of where the bottom is.
        """
        self.inset = inset
        self.page = page
        self.session = session
        self.fb = session.feedback
        self._strips: list[dict] = []
        self._contribution_widgets: dict[int, dict] = {}

        self.contributions = ft.Row(spacing=16, wrap=True)
        self.artifact_rows = ft.Column(spacing=14)
        # Where the rejection actually landed, under the settings that decide it — the one place a
        # threshold can be judged rather than guessed at. Untitled: ARTIFACTS is already the heading.
        self.quality = quality_pane.QualityPane(
            page, session,
            lambda reserve=0: max(220.0, (getattr(page, "width", None) or 1020) - reserve - 210),
            titled=False)
        self.signal_rows = ft.Column(spacing=14)
        self.model_rows = ft.Column(spacing=10)
        self.sound_rows = ft.Column(spacing=14)
        self.shadow_rows = ft.Column(spacing=14)
        self.click_rows = ft.Column(spacing=14)
        self.alert_rows = ft.Column(spacing=14)
        self.music_rows = ft.Column(spacing=14)

        self.music_path = ft.TextField(
            value=self.fb.music.path, hint_text="an audio file, or a folder of them",
            height=34, text_size=t.BODY, color=t.TEXT, bgcolor=t.SURFACE,
            border_color=t.BORDER, expand=True, content_padding=ft.Padding(10, 4, 10, 4))
        self.music_path.on_submit = self._commit_music_path
        self.music_path.on_blur = self._commit_music_path
        self._picker = ft.FilePicker()
        page.services.append(self._picker)

        # Only what this protocol offers, and in the order below. The alert is the case that matters:
        # see `TrainingSession.offers` — it is an *event* mechanism, so whether it means anything depends
        # on how often the protocol's thresholds are crossed, which is not a property of the mechanism.
        panes: dict[str, ft.Control] = {
            "sound": ft.Container(content=panel("SOUND", self.sound_rows)),
            "shadow": ft.Container(content=panel("SHADOW", self.shadow_rows), visible=False),
            "click": ft.Container(content=panel(self.fb.click.label.upper(), self.click_rows,
                                               help=help_button(
                page, self.fb.click.label.capitalize(),
                TONE_HELP if self.fb.click.keyed else CLICK_HELP)), visible=False),
            "alert": ft.Container(content=panel("ALERT", self.alert_rows, help=help_button(
                page, "Alert",
                "A sound plays when a target crosses to the wrong side of its threshold. This is the "
                "mirror of the click, which sounds when every target is met.\n\n"
                "This is what makes a time-above-threshold budget meaningful: the percentage is the "
                "fraction of the session you expect to hear something. With several inhibits running, "
                "you need to know which one is over, not only that one is. The sound is keyed to the "
                "failing target's frequency: slow bands make a low sound, fast ones make a high sound. "
                "Only the worst offender sounds, so a target barely over its threshold cannot hide one "
                "far over its threshold. A sound plays instead of pausing the display, because a pause "
                "is jerky and makes the event hard to see.")),
                visible=False),
            "music": ft.Container(content=panel("MUSIC", self.music_rows, help=help_button(
                page, "Music",
                "This is audio the session owns. Each screen offers some of these sources. The **source** row "
                "shows which ones.\n\n"
                "noise — generated sound with a tilted spectrum, instead of a looped file, so you never "
                "hear a repeating loop. It does not carry the signal. Pink matches the tilt of a healthy "
                "EEG signal.\n\n"
                "crossfade — two tilted sounds, with the signal fading between them.\n\n"
                "tone — one tone. Its pitch tracks the signal between two frequencies.\n\n"
                "shepard — a pitch that rises with the signal and has no top.\n\n"
                "file — an audio file, or a folder of files played in order and repeated. It does not "
                "carry the signal.\n\n"
                "eeg — the band's own rhythm becomes the pitch of one voice, and its size becomes the "
                "loudness. A faster rhythm sounds higher. It carries the waveform instead of the level.\n\n"
                "In the eeg source, **pitch** sets the two frequencies the bottom and the top of the band sound at. The note "
                "under the row shows what that pair does to the EEG's own rhythm. If the pitch range "
                "spans the same number of octaves as the band, you hear an exact transposition: the EEG "
                "played faster, and nothing else. A wider pitch range stretches every interval by a "
                "stated multiple.\n\n"
                "Choose a narrow band around the rhythm you are training. The sound plays the band's "
                "dominant frequency, so a band holding two rhythms at once gives one voice moving "
                "between them, instead of two voices.\n\n"
                "In the eeg source, there is no loudness setting. The band is heard against its own recent range of quiet "
                "and loud, learned from the signal the same way a threshold is. This learned range is "
                "printed in microvolts in the state line below. If the sound stays flat, change the "
                "pitch row instead.\n\n"
                "The volume here is separate from the system volume. Set a backing track's volume once, "
                "and leave it. The Sound mechanism then moves the master volume over the top of it.")),
                visible=False),
        }
        self.offered = [k for k in panes if k in session.offers]
        self._panes = [panes[k] for k in self.offered]
        # Grouped by the channel each one reads: the level, then the condition, then the music, which
        # carries neither. A flat row of five names showed nothing of that, so "60 to 100" on the sound
        # pane and "every 2 s" on the click pane looked like two settings of one kind of thing. The
        # dividers follow the channels rather than fixed positions, since a protocol offering fewer
        # mechanisms would otherwise put a rule in the middle of a group.
        reads = {"sound": LEVEL, "shadow": LEVEL, "click": CONDITION, "alert": CONDITION}
        dividers = tuple(i for i, k in enumerate(self.offered[:-1])
                         if reads.get(k) != reads.get(self.offered[i + 1]))
        # A keyed reward plays a *tone*, so it is called one — see `Click.label`. The tab said Click while
        # the speaker played a tone, which is the kind of copy that makes a screen untrustworthy.
        names = {k: self.fb.mechanisms[k].label.capitalize() if k in self.fb.mechanisms
                 else k.capitalize() for k in self.offered}
        self.tabs = chrome.Tabs([names[k] for k in self.offered],
                                self._select, inline=True, dividers=dividers)

        self.control = ft.Container(
            content=ft.Column([
                self.strip(),
                ft.Container(height=12),
                # **Absent rather than empty** where the protocol has no sample stream to gate, for the same
                # reason ARRANGEMENT is: the rows were being emptied and the titled panel left behind, so the
                # HRV screen showed an ARTIFACTS heading with a help button about blinks and µV and nothing
                # under it. Its beats *are* filtered, and that is reported beside the tachogram.
                *([panel("ARTIFACTS", self.artifact_rows, help=help_button(
                    page, "Artifact rejection",
                    "A blink reaches 100 to 500 µV, against 10 to 40 µV for EEG. Without this gate, a "
                    "blink goes straight into the band amplitude and drives the feedback.\n\n"
                    "The gate first limits the signal to 0.5 to 50 Hz. The threshold that decides what "
                    "counts as an artifact then comes from EEG only. Mains hum, muscle activity and "
                    "electrode drift do not set it. The lower edge matters: with the amplifier "
                    "set to its 0.01 Hz corner, drift reaches hundreds of microvolts, against 7 to 8 µV "
                    "for EEG.\n\n"
                    "Below 0.5 Hz the gate cannot see anything. Some screens train below 0.5 Hz: "
                    "infra-low synchrony, the waveform and slow cortical potentials. On those screens, "
                    "the gate never rejects a slow artifact. The panel on those screens "
                    "states this, instead of showing a rate that does not cover them.\n\n"
                    "The threshold adapts to the running level, using accepted samples only. This stops "
                    "a long artifact from raising the threshold high enough to pass through unrejected. "
                    "A floor and a ceiling bound the threshold. Without a floor, a quiet stretch starts "
                    "rejecting real signal. Without a ceiling, a noisy stretch rejects nothing.\n\n"
                    "The trace still shows rejected samples; only the measurement skips them. A channel "
                    "the amplifier already reports as unusable is not judged at all.")),
                   ft.Container(height=12)]
                  if session.conditions_signal else []),
                panel("SIGNAL", self.signal_rows, help=help_button(
                    page, "The signal", COMBINE_HELP + "\n\n" + NORMALISATION_HELP)),
                # **Only where the protocol offers them.** The three named arrangements belong to the CFC and
                # HRV methods, each loaded deliberately, and Quiet Reward on an amplitude
                # screen is a setting out of another protocol. So the panel is *absent* rather
                # than empty where `arrangements` is — a panel of three pills that mean nothing here is
                # the same mistake as drawing a crossfade with no two states to fade between, and the
                # amplitude screen carried it for as long as this pane has existed.
                *([panel("ARRANGEMENT", self.model_rows,
                         help=help_button(page, "Arrangements", FEEDBACK_MODELS_HELP))]
                  if session.arrangements else []),
                # The mechanism tabs sit *between* two panels, so they need room on both sides or the row
                # reads as attached to whichever panel it happens to touch. It had 12 above and 10 below,
                # which is less than the gap between the panels themselves — and this row carries its own
                # selected background, so it needs more rather than less.
                ft.Container(height=20),
                self.tabs.control,
                ft.Container(height=14),
                ft.Stack(self._panes),
            ], spacing=0, scroll=ft.ScrollMode.AUTO if inset else None, expand=True),
            padding=ft.Padding(14, 12, 14, 14) if inset else None,
            expand=True, visible=not inset)

        self.rebuild()

    # --- the readout ------------------------------------------------------------------------

    def strip(self, with_shares: bool = True) -> ft.Control:
        """A read-only readout of what is being fed back. A factory: see the module docstring."""
        headline = ft.Text("—", size=24, color=t.TEXT, weight=ft.FontWeight.W_300,
                           max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)
        fill = ft.Container(width=0, height=10, border_radius=5, bgcolor=t.ACCENT)
        note = ft.Text("", size=t.SMALL, color=t.FAINT,
                       max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)
        entry = dict(headline=headline, fill=fill, note=note)
        rest: list[ft.Control] = []
        if with_shares:
            shares = ft.Text("", size=t.SMALL, color=t.FAINT,
                             max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)
            entry["shares"] = shares
            rest = [ft.Container(height=14),
                    ft.Container(content=shares, height=NOTE_H),
                    ft.Container(height=8), self.contributions]
        self._strips.append(entry)
        return panel(
            "OUTPUT",
            ft.Row([ft.Column([ft.Container(content=headline, height=HEADLINE_H),
                               ft.Container(content=note, height=NOTE_H)],
                              spacing=0, expand=True),
                    ft.Container(content=ft.Row([fill], spacing=0),
                                 width=SIGNAL_BAR_W, height=10, border_radius=5,
                                 bgcolor=t.SURFACE_HI)],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            *rest,
            help=help_button(self.page, "Output", OUTPUT_HELP))

    def rebuild_contributions(self, targets) -> None:
        """One share bar per target. Called when the target set changes, not per frame."""
        self._contribution_widgets.clear()
        rows = []
        for tgt in targets:
            # A bandless target contributes to the level like any other, so it belongs in this list — but it
            # has no frequency and therefore no colour from `frequency_colour`. Neutral rather than absent: a
            # missing swatch beside present ones reads as a rendering fault, and a borrowed band colour would
            # claim agreement with a bar on the spectrum that does not exist.
            colour = tgt.colour or t.MUTED
            share_text = ft.Text("—", size=t.BODY, color=t.TEXT)
            share_fill = ft.Container(width=0, height=5, border_radius=3, bgcolor=colour)
            rows.append(ft.Column(
                [ft.Row([ft.Container(width=3, height=11, bgcolor=colour, border_radius=2),
                         # The role, because this list is *about* how each target reaches the level and
                         # the two groups are weighted differently; the arrow, because a reward can be
                         # trained either way and the share alone would not say which.
                         ft.Text(f"{tgt.label} {tgt.role} "
                                 f"{'↑' if tgt.direction == 'increase' else '↓'}",
                                 size=t.SMALL, color=t.MUTED),
                         share_text], spacing=6,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER),
                 ft.Container(height=4),
                 ft.Container(content=ft.Row([share_fill], spacing=0), width=132, height=5,
                              border_radius=3, bgcolor=t.SURFACE_HI)], spacing=0))
            self._contribution_widgets[id(tgt)] = dict(text=share_text, fill=share_fill)
        self.contributions.controls = rows

    def refresh(self) -> None:
        """Per frame. The one thing here that is not cosmetic: when nothing is enabled it must say so."""
        if not self.control.visible:
            return              # a hidden pane must cost nothing; see SpectrumPane.refresh
        session, fb = self.session, self.fb
        if session.conditions_signal:
            self.quality.refresh()
        targets = session.targets
        active = getattr(session, "active_targets", targets)
        # Readings, not observations: a railed channel produces readings that (rightly) never reach a
        # target's statistics, so judging arrival by `tgt.total` would read a connected-but-dead session
        # as "waiting for data" when the correct news is "not delivering signal", named below.
        arriving = bool(targets) and (bool(getattr(session, "amplitudes", None))
                                      or any(x.total for x in targets))
        # Three ways for the signal to be at zero, and they are not the same news. Excluding channels with
        # no signal from the output — which is right — created the middle one: a level of 0% with nothing
        # being output reads as "doing terribly" when it means "nothing is connected".
        live = arriving and bool(active)
        level = session.level if live else 0.0
        met = fb.on_target(active)
        anything_on = any(m.enabled for m in fb.mechanisms.values())
        for strip in self._strips:
            strip["fill"].width = SIGNAL_BAR_W * level if live else 0
            strip["fill"].bgcolor = t.GOOD if level >= 0.5 else t.ACCENT
            if not arriving:
                # "waiting for data" says nothing about what it is waiting for, and on a screen whose
                # instrument is plainly delivering — the chip reading "reading the pulse waveform" a few
                # inches above — it reads as a fault. A session that can say what it is counting says it.
                waiting = _holding(session)
                strip["headline"].value = waiting[0] if waiting else "—"
                strip["headline"].color = t.WARN if waiting else t.FAINT
                strip["note"].value = waiting[1] if waiting else "waiting for data"
            elif not active and (holding := _holding(session)) is not None:
                # **A level can be unavailable without anything being wrong.** A rolling scale that has not
                # filled enough of its window, or a metric that needs a spectrum it does not have yet,
                # produce no level — and the branch below then announced that no channel was delivering
                # signal, in red, while the beats were arriving and visible two panels away. A session that
                # knows why its own level is missing says so; the branch below is for when nothing is.
                strip["headline"].value, strip["note"].value = holding
                strip["headline"].color = t.WARN
            elif not active:
                # **Name the channels that are out, not "no channel".** With every target on a pair, one bad
                # channel makes every target dead — and the headline then said no channel was delivering
                # while the other one was visibly drawing EEG beside it, which is the sort of copy that makes
                # an operator distrust the screen rather than the electrode.
                dead = session.dead_channels()
                # `state` is a per-channel map on the amplifier screens and something else entirely on the
                # ones with a single instrument — the heart's `BeatState`, the headband's. Those have one
                # source and it is already named in `dead_channels`, so there is no "still delivering" list
                # to draw and asking for one crashed the draw loop on the HRV screen.
                by_channel = session.state if isinstance(session.state, dict) else {}
                alive = [f"ch{c}" for c, ok in by_channel.items() if ok is not False]
                strip["headline"].value = (f"{' and '.join(dead)} not delivering signal" if dead
                                           else "no channel is delivering signal")
                strip["headline"].color = t.BAD
                # The device's own word for this is "signal far below range for a long time, or the two
                # inputs shorted together" — so pointing only at contact was pointing at the wrong test,
                # and on the report that prompted this the contact check read 14 and 26 kOhm, i.e. fine.
                strip["note"].value = (
                    (f"{', '.join(alive)} still is · " if alive else "")
                    + "the device reports signal far below range, or that channel's two electrodes "
                      "bridged — look at the gel between them, then check contact on the device screen"
                ) if by_channel else (
                    "the source is chosen but nothing is arriving from it — the device screen says what "
                    "it is doing, and a strap or a clip that is not on the body reads exactly like this")
            elif not anything_on:
                strip["headline"].value = "nothing is being fed back"
                strip["headline"].color = t.WARN
                strip["note"].value = (f"turn on sound, shadow, click or alert · "
                                       f"signal is at {level * 100:.0f}%")
            else:
                strip["headline"].value = fb.describe_outputs(targets, level)
                strip["headline"].color = t.TEXT
                # Short enough not to need the reserved second line in the ordinary case. What the
                # percentage *means* — how far between each output's floor and its ceiling — is an
                # explanation, so it lives behind the panel's `?` rather than being re-read every frame.
                rule = f"{fb.combine} · " if session.scores_bands else ""
                strip["note"].value = (
                    f"level {level * 100:.0f}% · {rule}"
                    f"condition {'met' if met else 'not met'}")
            if "shares" in strip:
                strip["shares"].value = (session.level_source or
                                         (COMBINE_SHARES[fb.combine],))[0]
        self._refresh_fork(live, level, met)
        for tgt in targets:
            wid = self._contribution_widgets.get(id(tgt))
            if wid is None:
                continue
            # **The session's normalisation, not the default.** This read `tgt.score()` bare, so the bars
            # were computed on `sd` whatever the operator had chosen — the one place on the screen showing a
            # per-target contribution was ignoring the control that shapes it. No factor: the per-target
            # ramp has a fixed shape now and the gain is one dial at the end of the chain, on the combined
            # level, which is not a per-target quantity and so has nothing to do with these bars.
            share = tgt.score(self.fb.normalisation) if tgt.total else 0.0
            wid["text"].value = f"{share * 100:.0f}%" if tgt.total else "—"
            wid["fill"].width = 132 * share
        # `gates` is amplitude rejection on blocks of evenly spaced samples, and the HRV screen has no such
        # stream — see `conditions_signal`. Its beats are filtered too, and that is reported beside its
        # tachogram where it can be read against the signal it applies to.
        if session.conditions_signal:
            self.artifact_state.value = session.gates.describe()
            self.artifact_state.color = t.WARN if session.gates.rate > 0.2 else t.MUTED
        # What the player is doing *now* — see the note where this control is placed. A player starts,
        # stops, is held by a failing target and can fail outright, none of which a line written once at
        # build time can report.
        state = getattr(self, "music_state", None)
        if state is not None:
            state.value = session.music.describe(self.fb.music)

    def _refresh_fork(self, live: bool, level: float, met: bool) -> None:
        """The two channels' values and followers, per frame.

        Both lines are drawn even when nothing follows one of them, and say so: *no output on this
        channel* is the state worth naming — it is how a click can be the only thing enabled while the
        level moves visibly and nothing happens.
        """
        fb = self.fb

        def line(value: str, followers: list[str]) -> str:
            """The value, then who is carrying it. **The followers show even with no data**, because who
            follows which channel is the thing this fork exists to state and a paused screen is exactly
            when someone reads it — it said only "—" at first, which hid the wiring until a session ran.
            """
            if not followers:
                return "no output on this channel"
            return f"{value if live else '—'} → {', '.join(followers)}"

        self._level_line.value = line(f"{level * 100:.0f}%", fb.followers(LEVEL))
        self._condition_line.value = line("met" if met else "not met", fb.followers(CONDITION))
        if self.session.state_ends is not None:
            self._balance_line.value = self._balance_text(self.session.state_ends)

    # --- setting handlers -------------------------------------------------------------------

    def _select(self, index: int) -> None:
        for i, pane in enumerate(self._panes):
            pane.visible = i == index
        self.page.update()

    def _toggle(self, obj, attr: str) -> None:
        setattr(obj, attr, not getattr(obj, attr))
        self.rebuild()

    def _set(self, obj, attr: str, value) -> None:
        setattr(obj, attr, value)
        self.rebuild()

    def _set_hold(self, seconds: float) -> None:
        """Change the sample-and-hold. On the session rather than on `FeedbackSettings`, because it is a
        property of how the protocol measures rather than of how the level is dressed for output."""
        self.session.hold_seconds = float(seconds)
        self.rebuild()

    def _set_field(self, attr: str, value) -> None:
        setattr(self.fb, attr, value)
        # The envelope constant and convention are part of an estimator's identity, so changing either
        # has to build the new ones and drop the old — unlike everything else here, which only affects
        # how scores are combined.
        if attr in ("envelope_seconds", "envelope_convention"):
            for tgt in self.session.targets:
                self.session.retune_filter(tgt)
        self.rebuild()

    def _commit_music_path(self, _e=None) -> None:
        self.fb.music.path = str(self.music_path.value or "").strip()
        self.rebuild()

    def _browse(self, _e) -> None:
        async def run() -> None:
            try:
                files = await self._picker.pick_files(dialog_title="Music", allow_multiple=False)
            except Exception:
                return
            if files:
                self.music_path.value = files[0].path
                self._commit_music_path()
        self.page.run_task(run)

    def _click_sound(self) -> str:
        """The sound the click would play right now, so the audition and the session cannot disagree. Keyed,
        that is the earning target's pitch; unkeyed, the chosen character."""
        click = self.fb.click
        if click.keyed:
            earning = getattr(self.session, "earning_target", lambda: None)()
            rewards = [x for x in self.session.targets
                       if getattr(x, "role", "reward") == "reward" and getattr(x, "banded", True)]
            tgt = earning or (rewards[0] if rewards else None)
            if tgt is not None:
                return cues.tone_at(cues.target_pitch(tgt))
        return click.sound

    def _has_state(self) -> bool:
        """Whether this protocol supplies a state for a continuous mechanism to follow."""
        return getattr(self.session, "state_ends", None) is not None

    def _state_label(self) -> str:
        """"toward theta", from the protocol's own names for the two ends — because only it knows what they
        are, and "state" on a pill says nothing about which direction the screen will darken."""
        ends = getattr(self.session, "state_ends", None)
        return f"toward {ends[1]}" if ends else "the state"

    def _sensitivity_bounds(self) -> tuple[float, float] | None:
        """`getattr`, because the heart and headband screens are duck-typed rather than subclasses — see
        `HEGSession`. Every protocol hook this pane reads is reached the same way."""
        return getattr(self.session, "state_sensitivity_range", None)

    def _state_sensitivity(self) -> float:
        return float(getattr(self.session, "state_sensitivity", 1.0))

    def _set_state_sensitivity(self, value: float) -> None:
        self.session.set_state_sensitivity(value)
        self.rebuild()

    def _sensitivity_note(self) -> str:
        """A consequence, not a multiplier. "×2" alone says the output moves twice as much per unit of a
        quantity nobody is looking at; what an operator can check is the microvolts of wave it now takes to
        cross the whole range, which is the same number the slow plot's caption prints."""
        full = getattr(self.session, "state_full_uv", None)
        if full is None:
            # **What it is waiting for, not a paraphrase of the label.** This said "how much of the
            # brightness range today's wave asks for", which is vague exactly when the operator has least
            # to go on: the number is missing because the spread is still being measured, so say that.
            return "measuring the wave's size — the microvolts appear here once it has"
        return f"the whole brightness range over ±{full:.0f} µV of wave · 1× is the protocol's own"

    def _state_shadow_note(self) -> str:
        """Which end of the state the screen is bright at, in the protocol's own two names.

        It read "dims toward the second of the two states, which is how an alpha-theta display can darken
        its visuals", and that was wrong twice. The direction row *directly above* decides which end is
        bright, so under the default — brighter is better — the screen brightens toward the crest rather
        than dimming; and naming a tuple position to somebody who can see both names on the pills beside it
        says less than the pills do. What is left is the one thing the pills cannot say: which way round it
        runs, given the direction actually set.
        """
        low, high = getattr(self.session, "state_ends", None) or ("one end", "the other")
        dark, bright = (high, low) if self.fb.shadow.invert else (low, high)
        return f"brightest at {bright}, darkest at {dark} — the state itself, not a score"

    def _set_tone_hz(self, tgt, pitch_hz: float) -> None:
        """Pin one reward's pitch, and play it — the only way to judge a pitch is to hear it, which is why
        selecting a click sound plays it too. Typing back the derived value leaves it derived, so the box is
        not a one-way door."""
        derived = round(cues.tone_pitch(tgt.band.centre)) if tgt.banded else None
        tgt.tone_hz = None if derived is not None and round(pitch_hz) == derived else float(pitch_hz)
        cues.play(cues.tone_at(cues.target_pitch(tgt)), self.fb.click.volume)
        self.rebuild()

    def _pick_click_sound(self, name: str) -> None:
        """Selecting plays it. Choosing between clicks by name is hopeless, so the only useful control
        is one you hear."""
        self.fb.click.sound = name
        cues.play(name)
        self.rebuild()

    def _pick_alert_sound(self, name: str) -> None:
        self.fb.alert.sound = name
        cues.play(name)
        self.rebuild()

    def _apply_model(self, name: str) -> None:
        self.fb.apply_model(name)
        self.rebuild()

    # --- building ---------------------------------------------------------------------------

    def rebuild(self) -> None:
        fb, session = self.fb, self.session
        problem = cues.problem()
        self._build_artifacts()
        self._build_signal()
        self._build_models()
        self._build_sound()
        self._build_shadow()
        self._build_click(problem)
        self._build_alert(problem)
        self._build_music()
        self.page.update()

    def _build_artifacts(self) -> None:
        if not self.session.conditions_signal:
            self.artifact_state = getattr(self, "artifact_state", None) or ft.Text(
                "", size=t.SMALL, color=t.MUTED)
            self.artifact_rows.controls = []
            return
        policy = self.session.gates.policy
        self.artifact_state = getattr(self, "artifact_state", None) or ft.Text(
            "", size=t.SMALL, color=t.MUTED)
        rows = [
            labelled("rejection",
                     pill("on", policy.enabled, lambda _e: self._toggle(policy, "enabled")),
                     pill("off", not policy.enabled, lambda _e: self._toggle(policy, "enabled")),
),
        ]
        if policy.enabled:
            rows += [
                labelled("what to do",
                         *[pill(mode, policy.mode == mode,
                                (lambda m=mode: lambda _e: self._set(policy, "mode", m))())
                           for mode in REJECT_MODES],
                         note=REJECT_MODES[policy.mode]),
                labelled("threshold",
                         num_box(policy.tolerance,
                                 lambda v: self._set(policy, "tolerance", v),
                                 suffix="× level", step=0.5, low=1.0),
                         note=f"adapting over {policy.tau_seconds:g} s"),
                labelled("but within",
                         num_box(policy.floor_uv, lambda v: self._set(policy, "floor_uv", v),
                                 step=5, low=1),
                         ft.Text("to", size=t.SMALL, color=t.FAINT),
                         num_box(policy.ceiling_uv, lambda v: self._set(policy, "ceiling_uv", v),
                                 suffix="µV", step=25, low=10),
),
                # **The floor is stated and is not editable, and both halves are deliberate.** It is stated
                # because the row used to print one edge of a two-edged band, so "judged on 50 Hz" read as
                # "everything up to 50" — and what the gate does *not* look at is the more useful half on
                # half the screens here. It is not editable for the reason `JUDGE_FLOOR_HZ` gives: there is
                # no legitimate reason to move it, and another box on this pane is another thing to set
                # wrong.
                labelled("judged on",
                         ft.Text(f"{dsp_artifact.JUDGE_FLOOR_HZ:g} to", size=t.SMALL, color=t.FAINT),
                         num_box(policy.limit_hz, lambda v: self._set(policy, "limit_hz", v),
                                 suffix="Hz", step=5, low=10),
),
            ]
            # **What the gate cannot see, said where it says what it is doing.** An infra-low screen trains
            # underneath the judged band entirely, so "rejecting 2% of the session" is true and reads as
            # cover the measurement does not have. See `TrainingSession.trained_floor_hz`, which also says
            # why a sentence is not the fix.
            if self.session.trained_floor_hz < dsp_artifact.JUDGE_FLOOR_HZ:
                # One sentence, under the 110 characters a screen is held to — the rest of the argument is
                # behind this panel's `?`, which is where an explanation belongs.
                rows.append(labelled("", ft.Text(
                    f"This screen trains below {dsp_artifact.JUDGE_FLOOR_HZ:g} Hz, which the gate does "
                    f"not judge: drift there is not rejected.",
                    size=t.SMALL, color=t.WARN, expand=True)))
        rows.append(labelled("", self.artifact_state))
        rows.append(labelled("where", self.quality.control))
        self.artifact_rows.controls = rows

    # --- what the protocol owns -------------------------------------------------------------

    def pinned(self, name: str) -> bool:
        """Is this setting the protocol's rather than the operator's? See `pinned_feedback`."""
        return name in self.session.pinned_feedback

    def _because(self, note: str = "") -> str:
        """The reason beside a locked control. Never omitted — an inert row with no reason reads as a
        broken one, which is the same failure as a greyed launcher card that does not say why."""
        who = "set by the arrangement" if self.session.arrangements else "set by the protocol"
        return f"{note} · {who}" if note else who

    def _eeg_pitch_note(self, mus) -> str:
        """What this pair of pitch ends does to the band's own intervals, worked out for the operator.

        A ratio is not something to read off two pairs of numbers in your head, and it is the whole
        difference between an instrument that sounds like the EEG sped up and one that exaggerates it. Both
        are legitimate; only one of them is what somebody thinks they set.
        """
        band = music.Band(mus.low_hz, mus.high_hz, "music")
        stretch = music.stretch_factor(band, mus.eeg_pitch_lo_hz, mus.eeg_pitch_hi_hz)
        if abs(stretch - 1.0) < 0.05:
            return (f"{mus.low_hz:g} Hz sounds here, {mus.high_hz:g} Hz there · an exact transposition, "
                    f"×{mus.eeg_pitch_lo_hz / max(mus.low_hz, 1e-9):.1f}")
        wider = "wider apart than they are" if stretch > 1 else "closer together than they are"
        return (f"{mus.low_hz:g} Hz sounds here, {mus.high_hz:g} Hz there · intervals ×{stretch:.1f}, "
                f"so the band's rhythms sound {wider}")

    def _choice(self, name: str, options, chosen: str, on_pick, note: str = "",
                label: str = "", labels: dict[str, str] | None = None) -> ft.Control:
        """A row of pills, locked or live, with its note carrying the reason when locked.

        `labels` where the pill text is not the key — some of these dicts map a key to its label and
        others map it to its description, and the row has to read the same either way.
        """
        locked = self.pinned(name)
        options = list(options)
        # One option is **stated rather than offered**, as everywhere else here: a row of pills containing
        # a single pill reads as a setting whose other options have gone missing.
        controls = ([ft.Text((labels or {}).get(options[0], options[0]), size=t.BODY, color=t.TEXT)]
                    if len(options) == 1 else
                    [pill((labels or {}).get(k, k), chosen == k,
                          (lambda k_=k: lambda _e: on_pick(k_))(), enabled=not locked)
                     for k in options])
        return labelled(label, *controls, note=self._because(note) if locked else note)

    def _switch(self, name: str, label: str, on_label: str, obj, attr: str,
                *extra: ft.Control) -> ft.Control:
        """A mechanism's on/off pill. Locked where the protocol decided what it is."""
        locked = self.pinned(name)
        return labelled(label,
                        pill(on_label, getattr(obj, attr),
                             lambda _e: self._toggle(obj, attr), enabled=not locked),
                        *extra,
                        note=self._because() if locked else "")

    def _direction(self, name: str, better: str, worse: str, obj) -> ft.Control:
        """"louder is better" against "quieter is better" — which way the mechanism runs, so it belongs
        to the arrangement rather than to the room."""
        locked = self.pinned(name)
        return labelled("direction",
                        pill(better, not obj.invert,
                             lambda _e: self._set(obj, "invert", False), enabled=not locked),
                        pill(worse, obj.invert,
                             lambda _e: self._set(obj, "invert", True), enabled=not locked),
                        note=self._because() if locked else "")

    def _level_sensitivity_note(self) -> str:
        """What the dial is doing, and — while the level's range is still being measured — that it is.

        **One dial, and the note is where its direction lives.** The control this replaced said how big a
        move had to be before it counted, so turning it up made the feedback flatter; it was reported turned
        from 1 to 10 to get more movement, which removed what movement there was. This one is a gain on the
        finished level, so up is more, and the note says so in the words somebody would use to ask.

        The countdown is the honest picture of the first minute rather than a gate: the dial works
        throughout, on whatever range has been measured so far, exactly as a threshold is provisional and
        then settles. Kept inside the screens' 110-character limit, which is why the measurements behind the
        default live in `feedback.SENSITIVITY_HEADROOM`.
        """
        rng = getattr(self.session, "level_range", None)
        left = rng.remaining(self.session._level_t) if rng is not None else 0.0
        if left > 0:
            return (f"1 = an ordinary swing uses half of each output's travel · "
                    f"still measuring yours, {left:.0f} s")
        return "1 = an ordinary swing uses half of each output's travel · higher is livelier"

    def _stops_row(self, mus) -> ft.Control:
        """Whether the playback is gated, and on what. On the *condition* channel — the same one the
        click is on — which is what makes Quiet Reward one arrangement rather than two coincidences, and
        why it locks with the mechanism rather than with the volumes.

        Except where the protocol says the failing polarity means something else — `music_failing`, set
        where `music_hold` is overridden — in which case the pill and the note say what it actually does
        on this screen, because a control whose behaviour differs from its label is a defect however
        good the reason for the behaviour."""
        labels = dict(STOPS_WHEN)
        detail = dict(STOPS_WHEN_DETAIL)
        failing = getattr(self.session, "music_failing", None)
        if failing is not None:
            labels["failing"] = f"stops while {failing[0]}"
            detail["failing"] = failing[1]
        return self._choice("music", labels, mus.stops_when,
                            lambda k: self._set(mus, "stops_when", k), labels=labels,
                            note=detail[mus.stops_when], label="goes silent")

    def _build_signal(self) -> None:
        """The chain that turns targets into a level, and the output smoothing at the end of it.

        **Steps 1 to 4 only where the level is built that way** — see `TrainingSession.scores_bands`. On the
        HRV screen the level comes from a metric and a normalisation, and those four rows were live controls
        wired to fields nothing there reads: a band-averaging time with no band, a spread to score against
        when the scaling has already mapped the metric onto 0…1, and a combining rule over one target. The
        smoothing is the one step that applies to any level however it was arrived at, so it keeps its row and
        loses its number: there is no chain for it to be the fifth of.
        """
        fb, session = self.fb, self.session
        scored = session.scores_bands

        # The chain is numbered, and the numbers have to stay contiguous when a step is absent — a panel
        # that begins at "2 ·" reads as one that lost a row. A counter rather than literals, because there
        # are now two independent reasons for a step to be missing (`scores_bands`, `tunes_envelope`) and
        # hard-coded numbers were already conditional on one of them.
        step_no = [0]

        def step(name: str) -> str:
            step_no[0] += 1
            return f"{step_no[0]} · {name}" if scored else name

        # Whether any target actually runs an IIR estimator right now. The band-averaging step (and the
        # amplitude convention with it) lives inside the IIR estimators only — cFIR has no envelope
        # stage — and this row used to present itself as step 1 of the pipeline unconditionally, on a
        # screen whose default estimator is cFIR. A numbered step that is not running is not a setting,
        # it is a misdescription of the signal path.
        designs = getattr(session, "filter_for", None)
        iir_in_use = session.tunes_envelope and (
            designs is None or any(designs(t)[0] != "cfir" for t in session.targets))
        envelope_note = ("smooths each band's amplitude before scoring; meanabs is the older "
                         "convention, 0.9× the rms figure on a sine"
                         if iir_in_use else
                         "not in use: every target is on cFIR, which has no envelope stage")
        self.signal_rows.controls = [
            *([labelled(step("band averaging") if iir_in_use else "band averaging",
                       num_box(fb.envelope_seconds,
                               lambda v: self._set_field("envelope_seconds", float(v)),
                               suffix="s", step=0.05, low=0.05, high=2.0),
                       *[pill(k, fb.envelope_convention == k,
                              (lambda k_=k: lambda _e: self._set_field("envelope_convention", k_))())
                         for k in ("rms", "meanabs")],
                       note=envelope_note)]
              if session.tunes_envelope else []),
            *([labelled(step("scale from"),
                       *[pill(k, fb.normalisation == k,
                              (lambda k_=k: lambda _e: self._set_field("normalisation", k_))())
                         for k in NORMALISATIONS
                         if k in getattr(session, "normalisations", NORMALISATIONS)],
                       note=NORMALISATIONS[fb.normalisation]),
              # "driven by" overpromised: it is step 3 of a chain that ends in the *level*, and it does not
              # reach the click or the alert at all. A row that sounds like it governs every output is how
              # the two channels came to look like one.
              # Only the rules this protocol has — see `TrainingSession.combines`. `either` is a claim that
              # the rewards are alternatives rather than a way of mixing them, so it is not on offer where
              # that is false.
              self._choice("combine", [k for k in COMBINES if k in session.combines], fb.combine,
                           lambda k: self._set_field("combine", k),
                           note=fb.describe_combine(), label=step("level from")),
              *([labelled("ratio",
                          num_box(fb.reward_share * 100,
                                  lambda v: self._set_field("reward_share", v / 100),
                                  suffix="% rewards", step=1, low=0, high=100))]
                if fb.combine == "weighted" and not self.pinned("combine") else []),
              # **The condition gets a row, not a footnote.** With the level's rule in a row of pills and
              # the condition's in a remark below, the panel said there was one mixing strategy and an aside
              # — and the aside is the rule that governs every discrete output.
              #
              # **Stated, not offered.** It has exactly one form per protocol, because it is derived from the
              # rule above and `either` is only offered where the rewards really are alternatives. Two inert
              # pills was the first attempt and it read as broken: a control you cannot press, next to a live
              # row, whose second option could never apply on that screen.
              labelled(step("condition"),
                       ft.Text(CONDITIONS[fb.condition_rule], size=t.BODY, color=t.TEXT),
                       note=f"{CONDITION_DETAIL[fb.condition_rule]} · follows the rule above"),
              # **The one dial, and it is here because this is where the level is finished.** After
              # combining, because it is a gain on the combined number and not on any target's own ramp —
              # that was the old control and it could not widen anything, since a gain on an off-centre
              # level saturates. Before the smoothing, so the two settings stay separate questions: how
              # much output a swing produces, then how twitchy it is.
              labelled(step("sensitivity"),
                       num_box(fb.sensitivity,
                               lambda v: self._set_field("sensitivity", float(v)),
                               suffix="×", step=0.1,
                               low=SENSITIVITY_RANGE[0], high=SENSITIVITY_RANGE[1]),
                       note=self._level_sensitivity_note())]
              if scored else []),
            # **Where the protocol holds its value, that is step one of the chain and not a footnote.** It
            # decides how often every later step can change, and on the infra-low screen it also decides how
            # many values the threshold window holds — fifteen against six hundred, which is a different
            # threshold and not a smoother one. Offered only where a protocol has the choice; see
            # `TrainingSession.hold_choices`.
            *([self._choice("hold", list(session.hold_choices), session.hold_seconds,
                            lambda k: self._set_hold(k), labels=session.hold_choices,
                            label="value updates")]
              if session.hold_choices else []),
            labelled(step("output smoothing"),

                     num_box(fb.smoothing_seconds,
                             lambda v: self._set_field("smoothing_seconds", float(v)),
                             suffix="s", step=0.05, low=0.0),
                     note="smooths the level, not the condition — a yes or no has nothing to smooth",
),
            ft.Container(height=6),
            *self._fork_rows(),
            # Trailing space, or the last row sits hard against the panel's border while every other pair
            # is 14 apart — which reads as the panel having been cut off rather than having ended.
            ft.Container(height=6),
        ]

    def _fork_rows(self) -> list[ft.Control]:
        """Where the chain forks, printed with both channels' live values and their followers.

        The four rows above make one number. Only half the outputs read it — the click and the alert ask
        the targets directly — and with nothing saying so, the rule in row 3 reads as though it governs
        everything. So the chain does not end at "the level": it ends at two named channels, each with
        who is currently carrying it, which is also the answer to "the signal moved and this did not".

        A third line appears only where a protocol publishes a quantity of its own (`state_balance`, which
        is a *state* rather than a score). Stated where it exists, absent where it does not, for the same
        reason music is not offered on a screen with no two ends to fade between.
        """
        self._level_line = ft.Text("", size=t.SMALL, color=t.MUTED)
        self._condition_line = ft.Text("", size=t.SMALL, color=t.MUTED)
        self._balance_line = ft.Text("", size=t.SMALL, color=t.MUTED)
        source = self.session.level_source
        rows = [labelled("→ level", self._level_line,
                         note=source[1] if source else COMBINE_DETAIL[self.fb.combine]),
                labelled("→ condition", self._condition_line)]
        if self.session.state_ends is not None:
            rows.append(labelled("→ balance", self._balance_line))
        return rows

    def _build_models(self) -> None:
        """The arrangements this protocol offers, and **what each one has set**.

        Pressing one used to change five fields spread across nested tabs and show nothing at all: a
        control whose whole effect is off-screen, which is the same class of defect as the readout that
        changed the height of its own pane. So the panel prints one line per mechanism underneath, and
        pressing a pill visibly rewrites them in place — the consequences shown rather than a flash, the
        same reasoning as the filter dialog printing its delay and its ripple instead of offering a
        friendlier knob.
        """
        fb = self.fb
        if not self.session.arrangements:
            self.model_rows.controls = []
            return
        summary = [f"sound {'on' if fb.sound.enabled else 'off'}"
                   + (" · inverted" if fb.sound.enabled and fb.sound.invert else ""),
                   f"click {'on' if fb.click.enabled else 'off'}",
                   f"alert {'on' if fb.alert.enabled else 'off'}",
                   ("music " + (STOPS_WHEN[fb.music.stops_when] if fb.music.enabled else "off"))]
        self.model_rows.controls = [
            ft.Row([*[pill(name, fb.arrangement == name,
                           (lambda n=name: lambda _e: self._apply_model(n))())
                      for name in self.session.arrangements]], spacing=6),
            labelled("sets", ft.Text(" · ".join(summary), size=t.SMALL, color=t.MUTED)),
        ]

    def _build_sound(self) -> None:
        fb = self.fb
        trouble = sysvolume.problem()
        self.sound_rows.controls = [
            self._switch("sound", "output", "sound on", fb.sound, "enabled",
                         reads_chip(fb.sound)),
            self._direction("sound", "louder is better", "quieter is better", fb.sound),
            range_row("volume", fb.sound, self._set, note=""),
            labelled("", ft.Text(
                trouble if trouble else
                f"the machine's master volume · floor {sysvolume.HARD_FLOOR:.0%} · restored on stop",
                size=t.SMALL, color=t.WARN if trouble else t.FAINT)),
        ]

    def _build_shadow(self) -> None:
        fb = self.fb
        trouble = shadow.problem()
        note = (trouble if trouble else
                f"floor {shadow.HARD_FLOOR:.0%} even if the range allows less · restored on stop")
        self.shadow_rows.controls = [
            self._switch("shadow", "output", "shadow on", fb.shadow, "enabled",
                         reads_chip(fb.shadow)),
            # Only what exists. A pill reading "(soon)" was a control for a thing that is not there, on
            # every screen; the mechanism stays named in `SHADOW_MECHANISMS` for the day it is built.
            labelled("drawn by",
                     *[pill(k, fb.shadow.mechanism == k,
                            (lambda k_=k: lambda _e: self._set(fb.shadow, "mechanism", k_))(),
                            enabled=not self.pinned("shadow"))
                       for k in SUPPORTED_SHADOW],
                     note=(self._because(SHADOW_MECHANISMS[fb.shadow.mechanism])
                           if self.pinned("shadow") else SHADOW_MECHANISMS[fb.shadow.mechanism])),
            self._direction("shadow", "brighter is better", "dimmer is better", fb.shadow),
            # **Offered only where the protocol has a state**, the same gate the crossfade uses and for
            # the same reason: with no `state_balance` this would render as a brightness frozen at one end.
            # Stated rather than offered when there is nothing to choose between.
            *([labelled("follows",
                        *[pill(name, fb.shadow.follows == key,
                               (lambda k_=key: lambda _e: self._set(fb.shadow, "follows", k_))(),
                               enabled=not self.pinned("shadow"))
                          for key, name in ((LEVEL, "how well"), (STATE, self._state_label()))],
                        note=(self._state_shadow_note() if fb.shadow.follows == STATE
                              else "brightness tracks the combined score"))]
              if self._has_state() else []),
            # **Under the control it acts on, and only where the protocol offers it.** The brightness row
            # below sets where the output starts and stops; this sets how much of that a wave of today's
            # size asks for, which is a different question and the one behind "it barely moves". It lives
            # here rather than on the plot panel that draws the state because this is where somebody
            # watching a flat screen is looking, and it is absent unless the protocol supplies
            # `state_sensitivity_choices` — a state already bounded 0…1 by construction has nothing for a
            # gain to fix.
            *([labelled("sensitivity",
                        num_box(self._state_sensitivity(), self._set_state_sensitivity,
                                suffix="×", width=58, step=0.25,
                                low=self._sensitivity_bounds()[0],
                                high=self._sensitivity_bounds()[1]),
                        note=self._sensitivity_note())]
              if fb.shadow.follows == STATE and self._sensitivity_bounds() else []),
            range_row("brightness", fb.shadow, self._set, note=""),
            # **Its own smoothing, because how fast the level moves is a property of the protocol.** A screen
            # whose level crosses thresholds several times a second strobes the whole backlight otherwise;
            # the note prints what the current value actually leaves of a 3 Hz component, since choosing a
            # time constant by feel is how this shipped at a number that did nothing.
            labelled("smoothing",
                     num_box(fb.shadow.smoothing_seconds,
                             lambda v: self._set(fb.shadow, "smoothing_seconds", max(0.0, float(v))),
                             suffix="s", step=0.1, low=0.0, high=5.0),
                     note=(f"leaves {_flicker_left(fb.shadow.smoothing_seconds):.0%} of a 3 Hz flicker, "
                           f"over the display's own {shadow.APPROACH_SECONDS:g} s"
                           if fb.shadow.smoothing_seconds else
                           f"the display's own {shadow.APPROACH_SECONDS:g} s only")),
            labelled("screen",
                     num_box(fb.shadow.screen,
                             lambda v: self._set(fb.shadow, "screen", int(v)),
                             width=38, step=1, low=1),
                     # Muted, the display list is empty by design — see `shadow.displays` — so counting
                     # monitors here would print "0 attached" on a machine that plainly has one. The
                     # problem itself is the line under this one; saying it here as well was the same
                     # sentence twice on one pane.
                     note=("" if shadow.muted() else f"{len(shadow.displays())} attached")),
            labelled("", ft.Text(note, size=t.SMALL, color=t.WARN if trouble else t.FAINT)),
        ]

    def _build_click(self, problem) -> None:
        """Keyed, this pane is about **tones** and says so throughout — see `Click.label`.

        The character menu is *replaced* rather than disabled, because when the sound comes from the
        earning band there is no character to choose: it was a row of pills that changed nothing, which is
        worse than a missing control. What goes in its place is one audition per reward, since the pitch
        distinguishing them is the whole mechanism and hearing both together is the only way to judge it.
        """
        fb = self.fb
        click = fb.click
        # `banded` as well as the role: a keyed tone's pitch comes from a *frequency*, so a reward with no
        # band has no tone to audition. Alpha-theta's θ−α is such a reward, and `earning_target` never
        # returns it for exactly the same reason — the pitch says which band earned it, and a difference is
        # not a band. Without this the audition row read `.band.centre` off None.
        rewards = [x for x in self.session.targets
                   if getattr(x, "role", "reward") == "reward" and getattr(x, "banded", True)]
        if click.keyed:
            # **The pitch is settable per reward, and the ▶ is beside the number that sets it.** Derived is
            # a default: the whole job of these two pitches is to be told apart with the eyes closed, and
            # how far apart that takes is a property of the listener and the room, not of the bands. So the
            # box shows the pitch in force and typing one pins it — the same shape as a threshold, where the
            # derivation is the starting point and the operator owns the number.
            #
            # **One row per reward.** Both on one line put two labels, two number boxes and two ▶ across a
            # pane that already carries an interval, a hold and a volume, and the whole tab went wide enough
            # to push the panes beside it off the screen. A row each is the same information at half the width,
            # and the two auditions are still adjacent, which is the only reason they share a pane.
            derived = all(getattr(x, "tone_hz", None) in (None, 0) for x in rewards)
            tone_rows = [
                labelled(x.label,
                         num_box(round(cues.target_pitch(x)),
                                 (lambda g=x: lambda v: self._set_tone_hz(g, v))(),
                                 suffix="Hz", width=58, step=10,
                                 low=cues.TONE_LO_HZ, high=cues.TONE_HI_HZ),
                         test_button("", (lambda g=x: lambda _e: cues.play(
                             cues.tone_at(cues.target_pitch(g)), click.volume))(), problem is None),
                         )
                for x in rewards]
            if not tone_rows:
                tone_rows = [labelled("tones", ft.Text("no reward to sound like yet", size=t.SMALL,
                                                       color=t.FAINT))]
            sound_row = ft.Column(
                tone_rows + [ft.Text(problem or ("pitched by the band that earned it — press to hear each"
                                                 if derived else "set yourself — press to hear each"),
                                     size=t.SMALL, color=t.WARN if problem else t.FAINT)],
                spacing=8)
        else:
            sound_row = labelled("sound",
                                 *[pill(name, click.sound == name,
                                        (lambda n=name: lambda _e: self._pick_click_sound(n))())
                                   for name in cues.click_choices()],
                                 note=f"{cues.duration_ms(click.sound):.0f} ms")
        # Beside the interval, because the two are the same kind of thing and read as one sentence:
        # hold it for this long, then no faster than that. Hidden when zero would be wrong — a control that
        # says "0 s" is clearer than one that vanishes, since a missing dwell reads as a feature absent
        # rather than off.
        rows = [
            self._switch("click", "output", f"{click.label} on", click, "enabled",
                         reads_chip(click)),
            labelled(f"one {click.label} every",
                     num_box(click.interval_seconds,
                             lambda v: self._set(click, "interval_seconds", v),
                             suffix="s", step=0.5, low=0.1),
),
            # **Beside the interval, and with its own audition.** The two sounds are one channel in two
            # states, so the only question that matters is whether you can tell them apart — which you can
            # only answer by pressing one and then the other. Same reasoning as the two music ends sharing the
            # Music pane rather than being split across tabs.
            *([labelled("doubles after",
                     num_box(click.hold_seconds,
                             lambda v: self._set(click, "hold_seconds", max(0.0, v)),
                             suffix="s" if click.hold_seconds else "", step=1.0, low=0),
                     *([ft.Text("never", size=t.SMALL, color=t.FAINT)] if not click.hold_seconds else
                       [test_button("", (lambda: lambda _e: cues.play(
                           cues.held(self._click_sound()), click.volume))(), problem is None)]),
                     note=(f"one mark on arriving; two once the condition has been met for at least "
                           f"{HOLD_FRACTION:.0%} of the last {click.hold_seconds:g} s"
                           if click.hold_seconds else
                           f"the {click.label} never changes character"))]
              # Only where the protocol has a sustained reward beside its momentary one — see
              # `TrainingSession.sustains_reward`. Elsewhere this defaulted to 0 and read "never": a control
              # for a distinction the protocol does not make, in the panel of a mechanism every protocol has.
              if getattr(self.session, "sustains_reward", False) else []),
            sound_row,
            labelled("volume",
                     num_box(click.volume, lambda v: self._set(click, "volume", v),
                             step=5, low=0, high=100),
                     note="its own gain, like the music's — Sound moves the master over both"),
            labelled("when", ft.Text(self.fb.describe_condition(), size=t.SMALL, color=t.TEXT),
),
        ]
        bias = self.session.tone_bias
        if click.keyed and bias is not None:
            # Which reward the pitch marks when more than one is available. Protocol-supplied labels, like
            # the music's two ends — only the protocol knows what the preference means. It was on the Train
            # tab, which is for what you watch; this is sound configuration and belongs beside the sound.
            rows.append(labelled("marks",
                                 pill(bias[0], not self.session.unbiased,
                                      lambda _e: self._set(self.session, "unbiased", False)),
                                 pill(bias[1], self.session.unbiased,
                                      lambda _e: self._set(self.session, "unbiased", True)),
                                 note="when more than one reward is above threshold"))
        if not click.keyed:
            rows.append(labelled("audition",
                                 test_button("one click",
                                             lambda _e: cues.play(click.sound, click.volume),
                                             problem is None),
                                 note=(problem or "")))
        self.click_rows.controls = rows

    def _build_alert(self, problem) -> None:
        """The alert is CFC's, from the manual, and the pane has to say so — it is the one mechanism here
        whose *purpose* is not guessable from its controls."""
        fb = self.fb
        keyed = fb.alert.keyed
        examples = ((3.0, "a slow bin"), (10.0, "alpha"), (35.0, "a fast bin"))
        self.alert_rows.controls = [
            self._switch("alert", "output", "alert on", fb.alert, "enabled",
                         reads_chip(fb.alert)),

            labelled("which sound",
                     pill("by frequency", keyed, lambda _e: self._set(fb.alert, "keyed", True)),
                     pill("always the same", not keyed,
                          lambda _e: self._set(fb.alert, "keyed", False)),
                     note=("slow bands thock, fast ones tick" if keyed
                           else "one sound for every target")),
            *([labelled("audition",
                        *[test_button(name,
                                      (lambda h=hz: lambda _e:
                                       cues.play(cues.frequency_click(h), fb.alert.volume))(),
                                      problem is None)
                          for hz, name in examples],
                        note=(problem or ""))]
              if keyed else
              [labelled("sound",
                        *[pill(name, fb.alert.sound == name,
                               (lambda n=name: lambda _e: self._pick_alert_sound(n))())
                          for name in cues.click_choices()],
),
               labelled("audition",
                        test_button("one alert",
                                    lambda _e: cues.play(fb.alert.sound, fb.alert.volume),
                                    problem is None),
                        note=(problem or ""))]),
            labelled("no faster than",
                     num_box(fb.alert.interval_seconds,
                             lambda v: self._set(fb.alert, "interval_seconds", v),
                             suffix="s", step=0.5, low=0.2),
),
            labelled("volume",
                     num_box(fb.alert.volume, lambda v: self._set(fb.alert, "volume", v),
                             step=5, low=0, high=100),
                     note="louder than the click by default: the exception has to be noticed"),
            labelled("", ft.Text("one at a time, worst offender first", size=t.SMALL, color=t.FAINT)),
        ]

    def _colour_row(self, label: str, field: str, note: str | None = None,
                    pitch_field: str | None = None) -> ft.Control:
        """One tilt with a preview, and optionally where that end sits.

        The preview is the row's whole justification: "1/f²" does not tell anybody how deep brown sounds, and
        this is the one setting on the pane whose name cannot. **The pitch box is on the same row as the tilt**
        because they are two properties of one end of the music — what it is made of and where it sits — and the
        preview beside them plays both, or an audition would be of something not on screen.
        """
        mus = self.fb.music
        chosen = getattr(mus, field)
        problem = cues.problem()
        detail = note or music.NOISE_COLOURS[chosen]
        pitch = getattr(mus, pitch_field) if pitch_field else 0.0
        extra: list = []
        if pitch_field is not None:
            extra = [ft.Container(width=6),
                     ft.Text("pitch", size=t.SMALL, color=t.FAINT),
                     num_box(pitch, (lambda f=pitch_field: lambda v: self._set_pitch(f, v))(),
                             suffix="Hz" if pitch else "", width=52,
                             step=music.PITCH_LO_HZ, low=0, high=music.PITCH_HI_HZ),
                     ft.Text("" if pitch else "off", size=t.SMALL, color=t.FAINT)]
        return labelled(
            label,
            *[pill(k, chosen == k, (lambda k_=k: lambda _e: self._set(mus, field, k_))())
              for k in music.NOISE_COLOURS],
            *extra,
            ft.Container(width=6),
            test_button("hear it",
                        (lambda: lambda _e: cues.play_file(
                            music.preview_path(getattr(mus, field), pitch), mus.volume))(),
                        problem is None),
            note=f"{detail} · {problem}" if problem else detail)

    def _set_pitch(self, field: str, hz: float) -> None:
        """Set one end's pitch and play it, because a pitch is another setting whose name cannot describe it.

        **Zero is off; anything above zero is clamped up into the legal band.** An earlier version snapped
        *down* to off below the 70 Hz corner, on the reasoning that a pitch just above the corner is not a
        useful place to land by accident — which made the control unusable from its own default: the box
        starts at 0, the stepper adds 50, 50 is under the corner, so it snapped straight back to 0 and neither
        the arrows nor a typed value below 70 could ever do anything. Clamping up means 0 + one step lands on
        the lowest pitch the music can actually render.
        """
        hz = float(hz)
        hz = 0.0 if hz <= 0 else min(music.PITCH_HI_HZ, max(music.PITCH_LO_HZ, hz))
        setattr(self.fb.music, field, hz)
        colour = self.fb.music.colour if field == "pitch_hz" else self.fb.music.second_colour
        cues.play_file(music.preview_path(colour, hz), self.fb.music.volume)
        self.rebuild()

    def _balance_text(self, ends: tuple[str, str]) -> str:
        balance = self.session.state_balance
        if balance is None:
            return "waiting for data"
        return f"{balance:.0%} toward {ends[1] or 'the second tilt'}"

    def _build_music(self) -> None:
        fb, session = self.fb, self.session
        mus = fb.music
        # Survives a rebuild, so `refresh` keeps writing to the control the panel is actually showing —
        # and is filled in here as well, or a panel just rebuilt would show a blank line until the next
        # frame. Building it and refreshing it are the same statement, written twice on purpose.
        self.music_state = getattr(self, "music_state", None) or ft.Text(
            "", size=t.SMALL, color=t.MUTED)
        self.music_state.value = session.music.describe(mus)
        browse = ft.Container(content=ft.Text("browse…", size=t.SMALL, color=t.ACCENT),
                              padding=ft.Padding(11, 8, 11, 8), border_radius=6,
                              border=ft.Border.all(1, t.BORDER), on_click=self._browse)
        # Only the music sources this protocol can actually use — see `TrainingSession.music_sources`. With one
        # source there is no choice to present, so it is stated rather than offered: a row of pills with a
        # single pill in it reads as a setting whose other options have gone missing.
        sources = [k for k in MUSIC_SOURCES if k in session.music_sources]
        if not sources:
            sources = [mus.source]
        ends = session.state_ends or ("", "")
        self.music_rows.controls = [
            self._switch("music", "output", "music on", mus, "enabled"),
            labelled("source",
                     *([ft.Text(sources[0], size=t.BODY, color=t.TEXT)] if len(sources) == 1 else
                       [pill(k, mus.source == k,
                             (lambda k_=k: lambda _e: self._set(mus, "source", k_))())
                        for k in sources]),
                     note=MUSIC_SOURCES.get(mus.source, "")),
            *([self._colour_row("colour", "colour", pitch_field="pitch_hz"),
               self._stops_row(mus)]
              if mus.source == "noise" else
              # Both ends, each labelled with the state it stands for, because "crossfade" alone does not
              # say that it runs brown to pink — and which end is which is the only thing about it an
              # operator needs to know. The balance is the protocol's, so it is reported and not set.
              [self._colour_row(f"{ends[0]} end".strip(), "colour", pitch_field="pitch_hz"),
               self._colour_row(f"{ends[1]} end".strip(), "second_colour",
                                pitch_field="second_pitch_hz"),
               labelled("balance",
                        ft.Text(self._balance_text(ends), size=t.SMALL, color=t.MUTED),
                        note=self._because("the fade is the feedback, not the loudness"))]
              if mus.source == "crossfade" else
              # **What makes it a Shepard tone is stated; how far it travels is set.** Twelve partials an
              # octave apart and a bell over log-frequency are the instrument, so a box for either would be
              # a box for turning it into something else. The travel is not: it is this mechanism's output
              # range, the same setting the shadow's brightness span is, and it was a module constant
              # printed as a caption — which left the one mechanism carrying an infra-slow state as the
              # only continuous output with no say in how far it moves.
              [labelled("wheel",
                        ft.Text(f"{music.SHEPARD_SUBTONES} partials an octave apart, "
                                f"{music.SHEPARD_MIN_HZ:g}–{music.SHEPARD_MIN_HZ * 2 ** music.SHEPARD_SUBTONES:g} Hz",
                                size=t.SMALL, color=t.MUTED),
                        note=self._because("a pitch with no top is what an unbounded signal needs")),
               # **Named for what the listener hears, not for what the instrument does.** This row said
               # "6 oct across the range · half the wheel", which is three pieces of jargon in nine words:
               # *which* range (the signal's, not the plot's, not the pitch's), and the wheel is an internal
               # of the synthesis nobody outside this file has met. What an operator is choosing is how far
               # the pitch climbs between the quietest and the loudest the signal gets, so the row says that
               # and the number is in octaves because that is the unit a pitch interval has.
               labelled("how far it climbs",
                        num_box(mus.travel_octaves or music.SHEPARD_OCTAVES,
                                lambda v: self._set(mus, "travel_octaves", v),
                                suffix="oct", width=64, step=0.5,
                                low=0.25, high=music.SHEPARD_OCTAVES),
                        ft.Text(f"from the bottom of the signal to the top, "
                                f"over {mus.glide_seconds or music.SHEPARD_GLIDE_SECONDS:g} s",
                                size=t.SMALL, color=t.MUTED),
                        note=(f"{music.SHEPARD_OCTAVES:g} octaves is as far as it goes — climb further and "
                              f"the top starts sounding like the bottom again"
                              if (mus.travel_octaves or music.SHEPARD_OCTAVES) >= music.SHEPARD_OCTAVES
                              else f"a smaller climb is gentler to sit with; "
                                   f"{music.SHEPARD_OCTAVES:g} is the widest")),
               # The same gate the shadow's `follows` row uses, for the same reason: with no state
               # there is nothing to choose between and the pitch follows the level, stated by absence.
               *([labelled("follows",
                           *[pill(name, mus.follows == key,
                                  (lambda k_=key: lambda _e: self._set(mus, "follows", k_))())
                             for key, name in ((LEVEL, "how well"), (STATE, self._state_label()))],
                           note=("the pitch rides the signal itself, which is what the reference's "
                                 "own tone follows" if mus.follows == STATE
                                 else "the pitch tracks the combined score"))]
                 if self._has_state() else []),
               self._stops_row(mus)]
              if mus.source == "shepard" else
              # The two ends of the pitch, which are the whole of this source: unlike the Shepard wheel it is
              # *bounded*, so where its ends sit is a real choice — a wide range is easier to read and a
              # narrow one is easier to live with for an hour. Without this branch it fell through to the
              # file browser and offered to play an mp3, the same way the Shepard tone did.
              [labelled("pitch",
                        num_box(mus.tone_base_hz, lambda v: self._set(mus, "tone_base_hz", v),
                                step=10, low=50, high=mus.tone_top_hz - 50),
                        ft.Text("–", size=t.BODY, color=t.FAINT),
                        num_box(mus.tone_top_hz, lambda v: self._set(mus, "tone_top_hz", v),
                                suffix="Hz", step=10, low=mus.tone_base_hz + 50, high=4000),
                        note="low is the bottom of the signal, high is the top"),
               labelled("glide",
                        ft.Text(f"{music.TONE_GLIDE_SECONDS:g} s to reach a new pitch",
                                size=t.SMALL, color=t.MUTED),
                        note=self._because("a pitch that jumps sounds like a series of beeps")),
               self._stops_row(mus)]
              if mus.source == "tone" else
              [labelled("band",
                        num_box(mus.low_hz, lambda v: self._set(mus, "low_hz", v),
                                step=0.5, low=0.5, high=mus.high_hz - 0.5),
                        ft.Text("–", size=t.BODY, color=t.FAINT),
                        num_box(mus.high_hz, lambda v: self._set(mus, "high_hz", v),
                                suffix="Hz", step=0.5, low=mus.low_hz + 0.5),
),
               # **Where the movement actually lives**, and the row that replaced a carrier frequency. A
               # carrier could only be *offset* by the EEG, so the whole band arrived inside a few tens of
               # Hz and no setting on this panel could make it move; these two ends say how far the pitch
               # travels between the bottom of the band and the top of it. The note is the live arithmetic
               # rather than advice, because whether this pair is a transposition or a stretch is a fact
               # about the two ranges and nobody should have to work it out.
               labelled("pitch",
                        num_box(mus.eeg_pitch_lo_hz, lambda v: self._set(mus, "eeg_pitch_lo_hz", v),
                                step=10, low=50, high=mus.eeg_pitch_hi_hz - 50),
                        ft.Text("–", size=t.BODY, color=t.FAINT),
                        num_box(mus.eeg_pitch_hi_hz, lambda v: self._set(mus, "eeg_pitch_hi_hz", v),
                                suffix="Hz", step=10, low=mus.eeg_pitch_lo_hz + 50, high=4000),
                        note=self._eeg_pitch_note(mus))]
              if mus.source == "eeg" else
              [labelled("file or folder", self.music_path, browse,
                        note=f"{len(music.tracks(mus.path))} track(s)" if mus.path else
                             "an audio file, or a folder played in order and repeated"),
               self._stops_row(mus)]),
            labelled("volume",
                     num_box(mus.volume, lambda v: self._set(mus, "volume", v),
                             step=5, low=0, high=100),
                     note="its own gain, not the system volume"),
            # **Rebuilt with the panel, rewritten every frame**, and it was only the first — so the line
            # said whatever was true at the instant a control was last touched. Choosing `eeg` rebuilds
            # this pane immediately, and the stream starts on the next output apply a fraction of a second
            # later, so it froze at "waiting for a channel" and stayed there for the whole session with the
            # sound audibly playing. Same for noise and "not started". It is a live readout of a player's
            # state, which is the same reason `artifact_state` is refreshed and not built.
            labelled("", self.music_state),
        ]
