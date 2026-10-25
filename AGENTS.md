# Working on this repo

A macOS driver and neurofeedback console for the Neurobit Optima, a 4-channel EEG amplifier. The wire
protocol was recovered from the vendor's driver so this software can talk to the device. This file is the
operating guide and is loaded into every session, so it holds rules, not stories. Longer notes live beside
the repository in `CLAUDE.local.md` and what it points to.

## The device

- `make probe`, `make impedance`, `make live` need the amplifier switched on. USB presence is not power: it
  runs on two AA cells with a physical switch, and `find_port()` succeeding with nothing answering means it
  is off.
- The USB serial (the FTDI module's) is not the device serial. `find_port()` globs
  `/dev/cu.usbserial-*`, so pass `--port` if another FTDI device is attached.

## Boundaries

Three layers, and they stay separate: **acquisition** (`nfb.sources`, `nfb.decode`, `nfb.stream`) owns the
device and emits sample blocks with absolute indices; **processing** works on blocks and knows nothing
about devices; **feedback** maps values to output and is the only place a reward mapping belongs. Inside
`optima`, `protocol` and `tables` depend on nothing, `config` on `tables`, `stream` on those, `device` only
on `protocol`. If transport starts needing `config`, something has gone wrong.

## Running it

`make` with no arguments lists everything, with the variables each target takes. A recording is never
overwritten: `make record` writes a timestamped file and repoints `sessions/latest.jsonl.gz`.

## Checking your work

`make test` is the whole suite and takes about a minute. Spend it once, at the end.

- **Drive the code, never start it and wait.** `TrainingSession.feed_block` is the feed loop's body, so a
  test is the loop. Sources take `realtime=False`. Beats, frames and states are values: push them in.
- **No test waits on the clock.** `conftest.waitfor(condition, what=...)` is the only sanctioned wait, for
  genuinely concurrent behaviour. `time.sleep` over 50 ms raises inside the suite.
- **Convert samples to counts in one pass** with `conftest.counts`, and feed device-sized blocks only where
  blocking is the subject: a gap, a held block, an index discontinuity.
- **Run the narrowest thing that could fail first.** `make lint` is the check for mechanical edits; a
  rename or a prose change learns nothing from the suite.
- **Never re-run on an unchanged tree.** Committing is not a reason to ask again, and splitting a commit does
  not require re-testing each half unless an intermediate state holds logic that never existed in a tree
  that passed.
- **"Not run" and "not looked at" are fine things to report.** Say what was checked and what was not.

**Every string a user can read is Simplified Technical English (ASD-STE100):** captions, notes, help
dialogs, pills, tooltips, menu items. One topic per sentence, at most 25 words, present tense, active voice,
"must" and "can" and never "may/might/shall/should", no contractions, no Latin abbreviations, one word for
one thing. `tests/test_ste.py` enforces the mechanical half; word choice is a reading job. Code comments
are out of scope. In help text, `**word**` is bold and `*word*` is italic, and nothing else is markup; a
bold word in lower case names a label on that screen, and the test checks that the label exists.

**A change to a screen is not done until a frame of it has been looked at.** `tests/test_layout` catches
the faults that stay in the control tree while the suite is green; the rest needs eyes. `make shots`
renders every screen, cold and fed, into `shots/`; `make shots SCREEN=PAC` narrows it.

## Rules that cost an evening each

- **Every UI update runs on Flet's event loop.** Use `page.run_task(async_fn)`. Mutating controls from a
  plain thread and calling `page.update()` silently does nothing, and no test can see it. If a UI symptom
  cannot be reproduced in a test, set `NFB_DEBUG=1` and read `/tmp/nfb-panel.log` before guessing.
- **`NFB_NO_OUTPUTS=1` for every test run, benchmark and scratch script.** It switches the shadow, the
  system volume and the cues off for the process; `tests/conftest.py` sets it autouse. A session takes the
  operator's display and volume the moment it runs, which is right for a session and wrong for anything else.
- **The reference is `linked` by default**, one reference electrode serving all four channels, because
  cross-channel measures need a shared reference. The cost: one bad reference electrode makes every channel
  noisy at once, with no channel obviously at fault. The impedance test finds it.
- **A working channel reports signal state 0, not 1.** State 2 is "no signal"; 1 has never been observed on
  this hardware.
