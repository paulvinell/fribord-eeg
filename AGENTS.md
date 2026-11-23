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

Three checks. Use the smallest one that can fail.

- `make lint` for a mechanical edit. A rename, a moved line or a prose change learns nothing from a test.
- `make test` while you work and before a commit. It runs only the tests that the uncommitted changes can
  break (`tools/affected.py` reads the imports), and it runs nothing on an unchanged tree.
  `make test T=tests/test_pac_view.py` or `T="tests/test_dsp.py -k envelope"` runs one part.
- `make test ALL=1` is every test, on every performance core, in about 15 seconds. Run it once before the
  commit when the change is to a module that many others use, or crosses from one screen to another.

**Never re-run on an unchanged tree.** Committing is not a reason to ask again, and splitting a commit does
not require re-testing each half unless an intermediate state holds logic that never existed in a tree
that passed. **"Not run" and "not looked at" are fine things to report.** Say what was checked and what
was not.

### Where a test goes

Each screen has its own file, named for its module: `pac_view.py` is tested in `tests/test_pac_view.py`.
`tests/test_screens.py` holds only what two or more screens must share. `tests/harness.py` holds what the
screen tests share: the stub page, the sources that do not wait, the walks over a built screen.
`tests/test_layout.py` and `tests/test_ste.py` are rules that run over every screen.

### What earns a test

Add a test when a fault would show in a session (a wrong number, a crash, a dead control, a wrong sound)
and no test fails for it now. Put it at the lowest layer that shows the fault: if a function returns the
value, call the function and do not build a screen.

- **A class of fault gets a rule, not a test.** Text that does not fit, a label that is not there, a
  control that blinks: extend the rule in `test_layout` or `test_ste`, which then holds for every screen.
- **Do not copy a value.** A test that repeats a default, a band edge or a sentence from the program fails
  on every intended change and finds no fault. Assert the property the value must have.
- **Do not read private state** when a public reading shows the same thing. The test then breaks on a
  refactor and passes on a real fault.
- **Do not guard history.** "Untouched by the X work" means nothing once X is done.
- **The docstring says what must hold, in a sentence or two.** The story of the bug goes in the commit
  message.

### What keeps a test fast

- **A test that uses more than 1.5 s of processor time fails**, unless it carries
  `@pytest.mark.slow("the reason")`. A slow test is usually slow program code, which the operator pays for
  too: profile before you mark.
- **Drive the code, never start it and wait.** `TrainingSession.feed_block` is the feed loop's body, so a
  test is the loop. Sources take `realtime=False`. Beats, frames and states are values: push them in.
- **No test waits on the clock.** `conftest.waitfor(condition, what=...)` is the only sanctioned wait, for
  genuinely concurrent behaviour. `time.sleep` over 50 ms raises inside the suite.
- **Convert samples to counts in one pass** with `conftest.counts`, and feed device-sized blocks only where
  blocking is the subject: a gap, a held block, an index discontinuity.
- **A test that only reads a screen shares its build**: `harness.built_screen(module, code, channels, fed)`.
  A test that presses something builds its own.

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
- **Every generated sound goes through `nfb_console.audio`.** Render a short sound with `audio.finish` and
  play it with `audio.play`; stream a continuous one as an `audio.Stream`. `tests/test_audio.py` fails if
  another module opens a stream or starts `afplay`; a music file the operator chose is the one exception.
  Each rule there is a crackle that came back:
  - a step: a sound starts and ends at zero, a stop fades, a gain change ramps, a late block fades;
  - starvation: no callback, a writer thread fills a buffer the device drains without Python;
  - a clip: short sounds peak at -6.9 dBFS, and the sum bends below full scale;
  - the cone: nothing much under the 70 Hz speaker corner;
  - a device opened per sound, and a rate converted elsewhere.
- **The reference is `linked` by default**, one reference electrode serving all four channels, because
  cross-channel measures need a shared reference. The cost: one bad reference electrode makes every channel
  noisy at once, with no channel obviously at fault. The impedance test finds it.
- **A working channel reports signal state 0, not 1.** State 2 is "no signal"; 1 has never been observed on
  this hardware.
