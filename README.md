![Fribord](docs/banner.svg)

A from-scratch macOS driver and neurofeedback console for the **Neurobit Optima**, a 4-channel EEG
amplifier — with support for a few other biosignal devices alongside it. Written in Python, with a desktop
UI built on [Flet](https://flet.dev).

> **Independent project.** Fribord is not affiliated with, authorised by, or endorsed by the makers of any
> device it talks to. It is experimental software for personal research and self-experimentation — **not a
> medical device**. See [Safety](#safety) below.

## What it is

Fribord records from a biosignal device, processes the signal in real time, and gives the trainee
feedback — sound, music, or a visual display — driven by a chosen measure of their own physiology. It is a
single-operator console: the person running it is the person training.

## Supported devices

The wire protocol of each device was recovered for **interoperability**, so this software can talk to
hardware you own. No vendor code or binaries are part of this repository.

| Device | Signal | Connection |
|---|---|---|
| Neurobit Optima (4-channel) | EEG | USB (FTDI serial) |
| Mendi | HEG (hemoencephalography) | Bluetooth |
| HeartMath | heart rate / pulse | Bluetooth or USB |
| Tobii Eye Tracker 5 | pupil size | USB |

Every live screen also runs on a built-in **simulator**, so the UI and the training logic work with no
hardware attached.

## Training protocols

| Code | Screen | Trains |
|---|---|---|
| AMP | Amplitude | the amplitude of a frequency band, up or down |
| AT | Alpha-Theta | the alpha and theta bands, in the classic arrangement |
| SCP | Slow cortical potentials | trial-based up/down regulation of the slow potential |
| CFC | Cross-frequency coupling | coupling between two bands |
| PAC | Phase-amplitude coupling | a modulation index with a surrogate floor |
| ILF | ILF Synchrony | cross-channel synchrony at an infra-low frequency |
| WAVE | Waveform | the shape of a signed slow wave |
| SLEEP | Sleep | an overnight display, recording, and a tiered alarm |
| HEG | HEG | the Mendi blood-flow measure |
| HRV | HRV | heart-rate variability, with a breathing pacer |
| PUPIL | Pupil | voluntary control of pupil size, with passive and trial modes |

## Architecture

Two packages, three layers that stay separate:

- **`packages/nfb`** — acquisition (owns the device, emits stamped sample blocks), processing (works on
  blocks, knows nothing about devices), and the device drivers and DSP. No UI code.
- **`packages/console`** (`nfb-console`) — the Flet desktop UI: one screen per protocol, a shared feedback
  layer, and the device panel.

Feedback — the mapping from a measured value to sound or display — lives only in the console layer.

## Requirements

- macOS (the app uses CoreAudio and CoreGraphics)
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- The relevant hardware, for a live session (not needed for the simulator)

## Getting started

```sh
uv sync            # install dependencies
make               # list every target, with the variables each one takes
make test          # run the test suite (about a minute)
make console       # launch the UI
```

A live session needs the amplifier switched on — USB presence is not power, as the Optima runs on two AA
cells with a physical switch. `make probe` and `make impedance` check the connection and the electrodes.

## Development

- `make test` runs the whole suite. The tests drive the code directly and never wait on the clock, so they
  are fast and deterministic.
- `make lint` is the quick check for mechanical edits.
- `make shots` renders every screen to `shots/` for a visual look.

## How the device protocols were obtained

Each device protocol was recovered by observing the device and its official software — a lawful practice
for achieving interoperability with hardware you own. The committed code is an independent reimplementation
that speaks those protocols; it contains no third-party source, binaries, or documentation. Recorded
sessions, vendor material, and the notes from that work are kept out of this repository by design.

## Safety

This is experimental software for personal research and self-experimentation. It is **not** a medical
device and must not be used for diagnosis, treatment, or any clinical purpose. Biosignal hardware attaches
to your body and, in the case of the eye tracker, shines infrared light near your eyes — follow each
manufacturer's instructions and use it at your own risk.

## Contributing

Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, the pull-request flow, and
the sign-off requirement, and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## License

Fribord is free software, licensed under the **GNU General Public License v3.0** — see [`LICENSE`](LICENSE).
This keeps the project and every fork of it open: anyone may use, study, change, and share it, and any
distributed version must stay under the same license.

Copyright (C) 2026 Paul Vinell and contributors.
