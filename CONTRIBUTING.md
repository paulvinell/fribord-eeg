# Contributing to Fribord

Thanks for your interest. Fribord is meant to be built by a community, and contributions of all
sizes — bug reports, fixes, new protocols, hardware support, docs — are welcome.

## Before you start

- For anything larger than a small fix, open an issue or a discussion first so we can agree on the
  approach before you spend time on it.
- Keep pull requests focused: one logical change per PR is much easier to review and merge.

## Development setup

Fribord targets macOS and Python 3.12+, and uses [uv](https://docs.astral.sh/uv/).

```sh
uv sync            # install dependencies
make               # list every target
make test          # run the full suite (about a minute)
make lint          # the quick mechanical check
make shots         # render every screen to shots/ for a visual look
```

You do not need any hardware to develop: every live screen also runs on a built-in simulator, and the
tests drive the code directly without a device or a clock.

## What a good change looks like

- **Tests pass.** Run `make test` before you open the PR, and add tests for new behaviour. The suite is
  fast and deterministic by design — tests push values through the code rather than waiting on time.
- **On-screen text is plain and consistent.** Every string a user reads in the app follows Simplified
  Technical English; `tests/test_ste.py` enforces the mechanical half.
- **The three layers stay separate.** Acquisition owns the device, processing knows nothing about devices,
  and feedback (value → sound or display) lives only in the console layer. See `AGENTS.md`.

## Licensing of contributions (inbound = outbound)

Fribord is licensed under the **GNU General Public License v3.0** (see [`LICENSE`](LICENSE)). By submitting
a contribution, you agree that it is licensed under the same GPL-3.0 terms. This keeps the whole project —
including every fork — free and open: nobody can take Fribord or its improvements closed-source.

## Sign your commits (Developer Certificate of Origin)

Contributions are accepted under the [Developer Certificate of Origin](https://developercertificate.org/) —
a short statement that you wrote the change, or have the right to submit it under the project's license. You
certify it by adding a `Signed-off-by` line to each commit, which `git` adds for you:

```sh
git commit -s -m "Your message"
```

The line must use your real name and an email you can be reached at:

```
Signed-off-by: Jane Developer <jane@example.com>
```

No separate contributor-license-agreement paperwork is required — the sign-off is enough.

## Reporting bugs and asking questions

Open an issue for bugs (include your macOS version, the device, and steps to reproduce), and use
Discussions for questions and ideas. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).
