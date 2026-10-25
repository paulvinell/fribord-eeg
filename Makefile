SESSION ?= sessions/latest.jsonl.gz
SECONDS ?= 60
CHANNELS ?= 4
REFERENCE ?= linked
MAINS ?= 50
HZ      ?= 1000
HIGHPASS ?= 0.5
FILE    ?= $(SESSION)

.DEFAULT_GOAL := help
.PHONY: help setup console live play record sessions probe impedance capture test lint clean

help:  ## show this
	@echo "nfb — neurofeedback stack"
	@echo
	@grep -hE '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t22 | sed 's/^/  make /'
	@echo
	@echo "  variables: SESSION=$(SESSION) FILE=$(FILE) SECONDS=$(SECONDS) HZ=$(HZ) CHANNELS=$(CHANNELS) REFERENCE=$(REFERENCE) MAINS=$(MAINS) HIGHPASS=$(HIGHPASS)"

setup:  ## install everything (once)
	uv sync

console: $(SESSION)  ## open the console on a recording — no hardware needed
	uv run nfb-console $(SESSION)

.PHONY: shots
shots:  ## render every screen to shots/ so it can be looked at (SCREEN=PAC WIDTH=900 TAB=3 HELP="The run")
	uv run --group shots python tools/capture.py \
		$(if $(SCREEN),--screen $(SCREEN)) $(if $(STATE),--state $(STATE)) \
		$(if $(WIDTH),--width $(WIDTH)) $(if $(TAB),--tab $(TAB)) \
		$(if $(HELP),--help-title "$(HELP)")

live:  ## open the console on the live device
	uv run nfb-console --live --hz $(HZ) --reference $(REFERENCE) --mains $(MAINS) --high-pass $(HIGHPASS)

play:  ## open the console playing $(FILE) on loop — any sample export, no hardware
	uv run nfb-console --file "$(FILE)"

# Where a recording goes. Timestamped, and `sessions/latest.jsonl.gz` is a symlink to the newest — because
# the target used to be that fixed path, so every `make record` destroyed the previous capture. A recording
# is minutes of somebody's head that cannot be taken again. Passing SESSION= on the command line still names
# the file exactly, for when you do mean a particular one.
STAMP := $(shell date +%Y%m%d-%H%M%S)
RECORD_TO = $(if $(filter command line,$(origin SESSION)),$(SESSION),sessions/$(STAMP).jsonl.gz)

record:  ## record a session to sessions/<timestamp>.jsonl.gz (SESSION= to name it)
	@mkdir -p $(dir $(RECORD_TO))
	uv run nfb record $(RECORD_TO) --seconds $(SECONDS) --channels $(CHANNELS) --hz $(HZ) --reference $(REFERENCE) --mains $(MAINS) --high-pass $(HIGHPASS)
	@ln -sf $(notdir $(RECORD_TO)) sessions/latest.jsonl.gz
	@echo "recorded to $(RECORD_TO) — sessions/latest.jsonl.gz now points at it"

sessions:  ## list the recordings, newest first
	@ls -lht sessions/ 2>/dev/null || echo "no recordings yet"

probe:  ## identity, state and config from the device
	uv run optima probe

impedance:  ## electrode contact in kOhm
	uv run optima impedance --seconds 20

capture:  ## raw frame capture, for protocol work
	uv run optima capture --seconds 10

test:  ## run the test suite
	uv run pytest -q

latency:  ## measure the output half of the latency budget (silent, invisible — writes current values only)
	uv run python -m nfb_console.latency_bench --all

lint:  ## byte-compile everything as a smoke check
	uv run python -m compileall -q packages tests

clean:  ## remove caches
	find . -name __pycache__ -type d -prune -exec rm -rf {} + ; rm -rf .pytest_cache

# `make console` needs a recording; explain rather than fail obscurely.
$(SESSION):
	@echo "No recording at $(SESSION)."
	@echo "Plug the device in and run:  make record"
	@echo "Or point at an existing one: make console SESSION=path/to/file.jsonl.gz"
	@exit 1
