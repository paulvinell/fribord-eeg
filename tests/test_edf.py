"""EDF export: a header other tools can read, and samples landing where they belong in time.

The two things worth pinning are the ones an EDF writer usually gets wrong — channels at different
rates sharing one record grid, and a dropped frame leaving a hole at the right instant rather than
shifting everything after it earlier.
"""

from __future__ import annotations

import numpy as np
import pytest

from nfb.edf import EDFWriter, read_header, read_samples, record_seconds
from nfb.stream import Block, ChannelSpec


def specs(*rates: float) -> dict[int, ChannelSpec]:
    return {i: ChannelSpec(index=i, rate_hz=rate, unit="uV", scale=0.0244140625)
            for i, rate in enumerate(rates)}


def feed_seconds(writer: EDFWriter, spec_map, seconds: float, value=lambda c, i: 0) -> None:
    """Blocks in the device's own shape: eight samples at a time, one channel after another."""
    for channel, spec in spec_map.items():
        n = int(seconds * spec.rate_hz)
        for start in range(0, n, 8):
            samples = tuple(int(value(channel, i)) for i in range(start, min(start + 8, n)))
            writer.feed(Block(channel, start, samples, spec))


def test_the_header_says_what_every_reader_needs(tmp_path):
    path = tmp_path / "night.edf"
    spec_map = specs(250.0, 250.0)
    with EDFWriter(path, spec_map, prefilter="HP:0.5Hz N:50Hz") as writer:
        feed_seconds(writer, spec_map, 4.0)
    head = read_header(path)
    assert head["version"] == "0"
    assert head["signals"] == 2
    assert head["labels"] == ["EEG ch0", "EEG ch1"]
    assert head["units"] == ["uV", "uV"]
    assert head["record_seconds"] == 1.0
    assert head["samples_per_record"] == [250, 250]
    assert head["records"] == 4, "patched at close, not left at -1"
    assert head["header_bytes"] == 256 * 3
    assert head["digital_min"] == [-32768, -32768] and head["digital_max"] == [32767, 32767]
    assert head["physical_max"][0] == pytest.approx(32767 * 0.0244140625, rel=1e-6), \
        "counts written unchanged; the microvolts live in the physical scaling"
    assert head["prefilter"][0] == "HP:0.5Hz N:50Hz", "the corner and the notch travel with the file"


def test_a_decimated_channel_shares_the_record_grid(tmp_path):
    """62.5 sps is the device's deepest decimation and has no whole number of samples in a second, so
    the record duration has to stretch until every rate is integral."""
    assert record_seconds([250.0, 125.0]) == 1
    assert record_seconds([250.0, 62.5]) == 2

    path = tmp_path / "mixed.edf"
    spec_map = specs(250.0, 62.5)
    with EDFWriter(path, spec_map) as writer:
        feed_seconds(writer, spec_map, 8.0)
    head = read_header(path)
    assert head["record_seconds"] == 2.0
    assert head["samples_per_record"] == [500, 125], "a 4:1 rate ratio, in one grid"
    assert head["records"] == 4
    got = read_samples(path)
    assert len(got[0]) == 2000 and len(got[1]) == 500


def test_a_dropped_frame_becomes_a_hole_where_it_happened(tmp_path):
    """The reason samples are placed by absolute index rather than by arrival: a missing block must
    not pull everything after it earlier in time, which is what silently corrupts a night."""
    path = tmp_path / "gap.edf"
    spec_map = specs(250.0)
    spec = spec_map[0]
    with EDFWriter(path, spec_map) as writer:
        for start in range(0, 1000, 8):
            if 240 <= start < 320:                   # eighty samples never arrive
                continue
            writer.feed(Block(0, start, tuple([1000] * 8), spec))
    got = read_samples(path)[0]
    assert len(got) >= 750
    assert got[100] == 1000
    assert set(got[240:320]) == {0}, "the hole is zero-filled, at the instant it happened"
    assert got[400] == 1000, "and everything after it is still where it belongs"


def test_the_samples_survive_the_round_trip_unchanged(tmp_path):
    path = tmp_path / "wave.edf"
    spec_map = specs(250.0)
    times = np.arange(1000) / 250.0
    wave = (8000 * np.sin(2 * np.pi * 10.0 * times)).astype(int)
    with EDFWriter(path, spec_map) as writer:
        feed_seconds(writer, spec_map, 4.0, value=lambda _c, i: wave[i])
    got = read_samples(path)[0]
    assert np.array_equal(got, wave), "16-bit counts in, the same counts out — no requantisation"


def test_an_interrupted_file_is_still_readable(tmp_path):
    """A crash leaves the record count at -1, which readers take as "ongoing" — the data up to the
    last whole record is intact, which is the whole reason the count is written that way."""
    path = tmp_path / "cut.edf"
    spec_map = specs(250.0)
    writer = EDFWriter(path, spec_map)
    feed_seconds(writer, spec_map, 3.0)
    writer._fh.flush()                               # no close(): the file as a crash would leave it
    head = read_header(path)
    assert head["records"] == -1
    assert len(read_samples(path)[0]) == 750
    writer.close()
    assert read_header(path)["records"] == 3
