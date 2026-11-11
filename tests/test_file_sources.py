"""Playing from files: a text export of samples, and a frame recording on loop.

Both exist so a protocol screen can be worked on without wiring up a head, and both have to satisfy
the same contract as the live device — most importantly that per-channel sample indices are absolute
and monotonic, including across a loop seam. An index that resets would silently corrupt every
windowed metric downstream.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from nfb import Block, Gap, ReplaySource, SampleFileSource
from nfb_console.inputs import InputChoice, is_frame_recording

from conftest import BENCH

SAMPLE_EXPORT = Path("/Users/paul/personal/emotional-n-back/data/eeg/Amplitude.recording")
FRAMES = BENCH
"""The generated bench recording rather than `sessions/latest.jsonl.gz`, which is gitignored — so the frame
replay test was skipped on any clone that had not recorded one, which is every fresh one."""


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "samples.txt"
    p.write_text(text)
    return p


def test_reads_the_rate_from_the_header_and_the_samples_as_microvolts(tmp_path):
    path = write(tmp_path, "Sampling rate: 256\n1.5\t-2.25\n3.0\t-4.5\n")
    src = SampleFileSource(path, loop=False, realtime=False)
    assert src.rate_hz == 256.0
    assert src.samples == 2
    assert src.seconds == pytest.approx(2 / 256)
    blocks = [b for b in src.blocks() if isinstance(b, Block)]
    assert [round(v, 4) for v in blocks[0].values] == [1.5, 3.0]
    assert [round(v, 4) for v in blocks[1].values] == [-2.25, -4.5]


def test_an_explicit_rate_is_required_when_there_is_no_header(tmp_path):
    path = write(tmp_path, "1.0\t2.0\n")
    with pytest.raises(ValueError, match="rate_hz"):
        SampleFileSource(path, loop=False, realtime=False)
    assert SampleFileSource(path, rate_hz=250, loop=False, realtime=False).rate_hz == 250


def test_a_file_with_no_samples_is_an_error_not_an_empty_stream(tmp_path):
    path = write(tmp_path, "Sampling rate: 256\n")
    with pytest.raises(ValueError, match="no sample rows"):
        SampleFileSource(path, loop=False, realtime=False)


def test_an_all_zero_column_is_reported_inactive(tmp_path):
    """Exports pad unused channels with zeros, and a flat line is otherwise indistinguishable from a
    working channel on a dead subject."""
    path = write(tmp_path, "Sampling rate: 100\n1.0\t0\n2.0\t0\n")
    src = SampleFileSource(path, loop=False, realtime=False)
    assert src.active == [True, False]
    by_channel = {b.channel: b for b in src.blocks() if isinstance(b, Block)}
    assert by_channel[0].valid and not by_channel[1].valid


def test_ragged_rows_are_padded_rather_than_shifting_channels(tmp_path):
    path = write(tmp_path, "Sampling rate: 100\n1.0\t2.0\n3.0\n")
    src = SampleFileSource(path, loop=False, realtime=False)
    got = {b.channel: [round(v, 4) for v in b.values] for b in src.blocks()
           if isinstance(b, Block)}
    assert got == {0: [1.0, 3.0], 1: [2.0, 0.0]}


def test_a_missing_leading_cell_is_an_error_not_a_channel_shift(tmp_path):
    """Stripping the whole line ate the empty first field, so ch1's value slid into ch0 and the row read as
    one channel short. The leading tab is kept now, so the empty cell is seen and the row is rejected."""
    path = write(tmp_path, "Sampling rate: 100\n1.0\t2.0\n\t5.0\n")
    with pytest.raises(ValueError, match="non-numeric cell"):
        SampleFileSource(path, loop=False, realtime=False)


def test_a_corrupt_sample_cell_is_an_error_not_a_dropped_row(tmp_path):
    """A data row with one bad cell was dropped as if it were prose, so later samples took earlier times with
    no gap — time silently compressed. A multi-column row that does not parse is now a loud error."""
    path = write(tmp_path, "Sampling rate: 100\n1.0\t2.0\n3.0\tbad\n")
    with pytest.raises(ValueError, match="non-numeric cell"):
        SampleFileSource(path, loop=False, realtime=False)


def test_a_low_rate_text_file_keeps_its_block_within_a_metric_stride(tmp_path):
    """A fixed 8-sample block at a low rate spans more than one 20 Hz decimation stride, so the feed's
    one-reading-per-block step dropped readings and calibration depended on input packaging. The block is
    capped to a device frame's span, so at a low rate it shrinks below the stride."""
    rows = "".join(f"{i}.0\n" for i in range(40))
    low = SampleFileSource(write(tmp_path, "Sampling rate: 100\n" + rows), realtime=False)
    assert low.block_size <= round(100 / 20.0), "a 100 Hz block must stay within one metric stride"
    high = SampleFileSource(write(tmp_path, "Sampling rate: 250\n" + rows), realtime=False)
    assert high.block_size == 8, "at the amplifier's own rate the block is a full device frame"


def test_indices_stay_monotonic_across_a_loop_seam(tmp_path):
    """A loop is not a dropped sample, so no Gap — but the index must never rewind."""
    path = write(tmp_path, "Sampling rate: 100\n" + "".join(f"{i}.0\n" for i in range(20)))
    src = SampleFileSource(path, loop=True, realtime=False, block_size=8)
    items = list(itertools.islice(src.blocks(), 12))
    assert not any(isinstance(b, Gap) for b in items)
    starts = [b.start_index for b in items]
    assert starts == sorted(starts) and starts[-1] > 20, "must pass the end of the file"
    assert all(b.end_index == starts[i + 1] for i, b in enumerate(items[:-1])), "contiguous"


def test_not_looping_stops_at_the_end(tmp_path):
    path = write(tmp_path, "Sampling rate: 100\n" + "".join(f"{i}.0\n" for i in range(20)))
    src = SampleFileSource(path, loop=False, realtime=False, block_size=8)
    blocks = list(src.blocks())
    assert sum(len(b.samples) for b in blocks) == 20


def test_closing_ends_an_endless_loop(tmp_path):
    """Without this, closing a looping source would hang whatever was draining it."""
    path = write(tmp_path, "Sampling rate: 100\n" + "".join(f"{i}.0\n" for i in range(20)))
    src = SampleFileSource(path, loop=True, realtime=False)
    stream = src.blocks()
    next(stream)
    src.close()
    rest = list(stream)                      # terminates, rather than looping forever
    assert len(rest) < 4, "at most the remainder of the group in flight"


@pytest.mark.skipif(not FRAMES.exists(), reason="no recording to replay")
def test_a_looped_frame_recording_keeps_counting_without_a_seam():
    """A loop is a splice, not a hole, and the indices have to say so.

    The seam used to be reported as a Gap, argued as "what actually happened to the signal" — but a Gap
    means *lost samples*, and a replay starting its file over has lost nothing. The difference is
    expensive: every filter downstream resets on a gap and re-runs its settling time once per loop, so a
    short recording looped under a slow protocol never settled at all. The step the signal itself takes
    at the seam is a property of the data and rides through the filters like any other transient;
    `SampleFileSource` has documented the same contract for itself all along.
    """
    src = ReplaySource(FRAMES, loop=True)
    items = list(itertools.islice(src.blocks(), 20000))
    blocks0 = [b for b in items if isinstance(b, Block) and b.channel == 0]
    assert not any(isinstance(b, Gap) for b in items), "a loop is a splice, not a hole"
    for prev, nxt in zip(blocks0, blocks0[1:]):
        assert nxt.start_index == prev.end_index, "contiguous across the seam, no jump"
    assert blocks0[0].start_index == 0
    assert blocks0[-1].start_index > 30000, "must have played past the end of the file at least once"


@pytest.mark.skipif(not SAMPLE_EXPORT.exists(), reason="no sample export on this machine")
def test_the_real_export_reads_as_two_live_channels_at_256_hz():
    src = SampleFileSource(SAMPLE_EXPORT, loop=False, realtime=False)
    assert src.rate_hz == 256.0
    assert len(src.channels()) == 4
    assert src.active == [True, True, False, False]
    assert src.seconds == pytest.approx(74.4, abs=0.1)


def test_input_choice_dispatches_on_extension():
    assert is_frame_recording("a/b.jsonl.gz") and is_frame_recording("a/b.jsonl")
    assert not is_frame_recording("Amplitude.recording")
    assert "recorded frames" in InputChoice("file", "x.jsonl.gz").describe()
    assert "text samples" in InputChoice("file", "x.recording").describe()


def test_input_choice_reports_a_missing_file_before_a_session_starts(tmp_path):
    assert InputChoice().problem() is None
    assert "no such file" in InputChoice("file", str(tmp_path / "nope")).problem()
    assert "directory" in InputChoice("file", str(tmp_path)).problem()
    assert InputChoice("file", "").problem() is None, "an empty path is not yet a file choice"


def test_an_empty_path_falls_back_to_the_amplifier():
    assert not InputChoice("file", "").from_file
    with pytest.raises(RuntimeError, match="no device"):
        InputChoice("file", "").open(device=None)


def test_a_notes_file_stands_for_the_recording_beside_it(tmp_path):
    from nfb.notes import notes_path
    from nfb_console.inputs import recording_of

    recording = tmp_path / "x.jsonl.gz"
    recording.write_bytes(b"")
    assert recording_of(notes_path(recording)) == recording
    assert recording_of(recording) == recording
    alone = tmp_path / "y.jsonl.gz.notes.jsonl"
    assert recording_of(alone) == alone, "with no recording beside it, the choice stays what was chosen"


def test_input_choice_reports_a_file_that_is_not_a_recording_before_a_session_starts(tmp_path):
    notes = tmp_path / "y.jsonl.gz.notes.jsonl"
    notes.write_text('{"format": "nfb-notes/1"}\n')
    assert "not a recording" in InputChoice("file", str(notes)).problem()
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert "not a recording" in InputChoice("file", str(empty)).problem()
    assert InputChoice("file", str(FRAMES)).problem() is None
