"""What a source says about its channels and its seams, before any block is judged.

Two facts the rest of the stack must not guess. The input range is a property of the amplifier, so a
source that cannot know it says nothing, and no rule may apply a limit it was not given. A looped source
splices its end to its start, and the first block after that splice says so, because the signal steps
there while the sample indices run on.
"""

from __future__ import annotations

import pytest

from nfb import ReplaySource
from nfb.sources import SampleFileSource
from nfb.stream import Block

from conftest import BENCH_OK


def test_a_device_channel_states_its_input_range():
    import optima

    for spec in ReplaySource(str(BENCH_OK)).channels().values():
        assert spec.input_range == pytest.approx(abs(spec.scale) * optima.ADC_FULL_SCALE)


def test_a_sample_file_states_no_input_range(tmp_path):
    path = tmp_path / "samples.txt"
    path.write_text("Sampling rate: 250\n" + "".join(f"{i % 7}\t{-i % 5}\n" for i in range(40)))
    specs = SampleFileSource(path, realtime=False).channels()
    assert specs and all(spec.input_range is None for spec in specs.values())


def test_a_looped_sample_file_marks_its_seam(tmp_path):
    path = tmp_path / "samples.txt"
    path.write_text("Sampling rate: 250\n" + "".join(f"{i}\n" for i in range(1, 41)))
    seams = []
    for block in SampleFileSource(path, realtime=False, loop=True, block_size=8).blocks():
        if block.start_index >= 120:
            break
        seams.append((block.start_index, block.seam))
    assert [i for i, seam in seams if seam] == [40, 80], "the first block after each splice, and no other"


def test_a_looped_replay_marks_its_seam():
    source = ReplaySource(str(BENCH_OK), realtime=False, loop=True)
    length: dict[int, int] = {}
    first: dict[int, int] = {}
    seams: list[tuple[int, int]] = []
    try:
        for block in source.blocks():
            if not isinstance(block, Block):
                continue
            if block.seam:
                seams.append((block.channel, block.start_index))
            if block.channel in first and block.start_index < first[block.channel]:
                pytest.fail("a looped replay keeps its indices monotonic")
            first.setdefault(block.channel, block.start_index)
            length[block.channel] = block.end_index
            if all(end > 2600 for end in length.values()) and len(length) == len(source.channels()):
                break
    finally:
        source.close()
    assert sorted(ch for ch, _ in seams) == sorted(source.channels()), "one seam per channel"
    assert all(1900 <= index <= 2100 for _, index in seams), "at the end of the 8 s bench file"


@pytest.mark.parametrize("range_uv", [800.0, 6000.0, 24000.0])
def test_each_input_range_builds_its_code_and_decodes_to_its_own_scale(range_uv):
    import optima
    from nfb.decode import FrameDecoder

    table, _unit = optima.RANGE_BY_FUNCTION[optima.FUNC_VOLTAGE]
    blob = optima.build_eeg_config(4, range_uv=range_uv)
    for ch in range(4):
        code = optima.get_field(blob, optima.CHANNEL_FIELDS["measurement_range"], ch)
        assert table[code] == range_uv
    for corner in (0.5, 0.01):
        assert optima.validate_config(optima.build_eeg_config(4, range_uv=range_uv,
                                                              lower_limit_hz=corner)) == []
    for spec in FrameDecoder(blob).channels().values():
        assert spec.input_range == pytest.approx(range_uv)
        assert spec.scale == pytest.approx(range_uv / optima.ADC_FULL_SCALE)


def test_the_input_limit_follows_a_wide_range():
    """At ±6 mV a sample at 98% of the range is at the limit, and 900 µV, over the narrow range, is not."""
    import optima
    from nfb.decode import FrameDecoder
    from nfb.dsp.artifact import input_limit_mask

    spec = FrameDecoder(optima.build_eeg_config(4, range_uv=6000.0)).channels()[0]

    def at_limit(uv: float) -> bool:
        block = Block(channel=0, start_index=0, samples=(round(uv / spec.scale),), spec=spec, state=0)
        mask = input_limit_mask(block)
        return mask is not None and bool(mask[0])

    assert at_limit(0.98 * 6000.0) and at_limit(-0.99 * 6000.0)
    assert not at_limit(900.0) and not at_limit(-900.0)
