"""Tests for MarkerStripper."""

import pytest

from jitgen.markers import CodeSegment, MarkerStripper


@pytest.fixture
def fence():
    return MarkerStripper(start="```python", end="```")


def test_no_marker_yields_nothing(fence: MarkerStripper):
    segs = fence.process("just plain text")
    assert segs == []


def test_single_block_in_one_chunk(fence: MarkerStripper):
    segs = fence.process("```python\nprint('hi')\n```")
    texts = [s.text for s in segs]
    assert "".join(texts) == "\nprint('hi')\n"
    assert segs[-1].end_of_block is True


def test_code_before_end_marker_not_end_of_block(fence: MarkerStripper):
    fence.process("```python\n")
    segs = fence.process("line1\nline2\n")
    assert all(not s.end_of_block for s in segs)


def test_end_of_block_set_on_closing_marker():
    stripper = MarkerStripper(start="<code>", end="</code>")
    stripper.process("<code>body")
    segs = stripper.process("</code>")
    assert segs[-1].end_of_block is True


def test_marker_split_across_chunks():
    """Start marker split across two chunks must still be detected."""
    stripper = MarkerStripper(start="```python", end="```")
    stripper.process("```pyt")
    segs = stripper.process("hon\nprint(1)\n```")
    texts = "".join(s.text for s in segs)
    assert "print(1)" in texts
    assert segs[-1].end_of_block is True


def test_end_marker_split_across_chunks():
    stripper = MarkerStripper(start="```python", end="```")
    stripper.process("```python\ncode\n``")
    segs = stripper.process("`")
    assert segs[-1].end_of_block is True


def test_reset_clears_state(fence: MarkerStripper):
    fence.process("```python\nsome code")
    assert fence.inside_markers is True
    fence.reset()
    assert fence.inside_markers is False
    # After reset, start marker restarts detection
    segs = fence.process("```python\nx = 1\n```")
    assert any(s.end_of_block for s in segs)


def test_multiple_blocks_in_one_chunk(fence: MarkerStripper):
    raw = "```python\na=1\n```ignore```python\nb=2\n```"
    segs = fence.process(raw)
    end_of_blocks = [s for s in segs if s.end_of_block]
    assert len(end_of_blocks) == 2


def test_json_framing_markers():
    stripper = MarkerStripper(start='{"code":"', end='"}')
    segs = stripper.process('{"code":"print(1)"}')
    texts = "".join(s.text for s in segs)
    assert texts == "print(1)"
    assert segs[-1].end_of_block is True


def test_finalize_flushes_partial_marker_prefix():
    """A partial marker prefix retained in buffer must be flushed by finalize()."""
    stripper = MarkerStripper(start="<code>", end="</code>")
    # Feed text that ends mid-start-marker so process() retains the partial prefix.
    segs_process = stripper.process("<co")  # "<co" is a prefix of "<code>"
    assert segs_process == []  # held back waiting for marker completion
    assert not stripper.inside_markers
    # finalize() releases whatever's buffered
    segs_final = stripper.finalize()
    # The partial "<co" was outside markers — finalize discards it (no code segment)
    assert all(not s.end_of_block for s in segs_final)


def test_overlap_suffix_prefix_edge_case():
    stripper = MarkerStripper(start="STARTMARKER", end="ENDMARKER")
    # Feed start marker split across chunks
    stripper.process("prefix_START")
    segs = stripper.process("MARKER\ncode\nENDMARKER")
    assert any(s.end_of_block for s in segs)
