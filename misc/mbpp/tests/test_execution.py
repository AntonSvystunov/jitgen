import pytest

from mbpp.execution import _extract_code_block


def test_extract_code_block_single_block() -> None:
    raw = "some prose\n```python\nprint(1)\n```\nmore prose"
    assert _extract_code_block(raw) == "\nprint(1)\n"


def test_extract_code_block_concatenates_multiple_blocks() -> None:
    raw = "```python\nprint(13)\n```\nsome commentary\n```python\nprint(1)\n```"
    assert _extract_code_block(raw) == "\nprint(13)\n\nprint(1)\n"


def test_extract_code_block_tolerates_missing_newline_after_fence_tag() -> None:
    raw = "```python print(1)\n```"
    assert _extract_code_block(raw) == " print(1)\n"


def test_extract_code_block_unterminated_block_uses_rest_of_response() -> None:
    raw = "```python\nprint(1)\nprint(2)"
    assert _extract_code_block(raw) == "\nprint(1)\nprint(2)"


def test_extract_code_block_unterminated_last_of_multiple_blocks() -> None:
    raw = "```python\nprint(1)\n```\n```python\nprint(2)"
    assert _extract_code_block(raw) == "\nprint(1)\n\nprint(2)"


def test_extract_code_block_raises_when_no_fence_present() -> None:
    with pytest.raises(ValueError, match="no ```python code block found"):
        _extract_code_block("just some prose, no code fence at all")
