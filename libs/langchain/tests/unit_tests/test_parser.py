"""Tests for jitgen.langchain.parser module."""

import pytest
from collections.abc import Iterator
from langchain_core.messages import HumanMessage

from jitgen_langchain.parser import JITGenParser
from jitgen.prebuilt.python import create_python_jitgen


@pytest.fixture
def parser():
    """Create a JITGenParser instance."""
    return JITGenParser(
        jit_gen=create_python_jitgen(),
        start_marker="```python",
        end_marker="```",
    )


def test_parser_type(parser: JITGenParser):
    """Test that parser has correct type."""
    assert parser._type == "jitgen_parser"


def test_parser_parse(parser: JITGenParser):
    """Test the parse method (should return text as-is)."""
    text = "some text"
    result = parser.parse(text)
    assert result == text


def test_yield_code_blocks_string_input(parser: JITGenParser):
    """Test _yield_code_blocks with string input."""

    def input_stream() -> Iterator[str]:
        yield "```python\n"
        yield "print('Hello')\n"
        yield "print('World')\n"
        yield "```\n"

    blocks = list(parser._yield_code_blocks(input_stream()))
    assert len(blocks) == 2
    assert blocks[0] == "print('Hello')\n"
    assert blocks[1] == "print('World')\n"


def test_yield_code_blocks_message_input(parser: JITGenParser):
    """Test _yield_code_blocks with BaseMessage input."""

    def input_stream() -> Iterator[HumanMessage]:
        yield HumanMessage(content="```python\n")
        yield HumanMessage(content="print('Hello')\n")
        yield HumanMessage(content="```\n")

    blocks = list(parser._yield_code_blocks(input_stream()))
    assert len(blocks) == 1
    assert blocks[0] == "print('Hello')\n"


def test_yield_code_blocks_mixed_input(parser: JITGenParser):
    """Test _yield_code_blocks with mixed string and message input."""

    def input_stream() -> Iterator[str | HumanMessage]:
        yield "```python\n"
        yield HumanMessage(content="print('Hello')\n")
        yield "print('World')\n"
        yield "```\n"

    blocks = list(parser._yield_code_blocks(input_stream()))
    assert len(blocks) == 2
    assert blocks[0] == "print('Hello')\n"
    assert blocks[1] == "print('World')\n"


def test_yield_code_blocks_incomplete_lines(parser: JITGenParser):
    """Test _yield_code_blocks with incomplete lines."""

    def input_stream() -> Iterator[str]:
        yield "```python\n"
        yield "print('Hel"
        yield "lo')\n"
        yield "```\n"

    blocks = list(parser._yield_code_blocks(input_stream()))
    assert len(blocks) == 1
    assert blocks[0] == "print('Hello')\n"


def test_yield_code_blocks_no_start_marker(parser: JITGenParser):
    """Test _yield_code_blocks when start marker is not found."""

    def input_stream() -> Iterator[str]:
        yield "print('Hello')\n"
        yield "print('World')\n"

    blocks = list(parser._yield_code_blocks(input_stream()))
    assert len(blocks) == 0


def test_yield_code_blocks_custom_markers():
    """Test _yield_code_blocks with custom start/end markers."""
    custom_parser = JITGenParser(
        jit_gen=create_python_jitgen(),
        start_marker="```javascript",
        end_marker="```",
    )

    def input_stream() -> Iterator[str]:
        yield "```javascript\n"
        yield "console.log('Hello')\n"
        yield "```\n"

    blocks = list(custom_parser._yield_code_blocks(input_stream()))
    assert len(blocks) == 1
    assert blocks[0] == "console.log('Hello')\n"


def test_transform_sync(parser: JITGenParser):
    """Test synchronous transform method."""

    def input_stream() -> Iterator[str]:
        yield "```python\n"
        yield "print('Hello')\n"
        yield "```\n"

    results = list(parser.transform(input_stream()))
    assert len(results) == 1
    assert results[0] == "Hello\n"


def test_transform_with_multiple_statements(parser: JITGenParser):
    """Test transform with multiple statements."""

    def input_stream() -> Iterator[str]:
        yield "```python\n"
        yield "print('First')\n"
        yield "print('Second')\n"
        yield "```\n"

    results = list(parser.transform(input_stream()))
    assert len(results) == 2
    assert results[0] == "First\n"
    assert results[1] == "Second\n"


@pytest.mark.asyncio
async def test_ayield_code_blocks_string_input(parser: JITGenParser):
    """Test async _ayield_code_blocks with string input."""

    async def input_stream():
        yield "```python\n"
        yield "print('Hello')\n"
        yield "print('World')\n"
        yield "```\n"

    blocks = []
    async for block in parser._ayield_code_blocks(input_stream()):
        blocks.append(block)

    assert len(blocks) == 2
    assert blocks[0] == "print('Hello')\n"
    assert blocks[1] == "print('World')\n"


@pytest.mark.asyncio
async def test_ayield_code_blocks_message_input(parser: JITGenParser):
    """Test async _ayield_code_blocks with BaseMessage input."""

    async def input_stream():
        yield HumanMessage(content="```python\n")
        yield HumanMessage(content="print('Hello')\n")
        yield HumanMessage(content="```\n")

    blocks = []
    async for block in parser._ayield_code_blocks(input_stream()):
        blocks.append(block)

    assert len(blocks) == 1
    assert blocks[0] == "print('Hello')\n"


@pytest.mark.asyncio
async def test_atransform_async(parser: JITGenParser):
    """Test asynchronous _atransform method."""

    async def input_stream():
        yield "```python\n"
        yield "print('Hello')\n"
        yield "```\n"

    results = []
    async for result in parser._atransform(input_stream()):
        results.append(result)

    assert len(results) == 1
    assert results[0] == "Hello\n"


@pytest.mark.asyncio
async def test_atransform_with_multiple_statements(parser: JITGenParser):
    """Test async transform with multiple statements."""

    async def input_stream():
        yield "```python\n"
        yield "print('First')\n"
        yield "print('Second')\n"
        yield "```\n"

    results = []
    async for result in parser._atransform(input_stream()):
        results.append(result)

    assert len(results) == 2
    assert results[0] == "First\n"
    assert results[1] == "Second\n"


def test_yield_code_blocks_first_line_stripping(parser: JITGenParser):
    """Test that first line is stripped correctly."""

    def input_stream() -> Iterator[str]:
        yield "  ```python  \n"  # Should be stripped
        yield "print('Hello')\n"
        yield "```\n"

    blocks = list(parser._yield_code_blocks(input_stream()))
    assert len(blocks) == 1
    assert blocks[0] == "print('Hello')\n"


@pytest.mark.asyncio
async def test_ayield_code_blocks_first_line_stripping(parser: JITGenParser):
    """Test that first line is stripped correctly in async version."""

    async def input_stream():
        yield "  ```python  \n"  # Should be stripped
        yield "print('Hello')\n"
        yield "```\n"

    blocks = []
    async for block in parser._ayield_code_blocks(input_stream()):
        blocks.append(block)

    assert len(blocks) == 1
    assert blocks[0] == "print('Hello')\n"
