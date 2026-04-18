"""Tests for jitgen_langchain.python convenience factory."""

from jitgen.markers import MarkerStripper
from jitgen_core import Session
from jitgen_langchain.parser import JITGenParser
from jitgen_langchain.python import create_python_jitgen_parser


def test_create_python_jitgen_parser_returns_correct_type():
    parser = create_python_jitgen_parser()
    assert isinstance(parser, JITGenParser)


def test_create_python_jitgen_parser_has_session_and_stripper():
    parser = create_python_jitgen_parser()
    assert isinstance(parser.session, Session)
    assert isinstance(parser.stripper, MarkerStripper)
    assert parser.stripper.start == "```python"
    assert parser.stripper.end == "```"
