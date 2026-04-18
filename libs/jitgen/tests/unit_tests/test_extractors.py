"""Tests for PythonLarkExtractor."""

import pytest
from lark import Lark
from lark.indenter import PythonIndenter

from jitgen.extractors.python import PythonLarkExtractor

_PARSER = Lark.open_from_package(
    "lark",
    "python.lark",
    ["grammars"],
    parser="lalr",
    postlex=PythonIndenter(),
    start="file_input",
    propagate_positions=True,
)


@pytest.fixture
def extractor():
    return PythonLarkExtractor(_PARSER)


def test_complete_statement_extracted(extractor: PythonLarkExtractor):
    stmts, leftover = extractor.extract("x = 1\n", final=False)
    # Only one statement; needs ≥2 children to confirm it's complete non-finally
    # (so stmts may be empty here)
    stmts_final, _ = extractor.extract("x = 1\n", final=True)
    assert "x = 1" in stmts_final[0]


def test_two_statements_first_extracted_nonfinal(extractor: PythonLarkExtractor):
    source = "x = 1\ny = 2\n"
    stmts, leftover = extractor.extract(source, final=False)
    assert len(stmts) >= 1
    assert "x = 1" in stmts[0]
    assert leftover  # y = 2 still in buffer


def test_incomplete_unclosed_string_is_recoverable(extractor: PythonLarkExtractor):
    stmts, leftover = extractor.extract('x = "hello', final=False)
    assert stmts == []
    assert leftover == 'x = "hello'


def test_incomplete_if_block_is_recoverable(extractor: PythonLarkExtractor):
    stmts, leftover = extractor.extract("if True:\n    pass", final=False)
    assert stmts == []


def test_unrecoverable_syntax_error_raises(extractor: PythonLarkExtractor):
    with pytest.raises(SyntaxError):
        extractor.extract("x = ===\n", final=False)


def test_final_flush_extracts_all(extractor: PythonLarkExtractor):
    stmts, leftover = extractor.extract("a = 1\nb = 2\n", final=True)
    assert len(stmts) == 2
    assert leftover == ""


def test_empty_whitespace_returns_empty(extractor: PythonLarkExtractor):
    stmts, leftover = extractor.extract("   \n  ", final=False)
    assert stmts == []


def test_multiline_function_recoverable_until_closed(extractor: PythonLarkExtractor):
    incomplete = "def foo():\n    x = 1\n"
    stmts, _ = extractor.extract(incomplete, final=False)
    assert stmts == []  # function body might not be complete yet

    complete = incomplete + "\n"
    stmts, _ = extractor.extract(complete, final=True)
    assert len(stmts) == 1
