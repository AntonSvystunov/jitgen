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


# ── sealed trailing statements (no one-statement lag) ──────────────────────────

def test_trailing_simple_statement_released_at_its_newline(
    extractor: PythonLarkExtractor,
):
    """A newline-terminated simple statement must not wait for the next one.

    The N-1 rule alone would hold this back until a second statement started,
    which — since a generated script's ``print`` is its last line — defers all
    output past the end of generation.
    """
    stmts, leftover = extractor.extract("x = 1\n", final=False)
    assert stmts == ["x = 1"]
    assert leftover.strip() == ""


def test_trailing_print_released_before_final_flush(extractor: PythonLarkExtractor):
    stmts, _ = extractor.extract("a = 2\nprint(a)\n", final=False)
    assert "print(a)" in stmts


def test_statement_without_newline_is_not_released(extractor: PythonLarkExtractor):
    """``x = 1`` may still become ``x = 1 + 2``; only a real newline seals it."""
    stmts, leftover = extractor.extract("x = 1", final=False)
    assert stmts == []
    assert leftover == "x = 1"


def test_compound_statement_still_waits_for_the_next_statement(
    extractor: PythonLarkExtractor,
):
    """A ``for`` body can still grow, so the newline alone does not seal it."""
    stmts, _ = extractor.extract("for i in [1]:\n    print(i)\n", final=False)
    assert stmts == []

    # ...but once a dedented statement starts, the loop is provably closed.
    stmts, _ = extractor.extract("for i in [1]:\n    print(i)\nq", final=False)
    assert stmts == ["for i in [1]:\n    print(i)\n"]


def test_if_else_clause_not_split_across_extractions(extractor: PythonLarkExtractor):
    """``else`` must attach to its ``if`` — releasing the ``if`` early would
    execute the wrong branch and then fail on an orphaned ``else``."""
    stmts, _ = extractor.extract("if False:\n    x = 1\n", final=False)
    assert stmts == []
    stmts, _ = extractor.extract("if False:\n    x = 1\nelse:\n    x = 2\n", final=False)
    assert stmts == []


def test_decorated_function_not_split_from_its_decorator(
    extractor: PythonLarkExtractor,
):
    stmts, _ = extractor.extract("@staticmethod\n", final=False)
    assert stmts == []


# ── truncated tokens at the buffer tail ───────────────────────────────────────

def test_token_truncated_by_chunk_boundary_is_recoverable(
    extractor: PythonLarkExtractor,
):
    """``for p i`` is a half-delivered ``in``, not a syntax error.

    LALR's contextual lexer only offers terminals the parser state accepts, so
    the ``i`` is reported as an unexpected NAME.  Treating that as fatal aborts
    generation on perfectly valid code.
    """
    stmts, leftover = extractor.extract("for p i", final=False)
    assert stmts == []
    assert leftover == "for p i"


def test_truncated_token_resolves_once_the_rest_arrives(
    extractor: PythonLarkExtractor,
):
    stmts, _ = extractor.extract("for p in [1]:\n    print(p)\nq = 1\n", final=False)
    assert stmts[0] == "for p in [1]:\n    print(p)\n"


def test_error_with_input_after_it_still_raises(extractor: PythonLarkExtractor):
    """Only the *last* token can be truncated; anything followed by more input
    has been seen in full context and a rejection there is a real error."""
    with pytest.raises(SyntaxError):
        extractor.extract("x = ===\ny = 2\n", final=False)


def test_char_by_char_stream_yields_the_whole_program(
    extractor: PythonLarkExtractor,
):
    """Worst-case chunking must not lose statements or invent syntax errors."""
    program = "total = 0\nfor i in [1, 2, 3]:\n    total += i\n\nprint(total)\n"
    buffer = ""
    collected: list[str] = []
    for char in program:
        buffer += char
        stmts, buffer = extractor.extract(buffer, final=False)
        collected.extend(stmts)
    stmts, _ = extractor.extract(buffer, final=True)
    collected.extend(stmts)
    assert "".join(collected).replace("\n", "") == program.replace("\n", "")
