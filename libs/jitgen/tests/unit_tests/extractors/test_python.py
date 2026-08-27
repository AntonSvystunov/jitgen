import pytest
from lark import Lark

from jitgen.extractors.python import PythonLarkExtractor


class _UnsealedExtractor(PythonLarkExtractor):
    # Exercises `LarkStatementExtractor`'s fallback path: with no extensible
    # rules declared, sealing is disabled and the conservative N-1 rule alone
    # decides what's ready.
    @staticmethod
    def _extensible_rules() -> None:
        return None


def test_extract_releases_a_sealed_simple_statement_once_a_newline_follows(
    python_extractor: PythonLarkExtractor,
):
    statements, leftover = python_extractor.extract("x = 1\n", final=False)

    assert statements == ["x = 1"]
    assert leftover == "\n"


def test_extract_defers_a_compound_statement_until_a_new_statement_starts(
    python_extractor: PythonLarkExtractor,
):
    buffer = "if x:\n"
    statements, buffer = python_extractor.extract(buffer, final=False)
    assert statements == []  # still inside the `if` body

    buffer += "    y = 1\n"
    statements, buffer = python_extractor.extract(buffer, final=False)
    assert statements == []  # not sealed: no sibling statement has started yet

    buffer += "z = 2\n"
    statements, buffer = python_extractor.extract(buffer, final=False)

    # N-1: releasing "z = 2" proves the `if` block is what unblocked it.
    assert statements == ["if x:\n    y = 1\n", "z = 2"]
    assert buffer == "\n"


def test_extract_flushes_a_withheld_compound_statement_on_final(
    python_extractor: PythonLarkExtractor,
):
    buffer = "if x:\n    y = 1\n"
    statements, buffer = python_extractor.extract(buffer, final=False)
    assert statements == []

    statements, buffer = python_extractor.extract(buffer, final=True)

    assert statements == ["if x:\n    y = 1\n"]
    assert buffer == ""


@pytest.mark.parametrize(
    ("first_chunk", "second_chunk", "expected_statement"),
    [
        pytest.param("print(x", " + y)\n", "print(x + y)", id="open-paren"),
        pytest.param('x = "hello', ' world"\n', 'x = "hello world"', id="open-string"),
    ],
)
def test_extract_holds_back_input_ending_mid_token(
    python_extractor: PythonLarkExtractor,
    first_chunk: str,
    second_chunk: str,
    expected_statement: str,
):
    statements, buffer = python_extractor.extract(first_chunk, final=False)
    assert statements == []  # the token isn't finished yet, not a syntax error

    buffer += second_chunk
    statements, buffer = python_extractor.extract(buffer, final=False)

    assert statements == [expected_statement]


def test_extract_raises_syntax_error_for_invalid_syntax(
    python_extractor: PythonLarkExtractor,
):
    with pytest.raises(SyntaxError):
        python_extractor.extract("x = = 1\n", final=False)


def test_extract_raises_syntax_error_for_an_illegal_character(
    python_extractor: PythonLarkExtractor,
):
    with pytest.raises(SyntaxError):
        python_extractor.extract("x = 1 $\n", final=False)


def test_extract_raises_syntax_error_on_a_bad_dedent_at_final(
    python_extractor: PythonLarkExtractor,
):
    # Dedenting to a column that matches no open block is unconditionally
    # wrong, but recovery still defers to the final flush like every other
    # error class -- it never resolves early just because it's structural.
    buffer = "if x:\n    y = 1\n  z = 2\n"
    statements, buffer = python_extractor.extract(buffer, final=False)
    assert statements == []

    with pytest.raises(SyntaxError):
        python_extractor.extract(buffer, final=True)


def test_extract_raises_syntax_error_on_final_flush_of_incomplete_input(
    python_extractor: PythonLarkExtractor,
):
    buffer = "x = (1 + 2"
    statements, buffer = python_extractor.extract(buffer, final=False)
    assert statements == []  # still looks like it could be completed

    with pytest.raises(SyntaxError):
        python_extractor.extract(buffer, final=True)


@pytest.mark.parametrize("final", [False, True])
def test_extract_treats_blank_only_input_as_a_noop(
    python_extractor: PythonLarkExtractor, final: bool
):
    statements, leftover = python_extractor.extract("   \n", final=final)

    assert statements == []
    assert leftover == ("" if final else "   \n")


def test_extract_without_sealing_defers_even_a_simple_statement(
    python_lark_parser: Lark,
):
    extractor = _UnsealedExtractor(python_lark_parser)

    statements, buffer = extractor.extract("x = 1\n", final=False)
    assert statements == []  # only one statement so far: nothing to release yet

    buffer += "y = 2\n"
    statements, buffer = extractor.extract(buffer, final=False)

    assert statements == ["x = 1"]  # N-1: "y = 2" is still withheld
