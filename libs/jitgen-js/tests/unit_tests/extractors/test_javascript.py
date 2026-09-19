import pytest

from jitgen_js.extractors.javascript import JavaScriptAntlrExtractor


def test_extract_withholds_a_truncated_buffer_that_would_otherwise_parse_clean(
    extractor: JavaScriptAntlrExtractor,
):
    # The regression test for the extractor's central design fix: ANTLR's
    # `eos` rule accepts a synthetic EOF as a valid statement terminator, so
    # `"x = 1"` alone parses as one complete, error-free statement even
    # though the model may still be about to emit `+ 2`. If `_is_sealed`
    # were ever implemented by trusting "the parse succeeded" instead of
    # checking for a genuine trailing terminator in the raw text, this is
    # the test that fails immediately.
    statements, leftover = extractor.extract("x = 1", final=False)
    assert statements == []
    assert leftover == "x = 1"


def test_extract_flushes_the_withheld_statement_on_final(
    extractor: JavaScriptAntlrExtractor,
):
    statements, leftover = extractor.extract("x = 1", final=True)
    assert statements == ["x = 1"]
    assert leftover == ""


def test_extract_releases_a_sealed_statement_terminated_by_a_semicolon(
    extractor: JavaScriptAntlrExtractor,
):
    statements, leftover = extractor.extract("x = 1;\ny = 2;\n", final=False)
    assert statements == ["x = 1;", "y = 2;"]
    assert leftover == "\n"


def test_extract_releases_a_sealed_statement_terminated_by_asi_newline(
    extractor: JavaScriptAntlrExtractor,
):
    # No explicit semicolon — sealed via the bare-newline branch of the text
    # check instead, exercising a different path than the semicolon case.
    statements, leftover = extractor.extract("x = 1\ny = 2\n", final=False)
    assert statements == ["x = 1", "y = 2"]
    assert leftover == "\n"


def test_extract_defers_an_if_statement_until_a_sibling_starts(
    extractor: JavaScriptAntlrExtractor,
):
    # `if (x) { y(); }` alone could still grow a trailing `else` clause.
    statements, leftover = extractor.extract("if (x) { y(); }", final=False)
    assert statements == []
    assert leftover == "if (x) { y(); }"

    statements, leftover = extractor.extract("if (x) { y(); }\nz();\n", final=False)
    assert statements == ["if (x) { y(); }", "z();"]
    assert leftover == "\n"


def test_extract_flushes_a_withheld_if_statement_on_final(
    extractor: JavaScriptAntlrExtractor,
):
    statements, leftover = extractor.extract("if (x) { y(); }", final=True)
    assert statements == ["if (x) { y(); }"]
    assert leftover == ""

    statements, leftover = extractor.extract(
        "if (x) { y(); } else { z(); }", final=True
    )
    assert statements == ["if (x) { y(); } else { z(); }"]
    assert leftover == ""


def test_extract_defers_a_truncated_keyword_that_could_still_attach_backward(
    extractor: JavaScriptAntlrExtractor,
):
    # Regression test for the N-1-breaking bug found via chunked streaming
    # (see `_is_ambiguous_tail_fragment`'s docstring): a keyword truncated
    # mid-word lexes as a plain identifier and parses as its own trailing
    # statement, which must not be trusted to mark a real boundary — the
    # extensible statement right before it might still absorb it once the
    # keyword finishes arriving (`if`/`else`, `try`/`finally`).
    for src in (
        "if (x) { y(); } el",
        "try { a(); } catch (e) { b(); } fin",
    ):
        statements, leftover = extractor.extract(src, final=False)
        assert statements == []
        assert leftover == src


def test_extract_defers_a_try_finally_without_catch(
    extractor: JavaScriptAntlrExtractor,
):
    # Direct structural analogue of the `try_finally` bug that shipped in
    # `PythonLarkExtractor`: a `try { } finally { }` (no `catch`) could still
    # grow nothing further in this grammar (finally is already present), but
    # a `try { } catch (e) { }` could still grow a trailing `finally` — both
    # forms share the same `TryStatementContext` class, so both must be
    # withheld the same way.
    statements, leftover = extractor.extract(
        "try { a(); } finally { b(); }\nc();\n", final=False
    )
    assert statements == ["try { a(); } finally { b(); }", "c();"]
    assert leftover == "\n"

    statements, leftover = extractor.extract(
        "try { a(); } catch (e) { b(); }\nc();\n", final=False
    )
    assert statements == ["try { a(); } catch (e) { b(); }", "c();"]
    assert leftover == "\n"


def test_extract_seals_do_while_but_defers_plain_while(
    extractor: JavaScriptAntlrExtractor,
):
    # `DoStatementContext` ends in plain `eos`, not an optional trailing
    # clause — it should seal via the text check alone, immediately after
    # its own terminator. `WhileStatementContext` has no such terminator to
    # check and is always deferred via N-1 regardless of what follows. This
    # is the test that would catch classifying by rule-name string instead
    # of by context class name (`iterationStatement`'s five forms all share
    # one rule name / rule index in ANTLR).
    statements, leftover = extractor.extract("do { x(); } while (c);", final=False)
    assert statements == ["do { x(); } while (c);"]
    assert leftover == ""

    statements, leftover = extractor.extract("while (c) { x(); }", final=False)
    assert statements == []
    assert leftover == "while (c) { x(); }"


def test_extract_holds_back_an_unterminated_string_literal(
    extractor: JavaScriptAntlrExtractor,
):
    statements, leftover = extractor.extract('x = "hello', final=False)
    assert statements == []
    assert leftover == 'x = "hello'

    statements, leftover = extractor.extract('x = "hello world";\ny();\n', final=False)
    assert statements == ['x = "hello world";', "y();"]


def test_extract_holds_back_an_unterminated_template_literal(
    extractor: JavaScriptAntlrExtractor,
):
    statements, leftover = extractor.extract("x = `hello", final=False)
    assert statements == []
    assert leftover == "x = `hello"

    statements, leftover = extractor.extract("x = `hello`;\ny();\n", final=False)
    assert statements == ["x = `hello`;", "y();"]


def test_extract_holds_back_a_multi_char_operator_split_across_chunks(
    extractor: JavaScriptAntlrExtractor,
):
    # Simulates a chunk boundary landing between the two `=` characters of
    # `===`. The buffer as it stands after the first chunk ("x ==") must not
    # be misread as a complete, syntactically-wrong statement.
    statements, leftover = extractor.extract("x ==", final=False)
    assert statements == []
    assert leftover == "x =="


def test_extract_defers_every_parse_failure_while_streaming(
    extractor: JavaScriptAntlrExtractor,
):
    # The extractor's disambiguation policy for ANTLR's ALL(*) prediction
    # (documented on `_parse_or_recover`): since a failed parse's reported
    # offending token can land arbitrarily far from the true point of
    # truncation — confirmed for both an unterminated string containing a
    # `{` and an ordinary unclosed function call — no parse failure can be
    # trusted as "unambiguously wrong" while streaming. Every failure defers
    # until `final=True`, even one as clearly invalid as `x = = 1;`.
    for invalid in ("x = = 1;", "x = ;", "if (x) else {}"):
        statements, leftover = extractor.extract(invalid, final=False)
        assert statements == []
        assert leftover == invalid


def test_extract_raises_on_final_for_an_unrecoverable_buffer(
    extractor: JavaScriptAntlrExtractor,
):
    for invalid in ("x = = 1;", "x = ;", "if (x) else {}"):
        with pytest.raises(SyntaxError):
            extractor.extract(invalid, final=True)


@pytest.mark.parametrize("final", [False, True])
def test_extract_treats_blank_only_input_as_a_noop(
    extractor: JavaScriptAntlrExtractor, final: bool
):
    statements, leftover = extractor.extract("   \n  ", final=final)
    assert statements == []
    assert leftover == ("" if final else "   \n  ")


def test_extract_defers_a_nested_compound_statement_correctly(
    extractor: JavaScriptAntlrExtractor,
):
    statements, leftover = extractor.extract(
        "if (a) { if (b) { c(); } }\nd();\n", final=False
    )
    assert statements == ["if (a) { if (b) { c(); } }", "d();"]
    assert leftover == "\n"
