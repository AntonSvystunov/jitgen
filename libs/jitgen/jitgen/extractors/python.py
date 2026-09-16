from lark import (
    UnexpectedCharacters,
    UnexpectedEOF,
    UnexpectedInput,
    UnexpectedToken,
)
from lark.indenter import DedentError
from lark.tree import Tree

from jitgen.base import SourceCode

from .lark import LarkStatementExtractor

_INDENT_TOKENS: frozenset[str] = frozenset({"_DEDENT", "_NEWLINE", "$END"})

# Rules that a later line can still attach to: either by extending an indented
# suite, or by adding a clause (`elif` / `else` / `except` / `finally`).
# Everything else is a simple statement, complete at its newline.
_EXTENSIBLE_RULES: frozenset[str] = frozenset(
    {
        "if_stmt",
        "while_stmt",
        "for_stmt",
        "try_stmt",
        # The grammar aliases the except-less form: `"try" ":" suite finally
        # -> try_finally`. Omitting it here sealed a try/finally the moment a
        # newline followed, releasing it before its `finally` body had finished
        # streaming — and the orphaned indented remainder then failed to parse,
        # aborting generation on perfectly valid code.
        "try_finally",
        "with_stmt",
        "match_stmt",
        "funcdef",
        "async_funcdef",
        "async_stmt",
        "classdef",
        "decorated",
        "decorators",
        "suite",
    }
)


class PythonLarkExtractor(LarkStatementExtractor):
    """Python-grammar statement extractor with Python-specific recovery rules.

    Recoverable parse failures (incomplete input, not a syntax error):

    - `UnexpectedEOF` — input ends mid-statement
    - `DedentError` (`lark.indenter`) — mid-block dedent artefact
    - `UnexpectedToken` whose token type is in `{"_DEDENT", "_NEWLINE", "$END"}`
      — indentation continuation
    - `UnexpectedCharacters` on an unclosed string quote (`"` or `'`) — string
      literal not yet closed
    - `UnexpectedCharacters` or `UnexpectedToken` ending at the very end of the
      buffer — a token the stream has not finished delivering (see
      `_ends_at_buffer_tail`)

    Any other parse error is unrecoverable and raised as `SyntaxError`.

    Extend this class or create a sibling for other Lark-based grammars,
    implementing `_parse_or_recover` with the appropriate recovery set.
    """

    @staticmethod
    def _extensible_rules() -> frozenset[str] | None:
        return _EXTENSIBLE_RULES

    @staticmethod
    def _ends_at_buffer_tail(source: SourceCode, offset: int | None) -> bool:
        """Check whether the offending token/character runs to the very end of `source`.

        LALR drives a *contextual* lexer: only the terminals the current parser
        state accepts are offered. A chunk boundary landing inside a token can
        therefore make a perfectly good character look illegal — after `for p`,
        the `i` of a still-streaming `in` matches no permitted terminal, since
        NAME is not among them, and lark reports it as an
        `UnexpectedToken(NAME, 'i')`. That is incomplete input, not a syntax
        error, and rejecting it aborts generation on valid code.

        Only the buffer's final token can be truncated this way: anything with
        input after it has already been proven illegal in full context.

        Args:
            source: The full unexecuted-suffix buffer being parsed.
            offset: The offending token/character's end position, or `None`.

        Returns:
            `True` if `offset` falls at or past `source`'s last character.
        """
        # `source` is parsed as `source + "\n"`, so the last real character
        # sits at len(source) - 1 and a token ending there ends at len(source).
        return offset is not None and offset >= len(source) - 1

    def _parse_or_recover(self, source: SourceCode, *, final: bool) -> Tree | None:
        """Implement the recovery policy documented on the class.

        Args:
            source: The full unexecuted-suffix buffer accumulated so far.
            final: `True` when the stream has ended, so a parse error can no
                longer be explained away as "more input is still coming".

        Returns:
            A `Tree` on successful parse; `None` when the parse error is
            recoverable (more input needed).

        Raises:
            SyntaxError: on an unrecoverable grammar violation.
        """
        try:
            return self._parser.parse(source + "\n")
        except DedentError as exc:
            if final:
                raise SyntaxError(f"Unexpected dedent: {exc}") from exc
            return None
        except UnexpectedEOF as exc:
            if final:
                raise SyntaxError(f"Unexpected end of input: {exc}") from exc
            return None
        except UnexpectedCharacters as exc:
            if not final and (
                exc.char in ('"', "'")
                or self._ends_at_buffer_tail(
                    source, getattr(exc, "pos_in_stream", None)
                )
            ):
                return None
            raise SyntaxError(str(exc)) from exc
        except UnexpectedToken as exc:
            token = getattr(exc, "token", None)
            token_type = getattr(token, "type", None)
            if not final and (
                token_type in _INDENT_TOKENS
                or self._ends_at_buffer_tail(source, getattr(token, "end_pos", None))
            ):
                return None
            raise SyntaxError(str(exc)) from exc
        except UnexpectedInput as exc:
            if final:
                raise SyntaxError(str(exc)) from exc
            return None
