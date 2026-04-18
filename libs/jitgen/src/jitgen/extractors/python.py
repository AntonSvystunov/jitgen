from __future__ import annotations

from lark import (
    Lark,
    UnexpectedCharacters,
    UnexpectedEOF,
    UnexpectedInput,
    UnexpectedToken,
)
from lark.indenter import DedentError
from lark.tree import Tree

from jitgen_core import SourceCode

from .lark import LarkStatementExtractor

_INDENT_TOKENS: frozenset[str] = frozenset({"_DEDENT", "_NEWLINE", "$END"})


class PythonLarkExtractor(LarkStatementExtractor):
    """Python-grammar statement extractor with Python-specific recovery rules.

    Recoverable parse failures (incomplete input, not a syntax error):
    - :class:`~lark.UnexpectedEOF` — input ends mid-statement
    - :class:`~lark.indenter.DedentError` — mid-block dedent artefact
    - :class:`~lark.UnexpectedToken` whose token type is in
      ``{"_DEDENT", "_NEWLINE", "$END"}`` — indentation continuation
    - :class:`~lark.UnexpectedCharacters` on an unclosed string quote
      (``"`` or ``'``) — string literal not yet closed

    Any other parse error is unrecoverable and raised as :class:`SyntaxError`.

    Extend this class or create a sibling for other Lark-based grammars,
    implementing ``_parse_or_recover`` with the appropriate recovery set.
    """

    def _parse_or_recover(self, source: SourceCode, *, final: bool) -> Tree | None:
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
            if not final and exc.char in ('"', "'"):
                return None
            raise SyntaxError(str(exc)) from exc
        except UnexpectedToken as exc:
            token_type = getattr(exc.token, "type", None)
            if not final and token_type in _INDENT_TOKENS:
                return None
            raise SyntaxError(str(exc)) from exc
        except UnexpectedInput as exc:
            if final:
                raise SyntaxError(str(exc)) from exc
            return None
