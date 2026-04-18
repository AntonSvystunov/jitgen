from __future__ import annotations

from abc import ABC, abstractmethod

from lark import Lark, Token
from lark.tree import Branch, Tree

from jitgen_core import SourceCode


class LarkStatementExtractor(ABC):
    """Thin base for Lark-driven statement extraction.

    Handles the mechanics of converting a successful parse tree into a list
    of statement strings (via Lark meta positions) and the N-1 child slicing
    invariant.  Subclasses own the complete grammar-specific recovery policy
    by implementing :meth:`_parse_or_recover`.

    To support a new language, subclass this, set up the appropriate
    :class:`lark.Lark` parser, and implement ``_parse_or_recover`` with the
    exact set of Lark exceptions that mean "incomplete input, buffer more".
    """

    def __init__(self, parser: Lark) -> None:
        self._parser = parser

    @abstractmethod
    def _parse_or_recover(self, source: SourceCode, *, final: bool) -> Tree | None:
        """Attempt to parse *source* with grammar-specific recovery logic.

        Returns:
            A :class:`lark.Tree` on successful parse.
            ``None`` when the parse error is recoverable (more input needed).

        Raises:
            :class:`SyntaxError` on an unrecoverable grammar violation.
        """

    def extract(
        self, source: SourceCode, *, final: bool
    ) -> tuple[list[SourceCode], SourceCode]:
        """Extract ready top-level statements from *source*.

        Returns:
            A ``(statements, leftover)`` pair.  *leftover* is the portion of
            *source* not yet covered by a complete statement.

        Raises:
            :class:`SyntaxError` if :meth:`_parse_or_recover` raises.
        """
        if not source.strip():
            return [], ("" if final else source)

        tree = self._parse_or_recover(source, final=final)
        if tree is None:
            return [], source

        if final:
            statements = [
                self._statement_text(source, child) for child in tree.children
            ]
            return [s for s in statements if s.strip()], ""

        # Non-final: need ≥2 children to confirm the first N-1 are complete.
        if len(tree.children) < 2:
            return [], source

        statements: list[SourceCode] = []
        executed_upto = 0
        for child in tree.children[:-1]:
            stmt = self._statement_text(source, child)
            if stmt.strip():
                statements.append(stmt)
            meta = getattr(child, "meta", None)
            if meta is not None:
                executed_upto = getattr(meta, "end_pos", 0)
        return statements, source[executed_upto:]

    @staticmethod
    def _statement_text(buffer: str, node: Branch[Token]) -> SourceCode:
        meta = getattr(node, "meta", None)
        if meta is None:
            return ""
        start = getattr(meta, "start_pos", 0)
        end = getattr(meta, "end_pos", 0)
        return buffer[start:end]
