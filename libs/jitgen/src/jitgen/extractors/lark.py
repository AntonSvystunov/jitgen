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

        # Non-final: the first N-1 children are complete by construction — a
        # further child could only ever extend the *last* one.  The last child
        # is additionally safe when it cannot grow: see :meth:`_is_sealed`.
        children = list(tree.children)
        if children and self._is_sealed(source, children[-1]):
            ready = children
        elif len(children) < 2:
            return [], source
        else:
            ready = children[:-1]

        statements: list[SourceCode] = []
        executed_upto = 0
        for child in ready:
            stmt = self._statement_text(source, child)
            if stmt.strip():
                statements.append(stmt)
            meta = getattr(child, "meta", None)
            if meta is not None:
                executed_upto = getattr(meta, "end_pos", 0)
        return statements, source[executed_upto:]

    def _is_sealed(self, source: SourceCode, node: Branch[Token]) -> bool:
        """Can *node* still be extended by input that has not arrived yet?

        The N-1 rule is conservative: it waits for the *next* statement to start
        before releasing one, which costs a full line of latency per statement
        and — since a generated script's ``print`` is usually its last line —
        defers all output to the final flush.

        A statement is *sealed* (safe to release right away) when both hold:

        * its grammar rule cannot take further clauses or an indented body —
          :meth:`_extensible_rules` names the ones that can; and
        * a real newline follows it in *source*, so the terminating NEWLINE came
          from the model rather than from the ``"\\n"`` that
          :meth:`_parse_or_recover` appends.  Without this, ``x = 1`` would be
          released while the stream is still one chunk away from ``x = 1 + 2``.
        """
        extensible = self._extensible_rules()
        if extensible is None:
            return False
        rule = getattr(node, "data", None)
        if rule is None or str(rule) in extensible:
            return False
        meta = getattr(node, "meta", None)
        if meta is None:
            return False
        return "\n" in source[getattr(meta, "end_pos", len(source)) :]

    @staticmethod
    def _extensible_rules() -> frozenset[str] | None:
        """Rule names whose nodes may still grow when more input arrives.

        ``None`` — the default — means the grammar has not classified its rules,
        so sealing is disabled entirely and the conservative N-1 rule applies.
        Erring the other way would hand the executor a fragment of a statement,
        so a new grammar has to opt in by overriding this.
        """
        return None

    @staticmethod
    def _statement_text(buffer: str, node: Branch[Token]) -> SourceCode:
        meta = getattr(node, "meta", None)
        if meta is None:
            return ""
        start = getattr(meta, "start_pos", 0)
        end = getattr(meta, "end_pos", 0)
        return buffer[start:end]
