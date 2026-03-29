from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Protocol

from lark import (
    Lark,
    Token,
    UnexpectedCharacters,
    UnexpectedEOF,
    UnexpectedInput,
    UnexpectedToken,
)
from lark.indenter import DedentError
from lark.tree import Branch

type Statement = str


@dataclass(slots=True)
class MarkerSegment:
    """Executable code slice extracted from marker-aware stream processing."""

    text: str
    flush_after: bool = False


class StatementParser(Protocol):
    """Protocol for algorithms that detect executable statements from streamed input."""

    def ingest_chunk(self, chunk: str) -> list[MarkerSegment]: ...
    def finalize_ingest(self) -> list[MarkerSegment]: ...
    def append_code(self, text: str) -> None: ...
    def pop_ready_statements(self, *, flush: bool) -> list[Statement]: ...


class BaseStatefulAlgorithm(ABC):
    """Stateful algorithm for marker-aware buffering and statement extraction."""

    def __init__(
        self,
        *,
        parser: Lark,
        indentation_tokens: set[str],
        start_marker: str,
        end_marker: str,
    ) -> None:
        self.parser = parser
        self.indentation_tokens = indentation_tokens
        self.start_marker = start_marker
        self.end_marker = end_marker

        self._inside_markers = False
        self._raw_buffer = ""
        self._code_buffer = ""

    @property
    def inside_markers(self) -> bool:
        return self._inside_markers

    @property
    def code_buffer(self) -> str:
        return self._code_buffer

    def ingest_chunk(self, chunk: str) -> list[MarkerSegment]:
        """Ingest raw streamed text and return executable marker segments."""
        self._raw_buffer += chunk
        return self._drain_raw_buffer(final=False)

    def finalize_ingest(self) -> list[MarkerSegment]:
        """Finalize marker processing at stream/session end."""
        return self._drain_raw_buffer(final=True)

    def append_code(self, text: str) -> None:
        self._code_buffer += text

    def pop_ready_statements(self, *, flush: bool) -> list[Statement]:
        """Return executable statements from code buffer and update remaining buffer."""
        if not self._code_buffer.strip():
            if flush:
                self._code_buffer = ""
            return []

        try:
            tree = self.parser.parse(self._code_buffer)
        except DedentError:
            if flush:
                raise ValueError(
                    "Syntax error detected. Halting further processing. "
                    "Unexpected dedent during flush."
                )
            return []
        except UnexpectedEOF:
            if flush:
                raise ValueError(
                    "Syntax error detected. Halting further processing. "
                    "Unexpected end of input during flush."
                )
            return []
        except UnexpectedCharacters as e:
            raise ValueError(
                f"Syntax error detected. Halting further processing. {str(e)}"
            )
        except UnexpectedToken as e:
            token_type = getattr(e.token, "type", None)
            if not flush and token_type in self.indentation_tokens:
                return []
            raise ValueError(
                f"Syntax error detected. Halting further processing. {str(e)}"
            )
        except UnexpectedInput as e:
            if flush:
                raise ValueError(
                    f"Syntax error detected. Halting further processing. {str(e)}"
                )
            return []

        if flush:
            statements = [
                self._statement_text(self._code_buffer, statement)
                for statement in tree.children
            ]
            self._code_buffer = ""
            return [stmt for stmt in statements if stmt.strip()]

        if len(tree.children) < 2:
            return []

        statements: list[Statement] = []
        executed_upto = 0
        for statement in tree.children[:-1]:
            stmt = self._statement_text(self._code_buffer, statement)
            if stmt.strip():
                statements.append(stmt)
            meta = getattr(statement, "meta", None)  # type: ignore[arg-type]
            if meta is not None:
                executed_upto = getattr(meta, "end_pos", 0)  # type: ignore[arg-type]
        self._code_buffer = self._code_buffer[executed_upto:]
        return statements

    def _drain_raw_buffer(self, *, final: bool) -> list[MarkerSegment]:
        segments: list[MarkerSegment] = []

        while True:
            marker = self.end_marker if self._inside_markers else self.start_marker
            marker_idx = self._raw_buffer.find(marker)

            if marker_idx == -1:
                keep = 0 if final else self._overlap_suffix_prefix(self._raw_buffer, marker)
                stable_text = self._raw_buffer if keep == 0 else self._raw_buffer[:-keep]

                if self._inside_markers and stable_text:
                    segments.append(MarkerSegment(text=stable_text, flush_after=False))
                self._raw_buffer = "" if keep == 0 else self._raw_buffer[-keep:]
                return segments

            before_marker = self._raw_buffer[:marker_idx]
            self._raw_buffer = self._raw_buffer[marker_idx + len(marker) :]

            if self._inside_markers:
                segments.append(MarkerSegment(text=before_marker, flush_after=True))

            self._inside_markers = not self._inside_markers

    @staticmethod
    def _statement_text(buffer: str, statement: Branch[Token]) -> Statement:
        meta = getattr(statement, "meta", None)  # type: ignore[arg-type]
        if meta is None:
            return ""
        start = getattr(meta, "start_pos", 0)  # type: ignore[arg-type]
        end = getattr(meta, "end_pos", 0)  # type: ignore[arg-type]
        return buffer[start:end]

    @staticmethod
    def _overlap_suffix_prefix(text: str, marker: str) -> int:
        """Length of longest suffix in text that is a prefix of marker."""
        max_len = min(len(text), len(marker) - 1)
        for size in range(max_len, 0, -1):
            if text.endswith(marker[:size]):
                return size
        return 0


class MarkerStatefulAlgorithm(BaseStatefulAlgorithm):
    """Concrete marker-aware stateful algorithm."""
