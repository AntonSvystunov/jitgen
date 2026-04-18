"""MarkerStripper: client-side utility to extract code from a marker-delimited stream."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CodeSegment:
    """A slice of code text extracted from between start/end markers.

    ``end_of_block=True`` signals that this segment closes a marker block —
    the caller may then call ``await session.result()`` to collect the output
    and reset for the next block.
    """

    text: str
    end_of_block: bool = field(default=False)


class MarkerStripper:
    """Transform a raw LLM output stream into code-only segments.

    Scans for configurable start/end markers (e.g. markdown code fences or
    JSON tool-call framing) and yields only the text that lies between them.
    Handles chunks that split a marker across boundaries via suffix-prefix
    overlap detection.

    This class is fully independent of any grammar or executor — configure it
    client-side with whatever markers your LLM output uses::

        stripper = MarkerStripper(start='{"code":"', end='"}')
        for chunk in llm_stream:
            for seg in stripper.process(chunk):
                session.push(seg.text)
                if seg.end_of_block:
                    output = await session.result()
                    session.reset()
                    stripper.reset()
    """

    def __init__(self, *, start: str, end: str) -> None:
        self.start = start
        self.end = end
        self._buffer = ""
        self._inside = False

    @property
    def inside_markers(self) -> bool:
        """``True`` when the current position in the stream is inside a marker block."""
        return self._inside

    def reset(self) -> None:
        """Clear internal buffer and reset marker state for a new stream."""
        self._buffer = ""
        self._inside = False

    def process(self, chunk: str) -> list[CodeSegment]:
        """Append *chunk* and return any newly extractable code segments."""
        self._buffer += chunk
        return self._drain(final=False)

    def finalize(self) -> list[CodeSegment]:
        """Flush any remaining buffered text at end-of-stream."""
        return self._drain(final=True)

    # ── private ────────────────────────────────────────────────────────

    def _drain(self, *, final: bool) -> list[CodeSegment]:
        segments: list[CodeSegment] = []

        while True:
            marker = self.end if self._inside else self.start
            idx = self._buffer.find(marker)

            if idx == -1:
                # Marker not found; retain a suffix that could be a marker prefix.
                keep = 0 if final else self._overlap_suffix_prefix(self._buffer, marker)
                stable = self._buffer if keep == 0 else self._buffer[:-keep]

                if self._inside and stable:
                    segments.append(CodeSegment(text=stable, end_of_block=False))
                self._buffer = "" if keep == 0 else self._buffer[-keep:]
                return segments

            before = self._buffer[:idx]
            self._buffer = self._buffer[idx + len(marker):]

            if self._inside:
                # Text before the end marker closes the block.
                segments.append(CodeSegment(text=before, end_of_block=True))
            # Text before the start marker is outside a block — discard.

            self._inside = not self._inside

    @staticmethod
    def _overlap_suffix_prefix(text: str, marker: str) -> int:
        """Return the length of the longest suffix of *text* that is also a
        prefix of *marker* (to avoid discarding a partial marker at chunk end)."""
        max_len = min(len(text), len(marker) - 1)
        for size in range(max_len, 0, -1):
            if text.endswith(marker[:size]):
                return size
        return 0
