from jitgen.base import CodeSegment


class MarkerSegmenter:
    """Yield only the text that lies between a start and an end marker.

    Handles markers split across chunk boundaries by retaining any suffix that
    could still turn out to be the beginning of a marker. Text outside a block
    is discarded.

    Independent of grammar and executor — configure it with whatever framing
    the model's output uses:

    ```python
    segmenter = MarkerSegmenter(start="```python", end="```")
    ```

    Args:
        start: The marker text that opens a block.
        end: The marker text that closes a block.
    """

    def __init__(self, *, start: str, end: str) -> None:
        self.start = start
        self.end = end
        self._buffer = ""
        self._inside = False

    @property
    def inside_block(self) -> bool:
        """`True` when the stream position is inside a marker block."""
        return self._inside

    def reset(self) -> None:
        """Clear buffered state so the segmenter can be reused for a new stream."""
        self._buffer = ""
        self._inside = False

    def feed(self, chunk: str) -> list[CodeSegment]:
        """Append `chunk` and return any newly extractable code segments.

        Args:
            chunk: The next increment of raw model output.

        Returns:
            Code segments that became extractable as a result of `chunk`, in
            order; empty when nothing is ready yet.
        """
        self._buffer += chunk
        return self._drain(final=False)

    def finalize(self) -> list[CodeSegment]:
        """Flush whatever is buffered at end-of-stream.

        Used when the model ended its response without closing the block, so the
        code written so far is executed rather than silently discarded.

        Returns:
            Any code segments still held back, in order.
        """
        return self._drain(final=True)

    # ── private ────────────────────────────────────────────────────────

    def _drain(self, *, final: bool) -> list[CodeSegment]:
        """Advance the marker scan over the buffer, emitting closed segments.

        Args:
            final: `True` when the stream has ended, so any trailing content
                still inside a block should be emitted rather than withheld
                as a possible marker prefix.

        Returns:
            Newly extractable code segments, in order.
        """
        segments: list[CodeSegment] = []

        while True:
            marker = self.end if self._inside else self.start
            idx = self._buffer.find(marker)

            if idx == -1:
                # Marker not found; retain a suffix that could be a marker prefix.
                keep = 0 if final else _overlap_suffix_prefix(self._buffer, marker)
                stable = self._buffer if keep == 0 else self._buffer[:-keep]

                if self._inside and stable:
                    segments.append(CodeSegment(text=stable))
                self._buffer = "" if keep == 0 else self._buffer[-keep:]
                return segments

            before = self._buffer[:idx]
            self._buffer = self._buffer[idx + len(marker) :]

            if self._inside:
                # Text before the end marker closes the block.
                segments.append(CodeSegment(text=before, end_of_block=True))
            # Text before the start marker is outside a block — discard.

            self._inside = not self._inside


def _overlap_suffix_prefix(text: str, marker: str) -> int:
    """Find the longest suffix of `text` that is also a prefix of `marker`.

    Without this a marker split across two chunks would be missed entirely, and
    the block would never open (or never close).

    Args:
        text: The buffered text to inspect.
        marker: The marker `text` might be starting to match.

    Returns:
        The length of the longest matching suffix/prefix overlap; `0` when
        there is none.
    """
    max_len = min(len(text), len(marker) - 1)
    for size in range(max_len, 0, -1):
        if text.endswith(marker[:size]):
            return size
    return 0


def markdown_code(lang: str = "python") -> MarkerSegmenter:
    """Build a segmenter for a fenced markdown code block, e.g. ` ```python `.

    Args:
        lang: The language tag on the opening fence.

    Returns:
        A `MarkerSegmenter` matching that fence.
    """
    return MarkerSegmenter(start=f"```{lang}", end="```")
