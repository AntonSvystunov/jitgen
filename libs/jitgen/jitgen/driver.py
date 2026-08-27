from .base import CodeSegmenter
from .errors import JitGenError
from .session import Session


class StreamDriver:
    """Feed raw model output through a segmenter into a session.

    Every consumer of this library was writing the same loop — strip markers,
    push, poll for an error, resolve and reset at the block boundary — and the
    two subtleties below were re-derived (or missed) each time.

    Usage:

    ```python
    driver = StreamDriver(session, markdown_code("python"))
    async for chunk in model_stream:
        for output in await driver.apush(chunk):
            yield output
        if driver.has_error:
            break                    # stop paying for tokens after a failure
    for output in await driver.afinish():
        yield output
    ```

    `apush` returns output without raising, so a caller can decide when to
    stop; the error surfaces from `afinish`. The exception is a block that
    closes cleanly mid-stream, where the result is due immediately.

    Args:
        session: The session that extracts and executes statements.
        segmenter: The segmenter that recovers code text from the raw stream.
    """

    def __init__(self, session: Session, segmenter: CodeSegmenter) -> None:
        self._session = session
        self._segmenter = segmenter

    @property
    def session(self) -> Session:
        """The underlying session driving extraction and execution."""
        return self._session

    @property
    def has_error(self) -> bool:
        """`True` once the session has recorded a failure."""
        return self._session.has_error

    @property
    def error(self) -> JitGenError | None:
        """The first error recorded by the session, or `None`."""
        return self._session.error

    async def apush(self, text: str) -> list[str]:
        """Push one raw chunk; return whatever stdout is ready now.

        Args:
            text: The next increment of raw model output.

        Returns:
            Stdout produced so far that had not yet been returned, in order;
            empty when nothing new is ready.

        Raises:
            JitGenError: only when a code block closed in this chunk and its
                execution failed. Errors detected mid-block are left on the
                session for the caller to notice via `has_error`.
        """
        out: list[str] = []
        flushed = False
        for segment in self._segmenter.feed(text):
            self._session.push(segment.text)
            if self._session.has_error:
                return [o for o in out if o]
            if segment.end_of_block:
                out.append(await self._session.result())
                await self._session.reset()
                self._segmenter.reset()
                flushed = True

        # Drained per *chunk*, not per segment: while the segmenter withholds a
        # suffix that might turn out to be the end marker it emits no segments at
        # all, and those are precisely the final chunks — when the last
        # statement's output has just landed.  Holding output back until the
        # closing marker would pin time-to-first-output to the end of
        # generation, which is exactly what the non-incremental baseline does.
        if not flushed and not self._session.has_error:
            out.append(self._session.take_output())
        return [o for o in out if o]

    async def afinish(self) -> list[str]:
        """Resolve the stream: flush an unterminated block, then await the rest.

        Returns:
            The final stdout, as a single-item list, or empty when there was
            none.

        Raises:
            JitGenError: the first extraction or execution failure of the stream.
        """
        if self._session.has_error:
            await self._session.result()  # always raises

        # The stream ended while still inside a block — the model omitted the
        # closing marker, or inference was stopped early.  Execute what arrived
        # rather than discarding it.
        if self._segmenter.inside_block:
            for segment in self._segmenter.finalize():
                self._session.push(segment.text)

        output = await self._session.result()
        return [output] if output else []

    async def areset(self) -> None:
        """Clear both halves for the next turn.  Keeps executor REPL state."""
        self._segmenter.reset()
        await self._session.reset()
