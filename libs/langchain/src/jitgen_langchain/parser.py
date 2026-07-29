from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, override

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseTransformOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic import ConfigDict, Field

from jitgen_core import Session
from jitgen.markers import MarkerStripper


class JITGenParser(BaseTransformOutputParser[str]):
    """LangChain output parser that executes code blocks via a JITGen session.

    Each marker-delimited code block in the LLM stream is pushed incrementally
    to the session (fire-and-forget); when the end marker is detected, the
    aggregated stdout is yielded and the session is reset for the next block.

    The *session* and *stripper* are fully client-owned — configure markers,
    executor, and timeout before constructing this parser.  Ownership extends to
    teardown: the parser resets both after each stream (so a single parser may be
    reused across many ``astream`` calls) but never calls
    :meth:`~jitgen_core.Session.aclose`.  The client must do that when the
    session is no longer needed.

    Example::

        from jitgen.prebuilt.python import create_python_jitgen
        from jitgen.markers import MarkerStripper
        from jitgen_langchain.parser import JITGenParser

        session  = create_python_jitgen()
        stripper = MarkerStripper(start="```python", end="```")
        parser   = JITGenParser(session=session, stripper=stripper)
        chain    = prompt | llm | parser
    """

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    session: Session = Field(description="JITGen session that executes extracted code.")
    stripper: MarkerStripper = Field(
        description="Marker stripper that isolates code blocks from LLM output."
    )

    @property
    @override
    def _type(self) -> str:
        return "jitgen_parser"

    @override
    def parse(self, text: str) -> str:
        return text

    @override
    def transform(
        self,
        input: Iterator[str | BaseMessage],  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        raise NotImplementedError(
            "Synchronous transform is not supported. Use astream / atransform."
        )

    @override
    async def _atransform(
        self, input: AsyncIterator[str | BaseMessage]  # noqa: A002
    ) -> AsyncIterator[str]:
        try:
            async for chunk in input:
                if self.session.has_error:
                    break

                text = str(chunk.content) if isinstance(chunk, BaseMessage) else chunk
                for seg in self.stripper.process(text):
                    self.session.push(seg.text)
                    if self.session.has_error:
                        break
                    if seg.end_of_block:
                        try:
                            output = await self.session.result()
                        except Exception as exc:
                            yield f"[JITGen error: {exc}]"
                            await self.session.reset()
                            self.stripper.reset()
                            break
                        if output:
                            yield output
                        await self.session.reset()
                        self.stripper.reset()

                if self.session.has_error:
                    try:
                        await self.session.result()
                    except Exception as exc:
                        yield f"[JITGen error: {exc}]"
                    await self.session.reset()
                    self.stripper.reset()
                    break

            # Stream ended while still inside a code block (e.g. the model
            # omitted the closing marker), or an error surfaced after the last
            # handled boundary.  Flush rather than silently discarding.
            if self.stripper.inside_markers or self.session.has_error:
                for seg in self.stripper.finalize():
                    self.session.push(seg.text)
                try:
                    output = await self.session.result()
                except Exception as exc:
                    yield f"[JITGen error: {exc}]"
                else:
                    if output:
                        yield output
        finally:
            # Reset for the next stream.  The stripper reset is synchronous, so
            # do it first — it still runs if the await below is cancelled.
            # Session.aclose() is the client's responsibility, not ours.
            self.stripper.reset()
            await self.session.reset()
