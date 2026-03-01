from collections.abc import AsyncIterator, Iterator
from typing import Any, override

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseTransformOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic import ConfigDict, Field

from jitgen_core import JITGen


class JITGenParser(BaseTransformOutputParser[str]):
    model_config: ConfigDict = ConfigDict(
        arbitrary_types_allowed=True,
    )
    
    jit_gen: JITGen = Field(
        description="JITGen instance for parsing and executing code."
    )

    start_marker: str = Field(
        default="```",
        description="Marker indicating the start of a code block in the input stream.",
    )

    end_marker: str = Field(
        default="```",
        description="Marker indicating the end of a code block in the input stream.",
    )

    @property
    @override
    def _type(self) -> str:
        return "jitgen_parser"

    @override
    def parse(self, text: str) -> str:
        """Parse method required by base class but not used in streaming mode."""
        return text
    
    def _yield_code_blocks(
        self, input: Iterator[str | BaseMessage]  # noqa: A002
    ) -> Iterator[str]:
        inside_block = False  # Flag to know when we're inside the desired code block
        current_line = ""
        first = True
        for chunk in input:
            # Get text content from BaseMessage or use string directly
            chunk_text: str
            if isinstance(chunk, BaseMessage):
                chunk_text = chunk.text
            else:
                chunk_text = chunk
            
            # Split the incoming chunk into parts by newline.
            lines = chunk_text.split("\n")

            if first:
                first = False
                lines[0] = lines[0].strip()  # [1:]  # Adjust first line if needed

            # Process each line in the chunk.
            for i in range(len(lines)):
                # For all but the last part (which may be an incomplete line)
                if i < len(lines) - 1:
                    current_line += lines[i]
                    stripped = current_line.strip()
                    # If not yet inside the block, look for the starting marker.
                    if not inside_block:
                        if stripped == self.start_marker:
                            inside_block = True
                            # Clear the buffer so we start fresh.
                            current_line = ""
                    else:
                        # If inside the block, check for the end marker.
                        if stripped == self.end_marker:
                            return  # Stop yielding altogether. TODO: Handle this case properly.
                        else:
                            # Yield the complete line.
                            yield current_line + "\n"
                    # Clear the buffer after processing a complete line.
                    current_line = ""
                else:
                    # Last element may be an incomplete line; add it to the buffer.
                    current_line += lines[i]

    async def _ayield_code_blocks(
        self, input: AsyncIterator[str | BaseMessage]  # noqa: A002
    ) -> AsyncIterator[str]:
        """
        Asynchronously yield code blocks from the input stream.
        """
        inside_block = False  # Flag to know when we're inside the desired code block
        current_line = ""
        first = True
        async for chunk in input:
            # Get text content from BaseMessage or use string directly
            chunk_text: str
            if isinstance(chunk, BaseMessage):
                chunk_text = chunk.text
            else:
                chunk_text = chunk
            
            # Split the incoming chunk into parts by newline.
            lines = chunk_text.split("\n")

            if first:
                first = False
                lines[0] = lines[0].strip()  # [1:]  # Adjust first line if needed

            # Process each line in the chunk.
            for i in range(len(lines)):
                # For all but the last part (which may be an incomplete line)
                if i < len(lines) - 1:
                    current_line += lines[i]
                    stripped = current_line.strip()
                    # If not yet inside the block, look for the starting marker.
                    if not inside_block:
                        if stripped == self.start_marker:
                            inside_block = True
                            # Clear the buffer so we start fresh.
                            current_line = ""
                    else:
                        # If inside the block, check for the end marker.
                        if stripped == self.end_marker:
                            return  # Stop yielding altogether. TODO: Handle this case properly.
                        else:
                            # Yield the complete line.
                            yield current_line + "\n"
                    # Clear the buffer after processing a complete line.
                    current_line = ""
                else:
                    # Last element may be an incomplete line; add it to the buffer.
                    current_line += lines[i]

    @override
    def transform(
        self,
        input: Iterator[str | BaseMessage],  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Iterator[str]:
        """
        Transform the input stream into a list of code blocks.
        Each code block is yielded as a list containing the single string.
        """
        for output_chunk in self.jit_gen.run_from_stream(self._yield_code_blocks(input)):
            # Yield each output chunk as a list containing the single string.
            yield output_chunk

    @override
    async def _atransform(
        self, input: AsyncIterator[str | BaseMessage]  # noqa: A002
    ) -> AsyncIterator[str]:
        async for output_chunk in self.jit_gen.arun_from_stream(
            self._ayield_code_blocks(input)
        ):
            # Yield each output chunk as a list containing the single string.
            yield output_chunk
