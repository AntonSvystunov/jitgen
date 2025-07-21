from typing import AsyncIterator, Iterator, Type
from lark import (
    Lark,
    Token,
    UnexpectedCharacters,
    UnexpectedEOF,
    UnexpectedToken,
    UnexpectedInput,
)
from lark.tree import Branch
from pydantic import BaseModel, ConfigDict, Field

from jitgen.core.base import BaseExecutor


class JITGen(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    
    parser: Lark = Field(
        description="The Lark parser instance used for parsing the input code."
    )

    interpreter_type: Type[BaseExecutor] = Field(
        description="The interpreter instance that executes the parsed statements."
    )

    indentation_tokens: set[str] = Field(
        default_factory=set,
        description="Set of tokens that are used for indentation in the parser.",
    )

    async def _aexecute_statement(
        self, buffer: str, statement: Branch[Token], interpreter: BaseExecutor
    ) -> str:
        """
        Execute a single statement from the buffer using the interpreter.
        """
        start = statement.meta.start_pos
        end = statement.meta.end_pos
        stmt = buffer[start:end]

        if stmt.strip():
            execution_result = await interpreter.aexecute(stmt)
            if not execution_result.success:
                raise ValueError(
                    f"Error detected. Halting further processing. {execution_result.error}"
                )
            return execution_result.output
        return ""
    
    def _execute_statement(
        self, buffer: str, statement: Branch[Token], interpreter: BaseExecutor
    ) -> str:
        """ Execute a single statement from the buffer using the interpreter.
        """
        start = statement.meta.start_pos
        end = statement.meta.end_pos
        stmt = buffer[start:end]

        if stmt.strip():
            execution_result = interpreter.execute(stmt)
            if not execution_result.success:
                raise ValueError(
                    f"Error detected. Halting further processing. {execution_result.error}"
                )
            return execution_result.output
        return ""

    async def arun_from_stream(
        self, input_stream: AsyncIterator[str]
    ) -> AsyncIterator[str]:
        interpreter = self.interpreter_type()
        code_buffer = ""

        async for fragment in input_stream:
            code_buffer += fragment

            try:
                tree = self.parser.parse(code_buffer)
            except UnexpectedEOF:
                # Incomplete input; wait for more fragments.
                continue
            except UnexpectedCharacters as e:
                raise ValueError(
                    f"Syntax error detected. Halting further processing. {str(e)}"
                )
            except UnexpectedToken as e:
                if e.token.type in self.indentation_tokens:
                    # Incomplete input; wait for more fragments.
                    continue
                # A genuine syntax error: halt further processing.
                raise ValueError(
                    f"Syntax error detected. Halting further processing. {str(e)}"
                )

            # If the parse tree has at least 2 top-level children, we have more than one complete statement.
            if len(tree.children) >= 2:
                executed_upto = 0
                # Execute all nodes except the last one.
                for statement in tree.children[:-1]:
                    yield await self._aexecute_statement(
                        code_buffer, statement, interpreter
                    )
                    executed_upto = statement.meta.end_pos
                # The code corresponding to the last node remains in the buffer.
                code_buffer = code_buffer[executed_upto:]

        # After the generator is exhausted, flush the remaining code.
        if code_buffer.strip():
            try:
                tree = self.parser.parse(code_buffer)
            except (UnexpectedEOF, UnexpectedToken, UnexpectedInput) as e:
                raise ValueError(
                    f"Syntax error detected. Halting further processing. {str(e)}"
                )
            else:
                for statement in tree.children:
                    yield await self._aexecute_statement(
                        code_buffer, statement, interpreter
                    )


    def run_from_stream(self, input_stream: Iterator[str]) -> Iterator[str]:
        interpreter = self.interpreter_type()
        code_buffer = ""

        for fragment in input_stream:
            code_buffer += fragment

            try:
                tree = self.parser.parse(code_buffer)
            except UnexpectedEOF:
                # Incomplete input; wait for more fragments.
                continue
            except UnexpectedCharacters as e:
                raise ValueError(
                    f"Syntax error detected. Halting further processing. {str(e)}"
                )
            except UnexpectedToken as e:
                if e.token.type in self.indentation_tokens:
                    # Incomplete input; wait for more fragments.
                    continue
                # A genuine syntax error: halt further processing.
                raise ValueError(
                    f"Syntax error detected. Halting further processing. {str(e)}"
                )

            # If the parse tree has at least 2 top-level children, we have more than one complete statement.
            if len(tree.children) >= 2:
                executed_upto = 0
                # Execute all nodes except the last one.
                for statement in tree.children[:-1]:
                    yield self._execute_statement(
                        code_buffer, statement, interpreter
                    )
                    executed_upto = statement.meta.end_pos
                # The code corresponding to the last node remains in the buffer.
                code_buffer = code_buffer[executed_upto:]

        # After the generator is exhausted, flush the remaining code.
        if code_buffer.strip():
            try:
                tree = self.parser.parse(code_buffer)
            except (UnexpectedEOF, UnexpectedToken, UnexpectedInput) as e:
                raise ValueError(
                    f"Syntax error detected. Halting further processing. {str(e)}"
                )
            else:
                for statement in tree.children:
                    yield self._execute_statement(
                        code_buffer, statement, interpreter
                    )