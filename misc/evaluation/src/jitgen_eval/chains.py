import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .prompt import task_prompt, TaskInput

from langchain_core.runnables import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from jitgen.executors.python import InProcPythonExecutor
from jitgen.markers import MarkerStripper
from jitgen.prebuilt.python import create_python_jitgen
from jitgen_langchain.parser import JITGenParser


@dataclass
class ChainBundle:
    """A chain plus the teardown for the stateful resources it owns.

    Both chain flavours keep per-invocation state — a JITGen ``Session`` with a
    REPL namespace, or a bare executor with the same — so each test case must
    get its own bundle and dispose of it afterwards.  ``InProcPythonExecutor``
    starts a daemon thread per instance, so skipping ``aclose`` leaks threads.
    """

    chain: RunnableSerializable[TaskInput, str]
    aclose: Callable[[], Awaitable[None]]


def create_jitgen_chain(llm: BaseChatModel) -> ChainBundle:
    """
    Create a chain that combines a task prompt with a language model and a Python JITGen parser.

    The session, stripper and executor are built here (rather than via
    ``create_python_jitgen_parser()``) so that the caller can shut them down.

    Args:
        llm (BaseChatModel): The language model to use in the chain.

    Returns:
        ChainBundle: The constructed chain and its teardown callback.
    """

    executor = InProcPythonExecutor()
    session = create_python_jitgen(executor=executor)
    stripper = MarkerStripper(start="```python", end="```")
    parser = JITGenParser(session=session, stripper=stripper)

    async def _aclose() -> None:
        await session.aclose()
        await executor.aclose()

    return ChainBundle(
        chain=task_prompt | llm | parser | StrOutputParser(),
        aclose=_aclose,
    )


def create_sync_executor_chain(llm: BaseChatModel) -> ChainBundle:
    """
    Create a synchronous executor chain that processes input through a task prompt,
    a language model, and executes Python code.

    Args:
        llm (BaseChatModel): The language model to use in the chain.

    Returns:
        ChainBundle: The constructed chain and its teardown callback.
    """

    executor = InProcPythonExecutor()

    async def _execute_python_code(text: str) -> str:
        code_block = re.search(r"```python\s+(.*?)```", text, re.DOTALL)
        if not code_block:
            return ""

        source_code = code_block.group(1)

        result = await executor.aexecute(source_code)

        if result.success:
            return result.output
        else:
            raise ValueError(
                f"Error detected. Halting further processing. {result.error}"
            )

    return ChainBundle(
        chain=task_prompt | llm | StrOutputParser() | _execute_python_code,
        aclose=executor.aclose,
    )
