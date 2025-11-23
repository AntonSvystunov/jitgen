import re
from .prompt import task_prompt, TaskInput

from langchain_core.runnables import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from jitgen.executors.python import InProcPythonExecutor
from jitgen_langchain.python import create_python_jitgen_parser


def create_jitgen_chain(llm: BaseChatModel) -> RunnableSerializable[TaskInput, str]:
    """
    Create a chain that combines a task prompt with a language model and a Python JITGen parser.

    Args:
        llm (BaseChatModel): The language model to use in the chain.

    Returns:
        RunnableSerializable: The constructed chain.
    """

    return task_prompt | llm | create_python_jitgen_parser() | StrOutputParser()


def create_sync_executor_chain(llm: BaseChatModel) -> RunnableSerializable[TaskInput, str]:
    """
    Create a synchronous executor chain that processes input through a task prompt,
    a language model, and executes Python code.

    Args:
        llm (BaseChatModel): The language model to use in the chain.

    Returns:
        RunnableSerializable: The constructed synchronous executor chain.
    """

    async def _execute_python_code(text: str) -> str:
        code_block = re.search(r"```python\s+(.*?)```", text, re.DOTALL)
        if not code_block:
            return ""

        source_code = code_block.group(1)

        executor = InProcPythonExecutor()
        result = await executor.aexecute(source_code)

        if result.success:
            return result.output
        else:
            raise ValueError(
                f"Error detected. Halting further processing. {result.error}"
            )

    return task_prompt | llm | StrOutputParser() | _execute_python_code
