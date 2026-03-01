from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State for the CodeAct agent evaluation graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    """Conversation history (system + user + assistant + execution outputs)."""

    task_input: dict
    """Frozen MBPP task context (TaskInput fields)."""

    execution_count: int
    """Number of code execution turns used so far."""

    final_output: str
    """Last captured stdout from code execution."""

    has_error: bool
    """Whether an error occurred during the last execution."""

    has_timed_out: bool
    """Whether the last execution timed out."""

    error_message: str | None
    """Error message from the last execution, if any."""
