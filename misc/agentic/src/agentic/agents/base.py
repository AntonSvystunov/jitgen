from dataclasses import dataclass, field
from typing import Protocol

from langchain_core.messages import AnyMessage


@dataclass
class ExecutionResult:
    messages: list[AnyMessage]

    success: bool
    timeout: bool

    steps_taken: int
    max_steps: int

    output: str | None
    error: str | None = None

    model_name: str | None = None
    total_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None

    total_execution_time: float | None = None
    inference_time: float | None = None
    execution_time: float | None = None

    # Set to True by IncrementalAgent when session.has_error aborted the stream
    # before on_chat_model_end — meaning we saved the remaining output tokens.
    early_exit_on_error: bool = False


class AgentProtocol(Protocol):
    async def run(self, question: str, guidelines: str) -> ExecutionResult: ...
