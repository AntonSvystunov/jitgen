from abc import abstractmethod
from dataclasses import dataclass
import re
from time import perf_counter

from jitgen.executors.python import InProcPythonExecutor
from jitgen.prebuilt.python import create_python_async_jitgen_session
from langsmith import traceable

from .prompt import SYSTEM_PROMPT, HUMAN_PROMPT

from langchain_core.language_models import BaseChatModel


@dataclass
class ExecutionResult:
    success: bool
    output: str | None
    total_time: float
    steps_executed: int
    steps_duration: list[float]
    steps_left: int
    messages: list[dict[str, str]]


class AgentSession:
    def __init__(
        self,
        model: BaseChatModel,
        context_files: list[str],
        question: str,
        guidelines: str,
        max_steps: int = 5,
    ):
        self.model = model

        self._messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(context_files="\n".join(context_files)),
            },
            {
                "role": "user",
                "content": HUMAN_PROMPT.format(
                    question=question, guidelines=guidelines
                ),
            },
        ]

        self._max_steps = max_steps
        self._steps_left = max_steps
        self._steps_duration: list[float] = []
        self._last_response: str | None = None
        self._last_observation: str | None = None

        self._should_continue_streaming: bool = True

    @property
    def steps_left(self) -> int:
        return self._steps_left

    @property
    def steps_executed(self) -> int:
        return self._max_steps - self._steps_left

    @property
    def steps_duration(self) -> list[float]:
        return list(self._steps_duration)

    def get_messages(self, *, include_inflight: bool = False) -> list[dict[str, str]]:
        messages = [message.copy() for message in self._messages]

        if not include_inflight:
            return messages

        if self._last_response:
            messages.append({"role": "assistant", "content": self._last_response})

        if self._last_observation and not self.has_solution():
            messages.append(
                {"role": "user", "content": "Observation:\n" + self._last_observation}
            )

        return messages

    async def _get_solution(self) -> str | None:
        if self._last_response is None:
            return None

        solution_blocks = re.findall(
            r"<solution>(.*?)</solution>", self._last_response, re.DOTALL
        )
        return solution_blocks[0].strip() if solution_blocks else None

    @abstractmethod
    async def on_chunk(self, chunk: str) -> None: ...

    @abstractmethod
    async def on_after_step(self) -> None: ...

    @abstractmethod
    def has_solution(self) -> bool: ...

    async def run(self) -> ExecutionResult:
        self._steps_duration = []
        start_time = perf_counter()

        while self._steps_left > 0:
            self._last_response = ""
            self._last_observation = ""
            self._should_continue_streaming = True
            
            reasoning_block = ""

            async for chunk in self.model.astream(self._messages):
                reasoning = chunk.additional_kwargs.get("reasoning_content", "")
                if reasoning:
                    reasoning_block += reasoning
                if not self._should_continue_streaming:
                    break
                chunk_text = chunk.content
                await self.on_chunk(chunk_text)
            
            await self.on_after_step()

            self._steps_left -= 1
            self._steps_duration.append(perf_counter() - start_time)

            assistant_content = self._last_response or reasoning_block
            if assistant_content:
                self._messages.append({"role": "assistant", "content": assistant_content})

            if self.has_solution():
                break

            self._messages.append(
                {"role": "user", "content": "Observation:\n" + self._last_observation}
            )

        end_time = perf_counter()

        final_response = await self._get_solution()
        return ExecutionResult(
            success=final_response is not None,
            output=final_response,
            total_time=end_time - start_time,
            steps_executed=self._max_steps - self._steps_left,
            steps_duration=self.steps_duration,
            steps_left=self._steps_left,
            messages=self.get_messages(),
        )


class IncrementalAgentSession(AgentSession):
    def __init__(
        self,
        model: BaseChatModel,
        context_files: list[str],
        question: str,
        guidelines: str,
        max_steps: int = 5,
    ):
        super().__init__(model, context_files, question, guidelines, max_steps)

        self._session = create_python_async_jitgen_session(
            start_marker="<execute>", end_marker="</execute>", tools={"open": open}
        )
        self._handled_session_error: Exception | None = None
        
        @self._session.on_stdout
        def on_stdout(stdout: str):
            self._last_observation += stdout
        
        @self._session.on_error
        def on_error(stderr: Exception):
            self._handled_session_error = stderr
            if self._should_continue_streaming:
                self._last_observation += str(stderr)
            self._should_continue_streaming = False # Stop streaming further chunks if there's an error during code execution

    def _is_handled_session_error(self, error: Exception) -> bool:
        return self._handled_session_error is error

    def _reset_incremental_session(self, *, clear_buffers: bool) -> None:
        if clear_buffers:
            self._session._algorithm._code_buffer = "" # Clear any buffered code in the session to prevent it from being executed in the next step
            self._session._algorithm._raw_buffer = "" # Clear any raw buffered input as well

        self._session._algorithm._inside_markers = False # Reset marker state after each step to allow for new code blocks in subsequent steps FIXME: Expose a proper API for this in the algorithm/session instead of reaching into internals
        self._handled_session_error = None

    # @traceable(name="IncrementalAgentSession.on_chunk", run_type="tool")
    async def on_chunk(self, chunk: str) -> None:
        self._last_response += chunk
        try:
            await self._session.apush(chunk)
        except Exception as error:
            if not self._is_handled_session_error(error):
                raise

    @traceable(name="IncrementalAgentSession.on_after_step", run_type="tool")
    async def on_after_step(self) -> None:
        try:
            if self._should_continue_streaming:
                await self._session.aflush()
        except Exception as error:
            if not self._is_handled_session_error(error):
                raise
        finally:
            self._reset_incremental_session(
                clear_buffers=not self._should_continue_streaming
            )
        
        
    def has_solution(self) -> bool:
        return self._last_response is not None and "<solution>" in self._last_response
    
    @traceable(name="IncrementalAgentSession.run")
    async def run(self) -> ExecutionResult:
        return await super().run()

class SequentialAgentSession(AgentSession):
    def __init__(self, model, context_files, question, guidelines, max_steps = 5):
        super().__init__(model, context_files, question, guidelines, max_steps)
        
        self._executor = InProcPythonExecutor()
    
    # @traceable(name="SequentialAgentSession.on_chunk", run_type="tool")
    async def on_chunk(self, chunk: str) -> None:
        self._last_response += chunk

    @traceable(name="SequentialAgentSession.on_after_step", run_type="tool")
    async def on_after_step(self) -> None:
        if "<execute>" in self._last_response:
            code_block = re.search(r"<execute>(.+?)(</execute>|$)", self._last_response, re.DOTALL)
            if code_block:
                source_code = code_block.group(1)
                result = await self._executor.aexecute(source_code)

                self._last_observation = (
                    result.output or ""
                    if result.success
                    else ("Error detected. Halting further processing. " + (result.error or ""))
                )
        


    def has_solution(self) -> bool:
        return self._last_response is not None and "<solution>" in self._last_response
    
    @traceable(name="SequentialAgentSession.run")
    async def run(self) -> ExecutionResult:
        return await super().run()
