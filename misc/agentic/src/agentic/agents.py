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
        self._last_response: str | None = None
        self._last_observation: str | None = None

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
        steps_duration = []
        start_time = perf_counter()

        while self._steps_left > 0:
            self._last_response = ""
            self._last_observation = ""

            async for chunk in self.model.astream(self._messages):
                chunk_text = chunk.content
                await self.on_chunk(chunk_text)

            await self.on_after_step()

            self._steps_left -= 1
            steps_duration.append(perf_counter() - start_time)
            if self.has_solution():
                break

            self._messages.append({"role": "assistant", "content": self._last_response})
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
            steps_duration=steps_duration,
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
        
        @self._session.on_stdout
        def on_stdout(stdout: str):
            self._last_observation += stdout
        
        @self._session.on_error
        def on_error(stderr: str):
            self._last_observation += stderr

    @traceable(name="IncrementalAgentSession.on_chunk", run_type="tool")
    async def on_chunk(self, chunk: str) -> None:
        self._last_response += chunk
        await self._session.apush(chunk)

    @traceable(name="IncrementalAgentSession.on_after_step", run_type="tool")
    async def on_after_step(self) -> None:
        await self._session.aflush()
        
        self._session._algorithm._inside_markers = False # Reset marker state after each step to allow for new code blocks in subsequent steps FIXME: Expose a proper API for this in the algorithm/session instead of reaching into internals

    def has_solution(self) -> bool:
        return self._last_response is not None and "<solution>" in self._last_response
    
    @traceable(name="IncrementalAgentSession.run")
    async def run(self) -> ExecutionResult:
        return await super().run()

class SequentialAgentSession(AgentSession):
    def __init__(self, model, context_files, question, guidelines, max_steps = 5):
        super().__init__(model, context_files, question, guidelines, max_steps)
        
        self._executor = InProcPythonExecutor()
    
    @traceable(name="SequentialAgentSession.on_chunk", run_type="tool")
    async def on_chunk(self, chunk: str) -> None:
        self._last_response += chunk

    @traceable(name="SequentialAgentSession.on_after_step", run_type="tool")
    async def on_after_step(self) -> None:
        if "<execute>" in self._last_response:
            code_block = re.search(r"<execute>(.+)(</execute>|$)", self._last_response, re.DOTALL)
            if code_block:
                source_code = code_block.group(1)
                result = await self._executor.aexecute(source_code)

                self._last_observation = (
                    result.output or ""
                    if result.success
                    else (result.error or "Error during code execution.")
                )


    def has_solution(self) -> bool:
        return self._last_response is not None and "<solution>" in self._last_response
    
    @traceable(name="SequentialAgentSession.run")
    async def run(self) -> ExecutionResult:
        return await super().run()
