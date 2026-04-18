from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

from agentic.prompts import format_human_prompt, format_system_prompt
from jitgen.executors.python import InProcPythonExecutor
from langchain_openai import ChatOpenAI

from langchain.agents import create_agent
from langchain_core.tools import tool

from langchain_core.messages import AnyMessage
from langgraph.errors import GraphRecursionError
from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.checkpoint.memory import InMemorySaver

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

    
    



class LangchainAgent:
    def __init__(self, model: ChatOpenAI, context_file_names: list[str], max_steps: int):
        self._executor = InProcPythonExecutor()
        self._max_steps = max_steps
        self._steps = 0
        
        @tool
        async def execute_code(code: str) -> str:
            """Executes Python code and returns stdout or stderr."""
            self._steps += 1
            result = await self._executor.aexecute(code)

            return (
                result.output or ""
                if result.success
                else ("Error detected. Halting further processing. " + (result.error or ""))
            )

        self._agent = create_agent(
            model=model,
            tools=[execute_code],
            system_prompt=format_system_prompt(context_file_names),
            checkpointer=InMemorySaver(),
        ).with_config({"recursion_limit": self._max_steps * 2})
    
    async def run(self, question: str, guidelines: str) -> ExecutionResult:
        is_timeout = False
        thread_id = str(uuid4())
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        with get_usage_metadata_callback() as cb:
            start_time = perf_counter()
            output_state = None
            try:
                output_state = await self._agent.ainvoke(
                    {
                        "messages": [
                            {"role": "user", "content": format_human_prompt(question, guidelines)},
                        ]
                    },
                    config=config
                )
            except GraphRecursionError:
                is_timeout = True
                
            output_state = output_state or self._agent.get_state(config=config).values
            end_time = perf_counter()
            
            model_name = next(iter(cb.usage_metadata.keys()), None)
            
            return ExecutionResult(
                total_execution_time=end_time - start_time,
                messages=output_state["messages"],
                success=not is_timeout,
                timeout=is_timeout,
                steps_taken=self._steps + (1 if not is_timeout else 0),
                max_steps=self._max_steps,
                output=output_state["messages"][-1].content if not is_timeout else None,
                model_name=model_name,
                total_tokens=cb.usage_metadata[model_name]["total_tokens"] if model_name in cb.usage_metadata else None,
                input_tokens=cb.usage_metadata[model_name]["input_tokens"] if model_name in cb.usage_metadata else None,
                output_tokens=cb.usage_metadata[model_name]["output_tokens"] if model_name in cb.usage_metadata else None,
                reasoning_tokens=cb.usage_metadata[model_name]["output_token_details"]["reasoning"] if model_name in cb.usage_metadata else None,
            )