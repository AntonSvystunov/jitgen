from dataclasses import dataclass
from typing import Literal
from uuid import UUID
from langchain_core.messages import AnyMessage


RunType = Literal["incremental", "sequential", "sequential_langchain"]

@dataclass
class RunInfo:
    id: UUID
    model_name: str
    temperature: float
    seed: int
    type: RunType


@dataclass
class TaskRunStats:
    task_id: str
    run_id: UUID

    messages: list[AnyMessage]
    
    total_tool_calls_count: int
    
    successful_tool_calls_count: int
    failed_tool_calls_count: int
    
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    total_reasoning_tokens: int
    
    final_answer: str | None
    final_answer_parsed: str | None
    
    has_timeout: bool
    
    total_execution_time: float
    

@dataclass
class StepLogEntry:
    task_id: str
    run_id: UUID
    step_number: int
    
    message: str | None
    code: str | None
    observation: str | None
    
    is_error: bool
    
    tool_call_success: bool | None
    tool_call_error_message: str | None
    
    total_time: float
    inference_time: float | None
    tool_execution_time: float | None

@dataclass
class TaskRunResult:
    run_info: RunInfo
    messages: list[AnyMessage]
    task_run_stats: TaskRunStats
    steps: list[StepLogEntry]




class ResultsLogger:
    def __init__(self, id: UUID, model_name: str, temperature: float, seed: int, type: RunType):
        self.run_info = RunInfo(
            id=id,
            model_name=model_name,
            temperature=temperature,
            seed=seed,
            type=type
        )
        self.messages: list[AnyMessage] = []
        self.task_run_stats: TaskRunStats | None = None
        self.steps: list[StepLogEntry] = []
    
    
    def set_messages(self, messages: list[AnyMessage]) -> None:
        self.messages = messages
        
    def set_task_run_stats(
        self,
        total_tool_calls_count: int,
        successful_tool_calls_count: int,
        failed_tool_calls_count: int,
        total_tokens: int,
        total_input_tokens: int,
        total_output_tokens: int,
        total_reasoning_tokens: int,
        final_answer: str | None,
        final_answer_parsed: str | None,
        has_timeout: bool,
        total_execution_time: float
    ) -> None:
        self.task_run_stats = TaskRunStats(
            task_id="",
            run_id=self.run_info.id,
            messages=self.messages,
            total_tool_calls_count=total_tool_calls_count,
            successful_tool_calls_count=successful_tool_calls_count,
            failed_tool_calls_count=failed_tool_calls_count,
            total_tokens=total_tokens,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_reasoning_tokens=total_reasoning_tokens,
            final_answer=final_answer,
            final_answer_parsed=final_answer_parsed,
            has_timeout=has_timeout,
            total_execution_time=total_execution_time
        )
    
    def add_step(
        self,
        step_number: int,
        message: str | None,
        code: str | None,
        observation: str | None,
        is_error: bool,
        tool_call_success: bool | None,
        tool_call_error_message: str | None,
        total_time: float,
        tool_execution_time: float | None,
        inference_time: float | None = None
    ) -> None:
        step_entry = StepLogEntry(
            task_id="",
            run_id=self.run_info.id,
            step_number=step_number,
            message=message,
            code=code,
            observation=observation,
            is_error=is_error,
            tool_call_success=tool_call_success,
            tool_call_error_message=tool_call_error_message,
            total_time=total_time,
            tool_execution_time=tool_execution_time,
            inference_time=inference_time
        )
        self.steps.append(step_entry)
        
    def get_result(self) -> TaskRunResult:
        if self.task_run_stats is None:
            raise ValueError("Task run stats have not been set yet.")
        
        return TaskRunResult(
            run_info=self.run_info,
            messages=self.messages,
            task_run_stats=self.task_run_stats,
            steps=self.steps
        )
    
    

