import asyncio

import langsmith
import pytest

from jitgen_openai._tracing import EvalTracer


async def _echo(*args: object, **kwargs: object) -> str:
    return f"{args} {kwargs}"


async def _fail() -> str:
    raise ValueError("kaboom")


async def test_disabled_tracer_passes_calls_through_without_runs(traced_runs):
    tracer = EvalTracer(None)
    tool = tracer.wrap_tool("echo", _echo)

    tracer.start_eval()
    assert await tool(1, b=2) == "(1,) {'b': 2}"
    tracer.end_eval("code", "output", None)

    assert traced_runs.all() == []


async def test_eval_run_records_code_output_and_language(traced_runs):
    async with langsmith.trace("agent", run_type="chain") as parent:
        tracer = EvalTracer(parent)
        tracer.start_eval()
        tracer.end_eval("print(1)", "1\n", None)

    [agent] = traced_runs.named("agent")
    [eval_run] = traced_runs.named("eval")
    assert eval_run["run_type"] == "tool"
    assert eval_run["parent_run_id"] == agent["id"]
    assert eval_run["inputs"] == {"code": "print(1)"}
    assert eval_run["outputs"] == {"output": "1\n"}
    assert eval_run["extra"]["metadata"]["ls_code_input_language"] == "python"


async def test_eval_run_records_execution_error(traced_runs):
    async with langsmith.trace("agent", run_type="chain") as parent:
        tracer = EvalTracer(parent)
        tracer.start_eval()
        tracer.end_eval("raise X", "execution error: X", "X")

    [eval_run] = traced_runs.named("eval")
    assert eval_run["error"] == "X"


async def test_start_eval_opens_one_run_per_turn(traced_runs):
    async with langsmith.trace("agent", run_type="chain") as parent:
        tracer = EvalTracer(parent)
        for turn in range(2):
            tracer.start_eval()
            tracer.start_eval()
            tracer.end_eval(f"turn {turn}", "", None)

    assert [run["inputs"] for run in traced_runs.named("eval")] == [
        {"code": "turn 0"},
        {"code": "turn 1"},
    ]


async def test_tool_call_is_a_child_of_the_open_eval_run(traced_runs):
    async with langsmith.trace("agent", run_type="chain") as parent:
        tracer = EvalTracer(parent)
        tool = tracer.wrap_tool("echo", _echo)
        tracer.start_eval()
        # Tools are called from the executor's worker thread, in a fresh loop.
        await asyncio.to_thread(asyncio.run, tool(1, b=2))
        tracer.end_eval("await echo(1, b=2)", "", None)

    [eval_run] = traced_runs.named("eval")
    [tool_run] = traced_runs.named("echo")
    assert tool_run["run_type"] == "tool"
    assert tool_run["parent_run_id"] == eval_run["id"]
    assert tool_run["inputs"] == {"b": 2, "args": [1]}
    assert tool_run["outputs"] == {"output": "(1,) {'b': 2}"}


async def test_tool_call_outside_an_eval_run_is_not_traced(traced_runs):
    async with langsmith.trace("agent", run_type="chain") as parent:
        tool = EvalTracer(parent).wrap_tool("echo", _echo)
        await tool()

    assert traced_runs.named("echo") == []


async def test_failing_tool_call_records_error_and_reraises(traced_runs):
    async with langsmith.trace("agent", run_type="chain") as parent:
        tracer = EvalTracer(parent)
        tool = tracer.wrap_tool("fail", _fail)
        tracer.start_eval()
        with pytest.raises(ValueError, match="kaboom"):
            await tool()
        tracer.end_eval("await fail()", "", "kaboom")

    [tool_run] = traced_runs.named("fail")
    assert "kaboom" in tool_run["error"]


async def test_eval_run_records_the_given_language(traced_runs):
    async with langsmith.trace("agent", run_type="chain") as parent:
        tracer = EvalTracer(parent, "javascript")
        tracer.start_eval()
        tracer.end_eval("console.log(1);", "1\n", None)

    [eval_run] = traced_runs.named("eval")
    assert eval_run["extra"]["metadata"]["ls_code_input_language"] == "javascript"
