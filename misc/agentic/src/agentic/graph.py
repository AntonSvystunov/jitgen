"""
CodeAct Agent LangGraph definitions.

Two execution strategies sharing a common looping graph structure:
- sync:  invoke LLM → extract code from <execute> tags → run via InProcPythonExecutor
- async: stream LLM through JITGenParser (<execute> markers) for incremental execution

Graph structure::

    START → agent_loop ⟲ [should_continue?]
                            ├─ "agent_loop" → agent_loop  (loop)
                            └─ END                         (done)

Each iteration of ``agent_loop``:

1. Calls the LLM (invoke for sync, stream for async)
2. Appends the AIMessage to state
3. If an ``<execute>`` block is present, executes the code and appends an
   observation (``HumanMessage``) so the next iteration sees the result
4. If no ``<execute>`` block the model gave a final answer; routing ends the graph

The loop terminates when:

- ``execution_count >= max_execution_turns``
- The LLM response contains no ``<execute>`` block
- A timeout occurs
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from jitgen.executors.python import InProcPythonExecutor
from jitgen.prebuilt.python import create_python_jitgen
from jitgen_langchain.parser import JITGenParser

from .config import AgenticEvaluationConfig
from .state import AgentState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXECUTE_RE = re.compile(r"<execute>(.*?)(?:</execute>|$)", re.DOTALL)


def _extract_code(text: str) -> str | None:
    """Return code inside the first ``<execute>…</execute>`` block, or *None*."""
    m = _EXECUTE_RE.search(text)
    if m:
        code = m.group(1).strip()
        return code or None
    return None


# ---------------------------------------------------------------------------
# Sync agent-loop node
# ---------------------------------------------------------------------------


def _make_sync_agent_loop(
    llm: BaseChatModel, cfg: AgenticEvaluationConfig
):
    """Return the **sync** ``agent_loop`` node.

    Each call:

    1. ``llm.ainvoke`` → full AIMessage
    2. Extract code from ``<execute>`` tags
    3. Execute via ``InProcPythonExecutor``
    4. Append observation as ``HumanMessage``
    """

    async def sync_agent_loop(state: AgentState) -> dict:
        # --- 1. LLM inference ---
        response: AIMessage = await llm.ainvoke(state["messages"])
        content: str = response.content or ""

        # The stop token strips </execute>; restore it so the regex works
        if "<execute>" in content and "</execute>" not in content:
            content += "</execute>"

        new_messages: list[BaseMessage] = [AIMessage(content=content)]

        # --- 2. Extract code ---
        code = _extract_code(content)
        if code is None:
            # No code block → agent gave a final answer
            return {
                "messages": new_messages,
                "execution_count": state["execution_count"],
                "final_output": state.get("final_output", ""),
                "has_error": False,
                "has_timed_out": False,
                "error_message": None,
            }

        # --- 3. Execute ---
        executor = InProcPythonExecutor()
        result = await executor.aexecute(code, timeout=cfg.test_case_timeout)

        exec_count = state["execution_count"] + 1
        output = result.output or ""
        error_msg = result.error

        if result.success:
            feedback = f"Execution Output:\n{output}\n\n(Code executed successfully. If the output looks correct for the given input, provide your final answer without <execute> tags.)"
        elif result.has_timed_out:
            feedback = "Execution Output:\nError: Code execution timed out."
            error_msg = "Timeout"
        else:
            feedback = f"Execution Output:\nError: {error_msg}"

        # --- 4. Append observation ---
        new_messages.append(HumanMessage(content=feedback))

        return {
            "messages": new_messages,
            "execution_count": exec_count,
            "final_output": output,
            "has_error": not result.success,
            "has_timed_out": result.has_timed_out,
            "error_message": error_msg,
        }

    return sync_agent_loop


# ---------------------------------------------------------------------------
# Async (JITGen) agent-loop node
# ---------------------------------------------------------------------------


def _make_async_agent_loop(
    llm: BaseChatModel, cfg: AgenticEvaluationConfig
):
    """Return the **async** ``agent_loop`` node.

    Each call:

    1. Stream LLM output, capturing raw text
    2. Pipe the stream through ``JITGenParser`` (``<execute>``/``</execute>``
       markers) so JITGen parses and executes code incrementally
    3. Collect execution stdout
    4. Append AIMessage (raw LLM text) + observation (``HumanMessage``) to state
    """

    async def async_agent_loop(state: AgentState) -> dict:
        # JITGen parser configured with <execute> markers
        jitgen_parser = JITGenParser(
            jit_gen=create_python_jitgen(),
            start_marker="<execute>",
            end_marker="</execute>",
        )

        # --- 1. Stream LLM, tee-ing raw text while forwarding to JITGen ---
        raw_chunks: list[str] = []

        async def _capture_and_forward() -> AsyncIterator[BaseMessage]:
            """Yield LLM chunks for JITGen while collecting raw text."""
            async for chunk in llm.astream(state["messages"]):
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                raw_chunks.append(text)
                yield chunk

        # --- 2-3. JITGen incremental execution ---
        execution_outputs: list[str] = []
        has_error = False
        error_msg: str | None = None

        try:
            async for output_chunk in jitgen_parser._atransform(
                _capture_and_forward()
            ):
                if output_chunk:
                    execution_outputs.append(output_chunk)
        except ValueError as exc:
            has_error = True
            error_msg = str(exc)

        # --- 4. Reconstruct raw LLM response & build messages ---
        raw_content = "".join(raw_chunks)
        if "<execute>" in raw_content and "</execute>" not in raw_content:
            raw_content += "</execute>"

        code = _extract_code(raw_content)
        output = "".join(execution_outputs)

        new_messages: list[BaseMessage] = [AIMessage(content=raw_content)]

        if code is None:
            # No code block → agent gave a final answer
            return {
                "messages": new_messages,
                "execution_count": state["execution_count"],
                "final_output": state.get("final_output", ""),
                "has_error": False,
                "has_timed_out": False,
                "error_message": None,
            }

        exec_count = state["execution_count"] + 1

        if has_error:
            feedback = f"Execution Output:\nError: {error_msg}"
        else:
            feedback = f"Execution Output:\n{output}\n\n(Code executed successfully. If the output looks correct for the given input, provide your final answer without <execute> tags.)"

        new_messages.append(HumanMessage(content=feedback))

        return {
            "messages": new_messages,
            "execution_count": exec_count,
            "final_output": output,
            "has_error": has_error,
            "has_timed_out": False,
            "error_message": error_msg,
        }

    return async_agent_loop


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _make_route_after_loop(cfg: AgenticEvaluationConfig):
    """Return the routing function that decides loop vs END."""

    def _route(state: AgentState) -> str:
        last_msg = state["messages"][-1]

        # Last message is AIMessage → no observation appended → no code → done
        if isinstance(last_msg, AIMessage):
            return END

        # Code was executed; check termination conditions
        if state["execution_count"] >= cfg.max_execution_turns:
            return END
        if state.get("has_timed_out", False):
            return END

        # Otherwise loop back for the next agent turn
        return "agent_loop"

    return _route


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def create_agent_graph(
    llm: BaseChatModel,
    strategy: Literal["sync", "async"],
    eval_config: AgenticEvaluationConfig,
) -> CompiledStateGraph:
    """Build and compile a CodeAct agent evaluation graph.

    Parameters
    ----------
    llm:
        The chat model (e.g. ``ChatOllama`` wrapping codeact-agent-mistral).
    strategy:
        ``"sync"``  – invoke LLM, extract and execute code in one shot.
        ``"async"`` – stream LLM through JITGenParser for incremental execution.
    eval_config:
        Evaluation configuration (timeouts, max turns, …).
    """
    workflow = StateGraph(AgentState)

    # Single looping node — strategy determines the implementation
    if strategy == "sync":
        workflow.add_node(
            "agent_loop", _make_sync_agent_loop(llm, eval_config)
        )
    else:
        workflow.add_node(
            "agent_loop", _make_async_agent_loop(llm, eval_config)
        )

    # Entry
    workflow.set_entry_point("agent_loop")

    # Conditional edge: loop back or end
    workflow.add_conditional_edges(
        "agent_loop",
        _make_route_after_loop(eval_config),
        {
            "agent_loop": "agent_loop",
            END: END,
        },
    )

    return workflow.compile()
