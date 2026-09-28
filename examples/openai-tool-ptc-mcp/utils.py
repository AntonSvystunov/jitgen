# Shared setup for the MCP agent examples: client, progress printing, and the
# agent run itself.
import argparse
import io
import os
import sys
from collections.abc import Mapping
from typing import Any, get_args

from dotenv import load_dotenv
from jitgen_openai import CodeLanguage, IptcAgent, McpToolBridge, PtcAgent
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI

_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_RED = "\033[31m"
_RESET = "\033[0m"


def build_client() -> AsyncOpenAI:
    """Build an OpenAI client, defaulting to a local LM Studio server.

    LangSmith tracing is a no-op unless its environment variables are set.

    Returns:
        An `AsyncOpenAI` client wrapped for LangSmith tracing.
    """
    return wrap_openai(
        AsyncOpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "lm-studio"),
        )
    )


def require_model() -> str:
    """Read the model identifier from `OPENAI_MODEL`.

    Returns:
        The configured model identifier.

    Raises:
        RuntimeError: If `OPENAI_MODEL` is unset or empty.
    """
    model = os.environ.get("OPENAI_MODEL", "")
    if not model:
        msg = "set OPENAI_MODEL to the identifier of a tool-calling model"
        raise RuntimeError(msg)
    return model


def add_language_argument(parser: argparse.ArgumentParser) -> None:
    """Add the `--language` option shared by the example scripts.

    Args:
        parser: The script's argument parser.
    """
    parser.add_argument(
        "--language",
        choices=get_args(CodeLanguage),
        default="python",
        help="the language the model writes its `eval` code in",
    )


def _print_delta(text: str) -> None:
    print(text, end="", flush=True)


def _print_output(output: str) -> None:
    print(
        f"\n{_CYAN}--- jitgen output ---\n{output}--- end output ---{_RESET}\n",
        flush=True,
    )


def _print_tool_error(result: str) -> None:
    if result.startswith("execution error:"):
        print(f"\n{_RED}--- {result} ---{_RESET}\n", flush=True)


async def run_agent(
    bridge: McpToolBridge,
    task: str,
    *,
    system_prompt: str,
    agent_cls: type[IptcAgent] | type[PtcAgent] = IptcAgent,
    language: CodeLanguage = "python",
    max_turns: int = 8,
    executor_timeout: float = 60.0,
    completion_options: Mapping[str, Any] | None = None,
) -> str:
    """Solve `task` with an agent over `bridge`'s tools, printing progress.

    Args:
        bridge: An unstarted bridge; it is started and closed here.
        task: The user's task.
        system_prompt: The agent's system message.
        agent_cls: `IptcAgent`, or `PtcAgent` for the non-incremental baseline.
        language: The language the model writes its `eval` code in; must match
            what `system_prompt` asks for.
        max_turns: Model round trips allowed before giving up.
        executor_timeout: Seconds allowed per executed statement.
        completion_options: Extra arguments for every model call.

    Returns:
        The model's final answer.
    """
    load_dotenv()
    if isinstance(sys.stdout, io.TextIOWrapper):
        # Model output can hold characters the console encoding lacks (e.g.
        # cp1251 when redirected on Windows); don't crash on them.
        sys.stdout.reconfigure(errors="replace")
    client = build_client()
    model = require_model()
    async with bridge:
        agent = agent_cls(
            client,
            model,
            bridges=[bridge],
            system_prompt=system_prompt,
            language=language,
            max_turns=max_turns,
            executor_timeout=executor_timeout,
            on_delta=_print_delta,
            on_output=_print_output,
            on_tool_result=_print_tool_error,
            completion_options=completion_options,
        )
        answer = await agent.arun(task)
    print(f"\n{_MAGENTA}--- final answer ---{_RESET}\n{answer}")
    return answer
