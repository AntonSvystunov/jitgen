import json
from collections.abc import Awaitable, Callable
from typing import Any

from iptc_parcs.config import Scenario

TRANSIENT_COMPILE_ERROR = json.dumps(
    {
        "error": "Compilation failed: compilation service interrupted; "
        "resubmit the same source"
    },
    separators=(",", ":"),
)

Tool = Callable[..., Awaitable[str]]


class FaultInjector:
    """Decides, per tool call, whether to return an injected fault instead.

    `fault_compile` replaces only the first `create_session` call of a run with
    a transient compile error; every later call reaches the real server.

    Args:
        scenario: The experiment scenario.
    """

    def __init__(self, scenario: Scenario) -> None:
        self._scenario = scenario
        self._fired = False

    def fault_for(self, name: str) -> str | None:
        """Return the injected reply for this call, or `None` to pass through."""
        if (
            self._scenario is Scenario.FAULT_COMPILE
            and name == "create_session"
            and not self._fired
        ):
            self._fired = True
            return TRANSIENT_COMPILE_ERROR
        return None


def parse_reply(reply: str) -> dict[str, Any] | None:
    """Decode a tool's JSON reply, or `None` if it isn't a JSON object."""
    try:
        data = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None
