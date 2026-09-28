from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import langsmith
import pytest
from langsmith import Client


class TracedRuns:
    """Records the runs a mocked LangSmith client receives."""

    def __init__(self, client: MagicMock) -> None:
        self.client = client

    def all(self) -> list[dict[str, Any]]:
        """Every run, in creation order, with its later updates merged in."""
        runs: dict[Any, dict[str, Any]] = {}
        for method, _, kwargs in self.client.method_calls:
            if method == "create_run":
                runs[kwargs["id"]] = dict(kwargs)
            elif method == "update_run":
                updates = {k: v for k, v in kwargs.items() if v is not None}
                runs[kwargs["run_id"]].update(updates)
        return list(runs.values())

    def named(self, name: str) -> list[dict[str, Any]]:
        """Every run called `name`, in creation order."""
        return [run for run in self.all() if run["name"] == name]


@pytest.fixture
def traced_runs() -> Iterator[TracedRuns]:
    """Enable LangSmith tracing against a mocked client for one test."""
    client = MagicMock(spec=Client)
    with langsmith.tracing_context(enabled=True, client=client):
        yield TracedRuns(client)
