# LLM backends, adapted from misc/mbpp/mbpp/providers.py: how to reach each one,
# how to reset local model residency between runs, and provider-specific options.
import os
from abc import ABC, abstractmethod
from typing import Any

import httpx
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI

from iptc_parcs.metrics import TurnRecord


class Provider(ABC):
    """An OpenAI-compatible LLM backend."""

    name: str

    @abstractmethod
    def build_client(self) -> AsyncOpenAI:
        """Build a LangSmith-wrapped chat-completions client."""

    def request_options(self) -> dict[str, Any]:
        """Extra arguments for every model call: ask for a final usage chunk."""
        return {"stream_options": {"include_usage": True}}

    async def prepare(self, model: str, context_length: int) -> None:
        """Bring `model` to a fresh state before a run. No-op by default."""

    async def fill_native_usage(self, turns: list[TurnRecord]) -> None:
        """Add server-side token counts to `turns`. No-op by default."""


class LMStudioProvider(Provider):
    """A local LM Studio server (native REST API for model residency)."""

    name = "lmstudio"

    def __init__(
        self, *, base_url: str = "http://localhost:1234/v1", api_key: str = "lm-studio"
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key

    def build_client(self) -> AsyncOpenAI:
        return wrap_openai(AsyncOpenAI(base_url=self._base_url, api_key=self._api_key))

    async def prepare(self, model: str, context_length: int) -> None:
        """Reload `model` so no run inherits another's prompt cache.

        The context length must be explicit: LM Studio's default (8192) is
        too small for a reasoning model's multi-turn run.
        """
        native_url = self._base_url.removesuffix("/v1").removesuffix("/")
        async with httpx.AsyncClient(base_url=native_url, timeout=60.0) as http:
            response = await http.get("/api/v1/models")
            response.raise_for_status()
            for entry in response.json()["models"]:
                if entry["key"] != model:
                    continue
                for instance in entry.get("loaded_instances", []):
                    await http.post(
                        "/api/v1/models/unload", json={"instance_id": instance["id"]}
                    )
            response = await http.post(
                "/api/v1/models/load",
                # One slot: the agents send one request at a time.
                json={"model": model, "context_length": context_length, "parallel": 1},
                timeout=600.0,
            )
            response.raise_for_status()


class OpenRouterProvider(Provider):
    """OpenRouter; the API key is read lazily from the environment."""

    name = "openrouter"

    def __init__(
        self,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
    ) -> None:
        self._base_url = base_url
        self._api_key_env = api_key_env

    def _api_key(self) -> str:
        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            msg = f"{self._api_key_env} is not set"
            raise RuntimeError(msg)
        return api_key

    def build_client(self) -> AsyncOpenAI:
        return wrap_openai(
            AsyncOpenAI(base_url=self._base_url, api_key=self._api_key())
        )

    def request_options(self) -> dict[str, Any]:
        # OpenRouter always sends usage in the final chunk.
        return {}

    async def fill_native_usage(self, turns: list[TurnRecord]) -> None:
        """Add OpenRouter's per-generation counts, which include closed streams."""
        generated = [turn for turn in turns if turn.generation_id]
        if not generated:
            return
        headers = {"Authorization": f"Bearer {self._api_key()}"}
        async with httpx.AsyncClient(
            base_url=self._base_url, timeout=60.0, headers=headers
        ) as http:
            for turn in generated:
                response = await http.get(
                    "/generation", params={"id": turn.generation_id}
                )
                if response.status_code != 200:
                    continue
                data = response.json().get("data") or {}
                turn.native_input = data.get("native_tokens_prompt")
                turn.native_output = data.get("native_tokens_completion")
                turn.native_reasoning = data.get("native_tokens_reasoning")


LMSTUDIO = LMStudioProvider()
OPENROUTER = OpenRouterProvider()
PROVIDERS: dict[str, Provider] = {p.name: p for p in (LMSTUDIO, OPENROUTER)}
