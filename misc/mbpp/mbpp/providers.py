import os
from abc import ABC, abstractmethod

import httpx
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI


class ProviderUnavailableError(Exception):
    """A provider can't run right now (e.g. a required API key isn't set).

    Raised from `Provider.build_client`/`build_native_client` rather than at
    construction time, so a `ModelSpec` referencing an unconfigured provider
    can sit in a registry without breaking every other model's run — the
    orchestration loop catches this and skips just that one model.
    """


class Provider(ABC):
    """An LLM backend: how to reach it, and how to control model residency around it.

    `prepare`/`release` bracket one strategy pass: `prepare` runs before a
    pass starts, `release` after it ends. For a local server this reloads
    the model fresh so no pass inherits another's KV-cache state; for a
    hosted provider with no controllable residency, both are no-ops.
    """

    name: str
    needs_warmup: bool = True

    @property
    @abstractmethod
    def base_url(self) -> str:
        """The OpenAI-compatible `/v1`-style base URL for chat completions."""

    @property
    def native_base_url(self) -> str:
        """Base URL for this provider's own REST API (residency, listing).

        Defaults to `base_url` unchanged — correct for a hosted provider
        whose model-listing endpoint lives under the same `/v1` base as
        chat completions (OpenRouter). Local servers whose native API hangs
        off the bare server root instead (Ollama, LM Studio) override this.
        """
        return self.base_url

    @abstractmethod
    def build_client(self) -> AsyncOpenAI:
        """Build the chat-completions client for this provider."""

    def build_native_client(self) -> httpx.AsyncClient:
        """Build an httpx client scoped to this provider's native REST API."""
        return httpx.AsyncClient(base_url=self.native_base_url, timeout=60.0)

    def stream_options(self) -> dict[str, bool] | None:
        """Streaming request option asking for a final usage-only chunk.

        `None` for a provider where the parameter is rejected, or a
        confirmed no-op because usage is always included regardless
        (OpenRouter).
        """
        return {"include_usage": True}

    @abstractmethod
    async def available_model_ids(self, http: httpx.AsyncClient) -> set[str]:
        """List model identifiers this provider currently knows about.

        Args:
            http: Client scoped to `native_base_url`.
        """

    async def verify_model_present(self, http: httpx.AsyncClient, model: str) -> None:
        """Fail loudly before running anything if `model` isn't available.

        Implemented once here in terms of `available_model_ids` so every
        provider reports a missing model the same way.

        Args:
            http: Client scoped to `native_base_url`.
            model: The model identifier the run is configured to use.

        Raises:
            RuntimeError: `model` is not among the endpoint's known models.
        """
        available = await self.available_model_ids(http)
        if model not in available:
            msg = (
                f"configured model {model!r} not found on {self.name!r}; "
                f"available: {sorted(available)}"
            )
            raise RuntimeError(msg)

    async def prepare(self, http: httpx.AsyncClient, model: str) -> None:
        """Bring `model` to a known-fresh state before a pass starts. No-op by default."""

    async def release(self, http: httpx.AsyncClient, model: str) -> None:
        """Release `model`'s resources after a pass ends. No-op by default."""


class OllamaProvider(Provider):
    """A local Ollama server, reached over its OpenAI-compatible endpoint."""

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def native_base_url(self) -> str:
        return self._base_url.removesuffix("/v1").removesuffix("/")

    def build_client(self) -> AsyncOpenAI:
        return wrap_openai(AsyncOpenAI(base_url=self._base_url, api_key=self._api_key))

    async def available_model_ids(self, http: httpx.AsyncClient) -> set[str]:
        response = await http.get("/api/tags")
        response.raise_for_status()
        return {entry["name"] for entry in response.json()["models"]}

    async def prepare(self, http: httpx.AsyncClient, model: str) -> None:
        """Unload any resident copy of `model`, then load a fresh one with no carried-over KV cache."""
        await http.post("/api/generate", json={"model": model, "keep_alive": 0})
        await http.post(
            "/api/generate", json={"model": model, "prompt": "", "keep_alive": "30m"}
        )

    async def release(self, http: httpx.AsyncClient, model: str) -> None:
        await http.post("/api/generate", json={"model": model, "keep_alive": 0})


class LMStudioProvider(Provider):
    """A local LM Studio server, reached over its OpenAI-compatible endpoint.

    Uses LM Studio's native `/api/v1/models` REST surface (not the older
    `/api/v0/models`, which reports a bare `state` field but no
    `loaded_instances` — `/api/v1/models` is what actually exposes the
    `instance_id` values `/api/v1/models/unload` needs), confirmed live
    against a running LM Studio server.
    """

    name = "lmstudio"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def native_base_url(self) -> str:
        return self._base_url.removesuffix("/v1").removesuffix("/")

    def build_client(self) -> AsyncOpenAI:
        return wrap_openai(AsyncOpenAI(base_url=self._base_url, api_key=self._api_key))

    async def _model_entry(
        self, http: httpx.AsyncClient, model: str
    ) -> dict[str, object] | None:
        response = await http.get("/api/v1/models")
        response.raise_for_status()
        for entry in response.json()["models"]:
            if entry["key"] == model:
                return entry
        return None

    async def available_model_ids(self, http: httpx.AsyncClient) -> set[str]:
        response = await http.get("/api/v1/models")
        response.raise_for_status()
        return {entry["key"] for entry in response.json()["models"]}

    async def _unload_all_instances(self, http: httpx.AsyncClient, model: str) -> None:
        entry = await self._model_entry(http, model)
        if entry is None:
            return
        for instance in entry["loaded_instances"]:  # type: ignore[index]
            await http.post(
                "/api/v1/models/unload", json={"instance_id": instance["id"]}
            )

    async def prepare(self, http: httpx.AsyncClient, model: str) -> None:
        """Unload any resident instance of `model`, then load a fresh one."""
        await self._unload_all_instances(http, model)
        await http.post("/api/v1/models/load", json={"model": model})

    async def release(self, http: httpx.AsyncClient, model: str) -> None:
        await self._unload_all_instances(http, model)


class OpenRouterProvider(Provider):
    """A hosted OpenAI-compatible provider with no local model residency.

    The API key is resolved lazily (inside `build_client`/
    `build_native_client`, not `__init__`) so a `ModelSpec` referencing this
    provider can sit in a registry unexercised without an API key present —
    it only raises `ProviderUnavailableError` when actually used.
    """

    name = "openrouter"
    needs_warmup = False

    def __init__(
        self,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
    ) -> None:
        self._base_url = base_url
        self._api_key_env = api_key_env

    @property
    def base_url(self) -> str:
        return self._base_url

    def _api_key(self) -> str:
        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            msg = f"{self._api_key_env} is not set"
            raise ProviderUnavailableError(msg)
        return api_key

    def build_client(self) -> AsyncOpenAI:
        return wrap_openai(
            AsyncOpenAI(base_url=self._base_url, api_key=self._api_key())
        )

    def build_native_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.native_base_url,
            timeout=60.0,
            headers={"Authorization": f"Bearer {self._api_key()}"},
        )

    def stream_options(self) -> dict[str, bool] | None:
        # Confirmed via OpenRouter's usage-accounting docs: stream_options
        # is a deprecated no-op there -- usage (including reasoning tokens)
        # is always included in the final chunk regardless.
        return None

    async def available_model_ids(self, http: httpx.AsyncClient) -> set[str]:
        response = await http.get("/models")
        response.raise_for_status()
        return {entry["id"] for entry in response.json()["data"]}


OLLAMA = OllamaProvider()
LMSTUDIO = LMStudioProvider()
OPENROUTER = OpenRouterProvider()
