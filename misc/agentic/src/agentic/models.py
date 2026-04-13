from contextlib import asynccontextmanager, suppress
from typing import AsyncIterator

from langchain_openai import ChatOpenAI

import lmstudio as lms

# FIXME: move to config
LM_STUDIO_API_HOST = "localhost:1234"
OPENAI_API_BASE = f"http://{LM_STUDIO_API_HOST}/v1"
OPENAI_API_KEY = "lmstudio"


async def _unload_model(client: lms.AsyncClient, model_name: str) -> None:
    with suppress(lms.LMStudioModelNotFoundError):
        await client.llm.unload(model_name)


async def _load_fresh_model(
    client: lms.AsyncClient, model_name: str, seed: int
) -> None:
    await _unload_model(client, model_name)
    await client.llm.load_new_instance(
        model_name,
        config=lms.LlmLoadModelConfig(
            seed=seed,
        ),
    )


@asynccontextmanager
async def get_model(
    model_name: str, temperature: float, seed: int
) -> AsyncIterator[ChatOpenAI]:
    async with lms.AsyncClient(api_host=LM_STUDIO_API_HOST) as client:
        # await _load_fresh_model(client, model_name, seed)
        llm = ChatOpenAI(
            model=model_name,
            base_url=OPENAI_API_BASE,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
            seed=seed,
            stream_usage=True,
            streaming=True,
        )
        yield llm
        # await _unload_model(client, model_name)
