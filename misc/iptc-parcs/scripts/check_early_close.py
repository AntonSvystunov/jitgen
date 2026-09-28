# One-time check: does the model server stop generating when a client closes a
# stream early (as IPTC does on its first error)? If it keeps generating, a
# request sent right after the close queues behind the abandoned generation,
# so its time to first token grows by roughly the remaining generation time.
import argparse
import asyncio
import statistics
import time

import httpx
from openai import AsyncOpenAI

LONG_PROMPT = "Count from 1 to 3000, one number per line, with no other text."
SHORT_PROMPT = "Reply with the single word: ok"


async def _ttft(client: AsyncOpenAI, model: str) -> float:
    start = time.monotonic()
    stream = await client.chat.completions.create(
        model=model,
        stream=True,
        messages=[{"role": "user", "content": SHORT_PROMPT}],
        max_tokens=64,
    )
    async with stream:
        async for chunk in stream:
            if chunk.choices and (
                chunk.choices[0].delta.content
                or getattr(chunk.choices[0].delta, "reasoning_content", None)
            ):
                return time.monotonic() - start
    return time.monotonic() - start


async def _abandoned_long_request(
    client: AsyncOpenAI, model: str, chunks: int
) -> float:
    """Start a long generation, read `chunks` chunks, close; return its token rate."""
    start = time.monotonic()
    stream = await client.chat.completions.create(
        model=model,
        stream=True,
        messages=[{"role": "user", "content": LONG_PROMPT}],
        max_tokens=4000,
    )
    seen = 0
    async with stream:
        async for _chunk in stream:
            seen += 1
            if seen >= chunks:
                break
    return seen / (time.monotonic() - start)


async def _load_single_slot(base_url: str, model: str) -> None:
    """Reload `model` with one request slot, so a continued generation would
    block the next request (with several slots it would just take another)."""
    native = base_url.removesuffix("/v1").removesuffix("/")
    async with httpx.AsyncClient(base_url=native, timeout=600) as http:
        for entry in (await http.get("/api/v1/models")).json()["models"]:
            if entry["key"] == model:
                for instance in entry.get("loaded_instances", []):
                    await http.post(
                        "/api/v1/models/unload", json={"instance_id": instance["id"]}
                    )
        response = await http.post(
            "/api/v1/models/load",
            json={"model": model, "context_length": 8192, "parallel": 1},
        )
        response.raise_for_status()


async def main(base_url: str, model: str) -> None:
    await _load_single_slot(base_url, model)
    client = AsyncOpenAI(base_url=base_url, api_key="lm-studio")
    baseline = [await _ttft(client, model) for _ in range(3)]
    after_close = []
    rate = 0.0
    for _ in range(3):
        rate = await _abandoned_long_request(client, model, chunks=20)
        after_close.append(await _ttft(client, model))
        await asyncio.sleep(1)
    remaining = 4000 / rate if rate else float("inf")
    print(f"baseline TTFT:          {statistics.median(baseline):.2f}s {baseline}")
    print(
        f"TTFT right after close: {statistics.median(after_close):.2f}s {after_close}"
    )
    print(f"a continued generation would add ~{remaining:.0f}s ({rate:.1f} chunks/s)")
    stopped = statistics.median(after_close) < statistics.median(baseline) + 5
    print("verdict:", "server stops on close" if stopped else "server KEEPS generating")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--model", default="qwen/qwen3.6-27b")
    args = parser.parse_args()
    asyncio.run(main(args.base_url, args.model))
