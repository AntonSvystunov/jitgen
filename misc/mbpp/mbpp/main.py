import asyncio

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .dataset import IndexedCase, load_indexed_dataset
from .execution import run_case, warm_up
from .models import MODELS, ModelSpec
from .prompts import render
from .providers import ProviderUnavailableError
from .reporting import print_result
from .results import (
    Strategy,
    assert_unique_csv_stems,
    csv_path_for,
    to_csv_row,
    write_csv,
)
from .run_types import RunConfig, RunResult

# Fixed so decoding is deterministic: same model, same seed, same
# temperature, same prompt should yield the same completion, so any
# difference between strategies is attributable to execution strategy, not
# sampling noise.
SEED = 42
TEMPERATURE = 0.0
ROWS_TO_RUN = 113


async def _run_pass(
    spec: ModelSpec,
    client: AsyncOpenAI,
    http: httpx.AsyncClient,
    config: RunConfig,
    strategy: Strategy,
    rendered: list[tuple[IndexedCase, list[dict[str, str]]]],
) -> list[RunResult]:
    """Run every case for one (model, strategy) pass, bracketed by prepare/release.

    Args:
        spec: The model under test.
        client: The chat completions client to run cases through.
        http: Client scoped to `spec.provider.native_base_url`.
        config: The pinned run configuration for this model.
        strategy: Which execution strategy this pass measures.
        rendered: Every case paired with its rendered messages.

    Returns:
        One `RunResult` per case, in row order.
    """
    print(f"--- pass: {strategy} ---")
    await spec.provider.prepare(http, spec.model)
    pass_results: list[RunResult] = []
    try:
        if spec.provider.needs_warmup:
            warmup_case, warmup_messages = rendered[0]
            await warm_up(
                client,
                config,
                warmup_case,
                warmup_messages,
                strategy,
                spec.provider.stream_options(),
            )
        for case, messages in rendered:
            result = await run_case(
                client, config, case, messages, strategy, spec.provider.stream_options()
            )
            pass_results.append(result)
            print_result(case, result)
    finally:
        await spec.provider.release(http, spec.model)
    return pass_results


async def run_evaluation() -> None:
    """Compare incremental vs. sequential execution over every configured model.

    Each model's strategy runs as one full pass over every row before the
    other strategy starts — never interleaved — with the model unloaded and
    reloaded fresh at the start of each pass and released at the end, so
    neither pass inherits KV-cache state or load effects from the other.
    Each (model, strategy) pass writes its own CSV under `misc/mbpp/results/`.
    """
    assert_unique_csv_stems(MODELS)

    indexed_cases = load_indexed_dataset()[:ROWS_TO_RUN]
    # Rendered once and reused verbatim by both passes: there is no
    # per-strategy templating path that could drift.
    rendered = [(case, render(case.example)) for case in indexed_cases]

    for spec in MODELS:
        try:
            client = spec.provider.build_client()
        except ProviderUnavailableError as exc:
            print(f"skipping {spec.label}: {exc}\n")
            continue

        async with spec.provider.build_native_client() as http:
            await spec.provider.verify_model_present(http, spec.model)
            config = RunConfig(model=spec.model, seed=SEED, temperature=TEMPERATURE)
            print(f"=== model: {spec.label} ({spec.provider.name}) ===")

            for strategy in (Strategy.INCREMENTAL, Strategy.SEQUENTIAL):
                pass_results = await _run_pass(
                    spec, client, http, config, strategy, rendered
                )
                write_csv(
                    csv_path_for(spec, strategy),
                    [to_csv_row(result, spec) for result in pass_results],
                )
                print()


def main() -> None:
    """Entry point for the MBPP evaluation script."""
    load_dotenv()
    asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()
