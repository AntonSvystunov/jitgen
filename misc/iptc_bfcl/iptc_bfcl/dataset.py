import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DATASET_URL = (
    "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard"
)
# Pinned so every experiment grades against the same questions and answers.
BFCL_REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
# The single-turn categories graded by BFCL's AST checker. `irrelevance` is left
# out: every arm forces a tool call on its first turn.
CATEGORIES = (
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
)
DEFAULT_DATA_DIR = Path.home() / ".cache" / "iptc-bfcl" / BFCL_REVISION


@dataclass(frozen=True)
class BfclEntry:
    """One BFCL question with its functions and accepted answers.

    Attributes:
        id: BFCL's entry id, e.g. `parallel_12`.
        category: The category the entry comes from.
        system: The entry's own system message, empty when it has none.
        user: The user's request.
        functions: The BFCL function docs, exactly as in the dataset.
        ground_truth: One `{name: {param: [accepted values]}}` per expected call.
    """

    id: str
    category: str
    system: str
    user: str
    functions: tuple[dict[str, Any], ...]
    ground_truth: tuple[dict[str, Any], ...]


def question_path(category: str) -> str:
    """The dataset path of `category`'s questions."""
    return f"BFCL_v3_{category}.json"


def answer_path(category: str) -> str:
    """The dataset path of `category`'s accepted answers."""
    return f"possible_answer/BFCL_v3_{category}.json"


def ensure_downloaded(
    categories: list[str],
    data_dir: Path,
    *,
    client: httpx.Client | None = None,
) -> None:
    """Download the questions and answers of `categories` unless cached.

    Args:
        categories: The categories to fetch.
        data_dir: Where the files are cached, mirroring the dataset's layout.
        client: The HTTP client to use; a fresh one when `None`.

    Raises:
        httpx.HTTPStatusError: If a file can't be downloaded.
    """
    paths = [path for c in categories for path in (question_path(c), answer_path(c))]
    missing = [path for path in paths if not (data_dir / path).exists()]
    if not missing:
        return
    http = client or httpx.Client(follow_redirects=True, timeout=120.0)
    try:
        for path in missing:
            response = http.get(f"{DATASET_URL}/resolve/{BFCL_REVISION}/{path}")
            response.raise_for_status()
            target = data_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(response.content)
    finally:
        if client is None:
            http.close()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    # The dataset's `.json` files hold one JSON object per line.
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_entry(
    category: str, question: dict[str, Any], answer: dict[str, Any]
) -> BfclEntry:
    """Build an entry from one question row and its answer row.

    Args:
        category: The category both rows come from.
        question: The question row (`id`, `question`, `function`).
        answer: The answer row (`id`, `ground_truth`).

    Returns:
        The entry.

    Raises:
        ValueError: If the question has more than one turn, or no user message.
    """
    turns = question["question"]
    if len(turns) != 1:
        msg = f"{question['id']}: expected one turn, got {len(turns)}"
        raise ValueError(msg)
    messages = turns[0]
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    users = [m["content"] for m in messages if m["role"] == "user"]
    if not users:
        msg = f"{question['id']}: no user message"
        raise ValueError(msg)
    return BfclEntry(
        id=question["id"],
        category=category,
        system=system,
        user="\n\n".join(users),
        functions=tuple(question["function"]),
        ground_truth=tuple(answer["ground_truth"]),
    )


def load_entries(category: str, data_dir: Path) -> list[BfclEntry]:
    """Load every entry of `category`, joining questions with their answers.

    Args:
        category: The category to load.
        data_dir: The cache `ensure_downloaded` filled.

    Questions and answers are matched by `id`. A question whose id has no
    answer takes the answer on the same line, provided no question claims
    that answer's id: the pinned revision has one such typo
    (`live_multiple_1052-79-0` against `live_multiple_1052-279-0`).

    Returns:
        The entries, in dataset order.

    Raises:
        ValueError: If a question has no answer.
    """
    questions = _read_jsonl(data_dir / question_path(category))
    answer_rows = _read_jsonl(data_dir / answer_path(category))
    answers = {row["id"]: row for row in answer_rows}
    question_ids = {question["id"] for question in questions}
    entries = []
    for line, question in enumerate(questions):
        answer = answers.get(question["id"])
        if answer is None and line < len(answer_rows):
            same_line = answer_rows[line]
            if same_line["id"] not in question_ids:
                answer = same_line
        if answer is None:
            msg = f"{question['id']}: no possible answer"
            raise ValueError(msg)
        entries.append(parse_entry(category, question, answer))
    return entries


def select_entries(
    entries: list[BfclEntry],
    *,
    limit: int | None,
    sample_seed: int,
    ids: set[str] | None = None,
) -> list[BfclEntry]:
    """Pick the entries of one category to run.

    Args:
        entries: The category's entries.
        limit: How many to sample; `None` or 0 keeps all of them.
        sample_seed: Seeds the sample, so reruns pick the same entries.
        ids: Run exactly these entries instead of sampling.

    Returns:
        The selected entries, in dataset order.
    """
    if ids is not None:
        return [entry for entry in entries if entry.id in ids]
    if not limit or limit >= len(entries):
        return list(entries)
    category = entries[0].category
    picked = set(
        random.Random(f"{sample_seed}|{category}").sample(range(len(entries)), limit)
    )
    return [entry for index, entry in enumerate(entries) if index in picked]
