import ast
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

DATASET_REPO_ID = "AntonSvystunov/mbpp-jitgen-validation"
DATA_CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class MbppExample:
    """A single (task, test case) row of the MBPP JitGen validation dataset."""

    task_id: int
    text: str
    example_test_input: str
    example_test_output: str
    test_input: str
    test_output: str

    def is_correct_answer(self, answer: str) -> bool:
        """Check whether a candidate answer matches `test_output`.

        `test_output` is stored via `repr()`, not `str()` (see
        `scripts/create_dataset.py`), so both sides are recovered with
        `ast.literal_eval()` and compared as Python values rather than as
        raw text — otherwise equal values could be reported as mismatches
        over formatting differences alone (dict key order, float
        precision, etc.). Falls back to a raw string comparison if either
        side isn't a literal Python expression.

        Args:
            answer: The candidate answer, as a `repr()`-style string.

        Returns:
            `True` if `answer` evaluates to the same value as `test_output`.
        """
        try:
            expected = ast.literal_eval(self.test_output)
            actual = ast.literal_eval(answer)
        except (ValueError, SyntaxError):
            return answer.strip() == self.test_output.strip()
        return actual == expected


def load_mbpp_dataset(
    repo_id: str = DATASET_REPO_ID, *, cache_dir: Path = DATA_CACHE_DIR
) -> list[MbppExample]:
    """Load the MBPP JitGen validation dataset from the Hugging Face Hub.

    Args:
        repo_id: The Hugging Face Hub dataset repo to load, e.g. `owner/name`.
        cache_dir: Directory to cache the downloaded dataset in.

    Returns:
        One `MbppExample` per row of the dataset's `train` split.
    """
    dataset = load_dataset(repo_id, split="train", cache_dir=str(cache_dir))
    return [MbppExample(**row) for row in dataset]  # type: ignore


@dataclass(frozen=True)
class IndexedCase:
    """An `MbppExample` tagged with stable identity independent of any run's sampling.

    `dataset_row`/`case_index` must be computed over the *full* dataset before
    a run slices out a subset (e.g. `[:ROWS_TO_RUN]`) — otherwise they'd
    reflect the run's sampling order rather than the dataset's own, and stop
    being comparable across runs that sample differently.
    """

    dataset_row: int
    case_index: int
    example: MbppExample


def load_indexed_dataset(
    repo_id: str = DATASET_REPO_ID, *, cache_dir: Path = DATA_CACHE_DIR
) -> list[IndexedCase]:
    """Load the dataset with stable row/case identity attached.

    Args:
        repo_id: The Hugging Face Hub dataset repo to load, e.g. `owner/name`.
        cache_dir: Directory to cache the downloaded dataset in.

    Returns:
        One `IndexedCase` per row, in dataset order. `dataset_row` is the
        row's absolute position; `case_index` is its 0-based position among
        rows sharing the same `task_id`.
    """
    examples = load_mbpp_dataset(repo_id, cache_dir=cache_dir)
    case_counts: dict[int, int] = {}
    indexed: list[IndexedCase] = []
    for dataset_row, example in enumerate(examples):
        case_index = case_counts.get(example.task_id, 0)
        case_counts[example.task_id] = case_index + 1
        indexed.append(IndexedCase(dataset_row, case_index, example))
    return indexed
