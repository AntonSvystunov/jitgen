import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset"
CSV_NAME = "mbpp_jitgen_validation.csv"
REPO_ID = "AntonSvystunov/mbpp-jitgen-validation"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the publish step.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Publish the MBPP JitGen validation dataset to the Hugging Face Hub."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Directory containing the dataset CSV.",
    )
    parser.add_argument(
        "--csv-name",
        default=CSV_NAME,
        help="Name of the CSV file inside `--dataset-dir`.",
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help="Target Hugging Face Hub dataset repo, e.g. `owner/name`.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Publish the repo as private instead of public.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. Falls back to the `HF_TOKEN` env var or a cached login if omitted.",
    )
    return parser.parse_args()


def load_dataset_csv(csv_path: Path) -> pd.DataFrame:
    """Load the dataset CSV produced by `create_dataset.py`.

    Args:
        csv_path: Path to the dataset CSV file.

    Returns:
        The dataset as a `DataFrame`.

    Raises:
        FileNotFoundError: If `csv_path` doesn't exist.
    """
    if not csv_path.exists():
        msg = f"{csv_path} does not exist. Run create_dataset.py first to generate it."
        raise FileNotFoundError(msg)
    return pd.read_csv(csv_path)


def main() -> None:
    args = parse_args()
    csv_path = args.dataset_dir / args.csv_name

    df = load_dataset_csv(csv_path)
    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset.push_to_hub(args.repo_id, private=args.private, token=args.token)


if __name__ == "__main__":
    main()
