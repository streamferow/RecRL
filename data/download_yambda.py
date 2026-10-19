import random
import argparse
from typing import Any
from pathlib import Path
from datasets import load_dataset
import yaml


def run_download(config: dict[str, Any]) -> None:
    event = config["event"]
    dataset_size = config["dataset_size"]
    output_root = Path(config["output_dir"])
    users_fraction = config["users_fraction"]
    seed = config["seed"]
    if not 0 < users_fraction <= 1:
        raise ValueError("users_fraction must be between 0 and 1")

    dataset = load_dataset(
        "yandex/yambda",
        data_dir=f"flat/{dataset_size}",
        data_files=f"{event}.parquet",
    )["train"]
    expected_columns = {"uid", "timestamp", "item_id"}
    missing = expected_columns - set(dataset.column_names)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    if users_fraction < 1:
        users = dataset.unique("uid")
        count = max(1, round(len(users) * users_fraction))
        selected_users = set(random.Random(seed).sample(users, count))
        dataset = dataset.filter(lambda row: row["uid"] in selected_users)

    output_dir = output_root / event
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_dir.with_suffix(".parquet")
    dataset.to_parquet(output_path)
    print(f"Saved {len(dataset):,} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("train.yaml"))
    args = parser.parse_args()
    with args.config.open() as file:
        config = yaml.safe_load(file)
    run_download(config["data"]["download"])


if __name__ == "__main__":
    main()
