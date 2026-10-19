import json
import argparse
from typing import Any
from pathlib import Path
from typing import Iterator
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml



def sample_cutoffs(input_path: Path, train_fraction: float, validation_fraction: float, batch_size: int) -> tuple[int, int]:
    parquet = pq.ParquetFile(input_path)
    columns = ["timestamp", "event_type"]
    samples: list[pd.Series] = []

    for record_batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        frame = record_batch.to_pandas()
        frame = frame[frame["event_type"] == "listen"]
        timestamps = frame["timestamp"]
        if len(timestamps) > 256:
            timestamps = timestamps.sample(256, random_state=0)
        samples.append(timestamps)

    timestamps = pd.concat(samples, ignore_index=True)
    train_cutoff = int(timestamps.quantile(train_fraction))
    validation_cutoff = int(timestamps.quantile(train_fraction + validation_fraction))
    return train_cutoff, validation_cutoff



def iter_user_splits(
    input_path: Path,
    train_cutoff: int,
    validation_cutoff: int,
    batch_size: int,
    min_sequence_length: int,
) -> Iterator[tuple[list[int], list[int], list[int]]]:
    parquet = pq.ParquetFile(input_path)
    columns = ["uid", "timestamp", "item_id", "event_type", "is_organic"]

    train: list[int] = []
    test: list[int] = []
    validation: list[int] = []

    def flush() -> tuple[list[int], list[int], list[int]] | None:
        nonlocal train, validation, test, last_event
        if current_uid is None:
            return None
        if len(train) < min_sequence_length:
            train, validation, test = [], [], []
            last_event = None
            return None
        result = train, validation, test
        train, validation, test = [], [], []
        last_event = None
        return result

    current_uid = None
    last_event: tuple[int, int] | None = None

    for record_batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        frame = record_batch.to_pandas()
        frame = frame[frame["event_type"] == "listen"]

        if frame.empty:
            continue

        frame = frame.drop_duplicates(["uid", "timestamp", "item_id"], keep="first")
        frame = frame.sort_values(["uid", "timestamp"], kind="stable")

        for uid_value, user_frame in frame.groupby("uid", sort=False):
            uid = int(uid_value)
            if current_uid is not None and uid != current_uid:
                result = flush()
                if result is not None:
                    yield result

            current_uid = uid
            if last_event is not None and len(user_frame):
                first = user_frame.iloc[0]
                if (int(first["timestamp"]), int(first["item_id"])) == last_event:
                    user_frame = user_frame.iloc[1:]
            if user_frame.empty:
                continue

            last_row = user_frame.iloc[-1]
            last_event = (int(last_row["timestamp"]), int(last_row["item_id"]))
            timestamps = user_frame["timestamp"]

            train.extend(user_frame.loc[timestamps <= train_cutoff, "item_id"].astype(np.int64).tolist())
            validation.extend(user_frame.loc[(timestamps > train_cutoff) & (timestamps <= validation_cutoff), "item_id",].astype(np.int64).tolist())
            test.extend(user_frame.loc[timestamps > validation_cutoff, "item_id"].astype(np.int64).tolist())

    result = flush()
    if result is not None:
        yield result



def collect_statistics(
    input_path: Path,
    train_cutoff: int,
    validation_cutoff: int,
    batch_size: int,
    min_sequence_length: int,
) -> tuple[Counter, dict[str, int]]:
    counts: Counter = Counter()
    stats = Counter()

    def consume(train: list[int], validation: list[int], test: list[int]) -> None:
        if train or validation or test:
            stats["users"] += 1
        counts.update(train)
        stats["train_events"] += len(train)
        stats["validation_events"] += len(validation)
        stats["test_events"] += len(test)
        stats["train_users"] += bool(train)
        stats["validation_users"] += bool(validation)
        stats["test_users"] += bool(test)

    for split in iter_user_splits(
        input_path,
        train_cutoff,
        validation_cutoff,
        batch_size,
        min_sequence_length,
    ):
        consume(*split)
    return counts, dict(stats)



def count_training_windows(
    input_path: Path,
    train_cutoff: int,
    validation_cutoff: int,
    context_length: int,
    batch_size: int,
    min_sequence_length: int,
) -> int:
    count = 0
    for train, _, _ in iter_user_splits(input_path, train_cutoff, validation_cutoff, batch_size,min_sequence_length):
        count += sum(
            1
            for start in range(0, max(0, len(train) - 1), context_length)
            if len(train[start : start + context_length + 1]) > 2
        )
    return count



def encode_sequence(sequence: list[int], item_to_token: dict[int, int], unk_token_id: int) -> list[int]:
    return [item_to_token.get(item, unk_token_id) for item in sequence]


def write_prepared_data(
    input_path: Path,
    output_dir: Path,
    train_cutoff: int,
    validation_cutoff: int,
    item_to_token: dict[int, int],
    context_length: int,
    batch_size: int,
    min_sequence_length: int,
    stats: dict[str, int],
    preprocessing_config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unk_token_id = len(item_to_token)
    pad_token_id = unk_token_id + 1
    windows = count_training_windows(
        input_path,
        train_cutoff,
        validation_cutoff,
        context_length,
        batch_size,
        min_sequence_length,
    )
    input_memmap = np.memmap(
        output_dir / "train_input_ids.dat",
        mode="w+",
        dtype=np.int32,
        shape=(windows, context_length),
    )
    label_memmap = np.memmap(
        output_dir / "train_labels.dat",
        mode="w+",
        dtype=np.int32,
        shape=(windows, context_length),
    )
    mask_memmap = np.memmap(
        output_dir / "train_attention_mask.dat",
        mode="w+",
        dtype=np.uint8,
        shape=(windows, context_length),
    )
    validation_inputs = np.memmap(
        output_dir / "validation_input_ids.dat",
        mode="w+",
        dtype=np.int32,
        shape=(stats["users"], context_length),
    )
    validation_targets = np.memmap(
        output_dir / "validation_targets.dat",
        mode="w+",
        dtype=np.int32,
        shape=(stats["users"],),
    )
    test_inputs = np.memmap(
        output_dir / "test_input_ids.dat",
        mode="w+",
        dtype=np.int32,
        shape=(stats["users"], context_length),
    )
    test_targets = np.memmap(
        output_dir / "test_targets.dat",
        mode="w+",
        dtype=np.int32,
        shape=(stats["users"],),
    )
    row_index = 0
    validation_index = 0
    test_index = 0
    coverage = Counter()

    def encode_and_count(sequence: list[int], split: str) -> list[int]:
        encoded = encode_sequence(sequence, item_to_token, unk_token_id)
        coverage["events"] += len(encoded)
        known = sum(token != unk_token_id for token in encoded)
        coverage["known_events"] += known
        coverage[f"{split}_events"] += len(encoded)
        coverage[f"known_{split}_events"] += known
        return encoded

    for train, validation, test in iter_user_splits(
        input_path,
        train_cutoff,
        validation_cutoff,
        batch_size,
        min_sequence_length,
    ):
        encoded_train = encode_and_count(train, "train")
        encoded_validation = encode_and_count(validation, "validation")
        encoded_test = encode_and_count(test, "test")

        validation_history = encoded_train + encoded_validation
        if validation and len(validation_history) > 1:
            context = validation_history[:-1][-context_length:]
            pad = context_length - len(context)
            validation_inputs[validation_index] = [pad_token_id] * pad + context
            validation_targets[validation_index] = validation_history[-1]
            validation_index += 1

        test_history = validation_history + encoded_test
        if test and len(test_history) > 1:
            context = test_history[:-1][-context_length:]
            pad = context_length - len(context)
            test_inputs[test_index] = [pad_token_id] * pad + context
            test_targets[test_index] = test_history[-1]
            test_index += 1

        for start in range(0, max(0, len(encoded_train) - 1), context_length):
            window = encoded_train[start : start + context_length + 1]
            inputs, targets = window[:-1], window[1:]
            if len(inputs) <= 1:
                continue
            pad = context_length - len(inputs)
            input_memmap[row_index] = [pad_token_id] * pad + inputs
            label_memmap[row_index] = [-100] * pad + [-100 if token == unk_token_id else token for token in targets]
            mask_memmap[row_index] = [0] * pad + [1] * len(inputs)
            row_index += 1

    input_memmap.flush()
    label_memmap.flush()
    mask_memmap.flush()

    validation_inputs.flush()
    validation_targets.flush()

    test_inputs.flush()
    test_targets.flush()

    stats.update(
        {
            "train_windows": windows,
            "vocabulary_size": len(item_to_token),
            "unk_token_id": unk_token_id,
            "pad_token_id": pad_token_id,
            "min_sequence_length": min_sequence_length,
            **coverage,
        }
    )
    with (output_dir / "metadata.json").open("w") as file:
        json.dump(
            {
                "context_length": context_length,
                "train_shape": [windows, context_length],
                "validation_shape": [validation_index, context_length],
                "test_shape": [test_index, context_length],
                "item_ids": list(item_to_token),
                "train_cutoff": train_cutoff,
                "validation_cutoff": validation_cutoff,
                "preprocessing": {
                    **preprocessing_config,
                    "train_cutoff": train_cutoff,
                    "validation_cutoff": validation_cutoff,
                },
                "stats": stats,
            },
            file,
        )


def run_preprocessing(data_config: dict[str, Any], preprocessing_config: dict[str, Any]) -> None:
    input_path = Path(data_config["source_path"])
    output_dir = Path(data_config["prepared_dir"])
    context_length = preprocessing_config["context_length"]
    min_sequence_length = preprocessing_config["min_sequence_length"]
    max_items = preprocessing_config["max_items"]
    batch_size = preprocessing_config["batch_size"]
    train_fraction = preprocessing_config["train_fraction"]
    validation_fraction = preprocessing_config["validation_fraction"]
    configured_train_cutoff = preprocessing_config.get("train_cutoff")
    configured_validation_cutoff = preprocessing_config.get("validation_cutoff")
    if min_sequence_length < 1:
        raise ValueError("min_sequence_length must be positive")

    if configured_train_cutoff is None or configured_validation_cutoff is None:
        train_cutoff, validation_cutoff = sample_cutoffs(
            input_path,
            train_fraction,
            validation_fraction,
            batch_size,
        )
    else:
        train_cutoff, validation_cutoff = (
            configured_train_cutoff,
            configured_validation_cutoff,
        )
    counts, stats = collect_statistics(
        input_path,
        train_cutoff,
        validation_cutoff,
        batch_size,
        min_sequence_length,
    )
    item_ids = [item_id for item_id, _ in counts.most_common(max_items)]
    item_to_token = {item_id: token for token, item_id in enumerate(item_ids)}
    write_prepared_data(
        input_path,
        output_dir,
        train_cutoff,
        validation_cutoff,
        item_to_token,
        context_length,
        batch_size,
        min_sequence_length,
        stats,
        preprocessing_config,
    )
    print(json.dumps({"cutoffs": [train_cutoff, validation_cutoff], **stats}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("train.yaml"))
    args = parser.parse_args()
    with args.config.open() as file:
        config = yaml.safe_load(file)
    run_preprocessing(config["data"], config["preprocessing"])


if __name__ == "__main__":
    main()
