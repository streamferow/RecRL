"""Build a critic dataset of (context, candidate, engagement, like, dislike).

Reads the same multi-event parquet and the prepared vocabulary / time cutoffs
as pretraining. Training rows come only from timestamps up to the train cutoff.
Validation rows are kept for calibration. The test period is not written.

Labels are the signals observed on that event, not a carried preference state.
A listen has only y_eng. A like sets y_like=1, a dislike sets y_dislike=1.
unlike sets y_like=0 only after an observed like, undislike sets y_dislike=0
only after an observed dislike. Every other label stays -1. y_eng is NaN when
the event is not a listen. Candidates with item == unk are dropped and get no
pairwise pair; unk may still appear inside the context. Unheard candidates are
extra rows with every label missing and sampled_negative=1. Only the last
context_length listens are kept in the context; the heard-track set is complete.
If embeddings_path or artist_mapping_path is set, audio.npy and artist.npy are
aligned to pretrain token ids. Test-period events are not written.
"""

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

MISSING = -1
EVENT_CODE = {"listen": 0, "like": 1, "dislike": 2, "unlike": 3, "undislike": 4}


def load_config(path: Path) -> dict:
    with path.open() as file:
        return yaml.safe_load(file)


def load_prepared(prepared_dir: Path) -> tuple[dict, dict[int, int]]:
    with (prepared_dir / "metadata.json").open() as file:
        metadata = json.load(file)
    item_to_token = {int(item_id): token for token, item_id in enumerate(metadata["item_ids"])}
    return metadata, item_to_token


def context_row(history: list[int], context_length: int, pad_token_id: int) -> tuple[list[int], list[int]]:
    context = history[-context_length:]
    pad = context_length - len(context)
    return [pad_token_id] * pad + context, [0] * pad + [1] * len(context)


def remember(history: list[int], token: int, context_length: int) -> None:
    history.append(token)
    extra = len(history) - context_length
    if extra > 0:
        del history[:extra]


def keep(token: int | None, probability: float, tail_start: int, tail_multiplier: float, rng: np.random.Generator) -> bool:
    rate = probability
    if token is not None and token >= tail_start:
        rate = min(1.0, probability * tail_multiplier)
    return rng.random() < rate


def observed_labels(
    event_type: str,
    item_id: int,
    played_ratio: object,
    liked: set[int],
    disliked: set[int],
) -> tuple[float, int, int] | None:
    """Return (y_eng, y_like, y_dislike) for a directly observed signal."""
    if event_type == "listen":
        if pd.isna(played_ratio):
            y_eng = float("nan")
        else:
            y_eng = float(min(int(played_ratio), 100) / 100.0)
        return y_eng, MISSING, MISSING
    if event_type == "like":
        liked.add(item_id)
        disliked.discard(item_id)
        return float("nan"), 1, MISSING
    if event_type == "dislike":
        disliked.add(item_id)
        liked.discard(item_id)
        return float("nan"), MISSING, 1
    if event_type == "unlike":
        if item_id not in liked:
            return None
        liked.discard(item_id)
        return float("nan"), 0, MISSING
    if event_type == "undislike":
        if item_id not in disliked:
            return None
        disliked.discard(item_id)
        return float("nan"), MISSING, 0
    return None


class TableWriter:
    def __init__(self, path: Path, context_length: int):
        self.path = path
        self.context_length = context_length
        self._buffer: dict[str, list] = {name: [] for name in self._columns()}
        self._writer: pq.ParquetWriter | None = None
        self.rows = 0

    @staticmethod
    def _columns() -> tuple[str, ...]:
        return (
            "uid",
            "timestamp",
            "context",
            "attention_mask",
            "item",
            "y_eng",
            "y_like",
            "y_dislike",
            "is_organic",
            "event_type",
            "sampled_negative",
            "pair_id",
        )

    def _schema(self) -> pa.Schema:
        context = pa.list_(pa.int32(), self.context_length)
        mask = pa.list_(pa.uint8(), self.context_length)
        return pa.schema(
            [
                ("uid", pa.int64()),
                ("timestamp", pa.int64()),
                ("context", context),
                ("attention_mask", mask),
                ("item", pa.int32()),
                ("y_eng", pa.float32()),
                ("y_like", pa.int8()),
                ("y_dislike", pa.int8()),
                ("is_organic", pa.uint8()),
                ("event_type", pa.int8()),
                ("sampled_negative", pa.uint8()),
                ("pair_id", pa.int64()),
            ]
        )

    def add(self, row: dict) -> None:
        for name in self._columns():
            self._buffer[name].append(row[name])
        if len(self._buffer["uid"]) >= 65536:
            self.flush()

    def flush(self) -> None:
        if not self._buffer["uid"]:
            return
        table = pa.Table.from_pydict(self._buffer, schema=self._schema())
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.path, self._schema())
        self._writer.write_table(table)
        self.rows += table.num_rows
        for values in self._buffer.values():
            values.clear()

    def close(self) -> None:
        self.flush()
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            empty = {name: [] for name in self._columns()}
            pq.write_table(pa.Table.from_pydict(empty, schema=self._schema()), self.path)
            return
        self._writer.close()


def _keep_example(examples: list[dict], rng: np.random.Generator, stats: Counter, row: dict) -> None:
    stats["examples_seen"] += 1
    seen = stats["examples_seen"]
    if len(examples) < 5:
        examples.append(row)
        return
    slot = int(rng.integers(0, seen))
    if slot < 5:
        examples[slot] = row


def sample_negative(heard: set[int], num_items: int, rng: np.random.Generator) -> int | None:
    if len(heard) >= num_items:
        return None
    for _ in range(64):
        token = int(rng.integers(0, num_items))
        if token not in heard:
            return token
    start = int(rng.integers(0, num_items))
    for offset in range(num_items):
        token = (start + offset) % num_items
        if token not in heard:
            return token
    return None


def emit_user(
    uid: int,
    events: list[tuple],
    writers: dict[str, TableWriter],
    item_to_token: dict[int, int],
    unk_token_id: int,
    pad_token_id: int,
    context_length: int,
    train_cutoff: int,
    validation_cutoff: int,
    min_sequence_length: int,
    num_items: int,
    critic_config: dict,
    rng: np.random.Generator,
    stats: Counter,
    next_pair_id: list[int],
    examples: list[dict],
    example_rng: np.random.Generator,
) -> None:
    train_listens = sum(1 for timestamp, _, event_type, _, _ in events if event_type == "listen" and timestamp <= train_cutoff)
    if train_listens < min_sequence_length:
        stats["users_dropped_short"] += 1
        stats["test_events_skipped"] += sum(timestamp > validation_cutoff for timestamp, *_ in events)
        return
    stats["users_kept"] += 1

    liked: set[int] = set()
    disliked: set[int] = set()
    history: list[int] = []
    heard: set[int] = set()
    tail_start = int(critic_config["tail_token_start"])
    tail_multiplier = float(critic_config["tail_keep_multiplier"])
    negatives_per_listen = int(critic_config["negatives_per_listen"])

    for timestamp, item_id, event_type, is_organic, played_ratio in events:
        if timestamp > validation_cutoff:
            stats["test_events_skipped"] += 1
            continue
        if event_type not in EVENT_CODE:
            stats["unknown_event_type"] += 1
            continue
        stats[f"seen_{event_type}"] += 1
        split = "train" if timestamp <= train_cutoff else "validation"
        probability = float(critic_config[f"{split}_keep_probability"])
        token = item_to_token.get(int(item_id), unk_token_id)
        if token == unk_token_id:
            stats["unk_candidates_skipped"] += 1
            if event_type == "listen":
                remember(history, token, context_length)
            continue
        context, mask = context_row(history, context_length, pad_token_id)
        labels = observed_labels(event_type, int(item_id), played_ratio, liked, disliked)
        if labels is None:
            stats["unobserved_feedback_skipped"] += 1
            continue
        y_eng, y_like, y_dislike = labels

        if keep(token, probability, tail_start, tail_multiplier, rng):
            pair_id = -1
            if event_type == "listen":
                next_pair_id[0] += 1
                pair_id = next_pair_id[0]
            writers[split].add(
                {
                    "uid": uid,
                    "timestamp": int(timestamp),
                    "context": context,
                    "attention_mask": mask,
                    "item": token,
                    "y_eng": y_eng,
                    "y_like": y_like,
                    "y_dislike": y_dislike,
                    "is_organic": int(is_organic),
                    "event_type": EVENT_CODE[event_type],
                    "sampled_negative": 0,
                    "pair_id": pair_id,
                }
            )
            stats[f"{split}_rows"] += 1
            stats[f"{split}_{event_type}"] += 1
            stats[f"{split}_organic"] += int(is_organic)
            stats[f"{split}_recommended"] += int(not is_organic)
            real_tokens = sum(mask)
            stats[f"{split}_context_tokens"] += real_tokens
            stats[f"{split}_empty_context"] += int(real_tokens == 0)
            if y_eng == y_eng:
                stats[f"{split}_eng_count"] += 1
                stats[f"{split}_eng_sum"] += y_eng
            else:
                stats[f"{split}_eng_missing"] += 1
            for name, value in (("like", y_like), ("dislike", y_dislike)):
                if value == 1:
                    stats[f"{split}_{name}_positive"] += 1
                elif value == 0:
                    stats[f"{split}_{name}_zero"] += 1
                else:
                    stats[f"{split}_{name}_missing"] += 1
            if token >= tail_start:
                stats[f"{split}_tail"] += 1
            _keep_example(examples, example_rng, stats, {
                "split": split,
                "uid": uid,
                "timestamp": int(timestamp),
                "item": token,
                "context": [token_id for token_id, bit in zip(context, mask) if bit],
                "y_eng": None if y_eng != y_eng else y_eng,
                "y_like": y_like,
                "y_dislike": y_dislike,
                "is_organic": int(is_organic),
                "event_type": event_type,
                "sampled_negative": 0,
                "pair_id": pair_id,
            })
            if event_type == "listen" and pair_id:
                blocked = heard | {token}
                for _ in range(negatives_per_listen):
                    negative = sample_negative(blocked, num_items, rng)
                    if negative is None:
                        break
                    writers[split].add(
                        {
                            "uid": uid,
                            "timestamp": int(timestamp),
                            "context": context,
                            "attention_mask": mask,
                            "item": negative,
                            "y_eng": float("nan"),
                            "y_like": MISSING,
                            "y_dislike": MISSING,
                            "is_organic": int(is_organic),
                            "event_type": EVENT_CODE["listen"],
                            "sampled_negative": 1,
                            "pair_id": pair_id,
                        }
                    )
                    stats[f"{split}_sampled_negatives"] += 1
                    stats[f"{split}_rows"] += 1
                    stats[f"{split}_eng_missing"] += 1
                    stats[f"{split}_like_missing"] += 1
                    stats[f"{split}_dislike_missing"] += 1
                    stats[f"{split}_organic"] += int(is_organic)
                    stats[f"{split}_recommended"] += int(not is_organic)
                    stats[f"{split}_context_tokens"] += real_tokens
                    stats[f"{split}_empty_context"] += int(real_tokens == 0)
                    _keep_example(examples, example_rng, stats, {
                        "split": split,
                        "uid": uid,
                        "timestamp": int(timestamp),
                        "item": negative,
                        "context": [token_id for token_id, bit in zip(context, mask) if bit],
                        "y_eng": None,
                        "y_like": MISSING,
                        "y_dislike": MISSING,
                        "is_organic": int(is_organic),
                        "event_type": "listen",
                        "sampled_negative": 1,
                        "pair_id": pair_id,
                    })

        if event_type == "listen":
            remember(history, token, context_length)
            heard.add(token)


def write_audio(embeddings_path: Path, item_to_token: dict[int, int], output_path: Path) -> tuple[int, int]:
    parquet = pq.ParquetFile(embeddings_path)
    matrix = None
    matched = 0
    for batch in parquet.iter_batches(columns=["item_id", "normalized_embed"]):
        item_ids = batch.column("item_id").to_numpy()
        vectors = batch.column("normalized_embed")
        for item_id, vector in zip(item_ids, vectors, strict=True):
            values = None if vector is None else vector.as_py()
            if values is None:
                continue
            if matrix is None:
                matrix = np.zeros((len(item_to_token), len(values)), dtype=np.float32)
            token = item_to_token.get(int(item_id))
            if token is None:
                continue
            matrix[token] = np.asarray(values, dtype=np.float32)
            matched += 1
    if matrix is None:
        matrix = np.zeros((len(item_to_token), 0), dtype=np.float32)
    np.save(output_path, matrix)
    return matched, int(matrix.shape[1])


def write_artists(mapping_path: Path, item_to_token: dict[int, int], output_path: Path) -> int:
    artist = np.full(len(item_to_token), -1, dtype=np.int32)
    matched = 0
    parquet = pq.ParquetFile(mapping_path)
    for batch in parquet.iter_batches(columns=["artist_id", "item_id"]):
        artist_ids = batch.column("artist_id").to_numpy()
        item_ids = batch.column("item_id").to_numpy()
        for artist_id, item_id in zip(artist_ids, item_ids, strict=True):
            token = item_to_token.get(int(item_id))
            if token is None or artist[token] >= 0:
                continue
            artist[token] = int(artist_id)
            matched += 1
    np.save(output_path, artist)
    return matched


def iter_user_events(input_path: Path, batch_size: int) -> Iterator[tuple[int, list[tuple]]]:
    """Stream a uid-sorted multi-event parquet one user at a time.

    Each batch is deduplicated and ordered by uid, timestamp, the same way the
    pretrain prepare pass works. Only the user that spans the current batch is
    held in memory, so the file can be 5M, 500M, or 5B rows.
    """
    parquet = pq.ParquetFile(input_path)
    columns = ["uid", "timestamp", "item_id", "event_type", "is_organic", "played_ratio_pct"]
    current_uid = None
    events: list[tuple] = []
    last_key: tuple[int, int, str] | None = None

    def take(user_frame: pd.DataFrame) -> None:
        nonlocal last_key
        if user_frame.empty:
            return
        frame = user_frame
        if last_key is not None:
            first = frame.iloc[0]
            key = (int(first["timestamp"]), int(first["item_id"]), str(first["event_type"]))
            if key == last_key:
                frame = frame.iloc[1:]
        if frame.empty:
            return
        last = frame.iloc[-1]
        last_key = (int(last["timestamp"]), int(last["item_id"]), str(last["event_type"]))
        played = frame["played_ratio_pct"].to_numpy(dtype=np.float64, na_value=np.nan)
        timestamps = frame["timestamp"].to_numpy(dtype=np.int64, copy=False)
        item_ids = frame["item_id"].to_numpy(dtype=np.int64, copy=False)
        organic = frame["is_organic"].to_numpy(dtype=np.int64, copy=False)
        event_types = frame["event_type"].astype(str).to_numpy()
        for index in range(len(frame)):
            ratio = played[index]
            events.append(
                (
                    int(timestamps[index]),
                    int(item_ids[index]),
                    str(event_types[index]),
                    int(organic[index]),
                    None if np.isnan(ratio) else float(ratio),
                )
            )

    for record_batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        frame = record_batch.to_pandas()
        if frame.empty:
            continue
        frame = frame.drop_duplicates(["uid", "timestamp", "item_id", "event_type"], keep="first")
        frame = frame.sort_values(["uid", "timestamp"], kind="stable")
        for uid_value, user_frame in frame.groupby("uid", sort=False):
            uid = int(uid_value)
            if current_uid is not None and uid != current_uid:
                yield current_uid, events
                events = []
                last_key = None
            current_uid = uid
            take(user_frame)

    if current_uid is not None and events:
        yield current_uid, events


def _rate(part: float, whole: float) -> float | None:
    if not whole:
        return None
    return part / whole


def dataset_report(stats: Counter) -> dict:
    report = {"counts": {key: stats[key] for key in sorted(stats)}}
    for split in ("train", "validation"):
        rows = stats[f"{split}_rows"]
        negatives = stats[f"{split}_sampled_negatives"]
        report[split] = {
            "rows": rows,
            "labeled_rows": rows - negatives,
            "sampled_negatives": negatives,
            "by_event": {name: stats[f"{split}_{name}"] for name in EVENT_CODE},
            "organic_fraction": _rate(stats[f"{split}_organic"], rows),
            "recommended_fraction": _rate(stats[f"{split}_recommended"], rows),
            "negative_fraction": _rate(stats[f"{split}_sampled_negatives"], rows),
            "tail_fraction": _rate(stats[f"{split}_tail"], rows),
            "empty_context_fraction": _rate(stats[f"{split}_empty_context"], rows),
            "mean_context_tokens": _rate(stats[f"{split}_context_tokens"], rows),
            "mean_engagement": _rate(stats[f"{split}_eng_sum"], stats[f"{split}_eng_count"]),
            "engagement_labeled": stats[f"{split}_eng_count"],
            "engagement_missing": stats[f"{split}_eng_missing"],
            "like_positive": stats[f"{split}_like_positive"],
            "like_zero": stats[f"{split}_like_zero"],
            "like_missing": stats[f"{split}_like_missing"],
            "dislike_positive": stats[f"{split}_dislike_positive"],
            "dislike_zero": stats[f"{split}_dislike_zero"],
            "dislike_missing": stats[f"{split}_dislike_missing"],
        }
    report["users"] = {
        "seen": stats["users"],
        "kept": stats["users_kept"],
        "dropped_short": stats["users_dropped_short"],
    }
    report["skipped"] = {
        "test_events": stats["test_events_skipped"],
        "unk_candidates": stats["unk_candidates_skipped"],
        "feedback_without_prior_state": stats["unobserved_feedback_skipped"],
        "unknown_event_type": stats["unknown_event_type"],
        "seen_by_event": {name: stats[f"seen_{name}"] for name in EVENT_CODE},
    }
    return report


def run(config: dict) -> None:
    data_config = config["data"]
    preprocessing_config = config["preprocessing"]
    critic_config = config["critic"]
    prepared_dir = Path(data_config["prepared_dir"])
    metadata, item_to_token = load_prepared(prepared_dir)
    train_cutoff = int(metadata["train_cutoff"])
    validation_cutoff = int(metadata["validation_cutoff"])
    context_length = int(metadata["context_length"])
    unk_token_id = int(metadata["stats"]["unk_token_id"])
    pad_token_id = int(metadata["stats"]["pad_token_id"])
    min_sequence_length = int(preprocessing_config["min_sequence_length"])
    num_items = len(item_to_token)
    output_dir = Path(critic_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(critic_config.get("seed", config.get("seed", 0))))

    writers = {
        "train": TableWriter(output_dir / "train.parquet", context_length),
        "validation": TableWriter(output_dir / "validation.parquet", context_length),
    }
    stats: Counter = Counter()
    next_pair_id = [0]
    examples: list[dict] = []
    example_rng = np.random.default_rng(int(critic_config.get("seed", config.get("seed", 0))) + 1)

    users_seen = 0
    for uid, events in iter_user_events(Path(data_config["source_path"]), int(preprocessing_config["batch_size"])):
        stats["users"] += 1
        users_seen += 1
        emit_user(
            uid,
            events,
            writers,
            item_to_token,
            unk_token_id,
            pad_token_id,
            context_length,
            train_cutoff,
            validation_cutoff,
            min_sequence_length,
            num_items,
            critic_config,
            rng,
            stats,
            next_pair_id,
            examples,
            example_rng,
        )
        if users_seen % 5000 == 0:
            print(
                f"[critic] users={users_seen:,} "
                f"train_rows={writers['train'].rows + len(writers['train']._buffer['uid']):,} "
                f"validation_rows={writers['validation'].rows + len(writers['validation']._buffer['uid']):,}",
                flush=True,
            )
    for writer in writers.values():
        writer.close()

    catalog = {}
    embeddings_path = critic_config.get("embeddings_path")
    if embeddings_path:
        matched, width = write_audio(Path(embeddings_path), item_to_token, output_dir / "audio.npy")
        catalog["audio"] = {"rows_matched": matched, "width": width, "file": "audio.npy"}
        print(f"[critic] audio tokens={matched:,} dim={width}", flush=True)
    artist_path = critic_config.get("artist_mapping_path")
    if artist_path:
        matched = write_artists(Path(artist_path), item_to_token, output_dir / "artist.npy")
        catalog["artist"] = {"rows_matched": matched, "missing": -1, "file": "artist.npy"}
        print(f"[critic] artist tokens={matched:,}", flush=True)

    summary = {
        "train_cutoff": train_cutoff,
        "validation_cutoff": validation_cutoff,
        "context_length": context_length,
        "num_items": num_items,
        "unk_token_id": unk_token_id,
        "pad_token_id": pad_token_id,
        "missing_label": MISSING,
        "event_code": EVENT_CODE,
        "engagement": "min(played_ratio_pct, 100) / 100 on listen, null otherwise",
        "labels": "observed signal only; missing is -1 for like/dislike and null for y_eng; unk items are not candidates",
        "test_split": "excluded",
        "item_ids": [item_id for item_id, _ in sorted(item_to_token.items(), key=lambda pair: pair[1])],
        "catalog": catalog,
        "rows": {split: writers[split].rows for split in writers},
        "report": dataset_report(stats),
        "examples": examples,
        "config": critic_config,
    }
    with (output_dir / "metadata.json").open("w") as file:
        json.dump(summary, file, indent=2)
    (output_dir / "examples.json").write_text(json.dumps(examples, indent=2))
    print(json.dumps(summary["report"], indent=2))
    print(json.dumps(examples, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "posttrain" / "train.yaml",
    )
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
