import json
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset



class PreparedNextItemDataset(Dataset):
    def __init__(self, prepared_dir: Path):
        with (prepared_dir / "metadata.json").open() as file:
            metadata = json.load(file)
        rows, context_length = metadata["train_shape"]
        self.context_length = context_length
        self.input_ids = np.memmap(
            prepared_dir / "train_input_ids.dat",
            mode="r",
            dtype=np.int32,
            shape=(rows, context_length),
        )
        self.labels = np.memmap(
            prepared_dir / "train_labels.dat",
            mode="r",
            dtype=np.int32,
            shape=(rows, context_length),
        )
        self.attention_mask = np.memmap(
            prepared_dir / "train_attention_mask.dat",
            mode="r",
            dtype=np.uint8,
            shape=(rows, context_length),
        )
        if "train_tokens" in metadata["stats"]:
            self.token_count = int(metadata["stats"]["train_tokens"])
        elif {"train_events", "train_users"} <= metadata["stats"].keys():
            self.token_count = int(
                metadata["stats"]["train_events"]
                - metadata["stats"]["train_users"]
            )
        else:
            self.token_count = int(self.attention_mask.sum())

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.from_numpy(self.input_ids[index].astype(np.int64)),
            "attention_mask": torch.from_numpy(
                self.attention_mask[index].astype(np.int64)
            ),
            "labels": torch.from_numpy(self.labels[index].astype(np.int64)),
        }



class PreparedEvaluation:
    def __init__(self, prepared_dir: Path, split: str):
        with (prepared_dir / "metadata.json").open() as file:
            metadata = json.load(file)
        rows, context_length = metadata[f"{split}_shape"]
        self.input_ids = np.memmap(
            prepared_dir / f"{split}_input_ids.dat",
            mode="r",
            dtype=np.int32,
            shape=(rows, context_length),
        )
        self.targets = np.memmap(
            prepared_dir / f"{split}_targets.dat",
            mode="r",
            dtype=np.int32,
            shape=(rows,),
        )
        self.unk_token_id = metadata["stats"]["unk_token_id"]
        self.pad_token_id = metadata["stats"]["pad_token_id"]
        self.valid_count = int(np.count_nonzero(self.targets != self.unk_token_id))

    def __len__(self) -> int:
        return len(self.targets)

    def batch(self, start: int, end: int) -> list[tuple[list[int], int]]:
        result = []
        for index in range(start, min(end, len(self))):
            context = self.input_ids[index].tolist()
            while context and context[0] == self.pad_token_id:
                context.pop(0)
            target = int(self.targets[index])
            if target != self.unk_token_id:
                result.append((context, target))
        return result


def load_prepared_metadata(
    prepared_dir: Path,
) -> tuple[dict, list[int], int, int]:
    with (prepared_dir / "metadata.json").open() as file:
        metadata = json.load(file)
    stats = metadata["stats"]
    return (
        metadata,
        metadata["item_ids"],
        stats["unk_token_id"],
        stats["pad_token_id"],
    )


def load_prepared_sequences(
    prepared_dir: Path, split: str
) -> PreparedEvaluation:
    return PreparedEvaluation(prepared_dir, split)

