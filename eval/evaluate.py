
import json
from pathlib import Path
from datetime import datetime, timezone

import torch
from model.gpt2 import GPT2NextItem
from .metrics import map_at_k, ndcg_at_k, recall_at_k


def evaluate(
    model: GPT2NextItem,
    sequences: list[list[int]],
    context_length: int,
    batch_size: int,
    k_values: list[int],
    ignore_token_id: int | None = None,
) -> dict[str, float]:
    if hasattr(sequences, "batch"):
        examples = None
    else:
        examples = [
            (sequence[:-1][-context_length:], sequence[-1])
            for sequence in sequences
            if len(sequence) > 1
            and (ignore_token_id is None or sequence[-1] != ignore_token_id)
        ]
    pad_token_id = model.pad_token_id
    device = next(model.parameters()).device
    recommendations: list[list[int]] = []
    targets: list[int] = []

    model.eval()
    with torch.no_grad():
        sequence_count = len(examples) if examples is not None else len(sequences)
        for start in range(0, sequence_count, batch_size):
            if examples is None:
                batch = sequences.batch(start, start + batch_size)
            else:
                batch = examples[start : start + batch_size]
            batch = [
                example
                for example in batch
                if ignore_token_id is None or example[1] != ignore_token_id
            ]
            if not batch:
                continue
            contexts = [context[-context_length:] for context, _ in batch]
            input_ids = torch.tensor(
                [
                    [pad_token_id] * (context_length - len(context)) + context
                    for context in contexts
                ],
                dtype=torch.long,
            ).to(device)
            attention_mask = torch.tensor(
                [
                    [0] * (context_length - len(context)) + [1] * len(context)
                    for context in contexts
                ],
                dtype=torch.long,
            ).to(device)
            last_logits = model.next_item_logits(input_ids, attention_mask)
            if (
                ignore_token_id is not None
                and ignore_token_id < last_logits.shape[-1]
            ):
                last_logits[:, ignore_token_id] = float("-inf")
            max_k = min(max(k_values), last_logits.shape[-1])
            recommendations.extend(last_logits.topk(max_k).indices.cpu().tolist())
            targets.extend(target for _, target in batch)

    metrics = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(recommendations, targets, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(recommendations, targets, k)
        metrics[f"map@{k}"] = map_at_k(recommendations, targets, k)
    return metrics


def append_metrics(
    metrics_file: Path,
    metrics: dict[str, float],
    step: int,
    examples: int,
    split: str = "validation",
) -> None:
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "step": step,
        "examples": examples,
        **metrics,
    }
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("a") as file:
        file.write(json.dumps(result) + "\n")


