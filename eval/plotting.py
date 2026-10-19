import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def append_training_loss(path: Path, epoch: int, step: int, loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(
            json.dumps({"epoch": epoch, "step": step, "loss": loss}) + "\n"
        )


def plot_training_loss(history_path: Path, output_path: Path) -> None:
    history = _read_jsonl(history_path)
    if not history:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(
        [entry["step"] for entry in history],
        [entry["loss"] for entry in history],
        label="train loss",
    )
    plt.xlabel("Optimizer update")
    plt.ylabel("Loss")
    plt.title("GPT-2 next-item training loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_metrics_at_k(metrics_path: Path, output_path: Path, k: int) -> None:
    history = [
        entry
        for entry in _read_jsonl(metrics_path)
        if entry.get("split") == "validation"
    ]
    if not history:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    steps = [entry["step"] for entry in history]
    for name in ("recall", "ndcg", "map"):
        key = f"{name}@{k}"
        if key in history[0]:
            plt.plot(steps, [entry[key] for entry in history], label=key)
    plt.xlabel("Optimizer update")
    plt.ylabel("Metric")
    plt.title(f"Validation metrics at k={k}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
