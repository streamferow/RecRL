import json
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from eval.evaluate import append_metrics
from posttrain.critic import Critic, critic_loss


@torch.no_grad()
def evaluate_critic(model: Critic, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], int]:
    """Validation loss plus checks that the heads track the observed labels."""
    model.eval()
    totals = {name: 0.0 for name in ("loss", "engagement", "like", "dislike", "eng_abs", "eng_pred", "eng_target", "log_var")}
    counts = {name: 0.0 for name in ("rows", "eng", "like", "dislike", "like_correct", "dislike_correct", "like_pred", "dislike_pred", "like_pos", "dislike_pos")}
    for context, mask, item, y_eng, y_like, y_dislike in loader:
        context, mask, item = context.to(device), mask.to(device), item.to(device)
        y_eng, y_like, y_dislike = y_eng.to(device), y_like.to(device), y_dislike.to(device)
        prediction = model(context, mask, item)
        parts = critic_loss(prediction, y_eng, y_like, y_dislike)
        rows = context.size(0)
        for name in ("loss", "engagement", "like", "dislike"):
            totals[name] += float(parts[name]) * rows
        counts["rows"] += rows
        totals["log_var"] += float(prediction["log_var_eng"].sum())

        eng = torch.isfinite(y_eng)
        if eng.any():
            mu = prediction["mu_eng"][eng]
            target = y_eng[eng]
            totals["eng_abs"] += float((mu - target).abs().sum())
            totals["eng_pred"] += float(mu.sum())
            totals["eng_target"] += float(target.sum())
            counts["eng"] += int(eng.sum())
        _classification(prediction["like_logit"], y_like, "like", totals, counts)
        _classification(prediction["dislike_logit"], y_dislike, "dislike", totals, counts)

    rows = max(1.0, counts["rows"])
    metrics = {name: totals[name] / rows for name in ("loss", "engagement", "like", "dislike")}
    metrics["log_var"] = totals["log_var"] / rows
    metrics["eng_mae"] = totals["eng_abs"] / max(1.0, counts["eng"])
    metrics["eng_bias"] = (totals["eng_pred"] - totals["eng_target"]) / max(1.0, counts["eng"])
    for name in ("like", "dislike"):
        labeled = counts[name]
        if not labeled:
            metrics[f"{name}_accuracy"] = float("nan")
            metrics[f"{name}_pred_rate"] = float("nan")
            metrics[f"{name}_label_rate"] = float("nan")
            continue
        metrics[f"{name}_accuracy"] = counts[f"{name}_correct"] / labeled
        metrics[f"{name}_pred_rate"] = counts[f"{name}_pred"] / labeled
        metrics[f"{name}_label_rate"] = counts[f"{name}_pos"] / labeled
    return metrics, int(counts["rows"])


def _classification(logit, target, name, totals, counts) -> None:
    labeled = target != -1
    if not labeled.any():
        return
    probability = torch.sigmoid(logit[labeled])
    label = target[labeled].float()
    counts[name] += int(labeled.sum())
    counts[f"{name}_correct"] += float(((probability >= 0.5) == (label >= 0.5)).sum())
    counts[f"{name}_pred"] += float((probability >= 0.5).sum())
    counts[f"{name}_pos"] += float(label.sum())


def restore_history(checkpoint: Path, *paths: Path) -> None:
    """Copy logs out of a checkpoint when /kaggle/working was wiped."""
    for path in paths:
        saved = checkpoint / path.name
        if saved.exists() and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(saved, path)


def keep_history(checkpoint: Path, *paths: Path) -> None:
    for path in paths:
        if path.exists():
            shutil.copyfile(path, checkpoint / path.name)


def plot_critic(metrics_file: Path, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not metrics_file.exists():
        return
    rows = [json.loads(line) for line in metrics_file.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("split") == "validation"]
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = [row["step"] for row in rows]
    plt.figure(figsize=(10, 5))
    for name in ("loss", "engagement", "like", "dislike", "eng_mae"):
        plt.plot(steps, [row[name] for row in rows], label=name)
    plt.xlabel("Optimizer update")
    plt.ylabel("Validation")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def record_eval(model, loader, device, metrics_file: Path, step: int) -> dict[str, float]:
    metrics, examples = evaluate_critic(model, loader, device)
    append_metrics(metrics_file, metrics, step, examples)
    return metrics
