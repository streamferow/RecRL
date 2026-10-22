import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler, Subset, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.plotting import append_training_loss, plot_training_loss
from model.gpt2 import GPT2NextItem, get_device
from posttrain.critic import Critic, critic_loss
from posttrain.eval import keep_history, plot_critic, record_eval, restore_history


class _SkipFirstBatches(torch.utils.data.Sampler[int]):
    def __init__(self, sampler, skip_batches: int, batch_size: int):
        self.sampler = sampler
        self.skip = skip_batches * batch_size

    def __iter__(self):
        indices = iter(self.sampler)
        for _ in range(self.skip):
            next(indices, None)
        return indices

    def __len__(self) -> int:
        return max(0, len(self.sampler) - self.skip)


def _column(table, name: str, context_length: int | None = None) -> torch.Tensor:
    values = table.column(name).combine_chunks()
    if context_length is None:
        array = values.to_numpy(zero_copy_only=False)
        dtype = torch.float32 if name == "y_eng" else torch.long
        return torch.tensor(array, dtype=dtype)
    flat = values.values.to_numpy(zero_copy_only=False)
    return torch.tensor(flat.reshape(-1, context_length), dtype=torch.long)


def load_split(path: Path, context_length: int) -> TensorDataset:
    table = pq.read_table(path, columns=["context", "attention_mask", "item", "y_eng", "y_like", "y_dislike"])
    return TensorDataset(
        _column(table, "context", context_length),
        _column(table, "attention_mask", context_length),
        _column(table, "item"),
        _column(table, "y_eng"),
        _column(table, "y_like"),
        _column(table, "y_dislike"),
    )


def build_encoder(model_config: dict, metadata: dict, weights: Path, device: torch.device) -> GPT2NextItem:
    encoder = GPT2NextItem(
        num_items=int(metadata["num_items"]),
        context_length=int(metadata["context_length"]),
        hidden_size=model_config["hidden_size"],
        layers=model_config["layers"],
        heads=model_config["heads"],
        feed_forward_size=model_config["feed_forward_size"],
        dropout=model_config["dropout"],
        unk_token_id=int(metadata["unk_token_id"]),
        pad_token_id=int(metadata["pad_token_id"]),
    )
    encoder.load_state_dict(torch.load(weights / "model.pt", map_location=device))
    return encoder.to(device)


def save_checkpoint(model, optimizer, checkpoint_dir: Path, step: int, epoch: int, batch_index: int, logs: tuple[Path, ...]) -> Path:
    output = checkpoint_dir / f"checkpoint-{step}"
    output.mkdir(parents=True, exist_ok=True)
    trainable = {name: tensor for name, tensor in model.state_dict().items() if not name.startswith("encoder.")}
    torch.save(trainable, output / "critic.pt")
    torch.save({"optimizer": optimizer.state_dict(), "step": step, "epoch": epoch, "batch_index": batch_index}, output / "training_state.pt")
    keep_history(output, *logs)
    return output


def main() -> None:
    config_path = Path(__file__).resolve().with_name("train.yaml")
    config = yaml.safe_load(config_path.read_text())
    training = config["training"]
    torch.manual_seed(config["seed"])
    device = get_device()
    dataset_dir = Path(training["dataset_dir"])
    metadata = json.loads((dataset_dir / "metadata.json").read_text())
    pretrain = yaml.safe_load((ROOT / "pretrain" / "train.yaml").read_text())["model"]
    context_length = int(metadata["context_length"])

    print("[critic] loading rows", flush=True)
    train_data = load_split(dataset_dir / "train.parquet", context_length)
    validation = load_split(dataset_dir / "validation.parquet", context_length)
    eval_rows = min(int(training["eval_rows"]), len(validation))
    eval_data = Subset(validation, torch.randperm(len(validation), generator=torch.Generator().manual_seed(config["seed"]))[:eval_rows].tolist())
    eval_loader = DataLoader(eval_data, batch_size=training["batch_size"])

    encoder = build_encoder(pretrain, metadata, Path(training["encoder_from"]), device)
    model = Critic(encoder, dropout=pretrain["dropout"]).to(device)
    optimizer = torch.optim.AdamW(list(model.trainable_parameters()), lr=training["learning_rate"], weight_decay=training["weight_decay"])

    resume = training.get("resume_from")
    checkpoint_dir = Path(training["checkpoint_dir"])
    if resume == "auto":
        saved = [path for path in checkpoint_dir.glob("checkpoint-*") if (path / "training_state.pt").exists()]
        resume = max(saved, key=lambda path: int(path.name.rsplit("-", 1)[1]), default=None)
    step = 0
    start_epoch = 0
    resume_batch_index = 0
    loss_path = Path(training["training_loss_file"])
    metrics_path = Path(training["metrics_file"])
    plots = Path(training["plots_dir"])
    if resume:
        resume = Path(resume)
        model.load_state_dict(torch.load(resume / "critic.pt", map_location=device), strict=False)
        state = torch.load(resume / "training_state.pt", map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        start_epoch = state["epoch"]
        resume_batch_index = state["batch_index"]
        restore_history(resume, loss_path, metrics_path)
        print(f"[critic] resumed step={step} next_batch={resume_batch_index}", flush=True)

    batch_size = int(training["batch_size"])
    print(f"[critic] train_rows={len(train_data):,} eval_rows={eval_rows:,} device={device}", flush=True)
    for epoch in range(start_epoch, int(training["epochs"])):
        skip = resume_batch_index if epoch == start_epoch else 0
        sampler = RandomSampler(train_data, generator=torch.Generator().manual_seed(config["seed"] + epoch))
        if skip:
            sampler = _SkipFirstBatches(sampler, skip, batch_size)
            print(f"[critic] jumping to batch {skip}", flush=True)
        loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=training["num_workers"], drop_last=True)
        model.train()
        window = 0.0
        seen = 0
        for local_index, batch in enumerate(loader):
            batch_index = local_index + skip
            context, mask, item, y_eng, y_like, y_dislike = (value.to(device) for value in batch)
            parts = critic_loss(model(context, mask, item), y_eng, y_like, y_dislike)
            parts["loss"].backward()
            torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), training["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            window += float(parts["loss"])
            seen += 1
            if step == 1 or step % training["log_every_steps"] == 0:
                mean = window / max(1, seen)
                window = 0.0
                seen = 0
                append_training_loss(loss_path, epoch + 1, step, mean)
                print(
                    f"[critic] epoch={epoch + 1} update={step} loss={mean:.4f} "
                    f"eng={float(parts['engagement']):.4f} like={float(parts['like']):.4f} dislike={float(parts['dislike']):.4f}",
                    flush=True,
                )
            if step % training["eval_every_steps"] == 0:
                metrics = record_eval(model, eval_loader, device, metrics_path, step)
                model.train()
                plot_training_loss(loss_path, plots / "critic_loss.png")
                plot_critic(metrics_path, plots / "critic_metrics.png")
                print(f"[eval] step={step} {metrics}", flush=True)
            if step % training["checkpoint_every_steps"] == 0:
                saved = save_checkpoint(model, optimizer, checkpoint_dir, step, epoch, batch_index + 1, (loss_path, metrics_path))
                print(f"[checkpoint] saved to {saved}", flush=True)
        resume_batch_index = 0

    metrics = record_eval(model, eval_loader, device, metrics_path, step)
    plot_training_loss(loss_path, plots / "critic_loss.png")
    plot_critic(metrics_path, plots / "critic_metrics.png")
    output = Path(training["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    trainable = {name: tensor for name, tensor in model.state_dict().items() if not name.startswith("encoder.")}
    torch.save(trainable, output / "critic.pt")
    keep_history(output, loss_path, metrics_path)
    print(f"[done] step={step} metrics={metrics}", flush=True)


if __name__ == "__main__":
    main()
