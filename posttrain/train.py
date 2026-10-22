import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.plotting import append_training_loss, plot_training_loss
from model.gpt2 import GPT2NextItem, get_device
from posttrain.critic import Critic, critic_loss
from posttrain.eval import keep_history, plot_critic, record_eval, restore_history


COLUMNS = ["context", "attention_mask", "item", "y_eng", "y_like", "y_dislike"]


def _batch(table: pa.Table, context_length: int) -> tuple[torch.Tensor, ...]:
    rows = table.num_rows
    context = np.asarray(table.column("context").combine_chunks().values.to_numpy()).reshape(rows, context_length)
    mask = np.asarray(table.column("attention_mask").combine_chunks().values.to_numpy()).reshape(rows, context_length)
    def tensor(array, dtype):
        return torch.tensor(np.array(array, copy=True), dtype=dtype)

    return (
        tensor(context, torch.long),
        tensor(mask, torch.long),
        tensor(table.column("item").to_numpy(zero_copy_only=False), torch.long),
        tensor(np.asarray(table.column("y_eng").to_numpy(zero_copy_only=False), dtype=np.float32), torch.float32),
        tensor(table.column("y_like").to_numpy(zero_copy_only=False), torch.long),
        tensor(table.column("y_dislike").to_numpy(zero_copy_only=False), torch.long),
    )


def iter_train(path: Path, context_length: int, batch_size: int, skip_rows: int, seed: int):
    """One pass over the parquet. Only one decoded chunk is in memory."""
    parquet = pq.ParquetFile(path)
    rng = np.random.default_rng(seed)
    left = skip_rows
    for record in parquet.iter_batches(batch_size=65536, columns=COLUMNS):
        table = pa.Table.from_batches([record])
        if left:
            drop = min(left, table.num_rows)
            left -= drop
            if drop == table.num_rows:
                continue
            table = table.slice(drop)
        table = table.take(rng.permutation(table.num_rows))
        for start in range(0, table.num_rows - batch_size + 1, batch_size):
            yield _batch(table.slice(start, batch_size), context_length)


def load_eval(path: Path, context_length: int, eval_rows: int) -> TensorDataset:
    parquet = pq.ParquetFile(path)
    parts = []
    got = 0
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=COLUMNS)
        parts.append(table.slice(0, eval_rows - got))
        got += parts[-1].num_rows
        if got >= eval_rows:
            break
    return TensorDataset(*_batch(pa.concat_tables(parts), context_length))


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

    train_path = dataset_dir / "train.parquet"
    train_rows = pq.ParquetFile(train_path).metadata.num_rows
    eval_rows = min(int(training["eval_rows"]), pq.ParquetFile(dataset_dir / "validation.parquet").metadata.num_rows)
    print(f"[critic] train_rows={train_rows:,} eval_rows={eval_rows:,}", flush=True)
    eval_loader = DataLoader(load_eval(dataset_dir / "validation.parquet", context_length, eval_rows), batch_size=training["batch_size"])

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
    print(f"[critic] device={device}", flush=True)
    for epoch in range(start_epoch, int(training["epochs"])):
        skip = resume_batch_index if epoch == start_epoch else 0
        if skip:
            print(f"[critic] jumping to batch {skip}", flush=True)
        model.train()
        window = 0.0
        seen = 0
        batches = iter_train(train_path, context_length, batch_size, skip * batch_size, config["seed"] + epoch)
        for local_index, batch in enumerate(batches):
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
