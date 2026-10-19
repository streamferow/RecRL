import builtins
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from eval.evaluate import append_metrics, evaluate
from eval.plotting import (
    append_training_loss,
    plot_metrics_at_k,
    plot_training_loss,
)
from model.dataset import (
    PreparedNextItemDataset,
    load_prepared_metadata,
    load_prepared_sequences,
)
from model.gpt2 import (
    build_model,
    count_parameters,
    find_latest_checkpoint,
    get_device,
    load_checkpoint,
    optimizer_step,
    parameter_groups,
    save_checkpoint,
    save_model,
    train_batch,
)


def setup_distributed() -> tuple[int, int, int]:
    """Returns (rank, world_size, local_rank); initialises NCCL when launched by torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size < 2:
        return 0, 1, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    else:
        dist.init_process_group("gloo")
    return dist.get_rank(), world_size, local_rank


def main() -> None:
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    if not is_main:
        builtins.print = lambda *args, **kwargs: None

    print("[main] loading train.yaml", flush=True)
    with Path("train.yaml").open() as file:
        config = yaml.safe_load(file)

    torch.manual_seed(config["seed"])
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    loss_history_path = Path(evaluation_config["training_loss_file"])
    plots_dir = Path(evaluation_config["plots_dir"])

    print("[main] loading dataset", flush=True)
    prepared_path = Path(data_config["prepared_dir"])
    metadata, item_ids, unk_token_id, pad_token_id = (
        load_prepared_metadata(prepared_path)
    )
    if metadata["context_length"] != model_config["context_length"]:
        raise ValueError(
            "prepared context_length must equal model context_length"
        )
    train_data = PreparedNextItemDataset(prepared_path)
    validation_sequences = load_prepared_sequences(
        prepared_path, "validation"
    )
    test_sequences = load_prepared_sequences(prepared_path, "test")
    stats = metadata["stats"]
    train_users = stats["train_users"]
    validation_users = stats["validation_users"]
    test_users = stats["test_users"]
    coverage_value = stats["known_events"] / max(stats["events"], 1)
    print(
        f"[main] users={stats['users']:,} top_items={len(item_ids):,} "
        f"train_users={train_users:,} "
        f"validation_users={validation_users:,} "
        f"test_users={test_users:,} "
        f"coverage(all prepared events)={coverage_value:.3f}",
        flush=True,
    )
    max_epochs = training_config["epochs"]
    print(
        f"[data] train_tokens_per_epoch={train_data.token_count:,} "
        f"epochs={max_epochs} "
        f"total_tokens={train_data.token_count * max_epochs:,}",
        flush=True,
    )

    device = (
        torch.device("cuda", local_rank)
        if world_size > 1 and torch.cuda.is_available()
        else get_device()
    )
    checkpoint_dir = Path(training_config["checkpoint_dir"])
    resume_from = training_config.get("resume_from")
    if resume_from == "auto":
        resume_from = find_latest_checkpoint(checkpoint_dir)
    elif resume_from:
        resume_from = Path(resume_from)

    print("[model] building GPT-2 next-item architecture", flush=True)
    model = build_model(
        num_items=len(item_ids),
        context_length=model_config["context_length"],
        hidden_size=model_config["hidden_size"],
        layers=model_config["layers"],
        heads=model_config["heads"],
        feed_forward_size=model_config["feed_forward_size"],
        dropout=model_config["dropout"],
        unk_token_id=unk_token_id,
        pad_token_id=pad_token_id,
    ).to(device)
    print(
        f"[model] parameters={count_parameters(model):,}",
        flush=True,
    )

    print(f"[train] device={device} world_size={world_size}", flush=True)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, training_config["weight_decay"]),
        lr=training_config["learning_rate"],
        betas=tuple(training_config.get("betas", [0.9, 0.999])),
    )
    accumulation_steps = training_config["gradient_accumulation_steps"]
    batch_size = training_config["batch_size"]
    effective_batch_size = batch_size * accumulation_steps * world_size
    if effective_batch_size != training_config["effective_batch_size"]:
        raise ValueError(
            f"effective_batch_size must equal batch_size * "
            f"gradient_accumulation_steps * world_size: "
            f"{batch_size} * {accumulation_steps} * {world_size} = "
            f"{effective_batch_size}, config says "
            f"{training_config['effective_batch_size']}"
        )
    shard_size = len(train_data) // world_size
    num_batches = (
        shard_size // batch_size
        if training_config["drop_last"]
        else math.ceil(shard_size / batch_size)
    )
    updates_per_epoch = math.ceil(num_batches / accumulation_steps)
    total_updates = max_epochs * max(
        1, updates_per_epoch
    )
    warmup_steps = round(
        total_updates * training_config.get("warmup_ratio", 0.0)
    )

    def learning_rate_schedule(update: int) -> float:
        if update < warmup_steps:
            return update / max(1, warmup_steps)
        progress = (update - warmup_steps) / max(
            1, total_updates - warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, learning_rate_schedule
    )
    start_epoch = 0
    step = 0
    tokens_seen = 0
    resume_batch_index = 0
    if resume_from:
        state = load_checkpoint(
            model, resume_from, optimizer, device, scheduler
        )
        step = state["step"]
        start_epoch = state["epoch"]
        resume_batch_index = state.get("batch_index", 0)
        tokens_seen = state.get("tokens_seen", 0)

    # `model` stays the unwrapped module for evaluation, checkpointing and saving.
    trainee = (
        DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
        )
        if world_size > 1
        else model
    )

    print(
        f"[train] effective_batch_size={effective_batch_size} "
        f"updates_per_epoch={updates_per_epoch:,} "
        f"target_epochs={max_epochs}",
        flush=True,
    )
    use_amp = (
        training_config["mixed_precision"] == "fp16" and device.type == "cuda"
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    optimizer.zero_grad(set_to_none=True)

    window_loss = 0.0
    window_targets = 0
    pending_batches: list[tuple[int, dict[str, torch.Tensor]]] = []
    print("[train] starting", flush=True)
    for epoch in range(start_epoch, max_epochs):
        drop_last = training_config["drop_last"] or world_size > 1
        sampler = (
            DistributedSampler(
                train_data,
                shuffle=True,
                drop_last=drop_last,
                seed=config["seed"] + epoch,
            )
            if world_size > 1
            else None
        )
        if sampler is not None:
            sampler.set_epoch(epoch)
        loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=training_config["num_workers"],
            drop_last=drop_last,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(config["seed"] + epoch),
        )
        trainee.train()
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < resume_batch_index:
                continue
            pending_batches.append((batch_index, batch))
            if (
                len(pending_batches) < accumulation_steps
                and batch_index != len(loader) - 1
            ):
                continue

            local_valid_targets = sum(
                int(batch["labels"].ne(-100).sum())
                for _, batch in pending_batches
            )
            if world_size > 1:
                # DDP averages gradients, so every rank must divide by the same
                # global target count for the update to stay correctly weighted.
                counter = torch.tensor(
                    [local_valid_targets], dtype=torch.float64, device=device
                )
                dist.all_reduce(counter)
                total_valid_targets = int(counter.item()) / world_size
            else:
                total_valid_targets = local_valid_targets
            for position, (_, pending_batch) in enumerate(pending_batches):
                is_last = position == len(pending_batches) - 1
                skip_sync = world_size > 1 and not is_last
                with trainee.no_sync() if skip_sync else nullcontext():
                    loss_sum, valid_targets = train_batch(
                        trainee,
                        pending_batch,
                        device,
                        max(1.0, total_valid_targets),
                        scaler,
                    )
                window_loss += loss_sum
                window_targets += valid_targets
                tokens_seen += int(pending_batch["attention_mask"].sum())

            last_batch_index = pending_batches[-1][0]
            pending_batches = []
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_config["max_grad_norm"]
            )
            optimizer_step(optimizer, scaler, scheduler)
            mean_loss = window_loss / max(1, window_targets)
            step += 1
            if step == 1 or step % training_config["log_every_steps"] == 0:
                window_loss = 0.0
                window_targets = 0
                if is_main:
                    append_training_loss(
                        loss_history_path,
                        epoch + 1,
                        step,
                        mean_loss,
                    )
                print(
                    f"[train] epoch={epoch + 1} update={step} "
                    f"loss={mean_loss:.4f}",
                    flush=True,
                )

            if is_main and step % training_config["eval_every_steps"] == 0:
                print(f"[eval] running at update={step}", flush=True)
                metrics = evaluate(
                    model,
                    validation_sequences,
                    model_config["context_length"],
                    evaluation_config["batch_size"],
                    evaluation_config["k_values"],
                    unk_token_id,
                )
                trainee.train()
                append_metrics(
                    Path(evaluation_config["metrics_file"]),
                    metrics,
                    step,
                    validation_sequences.valid_count,
                )
                plot_training_loss(
                    loss_history_path, plots_dir / "training_loss.png"
                )
                plot_metrics_at_k(
                    Path(evaluation_config["metrics_file"]),
                    plots_dir / f"metrics_at_{evaluation_config['plot_k']}.png",
                    evaluation_config["plot_k"],
                )
                print(
                    f"step={step} loss={mean_loss:.4f} "
                    f"metrics={metrics}",
                    flush=True,
                )

            if is_main and step % training_config["checkpoint_every_steps"] == 0:
                checkpoint = save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    Path(training_config["checkpoint_dir"]),
                    step,
                    epoch,
                    last_batch_index + 1,
                    tokens_seen,
                )
                print(f"[checkpoint] saved to {checkpoint}", flush=True)
            # Rank 0 eval/checkpoint must not let the other rank enter the next all_reduce.
            if world_size > 1 and (
                step % training_config["eval_every_steps"] == 0
                or step % training_config["checkpoint_every_steps"] == 0
            ):
                dist.barrier()
        resume_batch_index = 0

        pending_batches = []

    if is_main:
        print("[eval] running final test evaluation", flush=True)
        test_metrics = evaluate(
            model,
            test_sequences,
            model_config["context_length"],
            evaluation_config["batch_size"],
            evaluation_config["k_values"],
            unk_token_id,
        )
        append_metrics(
            Path(evaluation_config["metrics_file"]),
            test_metrics,
            step,
            test_sequences.valid_count,
            split="test",
        )
        print(f"[eval] test metrics={test_metrics}", flush=True)
        plot_training_loss(loss_history_path, plots_dir / "training_loss.png")
        plot_metrics_at_k(
            Path(evaluation_config["metrics_file"]),
            plots_dir / f"metrics_at_{evaluation_config['plot_k']}.png",
            evaluation_config["plot_k"],
        )
        save_model(model, Path(training_config["output_dir"]))
        with (Path(training_config["output_dir"]) / "item_ids.json").open("w") as file:
            json.dump(item_ids, file)
        print(
            f"[done] saved final model to {training_config['output_dir']}",
            flush=True,
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
