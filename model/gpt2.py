import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F



class GPT2Block(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        heads: int, 
        feed_forward_size: int, 
        dropout: float
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, feed_forward_size),
            nn.GELU(),
            nn.Linear(feed_forward_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)


    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(hidden)
        length = hidden.shape[1]
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=hidden.device), diagonal=1)
        key_padding = ~attention_mask.bool()
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        hidden = hidden + self.dropout(attended)
        return hidden + self.feed_forward(self.norm2(hidden))



class GPT2NextItem(nn.Module):
    def __init__(
        self,
        num_items: int,
        context_length: int,
        hidden_size: int = 768,
        layers: int = 12,
        heads: int = 12,
        feed_forward_size: int = 3072,
        dropout: float = 0.1,
        unk_token_id: int | None = None,
        pad_token_id: int | None = None,
    ):
        super().__init__()

        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")

        self.num_items = num_items
        self.unk_token_id = unk_token_id
        self.pad_token_id = pad_token_id
        self.item_embedding = nn.Embedding(num_items + 2, hidden_size, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(context_length, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([GPT2Block(hidden_size, heads, feed_forward_size, dropout) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(hidden_size)
        self.apply(self._init_weights)
        nn.init.normal_(self.position_embedding.weight, std=0.01)

        residual_std = 0.02 / math.sqrt(2 * layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.feed_forward[2].weight, mean=0.0, std=residual_std)

        if pad_token_id is not None:
            with torch.no_grad():
                self.item_embedding.weight[pad_token_id].zero_()


    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)


    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)[None, :]
        hidden = self.dropout(self.item_embedding(input_ids) + self.position_embedding(positions))
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        return self.final_norm(hidden)


    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(input_ids, attention_mask)
        return hidden @ self.item_embedding.weight[: self.num_items].T


    def next_item_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(input_ids, attention_mask)
        return hidden[:, -1] @ self.item_embedding.weight[: self.num_items].T



def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            if isinstance(module, nn.Linear) and name == "weight":
                decay.append(parameter)
            elif isinstance(module, nn.MultiheadAttention) and "weight" in name:
                decay.append(parameter)
            else:
                no_decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_model(
    num_items: int,
    context_length: int,
    hidden_size: int,
    layers: int,
    heads: int,
    feed_forward_size: int,
    dropout: float,
    unk_token_id: int,
    pad_token_id: int,
) -> GPT2NextItem:
    return GPT2NextItem(
        num_items=num_items,
        context_length=context_length,
        hidden_size=hidden_size,
        layers=layers,
        heads=heads,
        feed_forward_size=feed_forward_size,
        dropout=dropout,
        unk_token_id=unk_token_id,
        pad_token_id=pad_token_id,
    )


def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def save_checkpoint(
    model: GPT2NextItem,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    checkpoint_dir: Path,
    step: int,
    epoch: int,
    batch_index: int,
    tokens_seen: int,
) -> Path:
    output_dir = checkpoint_dir / f"checkpoint-{step}"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "batch_index": batch_index,
            "tokens_seen": tokens_seen,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        },
        output_dir / "training_state.pt",
    )
    return output_dir


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = [
        path
        for path in checkpoint_dir.glob("checkpoint-*")
        if (path / "training_state.pt").exists()
        and (path / "model.pt").exists()
    ]
    return max(
        checkpoints,
        key=lambda path: int(path.name.rsplit("-", 1)[1]),
        default=None,
    )


def load_checkpoint(
    model: GPT2NextItem,
    checkpoint_dir: Path,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict:
    model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location=device))
    state = torch.load(
        checkpoint_dir / "training_state.pt",
        map_location=device,
        weights_only=False,
    )
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state


def save_model(model: GPT2NextItem, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")


def train_batch(
    model: GPT2NextItem,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    loss_normalizer: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> tuple[float, int]:
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler is not None):
        logits = model.logits(batch["input_ids"], batch["attention_mask"])
        labels = batch["labels"].reshape(-1)
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels,
            ignore_index=-100,
            reduction="sum",
        )
        loss = loss_sum / loss_normalizer
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    return loss_sum.item(), int((labels != -100).sum())


def optimizer_step(
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
