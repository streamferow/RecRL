import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.gpt2 import GPT2NextItem


NEEDS = ("interest", "explore", "trend")


class NeedPolicy(nn.Module):
    def __init__(self, backbone: GPT2NextItem):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.item_embedding.embedding_dim
        self.need_embedding = nn.Embedding(len(NEEDS), hidden)
        nn.init.zeros_(self.need_embedding.weight)

    def next_logits(self, input_ids: Tensor, attention_mask: Tensor, need: Tensor | str) -> Tensor:
        if isinstance(need, str):
            index = torch.full((input_ids.size(0),), NEEDS.index(need), device=input_ids.device, dtype=torch.long)
        else:
            index = need
        extra = self.need_embedding(index)[:, None, :]
        hidden = self.backbone.encode(input_ids, attention_mask, extra)
        return hidden[:, -1] @ self.backbone.item_embedding.weight[: self.backbone.num_items].T


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module


def unfreeze_tail(backbone: GPT2NextItem, n_blocks: int, *also: nn.Module) -> None:
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    for block in backbone.blocks[-n_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for module in also:
        for parameter in module.parameters():
            parameter.requires_grad = True


def select_candidates(
    logits: Tensor,
    novelty_score: Tensor,
    trend_score: Tensor,
    policy_count: int = 30,
    extra_count: int = 10,
) -> Tensor:
    policy = logits.topk(policy_count, dim=-1).indices
    bonus = novelty_score + trend_score
    bonus = bonus.scatter(-1, policy, float("-inf"))
    extra = bonus.topk(extra_count, dim=-1).indices
    return torch.cat((policy, extra), dim=-1)


def mask_logits(logits: Tensor, allowed: Tensor) -> Tensor:
    masked = torch.full_like(logits, float("-inf"))
    chosen = logits.gather(-1, allowed)
    return masked.scatter(-1, allowed, chosen)


def _append(input_ids: Tensor, attention_mask: Tensor, token: Tensor, context_length: int) -> tuple[Tensor, Tensor]:
    token = token.view(-1, 1)
    ones = torch.ones(token.shape, dtype=attention_mask.dtype, device=attention_mask.device)
    if input_ids.size(1) >= context_length:
        return torch.cat((input_ids[:, 1:], token), dim=1), torch.cat((attention_mask[:, 1:], ones), dim=1)
    return torch.cat((input_ids, token), dim=1), torch.cat((attention_mask, ones), dim=1)


def rollout_list(
    policy: NeedPolicy,
    input_ids: Tensor,
    attention_mask: Tensor,
    need: Tensor | str,
    novelty_score: Tensor,
    trend_score: Tensor,
    length: int = 10,
    policy_count: int = 30,
    extra_count: int = 10,
) -> dict[str, Tensor]:
    items = []
    log_probs = []
    ids, mask = input_ids, attention_mask
    context_length = policy.backbone.position_embedding.num_embeddings
    for _ in range(length):
        logits = policy.next_logits(ids, mask, need)
        pool = select_candidates(logits, novelty_score, trend_score, policy_count, extra_count)
        logits = mask_logits(logits, pool)
        if items:
            chosen = torch.stack(items, dim=1)
            logits = logits.scatter(-1, chosen, float("-inf"))
        distribution = Categorical(logits=logits)
        pick = distribution.sample()
        items.append(pick)
        log_probs.append(distribution.log_prob(pick))
        ids, mask = _append(ids, mask, pick, context_length)
    return {"items": torch.stack(items, dim=1), "log_prob": torch.stack(log_probs, dim=1)}


def counterfactual_credit(items: Tensor, reward_fn, outside: Tensor | None = None) -> Tensor:
    base = reward_fn(items)
    batch, length = items.shape
    credit = items.new_zeros(batch, length, dtype=base.dtype)
    for position in range(length - 1):
        deltas = []
        for other in range(position + 1, length):
            swapped = items.clone()
            left = swapped[:, position].clone()
            swapped[:, position] = swapped[:, other]
            swapped[:, other] = left
            deltas.append(base - reward_fn(swapped))
        credit[:, position] = torch.stack(deltas, dim=0).mean(dim=0)
    if outside is not None:
        replaced = items.clone()
        replaced[:, -1] = outside
        credit[:, -1] = base - reward_fn(replaced)
    else:
        credit[:, -1] = (base - base.mean()) / base.std(unbiased=False).clamp(min=1e-6)
    return credit


def confidence_weights(log_var: Tensor, eps: float = 1e-6, low: float = 0.1, high: float = 10.0) -> Tensor:
    precision = (1 / (log_var.exp() + eps)).clamp(low, high)
    return precision / precision.mean().clamp(min=1e-6)


def weighted_advantages(credit: Tensor, log_var: Tensor, eps: float = 1e-6, low: float = 0.1, high: float = 10.0) -> Tensor:
    centered = credit - credit.mean()
    normalized = centered / credit.std(unbiased=False).clamp(min=1e-6)
    return confidence_weights(log_var, eps, low, high) * normalized


def grpo_loss(
    log_prob: Tensor,
    log_prob_old: Tensor,
    log_prob_ref: Tensor,
    advantage: Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
) -> dict[str, Tensor]:
    ratio = (log_prob - log_prob_old).exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage
    policy = -torch.minimum(unclipped, clipped).mean()
    kl = (log_prob - log_prob_ref).mean()
    return {"loss": policy + kl_coef * kl, "policy": policy, "kl": kl}
