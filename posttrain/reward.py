from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor


WINDOW_24H = 24 * 60 * 60 // 5 


@dataclass
class RewardWeights:
    alpha: float = 1.0
    beta: float = 1.0
    lambda_novelty: float = 1.0
    lambda_trend: float = 1.0
    lambda_diversity: float = 0.1
    lambda_repeat: float = 1.0
    similar_cosine: float = 0.95


def engagement_stats(mu: Tensor) -> tuple[Tensor, Tensor]:
    return mu.mean(), mu.std(unbiased=False).clamp(min=1e-6)


def track_quality(
    mode: str,
    mu_eng: Tensor,
    like_logit: Tensor,
    dislike_logit: Tensor,
    mu_mean: Tensor,
    mu_std: Tensor,
    weights: RewardWeights,
    novelty: Tensor | None = None,
    trend: Tensor | None = None,
) -> Tensor:
    z = (mu_eng - mu_mean) / mu_std
    quality = z + weights.alpha * torch.sigmoid(like_logit) - weights.beta * torch.sigmoid(dislike_logit)
    if mode == "interest":
        return quality
    if mode == "explore":
        if novelty is None:
            raise ValueError("explore reward needs novelty")
        return quality + weights.lambda_novelty * novelty
    if mode == "trend":
        if trend is None:
            raise ValueError("trend reward needs trend")
        return quality + weights.lambda_trend * trend
    raise ValueError(f"unknown reward mode: {mode}")


def novelty(history: Tensor, candidates: Tensor, audio: Tensor, num_items: int) -> Tensor:
    single = candidates.dim() == 1
    if single:
        candidates = candidates.unsqueeze(1)
    history_ok = (history >= 0) & (history < num_items)
    hist = audio[history.clamp(0, num_items - 1)]
    cand = audio[candidates.clamp(0, num_items - 1)]
    cosine = torch.einsum("bkd,bld->bkl", cand, hist)
    cosine = cosine.masked_fill(~history_ok.unsqueeze(1), float("-inf"))
    nearest, _ = cosine.max(dim=-1)
    score = 1 - nearest
    score = score.masked_fill(~torch.isfinite(nearest), 1)
    return score.squeeze(1) if single else score


def diversity(audio_rows: Tensor) -> Tensor:
    batch, length, _ = audio_rows.shape
    if length < 2:
        return audio_rows.new_zeros(batch)
    cosine = torch.einsum("bid,bjd->bij", audio_rows, audio_rows)
    present = audio_rows.norm(dim=-1) > 0
    pair = present.unsqueeze(2) & present.unsqueeze(1)
    eye = torch.eye(length, dtype=torch.bool, device=audio_rows.device)
    pair = pair & ~eye.unsqueeze(0)
    distance = (1 - cosine).masked_fill(~pair, 0)
    count = pair.sum(dim=(1, 2)).clamp(min=1)
    return distance.sum(dim=(1, 2)) / count


def repeat_penalty(
    items: Tensor,
    audio_rows: Tensor,
    similar_cosine: float,
    artist: Tensor | None = None,
) -> Tensor:
    batch, length = items.shape
    if length < 2:
        return items.new_zeros(batch, dtype=audio_rows.dtype)
    unique = torch.tensor(
        [row.unique().numel() for row in items],
        device=items.device,
        dtype=audio_rows.dtype,
    )
    exact = 1 - unique / length
    cosine = torch.einsum("bid,bjd->bij", audio_rows, audio_rows)
    upper = torch.triu(torch.ones(length, length, dtype=torch.bool, device=items.device), diagonal=1)
    near = (cosine > similar_cosine) & upper.unsqueeze(0)
    penalty = exact + near.float().sum(dim=(1, 2)) / upper.sum().clamp(min=1)
    if artist is not None:
        artists = artist[items.clamp(0, artist.numel() - 1)]
        same = artists.unsqueeze(2).eq(artists.unsqueeze(1)) & upper
        known = (artists >= 0).unsqueeze(2) & (artists >= 0).unsqueeze(1)
        penalty = penalty + (same & known).float().sum(dim=(1, 2)) / upper.sum().clamp(min=1)
    return penalty


def position_weights(length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    ranks = torch.arange(1, length + 1, device=device, dtype=dtype)
    return 1 / torch.log2(ranks + 1)


def list_reward(
    quality: Tensor,
    items: Tensor,
    audio_rows: Tensor,
    weights: RewardWeights,
    artist: Tensor | None = None,
) -> Tensor:
    discount = position_weights(quality.size(1), quality.device, quality.dtype)
    ranked = (quality * discount).sum(dim=-1)
    varied = diversity(audio_rows)
    repeated = repeat_penalty(items, audio_rows, weights.similar_cosine, artist)
    return ranked + weights.lambda_diversity * varied - weights.lambda_repeat * repeated


class RecentPopularity:
    def __init__(self, num_items: int, window: int = WINDOW_24H):
        self.num_items = num_items
        self.window = window
        self.counts = torch.zeros(num_items)
        self._events: deque[tuple[int, int]] = deque()

    def update(self, timestamp: int, tokens: Tensor) -> None:
        self._expire(timestamp)
        for token in tokens.detach().cpu().tolist():
            token = int(token)
            if token < 0 or token >= self.num_items:
                continue
            self._events.append((int(timestamp), token))
            self.counts[token] += 1

    def score(self, tokens: Tensor) -> Tensor:
        return torch.log1p(self.counts[tokens.clamp(0, self.num_items - 1)].to(tokens.device))

    def _expire(self, timestamp: int) -> None:
        limit = timestamp - self.window
        while self._events and self._events[0][0] <= limit:
            _, token = self._events.popleft()
            self.counts[token] -= 1
