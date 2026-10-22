import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.gpt2 import GPT2NextItem


class Critic(nn.Module):
    def __init__(self, encoder: GPT2NextItem, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.item_embedding.embedding_dim
        for parameter in encoder.parameters():
            parameter.requires_grad = False

        self.user_projection = nn.Linear(hidden, 256)
        self.item_projection = nn.Linear(hidden, 256)
        self.mlp = nn.Sequential(
            nn.Linear(256 * 3 + 1, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.engagement_head = nn.Linear(256, 2)
        self.like_head = nn.Linear(256, 1)
        self.dislike_head = nn.Linear(256, 1)


    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self


    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)


    def forward(self, input_ids: Tensor, attention_mask: Tensor, item: Tensor) -> dict[str, Tensor]:
        if (item >= self.encoder.num_items).any():
            raise ValueError("candidate item must be a known track, not unk or pad")
        user = self.user_projection(self._user_state(input_ids, attention_mask))
        track = self.item_projection(self._track_embedding(item))
        similarity = (user * track).sum(dim=-1, keepdim=True)
        hidden = self.mlp(torch.cat((user, track, user * track, similarity), dim=-1))
        engagement = self.engagement_head(hidden)
        return {
            "mu_eng": engagement[:, 0],
            "log_var_eng": engagement[:, 1],
            "like_logit": self.like_head(hidden).squeeze(-1),
            "dislike_logit": self.dislike_head(hidden).squeeze(-1),
        }


    def _user_state(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        with torch.no_grad():
            hidden = self.encoder.encode(input_ids, attention_mask)
        rightmost = attention_mask.size(1) - 1 - attention_mask.long().flip(1).argmax(1)
        index = rightmost.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        state = hidden.gather(1, index).squeeze(1)
        empty = attention_mask.sum(1) == 0
        return state.masked_fill(empty.unsqueeze(1), 0)


    def _track_embedding(self, item: Tensor) -> Tensor:
        with torch.no_grad():
            return self.encoder.item_embedding(item)


def critic_loss(
    prediction: dict[str, Tensor],
    y_eng: Tensor,
    y_like: Tensor,
    y_dislike: Tensor,
) -> dict[str, Tensor]:
    engagement = _gaussian_nll(prediction["mu_eng"], prediction["log_var_eng"], y_eng)
    like = _binary_nll(prediction["like_logit"], y_like)
    dislike = _binary_nll(prediction["dislike_logit"], y_dislike)
    return {
        "loss": engagement + like + dislike,
        "engagement": engagement,
        "like": like,
        "dislike": dislike,
    }


def _gaussian_nll(mu: Tensor, log_var: Tensor, target: Tensor) -> Tensor:
    mask = torch.isfinite(target)
    if not mask.any():
        return mu.new_zeros(())
    mu = mu[mask]
    log_var = log_var[mask].clamp(-8, 8)
    nll = 0.5 * (log_var + (target[mask] - mu).square() * (-log_var).exp())
    return nll.mean()


def _binary_nll(logit: Tensor, target: Tensor) -> Tensor:
    mask = target != -1
    if not mask.any():
        return logit.new_zeros(())
    return F.binary_cross_entropy_with_logits(logit[mask], target[mask].float())
