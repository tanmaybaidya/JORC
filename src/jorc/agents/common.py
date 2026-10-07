"""Shared building blocks: 3x400 ReLU MLP backbone (Sec. 6.1) and a uniform replay buffer."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def mlp(in_dim: int, out_dim: int, hidden: list[int]) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class ReplayBuffer:
    """Uniform replay D (Alg. 1 line 9). Stored as float32 numpy arrays."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int, rng: np.random.Generator):
        self.cap = int(capacity)
        self.obs = np.zeros((self.cap, obs_dim), np.float32)
        self.act = np.zeros((self.cap, act_dim), np.float32)
        self.rew = np.zeros((self.cap, 1), np.float32)
        self.nxt = np.zeros((self.cap, obs_dim), np.float32)
        self.ptr, self.size, self.rng = 0, 0, rng

    def add(self, o, a, r, o2):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i, 0], self.nxt[i] = o, a, r, o2
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int):
        idx = self.rng.integers(0, self.size, batch)
        f = lambda x: torch.as_tensor(x[idx])
        return f(self.obs), f(self.act), f(self.rew), f(self.nxt)

    def __len__(self):
        return self.size


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """eq.(32): theta' <- tau theta + (1 - tau) theta'."""
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1.0 - tau).add_(sp, alpha=tau)
