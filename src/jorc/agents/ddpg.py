"""DDPG baseline (Lillicrap et al. 2016) - "DDPG with caching" in Sec. 6.2.

The paper gives no DDPG specifics beyond the shared backbone (3x400), Adam and the shared learning rate.
Standard choices: deterministic tanh actor, single critic, target networks with soft update tau,
Gaussian exploration noise (std 0.1, Fujimoto et al. 2018) instead of OU noise.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from .common import ReplayBuffer, mlp, soft_update
from .sac import QNet


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.body = mlp(obs_dim, act_dim, hidden)

    def forward(self, obs):
        return torch.tanh(self.body(obs))


class DDPGAgent:
    name = "ddpg"

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        h = cfg["hidden"]
        self.actor, self.critic = DeterministicActor(obs_dim, act_dim, h), QNet(obs_dim, act_dim, h)
        self.actor_t, self.critic_t = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        for p in list(self.actor_t.parameters()) + list(self.critic_t.parameters()):
            p.requires_grad_(False)
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=cfg["lr"])
        self.opt_q = torch.optim.Adam(self.critic.parameters(), lr=cfg["lr"])
        self.buffer = ReplayBuffer(obs_dim, act_dim, cfg["buffer_size"], rng)
        self.gamma, self.tau, self.batch = cfg["gamma"], cfg["tau"], cfg["batch_size"]
        self.noise = cfg["exploration_noise_std"]
        self.last_losses = {}

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        a = self.actor(torch.as_tensor(obs).unsqueeze(0)).squeeze(0).numpy()
        if not deterministic:
            a = np.clip(a + self.noise * self.rng.standard_normal(a.shape), -1.0, 1.0).astype(np.float32)
        return a

    def observe(self, o, a, r, o2):
        self.buffer.add(o, a, r, o2)

    def ready(self):
        return len(self.buffer) >= max(self.batch, self.cfg["learning_starts"])

    def update(self):
        o, a, r, o2 = self.buffer.sample(self.batch)
        with torch.no_grad():
            y = r.squeeze(-1) + self.gamma * self.critic_t(o2, self.actor_t(o2)).squeeze(-1)
        loss_q = ((self.critic(o, a).squeeze(-1) - y) ** 2).mean()
        self.opt_q.zero_grad()
        loss_q.backward()
        self.opt_q.step()
        for p in self.critic.parameters():
            p.requires_grad_(False)
        loss_pi = -self.critic(o, self.actor(o)).mean()
        self.opt_pi.zero_grad()
        loss_pi.backward()
        self.opt_pi.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)
        soft_update(self.actor_t, self.actor, self.tau)
        soft_update(self.critic_t, self.critic, self.tau)
        self.last_losses = {"loss_q": loss_q.item(), "loss_pi": loss_pi.item()}
        return self.last_losses

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
