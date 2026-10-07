"""Soft actor-critic with automatic temperature (Algorithm 1, eqs.(23)-(32); Haarnoja et al. [45]).

Fidelity notes (details in REPRODUCTION_ASSUMPTIONS.md):
* eq.(27) as printed omits alpha in front of log pi in the target; eq.(25) and [45] include it -> we include it.
* eq.(29) reparameterisation A = mu + psi * sigma; bounded actions use the tanh squashing of [45] with the
  log-det-Jacobian correction (the paper's actions lie in a bounded set but the squashing is not described).
* eq.(31) is written in alpha; [45]'s reference implementation optimises log(alpha) -> configurable.
* Clipped double-Q: min of two critics in the target and in the actor loss (Sec. 5.1.2).
* Time-limit truncation at n_s steps is not a terminal state -> no done-mask (bootstrapping always).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ReplayBuffer, mlp, soft_update

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.body = mlp(obs_dim, 2 * act_dim, hidden)
        self.act_dim = act_dim

    def forward(self, obs, deterministic=False, with_logp=True):
        mu, log_std = self.body(obs).chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)      # eq.(29)
        a = torch.tanh(u)
        if not with_logp:
            return a, None
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)   # tanh Jacobian
        return a, logp


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.q = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], -1))


class SACAgent:
    name = "sac"

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg
        h = cfg["hidden"]
        self.actor = GaussianActor(obs_dim, act_dim, h)
        self.q1, self.q2 = QNet(obs_dim, act_dim, h), QNet(obs_dim, act_dim, h)
        self.q1_t, self.q2_t = QNet(obs_dim, act_dim, h), QNet(obs_dim, act_dim, h)
        self.q1_t.load_state_dict(self.q1.state_dict())       # Alg.1 line 2
        self.q2_t.load_state_dict(self.q2.state_dict())
        for p in list(self.q1_t.parameters()) + list(self.q2_t.parameters()):
            p.requires_grad_(False)
        lr = cfg["lr"]
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.log_param = cfg["alpha_parameterization"] == "log"
        init = math.log(cfg["alpha0"]) if self.log_param else cfg["alpha0"]
        self.alpha_param = torch.tensor(init, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.alpha_param], lr=lr)   # same Adam + lr (Sec. 6.1)
        self.target_entropy = cfg["target_entropy_per_action_dim"] * act_dim   # H = -5N
        self.gamma, self.tau = cfg["gamma"], cfg["tau"]
        self.buffer = ReplayBuffer(obs_dim, act_dim, cfg["buffer_size"], rng)
        self.batch = cfg["batch_size"]
        self.last_losses = {}

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_param.exp() if self.log_param else self.alpha_param.clamp_min(1e-8)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        a, _ = self.actor(torch.as_tensor(obs).unsqueeze(0), deterministic, with_logp=False)
        return a.squeeze(0).numpy()

    def observe(self, o, a, r, o2):
        self.buffer.add(o, a, r, o2)

    def ready(self) -> bool:
        return len(self.buffer) >= max(self.batch, self.cfg["learning_starts"])

    def update(self) -> dict:
        o, a, r, o2 = self.buffer.sample(self.batch)                       # Alg.1 line 10
        alpha = self.alpha.detach()
        with torch.no_grad():                                              # target of eq.(26)/(27)
            a2, logp2 = self.actor(o2)
            q_t = torch.min(self.q1_t(o2, a2), self.q2_t(o2, a2)).squeeze(-1)
            y = r.squeeze(-1) + self.gamma * (q_t - alpha * logp2)
        q1, q2 = self.q1(o, a).squeeze(-1), self.q2(o, a).squeeze(-1)
        loss_q = 0.5 * ((q1 - y) ** 2).mean() + 0.5 * ((q2 - y) ** 2).mean()   # eq.(26), line 11
        self.opt_q.zero_grad()
        loss_q.backward()
        self.opt_q.step()

        for p in list(self.q1.parameters()) + list(self.q2.parameters()):
            p.requires_grad_(False)
        a_new, logp = self.actor(o)
        q_new = torch.min(self.q1(o, a_new), self.q2(o, a_new)).squeeze(-1)
        loss_pi = (alpha * logp - q_new).mean()                            # eq.(28), line 12
        self.opt_pi.zero_grad()
        loss_pi.backward()
        self.opt_pi.step()
        for p in list(self.q1.parameters()) + list(self.q2.parameters()):
            p.requires_grad_(True)

        a_term = self.alpha_param if self.log_param else self.alpha
        loss_alpha = -(a_term * (logp.detach() + self.target_entropy)).mean()   # eq.(31), line 13
        self.opt_alpha.zero_grad()
        loss_alpha.backward()
        self.opt_alpha.step()

        soft_update(self.q1_t, self.q1, self.tau)                          # eq.(32), line 14
        soft_update(self.q2_t, self.q2, self.tau)
        self.last_losses = {"loss_q": loss_q.item(), "loss_pi": loss_pi.item(),
                            "alpha": float(self.alpha.item()), "entropy": float(-logp.mean().item())}
        return self.last_losses

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
                "alpha": self.alpha_param.detach().clone()}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.q1.load_state_dict(sd["q1"])
        self.q2.load_state_dict(sd["q2"])
        self.alpha_param.data.copy_(sd["alpha"])
