"""PPO-clip baseline (Schulman et al. 2017) - "PPO with caching" in Sec. 6.2.

Standard continuous-control PPO: Gaussian policy with state-independent log-std, separate value network,
GAE(lambda), clipped surrogate, several epochs of mini-batch Adam per rollout. One rollout = one episode
(n_s steps). The environment clips actions to [-1, 1]; log-probabilities are of the unclipped sample.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from .common import mlp


class GaussianPolicyValue(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, init_log_std):
        super().__init__()
        self.mu = mlp(obs_dim, act_dim, hidden)
        self.v = mlp(obs_dim, 1, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def dist(self, obs):
        return Normal(self.mu(obs), self.log_std.exp())

    def value(self, obs):
        return self.v(obs).squeeze(-1)


class PPOAgent:
    name = "ppo"
    on_policy = True

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self.net = GaussianPolicyValue(obs_dim, act_dim, cfg["hidden"], cfg["init_log_std"])
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"])
        self._reset_rollout()
        self.last_losses = {}

    def _reset_rollout(self):
        self.ro = {"obs": [], "act": [], "logp": [], "rew": [], "val": []}
        self._last_obs2 = None

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        o = torch.as_tensor(obs).unsqueeze(0)
        d = self.net.dist(o)
        a = d.mean if deterministic else d.sample()
        if not deterministic:
            self._pending = (obs, a.squeeze(0).numpy(), d.log_prob(a).sum(-1).item(), self.net.value(o).item())
        return a.squeeze(0).numpy()

    def observe(self, o, a, r, o2):
        obs, act, logp, val = self._pending
        for k, v in zip(("obs", "act", "logp", "val", "rew"), (obs, act, logp, val, r)):
            self.ro[k].append(v)
        self._last_obs2 = o2

    def end_episode(self) -> dict:
        c = self.cfg
        obs = torch.as_tensor(np.array(self.ro["obs"]))
        act = torch.as_tensor(np.array(self.ro["act"]))
        old_logp = torch.as_tensor(self.ro["logp"], dtype=torch.float32)
        rew = np.array(self.ro["rew"], np.float32)
        val = np.array(self.ro["val"], np.float32)
        with torch.no_grad():   # time-limit truncation: bootstrap from V(s_T)
            last_v = self.net.value(torch.as_tensor(self._last_obs2).unsqueeze(0)).item()
        adv = np.zeros_like(rew)
        gae, nxt = 0.0, last_v
        for t in reversed(range(len(rew))):
            delta = rew[t] + c["gamma"] * nxt - val[t]
            gae = delta + c["gamma"] * c["gae_lambda"] * gae
            adv[t], nxt = gae, val[t]
        ret = torch.as_tensor(adv + val)
        adv = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8))
        n = len(rew)
        for _ in range(c["epochs"]):
            perm = self.rng.permutation(n)
            for s in range(0, n, c["minibatch_size"]):
                idx = torch.as_tensor(perm[s:s + c["minibatch_size"]])
                d = self.net.dist(obs[idx])
                logp = d.log_prob(act[idx]).sum(-1)
                ratio = (logp - old_logp[idx]).exp()
                l_clip = -torch.min(ratio * adv[idx], ratio.clamp(1 - c["clip"], 1 + c["clip"]) * adv[idx]).mean()
                l_v = ((self.net.value(obs[idx]) - ret[idx]) ** 2).mean()
                loss = l_clip + c["value_coef"] * l_v - c["entropy_coef"] * d.entropy().sum(-1).mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c["max_grad_norm"])
                self.opt.step()
        self.last_losses = {"loss_pi": l_clip.item(), "loss_v": l_v.item()}
        self._reset_rollout()
        return self.last_losses

    def state_dict(self):
        return {"net": self.net.state_dict()}

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd["net"])
