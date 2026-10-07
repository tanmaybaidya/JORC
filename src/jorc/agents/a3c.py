"""A3C baseline (Mnih et al. 2016) - "A3C with caching" in Sec. 6.2.

Asynchronous workers (torch.multiprocessing, Hogwild-style lock-free updates of a shared model with a shared
Adam optimiser). Each worker owns an environment copy with its own seed, collects t_max-step rollouts and
pushes gradients of  -log pi(a|s) A  - beta H[pi]  + c_v (R - V)^2  to the shared model.
The paper's training budget (n_e episodes) is interpreted as the TOTAL number of episodes over all workers,
so A3C sees the same number of environment interactions as the other methods ([E]).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from .ppo import GaussianPolicyValue


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state[p]
                st["step"] = torch.zeros(1)
                st["exp_avg"] = torch.zeros_like(p.data)
                st["exp_avg_sq"] = torch.zeros_like(p.data)
                st["step"].share_memory_()
                st["exp_avg"].share_memory_()
                st["exp_avg_sq"].share_memory_()

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                st["step"] += 1
                t = st["step"].item()
                st["exp_avg"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["exp_avg_sq"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                denom = (st["exp_avg_sq"] / (1 - b2**t)).sqrt_().add_(group["eps"])
                p.addcdiv_(st["exp_avg"], denom, value=-group["lr"] / (1 - b1**t))


def _worker(wid, shared, opt, cfg_agent, env_cfg, env_seed, episodes_total, steps_per_ep,
            counter, lock, queue, torch_seed):
    from ..env import UAVMECEnv, episode_seed

    torch.set_num_threads(1)
    torch.manual_seed(torch_seed)
    env = UAVMECEnv(env_cfg, env_seed)
    local = GaussianPolicyValue(env.obs_dim, env.action_dim, cfg_agent["hidden"], -0.5)
    g, beta, tmax = cfg_agent["gamma"], cfg_agent["entropy_beta"], cfg_agent["t_max"]
    while True:
        with lock:
            if counter.value >= episodes_total:
                return
            ep = counter.value
            counter.value += 1
        obs = env.reset(episode_seed(env_seed, ep))
        ep_rew, step = 0.0, 0
        infos = []
        while step < steps_per_ep:
            local.load_state_dict(shared.state_dict())
            logps, vals, ents, rews = [], [], [], []
            for _ in range(tmax):
                o = torch.as_tensor(obs).unsqueeze(0)
                d = local.dist(o)
                a = d.sample()
                obs, r, _, info = env.step(a.squeeze(0).numpy())
                logps.append(d.log_prob(a).sum())
                ents.append(d.entropy().sum())
                vals.append(local.value(o).squeeze())
                rews.append(r)
                infos.append(info)
                ep_rew += r
                step += 1
                if step >= steps_per_ep:
                    break
            with torch.no_grad():   # truncation is not terminal -> bootstrap
                R = local.value(torch.as_tensor(obs).unsqueeze(0)).item()
            loss = 0.0
            for t in reversed(range(len(rews))):
                R = rews[t] + g * R
                adv = R - vals[t]
                loss = loss - logps[t] * adv.detach() - beta * ents[t] + cfg_agent["value_coef"] * adv**2
            local.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(local.parameters(), cfg_agent["max_grad_norm"])
            for lp, sp in zip(local.parameters(), shared.parameters()):
                sp._grad = lp.grad.clone()
            opt.step()
        queue.put((ep, wid, ep_rew, infos_summary(infos)))


def infos_summary(infos):
    keys = ("n_tasks", "latency_sum", "energy_sum", "cost_sum", "n_success")
    return {k: float(sum(i[k] for i in infos)) for k in keys}


class A3CAgent:
    name = "a3c"
    asynchronous = True

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg
        self.net = GaussianPolicyValue(obs_dim, act_dim, cfg["hidden"], -0.5)
        self.net.share_memory()
        self.opt = SharedAdam(self.net.parameters(), lr=cfg["lr"])
        self.rng = rng

    def train(self, env_cfg: dict, base_seed: int, episodes: int, steps_per_ep: int, log=None):
        """Run asynchronous training; returns the list of episode returns ordered by episode index."""
        ctx = mp.get_context("fork")
        counter, lock, queue = ctx.Value("i", 0), ctx.Lock(), ctx.Queue()
        procs = []
        for w in range(int(self.cfg["n_workers"])):
            p = ctx.Process(target=_worker, args=(w, self.net, self.opt, self.cfg, env_cfg,
                                                  base_seed, episodes, steps_per_ep,
                                                  counter, lock, queue, base_seed * 7919 + w))
            p.start()
            procs.append(p)
        returns = [math.nan] * episodes
        for _ in range(episodes):
            ep, wid, ret, summ = queue.get()
            returns[ep] = ret
            if log:
                log(ep, ret, summ)
        for p in procs:
            p.join()
        return returns

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        d = self.net.dist(torch.as_tensor(obs).unsqueeze(0))
        return (d.mean if deterministic else d.sample()).squeeze(0).numpy()

    def state_dict(self):
        return {"net": self.net.state_dict()}

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd["net"])
