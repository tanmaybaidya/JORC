"""Training (Algorithm 1 outer loops) and evaluation (online use of the trained policy, Algorithm 2)."""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch

from .baselines import METHODS, make_agent
from .env import UAVMECEnv, episode_seed
from .metrics import MetricAccumulator
from .utils import seed_everything


def method_config(cfg: dict, method: str) -> dict:
    c = copy.deepcopy(cfg)
    c["caching"]["enabled"] = METHODS[method]["caching"]
    return c


def train(method: str, cfg: dict, seed: int, episodes: int | None = None, steps: int | None = None,
          verbose: bool = False):
    """Train one method with one seed. Returns (agent, history)."""
    episodes = int(episodes or cfg["training"]["episodes"])
    steps = int(steps or cfg["training"]["steps_per_episode"])
    torch.set_num_threads(int(cfg["training"]["torch_threads"]))
    seed_everything(seed)
    mcfg = method_config(cfg, method)
    algo = METHODS[method]["algo"]
    env = UAVMECEnv(mcfg, seed)
    agent = make_agent(algo, env.obs_dim, env.action_dim, mcfg, np.random.default_rng(seed + 1))
    hist = {"episode_return": [], "avg_cost": [], "stcr": [], "wall_s": []}
    t0 = time.time()

    def log(ep, ret, summ):
        hist["episode_return"].append(float(ret))
        hist["avg_cost"].append(summ["cost_sum"] / max(summ["n_tasks"], 1))
        hist["stcr"].append(summ["n_success"] / max(summ["n_tasks"], 1))
        hist["wall_s"].append(time.time() - t0)
        if verbose and (ep % max(1, episodes // 20) == 0 or ep == episodes - 1):
            print(f"  [{method}] ep {ep:5d} return {ret:10.2f} avg_cost {hist['avg_cost'][-1]:.4f} "
                  f"stcr {hist['stcr'][-1]:.3f} ({time.time() - t0:.0f}s)", flush=True)

    if algo == "a3c":
        rets = agent.train(mcfg, seed, episodes, steps, log=log)
        hist["episode_return_by_index"] = rets
        return agent, hist

    upd_every = int(mcfg["sac"]["update_every"]) if algo == "sac" else 1
    n_upd = int(mcfg["sac"]["updates_per_step"]) if algo == "sac" else 1
    for ep in range(episodes):                                     # Alg.1 line 4
        obs = env.reset(episode_seed(seed, ep))                    # line 5
        ret, summ = 0.0, {"n_tasks": 0, "cost_sum": 0.0, "n_success": 0}
        for t in range(steps):                                     # line 6
            a = agent.act(obs, deterministic=False)                # line 7
            obs2, r, _, info = env.step(a)                         # line 8
            agent.observe(obs, a, r, obs2)                         # line 9
            if algo in ("sac", "ddpg") and agent.ready() and t % upd_every == 0:
                for _ in range(n_upd):
                    agent.update()                                 # lines 10-14
            obs = obs2
            ret += r
            for k in summ:
                summ[k] += info[k]
        if algo == "ppo":
            agent.end_episode()
        log(ep, ret, summ)
    return agent, hist


def evaluate(agent, cfg: dict, method: str, seed: int, episodes: int | None = None,
             steps: int | None = None, cache_overrides: dict | None = None) -> dict:
    """Deterministic (mean-action) evaluation on held-out episode seeds with the same task catalogue."""
    episodes = int(episodes or cfg["training"]["eval_episodes"])
    steps = int(steps or cfg["training"]["eval_steps_per_episode"])
    mcfg = method_config(cfg, method)
    if cache_overrides:
        mcfg["caching"].update(cache_overrides)
    env = UAVMECEnv(mcfg, seed)
    acc = MetricAccumulator()
    off = int(cfg["training"]["eval_seed_offset"])
    for ep in range(episodes):
        obs = env.reset(episode_seed(seed + off, ep))
        for _ in range(steps):
            obs, _, _, info = env.step(agent.act(obs, deterministic=True))
            acc.add(info)
        if env.cache_enabled:
            for c in env.caches:
                c.check_invariants()
    out = acc.summary()
    out["deadline_violations"] = out["n_tasks"] * (1.0 - out["stcr"])
    return out


def save_agent(agent, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), path)
