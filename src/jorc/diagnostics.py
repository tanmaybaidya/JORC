"""Training-free reference policies used ONLY for numerical sanity checks and sensitivity analysis.
They are NOT baselines of the paper and are never plotted together with the paper's methods.

all_local      : every cache-miss task executed on the UD (eq.(8)-(9))
all_uav_equal  : every task offloaded to the assigned UAV at P_max, UAV CPU split equally (projection of s=1)
all_bs_equal   : every task forwarded to the BS at P_max, BS CPU split equally
uav_fixed_share: offload to UAV with a fixed requested share s (then projected to satisfy (19f))
"""
from __future__ import annotations

import numpy as np

from .env import UAVMECEnv, episode_seed
from .metrics import MetricAccumulator


def fixed_action(n_users: int, loc: int, power_level: float = 1.0, share: float = 1.0) -> np.ndarray:
    a = np.full((n_users, 5), -1.0)
    a[:, loc] = 1.0
    a[:, 3] = power_level
    a[:, 4] = 2.0 * share - 1.0
    return a.ravel()


POLICIES = {
    "all_local": lambda n: fixed_action(n, 0),
    "all_uav_equal": lambda n: fixed_action(n, 1, share=1.0),
    "all_bs_equal": lambda n: fixed_action(n, 2, share=1.0),
    "uav_share_0.05": lambda n: fixed_action(n, 1, share=0.05),
    "bs_share_0.02": lambda n: fixed_action(n, 2, share=0.02),
}


def run_fixed_policy(cfg: dict, policy: str, seed: int, episodes: int, steps: int) -> dict:
    env = UAVMECEnv(cfg, seed)
    act = POLICIES[policy](env.N)
    acc = MetricAccumulator()
    for ep in range(episodes):
        env.reset(episode_seed(seed + 777, ep))
        for _ in range(steps):
            _, _, _, info = env.step(act)
            acc.add(info)
    return acc.summary()
