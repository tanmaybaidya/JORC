"""SAC implementation checks: target entropy, tanh log-prob, learning on a toy problem, env determinism."""
import math

import numpy as np
import torch

from jorc.agents.sac import GaussianActor, SACAgent
from jorc.env import UAVMECEnv
from jorc.utils import load_config


def test_target_entropy_is_minus_5N():
    cfg = load_config(overrides=["system.n_users=7"])
    env = UAVMECEnv(cfg, 0)
    ag = SACAgent(env.obs_dim, env.action_dim, cfg["sac"] | {"buffer_size": 100}, np.random.default_rng(0))
    assert ag.target_entropy == -5 * 7
    assert math.isclose(ag.alpha.item(), 0.2, rel_tol=1e-6)


def test_tanh_logprob_matches_change_of_variables():
    torch.manual_seed(0)
    act = GaussianActor(3, 1, [8])
    o = torch.zeros(1000, 3)
    a, logp = act(o)
    mu, log_std = act.body(o).chunk(2, -1)
    u = torch.atanh(a.clamp(-1 + 1e-6, 1 - 1e-6))
    base = torch.distributions.Normal(mu, log_std.clamp(-20, 2).exp()).log_prob(u).sum(-1)
    ref = base - torch.log(1 - a.pow(2) + 1e-12).sum(-1)
    assert torch.allclose(logp, ref, atol=1e-2)


def test_sac_learns_toy_bandit():
    """Reward -(a-0.5)^2 on a constant state: the deterministic policy must approach 0.5."""
    torch.manual_seed(0)
    cfg = load_config()["sac"] | {"hidden": [32, 32], "lr": 3e-3, "buffer_size": 5000, "learning_starts": 64,
                                  "batch_size": 64}
    ag = SACAgent(2, 1, cfg, np.random.default_rng(0))
    o = np.ones(2, np.float32)
    for _ in range(1500):
        a = ag.act(o)
        ag.observe(o, a, -float((a[0] - 0.5) ** 2), o)
        if ag.ready():
            ag.update()
    assert abs(ag.act(o, deterministic=True)[0] - 0.5) < 0.15


def test_env_is_deterministic_for_a_seed():
    cfg = load_config(overrides=["system.n_users=10"])
    acts = np.random.default_rng(5).uniform(-1, 1, (20, 50))
    rets = []
    for _ in range(2):
        env = UAVMECEnv(cfg, 11)
        env.reset(123)
        rets.append([env.step(a)[1] for a in acts])
    assert rets[0] == rets[1]
