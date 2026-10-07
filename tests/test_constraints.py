"""Constraints (3)/(19b)/(19c), (19e), (19f), coverage, and (14) inside the environment."""
import numpy as np

from jorc.env import UAVMECEnv
from jorc.offloading import decode_action, project_shares, server_load
from jorc.utils import load_config


def test_decode_one_hot_and_power_bounds():
    rng = np.random.default_rng(0)
    a = rng.uniform(-3, 3, 5 * 50)                      # out-of-range inputs are clipped
    loc, p, s = decode_action(a, 50, 1e-3, 0.2, 0.01)
    x = np.eye(3)[loc]
    assert np.all(x.sum(1) == 1)                        # (19c) exactly one location
    assert np.all((p > 0) & (p <= 0.2))                 # (19e)
    assert np.all((s >= 0.01) & (s <= 1))


def test_projection_19f():
    rng = np.random.default_rng(1)
    n, m = 40, 3
    loc = rng.integers(0, 3, n)
    uav = rng.integers(0, m, n)
    s = rng.uniform(0.01, 1, n)
    act = rng.random(n) < 0.8
    sp = project_shares(s, loc, uav, m, act)
    assert np.all(server_load(sp, loc, uav, m, act) <= 1 + 1e-9)


def test_env_constraints_random_actions():
    cfg = load_config(overrides=["system.n_users=30", "task.library_size=200"])
    env = UAVMECEnv(cfg, 3)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(60):
        _, r, _, info = env.step(rng.uniform(-1, 1, env.action_dim))
        assert info["max_server_load"] <= 1 + 1e-9
        assert info["n_local"] + info["n_uav"] + info["n_bs"] + info["n_hits"] == 30
        assert r <= 0
        assert np.all(env.geo.in_coverage())
    for c in env.caches:
        c.check_invariants()                            # (14)


def test_out_of_coverage_users_cannot_offload():
    cfg = load_config(overrides=["system.n_users=40", "system.user_placement=uniform_area"])
    env = UAVMECEnv(cfg, 0)
    env.reset()
    a = np.tile([-1, 1, -1, 1, 1], 40).astype(float)  # everyone requests UAV offloading
    out = (~env._covered).sum()
    _, _, _, info = env.step(a)
    assert info["n_local"] >= out - info["n_hits"]
