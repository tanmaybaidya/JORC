"""Methods compared in Sec. 6.2, using the paper's labels.

JORC                 : SAC (Alg. 1) + hybrid LFU-LRU task-result caching (Alg. 2)
PPO with caching     : PPO   + the same caching module
DDPG with caching    : DDPG  + the same caching module
A3C with caching     : A3C   + the same caching module
SAC without caching  : SAC, caching module disabled (no lookup, no insertion)
"With caching" for the DRL baselines is read as "with JORC's hybrid LFU-LRU caching" ([B]: Sec. 7.1 says
PPO/DDPG/A3C are "equipped with the same caching mechanism").
The cache-replacement baselines (LFU, LRU, Random; Figs. 14-15) are values of caching.policy.
"""
from __future__ import annotations

METHODS = {
    "JORC": {"algo": "sac", "caching": True},
    "PPO with caching": {"algo": "ppo", "caching": True},
    "DDPG with caching": {"algo": "ddpg", "caching": True},
    "A3C with caching": {"algo": "a3c", "caching": True},
    "SAC without caching": {"algo": "sac", "caching": False},
}

SLUG = {"JORC": "jorc", "PPO with caching": "ppo_cache", "DDPG with caching": "ddpg_cache",
        "A3C with caching": "a3c_cache", "SAC without caching": "sac_nocache"}


def make_agent(algo: str, obs_dim: int, act_dim: int, cfg: dict, rng):
    if algo == "sac":
        from .agents.sac import SACAgent
        return SACAgent(obs_dim, act_dim, cfg["sac"], rng)
    if algo == "ddpg":
        from .agents.ddpg import DDPGAgent
        return DDPGAgent(obs_dim, act_dim, cfg["ddpg"], rng)
    if algo == "ppo":
        from .agents.ppo import PPOAgent
        return PPOAgent(obs_dim, act_dim, cfg["ppo"], rng)
    if algo == "a3c":
        from .agents.a3c import A3CAgent
        return A3CAgent(obs_dim, act_dim, cfg["a3c"], rng)
    raise ValueError(algo)
