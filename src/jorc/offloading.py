"""Mapping of the continuous SAC action to (X, P, F) - Sec. 5.1.1, eq.(21), (21a), (21b).

Action layout (paper Sec. 6.1, "three continuous offloading-preference scores per user ... plus one power
allocation value and one CPU frequency allocation value per user"): for user n the 5 entries are
    [z_UD, z_UAV, z_BS, a_P, a_F]   each in [-1, 1]  (tanh-squashed actor output)
The affine maps of a_P -> P_n and a_F -> s_n are not given in the paper ([E]).
"""
from __future__ import annotations

import numpy as np

from .cost_model import LOC_BS, LOC_UAV

ACTION_DIM_PER_USER = 5


def decode_action(action: np.ndarray, n_users: int, p_min: float, p_max: float, s_min: float):
    """Return (loc, power_w, share_raw). loc is the argmax of the preference scores (21a, 21b)."""
    a = np.clip(np.asarray(action, float).reshape(n_users, ACTION_DIM_PER_USER), -1.0, 1.0)
    loc = np.argmax(a[:, :3], axis=1)                 # 0 UD, 1 UAV, 2 BS ; mutual exclusivity (19c)
    power = p_min + 0.5 * (a[:, 3] + 1.0) * (p_max - p_min)   # (19e): 0 < P <= P_max
    share = np.clip(0.5 * (a[:, 4] + 1.0), s_min, 1.0)       # s in (0, 1]
    return loc, power, share


def project_shares(share: np.ndarray, loc: np.ndarray, uav_of_user: np.ndarray, n_uavs: int,
                   active: np.ndarray, mode: str = "proportional") -> np.ndarray:
    """Enforce (19f): sum_n s_{i,n} <= 1 at every server i (UAVs and BS) among active tasks.

    'proportional': if the shares requested at a server sum to S > 1, every share there is divided by S.
    Shares of tasks that are not executed at that server are irrelevant and left unchanged.
    """
    if mode != "proportional":
        raise ValueError(mode)
    s = share.copy()
    servers = np.where(loc == LOC_UAV, uav_of_user, np.where(loc == LOC_BS, n_uavs, -1))
    servers = np.where(active, servers, -1)
    for i in range(n_uavs + 1):
        idx = servers == i
        tot = s[idx].sum()
        if tot > 1.0:
            s[idx] /= tot
    return s


def server_load(share: np.ndarray, loc: np.ndarray, uav_of_user: np.ndarray, n_uavs: int,
                active: np.ndarray) -> np.ndarray:
    """sum_n s_{i,n} for i in UAVs + BS (used to verify (19f))."""
    servers = np.where(loc == LOC_UAV, uav_of_user, np.where(loc == LOC_BS, n_uavs, -1))
    servers = np.where(active, servers, -1)
    return np.array([share[servers == i].sum() for i in range(n_uavs + 1)])
