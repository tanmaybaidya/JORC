"""Offloading, computation and cost model (Sec. 3.4, 3.5, 4). All inputs are numpy arrays over tasks.

Execution location codes: LOC_UD = 0 (local), LOC_UAV = 1 (assigned UAV), LOC_BS = 2 (BS via assigned UAV).
"""
from __future__ import annotations

import numpy as np

LOC_UD, LOC_UAV, LOC_BS = 0, 1, 2


# ---- Sec. 3.4 transmission ----------------------------------------------------------
def uplink_latency(din_bits, rate_bps):
    """eq.(4) / eq.(6): D_in / R."""
    return np.asarray(din_bits, float) / np.asarray(rate_bps, float)


def uplink_energy(power_w, din_bits, rate_bps):
    """eq.(5) / eq.(7): P * D_in / R."""
    return np.asarray(power_w, float) * np.asarray(din_bits, float) / np.asarray(rate_bps, float)


# ---- Sec. 3.5 computation -----------------------------------------------------------
def local_latency(cycles, f_hz):
    """eq.(8): W / f."""
    return np.asarray(cycles, float) / np.asarray(f_hz, float)


def local_energy(kappa, f_hz, cycles):
    """eq.(9): k f^2 W."""
    f = np.asarray(f_hz, float)
    return kappa * f**2 * np.asarray(cycles, float)


def edge_latency(cycles, share, capacity_hz):
    """eq.(10) (UAV) / eq.(12) (BS): W / (s F)."""
    return np.asarray(cycles, float) / (np.asarray(share, float) * capacity_hz)


def edge_energy(kappa, share, capacity_hz, cycles):
    """eq.(11) (UAV) / eq.(13) (BS): k (s F)^2 W."""
    f = np.asarray(share, float) * capacity_hz
    return kappa * f**2 * np.asarray(cycles, float)


# ---- Sec. 4 totals ------------------------------------------------------------------
def task_latency_energy(loc, *, din, cycles, f_local, kappa, p_ud, rate_up, rate_bh, p_uav,
                        share, f_uav, f_bs):
    """eq.(15) and eq.(16) for a vector of (cache-miss) tasks.

    loc     : int array of LOC_* codes (one-hot x in the paper)
    rate_up : R_{n,m} of each task's UD to its assigned UAV
    rate_bh : R_{m,b} of each task's assigned UAV
    share   : s_{m,n} or s_{b,n} (already projected to satisfy (19f)); ignored for local tasks
    """
    loc = np.asarray(loc)
    L = np.zeros(loc.shape, float)
    E = np.zeros(loc.shape, float)

    ud = loc == LOC_UD
    L[ud] = local_latency(cycles[ud], f_local[ud])
    E[ud] = local_energy(kappa, f_local[ud], cycles[ud])

    off = ~ud
    L_up = uplink_latency(din[off], rate_up[off])
    E_up = uplink_energy(p_ud[off], din[off], rate_up[off])
    is_uav = loc[off] == LOC_UAV
    cap = np.where(is_uav, f_uav, f_bs)
    L_cmp = edge_latency(cycles[off], share[off], cap)
    E_cmp = edge_energy(kappa, share[off], cap, cycles[off])
    L_bh = np.where(is_uav, 0.0, uplink_latency(din[off], rate_bh[off]))
    E_bh = np.where(is_uav, 0.0, uplink_energy(p_uav, din[off], rate_bh[off]))
    L[off] = L_up + L_bh + L_cmp
    E[off] = E_up + E_bh + E_cmp
    return L, E


def task_cost(L, E, *, l_norm, e_norm, gamma_l, gamma_e):
    """eq.(17): C_n = gamma_L L/L^max + gamma_E E/E^max."""
    return gamma_l * np.asarray(L) / np.asarray(l_norm) + gamma_e * np.asarray(E) / e_norm
