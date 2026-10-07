"""Latency equations (4), (6), (8), (10), (12), (15)."""
import numpy as np

from jorc.cost_model import LOC_BS, LOC_UAV, LOC_UD, task_latency_energy


def _call(loc):
    one = lambda v: np.array([v], float)
    return task_latency_energy(np.array([loc]), din=one(4e6), cycles=one(4e6 * 600), f_local=one(0.6e9),
                               kappa=1e-27, p_ud=one(0.2), rate_up=one(8e6), rate_bh=one(100e6), p_uav=1.0,
                               share=one(0.1), f_uav=10e9, f_bs=50e9)


def test_local_latency_eq8():
    L, _ = _call(LOC_UD)
    assert np.isclose(L[0], 2.4e9 / 0.6e9)


def test_uav_latency_eq4_eq10_eq15():
    L, _ = _call(LOC_UAV)
    assert np.isclose(L[0], 4e6 / 8e6 + 2.4e9 / (0.1 * 10e9))


def test_bs_latency_includes_backhaul_eq6_eq15():
    L, _ = _call(LOC_BS)
    assert np.isclose(L[0], 4e6 / 8e6 + 4e6 / 100e6 + 2.4e9 / (0.1 * 50e9))
