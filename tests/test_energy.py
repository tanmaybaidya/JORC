"""Energy equations (5), (7), (9), (11), (13), (16) and cost (17)."""
import numpy as np

from jorc.cost_model import LOC_BS, LOC_UAV, LOC_UD, task_cost, task_latency_energy


def _call(loc):
    one = lambda v: np.array([v], float)
    return task_latency_energy(np.array([loc]), din=one(4e6), cycles=one(2.4e9), f_local=one(0.6e9),
                               kappa=1e-27, p_ud=one(0.2), rate_up=one(8e6), rate_bh=one(100e6), p_uav=1.0,
                               share=one(0.1), f_uav=10e9, f_bs=50e9)


def test_local_energy_eq9():
    _, E = _call(LOC_UD)
    assert np.isclose(E[0], 1e-27 * (0.6e9) ** 2 * 2.4e9)       # 0.864 J


def test_uav_energy_eq5_eq11():
    _, E = _call(LOC_UAV)
    assert np.isclose(E[0], 0.2 * 4e6 / 8e6 + 1e-27 * (1e9) ** 2 * 2.4e9)


def test_bs_energy_eq7_eq13_eq16():
    _, E = _call(LOC_BS)
    assert np.isclose(E[0], 0.2 * 4e6 / 8e6 + 1.0 * 4e6 / 100e6 + 1e-27 * (5e9) ** 2 * 2.4e9)


def test_cost_eq17():
    c = task_cost(np.array([2.0]), np.array([5.0]), l_norm=np.array([4.0]), e_norm=20.0, gamma_l=0.5, gamma_e=0.5)
    assert np.isclose(c[0], 0.5 * 0.5 + 0.5 * 0.25)
