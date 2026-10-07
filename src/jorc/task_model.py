"""Task model, Sec. 3.3 / 6.1.

A task is (T_n, L_n^max) with T_n = (D_in, W, D_out). Task-result caching (Sec. 3.6) requires that the
*same* task with *identical input* can recur, which the paper does not model explicitly. We therefore use a
finite library of K deterministic tasks; each library entry has fixed (D_in, cycles/bit, D_out[, L^max])
drawn once from the paper's uniform ranges, and requests draw a task id from a Zipf popularity law.
K and the Zipf exponent are reconstruction parameters ([E]/[D], see REPRODUCTION_ASSUMPTIONS.md).
"""
from __future__ import annotations

import numpy as np


class TaskLibrary:
    def __init__(self, cfg_task: dict, rng: np.random.Generator):
        k = int(cfg_task["library_size"])
        u = lambda lo_hi: rng.uniform(lo_hi[0], lo_hi[1], size=k)
        self.size = k
        self.din = u(cfg_task["din_bits"])                       # bits
        self.cycles_per_bit = u(cfg_task["cycles_per_bit"])      # cycles/bit
        self.cycles = self.din * self.cycles_per_bit             # W_n in CPU cycles
        self.dout = u(cfg_task["dout_bits"])                     # bits
        self.lmax = u(cfg_task["lmax_s"])                        # s
        self.lmax_range = tuple(cfg_task["lmax_s"])
        self.lmax_per_request = bool(cfg_task["lmax_per_request"])
        ranks = np.arange(1, k + 1, dtype=float)
        p = ranks ** (-float(cfg_task["zipf_exponent"]))
        p /= p.sum()
        # popularity rank is decoupled from task attributes by a random permutation
        self._perm = rng.permutation(k)
        self._cdf = np.cumsum(p)
        self._cdf[-1] = 1.0
        self.popularity = np.empty(k)
        self.popularity[self._perm] = p

    def sample(self, n: int, rng: np.random.Generator) -> dict:
        ranks = np.searchsorted(self._cdf, rng.random(n), side="right")
        ids = self._perm[np.minimum(ranks, self.size - 1)]
        lmax = rng.uniform(*self.lmax_range, size=n) if self.lmax_per_request else self.lmax[ids]
        return {"id": ids, "din": self.din[ids], "cycles": self.cycles[ids],
                "dout": self.dout[ids], "lmax": lmax}
