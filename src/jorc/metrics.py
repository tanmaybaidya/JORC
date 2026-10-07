"""Performance metrics, Sec. 6.3. Accumulated from per-step `info` dicts produced by UAVMECEnv.step.

(a) average task completion latency  = sum L_n / #tasks            (cache hits count with hit latency, 0 s)
(b) average energy consumption       = sum E_n / #tasks            (J per task)
(c) average cost                     = sum C_n / #tasks            (eq.(17) per task)
(d) STCR                             = #tasks with L_n <= L_n^max / #tasks   (hits count as successes)
(e) offloading ratio                 = #cache-miss tasks executed at UAV or BS / #tasks
(f) cache hit ratio at UAVs          = UAV-cache hits / UAV-cache lookups
    cache hit ratio at BS            = BS-cache hits / BS-cache lookups
How the authors average (per task vs per user vs per slot) is not stated; per-task averaging is used ([E]).
"""
from __future__ import annotations

from collections import defaultdict

SUM_KEYS = ("n_tasks", "latency_sum", "energy_sum", "cost_sum", "n_success", "n_offloaded", "n_hits",
            "n_local", "n_uav", "n_bs", "n_hits_uav", "n_hits_bs", "n_lookups_uav", "n_lookups_bs")


class MetricAccumulator:
    def __init__(self):
        self.s = defaultdict(float)
        self.max_load = 0.0
        self.min_power = float("inf")

    def add(self, info: dict) -> None:
        for k in SUM_KEYS:
            self.s[k] += info[k]
        self.max_load = max(self.max_load, info["max_server_load"])
        self.min_power = min(self.min_power, info["min_power"])

    def summary(self) -> dict:
        s, n = self.s, max(self.s["n_tasks"], 1.0)
        div = lambda a, b: a / b if b > 0 else float("nan")
        return {
            "avg_latency_s": s["latency_sum"] / n,
            "avg_energy_j": s["energy_sum"] / n,
            "avg_cost": s["cost_sum"] / n,
            "stcr": s["n_success"] / n,
            "offloading_ratio": s["n_offloaded"] / n,
            "local_ratio": s["n_local"] / n,
            "uav_exec_ratio": s["n_uav"] / n,
            "bs_exec_ratio": s["n_bs"] / n,
            "overall_hit_ratio": s["n_hits"] / n,
            "cache_hit_ratio_uav": div(s["n_hits_uav"], s["n_lookups_uav"]),
            "cache_hit_ratio_bs": div(s["n_hits_bs"], s["n_lookups_bs"]),
            "max_server_load": self.max_load,
            "min_power_w": self.min_power,
            "n_tasks": s["n_tasks"],
        }
