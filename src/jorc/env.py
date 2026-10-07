"""UAV-aided MEC environment with task-result caching (the MDP of Sec. 5.1.1).

One environment step = one time slot. Per slot (Sec. 6.1, "In each timeslot, a single user generates a single
computational task" - read as: every user generates one task; see REPRODUCTION_ASSUMPTIONS A-15):
  1. user positions are refreshed (quasi-static within the slot);
  2. each UD issues one task request; the request is checked in the assigned UAV's cache and (option)
     in the BS cache (Fig. 2, Alg. 2 lines 5-10). Hits are served at negligible cost (Sec. 4);
  3. the observation S_t (eq.(20)) is emitted; the agent returns A_t = {X, P, F} (eq.(21));
  4. cache-miss tasks are executed per eqs.(4)-(16); the reward is R_t = -C(t) (eq.(22));
  5. results of tasks executed at a UAV / the BS are inserted in that server's cache (Alg. 2 lines 14-20);
  6. delta is adapted (eq.(34)).
"""
from __future__ import annotations

import numpy as np

from .caching import ResultCache
from .channel_model import backhaul_rates, channel_gain, shannon_rate
from .cost_model import LOC_BS, LOC_UAV, LOC_UD, task_cost, task_latency_energy
from .offloading import ACTION_DIM_PER_USER, decode_action, project_shares, server_load
from .system_model import Geometry
from .task_model import TaskLibrary
from .utils import bytes_to_bits, db_to_linear, dbm_to_watt

USER_FEATURES = 10
UAV_FEATURES = 4
GLOBAL_FEATURES = 2


def episode_seed(run_seed: int, episode: int) -> int:
    """Deterministic per-episode seed shared by all training methods (common random numbers)."""
    return int(np.random.SeedSequence([int(run_seed), int(episode)]).generate_state(1)[0])


class UAVMECEnv:
    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        self.seed = int(seed)
        self.N = int(cfg["system"]["n_users"])
        self.M = int(cfg["system"]["n_uavs"])
        self.cache_enabled = bool(cfg["caching"]["enabled"])
        c = cfg["communication"]
        self.h0 = db_to_linear(c["h0_db"])
        self.n0 = dbm_to_watt(c["noise_psd_dbm_per_hz"])          # W/Hz
        self.action_dim = ACTION_DIM_PER_USER * self.N
        self.obs_dim = USER_FEATURES * self.N + UAV_FEATURES * self.M + GLOBAL_FEATURES
        ss = np.random.SeedSequence(self.seed)
        lib_seed, geo_seed, self._episode_ss = ss.spawn(3)
        # the task catalogue (identities and attributes of deterministic tasks) is fixed for the run
        self.library = TaskLibrary(cfg["task"], np.random.default_rng(lib_seed))
        # UAV placement is fixed for the run (only 'random' layout uses the rng)
        self.geo = Geometry(cfg["system"], np.random.default_rng(geo_seed))
        self.r_bh = backhaul_rates(c, self.h0, self.geo.H, self.geo.uav_xy, self.geo.bs_xy, self.n0)
        self.caches: list[ResultCache] = []
        self._episode = 0
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ episode control
    def _make_caches(self):
        ca = self.cfg["caching"]
        min_entry = self.cfg["task"]["dout_bits"][0]
        if ca["entry_size"] == "din_plus_dout":
            min_entry += self.cfg["task"]["din_bits"][0]
        caps = [bytes_to_bits(ca["uav_capacity_bytes"])] * self.M + [bytes_to_bits(ca["bs_capacity_bytes"])]
        self.caches = [ResultCache(cap, ca["policy"], ca, int(cap // min_entry) + 1,
                                   np.random.default_rng(self.rng.integers(2**63)))
                       for cap in caps]

    def reset(self, episode_seed: int | None = None) -> np.ndarray:
        if episode_seed is None:
            episode_seed = int(self._episode_ss.spawn(1)[0].generate_state(1)[0])
        self.rng = np.random.default_rng(episode_seed)
        self._episode += 1
        lo, hi = self.cfg["computation"]["ud_freq_hz"]
        self.f_local = self.rng.uniform(lo, hi, self.N)             # heterogeneous UDs (Sec. 6.1)
        if self.cache_enabled and (not self.caches or self.cfg["caching"]["reset_each_episode"]):
            self._make_caches()
        self.t = 0
        self.geo.resample_users(self.rng)
        return self._new_slot(resample=False)

    # ------------------------------------------------------------------ slot generation
    def _new_slot(self, resample: bool = True) -> np.ndarray:
        if resample and self.cfg["system"]["resample_positions_each_step"]:
            self.geo.resample_users(self.rng)
        self.task = self.library.sample(self.N, self.rng)
        self.d = self.geo.horizontal_distance()
        self.h = channel_gain(self.h0, self.geo.H, self.d)
        self._covered = self.geo.in_coverage()
        self.hit_at = np.full(self.N, -1)                           # -1 miss, m UAV index, M = BS
        if self.cache_enabled:
            look_bs = self.cfg["caching"]["lookup"] == "uav_then_bs"
            covered = self._covered
            for n in range(self.N):
                if not covered[n]:          # cannot reach any MEC server (only in 'uniform_area' mode)
                    continue
                m = self.geo.uav_of_user[n]
                tid = self.task["id"][n]
                if self.caches[m].lookup(tid, self.t):
                    self.hit_at[n] = m
                elif look_bs and self.caches[self.M].lookup(tid, self.t):
                    self.hit_at[n] = self.M
        return self._observe()

    def _observe(self) -> np.ndarray:
        cfg, N, M = self.cfg, self.N, self.M
        t = self.task
        half = self.geo.side / 2.0
        kappa = cfg["computation"]["kappa"]
        e_loc = kappa * self.f_local**2 * t["cycles"] / cfg["task"]["emax_j"]
        miss = self.hit_at < 0
        users = np.stack([
            t["din"] / cfg["task"]["din_bits"][1],
            t["dout"] / cfg["task"]["dout_bits"][1],
            t["cycles"] / (cfg["task"]["din_bits"][1] * cfg["task"]["cycles_per_bit"][1]),
            t["lmax"] / cfg["task"]["lmax_s"][1],
            (self.geo.user_xy[:, 0] - self.geo.bs_xy[0]) / half,
            (self.geo.user_xy[:, 1] - self.geo.bs_xy[1]) / half,
            self.f_local / cfg["computation"]["ud_freq_hz"][1],
            np.minimum(e_loc, 10.0),                                  # local energy status E_n^UD / E^max
            (10 * np.log10(self.h) + 90.0) / 10.0,                    # channel gain h_{n,m} in dB, shifted
            (~miss).astype(float),                                    # cache-hit flag (task handled)
        ], axis=1)
        workload = np.bincount(self.geo.uav_of_user[miss], minlength=M) / N
        uavs = np.stack([(self.geo.uav_xy[:, 0] - self.geo.bs_xy[0]) / half,
                         (self.geo.uav_xy[:, 1] - self.geo.bs_xy[1]) / half,
                         np.full(M, cfg["computation"]["uav_freq_hz"] / cfg["computation"]["bs_freq_hz"]),
                         workload], axis=1)
        glob = np.array([1.0, miss.mean()])                           # BS capacity (normalised), congestion
        obs = np.concatenate([users.ravel(), uavs.ravel(), glob]).astype(np.float32)
        assert obs.shape == (self.obs_dim,) and np.all(np.isfinite(obs))
        return obs

    # ------------------------------------------------------------------ bandwidth
    def _bandwidth(self, loc: np.ndarray, active: np.ndarray) -> np.ndarray:
        c = self.cfg["communication"]
        B = float(c["bandwidth_hz"])
        uav = self.geo.uav_of_user
        if c["bandwidth_split"] == "equal_associated":
            group = np.bincount(uav, minlength=self.M)[uav] if c["bandwidth_scope"] == "per_uav" \
                else np.full(self.N, self.N)
        elif c["bandwidth_split"] == "equal_offloaders":
            off = active & (loc != LOC_UD)
            cnt = np.bincount(uav[off], minlength=self.M)
            group = np.maximum(cnt[uav] if c["bandwidth_scope"] == "per_uav" else np.full(self.N, off.sum()), 1)
        else:
            raise ValueError(c["bandwidth_split"])
        return B / group

    # ------------------------------------------------------------------ step
    def step(self, action: np.ndarray):
        cfg, N, M = self.cfg, self.N, self.M
        comm, comp, cost_cfg = cfg["communication"], cfg["computation"], cfg["cost"]
        loc, power, share_raw = decode_action(action, N, comm["ud_pmin_w"], comm["ud_pmax_w"], comp["s_min"])
        active = self.hit_at < 0                                       # cache misses need execution
        # Sec. 3.2: offloading to UAV m requires d_{n,m} <= H tan(theta); otherwise execute locally
        loc = np.where(self._covered, loc, LOC_UD)
        uav = self.geo.uav_of_user
        share = project_shares(share_raw, loc, uav, M, active, comp["capacity_projection"])
        bw = self._bandwidth(loc, active)
        rate_up = shannon_rate(bw, power, self.h, self.n0)
        t = self.task
        L = np.full(N, float(cost_cfg["hit_latency_s"]))
        E = np.full(N, float(cost_cfg["hit_energy_j"]))
        a = active
        if a.any():
            L[a], E[a] = task_latency_energy(
                loc[a], din=t["din"][a], cycles=t["cycles"][a], f_local=self.f_local[a],
                kappa=comp["kappa"], p_ud=power[a], rate_up=rate_up[a], rate_bh=self.r_bh[uav[a]],
                p_uav=comm["uav_tx_power_w"], share=share[a], f_uav=comp["uav_freq_hz"], f_bs=comp["bs_freq_hz"])
        l_norm = t["lmax"] if cost_cfg["latency_normalizer"] == "deadline" \
            else np.full(N, float(cost_cfg["latency_normalizer_value_s"]))
        C = task_cost(L, E, l_norm=l_norm, e_norm=cfg["task"]["emax_j"],
                      gamma_l=cost_cfg["gamma_latency"], gamma_e=cost_cfg["gamma_energy"])
        success = L <= t["lmax"] + 1e-12
        C_total = C.sum() + cost_cfg["deadline_penalty"] * (~success).sum()
        reward = -(C_total if cost_cfg["reward_aggregation"] == "sum" else C_total / N)

        # ---- cache insertion (results of tasks executed at UAV or BS) ----
        if self.cache_enabled:
            size = t["dout"] + (t["din"] if cfg["caching"]["entry_size"] == "din_plus_dout" else 0.0)
            for n in np.flatnonzero(a & (loc != LOC_UD)):
                server = uav[n] if loc[n] == LOC_UAV else M
                self.caches[server].insert(t["id"][n], size[n], self.t)
            for c in self.caches:
                c.end_slot()

        loads = server_load(share, loc, uav, M, active)
        info = {
            "n_tasks": N, "latency_sum": float(L.sum()), "energy_sum": float(E.sum()),
            "cost_sum": float(C.sum()), "n_success": int(success.sum()),
            "n_offloaded": int((a & (loc != LOC_UD)).sum()), "n_hits": int((~a).sum()),
            "n_local": int((a & (loc == LOC_UD)).sum()), "n_uav": int((a & (loc == LOC_UAV)).sum()),
            "n_bs": int((a & (loc == LOC_BS)).sum()),
            "n_hits_uav": int(((self.hit_at >= 0) & (self.hit_at < M)).sum()),
            "n_hits_bs": int((self.hit_at == M).sum()),
            "n_lookups_uav": int(self._covered.sum()) if self.cache_enabled else 0,
            # BS is queried for every covered request that missed at the UAV (hit_at is -1 or M)
            "n_lookups_bs": int(((self.hit_at != uav) & self._covered).sum()) if self.cache_enabled
            and cfg["caching"]["lookup"] == "uav_then_bs" else 0,
            "max_server_load": float(loads.max()) if loads.size else 0.0,
            "min_power": float(power[a].min()) if a.any() else comm["ud_pmax_w"],
            "max_power": float(power[a].max()) if a.any() else 0.0,
            "rate_up_mean": float(rate_up[a & (loc != LOC_UD)].mean()) if (a & (loc != LOC_UD)).any() else 0.0,
            "reward": float(reward),
            "n_out_of_coverage": int((~self._covered).sum()),
        }
        self.t += 1
        obs = self._new_slot()
        return obs, float(reward), False, info
