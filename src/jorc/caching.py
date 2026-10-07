"""Task-result caching, Sec. 3.6 and 5.2, Algorithm 2.

Per cached entry: size (bits), access frequency F(T_n) and last access time L(T_n) (in time slots).
Hybrid score, eq.(33):  H(T) = delta F(T) + (1 - delta) / (t - L(T))
Adaptive weight, eq.(34): delta <- clip(delta + eta (r_t - r_th), delta_min, delta_max)
Eviction: lowest score first, repeated until the new entry fits (entries have different sizes).
Alg.2 lines 9-10 (hit: F += 1, L = t) and line 20 (insert: F = 1, L = t) are followed literally.
"""
from __future__ import annotations

from collections import Counter, deque

import numpy as np

POLICIES = ("hybrid", "lfu", "lru", "random")


class ResultCache:
    def __init__(self, capacity_bits: float, policy: str, cfg_cache: dict, max_entries: int,
                 rng: np.random.Generator):
        if policy not in POLICIES:
            raise ValueError(policy)
        self.capacity = float(capacity_bits)
        self.policy = policy
        self.rng = rng
        self.floor = float(cfg_cache["recency_floor_slots"])
        self.delta = float(cfg_cache["delta0"])
        self.eta = float(cfg_cache["eta"])
        self.r_th = float(cfg_cache["r_th"])
        self.d_min, self.d_max = float(cfg_cache["delta_min"]), float(cfg_cache["delta_max"])
        self.adaptive = bool(cfg_cache["adaptive_delta"])
        self.window = int(cfg_cache["window_slots"])
        n = int(max_entries) + 1
        self.size = np.zeros(n)
        self.freq = np.zeros(n)
        self.last = np.zeros(n)
        self.valid = np.zeros(n, bool)
        self.slot_id = np.full(n, -1, dtype=np.int64)
        self.index: dict[int, int] = {}
        self.free = list(range(n - 1, -1, -1))
        self.used = 0.0
        self._win: deque[list[int]] = deque()
        self._win_count: Counter = Counter()
        self._win_total = 0
        self._slot_req: list[int] = []
        self.r_t = 0.0
        self.stats = {"lookups": 0, "hits": 0, "inserts": 0, "evictions": 0}

    # ------------------------------------------------------------------ queries
    def lookup(self, task_id: int, t: float) -> bool:
        """Cache check of Alg.2 lines 6-10. Records the request for the repetition-rate window."""
        self.stats["lookups"] += 1
        self._slot_req.append(int(task_id))
        j = self.index.get(int(task_id))
        if j is None:
            return False
        self.freq[j] += 1.0
        self.last[j] = t
        self.stats["hits"] += 1
        return True

    def contains(self, task_id: int) -> bool:
        return int(task_id) in self.index

    # ------------------------------------------------------------------ scores
    def scores(self, t: float) -> np.ndarray:
        v = self.valid
        if self.policy == "hybrid":
            age = np.maximum(t - self.last[v], self.floor)
            return self.delta * self.freq[v] + (1.0 - self.delta) / age      # eq.(33)
        if self.policy == "lfu":
            return self.freq[v] + 1e-9 * self.last[v] / max(t, 1.0)          # LFU, ties -> LRU
        if self.policy == "lru":
            return self.last[v].copy()
        return self.rng.random(int(v.sum()))                                   # random eviction

    # ------------------------------------------------------------------ insertion
    def insert(self, task_id: int, size_bits: float, t: float) -> bool:
        """Alg.2 lines 14-20. Returns True if the entry is cached after the call."""
        task_id = int(task_id)
        if task_id in self.index:              # already stored (e.g. duplicate in the same slot)
            self.last[self.index[task_id]] = t
            return True
        if size_bits > self.capacity:
            return False
        while self.used + size_bits > self.capacity + 1e-6:
            slots = np.flatnonzero(self.valid)
            victim = slots[int(np.argmin(self.scores(t)))]
            self._remove(victim)
            self.stats["evictions"] += 1
        j = self.free.pop()
        self.valid[j], self.size[j], self.freq[j], self.last[j] = True, size_bits, 1.0, t
        self.index[task_id] = j
        self.slot_id[j] = task_id
        self.used += size_bits
        self.stats["inserts"] += 1
        return True

    def _remove(self, j: int) -> None:
        del self.index[int(self.slot_id[j])]
        self.slot_id[j] = -1
        self.valid[j] = False
        self.used -= self.size[j]
        self.free.append(int(j))

    # ------------------------------------------------------------------ eq.(34)
    def end_slot(self) -> None:
        """Close a time slot: update the W-slot window, r_t and delta (Alg.2 line 21)."""
        self._win.append(self._slot_req)
        self._win_count.update(self._slot_req)
        self._win_total += len(self._slot_req)
        self._slot_req = []
        while len(self._win) > self.window:
            old = self._win.popleft()
            self._win_count.subtract(old)
            self._win_total -= len(old)
            for k in old:
                if self._win_count[k] <= 0:
                    del self._win_count[k]
        # repetition rate: share of requests in the window that repeat an id already seen in the window
        self.r_t = 0.0 if self._win_total == 0 else 1.0 - len(self._win_count) / self._win_total
        if self.adaptive and self.policy == "hybrid":
            self.delta = float(np.clip(self.delta + self.eta * (self.r_t - self.r_th), self.d_min, self.d_max))

    def check_invariants(self) -> None:
        assert self.used <= self.capacity + 1e-6, "eq.(14) storage constraint violated"
        assert abs(self.size[self.valid].sum() - self.used) < 1e-3
        assert len(self.index) == int(self.valid.sum())
