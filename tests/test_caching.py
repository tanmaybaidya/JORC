"""Hybrid LFU-LRU score (33), adaptation (34), replacement policies, capacity (14)."""
import numpy as np

from jorc.caching import ResultCache
from jorc.utils import load_config


def _cache(policy, cap=3.0, **kw):
    ca = load_config()["caching"]
    ca.update(kw)
    return ResultCache(cap, policy, ca, 10, np.random.default_rng(0))


def test_hit_updates_freq_and_time():
    c = _cache("hybrid")
    c.insert(7, 1.0, t=0)
    assert c.lookup(7, t=5) and not c.lookup(8, t=5)
    j = c.index[7]
    assert c.freq[j] == 2 and c.last[j] == 5           # Alg.2: F=1 at insert, +1 per hit


def test_hybrid_score_eq33():
    c = _cache("hybrid", adaptive_delta=False, delta0=0.3)
    c.insert(1, 1.0, t=0)
    c.lookup(1, t=2)                                     # F=2, L=2
    assert np.isclose(c.scores(t=6)[0], 0.3 * 2 + 0.7 / 4)


def test_lru_and_lfu_evict_correctly():
    lru = _cache("lru")
    for k in range(3):
        lru.insert(k, 1.0, t=k)
    lru.lookup(0, t=5)
    lru.insert(9, 1.0, t=6)
    assert not lru.contains(1) and lru.contains(0)
    lfu = _cache("lfu")
    for k in range(3):
        lfu.insert(k, 1.0, t=k)
    lfu.lookup(2, t=4); lfu.lookup(0, t=4)
    lfu.insert(9, 1.0, t=5)
    assert not lfu.contains(1)


def test_capacity_never_exceeded_with_variable_sizes():
    c = _cache("random", cap=10.0)
    rng = np.random.default_rng(0)
    for t in range(200):
        c.insert(int(rng.integers(0, 50)), float(rng.uniform(0.5, 4)), t)
        c.check_invariants()


def test_delta_adaptation_eq34():
    c = _cache("hybrid", delta0=0.5, eta=0.05, r_th=0.5, window_slots=50)
    for _ in range(4):
        c.lookup(1, 0)                                   # one slot, 4 identical requests -> r = 0.75
    c.end_slot()
    assert np.isclose(c.r_t, 0.75) and np.isclose(c.delta, 0.5 + 0.05 * 0.25)
    for t in range(200):                                 # all-distinct requests -> r = 0 -> delta -> min
        c.lookup(1000 + t, t)
        c.end_slot()
    assert np.isclose(c.delta, c.d_min)
