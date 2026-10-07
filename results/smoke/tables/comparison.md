# Paper-stated values vs. reproduction (scale: smoke)

Reproduced: mean ± 95% CI over available seeds. '—' = point not run.

| Sweep | Metric | Method | x | Paper | Reproduced |
|---|---|---|---|---|---|
| vs_users | avg_latency_s | JORC | 20 | 1.6 | 2.24 ± 2.7 (n=2) |
| vs_users | avg_latency_s | JORC | 120 | 3.7 | 4.81 ± 2 (n=2) |
| vs_users | avg_latency_s | SAC without caching | 20 | 2.4 | 2.65 ± 1.3 (n=2) |
| vs_users | avg_latency_s | SAC without caching | 120 | 6.4 | 7.39 ± 2.6 (n=2) |
| vs_users | avg_latency_s | DRL with caching range | 20 | [1.7, 2.0] | [2.61, 20.9] (seed range) |
| vs_users | avg_latency_s | DRL with caching range | 120 | [4.0, 5.0] | [4.56, 55.4] (seed range) |
| vs_users | avg_energy_j | JORC | 20 | 2.8 | 73.5 ± 1.1e+02 (n=2) |
| vs_users | avg_energy_j | JORC | 120 | 4.55 | 4.32 ± 1.9 (n=2) |
| vs_users | avg_energy_j | SAC without caching | 20 | 4.1 | 69.3 ± 47 (n=2) |
| vs_users | avg_energy_j | SAC without caching | 120 | 7.2 | 3.51 ± 0.42 (n=2) |
| vs_users | avg_energy_j | DRL with caching range | 20 | [3.0, 3.55] | [84.6, 216] (seed range) |
| vs_users | avg_energy_j | DRL with caching range | 120 | [4.8, 5.5] | [3.55, 11.4] (seed range) |
| vs_users | avg_cost | JORC | 20 | 0.15 | 2.2 ± 3.1 (n=2) |
| vs_users | avg_cost | JORC | 120 | 0.43 | 0.881 ± 0.3 (n=2) |
| vs_users | avg_cost | SAC without caching | 20 | 0.31 | 2.17 ± 1.4 (n=2) |
| vs_users | avg_cost | SAC without caching | 120 | 0.8 | 1.27 ± 0.45 (n=2) |
| vs_users | avg_cost | DRL with caching range | 20 | [0.16, 0.2] | [2.56, 7.39] (seed range) |
| vs_users | avg_cost | DRL with caching range | 120 | [0.5, 0.7] | [0.829, 9.12] (seed range) |
| vs_users | stcr | JORC | 20 | 0.98 | 0.792 ± 0.28 (n=2) |
| vs_users | stcr | JORC | 120 | 0.88 | 0.462 ± 0.095 (n=2) |
| vs_users | stcr | SAC without caching | 20 | 0.92 | 0.753 ± 0.067 (n=2) |
| vs_users | stcr | SAC without caching | 120 | 0.7 | 0.259 ± 0.16 (n=2) |
| vs_users | stcr | DRL with caching range | 120 | [0.78, 0.85] | [0.382, 0.462] (seed range) |
| vs_uavs | avg_latency_s | JORC | 1 | 3.4 | 6.83 ± 1.2 (n=2) |
| vs_uavs | avg_latency_s | JORC | 5 | 1.8 | 3.62 ± 0.17 (n=2) |
| vs_uavs | avg_latency_s | JORC | 8 | 1.7 | 3.32 ± 1.2 (n=2) |
| vs_uavs | avg_energy_j | JORC | 1 | 4.9 | 4.65 ± 0.28 (n=2) |
| vs_uavs | avg_energy_j | JORC | 5 | 3.9 | 9.6 ± 6.2 (n=2) |
| vs_uavs | avg_cost | JORC | 1 | 0.46 | 1.21 ± 0.11 (n=2) |
| vs_uavs | avg_cost | JORC | 6 | 0.17 | 0.839 ± 0.059 (n=2) |
| vs_uavs | avg_cost | SAC without caching | 1 | 0.75 | 1.69 ± 1.9 (n=2) |
| vs_uavs | avg_cost | DRL with caching range | 8 | [0.18, 0.2] | [0.839, 4.48] (seed range) |
| vs_uavs | stcr | JORC | 1 | 0.84 | 0.415 ± 0.043 (n=2) |
| vs_uavs | stcr | JORC | 5 | 0.965 | 0.593 ± 0.02 (n=2) |
| vs_uavs | stcr | JORC | 8 | 0.97 | 0.639 ± 0.11 (n=2) |
| vs_uavs | stcr | SAC without caching | 1 | 0.7 | 0.211 ± 0.11 (n=2) |
| vs_uavs | stcr | SAC without caching | 8 | 0.86 | 0.532 ± 0.032 (n=2) |
| vs_uavs | stcr | DRL with caching range | 8 | [0.93, 0.96] | [0.37, 0.668] (seed range) |
| fig14 | cache_hit_ratio_uav | Hybrid LRU-LFU (JORC) | 2 GB | 0.4 | 0.0539 ± 0.063 |
| fig14 | cache_hit_ratio_uav | Hybrid LRU-LFU (JORC) | 32 GB | 0.9 | 0.0539 ± 0.063 |
| fig14 | cache_hit_ratio_uav | LFU | 2 GB | 0.33 | 0.0539 ± 0.063 |
| fig14 | cache_hit_ratio_uav | LFU | 32 GB | 0.82 | 0.0539 ± 0.063 |
| fig14 | cache_hit_ratio_uav | LRU | 2 GB | 0.3 | 0.0539 ± 0.063 |
| fig14 | cache_hit_ratio_uav | LRU | 32 GB | 0.8 | 0.0539 ± 0.063 |
| fig14 | cache_hit_ratio_uav | Random | 2 GB | 0.18 | 0.0539 ± 0.063 |
| fig14 | cache_hit_ratio_uav | Random | 32 GB | 0.72 | 0.0539 ± 0.063 |
| fig15 | cache_hit_ratio_bs | Hybrid LRU-LFU (JORC) | 10 GB | 0.6 | 0.145 ± 0.063 |
| fig15 | cache_hit_ratio_bs | Hybrid LRU-LFU (JORC) | 160 GB | 0.95 | 0.145 ± 0.063 |
| fig15 | cache_hit_ratio_bs | LFU | 10 GB | 0.53 | 0.145 ± 0.063 |
| fig15 | cache_hit_ratio_bs | LFU | 160 GB | 0.9 | 0.145 ± 0.063 |
| fig15 | cache_hit_ratio_bs | LRU | 10 GB | 0.51 | 0.145 ± 0.063 |
| fig15 | cache_hit_ratio_bs | LRU | 160 GB | 0.89 | 0.145 ± 0.063 |
| fig15 | cache_hit_ratio_bs | Random | 10 GB | 0.38 | 0.145 ± 0.063 |
| fig15 | cache_hit_ratio_bs | Random | 160 GB | 0.8 | 0.145 ± 0.063 |
