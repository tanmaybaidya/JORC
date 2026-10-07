# Sensitivity analysis

## Part 3: capacity bound (paper parameters)

| N | M | mean demand (Gcycles) | total capacity (GHz) | bound (s) |
|---|---|---|---|---|
| 20 | 2 | 81 | 82.5 | 0.98 |
| 60 | 2 | 243 | 107.5 | 2.26 |
| 100 | 2 | 405 | 132.5 | 3.06 |
| 120 | 2 | 486 | 145.0 | 3.35 |
| 100 | 1 | 405 | 122.5 | 3.31 |
| 100 | 5 | 405 | 162.5 | 2.49 |
| 100 | 8 | 405 | 192.5 | 2.10 |

## Part 1: training-free reference policies (N=100, M=2, seed 0)

| Variant | Policy | avg_latency_s | avg_energy_j | avg_cost | stcr | cache_hit_ratio_uav | cache_hit_ratio_bs |
|---|---|---|---|---|---|---|---|
| default | all_local | 6.53 | 1.61 | 1.07 | 0.238 | 0 | 0 |
| default | all_uav_equal | 13.7 | 0.358 | 2.2 | 0.21 | 0.205 | 0 |
| default | all_bs_equal | 5.27 | 1.55 | 0.885 | 0.404 | 0 | 0.256 |
| noise_psd_dbm_per_hz: -174 | all_local | 6.53 | 1.61 | 1.07 | 0.238 | 0 | 0 |
| noise_psd_dbm_per_hz: -174 | all_uav_equal | 13.7 | 0.358 | 2.2 | 0.21 | 0.205 | 0 |
| noise_psd_dbm_per_hz: -174 | all_bs_equal | 5.27 | 1.55 | 0.885 | 0.404 | 0 | 0.256 |
| noise_psd_dbm_per_hz: -164 | all_local | 6.53 | 1.61 | 1.07 | 0.238 | 0 | 0 |
| noise_psd_dbm_per_hz: -164 | all_uav_equal | 13.9 | 0.399 | 2.24 | 0.21 | 0.205 | 0 |
| noise_psd_dbm_per_hz: -164 | all_bs_equal | 5.48 | 1.6 | 0.919 | 0.392 | 0 | 0.256 |
| ud_pmax_w: 0.1 | all_local | 6.53 | 1.61 | 1.07 | 0.238 | 0 | 0 |
| ud_pmax_w: 0.1 | all_uav_equal | 13.8 | 0.287 | 2.21 | 0.21 | 0.205 | 0 |
| ud_pmax_w: 0.1 | all_bs_equal | 5.32 | 1.48 | 0.891 | 0.401 | 0 | 0.256 |
| ud_pmax_w: 0.5 | all_local | 6.53 | 1.61 | 1.07 | 0.238 | 0 | 0 |
| ud_pmax_w: 0.5 | all_uav_equal | 13.7 | 0.557 | 2.2 | 0.211 | 0.205 | 0 |
| ud_pmax_w: 0.5 | all_bs_equal | 5.22 | 1.73 | 0.88 | 0.406 | 0 | 0.256 |
| beamwidth_deg: 45 | all_local | 6.53 | 1.61 | 1.07 | 0.238 | 0 | 0 |
| beamwidth_deg: 45 | all_uav_equal | 13.7 | 0.352 | 2.2 | 0.211 | 0.205 | 0 |
| beamwidth_deg: 45 | all_bs_equal | 5.25 | 1.54 | 0.88 | 0.405 | 0 | 0.256 |
| beamwidth_deg: 75 | all_local | 6.53 | 1.61 | 1.07 | 0.238 | 0 | 0 |
| beamwidth_deg: 75 | all_uav_equal | 13.8 | 0.375 | 2.22 | 0.21 | 0.205 | 0 |
| beamwidth_deg: 75 | all_bs_equal | 5.35 | 1.56 | 0.898 | 0.4 | 0 | 0.256 |
| backhaul: fixed, 20.0e6 | all_local | 6.53 | 1.61 | 1.07 | 0.238 | 0 | 0 |
| backhaul: fixed, 20.0e6 | all_uav_equal | 13.7 | 0.358 | 2.2 | 0.21 | 0.205 | 0 |
| backhaul: fixed, 20.0e6 | all_bs_equal | 5.47 | 1.74 | 0.921 | 0.393 | 0 | 0.256 |
| backhaul: fixed, 100.0e6 | all_local | 6.53 | 1.61 | 1.07 | 0.238 | 0 | 0 |
| backhaul: fixed, 100.0e6 | all_uav_equal | 13.7 | 0.358 | 2.2 | 0.21 | 0.205 | 0 |
| backhaul: fixed, 100.0e6 | all_bs_equal | 5.29 | 1.57 | 0.888 | 0.403 | 0 | 0.256 |
| bandwidth_scope: shared | all_local | 6.53 | 1.61 | 1.07 | 0.238 | 0 | 0 |
| bandwidth_scope: shared | all_uav_equal | 14.4 | 0.492 | 2.31 | 0.209 | 0.205 | 0 |
| bandwidth_scope: shared | all_bs_equal | 5.9 | 1.67 | 0.988 | 0.371 | 0 | 0.256 |
| library_size: 1000 | all_local | 6.52 | 1.6 | 1.16 | 0.199 | 0 | 0 |
| library_size: 1000 | all_uav_equal | 1.49 | 1.83 | 0.296 | 0.871 | 0.816 | 0 |
| library_size: 1000 | all_bs_equal | 0.391 | 29.2 | 0.796 | 0.958 | 0 | 0.896 |
| library_size: 20000 | all_local | 6.54 | 1.61 | 1.14 | 0.223 | 0 | 0 |
| library_size: 20000 | all_uav_equal | 8.43 | 0.401 | 1.37 | 0.413 | 0.394 | 0 |
| library_size: 20000 | all_bs_equal | 2.81 | 2.25 | 0.511 | 0.657 | 0 | 0.487 |
| library_size: 2000000 | all_local | 6.63 | 1.64 | 1.13 | 0.216 | 0 | 0 |
| library_size: 2000000 | all_uav_equal | 17.2 | 0.354 | 2.8 | 0.106 | 0.105 | 0 |
| library_size: 2000000 | all_bs_equal | 6.99 | 1.37 | 1.17 | 0.259 | 0 | 0.134 |
| zipf_exponent: 0.5 | all_local | 6.6 | 1.62 | 1.1 | 0.227 | 0 | 0 |
| zipf_exponent: 0.5 | all_uav_equal | 20 | 0.356 | 3.24 | 0.0347 | 0.034 | 0 |
| zipf_exponent: 0.5 | all_bs_equal | 8.21 | 1.3 | 1.36 | 0.175 | 0 | 0.0581 |
| zipf_exponent: 1.0 | all_local | 6.46 | 1.59 | 1.03 | 0.276 | 0 | 0 |
| zipf_exponent: 1.0 | all_uav_equal | 6.37 | 0.425 | 1.02 | 0.502 | 0.474 | 0 |
| zipf_exponent: 1.0 | all_bs_equal | 2.37 | 2.36 | 0.434 | 0.705 | 0 | 0.527 |
| latency_normalizer: constant | all_local | 6.53 | 1.61 | 0.507 | 0.238 | 0 | 0 |
| latency_normalizer: constant | all_uav_equal | 13.7 | 0.358 | 0.989 | 0.21 | 0.205 | 0 |
| latency_normalizer: constant | all_bs_equal | 5.27 | 1.55 | 0.415 | 0.404 | 0 | 0.256 |
| reward_aggregation: mean | all_local | 6.53 | 1.61 | 1.07 | 0.238 | 0 | 0 |
| reward_aggregation: mean | all_uav_equal | 13.7 | 0.358 | 2.2 | 0.21 | 0.205 | 0 |
| reward_aggregation: mean | all_bs_equal | 5.27 | 1.55 | 0.885 | 0.404 | 0 | 0.256 |
| cache_entry_size: din_plus_dout | all_local | 6.53 | 1.61 | 1.07 | 0.238 | 0 | 0 |
| cache_entry_size: din_plus_dout | all_uav_equal | 14 | 0.357 | 2.24 | 0.202 | 0.196 | 0 |
| cache_entry_size: din_plus_dout | all_bs_equal | 5.27 | 1.55 | 0.885 | 0.404 | 0 | 0.256 |
| cache_units_MB: 2.0e6, 10.0e6 | all_local | 6.53 | 1.61 | 1.07 | 0.238 | 0 | 0 |
| cache_units_MB: 2.0e6, 10.0e6 | all_uav_equal | 19.6 | 0.353 | 3.13 | 0.0427 | 0.0422 | 0 |
| cache_units_MB: 2.0e6, 10.0e6 | all_bs_equal | 7.81 | 1.3 | 1.28 | 0.194 | 0 | 0.0757 |
| user_placement: uniform_area | all_local | 6.53 | 1.61 | 1.07 | 0.238 | 0 | 0 |
| user_placement: uniform_area | all_uav_equal | 6.33 | 5.74 | 1.14 | 0.264 | 0.0541 | 0 |
| user_placement: uniform_area | all_bs_equal | 6.3 | 35 | 1.87 | 0.269 | 0 | 0.0721 |
