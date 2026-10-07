# Parameter provenance (generated from configs/paper_default.yaml)

| Parameter | Value | Provenance | Note |
|---|---|---|---|
| system.n_users | `100` | A | Table 3, N = 100 |
| system.n_uavs | `2` | A | Table 3, M = 2 |
| system.uav_altitude_m | `100.0` | A | Table 3, H = 100 m |
| system.area_side_m | `2000.0` | A | Sec. 6.1, 2 km x 2 km |
| system.bs_position_m | `[0.0, 0.0]` | A | Sec. 3.1 / 6.1, q_b = (0, 0) |
| system.area_centered_on_bs | `true` | E | paper does not say whether (0,0) is a corner or the centre |
| system.uav_layout | `ring` | E | UAV positions not given; 'ring' | 'random' |
| system.uav_ring_radius_m | `500.0` | E |  |
| system.beamwidth_deg | `60.0` | C/E | theta used but no value given; coverage radius = H tan(theta) |
| system.user_placement | `coverage_disk` | E | 'coverage_disk' (user uniform in its UAV's disk) | 'uniform_area' |
| system.user_association | `round_robin` | E | 'round_robin' | 'random' |
| system.resample_positions_each_step | `true` | A | Sec. 6.1 "user positions are refreshed at every time step" |
| communication.bandwidth_hz | `20.0e6` | A | Table 3, B = 20 MHz |
| communication.bandwidth_scope | `per_uav` | E | 'per_uav' (each UAV owns B) | 'shared' (B split over all users) |
| communication.bandwidth_split | `equal_associated` | E | B_n = B / (#users associated with the UAV) |
| communication.h0_db | `-50.0` | A | Table 3, h0 = -50 dB at 1 m |
| communication.noise_psd_dbm_per_hz | `-174.0` | D | thermal noise PSD (3GPP; e.g. arXiv:2206.04302, 1912.07112) |
| communication.ud_pmax_w | `0.2` | D | 23 dBm UE class (3GPP TR 36.777 Table A.1-1 via arXiv:2206.04302) |
| communication.ud_pmin_w | `1.0e-3` | E | lower clip so that P > 0 strictly (constraint 19e) |
| communication.uav_tx_power_w | `1.0` | D | UAV transmit power 1 W (arXiv:2501.15164 Table III) |
| communication.backhaul_mode | `shannon_los` | E | 'shannon_los' (eq.(1)-(2) on UAV->BS LoS link) | 'fixed' |
| communication.backhaul_bandwidth_hz | `20.0e6` | E | separate backhaul band (Sec. 3.2 says bands are separate) |
| communication.backhaul_rate_bps | `100.0e6` | E | used only when backhaul_mode == fixed |
| computation.ud_freq_hz | `[0.5e9, 0.75e9]` | A | Sec. 6.1 / Table 3 (f_m in Table 3 is a typo for f_n) |
| computation.uav_freq_hz | `10.0e9` | A | F_u = 10 GHz |
| computation.bs_freq_hz | `50.0e9` | A | F_b = 50 GHz |
| computation.kappa | `1.0e-27` | A | k = 1e-27 |
| computation.s_min | `0.01` | E | lower clip on the computation share s to avoid division by zero |
| computation.capacity_projection | `proportional` | E | enforce (19f): scale shares at a server if they sum > 1 |
| task.din_bits | `[2.0e6, 10.0e6]` | A | [2,10] Mbit (Mbit = 1e6 bit, [B]) |
| task.dout_bits | `[0.1e6, 1.0e6]` | A | [0.1,1] Mbit |
| task.cycles_per_bit | `[600.0, 750.0]` | A |  |
| task.lmax_s | `[1.0, 7.0]` | A |  |
| task.emax_j | `20.0` | A | E_n^max = 20 J |
| task.library_size | `200000` | E | number of distinct deterministic tasks (not given) |
| task.zipf_exponent | `0.8` | D | task popularity skew (Zipf 0.5-1 used in arXiv:2101.07930, 2007.11501) |
| task.lmax_per_request | `false` | E | false: deadline is a property of the task id; true: drawn per request |
| cost.gamma_latency | `0.5` | A |  |
| cost.gamma_energy | `0.5` | A |  |
| cost.latency_normalizer | `deadline` | C | eq.(17) symbol L_n^max = deadline ('deadline') or a constant ('constant') |
| cost.latency_normalizer_value_s | `7.0` | E | used only for 'constant' |
| cost.hit_latency_s | `0.0` | B | Sec. 4: cache-query latency negligible |
| cost.hit_energy_j | `0.0` | B |  |
| cost.reward_aggregation | `sum` | A | eq.(18)+(22): R_t = -sum_n C_n ; alternative 'mean' [E] |
| cost.deadline_penalty | `0.0` | E | no penalty term in the paper; extra cost per missed deadline |
| caching.uav_capacity_bytes | `2.0e9` | A | 2 GB (decimal GB assumed, [C]) |
| caching.bs_capacity_bytes | `10.0e9` | A | 10 GB |
| caching.entry_size | `dout` | A | eq.(14) counts D_out only ('dout' | 'din_plus_dout') |
| caching.policy | `hybrid` | A | 'hybrid' | 'lfu' | 'lru' | 'random' |
| caching.window_slots | `50` | A | W = 50 |
| caching.eta | `0.05` | A |  |
| caching.r_th | `0.5` | A |  |
| caching.delta0 | `0.5` | E |  |
| caching.delta_min | `0.1` | E |  |
| caching.delta_max | `0.9` | E |  |
| caching.adaptive_delta | `true` | A | eq.(34) |
| caching.lookup | `uav_then_bs` | E | 'uav_then_bs' | 'uav_only' |
| caching.reset_each_episode | `true` | E | Alg.1 line 5 "reinitialize environment" |
| caching.recency_floor_slots | `1.0` | E | guards 1/(t - L) in eq.(33) when t == L |
| training.episodes | `2000` | A | n_e |
| training.steps_per_episode | `1000` | A | n_s |
| training.seeds | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` | A | ten seeds; values [E] |
| training.eval_episodes | `3` | E | evaluation protocol not given |
| training.eval_steps_per_episode | `1000` | E |  |
| training.eval_seed_offset | `100000` | E |  |
| training.torch_threads | `1` | E |  |
| sac.hidden | `[400, 400, 400]` | A |  |
| sac.lr | `1.0e-4` | A |  |
| sac.gamma | `0.8` | A |  |
| sac.tau | `0.005` | A |  |
| sac.buffer_size | `1000000` | A | (needs ~10 GB RAM at N=100; reduce if necessary) |
| sac.batch_size | `64` | A |  |
| sac.alpha0 | `0.2` | A |  |
| sac.target_entropy_per_action_dim | `-1.0` | A | H = -5N = -(action dim) |
| sac.updates_per_step | `1` | A | "single gradient update per environment step" |
| sac.update_every | `1` | C | text also says "updated every 100 time slots" |
| sac.learning_starts | `64` | E | start updates once one mini-batch is available |
| sac.alpha_parameterization | `log` | D | log-alpha as in Haarnoja et al. 2018b [45] |
| ddpg.exploration_noise_std | `0.1` | D | Gaussian N(0, 0.1) (Fujimoto et al. 2018) |
| ppo.gae_lambda | `0.95` | D | Schulman et al. 2017 |
| ppo.clip | `0.2` | D |  |
| ppo.epochs | `10` | D |  |
| ppo.minibatch_size | `64` | E | equal to SAC batch |
| ppo.value_coef | `0.5` | D |  |
| ppo.entropy_coef | `0.0` | D | (continuous-control default) |
| ppo.max_grad_norm | `0.5` | D |  |
| ppo.init_log_std | `-0.5` | E |  |
| a3c.n_workers | `4` | E | not given |
| a3c.t_max | `20` | D | n-step rollout (Mnih et al. 2016 use 5-20) |
| a3c.entropy_beta | `1.0e-4` | D | Mnih et al. 2016 continuous control |
| a3c.max_grad_norm | `40.0` | E |  |
A = explicit in paper, B = clearly implied, C = missing/ambiguous, D = external literature, E = engineering assumption. Details: REPRODUCTION_ASSUMPTIONS.md
