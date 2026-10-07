# Reproduction assumptions register

Paper: T. Baidya, S. Moh, "Joint Task Offloading and Resource Allocation with Data Caching in UAV-Aided Mobile
Edge Computing Networks for Latency-Sensitive Applications", *Sensors* 2026, 26, 4966.

Provenance categories: **A** explicitly specified · **B** clearly implied · **C** missing/ambiguous ·
**D** reconstructed from external literature · **E** engineering assumption.
Impact: how much a reasonable change could alter reported numbers / conclusions (**High / Medium / Low**),
judged from `results/sensitivity/sensitivity.md` where measured, otherwise from reasoning (marked *est.*).
Every C/D/E item is a config key in `configs/paper_default.yaml` and can be changed with `--set key=value`.

## 1. Facts taken directly from the paper (A)

| Item | Paper | Code |
|---|---|---|
| Topology | 1 BS at (0,0), M UAVs at H = 100 m, N UDs, 2 km × 2 km (Sec. 3.1, 6.1) | `system_model.py` |
| N, M defaults | N = 100, M = 2 (Table 3) | `system.*` |
| Channel | h = h0/(H² + d²), h0 = −50 dB (eq. 1, Table 3) | `channel_model.channel_gain` |
| Rate | R = B_n log2(1 + P h /(B_n N0)), OFDMA, B = 20 MHz (eq. 2) | `channel_model.shannon_rate` |
| Coverage | offloading requires d ≤ H tanθ (Sec. 3.2) | `env.step` (projection to local) |
| Task | D_in ∈ [2,10] Mbit, cycles/bit ∈ [600,750], D_out ∈ [0.1,1] Mbit, L^max ∈ [1,7] s | `task_model.py` |
| CPUs | f_n ∈ [0.5,0.75] GHz (heterogeneous), F_u = 10 GHz, F_b = 50 GHz, k = 1e−27 | `cost_model.py` |
| Latency/energy | eqs. (4)–(16); downlink neglected | `cost_model.py` |
| Cost | C_n = γ_L L/L^max + γ_E E/E^max, γ = 0.5/0.5, E^max = 20 J (eq. 17, Table 3) | `cost_model.task_cost` |
| Reward | R_t = −C(t), C = Σ_n C_n (eq. 18, 22) | `env.step` |
| Constraints | (19b)(19c) one-hot, (19e) 0 < P ≤ Pmax, (19f) Σ s ≤ 1, (14) cache size | `offloading.py`, `caching.py` |
| Action | 5 values/user: 3 preference scores + power + CPU share; argmax projection (21a, 21b) | `offloading.decode_action` |
| SAC | twin critics + targets, auto-α, α0 = 0.2, H = −5N, 3×400 ReLU, Adam 1e−4, γ = 0.8, τ = 0.005, buffer 1e6, batch 64, 1 update/step (Alg. 1, Sec. 6.1) | `agents/sac.py` |
| Training budget | n_e = 2000 episodes × n_s = 1000 steps, 10 seeds, 95 % CIs, paired Wilcoxon | `configs`, `experiments/` |
| Caching | cache result of every newly executed *offloaded* task at the executing server; hit → cost ≈ 0; hybrid score (33); δ adaptation (34), W = 50, η = 0.05, r_th = 0.5; evict lowest score | `caching.py`, `env.py` |
| Alg. 2 counters | F = 1 and L = t at insertion; F += 1 and L = t on hit | `caching.py` |
| Cache sizes | S_u = 2 GB, S_b = 10 GB; Fig. 14: 2–32 GB; Fig. 15: 10–160 GB | `configs/figures.yaml` |
| Baselines | PPO / DDPG / A3C with caching, SAC without caching; LFU / LRU / Random replacement | `baselines.py` |

## 2. Reconstructed items

### A-1 Task arrival per slot — **C**, impact **High**
*Paper:* "In each timeslot, a single user generates a single computational task" (Sec. 6.1).
*Missing:* whether this means each user generates one task or exactly one user is active per slot.
*Chosen:* every UD issues one task per slot (all N tasks share the slot's resources).
*Why:* (i) constraint (19f) Σ_n s_{i,n} F_i ≤ F_i, the joint 5N-dimensional action and target entropy −5N are
only meaningful if many tasks compete in a slot; (ii) the training-free capacity bound
(mean demand ÷ total CPU) is 0.98 s at N = 20 and 3.35 s at N = 120 (`results/sensitivity/sensitivity.md`
Part 3), closely matching the paper's JORC latency 1.6 → 3.7 s; with one task per slot the BS alone would finish a
task in ≈ 0.1 s and latency could not grow with N as in Fig. 4. *Literature:* none needed (internal evidence).

### A-2 Fig. 3 reward scale — **C**, impact **Medium** (plot scale only)
eq. (22) with C = Σ_n C_n over 1000 steps gives episode returns of order −10⁴ at N = 100, whereas Fig. 3 shows
≈ −230. Default follows the equations (`cost.reward_aggregation=sum`); `mean` divides by N. Neither choice is
tuned to match Fig. 3. The magnitude in Fig. 3 is irreproducible from the stated equations.

### A-3 Latency normaliser in eq. (17) — **C**, impact **Medium**
L^max_n denotes both the deadline (Sec. 3.3) and "the maximum value of L_n" (Sec. 4). Default: the deadline.
Option: `cost.latency_normalizer=constant`.

### A-4 UAV horizontal positions — **E**, impact **Low** (*est.*; positions only enter backhaul distance)
Not given. Default: ring of radius 500 m around the BS (`uav_layout=ring`); option `random`.

### A-5 Beamwidth θ — **C/E**, impact **Low** (measured: θ = 45°/75° changes diagnostic cost < 2 %)
θ is used but never given. Default 60° → coverage radius 173 m.

### A-6 User spatial distribution and association — **C/E**, impact **High** if changed to uniform
Paper says users are random over 2 km × 2 km *and* every user is within a UAV's range; with H tanθ coverage these
are incompatible (uniform placement leaves ≈ 95 % of users out of coverage at θ = 60°). Default: each user is
assigned round-robin to a UAV and placed uniformly in that UAV's coverage disk, re-drawn every slot
("positions refreshed at every time step", A). Option `user_placement=uniform_area` (nearest UAV; uncovered users
are forced to local execution and cannot query caches).

### A-7 Noise PSD N0 — **D**, impact **Low/Medium**
Not given. −174 dBm/Hz thermal noise (3GPP; used e.g. in arXiv:2206.04302, 1912.07112, 2301.08108).

### A-8 UD maximum transmit power P_max — **D**, impact **Low/Medium**
Not given. 0.2 W (23 dBm, 3GPP UE power class; arXiv:2206.04302 citing 3GPP TR 36.777). P_min = 1 mW (E) so that
P > 0 (19e).

### A-9 Backhaul R_{m,b} and UAV power P_{m,b} — **D/E**, impact **Low** (measured ±4 % diagnostic cost)
Not given. P_{m,b} = 1 W (UAV transmit power in arXiv:2501.15164). R_{m,b} computed with the same LoS model on a
dedicated 20 MHz band (≈ 178 Mbit/s at 500 m) and **not** shared among forwarded tasks (literal eq. 6).
Option `backhaul_mode=fixed`.

### A-10 Bandwidth split B_n — **E**, impact **Medium** (*est.*)
OFDMA is stated but not the allocation. Default: each UAV owns B = 20 MHz, split equally among its associated
users. Options: `bandwidth_scope=shared`, `bandwidth_split=equal_offloaders`.

### A-11 Action → (P, s) mapping and s_min — **E**, impact **Low/Medium**
Affine maps from [−1,1]: P = P_min + (a+1)/2 (P_max − P_min); s = clip((a+1)/2, 0.01, 1). The 0.01 floor avoids
W/(sF) → ∞.

### A-12 Enforcing (19f) — **E**, impact **Medium** (*est.*)
The paper lists (19f) but not how the actor respects it. Shares at a server are scaled proportionally if they
sum above 1.

### A-13 Deadline constraint (19d) — **C**, impact **Medium**
No penalty appears in the reward; violations only lower STCR. Default: no penalty (`cost.deadline_penalty=0`).
There is no queueing between slots; each slot's resources are fully available (paper gives no queue model).

### A-14 Repeated-task model (task catalogue, popularity) — **E/D**, impact **High** (largest of all)
Result caching needs recurring identical tasks; the paper gives no request model. Default: K = 200 000
deterministic tasks with fixed attributes and Zipf(0.8) popularity (Zipf 0.5–1 used in arXiv:2101.07930,
2007.11501). Measured: K = 1 000 → UAV hit ratio 0.82; K = 2·10⁶ → 0.10; Zipf 0.5 / 1.0 → 0.03 / 0.47
(diagnostic policy). Absolute hit ratios, latency and energy therefore cannot be pinned down from the paper.

### A-15 Cache units, entry size and persistence — **C/E**, impact **High** for Figs. 14–15
Eq. (14) counts D_out only; 2 GB then holds ≈ 29 000 results. With caches reset per episode (E) even a full
1000-slot episode produces ≈ 2 GB of results per UAV, so **capacity in GB never binds** and all replacement
policies tie (`results/cache_diagnostic/`). If the values were meant in **MB**, capacity binds and policies
differ. Options: `caching.entry_size=din_plus_dout`, `reset_each_episode=false`, MB capacities.

### A-16 Cache lookup path and storage location — **E**, impact **Medium**
Default: request checked at the assigned UAV, then at the BS (`lookup=uav_then_bs`); result stored at the server
that executed it (Alg. 2 line 14 "if executed at UAV or BS"). Hits cost 0 s/0 J (Sec. 4 "negligible", B).

### A-17 Hybrid-score details — **E**, impact **Medium**
δ0 = 0.5, δ_min = 0.1, δ_max = 0.9 (not given). r_t = 1 − (#distinct ids)/(#requests) over the last W = 50 slots
per cache. 1/(t − L) is floored at age 1 slot (eq. 33 divides by zero at t = L). F(T) is unnormalised, as printed.

### A-18 Cache-replacement baselines — **B/E**
LFU ties broken by recency (E); LRU by last access; Random uniform over entries.

### A-19 SAC details not in the paper — **D**
tanh-squashed Gaussian with log-det correction and log-α optimisation as in Haarnoja et al. [45]; α included in
the target although eq. (27) omits it (eq. 25 includes it); no done-mask at the time limit; updates start once one
mini-batch is stored. Contradiction: Sec. 6.1 says both "updated every 100 time slots" and "a single gradient
update per environment step"; default follows Alg. 1 (`sac.update_every=1`), option `update_every=100`.

### A-20 Baseline hyper-parameters — **D**
Paper: "followed their original publications", same 3×400 backbone, Adam, same lr. PPO: clip 0.2, GAE 0.95, 10
epochs, mini-batch 64, rollout = 1 episode (Schulman 2017). DDPG: Gaussian noise 0.1 (Fujimoto 2018).
A3C: 4 workers, t_max 20, entropy 1e−4 (Mnih 2016); the 2000-episode budget is the total over workers (E).
"With caching" = JORC's hybrid cache (B: Sec. 7.1 "the same caching mechanism").

### A-21 State features — **E**, impact **Low/Medium**
Eq. (20) lists the components but not the encoding. Per user: D_in, D_out, W, L^max, position, f_n, local-energy
status k f² W / E^max, channel gain (dB), cache-hit flag; per UAV: position, capacity, current workload;
global: BS capacity, congestion (share of cache misses). All normalised to O(1).

### A-22 Evaluation protocol and metric averaging — **E**
Not given. Deterministic (mean) actions, 3 held-out episodes × 1000 slots, same task catalogue as training;
metrics averaged per task, cache hits counted with 0 latency and as successful.

### A-23 Units — **B**
Mbit = 10⁶ bit; GB = 10⁹ byte = 8·10⁹ bit (decimal); GHz = 10⁹ cycles/s; log base 2 in eq. (2).
Table 3 labels UD frequency "f_m" and UAV "f_u"; read as f_n and F_u.

### A-24 Seeds — **E**
Seeds 0–9. Each run derives the task catalogue/UAV layout from its seed and per-episode seeds from
`SeedSequence([seed, episode])`, shared by all methods (common random numbers).
