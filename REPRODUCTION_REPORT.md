# Reproduction report — JORC (Sensors 2026, 26, 4966)

**Status.** The full method, all baselines and all experiment pipelines are implemented and tested. The paper's
training budget (2000 × 1000 steps per run, ≈ 26 CPU-hours per run at N = 100 on one core; ≈ 650 runs for all
figures × 10 seeds) was **not** executed. Only a *smoke* scale (4 episodes × 50 steps, 2 seeds) was run, to
validate the pipeline end to end. **Smoke numbers are not reproductions of the paper's results** and are not
presented as such. Conclusions below come from equation-level checks, training-free diagnostics and sensitivity
analysis, which do not depend on training length.

## 1. Paper → code mapping

| Paper | Implementation |
|---|---|
| Sec. 3.1 network model, quasi-static positions | `src/jorc/system_model.py::Geometry` |
| eq. (1) channel gain; coverage H tanθ | `channel_model.py::channel_gain`, `coverage_radius`; enforced in `env.py::step`/`_new_slot` |
| eq. (2) uplink rate, OFDMA | `channel_model.py::shannon_rate`; `env.py::_bandwidth` |
| R_{m,b} backhaul | `channel_model.py::backhaul_rates` |
| Sec. 3.3 task tuple (T_n, L^max) | `task_model.py::TaskLibrary` |
| eq. (3), (19b), (19c), (21a), (21b) | `offloading.py::decode_action` (argmax → one-hot) |
| eq. (4)/(6) uplink latency, (5)/(7) energy | `cost_model.py::uplink_latency`, `uplink_energy` |
| eq. (8)–(13) computation latency/energy | `cost_model.py::local_*`, `edge_*` |
| eq. (14) cache storage | `caching.py::ResultCache.insert`, `check_invariants` |
| eq. (15)–(16) per-task totals | `cost_model.py::task_latency_energy` |
| eq. (17)–(18) cost | `cost_model.py::task_cost`; sum in `env.py::step` |
| eq. (19d) deadline | STCR in `env.py::step`/`metrics.py` (no penalty, see A-13) |
| eq. (19e) power bounds | `offloading.py::decode_action` |
| eq. (19f) CPU capacity | `offloading.py::project_shares`, verified by `server_load` |
| eq. (20) state, (21) action, (22) reward | `env.py::_observe`, `decode_action`, `step` |
| eq. (23)–(32), Alg. 1 | `agents/sac.py::SACAgent.update`; loop in `training.py::train` |
| eq. (33) hybrid score, (34) δ adaptation | `caching.py::scores`, `end_slot` |
| Alg. 2 (lookup, hit update, insert, evict) | `env.py::_new_slot` (lookup), `env.py::step` (insert), `caching.py` |
| Sec. 6.2 baselines | `baselines.py`, `agents/{ppo,ddpg,a3c}.py`, `caching.POLICIES` |
| Sec. 6.3 metrics | `metrics.py::MetricAccumulator` |
| Figs. 3–15 | `experiments/reproduce_figures.py` + `configs/figures.yaml` |
| 95 % CIs, Wilcoxon | `utils.py::mean_ci95`, `experiments/reproduce_tables.py` |

## 2. Experiments

| Figure | x-axis | y-axis | Methods | Fixed | Script entry |
|---|---|---|---|---|---|
| 3 | Episodes (0–2000) | episode reward | SAC (JORC), PPO, DDPG, A3C (with caching) | N = 100*, M = 2 | `fig3` |
| 4,5,6,7,12 | Number of UDs 20–120 | latency, energy, cost, STCR, offloading ratio | 5 methods | M = 2 | `fig4_7_12` |
| 8,9,10,11,13 | Number of UAVs 1–8 | same | 5 methods | N = 100 | `fig8_11_13` |
| 14 | UAV capacity 2–32 GB | cache hit ratio at UAVs | Hybrid, LFU, LRU, Random | N = 100, M = 2 | `fig14` |
| 15 | BS capacity 10–160 GB | cache hit ratio at BS | Hybrid, LFU, LRU, Random | N = 100, M = 2 | `fig15` |

\*N for Fig. 3 is not stated. All figures: mean over 10 seeds with 95 % CIs. Figs. 14–15 evaluate the trained
JORC policy with each replacement policy/capacity (only the cache module changes). The paper has no results
tables besides parameters (Table 3) and the qualitative Table 1; `reproduce_tables.py` regenerates Table 3 with
provenance, a paper-vs-reproduction table, and the Wilcoxon tests.

## 3. Validation

**Level 1 — equations.** `tests/test_channel.py`, `test_latency.py`, `test_energy.py` check eqs. (1), (2),
(4)–(17) against hand-computed values, including unit conversions (dB, dBm, Hz, bits). A YAML pitfall was
found and fixed: PyYAML parses `20.0e6` as a *string*; a float resolver plus range assertions in
`utils.validate_config` now guard units.

**Level 2 — constraints.** `test_constraints.py` checks (19c) one-hot, (19e) power bounds, (19f) after
projection, coverage, and that every task is either a hit or executed exactly once; `test_caching.py` checks (14)
under random variable-size insertions and every evaluation calls `check_invariants`. Observed in all smoke runs:
`max_server_load ≤ 1`.

**Level 3 — numerical sanity** (training-free policies, N = 100, M = 2, `results/sensitivity/sensitivity.md`):
- Uplink rates ≈ 5–7 Mbit/s per user (400 kHz share, d ≤ 173 m); backhaul ≈ 178 Mbit/s.
- All-local: latency 6.5 s, energy 1.6 J, STCR 0.24. All-BS (equal share): latency 5.3 s, energy 1.5 J,
  STCR 0.40. All-UAV: latency 13.7 s (2 × 10 GHz for ≈ 50 tasks each).
- Energy formula (11)/(13): one task at full UAV speed costs ≈ 400 J (≫ E^max = 20 J), at full BS speed
  ≈ 10 kJ. Low CPU shares are therefore optimal for the energy term — a property of the paper's equations.
- Capacity bound (mean demand ÷ total CPU): 0.98, 2.26, 3.06, 3.35 s for N = 20, 60, 100, 120 (M = 2) and
  3.31 → 2.10 s for M = 1 → 8 (N = 100). The paper's JORC latencies (1.6 → 3.7 s; 3.4 → 1.7 s) follow the same
  scale and trend, supporting the system-model reading A-1.
- SAC sanity: `test_optimization.py` checks target entropy −5N, the tanh log-probability, convergence on a toy
  bandit, and determinism of the environment for a fixed seed.

**Level 4 — sensitivity** (diagnostic policies; one-at-a-time):
- Low impact: θ (45°/75°), backhaul (20/100 Mbit/s fixed), noise PSD, P_max.
- High impact: task-catalogue size K (UAV hit ratio 0.82 / 0.39 / 0.21 / 0.10 for K = 10³, 2·10⁴, 2·10⁵, 2·10⁶),
  Zipf exponent (0.03 at 0.5, 0.47 at 1.0), user placement (uniform over the area → most users uncovered),
  cache units (GB vs MB).
- The trained comparison of "caching lowers cost" per variant is available via
  `sensitivity_analysis.py --train` but was not run at a meaningful budget.

**Level 5 — comparison with the paper.**
- Smoke-scale tables (`results/smoke/tables/comparison.md`) are produced but carry no evidential weight (≈ 200
  gradient steps; e.g. JORC energy at N = 20 is 73 J ± 110 because the untrained policy still uses large CPU
  shares). Wilcoxon p-values with 2 seeds cannot be below 0.5.
- **Cache capacity (Figs. 14–15) — discrepancy found.** With the paper's GB capacities, eq. (14) entry sizes
  (D_out) and episodic caches, the caches never fill even over a full 1000-slot episode
  (`experiments/cache_capacity_diagnostic.py`): hit ratio 0.420 for every policy and every capacity 2–32 GB. The
  paper's strong capacity dependence (0.4 → 0.9) cannot arise under these readings. Reading the capacities as
  **MB**, capacity binds:

  | Policy (UAV, fixed all-UAV policy, 1000 slots, seed 0) | 2 MB | 4 | 8 | 16 | 32 MB |
  |---|---|---|---|---|---|
  | Hybrid (eq. 33 + 34) | 0.047 | 0.057 | 0.088 | 0.132 | 0.178 |
  | LFU | 0.085 | 0.107 | 0.131 | 0.159 | 0.190 |
  | LRU | 0.024 | 0.037 | 0.061 | 0.088 | 0.122 |
  | Random | 0.016 | 0.029 | 0.047 | 0.072 | 0.103 |

  Hybrid lies **between** LFU and LRU, not above both as in the paper. Cause traced: the observed repetition rate
  r_t ≈ 0.16 < r_th = 0.5, so eq. (34) drives δ to δ_min (LRU-leaning). With δ fixed at 0.9 the hybrid reaches
  0.130 at 8 MB, still not above LFU (0.131). Under a stationary Zipf stream LFU is near-optimal, so a hybrid
  outperforming LFU would need non-stationary popularity, which the paper does not describe. This is reported
  as an unresolved discrepancy, not tuned away.

## 4. Discrepancies and likely causes

| Observation | Likely cause |
|---|---|
| Fig. 3 reward magnitude (−230) inconsistent with eq. (18)+(22) | unreported reward scaling/averaging |
| Uniform users over 2 km² vs coverage H tanθ | unreported placement; resolved by in-coverage placement |
| Capacity insensitivity in GB | unreported request model / entry size / units / cache persistence |
| Hybrid ≯ LFU | stationary popularity; δ bounds and r_t definition unreported |
| Fig. 12 text: both JORC and SAC-w/o-caching "highest offloading ratio" | inconsistency in the text |
| "update every 100 slots" vs "one update per step" | contradiction in Sec. 6.1 |
| eq. (27) omits α | typo relative to eq. (25) |

## 5. Reproduction confidence

| Component | Paper specification | Implementation | Confidence |
|---|---|---|---|
| Network topology (BS, M UAVs, H) | Explicit | as stated | High |
| UAV positions, θ, user placement | Missing / inconsistent | ring 500 m, 60°, in-coverage disks | Low |
| Channel model eq. (1)–(2) | Explicit | exact | High |
| N0, P_max, P_{m,b}, R_{m,b}, B_n split | Missing | literature / assumption | Medium (low measured impact) |
| Task attribute ranges | Explicit | exact | High |
| Task arrival per slot | Ambiguous | all users each slot | Medium (supported by capacity bound) |
| Repeated-task / popularity model | Missing | catalogue + Zipf | Low (high impact) |
| Latency/energy eqs. (4)–(16) | Explicit | exact | High |
| Cost eq. (17)–(18) | Partial (normaliser ambiguous) | deadline normaliser | Medium |
| Constraint handling (19d), (19f) | Partial | no penalty; proportional projection | Medium |
| MDP state encoding | Partial | listed features, normalised | Medium |
| Action mapping (21a/b) | Explicit for X; maps for P, s missing | argmax + affine | Medium-High |
| SAC algorithm & hyper-parameters | Explicit | Alg. 1 per [45] | High |
| Hybrid LFU-LRU eq. (33)–(34) | Explicit formula, missing δ bounds/δ0, r_t definition | literal + assumptions | Medium |
| Cache units / entry size / persistence | Ambiguous | GB, D_out, per-episode reset | Low |
| PPO / DDPG / A3C baselines | "standard" only | standard implementations | Medium |
| Evaluation protocol / averaging | Missing | deterministic, per-task | Medium |
| Reported numeric results | — | not reproduced (budget) | not assessed |

## 6. Final audit (section by section)

- **Sec. 3.1–3.2.** Faithful except positions/θ/placement (A-4–A-6). The paper's quasi-static mobility is
  implemented as i.i.d. re-placement each slot; true random-walk mobility would correlate states — authors may differ.
- **Sec. 3.3–3.5.** Faithful. Task identity for caching is an addition required by Sec. 3.6.
- **Sec. 3.6 / 5.2 / Alg. 2.** Faithful to the formulas; uncertain on everything the formulas leave open
  (A-14–A-18). The authors' cache could persist across episodes or store at both UAV and BS — both would raise hit
  ratios.
- **Sec. 4.** Faithful; the deadline constraint is not enforced in the reward because the paper gives no mechanism.
- **Sec. 5.1 / Alg. 1.** Faithful SAC. Possible differences: update frequency (A-19), state encoding (A-21),
  whether the actor sees hit flags.
- **Sec. 6.1.** Budget, network sizes and hyper-parameters as stated; the 1e6-transition buffer needs ≈ 10 GB RAM
  at N = 100 (float32), suggesting the authors may have used a smaller state or more memory.
- **Sec. 6.2.** Baselines implemented as standard algorithms; their exact settings in the paper are unknown.
- **Sec. 6.3.** Metric definitions are the paper's prose; averaging conventions are assumed.

## 7. How to complete the reproduction

1. `bash scripts/run_all.sh paper` on a machine with ≥ 16 GB RAM (or set `sac.buffer_size` lower and record it).
   Budget ≈ 650 runs × ≈ 26 CPU-h; GPU or parallel seeds recommended.
2. Re-run `experiments/reproduce_tables.py --scale paper` and compare `comparison.md` with the paper.
3. Repeat Figs. 14–15 under the alternative cache readings (MB units, `reset_each_episode=false`,
   `entry_size=din_plus_dout`) and under several catalogue sizes, and report all of them.
