# JORC reproduction — Joint Task Offloading and Resource Allocation with Data Caching in UAV-Aided MEC

Independent reproduction of

> T. Baidya and S. Moh, "Joint Task Offloading and Resource Allocation with Data Caching in UAV-Aided Mobile
> Edge Computing Networks for Latency-Sensitive Applications," *Sensors*, vol. 26, 4966, 2026.
> doi:10.3390/s26154966

The authors did not release code. This repository implements the paper's system model, optimisation problem,
SAC-based JORC algorithm (Alg. 1), hybrid LFU-LRU task-result caching (Alg. 2), all baselines and all figure
pipelines, as close to the text as the paper allows. Everything the paper leaves open is recorded in
[`REPRODUCTION_ASSUMPTIONS.md`](REPRODUCTION_ASSUMPTIONS.md) and is a configuration switch.
Validation, discrepancies and confidence are in [`REPRODUCTION_REPORT.md`](REPRODUCTION_REPORT.md).

> **Status: implementation complete and tested; paper-scale training NOT run.** One paper-scale run
> (2000 × 1000 steps, N = 100) takes ≈ 26 CPU-hours; the full figure set is ≈ 650 runs. Results under
> `results/smoke/` come from a pipeline-validation budget (4 × 50 steps, 2 seeds) and must not be read as
> reproduced numbers.

## Repository structure

```
configs/paper_default.yaml   all parameters, each tagged [A]-[E] (provenance)
configs/figures.yaml         experiment definitions for Figs. 3-15
configs/baselines.yaml       compared methods and cache-replacement policies
configs/scales.yaml          compute profiles: paper | reduced | smoke
configs/sensitivity.yaml     one-at-a-time variations of reconstructed parameters
data/paper_reported_values.yaml   numbers stated in the paper's text (comparison only)
src/jorc/
  utils.py           config loading (YAML float fix), unit conversions, seeds, CIs
  channel_model.py   eq.(1)-(2), coverage, backhaul
  task_model.py      task catalogue + Zipf requests
  system_model.py    BS/UAV/UD geometry, association, mobility
  cost_model.py      eq.(4)-(17)
  offloading.py      action decoding (21a/b), (19e), (19f) projection
  caching.py         task-result cache: hybrid eq.(33)/(34), LFU, LRU, Random
  env.py             MDP: state (20), action (21), reward (22), Alg. 2 flow
  agents/            sac.py (Alg. 1), ppo.py, ddpg.py, a3c.py, common.py
  baselines.py       method registry with the paper's labels
  training.py        training / evaluation loops
  metrics.py         Sec. 6.3 metrics
  diagnostics.py     training-free reference policies (sanity checks only)
experiments/
  run_main_experiment.py   one method / configuration / seed(s)
  reproduce_figures.py     Figs. 3-15 (resumable; raw JSON per run)
  reproduce_tables.py      Table 3 with provenance, paper-vs-reproduction, Wilcoxon tests
  sensitivity_analysis.py  sensitivity of reconstructed parameters
  cache_capacity_diagnostic.py  cache module at paper request volume (Figs. 14-15 setting)
tests/                     25 unit/integration tests
scripts/run_all.sh         end-to-end pipeline
results/                   generated outputs
```

## Model summary

One BS at (0,0), M UAVs at altitude H = 100 m, N UDs. Each UD is served by one UAV; offloading requires
d ≤ H tanθ. Channel h = h0/(H² + d²); OFDMA uplink R = B_n log2(1 + P h/(B_n N0)). A task (D_in, W, D_out, L^max)
runs locally (W/f_n, k f_n² W), at the UAV (uplink + W/(sF_u), k(sF_u)²W) or at the BS via the UAV backhaul
(+ D_in/R_{m,b}, P_{m,b}D_in/R_{m,b}, W/(sF_b), k(sF_b)²W). Cost C_n = 0.5 L/L^max + 0.5 E/E^max; the problem
minimises Σ_n C_n subject to one-hot offloading, deadlines, 0 < P ≤ P_max and Σ s ≤ 1 per server.
Results of tasks executed at a UAV/BS are cached there; repeat requests are served at negligible cost.

## Algorithm summary

* **JORC.** Per slot, every request is first checked in the UAV (then BS) cache. For misses, a SAC actor outputs
  5 values per user (3 location scores → argmax, transmit power, CPU share). SAC: twin critics + target critics,
  automatic temperature (α0 = 0.2, target entropy −5N), 3×400 ReLU, Adam 1e−4, γ = 0.8, τ = 0.005, buffer 1e6,
  batch 64, one gradient step per environment step. Results are cached; when full, the entry with the lowest
  H(T) = δF(T) + (1−δ)/(t − L(T)) is evicted, and δ is adapted from the observed repetition rate.
* **Baselines.** PPO / DDPG / A3C with the same caching; SAC without caching; LFU / LRU / Random replacement.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: conda env create -f environment.yml
python -m pytest -q tests                # 25 tests, ~1 min
```
Tested with Python 3.12, PyTorch 2.5.1 (CPU), NumPy 2.x, SciPy 1.x.

## Configuration

All parameters live in `configs/paper_default.yaml`. Any script accepts overrides:
`--set system.n_users=60 caching.policy=lru`. Scales: `--scale smoke | reduced | paper`.

## Running

```bash
# main experiment (paper default point N=100, M=2), one seed, paper budget
python experiments/run_main_experiment.py --method JORC --scale paper --seeds 0 --checkpoint

# all figures (resumable; raw results in results/<scale>/runs, plots in results/<scale>/figures)
python experiments/reproduce_figures.py --scale paper
python experiments/reproduce_figures.py --scale paper --figures fig4_7_12      # Figs. 4, 5, 6, 7, 12
python experiments/reproduce_figures.py --scale paper --figures fig8_11_13     # Figs. 8, 9, 10, 11, 13
python experiments/reproduce_figures.py --scale paper --figures fig3 fig14 fig15

python experiments/reproduce_tables.py --scale paper     # tables in results/paper/tables
python experiments/sensitivity_analysis.py               # training-free part (minutes)
python experiments/sensitivity_analysis.py --train --scale reduced
python experiments/cache_capacity_diagnostic.py --units GB   # and --units MB
bash scripts/run_all.sh smoke                             # whole pipeline at smoke scale
```

### Expected outputs

| Command | Output |
|---|---|
| `reproduce_figures.py` | `results/<scale>/figures/fig{3..15}.png` and `.json` (mean, 95 % CI, per-seed values) |
| `reproduce_tables.py` | `parameter_provenance.md`, `comparison.md`, `wilcoxon.md` |
| `sensitivity_analysis.py` | `results/sensitivity/sensitivity.md` |
| `cache_capacity_diagnostic.py` | `results/cache_diagnostic/*.png/json` |

## Baselines

| Label (paper) | Difference from JORC |
|---|---|
| PPO with caching | PPO-clip instead of SAC (on-policy, no entropy-tuned off-policy learning) |
| DDPG with caching | deterministic actor + Gaussian exploration, single critic |
| A3C with caching | asynchronous n-step advantage actor-critic, 4 workers |
| SAC without caching | identical SAC; no cache lookup and no insertion |
| LFU / LRU / Random | replacement policy only (Figs. 14–15) |

## Random seeds

Seeds 0–9 (`training.seeds`). For each run the seed fixes the task catalogue, UAV layout and torch init;
episode e uses `SeedSequence([seed, e])` for every method (common random numbers); evaluation uses
`SeedSequence([seed + 100000, e])`. Raw per-run JSON includes the full config.

## Parameter table (abridged; full table with tags: `results/<scale>/tables/parameter_provenance.md`)

| Parameter | Value | Source |
|---|---|---|
| N, M, H | 100, 2, 100 m | paper |
| B, h0 | 20 MHz, −50 dB | paper |
| D_in, D_out, cycles/bit, L^max | [2,10] Mbit, [0.1,1] Mbit, [600,750], [1,7] s | paper |
| f_n, F_u, F_b, k | [0.5,0.75] GHz, 10 GHz, 50 GHz, 1e−27 | paper |
| γ_L = γ_E, E^max | 0.5, 20 J | paper |
| S_u, S_b, W, η, r_th | 2 GB, 10 GB, 50, 0.05, 0.5 | paper |
| SAC hyper-parameters | see Algorithm summary | paper |
| N0 | −174 dBm/Hz | literature (3GPP) |
| P_max (UD), P_{m,b} (UAV) | 0.2 W, 1 W | literature |
| θ, UAV positions, user placement | 60°, ring 500 m, in-coverage disks | assumption |
| Task catalogue, popularity | 200 000 tasks, Zipf 0.8 | assumption / literature |
| δ0, δ_min, δ_max | 0.5, 0.1, 0.9 | assumption |

## Assumptions and literature-derived parameters

See [`REPRODUCTION_ASSUMPTIONS.md`](REPRODUCTION_ASSUMPTIONS.md) (24 numbered items with impact ratings).
The highest-impact unknowns are: the repeated-task/popularity model, cache-size units/persistence,
user placement vs coverage, and the per-slot task-arrival reading.

## Known deviations from the paper

* Users are placed inside UAV coverage, not uniformly over 2 km × 2 km (the two statements are incompatible).
* α is included in the critic target (eq. 27 omits it).
* Cache recency term floored at one slot (eq. 33 divides by zero at t = L).
* Fig. 3's reward magnitude is not matched (it is inconsistent with eqs. 18 and 22).
* Under GB cache sizes, capacity does not affect hit ratio in our model; under MB sizes, the hybrid policy ranks
  between LFU and LRU rather than first (REPRODUCTION_REPORT.md §3).

## Reproducibility limitations

No source code, no request/popularity model, and no evaluation protocol were published; several values are
missing (REPRODUCTION_ASSUMPTIONS.md). Absolute numbers depend strongly on the reconstructed request model.
The paper's buffer size (1e6) needs ≈ 10 GB RAM at N = 100. Paper-scale runs were not executed here.

## Citation

```bibtex
@article{baidya2026jorc,
  author  = {Baidya, Tanmay and Moh, Sangman},
  title   = {Joint Task Offloading and Resource Allocation with Data Caching in {UAV}-Aided Mobile Edge
             Computing Networks for Latency-Sensitive Applications},
  journal = {Sensors}, volume = {26}, pages = {4966}, year = {2026}, doi = {10.3390/s26154966}
}
```
