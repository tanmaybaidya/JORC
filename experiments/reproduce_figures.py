"""Reproduce Figures 3-15 of the paper.

    python experiments/reproduce_figures.py --scale paper            # paper budget (very expensive)
    python experiments/reproduce_figures.py --scale smoke            # pipeline check only
    python experiments/reproduce_figures.py --scale reduced --figures fig4_7_12 fig14

Every (method, configuration, seed) result is stored as raw JSON under results/<scale>/runs/ and reused if
present, so interrupted sweeps resume. Aggregated values are written to results/<scale>/figures/*.json and
plots to results/<scale>/figures/*.png. Error bars: 95% t-intervals over seeds (Sec. 6.1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (GB, ROOT, aggregate, base_config, load_yaml, merged, plot_bars, plot_lines,  # noqa: E402
                    point_tag, run_point, save_json)

from jorc.baselines import METHODS, SLUG, make_agent  # noqa: E402
from jorc.env import UAVMECEnv  # noqa: E402
from jorc.training import evaluate, method_config  # noqa: E402
from jorc.utils import mean_ci95  # noqa: E402

YLABEL = {"avg_latency_s": "Average task completion latency (s)", "avg_energy_j": "Average energy consumption (J)",
          "avg_cost": "Average cost", "stcr": "STCR", "offloading_ratio": "Offloading ratio",
          "cache_hit_ratio_uav": "Cache hit ratio at UAVs", "cache_hit_ratio_bs": "Cache hit ratio at BS"}


def _set(cfg_updates: dict, dotted: str, value):
    node = cfg_updates
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    return cfg_updates


def sweep(defn: dict, base: dict, runs_dir: Path, figs_dir: Path, name: str, methods=None):
    methods = methods or defn["methods"]
    seeds = base["training"]["seeds"]
    xs = defn["sweep_values"]
    raw = {m: {} for m in methods}
    for x in xs:
        upd = json.loads(json.dumps(defn["fixed"]))
        cfg = merged(base, _set(upd, defn["sweep_key"], x))
        for m in methods:
            keep = m == "JORC" and cfg["system"]["n_users"] == 100 and cfg["system"]["n_uavs"] == 2
            raw[m][x] = [run_point(m, cfg, s, runs_dir, keep_checkpoint=keep) for s in seeds]
    xlabel = "Number of UDs" if defn["sweep_key"].endswith("n_users") else "Number of UAVs"
    out = {}
    for fig, metric in defn["metrics"].items():
        series, table = {}, {}
        for m in methods:
            agg = [aggregate(raw[m][x], metric) for x in xs]
            series[m] = ([a[0] for a in agg], [a[1] for a in agg])
            table[m] = {str(x): {"mean": a[0], "ci95": a[1], "per_seed": a[2]} for x, a in zip(xs, agg)}
        out[fig] = {"metric": metric, "x": xs, "xlabel": xlabel, "values": table}
        if metric == "offloading_ratio":
            plot_bars(figs_dir / f"{fig}.png", xs, series, xlabel, YLABEL[metric])
        else:
            plot_lines(figs_dir / f"{fig}.png", xs, series, xlabel, YLABEL[metric])
    save_json(out, figs_dir / f"{name}.json")
    return raw


def fig3(defn: dict, base: dict, runs_dir: Path, figs_dir: Path):
    cfg = merged(base, defn["point"])
    seeds = base["training"]["seeds"]
    curves = {}
    for m in defn["methods"]:
        res = [run_point(m, cfg, s, runs_dir, keep_checkpoint=(m == "JORC")) for s in seeds]
        arr = np.array([r["history"]["episode_return"] for r in res])
        curves[m] = {"mean": arr.mean(0).tolist(), "per_seed": arr.tolist()}
    save_json({"x": "episode", "y": "episode_return", "curves": curves, "config_point": point_tag(cfg)},
              figs_dir / "fig3.json")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"JORC": "red", "PPO with caching": "green", "DDPG with caching": "blue", "A3C with caching": "purple"}
    label = {"JORC": "SAC", "PPO with caching": "PPO", "DDPG with caching": "DDPG", "A3C with caching": "A3C"}
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for m, c in curves.items():
        ax.plot(np.arange(len(c["mean"])), c["mean"], color=colors[m], lw=0.8, label=label[m])
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Rewards")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "fig3.png", dpi=150)
    plt.close(fig)


def _load_jorc(cfg: dict, seed: int, runs_dir: Path):
    res = run_point("JORC", cfg, seed, runs_dir, keep_checkpoint=True)
    if "checkpoint" not in res:          # result existed without checkpoint -> retrain with checkpoint
        (runs_dir / SLUG["JORC"] / f"{point_tag(cfg)}_seed{seed}.json").unlink()
        res = run_point("JORC", cfg, seed, runs_dir, keep_checkpoint=True)
    import torch
    env = UAVMECEnv(method_config(cfg, "JORC"), seed)
    agent = make_agent("sac", env.obs_dim, env.action_dim, cfg, np.random.default_rng(seed))
    agent.load_state_dict(torch.load(ROOT / res["checkpoint"], weights_only=True))
    return agent


def cache_capacity(defn: dict, base: dict, runs_dir: Path, figs_dir: Path, name: str):
    """Figs. 14/15: the trained JORC policy (N=100, M=2) evaluated with each replacement policy/capacity.
    Only the cache module differs between curves, as in the paper's comparison of replacement policies."""
    cfg = merged(base, defn["point"])
    seeds = base["training"]["seeds"]
    agents = {s: _load_jorc(cfg, s, runs_dir) for s in seeds}
    key = defn["sweep_key"].split(".")[1]
    xs = defn["sweep_values_gb"]
    table, series = {}, {}
    for label, pol in defn["policies"].items():
        means, cis, table[label] = [], [], {}
        for x in xs:
            vals = [evaluate(agents[s], cfg, "JORC", s, cache_overrides={"policy": pol, key: x * GB})[defn["metric"]]
                    for s in seeds]
            m, h = mean_ci95(vals)
            means.append(m)
            cis.append(h)
            table[label][str(x)] = {"mean": m, "ci95": h, "per_seed": vals}
        series[label] = (means, cis)
        print(f"[{name}] {label}: " + " ".join(f"{v:.3f}" for v in means), flush=True)
    xlabel = "UAV caching capacity (GB)" if key == "uav_capacity_bytes" else "BS caching capacity (GB)"
    plot_lines(figs_dir / f"{name}.png", xs, series, xlabel, YLABEL[defn["metric"]], logx=True)
    save_json({"metric": defn["metric"], "x_gb": xs, "values": table}, figs_dir / f"{name}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", default="smoke", choices=["smoke", "reduced", "paper"])
    ap.add_argument("--figures", nargs="*", default=["fig3", "fig4_7_12", "fig8_11_13", "fig14", "fig15"])
    ap.add_argument("--methods", nargs="*", default=None, help="subset of methods (paper labels)")
    ap.add_argument("--set", nargs="*", default=[], help="extra config overrides a.b=value")
    args = ap.parse_args()
    base = base_config(args.scale, args.set)
    out = ROOT / "results" / args.scale
    runs_dir, figs_dir = out / "runs", out / "figures"
    save_json(base, out / "base_config.json")
    defs = load_yaml("figures.yaml")
    for f in args.figures:
        print(f"=== {f}: {defs[f]['title']} ===", flush=True)
        if f == "fig3":
            fig3(defs[f], base, runs_dir, figs_dir)
        elif f in ("fig4_7_12", "fig8_11_13"):
            sweep(defs[f], base, runs_dir, figs_dir, f, args.methods)
        else:
            cache_capacity(defs[f], base, runs_dir, figs_dir, f)


if __name__ == "__main__":
    main()
