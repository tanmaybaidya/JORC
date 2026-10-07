"""Helpers shared by the experiment scripts: scales, one (method, config, seed) run, aggregation, plots."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jorc.baselines import SLUG  # noqa: E402
from jorc.training import evaluate, save_agent, train  # noqa: E402
from jorc.utils import load_config, mean_ci95, merged, save_json, yaml_load  # noqa: E402

GB = 1.0e9  # decimal gigabyte in bytes (assumption C-cache-units)


def load_yaml(name: str) -> dict:
    with open(ROOT / "configs" / name) as f:
        return yaml_load(f)


def base_config(scale: str, extra: list[str] | None = None) -> dict:
    scales = load_yaml("scales.yaml")
    return load_config(ROOT / "configs" / "paper_default.yaml", scales[scale]["overrides"] + (extra or []))


def point_tag(cfg: dict) -> str:
    return f"N{cfg['system']['n_users']}_M{cfg['system']['n_uavs']}"


def run_point(method: str, cfg: dict, seed: int, out_dir: Path, *, keep_checkpoint: bool = False,
              verbose: bool = True) -> dict:
    """Train + evaluate one (method, configuration, seed); cached on disk as JSON (raw results)."""
    path = out_dir / SLUG[method] / f"{point_tag(cfg)}_seed{seed}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    if verbose:
        print(f"[run] {method} {point_tag(cfg)} seed={seed}", flush=True)
    t0 = time.time()
    agent, hist = train(method, cfg, seed, verbose=False)
    t_train = time.time() - t0
    ev = evaluate(agent, cfg, method, seed)
    res = {"method": method, "seed": seed, "config": cfg, "history": hist, "eval": ev,
           "train_seconds": t_train}
    if keep_checkpoint:
        ckpt = path.with_suffix(".pt")
        save_agent(agent, ckpt)
        res["checkpoint"] = str(ckpt.relative_to(ROOT))
    save_json(res, path)
    if verbose:
        print(f"      cost={ev['avg_cost']:.4f} L={ev['avg_latency_s']:.3f}s E={ev['avg_energy_j']:.3f}J "
              f"STCR={ev['stcr']:.3f} off={ev['offloading_ratio']:.3f} ({t_train:.0f}s)", flush=True)
    return res


def aggregate(results: list[dict], metric: str) -> tuple[float, float, list[float]]:
    vals = [r["eval"][metric] for r in results]
    m, h = mean_ci95(vals)
    return m, h, vals


STYLE = {  # mirrors the paper's legends (colour/marker)
    "JORC": ("red", "x"), "PPO with caching": ("green", "o"), "DDPG with caching": ("blue", "^"),
    "A3C with caching": ("purple", "+"), "SAC without caching": ("darkred", "*"),
    "Hybrid LRU-LFU (JORC)": ("red", "o"), "LFU": ("green", "s"), "LRU": ("blue", "^"), "Random": ("purple", "x"),
}


def plot_lines(path: Path, x, series: dict, xlabel: str, ylabel: str, logx: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for name, (mean, ci) in series.items():
        c, mk = STYLE.get(name, (None, "o"))
        ax.errorbar(x, mean, yerr=ci, color=c, marker=mk, capsize=3, linewidth=1, label=name)
    if logx:
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bars(path: Path, x, series: dict, xlabel: str, ylabel: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    k = len(series)
    w = 0.8 / k
    pos = np.arange(len(x))
    for i, (name, (mean, ci)) in enumerate(series.items()):
        ax.bar(pos + (i - (k - 1) / 2) * w, mean, w, yerr=ci, capsize=2, color=STYLE.get(name, (None,))[0],
               label=name)
    ax.set_xticks(pos)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


__all__ = ["ROOT", "GB", "load_yaml", "base_config", "run_point", "aggregate", "plot_lines", "plot_bars",
           "merged", "save_json", "evaluate", "point_tag"]
