"""Training-free check of the cache module at the paper's request volume (Figs. 14/15 setting).

    python experiments/cache_capacity_diagnostic.py [--steps 1000] [--units GB|MB] [--seeds 0 1]

A FIXED policy (every cache miss offloaded to its UAV, or to the BS for the BS figure) generates the request
stream of N = 100 UDs, M = 2 UAVs for one 1000-slot episode, so only the replacement policy and capacity
differ between curves. This isolates the caching mechanism from SAC training. It is NOT the paper's
experiment (which uses the trained JORC policy); it answers "can capacity/policy matter at all here?".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_yaml, plot_lines  # noqa: E402

from jorc.diagnostics import run_fixed_policy  # noqa: E402
from jorc.utils import load_config, mean_ci95, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--units", default="GB", choices=["GB", "MB"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--figures", nargs="*", default=["fig14", "fig15"])
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    unit = 1e9 if args.units == "GB" else 1e6
    defs = load_yaml("figures.yaml")
    out = ROOT / "results" / "cache_diagnostic"
    res = {}
    for fig in args.figures:
        d = defs[fig]
        key = d["sweep_key"].split(".")[1]
        pol_fixed = "all_uav_equal" if fig == "fig14" else "all_bs_equal"
        series, res[fig] = {}, {}
        for label, pol in d["policies"].items():
            ms, hs = [], []
            for x in d["sweep_values_gb"]:
                cfg = load_config(overrides=[f"caching.policy={pol}", f"caching.{key}={x * unit}"] + args.set)
                vals = [run_fixed_policy(cfg, pol_fixed, s, 1, args.steps)[d["metric"]] for s in args.seeds]
                m, h = mean_ci95(vals)
                ms.append(m)
                hs.append(h)
            series[label] = (ms, hs)
            res[fig][label] = dict(zip(map(str, d["sweep_values_gb"]), ms))
            print(f"[{fig} {args.units}] {label:22s} " + " ".join(f"{v:.3f}" for v in ms), flush=True)
        plot_lines(out / f"{fig}_{args.units}.png", d["sweep_values_gb"], series,
                   f"{'UAV' if fig == 'fig14' else 'BS'} caching capacity ({args.units})", d["metric"], logx=True)
    save_json(res, out / f"cache_diagnostic_{args.units}_{args.steps}.json")


if __name__ == "__main__":
    main()
