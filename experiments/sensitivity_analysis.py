"""Sensitivity of results to reconstructed parameters (configs/sensitivity.yaml).

    python experiments/sensitivity_analysis.py                       # training-free diagnostics only (fast)
    python experiments/sensitivity_analysis.py --train --scale smoke # + JORC vs SAC-without-caching per variant

Part 1 (always): for every one-at-a-time variant, the training-free reference policies of
src/jorc/diagnostics.py are evaluated (latency, energy, cost, STCR, hit ratios). This shows how strongly a
reconstructed parameter moves the physical operating point independently of any learning.
Part 2 (--train): JORC and SAC without caching are trained per variant; the table reports whether the paper's
qualitative conclusion "caching lowers average cost" survives the variant.
Part 3 (always): a capacity bound on the mean execution time per slot vs N (explains the latency scale).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, base_config, load_yaml, run_point  # noqa: E402

from jorc.diagnostics import run_fixed_policy  # noqa: E402
from jorc.utils import load_config, mean_ci95, save_json, set_by_path  # noqa: E402

COLS = ("avg_latency_s", "avg_energy_j", "avg_cost", "stcr", "cache_hit_ratio_uav", "cache_hit_ratio_bs")


def variants():
    yield "default", []
    for name, lists in load_yaml("sensitivity.yaml")["variations"].items():
        for ov in lists:
            yield f"{name}: {', '.join(o.split('=', 1)[1] for o in ov)}", ov


def capacity_bound(cfg: dict) -> list[str]:
    """Mean cycles per slot divided by total CPU capacity of UDs + UAVs + BS, ignoring caching/transmission.
    A slot in which every UD issues one task cannot finish its work faster than this on average."""
    t, c = cfg["task"], cfg["computation"]
    mean_cycles = np.mean(t["din_bits"]) * np.mean(t["cycles_per_bit"])
    rows = ["| N | M | mean demand (Gcycles) | total capacity (GHz) | bound (s) |", "|---|---|---|---|---|"]
    for n, m in [(20, 2), (60, 2), (100, 2), (120, 2), (100, 1), (100, 5), (100, 8)]:
        cap = n * np.mean(c["ud_freq_hz"]) + m * c["uav_freq_hz"] + c["bs_freq_hz"]
        rows.append(f"| {n} | {m} | {n * mean_cycles / 1e9:.0f} | {cap / 1e9:.1f} | {n * mean_cycles / cap:.2f} |")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--scale", default="smoke", choices=["smoke", "reduced", "paper"])
    ap.add_argument("--steps", type=int, default=200, help="steps for the training-free diagnostics")
    args = ap.parse_args()
    out = ROOT / "results" / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Sensitivity analysis\n", "## Part 3: capacity bound (paper parameters)\n", *capacity_bound(load_config()),
          "\n## Part 1: training-free reference policies (N=100, M=2, seed 0)\n",
          "| Variant | Policy | " + " | ".join(COLS) + " |", "|---|---|" + "---|" * len(COLS)]
    raw = {}
    for name, ov in variants():
        cfg = load_config(overrides=ov)
        for pol in ("all_local", "all_uav_equal", "all_bs_equal"):
            r = run_fixed_policy(cfg, pol, 0, 1, args.steps)
            raw[f"{name} | {pol}"] = r
            md.append(f"| {name} | {pol} | " + " | ".join(f"{r[c]:.3g}" for c in COLS) + " |")
        print(f"[diag] {name}", flush=True)
    if args.train:
        md += ["\n## Part 2: does 'caching lowers cost' survive? (scale: %s)\n" % args.scale,
               "| Variant | JORC cost | SAC w/o caching cost | JORC lower? |", "|---|---|---|---|"]
        for name, ov in variants():
            cfg = base_config(args.scale, ov)
            runs_dir = out / "runs" / name.replace(" ", "").replace(":", "_").replace(",", "_")
            j = [run_point("JORC", cfg, s, runs_dir)["eval"]["avg_cost"] for s in cfg["training"]["seeds"]]
            b = [run_point("SAC without caching", cfg, s, runs_dir)["eval"]["avg_cost"]
                 for s in cfg["training"]["seeds"]]
            (jm, jh), (bm, bh) = mean_ci95(j), mean_ci95(b)
            raw[f"{name} | train"] = {"jorc": j, "sac_nocache": b}
            md.append(f"| {name} | {jm:.3g} ± {jh:.2g} | {bm:.3g} ± {bh:.2g} | {'yes' if jm < bm else 'no'} |")
    (out / "sensitivity.md").write_text("\n".join(md) + "\n")
    save_json(raw, out / "sensitivity_raw.json")
    print(f"written {out / 'sensitivity.md'}")


if __name__ == "__main__":
    main()
