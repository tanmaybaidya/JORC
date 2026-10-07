"""Tables.

    python experiments/reproduce_tables.py --scale smoke

Produces in results/<scale>/tables/:
  parameter_provenance.md   Table 3 of the paper + every reconstructed parameter with its provenance tag
  comparison.md             values stated in the paper text vs. reproduced values (same points)
  wilcoxon.md               paired Wilcoxon signed-rank tests JORC vs each baseline over seeds (Sec. 6.1)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_yaml  # noqa: E402

from jorc.baselines import METHODS, SLUG  # noqa: E402
from jorc.utils import mean_ci95  # noqa: E402

TAG = re.compile(r"^\s*(\w+):\s*(.+?)\s*#\s*\[([A-E](?:/[A-E])?)\]\s*(.*)$")


def provenance_table() -> str:
    rows, section = [], None
    for line in open(ROOT / "configs" / "paper_default.yaml"):
        if re.match(r"^\w+:\s*$", line):
            section = line.split(":")[0]
            continue
        m = TAG.match(line)
        if m:
            key, val, tag, note = m.groups()
            rows.append(f"| {section}.{key} | `{val}` | {tag} | {note} |")
    hdr = ("| Parameter | Value | Provenance | Note |\n|---|---|---|---|\n")
    legend = ("\nA = explicit in paper, B = clearly implied, C = missing/ambiguous, D = external literature, "
              "E = engineering assumption. Details: REPRODUCTION_ASSUMPTIONS.md\n")
    return "# Parameter provenance (generated from configs/paper_default.yaml)\n\n" + hdr + "\n".join(rows) + legend


def load_runs(scale: str) -> dict:
    """{method: {(N, M): [eval dicts over seeds]}} (only default cache configuration)."""
    out = {}
    for m in METHODS:
        d = ROOT / "results" / scale / "runs" / SLUG[m]
        for f in sorted(d.glob("*.json")) if d.exists() else []:
            r = json.load(open(f))
            key = (r["config"]["system"]["n_users"], r["config"]["system"]["n_uavs"])
            out.setdefault(m, {}).setdefault(key, []).append((r["seed"], r["eval"]))
    return out


def comparison_table(runs: dict, scale: str) -> str:
    paper = load_yaml("../data/paper_reported_values.yaml")
    lines = ["# Paper-stated values vs. reproduction (scale: %s)\n" % scale,
             "Reproduced: mean ± 95% CI over available seeds. '—' = point not run.\n",
             "| Sweep | Metric | Method | x | Paper | Reproduced |", "|---|---|---|---|---|---|"]
    for sweep, fixed_key in (("vs_users", "M2"), ("vs_uavs", "N100")):
        for metric, per_m in paper[sweep].items():
            for method, pts in per_m.items():
                for x, pv in pts.items():
                    key = (x, 2) if sweep == "vs_users" else (100, x)
                    if method.startswith("DRL"):
                        vals = [ev[metric] for mm in ("PPO with caching", "DDPG with caching", "A3C with caching")
                                for _, ev in runs.get(mm, {}).get(key, [])]
                        rep = f"[{min(vals):.3g}, {max(vals):.3g}] (seed range)" if vals else "—"
                    else:
                        vals = [ev[metric] for _, ev in runs.get(method, {}).get(key, [])]
                        rep = "%.3g ± %.2g (n=%d)" % (*mean_ci95(vals), len(vals)) if vals else "—"
                    lines.append(f"| {sweep} | {metric} | {method} | {x} | {pv} | {rep} |")
    for fig in ("fig14", "fig15"):
        p = ROOT / "results" / scale / "figures" / f"{fig}.json"
        key = "cache_hit_ratio_uav" if fig == "fig14" else "cache_hit_ratio_bs"
        rep = json.load(open(p))["values"] if p.exists() else {}
        for pol, pts in paper[key].items():
            for x, pv in pts.items():
                r = rep.get(pol, {}).get(str(x))
                lines.append(f"| {fig} | {key} | {pol} | {x} GB | {pv} | "
                             f"{'%.3g ± %.2g' % (r['mean'], r['ci95']) if r else '—'} |")
    return "\n".join(lines) + "\n"


def wilcoxon_table(runs: dict) -> str:
    metrics = ("avg_latency_s", "avg_energy_j", "avg_cost", "stcr")
    lines = ["# Paired Wilcoxon signed-rank tests, JORC vs baselines (two-sided)\n",
             "With n seeds the smallest attainable two-sided p is 2/2^n (n=10 -> 0.002; n=2 -> 0.5).\n",
             "| (N, M) | Baseline | Metric | n | JORC mean | Baseline mean | p |", "|---|---|---|---|---|---|---|"]
    for key, jres in sorted(runs.get("JORC", {}).items()):
        js = dict(jres)
        for b in METHODS:
            if b == "JORC" or key not in runs.get(b, {}):
                continue
            bs = dict(runs[b][key])
            common = sorted(set(js) & set(bs))
            for met in metrics:
                a = np.array([js[s][met] for s in common])
                c = np.array([bs[s][met] for s in common])
                p = wilcoxon(a, c).pvalue if len(common) >= 2 and np.any(a != c) else float("nan")
                lines.append(f"| {key} | {b} | {met} | {len(common)} | {a.mean():.4g} | {c.mean():.4g} | {p:.3g} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", default="smoke", choices=["smoke", "reduced", "paper"])
    args = ap.parse_args()
    out = ROOT / "results" / args.scale / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "parameter_provenance.md").write_text(provenance_table())
    runs = load_runs(args.scale)
    (out / "comparison.md").write_text(comparison_table(runs, args.scale))
    (out / "wilcoxon.md").write_text(wilcoxon_table(runs))
    print(f"tables written to {out}")


if __name__ == "__main__":
    main()
