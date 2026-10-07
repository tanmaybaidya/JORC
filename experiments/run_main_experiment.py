"""Train and evaluate one method at one configuration for one or more seeds (the paper's default point:
N = 100 UDs, M = 2 UAVs).

    python experiments/run_main_experiment.py --method JORC --scale paper --seeds 0
    python experiments/run_main_experiment.py --method "SAC without caching" --scale smoke --set system.n_users=20

Writes raw JSON (config, training history, evaluation metrics) to results/<scale>/runs/<method>/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, base_config, run_point  # noqa: E402

from jorc.baselines import METHODS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="JORC", choices=list(METHODS))
    ap.add_argument("--scale", default="smoke", choices=["smoke", "reduced", "paper"])
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides a.b=value")
    ap.add_argument("--checkpoint", action="store_true")
    args = ap.parse_args()
    cfg = base_config(args.scale, args.set)
    seeds = args.seeds if args.seeds is not None else cfg["training"]["seeds"]
    for s in seeds:
        run_point(args.method, cfg, s, ROOT / "results" / args.scale / "runs", keep_checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
