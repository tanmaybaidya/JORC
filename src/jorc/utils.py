"""Configuration loading, unit conversions and seeding helpers."""
from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import re

import yaml


class _Loader(yaml.SafeLoader):
    """SafeLoader that also parses '1e6' / '20.0e6' as floats (PyYAML's YAML-1.1 resolver does not)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."),
)


def yaml_load(text_or_stream):
    return yaml.load(text_or_stream, Loader=_Loader)

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- units
MBIT = 1.0e6          # bits in one megabit (decimal, as used in MEC literature)
GHZ = 1.0e9
MHZ = 1.0e6
BITS_PER_BYTE = 8.0


def db_to_linear(db: float) -> float:
    """Power ratio in dB -> linear (10^(dB/10))."""
    return 10.0 ** (db / 10.0)


def dbm_to_watt(dbm: float) -> float:
    """Power in dBm -> W."""
    return 10.0 ** ((dbm - 30.0) / 10.0)


def bytes_to_bits(n_bytes: float) -> float:
    return n_bytes * BITS_PER_BYTE


# ---------------------------------------------------------------- config
def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Load YAML config (default: configs/paper_default.yaml) and apply 'a.b=value' overrides."""
    path = Path(path) if path else REPO_ROOT / "configs" / "paper_default.yaml"
    with open(path) as f:
        cfg = yaml_load(f)
    for ov in overrides or []:
        set_by_path(cfg, ov)
    validate_config(cfg)
    return cfg


def set_by_path(cfg: dict, assignment: str) -> None:
    key, _, raw = assignment.partition("=")
    if not _:
        raise ValueError(f"override must look like a.b=value, got {assignment!r}")
    node = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"unknown config key {key!r}")
    node[parts[-1]] = yaml_load(raw)


def merged(cfg: dict, updates: dict) -> dict:
    """Deep copy of cfg with nested dict `updates` applied."""
    out = copy.deepcopy(cfg)

    def rec(dst, src):
        for k, v in src.items():
            if isinstance(v, dict):
                if k not in dst:
                    raise KeyError(k)
                rec(dst[k], v)
            else:
                if k not in dst:
                    raise KeyError(k)
                dst[k] = v

    rec(out, updates)
    validate_config(out)
    return out


def validate_config(cfg: dict) -> None:
    """Unit / range sanity checks that catch common unit slips (Hz vs GHz, bits vs Mbit...)."""
    s, c, comp, t, ca = cfg["system"], cfg["communication"], cfg["computation"], cfg["task"], cfg["caching"]
    assert s["n_users"] >= 1 and s["n_uavs"] >= 1
    assert 0.0 < s["beamwidth_deg"] < 90.0, "beamwidth is the half-angle theta in degrees"
    assert c["bandwidth_hz"] > 1e5, "bandwidth must be given in Hz"
    assert -200 < c["noise_psd_dbm_per_hz"] < -100, "noise PSD must be in dBm/Hz"
    assert 0 < c["ud_pmin_w"] < c["ud_pmax_w"] < 10, "UD power must be in W"
    assert all(1e8 <= f <= 1e10 for f in comp["ud_freq_hz"]), "UD frequency must be in Hz"
    assert comp["uav_freq_hz"] > 1e9 and comp["bs_freq_hz"] > 1e9
    assert all(1e5 <= d <= 1e8 for d in t["din_bits"]), "task size must be in bits"
    assert all(1e4 <= d <= 1e8 for d in t["dout_bits"]), "result size must be in bits"
    assert ca["uav_capacity_bytes"] > 1e6 and ca["bs_capacity_bytes"] > 1e6, "cache capacity in bytes"
    assert abs(cfg["cost"]["gamma_latency"] + cfg["cost"]["gamma_energy"] - 1.0) < 1e-9
    assert 0.0 <= ca["delta_min"] <= ca["delta0"] <= ca["delta_max"] <= 1.0


# ---------------------------------------------------------------- seeding
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:                       # atomic write: interrupted runs never leave bad JSON
        json.dump(obj, f, indent=2, default=_json_default)
    tmp.replace(path)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def mean_ci95(values) -> tuple[float, float]:
    """Mean and half-width of a two-sided 95% t-interval over independent seeds."""
    from scipy import stats

    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return float(v.mean()) if v.size else math.nan, 0.0
    half = stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / math.sqrt(v.size)
    return float(v.mean()), float(half)
