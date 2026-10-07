"""Communication model, Sec. 3.2 of the paper.

eq.(1)  h_{n,m} = h0 / (H^2 + d_{n,m}^2)                      (LoS, free-space-like, exponent 2)
eq.(2)  R_{n,m} = B_n log2(1 + P_n h_{n,m} / (B_n N0))        (N0 = noise power spectral density, W/Hz)
coverage: d_{n,m} <= H tan(theta)
"""
from __future__ import annotations

import numpy as np


def channel_gain(h0_linear: float, altitude_m: float, horiz_dist_m):
    """eq.(1). Returns the dimensionless power gain. h0_linear is the gain at 1 m (e.g. 1e-5 for -50 dB)."""
    d = np.asarray(horiz_dist_m, dtype=float)
    assert np.all(d >= 0)
    return h0_linear / (altitude_m**2 + d**2)


def coverage_radius(altitude_m: float, beamwidth_deg: float) -> float:
    """Service radius H tan(theta) (Sec. 3.2)."""
    return altitude_m * np.tan(np.deg2rad(beamwidth_deg))


def shannon_rate(bandwidth_hz, power_w, gain, noise_psd_w_per_hz):
    """eq.(2): B log2(1 + P h / (B N0)) in bit/s. log base 2 as written in the paper."""
    b = np.asarray(bandwidth_hz, dtype=float)
    snr = np.asarray(power_w, dtype=float) * np.asarray(gain, dtype=float) / (b * noise_psd_w_per_hz)
    return b * np.log2(1.0 + snr)


def backhaul_rates(cfg_comm: dict, h0_linear: float, altitude_m: float, uav_xy: np.ndarray,
                   bs_xy: np.ndarray, noise_psd_w_per_hz: float) -> np.ndarray:
    """Backhaul capacity R_{m,b} per UAV.

    The paper treats R_{m,b} as a given constant per UAV (Sec. 3.2) without a value. Two options:
    'fixed'       -> cfg value for every UAV;
    'shannon_los' -> same LoS model as eq.(1)-(2) applied to the UAV->BS link with the UAV transmit
                     power P_{m,b} on a dedicated backhaul band (assumption, see REPRODUCTION_ASSUMPTIONS).
    The rate is NOT shared among tasks forwarded in the same slot (literal reading of eq.(6)).
    """
    m = uav_xy.shape[0]
    if cfg_comm["backhaul_mode"] == "fixed":
        return np.full(m, float(cfg_comm["backhaul_rate_bps"]))
    if cfg_comm["backhaul_mode"] == "shannon_los":
        d = np.linalg.norm(uav_xy - bs_xy[None, :], axis=1)
        g = channel_gain(h0_linear, altitude_m, d)
        return shannon_rate(cfg_comm["backhaul_bandwidth_hz"], cfg_comm["uav_tx_power_w"], g, noise_psd_w_per_hz)
    raise ValueError(cfg_comm["backhaul_mode"])
