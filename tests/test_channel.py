"""Eq.(1), eq.(2), coverage radius and unit conversions."""
import math

import numpy as np

from jorc.channel_model import channel_gain, coverage_radius, shannon_rate
from jorc.utils import db_to_linear, dbm_to_watt, load_config


def test_units():
    assert math.isclose(db_to_linear(-50), 1e-5)
    assert math.isclose(dbm_to_watt(23), 0.19952623, rel_tol=1e-6)
    assert math.isclose(dbm_to_watt(-174), 3.981e-21, rel_tol=1e-3)


def test_yaml_scientific_notation_is_float():
    cfg = load_config()
    assert isinstance(cfg["communication"]["bandwidth_hz"], float)
    assert cfg["communication"]["bandwidth_hz"] == 20e6
    assert cfg["task"]["din_bits"] == [2e6, 10e6]


def test_channel_gain_eq1():
    h = channel_gain(1e-5, 100.0, np.array([0.0, 100.0]))
    assert np.allclose(h, [1e-5 / 1e4, 1e-5 / 2e4])
    assert h[0] > h[1]


def test_coverage_radius():
    assert math.isclose(coverage_radius(100.0, 45.0), 100.0, rel_tol=1e-12)


def test_shannon_rate_log2_and_hz():
    # SNR = 1 -> rate = B * log2(2) = B
    assert math.isclose(float(shannon_rate(1e6, 1.0, 1e-6, 1e-12)), 1e6)
    # hand check of a paper-scale link: 0.2 W, d = 0, B = 400 kHz, -174 dBm/Hz
    snr = 0.2 * 1e-9 / (4e5 * dbm_to_watt(-174))
    assert math.isclose(float(shannon_rate(4e5, 0.2, 1e-9, dbm_to_watt(-174))), 4e5 * math.log2(1 + snr))
