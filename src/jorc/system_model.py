"""Network model, Sec. 3.1 / 6.1: one BS at (0,0), M UAVs at fixed altitude H, N ground UDs.

Paper facts: BS at (0,0); UAVs at H = 100 m; area 2 km x 2 km; each user served by one UAV within its
communication range (d <= H tan theta); positions refreshed every time step (quasi-static within a slot).
Missing: UAV horizontal positions, theta, user->UAV association rule, the spatial user distribution.
"""
from __future__ import annotations

import numpy as np

from .channel_model import coverage_radius


class Geometry:
    def __init__(self, cfg_sys: dict, rng: np.random.Generator):
        self.cfg = cfg_sys
        self.n_users = int(cfg_sys["n_users"])
        self.n_uavs = int(cfg_sys["n_uavs"])
        self.H = float(cfg_sys["uav_altitude_m"])
        self.side = float(cfg_sys["area_side_m"])
        self.bs_xy = np.asarray(cfg_sys["bs_position_m"], float)
        lo = -self.side / 2 if cfg_sys["area_centered_on_bs"] else 0.0
        self.bounds = (lo, lo + self.side)
        self.r_cov = coverage_radius(self.H, cfg_sys["beamwidth_deg"])
        self.uav_xy = self._place_uavs(rng)
        if cfg_sys["user_association"] == "round_robin":
            self.uav_of_user = np.arange(self.n_users) % self.n_uavs
        elif cfg_sys["user_association"] == "random":
            self.uav_of_user = rng.integers(0, self.n_uavs, self.n_users)
        else:
            raise ValueError(cfg_sys["user_association"])
        self.user_xy = np.zeros((self.n_users, 2))
        self.resample_users(rng)

    def _place_uavs(self, rng):
        m, lo, hi = self.n_uavs, *self.bounds
        if self.cfg["uav_layout"] == "ring":
            ang = 2 * np.pi * np.arange(m) / m
            xy = self.bs_xy + self.cfg["uav_ring_radius_m"] * np.stack([np.cos(ang), np.sin(ang)], 1)
        elif self.cfg["uav_layout"] == "random":
            xy = rng.uniform(lo + self.r_cov, hi - self.r_cov, size=(m, 2))
        else:
            raise ValueError(self.cfg["uav_layout"])
        assert np.all((xy >= lo) & (xy <= hi)), "UAVs must stay inside the area / BS coverage"
        return xy

    def resample_users(self, rng):
        """Quasi-static positions for one slot."""
        n = self.n_users
        if self.cfg["user_placement"] == "coverage_disk":
            r = self.r_cov * np.sqrt(rng.random(n))          # uniform in disk
            phi = 2 * np.pi * rng.random(n)
            xy = self.uav_xy[self.uav_of_user] + np.stack([r * np.cos(phi), r * np.sin(phi)], 1)
            self.user_xy = np.clip(xy, *self.bounds)
        elif self.cfg["user_placement"] == "uniform_area":
            self.user_xy = rng.uniform(*self.bounds, size=(n, 2))
            # nearest-UAV association replaces the static association in this mode
            d = np.linalg.norm(self.user_xy[:, None, :] - self.uav_xy[None], axis=2)
            self.uav_of_user = d.argmin(1)
        else:
            raise ValueError(self.cfg["user_placement"])

    def horizontal_distance(self) -> np.ndarray:
        """d_{n,m} to each user's assigned UAV."""
        return np.linalg.norm(self.user_xy - self.uav_xy[self.uav_of_user], axis=1)

    def in_coverage(self) -> np.ndarray:
        return self.horizontal_distance() <= self.r_cov + 1e-9
