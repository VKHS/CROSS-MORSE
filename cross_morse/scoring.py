"""Shared numeric policy scores; no labels, fitting or model calls."""
from __future__ import annotations

import math
import numpy as np

SCORE_VERSION = "mean_kappa_sigma_radius_transport_v2"


def calibration_radius(calibration: dict, stratum: int, action: int) -> float:
    if action == 0:
        return 0.0  # Nominal-relative target is exactly zero.
    entry = calibration.get("cells", {}).get(f"{stratum}:{action}")
    if entry is None:
        entry = calibration.get("actions", {}).get(str(action))
    value = entry.get("radius") if entry is not None else None
    return float(value) if value is not None else math.inf


def action_scores(means, sigmas, radii, transport_mass, *, kind="radius",
                  kappa=1.0, lambda_rho=0.0001):
    """Vectorized Eq. (16) arithmetic used by runtime and held-out assessment.

    Support, observer validity and deterministic ties are applied by the caller.
    Nonfinite scores are never viable candidates. No operational penalty is
    invented: the current implementation declares it to be exactly zero.
    """
    if kind not in ("radius", "one", "one-step", "one_step"):
        raise ValueError("kind must be radius or one-step")
    if not (np.isfinite(kappa) and kappa >= 0 and np.isfinite(lambda_rho) and lambda_rho >= 0):
        raise ValueError("Score coefficients must be finite and nonnegative")
    mean = np.asarray(means, dtype=np.float64)
    if kind != "radius":
        return np.where(np.isfinite(mean), mean, np.inf)
    score = mean + float(kappa) * np.asarray(sigmas) + np.asarray(radii)
    score = score + float(lambda_rho) * np.asarray(transport_mass)
    return np.where(np.isfinite(score), score, np.inf)
