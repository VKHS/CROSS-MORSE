"""One label-free A9 anchor contract shared by logging and execution.

Observer validity still controls whether a policy action may be chosen. It does
not redefine the recovery reference learned from exploration trajectories.
The entropy threshold is an execution convention, not a predictive guarantee.
"""
from __future__ import annotations

import numpy as np

ANCHOR_RULE = "finite_measurements_entropy_le_0.99_v2"


def label_free_safe(measurements) -> bool:
    y = np.asarray(measurements)
    return bool(y.ndim == 1 and len(y) > 0 and np.isfinite(y).all()
                and float(y[0]) <= 0.99)
