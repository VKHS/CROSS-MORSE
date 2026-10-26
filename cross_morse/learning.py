"""Cross-fitted observer, action critic and sequence consequence models.

All fit/calibration rows must come from Dexplore. Trajectories (including their
seed identity) are held out together. Confidence radii are explicitly empirical:
trajectory holdouts and logged-state diagnostics are not an independent-graph coverage theorem.
Unknown support cells are rejected, never repaired by optimistic scores.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from .core import ACTIONS, site_transport_accounting
from .scoring import action_scores, calibration_radius, SCORE_VERSION


class TrainingInterrupted(RuntimeError):
    """A resumable epoch/fold checkpoint was saved before stopping."""


def _setting(config, section, name, default):
    return config.get(section, {}).get(name, default)


def _atomic_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _write_json(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _groups(records):
    seed = np.asarray(records.get("seed_index", np.zeros(len(records["trajectory"]))), dtype=np.int64)
    pairs = np.column_stack((seed, np.asarray(records["trajectory"], dtype=np.int64)))
    return np.unique(pairs, axis=0, return_inverse=True)[1]


def _holdout_groups(records):
    """One complete independent graph stays in one development role."""
    if "graph_id" in records:
        graph = np.asarray(records["graph_id"], dtype=np.int64)
        if np.any(graph < 0):
            raise ValueError("Independent graph identifiers must be nonnegative")
        return np.unique(graph, return_inverse=True)[1]
    return _groups(records)


def _fingerprint(records, config):
    h = hashlib.sha256(b"cross-morse-learning-trajectory-holdout-v2\0" +
                       json.dumps(config, sort_keys=True, default=str).encode())
    for key in sorted(records):
        value = np.ascontiguousarray(records[key])
        h.update(key.encode())
        h.update(str((value.shape, value.dtype.str)).encode())
        h.update(memoryview(value).cast("B"))
    return h.hexdigest()


def _profile_y(y, profile):
    value = np.asarray(y, dtype=np.float64).copy()
    if profile.lower() == "lean":
        value[..., 9:] = 0.0
    return value


def _low_rank_diagonal(cov, rank, floor=1e-5):
    """PSD rank-r plus nonnegative diagonal, represented as a small dense matrix."""
    cov = np.nan_to_num((cov + cov.T) / 2.0, nan=0.0, posinf=1e6, neginf=-1e6)
    vals, vectors = np.linalg.eigh(cov)
    chosen = np.argsort(vals)[-min(rank, len(vals)):]
    low = (vectors[:, chosen] * np.maximum(vals[chosen], 0.0)) @ vectors[:, chosen].T
    diagonal = np.maximum(np.diag(cov - low), floor)
    return low + np.diag(diagonal)


def _quantile(values, level):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return float(np.quantile(values, min(1.0, max(0.0, level)), method="higher"))


@dataclass
class ControllerState:
    z: np.ndarray
    cov: np.ndarray
    nis: float
    bias: float
    whiteness: float
    y: np.ndarray
    depth: int
    previous_action: int
    valid: bool
    innovation: np.ndarray | None = None


class ConditionedObserver:
    """Robust PCA observation, conditioned ridge dynamics, low-rank Kalman filter.

    Reconstruction choices: median/MAD normalization; latent PCA fitted only on
    fitting trajectories; depth/action ridge regression with exogenous normalized
    depth/mass/hops; rank-r+diagonal covariances. The normalized measurement
    projected into PCA space has identity observation matrix. The equivalent
    raw-normalized observation map is the stored PCA basis.
    """
    def __init__(self, payload):
        self.payload = payload
        for name in ("median", "scale", "center", "basis", "global_coef", "q", "r"):
            setattr(self, name, np.asarray(payload[name], dtype=np.float64))
        self.conditioned = {k: np.asarray(v, dtype=np.float64) for k, v in payload["conditioned"].items()}
        self.dimension = self.basis.shape[1]
        self.rank = int(payload["rank"])
        self.max_depth = int(payload["max_depth"])
        self.depth_bins = int(payload["depth_bins"])
        self.period = int(payload["period"])
        self.profile = payload["profile"]
        self.nis_threshold = float(payload.get("nis_threshold", 1e12))

    def normalize(self, y):
        value = _profile_y(y, self.profile)
        value = np.where(np.isfinite(value), value, self.median)
        return np.clip((value - self.median) / self.scale, -5, 5)

    def transform(self, y):
        return (self.normalize(y) - self.center) @ self.basis

    def reconstruct(self, z):
        normalized = np.asarray(z) @ self.basis.T + self.center
        raw = normalized * self.scale + self.median
        return _profile_y(raw, self.profile)

    def bin(self, depth):
        return min(self.depth_bins - 1, max(0, int(depth) * self.depth_bins // self.max_depth))

    def coefficient(self, depth, action):
        return self.conditioned.get(f"{self.bin(depth)}:{int(action)}", self.global_coef)

    def transition(self, state, action):
        coef = self.coefficient(state.depth, action)
        u = self.normalize(state.y)[5:8]
        z = np.r_[state.z, u, 1.0] @ coef
        f = coef[:self.dimension].T
        cov = _low_rank_diagonal(f @ state.cov @ f.T + self.q, self.rank)
        return z, cov

    def start(self, y, previous_action=-1, depth=0):
        raw = _profile_y(y, self.profile)
        finite = bool(np.isfinite(raw).all())
        z = self.transform(raw)
        return ControllerState(z, self.r.copy(), 0.0, 0.0, 0.0, raw,
                               int(depth), int(previous_action), finite,
                               np.zeros(self.dimension))

    def observe(self, state, y, depth, previous_action):
        raw = _profile_y(y, self.profile)
        finite = bool(np.isfinite(raw).all())
        if depth < state.depth:
            raise ValueError("Observer depth cannot move backwards")
        if depth > state.depth + 1:
            # Missing actions cannot be invented: reinitialize to the measured
            # state and mark invalid so the executable controller recovers.
            result = self.start(raw, previous_action, depth)
            result.valid = False
            return result
        if depth == state.depth + 1:
            prior_z, prior_cov = self.transition(state, previous_action)
        else:
            prior_z, prior_cov = state.z.copy(), state.cov.copy()
        if depth % self.period:
            return ControllerState(prior_z, prior_cov, state.nis, state.bias,
                                   state.whiteness, raw, int(depth), int(previous_action),
                                   finite and np.isfinite(prior_z).all() and state.valid, state.innovation)
        observation = self.transform(raw)
        error = observation - prior_z
        innovation_cov = prior_cov + self.r
        solved = np.linalg.solve(innovation_cov, error)
        nis = max(0.0, float(error @ solved))
        gain = np.linalg.solve(innovation_cov, prior_cov.T).T
        z = prior_z + gain @ error
        identity_minus_gain = np.eye(self.dimension) - gain
        # Joseph form preserves PSD before the explicit low-rank compression.
        covariance = (identity_minus_gain @ prior_cov @ identity_minus_gain.T
                      + gain @ self.r @ gain.T)
        covariance = _low_rank_diagonal(covariance, self.rank)
        standardized = error / np.sqrt(np.maximum(np.diag(innovation_cov), 1e-12))
        bias = float(np.mean(standardized))
        previous = state.innovation if state.innovation is not None else np.zeros_like(error)
        whiteness = float(error @ previous / max(np.linalg.norm(error) * np.linalg.norm(previous), 1e-12))
        valid = finite and np.isfinite(z).all() and np.isfinite(nis) and nis <= self.nis_threshold
        return ControllerState(z, covariance, nis, bias, whiteness, raw,
                               int(depth), int(previous_action), bool(valid), error)


def _fit_observer(records, indices, config):
    profile = config.get("active_profile", "full").lower()
    raw = _profile_y(np.asarray(records["y"])[indices], profile)
    finite_raw = np.where(np.isfinite(raw), raw, np.nan)
    median = np.nanmedian(finite_raw, axis=0)
    median = np.nan_to_num(median)
    raw = np.where(np.isfinite(raw), raw, median)
    scale = 1.4826 * np.median(np.abs(raw - median), axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    normalized = np.clip((raw - median) / scale, -5, 5)
    center = normalized.mean(0)
    covariance = (normalized - center).T @ (normalized - center) / max(len(raw) - 1, 1)
    _, vectors = np.linalg.eigh(covariance)
    dimension = min(int(_setting(config, "controller", "state_dim", 16)), raw.shape[1])
    basis = vectors[:, -dimension:][:, ::-1]
    # PCA sign is fixed for stable serialization and repeatable replay.
    for k in range(dimension):
        if basis[np.argmax(np.abs(basis[:, k])), k] < 0:
            basis[:, k] *= -1
    z = (normalized - center) @ basis
    next_raw = _profile_y(np.asarray(records["y_next"])[indices], profile)
    next_raw = np.where(np.isfinite(next_raw), next_raw, median)
    next_z = (np.clip((next_raw - median) / scale, -5, 5) - center) @ basis
    design = np.column_stack((z, normalized[:, 5:8], np.ones(len(z))))
    ridge = float(_setting(config, "controller", "observer_ridge", 1e-2))
    regularizer = np.eye(design.shape[1]) * ridge
    regularizer[-1, -1] = ridge * 0.01
    def fit_matrix(rows):
        x, y = design[rows], next_z[rows]
        return np.linalg.solve(x.T @ x + regularizer, x.T @ y)
    global_coef = fit_matrix(np.ones(len(z), dtype=bool))
    max_depth = int(_setting(config, "model", "depth", 256))
    depth_bins = int(_setting(config, "controller", "depth_bins", 4))
    depth = np.asarray(records["depth"])[indices]
    action = np.asarray(records["action"])[indices]
    bins = np.clip(depth * depth_bins // max_depth, 0, depth_bins - 1)
    conditioned = {}
    residual = next_z - design @ global_coef
    for b in range(depth_bins):
        for a in range(10):
            rows = (bins == b) & (action == a)
            if rows.sum() >= max(dimension + 4, 16):
                coef = fit_matrix(rows)
                conditioned[f"{b}:{a}"] = coef.tolist()
                residual[rows] = next_z[rows] - design[rows] @ coef
    q = residual.T @ residual / max(len(residual) - 1, 1)
    rank = int(_setting(config, "controller", "covariance_rank", 4))
    q = _low_rank_diagonal(q + np.eye(dimension) * 1e-4, rank)
    # Measurement covariance is identifiable only with a modeling convention:
    # 1/4 of transition-residual covariance, with a positive diagonal floor.
    r = _low_rank_diagonal(0.25 * q + np.eye(dimension) * 1e-3, rank)
    payload = {"median": median.tolist(), "scale": scale.tolist(),
               "center": center.tolist(), "basis": basis.tolist(),
               "global_coef": global_coef.tolist(), "conditioned": conditioned,
               "q": q.tolist(), "r": r.tolist(), "rank": rank,
               "max_depth": max_depth, "depth_bins": depth_bins,
               "period": int(_setting(config, "controller", "observer_period", 2)),
               "profile": profile, "nis_threshold": 1e12,
               "identification": "ridge PCA dynamics; R=0.25Q+1e-3I; covariance rank+diagonal"}
    return ConditionedObserver(payload)


def _features(state, max_depth):
    history = np.asarray(state.y)[5:8].copy()
    history[0] = state.depth / max_depth
    previous = np.zeros(10)
    if 0 <= state.previous_action < 10:
        previous[state.previous_action] = 1.0
    return np.concatenate((state.z, np.sqrt(np.maximum(np.diag(state.cov), 0.0)),
                           [math.log1p(min(state.nis, 1e8)), state.bias, state.whiteness],
                           history, previous)).astype(np.float32)


def _encode_records(observer, records, indices, *, return_valid=False):
    """Apply the observer separately to each trajectory, never across groups."""
    indices = np.asarray(indices, dtype=np.int64)
    groups = _groups(records)
    depths = np.asarray(records["depth"])
    order = np.lexsort((depths[indices], groups[indices]))
    features = np.empty((len(indices), 2 * observer.dimension + 16), dtype=np.float32)
    next_latent = np.empty((len(indices), observer.dimension), dtype=np.float32)
    current_latent = np.empty_like(next_latent)
    nis_values = np.empty(len(indices), dtype=np.float64)
    valid_values = np.empty(len(indices), dtype=bool)
    last_group = None
    state = None
    for position in order:
        i = indices[position]
        group, depth = int(groups[i]), int(depths[i])
        previous = int(np.asarray(records["previous_action"])[i])
        y = np.asarray(records["y"])[i]
        if group != last_group:
            state = observer.start(y, previous, depth)
        elif depth != state.depth or not np.array_equal(state.y, _profile_y(y, observer.profile)):
            state = observer.observe(state, y, depth, previous)
        features[position] = _features(state, observer.max_depth)
        current_latent[position] = state.z
        target_state = observer.observe(state, np.asarray(records["y_next"])[i],
                                        depth + 1, int(np.asarray(records["action"])[i]))
        next_latent[position] = target_state.z
        nis_values[position] = state.nis
        valid_values[position] = state.valid
        last_group = group
    result = (features, current_latent, next_latent, nis_values)
    return (*result, valid_values) if return_valid else result


class ActionCritic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.SiLU(), nn.Dropout(0.10),
                                 nn.Linear(128, 128), nn.SiLU(), nn.Dropout(0.10),
                                 nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 2))
    def forward(self, x):
        output = self.net(x)
        return output[:, :1], output[:, 1:].clamp(-7, 3)


class ResidualModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input = nn.Linear(input_dim, 128)
        self.residual = nn.Sequential(nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
        self.output = nn.Linear(128, output_dim)
    def forward(self, x):
        h = self.input(x)
        return self.output(torch.nn.functional.silu(h + self.residual(h)))


def _strata(y, depth, config):
    y = np.asarray(y)
    depth = np.asarray(depth)
    max_depth = int(_setting(config, "model", "depth", 256))
    bins = int(_setting(config, "controller", "depth_bins", 4))
    edges = np.asarray(_setting(config, "controller", "entropy_bins", [1 / 3, 2 / 3]))
    depth_bin = np.clip(depth.astype(np.int64) * bins // max_depth, 0, bins - 1)
    entropy_bin = np.searchsorted(edges, y[..., 0], side="right")
    return depth_bin * (len(edges) + 1) + entropy_bin


def _support(records, indices, config):
    indices = np.asarray(indices)
    strata = _strata(np.asarray(records["y"])[indices], np.asarray(records["depth"])[indices], config)
    groups = _holdout_groups(records)[indices]
    actions = np.asarray(records["action"])[indices].astype(int)
    propensity = np.asarray(records["propensity"])[indices].astype(float)
    if not np.all(np.isfinite(propensity) & (propensity > 0) & (propensity <= 1)):
        raise ValueError("Every logged propensity must be in (0,1]")
    p_min = float(_setting(config, "support", "p_min", 0.05))
    n_min = float(_setting(config, "support", "n_min", 64))
    omega = float(_setting(config, "support", "max_normalized_weight", 0.20))
    cells = {}
    for stratum in np.unique(strata):
        for action in range(10):
            selected = (strata == stratum) & (actions == action)
            rows = np.flatnonzero(selected)
            if len(rows):
                unique, inverse = np.unique(groups[rows], return_inverse=True)
                # One equal-role trajectory unit per cell. Average its inverse
                # propensities; repeated depths never increase the unit count.
                counts = np.bincount(inverse)
                weights = np.bincount(inverse, weights=1 / propensity[rows]) / counts
                normalized = weights / weights.sum()
                ess = float(1 / np.square(normalized).sum())
                max_weight = float(normalized.max())
                minimum = float(propensity[rows].min())
                count = len(unique)
            else:
                ess, max_weight, minimum, count = 0.0, 1.0, 0.0, 0
            qualified = minimum >= p_min - 1e-12 and ess >= n_min - 1e-9 and max_weight <= omega + 1e-12
            cells[f"{int(stratum)}:{action}"] = {
                "stratum": int(stratum), "action": action, "row_count": int(len(rows)),
                "trajectory_count": int(count), "min_propensity": minimum,
                "ess": ess, "max_normalized_weight": max_weight, "qualified": bool(qualified)}
    return {"cells": cells, "registered_strata": [int(x) for x in np.unique(strata)],
            "qualified_fraction": sum(c["qualified"] for c in cells.values()) / max(len(cells), 1),
            "thresholds": {"p_min": p_min, "n_min": n_min, "max_normalized_weight": omega},
            "unit": ("complete graph across trajectories; mean inverse propensity within a cell" if "graph_id" in records
                     else "complete seed/trajectory; mean inverse propensity within a cell"),
            "dependence": "diagnostic support only; repeated graph trajectories are not independent graphs"}


def _with_actions(features, actions):
    return np.column_stack((features, np.eye(10, dtype=np.float32)[np.asarray(actions, dtype=int)]))


def _target_scaler(target):
    target = np.asarray(target, dtype=np.float32)
    if target.ndim == 1:
        target = target[:, None]
    mean = target.mean(0)
    scale = np.maximum(target.std(0), 1e-3)
    return {"mean": mean.tolist(), "scale": scale.tolist()}


def _train_network(net, x, target, train_rows, validation_rows, config, checkpoint,
                   fingerprint, device, stop_requested, seed, heteroscedastic=False):
    finite = np.isfinite(target).all(1) & np.isfinite(x).all(1)
    train_rows = np.asarray(train_rows)[finite[train_rows]]
    validation_rows = np.asarray(validation_rows)[finite[validation_rows]]
    if not len(train_rows):
        return None, {"status": "unavailable", "reason": "no finite fitting targets"}
    scaler = _target_scaler(target[train_rows])
    normalized = (target - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
    features = torch.from_numpy(np.asarray(x, dtype=np.float32))
    targets = torch.from_numpy(np.asarray(normalized, dtype=np.float32))
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(),
                                 lr=float(_setting(config, "controller", "lr", 1e-3)),
                                 weight_decay=float(_setting(config, "controller", "weight_decay", 1e-4)))
    epochs = int(_setting(config, "controller", "epochs", 200))
    patience = int(_setting(config, "controller", "patience", 20))
    batch_size = int(_setting(config, "controller", "batch_size", 512))
    generator = torch.Generator().manual_seed(seed)
    start, best_loss, best_epoch, bad_epochs = 0, math.inf, -1, 0
    best_state, history = None, []
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("fingerprint") != fingerprint:
            raise ValueError(f"Checkpoint fingerprint mismatch: {checkpoint}")
        net.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        start, best_loss, best_epoch, bad_epochs = saved["epoch"] + 1, saved["best_loss"], saved["best_epoch"], saved["bad_epochs"]
        best_state, history = saved["best_state"], saved["history"]
        generator.set_state(saved["generator_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if device.startswith("cuda") and saved.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
    def loss_fn(prediction, truth):
        if heteroscedastic:
            mean, log_scale = prediction
            return (0.5 * ((truth - mean) * (-log_scale).exp()).square() + log_scale).mean()
        return (prediction - truth).square().mean()
    def persist(epoch):
        _atomic_save({"fingerprint": fingerprint, "model": net.state_dict(),
                      "optimizer": optimizer.state_dict(), "epoch": epoch,
                      "best_loss": best_loss, "best_epoch": best_epoch,
                      "bad_epochs": bad_epochs, "best_state": best_state,
                      "history": history, "generator_state": generator.get_state(),
                      "torch_rng_state": torch.get_rng_state(),
                      "cuda_rng_state": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None}, checkpoint)
    for epoch in range(start, epochs):
        if stop_requested():
            persist(epoch - 1)
            raise TrainingInterrupted("Controller training interrupted after a saved epoch")
        if bad_epochs >= patience:
            break
        net.train()
        permutation = train_rows[torch.randperm(len(train_rows), generator=generator).numpy()]
        train_sum, train_count = 0.0, 0
        for offset in range(0, len(permutation), batch_size):
            batch = permutation[offset:offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net(features[batch].to(device)), targets[batch].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite learned consequence-model loss")
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            train_sum += float(loss.detach()) * len(batch)
            train_count += len(batch)
        net.eval()
        validation_sum, validation_count = 0.0, 0
        with torch.no_grad():
            for offset in range(0, len(validation_rows), batch_size):
                batch = validation_rows[offset:offset + batch_size]
                loss = loss_fn(net(features[batch].to(device)), targets[batch].to(device))
                validation_sum += float(loss) * len(batch)
                validation_count += len(batch)
        # A rare-target model may have no finite internal-validation examples.
        # Its fixed-epoch training loss selects the saved state, explicitly logged.
        validation_loss = validation_sum / validation_count if validation_count else train_sum / train_count
        improved = validation_loss < best_loss - 1e-7
        if improved:
            best_loss, best_epoch, bad_epochs = validation_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad_epochs += 1
        history.append({"epoch": epoch, "train_loss": train_sum / train_count,
                        "internal_validation_loss": validation_loss,
                        "validation_target_rows": validation_count})
        persist(epoch)
    if best_state is None:
        raise ValueError("Controller epochs must be positive")
    net.load_state_dict(best_state)
    net.eval()
    return scaler, {"status": "trained", "best_epoch": best_epoch,
                    "best_internal_validation_loss": best_loss,
                    "train_rows": int(len(train_rows)), "validation_rows": int(len(validation_rows)),
                    "validation_status": "heldout trajectories" if len(validation_rows) else "no finite heldout targets; fitting-loss checkpoint",
                    "epochs_run": len(history), "history": history}


class Controller:
    def __init__(self, payload, device="cpu"):
        self.payload = payload
        self.device = str(device)
        self.config = payload["config"]
        self.observer = ConditionedObserver(payload["observer"])
        self.support_audit = payload["support"]
        self.calibration = payload.get("calibration", {})
        self.scalers = payload["scalers"]
        self.feature_dim = 2 * self.observer.dimension + 16
        self.critic = ActionCritic(self.feature_dim + 10).to(device)
        self.transition_model = ResidualModel(self.feature_dim + 10, self.observer.dimension).to(device)
        self.stage_model = ResidualModel(self.feature_dim + 10, 1).to(device)
        self.terminal_model = ResidualModel(self.feature_dim, 1).to(device)
        mapping = {"critic": self.critic, "transition": self.transition_model,
                   "stage": self.stage_model, "terminal": self.terminal_model}
        for name, model in mapping.items():
            model.load_state_dict(payload["models"][name])
            model.eval().requires_grad_(False)
        self.terminal_available = self.scalers.get("terminal") is not None
        self.sequence_models_available = self.terminal_available and self.scalers.get("transition") is not None and self.scalers.get("stage") is not None
        self.reset_counters()

    def reset_counters(self):
        self.counters = {"model_forward_calls": 0, "observer_updates": 0,
                         "observer_predictions": 0, "approx_dense_multiply_adds": 0}

    def _charge_model(self, model, rows):
        self.counters["model_forward_calls"] += 1
        self.counters["approx_dense_multiply_adds"] += int(rows) * sum(
            layer.in_features * layer.out_features for layer in model.modules() if isinstance(layer, nn.Linear))

    @classmethod
    def load(cls, path, device="cpu"):
        path = Path(path)
        if path.is_dir():
            path = path / "controller.pt"
        return cls(torch.load(path, map_location="cpu", weights_only=False), device)

    def start(self, y, previous_action=-1):
        self.counters["observer_updates"] += 1
        self.counters["approx_dense_multiply_adds"] += 25 * self.observer.dimension
        return self.observer.start(y, previous_action)

    def observe(self, state, y, depth, previous_action):
        self.counters["observer_updates"] += 1
        d = self.observer.dimension
        self.counters["approx_dense_multiply_adds"] += 4 * d**3 + 25 * d
        return self.observer.observe(state, y, depth, previous_action)

    def features(self, state):
        return _features(state, self.observer.max_depth)

    def stratum(self, y, depth):
        return int(_strata(np.asarray(y), depth, self.config))

    def supported(self, y, depth):
        if not np.isfinite(y).all():
            return []
        stratum = int(_strata(np.asarray(y), depth, self.config))
        return [action for action in range(10)
                if self.support_audit["cells"].get(f"{stratum}:{action}", {}).get("qualified", False)]

    @torch.no_grad()
    def _critic_batch(self, features, actions):
        x = torch.as_tensor(_with_actions(features, actions), dtype=torch.float32, device=self.device)
        self._charge_model(self.critic, len(features))
        mean, log_scale = self.critic(x)
        scaler = self.scalers["critic"]
        mean = mean.cpu().numpy().reshape(-1) * scaler["scale"][0] + scaler["mean"][0]
        sigma = np.exp(log_scale.cpu().numpy().reshape(-1)) * scaler["scale"][0]
        nominal = np.asarray(actions) == 0
        mean[nominal], sigma[nominal] = 0.0, 0.0
        return mean, sigma

    def _radius(self, stratum, action):
        return calibration_radius(self.calibration, stratum, action)

    def score(self, state, actions, kind="radius", kappa=1.0, lambda_rho=0.0001):
        actions = [int(a) for a in actions]
        if not actions:
            return {}
        features = np.repeat(self.features(state)[None], len(actions), axis=0)
        means, sigmas = self._critic_batch(features, actions)
        support = set(self.supported(state.y, state.depth))
        stratum = int(_strata(state.y, state.depth, self.config))
        radii = np.asarray([self._radius(stratum, a) for a in actions])
        rho = np.asarray([ACTIONS[a].rho for a in actions])
        values = action_scores(means, sigmas, radii, rho, kind=kind,
                               kappa=kappa, lambda_rho=lambda_rho)
        result = {}
        for action, mean, sigma, radius, value in zip(actions, means, sigmas, radii, values):
            score = float(value) if action in support and state.valid else math.inf
            result[action] = {"mean": float(mean), "sigma": float(sigma),
                              "radius": float(radius), "penalty": 0.0,
                              "transport_penalty": float(lambda_rho * ACTIONS[action].rho),
                              "score": score}
        return result

    @torch.no_grad()
    def _regression(self, model, scaler, features):
        if scaler is None:
            raise RuntimeError("The requested consequence model has no fitted targets")
        x = torch.as_tensor(np.asarray(features)[None], dtype=torch.float32, device=self.device)
        self._charge_model(model, 1)
        standardized = model(x).cpu().numpy()[0]
        return standardized * np.asarray(scaler["scale"]) + np.asarray(scaler["mean"])

    def predict(self, state, action):
        features = _with_actions(self.features(state)[None], [action])[0]
        delta = self._regression(self.transition_model, self.scalers["transition"], features)
        z = state.z + delta
        _, covariance = self.observer.transition(state, action)
        self.counters["observer_predictions"] += 1
        self.counters["approx_dense_multiply_adds"] += 2 * self.observer.dimension**3
        raw = self.observer.reconstruct(z)
        raw[5] = (state.depth + 1) / self.observer.max_depth
        accounting = site_transport_accounting(action, state.depth, self.config)
        raw[6] = state.y[6] + accounting["mass"] / self.observer.max_depth
        raw[7] = state.y[7] + accounting["hops"] / self.observer.max_depth
        return ControllerState(z, covariance, 0.0, 0.0, 0.0, raw,
                               state.depth + 1, int(action), bool(np.isfinite(z).all()),
                               np.zeros(self.observer.dimension))

    def stage_cost(self, state, action):
        features = _with_actions(self.features(state)[None], [action])[0]
        return float(self._regression(self.stage_model, self.scalers["stage"], features)[0])

    def terminal_value(self, state):
        if not self.terminal_available:
            raise RuntimeError("No finite terminal_delta targets; MPC cannot be evaluated")
        if state.depth >= self.observer.max_depth:
            return 0.0
        return float(self._regression(self.terminal_model, self.scalers["terminal"], self.features(state))[0])


def _fit_one(records, fitting_indices, config, directory, fingerprint, device, stop_requested, seed):
    directory.mkdir(parents=True, exist_ok=True)
    completed = directory / "complete.pt"
    if completed.exists():
        payload = torch.load(completed, map_location="cpu", weights_only=False)
        if payload.get("fingerprint") != fingerprint:
            raise ValueError(f"Completed fold fingerprint mismatch: {directory}")
        return payload
    fitting_indices = np.asarray(fitting_indices, dtype=np.int64)
    all_groups = _holdout_groups(records)
    unique = np.unique(all_groups[fitting_indices])
    if len(unique) < 2:
        raise ValueError("Each fitting fold needs at least two trajectories for internal validation")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    validation_count = min(len(unique) - 1, max(1, int(math.ceil(0.10 * len(unique)))))
    validation_groups = set(shuffled[:validation_count].tolist())
    validation_mask = np.array([g in validation_groups for g in all_groups[fitting_indices]])
    fit_rows, validation_rows = np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)
    observer = _fit_observer(records, fitting_indices[fit_rows], config)
    features, z, z_next, nis = _encode_records(observer, records, fitting_indices)
    valid_nis = nis[validation_rows]
    positive_nis = valid_nis[np.isfinite(valid_nis) & (valid_nis > 0)]
    threshold = _quantile(positive_nis, 0.99)
    if threshold is None:
        threshold = float(4 * observer.dimension + 16)
    observer.nis_threshold = max(float(threshold), float(observer.dimension))
    observer.payload["nis_threshold"] = observer.nis_threshold
    actions = np.asarray(records["action"])[fitting_indices].astype(int)
    action_features = _with_actions(features, actions)
    inputs = {"critic": action_features, "transition": action_features,
              "stage": action_features, "terminal": features}
    targets = {"critic": np.asarray(records["delta"])[fitting_indices, None],
               "transition": z_next - z,
               "stage": np.asarray(records["stage_delta"])[fitting_indices, None],
               "terminal": np.asarray(records["terminal_delta"])[fitting_indices, None]}
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    models = {"critic": ActionCritic(features.shape[1] + 10),
              "transition": ResidualModel(features.shape[1] + 10, observer.dimension),
              "stage": ResidualModel(features.shape[1] + 10, 1),
              "terminal": ResidualModel(features.shape[1], 1)}
    scalers, model_audits = {}, {}
    for number, name in enumerate(("critic", "transition", "stage", "terminal")):
        scalers[name], model_audits[name] = _train_network(
            models[name], inputs[name], targets[name], fit_rows, validation_rows, config,
            directory / f"{name}.last.pt", fingerprint, str(device), stop_requested,
            seed + 17 * number, heteroscedastic=(name == "critic"))
    if scalers["critic"] is None:
        raise ValueError("The action critic needs finite fitting delta targets")
    support = _support(records, fitting_indices, config)
    payload = {"schema_version": "cross-morse-controller-2", "fingerprint": fingerprint,
               "config": config, "observer": observer.payload, "support": support,
               "scalers": scalers,
               "models": {name: {k: v.detach().cpu() for k, v in model.state_dict().items()}
                          for name, model in models.items()},
               "calibration": {"cells": {}, "actions": {}},
               "audit": {"fitting_trajectories": int(len(unique)),
                         "internal_optimization_trajectories": int(len(unique) - validation_count),
                         "internal_validation_trajectories": int(validation_count),
                         "internal_validation_groups": sorted(validation_groups),
                         "fitting_rows": int(len(fitting_indices)),
                         "nis_threshold": observer.nis_threshold,
                         "model_training": model_audits}}
    _atomic_save(payload, completed)
    return payload


def calibration_partition(records, config):
    """Split complete (seed, trajectory) groups before any model is fitted.

    Only Dexplore is subdivided. Calibration targets never fit any observer,
    normalization, network, or checkpoint selector. Assessment targets never
    fit a model or a radius. The graph is still shared: no i.i.d.-graph claim.
    """
    groups = _holdout_groups(records)
    unique = np.unique(groups)
    fraction = float(_setting(config, "controller", "calibration_fraction", 0.20))
    audit_fraction = float(_setting(config, "controller", "calibration_audit_fraction", 0.20))
    if not (0 < fraction < 1 and 0 < audit_fraction < 1 and fraction + audit_fraction < 1):
        raise ValueError("Calibration and assessment fractions must be positive and sum to less than one")
    rng = np.random.default_rng(int(config.get("seed", 1)) + 5717)
    shuffled = rng.permutation(unique)
    nc, na = max(1, int(len(unique) * fraction)), max(1, int(len(unique) * audit_fraction))
    minimum_fit = max(3, int(_setting(config, "controller", "crossfit_folds", 5)))
    if len(unique) - nc - na < minimum_fit:
        raise ValueError("Too few fitting trajectories after calibration/assessment holdouts; collect more trajectories")
    calibration_groups, audit_groups = shuffled[:nc], shuffled[nc:nc + na]
    role = np.zeros(len(groups), dtype=np.int8)
    role[np.isin(groups, calibration_groups)] = 1
    role[np.isin(groups, audit_groups)] = 2
    declaration = {
        "protocol": "trajectory_holdout_v2", "role_codes": {"0": "model_fit", "1": "radius_fit", "2": "assessment"},
        "split_seed": int(config.get("seed", 1)) + 5717,
        "calibration_fraction": fraction, "assessment_fraction": audit_fraction,
        "group_definition": ("complete graph across all seeds and trajectories" if "graph_id" in records
                             else "complete seed_index/trajectory pair; not independent graphs"),
        "groups": {name: sorted(np.unique(groups[role == code]).tolist())
                   for code, name in enumerate(("model_fit", "radius_fit", "assessment"))},
        "rows": {name: int(np.sum(role == code))
                 for code, name in enumerate(("model_fit", "radius_fit", "assessment"))},
        "shared_graph": "graph_id" not in records, "data_role": "Dexplore", "assessment_used_for_selection": False,
    }
    return role, declaration


def _calibrate(records, predictions, config, indices):
    """All-logged-action trajectory-max radii on a dedicated held-out set.

    No circular selected-action preference: selected-action performance is
    assessed AFTER freezing these radii, on a different group set.
    """
    alpha = float(_setting(config, "controller", "calibration_alpha", 0.05))
    minimum = int(_setting(config, "controller", "calibration_min_count", 20))
    if not 0 < alpha < 1 or minimum < 1:
        raise ValueError("Invalid calibration alpha or minimum trajectory count")
    action = np.asarray(records["action"]).astype(int)
    strata = _strata(np.asarray(records["y"]), np.asarray(records["depth"]), config)
    errors = np.abs(np.asarray(records["delta"]) - predictions)
    groups = _holdout_groups(records)
    eligible = np.zeros(len(action), dtype=bool)
    eligible[np.asarray(indices, dtype=int)] = True
    def entry(rows):
        if not len(rows):
            return None
        units, inverse = np.unique(groups[rows], return_inverse=True)
        per_unit = np.full(len(units), -np.inf)
        np.maximum.at(per_unit, inverse, errors[rows])
        return {"radius": _quantile(per_unit, 1 - alpha),
                "trajectory_count": int(len(units)), "row_count": int(len(rows)),
                "selected_action_only": False, "small_sample": len(units) < minimum,
                "source": "held-out all-logged-action trajectory maxima"}
    cells, actions = {}, {}
    for a in range(10):
        rows = np.flatnonzero(eligible & (action == a) & np.isfinite(errors))
        value = entry(rows)
        if value is not None:
            if a == 0:
                value["radius"], value["source"] = 0.0, "analytic nominal-relative identity"
            actions[str(a)] = value
    for st in np.unique(strata[eligible]):
        for a in range(10):
            rows = np.flatnonzero(eligible & (strata == st) & (action == a) & np.isfinite(errors))
            if len(np.unique(groups[rows])) >= minimum:
                value = entry(rows)
                if a == 0:
                    value["radius"] = 0.0
                cells[f"{int(st)}:{a}"] = value
    return {"alpha": alpha, "empirical_quantile_level": 1 - alpha,
            "minimum_trajectories_for_cell": minimum, "cells": cells, "actions": actions,
            "coverage_status": "EMPIRICAL_ONLY; held-out trajectories share a graph; no finite-sample coverage claim",
            "aggregation": "maximum absolute residual per calibration trajectory within each action/cell",
            "selection_preference": "none; all logged actions; separate post-freeze assessment",
            "fallback": "action-pooled radius if cell too small; infinity if action unobserved",
            "quantile_method": "numpy higher empirical quantile, not a conformal guarantee"}


def _assessment(controller, records, indices, config):
    """Evaluate the exact one-step selector on held-out logged histories.

    The realized target is known only for the action taken by the behaviour
    policy. Matching and inverse-propensity diagnostics therefore describe
    logged states, NOT the closed-loop policy state distribution or MPC coverage.
    """
    indices = np.asarray(indices, dtype=int)
    features, _, _, _, valid = _encode_records(controller.observer, records, indices, return_valid=True)
    actions = np.asarray(records["action"])[indices].astype(int)
    groups = _holdout_groups(records)[indices]
    propensity = np.asarray(records["propensity"])[indices].astype(float)
    target = np.asarray(records["delta"])[indices]
    specifications = [("one_step", 0.0)] + [("radius", float(k)) for k in
        _setting(config, "controller", "kappa_grid", [0.5, 1.0, 2.0])]
    chosen = np.full((len(specifications), len(indices)), -1, dtype=np.int64)
    means = np.full(len(indices), np.nan)
    sigmas = np.full(len(indices), np.nan)
    radii = np.full(len(indices), np.inf)
    for start in range(0, len(indices), 2048):
        ids = indices[start:start + 2048]
        x = features[start:start + len(ids)]
        all_mean, all_sigma = controller._critic_batch(np.repeat(x, 10, axis=0), np.tile(np.arange(10), len(ids)))
        all_mean, all_sigma = all_mean.reshape(-1, 10), all_sigma.reshape(-1, 10)
        allowed = np.zeros((len(ids), 10), dtype=bool)
        all_radius = np.zeros((len(ids), 10))
        for j, i in enumerate(ids):
            allowed[j, controller.supported(records["y"][i], records["depth"][i])] = True
            st = int(_strata(records["y"][i], records["depth"][i], config))
            all_radius[j] = [controller._radius(st, a) for a in range(10)]
        allowed &= valid[start:start + len(ids), None]
        observed = np.asarray(records["action"])[ids].astype(int)
        loc = np.arange(len(ids))
        means[start:start + len(ids)] = all_mean[loc, observed]
        sigmas[start:start + len(ids)] = all_sigma[loc, observed]
        radii[start:start + len(ids)] = all_radius[loc, observed]
        for q, (kind, kappa) in enumerate(specifications):
            scores = action_scores(all_mean, all_sigma, all_radius,
                                   np.asarray([a.rho for a in ACTIONS])[None], kind=kind,
                                   kappa=kappa, lambda_rho=float(_setting(config, "controller", "lambda_rho", 0.0001)))
            scores[~allowed] = np.inf
            good = np.isfinite(scores).any(axis=1)
            winner = scores.argmin(axis=1)  # canonical ascending action-ID tie order
            chosen[q, start:start + len(ids)] = np.where(good, winner, -1)
    residual = np.abs(target - means)
    covered = residual <= radii
    policies = []
    for q, (kind, kappa) in enumerate(specifications):
        matched = (chosen[q] == actions) & (chosen[q] >= 0)
        def summarize(mask):
            n = int(mask.sum())
            weights = 1 / propensity[mask]
            return {"matched_rows": n, "matched_trajectories": int(len(np.unique(groups[mask]))),
                    "covered_rows": int(covered[mask].sum()),
                    "matched_residual_radius_coverage": float(covered[mask].mean()) if n else None,
                    "ipw_residual_radius_coverage": float(np.dot(weights, covered[mask]) / weights.sum()) if n else None,
                    "finite_radius_rows": int(np.isfinite(radii[mask]).sum())}
        policies.append({"kind": kind, "kappa": kappa,
                         "eligible_rows": int(np.sum(chosen[q] >= 0)),
                         "recovery_or_unscorable_rows": int(np.sum(chosen[q] < 0)),
                         "all_actions": summarize(matched), "nonnominal_actions": summarize(matched & (actions != 0)),
                         "nominal_matches": int(np.sum(matched & (actions == 0)))})
    report = {"schema_version": "calibration-assessment-2", "data_role": "Dexplore assessment holdout",
              "rows": len(indices), "trajectories": int(len(np.unique(groups))),
              "score_contract": SCORE_VERSION, "policies": policies,
              "radii_frozen_before_assessment": True, "assessment_labels_used_for_fitting": False,
              "estimand": "absolute nominal-relative critic residual within empirical radius on matched logged states",
              "scope": "off-policy logged-history diagnostic; not complete-policy, sequence, or independent-graph coverage",
              "sequence_coverage": "NOT_MEASURED", "policy_state_distribution_coverage": "NOT_MEASURED"}
    arrays = {"row_index": indices, "trajectory_group": groups, "logged_action": actions,
              "propensity": propensity, "delta": target, "mean": means, "sigma": sigmas,
              "radius": radii, "covered": covered, "selected_actions": chosen,
              "observer_valid": valid, "policy_kind": np.asarray([x[0] for x in specifications]),
              "policy_kappa": np.asarray([x[1] for x in specifications])}
    return report, arrays


def fit_controller(records, config, output_dir: Path, device: str = "cpu",
                   stop_requested: Callable[[], bool] = lambda: False) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    required = ("y", "y_next", "action", "previous_action", "depth", "trajectory",
                "delta", "stage_delta", "terminal_delta", "propensity")
    missing = [name for name in required if name not in records]
    if missing:
        raise ValueError(f"Missing controller records: {missing}")
    records = {key: np.asarray(value) for key, value in records.items()}
    count = len(records["action"])
    if any(len(records[name]) != count for name in required):
        raise ValueError("Controller record arrays have inconsistent lengths")
    if records["y"].shape != (count, 25) or records["y_next"].shape != (count, 25):
        raise ValueError("Raw sensor schema must have exactly 25 coordinates")
    if not np.all(np.isfinite(records["delta"])):
        raise ValueError("Nonfinite action-delta targets cannot train the critic")
    if not np.all((records["action"] >= 0) & (records["action"] < 10)):
        raise ValueError("Invalid action ids")
    propensities = records["propensity"]
    if not np.all(np.isfinite(propensities) & (propensities > 0) & (propensities <= 1)):
        raise ValueError("Every logged propensity must be in (0,1]")
    if np.any(np.abs(records["delta"][records["action"] == 0]) > 1e-7):
        raise ValueError("Nominal-relative A0 target must be zero")
    fingerprint = _fingerprint(records, config)
    groups = _holdout_groups(records)
    unique = np.unique(groups)
    folds = int(_setting(config, "controller", "crossfit_folds", 5))
    if folds < 2 or len(unique) < max(folds, 3):
        raise ValueError("Need at least three trajectories and one per requested cross-fitting fold")
    roles, partition = calibration_partition(records, config)
    partition["fingerprint"] = fingerprint
    existing_partition = output_dir / "calibration_partition.json"
    if existing_partition.exists() and json.loads(existing_partition.read_text()).get("fingerprint") != fingerprint:
        raise ValueError("Calibration partition fingerprint mismatch; use a new output directory")
    _write_json(partition, output_dir / "calibration_partition.json")
    fit_indices, cal_indices, audit_indices = (np.flatnonzero(roles == k) for k in range(3))
    fitting_unique = np.unique(groups[fit_indices])
    rng = np.random.default_rng(int(config.get("seed", 1)) + 3109)
    assignment = {int(group): int(i % folds) for i, group in enumerate(rng.permutation(fitting_unique))}
    fold_ids = np.asarray([assignment[int(group)] if role == 0 else -int(role)
                           for group, role in zip(groups, roles)])
    predictions, sigmas = np.full(count, np.nan), np.full(count, np.nan)
    fold_audits = []
    for fold in range(folds):
        if stop_requested():
            raise TrainingInterrupted("Controller stopped before next cross-fitting fold")
        fitting = np.flatnonzero((roles == 0) & (fold_ids != fold))
        heldout = np.flatnonzero((roles == 0) & (fold_ids == fold))
        payload = _fit_one(records, fitting, config, output_dir / "folds" / f"fold_{fold:02d}",
                           fingerprint, device, stop_requested, 1009 + fold * 37)
        controller = Controller(payload, device)
        features, _, _, _ = _encode_records(controller.observer, records, heldout)
        for offset in range(0, len(heldout), 2048):
            indices = heldout[offset:offset + 2048]
            mean, sigma = controller._critic_batch(features[offset:offset + len(indices)], records["action"][indices])
            predictions[indices], sigmas[indices] = mean, sigma
        fitting_groups, heldout_groups = set(groups[fitting].tolist()), set(groups[heldout].tolist())
        assert fitting_groups.isdisjoint(heldout_groups)
        fold_audits.append({"fold": fold, "fitting_rows": int(len(fitting)), "heldout_rows": int(len(heldout)),
                            "fitting_trajectories": len(fitting_groups), "heldout_trajectories": len(heldout_groups),
                            "trajectory_overlap": 0, "early_stopping": "internal fitting trajectories; never the OOF fold",
                            "rmse": float(np.sqrt(np.mean((predictions[heldout] - records["delta"][heldout])**2)))})
    # This final model is never refitted after its calibration/assessment.
    final = _fit_one(records, fit_indices, config, output_dir / "final_fit", fingerprint,
                     device, stop_requested, 7919)
    controller = Controller(final, device)
    features, _, _, _ = _encode_records(controller.observer, records, cal_indices)
    for offset in range(0, len(cal_indices), 2048):
        indices = cal_indices[offset:offset + 2048]
        predictions[indices], sigmas[indices] = controller._critic_batch(features[offset:offset + len(indices)],
                                                                        records["action"][indices])
    calibration = _calibrate(records, predictions, config, cal_indices)
    final["calibration"] = calibration
    controller.calibration = calibration
    assessment, assessment_arrays = _assessment(controller, records, audit_indices, config)
    predictions[audit_indices], sigmas[audit_indices] = assessment_arrays["mean"], assessment_arrays["sigma"]
    if not np.isfinite(predictions).all():
        raise RuntimeError("Cross-fitting/holdouts did not predict every row")
    audit = {"schema_version": "cross-morse-learning-audit-2", "fingerprint": fingerprint,
             "profile": config.get("active_profile", "full"), "rows": count, "trajectories": int(len(unique)),
             "crossfit_folds": folds, "folds": fold_audits, "support": final["support"],
             "calibration": calibration, "calibration_partition": partition,
             "calibration_assessment": assessment,
             "oof_rmse": float(np.sqrt(np.mean((predictions[fit_indices] - records["delta"][fit_indices])**2))),
             "oof_rmse_scope": "cross-fitted model-fit partition only; holdouts reported separately",
             "final_fit": final["audit"], "terminal_available": final["scalers"]["terminal"] is not None,
             "train_data_role": "Dexplore only; trajectory-partitioned fitting, calibration, assessment",
             "implementation_choices": {
                 "normalization": "training median/MAD, replacement by median, clip[-5,5]",
                 "latent": "PCA16 unless configured; conditioned ridge; covariance rank4 plus diagonal",
                 "NIS": "empirical internal-validation 99th percentile, lower floor state_dim",
                 "operational_penalty": 0.0, "calibration": "held-out trajectory-max radii; disjoint assessment partition",
                 "support_unit": "mean inverse propensity within model-fit seed/trajectory/action/stratum cell",
                 "terminal": "remaining nominal CE increment; unavailable if no finite fitting targets"}}
    final["audit"] = audit
    _atomic_save(final, output_dir / "controller.pt")
    np.savez_compressed(output_dir / "oof_predictions.npz", mean=predictions, sigma=sigmas,
                        fold=fold_ids, role=roles, trajectory_group=groups, delta=records["delta"])
    np.savez_compressed(output_dir / "calibration_assessment.npz", **assessment_arrays)
    _write_json(assessment, output_dir / "calibration_assessment.json")
    _write_json(audit, output_dir / "audit.json")
    return audit
