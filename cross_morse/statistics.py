"""Paired whole-unit statistics for CROSS-MORSE policy selection.

A unit is a dependent core on a single transductive graph, or a complete
independent graph. Core-based intervals remain descriptive under dependence.
The independent-graph admission bound is explicitly conditional on its stated
population and frozen-policy assumptions. Resampling preserves original unit
membership and its fixed loss weight; it never treats within-unit nodes or
optimization seeds as additional independent units.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import math

import numpy as np


def _vector(values: Any, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    if out.ndim != 1 or not len(out) or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return out


def _weights(weights: Any, n: int) -> np.ndarray:
    w = np.ones(n, dtype=np.float64) if weights is None else _vector(weights, "weights")
    if w.shape != (n,) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be nonnegative, match units, and have positive sum")
    if (w == 0).any():
        raise ValueError("remove zero-weight units before resampling")
    return w


def _check_bootstrap(reps: int, alpha: float) -> None:
    if reps < 20:
        raise ValueError("at least 20 bootstrap replicates are required")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must lie strictly between zero and .5")


def cluster_summary(node_values: Any, cluster_ids: Any):
    """Return (core means, sorted original core IDs, scored-node counts).

    Only pass nodes that belong to the requested label stage and core. Buffer
    nodes must be excluded by the caller. Accepts [nodes] or [nodes, candidates]
    so the same membership and weights apply to every paired policy.
    """
    values = np.asarray(node_values, dtype=np.float64)
    ids = np.asarray(cluster_ids)
    if values.ndim not in (1, 2) or ids.ndim != 1 or len(values) != len(ids):
        raise ValueError("values must be [nodes] or [nodes,candidates], with one core ID per node")
    if not len(ids) or not np.isfinite(values).all():
        raise ValueError("core observations must be nonempty and finite")
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    totals = np.zeros((len(unique),) + values.shape[1:], dtype=np.float64)
    np.add.at(totals, inverse, values)
    divisor = counts if values.ndim == 1 else counts[:, None]
    return totals / divisor, unique, counts


def paired_bootstrap(
    differences: Any,
    weights: Any = None,
    reps: int = 10_000,
    alpha: float = 0.025,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired whole-core ratio bootstrap; lower/upper are alpha/1-alpha quantiles.

    The upper endpoint is a one-sided (1-alpha) descriptive limit; the pair of
    endpoints is a central (1-2*alpha) interval. P values use the null-centered
    bootstrap distribution, not an exact randomization or independent-node test.
    A single core has no estimable uncertainty and never passes a downstream
    gate: interval fields are None rather than misleading zero-width bounds.
    """
    _check_bootstrap(reps, alpha)
    d = _vector(differences, "differences")
    w = _weights(weights, len(d))
    mean = float(np.dot(w, d) / w.sum())
    base = dict(
        mean=mean,
        units=int(len(d)),
        weight_sum=float(w.sum()),
        weighting="equal_unit" if np.all(w == w[0]) else "fixed_unit_weight_ratio",
        method="paired whole-unit bootstrap; descriptive when units are dependent",
        independent_graph_guarantee=False,
        one_sided_alpha=float(alpha),
        interval_level=float(1 - 2 * alpha),
        bootstrap_replicates=int(reps),
        bootstrap_seed=int(seed),
    )
    if len(d) < 2:
        return dict(base, lower=None, upper=None, se=None, valid_interval=False,
                    p_less_zero=None, p_greater_zero=None)
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, 1024):
        ids = rng.integers(0, len(d), size=(min(1024, reps - start), len(d)))
        bw = w[ids]
        means[start:start + len(ids)] = (bw * d[ids]).sum(axis=1) / bw.sum(axis=1)
    centered = means - mean
    return dict(
        base,
        lower=float(np.quantile(means, alpha)),
        upper=float(np.quantile(means, 1 - alpha)),
        se=float(np.std(means, ddof=1)),
        valid_interval=True,
        p_less_zero=float((1 + np.count_nonzero(centered <= mean)) / (reps + 1)),
        p_greater_zero=float((1 + np.count_nonzero(centered >= mean)) / (reps + 1)),
    )


def _candidate_key(c: Mapping[str, Any], index: int) -> tuple:
    complexity = c.get("complexity", 0)
    if isinstance(complexity, (list, tuple)):
        complexity = tuple(complexity)
    else:
        complexity = (float(complexity),)
    return (
        complexity,
        int(c.get("ensemble", 1)),
        float(c.get("estimated_cost", c.get("cost", 0.0))),
        str(c.get("hash", f"index-{index:08d}")),
        index,
    )


def _argmin(scores: np.ndarray, ids: Sequence[int], candidates: Sequence[Mapping]) -> int:
    """Exact score comparison followed by the registered deterministic tie order."""
    return min(ids, key=lambda j: (float(scores[j]), _candidate_key(candidates[j], j)))


def _inner_labels(n: int, seed: int) -> np.ndarray:
    out = np.empty(n, dtype=np.int64)
    out[np.random.default_rng(seed).permutation(n)] = np.arange(n) % min(4, n)
    return out


def _reconstruct_comparator(
    losses: np.ndarray,
    training_weights: np.ndarray,
    inner_labels: np.ndarray,
    open_loop_ids: Sequence[int],
    candidates: Sequence[Mapping],
) -> tuple[int, dict[str, Any]]:
    """Cross-fit within each open-loop family, then fit its final configuration.

    Candidate predictions themselves are frozen. The inner folds select each
    family's configuration without its held-out units, compare families on inner
    held-out risk, then select a configuration in the winning family using all
    available outer-training units. Duplicate bootstrap units remain in their
    original folds because multiplicities are represented by weights.
    """
    active = training_weights > 0
    if np.count_nonzero(active) < 2:
        raise ValueError("at least two outer-training cores are needed")
    families: dict[str, list[int]] = {}
    for j in open_loop_ids:
        families.setdefault(str(candidates[j].get("route", "open_loop")), []).append(j)
    labels = np.unique(inner_labels[active])
    # Even with an unlucky bootstrap, the comparison never evaluates a family
    # on the same original unit that picked its within-family configuration.
    if len(labels) < 2:
        raise ValueError("bootstrap left fewer than two nonempty inner folds")
    risk_sum = {family: 0.0 for family in families}
    held_weight_sum = 0.0
    selection_trace: list[dict[str, Any]] = []
    for label in labels:
        held = active & (inner_labels == label)
        train = active & ~held
        w_train = training_weights * train
        w_held = training_weights * held
        train_mean = (w_train @ losses) / w_train.sum()
        chosen: dict[str, int] = {}
        for family, ids in families.items():
            j = _argmin(train_mean, ids, candidates)
            risk_sum[family] += float(w_held @ losses[:, j])
            chosen[family] = j
        held_weight_sum += float(w_held.sum())
        selection_trace.append({"inner_fold": int(label), "family_configs": chosen})
    family_risks = {family: value / held_weight_sum for family, value in risk_sum.items()}
    all_training_risk = (training_weights @ losses) / training_weights.sum()
    family_representatives = {
        family: _argmin(all_training_risk, ids, candidates) for family, ids in families.items()
    }
    family = min(families, key=lambda f: (
        family_risks[f], _candidate_key(candidates[family_representatives[f]], family_representatives[f])
    ))
    chosen = family_representatives[family]
    return chosen, dict(
        selected_family=family,
        candidate_index=int(chosen),
        inner_family_risks={k: float(v) for k, v in family_risks.items()},
        inner_fold_choices=selection_trace,
    )


def _max_t_interval(point: np.ndarray, draws: np.ndarray, alpha: float):
    se = np.std(draws, axis=0, ddof=1)
    nonzero = se > 1e-12
    maxima = np.zeros(len(draws), dtype=np.float64)
    if nonzero.any():
        maxima = np.max(np.abs(draws[:, nonzero] - point[nonzero]) / se[nonzero], axis=1)
    critical = float(np.quantile(maxima, 1 - alpha))
    radius = se * critical
    radius[~nonzero] = 0.0
    return point - radius, point + radius, se, critical


def _restore_candidate_indices(result: Mapping[str, Any], observed_ids: np.ndarray,
                               candidate_count: int) -> dict[str, Any]:
    """Restore registry positions without inventing failed-policy observations.

    The numerical selection ran only on successfully executed candidates. None
    therefore denotes an unavailable complete-policy observation, not a zero
    effect, a comparator prediction, or an extremely bad measured score.
    """
    out = dict(result)
    indices = observed_ids.astype(int).tolist()
    available = np.zeros(candidate_count, dtype=bool)
    available[observed_ids] = True

    def expand(values, missing):
        restored = [missing] * candidate_count
        for index, value in zip(indices, values):
            restored[index] = value
        return restored

    def restore_trace(trace):
        restored = dict(trace)
        restored["candidate_index"] = indices[trace["candidate_index"]]
        restored["inner_fold_choices"] = [
            dict(row, family_configs={family: indices[index]
                                      for family, index in row["family_configs"].items()})
            for row in trace["inner_fold_choices"]
        ]
        return restored

    for key in ("comparator_index", "qC_index", "provisional_index"):
        out[key] = indices[result[key]]
    for key in ("eligible_indices", "Q_s"):
        out[key] = [indices[index] for index in result[key]]
    for key in ("mean", "lower", "upper", "L", "U", "se", "metric_mean",
                "metric_lower", "metric_upper", "metric_se"):
        out[key] = expand(result[key], None)
    for key in ("metric_feasible", "joint_feasible"):
        out[key] = expand(result[key], False)
    out["bootstrap_final_comparator_counts"] = expand(result["bootstrap_final_comparator_counts"], 0)
    out["nested_comparator_fold_choices"] = [restore_trace(row)
                                                for row in result["nested_comparator_fold_choices"]]
    out["final_comparator_selection"] = restore_trace(result["final_comparator_selection"])
    out.update(candidate_count=int(candidate_count), observed_candidate_count=len(indices),
               observations_available=available.tolist(),
               execution_failure_indices=np.flatnonzero(~available).tolist())
    return out


def nested_selection(
    losses: Any,
    accuracies: Any,
    candidates: Sequence[Mapping[str, Any]],
    fold_ids: Any,
    inner_fold_ids: Any = None,
    reps: int = 10_000,
    alpha: float = 0.025,
    epsilon_r: float = 0.003,
    epsilon_eq: float = 0.002,
    epsilon_metric: float = 0.001,
    seed: int = 0,
    weights: Any = None,
    metric_evaluator: Any = None,
) -> dict[str, Any]:
    """Main Eq.(19), with nested qC reconstruction in each core resample.

    Arrays have shape [cores, fixed registered candidates]. The caller computes
    support and componentwise resource feasibility before selection. Official
    accuracy is in [0,1], not percent. ``fold_ids`` are fixed original-core outer
    assignments (normally five); inner assignments are fixed original-core IDs
    (normally four), optionally an [outer_folds,cores] matrix.

    Only feedback candidates can enter Q_s: the lifecycle compares Q_FB union
    {qC}. Open-loop candidate contrasts are diagnostic and do not receive credit
    for within-family re-selection optimism (Supplementary S12b). The comparator
    itself is selected by nested family/configuration risk, never by test data.

    A candidate marked execution_failure=True must also have feasible=False.
    Its complete-policy observations are unavailable: its columns may contain
    NaNs and are excluded before any comparator or interval computation. Such
    registry entries retain their indices and receive None-valued estimates.
    Every nonfailure candidate must still supply finite paired observations,
    including candidates that are otherwise infeasible under resource/support
    constraints. No failed candidate is assigned fabricated comparator scores.
    """
    _check_bootstrap(reps, alpha)
    loss = np.asarray(losses, dtype=np.float64)
    acc = np.zeros_like(loss) if accuracies is None and metric_evaluator is not None else np.asarray(accuracies, dtype=np.float64)
    if loss.ndim != 2 or acc.shape != loss.shape or loss.shape[1] != len(candidates):
        raise ValueError("losses/accuracies must have shape [cores,len(candidates)]")
    n, q = loss.shape
    if n < 4 or q == 0:
        raise ValueError("nested selection needs at least four cores and one candidate")
    feasible = np.asarray([bool(c.get("feasible", True)) for c in candidates])
    failed = np.asarray([bool(c.get("execution_failure", False)) for c in candidates])
    if (failed & feasible).any():
        raise ValueError("an execution_failure candidate must have feasible=False")
    observed_ids = np.flatnonzero(~failed)
    if not len(observed_ids):
        raise ValueError("at least one observed feasible open-loop comparator is required")
    if not np.isfinite(loss[:, observed_ids]).all() or not np.isfinite(acc[:, observed_ids]).all():
        raise ValueError("all nonfailure candidate observations must be finite")
    if (acc[:, observed_ids] < 0).any() or (acc[:, observed_ids] > 1).any():
        raise ValueError("accuracy must use fractions in [0,1]")
    if failed.any():
        # Remove failed columns, rather than carrying NaNs or synthetic losses
        # into cross-fitting or max-t. With the same seed, adding a failed
        # registry entry leaves every surviving numerical calculation intact.
        result = nested_selection(
            loss[:, observed_ids].copy(), acc[:, observed_ids].copy(),
            [candidates[j] for j in observed_ids], fold_ids, inner_fold_ids,
            reps=reps, alpha=alpha, epsilon_r=epsilon_r, epsilon_eq=epsilon_eq,
            epsilon_metric=epsilon_metric, seed=seed, weights=weights,
            metric_evaluator=(None if metric_evaluator is None else
                lambda candidate_ids, comparators, effective_weights: metric_evaluator(
                    observed_ids[candidate_ids], observed_ids[comparators], effective_weights)),
        )
        return _restore_candidate_indices(result, observed_ids, q)
    w = _weights(weights, n)
    folds = np.asarray(fold_ids, dtype=np.int64)
    if folds.shape != (n,):
        raise ValueError("fold_ids must have one original outer-fold ID per core")
    outer = np.unique(folds)
    if len(outer) < 2:
        raise ValueError("at least two outer folds are required")
    if inner_fold_ids is None:
        inner = _inner_labels(n, seed + 173)
    else:
        inner = np.asarray(inner_fold_ids, dtype=np.int64)
        if inner.shape not in ((n,), (len(outer), n)):
            raise ValueError("inner_fold_ids must be [cores] or [outer_folds,cores]")
    final_inner = _inner_labels(n, seed + 173) if inner.ndim == 2 else inner
    open_ids = [j for j, c in enumerate(candidates) if bool(c.get("is_open_loop", False)) and feasible[j]]
    if not open_ids:
        raise ValueError("at least one resource/support-feasible open-loop comparator is required")

    def crossfit(effective_weights: np.ndarray, trace: bool = False):
        c_loss = np.zeros(n, dtype=np.float64)
        c_acc = np.zeros(n, dtype=np.float64)
        c_indices = np.zeros(n, dtype=np.int64)
        traces = []
        for k, outer_id in enumerate(outer):
            held = folds == outer_id
            if effective_weights[held].sum() == 0:
                continue
            train_weights = effective_weights * (~held)
            labels = inner if inner.ndim == 1 else inner[k]
            j, info = _reconstruct_comparator(loss, train_weights, labels, open_ids, candidates)
            c_loss[held], c_acc[held] = loss[held, j], acc[held, j]
            c_indices[held] = j
            if trace:
                info["outer_fold"] = int(outer_id)
                info["held_out_original_units"] = np.flatnonzero(held).tolist()
                info["training_original_units"] = np.flatnonzero(train_weights > 0).tolist()
                traces.append(info)
        paired_loss = ((effective_weights[:, None] * (loss - c_loss[:, None])).sum(axis=0)
                       / effective_weights.sum())
        paired_acc = ((effective_weights[:, None] * (acc - c_acc[:, None])).sum(axis=0)
                      / effective_weights.sum())
        if metric_evaluator is not None:
            paired_acc = metric_evaluator(np.arange(q), c_indices, effective_weights)
            if np.asarray(paired_acc).shape != (q,) or not np.isfinite(paired_acc).all():
                raise ValueError("Nonfinite or incorrectly shaped metric contrast")
        return paired_loss, paired_acc, traces

    point, metric_point, fold_trace = crossfit(w, trace=True)
    final_c, final_trace = _reconstruct_comparator(loss, w, final_inner, open_ids, candidates)
    rng = np.random.default_rng(seed)
    loss_draws = np.empty((reps, q), dtype=np.float64)
    acc_draws = np.empty((reps, q), dtype=np.float64)
    choices_count = np.zeros(q, dtype=np.int64)
    rejected = 0
    completed = 0
    # Reject only structurally undefined fold resamples. This affects the finite
    # bootstrap approximation and is recorded; it never creates a coverage claim.
    while completed < reps:
        if rejected > max(1000, 10 * reps):
            raise ValueError("too few cores per fold for stable nested resampling")
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        bw = w * counts
        try:
            d, a, _ = crossfit(bw)
            boot_c, _ = _reconstruct_comparator(loss, bw, final_inner, open_ids, candidates)
        except ValueError:
            rejected += 1
            continue
        loss_draws[completed], acc_draws[completed] = d, a
        choices_count[boot_c] += 1
        completed += 1
    lower, upper, se, critical = _max_t_interval(point, loss_draws, alpha)
    metric_lower, metric_upper, metric_se, metric_critical = _max_t_interval(metric_point, acc_draws, alpha)
    metric_ok = metric_lower >= -epsilon_metric
    joint_feasible = feasible & metric_ok
    # qC has zero paired contrast to itself; including its implicit zero in the
    # near-best reference handles the no-benefit case without a fake candidate.
    feedback_feasible = joint_feasible & np.asarray([
        not bool(c.get("is_open_loop", False)) for c in candidates
    ])
    minimum_upper = min(0.0, float(np.min(upper[feedback_feasible]))) if feedback_feasible.any() else 0.0
    eligible = [
        j for j, c in enumerate(candidates)
        if not bool(c.get("is_open_loop", False))
        and joint_feasible[j]
        and upper[j] < -epsilon_r
        and upper[j] <= minimum_upper + epsilon_eq
    ]
    provisional = min(eligible, key=lambda j: _candidate_key(candidates[j], j)) if eligible else final_c
    return {
        "comparator_index": int(final_c),
        "qC_index": int(final_c),
        "provisional_index": int(provisional),
        "eligible_indices": [int(j) for j in eligible],
        "Q_s": [int(j) for j in eligible],
        "mean": point.tolist(), "lower": lower.tolist(), "upper": upper.tolist(),
        "L": lower.tolist(), "U": upper.tolist(), "se": se.tolist(),
        "metric_mean": metric_point.tolist(), "metric_lower": metric_lower.tolist(),
        "metric_upper": metric_upper.tolist(), "metric_se": metric_se.tolist(),
        "metric_feasible": metric_ok.tolist(), "joint_feasible": joint_feasible.tolist(),
        "minimum_upper": float(minimum_upper),
        "max_t_critical": float(critical), "metric_max_t_critical": float(metric_critical),
        "max_radius": float(np.max(upper - point)),
        "epsilon_r": float(epsilon_r), "epsilon_eq": float(epsilon_eq),
        "epsilon_metric": float(epsilon_metric), "alpha": float(alpha),
        "outer_folds": folds.tolist(), "inner_folds": inner.tolist(),
        "nested_comparator_fold_choices": fold_trace,
        "final_comparator_selection": final_trace,
        "bootstrap_final_comparator_counts": choices_count.tolist(),
        "bootstrap_replicates": int(reps), "bootstrap_seed": int(seed),
        "structurally_rejected_resamples": int(rejected),
        "candidate_count": int(q), "eligible_count": len(eligible), "units": int(n),
        "observed_candidate_count": int(q), "observations_available": [True] * q,
        "execution_failure_indices": [],
        "weighting": "equal_unit" if np.all(w == w[0]) else "fixed_unit_weight_ratio",
        "method": "nested family/configuration paired-core max-t bootstrap; descriptive",
        "independent_graph_guarantee": False,
        "candidate_predictions_fixed_before_selection": True,
        **({"metric_evaluation": "benchmark metric recomputed from whole-unit resampled observations"}
           if metric_evaluator is not None else {}),
        "bootstrap_preserves_original_unit_fold_membership": True,
        "open_loop_contrasts_are_diagnostic": True,
        "bootstrap_resamples_reconstruct_comparator": True,
        "rule": "main Eq19: U < -epsilon_r; U <= min_U + epsilon_eq; lexicographic simplicity",
    }


def holm_adjust(p_values: Any) -> np.ndarray:
    """Holm step-down adjusted P values in original contrast order."""
    p = _vector(p_values, "p_values")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("P values must lie in [0,1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p), dtype=np.float64)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return adjusted


def confirmation(
    provisional_index: int,
    comparator_index: int,
    eligible_indices: Sequence[int],
    contrasts: Sequence[Mapping[str, Any]],
    alpha: float = 0.05,
    reps: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Frozen Supplementary S11 map with a declared finite Holm family.

    Each contrast contains kind (state/radius/mpc/full), candidate_index,
    predecessor_index, differences and optional weights/margin/p_value.
    State differences are destroyed-minus-intact losses; full differences are
    Full-minus-Lean retained fractions (positive is beneficial). Radius/MPC
    differences are candidate-minus-predecessor losses (negative beneficial).
    Caller-supplied p_value must be the predeclared one-sided test against ZERO.

    All contrasts are tested before any downgrade. A predecessor is legal only
    inside Q_s or at qC; missing evidence fails closed. The family here is the
    actually supplied contrasts, without adding unexecuted experimental tests.

    A required contrast may instead declare available=False and provide an
    unavailable_reason, without differences. It remains in the Holm family with
    p=1, no estimated effect/interval and zero observed units. Its registered
    predecessor is retained, but the ordinary Q_s-or-qC downgrade constraint
    still applies. An explicitly invalid intervention also receives p=1 and
    cannot pass; any available differences are reported only as diagnostics,
    with intervention_valid/invalid_reason preserved in the output row.
    """
    kinds = {
        "state": (1, 0.002), "state_destruction": (1, 0.002),
        "radius": (-1, 0.0015), "radius_vs_one_step": (-1, 0.0015),
        "mpc": (-1, 0.002), "mpc_rh_vs_ol": (-1, 0.002),
        "full": (1, 0.02), "full_vs_lean": (1, 0.02),
    }
    rows = []
    pvals = []
    for k, contrast in enumerate(contrasts):
        kind = str(contrast["kind"])
        if kind not in kinds:
            raise ValueError(f"unknown confirmation contrast kind {kind!r}")
        direction, default_margin = kinds[kind]
        margin = float(contrast.get("margin", default_margin))
        available = bool(contrast.get("available", True))
        row = dict(
            kind=kind,
            candidate_index=int(contrast["candidate_index"]),
            predecessor_index=int(contrast.get("predecessor_index", comparator_index)),
            beneficial_direction=direction, practical_margin=margin, available=available,
        )
        if not available:
            row.update(mean=None, lower=None, upper=None, units=0,
                       valid_interval=False, point_beyond_margin=False,
                       p_value_zero=1.0, p_value_margin=None,
                       unavailable_reason=str(contrast.get("unavailable_reason",
                                                          "required contrast observations unavailable")))
        else:
            stat = paired_bootstrap(contrast["differences"], contrast.get("weights"),
                                    reps=reps, alpha=0.025, seed=seed + k)
            p = contrast.get("p_value")
            if p is None:
                p = stat["p_greater_zero" if direction > 0 else "p_less_zero"]
            p = 1.0 if p is None else float(p)
            # Practical-margin evidence is reported separately. It is not the
            # S11 retention test, which also requires significance at zero.
            shifted = direction * _vector(contrast["differences"], "differences") - margin
            against_margin = paired_bootstrap(shifted, contrast.get("weights"), reps=reps,
                                              alpha=0.025, seed=seed + 10_000 + k)
            row.update(mean=float(stat["mean"]), lower=stat["lower"], upper=stat["upper"],
                       units=stat["units"], point_beyond_margin=bool(direction * stat["mean"] > margin),
                       p_value_zero=p, p_value_margin=against_margin["p_greater_zero"],
                       valid_interval=bool(stat["valid_interval"]))
        if "intervention_valid" in contrast:
            row["intervention_valid"] = bool(contrast["intervention_valid"])
            if not row["intervention_valid"]:
                row["p_value_zero"] = 1.0
        if "invalid_reason" in contrast:
            row["invalid_reason"] = contrast["invalid_reason"]
        rows.append(row)
        pvals.append(row["p_value_zero"])
    adjusted = holm_adjust(pvals) if pvals else []
    for row, p in zip(rows, adjusted):
        row["holm_p_value_zero"] = float(p)
        row["passes"] = bool(row["available"] and row.get("intervention_valid", True)
                             and row["valid_interval"] and row["point_beyond_margin"] and p < alpha)
    eligible = set(int(x) for x in eligible_indices)
    current = int(provisional_index)
    events = []
    seen: set[int] = set()
    if current != comparator_index and current not in eligible:
        events.append({"from": current, "to": int(comparator_index), "reason": "provisional outside Q_s"})
        current = int(comparator_index)
    while current != comparator_index:
        if current in seen:
            events.append({"from": current, "to": int(comparator_index), "reason": "cyclic predecessor map"})
            current = int(comparator_index)
            break
        seen.add(current)
        current_rows = [row for row in rows if row["candidate_index"] == current]
        if not current_rows:
            events.append({"from": current, "to": int(comparator_index), "reason": "missing confirmation evidence"})
            current = int(comparator_index)
            break
        failed = [row for row in current_rows if not row["passes"]]
        if not failed:
            break
        predecessor = int(failed[0]["predecessor_index"])
        if predecessor not in eligible and predecessor != comparator_index:
            predecessor = int(comparator_index)
        events.append({"from": current, "to": predecessor,
                       "reason": f"failed {failed[0]['kind']} confirmation or equality at practical margin"})
        current = predecessor
    return dict(
        confirmed_index=int(current), qM_index=int(current),
        provisional_index=int(provisional_index), comparator_index=int(comparator_index),
        contrasts=rows, downgrades=events, alpha=float(alpha), holm_family_size=len(rows),
        holm_family="predeclared contrasts supplied for the registered dataset and policy family",
        membership_in_Qs_or_qC=bool(current in eligible or current == comparator_index),
        method="held-out point-margin plus Holm significance against zero; Supplementary S11",
        independent_graph_guarantee=False,
    )


def empirical_bernstein(
    differences: Any,
    clip: float = 0.10,
    alpha: float = 0.025,
    independent_graphs: bool = False,
) -> dict[str, Any]:
    """Main Eq.(27); conditional bound for independent graphs, diagnostic for dependent cores.

    Clipping is applied to the paired unit difference, not separately to the two
    losses. Even independent_graphs=True requires fresh IID graphs and a
    policy frozen before admission; this flag does not verify those assumptions.
    """
    if clip <= 0 or not 0 < alpha < 1:
        raise ValueError("clip must be positive and alpha between zero and one")
    raw = _vector(differences, "differences")
    d = np.clip(raw, -clip, clip)
    n = len(d)
    mean = float(np.mean(d))
    if n < 2:
        variance = penalty = upper = None
    else:
        variance = float(np.var(d, ddof=1))
        logterm = math.log(2 / alpha)
        penalty = math.sqrt(2 * variance * logterm / n) + 14 * clip * logterm / (3 * (n - 1))
        upper = mean + penalty
    return dict(
        mean=mean, sample_variance=variance, penalty=penalty, upper=upper,
        units=n, clip=float(clip), alpha=float(alpha),
        clipping_fraction=float(np.mean(np.abs(raw) > clip)),
        raw_maximum_harm=float(np.max(raw)),
        independent_graph_assumption_declared=bool(independent_graphs),
        applicability=("conditional on fresh IID graphs and frozen policies; caller must verify"
                       if independent_graphs else "diagnostic only: dependent graph cores give no coverage guarantee"),
    )


def admission(
    differences: Any,
    metric_differences: Any = None,
    weights: Any = None,
    reps: int = 10_000,
    alpha: float = 0.025,
    epsilon_a: float = 0.003,
    epsilon_metric: float = 0.001,
    clip: float = 0.10,
    seed: int = 0,
    metric_summary: Any = None,
    independent_graphs: bool = False,
) -> dict[str, Any]:
    """Dataset-appropriate admission, main Eq.(27) and Supplementary Table S1.

    Dependent cores require clipped mean<=-epsilon_a and descriptive upper<0.
    Independent graph losses instead require the conditional empirical-Bernstein
    upper<=-epsilon_a. Both require metric lower>=-epsilon_metric.
    Missing or degenerate metric evidence fails closed.
    Caller must return the stored comparator output exactly when passed=False.
    """
    if clip <= 0:
        raise ValueError("clip must be positive")
    raw = _vector(differences, "differences")
    w = _weights(weights, len(raw))
    clipped = np.clip(raw, -clip, clip)
    stat = paired_bootstrap(clipped, weights, reps, alpha, seed)
    raw_stat = paired_bootstrap(raw, weights, reps, alpha, seed)
    metric = None
    if metric_differences is not None:
        m = _vector(metric_differences, "metric_differences")
        if m.shape != raw.shape:
            raise ValueError("loss and metric differences must have the same core identities")
        metric = paired_bootstrap(m, weights, reps, alpha, seed)
    if metric_summary is not None:
        metric = metric_summary
    bound = None
    if independent_graphs:
        if not np.allclose(w, w[0]):
            raise ValueError("Independent-graph admission requires equally weighted graph losses")
        bound = empirical_bernstein(raw, clip=clip, alpha=alpha, independent_graphs=True)
    mean_gate = bool(stat["mean"] <= -epsilon_a)
    upper_gate = bool(stat["valid_interval"] and stat["upper"] < 0)
    if independent_graphs:
        upper_gate = bool(bound["upper"] is not None and bound["upper"] <= -epsilon_a)
    metric_gate = bool(metric is not None and metric["valid_interval"] and metric["lower"] is not None
                       and metric["lower"] >= -epsilon_metric)
    passed = mean_gate and upper_gate and metric_gate
    return dict(
        passed=bool(passed), Gadmit=int(passed),
        loss=stat, raw_loss=raw_stat, metric=metric,
        mean_gate=mean_gate, upper_gate=upper_gate, metric_gate=metric_gate,
        clip=float(clip), epsilon_a=float(epsilon_a), epsilon_metric=float(epsilon_metric),
        clipping_fraction=float(np.mean(np.abs(raw) > clip)),
        clipping_weight_fraction=float(np.dot(w, np.abs(raw) > clip) / w.sum()),
        raw_maximum_harm=float(np.max(raw)),
        raw_minimum_effect=float(np.min(raw)),
        independent_graph_guarantee=False,
        statistical_regime=("independent-graph clipped-loss bound; conditional on fresh graphs and frozen policies"
                            if independent_graphs else "one transductive graph; dependent buffered cores; descriptive admission"),
        rule=("empirical-Bernstein upper <= -epsilon_a AND metric lower >= -epsilon_metric"
              if independent_graphs else "clipped mean <= -epsilon_a AND bootstrap upper < 0 AND metric lower >= -epsilon_metric"),
        **({"empirical_bernstein": bound, "conditional_graph_bound": True} if independent_graphs else {}),
        fallback_rule="Gadmit=0 returns the exact stored qC predictions",
    )


# Readable aliases for entry points without changing the documented contracts.
confirm = confirmation
admit = admission



def global_candidate_oracle(losses, counts):
    """Best one executable global candidate, with core minima diagnostic only.

    This is a finite evaluated-family, development-label oracle; not exact
    sequence enumeration and not an out-of-sample performance guarantee.
    """
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Oracle losses must be finite [candidates, cores]")
    weights = _weights(counts, values.shape[1])
    means = values @ weights / weights.sum()
    winner = int(np.argmin(means))
    return winner, values[winner].copy(), np.min(values, axis=0)
