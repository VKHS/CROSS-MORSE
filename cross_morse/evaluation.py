"""Task-aware scores and whole-unit resampling of benchmark metrics.

ROC-AUC, average precision and macro F1 are recomputed from the resampled
observations. They are never replaced by threshold accuracy or an average of
undefined per-node scores. This module does not import PyTorch.
"""
from __future__ import annotations

import numpy as np

from .tasks import task_spec, sample_losses, sample_metric_values, valid_sample_mask, metric_score
from .statistics import cluster_summary


GRAPH_DATASETS = frozenset({"PascalVOC-SP", "COCO-SP", "Peptides-func"})


def independent_graphs(config):
    return config.get("benchmark_dataset", config.get("dataset")) in GRAPH_DATASETS


def score_observations(probabilities, labels, unit_ids, config):
    spec = task_spec(config)
    values = sample_losses(probabilities, labels, spec)
    valid = np.isfinite(values)
    if not valid.any():
        raise ValueError("The scoring role contains no observed targets")
    means, units, sample_counts = cluster_summary(values[valid], np.asarray(unit_ids)[valid])
    weights = np.ones(len(units), dtype=np.int64) if independent_graphs(config) else sample_counts
    metric = float(metric_score(probabilities, labels, spec))
    accuracy = None
    if spec.metric == "accuracy":
        correct = sample_metric_values(probabilities, labels, spec)
        accuracy, check_units, _ = cluster_summary(correct[valid], np.asarray(unit_ids)[valid])
        if not np.array_equal(check_units, units):
            raise ValueError("Metric and loss units differ")
    return {"loss": means, "accuracy": accuracy, "core_ids": units,
            "counts": weights, "sample_counts": sample_counts,
            "node_ce": float(np.average(means, weights=weights)),
            "node_accuracy": metric, "metric": metric, "metric_name": spec.metric,
            "loss_name": "binary_cross_entropy" if spec.task_type == "multilabel" else "cross_entropy"}


class MetricEvidence:
    """Paired prediction evidence for an exact nonlinear metric bootstrap.

The prediction list is candidate-major. Units must match the loss matrix's
sorted unit order. Effective loss weights are converted back to bootstrap
frequencies before expanding to samples; graph size therefore does not get
counted twice, and the benchmark evaluator retains its declared definition.
"""

    def __init__(self, predictions, labels, sample_units, units, unit_weights, config):
        self.predictions = predictions
        self.labels = np.asarray(labels)
        self.units = np.asarray(units)
        self.unit_weights = np.asarray(unit_weights, dtype=float)
        self.spec = task_spec(config)
        sample_units = np.asarray(sample_units)
        self.observed = valid_sample_mask(self.labels, self.spec)
        if not len(self.units) or sample_units.shape != self.observed.shape:
            raise ValueError("Prediction observations do not match the scored statistical units")
        self.inverse = np.searchsorted(self.units, sample_units)
        matched = self.inverse < len(self.units)
        self.inverse = np.minimum(self.inverse, len(self.units)-1)
        matched &= self.units[self.inverse] == sample_units
        if (len(self.unit_weights) != len(self.units) or (self.unit_weights <= 0).any()
                or not np.all(matched[self.observed])):
            raise ValueError("Prediction observations do not match the scored statistical units")
        if any(np.asarray(p).shape[0] != len(self.labels) for p in predictions):
            raise ValueError("All candidates must have the same ordered observation rows")

    def __call__(self, candidate_indices, comparator_indices, effective_weights):
        frequencies = np.asarray(effective_weights, dtype=float) / self.unit_weights
        weights = frequencies[self.inverse] * self.observed
        comparator_indices = np.asarray(comparator_indices, dtype=np.int64)
        if comparator_indices.shape != self.units.shape:
            raise ValueError("One cross-fitted comparator index is required per statistical unit")
        chosen = comparator_indices[self.inverse]
        reference = np.empty_like(np.asarray(self.predictions[int(candidate_indices[0])]))
        for candidate in np.unique(chosen[weights > 0]):
            mask = chosen == candidate
            reference[mask] = np.asarray(self.predictions[int(candidate)])[mask]
        # Zero-weight rows are ignored by the metric. Initialize them to a valid
        # probability so evaluator shape/range validation remains meaningful.
        reference[weights <= 0] = np.asarray(self.predictions[int(candidate_indices[0])])[weights <= 0]
        baseline = float(metric_score(reference, self.labels, self.spec, sample_weight=weights))
        values = [float(metric_score(self.predictions[int(j)], self.labels, self.spec,
                                    sample_weight=weights)) - baseline for j in candidate_indices]
        if not np.isfinite([baseline, *values]).all():
            raise ValueError("The resampled benchmark metric is undefined")
        return np.asarray(values)


def paired_metric_interval(candidate, comparator, labels, sample_units, config, *,
                           reps=10000, alpha=.025, seed=0, scale=1.0):
    """Recompute a fixed-policy metric contrast in each whole-unit resample."""
    if reps < 20 or not 0 < alpha < .5:
        raise ValueError("At least 20 resamples and 0 < alpha < .5 are required")
    spec = task_spec(config)
    observed = valid_sample_mask(labels, spec)
    sample_units = np.asarray(sample_units)
    units = np.unique(sample_units[observed])
    inverse = np.zeros(len(sample_units), dtype=np.int64)
    inverse[observed] = np.searchsorted(units, sample_units[observed])
    candidate, comparator = np.asarray(candidate), np.asarray(comparator)
    mean = scale * (float(metric_score(candidate, labels, spec))
                    - float(metric_score(comparator, labels, spec)))
    base = {"mean": mean if np.isfinite(mean) else None, "units": len(units),
            "metric": spec.metric, "scale": float(scale), "bootstrap_replicates": int(reps),
            "bootstrap_seed": int(seed), "one_sided_alpha": float(alpha),
            "interval_level": float(1-2*alpha),
            "method": "paired whole-unit bootstrap with benchmark metric recomputation",
            "unit": "graph" if independent_graphs(config) else "dependent graph core"}
    if len(units) < 2 or not np.isfinite(mean):
        return dict(base, lower=None, upper=None, se=None, valid_interval=False,
                    rejected_resamples=0, reason="insufficient units or undefined metric")
    rng, draws, rejected = np.random.default_rng(seed), [], 0
    while len(draws) < reps and rejected <= max(1000, reps*10):
        counts = np.bincount(rng.integers(0, len(units), size=len(units)), minlength=len(units))
        w = counts[inverse] * observed
        effect = scale * (float(metric_score(candidate, labels, spec, sample_weight=w))
                          - float(metric_score(comparator, labels, spec, sample_weight=w)))
        if np.isfinite(effect):
            draws.append(effect)
        else:
            rejected += 1
    if len(draws) < reps:
        return dict(base, lower=None, upper=None, se=None, valid_interval=False,
                    rejected_resamples=rejected, reason="too many undefined metric resamples")
    return dict(base, lower=float(np.quantile(draws, alpha)), upper=float(np.quantile(draws, 1-alpha)),
                se=float(np.std(draws, ddof=1)), valid_interval=True, rejected_resamples=rejected)
