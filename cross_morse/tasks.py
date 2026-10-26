"""Task mathematics shared by training, control targets, and numeric replay.

NumPy-only import is intentional: reviewing saved observations must not import
PyTorch or deserialize a checkpoint. Tensor helpers import torch lazily.

Dataset metrics follow the primary benchmark definitions:
https://ogb.stanford.edu/docs/nodeprop/ and
https://github.com/vijaydwivedi75/lrgb . ROC-AUC and average precision are macro
averages over observed tasks containing both positive and negative examples.
Macro F1 uses the declared output-class universe, including zero-support classes.
All metric values are fractions on [0, 1], never percentages. Nondecomposable
metrics must be recomputed from observations in each resample; an average of
per-node F1/AUC/AP surrogates is not an official benchmark metric.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping
import math

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    dataset: str = "ogbn-arxiv"
    task_type: str = "multiclass"
    prediction_level: str = "node"
    metric: str = "accuracy"
    num_outputs: int | None = None

    def __post_init__(self):
        if self.task_type not in {"multiclass", "multilabel"} or self.prediction_level not in {"node", "graph"}:
            raise ValueError("Unknown task type or prediction level")
        expected = {"accuracy", "macro_f1"} if self.task_type == "multiclass" else {"rocauc", "ap"}
        if self.metric not in expected:
            raise ValueError("Metric is incompatible with the task type")
        if self.num_outputs is not None and (int(self.num_outputs) != self.num_outputs or self.num_outputs < 1):
            raise ValueError("num_outputs must be a positive integer")

    @property
    def label_axis(self):
        return self.prediction_level

    @property
    def loss_name(self):
        return "bce" if self.task_type == "multilabel" else "ce"

    @property
    def metric_decomposable(self):
        return self.metric == "accuracy"

    def to_dict(self):
        return asdict(self)


_DATASETS = {
    "ogbn-arxiv": ("multiclass", "node", "accuracy", 40),
    "ogbn-products": ("multiclass", "node", "accuracy", 47),
    "ogbn-proteins": ("multilabel", "node", "rocauc", 112),
    "Roman-empire": ("multiclass", "node", "accuracy", 18),
    "PascalVOC-SP": ("multiclass", "node", "macro_f1", 21),
    "COCO-SP": ("multiclass", "node", "macro_f1", 81),
    "Peptides-func": ("multilabel", "graph", "ap", 10),
    "synthetic": ("multiclass", "node", "accuracy", None),
}
_ALIASES = {name.lower().replace("_", "-"): name for name in _DATASETS}


def task_spec(config_or_name=None, metadata=None) -> TaskSpec:
    """Resolve canonical task semantics without reading labels.

    Synthetic fixtures may set ``benchmark_dataset``. Explicit metadata controls
    output cardinality, so a small fixture need not have all official classes.
    Metadata accepts ``prediction_level`` or the equivalent ``label_axis``.
    """
    if isinstance(config_or_name, TaskSpec):
        return config_or_name
    config = dict(config_or_name) if isinstance(config_or_name, Mapping) else {}
    meta = dict(metadata or {})
    raw_name = (config.get("benchmark_dataset") or config.get("dataset")
                or (config_or_name if isinstance(config_or_name, str) else None)
                or meta.get("benchmark_dataset") or meta.get("dataset") or "ogbn-arxiv")
    name = _ALIASES.get(str(raw_name).lower().replace("_", "-"))
    if name is None:
        raise ValueError(f"Unsupported task dataset: {raw_name}")
    kind, level, metric, outputs = _DATASETS[name]
    if config.get("dataset") == "synthetic":
        outputs = config.get("data", {}).get("synthetic_classes", outputs)
    explicit = config.get("task", {})
    if isinstance(explicit, str):
        explicit = {"task_type": explicit}
    if not isinstance(explicit, Mapping):
        raise ValueError("task configuration must be a mapping or task-type string")
    values = {key: config[key] for key in ("task_type", "prediction_level", "label_axis", "metric",
                                          "num_outputs", "num_classes", "num_tasks") if key in config}
    values.update(explicit)
    values.update(meta.get("task", {}) if isinstance(meta.get("task"), Mapping) else {})
    values.update(meta)
    kind = str(values.get("task_type", kind))
    level = str(values.get("prediction_level", values.get("label_axis", level)))
    metric = str(values.get("metric", metric)).lower().replace("-", "_")
    metric = {"roc_auc": "rocauc", "average_precision": "ap", "f1": "macro_f1", "acc": "accuracy"}.get(metric, metric)
    outputs = values.get("num_outputs", values.get("num_classes", values.get("num_tasks", outputs)))
    if kind not in {"multiclass", "multilabel"} or level not in {"node", "graph"}:
        raise ValueError("Unknown task type or prediction level")
    if metric not in ({"accuracy", "macro_f1"} if kind == "multiclass" else {"rocauc", "ap"}):
        raise ValueError("Metric is incompatible with the task type")
    if outputs is not None and (int(outputs) != outputs or int(outputs) < 1):
        raise ValueError("num_outputs must be a positive integer")
    return TaskSpec(name, kind, level, metric, None if outputs is None else int(outputs))


def probabilities(logits, spec=None):
    """Softmax for multiclass tasks; independent sigmoids for multilabel tasks."""
    spec = task_spec(spec)
    if isinstance(logits, np.ndarray) or not hasattr(logits, "detach"):
        x = np.asarray(logits)
        if spec.task_type == "multilabel":
            out = np.empty_like(x, dtype=np.result_type(x.dtype, np.float32))
            positive = x >= 0
            out[positive] = 1 / (1 + np.exp(-x[positive]))
            ex = np.exp(x[~positive])
            out[~positive] = ex / (1 + ex)
            return out
        shifted = x - x.max(axis=-1, keepdims=True)
        ex = np.exp(shifted)
        return ex / ex.sum(axis=-1, keepdims=True)
    x = logits.float()
    return x.sigmoid() if spec.task_type == "multilabel" else x.softmax(-1)


def valid_target_mask(targets, spec=None):
    """Observed-label mask; NaN and negative missing-label sentinels are omitted.

    Infinities and nonnegative invalid class/binary values are malformed labels,
    not missing observations. The function returns the same array backend as its
    input (NumPy for lists). Multiclass accepts [N] or [N,1]; multilabel is [N,T].
    """
    spec = task_spec(spec)
    if isinstance(targets, np.ndarray) or not hasattr(targets, "detach"):
        y = np.asarray(targets)
        if spec.task_type == "multiclass" and y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != (1 if spec.task_type == "multiclass" else 2):
            raise ValueError("Target shape is incompatible with prediction rows/task type")
        if np.isinf(y).any():
            raise ValueError("Infinite target values are invalid")
        mask = np.isfinite(y) & (y >= 0)
        observed = y[mask]
        if spec.task_type == "multiclass":
            if np.any(observed != np.floor(observed)) or (spec.num_outputs is not None and np.any(observed >= spec.num_outputs)):
                raise ValueError("Multiclass target outside the declared class universe")
        elif np.any((observed != 0) & (observed != 1)):
            raise ValueError("Multilabel targets must be binary or missing")
        return mask
    import torch
    y = targets
    if spec.task_type == "multiclass" and y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != (1 if spec.task_type == "multiclass" else 2):
        raise ValueError("Target shape is incompatible with prediction rows/task type")
    if torch.isinf(y).any():
        raise ValueError("Infinite target values are invalid")
    mask = torch.isfinite(y) & (y >= 0)
    observed = y[mask]
    if spec.task_type == "multiclass":
        if torch.any(observed != observed.floor()) or (spec.num_outputs is not None and torch.any(observed >= spec.num_outputs)):
            raise ValueError("Multiclass target outside the declared class universe")
    elif torch.any((observed != 0) & (observed != 1)):
        raise ValueError("Multilabel targets must be binary or missing")
    return mask


def valid_sample_mask(targets, spec=None):
    spec = task_spec(spec)
    mask = valid_target_mask(targets, spec)
    return mask.any(-1) if spec.task_type == "multilabel" else mask


def loss(logits, targets, spec=None, reduction="mean"):
    """Differentiable mean CE/BCE, averaging observed tasks within each sample.

    Fully unlabelled rows have NaN in reduction='none' and are excluded from the
    mean. An entirely unlabelled batch raises, rather than reporting zero loss.
    With complete multiclass labels this calls the original PyTorch CE directly.
    """
    import torch
    import torch.nn.functional as F
    spec = task_spec(spec)
    if reduction not in {"none", "mean", "sum"} or logits.ndim != 2:
        raise ValueError("Expected matrix logits and none/mean/sum reduction")
    if logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("Cannot compute loss without prediction rows and outputs")
    y = torch.as_tensor(targets, device=logits.device)
    if spec.task_type == "multiclass" and y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    mask = valid_target_mask(y, spec)
    if len(y) != len(logits) or (spec.num_outputs is not None and logits.shape[1] != spec.num_outputs):
        raise ValueError("Prediction/target output shape mismatch")
    if spec.task_type == "multiclass":
        if mask.all():
            return F.cross_entropy(logits.float(), y.long(), reduction=reduction)
        safe_y = torch.where(mask, y, torch.zeros_like(y)).long()
        values = F.cross_entropy(logits.float(), safe_y, reduction="none")
        rows = mask
    else:
        if y.shape != logits.shape:
            raise ValueError("Multilabel targets must match logits")
        safe_y = torch.where(mask, y, torch.zeros_like(y)).float()
        entries = F.binary_cross_entropy_with_logits(logits.float(), safe_y, reduction="none")
        values = (entries * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        rows = mask.any(-1)
    if reduction == "none":
        return torch.where(rows, values, values.new_full(values.shape, float("nan")))
    if not rows.any():
        raise ValueError("Cannot compute loss without observed targets")
    return values[rows].mean() if reduction == "mean" else values[rows].sum()


def _numeric_observations(p, targets, spec):
    p, y = np.asarray(p), np.asarray(targets)
    if spec.task_type == "multiclass" and y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    mask = valid_target_mask(y, spec)
    if p.ndim != 2 or len(p) != len(y) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Expected finite probability rows aligned with targets")
    if spec.num_outputs is not None and p.shape[1] != spec.num_outputs:
        raise ValueError("Probabilities differ from the declared output universe")
    if spec.task_type == "multiclass":
        if not np.allclose(p.sum(-1), 1, atol=1e-5, rtol=1e-6):
            raise ValueError("Multiclass probabilities must sum to one")
        if np.any(y[mask] >= p.shape[1]):
            raise ValueError("Target class outside probability matrix")
    elif p.shape != y.shape:
        raise ValueError("Multilabel probabilities and labels must have equal shapes")
    return p, y, mask


def sample_losses(p, targets, spec=None):
    """NumPy CE/BCE per prediction row; NaN identifies fully unlabelled rows."""
    spec = task_spec(spec)
    p, y, mask = _numeric_observations(p, targets, spec)
    if spec.task_type == "multiclass":
        # Preserve the arxiv probability dtype and exact historical NLL path.
        result = np.full(len(p), np.nan, dtype=np.result_type(p.dtype, np.float32))
        result[mask] = -np.log(np.maximum(p[np.where(mask)[0], y[mask].astype(int)], 1e-30))
        return result
    pf = p.astype(np.float64, copy=False)
    safe_y = np.where(mask, y, 0)
    entries = -(safe_y * np.log(np.maximum(pf, 1e-30))
                + (1 - safe_y) * np.log(np.maximum(1 - pf, 1e-30)))
    count = mask.sum(-1)
    return np.divide((entries * mask).sum(-1), count,
                     out=np.full(len(p), np.nan), where=count > 0)


def sample_metric_values(p, targets, spec=None):
    """Decomposable accuracy only; rank/F1 metrics need complete observations."""
    spec = task_spec(spec)
    if not spec.metric_decomposable:
        raise ValueError(f"{spec.metric} has no per-sample additive metric; recompute metric_score in each resample")
    p, y, mask = _numeric_observations(p, targets, spec)
    return np.where(mask, (p.argmax(-1) == y).astype(float), np.nan)


def _binary_metric(scores, labels, weights, metric):
    positive = float(weights[labels == 1].sum())
    negative = float(weights[labels == 0].sum())
    if positive <= 0 or negative <= 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    s, y, w = scores[order], labels[order], weights[order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    pos = np.add.reduceat(w * (y == 1), starts)
    neg = np.add.reduceat(w * (y == 0), starts)
    if metric == "rocauc":
        preceding_negative = np.cumsum(neg) - neg
        return float(np.dot(pos, preceding_negative + 0.5 * neg) / (positive * negative))
    pos, neg = pos[::-1], neg[::-1]
    precision = np.cumsum(pos) / np.cumsum(pos + neg)
    return float(np.dot(pos / positive, precision))


def metric_score(p, targets, spec=None, sample_weight=None):
    """Official task metric with optional bootstrap multiplicity weights.

    Undefined rank metrics return NaN if no task has both observed classes.
    Callers must treat that as unavailable evidence, never substitute zero.
    F1 uses all declared output classes and zero for classes with zero denominator.
    """
    spec = task_spec(spec)
    p, y, mask = _numeric_observations(p, targets, spec)
    w = np.ones(len(p)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if w.shape != (len(p),) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Sample weights must be aligned finite nonnegative values")
    if spec.task_type == "multiclass":
        keep = mask & (w > 0)
        if not keep.any():
            return float("nan")
        predicted, labels, weight = p[keep].argmax(-1), y[keep].astype(int), w[keep]
        if spec.metric == "accuracy":
            return float(np.dot(weight, predicted == labels) / weight.sum())
        classes = spec.num_outputs or p.shape[1]
        true_count = np.bincount(labels, weights=weight, minlength=classes)
        pred_count = np.bincount(predicted, weights=weight, minlength=classes)
        same = predicted == labels
        true_positive = np.bincount(labels[same], weights=weight[same], minlength=classes)
        denominator = true_count + pred_count
        return float(np.divide(2 * true_positive, denominator, out=np.zeros(classes),
                               where=denominator > 0).mean())
    per_task = []
    for column in range(p.shape[1]):
        keep = mask[:, column] & (w > 0)
        value = _binary_metric(p[keep, column], y[keep, column], w[keep], spec.metric)
        if np.isfinite(value):
            per_task.append(value)
    return float(np.mean(per_task)) if per_task else float("nan")


def confidence_statistics(p, p0, spec=None):
    """Per-row normalized entropy, confidence margin, and source JS distance.

    Multilabel features average Bernoulli quantities over outputs. Their margin
    is mean |2p-1|, not a top-two softmax gap between unrelated binary tasks.
    """
    import torch
    spec = task_spec(spec)
    p, p0 = p.float(), p0.float()
    if p.shape != p0.shape or p.ndim != 2:
        raise ValueError("Current/source probabilities must be aligned matrices")
    eps = 1e-12
    if spec.task_type == "multilabel":
        def entropy_of(q):
            return -(q * q.clamp_min(eps).log() + (1 - q) * (1 - q).clamp_min(eps).log())
        entropy = entropy_of(p).mean(-1) / math.log(2)
        margin = (2 * p - 1).abs().mean(-1)
        js = (entropy_of((p + p0) / 2) - (entropy_of(p) + entropy_of(p0)) / 2).mean(-1) / math.log(2)
    else:
        p, p0 = p.clamp_min(eps), p0.clamp_min(eps)
        entropy = -(p * p.log()).sum(-1) / max(math.log(p.shape[1]), eps)
        margin = (p.topk(2, dim=-1).values.diff(dim=-1).neg().squeeze(-1)
                  if p.shape[1] > 1 else p.new_ones(len(p)))
        mean_p = 0.5 * (p + p0)
        js = (0.5 * (p * (p.log() - mean_p.log())).sum(-1)
              + 0.5 * (p0 * (p0.log() - mean_p.log())).sum(-1)) / math.log(2)
    return entropy, margin, js.clamp_min(0)
