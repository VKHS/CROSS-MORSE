"""Read-only numeric verification for review: no PyTorch or pickle loading.

This checks the arithmetic of submitted observations and their byte identities.
Hashes are integrity identifiers, not independent proof of historical execution.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np

from .statistics import (cluster_summary, paired_bootstrap, nested_selection,
                         confirmation, admission)
from .tasks import task_spec, valid_target_mask
from .evaluation import (score_observations, MetricEvidence, paired_metric_interval, independent_graphs)

STAGES = ("prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate")


class EvidenceError(ValueError):
    """Incomplete, modified, or numerically inconsistent review evidence."""


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def safe_file(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or "\\" in relative:
        raise EvidenceError("Evidence paths must be relative POSIX paths")
    p = PurePosixPath(relative)
    if p.is_absolute() or p.as_posix() != relative or not p.parts or any(v in (".", "..") for v in p.parts):
        raise EvidenceError(f"Unsafe evidence path: {relative}")
    file = root.joinpath(*p.parts)
    if any(parent.is_symlink() for parent in (file, *list(file.parents)[:len(p.parts)-1])):
        raise EvidenceError(f"Symlinks are not evidence files: {relative}")
    if not file.resolve().is_relative_to(root) or not file.is_file():
        raise EvidenceError(f"Missing/escaping evidence file: {relative}")
    return file


def json_read(path):
    def reject_constant(value):
        raise EvidenceError(f"Nonstandard JSON constant: {value}")
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Cannot read {path}: {exc}") from exc


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def assert_matches(actual, expected, path="result", *, atol=1e-10):
    """Require the complete computed object; tolerate only floating arithmetic."""
    actual, expected = plain(actual), plain(expected)
    if isinstance(actual, dict):
        if not isinstance(expected, dict):
            raise EvidenceError(f"{path}: expected an object")
        for key, value in actual.items():
            if key not in expected:
                raise EvidenceError(f"{path}.{key}: missing recorded field")
            assert_matches(value, expected[key], f"{path}.{key}", atol=atol)
    elif isinstance(actual, list):
        if not isinstance(expected, list) or len(actual) != len(expected):
            raise EvidenceError(f"{path}: sequence length mismatch")
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert_matches(a, b, f"{path}[{i}]", atol=atol)
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if isinstance(expected, bool) or not isinstance(expected, (int, float)) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=atol):
            raise EvidenceError(f"{path}: recomputed {actual!r}, recorded {expected!r}")
    elif isinstance(actual, bool) and type(expected) is not bool:
        raise EvidenceError(f"{path}: expected a boolean")
    elif actual != expected:
        raise EvidenceError(f"{path}: recomputed {actual!r}, recorded {expected!r}")


def sealed_inventory(run_root):
    """Verify parent/input linkage and return recorded output descriptors.

    Byte checking is performed lazily for every numerical input actually used.
    Missing model binaries are permissible for an analysis-only review bundle;
    this function never claims to audit unprovided checkpoints.
    """
    root = Path(run_root)
    directory = root / "provenance/stages"
    candidates = list(directory.glob("*.json"))
    parsed = {json_read(path).get("stage"): path for path in candidates}
    if set(parsed) != set(STAGES) or len(candidates) != len(STAGES):
        raise EvidenceError("Eight distinct sealed lifecycle stages are required")
    sealed, first, previous = {}, None, None
    for stage in STAGES:
        path = parsed[stage]
        value = json_read(path)
        identities = [value.get(k) for k in ("source_sha256", "config_sha256", "unit_sha256", "dataset")]
        if first is None:
            first = identities
        elif identities != first:
            raise EvidenceError("Run source/configuration/units/dataset changed across stages")
        if value.get("parent_manifest") != previous:
            raise EvidenceError(f"Broken parent manifest at {stage}")
        for item in value.get("inputs", []):
            if stage != "prepare" and sealed.get(item["path"]) != item:
                raise EvidenceError(f"Input without earlier sealed output: {item['path']}")
        for item in value.get("outputs", []):
            if set(item) != {"path", "bytes", "sha256"}:
                raise EvidenceError("Invalid artifact descriptor")
            if item["path"] in sealed:
                raise EvidenceError(f"Repeated output identity: {item['path']}")
            sealed[item["path"]] = item
        previous = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}
    config_path = safe_file(root, "config.json")
    if canonical(json_read(config_path)) != first[1]:
        raise EvidenceError("Configuration differs from the stage identity")
    return sealed, {"source_sha256": first[0], "config_sha256": first[1],
                    "unit_sha256": first[2], "dataset": first[3]}


class NumericEvidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.sealed, self.identities = sealed_inventory(self.root)
        self.checked = set()

    def file(self, relative):
        item = self.sealed.get(relative)
        if item is None:
            raise EvidenceError(f"Unsealed input: {relative}")
        path = safe_file(self.root, relative)
        if relative not in self.checked:
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise EvidenceError(f"Evidence hash/size mismatch: {relative}")
            self.checked.add(relative)
        return path

    def json(self, relative):
        return json_read(self.file(relative))

    def arrays(self, relative):
        with np.load(self.file(relative), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}


def replay_results(run_root, *, selection=True):
    """Recompute task metrics, whole-unit resampling and sealed decisions."""
    e = NumericEvidence(run_root)
    c = e.json("config.json")
    spec = task_spec(c)
    summary = e.json("final/results_summary.json")
    predictions = e.arrays("final/test_predictions.npz")
    sample_ids, labels, core = (predictions[k] for k in ("node_ids", "labels", "cluster_ids"))
    n = len(sample_ids)
    if not n or sample_ids.ndim != 1 or len(np.unique(sample_ids)) != n or sample_ids.dtype.kind not in "iu":
        raise EvidenceError("Test sample IDs must be unique nonempty integers")
    if labels.shape[0] != n or core.shape != (n,) or core.dtype.kind not in "iu":
        raise EvidenceError("Labels and statistical units must align with test samples")
    valid_target_mask(labels, spec)
    splits = e.arrays("prepared/splits.npz")
    clusters = e.arrays("prepared/clusters.npz")["cluster_ids"]
    if not np.array_equal(sample_ids, splits["test"]):
        raise EvidenceError("Test observations differ from the sealed test split")
    if np.any(sample_ids < 0) or np.any(sample_ids >= len(clusters)) or not np.array_equal(core, clusters[sample_ids]):
        raise EvidenceError("Test membership differs from the sealed unit partition")
    if summary.get("test_prediction_record_sha256") != e.sealed["final/test_predictions.npz"]["sha256"]:
        raise EvidenceError("Summary does not bind the test prediction record")
    if c["dataset"] != e.identities["dataset"] or summary["dataset"] != c["dataset"]:
        raise EvidenceError("Dataset identity differs between observations and manifest")
    if "task" in summary:
        assert_matches(spec.to_dict(), summary["task"], "task")
    units = e.arrays("final/paired_test_units.npz")
    metrics, scores = {}, {}
    for name in ("q0", "qN", "qC", "qD"):
        p = predictions[f"{name}_probabilities"]
        score = scores[name] = score_observations(p, labels, core, c)
        row = {"metric_name": spec.metric, "metric": score["metric"], "metric_percent": 100*score["metric"],
               "loss_name": score["loss_name"], "loss_nats": score["node_ce"],
               "test_samples": n, "test_units": len(score["core_ids"]), "prediction_level": spec.prediction_level}
        if spec.metric == "accuracy":
            row.update(accuracy=score["metric"], accuracy_percent=100*score["metric"],
                       cross_entropy_nats=score["node_ce"], test_nodes=n)
        else:
            row[spec.metric] = score["metric"]
            row["binary_cross_entropy_nats" if spec.task_type == "multilabel" else "cross_entropy_nats"] = score["node_ce"]
        metrics[name] = row
        assert_matches(row, summary["models"][name], name)
        if name in ("qC", "qD"):
            for field, actual in ((f"{name}_loss", score["loss"]), ("core_ids", score["core_ids"]),
                                  ("counts", score["counts"])):
                assert_matches(actual, units[field], f"paired_test_units.{field}")
            if spec.metric == "accuracy":
                for field, actual in ((f"{name}_ce", score["loss"]), (f"{name}_accuracy", score["accuracy"])):
                    assert_matches(actual, units[field], f"paired_test_units.{field}")
    kwargs = dict(weights=scores["qC"]["counts"], reps=c["admission"]["bootstrap_reps"], alpha=.025, seed=c["seed"]+7)
    loss_contrast = paired_bootstrap(scores["qD"]["loss"] - scores["qC"]["loss"], **kwargs)
    if spec.metric == "accuracy":
        metric_contrast = paired_bootstrap(100*(scores["qD"]["accuracy"] - scores["qC"]["accuracy"]), **kwargs)
        assert_matches(loss_contrast, summary["paired_qD_minus_qC_ce"], "test_ce")
        assert_matches(metric_contrast, summary["paired_qD_minus_qC_accuracy_pp"], "test_accuracy_pp")
    else:
        metric_contrast = paired_metric_interval(predictions["qD_probabilities"], predictions["qC_probabilities"],
            labels, core, c, reps=c["admission"]["bootstrap_reps"], alpha=.025, seed=c["seed"]+7, scale=100)
    assert_matches(loss_contrast, summary["paired_qD_minus_qC_loss"], "test_loss")
    assert_matches(metric_contrast, summary["paired_qD_minus_qC_metric_pp"], "test_metric_pp")
    candidates = e.json("select/candidates.json")
    selected = e.json("select/decision.json")
    if selection:
        data = e.arrays("select/unit_scores.npz")
        sc, metric_evaluator = c["selection"], None
        if spec.metric != "accuracy":
            raw = e.arrays("select/metric_observations.npz")
            if not np.array_equal(raw["sample_ids"], splits["select"]) or not np.array_equal(raw["sample_units"], clusters[raw["sample_ids"]]):
                raise EvidenceError("Selection metric observations differ from the sealed split/units")
            example = next(raw[f"candidate_{j}"] for j, candidate in enumerate(candidates) if not candidate.get("execution_failure"))
            values = []
            for j, candidate in enumerate(candidates):
                if candidate.get("execution_failure"):
                    if f"candidate_{j}" in raw:
                        raise EvidenceError("A failed candidate has fabricated metric observations")
                    values.append(np.full_like(example, np.nan))
                else:
                    p = raw[f"candidate_{j}"]
                    check = score_observations(p, raw["labels"], raw["sample_units"], c)
                    assert_matches(check["loss"], data["loss"][:, j], f"selection_loss_{j}")
                    assert_matches(check["core_ids"], data["core_ids"], "selection_units")
                    assert_matches(check["counts"], data["counts"], "selection_weights")
                    values.append(p)
            metric_evaluator = MetricEvidence(values, raw["labels"], raw["sample_units"],
                                              data["core_ids"], data["counts"], c)
        recomputed = nested_selection(data["loss"], data.get("accuracy"), candidates, data["outer_fold"],
            inner_fold_ids=data["inner_fold"], reps=sc["bootstrap_reps"], alpha=sc["alpha"],
            epsilon_r=sc["epsilon_r"], epsilon_eq=sc["epsilon_eq"], epsilon_metric=sc["epsilon_metric"],
            weights=data["counts"], seed=c["seed"]+4, metric_evaluator=metric_evaluator)
        assert_matches(recomputed, selected, "selection")
    contrasts = e.json("confirm/paired_contrasts.json")
    confirmed = confirmation(selected["provisional_index"], selected["comparator_index"], selected["eligible_indices"],
                              contrasts, reps=c["selection"]["bootstrap_reps"], seed=c["seed"]+5)
    assert_matches(confirmed, e.json("confirm/decision.json"), "confirmation")
    a, ac = e.arrays("admit/paired_units.npz"), c["admission"]
    metric_summary = None
    if spec.metric != "accuracy":
        if not np.array_equal(a["sample_ids"], splits["admit"]) or not np.array_equal(a["sample_units"], clusters[a["sample_ids"]]):
            raise EvidenceError("Admission observations differ from the sealed split/units")
        for name in ("candidate", "comparator"):
            score = score_observations(a[f"{name}_probabilities"], a["labels"], a["sample_units"], c)
            assert_matches(score["loss"], a[f"{name}_loss"], f"admission_{name}_loss")
            assert_matches(score["core_ids"], a["core_ids"], "admission_units")
            assert_matches(score["counts"], a["counts"], "admission_weights")
        metric_summary = paired_metric_interval(a["candidate_probabilities"], a["comparator_probabilities"],
            a["labels"], a["sample_units"], c, reps=ac["bootstrap_reps"], alpha=ac["alpha"], seed=c["seed"]+6)
    admitted = admission(a["candidate_loss"] - a["comparator_loss"],
        metric_differences=a["candidate_accuracy"] - a["comparator_accuracy"] if spec.metric == "accuracy" else None,
        weights=a["counts"], reps=ac["bootstrap_reps"], alpha=ac["alpha"], epsilon_a=ac["epsilon_a"],
        epsilon_metric=ac["epsilon_metric"], clip=ac["clip"], seed=c["seed"]+6,
        metric_summary=metric_summary, independent_graphs=independent_graphs(c))
    assert_matches(admitted, e.json("admit/decision.json"), "admission")
    lock = e.json("policy_lock.json")
    qc = selected["comparator_index"]
    qd = confirmed["confirmed_index"] if admitted["passed"] else qc
    if lock["comparator_index"] != qc or lock["deployed_index"] != qd or lock["registry_sha256"] != canonical(candidates):
        raise EvidenceError("Policy lock does not match reconstructed decisions")
    assert_matches(candidates[qc], summary["comparator"], "comparator")
    assert_matches(candidates[qd], summary["deployed"], "deployed")
    if lock["config_sha256"] != e.identities["config_sha256"] or lock["admission_sha256"] != e.sealed["admit/decision.json"]["sha256"]:
        raise EvidenceError("Policy lock configuration/admission identity mismatch")
    if lock["comparator_hash"] != candidates[qc]["hash"] or lock["deployed_hash"] != candidates[qd]["hash"]:
        raise EvidenceError("Policy hashes differ from their locked identities")
    if summary["admitted"] != admitted["passed"]:
        raise EvidenceError("Reported admission differs from reconstructed admission")
    if not admitted["passed"] and not np.array_equal(predictions["qD_probabilities"], predictions["qC_probabilities"]):
        raise EvidenceError("Rejected deployment is not identical to qC on test observations")
    report = {"status": "PASS", "schema_version": "review-replay-3", **e.identities,
            "dataset": c["dataset"], "benchmark_dataset": c.get("benchmark_dataset", c["dataset"]),
            "scientific_run": c["scientific_run"], "task": spec.to_dict(), "models": metrics,
            "paired_qD_minus_qC_loss": loss_contrast, "paired_qD_minus_qC_metric_pp": metric_contrast,
            "statistical_unit": "graph" if independent_graphs(c) else "dependent graph core",
            "selected_index": selected["provisional_index"], "comparator_index": qc, "deployed_index": qd,
            "admitted": admitted["passed"], "selection_replayed": bool(selection),
            "confirmation_replayed": True, "admission_replayed": True,
            "verified_numeric_files": sorted(e.checked),
            "scope": "numeric replay and sealed-input integrity; not checkpoint execution, historical authenticity, or manuscript agreement"}
    if spec.metric == "accuracy":
        report.update(paired_qD_minus_qC_ce=loss_contrast, paired_qD_minus_qC_accuracy_pp=metric_contrast)
    return report


def evidence_markdown(report):
    metric = report["task"]["metric"]
    rows = ["# CROSS-MORSE numerical replay", "", f"Status: **{report['status']}**", "",
            f"Dataset: `{report['benchmark_dataset']}`. Scientific-run flag: `{report['scientific_run']}`.", "",
            f"| Predictor | {metric} (%) | Loss (nats) |", "|---|---:|---:|"]
    for name, item in report["models"].items():
        rows.append(f"| {name} | {item['metric_percent']:.8f} | {item['loss_nats']:.10f} |")
    stat = report["paired_qD_minus_qC_loss"]
    rows += ["", f"Paired qD minus qC loss: **{stat['mean']:.10f}** nats.",
             f"Descriptive interval: [{stat['lower']}, {stat['upper']}].",
             f"Units: {stat['units']} {report['statistical_unit']} units; {stat['weighting']}.", "",
             f"Selection replayed: {report['selection_replayed']}; confirmation and admission replayed: True.",
             f"Admitted: {report['admitted']}. Selected/comparator/deployed indices: "
             f"{report['selected_index']}/{report['comparator_index']}/{report['deployed_index']}.", "",
             "## Interpretation", "", report["scope"], "",
             "A synthetic run verifies software behavior, not benchmark performance. "
             "Inference conditions are recorded separately from the numerical integrity checks.", ""]
    return "\n".join(rows)
