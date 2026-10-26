#!/usr/bin/env python3
"""Paired comparison of two archived runs.

Reads numeric artifacts without modifying either run.
This is a post-hoc fixed-ensemble comparison, not fresh confirmatory evidence.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_morse.review import NumericEvidence, EvidenceError
from cross_morse.statistics import paired_bootstrap
from cross_morse.tasks import task_spec
from cross_morse.evaluation import score_observations, paired_metric_interval


def observations(root):
    e = NumericEvidence(root)
    summary = e.json("final/results_summary.json")
    if "final/test_predictions.npz" in e.sealed:
        data = e.arrays("final/test_predictions.npz")
    else:
        # For archives without a combined observation file, derive test
        # observations from cached probabilities and prepared labels.
        ids = e.arrays("prepared/splits.npz")["test"]
        with np.load(e.file("prepared/graph.npz"), allow_pickle=False) as z:
            labels = z["y"][ids]
        clusters = e.arrays("prepared/clusters.npz")["cluster_ids"][ids]
        data = {"node_ids": ids, "labels": labels, "cluster_ids": clusters}
        for name in ("qC", "qD"):
            data[f"{name}_probabilities"] = np.load(e.file(f"final/{name}_probabilities.npy"),
                                                    allow_pickle=False, mmap_mode="r")[ids]
    return e, summary, data


def compare(before, after, *, reps=10000, seed=1147):
    eb, sb, b = observations(before)
    ea, sa, a = observations(after)
    cb, ca = eb.json("config.json"), ea.json("config.json")
    spec = task_spec(ca)
    if eb.identities["dataset"] != ea.identities["dataset"] or task_spec(cb) != spec:
        raise EvidenceError("Dataset or task identities differ")
    for name in ("node_ids", "labels", "cluster_ids"):
        if not np.array_equal(b[name], a[name], equal_nan=True):
            raise EvidenceError(f"Paired comparison requires identical ordered {name}")
    scores = {run: {name: score_observations(obs[f"{name}_probabilities"], obs["labels"], obs["cluster_ids"], config)
                       for name in ("qC", "qD")} for run, obs, config in (("before", b, cb), ("after", a, ca))}
    bs, ats = scores["before"], scores["after"]
    report = {"status": "COMPUTED", "dataset": ea.identities["dataset"], "task": spec.to_dict(),
            "before": {"identities": eb.identities, "deployed": sb["deployed"],
                       "admitted": sb["admitted"], "mean_qD_minus_own_qC": bs["qD"]["node_ce"]-bs["qC"]["node_ce"]},
            "after": {"identities": ea.identities, "deployed": sa["deployed"],
                      "admitted": sa["admitted"], "mean_qD_minus_own_qC": ats["qD"]["node_ce"]-ats["qC"]["node_ce"]},
            "paired_after_qD_minus_before_qD": paired_bootstrap(ats["qD"]["loss"]-bs["qD"]["loss"],
                weights=ats["qD"]["counts"], reps=reps, alpha=.025, seed=seed),
            "paired_after_qD_minus_before_qD_metric_pp": paired_metric_interval(
                a["qD_probabilities"], b["qD_probabilities"], a["labels"], a["cluster_ids"], ca,
                reps=reps, alpha=.025, seed=seed, scale=100),
            "max_probability_change": float(np.max(np.abs(a["qD_probabilities"]-b["qD_probabilities"]))),
            "scope": "post-hoc fixed-ensemble impact audit on identical observed test samples; no causal attribution to one change or fresh confirmation"}
    if spec.task_type == "multilabel":
        report["changed_test_threshold_count"] = int(np.sum((a["qD_probabilities"] >= .5) != (b["qD_probabilities"] >= .5)))
    else:
        report["changed_test_argmax_count"] = int(np.sum(a["qD_probabilities"].argmax(1) != b["qD_probabilities"].argmax(1)))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before", type=Path); p.add_argument("after", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap-reps", type=int, default=10000)
    args = p.parse_args(argv)
    try:
        if args.output.exists() or any(args.output.resolve().is_relative_to(root.resolve()) for root in (args.before, args.after)):
            raise EvidenceError("Choose a new report outside both runs")
        result = compare(args.before, args.after, reps=args.bootstrap_reps)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
