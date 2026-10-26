#!/usr/bin/env python3
"""Summarize ten sealed Roman-empire split runs without changing their policies.

The Student-t interval describes variation across the ten registered splits.
Masks share one graph, so this is not independent-graph population inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_morse.review import NumericEvidence, EvidenceError, canonical, replay_results


# Two-sided 95% Student-t critical value for exactly ten split observations.
T_975_DF9 = 2.2621571628540993


def split_summary(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise EvidenceError("Exactly ten finite split measurements are required")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    half = T_975_DF9*sd/np.sqrt(10)
    return {"mean": mean, "sample_standard_deviation": sd, "lower": mean-half,
            "upper": mean+half, "splits": 10, "degrees_of_freedom": 9,
            "interval_level": .95, "method": "paired split-level Student-t descriptive interval"}


def summarize(campaign):
    campaign = Path(campaign).resolve()
    records, sources, configurations, flags = [], set(), set(), set()
    for index in range(10):
        root = campaign/f"split_{index:02d}"
        if not root.is_dir():
            raise EvidenceError(f"Missing completed split: {root.name}")
        e = NumericEvidence(root)
        config = e.json("config.json")
        if config.get("benchmark_dataset", config["dataset"]) != "Roman-empire":
            raise EvidenceError(f"{root.name} is not a Roman-empire run")
        if config["data"].get("split_index") != index:
            raise EvidenceError(f"{root.name} has the wrong official split index")
        report = replay_results(root)
        sources.add(e.identities["source_sha256"])
        shared = dict(config, data=dict(config["data"]))
        shared["data"].pop("split_index")
        configurations.add(canonical(shared))
        flags.add(bool(config["scientific_run"]))
        records.append({"split_index": index, "run_identities": e.identities,
            "admitted": report["admitted"], "comparator_index": report["comparator_index"],
            "deployed_index": report["deployed_index"], "models": report["models"],
            "qD_minus_qC_loss": report["paired_qD_minus_qC_loss"]["mean"],
            "qD_minus_qC_accuracy_pp": report["paired_qD_minus_qC_metric_pp"]["mean"]})
    if len(sources) != 1 or len(configurations) != 1 or len(flags) != 1:
        raise EvidenceError("All splits must share source, non-split settings, and scientific-run status")
    return {"status": "COMPUTED_FROM_TEN_SEALED_SPLITS", "dataset": "Roman-empire",
        "scientific_run": next(iter(flags)), "source_sha256": next(iter(sources)),
        "common_config_without_split_sha256": next(iter(configurations)),
        "split_indices": list(range(10)), "splits": records,
        "models": {name: {"accuracy_percent": split_summary([r["models"][name]["metric_percent"] for r in records]),
                          "loss_nats": split_summary([r["models"][name]["loss_nats"] for r in records])}
                   for name in ("q0", "qN", "qC", "qD")},
        "paired_qD_minus_qC_loss": split_summary([r["qD_minus_qC_loss"] for r in records]),
        "paired_qD_minus_qC_accuracy_pp": split_summary([r["qD_minus_qC_accuracy_pp"] for r in records]),
        "admitted_split_count": sum(r["admitted"] for r in records),
        "changes_per_split_policy": False,
        "scope": "paired variation of the complete fitted procedure across ten official masks on one shared graph; overlapping splits are not ten independent graphs; this report does not replace the per-run admission decisions or estimate retraining-seed uncertainty"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.resolve().is_relative_to(args.campaign.resolve()):
            raise EvidenceError("Choose a new summary path outside the campaign")
        result = summarize(args.campaign)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": result["status"], "output": str(args.output), "splits": 10}))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
