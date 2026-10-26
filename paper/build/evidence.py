"""Extract one evidence record per completed run.

The record is a flat, JSON-serialisable summary of the sealed decision files of one run:
identities, data roles and seeds, the entry contract, the candidate family, selection,
confirmation, admission, final test scores and the cost ledger.  Every number is read
from the run; nothing is imputed.  Quantities that the released runner does not measure
(for example the serving-time input-fallback fraction) are recorded as ``None`` and the
table builders print them as "not measured".

A Roman-empire campaign (``split_00`` ... ``split_09``) is aggregated into one record whose
effects are the mean over splits with a small-sample t interval.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

STAGES = ("prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate")
DATASET_STEMS = {"ogbn-arxiv": "arxiv", "ogbn-products": "products", "ogbn-proteins": "proteins",
                 "Roman-empire": "roman_empire", "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp",
                 "Peptides-func": "peptides_func"}
DATASET_ORDER = list(DATASET_STEMS)
ROUTE_LABELS = {"nominal": "nominal", "best_depth": "best depth", "learned_stop": "learned stop",
                "phase": "phase schedule", "fixed_sequence": "fixed sequence", "mpc_ol": "MPC-OL",
                "one_step": "one-step", "radius": "radius", "mpc_rh": "MPC-RH"}
FEEDBACK_ROUTES = {"one_step", "radius", "mpc_rh"}
SCHEMA = "cross-morse-evidence-1"


class EvidenceError(ValueError):
    """A run directory is incomplete or inconsistent."""


def decision_roles(record: dict) -> dict:
    """Candidate index -> roles it plays in this run's decisions (q_C, selected, deployed)."""
    roles: dict[int, list[str]] = {}
    qc = record["selection"].get("qC_index")
    provisional = record["selection"].get("provisional_index")
    deployed = record["admission"].get("deployed_index")
    if qc is not None:
        roles.setdefault(qc, []).append("q_C")
    if provisional is not None and provisional != qc:
        roles.setdefault(provisional, []).append("selected")
    if deployed is not None and deployed != qc:
        roles.setdefault(deployed, []).append("deployed")
    return roles


def family_representatives(record: dict, key: str) -> list[dict]:
    """Rows of a one-row-per-route-family table, in registry order.

    Within a family, the comparator q_C, the provisionally selected rule and the deployed rule
    are always shown when they belong to it; a family that holds none of them is represented by
    the configuration with the smallest ``key`` (``select_upper`` or ``select_loss``).
    """
    roles = decision_roles(record)
    out = []
    for route in ROUTE_LABELS:
        members = [c for c in record["candidates"] if c["route"] == route]
        marked = [c for c in members if c["index"] in roles]
        if marked:
            out.extend(sorted(marked, key=lambda c: c["index"]))
            continue
        scored = [c for c in members if c.get(key) is not None]
        if scored:
            out.append(min(scored, key=lambda c: c[key]))
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise EvidenceError(f"missing run record: {path}") from None
    except ValueError as exc:
        raise EvidenceError(f"invalid JSON in {path}: {exc}") from None


def _optional_json(path: Path):
    return _json(path) if path.is_file() else None


def _float(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _interval(block: dict | None) -> dict | None:
    if not isinstance(block, dict):
        return None
    return {"mean": _float(block.get("mean")), "lower": _float(block.get("lower")),
            "upper": _float(block.get("upper")), "se": _float(block.get("se")),
            "units": block.get("units"), "method": block.get("method"),
            "valid_interval": block.get("valid_interval"),
            "independent_graph_guarantee": block.get("independent_graph_guarantee")}


def _stage_seconds(root: Path) -> dict:
    out = {}
    for stage in STAGES:
        record = _optional_json(root / "stages" / f"{stage}.json")
        if record is None:
            out[stage] = None
            continue
        out[stage] = {"seconds": _float(record.get("seconds_this_attempt")),
                      "status": record.get("status")}
    return out


def _candidates(root: Path) -> list[dict]:
    candidates = _json(root / "select" / "candidates.json")
    decision = _optional_json(root / "select" / "decision.json") or {}

    def per_candidate(key, index):
        values = decision.get(key)
        if not isinstance(values, list) or index >= len(values):
            return None
        return values[index]
    ledger_path = root / "select" / "candidate_ledger.csv"
    ledger = {}
    if ledger_path.is_file():
        with ledger_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                ledger[row["candidate_id"]] = row
    rows = []
    for index, candidate in enumerate(candidates):
        led = ledger.get(candidate.get("candidate_id"), {})

        def num(key):
            value = led.get(key)
            if value in (None, "", "nan"):
                return None
            try:
                return _float(value)
            except ValueError:
                return None
        rows.append({
            "index": index,
            "candidate_id": candidate.get("candidate_id"),
            "route": candidate.get("route"),
            "route_label": ROUTE_LABELS.get(candidate.get("route"), candidate.get("route")),
            "feedback": candidate.get("route") in FEEDBACK_ROUTES,
            "is_open_loop": candidate.get("is_open_loop"),
            "profile": candidate.get("profile"),
            "ensemble": candidate.get("ensemble"),
            "depth": candidate.get("depth"),
            "complexity": candidate.get("complexity"),
            "cost": _float(candidate.get("cost")),
            "cost_definition": candidate.get("cost_definition"),
            "qualification_cost": _float(candidate.get("qualification_cost")),
            "serial_wall_seconds": _float(candidate.get("serial_wall_seconds")),
            "feasible": candidate.get("feasible"),
            "execution_failure": candidate.get("execution_failure"),
            "select_loss": num("select_loss"),
            "select_metric": num("select_metric"),
            "select_lower": num("lower"),
            "select_upper": num("upper"),
            "eligible": (led.get("eligible") == "True") if led else None,
            # sealed statistics of the nested selection (select/decision.json), relative to the nested comparator
            "select_delta": _float(per_candidate("mean", index)),
            "joint_feasible": per_candidate("joint_feasible", index),
            "hash": candidate.get("hash"),
        })
    return rows


def _entry(root: Path) -> dict:
    initial = _optional_json(root / "train" / "entry_initial.json") or {}
    complete = _optional_json(root / "select" / "entry_complete.json") or {}
    instrumentation = initial.get("instrumentation") or {}
    headroom = _interval(complete.get("headroom"))
    return {
        "identity_pass": initial.get("identity_pass"),
        "depth_gate_pass": initial.get("depth_gate_pass"),
        "initial_entry_pass": initial.get("initial_entry_pass"),
        "depth_grid": initial.get("depth_grid"),
        "depth_loss": initial.get("depth_loss"),
        "depth_metric": initial.get("depth_metric"),
        "dil_gap": _float(initial.get("dil_gap")),
        "dil_log_depth": _float(initial.get("dil_log_depth")),
        "loss_name": initial.get("loss_name"),
        "labels": initial.get("labels"),
        "headroom": headroom,
        "headroom_definition": complete.get("headroom_definition"),
        "enforced": complete.get("enforced"),
        "complete_pass": complete.get("passed"),
        "registry_support_fraction": _float(complete.get("registry_support_fraction")),
        "instrumentation": {
            "backbone": instrumentation.get("backbone"),
            "controlled_depth": instrumentation.get("controlled_depth"),
            "native_site_count": instrumentation.get("native_site_count"),
            "inserted_site_count": instrumentation.get("inserted_site_count"),
            "parameter_count": instrumentation.get("parameter_count"),
        },
    }


def _seeds(root: Path, config: dict, summary: dict, metadata: dict) -> dict:
    training = (config.get("training") or {}).get("seeds")
    out = {"partition_seed": metadata.get("partition_seed", config.get("seed")),
           "training_seeds": training,
           "ensemble_seeds": summary.get("ensemble_seeds"),
           "bootstrap_seeds": {}}
    for label, path, nested in (("selection", root / "select" / "decision.json", None),
                                ("admission", root / "admit" / "decision.json", "loss"),
                                ("final", root / "final" / "results_summary.json", "paired_qD_minus_qC_loss")):
        record = _optional_json(path) or {}
        if nested is not None:
            record = record.get(nested) or {}
        out["bootstrap_seeds"][label] = record.get("bootstrap_seed")
    return out


def _data(metadata: dict, config: dict) -> dict:
    return {
        "benchmark_dataset": metadata.get("benchmark_dataset"),
        "is_benchmark_data": metadata.get("is_benchmark_data"),
        "source": metadata.get("source"),
        "num_nodes": metadata.get("num_nodes"),
        "num_graphs": metadata.get("num_graphs"),
        "undirected_edges": metadata.get("undirected_edges"),
        "num_features": metadata.get("num_features"),
        "num_outputs": metadata.get("num_outputs"),
        "split_unit": metadata.get("split_unit"),
        "split_rule": metadata.get("split_rule"),
        "split_sizes": metadata.get("split_sizes"),
        "role_hashes": metadata.get("role_hashes"),
        "official_split_hashes": metadata.get("official_split_hashes"),
        "statistical_unit": (metadata.get("buffer_audit") or {}).get("statistical_unit"),
        "independent_graphs": metadata.get("independent_graphs"),
        "num_clusters": (config.get("data") or {}).get("num_clusters"),
        "buffer_hops": (config.get("data") or {}).get("buffer_hops"),
        "split_index": (config.get("data") or {}).get("split_index"),
        "prepared_fingerprint": metadata.get("fingerprint"),
        "source_array_hashes": metadata.get("source_array_hashes"),
    }


def _environment(env: dict | None) -> dict:
    env = env or {}
    return {key: env.get(key) for key in ("python", "torch", "cuda_runtime", "gpu", "device",
                                          "platform", "machine", "numpy", "precision",
                                          "deterministic_algorithms_enabled")}


def _cost(candidates: list[dict], selection: dict, admission: dict, final: dict,
          stage_seconds: dict, config: dict) -> dict:
    by_index = {c["index"]: c for c in candidates}
    comparator = by_index.get(selection.get("qC_index"))
    deployed = by_index.get(admission.get("deployed_index"))
    nominal = next((c for c in candidates if c["route"] == "nominal" and c["ensemble"] == 1), None)
    c_qc = comparator["cost"] if comparator else None
    c_qd = deployed["cost"] if deployed else None
    overhead = None
    if c_qc not in (None, 0) and c_qd is not None:
        overhead = 100.0 * (c_qd - c_qc) / c_qc
    assure = None
    if c_qc is not None and c_qd is not None:
        assure = c_qd + c_qc if admission.get("Gadmit") == 1 else c_qc
    paired = (final.get("paired_qD_minus_qC_loss") or {})
    effect = _float(paired.get("mean"))
    exchange = None
    if effect is not None and c_qc is not None and c_qd is not None and c_qd > c_qc and effect < 0:
        exchange = 1000.0 * abs(effect) / (c_qd - c_qc)
    qualification = {stage: (stage_seconds.get(stage) or {}).get("seconds")
                     for stage in ("explore", "fit", "select", "confirm", "admit")}
    total = sum(v for v in qualification.values() if v is not None) if any(
        v is not None for v in qualification.values()) else None
    return {
        "unit": comparator["cost_definition"] if comparator else None,
        "C_qC": c_qc,
        "C_qD": c_qd,
        "C_expected": c_qd,
        "expected_cost_note": ("equal to C_qD: the released runner has no serving-time input gate, "
                               "so the input-fallback probability of main Eq. (32) is not measured"),
        "fallback_probability": None,
        "overhead_percent": overhead,
        "C_assure": assure,
        "wall_seconds": {"qC": comparator["serial_wall_seconds"] if comparator else None,
                         "qD": deployed["serial_wall_seconds"] if deployed else None,
                         "nominal_E1": nominal["serial_wall_seconds"] if nominal else None},
        "exchange_rate_mnats_per_nominal_unit": exchange,
        "qualification_seconds": qualification,
        "qualification_seconds_total": total,
        "training_seconds": (stage_seconds.get("train") or {}).get("seconds"),
    }


def evidence_from_run(run_root: Path) -> dict:
    root = Path(run_root).resolve()
    if not (root / "final" / "results_summary.json").is_file():
        raise EvidenceError(f"not a completed run (final/results_summary.json missing): {root}")
    state = _json(root / "run_state.json")
    config = _json(root / "config.json")
    metadata = _json(root / "prepared" / "metadata.json")
    summary = _json(root / "final" / "results_summary.json")
    selection = _json(root / "select" / "decision.json")
    confirmation = _json(root / "confirm" / "decision.json")
    admission = _json(root / "admit" / "decision.json")
    lock = _optional_json(root / "policy_lock.json") or {}
    environment = _optional_json(root / "environment.json")
    candidates = _candidates(root)
    stage_seconds = _stage_seconds(root)
    explore = _optional_json(root / "explore" / "manifest.json") or {}
    models = {}
    for name in ("q0", "qN", "qC", "qD"):
        block = (summary.get("models") or {}).get(name) or {}
        models[name] = {"metric": _float(block.get("metric")), "metric_percent": _float(block.get("metric_percent")),
                        "metric_name": block.get("metric_name"), "loss_nats": _float(block.get("loss_nats")),
                        "loss_name": block.get("loss_name"), "ensemble": block.get("ensemble"),
                        "test_units": block.get("test_units"), "test_samples": block.get("test_samples"),
                        "probability_sha256": block.get("probability_sha256")}
    by_index = {c["index"]: c for c in candidates}
    unit_sha = None
    if (root / "prepared" / "splits.npz").is_file():
        unit_sha = sha256_file(root / "prepared" / "splits.npz")
    record = {
        "schema": SCHEMA,
        "kind": "run",
        "dataset": state.get("task", {}).get("dataset") or summary.get("benchmark_dataset"),
        "benchmark_dataset": summary.get("benchmark_dataset"),
        "data_kind": state.get("dataset"),
        "profile": state.get("profile"),
        "scientific_run": bool(state.get("scientific_run")),
        "status": state.get("status"),
        "run_root": str(root),
        "run_name": root.name,
        "identities": {"source_sha256": state.get("source_sha256"), "config_sha256": state.get("config_sha256"),
                       "unit_sha256": unit_sha, "implementation_version": state.get("version"),
                       "deployed_hash": lock.get("deployed_hash"), "comparator_hash": lock.get("comparator_hash")},
        "task": {"metric": summary.get("metric_definition"), "loss_name": models["qC"]["loss_name"],
                 "prediction_level": (summary.get("task") or {}).get("prediction_level"),
                 "num_outputs": (summary.get("task") or {}).get("num_outputs"),
                 "task_type": (summary.get("task") or {}).get("task_type")},
        "environment": _environment(environment),
        "data": _data(metadata, config),
        "seeds": _seeds(root, config, summary, metadata),
        "entry": _entry(root),
        "exploration": {"trajectories": explore.get("trajectories"), "labels": explore.get("labels")},
        "candidates": candidates,
        "selection": {
            "qC_index": selection.get("qC_index"), "comparator_index": selection.get("comparator_index"),
            "provisional_index": selection.get("provisional_index"),
            "Q_s": selection.get("Q_s"), "eligible_indices": selection.get("eligible_indices"),
            "epsilon_r": _float(selection.get("epsilon_r")), "epsilon_eq": _float(selection.get("epsilon_eq")),
            "epsilon_metric": _float(selection.get("epsilon_metric")), "alpha": _float(selection.get("alpha")),
            "bootstrap_replicates": selection.get("bootstrap_replicates"),
            "max_radius": _float(selection.get("max_radius")), "units": selection.get("units"),
            "outer_folds": selection.get("outer_folds"), "inner_folds": selection.get("inner_folds"),
            "candidate_count": selection.get("candidate_count"), "rule": selection.get("rule"),
            "minimum_upper": _float(selection.get("minimum_upper")),
            "method": selection.get("method"),
            "final_comparator_selection": selection.get("final_comparator_selection"),
            "qC": by_index.get(selection.get("qC_index")),
            "provisional": by_index.get(selection.get("provisional_index")),
        },
        "confirmation": {
            "provisional_index": confirmation.get("provisional_index"),
            "confirmed_index": confirmation.get("confirmed_index"),
            "comparator_index": confirmation.get("comparator_index"),
            "alpha": _float(confirmation.get("alpha")),
            "holm_family_size": confirmation.get("holm_family_size"),
            "contrasts": confirmation.get("contrasts") or [],
            "downgrades": confirmation.get("downgrades") or [],
            "membership_in_Qs_or_qC": confirmation.get("membership_in_Qs_or_qC"),
            "confirmed": by_index.get(confirmation.get("confirmed_index")),
        },
        "admission": {
            "Gadmit": admission.get("Gadmit"), "passed": admission.get("passed"),
            "deployed_index": admission.get("deployed_index"),
            "candidate_already_comparator": admission.get("candidate_already_comparator"),
            "epsilon_a": _float(admission.get("epsilon_a")), "epsilon_metric": _float(admission.get("epsilon_metric")),
            "clip": _float(admission.get("clip")), "clipping_fraction": _float(admission.get("clipping_fraction")),
            "independent_graph_guarantee": admission.get("independent_graph_guarantee"),
            "official_validation_only": admission.get("official_validation_only"),
            "loss": _interval(admission.get("loss")),
            "raw_loss": _interval(admission.get("raw_loss")),
            "metric": _interval(admission.get("metric")),
            "mean_gate": admission.get("mean_gate"), "metric_gate": admission.get("metric_gate"),
            "empirical_bernstein": admission.get("empirical_bernstein"),
            "deployed": by_index.get(admission.get("deployed_index")),
        },
        "final": {
            "admitted": summary.get("admitted"), "exact_fallback": summary.get("exact_fallback"),
            "models": models,
            "paired_loss": _interval(summary.get("paired_qD_minus_qC_loss")),
            "paired_metric_pp": _interval(summary.get("paired_qD_minus_qC_metric_pp")),
            "statistical_scope": summary.get("statistical_scope"),
            "seed_scope": summary.get("seed_scope"),
            "test_prediction_record_sha256": summary.get("test_prediction_record_sha256"),
            "status": summary.get("status"),
        },
        "stages": stage_seconds,
        "cost": None,
    }
    record["cost"] = _cost(candidates, record["selection"], record["admission"], summary, stage_seconds, config)
    return record


def _t_interval(values: list[float], level: float = 0.95) -> dict:
    n = len(values)
    mean = statistics.fmean(values)
    if n < 2:
        return {"mean": mean, "lower": None, "upper": None, "se": None, "units": n,
                "method": "single split; no interval"}
    sd = statistics.stdev(values)
    se = sd / math.sqrt(n)
    # two-sided t quantiles for 95% at n-1 degrees of freedom (n <= 10 in a campaign)
    quantiles = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
    t = quantiles.get(n - 1, 1.96)
    return {"mean": mean, "lower": mean - t * se, "upper": mean + t * se, "se": se, "units": n,
            "method": "paired split effects; small-sample t interval over official splits",
            "valid_interval": True, "independent_graph_guarantee": False}


def evidence_from_campaign(campaign_root: Path) -> dict:
    root = Path(campaign_root).resolve()
    splits = sorted(p for p in root.glob("split_*") if (p / "final" / "results_summary.json").is_file())
    if not splits:
        raise EvidenceError(f"no completed split runs under {root}")
    records = [evidence_from_run(p) for p in splits]
    from .selection import selection_readings
    base = json.loads(json.dumps(records[0]))
    base["kind"] = "campaign"
    base["run_root"] = str(root)
    base["run_name"] = root.name
    base["campaign"] = {"splits": [r["data"]["split_index"] for r in records],
                        "split_runs": [r["run_name"] for r in records],
                        "per_split": [{"split_index": r["data"]["split_index"],
                                       "paired_loss": (r["final"]["paired_loss"] or {}).get("mean"),
                                       "paired_metric_pp": (r["final"]["paired_metric_pp"] or {}).get("mean"),
                                       "Gadmit": r["admission"]["Gadmit"],
                                       "qC_route": (r["selection"]["qC"] or {}).get("route"),
                                       "qD_route": (r["admission"]["deployed"] or {}).get("route")}
                                      for r in records],
                        "selection_readings": [selection_readings(r) for r in records]}
    losses = [(r["final"]["paired_loss"] or {}).get("mean") for r in records]
    metrics = [(r["final"]["paired_metric_pp"] or {}).get("mean") for r in records]
    if all(v is not None for v in losses):
        base["final"]["paired_loss"] = _t_interval(losses)
    if all(v is not None for v in metrics):
        base["final"]["paired_metric_pp"] = _t_interval(metrics)
    for name in ("q0", "qN", "qC", "qD"):
        vals = [r["final"]["models"][name]["metric_percent"] for r in records]
        if all(v is not None for v in vals):
            base["final"]["models"][name]["metric_percent"] = statistics.fmean(vals)
            base["final"]["models"][name]["metric"] = statistics.fmean(vals) / 100.0
        losses_n = [r["final"]["models"][name]["loss_nats"] for r in records]
        if all(v is not None for v in losses_n):
            base["final"]["models"][name]["loss_nats"] = statistics.fmean(losses_n)
    admits = [r["admission"]["Gadmit"] for r in records]
    base["admission"]["Gadmit_per_split"] = admits
    base["admission"]["Gadmit"] = 1 if admits and all(a == 1 for a in admits) else 0
    base["admission"]["loss"] = base["final"]["paired_loss"]
    base["seeds"]["split_indices"] = [r["data"]["split_index"] for r in records]
    for stage in base["stages"]:
        secs = [(r["stages"].get(stage) or {}).get("seconds") for r in records]
        if base["stages"][stage] is not None and all(s is not None for s in secs):
            base["stages"][stage] = {"seconds": sum(secs), "status": "aggregated over splits"}
    base["cost"] = _cost(base["candidates"], base["selection"], base["admission"],
                         {"paired_qD_minus_qC_loss": base["final"]["paired_loss"]}, base["stages"], {})
    return base


def load_evidence(path: Path) -> dict:
    record = _json(Path(path))
    if record.get("schema") != SCHEMA:
        raise EvidenceError(f"unexpected evidence schema in {path}")
    return record


def write_evidence(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def stem_for(dataset: str) -> str:
    try:
        return DATASET_STEMS[dataset]
    except KeyError:
        raise EvidenceError(f"unknown canonical dataset name: {dataset}") from None
