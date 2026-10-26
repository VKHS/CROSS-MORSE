#!/usr/bin/env python3
"""Inventory a complete run using the Python standard library only.

Every sealed artifact is hashed, including checkpoints. No checkpoint is
deserialized, and recorded logical tensor hashes are crosschecked, not recomputed.
Numeric archive structure and actual split membership are inspected without
NumPy. Use reproduce_results.py for statistical replay and evaluate_checkpoints.py
for model execution. An inventory is integrity evidence, not signed history.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import tempfile
import zipfile

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from cross_morse import provenance as pv

SCHEMA = "cross-morse-artifact-inventory-1"
ROLES = ("fit", "fit_train", "fit_val", "explore", "select", "confirm", "admit", "test")
LEAF_ROLES = ("fit_train", "fit_val", "explore", "select", "confirm", "admit", "test")
HEX = re.compile(r"[0-9a-f]{64}\Z")
DTYPE = re.compile(r"([<>=|])([biufcSU])([1-9][0-9]*)\Z")


class InventoryError(ValueError):
    """The submitted complete-run artifacts do not satisfy the inventory contract."""


def require(condition, message):
    if not condition:
        raise InventoryError(message)


def read_json(path):
    def reject(value):
        raise InventoryError(f"Non-finite JSON number in {path}: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def safe_file(root, relative):
    require(isinstance(relative, str) and "\\" not in relative, "Artifact path must be a POSIX relative path")
    parts = PurePosixPath(relative).parts
    require(parts and not relative.startswith("/") and all(p not in (".", "..") for p in relative.split("/")),
            f"Unsafe artifact path: {relative}")
    p = root
    for part in parts:
        p = p / part
        require(not p.is_symlink(), f"Symlink is not an inventory artifact: {relative}")
    require(p.resolve().is_relative_to(root) and p.is_file(), f"Missing or escaping artifact: {relative}")
    return p


def _baseline_audit(root, descriptor, cache):
    pv._verify_artifact(root, descriptor, cache)
    baseline = read_json(safe_file(root, descriptor["path"]))
    require(set(baseline) == {"schema_version", "captured_after_stage", "model_key", "head_prefix", "checkpoints", "hash_method"},
            "Malformed frozen-model baseline")
    require(baseline["schema_version"] == pv.SCHEMA_VERSION and baseline["captured_after_stage"] == "train",
            "Model baseline must be captured immediately after train")
    require(isinstance(baseline["checkpoints"], dict) and baseline["checkpoints"], "Empty model baseline")
    comparisons = {}
    for relative, record in baseline["checkpoints"].items():
        require(record.get("artifact", {}).get("path") == relative, "Malformed checkpoint baseline path")
        safe_file(root, relative)
        actual = pv._verify_artifact(root, record["artifact"], cache)
        for key in ("backbone_sha256", "head_sha256"):
            require(isinstance(record.get(key), str) and HEX.fullmatch(record[key]), f"Invalid recorded {key}")
        for key in ("backbone_tensors", "head_tensors"):
            require(type(record.get(key)) is int and record[key] > 0, f"Invalid recorded {key}")
        comparisons[relative] = {
            "checkpoint_sha256": actual["sha256"],
            "backbone_sha256_before": record["backbone_sha256"],
            "backbone_sha256_after": record["backbone_sha256"],
            "head_sha256_before": record["head_sha256"],
            "head_sha256_after": record["head_sha256"],
        }
    return baseline, {"baseline": descriptor, "checkpoints": comparisons, "matches_post_train_baseline": True}


def verify_run_bytes(run_root):
    """Verify a complete chain and all bytes without calling torch.load.

    This mirrors the lifecycle's provenance checks except for logical tensor
    recomputation. Unlike an analysis-only review bundle, no sealed file may be
    missing. A per-call cache hashes repeated input references just once.
    """
    root = Path(run_root).resolve()
    config = read_json(safe_file(root, "config.json"))
    expected = {pv.stage_manifest_path(root, s).name for s in pv.STAGES}
    actual = {p.name for p in (root / "provenance/stages").glob("*.json")}
    require(actual == expected, "A complete inventory requires exactly all eight stage manifests")
    cache, sealed, origin, chain, baselines = {}, {}, {}, [], {}
    identities = None
    for index, stage in enumerate(pv.STAGES):
        rel = pv.stage_manifest_path(root, stage).relative_to(root).as_posix()
        value = read_json(safe_file(root, rel))
        pv._validate_manifest_shape(value)
        require(value["stage"] == stage, "Stage filename/content mismatch")
        ids = {k: value[k] for k in ("source_sha256", "config_sha256", "unit_sha256")}
        if identities is None:
            for key, digest in ids.items():
                require(isinstance(digest, str) and HEX.fullmatch(digest), f"Malformed {key}")
            require(ids["source_sha256"] == pv.current_source_sha256(), "Run requires its exact runtime source snapshot")
            require(ids["config_sha256"] == pv._json_hash(config), "Configuration content changed")
            require(ids["unit_sha256"] == pv.sha256_file(safe_file(root, pv.UNIT_MANIFEST_PATH)), "Split identity changed")
            identities = ids
        require(ids == identities and value["dataset"] == config["dataset"], "Run identities differ across stages")
        parent = None if index == 0 else pv._cached_artifact(root, pv.stage_manifest_path(root, pv.STAGES[index-1]), cache)
        require(value["parent_manifest"] == parent, f"Parent manifest hash changed: {stage}")
        records = {}
        for field in ("inputs", "outputs"):
            records[field] = {}
            for declared in value[field]:
                safe_file(root, declared.get("path"))
                record = pv._verify_artifact(root, declared, cache)
                relative = record["path"]
                require(relative not in records[field], f"Duplicate {field}: {relative}")
                records[field][relative] = record
        if index:
            require(all(sealed.get(k) == v for k, v in records["inputs"].items()), "Stage input lacks an earlier sealed output")
        require(not (set(records["outputs"]) & set(sealed)), "A completed artifact was redeclared as a new output")
        if stage == "evaluate":
            require(set(records["outputs"]) == pv.FINAL_EVALUATION_OUTPUTS, "Final output allowlist changed")
            ev = value["evaluator"]
            safe_file(root, ev.get("path"))
            pv._verify_artifact(root, ev, cache)
            require(sealed.get(ev["path"]) == ev and records["inputs"].get(ev["path"]) == ev,
                    "Evaluator must be sealed before test and declared as an input")
            require(ev["sha256"] == pv.sha256_file(SOURCE / "cross_morse/pipeline.py"), "Frozen evaluator source differs")
        else:
            require(value["evaluator"] is None, "Only evaluate may record an evaluator")
        if stage == "prepare":
            require(value["model_audit"] is None, "Prepare cannot contain a model freeze audit")
        else:
            audit = value["model_audit"]
            require(isinstance(audit, dict) and set(audit) == {"scope", "baseline", "checkpoints", "matches_post_train_baseline"},
                    "Missing frozen-model audit")
            baseline_desc = audit["baseline"]
            baseline_path = baseline_desc.get("path")
            safe_file(root, baseline_path)
            if stage == "train":
                require(records["outputs"].get(baseline_path) == baseline_desc, "Train must seal the model baseline")
            else:
                require(sealed.get(baseline_path) == baseline_desc, "Later stage changed the frozen baseline")
            if baseline_path not in baselines:
                baselines[baseline_path] = _baseline_audit(root, baseline_desc, cache)
            expected_audit = dict(baselines[baseline_path][1], scope="post_train_capture" if stage == "train" else "unchanged_since_train")
            require(audit == expected_audit, f"Recorded model/head hash linkage changed: {stage}")
        sealed.update(records["outputs"])
        origin.update({p: stage for p in records["outputs"]})
        chain.append(value)
    pv._verify_completed_directory_inventory(root, chain, sealed)
    for directory in ("predictions", "member_predictions", "prediction_failures"):
        extra = {p.relative_to(root).as_posix() for p in (root / directory).rglob("*")
                 if p.is_file() and not p.name.endswith(pv.TEMPORARY_ARTIFACT_SUFFIXES)} - set(sealed)
        require(not extra, f"Completed run contains unsealed {directory} files: {sorted(extra)}")
    state_path = root / "run_state.json"
    if state_path.exists():
        state = read_json(safe_file(root, "run_state.json"))
        require(state.get("status") == "COMPLETE" and state.get("last_complete_stage") == "evaluate", "Run bookkeeping is not complete")
        require(all(state.get(k) == identities[k] for k in ("source_sha256", "config_sha256")), "Run bookkeeping identities differ")
    for stage in pv.STAGES:
        marker = root / "stages" / f"{stage}.json"
        if marker.exists():
            data = read_json(safe_file(root, marker.relative_to(root).as_posix()))
            require(data.get("stage") == stage and data.get("status") == "COMPLETE" and
                    data.get("manifest_sha256") == pv.sha256_file(pv.stage_manifest_path(root, stage)), f"Completion marker mismatch: {stage}")
    return root, config, identities, sealed, origin, chain, baselines, cache


def npy_header(stream, total_bytes):
    """Read a bounded NPY header; reject object and structured dtypes.

    Fixed-width strings are safe metadata (e.g. calibration policy names).
    Their bytes are never interpreted as Python objects.
    """
    magic = stream.read(8)
    require(len(magic) == 8 and magic[:6] == b"\x93NUMPY" and magic[6:8] in (b"\x01\x00", b"\x02\x00", b"\x03\x00"), "Invalid NPY magic/version")
    size = 2 if magic[6] == 1 else 4
    raw = stream.read(size)
    require(len(raw) == size, "Truncated NPY header length")
    length = struct.unpack("<H" if size == 2 else "<I", raw)[0]
    require(0 < length <= 1024 * 1024, "Unreasonable NPY header length")
    raw = stream.read(length)
    require(len(raw) == length, "Truncated NPY header")
    try:
        h = ast.literal_eval(raw.decode("utf-8" if magic[6] == 3 else "latin1"))
    except (ValueError, SyntaxError, UnicodeError) as exc:
        raise InventoryError("Invalid NPY header literal") from exc
    require(isinstance(h, dict) and set(h) == {"descr", "fortran_order", "shape"}, "Unexpected NPY header fields")
    require(isinstance(h["descr"], str) and DTYPE.fullmatch(h["descr"]), "Only primitive numeric/fixed-width string NPY dtypes are allowed")
    require(type(h["fortran_order"]) is bool and isinstance(h["shape"], tuple) and all(type(n) is int and n >= 0 for n in h["shape"]), "Invalid NPY shape/order")
    match = DTYPE.fullmatch(h["descr"])
    kind, width = match.group(2), int(match.group(3))
    require(kind in ("S", "U") or width in {"b": (1,), "i": (1,2,4,8), "u": (1,2,4,8), "f": (2,4,8,16), "c": (8,16,32)}[kind], "Unsupported NPY numeric width")
    itemsize = width * (4 if kind == "U" else 1)
    offset = 8 + size + length
    require(offset + math.prod(h["shape"]) * itemsize == total_bytes, "NPY payload size does not match its shape/dtype")
    return {"dtype": h["descr"], "shape": list(h["shape"]), "fortran_order": h["fortran_order"], "payload_bytes": total_bytes-offset}, offset


def numeric_headers(path):
    if path.suffix == ".npy":
        with path.open("rb") as f:
            return {"array": npy_header(f, path.stat().st_size)[0]}
    result = {}
    with zipfile.ZipFile(path) as archive:
        require(archive.infolist(), f"Empty NPZ archive: {path.name}")
        for info in archive.infolist():
            name = info.filename
            require("/" not in name and "\\" not in name and name.endswith(".npy") and not info.flag_bits & 1,
                    f"Unsafe/non-numeric NPZ member: {name}")
            key = name[:-4]
            require(key and key not in result, f"Duplicate NPZ member: {name}")
            with archive.open(info) as f:
                result[key] = npy_header(f, info.file_size)[0]
    return result


def integer_array(path, key):
    """Read a one-dimensional numeric index array, checking the ZIP CRC too."""
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(key + ".npy")
        with archive.open(info) as stream:
            header, _ = npy_header(stream, info.file_size)
            match = DTYPE.fullmatch(header["dtype"])
            require(len(header["shape"]) == 1 and match.group(2) in ("i", "u"), f"Expected integer index vector: {path.name}:{key}")
            size = int(match.group(3))
            require(size in (1, 2, 4, 8), "Unsupported index integer width")
            prefix = match.group(1)
            if prefix in ("=", "|"):
                prefix = "<" if sys.byteorder == "little" else ">"
            code = {1: "b", 2: "h", 4: "i", 8: "q"}[size]
            if match.group(2) == "u":
                code = code.upper()
            payload = stream.read()
            require(len(payload) == header["payload_bytes"], "Truncated integer payload")
    return [v[0] for v in struct.iter_unpack(prefix + code, payload)]


def _membership(path, role_names, n, metadata_sizes=None):
    headers = numeric_headers(path)
    require(set(headers) == set(role_names), f"Unexpected role names in {path.name}")
    values = {role: integer_array(path, role) for role in role_names}
    for role, ids in values.items():
        require(ids and len(set(ids)) == len(ids) and min(ids) >= 0 and max(ids) < n, f"Invalid or duplicate {role} indices")
        if metadata_sizes is not None:
            require(metadata_sizes.get(role) == len(ids), f"Metadata split count differs: {role}")
    return values


def split_inventory(root, metadata):
    n = int(metadata["num_samples"])
    roles = _membership(root / "prepared/splits.npz", ROLES, n, metadata["split_sizes"])
    leaf = [i for role in LEAF_ROLES for i in roles[role]]
    require(len(leaf) == n and len(set(leaf)) == n, "Leaf roles must be disjoint and cover the label axis")
    require(set(roles["fit"]) == set(roles["fit_train"]) | set(roles["fit_val"]), "fit is not the union of fit_train and fit_val")
    independent = bool(metadata["independent_graphs"])
    unit_n = int(metadata["num_graphs"]) if independent else n
    official = _membership(root / "prepared/official_splits.npz", ("train", "valid", "test"), unit_n)
    all_official = [i for group in official.values() for i in group]
    require(len(all_official) == unit_n and len(set(all_official)) == unit_n, "Official splits overlap or omit units")
    unit_roles = _membership(root / "prepared/graph_splits.npz", ROLES, unit_n, metadata["graph_split_sizes"]) if independent else roles
    if independent:
        leaf_graphs = [i for role in LEAF_ROLES for i in unit_roles[role]]
        require(len(leaf_graphs) == unit_n and len(set(leaf_graphs)) == unit_n, "Graph role splits overlap or omit graphs")
        require(set(unit_roles["fit"]) == set(unit_roles["fit_train"]) | set(unit_roles["fit_val"]), "Graph fit split is not its internal union")
        ptr = integer_array(root / "prepared/graph.npz", "graph_ptr")
        require(len(ptr) == unit_n+1 and ptr[0] == 0 and ptr[-1] == metadata["num_nodes"] and all(a < b for a,b in zip(ptr,ptr[1:])), "Invalid graph pointer boundaries")
        for role in ROLES:
            expected = set(unit_roles[role]) if metadata["label_axis"] == "graph" else {i for g in unit_roles[role] for i in range(ptr[g], ptr[g+1])}
            require(set(roles[role]) == expected, f"Partial graph or wrong graph/row mapping in {role}")
    require(set(unit_roles["admit"]) == set(official["valid"]) and set(unit_roles["test"]) == set(official["test"]), "Admission/test differs from official validation/test")
    dev = [i for role in ("fit", "explore", "select", "confirm") for i in unit_roles[role]]
    require(len(dev) == len(official["train"]) and set(dev) == set(official["train"]), "Development roles do not partition official training")
    clusters = integer_array(root / "prepared/clusters.npz", "cluster_ids")
    require(len(clusters) == n and min(clusters) >= 0, "Statistical unit vector does not match label rows")
    if independent:
        expected = list(range(unit_n)) if metadata["label_axis"] == "graph" else [g for g in range(unit_n) for _ in range(ptr[g], ptr[g+1])]
        require(clusters == expected, "Graph statistical units differ from graph membership")
    return {
        "label_axis": metadata["label_axis"], "split_axis": "graph" if independent else "node",
        "label_rows": n, "independent_graphs": independent,
        "statistical_unit_kind": metadata["partitioner"], "statistical_units": len(set(clusters)),
        "membership_checks": "disjoint complete leaf roles; internal fit union; official train/valid/test correspondence; whole-graph mapping where applicable",
        "roles": {r: {"label_rows": len(roles[r]), "split_units": len(unit_roles[r]),
                      "nonempty_statistical_units": len({clusters[i] for i in roles[r]}),
                      "recorded_array_sha256": metadata["role_hashes"][r]} for r in ROLES},
        "official_split_counts": {k: len(v) for k,v in official.items()},
        "official_split": metadata.get("official_split", "synthetic fixture train/validation/test"),
        "split_index": metadata.get("split_index"),
        "prepared_fingerprint_recorded": metadata["fingerprint"],
    }


def category(relative):
    if relative.startswith("provenance/stages/"):
        return "stage_manifests"
    if relative in ("config.json", "environment.json", "action_registry.json"):
        return "configuration_and_environment"
    if relative.startswith("prepared/"):
        return "prepared_data_and_splits"
    if relative.endswith(".pt"):
        return "model_and_recovery_checkpoints"
    if relative.startswith("explore/trajectory_"):
        return "offline_trajectories"
    if relative.startswith(("predictions/", "member_predictions/")):
        return "candidate_and_member_predictions"
    if relative.startswith("final/"):
        return "final_predictions_and_statistics"
    if relative.startswith(("select/", "confirm/", "admit/")) or relative == "policy_lock.json":
        return "decision_and_statistical_records"
    if relative.startswith(("train/", "fit/", "provenance/")):
        return "training_and_freeze_records"
    return "ancillary_records"


def build_inventory(run_root):
    root, config, identities, sealed, origin, chain, baselines, cache = verify_run_bytes(run_root)
    metadata = read_json(safe_file(root, "prepared/metadata.json"))
    summary = read_json(safe_file(root, "final/results_summary.json"))
    require(metadata["dataset"] == config["dataset"] == summary["dataset"], "Prepared/final dataset mismatch")
    benchmark = config.get("benchmark_dataset", config["dataset"])
    require(metadata["benchmark_dataset"] == benchmark and summary["benchmark_dataset"] == benchmark, "Benchmark routing label mismatch")
    synthetic = config["dataset"] == "synthetic"
    if synthetic:
        require(config.get("scientific_run") is False and metadata.get("is_benchmark_data") is False and summary.get("scientific_run") is False,
                "A synthetic run cannot be labeled benchmark evidence")
    for relative, expected_hash in metadata["artifact_sha256"].items():
        require(sealed.get("prepared/" + relative, {}).get("sha256") == expected_hash, "Prepared metadata artifact hash differs")
    split = split_inventory(root, metadata)
    files, arrays = [], {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"Symlink in completed run: {path.relative_to(root)}")
        if not path.is_file() or path.name.endswith(pv.TEMPORARY_ARTIFACT_SUFFIXES) or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        record = pv._cached_artifact(root, relative, cache)
        files.append(dict(record, category=category(relative),
                          seal="stage_output" if relative in sealed else "linked_manifest" if relative.startswith("provenance/stages/") else "unsealed_ancillary",
                          produced_by=origin.get(relative)))
        if path.suffix in (".npz", ".npy"):
            arrays[relative] = numeric_headers(path)
    seeds = config["training"]["seeds"]
    require(seeds and len(set(seeds)) == len(seeds) and all(type(s) is int for s in seeds), "Invalid configured training seeds")
    train = read_json(safe_file(root, "train/manifest.json"))
    checkpoints = []
    for row in train["checkpoints"]:
        prefix = "q0" if row["baseline"] else "plant"
        relative = f"train/{prefix}_seed{row['seed']}_best.pt"
        require(sealed.get(relative, {}).get("sha256") == row["checkpoint_sha256"], "Training checkpoint/seed manifest differs")
        checkpoints.append({"path": relative, "seed": row["seed"], "role": "q0" if row["baseline"] else "frozen_plant",
                            "sha256": row["checkpoint_sha256"], "epochs_completed": row["epochs_completed"],
                            "checkpoint_rule": row["checkpoint_rule"], "labels": row["checkpoint_selection_labels"]})
    require({(r["seed"], r["role"]) for r in checkpoints} == {(s, r) for s in seeds for r in ("q0", "frozen_plant")} and len(checkpoints) == 2*len(seeds),
            "Each configured seed requires separate q0 and frozen-plant checkpoints")
    trajectory_paths = sorted(p for p in arrays if re.fullmatch(r"explore/trajectory_\d{6}\.npz", p))
    require(len(trajectory_paths) == config["exploration"]["trajectories"], "Saved trajectory count differs from configuration")
    trajectories = []
    for number, relative in enumerate(trajectory_paths):
        require(relative == f"explore/trajectory_{number:06d}.npz", "Trajectory numbering has a gap")
        info = read_json(safe_file(root, relative[:-4] + ".json"))
        require(info.get("trajectory") == number and info.get("seed_index") == number % len(seeds), "Trajectory seed/index metadata mismatch")
        headers = arrays[relative]
        required = {"y", "y_next", "action", "depth", "trajectory", "seed_index", "delta", "propensity", "sequence", "safe_depths", "final_loss"}
        require(required <= set(headers), "Trajectory archive is missing numeric transition fields")
        rows = headers["depth"]["shape"][0]
        require(info["records"] == rows and headers["sequence"]["shape"] == [config["model"]["depth"]], "Trajectory depth/record count mismatch")
        require(set(integer_array(root / relative, "trajectory")) == {number} and set(integer_array(root / relative, "seed_index")) == {number % len(seeds)}, "Numeric trajectory identities differ")
        graph_id = info.get("graph_id")
        if metadata["independent_graphs"]:
            require(graph_id in integer_array(root / "prepared/graph_splits.npz", "explore"), "Trajectory uses a graph outside D_explore")
            require(set(integer_array(root / relative, "graph_id")) == {graph_id}, "Trajectory numeric graph identity differs")
        trajectories.append({"path": relative, "trajectory": number, "training_seed": seeds[number % len(seeds)],
                             "records": rows, "graph_id": graph_id})
    for name in ("q0", "qN", "qC", "qD"):
        require(arrays[f"final/{name}_probabilities.npy"]["array"]["shape"] == [metadata["num_samples"], metadata["num_outputs"]], f"{name} probability shape differs from the task")
    require(summary["test_prediction_record_sha256"] == sealed["final/test_predictions.npz"]["sha256"], "Final test-record hash differs")
    if summary.get("exact_fallback"):
        require(sealed["final/qC_probabilities.npy"]["sha256"] == sealed["final/qD_probabilities.npy"]["sha256"], "Failed admission must retain byte-identical qC/qD predictions")
    counts = Counter(row["category"] for row in files)
    sizes = Counter()
    for row in files:
        sizes[row["category"]] += row["bytes"]
    return {
        "schema_version": SCHEMA, "status": "PASS", "dataset": config["dataset"], "benchmark_dataset": benchmark,
        "profile": config["profile"], "scientific_run": bool(config["scientific_run"]),
        "evidence_class": "executed_synthetic_software_example" if synthetic else "submitted_benchmark_run",
        "manuscript_history_authenticated": False,
        "identities": identities, "stages": [{"stage": m["stage"], "scientific_role": m["scientific_role"],
            "manifest_sha256": pv.sha256_file(pv.stage_manifest_path(root, m["stage"])), "outputs": len(m["outputs"])} for m in chain],
        "verification_scope": {"all_sealed_artifact_bytes": True, "complete_parent_chain": True,
            "runtime_source_matches": True, "actual_role_membership": True, "numeric_archive_headers": True,
            "logical_tensor_hashes": "recorded before/after values crosschecked against sealed post-train baseline; not recomputed",
            "checkpoint_deserialization": False, "statistical_replay": False, "checkpoint_reexecution": False,
            "limitations": "Integrity of submitted bytes does not authenticate historical execution, reproduce manuscript measurements, or certify statistical coverage."},
        "seeds": {"split_and_master_seed": config["seed"], "training_seeds": seeds,
                  "historical_manuscript_seed_identity": "not asserted"},
        "data": split, "environment": read_json(safe_file(root, "environment.json")),
        "checkpoints": checkpoints,
        "recorded_frozen_tensor_baselines": {name: record[0] for name,record in baselines.items()},
        "trajectories": {"count": len(trajectories), "transition_records": sum(t["records"] for t in trajectories), "items": trajectories},
        "decisions": {"selected_comparator": summary["comparator"]["candidate_id"], "deployed": summary["deployed"]["candidate_id"],
                      "admitted": summary["admitted"], "exact_fallback": summary["exact_fallback"],
                      "statistical_scope": summary["statistical_scope"], "seed_scope": summary["seed_scope"]},
        "categories": {c: {"files": counts[c], "bytes": sizes[c]} for c in sorted(counts)},
        "sealed_artifact_count": len(sealed), "all_file_count": len(files), "all_file_bytes": sum(r["bytes"] for r in files),
        "numeric_archives": arrays, "files": files,
    }


def inventory_markdown(report):
    rows = ["# Complete-run artifact inventory", "", f"**{report['benchmark_dataset']} — {report['evidence_class']}**", "",
            f"Verified all eight stages and {report['sealed_artifact_count']} sealed files. The directory contains {report['all_file_count']} inventoried files ({report['all_file_bytes']:,} bytes).", "",
            "This report checks artifact integrity and actual split membership. It does not deserialize checkpoints, replay statistics, authenticate historical execution, or claim agreement with paper measurements.", "",
            "## Identities", "", "| Identity | SHA-256 |", "|---|---|"]
    rows += [f"| {k} | `{v}` |" for k,v in report["identities"].items()]
    rows += ["", "## Contents", "", "| Category | Files | Bytes |", "|---|---:|---:|"]
    rows += [f"| {k.replace('_',' ')} | {v['files']} | {v['bytes']:,} |" for k,v in report["categories"].items()]
    rows += ["", "## Data and seeds", "", f"Master/split seed: `{report['seeds']['split_and_master_seed']}`. Training seeds: `{report['seeds']['training_seeds']}`. These are this run's recorded seeds; historical manuscript seed identity is not asserted.", "",
             f"Prediction axis: **{report['data']['label_axis']}**; split axis: **{report['data']['split_axis']}**; statistical units: **{report['data']['statistical_unit_kind']}**.", "",
             "| Role | Label rows | Split units | Nonempty statistical units |", "|---|---:|---:|---:|"]
    rows += [f"| {k} | {v['label_rows']} | {v['split_units']} | {v['nonempty_statistical_units']} |" for k,v in report["data"]["roles"].items()]
    rows += ["", "## Models, trajectories, and decisions", "", f"The run contains {len(report['checkpoints'])} best checkpoints, {report['trajectories']['count']} complete offline trajectories and {report['trajectories']['transition_records']} recorded transitions. Every checkpoint file is present and matches its seal. Recorded backbone/head hashes agree across stages; logical tensor hashes are not independently recomputed by this stdlib tool.", "",
             f"Admission passed: `{report['decisions']['admitted']}`. Exact qC fallback: `{report['decisions']['exact_fallback']}`. Statistical scope: {report['decisions']['statistical_scope']}. Seed scope: {report['decisions']['seed_scope']}.", "",
             "The JSON companion contains every relative path, size, SHA-256, producing stage, numeric-array shape, trajectory identity, and the captured environment. `unsealed_ancillary` records are hashed for this inventory but are not stage-sealed scientific outputs.", ""]
    return "\n".join(rows)


def literal_assignment(path, name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise InventoryError(f"Cannot locate literal {name} in {path}")


def static_manifests(source=SOURCE):
    """Generate the inventory from current files, never inferred paper seed IDs."""
    source = Path(source)
    stems = literal_assignment(source / "run_cross_morse.py", "DATASET_CONFIG_STEMS")
    tasks = literal_assignment(source / "cross_morse/tasks.py", "_DATASETS")
    roman_url = literal_assignment(source / "cross_morse/datasets.py", "ROMAN_URL")
    lrgb = literal_assignment(source / "cross_morse/datasets.py", "LRGB_SOURCES")
    rows, datasets = [], []
    for name, stem in stems.items():
        kind, axis, metric, outputs = tasks[name]
        split = {"ogbn-arxiv": "time", "ogbn-products": "sales_ranking", "ogbn-proteins": "species", "Roman-empire": "ten official train/validation/test masks"}.get(name, "official train/val/test graph files")
        raw = f"{name.replace('-', '_')}/raw/" if name.startswith("ogbn-") else "roman_empire/raw/roman_empire.npz" if name == "Roman-empire" else f"{name.lower()}/raw/{{train,val,test}}.{lrgb[name][2]}"
        datasets.append({"dataset": name, "task_type": kind, "prediction_axis": axis, "metric": metric,
            "official_outputs": outputs, "official_split": split,
            "source_documentation": "https://ogb.stanford.edu/docs/nodeprop/" if name.startswith("ogbn-") else "https://github.com/yandex-research/heterophilous-graphs" if name == "Roman-empire" else "https://github.com/vijaydwivedi75/lrgb",
            "adapter_reference": "cross_morse/datasets.py",
            "download_url": None if name.startswith("ogbn-") else roman_url if name == "Roman-empire" else lrgb[name][0],
            "download_resolution": "OGB package metadata for the named dataset/version" if name.startswith("ogbn-") else "literal adapter URL; download requires authorization flag",
            "expected_raw_cache": raw, "provider_raw_data_bundled": False,
            "historical_author_data_sha256": None,
            "split_indices": list(range(10)) if name == "Roman-empire" else None,
            "statistical_unit": "complete graph" if name in lrgb else "degree-stratified core" if name == "ogbn-proteins" else "METIS core"})
        for profile in ("smoke", "pilot", "full"):
            relative = f"configs/{stem}_{profile}.json"
            config = read_json(source / relative)
            require(config["profile"] == profile and config.get("benchmark_dataset", config["dataset"]) == name, f"Config routing mismatch: {relative}")
            rows.append({"dataset": name, "profile": profile, "path": relative,
                         "file_sha256": pv.sha256_file(source / relative), "config_sha256": pv._json_hash(config),
                         "data_kind": config["dataset"], "scientific_run": config["scientific_run"],
                         "master_and_split_seed": config["seed"], "training_seeds": config["training"]["seeds"],
                         "split_index_default": config["data"].get("split_index", 0) if name == "Roman-empire" else None,
                         "task": {"type": kind, "prediction_axis": axis, "metric": metric,
                                  "num_outputs": config["data"]["synthetic_classes"] if config["dataset"] == "synthetic" else outputs},
                         "data": config["data"], "model": config["model"], "training": config["training"],
                         "offline_trajectories": config["exploration"]["trajectories"],
                         "selection": config["selection"], "admission": config["admission"]})
    require(len(rows) == 21, "Expected seven datasets and three profiles")
    sources = {"schema_version": "cross-morse-data-sources-1", "scope": "Official sources and current loader rules; provider raw archives and historical author split hashes are not bundled.",
               "loader_file_sha256": {p: pv.sha256_file(source / p) for p in ("cross_morse/data.py", "cross_morse/datasets.py", "cross_morse/tasks.py")},
               "datasets": datasets,
               "role_split_rule": {"development_source": "official training units", "unit": "complete graph for LRGB; node for the four transductive datasets",
                   "permutation": "NumPy default_rng(config.seed).permutation(sorted official train IDs)",
                   "cumulative_floor_fractions": [0.6, 0.8, 0.9], "roles_in_order": ["fit", "explore", "select", "confirm"],
                   "internal_fit_validation": "default_rng(SeedSequence([seed,137])); max(1,floor(internal_val_fraction*fit units))", "admit": "official validation", "test": "official test",
                   "scope": "executable separated-role protocol; not a claim that these are the historical reported reused-validation memberships"},
               "actual_run_identities": "prepared/official_splits.npz, prepared/splits.npz, prepared/graph_splits.npz when applicable, prepared/clusters.npz, prepared/metadata.json",
               "checksum_policy": "Each run records source-array and prepared-file hashes. Roman and LRGB loaders also record raw input-file hashes; OGB records its version/split and array hashes, not a historical ZIP checksum. No absent archive hash or historical seed identity is invented."}
    profiles = {"schema_version": "cross-morse-configured-profiles-1", "profile_count": len(rows), "dataset_count": len(datasets),
                "scope": "Generated from committed configuration bytes. Configured seeds are not historical manuscript seed identities; profiles are not evidence that their runs completed.",
                "roman_campaign": {"dataset": "Roman-empire", "official_split_indices": list(range(10)), "single_split_is_aggregate": False},
                "profiles": rows}
    return {"data/sources.json": sources, "experiments/profiles.json": profiles}


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False, encoding="utf-8") as f:
        f.write(text)
        tmp = Path(f.name)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_root", nargs="?", type=Path)
    p.add_argument("--output", type=Path, help="JSON inventory outside the sealed run; also writes same-stem .md")
    p.add_argument("--check", action="store_true", help="Compare existing JSON and Markdown; do not write")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--check-static", action="store_true", help="Check generated source/profile manifests against current code/configs")
    group.add_argument("--write-static", action="store_true", help="Regenerate the two static JSON manifests")
    args = p.parse_args(argv)
    try:
        if args.check_static or args.write_static:
            require(args.run_root is None and args.output is None and not args.check, "Static and per-run modes are separate")
            for relative, value in static_manifests().items():
                expected = json_text(value)
                if args.check_static:
                    require((SOURCE / relative).read_text(encoding="utf-8") == expected, f"Stale static manifest: {relative}; run --write-static after reviewing changes")
                else:
                    write_atomic(SOURCE / relative, expected)
            print("PASS: seven official source definitions and 21 configured profiles")
            return 0
        require(args.run_root is not None, "Provide a complete run directory or --check-static")
        require(not args.check or args.output is not None, "--check requires --output")
        report = build_inventory(args.run_root)
        if args.output is None:
            print(json_text(report), end="")
        else:
            output = args.output.resolve()
            require(output.suffix == ".json" and not output.is_relative_to(args.run_root.resolve()), "Inventory output must be a .json path outside the sealed run")
            rendered = {output: json_text(report), output.with_suffix(".md"): inventory_markdown(report)}
            for path, text in rendered.items():
                if args.check:
                    require(path.read_text(encoding="utf-8") == text, f"Inventory differs from submitted artifacts: {path.name}")
                else:
                    require(not path.is_symlink(), "Inventory output cannot be a symlink")
                    write_atomic(path, text)
            print(f"PASS: {report['sealed_artifact_count']} sealed artifacts; {report['dataset']} / {report['benchmark_dataset']}; tensor loading and numerical replay not performed")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
