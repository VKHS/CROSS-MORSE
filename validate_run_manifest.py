#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

SHA256 = re.compile(r"^[0-9a-f]{64}$")
DATASETS = {"ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func"}
STAGES = ("fit", "explore", "select", "confirm", "admit", "test")
REQUIRED_IO: dict[str, tuple[set[str], set[str]]] = {
    "fit": (set(), {"backbone_checkpoint", "prediction_head_checkpoint"}),
    "explore": (set(), {"logged_trajectories", "observer_model", "consequence_models"}),
    "select": (set(), {"comparator", "provisional_policy", "candidate_ledger"}),
    "confirm": ({"comparator", "provisional_policy", "confirmation_map"}, {"confirmed_policy"}),
    "admit": ({"comparator", "executable_policy"}, {"deployment_decision"}),
    "test": ({"comparator", "executable_policy", "deployment_decision"}, {"official_test_evaluation"}),
}


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def is_hash(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def validate_hash_map(value: Any, name: str, errors: list[str], *, nonempty: bool = False) -> dict[str, str]:
    if not isinstance(value, dict):
        errors.append(f"{name} must be an object mapping artifact names to SHA-256 values")
        return {}
    if nonempty and not value:
        errors.append(f"{name} must contain at least one hashed artifact")
    for key, digest in value.items():
        require(isinstance(key, str) and bool(key), f"{name} contains an invalid artifact name", errors)
        require(is_hash(digest), f"{name}.{key} must be a lowercase 64-character SHA-256", errors)
    return value


def validate(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version", "dataset", "stage", "run_id", "created_utc",
        "source_tree_sha256", "config_sha256", "dataset_version",
        "unit_manifest_sha256", "parent_manifest_sha256s", "inputs_frozen_before_stage",
        "model", "label_access", "inputs", "outputs",
    }
    require(required <= data.keys(), f"missing top-level fields: {sorted(required - data.keys())}", errors)
    require(data.get("schema_version") == "2.0", "schema_version must be 2.0", errors)
    require(data.get("dataset") in DATASETS, "unknown dataset", errors)
    stage = data.get("stage")
    require(stage in STAGES, "unknown stage", errors)
    require(isinstance(data.get("run_id"), str) and len(data.get("run_id", "")) >= 8, "run_id must contain at least 8 characters", errors)
    require(isinstance(data.get("dataset_version"), str) and bool(data.get("dataset_version")), "dataset_version is required", errors)
    try:
        datetime.fromisoformat(str(data.get("created_utc", "")).replace("Z", "+00:00"))
    except ValueError:
        errors.append("created_utc must be an ISO-8601 date-time")

    for key in ("source_tree_sha256", "config_sha256", "unit_manifest_sha256"):
        require(is_hash(data.get(key)), f"{key} must be a lowercase 64-character SHA-256", errors)

    parents = data.get("parent_manifest_sha256s")
    if not isinstance(parents, list):
        errors.append("parent_manifest_sha256s must be an array")
        parents = []
    else:
        for index, digest in enumerate(parents):
            require(is_hash(digest), f"parent_manifest_sha256s[{index}] must be a SHA-256", errors)
        require(len(parents) == len(set(parents)), "parent_manifest_sha256s must not contain duplicates", errors)

    model = data.get("model") if isinstance(data.get("model"), dict) else {}
    for key in ("backbone_sha256_before", "backbone_sha256_after", "head_sha256_before", "head_sha256_after"):
        require(is_hash(model.get(key)), f"model.{key} must be a SHA-256", errors)
    require(isinstance(model.get("frozen_during_stage"), bool), "model.frozen_during_stage must be boolean", errors)

    access = data.get("label_access") if isinstance(data.get("label_access"), dict) else {}
    require(access.get("online_decisions_label_free") is True, "label_access.online_decisions_label_free must be true", errors)
    require(access.get("test_results_used_for_decision") is False, "label_access.test_results_used_for_decision must be false", errors)
    require(isinstance(access.get("official_test_evaluation_performed"), bool), "label_access.official_test_evaluation_performed must be boolean", errors)
    require(access.get("official_test_labels_exposed_to_decision_code") is False, "label_access.official_test_labels_exposed_to_decision_code must be false", errors)

    inputs = validate_hash_map(data.get("inputs"), "inputs", errors)
    outputs = validate_hash_map(data.get("outputs"), "outputs", errors, nonempty=True)

    if stage in STAGES:
        required_inputs, required_outputs = REQUIRED_IO[stage]
        require(required_inputs <= inputs.keys(), f"stage {stage} is missing input artifacts: {sorted(required_inputs - inputs.keys())}", errors)
        require(required_outputs <= outputs.keys(), f"stage {stage} is missing output artifacts: {sorted(required_outputs - outputs.keys())}", errors)

    if stage == "fit":
        require(data.get("inputs_frozen_before_stage") is False, "fit inputs_frozen_before_stage must be false", errors)
        require(len(parents) == 0, "fit must not declare parent stage manifests", errors)
        require(access.get("official_test_evaluation_performed") is False, "fit performed official-test evaluation", errors)
        # Training may legitimately change the backbone and prediction head during D_fit.
    elif stage in STAGES:
        require(data.get("inputs_frozen_before_stage") is True, f"{stage} inputs must be frozen before the stage", errors)
        require(len(parents) >= 1, f"{stage} must cite at least one parent stage manifest", errors)
        require(model.get("frozen_during_stage") is True, f"model must remain frozen during {stage}", errors)
        require(model.get("backbone_sha256_before") == model.get("backbone_sha256_after"), f"backbone changed during {stage}", errors)
        require(model.get("head_sha256_before") == model.get("head_sha256_after"), f"prediction head changed during {stage}", errors)

    if stage == "test":
        require(access.get("official_test_evaluation_performed") is True, "test stage must explicitly record final official-test evaluation", errors)
        require(is_hash(data.get("evaluator_sha256")), "test stage requires evaluator_sha256", errors)
    elif stage in STAGES:
        require(access.get("official_test_evaluation_performed") is False, "pre-final stage performed official-test evaluation", errors)

    allowed = required | {"evaluator_sha256", "notes"}
    require(set(data) <= allowed, f"unknown top-level fields: {sorted(set(data) - allowed)}", errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a CROSS-MORSE stage manifest without external dependencies.")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        print("ERROR: manifest root must be a JSON object")
        return 1
    errors = validate(data)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("run manifest valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
