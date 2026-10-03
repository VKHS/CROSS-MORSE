from __future__ import annotations

import copy
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_run_manifest import validate

H = "0" * 64
P = "1" * 64


def base(stage: str) -> dict:
    outputs = {
        "fit": {"backbone_checkpoint": H, "prediction_head_checkpoint": P},
        "explore": {"logged_trajectories": H, "observer_model": P, "consequence_models": H},
        "select": {"comparator": H, "provisional_policy": P, "candidate_ledger": H},
        "confirm": {"confirmed_policy": P},
        "admit": {"deployment_decision": H},
        "test": {"official_test_evaluation": P},
    }[stage]
    inputs = {
        "fit": {},
        "explore": {"backbone_checkpoint": H, "prediction_head_checkpoint": P},
        "select": {"observer_model": P},
        "confirm": {"comparator": H, "provisional_policy": P, "confirmation_map": H},
        "admit": {"comparator": H, "executable_policy": P},
        "test": {"comparator": H, "executable_policy": P, "deployment_decision": H},
    }[stage]
    data = {
        "schema_version": "2.0",
        "dataset": "ogbn-arxiv",
        "stage": stage,
        "run_id": f"example-{stage}",
        "created_utc": "2026-09-01T12:00:00Z",
        "source_tree_sha256": H,
        "config_sha256": P,
        "dataset_version": "official-v1",
        "unit_manifest_sha256": H,
        "parent_manifest_sha256s": [] if stage == "fit" else [P],
        "inputs_frozen_before_stage": stage != "fit",
        "model": {
            "backbone_sha256_before": H,
            "backbone_sha256_after": P if stage == "fit" else H,
            "head_sha256_before": H,
            "head_sha256_after": P if stage == "fit" else H,
            "frozen_during_stage": stage != "fit",
        },
        "label_access": {
            "online_decisions_label_free": True,
            "official_test_evaluation_performed": stage == "test",
            "official_test_labels_exposed_to_decision_code": False,
            "test_results_used_for_decision": False,
        },
        "inputs": inputs,
        "outputs": outputs,
    }
    if stage == "test":
        data["evaluator_sha256"] = H
    return data


class RunManifestValidationTests(unittest.TestCase):
    def test_fit_may_change_backbone_and_head(self) -> None:
        self.assertEqual(validate(base("fit")), [])

    def test_post_fit_model_change_is_rejected(self) -> None:
        data = base("select")
        data["model"]["backbone_sha256_after"] = P
        self.assertTrue(any("backbone changed" in item for item in validate(data)))

    def test_pre_final_test_access_is_rejected(self) -> None:
        data = base("confirm")
        data["label_access"]["official_test_evaluation_performed"] = True
        self.assertTrue(any("official-test" in item for item in validate(data)))

    def test_test_requires_frozen_deployment_inputs(self) -> None:
        data = base("test")
        del data["inputs"]["deployment_decision"]
        self.assertTrue(any("deployment_decision" in item for item in validate(data)))

    def test_each_stage_example_is_valid(self) -> None:
        for stage in ("fit", "explore", "select", "confirm", "admit", "test"):
            with self.subTest(stage=stage):
                self.assertEqual(validate(copy.deepcopy(base(stage))), [])


if __name__ == "__main__":
    unittest.main()
