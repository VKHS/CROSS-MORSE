from __future__ import annotations

import json
import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from release_tools.guardrails import StageLedger, StageViolation
from release_tools.hashing import sha256_bytes, sha256_json


class HashingTests(unittest.TestCase):
    def test_sha256_bytes(self) -> None:
        self.assertEqual(sha256_bytes(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_json_hash_is_key_order_invariant(self) -> None:
        self.assertEqual(sha256_json({"a": 1, "b": 2}), sha256_json({"b": 2, "a": 1}))


class StageLedgerTests(unittest.TestCase):
    def test_full_stage_order_and_freezes(self) -> None:
        ledger = StageLedger()
        for stage in ("fit", "explore", "select", "confirm"):
            ledger.complete(stage)
        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp) / "policy.json"
            comparator = Path(tmp) / "comparator.json"
            decision = Path(tmp) / "deployment.json"
            policy.write_text(json.dumps({"route": "radius"}), encoding="utf-8")
            comparator.write_text(json.dumps({"route": "fixed-sequence"}), encoding="utf-8")
            decision.write_text(json.dumps({"G_admit": 1}), encoding="utf-8")
            ledger.freeze_executable_policy(policy, comparator)
            ledger.complete("admit")
            ledger.freeze_deployment_decision(decision)
            ledger.complete("test")
        self.assertEqual(ledger.completed[-1], "test")
        self.assertTrue(all(ledger.frozen_identities.values()))

    def test_admission_requires_frozen_policy(self) -> None:
        ledger = StageLedger()
        for stage in ("fit", "explore", "select", "confirm"):
            ledger.complete(stage)
        with self.assertRaises(StageViolation):
            ledger.complete("admit")

    def test_test_requires_frozen_deployment_decision(self) -> None:
        ledger = StageLedger()
        for stage in ("fit", "explore", "select", "confirm"):
            ledger.complete(stage)
        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp) / "policy.json"
            comparator = Path(tmp) / "comparator.json"
            policy.write_text("{}", encoding="utf-8")
            comparator.write_text("{}", encoding="utf-8")
            ledger.freeze_executable_policy(policy, comparator)
            ledger.complete("admit")
            with self.assertRaises(StageViolation):
                ledger.complete("test")

    def test_pre_final_test_label_access_fails(self) -> None:
        ledger = StageLedger()
        with self.assertRaises(StageViolation):
            ledger.assert_pre_final_test_blind(["D_fit", "D_test"])

    def test_post_test_decision_fails(self) -> None:
        ledger = StageLedger()
        for stage in ("fit", "explore", "select", "confirm"):
            ledger.complete(stage)
        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp) / "policy.json"
            comparator = Path(tmp) / "comparator.json"
            decision = Path(tmp) / "deployment.json"
            for path in (policy, comparator, decision):
                path.write_text("{}", encoding="utf-8")
            ledger.freeze_executable_policy(policy, comparator)
            ledger.complete("admit")
            ledger.freeze_deployment_decision(decision)
            ledger.complete("test")
        with self.assertRaises(StageViolation):
            ledger.assert_no_post_test_decision()


if __name__ == "__main__":
    unittest.main()
