"""Lifecycle checks for separated labels, locked decisions and exact fallback."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from cross_morse.common import atomic_json, canonical_hash, file_hash, read_json
from cross_morse.pipeline import Pipeline, atomic_npy, source_fingerprint
from cross_morse.provenance import capture_frozen_models


class LockGuardedTestLabels:
    def __init__(self, values, lock_path, expected_ids):
        self.values = torch.tensor(values)
        self.lock_path = lock_path
        self.expected_ids = np.asarray(expected_ids)
        self.reads = 0

    def __getitem__(self, ids):
        if not self.lock_path.exists():
            raise AssertionError("Test labels were read before the deployment lock")
        np.testing.assert_array_equal(np.asarray(ids), self.expected_ids)
        self.reads += 1
        return self.values[ids]


class PipelineGuardrailTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        cfg_path = Path(__file__).resolve().parents[1]/"configs"/"arxiv_smoke.json"
        self.config = json.loads(cfg_path.read_text())
        self.config["selection"]["bootstrap_reps"] = 20
        self.config["admission"]["bootstrap_reps"] = 20

    def lightweight_pipeline(self):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.out = self.root
        pipeline.config = copy.deepcopy(self.config)
        pipeline.config_hash = canonical_hash(pipeline.config)
        pipeline.source_hash = source_fingerprint()
        pipeline._active_stage = None
        pipeline.device = "cpu"
        pipeline.check_frozen = Mock()
        pipeline.lock_hashes = Mock(return_value={"train/nominal_best.pt": "frozen-weight-hash"})
        return pipeline

    def candidates(self):
        return [dict(candidate_id=f"candidate_{i}", hash=f"hash_{i}", behavior_id=f"{i:08x}",
                     route=route, profile=profile, complexity=i, is_open_loop=i == 0,
                     ensemble=1, feasible=True)
                for i, (route, profile) in enumerate((("nominal", "none"),
                                                    ("one_step", "lean"), ("radius", "full")))]

    def test_resume_requires_original_source_and_configuration(self):
        run = self.root/"run"
        with patch("cross_morse.pipeline.source_fingerprint", return_value="original-source"):
            Pipeline(self.config, self.root/"data", run)
            with self.assertRaises(FileExistsError):
                Pipeline(self.config, self.root/"data", run)
            Pipeline(self.config, self.root/"data", run, resume=True)
            altered = copy.deepcopy(self.config)
            altered["seed"] += 1
            with self.assertRaisesRegex(ValueError, "source/configuration"):
                Pipeline(altered, self.root/"data", run, resume=True)
        with patch("cross_morse.pipeline.source_fingerprint", return_value="modified-source"):
            with self.assertRaisesRegex(ValueError, "source/configuration"):
                Pipeline(self.config, self.root/"data", run, resume=True)

    def test_stage_dependencies_block_evaluation_without_prior_stages(self):
        pipeline = Pipeline(self.config, self.root/"data", self.root/"run")
        pipeline.stage_evaluate = Mock()
        with self.assertRaisesRegex(RuntimeError, "requires completed admit"):
            pipeline.run("evaluate")
        pipeline.stage_evaluate.assert_not_called()
        self.assertFalse((pipeline.out/"stages"/"evaluate.json").exists())

    def test_cached_candidate_rejects_changed_bytes_or_control_identity(self):
        pipeline = self.lightweight_pipeline()
        candidate = self.candidates()[0]
        path = pipeline.probability_path(candidate)
        atomic_npy(path, np.array([[.8, .2]], dtype=np.float32))
        meta = {"candidate": candidate, "prediction_sha256": file_hash(path)}
        atomic_json(path.with_suffix(".json"), meta)
        self.assertEqual(pipeline.predict_candidate(candidate), meta)
        with self.assertRaisesRegex(RuntimeError, "identity/hash"):
            pipeline.predict_candidate(dict(candidate, state_control="matched_shuffle"))
        atomic_npy(path, np.array([[.2, .8]], dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "identity/hash"):
            pipeline.predict_candidate(candidate)

    def test_cached_member_rejects_another_seed_before_reusing_predictions(self):
        pipeline = self.lightweight_pipeline()
        candidate = self.candidates()[0]
        (self.root/"explore").mkdir()
        np.savez(self.root/"explore"/"nominal_reference.npz", initial_y=np.zeros((1, 25)), losses=np.zeros((1, 7)))
        (self.root/"train").mkdir()
        checkpoint = self.root/"train"/"plant_seed1_best.pt"
        checkpoint.write_bytes(b"frozen checkpoint fixture")
        member = self.root/"member_predictions"/"00000000_seed1.npy"
        atomic_npy(member, np.array([[.8, .2]], dtype=np.float32))
        atomic_json(member.with_suffix(".json"), {"seed": 2, "behavior_id": "00000000",
                    "prediction_sha256": file_hash(member), "checkpoint_sha256": file_hash(checkpoint)})
        with self.assertRaisesRegex(RuntimeError, "Cached member prediction"):
            pipeline.predict_candidate(candidate)

    def test_role_scoring_rejects_prediction_corruption_before_reading_labels(self):
        class ForbiddenLabels:
            reads = 0

            def __getitem__(self, ids):
                self.reads += 1
                raise AssertionError("Corrupted predictions must be rejected before scoring labels")
        pipeline = self.lightweight_pipeline()
        candidate = self.candidates()[0]
        path = pipeline.probability_path(candidate)
        atomic_npy(path, np.array([[.8, .2]], dtype=np.float32))
        atomic_json(path.with_suffix(".json"), {"candidate": candidate, "prediction_sha256": file_hash(path)})
        atomic_npy(path, np.array([[.2, .8]], dtype=np.float32))
        labels = ForbiddenLabels()
        pipeline._data = SimpleNamespace(splits={"confirm": np.array([0])}, y=labels)
        with self.assertRaisesRegex(RuntimeError, "identity/hash"):
            pipeline.role_scores(candidate, "confirm")
        self.assertEqual(labels.reads, 0)

    def test_frozen_exploration_reference_cannot_change_after_fitting(self):
        pipeline = self.lightweight_pipeline()
        reference = self.root/"explore"/"nominal_reference.npz"
        reference.parent.mkdir()
        np.savez(reference, initial_y=np.zeros((1, 25)))
        atomic_json(self.root/"fit"/"frozen_models.json", pipeline.lock_hashes())
        atomic_json(self.root/"fit"/"exploration_hashes.json", {"explore/nominal_reference.npz": file_hash(reference)})
        checkpoint = self.root / "train" / "fixture.pt"
        checkpoint.parent.mkdir()
        torch.save({"model": {"encoder.weight": torch.ones(2), "head.weight": torch.ones(2)}}, checkpoint)
        capture_frozen_models(self.root, ["train/fixture.pt"])
        Pipeline.check_frozen(pipeline)
        np.savez(reference, initial_y=np.ones((1, 25)))
        with self.assertRaisesRegex(RuntimeError, "Frozen exploration/reference"):
            Pipeline.check_frozen(pipeline)

    def test_all_registered_predictions_precede_selection_label_access(self):
        pipeline = self.lightweight_pipeline()
        candidates = self.candidates()
        atomic_json(self.root/"fit"/"registry.json", {"candidates": candidates,
                                                    "frozen_before_selection": True})
        predicted = []
        accesses = []

        def predict(candidate):
            predicted.append(candidate["candidate_id"])
            return dict(multiply_add_work=100+candidate["complexity"], serial_wall_seconds=.1, feasible=True)

        def scores(candidate, role):
            self.assertEqual(predicted, [c["candidate_id"] for c in candidates])
            self.assertEqual(role, "select")
            accesses.append(role)
            i = candidate["complexity"]
            losses = np.linspace(.8, 1.2, 8)-.01*i
            accuracy = np.full(8, .6+.01*i)
            return dict(loss=losses, accuracy=accuracy, counts=np.arange(1, 9),
                        core_ids=np.arange(8), node_ce=float(np.average(losses, weights=np.arange(1, 9))),
                        node_accuracy=float(accuracy.mean()), loss_name="cross_entropy", metric_name="accuracy")

        pipeline.predict_candidate = predict
        pipeline.role_scores = scores
        pipeline.entry_completion = Mock(return_value={"passed": True})
        pipeline._stage_select()
        self.assertEqual(accesses, ["select"]*3)
        self.assertTrue((self.root/"select"/"decision.json").exists())
        self.assertTrue((self.root/"select"/"candidate_ledger.csv").exists())
        self.assertEqual(pipeline.check_frozen.call_count, 2)

    def test_failed_nominal_ensemble_is_excluded_before_selection_labels(self):
        pipeline = self.lightweight_pipeline()
        candidates = self.candidates()
        candidates[1].update(route="nominal", profile="none", ensemble=2, is_open_loop=True)
        candidates[2].update(route="learned_stop", profile="none", ensemble=2, is_open_loop=True)
        atomic_json(self.root / "fit/registry.json", {"candidates": candidates})
        predicted, scored = [], []

        def predict(candidate):
            i = candidate["complexity"]
            predicted.append(i)
            return dict(multiply_add_work=100, serial_wall_seconds=.1,
                        feasible=i != 1, execution_failure=i == 1, failure_reason="nonfinite_predictions")

        def scores(candidate, role):
            self.assertEqual(predicted, [0, 1, 2])
            self.assertNotEqual(candidate["complexity"], 1)
            scored.append(candidate["complexity"])
            loss = np.full(8, 1 - .1*candidate["complexity"])
            return dict(loss=loss, accuracy=np.full(8, .6), counts=np.ones(8), core_ids=np.arange(8),
                        node_ce=float(loss.mean()), node_accuracy=.6, loss_name="cross_entropy", metric_name="accuracy")

        pipeline.predict_candidate = predict
        pipeline.role_scores = scores
        pipeline.entry_completion = Mock(return_value={"passed": True})
        pipeline._stage_select()
        decided = read_json(self.root / "select/decision.json")
        recorded = read_json(self.root / "select/candidates.json")
        self.assertEqual(decided["comparator_index"], 0)
        self.assertEqual(scored, [0, 2])
        self.assertIsNone(decided["mean"][1])
        self.assertFalse(recorded[2]["feasible"])
        self.assertEqual(recorded[2]["ineligibility_reason"], "same-ensemble nominal reference unavailable")

    def test_failed_required_predecessor_is_unavailable_confirmation_evidence(self):
        pipeline = self.lightweight_pipeline()
        candidates = self.candidates()
        candidates[1].update(execution_failure=True, feasible=False)
        candidates[2].update(profile="lean", execution_failure=False)
        atomic_json(self.root / "select/candidates.json", candidates)
        atomic_json(self.root / "select/decision.json", {
            "comparator_index": 0, "provisional_index": 2, "eligible_indices": [2]})

        def scores(candidate, role):
            self.assertNotEqual(candidate["candidate_id"], candidates[1]["candidate_id"])
            loss = .5 if candidate.get("state_control") else .3 - .1*candidate["complexity"]
            return dict(loss=np.full(8, loss), accuracy=np.full(8, .6), counts=np.ones(8),
                        core_ids=np.arange(8), node_ce=loss, node_accuracy=.6)

        pipeline.role_scores = scores
        pipeline.predict_candidate = Mock(return_value={"shuffle_missing_sites": 0, "shuffle_matched_sites": 8,
                                                        "feasible": True, "state_control_valid": True})
        with patch("cross_morse.pipeline.load_records", return_value={}), \
             patch("cross_morse.pipeline.DonorBank", return_value=object()):
            pipeline._stage_confirm()
        decision = read_json(self.root / "confirm/decision.json")
        missing = next(c for c in decision["contrasts"] if c["kind"] == "radius")
        self.assertFalse(missing["available"])
        self.assertIsNone(missing["mean"])
        self.assertFalse(missing["passes"])
        self.assertEqual(missing["p_value_zero"], 1)
        self.assertEqual(decision["confirmed_index"], 0)

    def test_admission_locks_candidate_before_labels_and_rejects_to_comparator(self):
        pipeline = self.lightweight_pipeline()
        candidates = self.candidates()
        atomic_json(self.root/"select"/"candidates.json", candidates)
        atomic_json(self.root/"confirm"/"decision.json", {"comparator_index": 0, "confirmed_index": 1})
        accesses = []

        def scores(candidate, role):
            self.assertTrue((self.root/"admit"/"pre_admission_lock.json").exists())
            self.assertFalse(read_json(self.root/"admit"/"pre_admission_lock.json")["test_labels_accessed"])
            self.assertEqual(role, "admit")
            accesses.append(role)
            return dict(loss=np.ones(8)+.02*candidate["complexity"], accuracy=np.full(8, .6),
                        core_ids=np.arange(8), counts=np.ones(8))

        pipeline.role_scores = scores
        pipeline._stage_admit()
        decision = read_json(self.root/"admit"/"decision.json")
        self.assertFalse(decision["passed"])
        self.assertEqual(decision["deployed_index"], 0)
        self.assertTrue(decision["exact_comparator_fallback"])
        self.assertEqual(read_json(self.root/"policy_lock.json")["deployed_hash"], candidates[0]["hash"])
        self.assertEqual(accesses, ["admit", "admit"])

    def evaluation_fixture(self):
        pipeline = self.lightweight_pipeline()
        pipeline.config["data"]["synthetic_classes"] = 2
        pipeline.config_hash = canonical_hash(pipeline.config)
        candidates = self.candidates()
        atomic_json(self.root/"select"/"candidates.json", candidates)
        atomic_json(self.root/"admit"/"decision.json", {"passed": False})
        atomic_json(self.root/"policy_lock.json", {
            "config_sha256": pipeline.config_hash,
            "admission_sha256": file_hash(self.root/"admit"/"decision.json"),
            "registry_sha256": canonical_hash(candidates),
            "comparator_index": 0, "deployed_index": 0,
        })
        probabilities = np.array([[.8, .2], [.4, .6], [.6, .4], [.3, .7], [.8, .2], [.2, .8]], dtype=np.float32)
        path = pipeline.probability_path(candidates[0])
        atomic_npy(path, probabilities)
        atomic_json(path.with_suffix(".json"), {"candidate": candidates[0], "prediction_sha256": file_hash(path)})
        labels = LockGuardedTestLabels([0, 1, 0, 1, 0, 1], self.root/"policy_lock.json", [4, 5])
        pipeline._data = SimpleNamespace(x=torch.zeros((6, 4)), y=labels,
                                        splits={"test": np.array([4, 5])},
                                        cluster_ids=np.array([0, 1, 0, 1, 0, 1]))
        pipeline._graph = object()
        return pipeline, labels, probabilities

    def test_final_rejected_predictions_are_byte_identical_and_labels_read_after_lock(self):
        pipeline, labels, probabilities = self.evaluation_fixture()
        with patch("cross_morse.pipeline.load_plant", return_value=object()), \
             patch("cross_morse.pipeline.predict_dataset", return_value=(probabilities, {})):
            pipeline._stage_evaluate()
        self.assertEqual(labels.reads, 1)
        self.assertEqual((self.root/"final"/"qC_probabilities.npy").read_bytes(),
                         (self.root/"final"/"qD_probabilities.npy").read_bytes())
        summary = read_json(self.root/"final"/"results_summary.json")
        self.assertTrue(summary["exact_fallback"])
        self.assertFalse(summary["test_can_change_policy"])
        self.assertFalse(summary["original_manuscript_results_reproduced"])

    def test_modified_registry_is_rejected_before_any_test_label_read(self):
        pipeline, labels, _ = self.evaluation_fixture()
        candidates = read_json(self.root/"select"/"candidates.json")
        candidates[0]["ensemble"] = 2
        atomic_json(self.root/"select"/"candidates.json", candidates)
        with self.assertRaisesRegex(RuntimeError, "registry changed"):
            pipeline._stage_evaluate()
        self.assertEqual(labels.reads, 0)

    def test_final_copy_rejects_prediction_corruption_before_any_test_label_read(self):
        pipeline, labels, probabilities = self.evaluation_fixture()
        path = pipeline.probability_path(self.candidates()[0])
        atomic_npy(path, probabilities[:, ::-1].copy())
        with self.assertRaisesRegex(RuntimeError, "identity/hash"):
            pipeline._stage_evaluate()
        self.assertEqual(labels.reads, 0)
        self.assertFalse((self.root/"final"/"results_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
