"""Behavioral regression checks for the lifecycle and label-role boundaries."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cross_morse.common import atomic_json, canonical_hash, file_hash, read_json
from cross_morse.data import prepare_data
from cross_morse.pipeline import Pipeline, atomic_npy, source_fingerprint
from cross_morse.policies import PolicyExecutionError
from cross_morse.provenance import ProvenanceError, stage_manifest_path, verify_frozen_models


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = read_json(Path(__file__).resolve().parents[1] / "configs/arxiv_smoke.json")

    def prepare(self):
        pipeline = Pipeline(self.config, self.root / "cache", self.root / "run", device="cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.stage_prepare()
        return pipeline

    def test_label_roles_are_closed_outside_the_permitted_runtime_stage(self):
        pipeline = self.prepare()
        for role in ("fit_train", "explore", "admit", "test"):
            with self.assertRaises(PermissionError):
                pipeline.data.labels(role)
        with self.assertRaises(PermissionError):
            pipeline.data.y[pipeline.data.splits["test"]]
        with pipeline.data._label_scope("train"):
            self.assertEqual(len(pipeline.data.labels("fit_train")), len(pipeline.data.splits["fit_train"]))
            with self.assertRaises(PermissionError):
                pipeline.data.labels("test")
            with self.assertRaises(PermissionError):
                pipeline.data.labels("admit")
        with pipeline.data._label_scope("fit"):
            with self.assertRaises(PermissionError):
                pipeline.data.labels("explore")  # fitting uses recorded targets
        with self.assertRaises(PermissionError):
            pipeline.data.labels("fit_train")

    def test_direct_stage_calls_cannot_skip_the_verified_predecessor(self):
        pipeline = self.prepare()
        with self.assertRaisesRegex(RuntimeError, "requires completed admit"):
            pipeline.stage_evaluate()
        with self.assertRaisesRegex(RuntimeError, "requires completed confirm"):
            pipeline.stage_admit()
        self.assertFalse((pipeline.out / "final").exists())

    def test_resume_validates_completed_artifacts_instead_of_trusting_marker(self):
        pipeline = self.prepare()
        prepared = pipeline.out / "prepared/clusters.npz"
        prepared.write_bytes(prepared.read_bytes() + b"corruption")
        resumed = Pipeline(self.config, self.root / "cache", pipeline.out, resume=True)
        with self.assertRaisesRegex(ProvenanceError, "identity/hash"):
            resumed.run("prepare")

    def test_resume_does_not_overwrite_the_original_environment_record(self):
        pipeline = self.prepare()
        before = (pipeline.out / "environment.json").read_bytes()
        with patch("cross_morse.pipeline.platform.platform", return_value="different resume host"):
            resumed = Pipeline(self.config, self.root / "cache", pipeline.out, resume=True)
            with contextlib.redirect_stdout(io.StringIO()):
                resumed.run("prepare")
        self.assertEqual(before, (pipeline.out / "environment.json").read_bytes())

    def test_a_committed_stage_recovers_a_missing_bookkeeping_marker(self):
        pipeline = self.prepare()
        manifest = stage_manifest_path(pipeline.out, "prepare")
        expected = file_hash(manifest)
        (pipeline.out / "stages/prepare.json").unlink()
        state = read_json(pipeline.out / "run_state.json")
        state.update(status="CREATED", last_complete_stage=None)
        atomic_json(pipeline.out / "run_state.json", state)
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.stage_prepare()
        self.assertEqual(expected, file_hash(manifest))
        self.assertTrue(read_json(pipeline.out / "stages/prepare.json")["recovered_after_commit"])
        self.assertEqual(read_json(pipeline.out / "run_state.json")["last_complete_stage"], "prepare")
        self.assertEqual(read_json(pipeline.out / "run_state.json")["status"], "RUNNING")

    def test_model_baseline_is_captured_at_train_and_detects_exploration_mutation(self):
        pipeline = self.prepare()
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.stage_train()
        baseline = pipeline.out / "provenance/frozen_models.json"
        self.assertTrue(baseline.exists())
        verify_frozen_models(pipeline.out)
        checkpoint = pipeline.out / "train/plant_seed1_best.pt"

        def change_frozen_model(*args, **kwargs):
            value = torch.load(checkpoint, weights_only=False, map_location="cpu")
            key = next(k for k in value["model"] if k.startswith("head."))
            value["model"][key].add_(1)
            torch.save(value, checkpoint)

        with patch("cross_morse.pipeline.explore", side_effect=change_frozen_model), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ProvenanceError):
                pipeline.stage_explore()
        self.assertFalse(stage_manifest_path(pipeline.out, "explore").exists())

    def failure_fixture(self):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.out = self.root / "failure_run"
        pipeline.config = copy.deepcopy(self.config)
        pipeline.config_hash = canonical_hash(pipeline.config)
        pipeline.source_hash = source_fingerprint()
        pipeline.device = "cpu"
        pipeline._data = prepare_data(self.root / "cache", pipeline.out, self.config).protect_labels()
        pipeline._graph = object()
        (pipeline.out / "explore").mkdir()
        np.savez(pipeline.out / "explore/nominal_reference.npz", initial_y=np.zeros((2, 25)))
        candidate = {"candidate_id": "broken", "hash": "broken-hash", "behavior_id": "broken-behavior",
                     "route": "nominal", "ensemble": 1, "profile": "none"}
        error = PolicyExecutionError("numerical_execution_failure", {
            "actions": [0, 1], "failed_depth": 2, "multiply_add_work": 123,
            "failure_stage": "plant_step", "wall_seconds": .2,
        })
        return pipeline, candidate, error

    def test_failed_development_candidate_has_diagnostics_and_no_invented_prediction(self):
        pipeline, candidate, error = self.failure_fixture()
        with patch("cross_morse.pipeline.load_plant", return_value=object()), \
             patch("cross_morse.pipeline.predict_dataset", side_effect=error):
            result = pipeline.predict_candidate(candidate)
        self.assertFalse(result["feasible"])
        self.assertFalse(result["prediction_available"])
        self.assertFalse(result["fallback_applied"])
        self.assertEqual(result["multiply_add_work"], 123)
        self.assertEqual(result["members"][0]["failed_depth"], 2)
        self.assertFalse(pipeline.probability_path(candidate).exists())
        self.assertEqual(result, pipeline.predict_candidate(candidate))

    def test_failed_confirmation_control_copies_only_locked_comparator_bytes(self):
        pipeline, candidate, error = self.failure_fixture()
        comparator = dict(candidate, candidate_id="comparator", hash="comparator-hash", behavior_id="comparator")
        atomic_json(pipeline.out / "select/candidates.json", [comparator, candidate])
        atomic_json(pipeline.out / "select/decision.json", {"comparator_index": 0})
        reference = pipeline.probability_path(comparator)
        atomic_npy(reference, np.tile(np.array([[.1, .2, .3, .4]], dtype=np.float32), (160, 1)))
        atomic_json(reference.with_suffix(".json"), {"candidate": comparator, "feasible": True,
                    "prediction_sha256": file_hash(reference)})
        failure = pipeline._failed_prediction(candidate, error, [], 1)
        with self.assertRaisesRegex(RuntimeError, "locked selected comparator"):
            pipeline._comparator_fallback(candidate, failure, candidate)
        result = pipeline._comparator_fallback(candidate, failure, comparator)
        self.assertTrue(result["fallback_applied"])
        self.assertFalse(result["feasible"])
        self.assertFalse(result["state_control_valid"])
        self.assertEqual(reference.read_bytes(), pipeline.probability_path(candidate).read_bytes())
        pipeline.verify_prediction(candidate)
        atomic_json(pipeline.out / "select/decision.json", {"comparator_index": 1})
        with self.assertRaisesRegex(RuntimeError, "locked comparator"):
            pipeline.verify_prediction(candidate)


if __name__ == "__main__":
    unittest.main()
