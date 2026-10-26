"""Trajectory separation, support bookkeeping and controller integration."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from cross_morse.core import ACTIONS
from cross_morse.learning import Controller, fit_controller, _support


def records(seed=7, trajectories=15, depth=6):
    rng = np.random.default_rng(seed)
    data = {name: [] for name in ("y", "y_next", "action", "previous_action", "depth", "trajectory",
                                  "delta", "stage_delta", "terminal_delta", "propensity", "current_loss", "seed_index")}
    for trajectory in range(trajectories):
        y = rng.normal(0, 0.1, 25).astype(np.float32)
        y[0] = 0.6
        for d in range(depth):
            action = (trajectory + d) % 10
            y[5] = d / depth
            y[6] = 0.7 * d / depth
            y[7] = 0.7 * d / depth
            next_y = 0.95 * y + rng.normal(0, 0.01, 25)
            next_y[0] = 0.6
            next_y[5] = (d + 1) / depth
            delta = 0.0 if action == 0 else (-0.004 * action + 0.001 * y[1])
            entry = {"y": y.copy(), "y_next": next_y.astype(np.float32), "action": action,
                     "previous_action": -1 if d == 0 else (trajectory + d - 1) % 10,
                     "depth": d, "trajectory": trajectory, "delta": delta,
                     "stage_delta": -0.002 + delta, "terminal_delta": -0.002 * (depth - d),
                     "propensity": ACTIONS[action].probability, "current_loss": 0.8,
                     "seed_index": 0}
            for name, value in entry.items():
                data[name].append(value)
            y = next_y.astype(np.float32)
    return {key: np.asarray(value) for key, value in data.items()}


def configuration():
    return {"seed": 1, "active_profile": "full", "model": {"depth": 6},
            "controller": {"state_dim": 4, "covariance_rank": 2, "observer_period": 2,
                           "depth_bins": 2, "crossfit_folds": 5, "epochs": 2, "patience": 2,
                           "batch_size": 32, "lr": 0.001, "weight_decay": 0.0001,
                           "calibration_min_count": 2, "kappa_grid": [0.5, 1.0, 2.0]},
            "support": {"p_min": 0.05, "n_min": 1, "max_normalized_weight": 1.0}}


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temporary.name)
        cls.data = records()
        cls.config = configuration()
        cls.audit = fit_controller(cls.data, cls.config, cls.path)
        cls.controller = Controller.load(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_cross_fitting_keeps_complete_trajectories(self):
        out = np.load(self.path / "oof_predictions.npz")
        for group in np.unique(out["trajectory_group"]):
            self.assertEqual(len(np.unique(out["fold"][out["trajectory_group"] == group])), 1)
        self.assertEqual(len(self.audit["folds"]), 5)
        self.assertTrue(np.isfinite(out["mean"]).all())
        for entry in self.audit["folds"]:
            self.assertEqual(entry["trajectory_overlap"], 0)
            self.assertIn("never the OOF fold", entry["early_stopping"])

    def test_support_never_inflates_with_repeated_depths(self):
        original = _support(self.data, np.arange(len(self.data["action"])), self.config)
        repeated = {key: np.repeat(value, 9, axis=0) for key, value in self.data.items()}
        duplicate = _support(repeated, np.arange(len(repeated["action"])), self.config)
        for key, first in original["cells"].items():
            second = duplicate["cells"][key]
            self.assertEqual(first["trajectory_count"], second["trajectory_count"])
            self.assertAlmostEqual(first["ess"], second["ess"], places=8)
            self.assertAlmostEqual(first["max_normalized_weight"], second["max_normalized_weight"], places=8)
            if first["trajectory_count"]:
                self.assertLessEqual(first["ess"], first["trajectory_count"] + 1e-8)
                self.assertLessEqual(first["ess"], 1 / first["max_normalized_weight"]**2 + 1e-8)

    def test_loaded_controller_scores_predicts_and_counts_work(self):
        controller = self.controller
        controller.reset_counters()
        state = controller.start(self.data["y"][0])
        actions = controller.supported(state.y, state.depth)
        self.assertTrue(actions)
        scores = controller.score(state, actions)
        self.assertTrue(all(np.isfinite(v["mean"]) and np.isfinite(v["sigma"]) for v in scores.values()))
        if 0 in scores:
            self.assertEqual(scores[0]["mean"], 0.0)
            self.assertEqual(scores[0]["sigma"], 0.0)
            self.assertEqual(scores[0]["radius"], 0.0)
        next_state = controller.predict(state, actions[0])
        self.assertEqual(next_state.depth, 1)
        self.assertTrue(np.isfinite(next_state.z).all())
        self.assertTrue(np.isfinite(controller.stage_cost(state, actions[0])))
        self.assertTrue(np.isfinite(controller.terminal_value(state)))
        self.assertGreater(controller.counters["model_forward_calls"], 0)
        self.assertGreater(controller.counters["approx_dense_multiply_adds"], 0)
        terminal = controller.start(state.y)
        terminal.depth = 6
        self.assertEqual(controller.terminal_value(terminal), 0.0)

    def test_unseen_stratum_and_invalid_state_are_rejected(self):
        controller = self.controller
        y = self.data["y"][0].copy()
        y[0] = 0.1  # Logging used only the medium entropy stratum.
        self.assertEqual(controller.supported(y, 0), [])
        state = controller.start(self.data["y"][0])
        state.valid = False
        scores = controller.score(state, [0, 1])
        self.assertTrue(all(np.isinf(entry["score"]) for entry in scores.values()))

    def test_observer_psd_and_missing_history_recovery(self):
        controller = self.controller
        state = controller.start(self.data["y"][0])
        state = controller.observe(state, self.data["y_next"][0], 1, int(self.data["action"][0]))
        state = controller.observe(state, self.data["y_next"][1], 2, int(self.data["action"][1]))
        self.assertTrue(np.linalg.eigvalsh(state.cov).min() > 0)
        missing_history = controller.observe(state, self.data["y"][0], 5, 0)
        self.assertFalse(missing_history.valid)
        self.assertEqual(missing_history.depth, 5)

    def test_completed_fold_resume_preserves_weights(self):
        before = {p: p.stat().st_mtime_ns for p in (self.path / "folds").glob("*/complete.pt")}
        again = fit_controller(self.data, self.config, self.path)
        after = {p: p.stat().st_mtime_ns for p in before}
        self.assertEqual(before, after)
        self.assertEqual(again["fingerprint"], self.audit["fingerprint"])
        self.assertIn("EMPIRICAL_ONLY", again["calibration"]["coverage_status"])

    def test_fingerprint_rejects_modified_training_target(self):
        modified = {key: value.copy() for key, value in self.data.items()}
        modified["delta"][1] += 0.5
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            fit_controller(modified, self.config, self.path)


if __name__ == "__main__":
    unittest.main()
