"""Scientific-contract regressions; no manuscript values are targets."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cross_morse.exploration import collect_trajectory
from cross_morse.learning import Controller, calibration_partition, fit_controller
from cross_morse.policies import execute_policy, DonorBank
from cross_morse.recovery import label_free_safe, ANCHOR_RULE
from cross_morse.scoring import action_scores
from cross_morse.statistics import global_candidate_oracle
from test_learning import records, configuration
from test_policy_guardrails import FixedMeasurementBank, InstrumentedController
from cross_morse.core import FrozenPlant, GraphOperator


class SafetyAndScoreTests(unittest.TestCase):
    def test_anchor_predicate_is_one_shared_function(self):
        import cross_morse.exploration as exploration
        import cross_morse.policies as policies
        self.assertIs(exploration.label_free_safe, policies.label_free_safe)
        self.assertIs(exploration.label_free_safe, label_free_safe)
        self.assertFalse(label_free_safe([]))
        self.assertFalse(label_free_safe([float("nan")]))
        self.assertTrue(label_free_safe([.99, 0.0]))
        self.assertFalse(label_free_safe([.991, 0.0]))

    def test_observer_invalid_anchor_is_identical_in_exploration_and_execution(self):
        torch.set_num_threads(1)
        torch.manual_seed(83)
        x = torch.randn(10, 4)
        edges = torch.stack((torch.arange(10), (torch.arange(10) + 1) % 10))
        graph = GraphOperator(torch.cat((edges, edges.flip(0)), dim=1), 10, "cpu")
        model = FrozenPlant(4, 8, 3, dropout=0).freeze()
        class Bank(FixedMeasurementBank):
            names = [f"y{i}" for i in range(25)]
            def metadata(self):
                return {"fixture": True}
        class Choices:
            def choice(self, n, size, replace=True, p=None):
                return np.asarray([1, 0, 9, 0]) if p is not None else np.asarray([], dtype=int)
        class ControllerFixture(InstrumentedController):
            def observe(self, state, y, depth, previous_action):
                out = super().observe(state, y, depth, previous_action)
                out.valid = depth != 1
                return out
        config = {"seed": 7, "model": {"depth": 4, "precision": "fp32"},
            "sensor": {"bands": [0,2,5], "degree": 8, "sketch_dim": 0},
            "exploration": {"terminal_samples_per_trajectory": 0, "critic_horizon": 1},
            "controller": {"max_search_calls": 100, "lambda_rho": .0001}}
        with patch("cross_morse.exploration.sensor_bank", side_effect=lambda *a, **k: Bank()), \
             patch("cross_morse.exploration.np.random.default_rng", return_value=Choices()):
            data, meta = collect_trajectory(model, x, graph, torch.arange(8) % 3,
                  torch.arange(8), np.arange(8) % 4, config, 0, 0, "cpu")
        candidate = {"route": "radius", "profile": "full", "depth": 4, "kappa": 1.0}
        with patch("cross_morse.policies.sensor_bank", side_effect=lambda *a, **k: Bank()):
            predicted, trace = execute_policy(model, x, graph, config, candidate,
                                             ControllerFixture(desired=[1, 0, 9, 0]))
        np.testing.assert_array_equal(data["safe_depths"], [0, 1, 1, 1])
        np.testing.assert_array_equal(trace["safe_depths"], data["safe_depths"])
        np.testing.assert_array_equal(trace["actions"], data["sequence"])
        replay, _ = execute_policy(model, x, graph, config, candidate,
                                    replay_actions=data["sequence"], replay_safe_depths=data["safe_depths"])
        np.testing.assert_array_equal(predicted, replay)
        self.assertEqual(meta["anchor_rule"], ANCHOR_RULE)
        self.assertTrue(trace["decisions"][1]["anchor_would_fail_legacy_observer_check"])
        self.assertIn("selected_score", trace["decisions"][0])

    def test_radius_changes_winner_and_matches_runtime(self):
        means = np.array([-.006, -.004]); sigmas = np.array([0., 0.])
        radii = np.array([.010, .001])
        values = action_scores(means, sigmas, radii, [0., 0.])
        self.assertEqual(int(np.argmin(means)), 0)
        self.assertEqual(int(np.argmin(values)), 1)
        controller = Controller.__new__(Controller)
        controller.config = {"model": {"depth": 6}, "controller": {"depth_bins": 2}}
        controller.calibration = {"actions": {"1": {"radius": .01}, "2": {"radius": .001}}}
        controller.features = lambda state: np.zeros(3)
        controller.supported = lambda y, depth: [1, 2]
        controller._critic_batch = lambda features, actions: (means, sigmas)
        state = SimpleNamespace(y=np.zeros(25), depth=0, valid=True)
        scores = controller.score(state, [1,2], lambda_rho=0.)
        np.testing.assert_array_equal([scores[1]["score"], scores[2]["score"]], values)
        self.assertEqual(min(scores, key=lambda a: scores[a]["score"]), 2)

    def test_configured_transport_cost_not_hardcoded(self):
        a = action_scores([0., -.001], [0.,0.], [0.,0.], [0.,1.], lambda_rho=0.)
        b = action_scores([0., -.001], [0.,0.], [0.,0.], [0.,1.], lambda_rho=.01)
        self.assertEqual(int(np.argmin(a)), 1)
        self.assertEqual(int(np.argmin(b)), 0)

    def test_invalid_scoring_configuration_fails(self):
        for kappa in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                action_scores([0.], [0.], [0.], [0.], kappa=kappa)
        with self.assertRaises(ValueError):
            action_scores([0.], [0.], [0.], [0.], kind="unknown")

    def test_headroom_does_not_mix_incompatible_global_policies(self):
        index, executable, optimistic = global_candidate_oracle([[.4,.6],[.6,.4]], [10,10])
        self.assertEqual(index, 0)
        self.assertAlmostEqual(float(executable.mean()), .5)
        self.assertAlmostEqual(float(optimistic.mean()), .4)
        with self.assertRaises(ValueError):
            global_candidate_oracle([[float("nan"), .6]], [10,10])

    def test_donor_build_does_not_rescan_every_trajectory_and_distinguishes_seeds(self):
        y = np.zeros((4,25)); y[:,0] = [.2,.2,.8,.8]
        d = {"trajectory": np.array([0,0,0,0]), "seed_index": np.array([1,1,2,2]),
             "depth": np.array([0,1,0,1]), "previous_action": np.array([-1,1,-1,1]), "y": y}
        with patch("cross_morse.policies.np.flatnonzero", side_effect=AssertionError("quadratic scan")):
            bank = DonorBank(d)
        self.assertIn((0,-1,0), bank.donors)
        self.assertIn((0,-1,2), bank.donors)


class CalibrationSeparationTests(unittest.TestCase):
    def test_three_roles_are_complete_disjoint_trajectory_partitions(self):
        data, cfg = records(), configuration()
        roles, metadata = calibration_partition(data, cfg)
        sets = [set(v) for v in metadata["groups"].values()]
        self.assertTrue(all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i)))
        self.assertEqual(sum(len(s) for s in sets), 15)
        for trajectory in np.unique(data["trajectory"]):
            self.assertEqual(len(np.unique(roles[data["trajectory"] == trajectory])), 1)
        self.assertFalse(metadata["assessment_used_for_selection"])

    def test_invalid_partition_rejected(self):
        data, cfg = records(), configuration()
        cfg["controller"]["calibration_fraction"] = .8
        cfg["controller"]["calibration_audit_fraction"] = .3
        with self.assertRaises(ValueError):
            calibration_partition(data, cfg)

    def test_assessment_target_changes_do_not_change_weights_or_radii(self):
        torch.set_num_threads(1)
        data, cfg = records(trajectories=20), configuration()
        roles, _ = calibration_partition(data, cfg)
        altered = {k: v.copy() for k, v in data.items()}
        mask = (roles == 2) & (data["action"] != 0)
        altered["delta"][mask] += 1.0
        altered["stage_delta"][roles == 2] += 1.0
        altered["terminal_delta"][roles == 2] += 1.0
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/"a", Path(directory)/"b"
            report_a = fit_controller(data, cfg, a)
            report_b = fit_controller(altered, cfg, b)
            pa = torch.load(a/"controller.pt", weights_only=False)
            pb = torch.load(b/"controller.pt", weights_only=False)
            self.assertEqual(pa["observer"], pb["observer"])
            self.assertEqual(pa["calibration"], pb["calibration"])
            for model, state in pa["models"].items():
                for name, tensor in state.items():
                    self.assertTrue(torch.equal(tensor, pb["models"][model][name]))
            self.assertEqual(report_a["calibration_assessment"]["sequence_coverage"], "NOT_MEASURED")
            self.assertEqual(report_a["final_fit"]["fitting_trajectories"], 12)
            # Failed reuse must not overwrite the original calibration declaration.
            before = (a/"calibration_partition.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                fit_controller(altered, cfg, a)
            self.assertEqual(before, (a/"calibration_partition.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
