"""Nominal-relative counterfactual target and label-free action-state checks."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from cross_morse.core import FrozenPlant, GraphOperator
from cross_morse.exploration import collect_trajectory, label_free_safe


class FixedExplorationChoices:
    def choice(self, choices, size, replace=True, p=None):
        if p is not None:
            return np.array([0, 3, 8, 2, 9, 7], dtype=np.int64)
        return np.array([1, 4], dtype=np.int64)


class ExplorationTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._threads)

    def setUp(self):
        torch.manual_seed(84)
        self.x = torch.randn(12, 4)
        edges = torch.stack((torch.arange(12), (torch.arange(12) + 1) % 12))
        self.graph = GraphOperator(torch.cat((edges, edges.flip(0)), dim=1), 12, "cpu")
        self.model = FrozenPlant(4, 8, 3, dropout=0).freeze()
        self.ids = torch.arange(8)
        self.labels = torch.arange(8) % 3
        self.cores = np.arange(8) % 4
        self.config = {
            "seed": 7,
            "model": {"depth":6, "precision":"fp32"},
            "sensor": {"bands":[0,2,5], "degree":8, "sketch_dim":4},
            "exploration": {"terminal_samples_per_trajectory":2,
                            "critic_horizon":3, "record_stride":1},
        }

    def collect(self, labels=None):
        with patch("cross_morse.exploration.np.random.default_rng", return_value=FixedExplorationChoices()):
            return collect_trajectory(self.model, self.x, self.graph,
                                      self.labels if labels is None else labels,
                                      self.ids, self.cores, self.config, 0, 0, "cpu")

    def loss(self, h):
        return float(F.cross_entropy(self.model.logits(h)[self.ids].float(), self.labels))

    def check_targets(self, horizon):
        self.config["exploration"]["critic_horizon"] = horizon
        records, _ = self.collect()
        h0 = self.model.encode(self.x)
        h, safe = h0, h0
        for depth, action in enumerate(records["sequence"]):
            if label_free_safe(records["y"][depth]):
                safe = h.clone()
            actual = self.model.step(h, h0, self.graph, int(action), safe)
            alternate = actual
            nominal = self.model.native_step(h, h0, self.graph)
            for _ in range(1, min(horizon, 6-depth)):
                alternate = self.model.native_step(alternate, h0, self.graph)
                nominal = self.model.native_step(nominal, h0, self.graph)
            expected = self.loss(alternate) - self.loss(nominal)
            self.assertAlmostEqual(float(records["delta"][depth]), expected, places=6)
            self.assertAlmostEqual(float(records["stage_delta"][depth]), self.loss(actual)-self.loss(h), places=6)
            if depth in (1, 4):
                terminal = h
                for _ in range(6-depth):
                    terminal = self.model.native_step(terminal, h0, self.graph)
                self.assertAlmostEqual(float(records["terminal_delta"][depth]), self.loss(terminal)-self.loss(h), places=6)
            else:
                self.assertTrue(np.isnan(records["terminal_delta"][depth]))
            h = actual
        self.assertEqual(float(records["delta"][0]), 0.0)  # A0 against itself.

    def test_one_step_target_is_action_minus_nominal_not_before_after(self):
        self.check_targets(horizon=1)

    def test_multi_step_target_uses_nominal_tail_in_both_branches(self):
        self.check_targets(horizon=3)

    def test_offline_label_changes_leave_measurements_and_behavior_unchanged(self):
        original, _ = self.collect()
        altered, _ = self.collect((self.labels + 1) % 3)
        for name in ("sequence", "y", "y_next", "action", "propensity", "depth"):
            np.testing.assert_array_equal(original[name], altered[name])
        self.assertFalse(np.array_equal(original["delta"], altered["delta"]))
        self.assertEqual(self.graph.num_nodes, 12)


if __name__ == "__main__":
    unittest.main()
