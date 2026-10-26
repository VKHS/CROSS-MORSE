"""Training-label isolation, checkpoint semantics and gradient correctness."""
from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from cross_morse import common
from cross_morse.core import FrozenPlant, GraphOperator
from cross_morse.training import native_forward, new_plant, train_one


class FitLabelsOnly:
    """An executable firewall, rather than an assertion about a split manifest."""
    def __init__(self, labels, allowed):
        self._labels = labels
        self.allowed = set(int(x) for x in allowed)
        self.read_nodes = set()

    def __getitem__(self, index):
        ids = torch.as_tensor(index).long().cpu()
        actual = set(int(x) for x in ids.reshape(-1))
        if not actual <= self.allowed:
            raise AssertionError(f"Training read labels outside Dfit: {actual - self.allowed}")
        self.read_nodes |= actual
        return self._labels[ids]

    def max(self):
        raise AssertionError("Architecture may not infer class count from all-stage labels")


def fixture():
    generator = torch.Generator().manual_seed(314)
    n = 24
    edges = torch.stack((torch.arange(n), (torch.arange(n) + 1) % n))
    edges = torch.cat((edges, edges.flip(0)), dim=1)
    graph = GraphOperator(edges, n, "cpu")
    data = SimpleNamespace(
        x=torch.randn(n, 4, generator=generator),
        y=torch.arange(n) % 3,
        metadata={"num_classes": 3},
        splits={"fit_train": np.arange(12), "fit_val": np.arange(12, 16),
                "admit": np.arange(16, 20), "test": np.arange(20, 24)},
        cluster_ids=np.arange(n) % 4,
    )
    config = {
        "model": {"hidden_dim":8, "encoder_layers":4, "dropout":.15,
                  "activation":"relu", "depth":3, "precision":"fp32", "checkpoint_segment":2},
        "training": {"epochs":3, "patience":10, "lr":.01, "weight_decay":.0001,
                     "grad_clip":1., "seeds":[5]},
    }
    return data, graph, config


class TrainingGuardrailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._threads)

    def setUp(self):
        self._stop = common.STOP_REQUESTED
        common.STOP_REQUESTED = False

    def tearDown(self):
        common.STOP_REQUESTED = self._stop

    def test_model_architecture_uses_metadata_without_any_label_read(self):
        data, _, config = fixture()
        data.y = FitLabelsOnly(data.y, [])
        model = new_plant(data, config, "cpu")
        self.assertEqual(model.num_classes, 3)
        self.assertFalse(data.y.read_nodes)

    def test_training_never_reads_admission_or_test_labels(self):
        data, graph, config = fixture()
        labels = FitLabelsOnly(data.y, np.arange(16))
        data.y = labels
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            for baseline in (True, False):
                result = train_one(data, graph, config, directory, 5, "cpu", baseline)
                self.assertFalse(result["official_validation_used"])
        self.assertEqual(labels.read_nodes, set(range(16)))

    def test_changing_forbidden_labels_cannot_change_checkpoint_weights(self):
        data, graph, config = fixture()
        changed = copy.deepcopy(data)
        changed.y[16:] = (changed.y[16:] + 1) % 3
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            first, second = Path(directory) / "first", Path(directory) / "second"
            train_one(data, graph, config, first, 5, "cpu")
            train_one(changed, graph, config, second, 5, "cpu")
            a = torch.load(first / "plant_seed5_best.pt", map_location="cpu", weights_only=False)
            b = torch.load(second / "plant_seed5_best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(a["epoch"], b["epoch"])
            self.assertEqual(a["internal_fit_val_ce"], b["internal_fit_val_ce"])
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)

    def test_checkpointed_segments_preserve_outputs_and_gradients(self):
        data, graph, config = fixture()
        torch.manual_seed(91)
        ordinary = FrozenPlant(4, 8, 3, dropout=0)
        checkpointed = copy.deepcopy(ordinary)
        logits1 = native_forward(ordinary, data.x, graph, 5, checkpoint_segment=0)
        logits2 = native_forward(checkpointed, data.x, graph, 5, checkpoint_segment=2)
        torch.testing.assert_close(logits1, logits2)
        F.cross_entropy(logits1[:12], data.y[:12]).backward()
        F.cross_entropy(logits2[:12], data.y[:12]).backward()
        for (name1, p1), (name2, p2) in zip(ordinary.named_parameters(), checkpointed.named_parameters()):
            self.assertEqual(name1, name2)
            self.assertIsNotNone(p1.grad, name1)
            self.assertIsNotNone(p2.grad, name2)
            torch.testing.assert_close(p1.grad, p2.grad, atol=1e-7, rtol=2e-5)

    def test_safe_boundary_resume_matches_uninterrupted_training(self):
        data, graph, config = fixture()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            complete = Path(directory) / "complete"
            interrupted = Path(directory) / "interrupted"
            train_one(data, graph, config, complete, 5, "cpu")
            common.STOP_REQUESTED = True
            with self.assertRaises(InterruptedError):
                train_one(data, graph, config, interrupted, 5, "cpu")
            self.assertTrue((interrupted / "plant_seed5_last.pt").exists())
            self.assertFalse((interrupted / "plant_seed5_done.json").exists())
            common.STOP_REQUESTED = False
            train_one(data, graph, config, interrupted, 5, "cpu")
            a = torch.load(complete / "plant_seed5_last.pt", map_location="cpu", weights_only=False)
            b = torch.load(interrupted / "plant_seed5_last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(a["history"], b["history"])
            self.assertEqual(a["next_epoch"], b["next_epoch"])
            self.assertEqual(a["scheduler"], b["scheduler"])
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)


if __name__ == "__main__":
    unittest.main()
