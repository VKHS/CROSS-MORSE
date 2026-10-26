"""Dataset-task training, whole-graph minibatches, and AP checkpoint selection."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from cross_morse import common
from cross_morse.data import GraphData
from cross_morse.core import GraphOperator
from cross_morse.tasks import task_spec
from cross_morse.training import (_checkpoint_improved, audit_training,
                                   train_one)


class NoCorpusTransfer:
    """Permit CPU graph slicing but reject an accidental all-union device copy."""
    def __init__(self, tensor):
        self.tensor = tensor
        self.shape = tensor.shape

    def __getitem__(self, index):
        return self.tensor[index]

    def to(self, *args, **kwargs):
        raise AssertionError("Independent training must not move the whole corpus to a device")


def graph_fixture(dataset="Peptides-func"):
    graphs, nodes, outputs = 14, 4, 3
    gen = torch.Generator().manual_seed(17)
    ptr = np.arange(graphs + 1) * nodes
    local = torch.stack((torch.arange(nodes), (torch.arange(nodes) + 1) % nodes))
    local = torch.cat((local, local.flip(0)), 1)
    edge_index = torch.cat([local + i * nodes for i in range(graphs)], 1)
    edge_ptr = np.arange(graphs + 1) * local.shape[1]
    roles = {"fit_train": np.arange(4), "fit_val": np.arange(4, 8), "fit": np.arange(8),
             "explore": np.array([8]), "select": np.array([9]), "confirm": np.array([10]),
             "admit": np.array([11]), "test": np.arange(12, 14)}
    spec = task_spec(dataset, {"num_outputs": outputs})
    graph_level = spec.label_axis == "graph"
    y = ((torch.arange(graphs)[:, None] + torch.arange(outputs)[None, :]) % 2).float() if graph_level else torch.arange(graphs * nodes) % outputs
    splits = {role: ids.copy() if graph_level else np.concatenate([np.arange(ptr[g], ptr[g+1]) for g in ids]) for role, ids in roles.items()}
    data = GraphData(
        x=NoCorpusTransfer(torch.randn(graphs * nodes, 4, generator=gen)),
        edge_index=edge_index, y=y, splits=splits,
        cluster_ids=np.arange(graphs) if graph_level else np.repeat(np.arange(graphs), nodes),
        metadata={**spec.to_dict(), "label_axis": spec.label_axis, "num_classes": outputs,
                  "num_samples": graphs if graph_level else graphs * nodes, "independent_graphs": True},
        graph_ptr=ptr, edge_ptr=edge_ptr, graph_splits=roles,
        edge_weight=torch.ones(edge_index.shape[1]),
    )
    config = {
        "dataset": dataset, "seed": 44,
        "model": {"backbone": "linear_gine" if graph_level else "gps", "depth": 8,
                  "native_layers": 2, "hidden_dim": 8, "encoder_layers": 1,
                  "attention_heads": 2, "dropout": .1, "activation": "relu",
                  "precision": "fp32", "checkpoint_segment": 3},
        "training": {"epochs": 3, "patience": 4, "lr": .01, "weight_decay": .0001,
                     "graph_batch_size": 2, "grad_clip": 1., "seeds": [5]},
        "selection": {"bootstrap_reps": 32, "alpha": .025},
        "entry": {"plant_margin": .003, "pdd_max": 1., "dil_gap_max": 1.},
    }
    return data, config


class TaskTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.stop = common.STOP_REQUESTED
        common.STOP_REQUESTED = False

    def tearDown(self):
        common.STOP_REQUESTED = self.stop

    def test_peptides_checkpoint_rule_uses_ap_then_bce(self):
        spec = task_spec("Peptides-func")
        self.assertTrue(_checkpoint_improved(spec, .8, .7, .5, .6))
        self.assertFalse(_checkpoint_improved(spec, .2, .5, .5, .6))
        self.assertTrue(_checkpoint_improved(spec, .4, .6, .5, .6))
        self.assertFalse(_checkpoint_improved(spec, .7, .6, .5, .6))
        with self.assertRaisesRegex(ValueError, "checkpoint selection"):
            _checkpoint_improved(spec, .3, float("nan"), .5, .6)

    def test_complete_graph_training_and_audit_never_copy_the_union(self):
        for dataset, metric in [("Peptides-func", "ap"), ("PascalVOC-SP", "macro_f1")]:
            data, config = graph_fixture(dataset)
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                with data._label_scope("train") as accesses:
                    for baseline in (True, False):
                        result = train_one(data, None, config, directory, 5, "cpu", baseline)
                        self.assertEqual(result["task"]["metric"], metric)
                        self.assertEqual(result["graph_batch_size"], 2)
                        self.assertFalse(result["official_validation_used"])
                    audit = audit_training(data, None, config, directory, "cpu")
                self.assertEqual({row["role"] for row in accesses}, {"fit_train", "fit_val"})
                self.assertTrue(audit["identity_pass"])
                self.assertTrue(audit["metric_available"])
                self.assertEqual(audit["metric_name"], metric)
                self.assertEqual(audit["risk_weighting"], "equal graphs after within-graph mean")
                self.assertEqual(audit["instrumentation"]["native_site_count"], 2)
                self.assertEqual(audit["instrumentation"]["inserted_site_count"], 6)

    def test_graph_training_epoch_boundary_resume_is_exact(self):
        data, config = graph_fixture()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            complete, interrupted = Path(directory) / "complete", Path(directory) / "interrupted"
            with data._label_scope("train"):
                train_one(data, None, config, complete, 5, "cpu")
            common.STOP_REQUESTED = True
            with data._label_scope("train"), self.assertRaises(InterruptedError):
                train_one(data, None, config, interrupted, 5, "cpu")
            common.STOP_REQUESTED = False
            with data._label_scope("train"):
                train_one(data, None, config, interrupted, 5, "cpu")
            a = torch.load(complete / "plant_seed5_last.pt", weights_only=False)
            b = torch.load(interrupted / "plant_seed5_last.pt", weights_only=False)
            self.assertEqual(a["history"], b["history"])
            self.assertEqual(a["best_metric"], b["best_metric"])
            self.assertEqual(a["scheduler"], b["scheduler"])
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)

    def test_forbidden_graph_labels_cannot_change_checkpoint(self):
        first, config = graph_fixture()
        second = copy.deepcopy(first)
        second.y[8:] = 1 - second.y[8:]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            a_dir, b_dir = Path(directory) / "a", Path(directory) / "b"
            with first._label_scope("train"):
                train_one(first, None, config, a_dir, 5, "cpu")
            with second._label_scope("train"):
                train_one(second, None, config, b_dir, 5, "cpu")
            a = torch.load(a_dir / "plant_seed5_best.pt", weights_only=False)
            b = torch.load(b_dir / "plant_seed5_best.pt", weights_only=False)
            self.assertEqual(a["epoch"], b["epoch"])
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)

    def test_multilabel_node_training_records_bce_and_rocauc(self):
        nodes = 24
        gen = torch.Generator().manual_seed(77)
        edges = torch.stack((torch.arange(nodes), (torch.arange(nodes) + 1) % nodes))
        edges = torch.cat((edges, edges.flip(0)), 1)
        graph = GraphOperator(edges, nodes, edge_weight=torch.linspace(.2, 1.2, edges.shape[1]))
        roles = {"fit_train": np.arange(12), "fit_val": np.arange(12, 16), "fit": np.arange(16),
                 "explore": np.array([16]), "select": np.array([17]), "confirm": np.array([18]),
                 "admit": np.array([19]), "test": np.arange(20, 24)}
        spec = task_spec("ogbn-proteins", {"num_outputs": 3})
        y = ((torch.arange(nodes)[:, None] + torch.arange(3)[None, :]) % 2).float()
        y[2, 1] = float("nan")
        data = GraphData(torch.randn(nodes, 4, generator=gen), edges, y, roles, np.arange(nodes) % 4,
                         {**spec.to_dict(), "num_classes": 3}, edge_weight=graph.edge_weight)
        config = {"dataset": "ogbn-proteins",
                  "model": {"backbone": "edge_weighted_transport", "hidden_dim": 8,
                            "depth": 3, "encoder_layers": 2, "dropout": .1, "activation": "relu",
                            "precision": "fp32", "checkpoint_segment": 2},
                  "training": {"epochs": 2, "patience": 3, "lr": .01, "weight_decay": .0001, "seeds": [4]}}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), data._label_scope("train"):
            result = train_one(data, graph, config, directory, 4, "cpu")
            checkpoint = torch.load(Path(directory) / "plant_seed4_best.pt", weights_only=False)
        self.assertEqual(result["task"]["metric"], "rocauc")
        self.assertEqual(result["checkpoint_rule"], "min_loss")
        self.assertIn("best_internal_fit_val_bce", result)
        self.assertNotIn("best_internal_fit_val_ce", result)
        self.assertEqual(checkpoint["loss_name"], "bce")


if __name__ == "__main__":
    unittest.main()
