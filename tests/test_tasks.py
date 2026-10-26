"""Task loss/metric correctness, missing-label masks, and numeric-only replay."""
import json
import subprocess
import sys
import unittest

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from cross_morse.tasks import (TaskSpec, confidence_statistics, loss, metric_score,
                               probabilities, sample_losses, sample_metric_values,
                               task_spec, valid_target_mask)


class TaskMathTests(unittest.TestCase):
    def test_all_seven_task_definitions_and_synthetic_cardinality(self):
        expected = {
            "ogbn-arxiv": ("multiclass", "node", "accuracy", 40),
            "ogbn-products": ("multiclass", "node", "accuracy", 47),
            "ogbn-proteins": ("multilabel", "node", "rocauc", 112),
            "Roman-empire": ("multiclass", "node", "accuracy", 18),
            "PascalVOC-SP": ("multiclass", "node", "macro_f1", 21),
            "COCO-SP": ("multiclass", "node", "macro_f1", 81),
            "Peptides-func": ("multilabel", "graph", "ap", 10),
        }
        for name, values in expected.items():
            spec = task_spec(name)
            self.assertEqual((spec.task_type, spec.prediction_level, spec.metric, spec.num_outputs), values)
            fixture = task_spec({"dataset": "synthetic", "benchmark_dataset": name,
                                 "data": {"synthetic_classes": 4}})
            self.assertEqual(fixture.num_outputs, 4)
            self.assertEqual(fixture.metric, spec.metric)
            self.assertEqual(task_spec(fixture.to_dict()), fixture)
        self.assertEqual(task_spec({"dataset": "synthetic", "benchmark_dataset": "ogbn-proteins",
                                    "data": {"synthetic_classes": 4}}, {"num_outputs": 7}).num_outputs, 7)

    def test_multilabel_sigmoid_does_not_normalize_across_tasks(self):
        spec = TaskSpec(task_type="multilabel", metric="rocauc", num_outputs=3)
        logits = torch.tensor([[2., 2., 2.], [-2., 0., 2.]])
        p = probabilities(logits, spec)
        self.assertGreater(float(p[0].sum()), 2.)
        torch.testing.assert_close(p, logits.sigmoid())
        np.testing.assert_allclose(probabilities(logits.numpy(), spec), p.numpy(), rtol=1e-6)

    def test_tensor_and_numeric_losses_agree_and_missing_gradients_are_zero(self):
        for spec, labels in [
            (TaskSpec(num_outputs=3), np.array([0., np.nan, 2.])),
            (TaskSpec(task_type="multilabel", metric="rocauc", num_outputs=3),
             np.array([[1., np.nan, 0.], [np.nan, -1., np.nan], [0., 1., 1.]])),
        ]:
            logits = torch.tensor([[.1, .8, -.2], [.3, -.4, 1.], [1., -1., .2]], requires_grad=True)
            rows = loss(logits, labels, spec, reduction="none")
            expected = sample_losses(probabilities(logits, spec).detach().numpy(), labels, spec)
            np.testing.assert_allclose(rows.detach().numpy(), expected, rtol=1e-6, equal_nan=True)
            value = loss(logits, labels, spec)
            self.assertAlmostEqual(float(value.detach()), float(np.nanmean(expected)), places=6)
            value.backward()
            self.assertTrue(torch.equal(logits.grad[1], torch.zeros(3)))
            if spec.task_type == "multilabel":
                self.assertEqual(float(logits.grad[0, 1]), 0.)
        with self.assertRaisesRegex(ValueError, "without observed"):
            loss(torch.zeros(2, 3), np.full((2, 3), np.nan), spec)

    def test_complete_multiclass_loss_preserves_original_ce(self):
        logits = torch.tensor([[1., -.2, .5], [.3, .1, .6]], requires_grad=True)
        labels = torch.tensor([0, 2])
        self.assertTrue(torch.equal(loss(logits, labels, TaskSpec(num_outputs=3)),
                                    torch.nn.functional.cross_entropy(logits, labels)))
        with self.assertRaises(ValueError):
            valid_target_mask(np.array([0., 1.5]), TaskSpec(num_outputs=3))
        with self.assertRaises(ValueError):
            valid_target_mask(np.array([[1., np.inf]]), TaskSpec(task_type="multilabel", metric="ap", num_outputs=2))
        with self.assertRaises(ValueError):
            loss(torch.empty(0, 3), torch.empty(0, dtype=torch.long), TaskSpec(num_outputs=3))

    def test_macro_f1_declared_universe_and_weights_match_reference(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        predicted = np.array([0, 1, 1, 2, 2, 2])
        p = np.eye(4)[predicted]
        weights = np.array([1., 2., 0., 1., 3., 1.])
        spec = TaskSpec(metric="macro_f1", num_outputs=4)
        actual = metric_score(p, y, spec, sample_weight=weights)
        expected = f1_score(y, predicted, labels=np.arange(4), average="macro", sample_weight=weights, zero_division=0)
        self.assertAlmostEqual(actual, expected, places=14)
        with self.assertRaisesRegex(ValueError, "no per-sample"):
            sample_metric_values(p, y, spec)

    def test_rank_metrics_ties_missing_tasks_and_bootstrap_multiplicities(self):
        y = np.array([[1., 0., 1.], [0., 1., 1.], [1., np.nan, 1.], [0., 1., 1.], [1., 0., np.nan]])
        p = np.array([[.8, .2, .5], [.5, .7, .3], [.5, .3, .2], [.3, .7, .5], [.2, .8, .1]])
        weights = np.array([1, 2, 0, 3, 1])
        for metric, reference in [("rocauc", roc_auc_score), ("ap", average_precision_score)]:
            spec = TaskSpec(task_type="multilabel", metric=metric, num_outputs=3)
            expected = []
            for column in range(2):
                keep = np.isfinite(y[:, column]) & (weights > 0)
                expected.append(reference(y[keep, column], p[keep, column], sample_weight=weights[keep]))
            actual = metric_score(p, y, spec, sample_weight=weights)
            self.assertAlmostEqual(actual, float(np.mean(expected)), places=14)
            expanded = np.repeat(np.arange(len(y)), weights)
            self.assertAlmostEqual(actual, metric_score(p[expanded], y[expanded], spec), places=14)
            self.assertTrue(np.isnan(metric_score(p[:, 2:], np.ones((5, 1)),
                                                 TaskSpec(task_type="multilabel", metric=metric, num_outputs=1))))

    def test_multilabel_confidence_is_bernoulli_not_categorical(self):
        spec = TaskSpec(task_type="multilabel", metric="ap", num_outputs=3)
        p = torch.tensor([[.5, .5, .5], [0., 1., 1.]])
        entropy, margin, js = confidence_statistics(p, p, spec)
        torch.testing.assert_close(entropy, torch.tensor([1., 0.]))
        torch.testing.assert_close(margin, torch.tensor([0., 1.]))
        torch.testing.assert_close(js, torch.zeros(2))
        _, _, opposite = confidence_statistics(torch.tensor([[0., 1., 0.]]),
                                                torch.tensor([[1., 0., 1.]]), spec)
        torch.testing.assert_close(opposite, torch.ones(1))

    def test_numeric_task_import_never_imports_torch(self):
        source = """
import builtins, json, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise RuntimeError('Numeric review may not import torch')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from cross_morse.tasks import task_spec, metric_score, sample_losses
spec = task_spec({'dataset':'synthetic', 'benchmark_dataset':'Peptides-func', 'data':{'synthetic_classes':2}})
p = [[.8,.2],[.1,.9]]
y = [[1,0],[0,1]]
print(json.dumps({'metric':metric_score(p,y,spec),'loss':sample_losses(p,y,spec).tolist(),'torch': 'torch' in sys.modules}))
"""
        result = subprocess.run([sys.executable, "-c", source], text=True, capture_output=True, check=True)
        record = json.loads(result.stdout)
        self.assertEqual(record["metric"], 1.)
        self.assertFalse(record["torch"])


if __name__ == "__main__":
    unittest.main()
