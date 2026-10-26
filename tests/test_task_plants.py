"""Weighted transport, independent graph execution, and designated-site laws."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cross_morse.core import (ACTIONS, DesignatedSitePlant, EdgeWeightedPlant,
                               FrozenPlant, GraphOperator, HeterophilyPlant,
                               MeasurementBank, apply_transport, build_plant,
                               site_transport_accounting, site_action_summary)
from cross_morse.exploration import collect_trajectory, sensor_bank
from cross_morse.learning import Controller, ControllerState
from cross_morse.policies import beam_plan, execute_policy
from cross_morse.tasks import TaskSpec, task_spec
from cross_morse.training import native_forward


def independent_graphs():
    edges = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    return GraphOperator(edges, 5, graph_ptr=torch.tensor([0, 3, 5])), edges


def small_config(backbone, depth=8, layers=2):
    return {"backbone": backbone, "depth": depth, "native_layers": layers,
            "hidden_dim": 8, "encoder_layers": 1, "attention_heads": 2,
            "dropout": 0., "activation": "relu"}


class TaskPlantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(123)

    def test_weighted_random_walk_and_symmetric_similarity(self):
        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        weights = torch.tensor([1., 1., 3., 3.])
        graph = GraphOperator(edges, 4, edge_weight=weights)
        expected = torch.tensor([[0., 1., 0., 0.], [.25, 0., .75, 0.],
                                 [0., 1., 0., 0.], [0., 0., 0., 1.]])
        torch.testing.assert_close(graph.P.to_dense(), expected)
        torch.testing.assert_close(graph.S.to_dense(), graph.S.to_dense().T)
        state = torch.randn(4, 3)
        symmetric = graph.sqrt_degree[:, None] * graph.apply(state / graph.sqrt_degree[:, None])
        torch.testing.assert_close(symmetric, graph.symmetric(state))
        torch.testing.assert_close(graph.apply(torch.ones(4, 2)), torch.ones(4, 2))

    def test_duplicate_weights_average_and_zero_weight_nodes_are_safe(self):
        edges = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 2]])
        graph = GraphOperator(edges, 3, edge_weight=torch.tensor([1., 3., 2., 0.]))
        self.assertEqual(float(graph.degree[0]), 2.)
        self.assertEqual(float(graph.P.to_dense()[2, 2]), 1.)
        self.assertTrue(torch.isfinite(graph.S.values()).all())
        with self.assertRaises(ValueError):
            GraphOperator(edges, 3, edge_weight=torch.tensor([1., -3., 2., 0.]))

    def test_cross_graph_edges_are_rejected_before_propagation(self):
        with self.assertRaisesRegex(ValueError, "crosses"):
            GraphOperator(torch.tensor([[1, 2], [2, 1]]), 4, graph_ptr=torch.tensor([0, 2, 4]))
        with self.assertRaisesRegex(ValueError, "partition"):
            GraphOperator(torch.empty((2, 0), dtype=torch.long), 4, graph_ptr=torch.tensor([0, 2, 2, 4]))

    def test_protein_factory_requires_weighted_kernel_and_uses_sigmoid_task(self):
        spec = task_spec("ogbn-proteins", {"num_outputs": 3})
        model = build_plant(4, 3, small_config("edge_weighted_transport"), spec)
        self.assertIsInstance(model, EdgeWeightedPlant)
        self.assertEqual(model.task.task_type, "multilabel")
        graph, _ = independent_graphs()
        h = torch.randn(5, 8)
        with self.assertRaisesRegex(ValueError, "edge weights"):
            model.native_step(h, h, graph)
        weighted = GraphOperator(graph.edge_index, 5, edge_weight=torch.ones(graph.num_edges))
        torch.testing.assert_close(model.native_step(h, h, weighted), model.step(h, h, weighted, 0))

    def test_roman_has_separate_trainable_ego_and_neighbour_maps(self):
        model = build_plant(4, 3, small_config("heterophily"),
                            task_spec("Roman-empire", {"num_outputs": 3}))
        self.assertIsInstance(model, HeterophilyPlant)
        self.assertNotEqual(model.ego.weight.data_ptr(), model.shared.weight.data_ptr())
        graph, _ = independent_graphs()
        h, h0 = torch.randn(5, 8), torch.randn(5, 8)
        expected, adapted = model.native_step(h, h0, graph), model.step(h, h0, graph, 0)
        self.assertTrue(torch.equal(expected, adapted))
        expected[:, 0].sum().backward()
        self.assertIsNotNone(model.ego.weight.grad)
        self.assertIsNotNone(model.shared.weight.grad)
        self.assertGreater(float(model.ego.weight.grad.norm()), 0.)
        self.assertGreater(float(model.shared.weight.grad.norm()), 0.)

    def test_runtime_sensor_factory_preserves_designated_nominal_baselines(self):
        graph = GraphOperator(torch.tensor([[0, 1], [1, 0]]), 3,
                              graph_ptr=torch.tensor([0, 3]))
        config = {"dataset": "synthetic", "benchmark_dataset": "PascalVOC-SP",
                  "seed": 2, "data": {"synthetic_classes": 3},
                  "sensor": {"bands": [0, 2, 5], "degree": 8, "sketch_dim": 0}}
        for backbone, depth, layers in [("gps", 8, 2), ("linear_gine", 8, 2)]:
            model = build_plant(4, 3, small_config(backbone, depth, layers),
                                TaskSpec(num_outputs=3)).freeze()
            bank = sensor_bank(graph, model, config)
            for site in range(depth):
                expected = "native_mean" if model.is_native_site(site) else "inserted_identity"
                self.assertEqual(bank.nominal_baseline(site, depth), expected)

    def test_full_control_site_counts_and_literal_inserted_identity(self):
        graph, _ = independent_graphs()
        for backbone, depth, layers, native_count in [("gps", 128, 6, 6), ("linear_gine", 64, 8, 8)]:
            model = build_plant(4, 3, small_config(backbone, depth, layers), TaskSpec(num_outputs=3)).freeze()
            self.assertEqual(sum(model.is_native_site(i) for i in range(depth)), native_count)
            self.assertEqual(model.instrumentation_metadata()["inserted_site_count"], depth - native_count)
            h, h0 = torch.randn(5, 8), torch.randn(5, 8)
            for site in range(depth):
                if not model.is_native_site(site):
                    graph.reset_counters()
                    model.reset_counters()
                    self.assertIs(model.step(h, h0, graph, 0, site=site), h)
                    self.assertIs(model.native_step(h, h0, graph, site=site), h)
                    self.assertEqual(graph.sparse_products, 0)
                    self.assertEqual(model.counters["dense_updates"], 0)
            with self.assertRaisesRegex(ValueError, "site index"):
                model.native_step(h, h0, graph)

    def test_inserted_transport_formula_and_hybrid_resets(self):
        graph, _ = independent_graphs()
        model = build_plant(4, 3, small_config("gps"), TaskSpec(num_outputs=3))
        h, h0, safe = torch.randn(5, 8), torch.randn(5, 8), torch.randn(5, 8)
        p, eye = graph.P.to_dense(), torch.eye(graph.num_nodes)
        for action in ACTIONS[1:8]:
            kernel = p
            if action.eta is not None:
                resolvent = sum(action.eta * (1-action.eta)**power * torch.linalg.matrix_power(p, power)
                                for power in range(action.order + 1))
                resolvent += (1-action.eta)**(action.order+1) * eye
                kernel = (1-action.rho)*p + action.rho*resolvent
            expected = (1-action.nu)*h + action.nu*((1-action.alpha)*(kernel @ h) + action.alpha*h0)
            actual = model.step(h, h0, graph, action, site=0)
            torch.testing.assert_close(actual, expected)
            double_mass = (1-action.nu)*h + action.nu*((1-action.alpha)*apply_transport(graph, h, action) + action.alpha*h0)
            self.assertGreater(float((actual-double_mass).norm()), 1e-3,
                               "Inserted transport must not apply nu twice")
        torch.testing.assert_close(model.step(h, h0, graph, 8, safe, site=0), h0)
        torch.testing.assert_close(model.step(h, h0, graph, 9, safe, site=0), safe)

    def test_nominal_designated_trajectory_matches_native_blocks(self):
        graph, _ = independent_graphs()
        for backbone in ("gps", "linear_gine"):
            model = build_plant(4, 3, small_config(backbone), TaskSpec(num_outputs=3)).freeze()
            h0 = model.encode(torch.randn(5, 4))
            native, adapted = h0, h0
            for site in range(model.control_depth):
                native = model.native_step(native, h0, graph, site=site)
                adapted = model.step(adapted, h0, graph, 0, site=site)
                self.assertTrue(torch.equal(native, adapted), (backbone, site))
            # Native execution applies only trained blocks, independently of the adapter.
            direct = h0
            for layer in range(model.native_layers):
                direct = model._native_block(layer, direct, graph.apply(direct), graph)
            self.assertTrue(torch.equal(native, direct))

    def test_gps_local_action_does_not_replace_attention_and_graphs_cannot_mix(self):
        graph, edges = independent_graphs()
        model = build_plant(4, 3, small_config("gps"), TaskSpec(num_outputs=3)).freeze()
        h, h0 = torch.randn(5, 8), torch.randn(5, 8)
        calls = []
        hook = model.blocks[0].attention.register_forward_hook(lambda module, args, output: calls.append(output[0].clone()))
        model.step(h, h0, graph, 0, site=1)
        nominal = calls[:]
        calls.clear()
        model.step(h, h0, graph, 5, site=1)
        changed = calls[:]
        hook.remove()
        self.assertEqual(len(nominal), 2)
        for before, after in zip(nominal, changed):
            self.assertTrue(torch.equal(before, after))
        x = torch.randn(5, 4)
        first = native_forward(model, x, graph, 8)
        altered = x.clone()
        altered[3:] = 1000 * torch.randn_like(altered[3:])
        second = native_forward(model, altered, graph, 8)
        self.assertTrue(torch.equal(first[:3], second[:3]))
        local = GraphOperator(edges[:, :4], 3, graph_ptr=torch.tensor([0, 3]))
        independent = native_forward(model, x[:3], local, 8)
        torch.testing.assert_close(first[:3], independent, atol=1e-6, rtol=1e-6)

    def test_linear_gine_aggregation_has_no_nonlinear_edge_message(self):
        graph, _ = independent_graphs()
        model = build_plant(4, 3, small_config("linear_gine"), TaskSpec(num_outputs=3))
        h = -torch.ones(5, 8)
        model._native_block = lambda layer, state, aggregation, current_graph: aggregation
        torch.testing.assert_close(model.native_step(h, h, graph, site=0), graph.apply(h))
        self.assertTrue((model.native_step(h, h, graph, site=0) < 0).all())

    def test_graph_readout_is_mean_within_each_graph_and_needs_binding(self):
        graph, _ = independent_graphs()
        spec = TaskSpec(task_type="multilabel", prediction_level="graph", metric="ap", num_outputs=3)
        model = FrozenPlant(4, 8, 3, task=spec)
        h = torch.randn(5, 8)
        with self.assertRaisesRegex(RuntimeError, "Bind"):
            model.logits(h)
        model.bind_graph(graph)
        actual = model.logits(h)
        expected = model.head(torch.stack((h[:3].mean(0), h[3:].mean(0))))
        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(actual.shape), (2, 3))
        altered = h.clone()
        altered[3:] *= 100
        self.assertTrue(torch.equal(actual[0], model.logits(altered)[0]))
        self.assertFalse(any("graph_ptr" in key for key in model.state_dict()))

    def test_independent_sensor_rejects_union_and_accepts_graph_head(self):
        graph, edges = independent_graphs()
        with self.assertRaisesRegex(ValueError, "one graph"):
            MeasurementBank(graph, 8)
        local = GraphOperator(edges[:, :4], 3, graph_ptr=torch.tensor([0, 3]))
        spec = TaskSpec(task_type="multilabel", prediction_level="graph", metric="ap", num_outputs=3)
        bank = MeasurementBank(local, 8, task=spec)
        h = torch.randn(3, 8)
        bank.prepare(h, torch.zeros(1, 3))
        values = bank.measure(h, torch.zeros(1, 3), 0, 8)
        self.assertEqual(tuple(values.shape), (25,))
        self.assertAlmostEqual(float(values[0]), 1., places=6)
        self.assertAlmostEqual(float(values[1]), 0., places=6)
        with self.assertRaisesRegex(ValueError, "node pooling"):
            bank.measure(h, torch.zeros(1, 3), 0, 8, mask=torch.tensor([True, False, True]))

    def test_site_aware_checkpoint_segments_preserve_gradients(self):
        graph, _ = independent_graphs()
        for backbone in ("gps", "linear_gine"):
            first = build_plant(4, 3, small_config(backbone), TaskSpec(num_outputs=3))
            second = copy.deepcopy(first)
            x = torch.randn(5, 4)
            a = native_forward(first, x, graph, 8, 0)
            b = native_forward(second, x, graph, 8, 3)
            torch.testing.assert_close(a, b)
            a.square().mean().backward()
            b.square().mean().backward()
            for (name, p), (_, q) in zip(first.named_parameters(), second.named_parameters()):
                self.assertIsNotNone(p.grad, name)
                self.assertIsNotNone(q.grad, name)
                torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=3e-5)

    def test_dense_chunking_preserves_predictions_and_gradients(self):
        graph, _ = independent_graphs()
        first = FrozenPlant(4, 8, 3, dropout=0.)
        second = copy.deepcopy(first)
        second.dense_chunk_size = 2
        x = torch.randn(5, 4)
        a, b = native_forward(first, x, graph, 3), native_forward(second, x, graph, 3)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
        a.sum().backward()
        b.sum().backward()
        for p, q in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=1e-4)

    def test_site_accounting_model_and_sealed_config_agree(self):
        for backbone, depth, layers in [("gps", 128, 6), ("linear_gine", 64, 8), ("shared_transport", 8, 2)]:
            cfg = {"model": small_config(backbone, depth, layers)}
            model = build_plant(4, 3, cfg["model"], TaskSpec(num_outputs=3))
            for site in range(depth):
                for action in range(10):
                    actual = site_transport_accounting(action, site, model)
                    self.assertEqual(actual, site_transport_accounting(action, site, cfg))
                    if action >= 8 or (not model.is_native_site(site) and action == 0):
                        self.assertEqual((actual["mass"], actual["hops"]), (0., 0.))
            native = next(site for site in range(depth) if model.is_native_site(site))
            self.assertEqual(site_transport_accounting(0, native, model)["mass"],
                             .7 if backbone == "shared_transport" else 1.)

    def test_observed_predicted_and_replayed_site_histories_match(self):
        cfg = {"seed": 7, "dataset": "synthetic", "data": {"synthetic_classes": 3},
               "model": {**small_config("gps"), "precision": "fp32"},
               "sensor": {"bands": [0, 2, 5], "degree": 8, "sketch_dim": 0},
               "exploration": {"terminal_samples_per_trajectory": 0, "critic_horizon": 1, "record_stride": 1},
               "controller": {"max_search_calls": 100}}
        model = build_plant(4, 3, cfg["model"], TaskSpec(num_outputs=3)).freeze()
        ids = torch.arange(5)
        edge = torch.stack((ids, (ids + 1) % 5))
        graph = GraphOperator(torch.cat((edge, edge.flip(0)), 1), 5, graph_ptr=torch.tensor([0, 5]))
        x = torch.randn(5, 4)
        sequence = [0, 0, 1, 0, 0, 5, 8, 0]
        rng = SimpleNamespace(choice=lambda choices, size, replace=True, p=None:
                              np.array(sequence, dtype=np.int64) if p is not None else np.empty(size, dtype=np.int64))
        with patch("cross_morse.exploration.np.random.default_rng", return_value=rng):
            records, info = collect_trajectory(model, x, graph, ids % 3, ids,
                                               np.zeros(5, dtype=int), cfg, 0, 0, "cpu")
        controller = Controller.__new__(Controller)
        controller.config = cfg
        controller.observer = SimpleNamespace(max_depth=8, dimension=2,
            reconstruct=lambda z: np.zeros(25), transition=lambda state, action: (state.z, np.eye(2)))
        controller.counters = {"observer_predictions": 0, "approx_dense_multiply_adds": 0}
        controller.transition_model = None
        controller.scalers = {"transition": None}
        controller.features = lambda state: np.zeros(2)
        controller._regression = lambda *args: np.zeros(2)
        cumulative = np.zeros(2)
        for site, action in enumerate(sequence):
            acc = site_transport_accounting(action, site, cfg)
            cumulative += [acc["mass"], acc["hops"]]
            np.testing.assert_allclose(records["y_next"][site, 6:8], cumulative / 8, atol=1e-7)
            state = ControllerState(np.zeros(2), np.eye(2), 0., 0., 0., records["y"][site].copy(),
                                    site, -1, True, np.zeros(2))
            predicted = controller.predict(state, action)
            np.testing.assert_allclose(predicted.y[5:8], records["y_next"][site, 5:8], atol=1e-7)
        _, trace = execute_policy(model, x, graph, cfg, {"route": "nominal", "profile": "none", "depth": 8},
                                   replay_actions=sequence)
        expected = site_action_summary(sequence, cfg)
        for key, value in expected.items():
            self.assertEqual(info[key], value)
            self.assertEqual(trace[key], value)
        self.assertEqual(trace["native_site_count"], 2)
        self.assertEqual(trace["inserted_site_count"], 6)
        self.assertEqual(trace["inserted_a0_bypass_count"], 3)
        self.assertEqual(trace["decisions"][0]["transport_mass_increment"], 0.)
        self.assertEqual(trace["decisions"][1]["transport_mass_increment"], 1.)

    def test_mpc_transport_penalty_observes_inserted_bypass(self):
        cfg = {"model": small_config("gps"), "controller": {
            "max_search_calls": 100, "lambda_rho": 0., "lambda_transport": 1.,
            "lambda_terminal": 0., "gamma": 1., "top_k": 2, "beam_width": 2}}
        controller = SimpleNamespace(
            supported=lambda y, depth: [0, 1],
            score=lambda state, actions, **kwargs: {a: {"score": 0., "sigma": 0.} for a in actions},
            stage_cost=lambda state, action: .1 if action == 1 else 0.,
            predict=lambda state, action: SimpleNamespace(y=state.y.copy(), depth=state.depth+1, valid=True),
            terminal_value=lambda state: 0.)
        inserted = SimpleNamespace(y=np.zeros(25), depth=0, valid=True)
        native = SimpleNamespace(y=np.zeros(25), depth=1, valid=True)
        self.assertEqual(beam_plan(controller, inserted, 1, cfg, {"calls": 0}), [0])
        self.assertEqual(beam_plan(controller, native, 1, cfg, {"calls": 0}), [1])


if __name__ == "__main__":
    unittest.main()
