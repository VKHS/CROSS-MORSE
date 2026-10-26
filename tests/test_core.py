"""Mathematical invariants of the reconstructed frozen transport and sensors."""
import unittest
import torch

from cross_morse.core import (
    ACTIONS, FrozenPlant, GraphOperator, MeasurementBank, DesignatedSitePlant,
    action_operator_response, nominal_operator_response,
    apply_transport, chebyshev_window_coefficients, executed_hops,
    executed_identity_coefficient, identity_mass,
)


def graph():
    # Irregular undirected graph 0--1--2 and 1--3; node4 is isolated.
    edge_index = torch.tensor([[0, 1, 1, 2, 1, 3], [1, 0, 2, 1, 3, 1]])
    return GraphOperator(edge_index, 5, "cpu")


class RegistryTests(unittest.TestCase):
    def test_exact_probabilities_and_ranges(self):
        self.assertAlmostEqual(sum(a.probability for a in ACTIONS), 1.0)
        expected = [0.700, 0.550, 0.850, 0.697, 0.646, 0.964, 0.864, 0.603]
        for a, ref in zip(ACTIONS[:8], expected):
            self.assertAlmostEqual(executed_hops(a), ref, delta=0.0006)
        self.assertAlmostEqual(identity_mass(7), 0.4721613633, places=8)
        self.assertAlmostEqual(executed_identity_coefficient(7), 0.5690201704, places=8)
        self.assertEqual(executed_hops(8), 0.0)
        with self.assertRaises(ValueError):
            action_operator_response(8, torch.tensor([0.0]))

    def test_sparse_graph_mass_and_isolate(self):
        g = graph()
        p = g.P.to_dense()
        s = g.S.to_dense()
        torch.testing.assert_close(p.sum(1), torch.ones(5))
        torch.testing.assert_close(s, s.T)
        self.assertTrue(bool((p >= 0).all()))
        self.assertEqual(p[4, 4].item(), 1.0)
        torch.testing.assert_close(g.apply(torch.ones(5, 3)), torch.ones(5, 3))

    def test_transport_agrees_with_independent_dense_resolvent(self):
        g = graph()
        p = g.P.to_dense().double()
        eye = torch.eye(5, dtype=torch.float64)
        for a in ACTIONS[:8]:
            expected = p
            if a.eta is not None:
                kernel = sum(a.eta * (1-a.eta)**t * torch.linalg.matrix_power(p, t)
                             for t in range(a.order + 1))
                kernel += (1-a.eta)**(a.order+1) * eye
                expected = (1-a.rho)*p + a.rho*kernel
            expected = (1-a.nu)*eye + a.nu*expected
            g.reset_counters()
            actual = apply_transport(g, eye.float(), a).double()
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)
            torch.testing.assert_close(actual.sum(1), torch.ones(5, dtype=torch.float64), atol=2e-7, rtol=2e-7)
            self.assertTrue(bool((actual >= 0).all()))
            self.assertEqual(g.sparse_products, 8 if a.eta else 1)

    def test_polynomial_response_matches_eigenvectors(self):
        g = graph()
        vals, vecs = torch.linalg.eigh(torch.eye(5) - g.S.to_dense())
        # P is similar to S; D^-1/2 transforms symmetric eigenvectors.
        state = vecs / g.sqrt_degree[:, None]
        for action in ACTIONS[:8]:
            expected = state * action_operator_response(action, vals)[None, :]
            actual = apply_transport(g, state, action)
            torch.testing.assert_close(actual, expected, atol=5e-7, rtol=2e-6)

    def test_nominal_spectral_response_matches_each_site_operator(self):
        g = graph()
        vals, vecs = torch.linalg.eigh(torch.eye(5) - g.S.to_dense())
        state = vecs / g.sqrt_degree[:, None]
        actual = {"shared": .3 * state + .7 * g.apply(state),
                  "native_mean": g.apply(state), "inserted_identity": state}
        for baseline, transported in actual.items():
            expected = state * nominal_operator_response(vals, baseline)[None, :]
            torch.testing.assert_close(transported, expected, atol=5e-7, rtol=2e-6)
        with self.assertRaises(ValueError):
            nominal_operator_response(vals, "unknown")


class PlantTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.g = graph()
        self.model = FrozenPlant(4, 8, 3, dropout=0.3).freeze()
        self.x = torch.randn(5, 4)

    def test_nominal_identity_each_depth(self):
        h0 = self.model.encode(self.x)
        native = h0.clone()
        adapted = h0.clone()
        for _ in range(16):
            native = self.model.native_step(native, h0, self.g)
            adapted = self.model.step(adapted, h0, self.g, 0)
            self.assertTrue(torch.equal(native, adapted))
        self.assertTrue(torch.equal(self.model.logits(native), self.model.logits(adapted)))

    def test_nominal_identity_under_bf16_autocast(self):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            h0 = self.model.encode(self.x)
            native = h0.clone()
            adapted = h0.clone()
            for _ in range(3):
                native = self.model.native_step(native, h0, self.g)
                adapted = self.model.step(adapted, h0, self.g, 0)
                self.assertTrue(torch.equal(native, adapted))
            self.assertEqual(native.dtype, torch.bfloat16)

    def test_recovery_is_direct_assignment_without_dense_update(self):
        h0 = self.model.encode(self.x)
        h = self.model.native_step(h0, h0, self.g)
        last_safe = h.clone()
        h = self.model.step(h, h0, self.g, 5)
        self.model.reset_counters()
        self.g.reset_counters()
        source = self.model.step(h, h0, self.g, 8, last_safe)
        recovered = self.model.step(h, h0, self.g, 9, last_safe)
        self.assertTrue(torch.equal(source, h0))
        self.assertTrue(torch.equal(recovered, last_safe))
        self.assertNotEqual(source.data_ptr(), h0.data_ptr())
        self.assertNotEqual(recovered.data_ptr(), last_safe.data_ptr())
        self.assertEqual(self.g.sparse_products, 0)
        self.assertEqual(self.model.counters["dense_updates"], 0)
        self.assertEqual(self.model.counters["resets"], 2)

    def test_frozen_parameters_and_deterministic_replay(self):
        initial = {k: v.clone() for k, v in self.model.state_dict().items()}
        h0 = self.model.encode(self.x)
        sequence = [0, 3, 7, 9, 1, 8, 2]
        results = []
        for _ in range(2):
            h, safe = h0.clone(), h0.clone()
            for depth, action in enumerate(sequence):
                h = self.model.step(h, h0, self.g, action, safe)
                if depth == 1:
                    safe = h.clone()
            results.append(self.model.logits(h))
        self.assertTrue(torch.equal(*results))
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, initial[key]))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.model.parameters()))
        with self.assertRaises(RuntimeError):
            self.model.train()

    def test_training_gradients_flow_before_freeze(self):
        model = FrozenPlant(4, 8, 3, dropout=0.0)
        loss = model(self.x, self.g, depth=3).square().mean()
        loss.backward()
        self.assertIsNotNone(model.shared.weight.grad)
        self.assertTrue(torch.isfinite(model.shared.weight.grad).all())
        self.assertIsNotNone(model.encoder[0].weight.grad)


class SensorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.g = graph()
        self.h = torch.randn(5, 8)
        self.logits = torch.randn(5, 3)

    def test_windows_partition_and_shared_recurrence(self):
        coeff = chebyshev_window_coefficients()
        torch.testing.assert_close(coeff[:, 0].sum(), torch.tensor(1.0))
        torch.testing.assert_close(coeff[:, 1:].sum(0), torch.zeros(8), atol=1e-7, rtol=0)
        bank = MeasurementBank(self.g, 8)
        self.g.reset_counters()
        bank.prepare(self.h, self.logits)
        self.assertEqual(self.g.sparse_products, 8)
        self.g.reset_counters()
        values = bank.measure(self.h, self.logits, 0, 256)
        # One Dirichlet product plus two eight-product spectral streams.
        self.assertEqual(self.g.sparse_products, 17)
        self.assertEqual(values.shape, (25,))
        self.assertTrue(torch.isfinite(values).all())
        self.assertAlmostEqual(values[2].item(), 0.0, places=6)
        torch.testing.assert_close(values[12:15], torch.ones(3), atol=1e-5, rtol=1e-5)

    def test_zero_states_and_lean_schema_are_finite(self):
        for profile in ("full", "lean"):
            bank = MeasurementBank(self.g, 8, profile=profile)
            bank.prepare(torch.zeros_like(self.h), self.logits)
            values = bank.measure(torch.zeros_like(self.h), self.logits, 4, 8)
            self.assertTrue(torch.isfinite(values).all())
            self.assertEqual(values[5].item(), 0.5)
            if profile == "lean":
                self.assertTrue(torch.equal(values[9:], torch.zeros(16)))

    def test_reachability_uses_next_sites_actual_nominal_operator(self):
        # For A1, the gap is respectively .15*lambda, .45*lambda and
        # -.55*lambda. Its exact band maximum is attained at the right edge.
        model = DesignatedSitePlant(4, 8, 3, encoder_layers=1, dropout=0.,
                                    control_depth=8, native_layers=2,
                                    backbone="gps", attention_heads=2).freeze()
        bank = MeasurementBank(self.g, 8, plant=model)
        right_edges = torch.tensor([2 * (b + 1) / 6 for b in bank.bands])
        for baseline, slope in (("shared", .15), ("native_mean", .45),
                                ("inserted_identity", .55)):
            torch.testing.assert_close(bank.sensitivities_by_baseline[baseline][0],
                                       slope * right_edges, atol=1e-7, rtol=1e-6)
        self.assertEqual(bank.nominal_baseline(0, 8), "inserted_identity")
        self.assertEqual(bank.nominal_baseline(1, 8), "native_mean")
        self.assertEqual(bank.nominal_baseline(2, 8), "inserted_identity")
        bank.prepare(self.h, self.logits)
        inserted = bank.measure(self.h, self.logits, 0, 8)
        native = bank.measure(self.h, self.logits, 1, 8)
        torch.testing.assert_close(inserted[:5], native[:5])
        torch.testing.assert_close(inserted[9:18], native[9:18])
        # Same state/powers, different correct A0 gap; this fails if the old
        # shared baseline is accidentally used for both controlled sites.
        torch.testing.assert_close(inserted[18] / native[18], torch.tensor(.55 / .45))
        terminal = bank.measure(self.h, self.logits, 8, 8)
        torch.testing.assert_close(terminal[18:], torch.zeros(7))
        self.assertIsNone(bank.nominal_baseline(8, 8))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            bank.measure(self.h, self.logits, 0, 7)
        self.assertEqual(bank.metadata()["reachability_baseline_by_site"][1], "native_mean")

    def test_mask_changes_pooling_without_changing_graph(self):
        mask = torch.tensor([True, False, True, False, False])
        bank = MeasurementBank(self.g, 8, mask=mask, sketch_dim=4, seed=3)
        bank.prepare(self.h, self.logits)
        values1 = bank.measure(self.h, self.logits, 0, 256)
        values2 = bank.measure(self.h, self.logits, 0, 256)
        self.assertTrue(torch.equal(values1, values2))
        self.assertEqual(self.g.num_nodes, 5)
        self.assertFalse(values1.requires_grad)


if __name__ == "__main__":
    unittest.main()
