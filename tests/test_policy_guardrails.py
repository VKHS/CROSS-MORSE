"""Executable policy, recovery, replay and pre-selection registry contracts.

These tests assert algorithmic invariants, not agreement with manuscript scores.
See main Eqs. (14)--(17), Algorithm 1, and supplement S3/S6/S9/S11.
"""
from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cross_morse.core import FrozenPlant, GraphOperator
from cross_morse.policies import (PlanningBudgetExceeded, PolicyExecutionError, beam_plan,
                                  execute_policy, make_registry)


class FixedMeasurementBank:
    """Deterministic observations make recovery-anchor semantics testable."""
    def __init__(self):
        self.counters = {"dense_multiply_adds": 0}

    def prepare(self, h, logits):
        pass

    def measure(self, h, logits, depth, length, mass=0., hops=0.):
        y = torch.zeros(25, device=h.device)
        y[0] = .2 if depth < 2 else .999  # Only depths zero/one are safe.
        y[5] = depth / length
        self.counters["dense_multiply_adds"] += 7
        return y


class InstrumentedController:
    def __init__(self, desired=None, support=None):
        self.desired = list(desired or [1, 2, 9, 0])
        self.support_function = support
        self.reset_counters()

    def reset_counters(self):
        self.counters = {"model_forward_calls": 0,
                         "observer_updates": 0,
                         "approx_dense_multiply_adds": 0}
        self.starts = self.observations = 0
        self.score_kinds = []

    def start(self, y):
        self.starts += 1
        return SimpleNamespace(y=np.array(y), depth=0, valid=True)

    def observe(self, state, y, depth, previous_action):
        self.observations += 1
        return SimpleNamespace(y=np.array(y), depth=depth, valid=True)

    def supported(self, y, depth):
        return list(range(10)) if self.support_function is None else self.support_function(y, depth)

    def score(self, state, actions, kind="radius", kappa=1., lambda_rho=0.):
        self.counters["model_forward_calls"] += 1
        self.score_kinds.append(kind)
        desired = self.desired[min(state.depth, len(self.desired)-1)]
        return {a: {"score": 0. if a == desired else 1. + a / 100.,
                    "sigma": 0.} for a in actions}

    def predict(self, state, action):
        self.counters["model_forward_calls"] += 1
        return SimpleNamespace(y=state.y.copy(), depth=state.depth+1, valid=True)

    def stage_cost(self, state, action):
        self.counters["model_forward_calls"] += 1
        return 0.

    def terminal_value(self, state):
        if state.depth < 4:
            self.counters["model_forward_calls"] += 1
        return 0.


class ResetQueryController(InstrumentedController):
    """Catch the original bug at model queries, not only at executed action IDs."""
    def __init__(self, *args, forbid_a9=False, nonfinite_scores=False, **kwargs):
        self.forbid_a9 = forbid_a9
        self.nonfinite_scores = nonfinite_scores
        super().__init__(*args, **kwargs)

    def reset_counters(self):
        super().reset_counters()
        self.queries = []

    def record(self, kind, actions):
        self.queries.append((kind, list(actions)))
        if self.forbid_a9 and 9 in actions:
            raise AssertionError(f"Open-loop {kind} queried dynamic-anchor A9")

    def score(self, state, actions, **kwargs):
        self.record("critic", actions)
        result = super().score(state, actions, **kwargs)
        if self.nonfinite_scores:
            for row in result.values():
                row["score"] = np.inf
        return result

    def predict(self, state, action):
        self.record("transition", [action])
        return super().predict(state, action)

    def stage_cost(self, state, action):
        self.record("stage", [action])
        return super().stage_cost(state, action)


class PolicyGuardrailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(39)
        self.x = torch.randn(12, 4)
        edges = torch.stack((torch.arange(12), (torch.arange(12)+1) % 12))
        self.graph = GraphOperator(torch.cat((edges, edges.flip(0)), dim=1), 12)
        self.model = FrozenPlant(4, 8, 3, dropout=0).freeze()
        self.config = {
            "seed": 7,
            "model": {"depth": 4, "precision": "fp32"},
            "sensor": {"bands": [0, 2, 5], "degree": 8, "sketch_dim": 4,
                       "profiles": ["full", "lean"]},
            "controller": {"max_search_calls": 100, "gamma": .97,
                           "lambda_rho": 1e-4, "lambda_transport": 1e-5,
                           "lambda_terminal": 1., "top_k": 2, "beam_width": 2,
                           "horizons": [2, 4], "kappa_grid": [1.]},
            "registry": {"phase_boundaries": [.5], "phase_pairs": [[1, 0]],
                         "max_exploration_sequences": 2, "ensemble_sizes": [1, 2]},
            "training": {"seeds": [11, 29]},
        }

    def candidate(self, route="one_step", **extra):
        return dict(route=route, profile="full", depth=4, horizon=2,
                    behavior_id="12345678", **extra)

    def execute(self, candidate=None, controller=None, **extra):
        with patch("cross_morse.policies.sensor_bank", side_effect=lambda *a, **k: FixedMeasurementBank()):
            return execute_policy(self.model, self.x, self.graph, self.config,
                                  self.candidate() if candidate is None else candidate,
                                  controller=controller, **extra)

    def test_execution_accepts_no_labels_or_dataset_handle(self):
        names = set(inspect.signature(execute_policy).parameters)
        self.assertTrue({"model", "x", "graph", "candidate"} <= names)
        self.assertFalse(names & {"data", "dataset", "labels", "y", "targets", "ids"})
        self.assertFalse(any(p.kind == inspect.Parameter.VAR_KEYWORD
                             for p in inspect.signature(execute_policy).parameters.values()))

    def test_feedback_replay_preserves_action_nine_anchor_without_observer(self):
        controller = InstrumentedController()
        probabilities, trace = self.execute(controller=controller)
        self.assertEqual(trace["actions"], [1, 2, 9, 0])
        self.assertEqual(trace["safe_depths"], [0, 1, 1, 1])
        self.assertEqual(controller.score_kinds, ["one_step"]*4)
        replay, replay_trace = self.execute(controller=controller,
                                          replay_actions=trace["actions"],
                                          replay_safe_depths=trace["safe_depths"])
        np.testing.assert_array_equal(probabilities, replay)
        self.assertEqual(controller.starts, 0)
        self.assertEqual(controller.observations, 0)
        self.assertEqual(replay_trace["sensor_operations"], {})
        self.assertEqual(replay_trace["safe_depths"], trace["safe_depths"])
        self.assertLess(replay_trace["multiply_add_work"], trace["multiply_add_work"])
        with self.assertRaisesRegex(ValueError, "safe_depths"):
            self.execute(controller=controller, replay_actions=trace["actions"])

    def test_fixed_open_loop_replays_deterministically_and_never_assimilates(self):
        candidate = self.candidate("fixed_sequence", sequence=[1, 3, 8, 2])
        controller = InstrumentedController()
        probabilities, trace = self.execute(candidate, controller)
        again, again_trace = self.execute(candidate, controller)
        replay, replay_trace = self.execute(candidate, controller,
                                          replay_actions=trace["actions"],
                                          replay_safe_depths=trace["safe_depths"])
        np.testing.assert_array_equal(probabilities, again)
        np.testing.assert_array_equal(probabilities, replay)
        self.assertEqual(trace["actions"], again_trace["actions"])
        self.assertEqual(controller.observations, 0)
        self.assertEqual(replay_trace["terminal_depth"], 4)
        self.assertEqual(trace["multiply_add_work"], trace["deployed_multiply_add_work"]
                         + trace["qualification_multiply_add_work"])
        self.assertEqual(trace["deployed_multiply_add_work"], replay_trace["multiply_add_work"])
        self.assertGreater(trace["qualification_sensor_dense_macs"], 0)

    def test_open_loop_source_reset_cannot_depend_on_intermediate_measurements(self):
        class AlternateMeasurements(FixedMeasurementBank):
            def measure(self, *args, **kwargs):
                y = super().measure(*args, **kwargs)
                y[0] = .1  # Every intermediate state is nominally safe.
                return y
        controller = InstrumentedController()
        for route in ("fixed_sequence", "phase", "mpc_ol"):
            candidate = self.candidate(route, sequence=[1, 2, 8, 0])
            with patch("cross_morse.policies.beam_plan", return_value=[1, 2, 8, 0]):
                expected, trace = self.execute(candidate, controller)
                with patch("cross_morse.policies.sensor_bank", side_effect=lambda *a, **k: AlternateMeasurements()):
                    actual, altered_trace = execute_policy(self.model, self.x, self.graph, self.config,
                                                           candidate, controller)
            np.testing.assert_array_equal(expected, actual)
            self.assertEqual(trace["actions"], [1, 2, 8, 0])
            self.assertEqual(trace["safe_depths"], [0]*4)
            self.assertEqual(altered_trace["safe_depths"], [0]*4)
            self.assertEqual(controller.observations, 0)

    def test_open_loop_refuses_uncanonicalized_a9_but_explicit_replay_remains_legal(self):
        controller = InstrumentedController()
        for route in ("fixed_sequence", "phase", "mpc_ol"):
            with self.subTest(route=route):
                with self.assertRaisesRegex(ValueError, "source-reset A8"):
                    self.execute(self.candidate(route, sequence=[1, 2, 9, 0]), controller)
        candidate = self.candidate("mpc_ol")
        with patch("cross_morse.policies.beam_plan", return_value=[9]):
            with self.assertRaisesRegex(ValueError, "plan contains dynamic-anchor A9"):
                self.execute(candidate, controller)
        # Replay is an explicit historical execution, not new open-loop search.
        feedback, trace = self.execute(controller=controller)
        replay, _ = self.execute(self.candidate("fixed_sequence"), controller,
                                 replay_actions=trace["actions"],
                                 replay_safe_depths=trace["safe_depths"])
        np.testing.assert_array_equal(feedback, replay)

    def test_open_loop_beam_never_queries_a9_even_for_simulated_recovery(self):
        self.config["controller"].update(top_k=1, beam_width=1)
        cases = [([0, 1, 8, 9], False, 0), ([9], False, 8),
                 ([], False, 8), ([1, 9], True, 8)]
        for support, nonfinite, expected in cases:
            with self.subTest(support=support, nonfinite_scores=nonfinite):
                controller = ResetQueryController(desired=[9]*4, forbid_a9=True,
                            support=lambda y, d: support, nonfinite_scores=nonfinite)
                state = controller.start(np.zeros(25))
                plan = beam_plan(controller, state, 3, self.config, {"calls": 0}, open_loop=True)
                self.assertEqual(plan, [expected]*3)
                self.assertTrue(controller.queries)
                self.assertTrue(all(9 not in actions for _, actions in controller.queries))
                self.assertIn(("transition", [expected]), controller.queries)
                self.assertIn(("stage", [expected]), controller.queries)

    def test_mpc_open_loop_execution_uses_restricted_search_and_source_reset(self):
        self.config["controller"].update(top_k=1, beam_width=1)
        controller = ResetQueryController(desired=[9]*4, forbid_a9=True,
                                          support=lambda y, d: [9])
        probabilities, trace = self.execute(self.candidate("mpc_ol"), controller)
        self.assertEqual(trace["actions"], [8, 8, 0, 0])
        self.assertEqual(trace["safe_depths"], [0]*4)
        self.assertTrue(all(9 not in actions for _, actions in controller.queries))
        replay, _ = self.execute(self.candidate("mpc_ol"), controller,
                                 replay_actions=trace["actions"], replay_safe_depths=trace["safe_depths"])
        np.testing.assert_array_equal(probabilities, replay)

    def test_mpc_feedback_retains_dynamic_a9_and_exact_anchor_replay(self):
        self.config["controller"].update(top_k=1, beam_width=1)
        controller = ResetQueryController(desired=[1, 2, 9, 0])
        probabilities, trace = self.execute(self.candidate("mpc_rh"), controller)
        self.assertEqual(trace["actions"], [1, 2, 9, 0])
        self.assertEqual(trace["safe_depths"], [0, 1, 1, 1])
        self.assertIn(("transition", [9]), controller.queries)
        self.assertIn(("stage", [9]), controller.queries)
        replay, _ = self.execute(self.candidate("mpc_rh"), controller,
                                 replay_actions=trace["actions"], replay_safe_depths=trace["safe_depths"])
        np.testing.assert_array_equal(probabilities, replay)

    def test_nonfinite_output_has_typed_json_safe_failure_trace(self):
        with patch.object(self.model, "logits", return_value=torch.full((12, 3), float("nan"))):
            with self.assertRaises(PolicyExecutionError) as caught:
                self.execute(self.candidate("nominal"))
        error = caught.exception
        self.assertIsInstance(error, FloatingPointError)
        self.assertEqual(error.reason, "nonfinite_predictions")
        self.assertIs(error.diagnostics, error.trace)
        self.assertEqual(error.trace["actions"], [0]*4)
        self.assertEqual(error.trace["nonfinite_probability_count"], 36)
        self.assertEqual(error.trace["failure_stage"], "final_prediction")
        self.assertFalse(error.trace["feasible"])
        self.assertGreater(error.trace["multiply_add_work"], 0)
        json.dumps(error.trace, allow_nan=False)

    def test_mid_execution_numerical_failure_records_partial_trace(self):
        native = self.model.step
        calls = []
        def fail_second(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise FloatingPointError("injected arithmetic failure")
            return native(*args, **kwargs)
        with patch.object(self.model, "step", side_effect=fail_second):
            with self.assertRaises(PolicyExecutionError) as caught:
                self.execute(self.candidate("nominal"))
        error = caught.exception
        self.assertEqual(error.reason, "numerical_execution_failure")
        self.assertEqual(error.trace["actions"], [0])
        self.assertEqual(error.trace["safe_depths"], [0])
        self.assertEqual(error.trace["failed_depth"], 1)
        self.assertEqual(error.trace["failed_action"], 0)
        self.assertEqual(error.trace["failure_stage"], "plant_step")
        self.assertEqual(error.trace["exception_type"], "FloatingPointError")
        json.dumps(error.trace, allow_nan=False)

    def test_nonfinite_planning_cost_is_a_typed_failure_not_a_valid_plan(self):
        controller = InstrumentedController()
        with patch.object(controller, "stage_cost", return_value=float("nan")):
            with self.assertRaises(PolicyExecutionError) as caught:
                self.execute(self.candidate("mpc_ol"), controller)
        self.assertEqual(caught.exception.trace["failure_stage"], "initial_plan")
        self.assertEqual(caught.exception.trace["actions"], [])
        self.assertIn("Nonfinite stage cost", caught.exception.trace["exception_message"])

    def test_unrelated_runtime_errors_are_not_reclassified_as_numerical_failure(self):
        with patch.object(self.model, "step", side_effect=RuntimeError("unexpected interface error")):
            with self.assertRaisesRegex(RuntimeError, "unexpected interface") as caught:
                self.execute(self.candidate("nominal"))
        self.assertNotIsInstance(caught.exception, PolicyExecutionError)

    def test_zero_planning_budget_uses_only_legal_recovery_and_charges_no_calls(self):
        self.config["controller"]["max_search_calls"] = 0
        for support, expected in (([0, 1], 0), ([1, 2], 9)):
            controller = InstrumentedController(support=lambda y, d: support)
            _, trace = self.execute(controller=controller)
            self.assertEqual(trace["actions"], [expected]*4)
            self.assertEqual(trace["origins"], ["recovery_budget"]*4)
            self.assertTrue(trace["planning_budget_exhausted"])
            self.assertEqual(trace["model_calls_charged"], 0)
            self.assertEqual(controller.counters["model_forward_calls"], 0)
            self.assertTrue(trace["feasible"])

    def test_mpc_open_loop_plans_once_and_feedback_replans_each_depth(self):
        controller = InstrumentedController()
        with patch("cross_morse.policies.beam_plan", return_value=[1, 2]) as plan:
            _, trace = self.execute(self.candidate("mpc_ol"), controller)
        self.assertEqual(plan.call_count, 1)
        self.assertEqual(trace["actions"], [1, 2, 0, 0])
        self.assertEqual(controller.observations, 0)
        with patch("cross_morse.policies.beam_plan", return_value=[1, 2]) as plan:
            _, trace = self.execute(self.candidate("mpc_rh"), controller)
        self.assertEqual(plan.call_count, 4)
        self.assertEqual([call.args[2] for call in plan.call_args_list], [2, 2, 2, 1])
        self.assertEqual(trace["actions"], [1]*4)
        self.assertEqual(controller.observations, 3)

    def test_mpc_open_loop_actual_support_audit_cannot_change_frozen_plan(self):
        controller = InstrumentedController(support=lambda y, d: [0, 1] if d == 0 else [0])
        with patch("cross_morse.policies.beam_plan", return_value=[1, 1]) as plan:
            _, trace = self.execute(self.candidate("mpc_ol"), controller)
        self.assertEqual(plan.call_count, 1)
        self.assertEqual(trace["actions"], [1, 1, 0, 0])
        self.assertEqual(trace["support_violations"], 1)
        self.assertFalse(trace["feasible"])

    def test_real_sensor_qualification_ledger_balances_and_retains_initial_mpc_sensing(self):
        controller = InstrumentedController()
        candidate = self.candidate("fixed_sequence", sequence=[1, 3, 8, 2])
        _, trace = execute_policy(self.model, self.x, self.graph, self.config, candidate, controller)
        _, replay = execute_policy(self.model, self.x, self.graph, self.config, candidate, controller,
                                   replay_actions=trace["actions"], replay_safe_depths=trace["safe_depths"])
        self.assertGreater(trace["qualification_sensor_sparse_macs"], 0)
        self.assertGreater(trace["qualification_sensor_dense_macs"], 0)
        self.assertEqual(trace["qualification_multiply_add_work"],
                         trace["qualification_sensor_sparse_macs"] + trace["qualification_sensor_dense_macs"])
        self.assertEqual(trace["deployed_multiply_add_work"], replay["multiply_add_work"])
        self.assertEqual(trace["multiply_add_work"], trace["deployed_multiply_add_work"]
                         + trace["qualification_multiply_add_work"])
        with patch("cross_morse.policies.beam_plan", return_value=[1, 2]):
            _, mpc = execute_policy(self.model, self.x, self.graph, self.config, self.candidate("mpc_ol"), controller)
        _, mpc_replay = execute_policy(self.model, self.x, self.graph, self.config, self.candidate("mpc_ol"), controller,
                                      replay_actions=mpc["actions"], replay_safe_depths=mpc["safe_depths"])
        self.assertEqual(mpc["multiply_add_work"], mpc["deployed_multiply_add_work"]
                         + mpc["qualification_multiply_add_work"])
        self.assertGreater(mpc["deployed_multiply_add_work"], mpc_replay["multiply_add_work"])
        # All feedback sensing remains deployment work; none is subtracted.
        _, feedback = execute_policy(self.model, self.x, self.graph, self.config, self.candidate(), controller)
        self.assertEqual(feedback["qualification_multiply_add_work"], 0)
        self.assertEqual(feedback["deployed_multiply_add_work"], feedback["multiply_add_work"])

    def test_shuffled_support_mismatch_is_recorded_before_safe_intersection(self):
        donor = np.zeros(25, dtype=np.float32)
        donor[0] = .8
        donors = SimpleNamespace(draw=lambda *args: donor)
        controller = InstrumentedController(
            desired=[1, 2, 2, 2],
            support=lambda y, d: [0, 2] if float(y[0]) == float(donor[0]) else [0, 1])
        _, trace = self.execute(self.candidate(state_control="matched_shuffle"),
                                controller, donors=donors)
        self.assertEqual(trace["actions"], [1, 0, 0, 0])
        self.assertEqual(trace["shuffle_support_mismatch_sites"], 3)
        self.assertGreaterEqual(trace["support_violations"], 3)
        self.assertFalse(trace["state_control_valid"])
        self.assertFalse(trace["feasible"])

    def test_missing_donors_never_count_as_valid_state_destruction(self):
        controller = InstrumentedController()
        _, trace = self.execute(self.candidate(state_control="matched_shuffle"), controller)
        self.assertEqual(trace["shuffle_missing_sites"], 3)
        self.assertEqual(trace["shuffle_matched_sites"], 0)
        self.assertFalse(trace["state_control_valid"])

    def test_beam_model_budget_matches_executed_calls_and_terminal_boundary(self):
        self.config["controller"]["top_k"] = 1
        self.config["controller"]["beam_width"] = 1
        controller = InstrumentedController(support=lambda y, d: [0])
        state = controller.start(np.zeros(25))
        budget = {"calls": 0}
        actions = beam_plan(controller, state, 4, self.config, budget)
        self.assertEqual(actions, [0]*4)
        self.assertEqual(budget["calls"], 15)  # 4 score + 4 transition + 4 stage + 3 tail.
        self.assertEqual(controller.counters["model_forward_calls"], budget["calls"])
        controller.reset_counters()
        self.config["controller"]["max_search_calls"] = 2
        budget = {"calls": 0}
        with self.assertRaises(PlanningBudgetExceeded):
            beam_plan(controller, state, 4, self.config, budget)
        self.assertEqual(budget["calls"], 1)
        self.assertEqual(controller.counters["model_forward_calls"], 1)

    def test_registry_is_fixed_from_exploration_and_configuration_without_selection(self):
        config_before = copy.deepcopy(self.config)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            np.savez(path/"trajectory_00000.npz", final_loss=.8, sequence=[0, 1, 8, 0])
            np.savez(path/"trajectory_00001.npz", final_loss=.7, sequence=[1, 1, 0, 0])
            first = make_registry(self.config, path)
            second = make_registry(self.config, path)
        self.assertEqual(first, second)
        self.assertEqual(self.config, config_before)
        self.assertEqual({c["route"] for c in first}, {"nominal", "best_depth", "learned_stop", "phase",
                         "fixed_sequence", "mpc_ol", "one_step", "radius", "mpc_rh"})
        self.assertEqual(len({c["candidate_id"] for c in first}), len(first))
        # Ensemble variants share a frozen behavior but have distinct candidate hashes.
        paired = [c for c in first if c["behavior_id"] == first[0]["behavior_id"]]
        self.assertEqual({c["ensemble"] for c in paired}, {1, 2})
        self.assertEqual(len({c["hash"] for c in paired}), 2)
        self.assertEqual(list(inspect.signature(make_registry).parameters),
                         ["config", "exploration_directory"])

    def test_registry_normalizes_open_loop_resets_and_preserves_original_provenance(self):
        self.config["registry"]["phase_pairs"] = [[9, 0]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            np.savez(path/"trajectory_00000.npz", final_loss=.5, sequence=[1, 9, 0, 0])
            np.savez(path/"trajectory_00001.npz", final_loss=.6, sequence=[1, 8, 0, 0])
            registry = make_registry(self.config, path)
        fixed = [c for c in registry if c["route"] == "fixed_sequence"]
        self.assertEqual(len(fixed), 2)  # One normalized behavior, two ensembles.
        for c in fixed:
            self.assertEqual(c["sequence"], [1, 8, 0, 0])
            self.assertEqual(c["reset_normalization"]["original_sequence"], [1, 9, 0, 0])
            self.assertEqual(c["reset_normalization"]["a9_to_a8_depths"], [1])
            self.assertEqual(c["exploration_source"], "trajectory_00000.npz")
            self.assertEqual(c["source_trajectory_loss"], .5)
            self.assertIn("not an evaluation", c["proposal_ranking"])
        for c in registry:
            if c["is_open_loop"] and "sequence" in c:
                self.assertNotIn(9, c["sequence"])
            if c["route"] == "phase":
                self.assertEqual(c["sequence"], [8, 8, 0, 0])
                self.assertEqual(c["reset_normalization"]["a9_to_a8_depths"], [0, 1])
            if c["route"] == "mpc_ol":
                self.assertNotIn(9, c["planning_actions"])
                self.assertEqual(c["recovery_reset"], 8)


if __name__ == "__main__":
    unittest.main()
