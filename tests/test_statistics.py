"""Meaningful numerical/protocol checks; no downloaded data or GPU required."""
import unittest
import math

import numpy as np

from cross_morse.statistics import (
    cluster_summary, paired_bootstrap, nested_selection,
    confirmation, holm_adjust, admission, empirical_bernstein,
)


class StatisticsTests(unittest.TestCase):
    def test_core_means_and_node_weighted_risk(self):
        means, ids, counts = cluster_summary([1., 3., 10.], [4, 4, 9])
        np.testing.assert_array_equal(ids, [4, 9])
        np.testing.assert_array_equal(counts, [2, 1])
        np.testing.assert_allclose(means, [2., 10.])
        result = paired_bootstrap(means, counts, reps=100, seed=1)
        self.assertAlmostEqual(result["mean"], 14/3)
        self.assertFalse(result["independent_graph_guarantee"])

    def test_degenerate_core_cannot_pass_admission(self):
        result = admission([-0.02], [0.01], reps=50)
        self.assertFalse(result["passed"])
        self.assertIsNone(result["loss"]["upper"])

    def test_bootstrap_paired_sign_and_reproducibility(self):
        x = np.linspace(-.021, -.019, 32)
        a = paired_bootstrap(x, reps=200, seed=12)
        b = paired_bootstrap(x, reps=200, seed=12)
        self.assertEqual(a, b)
        self.assertLess(a["upper"], 0)
        self.assertLess(a["p_less_zero"], .01)

    def test_admission_clips_differences_and_uses_metric(self):
        raw = np.array([-.2, -.02, -.02, -.02])
        good = admission(raw, np.full(4, .01), reps=200, seed=9)
        self.assertAlmostEqual(good["loss"]["mean"], -.04)
        self.assertAlmostEqual(good["raw_loss"]["mean"], -.065)
        self.assertEqual(good["clipping_fraction"], .25)
        self.assertTrue(good["passed"])
        bad = admission(raw, np.full(4, -.01), reps=200, seed=9)
        self.assertFalse(bad["passed"])
        self.assertFalse(admission(raw, reps=200)["passed"])

    def test_empirical_bernstein_formula_is_diagnostic(self):
        x = np.array([-.02, -.04, -.01, -.03])
        out = empirical_bernstein(x, clip=.1, alpha=.025)
        expected = np.mean(x) + math.sqrt(2*np.var(x, ddof=1)*math.log(80)/4) + 1.4*math.log(80)/9
        self.assertAlmostEqual(out["upper"], expected)
        self.assertFalse(out["independent_graph_assumption_declared"])
        self.assertIn("no coverage", out["applicability"])

    def test_holm_adjustment(self):
        np.testing.assert_allclose(holm_adjust([.01, .04, .03]), [.03, .06, .06])

    @staticmethod
    def fixture():
        n = 30
        base = 1 + np.linspace(-.03, .03, n)
        losses = np.stack([base, base-.005, base-.025, base-.026], axis=1)
        accuracies = np.full_like(losses, .70)
        accuracies[:, 2:] += .005
        candidates = [
            {"route":"nominal", "is_open_loop":True, "complexity":0, "hash":"a"},
            {"route":"fixed_sequence", "is_open_loop":True, "complexity":4, "hash":"b"},
            {"route":"radius", "is_open_loop":False, "complexity":7, "hash":"c"},
            {"route":"mpc_rh", "is_open_loop":False, "complexity":8, "hash":"d"},
        ]
        return losses, accuracies, candidates, np.arange(n) % 5

    def test_nested_rule_cheapest_near_best_and_comparator_trace(self):
        loss, acc, cands, folds = self.fixture()
        out = nested_selection(loss, acc, cands, folds, reps=100, seed=11)
        self.assertEqual(out["comparator_index"], 1)
        self.assertEqual(out["provisional_index"], 2)
        self.assertEqual(out["eligible_indices"], [2, 3])
        self.assertTrue(out["bootstrap_resamples_reconstruct_comparator"])
        for row in out["nested_comparator_fold_choices"]:
            self.assertFalse(set(row["held_out_original_units"]) & set(row["training_original_units"]))
            self.assertEqual(row["candidate_index"], 1)
        self.assertEqual(sum(out["bootstrap_final_comparator_counts"]), 100)

    def test_nested_candidate_feasibility_and_empty_Qs(self):
        loss, acc, cands, folds = self.fixture()
        cands[2]["feasible"] = False
        # Remaining feedback arm is harmful in official accuracy.
        acc[:, 3] = .60
        out = nested_selection(loss, acc, cands, folds, reps=80, seed=2)
        self.assertEqual(out["eligible_indices"], [])
        self.assertEqual(out["provisional_index"], out["comparator_index"])

    def test_failed_candidate_columns_do_not_change_observed_selection(self):
        loss, acc, cands, folds = self.fixture()
        n = len(loss)
        # Nonconstant effects exercise genuine max-t radii, metric intervals,
        # weighted risks and comparator reconstruction rather than zero SEs.
        loss[:, 0] += .025 * np.cos(np.arange(n) * .73)
        loss[:, 2] += .004 * np.sin(np.arange(n) * .67)
        acc[:, 1] += .001 * np.sin(np.arange(n) * .41)
        acc[:, 2] += .001 * np.cos(np.arange(n) * .29)
        weights = np.arange(1, n + 1)
        baseline = nested_selection(loss, acc, cands, folds, weights=weights, reps=100, seed=19)
        self.assertGreater(baseline["max_t_critical"], 0)
        self.assertGreater(baseline["metric_max_t_critical"], 0)

        # Inserting failed open-loop AND feedback entries catches accidental
        # inclusion in either comparator selection or the simultaneous family.
        observed = [1, 2, 4, 5]
        failed = [0, 3]
        expanded_loss = np.full((n, 6), np.nan)
        expanded_acc = np.full_like(expanded_loss, np.nan)
        expanded_loss[:, observed], expanded_acc[:, observed] = loss, acc
        bad_open = {"route": "phase", "is_open_loop": True, "hash": "failed-open",
                    "complexity": 0, "execution_failure": True, "feasible": False}
        bad_feedback = {"route": "radius", "is_open_loop": False, "hash": "failed-feedback",
                        "complexity": 0, "execution_failure": True, "feasible": False}
        expanded_candidates = [bad_open, cands[0], cands[1], bad_feedback, cands[2], cands[3]]
        actual = nested_selection(expanded_loss, expanded_acc, expanded_candidates, folds,
                                  weights=weights, reps=100, seed=19)

        for key in ("comparator_index", "qC_index", "provisional_index"):
            self.assertEqual(actual[key], observed[baseline[key]])
        for key in ("eligible_indices", "Q_s"):
            self.assertEqual(actual[key], [observed[j] for j in baseline[key]])
        for key in ("max_t_critical", "metric_max_t_critical", "minimum_upper", "max_radius",
                    "structurally_rejected_resamples"):
            self.assertEqual(actual[key], baseline[key])
        for key in ("mean", "lower", "upper", "L", "U", "se", "metric_mean",
                    "metric_lower", "metric_upper", "metric_se"):
            self.assertEqual([actual[key][j] for j in observed], baseline[key])
            self.assertTrue(all(actual[key][j] is None for j in failed))
        for key in ("metric_feasible", "joint_feasible", "bootstrap_final_comparator_counts"):
            self.assertEqual([actual[key][j] for j in observed], baseline[key])
            self.assertTrue(all(not actual[key][j] for j in failed))
        self.assertEqual(actual["observations_available"], [False, True, True, False, True, True])
        self.assertEqual(actual["execution_failure_indices"], failed)
        self.assertEqual(actual["candidate_count"], 6)
        self.assertEqual(actual["observed_candidate_count"], 4)
        for original, restored in zip(baseline["nested_comparator_fold_choices"] + [baseline["final_comparator_selection"]],
                                      actual["nested_comparator_fold_choices"] + [actual["final_comparator_selection"]]):
            self.assertEqual(restored["candidate_index"], observed[original["candidate_index"]])
            self.assertEqual(restored["inner_family_risks"], original["inner_family_risks"])
            for old_fold, new_fold in zip(original["inner_fold_choices"], restored["inner_fold_choices"]):
                self.assertEqual(new_fold["family_configs"],
                                 {family: observed[j] for family, j in old_fold["family_configs"].items()})

    def test_execution_failure_requires_infeasibility_and_nonfailures_require_finite_observations(self):
        for feasible in (True, None):
            loss, acc, cands, folds = self.fixture()
            cands[2]["execution_failure"] = True
            if feasible is not None:
                cands[2]["feasible"] = feasible
            loss[:, 2] = np.nan
            acc[:, 2] = np.nan
            with self.subTest(feasible=feasible):
                with self.assertRaisesRegex(ValueError, "execution_failure.*feasible=False"):
                    nested_selection(loss, acc, cands, folds, reps=20)
        for value in (np.nan, np.inf):
            loss, acc, cands, folds = self.fixture()
            cands[2]["feasible"] = False
            loss[0, 2] = value
            with self.subTest(nonfailure_observation=value):
                with self.assertRaisesRegex(ValueError, "nonfailure.*finite"):
                    nested_selection(loss, acc, cands, folds, reps=20)

    def test_bootstrap_reselects_comparator_instead_of_caching_winner(self):
        # Two crossing open-loop rules have equal aggregate risk. Resampling
        # cores must change which rule is best; a cached comparator fails this.
        n = 30
        drift = np.linspace(-.03, .03, n)
        losses = np.column_stack([1+drift, 1-drift, np.full(n, .97)])
        candidates = [
            {"route":"depth", "is_open_loop":True, "complexity":1},
            {"route":"phase", "is_open_loop":True, "complexity":2},
            {"route":"radius", "is_open_loop":False, "complexity":7},
        ]
        out = nested_selection(losses, np.full_like(losses, .7), candidates,
                               np.arange(n) % 5, reps=100, seed=13)
        self.assertGreater(out["bootstrap_final_comparator_counts"][0], 10)
        self.assertGreater(out["bootstrap_final_comparator_counts"][1], 10)
        self.assertEqual(out["bootstrap_final_comparator_counts"][2], 0)

    def test_selection_strict_improvement_margin(self):
        loss, acc, cands, folds = self.fixture()
        loss[:, 2] = loss[:, 1]-.00299
        loss[:, 3] = loss[:, 1]-.002
        out = nested_selection(loss, acc, cands, folds, reps=80, seed=3)
        self.assertEqual(out["eligible_indices"], [])

    def test_confirmation_keeps_only_qualified_and_margin_strict(self):
        ok = confirmation(2, 0, [1, 2], [
            {"kind":"state", "candidate_index":2, "predecessor_index":0,
             "differences":np.full(12, .01), "p_value":.001},
            {"kind":"radius", "candidate_index":2, "predecessor_index":1,
             "differences":np.full(12, -.01), "p_value":.001},
        ], reps=100)
        self.assertEqual(ok["confirmed_index"], 2)
        fail = confirmation(2, 0, [2], [
            {"kind":"radius", "candidate_index":2, "predecessor_index":1,
             "differences":np.full(12, -.0015), "p_value":.001},
        ], reps=100)
        self.assertEqual(fail["confirmed_index"], 0)
        self.assertTrue(fail["membership_in_Qs_or_qC"])

    def test_missing_confirmation_fails_closed(self):
        result = confirmation(2, 0, [2], [], reps=100)
        self.assertEqual(result["confirmed_index"], 0)

    def test_unavailable_required_contrast_keeps_holm_membership_without_observations(self):
        contrasts = [
            {"kind": "radius", "candidate_index": 2, "predecessor_index": 1,
             "available": False, "unavailable_reason": "one-step execution failed"},
            {"kind": "state", "candidate_index": 1, "predecessor_index": 0,
             "differences": np.full(12, .01), "p_value": .001},
        ]
        result = confirmation(2, 0, [1, 2], contrasts, reps=100)
        self.assertEqual(result["confirmed_index"], 1)
        self.assertEqual(result["holm_family_size"], 2)
        row = result["contrasts"][0]
        self.assertFalse(row["available"])
        self.assertEqual(row["unavailable_reason"], "one-step execution failed")
        self.assertEqual(row["predecessor_index"], 1)
        for field in ("mean", "lower", "upper", "p_value_margin"):
            self.assertIsNone(row[field])
        self.assertEqual(row["units"], 0)
        self.assertFalse(row["valid_interval"])
        self.assertFalse(row["point_beyond_margin"])
        self.assertFalse(row["passes"])
        self.assertEqual(row["p_value_zero"], 1.)
        self.assertEqual(row["holm_p_value_zero"], 1.)
        # The unavailable required comparison is still a family member, so the
        # valid comparison receives the two-test Holm adjustment.
        self.assertAlmostEqual(result["contrasts"][1]["holm_p_value_zero"], .002)
        self.assertEqual(result["downgrades"][0]["to"], 1)
        # Retaining its registered predecessor does not make it eligible.
        ineligible = confirmation(2, 0, [2], contrasts, reps=100)
        self.assertEqual(ineligible["confirmed_index"], 0)
        self.assertEqual(ineligible["contrasts"][0]["predecessor_index"], 1)

    def test_invalid_intervention_differences_remain_diagnostic_and_cannot_pass(self):
        result = confirmation(2, 0, [2], [
            {"kind": "state", "candidate_index": 2, "predecessor_index": 0,
             "differences": np.full(12, .02), "p_value": .0001,
             "intervention_valid": False, "invalid_reason": "numerical control failure; qC bytes returned"},
        ], reps=100)
        row = result["contrasts"][0]
        self.assertTrue(row["available"])
        self.assertAlmostEqual(row["mean"], .02)
        self.assertEqual(row["units"], 12)
        self.assertFalse(row["intervention_valid"])
        self.assertIn("qC bytes returned", row["invalid_reason"])
        self.assertEqual(row["p_value_zero"], 1.)
        self.assertEqual(row["holm_p_value_zero"], 1.)
        self.assertFalse(row["passes"])
        self.assertEqual(result["confirmed_index"], 0)


if __name__ == "__main__":
    unittest.main()
