"""The manuscript registry: executable constants equal registered constants, and the reported ledgers are arithmetically coherent.

These tests need only the standard library. They are the mechanised form of the
checks a reviewer performs by hand: that the action registry in ``core.py`` is
the registry of Supplementary Table S6, that its formulas reproduce Table S6b and
Figure S15b, that the seven full profiles carry the registered thresholds, and
that main Tables 1 and 2 follow from the supplementary ledgers they summarize.
When PyTorch is available, the registry formulas are additionally compared with
``cross_morse.core`` itself.
"""
from __future__ import annotations

import json
import math
import copy
import csv
import hashlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts import check_paper_constants
from scripts import render_paper_results
from scripts._paper_common import (BCE_DATASETS, DATASETS, PAPER, REGISTERED, REPORTED, ROOT, STEMS, band_sensitivity,
                                   bernstein_penalty, executed_hops, executed_identity_coefficient, full_config,
                                   identity_mass, load_registered, load_reported, minimum_admission_J,
                                   parse_actions_from_core, printed_half_unit, runtime_source_fingerprint)

try:  # optional: the torch-backed registry
    from cross_morse import core as torch_core
except Exception:  # pragma: no cover - torch absent
    torch_core = None


class RegistryFilesTests(unittest.TestCase):
    def test_registry_files_exist_and_name_their_provenance(self):
        for path in (REGISTERED, REPORTED):
            self.assertTrue(path.is_file(), path)
            value = json.loads(path.read_text(encoding="utf-8"))
            provenance = value["_provenance"]
            self.assertEqual(provenance["repository"], "https://github.com/vkhs/CROSS-MORSE")
            self.assertIn("transcription_rule", provenance)
            self.assertIn("Scientific Reports", provenance["venue"])

    def test_every_dataset_has_rows_in_the_main_ledgers(self):
        reported = load_reported()
        for key in ("main_table_1_policy_and_deployment_ledger", "main_table_2_gain_and_cost_ledger",
                    "supplementary_table_S4_target_qualification", "supplementary_table_S12_selection_ledger",
                    "supplementary_table_S15_method_comparison", "supplementary_table_S20_selection_theorem_instantiation",
                    "supplementary_table_S23_assurance_and_expected_compute", "supplementary_table_S41_qualification_cost_components"):
            self.assertEqual({k for k in reported[key] if not k.startswith("_")}, set(DATASETS), key)

    def test_rendered_markdown_is_current(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "render_reported_results.py"), "--check"],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_bindings_reference_registry_values(self):
        reported = load_reported()
        t1 = reported["main_table_1_policy_and_deployment_ledger"]
        for dataset in DATASETS:
            path = PAPER / "bindings" / f"{STEMS[dataset]}_full.json"
            self.assertTrue(path.is_file(), path)
            binding = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(binding["schema_version"], "result-binding-1")
            self.assertEqual(binding["status"], "template_prefilled_with_reported_values")
            self.assertEqual(binding["run_identities"]["dataset"], dataset)
            for entry in binding["results"]:
                self.assertTrue(entry["json_pointer"].startswith("/"))
                self.assertIsInstance(entry["absolute_tolerance"], float)
                # an exact zero (the comparator-return case) binds with zero tolerance; everything else with its printed half-unit
                self.assertGreaterEqual(entry["absolute_tolerance"], 0)
                self.assertEqual(entry["absolute_tolerance"] == 0, entry["reported_value"] == 0)
            pointers = {e["json_pointer"]: e["reported_value"] for e in binding["results"]}
            self.assertEqual(pointers["/paired_qD_minus_qC_loss/mean"], t1[dataset]["loss_effect_nats"])
            self.assertEqual(pointers["/paired_qD_minus_qC_metric_pp/mean"], t1[dataset]["official_gain_pp"])


class ActionRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registered = load_registered()
        self.code = {a["id"]: a for a in parse_actions_from_core()}
        self.table = {r["id"]: r for r in self.registered["action_registry_table_S6"]["actions"]}

    def test_ten_actions_with_the_registered_parameters(self):
        self.assertEqual(sorted(self.code), list(range(10)))
        for identifier, row in self.table.items():
            code = self.code[identifier]
            self.assertAlmostEqual(code["probability"], row["exploration_probability"], places=12, msg=f"A{identifier}")
            if row["mode"] == "direct":
                self.assertIn(code["mode"], ("source", "last_safe"), f"A{identifier}")
                continue
            self.assertEqual(code["mode"], "transport", f"A{identifier}")
            self.assertEqual((code["rho"], code["nu"], code["alpha"]), (row["rho"], row["nu"], row["alpha"]), f"A{identifier}")
            self.assertEqual(code["eta"], row["eta"], f"A{identifier}")
            if row["eta"] is not None:
                self.assertEqual(code["order"], row["T"], f"A{identifier}")
        self.assertAlmostEqual(sum(a["probability"] for a in self.code.values()), 1.0, places=12)

    def test_formulas_reproduce_table_S6b(self):
        derived = self.registered["action_registry_table_S6"]["derived_table_S6b"]
        for key, row in derived.items():
            code = self.code[int(key[1:])]
            self.assertAlmostEqual(executed_hops(code["rho"], code["nu"], code["eta"], code["order"]), row["executed_range_hops_per_step"], delta=0.0005, msg=key)
            self.assertAlmostEqual(executed_identity_coefficient(code["rho"], code["nu"], code["eta"], code["order"]), row["executed_identity_coefficient"], delta=0.0005, msg=key)
            mass = identity_mass(code["eta"], code["order"])
            if row["kernel_identity_mass"] is None:
                self.assertIsNone(mass, key)
            else:
                self.assertAlmostEqual(mass, row["kernel_identity_mass"], delta=0.00005, msg=key)

    def test_band_sensitivities_reproduce_figure_S15b(self):
        figure = self.registered["action_registry_table_S6"]["band_sensitivity_figure_S15b"]
        for key, printed in figure.items():
            if not key.startswith("A"):
                continue
            values = band_sensitivity(self.code[int(key[1:])], self.code[0])
            for b, (value, target) in enumerate(zip(values, printed)):
                self.assertAlmostEqual(value, target, delta=0.00501, msg=f"{key} band {b}")

    def test_local_actions_have_closed_form_sensitivity(self):
        # For one-step kernels g_a(lambda) = 1 - nu lambda, so s_ab = (nu_A0 - nu_a) * right band edge.
        for identifier in (1, 2):
            values = band_sensitivity(self.code[identifier], self.code[0])
            for b, value in enumerate(values):
                self.assertAlmostEqual(value, abs(self.code[0]["nu"] - self.code[identifier]["nu"]) * 2.0 * (b + 1) / 6.0, delta=1e-9)

    @unittest.skipIf(torch_core is None, "PyTorch is not installed; the torch-backed registry is checked in the full test suite")
    def test_pure_python_formulas_agree_with_core(self):
        import torch
        for action in torch_core.ACTIONS[:8]:
            self.assertAlmostEqual(torch_core.executed_hops(action), executed_hops(action.rho, action.nu, action.eta, action.order), places=12)
            self.assertAlmostEqual(torch_core.executed_identity_coefficient(action), executed_identity_coefficient(action.rho, action.nu, action.eta, action.order), places=12)
            self.assertAlmostEqual(torch_core.identity_mass(action), identity_mass(action.eta, action.order) or 0.0, places=12)
            grid = torch.linspace(0.0, 2.0, 41, dtype=torch.float64)
            response = torch_core.action_operator_response(action, grid)
            for lam, value in zip(grid.tolist(), response.tolist()):
                self.assertAlmostEqual(value, torch_core.action_operator_response(action, torch.tensor([lam], dtype=torch.float64)).item(), places=12)
                from scripts._paper_common import spectral_response
                self.assertAlmostEqual(value, spectral_response(action.rho, action.nu, action.eta, lam, action.order), places=9)


class FullProfileTests(unittest.TestCase):
    def test_registered_margins_depend_on_the_loss_type(self):
        for dataset in DATASETS:
            c = full_config(dataset)
            expected = 0.0015 if dataset in BCE_DATASETS else 0.003
            self.assertEqual(c["selection"]["epsilon_r"], expected, dataset)
            self.assertEqual(c["admission"]["epsilon_a"], expected, dataset)
            self.assertEqual(c["selection"]["epsilon_eq"], 0.002, dataset)
            self.assertEqual(c["admission"]["clip"], 0.10, dataset)
            self.assertEqual((c["selection"]["alpha"], c["admission"]["alpha"]), (0.025, 0.025), dataset)
            self.assertEqual(c["selection"]["bootstrap_reps"], 10000, dataset)

    def test_check_paper_constants_passes(self):
        exit_code = check_paper_constants.main(["--quiet"])
        self.assertEqual(exit_code, 0)


class ReportedArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.reported = load_reported()
        self.registered = load_registered()

    def test_admission_bound_arithmetic_table_S13(self):
        for dataset, r in self.reported["supplementary_table_S13_independent_graph_admission"].items():
            penalty = bernstein_penalty(r["S2"], r["J"], r["B"], r["delta_a"])
            self.assertAlmostEqual(penalty, r["penalty"], delta=5.1e-6, msg=dataset)
            self.assertAlmostEqual(r["D_bar"] + penalty, r["U_a"], delta=5.1e-6, msg=dataset)
            self.assertEqual(int(r["U_a"] <= -r["epsilon_a"]), r["G_admit"], dataset)

    def test_corollary_2_minimum_sample_sizes(self):
        t1 = self.reported["main_table_1_policy_and_deployment_ledger"]
        for dataset, expected in (("ogbn-arxiv", 1279), ("ogbn-proteins", 853), ("Roman-empire", 602)):
            eps = 0.0015 if dataset in BCE_DATASETS else 0.003
            # The main text prints the integer part of the threshold: "J > 602 (Roman), 853 (proteins) and 1279 (arxiv)".
            self.assertEqual(math.floor(minimum_admission_J(t1[dataset]["admission_D_bar"], eps)), expected, dataset)

    def test_table_2_follows_from_tables_S23_and_S41(self):
        t1 = self.reported["main_table_1_policy_and_deployment_ledger"]
        t2 = self.reported["main_table_2_gain_and_cost_ledger"]
        s23 = self.reported["supplementary_table_S23_assurance_and_expected_compute"]
        s41 = self.reported["supplementary_table_S41_qualification_cost_components"]
        for dataset in DATASETS:
            if dataset == "ogbn-products":
                continue
            expected, comparator = s23[dataset]["expected_serial_cost"], s23[dataset]["comparator_cost"]
            self.assertAlmostEqual(expected, t2[dataset]["C_expected"], delta=0.005, msg=dataset)
            self.assertAlmostEqual(1000 * abs(t1[dataset]["loss_effect_nats"]) / (expected - comparator), t2[dataset]["lambda_star_mnats"], delta=0.01, msg=dataset)
            amortization = 3600 * s41[dataset]["total_qualification_increment_GPU_h"] / (0.1 * (expected - comparator)) / 1000
            self.assertAlmostEqual(amortization, t2[dataset]["M_star_t_nom_thousand_input_seconds"], delta=0.5, msg=dataset)

    def test_table_1_gains_match_or_preserve_documented_discrepancy(self):
        t1 = self.reported["main_table_1_policy_and_deployment_ledger"]
        s15 = self.reported["supplementary_table_S15_method_comparison"]
        for dataset in DATASETS:
            deployed = next(r for r in s15[dataset]["rows"] if r["family"] == "CROSS-MORSE deployment")
            comparator = next(r for r in s15[dataset]["rows"] if "(qC" in r["method"])
            scale = 100.0 if s15[dataset]["score_unit"].startswith("AP") else 1.0
            gain = (deployed["official_score"] - comparator["official_score"]) * scale
            if dataset == "Peptides-func":
                # S26 prints four-decimal scores and explicitly rejects rounding
                # alone as an explanation for this 0.0002 native-AP difference.
                note = self.reported["_interpretation_notes"]["peptides_score_discrepancy"]
                self.assertAlmostEqual((gain - t1[dataset]["official_gain_pp"]) / 100,
                                       note["difference_minus_main_paired_gain_AP"], places=12)
                self.assertGreater(note["difference_minus_main_paired_gain_AP"], 0.0001)
            else:
                self.assertAlmostEqual(gain, t1[dataset]["official_gain_pp"], delta=0.011, msg=dataset)

    def test_printed_half_unit(self):
        self.assertEqual(printed_half_unit(-0.0045), 0.00005)
        self.assertEqual(printed_half_unit(73.22), 0.005)
        self.assertEqual(printed_half_unit(112), 0.5)
        self.assertEqual(printed_half_unit(0.0), 0.0)


class RegistryFigureTests(unittest.TestCase):
    def test_registry_figures_render(self):
        try:
            import matplotlib  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib/numpy are not installed")
        import tempfile
        from scripts import plot_registry_figures
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(plot_registry_figures.main(["--output-dir", tmp]), 0)
            names = sorted(p.name for p in Path(tmp).glob("registry_*.png"))
            self.assertEqual(names, ["registry_admission_arithmetic.png", "registry_deployment_economics.png", "registry_gain_and_cost.png",
                                     "registry_mechanism_contrasts.png", "registry_selection_geometry.png", "registry_spectral_fingerprint.png"])


class SourceIdentityTests(unittest.TestCase):
    def test_runtime_fingerprint_matches_the_validation_record(self):
        record = json.loads((ROOT / "docs" / "VALIDATION.json").read_text(encoding="utf-8"))
        recorded = {run["identities"]["source_sha256"] for run in record["smoke_runs"]}
        self.assertEqual(len(recorded), 1)
        self.assertEqual(runtime_source_fingerprint(ROOT), recorded.pop())


class PaperSourceProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.reported = load_reported()
        self.index = json.loads((PAPER / "sources.json").read_text(encoding="utf-8"))

    def test_recorded_manuscript_identities_cover_every_table_group(self):
        result = render_paper_results.validate_sources(ROOT, self.reported, self.index)
        self.assertEqual(result["pdf_sha256"], {
            "main": "82c8c3fada494c25670095dd9a9a31e504a2da464fe41ec6fbe1511761b4ba8c",
            "supplementary": "2ea7c8f041777939c7ced7f73b9d593db15e96072c3ce86ae2c342d30800fd65"})
        # the manuscript is not distributed: identities are checked as recorded, never claimed as byte-verified
        self.assertEqual(result["pdf_bytes_verified"], {"main": False, "supplementary": False})
        for entry in self.index["documents"].values():
            self.assertIsNone(entry["bundled_path"])
            self.assertFalse((ROOT / "paper" / "manuscript.pdf").exists())
        self.assertEqual(result["table_groups_checked"], len([k for k in self.reported if not k.startswith("_")]))

    def test_a_local_copy_is_verified_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = copy.deepcopy(self.index)
            copies = {}
            for name, entry in index["documents"].items():
                payload = b"%PDF-1.7\nlocal copy fixture\n" + name.encode()
                copies[name] = root / f"{name}.pdf"
                copies[name].write_bytes(payload)
                entry["byte_size"] = len(payload)
                entry["sha256"] = hashlib.sha256(payload).hexdigest()
            result = render_paper_results.validate_sources(ROOT, self.reported, index, copies)
            self.assertEqual(result["pdf_bytes_verified"], {"main": True, "supplementary": True})
            copies["main"].write_bytes(b"%PDF-1.7\nanother version\n")
            with self.assertRaisesRegex(ValueError, "content identity changed"):
                render_paper_results.validate_sources(ROOT, self.reported, index, copies)

    def temporary_sources(self, root):
        """Small byte fixtures exercise validation without copying research PDFs."""
        index = copy.deepcopy(self.index)
        (root / "paper").mkdir()
        for name, entry in index["documents"].items():
            payload = b"%PDF-1.7\nprovenance byte fixture\n" + name.encode()
            entry["bundled_path"] = f"paper/{name}.pdf"
            path = root / entry["bundled_path"]
            path.write_bytes(payload)
            entry["byte_size"] = len(payload)
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
        return index

    def test_changed_pdf_is_rejected_even_when_manifest_shape_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = self.temporary_sources(root)
            render_paper_results.validate_sources(root, self.reported, index)
            (root / index["documents"]["main"]["bundled_path"]).write_bytes(b"%PDF-1.7\nchanged content\n")
            with self.assertRaisesRegex(ValueError, "content identity changed"):
                render_paper_results.validate_sources(root, self.reported, index)

    def test_table_provenance_cannot_silently_omit_a_group(self):
        index = copy.deepcopy(self.index)
        del index["table_groups"]["main_table_1_policy_and_deployment_ledger"]
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            render_paper_results.validate_sources(ROOT, self.reported, index)

    def test_page_reference_must_be_inside_the_named_source(self):
        index = copy.deepcopy(self.index)
        index["table_groups"]["main_table_1_policy_and_deployment_ledger"]["pages"] = [21]
        with self.assertRaisesRegex(ValueError, "Invalid source page"):
            render_paper_results.validate_sources(ROOT, self.reported, index)

    def test_pdf_path_cannot_escape_repository(self):
        index = copy.deepcopy(self.index)
        index["documents"]["main"]["bundled_path"] = "../manuscript.pdf"
        with self.assertRaisesRegex(ValueError, "Invalid relative PDF path"):
            render_paper_results.validate_sources(ROOT, self.reported, index)

    def test_numeric_registry_rejects_nonfinite_values(self):
        reported = copy.deepcopy(self.reported)
        reported["main_table_1_policy_and_deployment_ledger"]["ogbn-arxiv"]["loss_effect_nats"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            render_paper_results.validate_sources(ROOT, reported, self.index)

    def test_measurements_are_distinguished_from_sensitivity_models(self):
        groups = self.index["table_groups"]
        self.assertEqual(groups["supplementary_table_S14_mechanism_contrasts_submitted_protocol"]["evidence_kind"],
                         "author_reported_measurements")
        for key in ("supplementary_table_S14b_separated_data_sensitivity", "supplementary_table_S42_power_of_budget_matched_test",
                    "supplementary_table_S44_per_input_cost_model"):
            self.assertEqual(groups[key]["evidence_kind"], "analytical_sensitivity")
        self.assertEqual(groups["supplementary_table_S46_budget_matched_frontier"]["evidence_kind"],
                         "author_reported_measurements")

    def test_unavailable_q0_and_corrected_support_are_not_fabricated(self):
        availability = self.reported["_availability"]
        for key in ("q0_absolute_test_loss", "q0_absolute_official_test_score", "qN_qC_qD_absolute_test_losses",
                    "corrected_support_ESS_and_max_weight"):
            self.assertEqual(availability[key]["status"], "not_tabulated")
            self.assertIsNone(availability[key]["value"])
        for row in self.reported["supplementary_table_S7_support_audit"].values():
            self.assertIsNone(row["corrected_minimum_ESS"])
            self.assertIsNone(row["corrected_maximum_normalized_weight"])

    def test_csv_preserves_missing_values_and_exact_comparator_return(self):
        rows = list(csv.DictReader(io.StringIO(render_paper_results.render_csv(self.reported, self.index))))
        self.assertEqual([r["dataset"] for r in rows], list(DATASETS))
        by_dataset = {r["dataset"]: r for r in rows}
        self.assertTrue(all(r["q0_score"] == "" for r in rows))
        products = by_dataset["ogbn-products"]
        self.assertEqual(float(products["paired_qD_minus_qC_loss_nats"]), 0.0)
        self.assertEqual(products["loss_CI95_lower"], "")
        self.assertEqual(products["loss_CI95_upper"], "")
        peptides = by_dataset["Peptides-func"]
        self.assertAlmostEqual(float(peptides["metric_gain_pp"]), 100 * self.reported["main_table_1_policy_and_deployment_ledger"]["Peptides-func"]["official_gain_native"])
        self.assertEqual(float(peptides["qN_score"]), 0.704)

    def test_overview_and_csv_render_from_the_canonical_values(self):
        reported = copy.deepcopy(self.reported)
        reported["main_table_1_policy_and_deployment_ledger"]["ogbn-arxiv"]["loss_effect_nats"] = -0.00451
        overview = render_paper_results.render_overview(reported, self.index)
        self.assertIn("−0.00451", overview)
        rows = list(csv.DictReader(io.StringIO(render_paper_results.render_csv(reported, self.index))))
        self.assertEqual(float(rows[0]["paired_qD_minus_qC_loss_nats"]), -0.00451)

    def test_generated_summary_views_are_current(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/render_paper_results.py"), "--check"],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
