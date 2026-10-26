"""Run-driven paper build: evidence extraction, tables and the check mode (NumPy only)."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from paper.build.evidence import EvidenceError, evidence_from_campaign, evidence_from_run  # noqa: E402
from paper.build.tables import TABLE_IDS, build_tables  # noqa: E402

EXAMPLES = PROJECT / "examples" / "runs"
STEMS = ("arxiv", "products", "proteins", "roman_empire", "pascalvoc_sp", "coco_sp", "peptides_func")


@unittest.skipUnless((EXAMPLES / "arxiv" / "final" / "results_summary.json").is_file(), "bundled examples absent")
class EvidenceTests(unittest.TestCase):
    def test_every_bundled_example_yields_a_complete_record(self):
        for stem in STEMS:
            with self.subTest(stem=stem):
                record = evidence_from_run(EXAMPLES / stem)
                self.assertEqual(record["schema"], "cross-morse-evidence-1")
                self.assertFalse(record["scientific_run"])
                self.assertEqual(len(record["candidates"]), record["selection"]["candidate_count"])
                self.assertIsNotNone(record["final"]["models"]["qD"]["metric"])
                self.assertIsNotNone(record["cost"]["C_qC"])
                self.assertIn(record["admission"]["Gadmit"], (0, 1))
                # nothing invented: the unmeasured fallback probability stays None
                self.assertIsNone(record["cost"]["fallback_probability"])
                json.dumps(record, allow_nan=False)

    def test_values_are_read_not_recomputed(self):
        record = evidence_from_run(EXAMPLES / "arxiv")
        summary = json.loads((EXAMPLES / "arxiv" / "final" / "results_summary.json").read_text())
        self.assertEqual(record["final"]["models"]["qC"]["metric"], summary["models"]["qC"]["metric"])
        self.assertEqual(record["final"]["paired_loss"]["mean"], summary["paired_qD_minus_qC_loss"]["mean"])
        state = json.loads((EXAMPLES / "arxiv" / "run_state.json").read_text())
        self.assertEqual(record["identities"]["source_sha256"], state["source_sha256"])

    def test_incomplete_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(EvidenceError):
                evidence_from_run(Path(tmp))

    def test_campaign_aggregates_split_effects_with_a_t_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / "campaign"
            for index in range(3):
                target = campaign / f"split_{index:02d}"
                import shutil
                shutil.copytree(EXAMPLES / "roman_empire", target)
            record = evidence_from_campaign(campaign)
            self.assertEqual(record["kind"], "campaign")
            self.assertEqual(len(record["campaign"]["per_split"]), 3)
            self.assertEqual(record["final"]["paired_loss"]["units"], 3)
            self.assertIn("t interval", record["final"]["paired_loss"]["method"])


@unittest.skipUnless((EXAMPLES / "arxiv" / "final" / "results_summary.json").is_file(), "bundled examples absent")
class TableTests(unittest.TestCase):
    def test_all_tables_are_written_in_three_formats_with_a_synthetic_banner(self):
        records = [evidence_from_run(EXAMPLES / stem) for stem in STEMS]
        with tempfile.TemporaryDirectory() as tmp:
            written = build_tables(records, Path(tmp), "unit-test")
            self.assertEqual(set(written), set(TABLE_IDS))
            for name, paths in written.items():
                for suffix in ("tex", "md", "csv"):
                    path = Path(tmp) / paths[suffix]
                    self.assertTrue(path.is_file(), path)
                tex = (Path(tmp) / paths["tex"]).read_text()
                self.assertIn("SYNTHETIC SOFTWARE EXAMPLES", tex)
                self.assertIn(r"\begin{tabular}", tex)
                self.assertIn("Synthetic software examples", (Path(tmp) / paths["md"]).read_text())
                first_csv_line = (Path(tmp) / paths["csv"]).read_text().splitlines()[0]
                self.assertIn("SYNTHETIC SOFTWARE EXAMPLES", first_csv_line)
            index = json.loads((Path(tmp) / "INDEX.json").read_text())
            self.assertTrue(index["synthetic_examples"])
            self.assertEqual(len(index["datasets"]), 7)

    def test_unmeasured_cells_are_dashes_never_numbers(self):
        records = [evidence_from_run(EXAMPLES / "arxiv")]
        with tempfile.TemporaryDirectory() as tmp:
            build_tables(records, Path(tmp), "unit-test", tables=("T1", "T2", "S23"))
            import csv
            with (Path(tmp) / "T1.csv").open(newline="") as handle:
                rows = list(csv.reader(handle))
            header = rows[2]
            row = next(r for r in rows[3:] if r and r[0] == "ogbn-arxiv")
            self.assertEqual(row[header.index("input fallback")], "")
            s23 = (Path(tmp) / "S23.md").read_text()
            self.assertIn("| — |", s23)

    def test_route_tables_always_show_the_comparator_of_every_run(self):
        records = [evidence_from_run(EXAMPLES / stem) for stem in STEMS]
        with tempfile.TemporaryDirectory() as tmp:
            build_tables(records, Path(tmp), "unit-test", tables=("S12", "S15"))
            for name in ("S12", "S15"):
                text = (Path(tmp) / f"{name}.md").read_text()
                for record in records:
                    with self.subTest(table=name, dataset=record["dataset"]):
                        rows = [line for line in text.splitlines() if line.startswith(f"| {record['dataset']} |")]
                        self.assertTrue(any("q_C" in line for line in rows), rows)

    def test_cli_check_mode_detects_a_stale_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "generated"
            cmd = [sys.executable, "-B", str(PROJECT / "scripts/build_paper.py"), "tables", "--runs-root", str(EXAMPLES),
                   "--label", "examples", "--output-dir", str(out)]
            subprocess.run(cmd, check=True, cwd=PROJECT, capture_output=True, text=True, timeout=600)
            # evidence must exist for --check; write it as `all` would
            subprocess.run(cmd[:3] + ["evidence"] + cmd[4:], check=True, cwd=PROJECT, capture_output=True, text=True, timeout=600)
            result = subprocess.run(cmd + ["--check"], cwd=PROJECT, capture_output=True, text=True, timeout=600)
            self.assertEqual(result.returncode, 0, result.stderr)
            (out / "tables" / "T2.md").write_text("edited by hand\n")
            result = subprocess.run(cmd + ["--check"], cwd=PROJECT, capture_output=True, text=True, timeout=600)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("T2.md", result.stderr)


@unittest.skipUnless((EXAMPLES / "arxiv" / "final" / "results_summary.json").is_file(), "bundled examples absent")
class SelectionReadingTests(unittest.TestCase):
    """Supplementary Table S12c: the three equivalence readings of main Eq. (19)."""

    def fixture(self, provisional_route: str):
        record = evidence_from_run(EXAMPLES / "arxiv")
        record = json.loads(json.dumps(record))
        for c in record["candidates"]:
            c["joint_feasible"] = False
            c["select_upper"], c["select_delta"] = 0.05, 0.01
        radius = next(c for c in record["candidates"] if c["route"] == "radius" and c["ensemble"] == 1)
        mpc_rh = next(c for c in record["candidates"] if c["route"] == "mpc_rh" and c["ensemble"] == 1)
        # Roman/Pascal/COCO pattern of Table S12c: both near-best on U_s, only MPC-RH on the point estimate at 0.001
        radius.update(joint_feasible=True, select_upper=-0.0040, select_delta=-0.0035)
        mpc_rh.update(joint_feasible=True, select_upper=-0.0050, select_delta=-0.0052)
        chosen = radius if provisional_route == "radius" else mpc_rh
        record["selection"]["provisional_index"] = chosen["index"]
        return record, radius, mpc_rh

    def test_three_readings_follow_the_registered_margins(self):
        from paper.build.selection import selection_readings
        record, radius, mpc_rh = self.fixture("mpc_rh")
        audit = selection_readings(record)
        readings = audit["readings"]
        self.assertAlmostEqual(audit["u_min"], -0.0050)
        self.assertEqual(readings["registered"]["near_best"], [radius["index"], mpc_rh["index"]])
        self.assertEqual(readings["registered"]["choice_route"], "radius")
        self.assertEqual(readings["point-0.002"]["choice_route"], "radius")
        self.assertEqual(readings["point-0.001"]["near_best_routes"], ["mpc_rh"])
        self.assertEqual(readings["point-0.001"]["choice_route"], "mpc_rh")
        self.assertFalse(audit["registered_consistent"])
        # the runner's own rule over every configuration picks radius; a record that deployed MPC-RH is flagged
        self.assertEqual(audit["registered_all_configurations_index"], radius["index"])
        self.assertFalse(audit["registered_all_configurations_reproduces_selection"])

    def test_registered_selection_is_reproduced_and_printed_in_s12c(self):
        from paper.build.selection import selection_readings
        record, radius, _ = self.fixture("radius")
        audit = selection_readings(record)
        self.assertTrue(audit["registered_consistent"])
        self.assertTrue(audit["registered_all_configurations_reproduces_selection"])
        with tempfile.TemporaryDirectory() as tmp:
            build_tables([record], Path(tmp), "unit-test", tables=("S12c",))
            row = next(line for line in (Path(tmp) / "S12c.md").read_text().splitlines() if line.startswith("| ogbn-arxiv |"))
            self.assertIn("radius, MPC-RH -> radius", row)
            self.assertIn("MPC-RH -> MPC-RH", row)
            self.assertIn("reproduces the sealed selection", row)

    def test_bundled_examples_reproduce_their_sealed_selection(self):
        from paper.build.selection import selection_readings
        for stem in STEMS:
            with self.subTest(stem=stem):
                audit = selection_readings(evidence_from_run(EXAMPLES / stem))
                self.assertTrue(audit["registered_all_configurations_reproduces_selection"])


def _rewrite(path: Path, update) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    update(data)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def admitted_copy(target: Path) -> int:
    """Copy the arxiv example and rewrite its decision records as if a radius rule had been admitted.

    The bundled synthetic examples all end with G_admit = 0, so this is the only exercise of the
    deployment branch of the builders before official archives exist.  The rewritten numbers are
    test fixtures, not results.
    """
    shutil.copytree(EXAMPLES / "arxiv", target)
    candidates = json.loads((target / "select" / "candidates.json").read_text(encoding="utf-8"))
    rule = next(i for i, c in enumerate(candidates) if c.get("route") == "radius")

    def interval(mean, lower, upper, units):
        return {"mean": mean, "lower": lower, "upper": upper, "se": (upper - lower) / 3.92, "units": units,
                "method": "fixture", "valid_interval": True, "independent_graph_guarantee": False}

    def selection(d):
        d.update(provisional_index=rule, Q_s=[rule], eligible_indices=[rule])

    def confirmation(d):
        d.update(provisional_index=rule, confirmed_index=rule, holm_family_size=1, downgrades=[],
                 membership_in_Qs_or_qC=True,
                 contrasts=[{"kind": "radius_vs_one_step", "candidate_index": rule, "predecessor_index": d["comparator_index"],
                             "beneficial_direction": -1, "practical_margin": 0.0015, "available": True, "mean": -0.004,
                             "lower": -0.006, "upper": -0.002, "units": 10, "point_beyond_margin": True,
                             "p_value_zero": 0.001, "p_value_margin": 0.01, "valid_interval": True,
                             "holm_p_value_zero": 0.001, "passes": True}])

    def admission(d):
        d.update(Gadmit=1, passed=True, deployed_index=rule, candidate_already_comparator=False,
                 loss=interval(-0.0042, -0.0061, -0.0019, 8))

    def summary(d):
        d.update(admitted=True, exact_fallback=False,
                 paired_qD_minus_qC_loss=interval(-0.0045, -0.0070, -0.0020, 8),
                 paired_qD_minus_qC_metric_pp=interval(0.62, 0.10, 1.10, 8))
        d["models"]["qD"]["metric_percent"] = d["models"]["qC"]["metric_percent"] + 0.62
        d["models"]["qD"]["loss_nats"] = d["models"]["qC"]["loss_nats"] - 0.0045

    _rewrite(target / "select" / "decision.json", selection)
    _rewrite(target / "confirm" / "decision.json", confirmation)
    _rewrite(target / "admit" / "decision.json", admission)
    _rewrite(target / "final" / "results_summary.json", summary)
    return rule


@unittest.skipUnless((EXAMPLES / "arxiv" / "final" / "results_summary.json").is_file(), "bundled examples absent")
class AdmittedRunTests(unittest.TestCase):
    """The deployment branch: a feedback rule selected, confirmed and admitted."""

    def test_evidence_tables_and_figures_follow_an_admitted_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "arxiv"
            rule = admitted_copy(run)
            record = evidence_from_run(run)
            self.assertEqual(record["admission"]["Gadmit"], 1)
            self.assertEqual(record["admission"]["deployed"]["route"], "radius")
            cost = record["cost"]
            self.assertAlmostEqual(cost["C_assure"], cost["C_qD"] + cost["C_qC"])
            self.assertAlmostEqual(cost["overhead_percent"], 100.0 * (cost["C_qD"] - cost["C_qC"]) / cost["C_qC"])
            if cost["C_qD"] > cost["C_qC"]:
                self.assertAlmostEqual(cost["exchange_rate_mnats_per_nominal_unit"],
                                       1000.0 * 0.0045 / (cost["C_qD"] - cost["C_qC"]))
            others = [evidence_from_run(EXAMPLES / stem) for stem in STEMS if stem != "arxiv"]
            out = Path(tmp) / "tables"
            build_tables([record] + others, out, "unit-test")

            def rows(name):
                return [line for line in (out / f"{name}.md").read_text().splitlines() if line.startswith("| ogbn-arxiv |")]

            t1 = rows("T1")[0]
            self.assertIn("G_admit = 1", t1)
            self.assertIn("radius_vs_one_step -0.00400 (Holm P = 0.001; pass)", t1)
            self.assertIn("+0.62 [+0.10, +1.10] accuracy pp", t1)
            t2 = rows("T2")[0]
            self.assertIn("radius", t2)
            self.assertIn(f"{cost['C_assure']:.3f}", t2)
            s12 = rows("S12")
            self.assertTrue(any("| radius |" in line and "selected; deployed" in line for line in s12), s12)
            self.assertTrue(any("q_C" in line for line in s12), s12)
            self.assertTrue(any("radius_vs_one_step" in line for line in rows("S14")))
            s15 = rows("S15")
            deployed = next(line for line in s15 if "(selected; deployed)" in line)
            self.assertIn(f"{record['final']['models']['qD']['metric_percent']:.2f}", deployed)
            self.assertTrue(any("(q_C)" in line for line in s15), s15)
            self.assertIn(f"{rule}", (out / "S14.md").read_text())

            from paper.build.figures import FIGURE_IDS, build_figures
            written = build_figures([record] + others, Path(tmp) / "figures", "unit-test")
            self.assertEqual(set(written), set(FIGURE_IDS))
            for paths in written.values():
                self.assertGreater((Path(tmp) / "figures" / paths["png"]).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
