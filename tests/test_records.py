"""Measurement-record contract (records/): manifests, traceability of every value, printed tables.

All values below are test fixtures with artificial names; none is a measurement.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.records import (RecordsError, build_table, cmd_check, cmd_index, cmd_tables, ensemble_law,  # noqa: E402
                             load_specs, verify_set)

SPECS = PROJECT / "records" / "specs.json"
PROVENANCE = {"experiment": "fixture", "produced_by": "tests/test_records.py", "hardware": "none",
              "period": "not applicable", "protocol": "fixture"}


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def fixture_root(tmp: Path) -> Path:
    root = tmp / "records"
    root.mkdir()
    shutil.copy(SPECS, root / "specs.json")
    return root


def baseline_set(root: Path) -> Path:
    """Per-method quantities from a JSON log, a scheduler CSV and a text log; per-seed scores from JSON."""
    path = root / "baselines"
    raw = path / "raw"
    raw.mkdir(parents=True)
    (raw / "fixture_run.json").write_text(json.dumps({"cost": {"serial": 4.5012}, "params": 1.25}), encoding="utf-8")
    write_csv(raw / "fixture_accounting.csv", ["JobName", "GPUHours"], [["fixture-job", "13.61"], ["other", "1.0"]])
    (raw / "fixture_timing.log").write_text("warm-up done\nwall_time_nominal=4.663\n", encoding="utf-8")
    seeds = {1: 70.10, 2: 70.30, 3: 70.50}
    for seed, score in seeds.items():
        (raw / f"fixture_seed{seed}.json").write_text(json.dumps({"test": {"score": score}}), encoding="utf-8")
    rows = [["fixture-dataset", "fixture-method", "", "serial_cost", "4.50", "nominal", "raw/fixture_run.json", "json:/cost/serial"],
            ["fixture-dataset", "fixture-method", "", "params_millions", "1.25", "M", "raw/fixture_run.json", "json:/params"],
            ["fixture-dataset", "fixture-method", "", "gpu_hours", "13.6", "GPU-h", "raw/fixture_accounting.csv",
             "csv:JobName=fixture-job;GPUHours"],
            ["fixture-dataset", "fixture-method", "", "wall_time", "4.66", "nominal", "raw/fixture_timing.log",
             r"regex:wall_time_nominal=([0-9.]+)"]]
    rows += [["fixture-dataset", "fixture-method", str(seed), "score", f"{score:.2f}", "%", f"raw/fixture_seed{seed}.json",
              "json:/test/score"] for seed, score in seeds.items()]
    write_csv(path / "summary.csv", ["dataset", "method", "seed", "quantity", "value", "unit", "source_file", "locator"], rows)
    (path / "provenance.json").write_text(json.dumps(PROVENANCE), encoding="utf-8")
    return path


class RecordsTests(unittest.TestCase):
    def test_repository_index_matches_the_deposited_sets(self):
        self.assertEqual(cmd_check(PROJECT / "records"), 0)

    def test_traced_set_indexes_checks_and_prints_its_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_root(Path(tmp))
            path = baseline_set(root)
            self.assertEqual(cmd_index(root), 0)
            self.assertEqual(cmd_check(root), 0)
            index = json.loads((root / "index.json").read_text())
            self.assertEqual([e["set"] for e in index["sets"]], ["baselines"])
            specs = load_specs(root)
            header, rows, _ = build_table(path, specs["sets"]["baselines"], specs)
            row = dict(zip(header, rows[0]))
            self.assertEqual(row["serial_cost"], "4.50")
            self.assertEqual(row["gpu_hours"], "13.6")
            self.assertEqual(row["score (mean over seeds)"], "70.3")
            self.assertEqual(row["score SD"], "0.2")
            self.assertEqual(row["seeds"], "3")
            out = Path(tmp) / "tables"
            self.assertEqual(cmd_tables(root, out), 0)
            for suffix in ("tex", "md", "csv"):
                self.assertTrue((out / f"baselines.{suffix}").is_file())
            self.assertIn("Supplementary Table S15b", (out / "baselines.md").read_text())

    def test_raw_file_changed_after_indexing_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_root(Path(tmp))
            path = baseline_set(root)
            cmd_index(root)
            (path / "raw" / "fixture_timing.log").write_text("wall_time_nominal=4.663\nedited\n", encoding="utf-8")
            self.assertEqual(cmd_check(root), 1)

    def test_value_not_found_in_its_raw_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_root(Path(tmp))
            path = baseline_set(root)
            summary = (path / "summary.csv").read_text().replace(",13.6,GPU-h,", ",12.9,GPU-h,")
            (path / "summary.csv").write_text(summary)
            with self.assertRaisesRegex(RecordsError, "differs from the raw value"):
                cmd_index(root)

    def test_printed_precision_is_the_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_root(Path(tmp))
            path = baseline_set(root)
            # raw 4.663 printed as 4.66 passes; printed as 4.6600 it no longer agrees
            summary = (path / "summary.csv").read_text().replace(",4.66,nominal,", ",4.6600,nominal,")
            (path / "summary.csv").write_text(summary)
            with self.assertRaisesRegex(RecordsError, "differs from the raw value"):
                cmd_index(root)

    def test_contract_violations_are_rejected(self):
        cases = {
            "unknown quantity": (lambda p: p.joinpath("summary.csv").write_text(
                p.joinpath("summary.csv").read_text().replace("params_millions", "parameter_guess")), "not defined"),
            "missing provenance": (lambda p: p.joinpath("provenance.json").write_text(json.dumps({"experiment": "x"})),
                                   "provenance.json lacks"),
            "per-seed value without seed": (lambda p: p.joinpath("summary.csv").write_text(
                p.joinpath("summary.csv").read_text().replace("fixture-method,1,score", "fixture-method,,score")),
                "per-seed quantity"),
            "source outside the set": (lambda p: p.joinpath("summary.csv").write_text(
                p.joinpath("summary.csv").read_text().replace("raw/fixture_run.json,json:/params", "../x.json,json:/params")),
                "not a deposited file"),
        }
        for name, (mutate, message) in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp:
                root = fixture_root(Path(tmp))
                mutate(baseline_set(root))
                with self.assertRaisesRegex(RecordsError, message):
                    cmd_index(root)

    def test_unknown_set_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_root(Path(tmp))
            (root / "not_a_set").mkdir()
            with self.assertRaisesRegex(RecordsError, "not a set named"):
                cmd_index(root)

    def test_ensemble_law_is_the_least_squares_fit_of_table_s26b(self):
        points = [(e, 0.80 + 0.03 / e) for e in (1, 2, 4, 8, 12, 16, 20)]
        fit = ensemble_law(points)
        self.assertAlmostEqual(fit["l_inf"], 0.80, places=12)
        self.assertAlmostEqual(fit["D_inf"], 0.03, places=12)
        self.assertLess(fit["max_abs_residual"], 1e-12)
        with self.assertRaises(RecordsError):
            ensemble_law([(4, 0.81)])

    def test_configuration_range_reports_the_selected_configuration_and_the_span(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_root(Path(tmp))
            path = root / "budget_matched"
            (path / "raw").mkdir(parents=True)
            effects = {"selected": -1.6, "alt-a": -0.9, "alt-b": -2.5}
            (path / "raw" / "arms.json").write_text(json.dumps(effects))
            write_csv(path / "summary.csv", ["dataset", "arm", "configuration", "quantity", "value", "unit", "source_file", "locator"],
                      [["fixture-dataset", "E3", name, "effect_mnats", f"{value:.1f}", "mnats", "raw/arms.json", f"json:/{name}"]
                       for name, value in effects.items()])
            (path / "provenance.json").write_text(json.dumps(PROVENANCE))
            self.assertEqual(cmd_index(root), 0)
            specs = load_specs(root)
            header, rows, _ = build_table(path, specs["sets"]["budget_matched"], specs)
            row = dict(zip(header, rows[0]))
            self.assertEqual(row["effect_mnats (selected)"], "-1.6")
            self.assertEqual(row["effect_mnats range"], "[-2.5, -0.9]")
            verify_set(path, specs["sets"]["budget_matched"], specs)


if __name__ == "__main__":
    unittest.main()
