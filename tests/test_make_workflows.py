"""Exercise Make routing with a recording interpreter, never real training."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
STEMS = {"arxiv", "products", "proteins", "roman_empire", "pascalvoc_sp", "coco_sp", "peptides_func"}


@unittest.skipUnless(shutil.which("make"), "GNU Make is not available")
class MakeWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "source with spaces"
        self.project.mkdir()
        shutil.copyfile(PROJECT / "Makefile", self.project / "Makefile")
        self.capture = self.root / "arguments.jsonl"
        self.interpreter = self.root / "recording python"
        self.interpreter.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                                    "with open(os.environ['CM_MAKE_CAPTURE'], 'a') as stream:\n"
                                    "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n")
        self.interpreter.chmod(0o755)

    def invoke(self, target, **variables):
        command = ["make", "--no-print-directory", target, f"PYTHON={self.interpreter}"]
        command += [f"{key}={value}" for key, value in variables.items()]
        result = subprocess.run(command, cwd=self.project, capture_output=True, text=True,
                                env=dict(os.environ, CM_MAKE_CAPTURE=str(self.capture)), timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return [json.loads(line) for line in self.capture.read_text().splitlines()] if self.capture.exists() else []

    def test_default_help_never_starts_an_interpreter_or_training(self):
        self.assertEqual(self.invoke("help"), [])
        self.assertFalse(self.capture.exists())

    def test_reproduce_verifies_seven_examples_and_preserves_paths_with_spaces(self):
        runs = self.root / "bundled runs"
        inventories = self.root / "bound inventories"
        reports = self.root / "replayed reports"
        calls = self.invoke("reproduce", EXAMPLE_RUNS_ROOT=runs,
                            EXAMPLE_INVENTORY_ROOT=inventories, REPORT_ROOT=reports)
        self.assertEqual(calls[0], ["-B", "scripts/artifact_inventory.py", "--check-static"])
        self.assertEqual(len(calls), 9)
        for call, stem in zip(calls[1:8], ("arxiv", "products", "proteins", "roman_empire", "pascalvoc_sp", "coco_sp", "peptides_func")):
            self.assertEqual(call, ["-B", "scripts/artifact_inventory.py", str(runs / stem),
                                    "--output", str(inventories / f"{stem}.json"), "--check"])
        self.assertEqual(calls[-1], ["-B", "scripts/reproduce_examples.py", "--runs-root", str(runs),
                                     "--output-dir", str(reports), "--figures"])
        self.assertFalse(any("run_cross_morse" in arg for call in calls for arg in call))
        self.assertFalse(reports.exists())

    def test_registry_views_target_only_regenerates_registry_views_and_explanatory_diagrams(self):
        figures = self.root / "paper figures"
        diagrams = self.root / "method diagrams"
        calls = self.invoke("registry-views", REGISTRY_FIGURE_ROOT=figures, DIAGRAM_ROOT=diagrams)
        self.assertEqual([call[1] for call in calls], ["scripts/render_reported_results.py", "scripts/render_paper_results.py",
                         "scripts/plot_paper_summary.py", "scripts/plot_registry_figures.py", "scripts/draw_diagrams.py"])
        self.assertEqual(calls[2][-1], str(figures))
        self.assertEqual(calls[3][-1], str(figures / "registry"))
        self.assertEqual(calls[4][-1], str(diagrams))

    def test_paper_targets_build_from_run_archives_not_from_the_registry(self):
        index = self.root / "official runs" / "index.json"
        out = self.root / "paper output"
        calls = self.invoke("paper", RUNS_INDEX=index, PAPER_OFFICIAL_ROOT=out)
        self.assertEqual(calls, [["-B", "scripts/build_paper.py", "all", "--index", str(index), "--label", "official",
                                  "--output-dir", str(out)],
                                 ["-B", "scripts/records.py", "tables", "--output-dir", str(out / "records")]])
        self.capture.unlink()
        calls = self.invoke("tables", RUNS_INDEX=index, PAPER_OFFICIAL_ROOT=out)
        self.assertEqual(calls[0][1:3], ["scripts/build_paper.py", "tables"])
        self.capture.unlink()
        calls = self.invoke("example-paper-check", EXAMPLE_RUNS_ROOT=self.root / "ex runs", PAPER_EXAMPLES_ROOT=out)
        self.assertEqual(calls, [["-B", "scripts/build_paper.py", "all", "--runs-root", str(self.root / "ex runs"),
                                  "--label", "examples", "--output-dir", str(out), "--check"]])
        self.assertFalse(any("render_reported_results" in arg or "reported_results" in arg for call in calls for arg in call))

    def test_ingest_run_verifies_extracts_and_refreshes_manifests_and_availability(self):
        archive = self.root / "deposit archive.tar.gz"
        index = self.root / "runs index.json"
        calls = self.invoke("ingest-run", ARCHIVE=archive, DOI="10.5281/zenodo.1", RUNS_INDEX=index)
        self.assertEqual([call[1] for call in calls], ["scripts/ingest_run.py", "scripts/run_manifests.py", "scripts/availability.py"])
        self.assertEqual(calls[0], ["-B", "scripts/ingest_run.py", str(archive), "--doi", "10.5281/zenodo.1"])
        self.assertEqual(calls[1], ["-B", "scripts/run_manifests.py", "runs", "--index", str(index), "--label", "official", "--with-indices"])
        self.capture.unlink()
        result = subprocess.run(["make", "--no-print-directory", "ingest-run", f"PYTHON={self.interpreter}"],
                                cwd=self.project, capture_output=True, text=True,
                                env=dict(os.environ, CM_MAKE_CAPTURE=str(self.capture)), timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("usage: make ingest-run", result.stderr)

    def test_explicit_train_samples_routes_all_seven_to_external_cpu_work_paths(self):
        work = self.root / "external work"
        data = self.root / "fixture cache"
        calls = self.invoke("train-samples", WORK_ROOT=work, DATA_ROOT=data)
        self.assertEqual({Path(call[1]).stem.removeprefix("run_cross_morse_") for call in calls}, STEMS)
        self.assertEqual(len(calls), 7)
        for call in calls:
            stem = Path(call[1]).stem.removeprefix("run_cross_morse_")
            self.assertEqual(call[call.index("--profile") + 1], "smoke")
            self.assertEqual(call[call.index("--device") + 1], "cpu")
            self.assertEqual(call[call.index("--threads") + 1], "2")
            self.assertEqual(call[call.index("--data-root") + 1], str(data))
            self.assertEqual(call[call.index("--output-dir") + 1], str(work / "runs" / stem))
        self.assertFalse(work.exists())

    def test_official_preparation_routes_only_prepare_and_rejects_smoke(self):
        output = self.root / "official preparation"
        data = self.root / "official cache"
        calls = self.invoke("prepare-data", DATASET="Roman-empire", PROFILE="pilot", RUN_ROOT=output, DATA_ROOT=data)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ["-B", "run_cross_morse.py", "--dataset", "Roman-empire", "--profile", "pilot",
                                    "--device", "cpu", "--threads", "2", "--stage", "prepare",
                                    "--data-root", str(data), "--output-dir", str(output)])
        self.capture.unlink()
        result = subprocess.run(["make", "--no-print-directory", "prepare-data", "PROFILE=smoke",
                                 f"PYTHON={self.interpreter}"], cwd=self.project, capture_output=True, text=True,
                                env=dict(os.environ, CM_MAKE_CAPTURE=str(self.capture)), timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires PROFILE=pilot or PROFILE=full", result.stderr)
        self.assertFalse(self.capture.exists())


if __name__ == "__main__":
    unittest.main()
