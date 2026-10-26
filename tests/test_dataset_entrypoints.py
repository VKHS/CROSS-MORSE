"""Dataset routing and snapshot tests; no downloads, training or real Slurm jobs."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from run_cross_morse import (DATASET_CONFIG_STEMS, ROOT, create_launcher_snapshot,
                            create_roman_campaign, main, resolve_configuration,
                            verify_launcher_snapshot, verify_roman_campaign)


class DatasetRoutingTests(unittest.TestCase):
    def test_all_supplied_profiles_route_without_importing_torch(self):
        with patch.dict(sys.modules, {"torch": None}):
            for dataset in DATASET_CONFIG_STEMS:
                for profile in ("smoke", "pilot", "full"):
                    with self.subTest(dataset=dataset, profile=profile):
                        output = io.StringIO()
                        with contextlib.redirect_stdout(output):
                            result = main(["--dataset", dataset, "--profile", profile, "--print-config"])
                        self.assertEqual(result, 0)
                        config = json.loads(output.getvalue())
                        self.assertEqual(config["profile"], profile)
                        if profile == "smoke":
                            self.assertEqual(config["dataset"], "synthetic")
                            self.assertEqual(config["benchmark_dataset"], dataset)
                            self.assertIs(config["scientific_run"], False)
                            self.assertIs(config["data"]["download"], False)
                        else:
                            self.assertEqual(config["dataset"], dataset)

    def test_python_wrappers_select_their_own_dataset(self):
        for dataset, stem in DATASET_CONFIG_STEMS.items():
            with self.subTest(dataset=dataset):
                result = subprocess.run([sys.executable, "-B", str(ROOT / f"run_cross_morse_{stem}.py"),
                                         "--profile", "smoke", "--print-config"],
                                        text=True, capture_output=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["benchmark_dataset"], dataset)

    def test_config_without_dataset_flag_selects_declared_real_dataset(self):
        result = resolve_configuration(config_path=ROOT / "configs/proteins_pilot.json")
        self.assertEqual(result["dataset"], "ogbn-proteins")

    def test_dataset_profile_and_wrapper_mismatches_are_errors(self):
        cases = (
            dict(dataset="ogbn-products", config_path=ROOT / "configs/arxiv_full.json"),
            dict(dataset="ogbn-arxiv", profile="pilot", config_path=ROOT / "configs/arxiv_full.json"),
            dict(dataset="ogbn-products", default_dataset="ogbn-proteins", profile="smoke"),
            dict(dataset="Roman-empire", config_path=ROOT / "configs/products_smoke.json"),
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                resolve_configuration(**kwargs)

    def test_roman_split_override_is_effective_without_mutating_source_config(self):
        path = ROOT / "configs/roman_empire_full.json"
        original = path.read_bytes()
        for index in range(10):
            result = resolve_configuration(config_path=path, split_index=index)
            self.assertEqual(result["data"]["split_index"], index)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(resolve_configuration(config_path=path)["data"]["split_index"], 0)

    def test_invalid_or_inapplicable_split_is_rejected(self):
        for index in (-1, 10, True, "2", 0.5):
            with self.subTest(index=index), self.assertRaises(ValueError):
                resolve_configuration(dataset="Roman-empire", profile="smoke", split_index=index)
        with self.assertRaises(ValueError):
            resolve_configuration(dataset="ogbn-arxiv", profile="smoke", split_index=0)

    def test_fixtures_cannot_claim_benchmark_execution(self):
        source = resolve_configuration(dataset="PascalVOC-SP", profile="smoke")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for change in ({"scientific_run": True}, {"profile": "full"}, {"benchmark_dataset": "not-a-dataset"}):
                with self.subTest(change=change):
                    path.write_text(json.dumps(dict(source, **change)))
                    with self.assertRaises(ValueError):
                        resolve_configuration(config_path=path)

    def test_nonfinite_json_is_rejected_before_runtime_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            source = (ROOT / "configs/arxiv_smoke.json").read_text().replace('"seed": 314159', '"seed": NaN')
            path.write_text(source)
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                resolve_configuration(config_path=path)

    def test_independent_graph_profiles_use_graph_units_and_compatible_native_sites(self):
        for dataset in ("PascalVOC-SP", "COCO-SP", "Peptides-func"):
            for profile in ("smoke", "pilot", "full"):
                c = resolve_configuration(dataset=dataset, profile=profile)
                self.assertEqual(c["data"]["partitioner"], "graphs")
                self.assertNotIn("num_clusters", c["data"])
                self.assertEqual(c["admission"]["rule"], "empirical_bernstein")
                depth, layers = c["model"]["depth"], c["model"]["native_layers"]
                self.assertEqual((depth - (2 if c["model"]["backbone"] == "gps" else 0)) % layers, 0)
                if profile == "smoke":
                    self.assertGreaterEqual(c["data"]["synthetic_graphs"], 80)


class LauncherSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cm snapshot test ")
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name)
        self.source = self.parent / "source with spaces"
        self.source.mkdir()
        (self.source / "cross_morse").mkdir()
        (self.source / "configs").mkdir()
        for name in ("run_cross_morse.py", "run_cross_morse.sh", "run_cross_morse_roman_empire.py"):
            shutil.copyfile(ROOT / name, self.source / name)
        shutil.copyfile(ROOT / "cross_morse/__init__.py", self.source / "cross_morse/__init__.py")
        self.config = self.source / "configs/roman_empire_smoke.json"
        shutil.copyfile(ROOT / "configs/roman_empire_smoke.json", self.config)
        shutil.copyfile(ROOT / "configs/public.env.example", self.source / "configs/public.env.example")
        self.suite = self.parent / "new suite"

    def snapshot(self, split=4):
        return create_launcher_snapshot(self.source, self.suite, self.config, "smoke",
                                        self.parent / "cache with spaces", self.source / "run_cross_morse.sh",
                                        dataset="Roman-empire", split_index=split)

    def test_resolved_and_submitted_config_are_both_captured(self):
        self.snapshot()
        record = verify_launcher_snapshot(self.suite)
        self.assertEqual(record["split_index"], 4)
        self.assertEqual(json.loads((self.suite / "source/configs/resolved.json").read_text())["data"]["split_index"], 4)
        self.assertEqual((self.suite / "source/configs/submitted.json").read_bytes(), self.config.read_bytes())
        self.assertIn("CM_SPLIT_INDEX=4", (self.suite / "source/configs/active.env.example").read_text())

    def test_snapshot_refuses_source_descendant_and_ancestor(self):
        for target in (self.source, self.source / "nested output", self.parent):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "outside"):
                create_launcher_snapshot(self.source, target, self.config, "smoke", self.parent / "cache",
                                         self.source / "run_cross_morse.sh", dataset="Roman-empire")
        self.assertFalse((self.source / "nested output").exists())
        self.assertTrue(self.config.is_file())

    def test_snapshot_never_replaces_an_existing_snapshot(self):
        self.snapshot()
        before = (self.suite / "launcher/source_sha256.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.snapshot(5)
        self.assertEqual(before, (self.suite / "launcher/source_sha256.json").read_bytes())
        self.assertEqual(verify_launcher_snapshot(self.suite)["split_index"], 4)

    def test_changed_missing_and_extra_files_are_detected(self):
        self.snapshot()
        target = self.suite / "source/run_cross_morse.py"
        original = target.read_bytes()
        for mutation in ("changed", "missing", "extra"):
            with self.subTest(mutation=mutation):
                target.write_bytes(original)
                extra = self.suite / "source/unrecorded.py"
                extra.unlink(missing_ok=True)
                if mutation == "changed":
                    target.write_bytes(original + b"\n# changed\n")
                elif mutation == "missing":
                    target.unlink()
                else:
                    extra.write_text("# extra source\n")
                with self.assertRaisesRegex(ValueError, "Source snapshot changed"):
                    verify_launcher_snapshot(self.suite)

    def test_campaign_source_stays_fixed_when_original_changes(self):
        campaign = self.parent / "campaign"
        create_roman_campaign(self.source, campaign, self.config, "smoke", self.parent / "cache",
                              self.source / "run_cross_morse.sh")
        before = (campaign / "launcher/source_sha256.json").read_bytes()
        (self.source / "run_cross_morse.py").write_text("# later original-source edit\n")
        self.assertEqual(verify_roman_campaign(campaign)["split_indices"], list(range(10)))
        self.assertEqual((campaign / "launcher/source_sha256.json").read_bytes(), before)
        with self.assertRaises(FileExistsError):
            create_roman_campaign(self.source, campaign, self.config, "smoke", self.parent / "cache",
                                  self.source / "run_cross_morse.sh")
        self.assertEqual((campaign / "launcher/source_sha256.json").read_bytes(), before)

    def test_failed_campaign_creation_does_not_leave_a_partial_destination(self):
        campaign = self.parent / "failed campaign"
        self.config.write_text('{"dataset":"ogbn-products","profile":"full"}')
        with self.assertRaises(ValueError):
            create_roman_campaign(self.source, campaign, self.config, "smoke", self.parent / "cache",
                                  self.source / "run_cross_morse.sh")
        self.assertFalse(campaign.exists())
        self.assertFalse(list(self.parent.glob(".failed campaign.build_*")))


if __name__ == "__main__":
    unittest.main()
