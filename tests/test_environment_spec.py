"""CPU environment pins and no-write installer checks; no packages are installed."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


PROJECT = Path(__file__).resolve().parents[1]
SPEC = PROJECT / "environment"
module_spec = importlib.util.spec_from_file_location("cpu_environment_checker", SPEC / "check_environment.py")
checker = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(checker)


class EnvironmentSpecTests(unittest.TestCase):
    def test_lock_contains_exactly_the_observed_distribution_versions(self):
        metadata, pins = checker.load_spec(SPEC)
        self.assertEqual(len(pins), metadata["package_count"])
        self.assertEqual(pins["torch"], metadata["torch_distribution"])
        self.assertEqual(hashlib.sha256((SPEC / "requirements-cpu.lock").read_bytes()).hexdigest(),
                         metadata["requirements_sha256"])

    def test_changed_lock_is_rejected_even_if_its_json_hash_is_updated(self):
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "environment"
            shutil.copytree(SPEC, copied)
            lock = copied / "requirements-cpu.lock"
            lock.write_text(lock.read_text() + "unexpected-package==1.0\n")
            with self.assertRaisesRegex(ValueError, "lock bytes"):
                checker.load_spec(copied)
            metadata_path = copied / "observed_environment.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["requirements_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "distribution inventory"):
                checker.load_spec(copied)

    def test_checker_reports_missing_and_mismatched_pins_without_importing_torch(self):
        _, pins = checker.load_spec(SPEC)
        distributions = [SimpleNamespace(metadata={"Name": name}, version=version)
                         for name, version in pins.items() if name != "torch"]
        distributions[0].version = "0.invalid"
        stdout = io.StringIO()
        with mock.patch.object(checker, "platform_differences", return_value=[]), \
             mock.patch.object(checker.importlib.metadata, "distributions", return_value=distributions), \
             contextlib.redirect_stdout(stdout):
            self.assertEqual(checker.main([]), 1)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["status"], "ENVIRONMENT_DIFFERS")
        self.assertEqual(len(result["differences"]), 2)
        self.assertTrue(any("torch" in line and "missing" in line for line in result["differences"]))

    def test_installer_dry_run_preserves_spaces_and_rejects_existing_or_inside_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source with spaces"
            copied = source / "environment"
            shutil.copytree(SPEC, copied)
            metadata_path = copied / "observed_environment.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["python"].update(version=platform.python_version(), version_info=list(sys.version_info[:3]))
            metadata["platform"].update(system=platform.system(), machine=platform.machine())
            metadata_path.write_text(json.dumps(metadata))
            command = ["bash", str(copied / "create_cpu_env.sh"), "--python", sys.executable, "--dry-run", "--venv"]
            target = root / "new environment"
            result = subprocess.run(command + [str(target)], text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("DRY-RUN", result.stdout)
            self.assertIn("download.pytorch.org/whl/cpu", result.stdout)
            self.assertFalse(target.exists())
            existing = root / "existing environment"
            existing.mkdir()
            marker = existing / "preserve.txt"
            marker.write_text("preserve")
            for path in (existing, source / "nested environment", source):
                result = subprocess.run(command + [str(path)], text=True, capture_output=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
            self.assertEqual(marker.read_text(), "preserve")
            self.assertFalse((source / "nested environment").exists())


if __name__ == "__main__":
    unittest.main()
