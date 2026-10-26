"""Launcher tests use a mock sbatch and never contact a cluster."""
from __future__ import annotations
import os
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cm launcher test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "source with spaces"
        self.project.mkdir()
        (self.project / "configs").mkdir()
        self.shell = self.project / "run_cross_morse.sh"
        for original in PROJECT.glob("run_cross_morse*.py"):
            shutil.copyfile(original, self.project / original.name)
        for original in PROJECT.glob("run_cross_morse*.sh"):
            shutil.copyfile(original, self.project / original.name)
        (self.project / "cross_morse").mkdir()
        shutil.copyfile(PROJECT / "cross_morse/__init__.py", self.project / "cross_morse/__init__.py")
        for original in (PROJECT / "configs").iterdir():
            if original.is_file():
                shutil.copyfile(original, self.project / "configs" / original.name)
        self.capture = self.root / "captured arguments.bin"
        mock_bin = self.root / "mock commands"
        mock_bin.mkdir()
        stub = mock_bin / "sbatch"
        stub.write_text('#!/bin/bash\nprintf "%s\\0" "$@" > "$CM_MOCK_CAPTURE"\nprintf "MOCK ONLY\\n"\n')
        stub.chmod(0o755)
        self.base = self.root / "storage with spaces"
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("CM_", "SLURM_"))}
        self.env.update(CM_PROJECT_DIR=str(self.project), CM_BASE=str(self.base), CM_MOCK_CAPTURE=str(self.capture),
                        PATH=str(mock_bin) + os.pathsep + self.env.get("PATH", ""))

    def run_shell(self, *args, env=None):
        return subprocess.run(["bash", str(self.shell), *args], env=env or self.env, text=True,
                              capture_output=True, timeout=30, cwd=self.project)

    def recorded(self):
        return self.capture.read_bytes().decode().rstrip("\0").split("\0")

    def test_syntax_help_and_account_directive(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(self.shell)], capture_output=True).returncode, 0)
        result = self.run_shell("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CM_ENV_FILE", result.stdout)
        self.assertIsNone(re.search(r"^#SBATCH\s+(?:--account|-A)", self.shell.read_text(), re.M))

    def test_dry_run_preserves_spaces_and_performs_no_writes(self):
        result = self.run_shell("dry-run", "pilot")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"base={self.base}\n", result.stdout)
        self.assertIn(f"project={self.project}\n", result.stdout)
        self.assertIn("No directories, packages or jobs were created.", result.stdout)
        self.assertFalse(self.base.exists())
        self.assertFalse(self.capture.exists())

    def test_exported_overrides_win_over_environment_file_including_empty_values(self):
        alternate = self.root / "alternate profile with spaces.env"
        alternate.write_text("CM_ACCOUNT=file-account\nCM_THREADS=3\nCM_BUILD_MODULE=file-module\n"
                             + f"CM_BASE={shlex.quote(str(self.root / 'file base'))}\n")
        env = dict(self.env, CM_ENV_FILE=str(alternate), CM_ACCOUNT="exported-account", CM_THREADS="6", CM_BUILD_MODULE="")
        result = self.run_shell("dry-run", "pilot", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("account=exported-account\n", result.stdout)
        self.assertIn(f"base={self.base}\n", result.stdout)
        self.assertIn("--threads 6", result.stdout)
        self.assertNotIn("file-module", result.stdout)
        self.assertFalse(self.base.exists())

    def test_mock_submissions_route_profiles_and_preserve_quoted_arguments(self):
        for requested, mode, time in (((), "pilot", "04:00:00"), (("pilot",), "pilot", "04:00:00"),
                                      (("full",), "full", "24:00:00"), (("setup",), "setup", "01:00:00")):
            with self.subTest(mode=requested):
                result = self.run_shell("submit", *requested, env=dict(self.env, CM_ACCOUNT="fixture-account"))
                self.assertEqual(result.returncode, 0, result.stderr)
                arguments = self.recorded()
                self.assertIn(str(self.shell), arguments)
                self.assertEqual(arguments[-1], mode)
                self.assertEqual(arguments[arguments.index("--dataset") + 1], "ogbn-arxiv")
                self.assertIn("--account=fixture-account", arguments)
                self.assertIn("--cpus-per-task=16", arguments)
                self.assertIn("--time=" + time, arguments)
                self.assertIn(f"--output={self.base}/logs/CROSS_MORSE_ARXIV_%j.out", arguments)

    def test_private_defaults_are_loaded_when_profile_is_present(self):
        private = self.project / "configs/arrhenius.env.example"
        if not private.exists():
            self.skipTest("public export intentionally omits the private profile")
        expected = re.search(r"CM_ACCOUNT:=([^}]+)", private.read_text()).group(1)
        result = self.run_shell("submit", "pilot")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--account=" + expected, self.recorded())

    def test_public_fallback_requires_explicit_account_before_submission(self):
        (self.project / "configs/arrhenius.env.example").unlink(missing_ok=True)
        result = self.run_shell("submit", "pilot")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Set CM_ACCOUNT", result.stderr)
        self.assertFalse(self.capture.exists())
        self.assertFalse(self.base.exists())
        result = self.run_shell("submit", "pilot", env=dict(self.env, CM_ACCOUNT="public-allocation"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--account=public-allocation", self.recorded())

    def test_effective_environment_is_in_hashed_snapshot_and_reused(self):
        from run_cross_morse import create_launcher_snapshot, verify_launcher_snapshot
        from unittest.mock import patch
        suite = self.root / "suite with spaces"
        config = self.project / "configs/arxiv_pilot.json"
        env = dict(self.env, CM_ACCOUNT="captured-allocation", CM_THREADS="5", CM_BUILD_MODULE="",
                   CM_PYTHON_MODULE="", CM_EXPECT_ARCH="", CM_DATA_ROOT=str(self.base / "data"),
                   CM_VENV_DIR=str(self.base / "venv"))
        with patch.dict(os.environ, env, clear=True):
            create_launcher_snapshot(self.project, suite, config, "pilot", self.base / "data",
                                     self.shell, dataset="ogbn-arxiv")
        record = verify_launcher_snapshot(suite)
        self.assertEqual(record["dataset"], "ogbn-arxiv")
        active = suite / "source/configs/active.env.example"
        resume_env = {k: v for k, v in self.env.items() if k != "CM_PROJECT_DIR"}
        result = subprocess.run(["bash", str(suite / "source" / self.shell.name), "dry-run", "pilot"],
                                env=resume_env, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("account=captured-allocation\n", result.stdout)
        self.assertIn("--threads 5", result.stdout)
        self.assertIn(f"env_file={active}\n", result.stdout)

    def test_all_dataset_wrappers_and_profiles_route_without_writes(self):
        from run_cross_morse import DATASET_CONFIG_STEMS
        for dataset, stem in DATASET_CONFIG_STEMS.items():
            for profile in ("smoke", "pilot", "full"):
                with self.subTest(dataset=dataset, profile=profile):
                    wrapper = self.project / f"run_cross_morse_{stem}.sh"
                    result = subprocess.run(["bash", str(wrapper), "dry-run", profile], env=self.env,
                                            text=True, capture_output=True, timeout=30, cwd=self.project)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f"dataset={dataset}\n", result.stdout)
                    self.assertIn(f"configs/{stem}_{profile}.json", result.stdout)
        self.assertFalse(self.base.exists())
        self.assertFalse(self.capture.exists())

    def test_dataset_and_split_errors_precede_submission(self):
        env = dict(self.env, CM_ACCOUNT="fixture-account")
        cases = [
            ("--dataset", "bad-name", "submit", "pilot"),
            ("--split-index", "1", "submit", "pilot"),
            ("--dataset", "Roman-empire", "--split-index", "10", "submit", "pilot"),
            ("--dataset", "Roman-empire", "--split-index", "01", "submit", "pilot"),
            ("--dataset", "ogbn-products", "--config", str(self.project / "configs/arxiv_full.json"), "submit", "full"),
        ]
        for args in cases:
            with self.subTest(args=args):
                result = self.run_shell(*args, env=env)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.capture.exists())
                self.assertFalse(self.base.exists())

    def test_wrapper_refuses_conflicting_explicit_dataset(self):
        wrapper = self.project / "run_cross_morse_products.sh"
        result = subprocess.run(["bash", str(wrapper), "--dataset", "ogbn-arxiv", "dry-run", "pilot"],
                                env=self.env, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Conflicting", result.stderr)
        self.assertFalse(self.base.exists())

    def test_roman_split_and_custom_cache_are_quoted_in_mock_submission(self):
        cache = self.root / "custom cache with spaces"
        result = self.run_shell("submit", "pilot", "--dataset", "Roman-empire", "--split-index", "7",
                                "--data-root", str(cache), env=dict(self.env, CM_ACCOUNT="fixture-account"))
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.recorded()
        self.assertEqual(args[args.index("--split-index") + 1], "7")
        self.assertEqual(args[args.index("--data-root") + 1], str(cache))
        self.assertEqual(args[args.index("--dataset") + 1], "Roman-empire")

    def test_custom_roman_config_split_is_not_silently_reset_to_zero(self):
        config = json.loads((self.project / "configs/roman_empire_pilot.json").read_text())
        config["data"]["split_index"] = 6
        custom = self.root / "custom Roman split.json"
        custom.write_text(json.dumps(config))
        result = self.run_shell("--dataset", "Roman-empire", "--config", str(custom), "submit", "pilot",
                                env=dict(self.env, CM_ACCOUNT="fixture-account"))
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.recorded()
        self.assertEqual(args[args.index("--split-index") + 1], "6")

    def test_campaign_submission_freezes_source_and_resume_uses_same_campaign(self):
        from run_cross_morse import verify_roman_campaign
        campaign = self.root / "Roman campaign with spaces"
        env = dict(self.env, CM_ACCOUNT="fixture-account")
        result = self.run_shell("--dataset", "Roman-empire", "submit", "campaign", "smoke", str(campaign), env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = verify_roman_campaign(campaign)
        self.assertEqual(record["split_indices"], list(range(10)))
        args = self.recorded()
        self.assertIn("--array=0-9", args)
        self.assertIn(str(campaign / "source/run_cross_morse.sh"), args)
        self.assertEqual(args[-3:], ["campaign", "smoke", str(campaign)])
        snapshot = (campaign / "launcher/source_sha256.json").read_bytes()
        active = (campaign / "source/configs/active.env.example").read_text()
        self.assertIn("CM_SPLIT_INDEX=''", active)
        resumed_env = dict(env, CM_PROJECT_DIR=str(campaign / "source"),
                           CM_ENV_FILE=str(campaign / "source/configs/active.env.example"))
        result = self.run_shell("--dataset", "Roman-empire", "submit", "campaign", "resume", str(campaign), env=resumed_env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(snapshot, (campaign / "launcher/source_sha256.json").read_bytes())
        self.assertIn("--array=0-9", self.recorded())

    def test_campaign_refuses_other_datasets_and_single_split_override(self):
        for args in (("--dataset", "ogbn-products"),
                     ("--dataset", "Roman-empire", "--split-index", "2")):
            with self.subTest(args=args):
                campaign = self.root / "must not exist"
                result = self.run_shell(*args, "submit", "campaign", "full", str(campaign),
                                        env=dict(self.env, CM_ACCOUNT="fixture-account"))
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(campaign.exists())
                self.assertFalse(self.capture.exists())


if __name__ == "__main__":
    unittest.main()
