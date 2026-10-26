"""Source identity checks read an explicit validation record without runtime imports."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts._paper_common import runtime_source_fingerprint


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "source_fingerprint.py"


class SourceFingerprintCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "source with spaces"
        (self.root / "cross_morse").mkdir(parents=True)
        (self.root / "run_cross_morse.py").write_text("print('fixture')\n")
        (self.root / "cross_morse" / "__init__.py").write_text('"""Fixture."""\n')
        self.identity = runtime_source_fingerprint(self.root)
        self.record = Path(self.temporary.name) / "validation record.json"

    def invoke(self, *arguments):
        return subprocess.run([sys.executable, "-B", str(SCRIPT), "--root", str(self.root),
                               *map(str, arguments)], capture_output=True, text=True)

    def test_record_matches_current_source_without_modifying_it(self):
        self.record.write_text(json.dumps({"source_sha256": self.identity}))
        before = self.record.read_bytes()
        result = self.invoke("--validation-record", self.record)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), self.identity)
        self.assertEqual(self.record.read_bytes(), before)

    def test_changed_source_does_not_pass_record_check(self):
        self.record.write_text(json.dumps({"source_sha256": self.identity}))
        (self.root / "cross_morse" / "__init__.py").write_text('"""Changed fixture."""\n')
        result = self.invoke("--validation-record", self.record)
        self.assertEqual(result.returncode, 1)
        self.assertIn("MISMATCH", result.stderr)

    def test_missing_or_invalid_record_fails_without_silent_comparison_skip(self):
        for value in ({}, {"source_sha256": ""}, {"source_sha256": None},
                      {"source_sha256": "x" * 64}, []):
            with self.subTest(value=value):
                self.record.write_text(json.dumps(value))
                result = self.invoke("--validation-record", self.record)
                self.assertEqual(result.returncode, 2)
                self.assertIn("ERROR", result.stderr)
        self.record.unlink()
        self.assertEqual(self.invoke("--validation-record", self.record).returncode, 2)

    def test_explicit_identity_and_record_are_mutually_exclusive(self):
        self.record.write_text(json.dumps({"source_sha256": self.identity}))
        self.assertEqual(self.invoke("--expect", self.identity).returncode, 0)
        result = self.invoke("--expect", self.identity, "--validation-record", self.record)
        self.assertEqual(result.returncode, 2)
        self.assertIn("not allowed with argument", result.stderr)


if __name__ == "__main__":
    unittest.main()
