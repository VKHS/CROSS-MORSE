#!/usr/bin/env python3
"""Print the runtime source identity recorded in every run and in the validation record.

The identity is the one computed by ``cross_morse.pipeline.source_fingerprint``:
sha256 of the canonical JSON of {relative path: sha256(bytes)} over
``run_cross_morse*.py`` and ``cross_morse/*.py``. Documentation, configs, tests and
release metadata are covered separately by ``RELEASE_MANIFEST.json``.

Use ``--expect <sha256>`` to assert a particular identity, for example the one in
an archived run's ``run_state.json``. Use ``--validation-record PATH`` to compare
with the ``source_sha256`` field of a machine-readable validation record.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._paper_common import ROOT, runtime_source_fingerprint


def _expected_identity(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value.strip()) is None:
        raise ValueError("Expected source identity must be a 64-character SHA-256 hex string")
    return value.strip().lower()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    expected = parser.add_mutually_exclusive_group()
    expected.add_argument("--expect", help="sha256 to compare against; exit 1 on mismatch")
    expected.add_argument("--validation-record", type=Path,
                          help="JSON record containing the expected source_sha256")
    args = parser.parse_args(argv)
    try:
        expected_identity = None
        if args.validation_record is not None:
            record = json.loads(args.validation_record.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or "source_sha256" not in record:
                raise ValueError("Validation record must contain source_sha256")
            expected_identity = _expected_identity(record["source_sha256"])
        elif args.expect is not None:
            expected_identity = _expected_identity(args.expect)
        identity = runtime_source_fingerprint(args.root.resolve())
        print(identity)
        if expected_identity is not None and expected_identity != identity:
            print(f"MISMATCH: expected {expected_identity}", file=sys.stderr)
            return 1
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
