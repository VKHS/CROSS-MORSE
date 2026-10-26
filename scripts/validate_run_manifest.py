#!/usr/bin/env python3
"""Verify a real run's sealed artifact/parent chain, not declarations alone."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cross_morse.provenance import ProvenanceError, STAGES, verify_stage_chain


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(
        "Verify actual stage artifacts, original source/config/splits, frozen model states, "
        "and the complete hash-linked parent chain. Run from the original source snapshot."))
    parser.add_argument("run_root", type=Path, help="Run directory containing config.json and provenance/stages")
    parser.add_argument("--through", choices=STAGES, help="Require and verify through this executable stage")
    parser.add_argument("--require-complete", action="store_true", help="Require a sealed final evaluation")
    parser.add_argument("--json", action="store_true", help="Print machine-readable validation results")
    args = parser.parse_args(argv)
    if args.require_complete and args.through not in (None, "evaluate"):
        parser.error("--require-complete cannot be combined with an earlier --through stage")
    try:
        manifests = verify_stage_chain(args.run_root, "evaluate" if args.require_complete else args.through)
        result = {"valid": True, "run_root": str(args.run_root.resolve()),
                  "stages": [item["stage"] for item in manifests],
                  "complete": len(manifests) == len(STAGES),
                  "scope": "actual file/model/parent integrity; not scientific-result reproduction or a security sandbox"}
        if not manifests:
            result["note"] = "No sealed stages yet; empty partial run, not a completed scientific execution."
        print(json.dumps(result, indent=2) if args.json else
              f"Verified {len(manifests)}/{len(STAGES)} sealed stages; complete={result['complete']}")
        return 0
    except (ProvenanceError, OSError, ValueError, KeyError) as exc:
        result = {"valid": False, "error": str(exc)}
        print(json.dumps(result, indent=2) if args.json else f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
