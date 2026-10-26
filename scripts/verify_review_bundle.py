#!/usr/bin/env python3
"""Verify an extracted review bundle's file inventory without loading models."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_review_bundle import verify_bundle


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle_root", type=Path)
    args = p.parse_args(argv)
    try:
        result = verify_bundle(args.bundle_root)
        print(json.dumps({"status": "PASS", "files_verified": len(result["files"]),
                          "dataset": result["dataset"], "scientific_run": result["scientific_run"],
                          "scope": "file integrity only; run reproduce_results.py to verify arithmetic"}, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
