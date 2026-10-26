#!/usr/bin/env python3
"""Recompute sealed numerical results and lifecycle decisions without training."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_morse.review import EvidenceError, replay_results, evidence_markdown


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_root", type=Path)
    p.add_argument("--output", type=Path, help="New JSON report outside the immutable run")
    p.add_argument("--skip-selection-replay", action="store_true", help="Omit costly nested bootstrap; clearly marked in report")
    args = p.parse_args(argv)
    try:
        if args.output and (args.output.resolve().is_relative_to(args.run_root.resolve()) or args.output.exists()
                            or args.output.with_suffix(".md").exists()):
            raise EvidenceError("Report destination must be new and outside the immutable run")
        result = replay_results(args.run_root, selection=not args.skip_selection_replay)
        text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as f:
                f.write(text)
            with args.output.with_suffix(".md").open("x") as f:
                f.write(evidence_markdown(result))
        print(text)
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
