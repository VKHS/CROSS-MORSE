#!/usr/bin/env python3
"""Check explicit manuscript-to-run declarations against sealed actual results.

A match is a consistency check, not proof of preregistration or chronology.
The template deliberately contains nulls until the author supplies the records.
"""
from pathlib import Path
import argparse
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_morse.review import NumericEvidence, EvidenceError, json_read


def check_binding(root, binding):
    if binding.get("schema_version") != "result-binding-1":
        raise EvidenceError("Unsupported result binding schema")
    if binding.get("status") != "author_completed":
        raise EvidenceError("Complete the author declaration first; a template is not evidence")
    if binding.get("relationship_to_manuscript") not in (
        "exact_result_producing_run", "refactor_with_prediction_equivalence", "post_result_revision"):
        raise EvidenceError("Declare the actual relationship between this run and the manuscript")
    if not isinstance(binding.get("manuscript_identifier"), str) or not binding["manuscript_identifier"].strip():
        raise EvidenceError("A manuscript version/identifier is required")
    e = NumericEvidence(root)
    for key in ("source_sha256", "config_sha256", "unit_sha256", "dataset"):
        if binding.get("run_identities", {}).get(key) != e.identities[key]:
            raise EvidenceError(f"Binding differs from the actual run: {key}")
    checks = binding.get("results", [])
    if not checks:
        raise EvidenceError("At least one explicit numerical result is required")
    results = []
    for entry in checks:
        if not isinstance(entry.get("manuscript_location"), str) or not entry["manuscript_location"]:
            raise EvidenceError("Every result needs a manuscript location")
        value = e.json(entry["artifact"])
        pointer = entry["json_pointer"]
        if not isinstance(pointer, str) or not pointer.startswith("/"):
            raise EvidenceError("Use an absolute JSON pointer such as /paired_qD_minus_qC_ce/mean")
        for segment in pointer[1:].split("/"):
            key = segment.replace("~1", "/").replace("~0", "~")
            value = value[int(key)] if isinstance(value, list) else value[key]
        expected, tolerance = entry.get("reported_value"), entry.get("absolute_tolerance")
        if (isinstance(expected, bool) or not isinstance(expected, (int, float)) or not math.isfinite(expected)
                or not isinstance(tolerance, (int,float)) or not math.isfinite(tolerance) or tolerance < 0):
            raise EvidenceError("Every reported value and nonnegative tolerance must be explicit and finite")
        if not isinstance(value, (int,float)) or not math.isfinite(value):
            raise EvidenceError("The bound field is not a finite numerical result")
        agrees = abs(value - expected) <= tolerance
        results.append(dict(entry, actual_value=value, agrees=agrees))
    return {"status": "MATCH" if all(x["agrees"] for x in results) else "MISMATCH", "results": results,
            "relationship_to_manuscript": binding["relationship_to_manuscript"],
            "scope": "author-declared mapping checked against actual sealed numbers; not verified historical provenance"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_root", type=Path); p.add_argument("binding", type=Path)
    args = p.parse_args(argv)
    try:
        result = check_binding(args.run_root, json_read(args.binding))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "MATCH" else 1
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        print(json.dumps({"status": "INCOMPLETE_OR_INVALID", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
