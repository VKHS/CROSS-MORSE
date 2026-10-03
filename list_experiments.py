#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="List CROSS-MORSE implementation work packages.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true", help="emit the registry as JSON")
    args = parser.parse_args()

    path = args.root / "release" / "experiment_registry.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0

    for entry in data["entries"]:
        locations = "; ".join(entry["paper_locations"])
        print(f"{entry['id']:8s}  {entry['name']}  [{locations}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
