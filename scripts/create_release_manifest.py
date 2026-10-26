#!/usr/bin/env python3
"""Create/verify complete source identity; adapted from BSD upstream, see NOTICE."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._release_common import MANIFEST, ReleaseError, normalized_root, verify_checksums, verify_manifest, write_integrity


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path, help="write RELEASE_MANIFEST.json and SHA256SUMS.txt")
    group.add_argument("--verify", type=Path, help="verify RELEASE_MANIFEST.json and SHA256SUMS.txt")
    parser.add_argument("--profile", choices=("private", "public"), default=None)
    args = parser.parse_args(argv)
    try:
        root = normalized_root(args.root)
        requested = args.output or args.verify
        target = requested if requested.is_absolute() else root / requested
        if target.resolve() != root / MANIFEST:
            raise ReleaseError(f"integrity files use the fixed name {MANIFEST} inside --root")
        if args.output:
            profile = args.profile or ("private" if (root / "configs/arrhenius.env.example").exists() else "public")
            data = write_integrity(root, profile)
            print(f"wrote {MANIFEST} and SHA256SUMS.txt: {len(data['files'])} source files; {profile} profile")
        errors = verify_manifest(root) + verify_checksums(root)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print("source manifest and complete checksums verified")
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
