#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv"}
EXCLUDED_FILES = {"RELEASE_MANIFEST.json", "UPGRADE_REPORT.json"}


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in rel.parts):
            continue
        if rel.as_posix() in EXCLUDED_FILES:
            continue
        yield path


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build(root: Path) -> dict:
    files = []
    tree_digest = hashlib.sha256()
    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        digest = hash_file(path)
        size = path.stat().st_size
        files.append({"path": rel, "sha256": digest, "bytes": size})
        tree_digest.update(rel.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(digest.encode("ascii"))
        tree_digest.update(b"\n")
    return {"schema_version": "1.0", "tree_sha256": tree_digest.hexdigest(), "files": files}


def verify(root: Path, manifest_path: Path) -> list[str]:
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = build(root)
    errors: list[str] = []
    if expected.get("tree_sha256") != actual.get("tree_sha256"):
        errors.append("tree_sha256 mismatch")
    expected_files = {x["path"]: x for x in expected.get("files", [])}
    actual_files = {x["path"]: x for x in actual.get("files", [])}
    for path in sorted(set(expected_files) | set(actual_files)):
        if path not in expected_files:
            errors.append(f"unexpected file: {path}")
        elif path not in actual_files:
            errors.append(f"missing file: {path}")
        elif expected_files[path]["sha256"] != actual_files[path]["sha256"]:
            errors.append(f"hash mismatch: {path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or verify a deterministic source-tree manifest.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()

    if args.output:
        manifest = build(root)
        output = args.output if args.output.is_absolute() else root / args.output
        output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output} ({len(manifest['files'])} files; tree {manifest['tree_sha256']})")
        return 0

    manifest_path = args.verify if args.verify.is_absolute() else root / args.verify
    errors = verify(root, manifest_path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("release manifest verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
