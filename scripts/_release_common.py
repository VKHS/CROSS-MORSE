"""Deterministic release inventory and integrity helpers.

Our CROSS-MORSE BSD release utilities; see NOTICE.
No functions in this module execute scientific code or submit cluster jobs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

MANIFEST = "RELEASE_MANIFEST.json"
CHECKSUMS = "SHA256SUMS.txt"
INTEGRITY_FILES = {MANIFEST, CHECKSUMS}
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv"}
SKIP_ROOTS = {"data", "datasets", "outputs", "runs", "cache", "logs", "checkpoints", "trajectories", "analysis", "upload", "build", "dist", "exports", "reports"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".tmp", ".part", ".swp", ".bak"}
GENERATED_SUFFIXES = {".pt", ".pth", ".ckpt", ".npz", ".npy", ".pkl", ".pickle", ".safetensors", ".parquet", ".zip", ".tar", ".tgz", ".7z"}
PUBLIC_EXCLUDES = {"configs/arrhenius.env.example", "configs/active.env.example"}
# Static declarations kept in a release although their directories hold untracked payloads.
STATIC_DATA_FILES = {"data/README.md", "data/sources.json", "runs/README.md", "runs/index.json"}
EXAMPLE_DATASETS = {"arxiv": "ogbn-arxiv", "products": "ogbn-products", "proteins": "ogbn-proteins",
                    "roman_empire": "Roman-empire", "pascalvoc_sp": "PascalVOC-SP",
                    "coco_sp": "COCO-SP", "peptides_func": "Peptides-func"}
HASH = re.compile(r"^[0-9a-f]{64}$")


class ReleaseError(ValueError):
    """A source tree or requested export cannot be validated safely."""


def normalized_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ReleaseError(f"source directory does not exist: {root}")
    return root


def excluded(rel: Path) -> bool:
    generated_binary = (rel.suffix in GENERATED_SUFFIXES or rel.name.endswith(".tar.gz"))
    # Data-source declarations are source files; actual dataset caches stay out.
    static_data = rel.as_posix() in {"data", "runs"} or rel.as_posix() in STATIC_DATA_FILES
    # Only explicitly named synthetic examples may ship generated artifacts.
    # release_audit checks their task/configuration and complete sealed chain.
    fixture = rel.parts[:2] == ("tests", "fixtures")
    example = (len(rel.parts) >= 4 and rel.parts[:2] == ("examples", "runs")
               and rel.parts[2] in EXAMPLE_DATASETS)
    return (any(part in SKIP_PARTS or part.endswith((".egg-info", ".dist-info")) for part in rel.parts)
            or bool(rel.parts and rel.parts[0] in SKIP_ROOTS and not static_data)
            or rel.suffix in SKIP_SUFFIXES
            or (generated_binary and not fixture and not example)
            or rel.name == ".env" or rel.name.endswith(".env"))


def public_excluded(rel: Path) -> bool:
    return rel.as_posix() in PUBLIC_EXCLUDES or rel.parts[:2] == ("docs", "private")


def files(root: Path, *, integrity: bool = False) -> list[Path]:
    """Inventory every distributable file, rejecting symlinks and odd paths."""
    result: list[Path] = []
    for folder, directories, names in os.walk(root, followlinks=False):
        directories.sort()
        kept = []
        for name in directories:
            path = Path(folder) / name
            rel = path.relative_to(root)
            if excluded(rel):
                continue
            if path.is_symlink():
                raise ReleaseError(f"symlink directory is not permitted in a release: {rel}")
            kept.append(name)
        directories[:] = kept
        for name in sorted(names):
            path = Path(folder) / name
            rel = path.relative_to(root)
            if excluded(rel) or (not integrity and rel.as_posix() in INTEGRITY_FILES):
                continue
            if path.is_symlink() or not path.is_file():
                raise ReleaseError(f"non-regular release file: {rel}")
            if any(c in rel.as_posix() for c in "\r\n\0"):
                raise ReleaseError(f"unsupported newline/NUL in release path: {rel!s}")
            result.append(path)
    return sorted(result, key=lambda path: path.relative_to(root).as_posix())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (not path.is_absolute() and all(part not in {".", ".."} for part in path.parts)
            and path.as_posix() == value and not any(c in value for c in "\r\n\0"))


def inventory(root: Path) -> dict:
    entries = []
    digest = hashlib.sha256()
    for path in files(root):
        rel = path.relative_to(root).as_posix()
        item = {"path": rel, "sha256": sha256_file(path), "bytes": path.stat().st_size,
                "executable": bool(path.stat().st_mode & 0o111)}
        entries.append(item)
        digest.update(json.dumps(item, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
    return {"schema_version": "2.0", "hash_algorithm": "sha256", "tree_sha256": digest.hexdigest(), "files": entries}


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".release-write-", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temp.chmod(0o644)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_integrity(root: Path, profile: str = "private") -> dict:
    if profile not in {"private", "public"}:
        raise ReleaseError(f"unknown release profile: {profile}")
    data = dict(inventory(root), release_profile=profile)
    atomic_text(root / MANIFEST, json.dumps(data, indent=2, sort_keys=True) + "\n")
    # No circular hashes: the manifest excludes both integrity files; this list
    # includes the manifest and excludes only itself.
    records = [(path.relative_to(root).as_posix(), sha256_file(path))
               for path in files(root, integrity=True) if path.relative_to(root).as_posix() != CHECKSUMS]
    atomic_text(root / CHECKSUMS, "".join(f"{digest}  {rel}\n" for rel, digest in records))
    return data


def verify_manifest(root: Path) -> list[str]:
    path = root / MANIFEST
    if not path.is_file() or path.is_symlink():
        return [f"missing regular {MANIFEST}"]
    try:
        expected = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(expected, dict) or expected.get("schema_version") != "2.0":
            return ["invalid release manifest schema"]
        if expected.get("release_profile") not in {"private", "public"}:
            return ["invalid release manifest profile"]
        records = expected.get("files")
        if not isinstance(records, list):
            return ["release manifest files must be a list"]
        old = {}
        for record in records:
            if (not isinstance(record, dict) or not safe_relative(record.get("path"))
                    or not isinstance(record.get("sha256"), str) or not HASH.fullmatch(record["sha256"])
                    or type(record.get("bytes")) is not int or record["bytes"] < 0
                    or type(record.get("executable")) is not bool):
                return ["invalid release manifest file entry"]
            if record["path"] in old:
                return [f"duplicate release manifest path: {record['path']}"]
            old[record["path"]] = record
        actual = inventory(root)
        new = {record["path"]: record for record in actual["files"]}
        errors = []
        if expected.get("tree_sha256") != actual["tree_sha256"]:
            errors.append("release tree_sha256 mismatch")
        for rel in sorted(set(old) | set(new)):
            if rel not in old:
                errors.append(f"unexpected source file: {rel}")
            elif rel not in new:
                errors.append(f"missing source file: {rel}")
            elif old[rel] != new[rel]:
                errors.append(f"source identity mismatch: {rel}")
        return errors
    except (OSError, ValueError, TypeError) as exc:
        return [f"cannot verify release manifest: {exc}"]


def verify_checksums(root: Path) -> list[str]:
    path = root / CHECKSUMS
    if not path.is_file() or path.is_symlink():
        return [f"missing regular {CHECKSUMS}"]
    try:
        expected = {}
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
            if not match or not safe_relative(match[2]) or match[2] == CHECKSUMS:
                return [f"invalid checksum record at line {number}"]
            if match[2] in expected:
                return [f"duplicate checksum path: {match[2]}"]
            expected[match[2]] = match[1]
        actual = {p.relative_to(root).as_posix(): sha256_file(p) for p in files(root, integrity=True)
                  if p.relative_to(root).as_posix() != CHECKSUMS}
        errors = []
        for rel in sorted(set(expected) | set(actual)):
            if rel not in expected:
                errors.append(f"unexpected checksummed file: {rel}")
            elif rel not in actual:
                errors.append(f"missing checksummed file: {rel}")
            elif expected[rel] != actual[rel]:
                errors.append(f"checksum mismatch: {rel}")
        return errors
    except (OSError, ValueError, TypeError) as exc:
        return [f"cannot verify checksums: {exc}"]
