#!/usr/bin/env python3
"""Build a validated private/public ZIP without changing source or prior output.

Archive mechanics follow our CROSS-MORSE BSD tooling; see NOTICE.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import zipfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._release_common import (MANIFEST, ReleaseError, files, inventory, normalized_root,
                                    public_excluded, sha256_file, write_integrity)
from scripts.release_audit import audit

ARCHIVE_ROOT = "CROSS-MORSE"


def validate_destination(source: Path, output: Path, *, replace: bool = False) -> Path:
    """Check both containment directions before touching any destination."""
    source = normalized_root(source)
    raw = output.expanduser()
    if raw.is_symlink():
        raise ReleaseError("output must not be a symlink")
    output = raw.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ReleaseError("output must be outside source and must not be an ancestor of source")
    if output.suffix.lower() != ".zip":
        raise ReleaseError("--output must name a ZIP file, not a directory")
    if output.exists() and not output.is_file():
        raise ReleaseError("existing output is not a regular file")
    if output.exists() and not replace:
        raise ReleaseError("output already exists; choose another path or use --replace")
    return output


def _assert_audit(root: Path, profile: str | None = None) -> None:
    findings = audit(root, profile=profile, require_integrity=True)
    if findings:
        lines = [f"{f.code}: {f.path}: {f.message}" for f in findings]
        raise ReleaseError("source release audit failed:\n" + "\n".join(lines))


def _write_zip(root: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files(root, integrity=True):
            rel = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{rel}", date_time=(2020, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o100755 if path.stat().st_mode & 0o111 else 0o100644
            info.external_attr = mode << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _verify_zip(root: Path, archive_path: Path) -> None:
    expected = {f"{ARCHIVE_ROOT}/{p.relative_to(root).as_posix()}": p for p in files(root, integrity=True)}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ReleaseError("ZIP contains duplicate, missing or unexpected members")
        if archive.testzip() is not None:
            raise ReleaseError("ZIP CRC verification failed")
        for name, path in expected.items():
            info = archive.getinfo(name)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ReleaseError(f"ZIP member is not a regular file: {name}")
            if bool(mode & 0o111) != bool(path.stat().st_mode & 0o111):
                raise ReleaseError(f"ZIP executable mode changed: {name}")
            if hashlib.sha256(archive.read(name)).hexdigest() != sha256_file(path):
                raise ReleaseError(f"ZIP file bytes changed: {name}")


def build_release(source: Path, output: Path, *, profile: str = "private", replace: bool = False) -> dict:
    source = normalized_root(source)
    output = validate_destination(source, output, replace=replace)
    if profile not in {"private", "public"}:
        raise ReleaseError(f"unknown profile: {profile}")
    _assert_audit(source)
    before = inventory(source)
    identities = {item["path"]: item for item in before["files"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    # A temporary sibling ensures the final file promotion stays on one
    # filesystem. Only this newly created directory is removed on failure.
    with tempfile.TemporaryDirectory(prefix=".cross-morse-release-", dir=output.parent) as temporary:
        stage = Path(temporary) / ARCHIVE_ROOT
        stage.mkdir()
        retained = []
        for path in files(source):
            rel = path.relative_to(source)
            if profile == "public" and public_excluded(rel):
                continue
            target = stage / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if sha256_file(target) != identities[rel.as_posix()]["sha256"]:
                raise ReleaseError(f"source changed during copying: {rel}")
            retained.append(rel.as_posix())
        if inventory(source) != before:
            raise ReleaseError("source changed while the release was being built")
        manifest = write_integrity(stage, profile)
        _assert_audit(stage, profile)
        archive_path = Path(temporary) / "validated-release.zip"
        _write_zip(stage, archive_path)
        _verify_zip(stage, archive_path)
        with archive_path.open("rb") as handle:
            os.fsync(handle.fileno())
        if replace:
            # This replaces a file atomically, after every validation succeeds.
            # No recursive deletion of the output or input exists in this tool.
            if output.is_symlink() or (output.exists() and not output.is_file()):
                raise ReleaseError("output changed type during release creation")
            os.replace(archive_path, output)
        else:
            # Atomic no-clobber creation, including an output created by another
            # process after our initial check. Both files are on one filesystem.
            try:
                os.link(archive_path, output)
            except FileExistsError as exc:
                raise ReleaseError("output appeared during the build; existing output was preserved") from exc
        return {"output": str(output), "profile": profile, "archive_root": ARCHIVE_ROOT,
                "sha256": sha256_file(output), "source_tree_sha256": before["tree_sha256"],
                "export_tree_sha256": manifest["tree_sha256"], "retained_source_files": len(retained),
                "computational_source_rewritten": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("private", "public"), default="private")
    parser.add_argument("--replace", action="store_true", help="atomically replace an existing ZIP only after validation")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build_release(args.source, args.output, profile=args.profile, replace=args.replace), indent=2))
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
