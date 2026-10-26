#!/usr/bin/env python3
"""Pack a completed run (or a Roman-empire campaign) into a verifiable archive for deposit.

    python -B scripts/archive_run.py /abs/path/to/suite --output ../deposits/arxiv_full.tar.gz
    python -B scripts/archive_run.py /abs/path/to/campaign --campaign --output ../deposits/roman_empire_full.tar.gz
    python -B scripts/archive_run.py --verify ../deposits/arxiv_full.tar.gz

The archive contains the complete run directory (prepared data and splits, checkpoints,
exploration trajectories, candidate predictions, decision records, final predictions, stage
manifests, environment record) plus ``ARCHIVE_MANIFEST.json``: one SHA-256 per file, the run's
source/config/unit identities, dataset, profile and the stage-chain verification result.

Before packing, the stage chain is checked: every artifact named in a sealed stage manifest must
exist with the recorded hash and size, inputs must be earlier sealed outputs, and the chain must
run from ``prepare`` to ``evaluate``.  The run's own source identity is recorded; it is *not*
required to equal the identity of the checkout that packs the archive, because archives are
typically packed on the cluster after the code has moved on.  ``scripts/ingest_run.py`` records
both identities so that the relation is visible.

Standard library only.  Nothing is unpickled.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path

STAGES = ("prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate")
SCHEMA = "cross-morse-run-archive-1"
MANIFEST_NAME = "ARCHIVE_MANIFEST.json"


class ArchiveError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ArchiveError(f"missing {path}") from None
    except ValueError as exc:
        raise ArchiveError(f"invalid JSON {path}: {exc}") from None


def verify_stage_chain(run_root: Path) -> dict:
    """Lightweight chain check that does not require the packing checkout's source identity."""
    stages_dir = run_root / "provenance" / "stages"
    sealed: dict[str, dict] = {}
    identities = None
    parent = None
    present = []
    for i, stage in enumerate(STAGES):
        path = stages_dir / f"{i:02d}_{stage}.json"
        if not path.is_file():
            break
        manifest = _json(path)
        current = (manifest.get("source_sha256"), manifest.get("config_sha256"), manifest.get("unit_sha256"))
        if identities is not None and current != identities:
            raise ArchiveError(f"identities change between stages at {stage}")
        identities = current
        if i > 0:
            if manifest.get("parent_manifest") is None or parent is None:
                raise ArchiveError(f"stage {stage} has no parent manifest link")
            expected = manifest["parent_manifest"]
            if expected.get("sha256") != parent["sha256"]:
                raise ArchiveError(f"parent manifest hash mismatch at {stage}")
        for item in manifest.get("inputs", []):
            rel = item["path"]
            if stage != "prepare" and rel in sealed and sealed[rel] != item:
                raise ArchiveError(f"input {rel} of {stage} differs from its sealed output")
        for item in manifest.get("outputs", []):
            rel = item["path"]
            file_path = run_root / rel
            if not file_path.is_file():
                raise ArchiveError(f"sealed output missing: {rel}")
            if file_path.stat().st_size != item["bytes"] or sha256_file(file_path) != item["sha256"]:
                raise ArchiveError(f"sealed output changed: {rel}")
            if rel in sealed and sealed[rel] != item:
                raise ArchiveError(f"earlier output replaced: {rel}")
            sealed[rel] = item
        parent = {"path": path.relative_to(run_root).as_posix(), "sha256": sha256_file(path)}
        present.append(stage)
    if present != list(STAGES):
        raise ArchiveError(f"incomplete stage chain: {present}")
    return {"stages": present, "sealed_artifacts": len(sealed),
            "source_sha256": identities[0], "config_sha256": identities[1], "unit_sha256": identities[2]}


def inventory(root: Path) -> list[dict]:
    entries = []
    for folder, dirs, names in os.walk(root):
        dirs.sort()
        for name in sorted(names):
            path = Path(folder) / name
            if path.is_symlink():
                raise ArchiveError(f"symlinks are not archived: {path}")
            rel = path.relative_to(root).as_posix()
            if rel == MANIFEST_NAME:
                continue
            entries.append({"path": rel, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return entries


def describe(run_root: Path) -> dict:
    state = _json(run_root / "run_state.json")
    meta = _json(run_root / "prepared" / "metadata.json")
    env = _json(run_root / "environment.json") if (run_root / "environment.json").is_file() else {}
    return {"dataset": (state.get("task") or {}).get("dataset"), "data_kind": state.get("dataset"),
            "benchmark_dataset": meta.get("benchmark_dataset"), "is_benchmark_data": meta.get("is_benchmark_data"),
            "profile": state.get("profile"), "scientific_run": state.get("scientific_run"), "status": state.get("status"),
            "implementation_version": state.get("version"),
            "environment": {k: env.get(k) for k in ("python", "torch", "cuda_runtime", "gpu", "device", "platform", "machine")}}


def build_manifest(root: Path, *, campaign: bool) -> dict:
    if campaign:
        splits = sorted(p for p in root.glob("split_*") if (p / "run_state.json").is_file())
        if not splits:
            raise ArchiveError(f"no split_* runs under {root}")
        chains = {p.name: verify_stage_chain(p) for p in splits}
        descriptions = {p.name: describe(p) for p in splits}
        first = descriptions[splits[0].name]
        return {"schema": SCHEMA, "kind": "campaign", "name": root.name, "dataset": first["dataset"],
                "profile": first["profile"], "scientific_run": first["scientific_run"],
                "splits": {name: dict(descriptions[name], chain=chains[name]) for name in chains},
                "source_sha256": sorted({c["source_sha256"] for c in chains.values()}),
                "files": inventory(root), "packed_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()}
    chain = verify_stage_chain(root)
    desc = describe(root)
    return {"schema": SCHEMA, "kind": "run", "name": root.name, **desc, "chain": chain,
            "source_sha256": chain["source_sha256"], "config_sha256": chain["config_sha256"],
            "unit_sha256": chain["unit_sha256"], "files": inventory(root),
            "packed_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()}


def pack(root: Path, output: Path, *, campaign: bool) -> dict:
    root = root.resolve()
    if output.exists():
        raise ArchiveError(f"refusing to overwrite {output}")
    manifest = build_manifest(root, campaign=campaign)
    output.parent.mkdir(parents=True, exist_ok=True)
    top = root.name
    with tarfile.open(output, "w:gz", compresslevel=6) as tar:
        data = json.dumps(manifest, indent=1, sort_keys=True).encode("utf-8")
        info = tarfile.TarInfo(f"{top}/{MANIFEST_NAME}")
        info.size = len(data)
        info.mtime = 0
        tar.addfile(info, io.BytesIO(data))
        for entry in manifest["files"]:
            tar.add(root / entry["path"], arcname=f"{top}/{entry['path']}", recursive=False)
    manifest["archive_sha256"] = sha256_file(output)
    manifest["archive_bytes"] = output.stat().st_size
    return manifest


def verify_archive(archive: Path) -> dict:
    with tarfile.open(archive, "r:*") as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        tops = {name.split("/", 1)[0] for name in members}
        if len(tops) != 1:
            raise ArchiveError(f"archive must contain exactly one top-level directory, found {sorted(tops)}")
        top = tops.pop()
        manifest_member = members.get(f"{top}/{MANIFEST_NAME}")
        if manifest_member is None:
            raise ArchiveError(f"{MANIFEST_NAME} missing from archive")
        manifest = json.loads(tar.extractfile(manifest_member).read().decode("utf-8"))
        if manifest.get("schema") != SCHEMA:
            raise ArchiveError("unexpected archive manifest schema")
        expected = {f"{top}/{e['path']}": e for e in manifest["files"]}
        unexpected = set(members) - set(expected) - {f"{top}/{MANIFEST_NAME}"}
        if unexpected:
            raise ArchiveError(f"files not listed in the manifest: {sorted(unexpected)[:5]}")
        for name, entry in expected.items():
            member = members.get(name)
            if member is None:
                raise ArchiveError(f"listed file missing from archive: {name}")
            digest = hashlib.sha256()
            handle = tar.extractfile(member)
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
            if member.size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                raise ArchiveError(f"archived file differs from its manifest entry: {name}")
    manifest["archive_sha256"] = sha256_file(archive)
    manifest["archive_bytes"] = archive.stat().st_size
    manifest["top_level_directory"] = top
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_root", nargs="?", type=Path, help="completed run directory (or campaign with --campaign)")
    parser.add_argument("--campaign", action="store_true", help="the directory holds split_00 ... split_09")
    parser.add_argument("--output", type=Path, help="archive to write (.tar.gz)")
    parser.add_argument("--verify", type=Path, help="verify an existing archive instead of packing")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            manifest = verify_archive(args.verify)
            print(json.dumps({k: manifest[k] for k in ("kind", "name", "dataset", "profile", "scientific_run",
                                                       "archive_sha256", "archive_bytes")}, indent=1))
            print(f"verified {len(manifest['files'])} files")
            return 0
        if not args.run_root or not args.output:
            parser.error("run_root and --output are required for packing")
        manifest = pack(args.run_root, args.output, campaign=args.campaign)
        print(json.dumps({k: manifest.get(k) for k in ("kind", "name", "dataset", "profile", "scientific_run",
                                                       "source_sha256", "archive_sha256", "archive_bytes")}, indent=1))
        print(f"packed {len(manifest['files'])} files into {args.output}")
        return 0
    except (ArchiveError, OSError, ValueError, KeyError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
