#!/usr/bin/env python3
"""Verify a deposited run archive and register it under ``runs/`` for the paper build.

    python -B scripts/ingest_run.py ../deposits/arxiv_full.tar.gz --doi 10.5281/zenodo.XXXXXXX
    python -B scripts/ingest_run.py --check            # re-verify every registered run
    python -B scripts/ingest_run.py --list

The archive is verified (``scripts/archive_run.py --verify``), extracted into
``runs/<stem>/`` (``runs/roman_empire/`` for a campaign) and recorded in ``runs/index.json``
with: dataset, profile, whether the run is an official-data scientific run, the archive
SHA-256 and byte size, the deposit DOI, the run's own runtime source identity and the source
identity of this checkout.  ``make tables`` and ``make figures`` read ``runs/index.json``;
``scripts/availability.py`` derives the availability statement from it.  Only archives whose
stage chain verifies are registered; an index entry is never written by hand.

Standard library only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.archive_run import ArchiveError, verify_archive, verify_stage_chain  # noqa: E402
from scripts._paper_common import runtime_source_fingerprint  # noqa: E402

STEMS = {"ogbn-arxiv": "arxiv", "ogbn-products": "products", "ogbn-proteins": "proteins", "Roman-empire": "roman_empire",
         "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp", "Peptides-func": "peptides_func"}
INDEX = ROOT / "runs" / "index.json"
SCHEMA = "cross-morse-runs-index-1"


def load_index(path: Path = INDEX) -> dict:
    if path.is_file():
        index = json.loads(path.read_text(encoding="utf-8"))
        if index.get("schema") != SCHEMA:
            raise ArchiveError(f"unexpected index schema in {path}")
        return index
    return {"schema": SCHEMA, "runs": []}


def save_index(index: dict, path: Path = INDEX) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    index["runs"].sort(key=lambda e: e["stem"])
    path.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if not str(target).startswith(str(destination)):
            raise ArchiveError(f"archive member escapes the destination: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise ArchiveError(f"unsupported archive member type: {member.name}")
    tar.extractall(destination, filter="data")


def ingest(archive: Path, *, doi: str | None, runs_root: Path, index_path: Path, replace: bool) -> dict:
    manifest = verify_archive(archive)
    dataset = manifest.get("dataset")
    if dataset not in STEMS:
        raise ArchiveError(f"archive dataset {dataset!r} is not one of the seven benchmarks")
    stem = STEMS[dataset]
    target = runs_root / stem
    if target.exists():
        if not replace:
            raise ArchiveError(f"{target} already exists; pass --replace to re-ingest")
        shutil.rmtree(target)
    with tarfile.open(archive, "r:*") as tar:
        staging = runs_root / f".ingest_{stem}"
        if staging.exists():
            shutil.rmtree(staging)
        _safe_extract(tar, staging)
        extracted = staging / manifest["top_level_directory"]
        extracted.rename(target)
        shutil.rmtree(staging, ignore_errors=True)
    # re-verify the extracted chain from disk
    if manifest["kind"] == "campaign":
        chains = {p.name: verify_stage_chain(p) for p in sorted(target.glob("split_*")) if (p / "run_state.json").is_file()}
        run_sources = sorted({c["source_sha256"] for c in chains.values()})
    else:
        chain = verify_stage_chain(target)
        run_sources = [chain["source_sha256"]]
    entry = {"stem": stem, "dataset": dataset, "path": stem, "kind": manifest["kind"],
             "profile": manifest.get("profile"), "scientific_run": bool(manifest.get("scientific_run")),
             "archive_name": archive.name, "archive_sha256": manifest["archive_sha256"],
             "archive_bytes": manifest["archive_bytes"], "deposit_doi": doi,
             "run_source_sha256": run_sources, "checkout_source_sha256": runtime_source_fingerprint(ROOT),
             "files": len(manifest["files"]), "verified": True,
             "ingested_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()}
    index = load_index(index_path)
    index["runs"] = [e for e in index["runs"] if e["stem"] != stem] + [entry]
    save_index(index, index_path)
    return entry


def check(runs_root: Path, index_path: Path) -> int:
    index = load_index(index_path)
    problems = []
    for entry in index["runs"]:
        path = runs_root / entry["path"]
        try:
            if entry["kind"] == "campaign":
                for p in sorted(path.glob("split_*")):
                    if (p / "run_state.json").is_file():
                        verify_stage_chain(p)
            else:
                verify_stage_chain(path)
        except (ArchiveError, OSError, KeyError) as exc:
            problems.append(f"{entry['stem']}: {exc}")
    for p in problems:
        print("CHECK FAILED:", p, file=sys.stderr)
    if problems:
        return 1
    print(f"{len(index['runs'])} registered run(s) verify against their sealed stage chains")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", nargs="?", type=Path)
    parser.add_argument("--doi", help="deposit DOI of the archive (recorded in the index)")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.list:
            for entry in load_index(args.index)["runs"]:
                print(f"{entry['stem']:14s} {entry['dataset']:14s} profile={entry['profile']} scientific={entry['scientific_run']} "
                      f"doi={entry.get('deposit_doi')} archive={entry['archive_sha256'][:12]}")
            return 0
        if args.check:
            return check(args.runs_root, args.index)
        if not args.archive:
            parser.error("an archive path is required")
        entry = ingest(args.archive, doi=args.doi, runs_root=args.runs_root, index_path=args.index, replace=args.replace)
        print(json.dumps(entry, indent=1))
        return 0
    except (ArchiveError, OSError, ValueError, KeyError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
