#!/usr/bin/env python3
"""Build the paper's tables and figures from completed run directories.

    python -B scripts/build_paper.py all --runs-root examples/runs --label examples \
        --output-dir paper/generated/examples
    python -B scripts/build_paper.py all --index runs/index.json --label official \
        --output-dir paper/generated/official
    python -B scripts/build_paper.py all --runs-root examples/runs --label examples \
        --output-dir paper/generated/examples --check

Input is either ``--runs-root DIR`` (every immediate subdirectory that is a completed run, or a
Roman-empire campaign directory containing ``split_00`` ... ``split_09``) or ``--index FILE``
(``runs/index.json`` written by ``scripts/ingest_run.py``).  Output is
``<output-dir>/evidence/<stem>.json``, ``<output-dir>/tables/<id>.{tex,md,csv}`` and
``<output-dir>/figures/<id>.{png,svg}``.  ``--check`` rebuilds evidence and tables into a
temporary directory and fails if they differ from the committed ones (figures are not compared
byte-for-byte because font rendering differs across platforms).

NumPy is required; Matplotlib only for figures.  No PyTorch, no checkpoint loading, no training.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.build.evidence import (DATASET_ORDER, EvidenceError, evidence_from_campaign,  # noqa: E402
                                  evidence_from_run, load_evidence, stem_for, write_evidence)
from paper.build.tables import TABLE_IDS, build_tables  # noqa: E402

STEM_TO_DATASET = {"arxiv": "ogbn-arxiv", "products": "ogbn-products", "proteins": "ogbn-proteins",
                   "roman_empire": "Roman-empire", "pascalvoc_sp": "PascalVOC-SP", "coco_sp": "COCO-SP",
                   "peptides_func": "Peptides-func"}


def is_run(path: Path) -> bool:
    return (path / "final" / "results_summary.json").is_file()


def is_campaign(path: Path) -> bool:
    return any(is_run(p) for p in path.glob("split_*"))


def discover(runs_root: Path) -> list[tuple[str, Path]]:
    found = []
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        if is_run(child) or is_campaign(child):
            found.append((child.name, child))
    if not found:
        raise EvidenceError(f"no completed runs or campaigns under {runs_root}")
    return found


def from_index(index_path: Path) -> list[tuple[str, Path]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    found = []
    for entry in index.get("runs", []):
        if not entry.get("verified"):
            continue
        path = (index_path.parent / entry["path"]).resolve()
        if is_run(path) or is_campaign(path):
            found.append((entry.get("stem") or path.name, path))
    if not found:
        raise EvidenceError(f"no verified runs listed in {index_path}")
    return found


def collect(found) -> list[dict]:
    records = []
    for stem, path in found:
        record = evidence_from_campaign(path) if (is_campaign(path) and not is_run(path)) else evidence_from_run(path)
        record["stem"] = stem_for(record["dataset"]) if record["dataset"] in DATASET_ORDER else stem
        records.append(record)
    records.sort(key=lambda r: DATASET_ORDER.index(r["dataset"]) if r["dataset"] in DATASET_ORDER else 99)
    return records


def write_all(records, output_dir: Path, label: str, *, figures: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    for record in records:
        write_evidence(record, evidence_dir / f"{record['stem']}.json")
    tables = build_tables(records, output_dir / "tables", label)
    figs = {}
    if figures:
        from paper.build.figures import build_figures
        figs = build_figures(records, output_dir / "figures", label)
    summary = {"label": label, "datasets": [r["dataset"] for r in records],
               "synthetic_examples": any(not r.get("scientific_run") for r in records),
               "evidence": {r["stem"]: f"evidence/{r['stem']}.json" for r in records},
               "tables": tables, "figures": figs}
    (output_dir / "BUILD.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _relativize(text: str, base: Path) -> str:
    return text.replace(str(base.resolve()) + "/", "")


def check(records, output_dir: Path, label: str) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "fresh"
        write_all(records, fresh, label, figures=False)
        problems = []
        for sub in ("evidence", "tables"):
            fresh_dir, committed_dir = fresh / sub, output_dir / sub
            if not committed_dir.is_dir():
                problems.append(f"missing committed directory {committed_dir}")
                continue
            fresh_files = sorted(p.name for p in fresh_dir.iterdir())
            committed_files = sorted(p.name for p in committed_dir.iterdir())
            if fresh_files != committed_files:
                problems.append(f"{sub}: file set differs ({sorted(set(fresh_files) ^ set(committed_files))})")
            for name in set(fresh_files) & set(committed_files):
                a, b = fresh_dir / name, committed_dir / name
                if sub == "evidence":
                    # run_root is an absolute path; compare everything else
                    ja, jb = json.loads(a.read_text()), json.loads(b.read_text())
                    ja.pop("run_root", None); jb.pop("run_root", None)
                    if ja != jb:
                        problems.append(f"evidence/{name} differs from a fresh extraction")
                elif not filecmp.cmp(a, b, shallow=False):
                    problems.append(f"{sub}/{name} differs from a fresh build")
        if problems:
            for p in problems:
                print("CHECK FAILED:", p, file=sys.stderr)
            return 1
    print(f"{label}: committed evidence and tables match a fresh build ({len(records)} runs)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("evidence", "tables", "figures", "all"))
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs-root", type=Path, help="directory of completed runs or campaigns")
    src.add_argument("--index", type=Path, help="runs/index.json written by scripts/ingest_run.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True, help="short label recorded in every output (e.g. examples, official)")
    parser.add_argument("--check", action="store_true", help="compare committed evidence/tables with a fresh build")
    args = parser.parse_args(argv)
    try:
        found = from_index(args.index) if args.index else discover(args.runs_root)
        records = collect(found)
        for record in records:
            record["run_root"] = _relativize(record["run_root"], ROOT)
        if args.check:
            return check(records, args.output_dir, args.label)
        if args.what == "evidence":
            out = args.output_dir / "evidence"
            out.mkdir(parents=True, exist_ok=True)
            for record in records:
                write_evidence(record, out / f"{record['stem']}.json")
            print(f"wrote {len(records)} evidence records to {out}")
            return 0
        if args.what == "tables":
            written = build_tables(records, args.output_dir / "tables", args.label)
            print(f"wrote {len(written)} tables ({', '.join(TABLE_IDS)}) to {args.output_dir / 'tables'}")
            return 0
        if args.what == "figures":
            from paper.build.figures import build_figures
            written = build_figures(records, args.output_dir / "figures", args.label)
            print(f"wrote {len(written)} figures to {args.output_dir / 'figures'}")
            return 0
        summary = write_all(records, args.output_dir, args.label, figures=True)
        print(f"built evidence, {len(summary['tables'])} tables and {len(summary['figures'])} figures for "
              f"{len(records)} run(s) under {args.output_dir}")
        if summary["synthetic_examples"]:
            print("note: at least one run is a synthetic software example; outputs carry that banner")
        return 0
    except (EvidenceError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
