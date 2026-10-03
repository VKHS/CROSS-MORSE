#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

PACKAGE_DIR = Path(__file__).resolve().parent
OVERLAY_DIR = PACKAGE_DIR / "overlay"
LICENSE_DIR = PACKAGE_DIR / "license_templates"

PUBLICATION_SUFFIXES = {
    ".tex", ".pdf", ".aux", ".bbl", ".blg", ".fls", ".fdb_latexmk",
    ".toc", ".lof", ".lot", ".out",
}
SKIP_DIR_NAMES = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "outputs", "runs", "logs", "wandb", "assembled",
}
ROOT_GENERATED_DIR_NAMES = {
    "checkpoints", "data", "outputs", "predictions", "runs", "trajectories", "wandb",
}
MIXED_RESULT_DIR_NAMES = {"artifacts", "results"}
MIXED_RESULT_SOURCE_SUFFIXES = {".py", ".sh", ".bash", ".md", ".txt", ".yaml", ".yml", ".toml"}
GENERATED_BINARY_SUFFIXES = {
    ".ckpt", ".npz", ".npy", ".parquet", ".pkl", ".pickle", ".pt", ".pth",
    ".safetensors", ".tar", ".tgz", ".zip", ".7z",
}
STALE_RELATIVE_PATHS = {
    "paper",
    "manuscript",
    "scripts/check_manuscript_alignment.py",
    "scripts/check_stage_independence.py",
    "scripts/validate_public_snapshot.py",
    "docs/manuscript-alignment.md",
    "docs/numerical-code-preservation.md",
    "results/manuscript_results.json",
    "artifacts/CLAIM_TO_ARTIFACT.csv",
}
DATASET_ALIASES = {
    "ogbn-arxiv": ("ogbn-arxiv", "ogbn_arxiv", "arxiv"),
    "ogbn-proteins": ("ogbn-proteins", "ogbn_proteins", "proteins"),
    "ogbn-products": ("ogbn-products", "ogbn_products", "products"),
    "Roman-empire": ("roman-empire", "roman_empire", "roman"),
    "PascalVOC-SP": ("pascalvoc-sp", "pascalvoc_sp", "pascal", "voc-sp", "voc_sp"),
    "COCO-SP": ("coco-sp", "coco_sp", "coco"),
    "Peptides-func": ("peptides-func", "peptides_func", "peptides"),
}
RELEASE_TOOL_SCRIPTS = {
    "scripts/release_audit.py", "scripts/list_experiments.py",
    "scripts/create_release_manifest.py", "scripts/validate_run_manifest.py",
}
SCIENTIFIC_SOURCE_SUFFIXES = {".py", ".sh", ".bash", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}


@dataclass
class CopyRecord:
    path: str
    status: str
    reason: str | None = None
    source_sha256: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a code-only public CROSS-MORSE repository from a complete source tree."
    )
    parser.add_argument("source", type=Path, help="source directory or ZIP containing the complete code")
    parser.add_argument("--output", type=Path, help="output directory; defaults to a sibling CROSS-MORSE-public")
    parser.add_argument("--article-url", default="", help="final journal article or DOI URL")
    parser.add_argument("--supplement-url", default="", help="final journal Supplementary Information URL")
    parser.add_argument("--doi", default="", help="final journal DOI without the https://doi.org/ prefix")
    parser.add_argument(
        "--implementation-map", type=Path,
        help="optional JSON mapping each paper dataset to one or more executable Python entrypoints",
    )
    parser.add_argument(
        "--license",
        choices=("keep", "bsd-3-clause", "mit"),
        default="keep",
        help="preserve the existing LICENSE or install a reviewed license template",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing output directory")
    parser.add_argument("--draft", action="store_true", help="produce an inspectable tree even when blocking checks fail")
    parser.add_argument("--make-zip", action="store_true", help="create a ZIP only after strict validation passes")
    return parser.parse_args()


def is_publication_file(path: Path) -> bool:
    low = path.name.lower()
    return path.suffix.lower() in PUBLICATION_SUFFIXES or low.endswith(".synctex.gz")


def select_extracted_root(extract_dir: Path) -> Path:
    children = [p for p in extract_dir.iterdir() if p.name not in {"__MACOSX"}]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return extract_dir


def source_context(source: Path, temp_root: Path) -> Path:
    source = source.resolve()
    if source.is_dir():
        return source
    if source.is_file() and source.suffix.lower() == ".zip":
        extract_dir = temp_root / "source"
        extract_dir.mkdir(parents=True)
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                target = (extract_dir / info.filename).resolve()
                if not str(target).startswith(str(extract_dir.resolve()) + os.sep) and target != extract_dir.resolve():
                    raise ValueError(f"unsafe ZIP member: {info.filename}")
            archive.extractall(extract_dir)
        return select_extracted_root(extract_dir)
    raise FileNotFoundError(f"source must be a directory or ZIP: {source}")


def should_skip_directory(rel: Path) -> bool:
    if any(part in SKIP_DIR_NAMES for part in rel.parts):
        return True
    if rel.parts and rel.parts[0].lower() in ROOT_GENERATED_DIR_NAMES:
        return True
    return rel.as_posix() in STALE_RELATIVE_PATHS


def should_skip_file(rel: Path) -> tuple[bool, str | None]:
    if should_skip_directory(rel.parent):
        return True, "generated/private directory excluded"
    if is_publication_file(rel):
        return True, "publication format excluded"
    if rel.parts and rel.parts[0].lower() in MIXED_RESULT_DIR_NAMES:
        if rel.suffix.lower() not in MIXED_RESULT_SOURCE_SUFFIXES:
            return True, "generated numerical result excluded; source/generator files are retained"
    if rel.suffix.lower() in GENERATED_BINARY_SUFFIXES or rel.name.lower().endswith(".tar.gz"):
        return True, "precomputed/generated binary artifact excluded"
    if rel.as_posix() in STALE_RELATIVE_PATHS:
        return True, "stale alignment artifact excluded"
    low = rel.as_posix().lower()
    if "for_chatgpt" in low or "for-chatgpt" in low:
        return True, "recipient-specific artifact excluded"
    return False, None


def copy_source_tree(source: Path, destination: Path) -> list[CopyRecord]:
    records: list[CopyRecord] = []
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        if path.is_dir():
            if should_skip_directory(rel):
                records.append(CopyRecord(rel.as_posix(), "skipped", "generated/private directory excluded"))
            continue
        skip, reason = should_skip_file(rel)
        if skip:
            records.append(CopyRecord(rel.as_posix(), "skipped", reason))
            continue
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        records.append(CopyRecord(rel.as_posix(), "copied", source_sha256=sha256_file(path)))
    return records


def merge_gitignore(existing: Path, incoming: Path) -> None:
    blocks: list[str] = []
    seen: set[str] = set()
    for path in (existing, incoming):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            key = line.rstrip()
            if key in seen:
                continue
            seen.add(key)
            blocks.append(key)
    existing.write_text("\n".join(blocks).rstrip() + "\n", encoding="utf-8")


def overlay_tree(destination: Path) -> None:
    for path in sorted(OVERLAY_DIR.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(OVERLAY_DIR)
        if rel.as_posix() in {"README.template.md", "CITATION.cff.template"}:
            continue
        target_rel = rel
        if rel.as_posix() == "Makefile" and (destination / "Makefile").exists():
            target_rel = Path("Makefile.release")
        target = destination / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel.as_posix() == ".gitignore" and target.exists():
            merge_gitignore(target, path)
        else:
            shutil.copy2(path, target)


def verify_source_preservation(destination: Path, records: list[CopyRecord]) -> list[str]:
    managed = {
        path.relative_to(OVERLAY_DIR).as_posix()
        for path in OVERLAY_DIR.rglob("*")
        if path.is_file() and path.name not in {"README.template.md", "CITATION.cff.template"}
    }
    managed.update({"README.md", "CITATION.cff", "LICENSE", "LICENSE_REVIEW_REQUIRED.md"})
    mutations: list[str] = []
    for record in records:
        if record.status != "copied" or record.source_sha256 is None:
            continue
        rel = Path(record.path)
        if rel.suffix.lower() not in SCIENTIFIC_SOURCE_SUFFIXES:
            continue
        if record.path in managed:
            continue
        target = destination / rel
        if not target.is_file() or sha256_file(target) != record.source_sha256:
            mutations.append(record.path)
    return mutations


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""


def detect_entrypoints(root: Path) -> dict[str, list[str]]:
    candidates: dict[str, list[tuple[int, str]]] = {name: [] for name in DATASET_ALIASES}
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        low = rel.lower()
        if rel in RELEASE_TOOL_SCRIPTS or low.startswith(("release_tools/", "release/", "tests/", ".github/")):
            continue
        stem = path.name.lower()
        for dataset, aliases in DATASET_ALIASES.items():
            if not any(alias in low for alias in aliases):
                continue
            score = 0
            if any(token in stem for token in ("run", "train", "experiment", "engine", "main")):
                score += 4
            if "test" in stem:
                score -= 5
            text = read_text(path)
            if "if __name__" in text:
                score += 2
            if any(token in text for token in ("argparse", "click.command", "typer")):
                score += 1
            if score > 0:
                candidates[dataset].append((score, rel))
    return {name: [path for _, path in sorted(items, key=lambda x: (-x[0], x[1]))] for name, items in candidates.items()}


def load_implementation_map(path: Path | None, destination: Path) -> tuple[dict[str, list[str]], str]:
    if path is None:
        return detect_entrypoints(destination), "automatic filename/content detection"
    raw = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    mapping = raw.get("datasets", raw) if isinstance(raw, dict) else None
    if not isinstance(mapping, dict):
        raise ValueError("implementation map must be a JSON object or contain a 'datasets' object")
    normalized: dict[str, list[str]] = {}
    for dataset in DATASET_ALIASES:
        value = mapping.get(dataset, [])
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError(f"implementation map entry for {dataset!r} must be a string list")
        paths = [Path(item).as_posix() for item in value]
        for rel in paths:
            target = (destination / rel).resolve()
            try:
                target.relative_to(destination.resolve())
            except ValueError as exc:
                raise ValueError(f"implementation path escapes repository: {rel}") from exc
            if not target.is_file() or target.suffix.lower() != ".py":
                raise ValueError(f"implementation entrypoint is not an existing Python file: {rel}")
        normalized[dataset] = paths
    return normalized, f"author-supplied map: {path.name}"


def implementation_table(entrypoints: dict[str, list[str]]) -> str:
    lines = ["| Dataset | Public implementation entrypoints |", "|---|---|"]
    for dataset, paths in entrypoints.items():
        cell = "<br>".join(f"`{p}`" for p in paths[:3]) if paths else "**Not detected - release audit fails**"
        lines.append(f"| {dataset} | {cell} |")
    return "\n".join(lines)


def render_templates(
    destination: Path, article_url: str, supplement_url: str, doi: str,
    license_id: str, implementation_map_path: Path | None,
) -> None:
    entrypoints, mapping_source = load_implementation_map(implementation_map_path, destination)
    article_link = article_url.strip() or "Journal article link pending"
    supplement_link = supplement_url.strip() or "Journal Supplementary Information link pending"
    doi_text = doi.strip() or "pending"
    replacements = {
        "{{ARTICLE_LINK}}": article_link,
        "{{SUPPLEMENT_LINK}}": supplement_link,
        "{{DOI_TEXT}}": doi_text,
        "{{IMPLEMENTATION_TABLE}}": implementation_table(entrypoints),
        "{{LICENSE_ID}}": license_id,
    }
    for template_name, output_name in (("README.template.md", "README.md"), ("CITATION.cff.template", "CITATION.cff")):
        text = (OVERLAY_DIR / template_name).read_text(encoding="utf-8")
        for key, value in replacements.items():
            text = text.replace(key, value)
        (destination / output_name).write_text(text.rstrip() + "\n", encoding="utf-8")

    implementation_map = {
        "schema_version": "1.0",
        "source": mapping_source,
        "datasets": entrypoints,
    }
    (destination / "release" / "implementation_map.json").write_text(
        json.dumps(implementation_map, indent=2) + "\n", encoding="utf-8"
    )

    publication = {
        "schema_version": "1.0",
        "article_url": article_url.strip(),
        "supplement_url": supplement_url.strip(),
        "doi": doi.strip(),
    }
    (destination / "release" / "publication.json").write_text(
        json.dumps(publication, indent=2) + "\n", encoding="utf-8"
    )


def install_license(destination: Path, choice: str) -> str:
    target = destination / "LICENSE"
    if choice == "bsd-3-clause":
        shutil.copy2(LICENSE_DIR / "BSD-3-Clause.txt", target)
        return "BSD-3-Clause"
    if choice == "mit":
        shutil.copy2(LICENSE_DIR / "MIT.txt", target)
        return "MIT"
    if target.exists() and target.stat().st_size > 0:
        text = target.read_text(encoding="utf-8", errors="ignore").lower()
        if "bsd 3-clause" in text or ("redistribution and use in source and binary forms" in text and "neither the name" in text):
            return "BSD-3-Clause"
        if "mit license" in text or ("permission is hereby granted, free of charge" in text and "substantial portions" in text):
            return "MIT"
        if "apache license" in text and "version 2.0" in text:
            return "Apache-2.0"
        if "gnu general public license" in text and "version 3" in text:
            return "GPL-3.0-only"
        if "gnu general public license" in text and "version 2" in text:
            return "GPL-2.0-only"
        if "mozilla public license" in text and "2.0" in text:
            return "MPL-2.0"
        (destination / "LICENSE_REVIEW_REQUIRED.md").write_text(
            "# License review required\n\nThe existing LICENSE was not recognized as a standard public software license. "
            "Select an institution-approved license explicitly before release.\n",
            encoding="utf-8",
        )
        return "LicenseRef-ReviewRequired"
    (destination / "LICENSE_REVIEW_REQUIRED.md").write_text(
        "# License review required\n\nNo existing license was found. Choose an institution-approved license before public release.\n",
        encoding="utf-8",
    )
    return "LicenseRef-ReviewRequired"


def run(command: list[str], cwd: Path, *, check: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def make_zip(source_dir: Path, zip_path: Path) -> None:
    """Create a deterministic, portable release ZIP."""
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            arcname = (Path(source_dir.name) / path.relative_to(source_dir)).as_posix()
            payload = path.read_bytes()
            info = zipfile.ZipInfo(arcname, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            executable = path.suffix.lower() in {".sh", ".bash"} or payload.startswith(b"#!")
            mode = 0o100755 if executable else 0o100644
            info.external_attr = (mode & 0xFFFF) << 16
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main() -> int:
    args = parse_args()
    source_arg = args.source.resolve()
    default_output = source_arg.parent / "CROSS-MORSE-public"
    output = (args.output or default_output).resolve()
    if source_arg.is_dir():
        try:
            output.relative_to(source_arg)
        except ValueError:
            pass
        else:
            print("ERROR: output must not be inside the source tree", file=sys.stderr)
            return 2
    if output.exists():
        if not args.force:
            print(f"ERROR: output exists; use --force: {output}", file=sys.stderr)
            return 2
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cross_morse_public_upgrade_") as tmp_name:
        temp_root = Path(tmp_name)
        source = source_context(source_arg, temp_root)
        staging = temp_root / "public"
        staging.mkdir()
        records = copy_source_tree(source, staging)
        overlay_tree(staging)
        license_id = install_license(staging, args.license)
        render_templates(
            staging, args.article_url, args.supplement_url, args.doi,
            license_id, args.implementation_map,
        )
        source_mutations = verify_source_preservation(staging, records)

        report_path = output.parent / f"{output.name}_UPGRADE_REPORT.json"
        audit_report_path = output.parent / f"{output.name}_AUDIT_REPORT.json"
        audit_cmd = [
            sys.executable, "scripts/release_audit.py", "--root", str(staging),
            "--json-output", str(audit_report_path), "--strict",
        ]
        audit_result = run(audit_cmd, staging)

        test_result = run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_release_*.py"],
            staging,
        )

        manifest_result = None
        if audit_result.returncode == 0 and test_result.returncode == 0:
            manifest_result = run(
                [sys.executable, "scripts/create_release_manifest.py", "--root", str(staging), "--output", "RELEASE_MANIFEST.json"],
                staging,
            )
            if manifest_result.returncode == 0:
                manifest_result = run(
                    [sys.executable, "scripts/create_release_manifest.py", "--root", str(staging), "--verify", "RELEASE_MANIFEST.json"],
                    staging,
                )

        blocking = (
            bool(source_mutations)
            or audit_result.returncode != 0
            or test_result.returncode != 0
            or manifest_result is None
            or manifest_result.returncode != 0
        )
        if blocking and not args.draft:
            summary = {
                "schema_version": "2.0",
                "source": str(source_arg),
                "output_requested": str(output),
                "status": "blocked",
                "audit_returncode": audit_result.returncode,
                "test_returncode": test_result.returncode,
                "manifest_returncode": None if manifest_result is None else manifest_result.returncode,
                "unexpected_source_mutations": source_mutations,
                "copied_files": sum(r.status == "copied" for r in records),
                "skipped_files": sum(r.status == "skipped" for r in records),
                "audit_report": str(audit_report_path),
                "copy_records": [asdict(r) for r in records if r.status == "skipped"],
            }
            report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            for rel in source_mutations:
                print(f"ERROR: scientific source changed unexpectedly during migration: {rel}", file=sys.stderr)
            print(f"ERROR: public release blocked. See {report_path} and {audit_report_path}", file=sys.stderr)
            return 1

        shutil.copytree(staging, output)
        summary = {
            "schema_version": "2.0",
            "source": str(source_arg),
            "output": str(output),
            "status": "draft" if blocking else "validated",
            "audit_returncode": audit_result.returncode,
            "test_returncode": test_result.returncode,
            "manifest_returncode": None if manifest_result is None else manifest_result.returncode,
            "unexpected_source_mutations": source_mutations,
            "license": license_id,
            "article_url_set": bool(args.article_url.strip()),
            "supplement_url_set": bool(args.supplement_url.strip()),
            "doi_set": bool(args.doi.strip()),
            "copied_files": sum(r.status == "copied" for r in records),
            "skipped_files": sum(r.status == "skipped" for r in records),
            "output_tree_sha256": sha256_file(output / "RELEASE_MANIFEST.json") if (output / "RELEASE_MANIFEST.json").exists() else None,
            "audit_report": str(audit_report_path),
            "copy_records": [asdict(r) for r in records if r.status == "skipped"],
        }
        report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")
        print(f"upgrade report: {report_path}")

        if args.make_zip:
            if blocking:
                print("draft output created; ZIP suppressed because validation did not pass", file=sys.stderr)
            else:
                zip_path = output.parent / f"{output.name}_public_release.zip"
                make_zip(output, zip_path)
                print(f"wrote {zip_path}")
        return 0 if not blocking else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
