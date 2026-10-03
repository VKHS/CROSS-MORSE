#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

DATASET_ALIASES = {
    "ogbn-arxiv": ("ogbn-arxiv", "ogbn_arxiv", "arxiv"),
    "ogbn-proteins": ("ogbn-proteins", "ogbn_proteins", "proteins"),
    "ogbn-products": ("ogbn-products", "ogbn_products", "products"),
    "Roman-empire": ("roman-empire", "roman_empire", "roman"),
    "PascalVOC-SP": ("pascalvoc-sp", "pascalvoc_sp", "pascal", "voc-sp", "voc_sp"),
    "COCO-SP": ("coco-sp", "coco_sp", "coco"),
    "Peptides-func": ("peptides-func", "peptides_func", "peptides"),
}

FORBIDDEN_SUFFIXES = {".tex", ".pdf", ".aux", ".bbl", ".blg", ".fls", ".fdb_latexmk"}
GENERATED_BINARY_SUFFIXES = {
    ".ckpt", ".npz", ".npy", ".parquet", ".pkl", ".pickle", ".pt", ".pth",
    ".safetensors", ".tar", ".tgz", ".zip", ".7z",
}
ROOT_GENERATED_DIR_NAMES = {
    "checkpoints", "data", "outputs", "predictions", "runs", "trajectories", "wandb",
}
MIXED_RESULT_DIR_NAMES = {"artifacts", "results"}
MIXED_RESULT_SOURCE_SUFFIXES = {".py", ".sh", ".bash", ".md", ".txt", ".yaml", ".yml", ".toml"}
FORBIDDEN_NAME_PATTERNS = (
    re.compile(r"for[_-]?chatgpt", re.I),
    re.compile(r"manuscript", re.I),
)
PRIVATE_PATTERNS = (
    ("private scratch path", re.compile(r"/(?:nobackup|scratch|proj/disk)/[^\s\"']+", re.I)),
    ("private home path", re.compile(r"/(?:home|users)/[A-Za-z0-9._-]+(?:/[^\s\"']*)?", re.I)),
    ("private Windows user path", re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9._-]+", re.I)),
    ("private cluster account/allocation", re.compile(r"\b(?:naiss\d{4}|khalilis|from-alvis)\b", re.I)),
    ("literal Slurm account", re.compile(r"(?m)^\s*#SBATCH\s+(?:--account(?:=|\s+)|-A\s+)(?![$<{])[A-Za-z0-9._-]+")),
    ("private login host", re.compile(r"\b[a-z0-9._-]+@login\.[a-z0-9._-]+", re.I)),
    ("literal remote-copy target", re.compile(r"\b(?:scp|rsync)[^\n]{0,200}\b[A-Za-z0-9._-]+@[A-Za-z0-9._-]+:", re.I)),
    ("credential-like token", re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{20,}")),
)
OVERCLAIM_PATTERNS = (
    re.compile(r"one[- ]command reconstruction of every reported table and figure", re.I),
    re.compile(r"complete experiment package contains.*(?:checkpoints|predictions|trajectories)", re.I | re.S),
    re.compile(r"checkpoints, predictions, propensity[- ]logged trajectories, statistical outputs", re.I),
)
KNOWN_SCIENTIFIC_ANTI_PATTERNS = (
    ("joint backbone fine-tuning defaults on", re.compile(r"joint_finetune(?:_encoder)?[^\n]{0,160}default\s*=\s*True", re.I)),
    ("test blindness defaults off", re.compile(r"blind[_-]?test[^\n]{0,160}default\s*=\s*False", re.I)),
    ("test metrics enabled by negating blind-test flag", re.compile(r"include_test_metrics\s*=\s*not\s+blind[_-]?test", re.I)),
    ("encoder/head explicitly made trainable", re.compile(r"(?:encoder|source_head|prediction_head|backbone)[^\n]{0,120}requires_grad(?:_|)\s*\(?\s*True", re.I)),
)
TEXT_SUFFIXES = {".py", ".sh", ".bash", ".json", ".yaml", ".yml", ".toml", ".md", ".txt", ".cfg", ".ini", ".bat"}
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv"}


@dataclass
class Finding:
    severity: str
    code: str
    path: str
    message: str
    line: int | None = None


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in SKIP_PARTS for part in rel.parts):
            continue
        yield path


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def detect_entrypoints(root: Path) -> dict[str, list[str]]:
    candidates: dict[str, list[tuple[int, str]]] = {name: [] for name in DATASET_ALIASES}
    for path in iter_files(root):
        if path.suffix.lower() != ".py":
            continue
        rel = path.relative_to(root).as_posix()
        low = rel.lower()
        stem = path.name.lower()
        if rel in {
            "scripts/release_audit.py", "scripts/list_experiments.py",
            "scripts/create_release_manifest.py", "scripts/validate_run_manifest.py"
        } or low.startswith(("release_tools/", "release/", "tests/", ".github/")):
            continue
        for dataset, aliases in DATASET_ALIASES.items():
            if not any(alias in low for alias in aliases):
                continue
            score = 0
            if any(token in stem for token in ("run", "train", "experiment", "engine", "main")):
                score += 4
            if "test" in stem:
                score -= 5
            if "release_tools" in low or "/tests/" in f"/{low}":
                score -= 5
            text = read_text(path) or ""
            if "if __name__" in text:
                score += 2
            if "argparse" in text or "click.command" in text or "typer" in text:
                score += 1
            if score > 0:
                candidates[dataset].append((score, rel))
    return {dataset: [path for _, path in sorted(items, key=lambda x: (-x[0], x[1]))] for dataset, items in candidates.items()}


def load_implementation_map(root: Path, findings: list[Finding]) -> dict[str, list[str]]:
    path = root / "release" / "implementation_map.json"
    if not path.is_file():
        return detect_entrypoints(root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        findings.append(Finding("error", "implementation-map", path.relative_to(root).as_posix(), str(exc)))
        return {name: [] for name in DATASET_ALIASES}
    mapping = data.get("datasets") if isinstance(data, dict) else None
    if not isinstance(mapping, dict):
        findings.append(Finding("error", "implementation-map", path.relative_to(root).as_posix(), "datasets must be an object"))
        return {name: [] for name in DATASET_ALIASES}
    normalized: dict[str, list[str]] = {}
    for dataset in DATASET_ALIASES:
        value = mapping.get(dataset, [])
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
            findings.append(Finding("error", "implementation-map", path.relative_to(root).as_posix(), f"invalid path list for {dataset}"))
            normalized[dataset] = []
            continue
        normalized[dataset] = []
        for rel in value:
            candidate = (root / rel).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                findings.append(Finding("error", "implementation-map", rel, "entrypoint escapes the repository"))
                continue
            if not candidate.is_file() or candidate.suffix.lower() != ".py":
                findings.append(Finding("error", "implementation-map", rel, "entrypoint must be an existing Python file"))
                continue
            candidate_rel = candidate.relative_to(root).as_posix()
            if candidate_rel in {
                "scripts/release_audit.py", "scripts/list_experiments.py",
                "scripts/create_release_manifest.py", "scripts/validate_run_manifest.py",
            } or candidate_rel.startswith(("release_tools/", "tests/")):
                findings.append(Finding("error", "implementation-map", candidate_rel, "release tooling is not a scientific dataset entrypoint"))
                continue
            normalized[dataset].append(candidate_rel)
    unknown = sorted(set(mapping) - set(DATASET_ALIASES))
    if unknown:
        findings.append(Finding("error", "implementation-map", path.relative_to(root).as_posix(), f"unknown datasets: {unknown}"))
    return normalized


def audit(root: Path) -> tuple[list[Finding], dict[str, list[str]]]:
    findings: list[Finding] = []
    entrypoints = load_implementation_map(root, findings)

    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        low_name = path.name.lower()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or low_name.endswith(".synctex.gz"):
            findings.append(Finding("error", "publication-file", rel, "publication/manuscript format is excluded from the source-code release"))
        if (path.suffix.lower() in GENERATED_BINARY_SUFFIXES or low_name.endswith(".tar.gz")) and not rel.startswith("tests/fixtures/"):
            findings.append(Finding("error", "generated-artifact", rel, "precomputed/generated binary artifact is excluded from this source-only release"))
        rel_parts = Path(rel).parts
        if rel_parts and rel_parts[0].lower() in ROOT_GENERATED_DIR_NAMES:
            findings.append(Finding("error", "generated-artifact", rel, "generated result/data directory is excluded from this source-only release"))
        if rel_parts and rel_parts[0].lower() in MIXED_RESULT_DIR_NAMES and path.suffix.lower() not in MIXED_RESULT_SOURCE_SUFFIXES:
            findings.append(Finding("error", "generated-artifact", rel, "numerical result artifact is excluded; retain only source/generator documentation in this directory"))
        if any(pattern.search(rel) for pattern in FORBIDDEN_NAME_PATTERNS):
            findings.append(Finding("error", "private-or-manuscript-name", rel, "filename contains a manuscript or recipient-specific term"))

        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = read_text(path)
        if text is None:
            continue
        if path.suffix.lower() == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                findings.append(Finding("error", "json-syntax", rel, exc.msg, exc.lineno))
        # Do not scan this audit script against its own pattern literals. All other
        # public text files remain subject to the private-infrastructure scan.
        if rel != "scripts/release_audit.py":
            for label, pattern in PRIVATE_PATTERNS:
                for match in pattern.finditer(text):
                    findings.append(Finding("error", "private-infrastructure", rel, label, line_number(text, match.start())))
        if path.suffix.lower() in {".md", ".txt", ".json", ".yaml", ".yml"} and rel != "docs/code-availability-text.md":
            for pattern in OVERCLAIM_PATTERNS:
                for match in pattern.finditer(text):
                    findings.append(Finding("error", "scope-overclaim", rel, "source-only release promises artifacts that may be absent", line_number(text, match.start())))
        if path.suffix.lower() in {".py", ".sh", ".bash"} and rel not in {
            "scripts/release_audit.py", "scripts/list_experiments.py",
            "scripts/create_release_manifest.py", "scripts/validate_run_manifest.py"
        } and not rel.startswith(("release_tools/", "tests/")):
            for label, pattern in KNOWN_SCIENTIFIC_ANTI_PATTERNS:
                for match in pattern.finditer(text):
                    findings.append(Finding("error", "scientific-anti-pattern", rel, label, line_number(text, match.start())))

    required = [
        "README.md", "LICENSE", "CITATION.cff",
        "docs/repository-scope.md", "docs/scientific-contract.md",
        "docs/experiment-map.md", "docs/reproducibility.md",
        "docs/code-availability-text.md", "docs/release-checklist.md",
        "release/release_scope.json", "release/scientific_contract.json",
        "release/experiment_registry.json", "release/implementation_map.json",
        "release/publication.json", "release/schemas/run_manifest.schema.json",
        "release_tools/guardrails.py", "release_tools/hashing.py",
        "scripts/create_release_manifest.py", "scripts/validate_run_manifest.py",
        "scripts/list_experiments.py", ".github/workflows/quality.yml",
        "tests/test_release_files.py", "tests/test_release_guardrails.py",
        "tests/test_release_run_manifest.py",
    ]
    for rel in required:
        if not (root / rel).is_file():
            findings.append(Finding("error", "missing-release-file", rel, "required release file is missing"))

    environment_candidates = (
        "pyproject.toml", "requirements.txt", "environment.yml", "environment.yaml",
        "conda-lock.yml", "uv.lock", "poetry.lock",
    )
    if not any((root / rel).is_file() for rel in environment_candidates):
        findings.append(Finding(
            "error", "missing-environment-spec", ".",
            "provide pyproject.toml, requirements.txt, environment.yml, conda-lock.yml, uv.lock, or poetry.lock",
        ))

    for dataset, paths in entrypoints.items():
        if not paths:
            findings.append(Finding("error", "missing-dataset-entrypoint", dataset, "no executable Python entrypoint was detected"))

    readme = root / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        for required_phrase in ("source-code release", "journal", "not included", "official test"):
            if required_phrase.lower() not in text.lower():
                findings.append(Finding("error", "readme-scope", "README.md", f"missing required scope phrase: {required_phrase!r}"))
        for placeholder in (
            "Journal article link pending",
            "Journal Supplementary Information link pending",
            "DOI: pending",
        ):
            if placeholder.lower() in text.lower():
                findings.append(Finding("error", "publication-link", "README.md", f"replace publication placeholder before release: {placeholder!r}"))

    if not (root / "LICENSE").is_file():
        findings.append(Finding("error", "license", "LICENSE", "an approved public software license is required"))
    if (root / "LICENSE_REVIEW_REQUIRED.md").exists():
        findings.append(Finding("error", "license", "LICENSE_REVIEW_REQUIRED.md", "license review is unresolved"))

    citation = root / "CITATION.cff"
    if citation.is_file():
        citation_text = citation.read_text(encoding="utf-8")
        if "{{" in citation_text or "pending" in citation_text.lower():
            findings.append(Finding("error", "citation-metadata", "CITATION.cff", "citation metadata contains an unresolved placeholder"))

    publication = root / "release" / "publication.json"
    if publication.is_file():
        try:
            metadata = json.loads(publication.read_text(encoding="utf-8"))
            for field in ("article_url", "supplement_url"):
                value = metadata.get(field)
                parsed = urlparse(value) if isinstance(value, str) else None
                if not value or parsed is None or parsed.scheme != "https" or not parsed.netloc:
                    findings.append(Finding("error", "publication-link", publication.relative_to(root).as_posix(), f"{field} must be a final HTTPS URL"))
            doi = metadata.get("doi")
            if not isinstance(doi, str) or re.fullmatch(r"10\.\d{4,9}/\S+", doi.strip()) is None:
                findings.append(Finding("error", "publication-doi", publication.relative_to(root).as_posix(), "doi must be a final DOI such as 10.1234/example"))
        except (json.JSONDecodeError, OSError) as exc:
            findings.append(Finding("error", "publication-json", publication.relative_to(root).as_posix(), str(exc)))

    contract = root / "release" / "scientific_contract.json"
    if contract.is_file():
        try:
            data = json.loads(contract.read_text(encoding="utf-8"))
            identity = data.get("method_identity", {})
            firewall = data.get("test_firewall", {})
            if identity.get("backbone_parameters_frozen_during_controller_development") is not True:
                findings.append(Finding("error", "contract", contract.relative_to(root).as_posix(), "backbone frozen requirement is not true"))
            if identity.get("prediction_head_frozen_during_controller_development") is not True:
                findings.append(Finding("error", "contract", contract.relative_to(root).as_posix(), "prediction head frozen requirement is not true"))
            if firewall.get("post_test_decisions_prohibited") is not True:
                findings.append(Finding("error", "contract", contract.relative_to(root).as_posix(), "post-test decisions must be prohibited"))
            role_names = [role.get("name") for role in data.get("data_roles", [])]
            if role_names != ["D_fit", "D_explore", "D_select", "D_confirm", "D_admit", "D_test"]:
                findings.append(Finding("error", "contract", contract.relative_to(root).as_posix(), "data-role order does not match the six-stage lifecycle"))
        except (json.JSONDecodeError, OSError) as exc:
            findings.append(Finding("error", "contract-json", contract.relative_to(root).as_posix(), str(exc)))

    for path in iter_files(root):
        if path.suffix.lower() == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, UnicodeDecodeError) as exc:
                findings.append(Finding("error", "python-syntax", path.relative_to(root).as_posix(), str(exc), getattr(exc, "lineno", None)))

    bash = shutil.which("bash")
    if bash:
        for path in iter_files(root):
            is_shell = path.suffix.lower() in {".sh", ".bash"} or path.name.lower().endswith(".env.example")
            if not is_shell:
                continue
            result = subprocess.run([bash, "-n", str(path)], text=True, capture_output=True)
            if result.returncode != 0:
                findings.append(Finding("error", "bash-syntax", path.relative_to(root).as_posix(), result.stderr.strip() or "bash -n failed"))

    return findings, entrypoints


def implementation_markdown(entrypoints: dict[str, list[str]]) -> str:
    lines = ["# Implementation status", "", "Generated by `scripts/release_audit.py`.", "", "| Dataset | Detected executable entrypoints |", "|---|---|"]
    for dataset, paths in entrypoints.items():
        display = "<br>".join(f"`{path}`" for path in paths[:5]) if paths else "**MISSING**"
        lines.append(f"| {dataset} | {display} |")
    lines.extend(["", "Detection verifies that a plausible executable Python entrypoint exists. It does not establish that the full experiment has been run or that its numerical output matches the publication.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a CROSS-MORSE code-only public release.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--strict", action="store_true", help="return nonzero when any error is found")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    findings, entrypoints = audit(root)
    (root / "IMPLEMENTATION_STATUS.md").write_text(implementation_markdown(entrypoints), encoding="utf-8")

    report = {
        "schema_version": "1.0",
        "root": str(root),
        "entrypoints": entrypoints,
        "findings": [asdict(item) for item in findings],
        "blocking_count": sum(item.severity == "error" for item in findings),
    }
    output = args.json_output
    if output:
        output = output if output.is_absolute() else root / output
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if findings:
        for item in findings:
            location = f"{item.path}:{item.line}" if item.line else item.path
            print(f"{item.severity.upper():5s} [{item.code}] {location} - {item.message}")
    else:
        print("public release audit passed")

    if args.strict and any(item.severity == "error" for item in findings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
