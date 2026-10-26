#!/usr/bin/env python3
"""Read-only source-release audit for the declared seven-dataset implementation.

Our CROSS-MORSE BSD release audit; see NOTICE. A passing
audit establishes source structure/integrity, not scientific reproduction.
"""
from __future__ import annotations
import argparse
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._release_common import (EXAMPLE_DATASETS, HASH, MANIFEST, ReleaseError, excluded,
                                    files, normalized_root, public_excluded, safe_relative,
                                    sha256_file, verify_checksums, verify_manifest)

DATASETS = {'ogbn-arxiv': 'arxiv', 'ogbn-products': 'products', 'ogbn-proteins': 'proteins', 'Roman-empire': 'roman_empire', 'PascalVOC-SP': 'pascalvoc_sp', 'COCO-SP': 'coco_sp', 'Peptides-func': 'peptides_func'}

REQUIRED = (
    "README.md", "LICENSE", "NOTICE", "requirements.txt", "run_cross_morse_arxiv.py",
    "run_cross_morse_arxiv.sh", "configs/public.env.example", "configs/arxiv_full.json",
    "configs/arxiv_pilot.json", "configs/arxiv_smoke.json", "release/coverage.json",
    "scripts/_release_common.py", "scripts/release_audit.py", "scripts/create_release_manifest.py",
    "scripts/build_release.py", "scripts/validate_run_manifest.py", "Makefile",
    "scripts/reproduce_results.py", "scripts/evaluate_checkpoints.py", "scripts/build_review_bundle.py",
    "scripts/verify_review_bundle.py", "scripts/compare_review_runs.py", "scripts/check_result_binding.py",
    "docs/DATASETS.md", "scripts/summarize_roman_splits.py", "run_cross_morse.py", "run_cross_morse.sh",
    "docs/REVIEWER_GUIDE.md", "docs/IMPLEMENTATION_CONTRACT.md", "docs/GETTING_STARTED.md",
    "CITATION.cff", "requirements-review.txt", "pyproject.toml",
    "environment/README.md", "environment/requirements-cpu.lock", "environment/observed_environment.json",
    "environment/check_environment.py", "environment/create_cpu_env.sh",
    "data/README.md", "data/sources.json", "experiments/README.md", "experiments/profiles.json",
    "examples/README.md", "docs/ARTIFACTS.md", "scripts/artifact_inventory.py", "scripts/reproduce_examples.py",
    "release/scientific_contract.json", "release/release_scope.json", "release/experiment_map.json",
    "release/schemas/stage_manifest.schema.json",
    "paper/README.md", "paper/registered_constants.json", "paper/reported_results.json", "paper/REPORTED_RESULTS.md",
    "paper/claims_ledger.md", "paper/KNOWN_DIFFERENCES.md",
    "scripts/_paper_common.py", "scripts/check_paper_constants.py", "scripts/compare_with_manuscript.py",
    "scripts/source_fingerprint.py", "scripts/render_reported_results.py",
    "scripts/plot_registry_figures.py", "scripts/plot_run_figures.py", "scripts/draw_diagrams.py",
    "docs/assets/README.md", "docs/assets/hero.svg", "docs/assets/reported_results.svg",
    *(f"docs/assets/{name}.svg" for name in ("architecture", "lifecycle", "control_depth", "policy_families", "review_workflow",
                                              "evidence_chain", "decision_chain")),
    "docs/METHOD_MAPPING.md", "docs/RESULTS.md", "docs/COMPUTE_BUDGET.md", "docs/REPRODUCIBILITY_CHECKLIST.md",
    "docs/VALIDATION.md", "docs/VALIDATION.json",
    # run-driven paper build, run-archive contract, manifests and the generated availability statement
    "AVAILABILITY.md", "CHANGELOG.md", "docs/RUN_ARCHIVES.md", "docs/ARRHENIUS.md", "docs/STATUS.md",
    "configs/arrhenius-site.env.example", "environment/arrhenius/capture_environment.sh",
    "paper/build/__init__.py", "paper/build/evidence.py", "paper/build/tables.py", "paper/build/figures.py",
    "paper/build/selection.py", "paper/sources.json",
    # measurement records of the experiments outside the eight-stage run
    "records/README.md", "records/specs.json", "records/index.json", "scripts/records.py",
    "scripts/build_paper.py", "scripts/archive_run.py", "scripts/ingest_run.py", "scripts/run_manifests.py",
    "scripts/availability.py", "runs/README.md", "manifests/registered/INDEX.json",
    "paper/generated/examples/BUILD.json", "paper/generated/examples/tables/T1.tex",
    "tests/test_paper_build.py", "tests/test_run_archives.py", "tests/test_availability.py",
    *(f"paper/bindings/{stem}_full.json" for stem in DATASETS.values()),
    *(f"cross_morse/{name}.py" for name in (
        "__init__", "core", "data", "training", "exploration", "learning", "policies", "statistics", "pipeline", "common", "provenance", "recovery", "scoring", "review", "reexecution", "datasets", "tasks", "evaluation", "runtime")),
    *(f"run_cross_morse_{stem}.{ext}" for stem in DATASETS.values() for ext in ("py", "sh")),
    *(f"configs/{stem}_{profile}.json" for stem in DATASETS.values() for profile in ("smoke", "pilot", "full")),
)
PRIVATE_PATTERNS = (
    re.compile(r"\bnaiss\d{4}-\d+-\d+(?:-gpu)?\b", re.I),
    # A project-disk path is private only when it names a concrete allocation; the generic
    # layout with a ${CM_STORAGE_PROJECT} placeholder is part of the public site profile.
    # The prefix is split so this rule does not flag its own source in an export.
    re.compile("/" + r"nobackup/[^\s\"']*(?:naiss|snic)\d{4}", re.I),
    re.compile(r"\b[A-Za-z0-9._-]+@login\.[A-Za-z0-9._-]+"),
)
TEXT_SUFFIXES = {".py", ".sh", ".bash", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".example", ".cff", ".lock"}


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    message: str
    line: int | None = None


def audit_examples(root: Path) -> list[Finding]:
    """Check shipped synthetic examples without loading tensors or running models.

    This verifies the artifact bytes and lifecycle links. Numerical replay and
    checkpoint reexecution remain separate, explicitly invoked checks.
    """
    directory = root / "examples/runs"
    if not directory.exists():
        return []
    findings = []
    if directory.is_symlink() or not directory.is_dir():
        return [Finding("example-artifacts", "examples/runs", "expected a regular example directory")]
    stages = ("prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate")
    for run in sorted(directory.iterdir()):
        relative = run.relative_to(root).as_posix()
        try:
            if run.name not in EXAMPLE_DATASETS or run.is_symlink() or not run.is_dir():
                raise ReleaseError("only the seven declared synthetic example directories may be shipped")
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            if (not isinstance(config, dict) or config.get("dataset") != "synthetic"
                    or config.get("benchmark_dataset") != EXAMPLE_DATASETS[run.name]
                    or config.get("profile") != "smoke" or config.get("scientific_run") is not False
                    or not isinstance(config.get("data"), dict)
                    or config["data"].get("download") is not False):
                raise ReleaseError("bundled runs must be synthetic smoke examples with downloads disabled, never benchmark results")
            config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            expected_paths = [run / f"provenance/stages/{i:02d}_{stage}.json" for i, stage in enumerate(stages)]
            if set((run / "provenance/stages").glob("*.json")) != set(expected_paths):
                raise ReleaseError("all eight sealed stage manifests are required")
            checked = {}
            def descriptor(value):
                if (not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}
                        or not safe_relative(value["path"]) or type(value["bytes"]) is not int
                        or value["bytes"] < 0 or not isinstance(value["sha256"], str)
                        or not HASH.fullmatch(value["sha256"])):
                    raise ReleaseError("malformed example artifact descriptor")
                path = run / value["path"]
                if (not path.resolve().is_relative_to(run.resolve()) or not path.is_file()
                        or excluded(path.relative_to(root))):
                    raise ReleaseError(f"sealed example artifact is missing, escaping or excluded: {value['path']}")
                if value["path"] not in checked:
                    checked[value["path"]] = {"path": value["path"], "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                if checked[value["path"]] != value:
                    raise ReleaseError(f"example artifact hash/size changed: {value['path']}")
                return value
            sealed, identities, previous = {}, None, None
            for stage, path in zip(stages, expected_paths):
                value = json.loads(path.read_text(encoding="utf-8"))
                current = [value.get(k) for k in ("source_sha256", "config_sha256", "unit_sha256")]
                if (value.get("stage") != stage or value.get("dataset") != "synthetic"
                        or any(not isinstance(v, str) or not HASH.fullmatch(v) for v in current)
                        or value.get("config_sha256") != config_hash
                        or value.get("parent_manifest") != previous):
                    raise ReleaseError(f"invalid example stage identities or parent link: {stage}")
                if identities is not None and current != identities:
                    raise ReleaseError("example source/configuration/units changed across stages")
                identities = current
                for item in value["inputs"]:
                    item = descriptor(item)
                    if stage != "prepare" and sealed.get(item["path"]) != item:
                        raise ReleaseError(f"example input has no earlier sealed output: {item['path']}")
                for item in value["outputs"]:
                    item = descriptor(item)
                    if item["path"] in sealed and sealed[item["path"]] != item:
                        raise ReleaseError(f"an earlier example output changed: {item['path']}")
                    sealed[item["path"]] = item
                previous = {"path": path.relative_to(run).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for required in ("config.json", "prepared/graph.npz", "prepared/splits.npz",
                             "final/results_summary.json", "final/test_predictions.npz"):
                if required not in sealed:
                    raise ReleaseError(f"complete example payload is missing: {required}")
            if not any(p.startswith("train/") and p.endswith(".pt") for p in sealed):
                raise ReleaseError("bundled examples must retain their sealed predictor checkpoints")
            if identities[2] != sealed["prepared/splits.npz"]["sha256"]:
                raise ReleaseError("example split bytes differ from the unit identity")
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            findings.append(Finding("example-artifacts", relative, str(exc)))
    return findings


def audit(root: Path, *, profile: str | None = None, require_integrity: bool = True) -> list[Finding]:
    root = normalized_root(root)
    findings: list[Finding] = []
    if profile is None:
        profile = "private" if (root / "configs/arrhenius.env.example").exists() else "public"
    if profile not in {"private", "public"}:
        raise ReleaseError(f"unknown profile: {profile}")
    for rel in REQUIRED:
        if not (root / rel).is_file():
            findings.append(Finding("missing-file", rel, "required implementation/release file is absent"))
    try:
        source_files = files(root, integrity=True)
    except ReleaseError as exc:
        return findings + [Finding("unsafe-file", ".", str(exc))]
    findings.extend(audit_examples(root))
    for path in source_files:
        rel = path.relative_to(root).as_posix()
        if profile == "public" and public_excluded(path.relative_to(root)):
            findings.append(Finding("private-profile", rel, "private profile/document must be excluded from public export"))
        if path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            findings.append(Finding("text-encoding", rel, "expected a UTF-8 text source"))
            continue
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=rel)
            except SyntaxError as exc:
                findings.append(Finding("python-syntax", rel, exc.msg, exc.lineno))
        if path.suffix == ".json":
            try:
                json.loads(text)
            except ValueError as exc:
                findings.append(Finding("json-syntax", rel, str(exc)))
        if profile == "public":
            for pattern in PRIVATE_PATTERNS:
                match = pattern.search(text)
                if match:
                    findings.append(Finding("private-infrastructure", rel,
                                            "literal allocation, storage path or login address in public source",
                                            text.count("\n", 0, match.start()) + 1))
    bash = shutil.which("bash")
    shell_files = [p for p in source_files if p.suffix in {".sh", ".bash"} or p.name.endswith(".env.example")]
    if shell_files and bash is None:
        findings.append(Finding("missing-validator", ".", "bash is required to validate shell syntax"))
    elif bash:
        for path in shell_files:
            result = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True, timeout=30)
            if result.returncode:
                findings.append(Finding("bash-syntax", path.relative_to(root).as_posix(), result.stderr.strip()))
    coverage_path = root / "release/coverage.json"
    if coverage_path.is_file():
        try:
            coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
            implemented = coverage.get("implemented_datasets")
            test_only = coverage.get("test_only_datasets")
            absent = coverage.get("not_implemented_datasets")
            if set(implemented or []) != set(DATASETS) or test_only != ["synthetic"] or absent != []:
                findings.append(Finding("coverage", "release/coverage.json", "all seven dataset adapters and the synthetic fixture must be declared"))
            for dataset in (*DATASETS, "synthetic"):
                rel = coverage.get("entrypoints", {}).get(dataset)
                if not safe_relative(rel) or not (root / rel).is_file():
                    findings.append(Finding("coverage", "release/coverage.json", f"missing declared entrypoint for {dataset}"))
            for dataset, stem in DATASETS.items():
                shell = coverage.get("shell_entrypoints", {}).get(dataset)
                if shell != f"run_cross_morse_{stem}.sh" or not (root / shell).is_file():
                    findings.append(Finding("coverage", "release/coverage.json", f"missing shell entrypoint for {dataset}"))
                for name in ("full", "pilot", "smoke"):
                    declared = coverage.get("profiles", {}).get(dataset, {}).get(name, {})
                    rel = declared.get("config")
                    if not safe_relative(rel) or not (root / rel).is_file():
                        findings.append(Finding("coverage", "release/coverage.json", f"missing {dataset}/{name} configuration"))
                        continue
                    config = json.loads((root / rel).read_text(encoding="utf-8"))
                    expected = "synthetic" if name == "smoke" else dataset
                    if declared.get("dataset") != expected or config.get("dataset") != expected or config.get("benchmark_dataset", dataset) != dataset:
                        findings.append(Finding("coverage", rel, f"incorrect dataset for {dataset}/{name}"))
                    if name == "smoke" and config.get("scientific_run") is not False:
                        findings.append(Finding("coverage", rel, "synthetic fixtures cannot be scientific benchmark runs"))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            findings.append(Finding("coverage", "release/coverage.json", f"invalid coverage declaration: {exc}"))
    if require_integrity:
        for error in verify_manifest(root) + verify_checksums(root):
            findings.append(Finding("integrity", MANIFEST, error))
    return findings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", choices=("private", "public"))
    parser.add_argument("--strict", action="store_true", help="return failure for any finding")
    parser.add_argument("--json-output", type=Path, help="optional report path outside the audited source tree")
    args = parser.parse_args(argv)
    try:
        root = normalized_root(args.root)
        output = args.json_output.expanduser().resolve() if args.json_output else None
        if output is not None and output.is_relative_to(root):
            raise ReleaseError("--json-output must be outside the source tree; the audit is read-only")
        findings = audit(root, profile=args.profile)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"schema_version": "1.0", "blocking_count": len(findings),
                                          "findings": [asdict(f) for f in findings]}, indent=2) + "\n", encoding="utf-8")
        for finding in findings:
            location = f"{finding.path}:{finding.line}" if finding.line else finding.path
            print(f"ERROR [{finding.code}] {location}: {finding.message}")
        if not findings:
            print("source structure, declared seven-dataset scope, syntax and complete checksums verified")
            print("This audit does not certify published numerical results or GPU execution.")
        return 1 if args.strict and findings else 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
