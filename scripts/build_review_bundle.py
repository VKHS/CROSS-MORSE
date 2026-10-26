#!/usr/bin/env python3
"""Create a portable, analysis-only review ZIP from a completed run.

Copies saved observations, decision records, identity manifests and an
analysis-only public source subset. It does not copy the full research package
or its bundled examples, create observations, overwrite a run, or unpickle weights.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from cross_morse.review import (EvidenceError, NumericEvidence, digest, replay_results,
                                evidence_markdown, json_read, safe_file)
from cross_morse.provenance import current_source_sha256
from scripts._release_common import GENERATED_SUFFIXES, files, public_excluded, write_integrity
from scripts._paper_common import runtime_source_fingerprint

CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".pkl", ".pickle"}


def omitted_review_source(relative):
    """The public research package and a numeric review snapshot have different scopes."""
    relative = Path(relative)
    return (public_excluded(relative) or relative.parts[:1] == ("examples",)
            or relative.suffix.lower() in GENERATED_SUFFIXES or relative.name.endswith(".tar.gz"))


def describe_source_subset(source, scope):
    """Replace only distribution metadata; computational source bytes stay exact."""
    (source / "SOURCE_SNAPSHOT_SCOPE.json").write_text(json.dumps(scope, indent=2, sort_keys=True) + "\n")
    for relative in ("release/release_scope.json", "release/coverage.json"):
        path = source / relative
        if not path.is_file():
            continue
        data = json_read(path)
        data.update(source_snapshot_scope="analysis-only source subset; not the complete research-source package",
                    bundled_synthetic_examples_included=False,
                    bundled_example_run_roots=[], bundled_example_inventories=None,
                    complete_bundled_resume_artifacts=False,
                    full_package_audit_applicable=False,
                    artifact_policy="Only this export's numerical evidence is included in ../run. The full release's examples, checkpoint assets and generated example views are intentionally omitted from this source subset.")
        if relative.endswith("release_scope.json"):
            data["release_type"] = "analysis-source-subset"
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    (source / "README.md").write_text('''# Analysis-only source snapshot

This directory is the source subset accompanying one numerical review bundle.
Its runtime Python files are copied byte-for-byte from the exporting source.
`SOURCE_SNAPSHOT_SCOPE.json` records the runtime identity, omissions and metadata
adaptations. The complete research release's seven synthetic example runs,
their inventories/generated figures, and binary source fixtures are omitted.
No trained checkpoint is included anywhere in this review bundle.

From the enclosing `cross_morse_review` directory:

```bash
python -m pip install -r source/requirements-review.txt
python -B source/scripts/verify_review_bundle.py .
python -B source/scripts/reproduce_results.py run
```

`RELEASE_MANIFEST.json` and `SHA256SUMS.txt` identify this **subset** only; their
presence does not make it the complete research package. Use the bundle
verification command above. Full-package targets such as `make verify`,
`make audit-artifacts` and `make reproduce` require intentionally omitted files
and are not validation commands for this snapshot. Retained documentation
describes the complete research release where stated; its seven bundled examples
are available in that full release, not inside this compact export.
''')


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = json_read(root / "REVIEW_BUNDLE_MANIFEST.json")
    if manifest.get("schema_version") != "review-bundle-2":
        raise EvidenceError("Unsupported review bundle schema")
    if manifest.get("checkpoints_included") is not False:
        raise EvidenceError("Analysis-only review bundles cannot include checkpoints")
    paths = set()
    for record in manifest["files"]:
        relative = record["path"]
        if not isinstance(relative, str):
            raise EvidenceError("Review bundle file paths must be strings")
        if (Path(relative).suffix.lower() in CHECKPOINT_SUFFIXES
                or Path(relative).parts[:2] == ("source", "examples")):
            raise EvidenceError("Analysis-only review bundles cannot include checkpoint assets or bundled example archives")
        if relative in paths or relative == "REVIEW_BUNDLE_MANIFEST.json":
            raise EvidenceError("Repeated/circular review bundle file")
        paths.add(relative)
        p = safe_file(root, relative)
        if p.stat().st_size != record["bytes"] or digest(p) != record["sha256"]:
            raise EvidenceError(f"Review bundle hash mismatch: {relative}")
    if any(p.is_symlink() for p in root.rglob("*")):
        raise EvidenceError("Symlinks are not allowed in review bundles")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
              and p.name != "REVIEW_BUNDLE_MANIFEST.json" and "__pycache__" not in p.parts}
    if actual != paths:
        raise EvidenceError("Review bundle has missing or unexpected files")
    return manifest


def build_bundle(run_root, output, *, skip_selection=False):
    run_root, output = Path(run_root).resolve(), Path(output).absolute()
    if output.suffix.lower() != ".zip" or output.exists() or output.is_symlink():
        raise EvidenceError("Choose a new .zip output; existing files are never overwritten")
    if output.resolve().is_relative_to(run_root) or output.resolve().is_relative_to(SOURCE):
        raise EvidenceError("Review ZIP must be outside the run and source trees")
    e = NumericEvidence(run_root)
    if current_source_sha256() != e.identities["source_sha256"]:
        raise EvidenceError("Run the exporter from the exact source snapshot of this run")
    report = replay_results(run_root, selection=not skip_selection)
    required = set(report["verified_numeric_files"])
    required |= {"environment.json", "prepared/metadata.json", "train/model_inventory.json",
                 "select/entry_complete.json", "select/candidate_ledger.csv", "confirm/contrast_registry.json",
                 "admit/pre_admission_lock.json", "provenance/frozen_models.json", "provenance/evaluator.py"}
    # Source-replay can still verify selection when export validation skipped it.
    required.add("select/unit_scores.npz")
    if "select/metric_observations.npz" in e.sealed:
        required.add("select/metric_observations.npz")
    for relative in e.sealed:
        if relative.startswith("fit/") and Path(relative).name in {
            "audit.json", "calibration_partition.json", "calibration_assessment.json", "calibration_assessment.npz"}:
            required.add(relative)
    summary = e.json("final/results_summary.json")
    for candidate in (summary["comparator"], summary["deployed"]):
        required.add(f"predictions/{candidate['candidate_id']}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cm-review-", dir=output.parent) as temp:
        staging = Path(temp) / "cross_morse_review"
        staging.mkdir()
        for relative in sorted(required):
            source = e.file(relative)
            target = staging / "run" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if digest(target) != e.sealed[relative]["sha256"]:
                raise EvidenceError("Source changed during evidence export")
        for source in sorted((run_root / "provenance/stages").glob("*.json")):
            target = staging / "run/provenance/stages" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        omitted_paths = []
        omitted_example_bytes = 0
        for source in files(SOURCE):
            relative = source.relative_to(SOURCE)
            if omitted_review_source(relative):
                omitted_paths.append(relative.as_posix())
                if relative.parts[:1] == ("examples",):
                    omitted_example_bytes += source.stat().st_size
                continue
            target = staging / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        if runtime_source_fingerprint(staging / "source") != e.identities["source_sha256"]:
            raise EvidenceError("Copied runtime source differs from the run identity")
        source_scope = {"schema_version": "analysis-source-subset-1",
            "scope": "runtime and review source plus documentation; not the complete research-source package",
            "runtime_source_sha256": e.identities["source_sha256"], "computational_source_rewritten": False,
            "full_research_package_included": False, "full_package_audit_applicable": False,
            "omitted_source_roots": ["examples/"],
            "omitted_example_run_directories": sorted({"/".join(Path(p).parts[:3]) for p in omitted_paths
                if Path(p).parts[:2] == ("examples", "runs") and len(Path(p).parts) > 3}),
            "omitted_example_bytes": omitted_example_bytes, "omitted_source_file_count": len(omitted_paths),
            "omitted_binary_source_assets": [p for p in omitted_paths if Path(p).parts[:1] != ("examples",)
                and (Path(p).suffix.lower() in GENERATED_SUFFIXES or p.endswith(".tar.gz"))],
            "metadata_adapted": ["README.md", "release/release_scope.json", "release/coverage.json"],
            "source_snapshot_integrity_scope": "all retained subset files, without a claim to full-package structure"}
        describe_source_subset(staging / "source", source_scope)
        write_integrity(staging / "source", "public")
        (staging / "replay_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (staging / "replay_report.md").write_text(evidence_markdown(report))
        (staging / "README.md").write_text('''# CROSS-MORSE review evidence

This is an **analysis-only** package exported from one completed run.
It contains test observations (including official test labels), numerical
stage records, model identities and an analysis-only public source subset.
Check your data-provider sharing terms before publishing an official-data bundle.
It does not contain the trained checkpoints or the complete prepared graph.
The full research package's bundled synthetic example runs, their generated
views and binary source fixtures are deliberately omitted too. See
`source/SOURCE_SNAPSHOT_SCOPE.json` for the exact source-subset scope.

From this directory:

```bash
python -m pip install -r source/requirements-review.txt
python -B source/scripts/verify_review_bundle.py .
python -B source/scripts/reproduce_results.py run
```

No PyTorch import, model training, or pickle loading is needed for numeric replay.
A PASS verifies these submitted observations and arithmetic, not historical
execution, the paper's reported numbers, or independent-graph coverage.
A synthetic bundle is a software example, not benchmark evidence.
The source checksum manifest covers the retained subset. Full-package audits
and bundled-example Make targets require the separate complete research release.

For checkpoint evaluation, use `source/scripts/evaluate_checkpoints.py` against
the separate full run and its exact source snapshot. See the source review guide.
''')
        bundle_files = [{"path": p.relative_to(staging).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
                        for p in sorted(staging.rglob("*")) if p.is_file()]
        manifest = {"schema_version": "review-bundle-2", "run_identities": e.identities,
                    "dataset": report["dataset"], "scientific_run": report["scientific_run"],
                    "checkpoints_included": False, "contains_test_labels": True,
                    "source_snapshot_scope": source_scope,
                    "unshipped_sealed_artifacts": sorted(set(e.sealed) - required),
                    "exporter_selection_replayed": not skip_selection, "files": bundle_files,
                    "scope": "integrity and numerical auditability; no externally authenticated history"}
        (staging / "REVIEW_BUNDLE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        verify_bundle(staging)
        # Validate the copied records, not only the originals.
        copied_report = replay_results(staging / "run", selection=not skip_selection)
        if copied_report != report:
            raise EvidenceError("Copied evidence changed during export")
        if current_source_sha256() != e.identities["source_sha256"]:
            raise EvidenceError("Source changed while exporting the evidence")
        archive_path = Path(temp) / "payload.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for source in sorted(staging.rglob("*")):
                if source.is_file():
                    info = zipfile.ZipInfo("cross_morse_review/" + source.relative_to(staging).as_posix(), (2020,1,1,0,0,0))
                    info.external_attr = 0o100644 << 16
                    z.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        with zipfile.ZipFile(archive_path) as z:
            if z.testzip() is not None:
                raise EvidenceError("ZIP validation failed")
        # Same-filesystem hard link publishes atomically without clobbering a race.
        os.link(archive_path, output)
    return {"path": str(output), "sha256": digest(output), "status": "PASS",
            "dataset": report["dataset"], "scientific_run": report["scientific_run"],
            "source_and_run_unmodified": True, "checkpoints_included": False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--skip-selection-replay", action="store_true")
    args = p.parse_args(argv)
    try:
        print(json.dumps(build_bundle(args.run_root, args.output, skip_selection=args.skip_selection_replay), indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
