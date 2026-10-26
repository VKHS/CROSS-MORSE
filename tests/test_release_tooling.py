"""Release safety tests use disposable fixtures, never a real output/source tree."""
from __future__ import annotations
import json
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import zipfile

from scripts import build_release as exporter
from scripts._release_common import (ReleaseError, inventory, sha256_file, verify_checksums,
                                    verify_manifest, write_integrity)
from scripts.release_audit import REQUIRED, audit

PROJECT = Path(__file__).resolve().parents[1]


def fixture_tree(root: Path) -> Path:
    root.mkdir(parents=True)
    for rel in REQUIRED:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".py":
            text = '"""Disposable packaging fixture; no scientific execution."""\n'
        elif path.suffix == ".json":
            text = json.dumps({"dataset": "synthetic" if "smoke" in path.name else "ogbn-arxiv"}) + "\n"
        elif path.suffix == ".sh":
            text = "#!/bin/bash\nexit 0\n"
        elif path.name.endswith(".env.example"):
            text = 'CM_ACCOUNT=""\n'
        else:
            text = "Disposable release fixture.\n"
        path.write_text(text, encoding="utf-8")
    shutil.copyfile(PROJECT / "release/coverage.json", root / "release/coverage.json")
    for config in (PROJECT / "configs").glob("*.json"):
        shutil.copyfile(config, root / "configs" / config.name)
    (root / "run_cross_morse_arxiv.sh").chmod(0o755)
    (root / "configs/arrhenius.env.example").write_text('CM_ACCOUNT="fixture-allocation"\n', encoding="utf-8")
    write_integrity(root, "private")
    return root


def synthetic_example(root: Path) -> Path:
    """Tiny byte-level packaging fixture; its placeholder tensors are never loaded."""
    run = root / "examples/runs/arxiv"
    config = {"dataset": "synthetic", "benchmark_dataset": "ogbn-arxiv", "profile": "smoke",
              "scientific_run": False, "data": {"download": False}}
    payloads = {"config.json": json.dumps(config).encode(), "prepared/graph.npz": b"fixture graph",
                "prepared/splits.npz": b"fixture split", "train/seed_0/model.pt": b"fixture checkpoint",
                "final/test_predictions.npz": b"fixture observations", "final/qD_probabilities.npy": b"fixture probabilities",
                "final/results_summary.json": b'{"dataset":"synthetic"}'}
    for relative, content in payloads.items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    def descriptor(path):
        return {"path": path.relative_to(run).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    stages = ("prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate")
    sealed, previous = [], None
    for i, stage in enumerate(stages):
        selected = [name for name in payloads if
                    (stage == "prepare" and (name == "config.json" or name.startswith("prepared/")))
                    or (stage == "train" and name.startswith("train/"))
                    or (stage == "evaluate" and name.startswith("final/"))]
        outputs = [descriptor(run / name) for name in selected]
        value = {"stage": stage, "dataset": "synthetic", "source_sha256": "a" * 64,
                 "config_sha256": config_hash, "unit_sha256": sha256_file(run / "prepared/splits.npz"),
                 "parent_manifest": previous, "inputs": sealed[:], "outputs": outputs}
        path = run / f"provenance/stages/{i:02d}_{stage}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        previous = descriptor(path)
        sealed.extend(outputs)
    return run


class ManifestIntegrityTests(unittest.TestCase):
    def test_editable_install_metadata_is_not_a_release_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_tree(Path(tmp) / "source")
            before = inventory(root)
            for name in ("cross_morse.egg-info", "cross_morse-1.0.0.dist-info"):
                directory = root / name
                directory.mkdir()
                (directory / "METADATA").write_text("Generated installation metadata\n")
            self.assertEqual(inventory(root), before)
            self.assertEqual(verify_manifest(root), [])
            self.assertEqual(verify_checksums(root), [])

    def test_public_audit_can_include_its_own_privacy_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_tree(Path(tmp) / "source")
            (root / "configs/arrhenius.env.example").unlink()
            shutil.copyfile(PROJECT / "scripts/release_audit.py", root / "scripts/release_audit.py")
            write_integrity(root, "public")
            self.assertEqual(audit(root, profile="public"), [])

    def test_changed_missing_and_extra_files_are_rejected(self):
        for mutation in ("changed", "missing", "extra"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = fixture_tree(Path(tmp) / "source")
                self.assertEqual(verify_manifest(root) + verify_checksums(root), [])
                if mutation == "changed":
                    (root / "README.md").write_text("changed\n")
                elif mutation == "missing":
                    (root / "README.md").unlink()
                else:
                    (root / "unexpected.py").write_text("new_file = True\n")
                self.assertTrue(verify_manifest(root))
                self.assertTrue(verify_checksums(root))

    def test_manifest_and_checksum_files_are_bound_without_circularity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_tree(Path(tmp) / "source")
            manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text())
            paths = {x["path"] for x in manifest["files"]}
            self.assertNotIn("RELEASE_MANIFEST.json", paths)
            self.assertNotIn("SHA256SUMS.txt", paths)
            self.assertIn("RELEASE_MANIFEST.json", (root / "SHA256SUMS.txt").read_text())
            (root / "RELEASE_MANIFEST.json").write_text(json.dumps(dict(manifest, release_profile="public")))
            self.assertTrue(verify_checksums(root))

    def test_cache_files_do_not_change_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_tree(Path(tmp) / "source")
            before = inventory(root)
            cached = root / "cross_morse/__pycache__/module.pyc"
            cached.parent.mkdir()
            cached.write_bytes(b"cache")
            output = root / "outputs/generated.json"
            output.parent.mkdir()
            output.write_text("{}")
            (root / "temporary_checkpoint.pt").write_bytes(b"generated checkpoint")
            self.assertEqual(before, inventory(root))
            self.assertEqual(verify_manifest(root) + verify_checksums(root), [])

    def test_explicit_binary_test_fixtures_are_retained_and_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_tree(Path(tmp) / "source")
            asset = root / "tests/fixtures/tiny.npy"
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b"small source fixture")
            write_integrity(root, "private")
            self.assertIn("tests/fixtures/tiny.npy", {r["path"] for r in inventory(root)["files"]})
            self.assertEqual(verify_manifest(root) + verify_checksums(root), [])
            output = Path(tmp) / "with_fixture.zip"
            exporter.build_release(root, output)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.read("CROSS-MORSE/tests/fixtures/tiny.npy"), asset.read_bytes())

    def test_static_dataset_manifests_are_retained_but_raw_caches_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_tree(Path(tmp) / "source")
            for relative in ("data/README.md", "data/sources.json", "data/raw/large.npy", "data/download.json"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n")
            paths = {item["path"] for item in inventory(root)["files"]}
            self.assertIn("data/README.md", paths)
            self.assertIn("data/sources.json", paths)
            self.assertNotIn("data/raw/large.npy", paths)
            self.assertNotIn("data/download.json", paths)

    def test_complete_synthetic_examples_retain_and_hash_checkpoint_and_probability_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_tree(Path(tmp) / "source")
            run = synthetic_example(root)
            write_integrity(root, "private")
            self.assertEqual(audit(root), [])
            output = Path(tmp) / "examples.zip"
            exporter.build_release(root, output, profile="public")
            with zipfile.ZipFile(output) as archive:
                records = {r["path"] for r in json.loads(archive.read("CROSS-MORSE/RELEASE_MANIFEST.json"))["files"]}
                sums = archive.read("CROSS-MORSE/SHA256SUMS.txt").decode()
                for path in run.rglob("*"):
                    if path.is_file():
                        relative = path.relative_to(root).as_posix()
                        self.assertIn(relative, records)
                        self.assertIn(f"{sha256_file(path)}  {relative}\n", sums)
                        self.assertEqual(archive.read("CROSS-MORSE/" + relative), path.read_bytes())

    def test_benchmark_run_cannot_be_disguised_as_a_shipped_synthetic_example(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_tree(Path(tmp) / "source")
            run = synthetic_example(root)
            config = json.loads((run / "config.json").read_text())
            config.update(dataset="ogbn-arxiv", scientific_run=True, profile="full")
            (run / "config.json").write_text(json.dumps(config))
            write_integrity(root, "private")
            findings = audit(root)
            self.assertTrue(any(f.code == "example-artifacts" and "synthetic smoke" in f.message for f in findings))
            with self.assertRaisesRegex(ReleaseError, "synthetic smoke"):
                exporter.build_release(root, Path(tmp) / "should_not_exist.zip")
            self.assertFalse((Path(tmp) / "should_not_exist.zip").exists())

    def test_changed_or_missing_example_checkpoint_fails_even_after_rehashing_release(self):
        for mutation in ("changed", "missing", "missing_stage"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = fixture_tree(Path(tmp) / "source")
                run = synthetic_example(root)
                checkpoint = run / "train/seed_0/model.pt"
                if mutation == "changed":
                    checkpoint.write_bytes(b"different checkpoint")
                elif mutation == "missing":
                    checkpoint.unlink()
                else:
                    (run / "provenance/stages/01_train.json").unlink()
                write_integrity(root, "private")
                self.assertTrue(any(f.code == "example-artifacts" for f in audit(root)))

    def test_symlinks_cannot_import_files_from_outside_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_tree(Path(tmp) / "source")
            external = Path(tmp) / "outside.txt"
            external.write_text("must not be copied")
            (root / "linked.txt").symlink_to(external)
            with self.assertRaises(ReleaseError):
                inventory(root)


class ExportSafetyTests(unittest.TestCase):
    def test_rejects_inside_same_and_ancestor_paths_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ancestor = Path(tmp) / "parent.zip"
            root = fixture_tree(ancestor / "source")
            before = inventory(root)
            for destination in (root / "nested.zip", root, ancestor):
                with self.subTest(destination=destination), self.assertRaisesRegex(ReleaseError, "ancestor"):
                    exporter.build_release(root, destination, replace=True)
                self.assertEqual(inventory(root), before)
            self.assertFalse((root / "nested.zip").exists())

    def test_refuses_existing_output_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_tree(Path(tmp) / "source")
            target = Path(tmp) / "previous.zip"
            target.write_bytes(b"previous release")
            with self.assertRaisesRegex(ReleaseError, "already exists"):
                exporter.build_release(root, target)
            self.assertEqual(target.read_bytes(), b"previous release")

    def test_failed_zip_build_preserves_prior_output_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_tree(Path(tmp) / "source")
            before = inventory(root)
            target = Path(tmp) / "previous.zip"
            target.write_bytes(b"previous release")
            with mock.patch.object(exporter, "_write_zip", side_effect=ReleaseError("simulated write failure")):
                with self.assertRaisesRegex(ReleaseError, "simulated"):
                    exporter.build_release(root, target, replace=True)
            self.assertEqual(target.read_bytes(), b"previous release")
            self.assertEqual(inventory(root), before)

    def test_public_audit_failure_preserves_prior_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_tree(Path(tmp) / "source")
            # Assembled at runtime so that the public-source scan does not flag this test file itself.
            (root / "README.md").write_text("/" + "nobackup/proj/disk/" + "naiss" + "2099-1-1/fictional")
            write_integrity(root, "private")
            target = Path(tmp) / "previous.zip"
            target.write_bytes(b"previous release")
            with self.assertRaisesRegex(ReleaseError, "private-infrastructure"):
                exporter.build_release(root, target, profile="public", replace=True)
            self.assertEqual(target.read_bytes(), b"previous release")

    def test_no_clobber_is_preserved_when_output_appears_during_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_tree(Path(tmp) / "source")
            target = Path(tmp) / "raced.zip"
            original = exporter._write_zip
            def concurrent_output(stage, archive):
                original(stage, archive)
                target.write_bytes(b"other writer")
            with mock.patch.object(exporter, "_write_zip", side_effect=concurrent_output):
                with self.assertRaisesRegex(ReleaseError, "appeared"):
                    exporter.build_release(root, target)
            self.assertEqual(target.read_bytes(), b"other writer")

    def test_public_export_preserves_code_bytes_and_exact_archive_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_tree(Path(tmp) / "source")
            private = root / "docs/private/cluster.md"
            private.parent.mkdir(parents=True)
            private.write_text("private fixture note")
            write_integrity(root, "private")
            before = inventory(root)
            output = Path(tmp) / "public.zip"
            result = exporter.build_release(root, output, profile="public")
            with zipfile.ZipFile(output) as archive:
                names = archive.namelist()
                self.assertTrue(all(n.startswith("CROSS-MORSE/") for n in names))
                self.assertNotIn("CROSS-MORSE/configs/arrhenius.env.example", names)
                self.assertFalse(any("/docs/private/" in n for n in names))
                for item in before["files"]:
                    if item["path"].endswith((".py", ".sh")):
                        self.assertEqual(archive.read("CROSS-MORSE/" + item["path"]), (root / item["path"]).read_bytes())
                manifest = json.loads(archive.read("CROSS-MORSE/RELEASE_MANIFEST.json"))
                self.assertEqual(manifest["release_profile"], "public")
            self.assertFalse(result["computational_source_rewritten"])
            self.assertEqual(before, inventory(root))

    def test_archives_are_deterministic_and_replace_is_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_tree(Path(tmp) / "source")
            first, second = Path(tmp) / "first.zip", Path(tmp) / "second.zip"
            exporter.build_release(root, first)
            exporter.build_release(root, second)
            self.assertEqual(sha256_file(first), sha256_file(second))
            old = first.read_bytes()
            (root / "README.md").write_text("updated fixture\n")
            write_integrity(root, "private")
            exporter.build_release(root, first, replace=True)
            self.assertNotEqual(old, first.read_bytes())
            with zipfile.ZipFile(first) as archive:
                self.assertIsNone(archive.testzip())

    def test_audit_accepts_complete_seven_dataset_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_tree(Path(tmp) / "source")
            self.assertEqual(audit(root), [])


if __name__ == "__main__":
    unittest.main()
