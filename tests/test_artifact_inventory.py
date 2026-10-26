"""Byte-level audit regressions; test fixtures are not experiment results."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import artifact_inventory as inv
from cross_morse import provenance as pv


def npy_bytes(dtype="<i8", shape=(3,), payload=None):
    """Write a tiny own-format NPY fixture with no NumPy dependency."""
    if payload is None:
        payload = struct.pack("<3q", 0, 1, 2)
    header = repr({"descr": dtype, "fortran_order": False, "shape": shape}).encode()
    header += b" " * ((64 - ((10 + len(header) + 1) % 64)) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + payload


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def chain_fixture(root):
    """A byte-integrity fixture deliberately uses opaque, non-executable .pt bytes.

    The inventory must hash them without deserializing. This fixture exercises
    verify_run_bytes only; positive numerical-structure coverage uses the real
    bundled software example below.
    """
    config = {"dataset": "synthetic", "scientific_run": False}
    write_json(root / "config.json", config)
    (root / "prepared").mkdir()
    (root / "prepared/splits.npz").write_bytes(b"split identity test fixture")
    (root / "provenance").mkdir()
    shutil.copyfile(inv.SOURCE / "cross_morse/pipeline.py", root / "provenance/evaluator.py")
    (root / "train").mkdir()
    (root / "train/plant_seed1_best.pt").write_bytes(b"opaque checkpoint fixture; never deserialize")
    checkpoint = pv._artifact(root, "train/plant_seed1_best.pt")
    baseline = {"schema_version": pv.SCHEMA_VERSION, "captured_after_stage": "train",
                "model_key": "model", "head_prefix": "head.", "hash_method": "fixture recorded values",
                "checkpoints": {checkpoint["path"]: {"artifact": checkpoint, "backbone_sha256": "1"*64,
                    "head_sha256": "2"*64, "backbone_tensors": 1, "head_tensors": 1}}}
    write_json(root / pv.DEFAULT_MODEL_BASELINE, baseline)
    identity = {"source_sha256": pv.current_source_sha256(), "config_sha256": pv._json_hash(config),
                "unit_sha256": pv.sha256_file(root / "prepared/splits.npz")}
    sealed = {}
    for index, stage in enumerate(pv.STAGES):
        if stage == "prepare":
            paths = ["config.json", "prepared/splits.npz", "provenance/evaluator.py"]
        elif stage == "train":
            paths = ["train/plant_seed1_best.pt", pv.DEFAULT_MODEL_BASELINE]
        elif stage == "evaluate":
            paths = sorted(pv.FINAL_EVALUATION_OUTPUTS)
            for relative in paths:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"opaque final fixture")
        else:
            relative = f"{stage}/fixture.json"
            write_json(root / relative, {"stage": stage})
            paths = [relative]
        outputs = [pv._artifact(root, p) for p in paths]
        audit = None
        if index:
            desc = pv._artifact(root, pv.DEFAULT_MODEL_BASELINE)
            audit = dict(inv._baseline_audit(root, desc, {})[1],
                         scope="post_train_capture" if stage == "train" else "unchanged_since_train")
        manifest = {"schema_version": pv.SCHEMA_VERSION, "stage": stage, "stage_index": index,
                    "scientific_role": pv.SCIENTIFIC_ROLES[stage], "dataset": "synthetic",
                    "created_utc": "2026-01-01T00:00:00+00:00", **identity,
                    "unit_manifest_path": pv.UNIT_MANIFEST_PATH,
                    "parent_manifest": None if index == 0 else pv._artifact(root, pv.stage_manifest_path(root, pv.STAGES[index-1])),
                    "inputs": sorted(sealed.values(), key=lambda v: v["path"]), "outputs": outputs,
                    "model_audit": audit,
                    "evaluator": pv._artifact(root, "provenance/evaluator.py") if stage == "evaluate" else None,
                    "claim_scope": "unit-test fixture, not scientific evidence"}
        write_json(pv.stage_manifest_path(root, stage), manifest)
        sealed.update({r["path"]: r for r in outputs})


class ArtifactInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "run"
        self.root.mkdir()
        chain_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_byte_verification_does_not_deserialize_opaque_checkpoint(self):
        result = inv.verify_run_bytes(self.root)
        self.assertEqual(len(result[5]), 8)
        self.assertIn("train/plant_seed1_best.pt", result[3])

    def test_changed_checkpoint_rejected_on_every_call(self):
        inv.verify_run_bytes(self.root)
        p = self.root / "train/plant_seed1_best.pt"
        p.write_bytes(p.read_bytes() + b"tamper")
        with self.assertRaisesRegex(pv.ProvenanceError, "identity/hash changed"):
            inv.verify_run_bytes(self.root)

    def test_missing_checkpoint_is_not_an_acceptable_analysis_bundle(self):
        (self.root / "train/plant_seed1_best.pt").unlink()
        with self.assertRaisesRegex(inv.InventoryError, "Missing"):
            inv.verify_run_bytes(self.root)

    def test_missing_final_stage_rejected(self):
        pv.stage_manifest_path(self.root, "evaluate").unlink()
        with self.assertRaisesRegex(inv.InventoryError, "all eight"):
            inv.verify_run_bytes(self.root)

    def test_parent_hash_checks_actual_manifest_bytes(self):
        p = pv.stage_manifest_path(self.root, "prepare")
        p.write_text(p.read_text() + " ")
        with self.assertRaisesRegex(inv.InventoryError, "Parent manifest hash"):
            inv.verify_run_bytes(self.root)

    def test_recorded_head_hash_disagreement_rejected(self):
        p = pv.stage_manifest_path(self.root, "evaluate")
        value = inv.read_json(p)
        value["model_audit"]["checkpoints"]["train/plant_seed1_best.pt"]["head_sha256_after"] = "3"*64
        write_json(p, value)
        with self.assertRaisesRegex(inv.InventoryError, "model/head hash linkage"):
            inv.verify_run_bytes(self.root)

    def test_evaluation_cannot_add_policy_output(self):
        p = pv.stage_manifest_path(self.root, "evaluate")
        value = inv.read_json(p)
        write_json(self.root / "policy_update.json", {"changed": True})
        value["outputs"].append(pv._artifact(self.root, "policy_update.json"))
        write_json(p, value)
        with self.assertRaisesRegex(inv.InventoryError, "allowlist"):
            inv.verify_run_bytes(self.root)

    def test_unsealed_completed_directory_file_rejected(self):
        (self.root / "explore/extra.npz").write_bytes(b"added")
        with self.assertRaisesRegex(pv.ProvenanceError, "inventory changed"):
            inv.verify_run_bytes(self.root)

    def test_shared_prediction_directory_frozen_in_complete_inventory(self):
        (self.root / "predictions").mkdir()
        (self.root / "predictions/extra.npy").write_bytes(npy_bytes())
        with self.assertRaisesRegex(inv.InventoryError, "unsealed predictions"):
            inv.verify_run_bytes(self.root)

    def test_path_traversal_and_symlink_rejected(self):
        for path in ("../outside", "/tmp/outside", "a/../b", "a\\b", "a//b"):
            with self.subTest(path=path), self.assertRaises(inv.InventoryError):
                inv.safe_file(self.root, path)
        (self.root / "alias").symlink_to(self.root / "config.json")
        with self.assertRaisesRegex(inv.InventoryError, "Symlink"):
            inv.safe_file(self.root, "alias")

    def test_numpy_header_rejects_object_and_truncation(self):
        obj = npy_bytes("|O8", (3,))
        with self.assertRaisesRegex(inv.InventoryError, "primitive"):
            inv.npy_header(io.BytesIO(obj), len(obj))
        good = npy_bytes()
        with self.assertRaisesRegex(inv.InventoryError, "payload size"):
            inv.npy_header(io.BytesIO(good[:-1]), len(good)-1)
        h, _ = inv.npy_header(io.BytesIO(good), len(good))
        self.assertEqual(h["shape"], [3])

    def test_safe_fixed_width_metadata_strings_are_supported(self):
        raw = npy_bytes("<U2", (1,), "RH".encode("utf-32-le"))
        h, _ = inv.npy_header(io.BytesIO(raw), len(raw))
        self.assertEqual(h["payload_bytes"], 8)

    def test_npz_duplicate_and_traversal_members_rejected(self):
        p = Path(self.temp.name) / "bad.npz"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("../row.npy", npy_bytes())
        with self.assertRaisesRegex(inv.InventoryError, "Unsafe"):
            inv.numeric_headers(p)
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("rows.npy", npy_bytes())
        self.assertEqual(inv.integer_array(p, "rows"), [0,1,2])

    def test_static_manifest_matches_actual_21_profiles(self):
        values = inv.static_manifests()
        profiles = values["experiments/profiles.json"]["profiles"]
        self.assertEqual(len(profiles), 21)
        self.assertEqual({p["profile"] for p in profiles}, {"smoke", "pilot", "full"})
        self.assertEqual(values["experiments/profiles.json"]["roman_campaign"]["official_split_indices"], list(range(10)))
        for relative, content in values.items():
            self.assertEqual(inv.read_json(inv.SOURCE / relative), content)
        smoke = [p for p in profiles if p["profile"] == "smoke"]
        self.assertTrue(all(p["data_kind"] == "synthetic" and p["scientific_run"] is False for p in smoke))

    def test_import_and_chain_verification_work_with_numpy_torch_blocked(self):
        script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'torch', 'numpy'}:
            raise AssertionError('numerical import forbidden: '+name)
sys.meta_path.insert(0, Block())
from scripts.artifact_inventory import verify_run_bytes, static_manifests
verify_run_bytes(sys.argv[1])
assert static_manifests()['experiments/profiles.json']['profile_count'] == 21
'''
        proc = subprocess.run([sys.executable, "-B", "-c", script, str(self.root)], cwd=inv.SOURCE,
                              text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_check_mode_does_not_rewrite_stale_report(self):
        report = {"sealed_artifact_count": 1, "dataset": "synthetic", "benchmark_dataset": "fixture"}
        output = Path(self.temp.name) / "inventory.json"
        output.write_text("stale\n")
        output.with_suffix(".md").write_text("stale md\n")
        with patch.object(inv, "build_inventory", return_value=report), patch.object(inv, "inventory_markdown", return_value="new md\n"), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(inv.main([str(self.root), "--output", str(output), "--check"]), 1)
        self.assertEqual(output.read_text(), "stale\n")
        self.assertEqual(output.with_suffix(".md").read_text(), "stale md\n")

    def test_bundled_example_complete_inventory_when_available(self):
        root = inv.SOURCE / "examples/runs/arxiv"
        if not root.is_dir():
            self.skipTest("Bundled examples are not present in this development tree")
        result = inv.build_inventory(root)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["scientific_run"])
        self.assertEqual(result["evidence_class"], "executed_synthetic_software_example")
        self.assertFalse(result["verification_scope"]["checkpoint_deserialization"])
        self.assertGreater(result["trajectories"]["transition_records"], 0)

    def test_complete_inventory_is_portable_after_moving_run(self):
        root = inv.SOURCE / "examples/runs/arxiv"
        if not root.is_dir():
            self.skipTest("Bundled examples are not present in this development tree")
        expected = inv.build_inventory(root)
        moved = Path(self.temp.name) / "another_location/renamed_run"
        shutil.copytree(root, moved)
        actual = inv.build_inventory(moved)
        self.assertEqual(actual, expected)
        self.assertEqual(inv.inventory_markdown(actual), inv.inventory_markdown(expected))
        self.assertNotIn(str(root.resolve()), inv.json_text(actual))
        self.assertNotIn(str(moved.resolve()), inv.json_text(actual))


if __name__ == "__main__":
    unittest.main()
