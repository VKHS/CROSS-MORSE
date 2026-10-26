"""Actual file, parent-chain, frozen-state and final-output provenance checks."""
from __future__ import annotations

from collections import Counter
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import runpy
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cross_morse import provenance as p


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)/"run"
        (self.root/"prepared").mkdir(parents=True)
        (self.root/"provenance").mkdir()
        self.config = {"dataset": "synthetic", "seed": 7}
        (self.root/"config.json").write_text(json.dumps(self.config))
        np.savez(self.root/"prepared"/"splits.npz", fit=np.array([0, 1]), test=np.array([2, 3]))
        np.savez(self.root/"prepared"/"clusters.npz", cores=np.array([0, 0, 1, 1]))
        shutil.copyfile(Path(p.__file__).with_name("pipeline.py"), self.root/"provenance"/"evaluator.py")
        self.identities = dict(dataset="synthetic", source_sha256=p.current_source_sha256(),
                               config_sha256=json_hash(self.config),
                               unit_sha256=p.sha256_file(self.root/"prepared"/"splits.npz"))
        self.outputs = {}
        self.state = {
            "encoder.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
            "encoder.count": torch.tensor(2, dtype=torch.int64),
            "encoder.zero": torch.tensor(.25),
            "encoder.empty": torch.empty(0, dtype=torch.bfloat16),
            "head.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "head.bias": torch.zeros(2),
        }

    def write_artifact(self, relative, content=b"fixture"):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return relative

    def seal_next(self, stage):
        previous = list(self.outputs)
        inputs = sorted({path for outputs in self.outputs.values() for path in outputs})
        extra = {}
        if stage == "prepare":
            outputs = ["config.json", "prepared/splits.npz", "prepared/clusters.npz", "provenance/evaluator.py"]
        elif stage == "train":
            checkpoint = self.root/"train"/"plant_seed7_best.pt"
            checkpoint.parent.mkdir()
            torch.save({"model": self.state}, checkpoint)
            p.capture_frozen_models(self.root, [checkpoint])
            outputs = ["train/plant_seed7_best.pt", p.DEFAULT_MODEL_BASELINE]
            extra["model_baseline"] = p.DEFAULT_MODEL_BASELINE
        elif stage == "evaluate":
            outputs = sorted(p.FINAL_EVALUATION_OUTPUTS)
            for relative in outputs:
                self.write_artifact(relative, b"synthetic test fixture, not numerical experiment evidence")
            extra.update(model_baseline=p.DEFAULT_MODEL_BASELINE, evaluator_path="provenance/evaluator.py")
        else:
            outputs = [self.write_artifact(f"{stage}/result.json", json.dumps({"stage": stage}).encode())]
            extra["model_baseline"] = p.DEFAULT_MODEL_BASELINE
        manifest = p.write_stage_manifest(self.root, stage, inputs=inputs, outputs=outputs,
                                           **self.identities, **extra)
        self.outputs[stage] = outputs
        self.assertEqual(manifest["stage_index"], len(previous))
        return manifest

    def seal_through(self, stage):
        for current in p.STAGES[:p.STAGES.index(stage)+1]:
            if current not in self.outputs:
                self.seal_next(current)

    def test_bfloat16_scalar_integer_empty_and_noncontiguous_state_hashes(self):
        digest = p.sha256_state_dict(self.state)
        self.assertEqual(digest, p.sha256_state_dict(dict(reversed(list(self.state.items())))))
        noncontiguous = {"weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).t()}
        self.assertEqual(p.sha256_state_dict(noncontiguous),
                         p.sha256_state_dict({"weight": noncontiguous["weight"].contiguous()}))
        changed = copy.deepcopy(self.state)
        changed["encoder.count"] += 1
        self.assertNotEqual(digest, p.sha256_state_dict(changed))
        changed = copy.deepcopy(self.state)
        changed["encoder.weight"][0, 0] += 1
        self.assertNotEqual(digest, p.sha256_state_dict(changed))

    def test_head_and_backbone_are_hashed_separately(self):
        original = p.frozen_model_hashes(self.state)
        changed = copy.deepcopy(self.state)
        changed["head.weight"][0, 0] += 1
        after = p.frozen_model_hashes(changed)
        self.assertEqual(original["backbone_sha256"], after["backbone_sha256"])
        self.assertNotEqual(original["head_sha256"], after["head_sha256"])

    def test_capture_and_verify_actual_bfloat16_checkpoint_and_reject_rebaseline(self):
        self.seal_through("train")
        audit = p.verify_frozen_models(self.root)
        self.assertTrue(audit["matches_post_train_baseline"])
        checkpoint = self.root/"train"/"plant_seed7_best.pt"
        p.capture_frozen_models(self.root, [checkpoint])  # Idempotent unchanged capture.
        changed = copy.deepcopy(self.state)
        changed["head.bias"] += 1
        torch.save({"model": changed}, checkpoint)
        with self.assertRaisesRegex(p.ProvenanceError, "identity/hash"):
            p.verify_frozen_models(self.root)
        with self.assertRaises(p.ProvenanceError):
            p.capture_frozen_models(self.root, [checkpoint])

    def test_complete_chain_has_eight_stages_actual_parents_and_scientific_roles(self):
        self.seal_through("evaluate")
        chain = p.verify_stage_chain(self.root, "evaluate", expected_source_sha256=self.identities["source_sha256"])
        self.assertEqual([item["stage"] for item in chain], list(p.STAGES))
        self.assertEqual(chain[1]["scientific_role"], "D_fit")
        self.assertEqual(chain[3]["scientific_role"], "D_explore")
        self.assertEqual(chain[-1]["scientific_role"], "D_test")
        for index, manifest in enumerate(chain[1:], 1):
            self.assertEqual(manifest["parent_manifest"]["sha256"],
                             p.sha256_file(p.stage_manifest_path(self.root, p.STAGES[index-1])))
        self.assertEqual({item["path"] for item in chain[-1]["outputs"]}, p.FINAL_EVALUATION_OUTPUTS)

    def test_empty_partial_chain_is_allowed_but_gaps_and_missing_required_stage_fail(self):
        self.assertEqual(p.verify_stage_chain(self.root), [])
        with self.assertRaisesRegex(p.ProvenanceError, "missing"):
            p.verify_stage_chain(self.root, "evaluate")
        self.seal_through("train")
        p.stage_manifest_path(self.root, "prepare").unlink()
        with self.assertRaisesRegex(p.ProvenanceError, "gap|parent"):
            p.verify_stage_chain(self.root)

    def test_parent_mutation_is_rechecked_instead_of_trusting_stored_digest(self):
        self.seal_through("train")
        parent = p.stage_manifest_path(self.root, "prepare")
        value = json.loads(parent.read_text())
        value["claim_scope"] += " changed"
        parent.write_text(json.dumps(value))
        with self.assertRaisesRegex(p.ProvenanceError, "Parent manifest"):
            p.verify_stage_chain(self.root)

    def test_existing_manifest_is_idempotent_but_cannot_be_replaced(self):
        self.seal_through("prepare")
        path = p.stage_manifest_path(self.root, "prepare")
        before = path.read_bytes()
        p.write_stage_manifest(self.root, "prepare", inputs=[], outputs=self.outputs["prepare"], **self.identities)
        self.assertEqual(path.read_bytes(), before)
        extra = self.write_artifact("prepared/extra.bin")
        with self.assertRaisesRegex(p.ProvenanceError, "immutable"):
            p.write_stage_manifest(self.root, "prepare", inputs=[],
                                   outputs=self.outputs["prepare"]+[extra], **self.identities)
        self.assertEqual(path.read_bytes(), before)

    def test_input_must_be_an_actual_earlier_sealed_output(self):
        self.seal_through("train")
        unknown = self.write_artifact("unsealed_input.npz")
        output = self.write_artifact("explore/result.json")
        with self.assertRaisesRegex(p.ProvenanceError, "earlier sealed output"):
            p.write_stage_manifest(self.root, "explore", inputs=[unknown], outputs=[output],
                                   model_baseline=p.DEFAULT_MODEL_BASELINE, **self.identities)
        self.assertFalse(p.stage_manifest_path(self.root, "explore").exists())

    def test_added_trajectory_in_completed_explore_is_rejected_but_atomic_tmp_is_ignored(self):
        self.seal_through("explore")
        temporary = self.root/"explore"/"trajectory_partial.npz.tmp"
        temporary.write_bytes(b"uncommitted atomic-write fixture")
        self.assertEqual(len(p.verify_stage_chain(self.root)), 3)
        np.savez(self.root/"explore"/"trajectory_99999.npz", action=np.array([0]), y=np.zeros((1, 25)))
        with self.assertRaisesRegex(p.ProvenanceError, "inventory changed: explore"):
            p.verify_stage_chain(self.root)

    def test_final_stage_cannot_declare_policy_or_decision_outputs(self):
        self.seal_through("admit")
        outputs = [self.write_artifact(relative) for relative in sorted(p.FINAL_EVALUATION_OUTPUTS)]
        with self.assertRaisesRegex(p.ProvenanceError, "allowlist"):
            p.write_stage_manifest(self.root, "evaluate", inputs=self.outputs["admit"],
                                   outputs=outputs+self.outputs["select"],
                                   model_baseline=p.DEFAULT_MODEL_BASELINE,
                                   evaluator_path="provenance/evaluator.py", **self.identities)
        self.assertFalse(p.stage_manifest_path(self.root, "evaluate").exists())

    def test_hidden_policy_mutation_is_rejected_even_when_not_declared_as_test_output(self):
        self.seal_through("admit")
        self.write_artifact("select/result.json", b"changed during final evaluation")
        outputs = [self.write_artifact(relative) for relative in sorted(p.FINAL_EVALUATION_OUTPUTS)]
        with self.assertRaisesRegex(p.ProvenanceError, "identity/hash"):
            p.write_stage_manifest(self.root, "evaluate", inputs=[], outputs=outputs,
                                   model_baseline=p.DEFAULT_MODEL_BASELINE,
                                   evaluator_path="provenance/evaluator.py", **self.identities)

    def test_actual_unit_and_configuration_corruption_are_rejected(self):
        self.seal_through("prepare")
        config_path = self.root/"config.json"
        before = config_path.read_bytes()
        config_path.write_text(json.dumps({"dataset": "synthetic", "seed": 8}))
        with self.assertRaisesRegex(p.ProvenanceError, "Configuration"):
            p.verify_stage_chain(self.root)
        config_path.write_bytes(before)
        self.write_artifact("prepared/splits.npz", b"changed split IDs")
        with self.assertRaisesRegex(p.ProvenanceError, "Unit/split"):
            p.verify_stage_chain(self.root)

    def test_source_identity_is_recomputed_not_only_compared_between_declarations(self):
        self.seal_through("prepare")
        with patch.object(p, "current_source_sha256", return_value="a"*64):
            with self.assertRaisesRegex(p.ProvenanceError, "Source identity"):
                p.verify_stage_chain(self.root)

    def test_repeated_inputs_hash_once_per_call_and_corruption_is_detected_next_call(self):
        self.seal_through("evaluate")
        counts = Counter()
        original = p.sha256_file

        def counted(path):
            counts[str(Path(path).resolve())] += 1
            return original(path)

        with patch.object(p, "sha256_file", side_effect=counted):
            p.verify_stage_chain(self.root)
        for relative in ("config.json", "prepared/splits.npz", "prepared/clusters.npz", "train/plant_seed7_best.pt"):
            self.assertEqual(counts[str((self.root/relative).resolve())], 1)
        self.write_artifact("prepared/clusters.npz", b"changed core IDs")
        with self.assertRaisesRegex(p.ProvenanceError, "identity/hash"):
            p.verify_stage_chain(self.root)

    def test_artifact_cannot_escape_the_run_directory(self):
        outside = Path(self.temporary.name)/"outside.bin"
        outside.write_bytes(b"outside fixture")
        with self.assertRaisesRegex(p.ProvenanceError, "escapes"):
            p.write_stage_manifest(self.root, "prepare", inputs=[], outputs=[outside], **self.identities)

    def test_standalone_validator_requires_actual_complete_chain(self):
        script = Path(p.__file__).resolve().parents[1]/"scripts"/"validate_run_manifest.py"
        main = runpy.run_path(str(script))["main"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root), "--require-complete", "--json"]), 1)
        self.seal_through("evaluate")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root), "--require-complete", "--json"]), 0)
        self.write_artifact("final/results.csv", b"corrupted report")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root), "--require-complete", "--json"]), 1)


if __name__ == "__main__":
    unittest.main()
