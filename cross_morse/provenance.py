"""Artifact-verified provenance for the eight-stage dataset lifecycle.

The lifecycle/hash concepts adapt the supplied CROSS-MORSE BSD-3-Clause
release helpers; see this repository's attribution notices. This module adds
actual artifact/parent verification, immutable creation, BF16-safe tensor
hashing and a restricted final-evaluation output namespace. These are integrity
checks within the executable workflow, not a security sandbox or evidence that
the scientific results reproduce the publication.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


STAGES = ("prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate")
SCIENTIFIC_ROLES = {
    "prepare": "auxiliary", "train": "D_fit", "explore": "D_explore",
    "fit": "D_explore", "select": "D_select", "confirm": "D_confirm",
    "admit": "D_admit", "evaluate": "D_test",
}
FINAL_EVALUATION_OUTPUTS = frozenset({
    "final/q0_probabilities.npy", "final/qN_probabilities.npy",
    "final/qC_probabilities.npy", "final/qD_probabilities.npy",
    "final/results_summary.json", "final/paired_test_units.npz", "final/results.csv",
    "final/test_predictions.npz",
})
SCHEMA_VERSION = "1.1"
DEFAULT_MODEL_BASELINE = "provenance/frozen_models.json"
UNIT_MANIFEST_PATH = "prepared/splits.npz"
TEMPORARY_ARTIFACT_SUFFIXES = (".tmp",)
_EXCLUSIVE_STAGE_DIRECTORIES = {
    "prepare": "prepared", "train": "train", "explore": "explore", "fit": "fit",
    "select": "select", "confirm": "confirm", "admit": "admit", "evaluate": "final",
}
_HASH = re.compile(r"^[0-9a-f]{64}$")


class ProvenanceError(RuntimeError):
    """An immutable artifact or lifecycle provenance contract was violated."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def current_source_sha256() -> str:
    """Same code-file recipe as pipeline.source_fingerprint; no mutable outputs."""
    root = Path(__file__).resolve().parents[1]
    paths = sorted(root.glob("run_cross_morse*.py")) + sorted((root / "cross_morse").glob("*.py"))
    return _json_hash({str(path.relative_to(root)): sha256_file(path)
                       for path in paths if path.is_file()})


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProvenanceError(f"Cannot read provenance JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProvenanceError(f"Provenance JSON must be an object: {path}")
    return value


def _relative_file(root: Path, path: str | Path, *, must_exist: bool = True) -> tuple[str, Path]:
    path = Path(path)
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=must_exist)
        relative = resolved.relative_to(root.resolve()).as_posix()
    except (OSError, ValueError) as exc:
        raise ProvenanceError(f"Artifact is missing or escapes the run directory: {path}") from exc
    if must_exist and not resolved.is_file():
        raise ProvenanceError(f"Artifact is not a regular file: {path}")
    if relative in ("", "."):
        raise ProvenanceError("The run directory is not a file artifact")
    return relative, resolved


def _artifact(root: Path, path: str | Path) -> dict:
    relative, resolved = _relative_file(root, path)
    return {"path": relative, "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def _cached_artifact(root: Path, path: str | Path, cache: dict | None = None) -> dict:
    if cache is None:
        return _artifact(root, path)
    relative, resolved = _relative_file(root, path)
    if relative not in cache:
        cache[relative] = {"path": relative, "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}
    return cache[relative]


def _verify_artifact(root: Path, descriptor: Any, cache: dict | None = None) -> dict:
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256", "bytes"}:
        raise ProvenanceError("Artifact records require exactly path, sha256 and bytes")
    if (not isinstance(descriptor["path"], str) or Path(descriptor["path"]).is_absolute()
            or not isinstance(descriptor["sha256"], str) or not _HASH.fullmatch(descriptor["sha256"])
            or type(descriptor["bytes"]) is not int or descriptor["bytes"] < 0):
        raise ProvenanceError("Malformed artifact path, SHA-256 or size")
    actual = _cached_artifact(root, descriptor["path"], cache)
    if actual != descriptor:
        raise ProvenanceError(f"Artifact identity/hash changed: {descriptor['path']}")
    return actual


def _artifacts(root: Path, paths: Sequence[str | Path], cache: dict | None = None) -> list[dict]:
    if isinstance(paths, (str, bytes, Path)):
        raise ProvenanceError("Artifact paths must be a sequence of individual files")
    values = [_cached_artifact(root, path, cache) for path in paths]
    if len({item["path"] for item in values}) != len(values):
        raise ProvenanceError("Repeated artifact paths are not allowed")
    for item in values:
        if item["path"].startswith("provenance/stages/"):
            raise ProvenanceError("Stage manifests are linked as parents, not circular file artifacts")
    return sorted(values, key=lambda item: item["path"])


def _atomic_create_json(path: Path, value: dict) -> None:
    """Publish a complete JSON file without replacing an existing identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False,
                                         encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        # A hard link publishes atomically and fails if another writer already
        # created the destination. Unlike replace(), it cannot overwrite a seal.
        os.link(temporary, path)
    except FileExistsError as exc:
        raise ProvenanceError(f"Immutable provenance file already exists: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sha256_state_dict(state_dict_or_module: Any) -> str:
    """Canonical named-tensor hash, including dtype/shape and raw storage bytes.

    CPU contiguous bytes are viewed as uint8 *before* NumPy conversion. Thus
    actual BF16 parameters, zero-dimensional floating/integer buffers and empty
    tensors work without converting their values or reducing their precision.
    Tensor keys and length-prefixed metadata prevent ambiguous concatenation.
    """
    import torch
    state = (state_dict_or_module.state_dict() if hasattr(state_dict_or_module, "state_dict")
             else state_dict_or_module)
    if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
        raise ProvenanceError("Expected a module or a state dictionary with string keys")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise ProvenanceError(f"State entry {name!r} must be a tensor (including scalar buffers)")
        if value.layout != torch.strided or value.is_quantized:
            raise ProvenanceError(f"Unsupported sparse/quantized state entry: {name}")
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps({"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                              sort_keys=True, separators=(",", ":")).encode()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        for payload in (metadata, raw):
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def frozen_model_hashes(state_dict_or_module: Any, *, head_prefix: str = "head.") -> dict:
    state = (state_dict_or_module.state_dict() if hasattr(state_dict_or_module, "state_dict")
             else state_dict_or_module)
    if not isinstance(state, Mapping) or not head_prefix:
        raise ProvenanceError("A state dictionary and a nonempty head prefix are required")
    head = {name: tensor for name, tensor in state.items() if name.startswith(head_prefix)}
    backbone = {name: tensor for name, tensor in state.items() if not name.startswith(head_prefix)}
    if not head or not backbone:
        raise ProvenanceError("Checkpoint must contain both backbone and prediction-head tensors")
    return {"backbone_sha256": sha256_state_dict(backbone), "head_sha256": sha256_state_dict(head),
            "backbone_tensors": len(backbone), "head_tensors": len(head)}


def _checkpoint_hashes(path: Path, model_key: str, head_prefix: str) -> dict:
    import torch
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload[model_key]
    except Exception as exc:
        raise ProvenanceError(f"Cannot load tensor-only checkpoint {path}: {exc}") from exc
    return frozen_model_hashes(state, head_prefix=head_prefix)


def capture_frozen_models(run_root: str | Path, checkpoints: Sequence[str | Path], *,
                          baseline_path: str | Path = DEFAULT_MODEL_BASELINE,
                          model_key: str = "model", head_prefix: str = "head.") -> dict:
    """Capture immediately after train/D_fit, before exploration or controller fit.

    Existing baselines are never replaced. An identical request verifies and
    reuses the seal, which supports a crash between capture and stage sealing.
    """
    root = Path(run_root).resolve()
    artifacts = _artifacts(root, checkpoints)
    if not artifacts:
        raise ProvenanceError("At least one frozen checkpoint is required")
    _, destination = _relative_file(root, baseline_path, must_exist=False)
    if destination.exists():
        existing = load_frozen_models(root, baseline_path)
        if (existing["model_key"] != model_key or existing["head_prefix"] != head_prefix
                or set(existing["checkpoints"]) != {item["path"] for item in artifacts}):
            raise ProvenanceError("Frozen baseline request changed after capture")
        verify_frozen_models(root, baseline_path)
        return existing
    entries = {}
    for artifact in artifacts:
        entries[artifact["path"]] = {"artifact": artifact,
                                    **_checkpoint_hashes(root / artifact["path"], model_key, head_prefix)}
    result = {"schema_version": SCHEMA_VERSION, "captured_after_stage": "train",
              "model_key": model_key, "head_prefix": head_prefix, "checkpoints": entries,
              "hash_method": "sorted named tensors; dtype/shape and length-prefixed CPU uint8 storage bytes"}
    _atomic_create_json(destination, result)
    return result


def load_frozen_models(run_root: str | Path, baseline_path: str | Path = DEFAULT_MODEL_BASELINE) -> dict:
    root = Path(run_root).resolve()
    _, path = _relative_file(root, baseline_path)
    value = _read_json(path)
    required = {"schema_version", "captured_after_stage", "model_key", "head_prefix", "checkpoints", "hash_method"}
    if (set(value) != required or value["schema_version"] != SCHEMA_VERSION
            or value["captured_after_stage"] != "train" or not isinstance(value["checkpoints"], dict)
            or not value["checkpoints"] or not isinstance(value["model_key"], str)
            or not isinstance(value["head_prefix"], str) or not value["head_prefix"]):
        raise ProvenanceError("Malformed post-training frozen-model baseline")
    return value


def verify_frozen_models(run_root: str | Path, baseline_path: str | Path = DEFAULT_MODEL_BASELINE,
                         *, _artifact_cache: dict | None = None) -> dict:
    """Re-read checkpoint bytes and logical backbone/head state at every call."""
    root = Path(run_root).resolve()
    baseline = load_frozen_models(root, baseline_path)
    comparisons = {}
    for relative, before in baseline["checkpoints"].items():
        if not isinstance(before, dict) or before.get("artifact", {}).get("path") != relative:
            raise ProvenanceError("Malformed frozen checkpoint record")
        _verify_artifact(root, before["artifact"], _artifact_cache)
        after = _checkpoint_hashes(root / relative, baseline["model_key"], baseline["head_prefix"])
        if any(before.get(key) != value for key, value in after.items()):
            raise ProvenanceError(f"Frozen backbone/head state changed: {relative}")
        comparisons[relative] = {
            "checkpoint_sha256": before["artifact"]["sha256"],
            "backbone_sha256_before": before["backbone_sha256"],
            "backbone_sha256_after": after["backbone_sha256"],
            "head_sha256_before": before["head_sha256"],
            "head_sha256_after": after["head_sha256"],
        }
    return {"baseline": _cached_artifact(root, baseline_path, _artifact_cache), "checkpoints": comparisons,
            "matches_post_train_baseline": True}


def stage_manifest_path(run_root: str | Path, stage: str) -> Path:
    if stage not in STAGES:
        raise ProvenanceError(f"Unknown executable stage: {stage}")
    return Path(run_root).resolve() / "provenance" / "stages" / f"{STAGES.index(stage):02d}_{stage}.json"


def _verify_identities(root: Path, source_sha256: str, config_sha256: str, unit_sha256: str,
                       cache: dict | None = None) -> None:
    for name, value in (("source", source_sha256), ("config", config_sha256), ("unit", unit_sha256)):
        if not isinstance(value, str) or not _HASH.fullmatch(value):
            raise ProvenanceError(f"{name} identity must be a lowercase SHA-256")
    if source_sha256 != current_source_sha256():
        raise ProvenanceError("Source identity changed; validate with the original source snapshot")
    config = _read_json(root / "config.json")
    if _json_hash(config) != config_sha256:
        raise ProvenanceError("Configuration identity/hash changed")
    if _cached_artifact(root, UNIT_MANIFEST_PATH, cache)["sha256"] != unit_sha256:
        raise ProvenanceError("Unit/split manifest identity/hash changed")


def _validate_manifest_shape(value: dict) -> None:
    required = {"schema_version", "stage", "stage_index", "scientific_role", "dataset", "created_utc",
                "source_sha256", "config_sha256", "unit_sha256", "unit_manifest_path", "parent_manifest",
                "inputs", "outputs", "model_audit", "evaluator", "claim_scope"}
    if set(value) != required:
        raise ProvenanceError(f"Unexpected stage-manifest fields: {sorted(set(value) ^ required)}")
    stage = value["stage"]
    if (value["schema_version"] != SCHEMA_VERSION or stage not in STAGES
            or type(value["stage_index"]) is not int or value["stage_index"] != STAGES.index(stage)
            or value["scientific_role"] != SCIENTIFIC_ROLES[stage]
            or value["dataset"] not in ("ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func", "synthetic")
            or value["unit_manifest_path"] != UNIT_MANIFEST_PATH):
        raise ProvenanceError("Invalid manifest version, stage, scientific role or dataset")
    try:
        timestamp = datetime.fromisoformat(value["created_utc"].replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("timezone required")
    except (ValueError, AttributeError, TypeError) as exc:
        raise ProvenanceError("Manifest creation time must include a timezone") from exc
    if not isinstance(value["inputs"], list) or not isinstance(value["outputs"], list) or not value["outputs"]:
        raise ProvenanceError("Manifest inputs/outputs must be arrays, with at least one output")


def _verify_completed_directory_inventory(root: Path, manifests: Sequence[dict], sealed: dict) -> None:
    """Detect new glob-consumed inputs added to an already completed stage.

    Only basename suffix .tmp is exempt; it is the documented atomic-write
    temporary suffix and is not consumed by scientific loaders. Shared
    predictions/member_predictions directories are deliberately excluded:
    confirmation may add its own newly sealed control predictions there.
    """
    for manifest in manifests:
        stage = manifest["stage"]
        directory = _EXCLUSIVE_STAGE_DIRECTORIES[stage]
        expected = {name for name in sealed if name.startswith(directory + "/")}
        actual = set()
        for path in (root / directory).rglob("*"):
            if path.is_symlink():
                # Resolve even an exempt temporary symlink so it cannot hide
                # an escape or a dangling directory/file reference.
                _relative_file(root, path)
            if not path.is_file() or path.name.endswith(TEMPORARY_ARTIFACT_SUFFIXES):
                continue
            actual.add(path.relative_to(root).as_posix())
        if actual != expected:
            raise ProvenanceError(
                f"immutable completed-stage inventory changed: {stage}; "
                f"unsealed={sorted(actual - expected)}, missing={sorted(expected - actual)}")


def verify_stage_chain(run_root: str | Path, through_stage: str | None = None, *,
                       expected_source_sha256: str | None = None,
                       expected_config_sha256: str | None = None,
                       expected_unit_sha256: str | None = None,
                       _artifact_cache: dict | None = None) -> list[dict]:
    """Validate the contiguous chain, actual artifacts and current frozen state.

    Empty runs are valid when through_stage is omitted. A partial run may have
    unsealed next-stage files, but existing seals and all their artifacts remain
    immutable. Parent digests must identify the actual preceding manifest, and
    each declared input after prepare must come from an earlier sealed output.
    """
    root = Path(run_root).resolve()
    if through_stage is not None and through_stage not in STAGES:
        raise ProvenanceError(f"Unknown through_stage: {through_stage}")
    directory = root / "provenance" / "stages"
    expected_paths = {stage_manifest_path(root, stage).name: stage for stage in STAGES}
    present = {path.name for path in directory.glob("*.json")} if directory.exists() else set()
    if present - set(expected_paths):
        raise ProvenanceError("Unrecognized manifest files in the immutable stage directory")
    indices = sorted(STAGES.index(expected_paths[name]) for name in present)
    if indices and indices != list(range(max(indices) + 1)):
        raise ProvenanceError("Stage-manifest chain has a gap or missing parent")
    if through_stage is not None:
        requested = STAGES.index(through_stage)
        if requested not in indices:
            raise ProvenanceError(f"Required stage manifest is missing: {through_stage}")
        indices = [index for index in indices if index <= requested]
    sealed, result, first = {}, [], None
    artifact_cache = {} if _artifact_cache is None else _artifact_cache
    model_cache = {}
    for index in indices:
        stage = STAGES[index]
        path = stage_manifest_path(root, stage)
        value = _read_json(path)
        _validate_manifest_shape(value)
        if value["stage"] != stage:
            raise ProvenanceError("Manifest content does not match its registered filename")
        identities = (value["source_sha256"], value["config_sha256"], value["unit_sha256"])
        if first is None:
            _verify_identities(root, *identities, cache=artifact_cache)
            first = identities + (value["dataset"],)
            expectations = (expected_source_sha256, expected_config_sha256, expected_unit_sha256)
            if any(expected is not None and expected != actual for expected, actual in zip(expectations, identities)):
                raise ProvenanceError("Run identities differ from the requested source/config/unit identities")
            if _read_json(root / "config.json").get("dataset") != value["dataset"]:
                raise ProvenanceError("Manifest dataset differs from the frozen configuration")
        elif identities + (value["dataset"],) != first:
            raise ProvenanceError("Source/config/unit/dataset identity changed between stages")
        expected_parent = None if index == 0 else _cached_artifact(root, stage_manifest_path(root, STAGES[index-1]), artifact_cache)
        if value["parent_manifest"] != expected_parent:
            raise ProvenanceError(f"Parent manifest identity/hash mismatch at {stage}")
        inputs, outputs = {}, {}
        for field, target in (("inputs", inputs), ("outputs", outputs)):
            for record in value[field]:
                actual = _verify_artifact(root, record, artifact_cache)
                if actual["path"] in target:
                    raise ProvenanceError(f"Repeated {field} artifact: {actual['path']}")
                target[actual["path"]] = actual
        for relative, record in inputs.items():
            if index and sealed.get(relative) != record:
                raise ProvenanceError(f"Input has no matching earlier sealed output: {relative}")
        for relative, record in outputs.items():
            if relative in sealed and sealed[relative] != record:
                raise ProvenanceError(f"An earlier stage artifact was replaced: {relative}")
        if stage == "evaluate":
            if set(outputs) != FINAL_EVALUATION_OUTPUTS:
                raise ProvenanceError("Evaluation outputs must exactly match the final-report allowlist")
            evaluator = _verify_artifact(root, value["evaluator"], artifact_cache)
            if sealed.get(evaluator["path"]) != evaluator or inputs.get(evaluator["path"]) != evaluator:
                raise ProvenanceError("Evaluator must be an earlier sealed artifact and a declared test input")
            expected_evaluator = Path(__file__).resolve().with_name("pipeline.py")
            if evaluator["sha256"] != sha256_file(expected_evaluator):
                raise ProvenanceError("Evaluator snapshot differs from the current frozen evaluator source")
        elif value["evaluator"] is not None:
            raise ProvenanceError("Only final evaluation records an evaluator identity")
        if stage == "prepare":
            if value["model_audit"] is not None:
                raise ProvenanceError("Preparation cannot declare a trained-model freeze audit")
        else:
            audit = value["model_audit"]
            if not isinstance(audit, dict) or set(audit) != {"scope", "baseline", "checkpoints", "matches_post_train_baseline"}:
                raise ProvenanceError("Post-training stages require an actual frozen-model audit")
            baseline = _verify_artifact(root, audit["baseline"], artifact_cache)
            if stage == "train":
                if audit["scope"] != "post_train_capture" or outputs.get(baseline["path"]) != baseline:
                    raise ProvenanceError("Train must seal its post-training baseline as an output")
            elif audit["scope"] != "unchanged_since_train" or sealed.get(baseline["path"]) != baseline:
                raise ProvenanceError("Later stages must reference the previously sealed model baseline")
            key = baseline["sha256"]
            if key not in model_cache:
                model_cache[key] = verify_frozen_models(root, baseline["path"], _artifact_cache=artifact_cache)
            expected_audit = dict(model_cache[key], scope="post_train_capture" if stage == "train" else "unchanged_since_train")
            if audit != expected_audit:
                raise ProvenanceError(f"Declared model audit differs from actual checkpoint state at {stage}")
        sealed.update(outputs)
        result.append(value)
    _verify_completed_directory_inventory(root, result, sealed)
    return result


def write_stage_manifest(run_root: str | Path, stage: str, *, dataset: str,
                         source_sha256: str, config_sha256: str, unit_sha256: str,
                         inputs: Sequence[str | Path], outputs: Sequence[str | Path],
                         model_baseline: str | Path | None = None,
                         evaluator_path: str | Path | None = None) -> dict:
    """Seal a completed stage using recomputed artifacts and verified ancestry.

    A byte-identical semantic re-request is idempotent; altered existing seals
    are never replaced. Root integration must call this after stage execution
    and verify the chain before resume and before exposing later label roles.
    """
    root = Path(run_root).resolve()
    destination = stage_manifest_path(root, stage)
    index = STAGES.index(stage)
    artifact_cache = {}
    prior = verify_stage_chain(root, expected_source_sha256=source_sha256,
                              expected_config_sha256=config_sha256, expected_unit_sha256=unit_sha256,
                              _artifact_cache=artifact_cache)
    if not destination.exists() and len(prior) != index:
        raise ProvenanceError(f"Stage {stage} cannot be sealed before its immediate predecessor")
    _verify_identities(root, source_sha256, config_sha256, unit_sha256, cache=artifact_cache)
    input_records, output_records = _artifacts(root, inputs, artifact_cache), _artifacts(root, outputs, artifact_cache)
    if not output_records:
        raise ProvenanceError("A completed stage must have at least one actual output artifact")
    evaluator = _cached_artifact(root, evaluator_path, artifact_cache) if evaluator_path is not None else None
    if stage == "evaluate":
        if {item["path"] for item in output_records} != FINAL_EVALUATION_OUTPUTS:
            raise ProvenanceError("Evaluation outputs must exactly match the final-report allowlist")
        if evaluator is None:
            raise ProvenanceError("Final evaluation requires its frozen evaluator snapshot")
        if evaluator["path"] not in {item["path"] for item in input_records}:
            input_records = sorted(input_records + [evaluator], key=lambda item: item["path"])
    elif evaluator is not None:
        raise ProvenanceError("An evaluator identity belongs only to final evaluation")
    if stage == "prepare":
        if model_baseline is not None:
            raise ProvenanceError("Preparation cannot capture a trained-model baseline")
        model_audit = None
    else:
        if model_baseline is None:
            raise ProvenanceError("Every stage from train onward requires the post-training model baseline")
        model_audit = dict(verify_frozen_models(root, model_baseline, _artifact_cache=artifact_cache),
                           scope="post_train_capture" if stage == "train" else "unchanged_since_train")
    parent = None if index == 0 else _cached_artifact(root, stage_manifest_path(root, STAGES[index-1]), artifact_cache)
    value = {"schema_version": SCHEMA_VERSION, "stage": stage, "stage_index": index,
             "scientific_role": SCIENTIFIC_ROLES[stage], "dataset": dataset,
             "created_utc": datetime.now(timezone.utc).isoformat(),
             "source_sha256": source_sha256, "config_sha256": config_sha256,
             "unit_sha256": unit_sha256, "unit_manifest_path": UNIT_MANIFEST_PATH,
             "parent_manifest": parent, "inputs": input_records, "outputs": output_records,
             "model_audit": model_audit, "evaluator": evaluator,
             "claim_scope": "registered dataset execution or synthetic validation; manuscript agreement requires result artifact bindings"}
    _validate_manifest_shape(value)
    if destination.exists():
        existing = _read_json(destination)
        compare = dict(value, created_utc=existing["created_utc"])
        if compare != existing:
            raise ProvenanceError(f"Cannot replace immutable stage manifest: {stage}")
        return existing
    # Validate new relationships before publishing, including unsealed inputs.
    sealed = {item["path"]: item for previous in prior for item in previous["outputs"]}
    if index and any(sealed.get(item["path"]) != item for item in input_records):
        raise ProvenanceError("A stage input has no matching earlier sealed output")
    if stage == "train" and model_audit["baseline"] not in output_records:
        raise ProvenanceError("Train outputs must include the immutable frozen-model baseline")
    if stage not in ("prepare", "train") and sealed.get(model_audit["baseline"]["path"]) != model_audit["baseline"]:
        raise ProvenanceError("The frozen-model baseline was not sealed by train")
    if evaluator is not None:
        if sealed.get(evaluator["path"]) != evaluator:
            raise ProvenanceError("Evaluator was not sealed before final evaluation")
        if evaluator["sha256"] != sha256_file(Path(__file__).resolve().with_name("pipeline.py")):
            raise ProvenanceError("Evaluator snapshot differs from the frozen evaluator source")
    _atomic_create_json(destination, value)
    return value
