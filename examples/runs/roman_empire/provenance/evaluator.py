"""Separated-data CROSS-MORSE lifecycle for node and graph benchmarks."""
from __future__ import annotations

from pathlib import Path
import copy
import contextlib
import csv
import json
import os
import platform
import importlib.metadata
import sys
import shutil
import tempfile
import time

import numpy as np
import torch

from . import __version__
from .common import (atomic_json, atomic_npz, canonical_hash, file_hash, read_json,
                     precision_context, seed_everything, stop_at_boundary, stop_requested)
from .core import ACTIONS, GraphOperator
from .recovery import ANCHOR_RULE
from .scoring import SCORE_VERSION
from .data import prepare_data, load_data, role_labels
from .exploration import explore, load_records
from .learning import fit_controller, Controller
from .policies import make_registry, execute_policy, DonorBank, PolicyExecutionError
from .provenance import (capture_frozen_models, verify_frozen_models, write_stage_manifest,
                         verify_stage_chain, stage_manifest_path, FINAL_EVALUATION_OUTPUTS)
from .statistics import cluster_summary, paired_bootstrap, nested_selection, confirmation, admission, global_candidate_oracle
from .training import train, load_plant, native_forward
from .tasks import task_spec
from .evaluation import score_observations, MetricEvidence, paired_metric_interval, independent_graphs
from .runtime import graph_operator, is_graph_collection, predict_dataset

STAGES = ("prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate")


def atomic_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npy")
    os.close(fd)
    try:
        np.save(tmp, array, allow_pickle=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    files = sorted(root.glob("run_cross_morse*.py")) + sorted((root / "cross_morse").glob("*.py"))
    return canonical_hash({str(p.relative_to(root)): file_hash(p) for p in files if p.exists()})


def validate_config(config):
    if config.get("schema_version") != 1 or config["dataset"] not in ("ogbn-arxiv", "ogbn-products", "ogbn-proteins", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func", "synthetic"):
        raise ValueError("Unsupported configuration schema or dataset")
    if min(config["model"]["depth"], config["model"]["hidden_dim"], config["training"]["epochs"]) <= 0:
        raise ValueError("Depth, width, and training epochs must be positive")
    if len(set(config["training"]["seeds"])) != len(config["training"]["seeds"]):
        raise ValueError("Checkpoint seeds must be unique")
    if config["exploration"].get("record_stride", 1) != 1:
        raise ValueError("The conditioned observer requires every action; record_stride must be 1")
    if config["exploration"]["trajectories"] < config["controller"]["crossfit_folds"] * 2:
        raise ValueError("At least two complete trajectories per controller fold are required")
    if config.get("scientific_run") and config["dataset"] == "synthetic":
        raise ValueError("Synthetic data may never be marked as a scientific benchmark run")
    cal = config["controller"]
    cf, af = float(cal.get("calibration_fraction", .2)), float(cal.get("calibration_audit_fraction", .2))
    if not (0 < cf < 1 and 0 < af < 1 and cf + af < 1):
        raise ValueError("Invalid calibration/assessment trajectory fractions")
    if cal.get("calibration_protocol", "trajectory_holdout_v2") != "trajectory_holdout_v2":
        raise ValueError("Legacy calibration requires its original source snapshot")
    n = config["exploration"]["trajectories"]
    if n - max(1, int(n * cf)) - max(1, int(n * af)) < max(3, cal["crossfit_folds"]):
        raise ValueError("Too few model-fit trajectories after calibration/assessment holdouts")
    if not config["registry"]["ensemble_sizes"] or min(config["registry"]["ensemble_sizes"]) != 1:
        raise ValueError("The registered ensemble grid must include E=1")


class Pipeline:
    def __init__(self, config, data_root, output_dir, device="cpu", resume=False):
        validate_config(config)
        self.config = config
        self.data_root = Path(data_root).resolve()
        self.out = Path(output_dir).resolve()
        self.device = str(device)
        self.resume = resume
        self.out.mkdir(parents=True, exist_ok=True)
        self.config_hash = canonical_hash(config)
        self.source_hash = source_fingerprint()
        state_path = self.out / "run_state.json"
        if state_path.exists():
            prior = read_json(state_path)
            if not resume:
                raise FileExistsError("Run already exists. Use --resume for this directory, or choose a new output directory.")
            if prior["config_sha256"] != self.config_hash or prior["source_sha256"] != self.source_hash:
                raise ValueError("Run source/configuration changed. Resume its original source snapshot, or use a new output directory.")
            if canonical_hash(read_json(self.out / "config.json")) != self.config_hash:
                raise ValueError("Stored run configuration changed; refusing to overwrite it during resume")
        else:
            atomic_json(state_path, {"version": __version__, "config_sha256": self.config_hash,
                                    "source_sha256": self.source_hash, "dataset": config["dataset"],
                                    "profile": config["profile"], "scientific_run": config["scientific_run"],
                                    "implementation": "CROSS-MORSE",
                                    "task": vars(task_spec(config)),
                                    "status": "CREATED"})
        if not (self.out / "config.json").exists():
            atomic_json(self.out / "config.json", config)
        if not (self.out / "environment.json").exists():
            atomic_json(self.out / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
                    "torch": torch.__version__, "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
                    "device": device, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
                    "precision": config["model"]["precision"], "version": __version__,
                    "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
                    "python_implementation": platform.python_implementation(),
                    "machine": platform.machine(), "torch_threads": torch.get_num_threads(),
                    "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                    "torch_build_configuration": torch.__config__.show()})
        self._data = None
        self._graph = None
        self.controllers = {}
        self._active_stage = None

    @property
    def data(self):
        if self._data is None:
            self._data = load_data(self.out).protect_labels()
        return self._data

    @property
    def graph(self):
        if is_graph_collection(self.data):
            return None
        if self._graph is None:
            self._graph = graph_operator(self.data, self.device)
        return self._graph

    def controller(self, profile):
        profile = "full" if profile == "none" else profile
        if profile not in self.controllers:
            # Small decision networks run on CPU; graph/plant remains on the
            # requested device. Their work is in the serial wall-time ledger.
            self.controllers[profile] = Controller.load(self.out / "fit" / profile, "cpu")
        return self.controllers[profile]

    def lock_hashes(self):
        paths = sorted((self.out / "train").glob("*_best.pt"))
        paths += sorted((self.out / "fit").glob("*/controller.pt"))
        return {str(p.relative_to(self.out)): file_hash(p) for p in paths}

    def check_frozen(self):
        verify_frozen_models(self.out)
        locked = read_json(self.out / "fit" / "frozen_models.json")
        now = self.lock_hashes()
        if now != locked:
            raise RuntimeError("A frozen backbone/controller checkpoint changed after fitting")
        exploration = read_json(self.out / "fit" / "exploration_hashes.json")
        for relative, expected in exploration.items():
            if file_hash(self.out / relative) != expected:
                raise RuntimeError(f"Frozen exploration/reference data changed: {relative}")

    def _verify_chain(self):
        unit_path = self.out / "prepared" / "splits.npz"
        chain = verify_stage_chain(self.out, expected_source_sha256=self.source_hash,
                     expected_config_sha256=self.config_hash,
                     expected_unit_sha256=file_hash(unit_path) if unit_path.exists() else None)
        completed = {m["stage"] for m in chain}
        for stage in STAGES:
            marker = self.out / "stages" / f"{stage}.json"
            if marker.exists():
                if stage not in completed:
                    raise RuntimeError(f"Completion marker has no verified stage manifest: {stage}")
                if read_json(marker).get("manifest_sha256") != file_hash(stage_manifest_path(self.out, stage)):
                    raise RuntimeError(f"Completion marker disagrees with stage manifest: {stage}")
        return chain

    def _stage_output_paths(self, stage, chain):
        directories = {
            "prepare": ("prepared",), "train": ("train",), "explore": ("explore",), "fit": ("fit",),
            "select": ("select", "predictions", "member_predictions", "prediction_failures"),
            "confirm": ("confirm", "predictions", "member_predictions", "prediction_failures"),
            "admit": ("admit",), "evaluate": ("final",),
        }
        paths = {p.relative_to(self.out).as_posix() for folder in directories[stage]
                 for p in (self.out / folder).rglob("*") if p.is_file() and not p.name.endswith(".tmp")}
        extras = {"prepare": {"config.json", "environment.json", "action_registry.json", "provenance/evaluator.py"},
                  "train": {"provenance/frozen_models.json"}, "admit": {"policy_lock.json"}}
        paths.update(extras.get(stage, ()))
        sealed = {a["path"] for manifest in chain for a in manifest["outputs"]}
        return sorted(paths - sealed)

    def run(self, requested_stage="all"):
        if requested_stage not in ("all", *STAGES):
            raise ValueError(f"Unknown lifecycle stage: {requested_stage}")
        if self._active_stage is not None:
            raise RuntimeError("Lifecycle stages cannot be nested")
        chain = self._verify_chain()
        if chain:
            # A manifest is the scientific commit. Repair stale bookkeeping if
            # interruption occurred between that commit and the state update.
            state = read_json(self.out / "run_state.json")
            last = chain[-1]["stage"]
            expected = {"last_complete_stage": last,
                        "status": "COMPLETE" if last == "evaluate" else "RUNNING"}
            if any(state.get(key) != value for key, value in expected.items()):
                state.update(expected)  # preserve the sticky test-access flag
                atomic_json(self.out / "run_state.json", state)
        stages = STAGES if requested_stage == "all" else (requested_stage,)
        for stage in stages:
            marker = self.out / "stages" / f"{stage}.json"
            sealed_stage = next((m for m in chain if m["stage"] == stage), None)
            if sealed_stage is not None:
                if not marker.exists():
                    atomic_json(marker, {"stage": stage, "status": "COMPLETE", "recovered_after_commit": True,
                              "manifest_sha256": file_hash(stage_manifest_path(self.out, stage)),
                              "config_sha256": self.config_hash, "source_sha256": self.source_hash})
                print(f"[resume] stage {stage} is complete", flush=True)
                continue
            index = STAGES.index(stage)
            if index != len(chain):
                raise RuntimeError(f"Stage {stage} requires completed {STAGES[index-1]}")
            if read_json(self.out / "run_state.json").get("test_labels_accessed") and stage != "evaluate":
                raise RuntimeError("Policy-changing stages are locked after test-label access")
            print(f"[stage] {stage}", flush=True)
            started = time.perf_counter()
            if index >= 2:
                verify_frozen_models(self.out)
            inputs = sorted({a["path"] for manifest in chain for a in manifest["outputs"]})
            access_start = len(self.data.y.accesses) if stage != "prepare" else 0
            scope = self.data._label_scope(stage) if stage != "prepare" else contextlib.nullcontext([])
            self._active_stage = stage
            try:
                with scope as accesses:
                    getattr(self, f"_stage_{stage}")()
                if index >= 1:
                    verify_frozen_models(self.out)
                # Rechecking all parents catches accidental writes by the stage
                # itself, including policy/comparator mutations by evaluation.
                self._verify_chain()
                manifest = write_stage_manifest(self.out, stage, dataset=self.config["dataset"],
                      source_sha256=self.source_hash, config_sha256=self.config_hash,
                      unit_sha256=file_hash(self.out / "prepared" / "splits.npz"), inputs=inputs,
                      outputs=self._stage_output_paths(stage, chain),
                      model_baseline="provenance/frozen_models.json" if index >= 1 else None,
                      evaluator_path="provenance/evaluator.py" if stage == "evaluate" else None)
            finally:
                self._active_stage = None
            atomic_json(marker, {"stage": stage, "status": "COMPLETE", "seconds_this_attempt": time.perf_counter() - started,
                                 "config_sha256": self.config_hash, "source_sha256": self.source_hash,
                                 "manifest_sha256": file_hash(stage_manifest_path(self.out, stage)),
                                 "label_accesses": accesses[access_start:]})
            chain.append(manifest)
            state = read_json(self.out / "run_state.json")
            state.update(status="COMPLETE" if stage == "evaluate" else "RUNNING", last_complete_stage=stage)
            atomic_json(self.out / "run_state.json", state)
            stop_at_boundary()

    def _stage_prepare(self):
        self._data = prepare_data(self.data_root, self.out, self.config).protect_labels()
        atomic_json(self.out / "action_registry.json", [vars(a) for a in ACTIONS])
        evaluator = self.out / "provenance" / "evaluator.py"
        evaluator.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__), evaluator)

    def _stage_train(self):
        train(self.data, self.graph, self.config, self.out, self.device)
        capture_frozen_models(self.out, [p.relative_to(self.out).as_posix()
                              for p in sorted((self.out / "train").glob("*_best.pt"))])
        inventory = []
        for path in sorted((self.out / "train").glob("*_best.pt")):
            # These checkpoints were created and frozen by this run.
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload["model"]
            tensors = [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                        "elements": value.numel()} for name, value in sorted(state.items())]
            inventory.append({"path": path.relative_to(self.out).as_posix(),
                "sha256": file_hash(path), "named_tensor_elements": sum(x["elements"] for x in tensors),
                "encoder_elements": sum(x["elements"] for x in tensors if x["name"].startswith("encoder.")),
                "head_elements": sum(x["elements"] for x in tensors if x["name"].startswith("head.")),
                "tensors": tensors})
        atomic_json(self.out / "train" / "model_inventory.json", {
            "architecture": "FrozenPlant", "architecture_module": "cross_morse/core.py",
            "configuration": self.config["model"], "checkpoints": inventory,
            "count_scope": "named checkpoint tensor elements; this FrozenPlant has no non-parameter buffers",
            "controller_parameters_included": False, "ensemble_counting": "each checkpoint listed separately"})

    def _stage_explore(self):
        explore(self.data, self.graph, self.config, self.out, self.device)

    def _stage_fit(self):
        records = load_records(self.out / "explore")
        for profile in self.config["sensor"]["profiles"]:
            c = copy.deepcopy(self.config)
            c["active_profile"] = profile
            fit_controller(records, c, self.out / "fit" / profile, device=self.device, stop_requested=stop_requested)
            stop_at_boundary()
        registry = make_registry(self.config, self.out / "explore")
        # This snapshot is written before any D_select labels are accessed.
        atomic_json(self.out / "fit" / "registry.json", {"candidates": registry,
                    "frozen_before_selection": True, "registry_sha256": canonical_hash(registry)})
        atomic_json(self.out / "fit" / "frozen_models.json", self.lock_hashes())
        artifacts = sorted((self.out / "explore").glob("*.npz"))
        atomic_json(self.out / "fit" / "exploration_hashes.json",
                    {str(p.relative_to(self.out)): file_hash(p) for p in artifacts})

    def probability_path(self, candidate):
        return self.out / "predictions" / f"{candidate['candidate_id']}.npy"

    def verify_prediction(self, candidate):
        """Validate cached identity and bytes before scoring or final copying."""
        path = self.probability_path(candidate)
        metadata_path = path.with_suffix(".json")
        if not path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"Cached candidate prediction is incomplete: {path}")
        cached = read_json(metadata_path)
        identity = cached.get("candidate", {})
        identity_ok = (identity.get("candidate_id") == candidate["candidate_id"]
                       and identity.get("hash") == candidate["hash"]
                       and identity.get("state_control") == candidate.get("state_control"))
        if not identity_ok or file_hash(path) != cached.get("prediction_sha256"):
            raise RuntimeError(f"Cached candidate prediction failed identity/hash verification: {path}")
        if cached.get("fallback_applied"):
            origin = cached.get("fallback_source", {})
            decision_path = self.out / "select" / "decision.json"
            decision = read_json(decision_path)
            family = read_json(self.out / "select" / "candidates.json")
            comparator = family[decision["comparator_index"]]
            if (origin.get("candidate_id") != comparator["candidate_id"]
                    or origin.get("hash") != comparator["hash"]
                    or origin.get("selection_decision_sha256") != file_hash(decision_path)
                    or origin.get("prediction_sha256") != file_hash(self.probability_path(comparator))
                    or cached["prediction_sha256"] != origin["prediction_sha256"]):
                raise RuntimeError("Cached numerical fallback no longer matches the locked comparator")
        return cached

    def _failed_prediction(self, candidate, error, traces, seed):
        """Keep a failed attempt explicit; it has no measured policy loss."""
        failed = dict(error.trace, seed=seed)
        attempts = traces + [failed]
        meta = {"candidate": candidate, "members": attempts, "execution_failure": True,
                "failure_reason": error.reason, "feasible": False,
                "prediction_available": False, "fallback_applied": False,
                "multiply_add_work": sum(t.get("multiply_add_work", 0) for t in attempts),
                "deployed_multiply_add_work": sum(t.get("deployed_multiply_add_work", t.get("multiply_add_work", 0)) for t in attempts),
                "qualification_multiply_add_work": sum(t.get("qualification_multiply_add_work", 0) for t in attempts),
                "serial_wall_seconds": sum(t.get("wall_seconds", 0) for t in attempts),
                "support_violations": sum(t.get("support_violations", 0) for t in attempts),
                "shuffle_matched_sites": sum(t.get("shuffle_matched_sites", 0) for t in attempts),
                "shuffle_missing_sites": sum(t.get("shuffle_missing_sites", 0) for t in attempts),
                "state_control_valid": False, "cost_is_partial": True,
                "labels_accessed_during_prediction": False,
                "config_sha256": self.config_hash, "source_sha256": self.source_hash}
        path = self.out / "prediction_failures" / f"{candidate['candidate_id']}.json"
        atomic_json(path, meta)
        return meta

    def _comparator_fallback(self, candidate, failure, comparator):
        """A failed control may return only the already selected qC bytes.

        This keeps an invalid mechanism intervention executable and auditable;
        its failure remains infeasible, and confirmation forces p=1. It makes
        no promise to recover a damaged CUDA context or an OOM condition.
        """
        decision = read_json(self.out / "select" / "decision.json")
        family = read_json(self.out / "select" / "candidates.json")
        expected = family[decision["comparator_index"]]
        if (comparator["candidate_id"], comparator["hash"]) != (expected["candidate_id"], expected["hash"]):
            raise RuntimeError("Numerical fallback must use the locked selected comparator")
        ref_meta = self.verify_prediction(expected)
        if ref_meta.get("execution_failure") or not ref_meta.get("feasible", True):
            raise RuntimeError("The comparator itself has no valid successful prediction")
        source, path = self.probability_path(expected), self.probability_path(candidate)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
        if file_hash(path) != file_hash(source):
            raise AssertionError("Numerical fallback must preserve comparator bytes exactly")
        meta = dict(failure, prediction_available=True, fallback_applied=True,
                    prediction_sha256=file_hash(path),
                    fallback_source={"candidate_id": expected["candidate_id"], "hash": expected["hash"],
                        "prediction_sha256": ref_meta["prediction_sha256"],
                        "selection_decision_sha256": file_hash(self.out / "select" / "decision.json")},
                    interpretation="invalid intervention; comparator bytes supplied, no mechanism evidence")
        atomic_json(path.with_suffix(".json"), meta)
        return meta

    def predict_candidate(self, candidate, donors=None, fallback=None):
        path = self.probability_path(candidate)
        metadata_path = path.with_suffix(".json")
        if path.exists() and metadata_path.exists():
            return self.verify_prediction(candidate)
        failure_path = self.out / "prediction_failures" / f"{candidate['candidate_id']}.json"
        if failure_path.exists():
            failure = read_json(failure_path)
            if (failure.get("candidate") != candidate or failure.get("config_sha256") != self.config_hash
                    or failure.get("source_sha256") != self.source_hash):
                raise RuntimeError("Cached numerical failure changed identity")
            return self._comparator_fallback(candidate, failure, fallback) if fallback is not None else failure
        behavior = candidate["behavior_id"] + ("_shuffle" if candidate.get("state_control") else "")
        reference = dict(np.load(self.out / "explore" / "nominal_reference.npz", allow_pickle=False))
        route = candidate["route"]
        needs_controller = route in ("phase", "fixed_sequence", "mpc_ol", "mpc_rh", "one_step", "radius")
        controller = self.controller(candidate["profile"]) if needs_controller else None
        if route.startswith("mpc") and not controller.sequence_models_available:
            raise RuntimeError("MPC configuration requires fitted transition/stage/terminal targets")
        aggregate, traces = None, []
        for seed in self.config["training"]["seeds"][:candidate["ensemble"]]:
            member_path = self.out / "member_predictions" / f"{behavior}_seed{seed}.npy"
            trace_path = member_path.with_suffix(".json")
            if not (member_path.exists() and trace_path.exists()):
                model = load_plant(self.data, self.config, self.out / "train", seed, self.device)
                if self.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                try:
                    probs, trace = predict_dataset(model, self.data, self.graph, self.config,
                                                  candidate, controller, reference, donors=donors,
                                                  device=self.device, output_path=member_path)
                except PolicyExecutionError as error:
                    failure = self._failed_prediction(candidate, error, traces, seed)
                    return self._comparator_fallback(candidate, failure, fallback) if fallback is not None else failure
                if not member_path.exists():
                    atomic_npy(member_path, probs)
                trace.update(seed=seed, behavior_id=behavior, prediction_sha256=file_hash(member_path),
                             checkpoint_sha256=file_hash(self.out / "train" / f"plant_seed{seed}_best.pt"))
                atomic_json(trace_path, trace)
            trace = read_json(trace_path)
            if (trace.get("seed") != seed or trace.get("behavior_id") != behavior
                    or trace.get("prediction_sha256") != file_hash(member_path)
                    or trace.get("checkpoint_sha256") != file_hash(self.out / "train" / f"plant_seed{seed}_best.pt")):
                raise RuntimeError(f"Cached member prediction failed identity/hash verification: {member_path}")
            p = np.load(member_path, mmap_mode="r", allow_pickle=False)
            aggregate = np.array(p, dtype=np.float64) if aggregate is None else aggregate + p
            traces.append(trace)
            stop_at_boundary()
        probabilities = (aggregate / candidate["ensemble"]).astype(np.float32)
        atomic_npy(path, probabilities)
        meta = {"candidate": candidate, "members": traces, "prediction_sha256": file_hash(path),
                "multiply_add_work": sum(t["multiply_add_work"] for t in traces),
                "deployed_multiply_add_work": sum(t.get("deployed_multiply_add_work", t["multiply_add_work"]) for t in traces),
                "qualification_multiply_add_work": sum(t.get("qualification_multiply_add_work", 0) for t in traces),
                "serial_wall_seconds": sum(t["wall_seconds"] for t in traces),
                "support_violations": sum(t["support_violations"] for t in traces),
                "feasible": all(t["feasible"] for t in traces),
                "shuffle_matched_sites": sum(t["shuffle_matched_sites"] for t in traces),
                "shuffle_missing_sites": sum(t["shuffle_missing_sites"] for t in traces),
                "state_control_valid": all(t.get("state_control_valid", True) for t in traces),
                "labels_accessed_during_prediction": False}
        atomic_json(metadata_path, meta)
        return meta

    def role_scores(self, candidate, role):
        self.verify_prediction(candidate)
        ids = np.asarray(self.data.splits[role])
        y = role_labels(self.data, role)
        p = np.load(self.probability_path(candidate), mmap_mode="r", allow_pickle=False)[ids]
        units = self.data.cluster_ids[ids]
        result = score_observations(p, y, units, self.config)
        result.update(probabilities=p, labels=y, sample_units=units, sample_ids=ids)
        return result


    def entry_completion(self, candidates):
        initial = read_json(self.out / "train" / "entry_initial.json")
        full_support = self.controller("full").support_audit
        E = 1
        relevant = [i for i, c in enumerate(candidates) if c["ensemble"] == E and c["feasible"]]
        opened = [i for i in relevant if candidates[i]["is_open_loop"]]
        if not opened:
            raise RuntimeError("No valid nominal/open-loop comparator exists")
        scores = {i: self.role_scores(candidates[i], "explore") for i in relevant}
        reference = min(opened, key=lambda i: (scores[i]["node_ce"], candidates[i]["complexity"], candidates[i]["hash"]))
        winner, global_oracle, per_core_oracle = global_candidate_oracle(
            np.stack([scores[i]["loss"] for i in relevant]), scores[reference]["counts"])
        difference = global_oracle - scores[reference]["loss"]
        headroom = paired_bootstrap(difference, weights=scores[reference]["counts"],
                      reps=self.config["selection"]["bootstrap_reps"], alpha=self.config["selection"]["alpha"], seed=self.config["seed"] + 2)
        thresholds = self.config["entry"]
        passed = bool(initial["initial_entry_pass"] and
                      full_support["qualified_fraction"] >= thresholds["registry_fraction_min"] and
                      headroom["upper"] is not None and headroom["upper"] < -thresholds["headroom_margin"])
        obj = {"initial": initial, "registry_support_fraction": full_support["qualified_fraction"],
               "headroom": headroom, "headroom_ensemble": E, "headroom_reference_index": reference,
               "headroom_definition": "D_explore best single global candidate at E=1; finite evaluated-family diagnostic, not exact sequence enumeration",
               "headroom_candidate_index": relevant[winner],
               "optimistic_corewise_oracle_effect_diagnostic_only": float(np.average(
                   per_core_oracle - scores[reference]["loss"], weights=scores[reference]["counts"])),
               "passed": passed, "enforced": self.config["selection"]["enforce_entry_contract"],
               "statistical_scope": "training-only descriptive diagnostics; one global candidate chosen; core-wise minimum is not a qualification statistic"}
        atomic_json(self.out / "select" / "entry_complete.json", obj)
        return obj

    def _stage_select(self):
        self.check_frozen()
        candidates = copy.deepcopy(read_json(self.out / "fit" / "registry.json")["candidates"])
        # Execute the complete frozen family before looking at its selection
        # losses. Predictions can include unlabelled test nodes transductively.
        nominal_work = None
        for number, candidate in enumerate(candidates):
            print(f"[select/predict] {number+1}/{len(candidates)} {candidate['route']} E={candidate['ensemble']} {candidate['profile']}", flush=True)
            meta = self.predict_candidate(candidate)
            if candidate["route"] == "nominal" and candidate["ensemble"] == 1:
                if meta.get("execution_failure"):
                    raise RuntimeError("Nominal E=1 execution failed before any comparator was selected; inspect prediction_failures")
                nominal_work = meta.get("deployed_multiply_add_work", meta["multiply_add_work"])
            if nominal_work is None:
                raise RuntimeError("Registry must begin with nominal E=1")
            candidate["cost"] = meta.get("deployed_multiply_add_work", meta["multiply_add_work"]) / max(nominal_work, 1)
            candidate["qualification_cost"] = meta.get("qualification_multiply_add_work", 0) / max(nominal_work, 1)
            candidate["cost_definition"] = "deployed sparse/dense multiply-add count normalized by nominal E=1; fixed-plan qualification sensing recorded separately"
            candidate["serial_wall_seconds"] = meta["serial_wall_seconds"]
            candidate["feasible"] = bool(meta["feasible"] and candidate["cost"] <= self.config["selection"]["max_normalized_work"])
            candidate["execution_failure"] = bool(meta.get("execution_failure", False))
            if candidate["execution_failure"]:
                candidate["ineligibility_reason"] = meta["failure_reason"]
                candidate["cost_is_partial"] = True
            stop_at_boundary()
        # This release reports qN at the comparator's ensemble size. Establish
        # that executable reference prerequisite before any D_select labels,
        # rather than discovering an unavailable baseline after policy locking.
        missing_nominal = {c["ensemble"] for c in candidates
                           if c["route"] == "nominal" and c["execution_failure"]}
        for candidate in candidates:
            if candidate["ensemble"] in missing_nominal:
                candidate["feasible"] = False
                candidate["nominal_reference_available"] = False
                if not candidate["execution_failure"]:
                    candidate["ineligibility_reason"] = "same-ensemble nominal reference unavailable"
        entry = self.entry_completion(candidates)
        if self.config["selection"]["enforce_entry_contract"] and not entry["passed"]:
            for candidate in candidates:
                if not candidate["is_open_loop"]:
                    candidate["feasible"] = False
                    candidate["ineligibility_reason"] = "training-only entry contract failed"
        scores = [None if c["execution_failure"] else self.role_scores(c, "select") for c in candidates]
        template = next(score for score in scores if score is not None)
        for i, score in enumerate(scores):
            if score is None:
                # Missing measurements stay NaN in the NPZ and become null in
                # result JSON. Statistics explicitly excludes failed columns.
                scores[i] = {"loss": np.full_like(template["loss"], np.nan),
                             "accuracy": (None if template["accuracy"] is None else np.full_like(template["accuracy"], np.nan)),
                             "counts": template["counts"], "core_ids": template["core_ids"],
                             "node_ce": None, "node_accuracy": None}
        loss = np.stack([r["loss"] for r in scores], 1)
        accuracy = None if template["accuracy"] is None else np.stack([r["accuracy"] for r in scores], 1)
        counts, core_ids = scores[0]["counts"], scores[0]["core_ids"]
        sc = self.config["selection"]
        if len(core_ids) < sc["outer_folds"]:
            raise ValueError("Too few nonempty selection cores for the registered outer folds")
        rng = np.random.default_rng(self.config["seed"] + 3)
        order = rng.permutation(len(core_ids))
        folds = np.empty(len(core_ids), dtype=np.int64)
        folds[order] = np.arange(len(core_ids)) % sc["outer_folds"]
        inner = np.empty_like(folds)
        inner[order] = np.arange(len(core_ids)) % sc["inner_folds"]
        metric_evaluator = None
        if accuracy is None:
            # Preserve precisely the stage observations required to replay the
            # nonlinear metric, including cross-fitted comparator reconstruction.
            observed = [np.asarray(r["probabilities"]) if not c["execution_failure"] else
                        np.full_like(template["probabilities"], np.nan) for r, c in zip(scores, candidates)]
            metric_evaluator = MetricEvidence(observed, template["labels"], template["sample_units"],
                                               core_ids, counts, self.config)
            atomic_npz(self.out / "select" / "metric_observations.npz",
                       labels=template["labels"], sample_units=template["sample_units"],
                       sample_ids=template["sample_ids"],
                       **{f"candidate_{j}": p for j, p in enumerate(observed) if not candidates[j]["execution_failure"]})
        result = nested_selection(loss, accuracy, candidates, folds, inner_fold_ids=inner,
                 reps=sc["bootstrap_reps"], alpha=sc["alpha"], epsilon_r=sc["epsilon_r"],
                 epsilon_eq=sc["epsilon_eq"], epsilon_metric=sc["epsilon_metric"],
                 seed=self.config["seed"] + 4, weights=counts, metric_evaluator=metric_evaluator)
        arrays = dict(loss=loss, core_ids=core_ids, counts=counts, outer_fold=folds, inner_fold=inner)
        if accuracy is not None:
            arrays["accuracy"] = accuracy
        atomic_npz(self.out / "select" / "unit_scores.npz", **arrays)
        atomic_json(self.out / "select" / "candidates.json", candidates)
        atomic_json(self.out / "select" / "decision.json", result)
        with open(self.out / "select" / "candidate_ledger.csv", "w", newline="") as f:
            fields = ["candidate_id", "route", "profile", "ensemble", "cost", "feasible", "loss_name", "metric_name", "select_loss", "select_metric", "lower", "upper", "eligible"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i, c in enumerate(candidates):
                row = {k: c[k] for k in fields[:6]}
                row.update(loss_name=template["loss_name"], metric_name=template["metric_name"],
                           select_loss=scores[i]["node_ce"], select_metric=scores[i]["node_accuracy"],
                           lower=result["lower"][i], upper=result["upper"][i], eligible=i in result["eligible_indices"])
                writer.writerow(row)
        self.check_frozen()

    @staticmethod
    def predecessor(candidates, current, route=None, profile=None):
        c = candidates[current]
        desired = dict(c, route=route or c["route"], profile=profile or c["profile"])
        for i, other in enumerate(candidates):
            if other["route"] != desired["route"] or other["profile"] != desired["profile"] or other["ensemble"] != c["ensemble"]:
                continue
            if desired["route"] in ("mpc_ol", "mpc_rh") and other.get("horizon") != c.get("horizon"):
                continue
            if desired["route"] == "radius" and other.get("kappa") != c.get("kappa"):
                continue
            return i
        return None

    def _stage_confirm(self):
        self.check_frozen()
        candidates = read_json(self.out / "select" / "candidates.json")
        selected = read_json(self.out / "select" / "decision.json")
        qC, provisional = selected["comparator_index"], selected["provisional_index"]
        eligible, pending, closure = set(selected["eligible_indices"]), [provisional], set()
        while pending:
            i = pending.pop()
            if i in closure or i == qC or i not in eligible:
                continue
            closure.add(i)
            c = candidates[i]
            predecessors = []
            if c["route"] == "radius":
                predecessors.append(self.predecessor(candidates, i, route="one_step"))
            if c["route"] == "mpc_rh":
                predecessors.append(self.predecessor(candidates, i, route="mpc_ol"))
            if c["profile"] == "full":
                predecessors.append(self.predecessor(candidates, i, profile="lean"))
            pending.extend(j for j in predecessors if j is not None)
        # Freeze which confirmation comparisons exist before reading D_confirm.
        atomic_json(self.out / "confirm" / "contrast_registry.json", {"candidate_indices": sorted(closure),
                     "selection_fixed": True, "state_matching": "exact depth, previous action, initial entropy bin",
                     "family_scope": "selected route and eligible simpler predecessors on the declared dataset"})
        contrasts = []
        donors = DonorBank(load_records(self.out / "explore")) if closure else None
        reference = self.role_scores(candidates[qC], "confirm")
        for i in sorted(closure):
            c = candidates[i]
            score = self.role_scores(c, "confirm")
            weights = score["counts"]
            control = dict(c, state_control="matched_shuffle", candidate_id="shuffle_" + c["candidate_id"])
            meta = self.predict_candidate(control, donors=donors, fallback=candidates[qC])
            destroyed = self.role_scores(control, "confirm")
            contrast = dict(kind="state", candidate_index=i, predecessor_index=qC,
                            differences=destroyed["loss"] - score["loss"], weights=weights)
            # An unmatched/invalid intervention cannot establish the mechanism.
            if (meta["shuffle_missing_sites"] or not meta["feasible"]
                    or not meta["shuffle_matched_sites"] or not meta["state_control_valid"]):
                contrast["p_value"] = 1.0
                contrast["intervention_valid"] = False
                contrast["invalid_reason"] = meta.get("failure_reason", "missing/mismatched state donors or infeasible execution")
            contrasts.append(contrast)
            for kind, route, profile in (("radius", "one_step", None), ("mpc", "mpc_ol", None), ("full", None, "lean")):
                if (kind == "radius" and c["route"] != "radius") or (kind == "mpc" and c["route"] != "mpc_rh") or (kind == "full" and c["profile"] != "full"):
                    continue
                pred = self.predecessor(candidates, i, route, profile)
                if pred is None or candidates[pred].get("execution_failure"):
                    contrasts.append(dict(kind=kind, candidate_index=i,
                              predecessor_index=qC if pred is None else pred,
                              available=False,
                              unavailable_reason="required simpler policy is absent or failed execution"))
                    continue
                simpler = self.role_scores(candidates[pred], "confirm")
                if kind == "full":
                    gain = reference["node_ce"] - score["node_ce"]
                    differences = (simpler["loss"] - score["loss"]) / gain if gain > 0 else np.zeros_like(score["loss"])
                else:
                    differences = score["loss"] - simpler["loss"]
                contrasts.append(dict(kind=kind, candidate_index=i, predecessor_index=pred,
                                      differences=differences, weights=weights))
        result = confirmation(provisional, qC, selected["eligible_indices"], contrasts,
                              reps=self.config["selection"]["bootstrap_reps"], seed=self.config["seed"] + 5)
        atomic_json(self.out / "confirm" / "decision.json", result)
        atomic_json(self.out / "confirm" / "paired_contrasts.json", contrasts)
        self.check_frozen()

    def _stage_admit(self):
        self.check_frozen()
        candidates = read_json(self.out / "select" / "candidates.json")
        confirmed = read_json(self.out / "confirm" / "decision.json")
        qC, qM = confirmed["comparator_index"], confirmed["confirmed_index"]
        lock = {"comparator_index": qC, "confirmed_index": qM,
                "comparator_hash": candidates[qC]["hash"], "confirmed_hash": candidates[qM]["hash"],
                "model_hashes": self.lock_hashes(), "config_sha256": self.config_hash,
                "registry_sha256": canonical_hash(candidates),
                "static_gate": "exact prepared graph schema and feature fingerprint; no unseen-input generalization claimed",
                "test_labels_accessed": False, "anchor_rule": ANCHOR_RULE,
                "score_contract": SCORE_VERSION, "calibration_protocol": "trajectory_holdout_v2"}
        atomic_json(self.out / "admit" / "pre_admission_lock.json", lock)
        ref, policy = self.role_scores(candidates[qC], "admit"), self.role_scores(candidates[qM], "admit")
        ac = self.config["admission"]
        nonlinear = policy["accuracy"] is None
        metric_summary = (paired_metric_interval(policy["probabilities"], ref["probabilities"],
                    ref["labels"], ref["sample_units"], self.config, reps=ac["bootstrap_reps"],
                    alpha=ac["alpha"], seed=self.config["seed"]+6) if nonlinear else None)
        result = admission(policy["loss"] - ref["loss"],
                           metric_differences=None if nonlinear else policy["accuracy"] - ref["accuracy"],
                           weights=ref["counts"], reps=ac["bootstrap_reps"], alpha=ac["alpha"],
                           epsilon_a=ac["epsilon_a"], epsilon_metric=ac["epsilon_metric"],
                           clip=ac["clip"], seed=self.config["seed"] + 6,
                           metric_summary=metric_summary, independent_graphs=independent_graphs(self.config))
        passed = bool(result["passed"])
        qD = qM if passed else qC
        result.update(comparator_index=qC, confirmed_index=qM, deployed_index=qD,
                      dataset=self.config["dataset"], official_validation_only=True,
                      exact_comparator_fallback=not passed, candidate_already_comparator=qM == qC)
        atomic_json(self.out / "admit" / "decision.json", result)
        arrays = dict(candidate_loss=policy["loss"], comparator_loss=ref["loss"],
                      core_ids=ref["core_ids"], counts=ref["counts"])
        if nonlinear:
            arrays.update(candidate_probabilities=policy["probabilities"],
                          comparator_probabilities=ref["probabilities"], labels=ref["labels"],
                          sample_units=ref["sample_units"], sample_ids=ref["sample_ids"])
        else:
            arrays.update(candidate_accuracy=policy["accuracy"], comparator_accuracy=ref["accuracy"])
        atomic_npz(self.out / "admit" / "paired_units.npz", **arrays)
        final_lock = dict(lock, deployed_index=qD, deployed_hash=candidates[qD]["hash"],
                          admission_sha256=file_hash(self.out / "admit" / "decision.json"))
        atomic_json(self.out / "policy_lock.json", final_lock)

    @torch.no_grad()
    def _stage_evaluate(self):
        """Open official test labels only after the deployment policy is sealed."""
        self.check_frozen()
        lock = read_json(self.out / "policy_lock.json")
        if lock["config_sha256"] != self.config_hash or lock["admission_sha256"] != file_hash(self.out / "admit" / "decision.json"):
            raise RuntimeError("Deployment/admission lock was modified before evaluation")
        candidates = read_json(self.out / "select" / "candidates.json")
        if lock["registry_sha256"] != canonical_hash(candidates):
            raise RuntimeError("Candidate registry changed after policy locking")
        qC, qD = lock["comparator_index"], lock["deployed_index"]
        E = candidates[qC]["ensemble"]
        nominal = next(c for c in candidates if c["route"] == "nominal" and c["ensemble"] == E)
        for candidate in (candidates[qC], candidates[qD], nominal):
            self.verify_prediction(candidate)
        paths = {"qC": self.probability_path(candidates[qC]), "qD": self.probability_path(candidates[qD]),
                 "qN": self.probability_path(nominal)}
        final = self.out / "final"
        final.mkdir(exist_ok=True)
        for name, path in paths.items():
            shutil.copyfile(path, final / f"{name}_probabilities.npy")
        q0sum = None
        for seed in self.config["training"]["seeds"][:E]:
            model = load_plant(self.data, self.config, self.out / "train", seed, self.device, baseline=True)
            p, _ = predict_dataset(model, self.data, self.graph, self.config, device=self.device)
            if not np.isfinite(p).all():
                raise FloatingPointError("Feature-only reference q0 is nonfinite; test labels remain unopened")
            q0sum = p.astype(np.float64) if q0sum is None else q0sum + p
        atomic_npy(final / "q0_probabilities.npy", (q0sum / E).astype(np.float32))
        ids = self.data.splits["test"]
        state_path = self.out / "run_state.json"
        if state_path.exists():
            state = read_json(state_path)
            state["test_labels_accessed"] = True
            atomic_json(state_path, state)
        labels = role_labels(self.data, "test")
        spec = task_spec(self.config)
        results, scores, observed = {}, {}, {}
        units = self.data.cluster_ids[ids]
        observations = {"node_ids": np.asarray(ids), "sample_ids": np.asarray(ids), "labels": labels,
                        "cluster_ids": units}
        for name in ("q0", "qN", "qC", "qD"):
            prob = np.load(final / f"{name}_probabilities.npy", mmap_mode="r", allow_pickle=False)[ids]
            observations[f"{name}_probabilities"] = np.asarray(prob)
            observed[name] = prob
            score = scores[name] = score_observations(prob, labels, units, self.config)
            row = {"metric_name": spec.metric, "metric": score["metric"], "metric_percent": 100*score["metric"],
                   "loss_name": score["loss_name"], "loss_nats": score["node_ce"],
                   "test_samples": len(ids), "test_units": len(score["core_ids"]),
                   "prediction_level": spec.prediction_level,
                   "ensemble": candidates[qD]["ensemble"] if name == "qD" else E,
                   "probability_sha256": file_hash(final / f"{name}_probabilities.npy")}
            if spec.metric == "accuracy":
                row.update(accuracy=score["metric"], accuracy_percent=100*score["metric"],
                           cross_entropy_nats=score["node_ce"], test_nodes=len(ids))
            else:
                row[spec.metric] = score["metric"]
                row["binary_cross_entropy_nats" if spec.task_type == "multilabel" else "cross_entropy_nats"] = score["node_ce"]
            results[name] = row
        weights = scores["qC"]["counts"]
        contrast = paired_bootstrap(scores["qD"]["loss"] - scores["qC"]["loss"], weights=weights,
                    reps=self.config["admission"]["bootstrap_reps"], alpha=.025, seed=self.config["seed"]+7)
        if spec.metric == "accuracy":
            metric_contrast = paired_bootstrap(100*(scores["qD"]["accuracy"] - scores["qC"]["accuracy"]),
                        weights=weights, reps=self.config["admission"]["bootstrap_reps"],
                        alpha=.025, seed=self.config["seed"]+7)
        else:
            metric_contrast = paired_metric_interval(observed["qD"], observed["qC"], labels, units,
                        self.config, reps=self.config["admission"]["bootstrap_reps"],
                        alpha=.025, seed=self.config["seed"]+7, scale=100)
        admitted = read_json(self.out / "admit" / "decision.json")
        if not admitted["passed"] and file_hash(final / "qD_probabilities.npy") != file_hash(final / "qC_probabilities.npy"):
            raise AssertionError("Rejected deployment must be byte-identical to comparator predictions")
        collection = independent_graphs(self.config)
        result = {"version": __version__, "dataset": self.config["dataset"], "profile": self.config["profile"],
                  "benchmark_dataset": self.config.get("benchmark_dataset", self.config["dataset"]),
                  "scientific_run": self.config["scientific_run"], "status": "COMPUTED_FROM_THIS_RUN",
                  "task": vars(spec), "models": results,
                  "paired_qD_minus_qC_loss": contrast, "paired_qD_minus_qC_metric_pp": metric_contrast,
                  "original_manuscript_results_reproduced": False, "manuscript_agreement": "NOT_ASSESSED",
                  "execution_unit": "complete independent graph with fresh policy state" if collection else "complete transductive graph",
                  "scoring_unit": "equally weighted complete graph losses" if collection else "node-count-weighted core loss means",
                  "ensemble_seeds": {"qC": self.config["training"]["seeds"][:E],
                                     "qD": self.config["training"]["seeds"][:candidates[qD]["ensemble"]]},
                  "comparator": candidates[qC], "deployed": candidates[qD],
                  "admitted": admitted["passed"], "exact_fallback": not admitted["passed"],
                  "statistical_scope": "paired graph bootstrap conditional on the fixed trained ensemble" if collection else "descriptive dependent-core inference",
                  "metric_definition": spec.metric,
                  "seed_scope": "one fixed prefix ensemble; retraining-seed uncertainty is not estimated",
                  "cost_scope": "measured serial execution and explicit sparse/dense work ledger",
                  "test_can_change_policy": False}
        paired = dict(core_ids=scores["qC"]["core_ids"], counts=weights,
                      qC_loss=scores["qC"]["loss"], qD_loss=scores["qD"]["loss"])
        if spec.metric == "accuracy":
            result.update(paired_qD_minus_qC_ce=contrast, paired_qD_minus_qC_accuracy_pp=metric_contrast,
                          accuracy_definition="fraction of correct labels on the declared official test split")
            paired.update(qC_ce=scores["qC"]["loss"], qD_ce=scores["qD"]["loss"],
                          qC_accuracy=scores["qC"]["accuracy"], qD_accuracy=scores["qD"]["accuracy"])
        atomic_npz(final / "test_predictions.npz", **observations)
        result["test_prediction_record_sha256"] = file_hash(final / "test_predictions.npz")
        atomic_json(final / "results_summary.json", result)
        atomic_npz(final / "paired_test_units.npz", **paired)
        with open(final / "results.csv", "w", newline="") as stream:
            fields = ["model", "ensemble", "metric_name", "metric", "metric_percent", "loss_name", "loss_nats", "test_samples", "test_units"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for name, row in results.items():
                writer.writerow({"model": name, **{key: row[key] for key in fields if key != "model"}})
        print(json.dumps({"dataset": result["dataset"], "profile": result["profile"], "admitted": result["admitted"],
                          "metric": spec.metric, "test_metric": results["qD"]["metric"],
                          "route": candidates[qD]["route"]}, indent=2), flush=True)

    # Public stage entry points always pass through the same verified lifecycle.
    # The private _stage_* functions are deliberately separate for focused tests.
    def stage_prepare(self):
        return self.run("prepare")

    def stage_train(self):
        return self.run("train")

    def stage_explore(self):
        return self.run("explore")

    def stage_fit(self):
        return self.run("fit")

    def stage_select(self):
        return self.run("select")

    def stage_confirm(self):
        return self.run("confirm")

    def stage_admit(self):
        return self.run("admit")

    def stage_evaluate(self):
        return self.run("evaluate")
