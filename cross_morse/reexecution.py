"""Label-free checkpoint reexecution and repeat-timing helpers.

Call only on trusted local checkpoints. Numeric review in review.py never loads
these Python-pickled model payloads. This module never refits or changes policy.
"""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import torch

from .review import NumericEvidence, EvidenceError
from .provenance import current_source_sha256
from .data import load_data
from .learning import Controller
from .training import load_plant
from .common import synchronize
from .tasks import task_spec
from .runtime import graph_operator, is_graph_collection, predict_dataset


class FrozenRunEvaluator:
    def __init__(self, run_root, device="cpu", names=("q0", "qN", "qC", "qD")):
        self.root = Path(run_root).resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise EvidenceError("CUDA requested but unavailable; no automatic CPU substitution")
        self.evidence = e = NumericEvidence(self.root)
        if current_source_sha256() != e.identities["source_sha256"]:
            raise EvidenceError("Checkpoint replay requires the original source snapshot")
        self.config = e.json("config.json")
        self.summary = e.json("final/results_summary.json")
        candidates = e.json("select/candidates.json")
        lock = e.json("policy_lock.json")
        self.candidates = {"qC": candidates[lock["comparator_index"]], "qD": candidates[lock["deployed_index"]]}
        self.candidates["qN"] = next(c for c in candidates if c["route"] == "nominal"
                                          and c["ensemble"] == self.candidates["qC"]["ensemble"])
        for rel in ("prepared/metadata.json", "prepared/graph.npz", "prepared/splits.npz", "prepared/clusters.npz"):
            e.file(rel)
        self.data = load_data(self.root).protect_labels()
        self.graph = None if is_graph_collection(self.data) else graph_operator(self.data, self.device)
        self.task = task_spec(self.config, self.data.metadata)
        self.reference = e.arrays("explore/nominal_reference.npz")
        self.models, self.controllers = {}, {}
        self.names = tuple(names)
        for name in names:
            if name not in ("q0", "qN", "qC", "qD"):
                raise ValueError(f"Unknown predictor: {name}")
            candidate = self.candidates["qC" if name == "q0" else name]
            for seed in self.config["training"]["seeds"][:candidate["ensemble"]]:
                key = (name == "q0", seed)
                if key not in self.models:
                    prefix = "q0" if name == "q0" else "plant"
                    e.file(f"train/{prefix}_seed{seed}_best.pt")
                    self.models[key] = load_plant(self.data, self.config, self.root / "train", seed,
                                                   self.device, baseline=name == "q0")
            if name != "q0" and candidate["route"] in ("phase", "fixed_sequence", "mpc_ol", "mpc_rh", "one_step", "radius"):
                profile = candidate["profile"] if candidate["profile"] != "none" else "full"
                if profile not in self.controllers:
                    path = e.file(f"fit/{profile}/controller.pt")
                    self.controllers[profile] = Controller.load(path, "cpu")

    @torch.no_grad()
    def predict(self, name):
        if name not in self.names:
            raise ValueError("Predictor was not preloaded")
        c = self.candidates["qC" if name == "q0" else name]
        aggregate, traces = None, []
        synchronize(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        for seed in self.config["training"]["seeds"][:c["ensemble"]]:
            model = self.models[(name == "q0", seed)]
            profile = c["profile"] if c["profile"] != "none" else "full"
            controller = self.controllers.get(profile) if name != "q0" and c["route"] in (
                "phase", "fixed_sequence", "mpc_ol", "mpc_rh", "one_step", "radius") else None
            p, trace = predict_dataset(model, self.data, self.graph, self.config,
                candidate=None if name == "q0" else c, controller=controller,
                reference=self.reference, device=self.device)
            trace["seed"] = seed
            aggregate = p.astype(np.float64) if aggregate is None else aggregate + p
            traces.append(trace)
        out = (aggregate / c["ensemble"]).astype(np.float32)
        synchronize(self.device)
        report = {"wall_seconds": time.perf_counter() - start, "ensemble": c["ensemble"],
                  "multiply_add_work": sum(t.get("multiply_add_work", t.get("plant_operations", {}).get("dense_multiply_adds", 0)) for t in traces),
                  "qualification_work": sum(t.get("qualification_multiply_add_work", 0) for t in traces),
                  "a9_executions": sum(t.get("a9_executions", t.get("actions", []).count(9)) for t in traces),
                  "anchor_legacy_predicate_disagreements": sum(
                      bool(d.get("anchor_would_fail_legacy_observer_check")) for t in traces for d in t.get("decisions", [])),
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
                  "no_test_labels_used_for_execution": True,
                  "timing_scope": "preloaded models; synchronized full policy execution, CPU copy and ensemble aggregation; includes qualification sensing and trace collection"}
        return out, report

    def compare(self, name, atol=1e-6, rtol=1e-5):
        expected = np.load(self.evidence.file(f"final/{name}_probabilities.npy"), mmap_mode="r", allow_pickle=False)
        actual, trace = self.predict(name)
        if actual.shape != expected.shape:
            raise EvidenceError(f"Prediction shape mismatch for {name}")
        trace.update(predictor=name, agrees=bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
                     max_absolute_probability_error=float(np.max(np.abs(actual.astype(float) - expected))),
                     atol=atol, rtol=rtol, rows=int(len(actual)))
        if self.task.task_type == "multilabel":
            trace["threshold_disagreements"] = int(np.sum((actual >= .5) != (expected >= .5)))
        else:
            trace["argmax_disagreements"] = int(np.sum(actual.argmax(1) != expected.argmax(1)))
        return trace
