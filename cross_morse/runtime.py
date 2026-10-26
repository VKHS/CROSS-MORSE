"""Graph-local execution shared by training-independent policy and replay paths."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .common import precision_context, stop_at_boundary
from .core import GraphOperator
from .policies import execute_policy
from .tasks import probabilities, task_spec
from .training import native_forward


def is_graph_collection(data):
    ptr = getattr(data, "graph_ptr", None)
    return ptr is not None and len(ptr) > 2


def graph_operator(data, device):
    return GraphOperator(data.edge_index, data.num_nodes, device,
                         edge_weight=getattr(data, "edge_weight", None),
                         graph_ptr=getattr(data, "graph_ptr", None))


def prediction_rows(data):
    if data.metadata.get("label_axis", "node") == "graph":
        return len(data.graph_ptr)-1
    return data.num_nodes


@torch.no_grad()
def predict_dataset(model, data, graph, config, candidate=None, controller=None,
                    reference=None, donors=None, device="cpu", output_path=None):
    """Execute each independent graph with fresh sensor, observer and policy state.

No label accessor is called here. An optional output path streams corpus
predictions into an atomic NumPy memmap rather than retaining every graph's
probabilities in accelerator or host memory. Checkpoint replay uses the same
episode boundaries and ordering.
    """
    if not is_graph_collection(data):
        graph = graph if graph is not None else graph_operator(data, device)
        if candidate is not None:
            return execute_policy(model, data.x.to(device), graph, config, candidate,
                                  controller, reference, donors=donors)
        graph.reset_counters()
        model.reset_counters()
        started = time.perf_counter()
        with precision_context(config, device):
            p = probabilities(native_forward(model, data.x.to(device), graph, 0).float(),
                              getattr(model, "task", task_spec(config))).cpu().numpy()
        return p, {"wall_seconds": time.perf_counter()-started, "actions": [],
                   "plant_operations": dict(model.counters),
                   "multiply_add_work": model.counters["dense_multiply_adds"]}

    rows, outputs = prediction_rows(data), model.num_classes
    temporary = None
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
        os.close(fd)
        result = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(rows, outputs))
    else:
        result = np.empty((rows, outputs), dtype=np.float32)
    totals = {key: 0 for key in ("multiply_add_work", "deployed_multiply_add_work",
              "qualification_multiply_add_work", "support_violations", "shuffle_matched_sites",
              "shuffle_missing_sites", "a9_executions", "native_site_count",
              "inserted_site_count", "inserted_a0_bypass_count")}
    totals.update(feasible=True, state_control_valid=True,
                  native_site_action_counts=[0]*10, inserted_site_action_counts=[0]*10)
    action_digest, written = hashlib.sha256(), 0
    started = time.perf_counter()
    try:
        for graph_id in range(len(data.graph_ptr)-1):
            episode = data.graph(graph_id)
            op = graph_operator(episode, device)
            x = episode.x.to(device)
            if candidate is None:
                op.reset_counters()
                model.reset_counters()
                with precision_context(config, device):
                    p = probabilities(native_forward(model, x, op, 0).float(), getattr(model, "task", task_spec(config))).cpu().numpy()
                trace = {"multiply_add_work": model.counters["dense_multiply_adds"], "actions": []}
            else:
                p, trace = execute_policy(model, x, op, config, candidate, controller, reference, donors=donors)
            ids = np.asarray(episode.global_sample_ids, dtype=np.int64)
            if p.shape != (len(ids), outputs) or not np.isfinite(p).all():
                raise FloatingPointError(f"Invalid predictions for graph {graph_id}")
            result[ids] = p
            written += len(ids)
            for key in ("multiply_add_work", "qualification_multiply_add_work", "support_violations",
                        "shuffle_matched_sites", "shuffle_missing_sites", "native_site_count",
                        "inserted_site_count", "inserted_a0_bypass_count"):
                totals[key] += trace.get(key, 0)
            totals["deployed_multiply_add_work"] += trace.get("deployed_multiply_add_work", trace.get("multiply_add_work", 0))
            totals["a9_executions"] += trace.get("actions", []).count(9)
            for key in ("native_site_action_counts", "inserted_site_action_counts"):
                totals[key] = [a+b for a,b in zip(totals[key], trace.get(key, [0]*10))]
            totals["feasible"] &= trace.get("feasible", True)
            control_valid = trace.get("state_control_valid")
            totals["state_control_valid"] &= True if control_valid is None else bool(control_valid)
            action_digest.update(json.dumps({"graph_id": graph_id, "actions": trace.get("actions", []),
                                "safe_depths": trace.get("safe_depths", [])}, sort_keys=True).encode())
            del x, op, episode
            stop_at_boundary()
        if written != rows:
            raise ValueError("Graph prediction rows do not cover the prepared label axis")
        totals.update(wall_seconds=time.perf_counter()-started, graphs=len(data.graph_ptr)-1,
                      prediction_rows=rows, actions_sha256=action_digest.hexdigest(),
                      execution_unit="one complete graph with fresh controller state",
                      trace_scope="summed work and failures; ordered per-graph actions and anchors bound by digest",
                      labels_accessed_during_prediction=False,
                      transport_history_definition="site-aware kernel mass and expected kernel hops before source retention; inserted A0 is zero")
        if temporary is not None:
            result.flush()
            del result
            os.replace(temporary, path)
            temporary = None
            result = np.load(path, mmap_mode="r", allow_pickle=False)
        return result, totals
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
