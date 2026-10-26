"""Train before instrumentation; all later stages load immutable checkpoints."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .common import (atomic_json, atomic_torch, atomic_npz, file_hash, read_json,
                     precision_context, seed_everything, stop_at_boundary, synchronize)
from .core import GraphOperator, build_plant
from .data import role_labels
from .statistics import cluster_summary, paired_bootstrap
from .tasks import (task_spec, loss as task_loss, probabilities, sample_losses,
                    metric_score, valid_sample_mask)


def new_plant(data, config, device):
    m = config["model"]
    spec = task_spec(config, data.metadata)
    if spec.num_outputs is None:
        raise ValueError("Prepared data must declare the output count without inspecting labels")
    return build_plant(data.x.shape[1], spec.num_outputs, m, spec).to(device)


def load_plant(data, config, directory, seed, device, baseline=False):
    model = new_plant(data, config, device)
    prefix = "q0" if baseline else "plant"
    payload = torch.load(Path(directory) / f"{prefix}_seed{seed}_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.freeze()
    return model


def native_forward(model, x, graph, depth, checkpoint_segment=0):
    model.bind_graph(graph)
    h0 = model.encode(x)
    h = h0
    if checkpoint_segment and model.training and depth:
        for start in range(0, depth, checkpoint_segment):
            count = min(checkpoint_segment, depth - start)
            def segment(state, source, count=count, start=start):
                for site in range(start, start + count):
                    state = model.native_step(state, source, graph, site=site)
                return state
            h = checkpoint(segment, h, h0, use_reentrant=False)
    else:
        for site in range(depth):
            h = model.native_step(h, h0, graph, site=site)
    return model.logits(h)


def _independent_graphs(data):
    return getattr(data, "graph_ptr", None) is not None


def _episode_operator(episode, device):
    return GraphOperator(episode.edge_index, len(episode.x), device,
                         edge_weight=getattr(episode, "edge_weight", None),
                         graph_ptr=getattr(episode, "graph_ptr", None))


def _independent_epoch(model, data, config, depth, optimizer, params, device):
    """Accumulate a minibatch over whole graphs without a union GPU tensor.

    A batch is weighted by its observed prediction rows (nodes for superpixels,
    graphs for Peptides); each graph's activations are released after backward.
    The optimizer steps after graph_batch_size complete graph episodes.
    """
    spec, tr = model.task, config["training"]
    graph_ids = np.asarray(data.graph_splits["fit_train"], dtype=np.int64)
    if len(graph_ids) == 0:
        raise ValueError("D_fit training contains no complete graphs")
    graph_ids = graph_ids[torch.randperm(len(graph_ids)).numpy()]
    batch_size = int(tr.get("graph_batch_size", 1))
    if batch_size < 1:
        raise ValueError("graph_batch_size must be positive")
    total_loss, total_rows = 0., 0
    for start in range(0, len(graph_ids), batch_size):
        batch = []
        for graph_id in graph_ids[start:start + batch_size]:
            episode = data.graph(int(graph_id))
            labels = role_labels(episode, "fit_train")
            count = int(valid_sample_mask(labels, spec).sum())
            if count:
                batch.append((episode, labels, count))
        rows = sum(count for _, _, count in batch)
        if not rows:
            continue
        optimizer.zero_grad(set_to_none=True)
        for episode, labels, count in batch:
            graph = _episode_operator(episode, device)
            ids = torch.as_tensor(episode.splits["fit_train"], dtype=torch.long, device=device)
            target = torch.as_tensor(labels, device=device)
            with precision_context(config, device):
                logits = native_forward(model, episode.x.to(device), graph, depth,
                                        config["model"].get("checkpoint_segment", 0))
                value = task_loss(logits[ids], target, spec)
            if not torch.isfinite(value):
                raise FloatingPointError("Nonfinite training loss in a complete graph episode")
            (value * (count / rows)).backward()
            total_loss += float(value.detach()) * count
            total_rows += count
        torch.nn.utils.clip_grad_norm_(params, tr.get("grad_clip", 1.))
        optimizer.step()
    if not total_rows:
        raise ValueError("D_fit training has no observed targets")
    return total_loss / total_rows


@torch.no_grad()
def _validation(model, data, graph, config, depth, device, cached=None):
    """Evaluate only internal-fit labels and recompute the complete task metric."""
    spec = model.task
    if not _independent_graphs(data):
        x, ids, target = cached
        with precision_context(config, device):
            logits = native_forward(model, x, graph, depth)[ids]
            value = float(task_loss(logits, target, spec))
        p = probabilities(logits, spec).cpu().numpy()
        return value, metric_score(p, target.cpu().numpy(), spec)
    predictions, targets = [], []
    total_loss, total_rows = 0., 0
    for graph_id in data.graph_splits["fit_val"]:
        episode = data.graph(int(graph_id))
        labels = role_labels(episode, "fit_val")
        count = int(valid_sample_mask(labels, spec).sum())
        graph = _episode_operator(episode, device)
        ids = torch.as_tensor(episode.splits["fit_val"], dtype=torch.long, device=device)
        with precision_context(config, device):
            logits = native_forward(model, episode.x.to(device), graph, depth)[ids]
            if count:
                total_loss += float(task_loss(logits, torch.as_tensor(labels, device=device), spec)) * count
                total_rows += count
        predictions.append(probabilities(logits, spec).cpu().numpy())
        targets.append(labels)
    if not total_rows:
        raise ValueError("Internal D_fit validation has no observed targets")
    return total_loss / total_rows, metric_score(np.concatenate(predictions), np.concatenate(targets), spec)


def _checkpoint_improved(spec, value, metric, best_loss, best_metric):
    if spec.metric == "ap":
        if not np.isfinite(metric):
            raise ValueError("Peptides AP checkpoint selection needs both label classes in an observed validation task")
        return metric > best_metric or (metric == best_metric and value < best_loss)
    return value < best_loss


def train_one(data, graph, config, directory, seed, device, baseline=False):
    """Resume exactly at epoch boundaries, including optimizer/scheduler/RNG."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "q0" if baseline else "plant"
    done = directory / f"{prefix}_seed{seed}_done.json"
    if done.exists():
        return read_json(done)
    last = directory / f"{prefix}_seed{seed}_last.pt"
    best = directory / f"{prefix}_seed{seed}_best.pt"
    seed_everything(seed + (100_000 if baseline else 0))
    model = new_plant(data, config, device)
    spec = model.task
    tr = config["training"]
    depth = 0 if baseline else config["model"]["depth"]
    # The feature-only baseline has the same encoder/head schema but no graph
    # update. Unused update weights do not enter its optimization.
    params = list(model.encoder.parameters()) + list(model.head.parameters()) if baseline else list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=tr["lr"], weight_decay=tr["weight_decay"])
    schedule = tr.get("schedule", "cosine" if spec.dataset in {"ogbn-arxiv", "ogbn-proteins", "ogbn-products", "synthetic"} else "constant")
    if schedule not in {"cosine", "constant"}:
        raise ValueError("training.schedule must be cosine or constant")
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tr["epochs"])
                 if schedule == "cosine" else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.))
    independent = _independent_graphs(data)
    cached_validation = None
    if not independent:
        x = data.x.to(device)
        train_ids = torch.as_tensor(data.splits["fit_train"], dtype=torch.long, device=device)
        valid_ids = torch.as_tensor(data.splits["fit_val"], dtype=torch.long, device=device)
        train_y = torch.as_tensor(role_labels(data, "fit_train"), device=device)
        valid_y = torch.as_tensor(role_labels(data, "fit_val"), device=device)
        cached_validation = (x, valid_ids, valid_y)
    epoch0, best_loss, best_metric, stale, history = 0, float("inf"), -float("inf"), 0, []
    if last.exists():
        obj = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(obj["model"])
        optimizer.load_state_dict(obj["optimizer"])
        scheduler.load_state_dict(obj["scheduler"])
        epoch0, best_loss, stale, history = obj["next_epoch"], obj["best_loss"], obj["stale"], obj["history"]
        best_metric = obj.get("best_metric", -float("inf"))
        torch.set_rng_state(obj["torch_rng"].cpu())
        if torch.device(device).type == "cuda" and obj.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in obj["cuda_rng"]])
    started = time.perf_counter()
    for epoch in range(epoch0, tr["epochs"]):
        if stale >= tr["patience"]:
            break
        model.train()
        if independent:
            train_loss = _independent_epoch(model, data, config, depth, optimizer, params, device)
        else:
            optimizer.zero_grad(set_to_none=True)
            with precision_context(config, device):
                logits = native_forward(model, x, graph, depth, config["model"].get("checkpoint_segment", 0))
                value = task_loss(logits[train_ids], train_y, spec)
            if not torch.isfinite(value):
                raise FloatingPointError(f"Nonfinite training loss at {prefix}, seed {seed}, epoch {epoch}")
            value.backward()
            torch.nn.utils.clip_grad_norm_(params, tr.get("grad_clip", 1.0))
            optimizer.step()
            train_loss = float(value.detach())
        scheduler.step()
        model.eval()
        val_loss, val_metric = _validation(model, data, graph, config, depth, device, cached_validation)
        if not np.isfinite(val_loss):
            raise FloatingPointError("Nonfinite internal D_fit validation loss")
        improved = _checkpoint_improved(spec, val_loss, val_metric, best_loss, best_metric)
        stale = 0 if improved else stale + 1
        if improved:
            best_loss = val_loss
            best_metric = val_metric
            payload = {"model": model.state_dict(), "seed": seed, "epoch": epoch,
                       "internal_fit_val_loss": val_loss, "internal_fit_val_metric": val_metric,
                       "loss_name": spec.loss_name, "metric_name": spec.metric, "task": spec.to_dict(),
                       "depth": depth, "parameter_count": model.parameter_count,
                       "model_config": config["model"], "instrumentation": model.instrumentation_metadata()}
            payload[f"internal_fit_val_{spec.loss_name}"] = val_loss
            atomic_torch(best, payload)
        row = {"epoch": epoch, "train_loss": train_loss, "internal_fit_val_loss": val_loss,
               "internal_fit_val_metric": val_metric, "loss_name": spec.loss_name, "metric_name": spec.metric,
               "lr": scheduler.get_last_lr()[0]}
        row.update({f"train_{spec.loss_name}": train_loss, f"internal_fit_val_{spec.loss_name}": val_loss,
                    f"internal_fit_val_{spec.metric}": val_metric})
        history.append(row)
        atomic_torch(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "next_epoch": epoch + 1,
                           "best_loss": best_loss, "best_metric": best_metric, "stale": stale, "history": history,
                           "torch_rng": torch.get_rng_state(),
                           "cuda_rng": torch.cuda.get_rng_state_all() if torch.device(device).type == "cuda" else None})
        atomic_json(directory / f"{prefix}_seed{seed}_history.json", history)
        print(f"[train] {prefix} seed={seed} epoch={epoch+1}/{tr['epochs']} fit-val {spec.loss_name.upper()}={val_loss:.6f} {spec.metric}={val_metric:.6f}", flush=True)
        stop_at_boundary()
    result = {"seed": seed, "baseline": baseline, "epochs_completed": len(history),
              "best_internal_fit_val_loss": best_loss, "best_internal_fit_val_metric": best_metric,
              f"best_internal_fit_val_{spec.loss_name}": best_loss, "parameter_count": model.parameter_count,
              "active_parameter_count": sum(p.numel() for p in params),
              "checkpoint_sha256": file_hash(best), "seconds_this_attempt": time.perf_counter() - started,
              "checkpoint_selection_labels": "fit_val only", "official_validation_used": False,
              "checkpoint_rule": "max_ap_bce_tiebreak" if spec.metric == "ap" else "min_loss",
              "schedule": schedule, "task": spec.to_dict(), "instrumentation": model.instrumentation_metadata(),
              "execution": "whole-graph episodes with gradient accumulation" if independent else "exact full-graph sparse transport",
              "graph_batch_size": int(tr.get("graph_batch_size", 1)) if independent else None}
    atomic_json(done, result)
    return result


@torch.no_grad()
def _audit_member(plant, q0, data, graph, config, depths, ids, device):
    """Gather depth predictions in label-row order from graph-local executions."""
    outputs = plant.num_classes
    native_probs = np.empty((len(depths), len(ids), outputs), dtype=np.float32)
    adapter_probs = np.empty((len(ids), outputs), dtype=np.float32)
    q0_probs = np.empty_like(adapter_probs)
    order = np.argsort(ids)
    sorted_ids = ids[order]
    depth_lookup = {depth: index for index, depth in enumerate(depths)}
    independent = _independent_graphs(data)
    graph_ids = data.graph_splits["fit_val"] if independent else [None]
    max_error = 0.
    seen = np.zeros(len(ids), dtype=bool)
    for graph_id in graph_ids:
        episode = data.graph(int(graph_id)) if independent else data
        current_graph = _episode_operator(episode, device) if independent else graph
        local_ids = np.asarray(episode.splits["fit_val"], dtype=np.int64)
        global_ids = (np.asarray(episode.global_sample_ids)[local_ids] if independent else local_ids)
        where = np.searchsorted(sorted_ids, global_ids)
        if np.any(where >= len(ids)) or not np.array_equal(sorted_ids[where], global_ids):
            raise ValueError("Graph episode validation rows differ from the declared D_fit split")
        positions = order[where]
        if seen[positions].any():
            raise ValueError("A validation prediction row appears in multiple graph episodes")
        seen[positions] = True
        tid = torch.as_tensor(local_ids, dtype=torch.long, device=device)
        x = episode.x.to(device)
        plant.bind_graph(current_graph)
        q0.bind_graph(current_graph)
        with precision_context(config, device):
            q0_probs[positions] = probabilities(q0.logits(q0.encode(x))[tid], plant.task).cpu().numpy()
            h0 = plant.encode(x)
            h_native, h_adapter = h0, h0
            for site in range(config["model"]["depth"]):
                h_native = plant.native_step(h_native, h0, current_graph, site=site)
                h_adapter = plant.step(h_adapter, h0, current_graph, 0, site=site)
                denom = max(float(h_native.float().norm()), 1e-12)
                max_error = max(max_error, float((h_native.float() - h_adapter.float()).norm()) / denom)
                if site + 1 in depth_lookup:
                    native_probs[depth_lookup[site + 1], positions] = probabilities(
                        plant.logits(h_native)[tid], plant.task).cpu().numpy()
            adapter_probs[positions] = probabilities(plant.logits(h_adapter)[tid], plant.task).cpu().numpy()
    if not seen.all():
        raise ValueError("D_fit validation contains rows omitted by graph-local execution")
    return native_probs, adapter_probs, q0_probs, max_error


def _risk_mean(values, clusters, independent):
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Cannot qualify a plant without observed validation targets")
    if independent:
        means, _, _ = cluster_summary(values[finite], clusters[finite])
        return float(means.mean())
    return float(values[finite].mean())


@torch.no_grad()
def audit_training(data, graph, config, directory, device):
    """Training-only E1/depth qualification with official task metrics.

    Independent graphs are never concatenated for GPU execution. Their risk
    averages within each graph and then equally across graphs; transductive risk
    remains the node-weighted core mean. F1/AUC/AP are recomputed from complete
    prediction observations, not from a per-row accuracy surrogate.
    """
    directory = Path(directory)
    depth = int(config["model"]["depth"])
    depths = sorted(set([1, depth] + [2**i for i in range(depth.bit_length()) if 2**i <= depth]))
    ids = np.asarray(data.splits["fit_val"], dtype=np.int64)
    labels = role_labels(data, "fit_val")
    clusters = data.cluster_ids[ids]
    spec = task_spec(config, data.metadata)
    independent = _independent_graphs(data)
    sum_probs, sum_q0 = None, None
    identity_records = []
    instrumentation = None
    for seed in config["training"]["seeds"]:
        plant = load_plant(data, config, directory, seed, device)
        q0 = load_plant(data, config, directory, seed, device, baseline=True)
        probs, adapter_probs, p0, max_error = _audit_member(plant, q0, data, graph, config, depths, ids, device)
        native_loss = sample_losses(probs[-1], labels, spec)
        adapter_loss = sample_losses(adapter_probs, labels, spec)
        deviation = abs(_risk_mean(native_loss, clusters, independent) - _risk_mean(adapter_loss, clusters, independent))
        state_tolerance = .008 if config["model"]["precision"] == "bf16" else 1e-5
        loss_tolerance = 1e-6 if spec.dataset == "Roman-empire" else 1e-4
        identity_records.append({"seed": seed, "max_relative_state_error": max_error,
                                 "absolute_fit_val_loss_deviation": float(deviation),
                                 "state_tolerance": state_tolerance, "loss_tolerance": loss_tolerance,
                                 "pass": bool(max_error <= state_tolerance and deviation <= loss_tolerance)})
        sum_probs = probs if sum_probs is None else sum_probs + probs
        sum_q0 = p0 if sum_q0 is None else sum_q0 + p0
        instrumentation = plant.instrumentation_metadata()
    members = len(config["training"]["seeds"])
    if not members:
        raise ValueError("Plant qualification requires at least one frozen seed")
    probs, q0p = sum_probs / members, sum_q0 / members
    losses = np.stack([sample_losses(p, labels, spec) for p in probs])
    q0loss = sample_losses(q0p, labels, spec)
    finite = np.isfinite(losses[-1]) & np.isfinite(q0loss)
    means, core_ids, counts = cluster_summary((losses[-1] - q0loss)[finite], clusters[finite])
    bound = paired_bootstrap(means, weights=None if independent else counts,
                             reps=config["selection"]["bootstrap_reps"],
                             alpha=config["selection"]["alpha"], seed=config["seed"] + 1)
    metrics = np.array([metric_score(p, labels, spec) for p in probs])
    metric_available = bool(np.isfinite(metrics).all())
    pdd = float(metrics.max() - metrics[-1]) if metric_available else float("nan")
    risk = np.array([_risk_mean(value, clusters, independent) for value in losses])
    q0risk = _risk_mean(q0loss, clusters, independent)
    log_depth = np.log2(depths)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    dil = float(integrate(risk, log_depth) / (log_depth[-1] - log_depth[0])) if len(depths) > 1 else float(risk[-1])
    dil_gap = dil - q0risk
    entry = config["entry"]
    result = {"nominal_identity": identity_records, "identity_pass": all(r["pass"] for r in identity_records),
              "plant_vs_q0": bound, "depth_grid": depths, "depth_loss": risk, "depth_metric": metrics,
              "loss_name": spec.loss_name, "metric_name": spec.metric, "metric_available": metric_available,
              "task": spec.to_dict(), "instrumentation": instrumentation,
              "pdd_fraction": pdd, "dil_log_depth": dil, "dil_gap": dil_gap,
              "dil_reference": "feature-only q0 loss repeated on the depth grid (reconstruction choice)",
              "plant_gate_pass": bound["upper"] < -entry["plant_margin"],
              "depth_gate_pass": metric_available and pdd <= entry["pdd_max"] and dil_gap <= entry["dil_gap_max"],
              "labels": "internal D_fit validation",
              "risk_weighting": "equal graphs after within-graph mean" if independent else "node-weighted statistical cores",
              "statistical_scope": "descriptive whole-graph bootstrap" if independent else "descriptive dependent-core bootstrap"}
    result[f"depth_{spec.loss_name}"] = risk
    result[f"depth_{spec.metric}"] = metrics
    result["initial_entry_pass"] = result["identity_pass"] and result["plant_gate_pass"] and result["depth_gate_pass"]
    atomic_json(directory / "entry_initial.json", result)
    atomic_npz(directory / "entry_depth_values.npz", sample_ids=ids, node_ids=ids, core_ids=clusters,
               depth_grid=np.array(depths), losses=losses, q0_loss=q0loss)
    return result


def train(data, graph, config, output_dir, device):
    directory = Path(output_dir) / "train"
    results = []
    for seed in config["training"]["seeds"]:
        for baseline in (True, False):
            results.append(train_one(data, graph, config, directory, seed, device, baseline))
            stop_at_boundary()
    audit = audit_training(data, graph, config, directory, device)
    if not audit["identity_pass"]:
        raise RuntimeError("Nominal adapter identity failed; downstream control is invalid.")
    atomic_json(directory / "manifest.json", {"checkpoints": results, "entry_audit": audit,
                                              "weights_frozen_after_training": True})
    return audit
