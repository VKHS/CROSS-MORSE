"""Propensity-logged trajectories and actual frozen-plant counterfactuals.

Only D_explore labels reach this module. One trajectory is one complete
full-graph rollout of one frozen checkpoint. Repeated steps are not treated as
independent support observations. No reported manuscript results are injected.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .common import atomic_json, atomic_npz, precision_context, seed_everything, stop_at_boundary
from .core import (ACTIONS, MeasurementBank, GraphOperator,
                   site_transport_accounting, site_action_summary)
from .data import role_labels
from .tasks import probabilities, loss as task_loss, sample_losses, task_spec
from .statistics import cluster_summary
from .training import load_plant
from .recovery import ANCHOR_RULE, label_free_safe


def sensor_bank(graph, model, config, profile="full"):
    s = config["sensor"]
    return MeasurementBank(graph, model.hidden_dim, bands=s["bands"], degree=s["degree"],
                           sketch_dim=s["sketch_dim"], seed=config["seed"], profile=profile,
                           task=getattr(model, "task", task_spec(config)), plant=model)



@torch.no_grad()
def collect_trajectory(model, x, graph, labels, ids, core_ids, config, trajectory, seed_index, device):
    depth = config["model"]["depth"]
    spec = getattr(model, "task", task_spec(config))
    model.bind_graph(graph)
    rng = np.random.default_rng(config["seed"] + 1_000_003 * (trajectory + 1))
    propensities = np.array([a.probability for a in ACTIONS], dtype=np.float64)
    sampled_actions = rng.choice(len(ACTIONS), size=depth, p=propensities)
    n_terminal = min(depth, config["exploration"]["terminal_samples_per_trajectory"])
    terminal_sites = set(rng.choice(depth, size=n_terminal, replace=False).tolist())
    horizon = config["exploration"]["critic_horizon"]
    stride = config["exploration"].get("record_stride", 1)
    records = {k: [] for k in ("y", "y_next", "action", "previous_action", "depth", "trajectory",
                               "delta", "stage_delta", "terminal_delta", "propensity", "current_loss", "seed_index",
                               "safe_depth", "anchor_updated")}
    start = time.perf_counter()
    graph.reset_counters()
    model.reset_counters()
    with precision_context(config, device):
        h0 = model.encode(x)
        h, last_safe = h0, h0
        last_safe_depth = 0
        all_safe_depths = []
        logits0 = model.logits(h0)
        bank = sensor_bank(graph, model, config)
        bank.prepare(h0, logits0)
        y = bank.measure(h, logits0, 0, depth).cpu().numpy()
        current_loss = float(task_loss(logits0[ids].float(), labels, spec))
        mass, hops, previous_action = 0., 0., -1
        for d, a in enumerate(sampled_actions.tolist()):
            anchor_updated = label_free_safe(y)
            if anchor_updated:
                last_safe, last_safe_depth = h, d
            all_safe_depths.append(last_safe_depth)
            actual = model.step(h, h0, graph, a, last_safe, site=d)
            actual_logits = model.logits(actual)
            next_loss = float(task_loss(actual_logits[ids].float(), labels, spec))
            nominal = model.native_step(h, h0, graph, site=d)
            alternate = actual
            for offset in range(1, min(horizon, depth - d)):
                nominal = model.native_step(nominal, h0, graph, site=d+offset)
                alternate = model.native_step(alternate, h0, graph, site=d+offset)
            nominal_loss = float(task_loss(model.logits(nominal)[ids].float(), labels, spec))
            alternate_loss = next_loss if min(horizon, depth - d) == 1 else float(task_loss(model.logits(alternate)[ids].float(), labels, spec))
            terminal = np.nan
            if d in terminal_sites:
                continuation = h
                for offset in range(depth - d):
                    continuation = model.native_step(continuation, h0, graph, site=d+offset)
                terminal = float(task_loss(model.logits(continuation)[ids].float(), labels, spec)) - current_loss
            accounting = site_transport_accounting(a, d, model)
            mass += accounting["mass"]
            hops += accounting["hops"]
            next_y = bank.measure(actual, actual_logits, d + 1, depth, mass, hops).cpu().numpy()
            if d % stride == 0:
                for name, value in {"y": y, "y_next": next_y, "action": a,
                                    "previous_action": previous_action, "depth": d, "trajectory": trajectory,
                                    "delta": alternate_loss - nominal_loss, "stage_delta": next_loss - current_loss,
                                    "terminal_delta": terminal, "propensity": propensities[a],
                                    "current_loss": current_loss, "seed_index": seed_index,
                                    "safe_depth": last_safe_depth, "anchor_updated": int(anchor_updated)}.items():
                    records[name].append(value)
            h, y, current_loss, previous_action = actual, next_y, next_loss, a
        probabilities_final = probabilities(model.logits(h)[ids].float(), spec).cpu().numpy()
    label_np = labels.cpu().numpy()
    losses = sample_losses(probabilities_final, label_np, spec)
    valid = np.isfinite(losses)
    if not valid.any():
        raise ValueError("Exploration episode has no observed targets")
    losses, core_ids = losses[valid], np.asarray(core_ids)[valid]
    core_loss, unique_ids, core_counts = cluster_summary(losses, core_ids)
    arrays = {}
    integer_names = {"action", "previous_action", "depth", "trajectory", "seed_index", "safe_depth", "anchor_updated"}
    for name, values in records.items():
        arrays[name] = np.asarray(values, dtype=np.int64 if name in integer_names else np.float32)
    arrays.update(sequence=sampled_actions, safe_depths=np.asarray(all_safe_depths, dtype=np.int64),
                  final_core_loss=core_loss,
                  core_ids=unique_ids, core_counts=core_counts, final_loss=np.array(losses.mean()))
    info = {"trajectory": trajectory, "seed_index": seed_index, "seconds": time.perf_counter() - start,
            "graph_operations": graph.counters, "plant_operations": model.counters,
            "records": len(arrays["depth"]), "counterfactual_horizon": horizon,
            "terminal_samples": sorted(terminal_sites), "sensor_schema": bank.names,
            "sensor": bank.metadata(), "labels": "D_explore only", "input": "one full graph",
            "anchor_rule": ANCHOR_RULE, "anchor_observer_dependency": False,
            **site_action_summary(sampled_actions, model)}
    return arrays, info


def _graph_collection(data):
    return getattr(data, "graph_ptr", None) is not None and len(data.graph_ptr) > 2


def _operator(data, device):
    return GraphOperator(data.edge_index, data.num_nodes, device,
                         edge_weight=getattr(data, "edge_weight", None),
                         graph_ptr=getattr(data, "graph_ptr", None))


@torch.no_grad()
def nominal_reference(data, graph, config, output_dir, device):
    """Initial-input stopping targets using only development-role labels."""
    path = Path(output_dir) / "explore" / "nominal_reference.npz"
    if path.exists():
        return
    spec, L = task_spec(config), config["model"]["depth"]
    corpus = _graph_collection(data)
    graph_ids = list(data.graph_splits["explore"]) if corpus else [None]
    if corpus:
        # The same deterministic development subset used for logged episodes.
        graph_ids = graph_ids[:min(len(graph_ids), config["exploration"]["trajectories"])]
    results, initial, core_results, template_graphs, template_seeds = [], [], [], [], []
    for seed in config["training"]["seeds"]:
        model = load_plant(data, config, Path(output_dir) / "train", seed, device)
        for graph_id in graph_ids:
            episode = data.graph(int(graph_id)) if corpus else data
            op = _operator(episode, device) if corpus else graph
            model.bind_graph(op)
            ids_np = episode.splits["explore"]
            ids = torch.as_tensor(ids_np, device=device)
            label_np = role_labels(episode, "explore")
            cores = episode.cluster_ids[ids_np]
            with precision_context(config, device):
                h0 = model.encode(episode.x.to(device))
                h = h0
                logits0 = model.logits(h0)
                bank = sensor_bank(op, model, config)
                bank.prepare(h0, logits0)
                initial.append(bank.measure(h0, logits0, 0, L).cpu().numpy())
                losses, core_loss = [], []
                for d in range(L+1):
                    p = probabilities(model.logits(h)[ids].float(), spec).cpu().numpy()
                    ce = sample_losses(p, label_np, spec)
                    valid = np.isfinite(ce)
                    if not valid.any():
                        raise ValueError("Development graph has no scored labels")
                    means, unique_cores, counts = cluster_summary(ce[valid], cores[valid])
                    losses.append(float(ce[valid].mean()))
                    core_loss.append(means)
                    if d < L:
                        h = model.native_step(h, h0, op, site=d)
            results.append(losses)
            core_results.append(core_loss)
            template_graphs.append(-1 if graph_id is None else int(graph_id))
            template_seeds.append(int(seed))
            stop_at_boundary()
    arrays = dict(losses=np.asarray(results), initial_y=np.asarray(initial),
                  core_losses=np.asarray(core_results), core_ids=unique_cores, core_counts=counts)
    if corpus:
        arrays.update(graph_ids=np.asarray(template_graphs), seed_ids=np.asarray(template_seeds),
                      core_ids=np.asarray(template_graphs)[:, None], core_counts=np.ones((len(template_graphs),1), dtype=int))
    atomic_npz(path, **arrays)


def explore(data, graph, config, output_dir, device):
    directory = Path(output_dir) / "explore"
    directory.mkdir(parents=True, exist_ok=True)
    corpus = _graph_collection(data)
    graph_ids = list(data.graph_splits["explore"]) if corpus else [None]
    if not graph_ids:
        raise ValueError("No graphs in the exploration role")
    seeds = config["training"]["seeds"]
    model, current_seed = None, None
    for trajectory in range(config["exploration"]["trajectories"]):
        path = directory / f"trajectory_{trajectory:06d}.npz"
        meta = path.with_suffix(".json")
        if path.exists() and meta.exists():
            continue
        member = trajectory % len(seeds)
        if current_seed != seeds[member]:
            model = load_plant(data, config, Path(output_dir) / "train", seeds[member], device)
            current_seed = seeds[member]
        graph_id = graph_ids[trajectory % len(graph_ids)]
        episode = data.graph(int(graph_id)) if corpus else data
        op = _operator(episode, device) if corpus else graph
        ids_np = episode.splits["explore"]
        ids = torch.as_tensor(ids_np, device=device)
        labels = torch.as_tensor(role_labels(episode, "explore"), device=device)
        arrays, info = collect_trajectory(model, episode.x.to(device), op, labels, ids,
                    episode.cluster_ids[ids_np], config, trajectory, member, device)
        if corpus:
            arrays["graph_id"] = np.full(len(arrays["depth"]), int(graph_id), dtype=np.int64)
            info.update(graph_id=int(graph_id), input="one independent development graph")
        atomic_npz(path, **arrays)
        atomic_json(meta, info)
        print(f"[explore] trajectory {trajectory+1}/{config['exploration']['trajectories']} seed={current_seed} loss={float(arrays['final_loss']):.6f}", flush=True)
        stop_at_boundary()
    nominal_reference(data, graph, config, output_dir, device)
    atomic_json(directory / "manifest.json", {"trajectories": config["exploration"]["trajectories"],
                "full_graph_execution": True, "labels": "D_explore", "propensity_source": "Supplementary Table S6",
                "counting_unit": "complete graph across all seeds and trajectories" if corpus else "one checkpoint, one full randomized action trajectory",
                "independent_graph_episodes": corpus})


def load_records(directory):
    """Read only numeric own-format shards; never unpickle uploaded graph data."""
    files = sorted(Path(directory).glob("trajectory_*.npz"))
    if not files:
        raise FileNotFoundError("No complete exploration trajectories; run --stage explore first")
    names = ("y", "y_next", "action", "previous_action", "depth", "trajectory", "delta", "stage_delta",
             "terminal_delta", "propensity", "current_loss", "seed_index")
    with np.load(files[0], allow_pickle=False) as first:
        if "graph_id" in first.files:
            names = (*names, "graph_id")
    chunks = {k: [] for k in names}
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            for k in names:
                chunks[k].append(z[k])
    return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
