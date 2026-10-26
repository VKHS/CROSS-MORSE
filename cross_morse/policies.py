"""Registered policy definitions and strictly label-free execution.

This module imports no dataset label accessor. The exact same frozen head is
used at every depth. A plan-once policy never replans after initial measurement.
"""
from __future__ import annotations

from pathlib import Path
import copy
import time

import numpy as np
import torch

from .common import canonical_hash, jsonable, precision_context, synchronize
from .core import ACTIONS, site_transport_accounting, site_action_summary
from .exploration import sensor_bank
from .tasks import probabilities, task_spec
from .recovery import ANCHOR_RULE, label_free_safe
from .scoring import SCORE_VERSION


class PlanningBudgetExceeded(RuntimeError):
    pass


class PolicyExecutionError(FloatingPointError):
    """A numerical policy failure with serializable partial execution evidence.

    The caller decides whether to invalidate a development candidate or execute
    a previously locked comparator. This exception never chooses a comparator,
    changes a policy identity, or substitutes fabricated probabilities.
    """

    def __init__(self, reason, trace=None, message=None):
        super().__init__(message or reason)
        self.reason = str(reason)
        self.trace = jsonable(dict(trace or {}))
        self.trace.update(status="FAILED", feasible=False,
                          execution_failure=True, failure_reason=self.reason)
        self.diagnostics = self.trace


def _source_reset_sequence(sequence):
    """Canonical open-loop actions plus honest dynamic-reset provenance."""
    original = [int(a) for a in sequence]
    positions = [i for i, action in enumerate(original) if action == 9]
    normalized = [8 if action == 9 else action for action in original]
    provenance = {
        "rule": "dynamic A9 replaced by source-reset A8 before registration",
        "original_sequence": original,
        "original_sequence_sha256": canonical_hash(original),
        "a9_to_a8_depths": positions,
    }
    return normalized, provenance


def make_registry(config, exploration_directory):
    """Freeze all route/configuration identities before D_select is evaluated."""
    L = config["model"]["depth"]
    entries = []
    def add(route, complexity, is_open_loop, profile="none", **kwargs):
        item = dict(route=route, complexity=complexity, is_open_loop=is_open_loop, profile=profile, **kwargs)
        item["behavior_id"] = canonical_hash(item)[:16]
        entries.append(item)
    add("nominal", 0, True, depth=L)
    for depth in sorted(set([1, L] + [2**i for i in range(L.bit_length()) if 2**i < L])):
        if depth != L:
            add("best_depth", 1, True, depth=depth)
    add("learned_stop", 2, True, profile="full", depth=L)
    for boundary in config["registry"]["phase_boundaries"]:
        for a, b in config["registry"]["phase_pairs"]:
            switch = max(1, min(L-1, int(round(boundary * L))))
            sequence, reset_provenance = _source_reset_sequence([int(a)] * switch + [int(b)] * (L - switch))
            add("phase", 3, True, sequence=sequence,
                reset_normalization=reset_provenance)
    # Candidates are selected using exploration outcomes, before selection labels
    # exist in this code path. Nominal remains in the open-loop family.
    traces = []
    for path in sorted(Path(exploration_directory).glob("trajectory_*.npz")):
        with np.load(path, allow_pickle=False) as z:
            traces.append((float(z["final_loss"]), z["sequence"].astype(int).tolist(), path.name))
    seen = set()
    for source_loss, original_sequence, provenance in sorted(traces, key=lambda t: (t[0], t[2])):
        sequence, reset_provenance = _source_reset_sequence(original_sequence)
        if tuple(sequence) in seen:
            continue
        seen.add(tuple(sequence))
        add("fixed_sequence", 4, True, sequence=sequence, exploration_source=provenance,
            reset_normalization=reset_provenance,
            source_trajectory_loss=source_loss,
            proposal_ranking="original exploration loss; not an evaluation of the normalized source-reset schedule")
        if len(seen) >= config["registry"]["max_exploration_sequences"]:
            break
    for profile in config["sensor"]["profiles"]:
        for horizon in config["controller"]["horizons"]:
            add("mpc_ol", 5, True, profile=profile, horizon=min(horizon, L), depth=L,
                planning_actions=list(range(9)), recovery_reset=8)
        add("one_step", 6, False, profile=profile, depth=L)
        for kappa in config["controller"]["kappa_grid"]:
            add("radius", 7, False, profile=profile, kappa=kappa, depth=L)
        for horizon in config["controller"]["horizons"]:
            add("mpc_rh", 8, False, profile=profile, horizon=min(horizon, L), depth=L)
    candidates = []
    for item in entries:
        for ensemble in config["registry"]["ensemble_sizes"]:
            if ensemble > len(config["training"]["seeds"]):
                raise ValueError("Ensemble size exceeds the declared checkpoint set")
            candidate = dict(item, ensemble=ensemble)
            candidate["hash"] = canonical_hash(candidate)
            candidate["candidate_id"] = candidate["hash"][:20]
            candidates.append(candidate)
    return candidates


class DonorBank:
    """Offline observation donors, matched without labels.

    Matching uses exact depth, previous action and initial entropy bin. These
    are explicit coarse reconstruction strata. No match => no valid state
    destruction contrast, and confirmation must downgrade rather than assert
    that an unmatched shuffle establishes feedback value.
    """
    def __init__(self, records, entropy_edges=(1/3, 2/3)):
        self.donors = {}
        trajectory = np.asarray(records["trajectory"])
        seeds = np.asarray(records.get("seed_index", np.zeros(len(trajectory), dtype=int)))
        _, first_indices, ids = np.unique(np.column_stack((seeds, trajectory)), axis=0,
                                          return_index=True, return_inverse=True)
        unique_ids = np.arange(len(first_indices))
        first = dict(zip(unique_ids, first_indices))
        initial_bin = {t: int(np.searchsorted(entropy_edges, records["y"][i, 0])) for t, i in first.items()}
        for i, t in enumerate(ids):
            key = (int(records["depth"][i]), int(records["previous_action"][i]), initial_bin[t])
            self.donors.setdefault(key, []).append(np.asarray(records["y"][i], dtype=np.float32))
        self.edges = entropy_edges

    def draw(self, depth, previous_action, initial_y, rng):
        key = (depth, previous_action, int(np.searchsorted(self.edges, float(initial_y[0]))))
        rows = self.donors.get(key, [])
        return None if not rows else rows[int(rng.integers(len(rows)))].copy()


def recovery_action(supported, last_safe_available=True):
    if 0 in supported:
        return 0
    return 9 if last_safe_available else 8


def beam_plan(controller, state, horizon, config, budget, *, open_loop=False):
    """Eq17 search on learned latent transitions, with deterministic beam ties.

    Separate stage_delta and nominal-suffix terminal_delta models avoid
    treating a one-action critic as an untrained multi-step consequence model.
    A plan of H sites is followed by nominal completion for MPC-OL.

    A9 targets describe a dynamic last-safe anchor. Open-loop planning excludes
    A9 from every critic/transition/stage query, including simulated recovery;
    its modeled reset is A8, whose H0 semantics match actual execution.
    """
    c = config["controller"]
    def charge(n=1):
        if budget["calls"] + n > c["max_search_calls"]:
            raise PlanningBudgetExceeded("registered model-call budget expired")
        budget["calls"] += n
    def finite_cost(value, component):
        if not np.isfinite(value):
            raise FloatingPointError(f"Nonfinite {component} in learned policy planning")
        return float(value)
    beams = [(0., [], copy.deepcopy(state))]
    for offset in range(horizon):
        expanded = []
        for cost, sequence, simulated in beams:
            supported = controller.supported(simulated.y, simulated.depth)
            if open_loop:
                supported = [a for a in supported if a != 9]
            if not supported:
                a = recovery_action(supported, last_safe_available=not open_loop)
                charge(2)
                nxt = controller.predict(simulated, a)
                value = finite_cost(controller.stage_cost(simulated, a), "recovery stage cost")
                expanded.append((cost + c["gamma"]**offset * value, sequence + [a], nxt))
                continue
            charge()
            scores = controller.score(simulated, supported, kind="radius", kappa=1., lambda_rho=c["lambda_rho"])
            finite_actions = [a for a in supported if np.isfinite(scores[a]["score"])]
            if not finite_actions:
                a = recovery_action(supported, last_safe_available=not open_loop)
                charge(2)
                nxt = controller.predict(simulated, a)
                value = finite_cost(controller.stage_cost(simulated, a), "recovery stage cost")
                expanded.append((cost + c["gamma"]**offset * value, sequence + [a], nxt))
                continue
            ordered = sorted(finite_actions, key=lambda a: (scores[a]["score"], a))[:c["top_k"]]
            for a in ordered:
                charge(2)
                stage = controller.stage_cost(simulated, a)
                # Model uncertainty and operational transport costs are charged
                # explicitly. These coefficients are declared reconstructions.
                transport = site_transport_accounting(a, simulated.depth, config)
                stage += scores[a]["sigma"] + c["lambda_rho"] * ACTIONS[a].rho + c["lambda_transport"] * transport["hops"]
                stage = finite_cost(stage, "stage cost")
                nxt = controller.predict(simulated, a)
                expanded.append((cost + c["gamma"]**offset * stage, sequence + [a], nxt))
        if not expanded:
            return []
        ranked = []
        for cost, sequence, simulated in expanded:
            # Controller.terminal_value returns the exact zero boundary value
            # without a neural forward call at the nominal terminal depth.
            if simulated.depth < config["model"]["depth"]:
                charge()
            tail = finite_cost(controller.terminal_value(simulated), "terminal value")
            ranked.append((cost + c["gamma"]**len(sequence) * c["lambda_terminal"] * tail, cost, sequence, simulated))
        ranked.sort(key=lambda item: (item[0], tuple(item[2])))
        beams = [(cost, sequence, simulated) for _, cost, sequence, simulated in ranked[:c["beam_width"]]]
    return beams[0][1]


@torch.no_grad()
def execute_policy(model, x, graph, config, candidate, controller=None,
                   nominal_reference=None, donors=None, replay_actions=None,
                   replay_safe_depths=None):
    """Execute label-free, returning probabilities/trace or typed failure.

    ``PolicyExecutionError.trace`` contains partial actions, reset anchors and
    work counters. A caller with a locked comparator can use that evidence to
    perform its registered fallback. Configuration/programming errors, resource
    exhaustion and device failures are not swallowed as numerical policy exits.
    """
    diagnostics = {
        "route": candidate.get("route"), "profile": candidate.get("profile"),
        "candidate_id": candidate.get("candidate_id"),
        "candidate_hash": candidate.get("hash"),
        "behavior_id": candidate.get("behavior_id"),
        "label_free": True, "failure_stage": "initialization",
        "failed_depth": 0, "failed_action": None,
    }
    started = time.perf_counter()
    try:
        return _execute_policy(model, x, graph, config, candidate, controller,
                               nominal_reference, donors, replay_actions,
                               replay_safe_depths, diagnostics)
    except PolicyExecutionError:
        raise
    except (FloatingPointError, np.linalg.LinAlgError,
            getattr(torch.linalg, "LinAlgError", np.linalg.LinAlgError)) as exc:
        graph_counts = dict(getattr(graph, "counters", {}))
        plant_counts = dict(getattr(model, "counters", {}))
        controller_counts = dict(getattr(controller, "counters", {}))
        sensor_counts = dict(diagnostics.get("sensor_operations", {}))
        partial_work = (int(getattr(graph, "sparse_multiply_adds", 0))
                        + int(plant_counts.get("dense_multiply_adds", 0))
                        + int(controller_counts.get("approx_dense_multiply_adds", 0))
                        + int(sensor_counts.get("dense_multiply_adds", 0)))
        diagnostics.update(
            exception_type=type(exc).__name__, exception_message=str(exc),
            graph_operations=graph_counts, plant_operations=plant_counts,
            controller_operations=controller_counts, sensor_operations=sensor_counts,
            multiply_add_work=partial_work, partial_execution=True,
            work_scope="partial executed product counters; excludes unfinished work and elementwise operations",
            wall_seconds=time.perf_counter() - started,
            timing_note="partial host elapsed time; a numerical failure does not add a CUDA synchronization",
            **site_action_summary(diagnostics.get("actions", []),
                                  model if hasattr(model, "is_native_site") else config),
        )
        raise PolicyExecutionError("numerical_execution_failure", diagnostics,
                                   "Numerical failure during frozen policy execution") from exc


def _execute_policy(model, x, graph, config, candidate, controller,
                    nominal_reference, donors, replay_actions,
                    replay_safe_depths, diagnostics):
    """Execute without labels. Returns probabilities and an auditable trace.

    A0 nominal replay uses the adapter; `training.audit_training` independently
    verifies it against the native aggregation path. Support auditing of fixed
    nonnominal schedules is qualification work, reported separately below.

    Exact replay of A9 requires ``replay_safe_depths=trace['safe_depths']`` as
    well as the action trace. The anchors are replayed without measurements or
    observer calls. An action-only trace cannot identify a last-safe reset.
    """
    device = x.device
    if hasattr(model, "bind_graph"):
        model.bind_graph(graph)
    L = config["model"]["depth"]
    route = candidate["route"]
    max_depth = int(candidate.get("depth", L))
    if replay_actions is not None:
        replay_actions = [int(a) for a in replay_actions]
        if not replay_actions or len(replay_actions) > L:
            raise ValueError("Replay must contain between one and L actions")
        max_depth = len(replay_actions)
        if any(a == 9 for a in replay_actions) and replay_safe_depths is None:
            raise ValueError("An A9 replay requires the recorded safe_depths trace")
        if replay_safe_depths is not None and len(replay_safe_depths) != max_depth:
            raise ValueError("Replay action and safe-depth traces must have the same length")
    graph.reset_counters()
    model.reset_counters()
    if controller is not None and hasattr(controller, "reset_counters"):
        controller.reset_counters()
    state, bank, initial_y = None, None, None
    previous, mass, hops = -1, 0., 0.
    actions, origins, safe_depths, support_violations, observer_failures = [], [], [], 0, 0
    decisions = []
    diagnostics.update(actions=actions, origins=origins, safe_depths=safe_depths,
                       decisions=decisions, anchor_rule=ANCHOR_RULE)
    shuffle_matched, shuffle_missing, plan_budget = 0, 0, {"calls": 0}
    shuffle_support_mismatch = 0
    plan, plan_exhausted = [], False
    schedule = candidate.get("sequence")
    is_feedback = route in ("one_step", "radius", "mpc_rh")
    if replay_actions is None and not is_feedback and schedule is not None and 9 in schedule:
        raise ValueError("Open-loop schedules must register source-reset A8 instead of dynamic-anchor A9")
    needs_sensor = replay_actions is None and (is_feedback or route in ("mpc_ol", "learned_stop"))
    # Fixed schedules are pre-qualified using a sensor pass. Decisions remain
    # fixed; qualification measurements never modify the declared sequence.
    audit_schedule = route in ("phase", "fixed_sequence", "mpc_ol") and replay_actions is None
    rng = np.random.default_rng(config["seed"] + int(candidate.get("behavior_id", "0")[:8], 16))
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    sensor_seconds, policy_seconds, plant_seconds = 0., 0., 0.
    audit_sparse_macs, audit_dense_macs, audit_seconds = 0, 0, 0.
    with precision_context(config, device):
        diagnostics["failure_stage"] = "encoder"
        h0 = model.encode(x)
        h, last_safe = h0, h0
        last_safe_depth = 0
        if needs_sensor or audit_schedule:
            diagnostics["failure_stage"] = "initial_measurement"
            begin = time.perf_counter()
            before_spmm = graph.sparse_multiply_adds
            before_plant_dense = model.counters["dense_multiply_adds"]
            bank = sensor_bank(graph, model, config, candidate["profile"] if candidate["profile"] != "none" else "full")
            diagnostics["sensor_operations"] = bank.counters
            source_logits = model.logits(h0)
            bank.prepare(h0, source_logits)
            initial_y = bank.measure(h0, source_logits, 0, L).cpu().numpy()
            elapsed = time.perf_counter() - begin
            sensor_seconds += elapsed
            # MPC-OL needs this initial measurement for its one-time plan.
            # Fixed sequences need it only for qualification; their A8 source
            # reset never uses a measurement-selected anchor.
            if audit_schedule and route != "mpc_ol":
                audit_sparse_macs += graph.sparse_multiply_adds - before_spmm
                audit_dense_macs += (model.counters["dense_multiply_adds"] - before_plant_dense
                                     + int(bank.counters.get("dense_multiply_adds", 0)))
                audit_seconds += elapsed
            if controller is not None and (is_feedback or route == "mpc_ol"):
                diagnostics["failure_stage"] = "observer_initialization"
                begin = time.perf_counter()
                state = controller.start(initial_y)
                policy_seconds += time.perf_counter() - begin
        if route == "learned_stop" and replay_actions is None:
            if nominal_reference is None:
                raise ValueError("learned_stop requires D_explore initial-input targets")
            templates = nominal_reference["initial_y"]
            scale = np.maximum(np.std(templates, axis=0), .05)
            distances = np.mean(((templates - initial_y[None, :]) / scale)**2, axis=1)
            neighbor = int(np.argmin(distances))
            # Depth zero is excluded from the shared-plant stopping family.
            max_depth = 1 + int(np.argmin(nominal_reference["losses"][neighbor, 1:]))
        if route == "mpc_ol" and replay_actions is None:
            diagnostics["failure_stage"] = "initial_plan"
            begin = time.perf_counter()
            try:
                plan = beam_plan(controller, state, min(candidate["horizon"], L), config, plan_budget,
                                 open_loop=True)
            except PlanningBudgetExceeded:
                plan_exhausted = True
                plan = []
            policy_seconds += time.perf_counter() - begin
            # Explicitly frozen completion rule: nominal after the initial H.
            schedule = plan + [0] * (L - len(plan))
            if 9 in schedule:
                raise ValueError("MPC-OL plan contains dynamic-anchor A9; only A8 source reset is permitted")
        for depth in range(max_depth):
            diagnostics.update(failed_depth=depth, failed_action=None,
                               failure_stage="policy_decision",
                               model_calls_charged=plan_budget["calls"])
            supported = list(range(10))
            decision = {"depth": depth, "score_contract": SCORE_VERSION,
                        "anchor_updated": False, "scores": None,
                        "coverage_status": "not assessed by a label-free execution trace"}
            y = initial_y
            if replay_actions is not None:
                if replay_safe_depths is not None:
                    requested_safe_depth = int(replay_safe_depths[depth])
                    if not 0 <= requested_safe_depth <= depth:
                        raise ValueError("Replay anchor must be a previously reached state")
                    if requested_safe_depth == depth:
                        last_safe, last_safe_depth = h.clone(), depth
                    elif requested_safe_depth != last_safe_depth:
                        raise ValueError("Replay skipped or reversed a recorded last-safe update")
                a, origin = int(replay_actions[depth]), "replay"
            else:
                if is_feedback or audit_schedule:
                    begin = time.perf_counter()
                    before_spmm = graph.sparse_multiply_adds
                    before_plant_dense = model.counters["dense_multiply_adds"]
                    before_sensor_dense = int(bank.counters.get("dense_multiply_adds", 0))
                    if depth == 0:
                        y = initial_y
                    else:
                        y = bank.measure(h, model.logits(h), depth, L, mass, hops).cpu().numpy()
                    elapsed = time.perf_counter() - begin
                    sensor_seconds += elapsed
                    if audit_schedule:
                        audit_sparse_macs += graph.sparse_multiply_adds - before_spmm
                        audit_dense_macs += (model.counters["dense_multiply_adds"] - before_plant_dense
                                             + int(bank.counters.get("dense_multiply_adds", 0)) - before_sensor_dense)
                        audit_seconds += elapsed
                    supported = controller.supported(y, depth) if controller is not None else []
                    if is_feedback:
                        used_y = y
                        if candidate.get("state_control") == "matched_shuffle" and depth > 0:
                            donor = donors.draw(depth, previous, initial_y, rng) if donors is not None else None
                            if donor is None:
                                shuffle_missing += 1
                            else:
                                used_y = donor
                                shuffle_matched += 1
                                donor_supported = controller.supported(used_y, depth)
                                if set(donor_supported) != set(supported):
                                    # This intervention crossed a registered
                                    # support boundary: record it before using
                                    # the intersection for executable safety.
                                    shuffle_support_mismatch += 1
                                    support_violations += 1
                                supported = sorted(set(supported) & set(donor_supported))
                        begin = time.perf_counter()
                        if depth > 0:
                            state = controller.observe(state, used_y, depth, previous)
                        policy_seconds += time.perf_counter() - begin
                    # Only feedback routes use a measurement-selected A9
                    # anchor. Open-loop schedules register A8 source resets.
                    if is_feedback and label_free_safe(y):
                        last_safe, last_safe_depth = h.clone(), depth
                        decision["anchor_updated"] = True
                    if is_feedback:
                        decision.update(observer_valid=bool(state.valid),
                            anchor_would_fail_legacy_observer_check=bool(label_free_safe(y) and not state.valid),
                            observer_nis=getattr(state, "nis", None),
                            observer_bias=getattr(state, "bias", None),
                            observer_whiteness=getattr(state, "whiteness", None))
                if is_feedback:
                    begin = time.perf_counter()
                    if not supported:
                        a, origin = recovery_action(supported), "recovery_no_support"
                    elif not state.valid:
                        a, origin = recovery_action(supported), "recovery_observer"
                        observer_failures += 1
                    elif plan_exhausted:
                        a, origin = recovery_action(supported), "recovery_budget"
                    else:
                        try:
                            if route == "mpc_rh":
                                plan = beam_plan(controller, state, min(candidate["horizon"], L-depth), config, plan_budget)
                                a = plan[0] if plan else recovery_action(supported)
                                decision["planned_sequence"] = list(plan)
                            else:
                                if plan_budget["calls"] + 1 > config["controller"]["max_search_calls"]:
                                    raise PlanningBudgetExceeded()
                                plan_budget["calls"] += 1
                                scores = controller.score(state, supported, kind=route,
                                            kappa=candidate.get("kappa", 1.), lambda_rho=config["controller"]["lambda_rho"])
                                decision["scores"] = scores
                                finite_actions = [a for a in supported if np.isfinite(scores[a]["score"])]
                                a = (min(finite_actions, key=lambda a: (scores[a]["score"], a))
                                     if finite_actions else recovery_action(supported))
                            origin = "policy"
                            if route != "mpc_rh" and not finite_actions:
                                origin = "recovery_no_finite_score"
                            if a not in supported and a not in (8, 9):
                                support_violations += 1
                                a, origin = recovery_action(supported), "recovery_support_mismatch"
                        except PlanningBudgetExceeded:
                            plan_exhausted = True
                            a, origin = recovery_action(supported), "recovery_budget"
                    policy_seconds += time.perf_counter() - begin
                elif schedule is not None:
                    a, origin = int(schedule[depth]), "fixed_schedule"
                    if audit_schedule and a not in supported and a not in (8, 9):
                        support_violations += 1
                else:
                    a, origin = 0, "nominal"
            accounting = site_transport_accounting(a, depth, model if hasattr(model, "is_native_site") else config)
            decision.update(action=int(a), origin=origin, supported_actions=list(supported),
                            anchor_depth=int(last_safe_depth), model_calls=plan_budget["calls"],
                            selection_mode="replay" if replay_actions is not None else route,
                            site_kind=accounting["site_kind"],
                            transport_mass_increment=accounting["mass"],
                            transport_hops_increment=accounting["hops"],
                            inserted_a0_bypass=accounting["inserted_a0_bypass"])
            if decision["scores"] is not None:
                finite_scores = sorted(v["score"] for v in decision["scores"].values() if np.isfinite(v["score"]))
                decision["score_gap"] = (float(finite_scores[1] - finite_scores[0])
                                          if len(finite_scores) > 1 else None)
                decision["selected_score"] = decision["scores"].get(int(a))
                if not config.get("recording", {}).get("all_action_scores", False):
                    decision.pop("scores")
            if is_feedback and replay_actions is None and hasattr(controller, "stratum"):
                decision["support_stratum"] = controller.stratum(state.y, state.depth)
            decisions.append(jsonable(decision))
            begin = time.perf_counter()
            diagnostics.update(failure_stage="plant_step", failed_action=int(a),
                               model_calls_charged=plan_budget["calls"])
            h = model.step(h, h0, graph, a, last_safe, site=depth)
            plant_seconds += time.perf_counter() - begin
            actions.append(a)
            origins.append(origin)
            safe_depths.append(last_safe_depth)
            mass += accounting["mass"]
            hops += accounting["hops"]
            previous = a
        diagnostics.update(failure_stage="final_prediction", failed_depth=max_depth,
                           failed_action=None, model_calls_charged=plan_budget["calls"])
        probs = probabilities(model.logits(h).float(), getattr(model, "task", task_spec(config))).cpu().numpy()
    synchronize(device)
    elapsed = time.perf_counter() - started
    control_counts = dict(getattr(controller, "counters", {})) if controller is not None else {}
    sensor_counts = dict(bank.counters) if bank is not None else {}
    raw_macs = (graph.sparse_multiply_adds + model.counters["dense_multiply_adds"]
                + int(control_counts.get("approx_dense_multiply_adds", 0))
                + int(sensor_counts.get("dense_multiply_adds", 0)))
    audit_macs = audit_sparse_macs + audit_dense_macs
    if audit_macs > raw_macs:
        raise AssertionError("Qualification products cannot exceed total executed products")
    trace = {"route": route, "profile": candidate["profile"], "actions": actions, "origins": origins,
             **site_action_summary(actions, model if hasattr(model, "is_native_site") else config),
             "safe_depths": safe_depths, "decisions": decisions,
             "anchor_rule": ANCHOR_RULE, "anchor_observer_dependency": False,
             "terminal_depth": max_depth, "support_violations": support_violations,
             "observer_failures": observer_failures, "planning_budget_exhausted": plan_exhausted,
             "model_calls_charged": plan_budget["calls"], "graph_operations": dict(graph.counters),
             "plant_operations": dict(model.counters), "controller_operations": control_counts,
             "sensor_operations": sensor_counts,
             "multiply_add_work": raw_macs,
             "deployed_multiply_add_work": raw_macs - audit_macs,
             "qualification_multiply_add_work": audit_macs,
             "qualification_sensor_sparse_macs": audit_sparse_macs,
             "qualification_sensor_dense_macs": audit_dense_macs,
             "work_scope": "executed sparse and dense products; elementwise/reduction costs excluded; use wall time too",
             "deployment_work_scope": ("total products minus support-qualification-only sensing/head products; "
                 "MPC-OL initial planning measurements retained; static graph qualification; estimated operation ledger, not a hard resource guarantee"),
             "wall_seconds": elapsed, "sensor_host_seconds": sensor_seconds,
             "policy_host_seconds": policy_seconds, "plant_host_seconds": plant_seconds,
             "timing_note": "total CUDA-synchronized; component host dispatch times are not isolated CUDA-kernel durations",
             "qualification_sensor_host_seconds": audit_seconds,
             "runtime_host_seconds_excluding_qualification_estimate": max(0., elapsed - audit_seconds),
             "runtime_estimate_warning": "host subtraction is not an isolated CUDA runtime measurement; total wall time is authoritative for this executed pass",
             "shuffle_matched_sites": shuffle_matched, "shuffle_missing_sites": shuffle_missing,
             "shuffle_support_mismatch_sites": shuffle_support_mismatch,
             "state_control_valid": (shuffle_matched > 0 and shuffle_missing == 0 and shuffle_support_mismatch == 0
                                     if candidate.get("state_control") == "matched_shuffle" else None),
             "label_free": True,
             "state_assignment": ("explicitly recorded reset anchors from replay" if replay_actions is not None
                                  else "source H0 or recorded label-free-safe state" if is_feedback
                                  else "source-reset A8; dynamic-anchor A9 excluded from open-loop execution"),
             "planning_action_contract": ("action/anchor replay; no learned consequence queries" if replay_actions is not None
                                          else "A0--A8, including A8 simulated recovery" if route == "mpc_ol"
                                          else "registered supported actions and legal recovery"),
             "feasible": support_violations == 0,
             "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    if not np.isfinite(probs).all():
        trace.update(candidate_id=candidate.get("candidate_id"), candidate_hash=candidate.get("hash"),
                     behavior_id=candidate.get("behavior_id"), failure_stage="final_prediction",
                     failed_depth=max_depth, failed_action=None,
                     nonfinite_probability_count=int((~np.isfinite(probs)).sum()))
        raise PolicyExecutionError("nonfinite_predictions", trace,
                                   "A frozen policy produced nonfinite predictions")
    return probs, trace
