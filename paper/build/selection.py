"""The selection rule of main Eq. (19) under the three equivalence readings of Supplementary Table S12c.

Main Eq. (19) selects, among feedback configurations whose simultaneous upper limit against the
nested comparator is below −ε_R, the simplest configuration within ε_eq of the best one.
Supplementary Table S12c audits that rule on the representative row of each route family under
three readings of the equivalence margin:

    registered     ε_eq = 0.002 applied to the upper limits U_s           (the rule the release deploys)
    point-0.002    ε_eq = 0.002 applied to the point estimates Δ̂_s
    point-0.001    ε_eq = 0.001 applied to the point estimates Δ̂_s

Everything here is read from the sealed selection record of a run (``select/decision.json`` and
``select/candidates.json``, carried into the evidence record); nothing is re-estimated.  The same
code also applies the registered rule to the complete configuration set and checks that it
reproduces the run's own provisional selection, which ties the table to the deployed decision.
"""
from __future__ import annotations

from .evidence import FEEDBACK_ROUTES, ROUTE_LABELS

READINGS = (
    ("registered", "select_upper", None, "ε_eq = 0.002 on U_s (registered)"),
    ("point-0.002", "select_delta", 0.002, "ε_eq = 0.002 on Δ̂_s"),
    ("point-0.001", "select_delta", 0.001, "ε_eq = 0.001 on Δ̂_s"),
)


def simplicity_key(candidate: dict) -> tuple:
    """The registered lexicographic order of ``cross_morse.statistics._candidate_key``."""
    complexity = candidate.get("complexity", 0)
    complexity = tuple(complexity) if isinstance(complexity, (list, tuple)) else (float(complexity or 0),)
    return (complexity, int(candidate.get("ensemble") or 1), float(candidate.get("cost") or 0.0),
            str(candidate.get("hash") or f"index-{candidate['index']:08d}"), candidate["index"])


def _feedback_feasible(candidate: dict) -> bool:
    if candidate.get("route") not in FEEDBACK_ROUTES:
        return False
    if candidate.get("joint_feasible") is None:          # older records: the ledger's feasibility flag
        return bool(candidate.get("feasible"))
    return bool(candidate["joint_feasible"])


def _eligible(rows: list[dict], epsilon_r: float) -> list[dict]:
    return [c for c in rows if _feedback_feasible(c) and c.get("select_upper") is not None
            and c["select_upper"] < -epsilon_r]


def representative_rows(record: dict) -> list[dict]:
    """One configuration per feedback route family: the one with the smallest U_s (Table S12)."""
    rows = []
    for route in ROUTE_LABELS:
        if route not in FEEDBACK_ROUTES:
            continue
        members = [c for c in record["candidates"] if c["route"] == route and c.get("select_upper") is not None]
        if members:
            rows.append(min(members, key=lambda c: (c["select_upper"], simplicity_key(c))))
    return rows


def apply_reading(rows: list[dict], statistic: str, margin: float, epsilon_r: float) -> dict:
    """Near-best set and lexicographic choice of one reading; ``choice`` None returns the comparator."""
    eligible = _eligible(rows, epsilon_r)
    if not eligible or any(c.get(statistic) is None for c in eligible):
        return {"eligible": [c["index"] for c in eligible], "reference": None, "near_best": [], "choice": None,
                "available": bool(not eligible or all(c.get(statistic) is not None for c in eligible))}
    reference = min(c[statistic] for c in eligible)
    near = [c for c in eligible if c[statistic] <= reference + margin + 1e-15]
    choice = min(near, key=simplicity_key)
    return {"eligible": [c["index"] for c in eligible], "reference": reference,
            "near_best": [c["index"] for c in sorted(near, key=simplicity_key)], "choice": choice["index"],
            "available": True}


def registered_rule_on_all_configurations(record: dict) -> int:
    """Main Eq. (19) exactly as ``cross_morse.statistics.nested_selection`` applies it."""
    sel = record["selection"]
    epsilon_r, epsilon_eq = sel["epsilon_r"], sel["epsilon_eq"]
    candidates = record["candidates"]
    feasible = [c for c in candidates if _feedback_feasible(c) and c.get("select_upper") is not None]
    minimum_upper = min([0.0] + [c["select_upper"] for c in feasible])
    eligible = [c for c in feasible if c["select_upper"] < -epsilon_r and c["select_upper"] <= minimum_upper + epsilon_eq]
    if not eligible:
        return sel["qC_index"]
    return min(eligible, key=simplicity_key)["index"]


def selection_readings(record: dict) -> dict:
    """All three readings on the representative rows, plus the registered rule on every configuration."""
    sel = record["selection"]
    epsilon_r = sel.get("epsilon_r")
    if epsilon_r is None or not record.get("candidates"):
        return {"available": False}
    rows = representative_rows(record)
    by_index = {c["index"]: c for c in record["candidates"]}
    out = {"available": True, "epsilon_r": epsilon_r, "readings": {}}
    for name, statistic, margin, label in READINGS:
        margin = sel.get("epsilon_eq") if margin is None else margin
        result = apply_reading(rows, statistic, margin, epsilon_r)
        result["label"] = label
        result["margin"] = margin
        result["choice_route"] = by_index[result["choice"]]["route"] if result["choice"] is not None else None
        result["near_best_routes"] = [by_index[i]["route"] for i in result["near_best"]]
        out["readings"][name] = result
    registered = out["readings"]["registered"]
    out["u_min"] = registered["reference"]
    provisional = sel.get("provisional_index")
    out["selected_index"] = provisional
    selected = by_index.get(provisional) or {}
    out["selected_route"] = None if provisional == sel.get("qC_index") else selected.get("route")
    out["registered_consistent"] = registered["choice_route"] == out["selected_route"]
    all_configurations = registered_rule_on_all_configurations(record)
    out["registered_all_configurations_index"] = all_configurations
    out["registered_all_configurations_reproduces_selection"] = all_configurations == provisional
    return out
