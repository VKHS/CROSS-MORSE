#!/usr/bin/env python3
"""Compare a completed run with the values the manuscript reports for the same dataset.

NumPy-only verification; no PyTorch or checkpoint loading. The script reads the sealed decision records of one
completed run (``config.json``, ``select/``, ``confirm/``, ``admit/``,
``policy_lock.json`` and ``final/results_summary.json``) and places every
quantity next to the corresponding entry of ``paper/reported_results.json``.

Three agreement levels are reported for every numerical quantity:

  exact      the run value equals the printed value at the manuscript's printed precision
             (the level a binding to the result-producing run must reach);
  interval   the run value lies inside the manuscript's 95% interval for that effect, or
             within the reported across-seed standard deviation for an official score
             (descriptive numerical compatibility, not a replication test or a prediction interval);
  outside    neither.

With ``--emit-binding PATH`` the script also writes a completed result-binding file
for ``scripts/check_result_binding.py`` (the formal, hash-bound declaration); the
author states the relationship between the run and the manuscript explicitly.

This comparison establishes consistency at reporting precision. It does not by
itself establish that the run is the manuscript's result-producing run; that is
what the binding, the sealed identities and the archived run records are for.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._paper_common import MULTI_GRAPH, ROUTE_NAMES, load_reported, printed_half_unit, read_json

STAGES = ("prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate")


class RunReader:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        if not (self.root / "config.json").is_file():
            raise FileNotFoundError(f"not a run directory (config.json missing): {self.root}")
        self.config = read_json(self.root / "config.json")
        self.dataset = self.config.get("benchmark_dataset", self.config["dataset"])
        self.evidence = None

    def verify(self) -> None:
        """Verify the complete parent chain and every JSON record used below."""
        from cross_morse.review import NumericEvidence
        evidence = NumericEvidence(self.root)
        if evidence.json("config.json") != self.config:
            raise ValueError("Configuration changed while opening the run")
        if self.config.get("scientific_run") is not True or self.config.get("dataset") == "synthetic":
            raise ValueError("Synthetic fixtures are software checks, not manuscript benchmark comparisons")
        for relative in ("select/candidates.json", "select/decision.json", "confirm/decision.json",
                         "admit/decision.json", "policy_lock.json", "final/results_summary.json"):
            evidence.json(relative)
        candidates = evidence.json("select/candidates.json")
        if not isinstance(candidates, list) or not candidates or any(
            not isinstance(c, dict) or not isinstance(c.get("route"), str)
            or type(c.get("ensemble")) is not int or c["ensemble"] < 1 for c in candidates
        ):
            raise ValueError("A nonempty candidate ledger with route and ensemble identities is required")
        records = {}
        for relative, keys in (
            ("select/decision.json", ("comparator_index", "provisional_index")),
            ("confirm/decision.json", ("comparator_index", "confirmed_index")),
            ("admit/decision.json", ("comparator_index", "confirmed_index", "deployed_index")),
            ("policy_lock.json", ("comparator_index", "deployed_index")),
        ):
            value = evidence.json(relative)
            if not isinstance(value, dict) or not value:
                raise ValueError(f"Nonempty decision record required: {relative}")
            for key in keys:
                if type(value.get(key)) is not int or not 0 <= value[key] < len(candidates):
                    raise ValueError(f"Missing or invalid candidate index: {relative}:{key}")
            records[relative] = value
        if type(records["admit/decision.json"].get("passed")) is not bool:
            raise ValueError("Admission must record a boolean passed decision")
        if not isinstance(records["confirm/decision.json"].get("contrasts"), list):
            raise ValueError("Confirmation must record its contrast list (empty is allowed for a skipped gate)")
        if len({record["comparator_index"] for record in records.values()}) != 1:
            raise ValueError("Comparator identity changes across the locked decisions")
        if records["admit/decision.json"]["deployed_index"] != records["policy_lock.json"]["deployed_index"]:
            raise ValueError("Admission and deployment lock disagree")
        summary = evidence.json("final/results_summary.json")
        if (not isinstance(summary, dict) or not summary or type(summary.get("admitted")) is not bool
                or summary.get("admitted") != records["admit/decision.json"]["passed"]
                or summary.get("dataset") != self.config["dataset"]):
            raise ValueError("A complete final summary consistent with admission is required")
        for key in ("paired_qD_minus_qC_loss", "paired_qD_minus_qC_metric_pp"):
            value = summary.get(key)
            if not isinstance(value, dict) or any(
                isinstance(value.get(k), bool) or not isinstance(value.get(k), (int, float))
                or not math.isfinite(value[k]) for k in ("mean", "lower", "upper")
            ):
                raise ValueError(f"A finite mean and interval are required: {key}")
        models = summary.get("models")
        if not isinstance(models, dict):
            raise ValueError("The final summary must record a models object")
        for name in ("qN", "qC", "qD"):
            model = models.get(name)
            if not isinstance(model, dict):
                raise ValueError(f"A model record is required: {name}")
            value = model.get("metric")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"A finite official metric is required: {name}")
        self.evidence = evidence

    def optional(self, relative: str):
        path = self.root / relative
        if not path.is_file():
            return None
        return self.evidence.json(relative) if self.evidence is not None else read_json(path)

    def identities(self) -> dict | None:
        if self.evidence is not None:
            return dict(self.evidence.identities)
        manifests = sorted((self.root / "provenance" / "stages").glob("*.json")) if (self.root / "provenance" / "stages").is_dir() else []
        for path in manifests:
            value = read_json(path)
            if all(k in value for k in ("source_sha256", "config_sha256", "unit_sha256", "dataset")):
                return {k: value[k] for k in ("source_sha256", "config_sha256", "unit_sha256", "dataset")}
        state = self.optional("run_state.json")
        if state:
            return {"source_sha256": state.get("source_sha256"), "config_sha256": state.get("config_sha256"), "unit_sha256": None, "dataset": state.get("dataset")}
        return None


def level(actual, reported, half_unit, interval=None, spread=None) -> str:
    if reported is None:
        return "not reported"
    if actual is None:
        return "missing"
    values = [actual, reported, half_unit, *(interval or []), *([] if spread is None else [spread])]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError("Comparison values and tolerances must be finite numbers")
    if half_unit < 0 or (spread is not None and spread < 0):
        raise ValueError("Comparison tolerances must be nonnegative")
    if abs(float(actual) - float(reported)) <= half_unit + 1e-12:
        return "exact"
    if interval is not None and min(interval) - 1e-12 <= float(actual) <= max(interval) + 1e-12:
        return "interval"
    if spread is not None and abs(float(actual) - float(reported)) <= spread + 1e-12:
        return "interval"
    return "outside"


def effect_half_unit(point, interval=None) -> float:
    """Retain trailing-zero precision using the complete printed effect row."""
    values = [v for v in [point, *(interval or [])] if v is not None and v != 0]
    return min((printed_half_unit(v) for v in values), default=0.0)


def comparator_status(candidate: dict, identity: str, ensemble: int) -> tuple[str, str]:
    """Check settings actually named by the paper without inventing a full identity."""
    if candidate.get("route") not in expected_routes(identity) or candidate.get("ensemble") != ensemble:
        return "differs", "Route or ensemble differs from the reported setting."
    match = re.search(r"\bH\s*=\s*(\d+)", identity)
    if match and candidate.get("horizon") != int(match.group(1)):
        return "differs", "The planning horizon differs from the reported comparator horizon."
    if candidate.get("route") == "phase":
        sequence = candidate.get("sequence") or []
        boundaries = [i for i in range(1, len(sequence)) if sequence[i] != sequence[i - 1]]
        return "informational", (f"Route and ensemble agree; actual action-change positions are {boundaries}. "
            "The paper's phase-boundary label does not provide the complete action sequence or its hash, so exact comparator identity is not established.")
    return "match", "Checks the reported route, ensemble and any stated horizon; this is not a match of a full policy hash or checkpoint identity."


def describe_route(candidate: dict | None) -> str:
    if candidate is None:
        return "missing"
    name = ROUTE_NAMES.get(candidate.get("route"), str(candidate.get("route")))
    extra = []
    if candidate.get("route") in ("mpc_ol", "mpc_rh") and candidate.get("horizon") is not None:
        extra.append(f"H = {candidate['horizon']}")
    if candidate.get("route") == "radius" and candidate.get("kappa") is not None:
        extra.append(f"kappa = {candidate['kappa']}")
    if candidate.get("profile") not in (None, "none"):
        extra.append(str(candidate["profile"]))
    return name + (f" ({', '.join(extra)})" if extra else "") + f", E = {candidate.get('ensemble')}"


def expected_routes(identity: str) -> set[str]:
    identity = identity.lower()
    found = set()
    for route, name in ROUTE_NAMES.items():
        if name.lower() in identity:
            found.add(route)
    if "qc returned" in identity:
        found.add("comparator")
    return found


def compare(run: RunReader, reported: dict) -> dict:
    dataset = run.dataset
    t1 = reported["main_table_1_policy_and_deployment_ledger"].get(dataset)
    if t1 is None:
        raise KeyError(f"{dataset} is not a dataset of the manuscript")
    t2 = reported["main_table_2_gain_and_cost_ledger"][dataset]
    s15 = reported["supplementary_table_S15_method_comparison"][dataset]
    s20 = reported["supplementary_table_S20_selection_theorem_instantiation"][dataset]
    s23 = reported["supplementary_table_S23_assurance_and_expected_compute"][dataset]
    s12c = reported["supplementary_table_S12c_selection_rule_audit"][dataset]
    s14 = [r for r in reported["supplementary_table_S14_mechanism_contrasts_submitted_protocol"] if r["dataset"] == dataset]
    s14b = {r["contrast"]: r for r in reported["supplementary_table_S14b_separated_data_sensitivity"]}
    summary = run.optional("final/results_summary.json")
    if not isinstance(summary, dict) or not summary:
        raise ValueError("A completed final/results_summary.json is required")
    candidates = run.optional("select/candidates.json") or []
    selection = run.optional("select/decision.json")
    confirmation = run.optional("confirm/decision.json")
    admission = run.optional("admit/decision.json")
    lock = run.optional("policy_lock.json")
    rows: list[dict] = []

    def add(section, quantity, actual, reported_value, status, location, note=""):
        rows.append({"section": section, "quantity": quantity, "run_value": actual, "reported_value": reported_value,
                     "status": status, "manuscript_location": location, "note": note})

    # --- deployment identities -------------------------------------------------------------
    deployed = candidates[lock["deployed_index"]] if lock and candidates else (summary or {}).get("deployed")
    comparator = candidates[lock["comparator_index"]] if lock and candidates else (summary or {}).get("comparator")
    admitted = (admission or {}).get("passed", (summary or {}).get("admitted"))
    if deployed is not None:
        if admitted is False:
            status = "match" if t1["G_admit"] == 0 else "differs"
            add("Table 1", "deployed rule", "qC returned (admission rejected)", t1["deployed_rule"], status, "main Table 1, final column")
        else:
            expected = expected_routes(t1["deployed_rule"])
            status = "match" if deployed.get("route") in expected and deployed.get("ensemble") == t1["deployed_E"] else "differs"
            note = ""
            if status == "differs" and deployed.get("route") == "radius" and dataset in ("Roman-empire", "PascalVOC-SP", "COCO-SP"):
                note = ("the registered reading of Eq. (19) selects the cheaper radius route on this dataset (Supplementary Table S12c; "
                        "paper/KNOWN_DIFFERENCES.md D12); the manuscript reports the MPC-RH deployment")
            add("Table 1", "deployed rule", describe_route(deployed), f"{t1['deployed_rule']}, E = {t1['deployed_E']}", status, "main Table 1, deployed rule", note)
    else:
        add("Table 1", "deployed rule", None, f"{t1['deployed_rule']}, E = {t1['deployed_E']}", "missing", "main Table 1")
    if comparator is not None:
        status, note = comparator_status(comparator, t1["qC_identity"], t1["qC_E"])
        add("Table 1", "comparator qC route and reported settings", describe_route(comparator), f"{t1['qC_identity']}, E = {t1['qC_E']}", status, "main Table 1, qC identity", note)
    add("Table 1", "global admission G_admit", None if admitted is None else int(bool(admitted)), t1["G_admit"],
        "missing" if admitted is None else ("match" if int(bool(admitted)) == t1["G_admit"] else "differs"), "main Table 1, global admission")
    add("Table 1", "input fallback fraction (g_input)", None, t1["input_fallback_fraction"], "not produced by this source",
        "main Table 1, input fallback", "the general serving-time input gate is outside this source release (docs/IMPLEMENTATION_CONTRACT.md)")

    # --- test-stage effects --------------------------------------------------------------------
    if summary:
        loss = summary.get("paired_qD_minus_qC_loss", {})
        metric = summary.get("paired_qD_minus_qC_metric_pp", {})
        rep, interval = t1["loss_effect_nats"], t1["loss_effect_interval_nats"]
        tolerance = effect_half_unit(rep, interval)
        add("Table 1", f"paired {t1['loss_name']} effect qD - qC (nats), mean", loss.get("mean"), rep, level(loss.get("mean"), rep, tolerance, interval), "main Table 1 / Table 2, test-stage effect")
        if interval:
            add("Table 1", "paired loss effect, lower 95% end", loss.get("lower"), interval[0], level(loss.get("lower"), interval[0], tolerance), "main Table 1")
            add("Table 1", "paired loss effect, upper 95% end", loss.get("upper"), interval[1], level(loss.get("upper"), interval[1], tolerance), "main Table 1")
        rep, interval = t1["official_gain_pp"], t1["official_gain_interval_pp"]
        tolerance = effect_half_unit(rep, interval)
        add("Table 1", f"official {t1['official_metric']} gain qD - qC (pp), mean", metric.get("mean"), rep, level(metric.get("mean"), rep, tolerance, interval), "main Table 1, final effect",
            "Peptides: the manuscript prints native AP (+0.0058); the run reports percentage points" if dataset == "Peptides-func" else "")
        if interval:
            add("Table 1", "official gain, lower 95% end", metric.get("lower"), interval[0], level(metric.get("lower"), interval[0], tolerance), "main Table 1")
            add("Table 1", "official gain, upper 95% end", metric.get("upper"), interval[1], level(metric.get("upper"), interval[1], tolerance), "main Table 1")
        models = summary.get("models", {})
        scale = 1.0 if s15["score_unit"].startswith("AP") else 100.0
        spread = max(s15["seed_SD_range_pp"]) * (0.01 if scale == 1.0 else 1.0)
        deployed_row = next(r for r in s15["rows"] if r["family"] == "CROSS-MORSE deployment")
        comparator_row = next(r for r in s15["rows"] if "(qC" in r["method"])
        plant_row = next(r for r in s15["rows"] if r["method"] == "full-depth plant")
        for name, row in (("qD", deployed_row), ("qC", comparator_row), ("qN", plant_row)):
            value = models.get(name, {}).get("metric")
            value = None if value is None else value * scale
            add("Table S15", f"official test score of {name} ({row['method']})", value, row["official_score"],
                level(value, row["official_score"], 0.0005 if scale == 1.0 else 0.005, spread=spread), "Supplementary Table S15",
                "qN is the full-depth plant at the comparator's ensemble size" if name == "qN" else "")

    # --- admission statistics -------------------------------------------------------------------
    if admission:
        if dataset in MULTI_GRAPH:
            eb = admission.get("empirical_bernstein") or {}
            add("Table 1 / S13", "independent-graph admission units J", eb.get("units"), t1["admission_J"], "match" if eb.get("units") == t1["admission_J"] else ("missing" if eb.get("units") is None else "differs"), "main Table 1; Supplementary Table S13")
            add("Table 1 / S13", "clipped mean D_bar (admission split)", eb.get("mean"), t1["admission_D_bar"], level(eb.get("mean"), t1["admission_D_bar"], printed_half_unit(t1["admission_D_bar"])), "Supplementary Table S13")
            add("Table 1 / S13", "empirical-Bernstein upper limit U_a", eb.get("upper"), t1["admission_U_a"], level(eb.get("upper"), t1["admission_U_a"], printed_half_unit(t1["admission_U_a"])), "main Table 1; Supplementary Table S13",
                "this source admits on the untouched validation split after confirming on D_confirm (separated sequence, E10); the manuscript's U_a was computed under the submitted protocol on the same split")
        elif t1["admission_D_bar"] is not None:
            clipped = admission.get("loss") or {}
            add("Table 1", "clipped mean D_bar on the dependent admission unit", clipped.get("mean"), t1["admission_D_bar"], level(clipped.get("mean"), t1["admission_D_bar"], printed_half_unit(t1["admission_D_bar"])), "main Table 1, global admission")
            add("Table 1", "descriptive upper limit U_desc", clipped.get("upper"), t1["admission_U_desc"], level(clipped.get("upper"), t1["admission_U_desc"], printed_half_unit(t1["admission_U_desc"])), "main Table 1, global admission")

    # --- selection ------------------------------------------------------------------------------
    if selection:
        eligible = selection.get("eligible_indices", [])
        add("Table S20", "|Q_s| (eligible feedback configurations)", len(eligible), s20["Q_s_size"], "match" if len(eligible) == s20["Q_s_size"] else "differs", "Supplementary Table S20")
        add("Table S20", "r_max (largest simultaneous radius)", selection.get("max_radius"), s20["r_max"], level(selection.get("max_radius"), s20["r_max"], printed_half_unit(s20["r_max"])), "Supplementary Table S20")
        uppers = selection.get("upper") or []
        feedback = [u for j, u in enumerate(uppers) if u is not None and j < len(candidates) and not candidates[j].get("is_open_loop", True) and (selection.get("joint_feasible") or [True] * len(uppers))[j]]
        u_min = selection.get("minimum_upper", min(0.0, min(feedback)) if feedback else 0.0)
        add("Table S12c", "U_min over feasible configurations", u_min, s12c["U_min"], level(u_min, s12c["U_min"], printed_half_unit(s12c["U_min"]) if s12c["U_min"] is not None else 0.0), "Supplementary Table S12c",
            "Includes the zero-effect comparator as in Eq. (19); this value precedes the practical-margin eligibility test.")
        provisional = candidates[selection["provisional_index"]] if candidates and selection.get("provisional_index") is not None else None
        add("Table 1", "provisional policy q^_s", describe_route(provisional), t1["provisional"], "informational", "main Table 1, provisional q^_s")

    # --- confirmation -------------------------------------------------------------------------
    if confirmation:
        for row in confirmation.get("contrasts", []):
            kind = row.get("kind")
            key = {"state": "state", "state_destruction": "state", "radius": "radius", "radius_vs_one_step": "radius", "mpc": "RH", "mpc_rh_vs_ol": "RH"}.get(kind)
            if key is None:
                continue
            target = next((r for r in s14 if r["contrast"].endswith("/" + key)), None)
            location = "Supplementary Table S14"
            if dataset in MULTI_GRAPH:
                alt = s14b.get({"PascalVOC-SP": "Pascal", "COCO-SP": "COCO", "Peptides-func": "Peptides"}[dataset] + "/" + key)
                if alt is not None:
                    target = dict(target or {}, effect=alt["effect"], interval=alt["interval"])
                    location = "Supplementary Table S14b (analytical sample-size sensitivity, not a rerun)"
            if target is None:
                continue
            add("Table S14", f"held-out {key} contrast ({kind}), mean", row.get("mean"), target["effect"],
                "informational" if dataset in MULTI_GRAPH else level(row.get("mean"), target["effect"], printed_half_unit(target["effect"]), target.get("interval")), location,
                "S14b rescales uncertainty under stated assumptions; its values are not outcomes of independently repeated confirmation experiments."
                if dataset in MULTI_GRAPH else "")
            add("Table S14", f"held-out {key} contrast, Holm-adjusted P (passes: {row.get('passes')})", row.get("holm_p_value_zero"), target.get("holm_P"), "informational", location)
        confirmed = candidates[confirmation["confirmed_index"]] if candidates else None
        add("Table 1", "confirmed policy q_M", describe_route(confirmed), t1["confirmation"], "informational", "main Table 1, confirmation output")

    # --- costs (informational: the manuscript measures serial execution; the run records a work ledger) ---
    if candidates and lock:
        add("Table 2", "qC work ledger (multiply-adds / nominal E=1)", candidates[lock["comparator_index"]].get("cost"), t2["C_qC"], "informational", "main Table 2, C_qC",
            "the ledger counts sparse/dense products normalized by nominal E=1; the manuscript's cost unit is measured serial execution (Table S23)")
        add("Table 2", "deployed-rule work ledger (multiply-adds / nominal E=1)", candidates[lock["deployed_index"]].get("cost"), s23["controller_cost"], "informational", "Supplementary Table S23, controller cost")
    single_roman = dataset == "Roman-empire"
    if single_roman:
        for row in rows:
            row["status"] = "diagnostic only"
            row["note"] = (row.get("note", "") + " A single official split is not the paper's ten-split aggregate.").strip()
    statuses = ("exact", "interval", "outside", "match", "differs", "missing", "informational",
                "not produced by this source", "not reported", "diagnostic only")
    return {"dataset": dataset, "run": str(run.root), "rows": rows,
            "run_identities": run.identities(),
            "input_verification": "sealed numerical inputs verified" if run.evidence is not None else "not verified (in-memory comparison only)",
            "paper_comparison_eligible": not single_roman,
            "comparison_scope": "single-split diagnostic; use --roman-campaign for the aggregate" if single_roman else "numerical comparison with the observations we report in the manuscript",
            "levels": {k: sum(1 for r in rows if r["status"] == k) for k in statuses},
            "interpretation": ("'exact' means equal at the displayed precision. 'interval' means inside the reported interval or one reported "
                               "across-seed standard deviation; neither is a formal replication test, a prediction interval for a fresh run, "
                               "or evidence of historical provenance. Different architectures, splits and execution protocols can change the result.")}


def emit_binding(run: RunReader, reported: dict, comparison: dict, path: Path, relationship: str, identifier: str, tolerance_scale: float) -> None:
    if run.evidence is None:
        run.verify()
    if not comparison.get("paper_comparison_eligible", True):
        raise ValueError("A single Roman split cannot be bound to the paper's ten-split result")
    if not isinstance(tolerance_scale, (float, int)) or not math.isfinite(tolerance_scale) or tolerance_scale <= 0:
        raise ValueError("Binding tolerance scale must be positive and finite")
    if not relationship:
        raise ValueError("--emit-binding requires an explicit --relationship declaration")
    path = output_path(path, run.root)
    ids = run.identities()
    if ids is None or any(ids.get(k) is None for k in ("source_sha256", "config_sha256", "unit_sha256", "dataset")):
        raise ValueError("A completed run with sealed stage manifests (provenance/stages) is required to emit a binding")
    t1 = reported["main_table_1_policy_and_deployment_ledger"][run.dataset]
    results = []
    # JSON drops trailing zeros (0.60 -> 0.6), so a row's tolerance is the finest printed precision among its point and interval ends.
    loss_tol = tolerance_scale * min(printed_half_unit(v) for v in [t1["loss_effect_nats"], *(t1["loss_effect_interval_nats"] or [])] if v != 0) if t1["loss_effect_nats"] != 0 else 0.0
    metric_tol = tolerance_scale * min(printed_half_unit(v) for v in [t1["official_gain_pp"], *(t1["official_gain_interval_pp"] or [])] if v != 0) if t1["official_gain_pp"] != 0 else 0.0
    results.append({"manuscript_location": f"Main Table 1, {run.dataset}, final effect over qC: paired {t1['loss_name']} (nats)", "artifact": "final/results_summary.json",
                    "json_pointer": "/paired_qD_minus_qC_loss/mean", "reported_value": t1["loss_effect_nats"], "absolute_tolerance": loss_tol})
    if t1["loss_effect_interval_nats"]:
        for end, name in zip(t1["loss_effect_interval_nats"], ("lower", "upper")):
            results.append({"manuscript_location": f"Main Table 1, {run.dataset}, {name} end of the paired loss interval", "artifact": "final/results_summary.json",
                            "json_pointer": f"/paired_qD_minus_qC_loss/{name}", "reported_value": end, "absolute_tolerance": loss_tol})
    results.append({"manuscript_location": f"Main Table 1, {run.dataset}, official {t1['official_metric']} gain (percentage points)", "artifact": "final/results_summary.json",
                    "json_pointer": "/paired_qD_minus_qC_metric_pp/mean", "reported_value": t1["official_gain_pp"], "absolute_tolerance": metric_tol})
    if t1["official_gain_interval_pp"]:
        for end, name in zip(t1["official_gain_interval_pp"], ("lower", "upper")):
            results.append({"manuscript_location": f"Main Table 1, {run.dataset}, {name} end of the official-gain interval (pp)", "artifact": "final/results_summary.json",
                            "json_pointer": f"/paired_qD_minus_qC_metric_pp/{name}", "reported_value": end, "absolute_tolerance": metric_tol})
    s15 = reported["supplementary_table_S15_method_comparison"][run.dataset]
    deployed_row = next(r for r in s15["rows"] if r["family"] == "CROSS-MORSE deployment")
    comparator_row = next(r for r in s15["rows"] if "(qC" in r["method"])
    native_ap = s15["score_unit"].startswith("AP")
    for name, row in (("qD", deployed_row), ("qC", comparator_row)):
        results.append({"manuscript_location": f"Supplementary Table S15, {run.dataset}, {row['method']} official score", "artifact": "final/results_summary.json",
                        "json_pointer": f"/models/{name}/{'metric' if native_ap else 'metric_percent'}", "reported_value": row["official_score"],
                        "absolute_tolerance": tolerance_scale * (0.0005 if native_ap else 0.005)})
    binding = {"schema_version": "result-binding-1", "status": "author_completed", "manuscript_identifier": identifier,
               "relationship_to_manuscript": relationship, "run_identities": ids, "results": results,
               "generated_by": "scripts/compare_with_manuscript.py --emit-binding", "comparison_levels": comparison["levels"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(binding, stream, indent=2, allow_nan=False)
        stream.write("\n")


def output_path(path: Path, input_root: Path) -> Path:
    """Preserve source, evidence, and prior reports when creating an output."""
    path = Path(path).absolute()
    if path.is_symlink() or path.exists():
        raise ValueError(f"Choose a new output file; existing paths are preserved: {path}")
    resolved = path.resolve()
    source = Path(__file__).resolve().parents[1]
    if resolved.is_relative_to(input_root.resolve()) or resolved.is_relative_to(source):
        raise ValueError("Outputs must be outside the run/campaign and source directories")
    return resolved


def compare_roman_campaign(campaign: Path, reported: dict) -> dict:
    """Compare the actual aggregate of ten sealed official split runs."""
    from scripts.summarize_roman_splits import summarize
    aggregate = summarize(campaign)
    if aggregate.get("scientific_run") is not True:
        raise ValueError("A manuscript campaign comparison requires official-data scientific runs")
    t1 = reported["main_table_1_policy_and_deployment_ledger"]["Roman-empire"]
    s15 = reported["supplementary_table_S15_method_comparison"]["Roman-empire"]
    rows = []
    for field, actual, point, interval in (
        ("paired CE effect qD - qC (nats)", aggregate["paired_qD_minus_qC_loss"], t1["loss_effect_nats"], t1["loss_effect_interval_nats"]),
        ("paired accuracy gain qD - qC (pp)", aggregate["paired_qD_minus_qC_accuracy_pp"], t1["official_gain_pp"], t1["official_gain_interval_pp"]),
    ):
        for statistic, target in (("mean", point), ("lower", interval[0]), ("upper", interval[1])):
            rows.append({"section": "Table 1", "quantity": f"{field}, {statistic}", "run_value": actual[statistic],
                         "reported_value": target, "status": level(actual[statistic], target, effect_half_unit(point, interval), interval if statistic == "mean" else None),
                         "manuscript_location": "main Table 1, Roman-empire, ten official splits", "note": "Paired split-level Student-t descriptive interval; masks share one graph."})
    for name, target in (
        ("qN", next(r for r in s15["rows"] if r["method"] == "full-depth plant")),
        ("qC", next(r for r in s15["rows"] if "(qC" in r["method"])),
        ("qD", next(r for r in s15["rows"] if r["family"] == "CROSS-MORSE deployment")),
    ):
        value = aggregate["models"][name]["accuracy_percent"]["mean"]
        rows.append({"section": "Table S15", "quantity": f"official test accuracy of {name} (percent)", "run_value": value,
                     "reported_value": target["official_score"], "status": level(value, target["official_score"], 0.005),
                     "manuscript_location": "Supplementary Table S15, Roman-empire", "note": "Mean across ten official splits."})
    return {"dataset": "Roman-empire", "run": str(campaign.resolve()), "rows": rows,
            "input_verification": "ten sealed split runs numerically replayed", "paper_comparison_eligible": True,
            "comparison_scope": "ten-split campaign aggregate", "campaign_summary": aggregate,
            "levels": {key: sum(row["status"] == key for row in rows) for key in ("exact", "interval", "outside", "match", "differs", "missing")},
            "interpretation": "Numerical compatibility is descriptive, not a replication guarantee or provenance claim. Ten official masks on one graph are not independent-graph admission units."}


def markdown(comparison: dict) -> str:
    lines = [f"# Run versus manuscript: {comparison['dataset']}", "", f"Run: `{comparison['run']}`", "",
             f"Scope: {comparison.get('comparison_scope', 'numerical comparison')}. Input check: {comparison.get('input_verification', 'not verified')}.", "",
             "| Section | Quantity | Run value | Reported value | Status | Manuscript location |", "|---|---|---|---|---|---|"]
    for row in comparison["rows"]:
        fmt = lambda v: "—" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))
        note = f" ({row['note']})" if row.get("note") else ""
        cells = [row['section'], row['quantity'] + note, fmt(row['run_value']), fmt(row['reported_value']), row['status'], row['manuscript_location']]
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells) + " |")
    lines += ["", f"Levels: {json.dumps(comparison['levels'])}", "", comparison["interpretation"], ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_root", type=Path, help="a completed run directory (eight stages)")
    parser.add_argument("--roman-campaign", action="store_true", help="read split_00 through split_09 and compare their verified aggregate")
    parser.add_argument("--registry", type=Path, help="alternative reported_results.json")
    parser.add_argument("--output", type=Path, help="write the comparison as JSON (and .md beside it); must be outside the run")
    parser.add_argument("--emit-binding", type=Path, help="write a completed result binding for scripts/check_result_binding.py")
    parser.add_argument("--relationship", choices=("exact_result_producing_run", "refactor_with_prediction_equivalence", "post_result_revision"),
                        help="required with --emit-binding: our explicit declaration of the run-to-manuscript relationship")
    parser.add_argument("--manuscript-identifier", default="CROSS-MORSE manuscript identified by SHA-256 in paper/sources.json")
    parser.add_argument("--binding-tolerance-scale", type=float, default=1.0, help="multiple of the printed half-unit used as binding tolerance")
    args = parser.parse_args(argv)
    try:
        reported = read_json(args.registry) if args.registry else load_reported()
        destinations = []
        if args.output:
            if args.output.suffix != ".json":
                raise ValueError("--output must name a .json file; a Markdown companion is written beside it")
            destinations += [output_path(args.output, args.run_root), output_path(args.output.with_suffix(".md"), args.run_root)]
        if args.emit_binding:
            if args.roman_campaign:
                raise ValueError("Campaign comparison does not create a single-run binding; retain the ten sealed split identities")
            if not args.relationship:
                raise ValueError("--emit-binding requires an explicit --relationship declaration")
            if not math.isfinite(args.binding_tolerance_scale) or args.binding_tolerance_scale <= 0:
                raise ValueError("Binding tolerance scale must be positive and finite")
            destinations.append(output_path(args.emit_binding, args.run_root))
        if len(set(destinations)) != len(destinations):
            raise ValueError("Comparison, Markdown and binding outputs must be different files")
        if any(a.is_relative_to(b) or b.is_relative_to(a)
               for i, a in enumerate(destinations) for b in destinations[i + 1:]):
            raise ValueError("Output files cannot be ancestors or descendants of one another")
        if args.roman_campaign:
            comparison = compare_roman_campaign(args.run_root, reported)
        else:
            run = RunReader(args.run_root)
            run.verify()
            comparison = compare(run, reported)
        if args.output:
            output = args.output.resolve()
            if output.is_relative_to(args.run_root.resolve()):
                raise ValueError("--output must be outside the run directory")
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                json.dump(comparison, stream, indent=2, allow_nan=False)
                stream.write("\n")
            with output.with_suffix(".md").open("x", encoding="utf-8") as stream:
                stream.write(markdown(comparison))
        if args.emit_binding:
            emit_binding(run, reported, comparison, args.emit_binding.resolve(), args.relationship, args.manuscript_identifier, args.binding_tolerance_scale)
        print(markdown(comparison))
        if not comparison["paper_comparison_eligible"] or comparison["levels"]["missing"]:
            return 2
        return 0 if not comparison["levels"]["outside"] and not comparison["levels"]["differs"] else 1
    except (OSError, ValueError, KeyError, TypeError, IndexError, ImportError) as exc:
        print(json.dumps({"status": "INCOMPLETE_OR_INVALID", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
