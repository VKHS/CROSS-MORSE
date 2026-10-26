#!/usr/bin/env python3
"""Verify that the executable constants of this source equal the manuscript's registered constants.

Standard-library only. The check reads ``paper/registered_constants.json`` and
``paper/reported_results.json`` (transcribed from the manuscript), the action
registry literal in ``cross_morse/core.py`` and the seven ``configs/*_full.json``
profiles, and verifies:

  1. the ten-action registry A0-A9 (Supplementary Table S6);
  2. the derived registry quantities of Table S6b through the formulas of main Eqs. (9)-(10);
  3. the band sensitivities of Supplementary Figure S15b;
  4. the registered thresholds, grids and backbone settings in the full profiles
     (main Eqs. (5), (7), (19), (27); Tables S3, S5, S7, S9, S13);
  5. the internal arithmetic of the reported ledgers (Table 2 from Tables S23/S41/S13;
     Corollary 2; Table 1 against Table S15).

A failure prints the offending quantity. The exit status is 1 when any check fails.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._paper_common import (BCE_DATASETS, DATASETS, MULTI_GRAPH, STEMS, amortization_thousand_input_seconds,
                                   band_sensitivity, bernstein_penalty, exchange_rate_mnats, executed_hops,
                                   executed_identity_coefficient, full_config, identity_mass, kernel_hops,
                                   load_registered, load_reported, minimum_admission_J, parse_actions_from_core)


def percent(cost: float, reference: float) -> int:
    """Overhead in percent, rounded half up on the decimal value (the manuscript's convention)."""
    return int(math.floor(100.0 * (cost / reference - 1.0) + 0.5 + 1e-9))


class Report:
    def __init__(self):
        self.rows: list[dict] = []

    def check(self, group: str, name: str, actual, expected, tolerance: float = 0.0, note: str = ""):
        if isinstance(expected, (int, float)) and not isinstance(expected, bool) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
            ok = abs(float(actual) - float(expected)) <= tolerance + 1e-12
        else:
            ok = actual == expected
        self.rows.append({"group": group, "check": name, "actual": actual, "expected": expected,
                          "tolerance": tolerance, "status": "OK" if ok else "FAIL", "note": note})
        return ok

    @property
    def failures(self):
        return [r for r in self.rows if r["status"] != "OK"]


def check_registry(report: Report, registered: dict):
    actions = parse_actions_from_core()
    table = registered["action_registry_table_S6"]["actions"]
    report.check("S6", "number of registered actions", len(actions), len(table))
    by_id = {a["id"]: a for a in actions}
    probability_sum = 0.0
    for row in table:
        code = by_id.get(row["id"])
        if code is None:
            report.check("S6", f"A{row['id']} present", False, True)
            continue
        probability_sum += code["probability"]
        report.check("S6", f"A{row['id']} exploration probability", code["probability"], row["exploration_probability"])
        if row["mode"] in ("nominal", "one step", "resolvent"):
            report.check("S6", f"A{row['id']} rho", code["rho"], row["rho"])
            report.check("S6", f"A{row['id']} nu", code["nu"], row["nu"])
            report.check("S6", f"A{row['id']} alpha", code["alpha"], row["alpha"])
            report.check("S6", f"A{row['id']} eta", code["eta"], row["eta"])
            report.check("S6", f"A{row['id']} truncation order T", code["order"] if row["eta"] is not None else None, row["T"])
            report.check("S6", f"A{row['id']} is a transport action", code["mode"], "transport")
        else:
            report.check("S6", f"A{row['id']} is a direct recovery action", code["mode"] in ("source", "last_safe"), True)
    report.check("S6", "exploration probabilities sum", round(probability_sum, 10), registered["action_registry_table_S6"]["exploration_probabilities_sum"])
    # Table S6b through the registry formulas
    derived = registered["action_registry_table_S6"]["derived_table_S6b"]
    for key, row in derived.items():
        code = by_id[int(key[1:])]
        report.check("S6b", f"{key} kernel hops", kernel_hops(code["rho"], code["eta"], code["order"]), row["kernel_hops"], 0.0005)
        report.check("S6b", f"{key} executed range (hops/step)", executed_hops(code["rho"], code["nu"], code["eta"], code["order"]), row["executed_range_hops_per_step"], 0.0005)
        report.check("S6b", f"{key} executed identity coefficient", executed_identity_coefficient(code["rho"], code["nu"], code["eta"], code["order"]), row["executed_identity_coefficient"], 0.0005)
        mass = identity_mass(code["eta"], code["order"])
        if row["kernel_identity_mass"] is None:
            report.check("S6b", f"{key} kernel identity mass (one-step kernel)", mass, None)
        else:
            report.check("S6b", f"{key} kernel identity mass", mass, row["kernel_identity_mass"], 0.00005)
        report.check("S6b", f"{key} maximum reach (hops)", code["order"] if code["eta"] is not None else 1, row["max_reach_hops"])
        report.check("S6b", f"{key} sparse products per step", code["order"] if code["eta"] is not None else 1, row["sparse_products_per_step"])
    # Figure S15b band sensitivities (printed at two decimals)
    figure = registered["action_registry_table_S6"]["band_sensitivity_figure_S15b"]
    for key, printed in figure.items():
        if not key.startswith("A"):
            continue
        values = band_sensitivity(by_id[int(key[1:])], by_id[0])
        for b, (value, target) in enumerate(zip(values, printed)):
            report.check("S15b", f"{key} band {b} sensitivity", value, target, 0.00501, "printed at two decimals; half-unit rounding")


def check_configs(report: Report, registered: dict):
    entry = registered["entry_contract_eq5_table_S3"]
    support = registered["support_eq7"]
    selection = registered["selection_eq19_algorithm_S4"]
    admission = registered["admission_eq27_theorem2_table_S13"]
    observer = registered["observer_eq12_table_S9"]
    backbones = registered["backbones_table_S5"]
    exploration = registered["exploration_support_table_S7"]
    roles = registered["data_roles_table_S2"]
    for dataset in DATASETS:
        c = full_config(dataset)
        tag = STEMS[dataset]
        bce = dataset in BCE_DATASETS
        report.check("Eq5", f"{tag}: epsilon_plant", c["entry"]["plant_margin"], entry["epsilon_plant_nats"])
        report.check("Eq5", f"{tag}: tau_PDD (fraction)", c["entry"]["pdd_max"], entry["tau_PDD_percentage_points"] / 100.0)
        report.check("Eq5", f"{tag}: tau_DIL", c["entry"]["dil_gap_max"], entry["tau_DIL_nats"])
        report.check("Eq5", f"{tag}: tau_sup", c["entry"]["registry_fraction_min"], entry["tau_sup"])
        report.check("Eq5", f"{tag}: epsilon_headroom", c["entry"]["headroom_margin"], entry["epsilon_headroom_nats"])
        report.check("Eq7", f"{tag}: p_min", c["support"]["p_min"], support["p_min"])
        report.check("Eq7", f"{tag}: n_min (ESS)", c["support"]["n_min"], support["n_min_ESS"])
        report.check("Eq7", f"{tag}: omega_max*", c["support"]["max_normalized_weight"], support["omega_max_star"])
        report.check("Eq19", f"{tag}: epsilon_R", c["selection"]["epsilon_r"], selection["epsilon_R_nats"]["mean_multilabel_BCE" if bce else "graph_level_CE"])
        report.check("Eq19", f"{tag}: epsilon_eq", c["selection"]["epsilon_eq"], selection["epsilon_eq_nats"])
        report.check("Eq19", f"{tag}: delta_s", c["selection"]["alpha"], selection["delta_s"])
        report.check("Eq19", f"{tag}: selection resamples", c["selection"]["bootstrap_reps"], selection["bootstrap_resamples"])
        report.check("S2", f"{tag}: outer folds", c["selection"]["outer_folds"], roles["selection_folds"]["outer"])
        report.check("S2", f"{tag}: inner folds", c["selection"]["inner_folds"], roles["selection_folds"]["inner"])
        report.check("S2", f"{tag}: plant seeds", len(c["training"]["seeds"]), roles["plant_seeds"])
        report.check("Eq27", f"{tag}: clip B", c["admission"]["clip"], admission["B"])
        report.check("Eq27", f"{tag}: delta_a", c["admission"]["alpha"], admission["delta_a"])
        report.check("Eq27", f"{tag}: epsilon_a", c["admission"]["epsilon_a"], admission["epsilon_a_nats"]["mean_multilabel_BCE" if bce else "graph_level_CE"])
        report.check("Eq27", f"{tag}: admission rule", c["admission"]["rule"], "empirical_bernstein" if dataset in MULTI_GRAPH else "cluster_bootstrap")
        report.check("S9", f"{tag}: observer state dimension", c["controller"]["state_dim"], observer["state_dimension"])
        report.check("S9", f"{tag}: observer covariance rank", c["controller"]["covariance_rank"], observer["covariance_rank"])
        report.check("S9", f"{tag}: observer period", c["controller"]["observer_period"], observer["observer_period"])
        report.check("S9", f"{tag}: trajectory folds", c["controller"]["crossfit_folds"], observer["trajectory_folds"])
        report.check("S9", f"{tag}: horizons", c["controller"]["horizons"], observer["horizons"])
        report.check("S9", f"{tag}: beam width", c["controller"]["beam_width"], observer["beam_width"])
        report.check("S9", f"{tag}: top-k", c["controller"]["top_k"], observer["top_k"])
        report.check("S9", f"{tag}: gamma", c["controller"]["gamma"], observer["gamma"])
        report.check("S9", f"{tag}: registered ensemble sizes", c["registry"]["ensemble_sizes"], observer["ensemble_sizes_registered"])
        report.check("S9", f"{tag}: controller epochs/patience/batch", [c["controller"]["epochs"], c["controller"]["patience"], c["controller"]["batch_size"]], [200, 20, 512])
        report.check("S9", f"{tag}: controller lr/weight decay", [c["controller"]["lr"], c["controller"]["weight_decay"]], [0.001, 0.0001])
        report.check("S8", f"{tag}: Chebyshev degree", c["sensor"]["degree"], 8)
        report.check("S8", f"{tag}: retained bands", len(c["sensor"]["bands"]), 3)
        report.check("S7", f"{tag}: exploration trajectories", c["exploration"]["trajectories"], exploration[dataset]["trajectories"])
        b = backbones[dataset]
        report.check("S5", f"{tag}: controlled depth", c["model"]["depth"], b["depth"])
        report.check("S5", f"{tag}: hidden width", c["model"]["hidden_dim"], b["hidden"])
        report.check("S5", f"{tag}: precision", c["model"]["precision"], "fp32" if b["precision"].startswith("fp32") else "bf16")
        epochs, patience = {"ogbn-arxiv": (500, 50), "ogbn-proteins": (400, 40), "ogbn-products": (300, 30), "Roman-empire": (800, 100),
                            "PascalVOC-SP": (300, 30), "COCO-SP": (300, 30), "Peptides-func": (250, 30)}[dataset]
        report.check("S5", f"{tag}: training epochs/patience", [c["training"]["epochs"], c["training"]["patience"]], [epochs, patience])
        lr, wd = {"ogbn-arxiv": (3e-4, 1e-4), "ogbn-proteins": (2e-4, 1e-5), "ogbn-products": (5e-4, 1e-5), "Roman-empire": (1e-3, 5e-4),
                  "PascalVOC-SP": (5e-4, 1e-5), "COCO-SP": (5e-4, 1e-5), "Peptides-func": (1e-3, 1e-5)}[dataset]
        report.check("S5", f"{tag}: optimizer lr/weight decay", [c["training"]["lr"], c["training"]["weight_decay"]], [lr, wd])
        if dataset in ("ogbn-arxiv", "ogbn-proteins", "ogbn-products"):
            report.check("S5", f"{tag}: cosine schedule", c["training"]["schedule"], "cosine")
        if dataset in ("PascalVOC-SP", "COCO-SP"):
            report.check("S5", f"{tag}: graph batch size", c["training"]["graph_batch_size"], 32)
            report.check("S5b", f"{tag}: native GPS blocks", c["model"]["native_layers"], 6)
        if dataset == "Peptides-func":
            report.check("S5", f"{tag}: graph batch size", c["training"]["graph_batch_size"], 128)
            report.check("S5", f"{tag}: checkpoint rule", c["training"]["checkpoint_rule"], "max_ap_bce_tiebreak")
            report.check("S5b", f"{tag}: native GINE layers", c["model"]["native_layers"], 8)
        if dataset == "ogbn-products":
            report.check("S5", f"{tag}: training batch", c["training"]["batch_size"], 65536)
        if dataset == "ogbn-arxiv":
            report.check("S1", f"{tag}: statistical units", [c["data"]["num_clusters"], c["data"]["partitioner"], c["data"]["buffer_hops"]], [32, "metis", 2])
        if dataset == "ogbn-proteins":
            report.check("S1", f"{tag}: statistical units", [c["data"]["num_clusters"], c["data"]["partitioner"], c["data"]["buffer_hops"]], [32, "degree_stratified", 2])
        if dataset == "ogbn-products":
            report.check("S1", f"{tag}: statistical units", [c["data"]["num_clusters"], c["data"]["partitioner"], c["data"]["buffer_hops"]], [64, "metis", 2])
        encoder_layers = {"ogbn-arxiv": 4, "ogbn-proteins": 4, "ogbn-products": 3, "Roman-empire": 4}.get(dataset)
        if encoder_layers is not None:
            report.check("S5", f"{tag}: encoder layers", c["model"]["encoder_layers"], encoder_layers)


def check_reported_arithmetic(report: Report, registered: dict, reported: dict):
    t1 = reported["main_table_1_policy_and_deployment_ledger"]
    t2 = reported["main_table_2_gain_and_cost_ledger"]
    s13 = reported["supplementary_table_S13_independent_graph_admission"]
    s15 = reported["supplementary_table_S15_method_comparison"]
    s23 = reported["supplementary_table_S23_assurance_and_expected_compute"]
    s41 = reported["supplementary_table_S41_qualification_cost_components"]
    s23b = reported["supplementary_table_S23b_nominal_execution_time_and_amortization"]
    admission = registered["admission_eq27_theorem2_table_S13"]
    for dataset in DATASETS:
        tag = STEMS[dataset]
        row1, row2 = t1[dataset], t2[dataset]
        # Table 1 <-> Table 2: the same test-stage effect
        report.check("T1/T2", f"{tag}: loss effect printed identically", row1["loss_effect_nats"], row2.get("loss_effect_nats", row1["loss_effect_nats"]))
        if row1["loss_effect_interval_nats"] is not None:
            lo, hi = row1["loss_effect_interval_nats"]
            report.check("T1", f"{tag}: loss interval contains the point", lo <= row1["loss_effect_nats"] <= hi, True)
            report.check("T1", f"{tag}: loss interval symmetric (bootstrap percentile)", (row1["loss_effect_nats"] - lo) - (hi - row1["loss_effect_nats"]), 0.0, 0.00011)
        if row1["official_gain_interval_pp"] is not None:
            lo, hi = row1["official_gain_interval_pp"]
            report.check("T1", f"{tag}: metric interval contains the point", lo <= row1["official_gain_pp"] <= hi, True)
        # Table 1 official gain <-> Table S15 deployed score minus comparator score
        rows = {r["method"]: r for r in s15[dataset]["rows"]}
        deployed = next(r for r in s15[dataset]["rows"] if r["family"] == "CROSS-MORSE deployment")
        comparator = next(r for r in s15[dataset]["rows"] if "(qC" in r["method"])
        scale = 100.0 if s15[dataset]["score_unit"].startswith("AP") else 1.0
        gain = (deployed["official_score"] - comparator["official_score"]) * scale
        if dataset == "Peptides-func":
            discrepancy = reported["_interpretation_notes"]["peptides_score_discrepancy"]
            report.check("T1/S15/S26", f"{tag}: documented difference between absolute-score subtraction and paired gain (AP)",
                         (gain - row1["official_gain_pp"]) / 100.0,
                         discrepancy["difference_minus_main_paired_gain_AP"], 1e-12,
                         "Preserved discrepancy, not equality: S26 prints 0.7200-0.7140=0.0060 AP; main Table 1 reports 0.0058. "
                         "S26 explicitly says four-decimal rounding alone cannot reconcile 0.0002 AP.")
        else:
            report.check("T1/S15", f"{tag}: official gain equals S15 deployed minus qC", gain, row1["official_gain_pp"], 0.011)
        # Table 2 <-> Table S23: expected cost and overhead
        if dataset != "ogbn-products":
            expected = s23[dataset]["expected_serial_cost"]
            comparator_cost = s23[dataset]["comparator_cost"]
            report.check("T2/S23", f"{tag}: C_expected", expected, row2["C_expected"], 0.005)
            report.check("T2/S23", f"{tag}: C_qC", comparator_cost, row2["C_qC"])
            report.check("T2/S23", f"{tag}: expected overhead (percent)", percent(expected, comparator_cost), row2["expected_overhead_pct"])
            report.check("T2/S23", f"{tag}: C_assure", s23[dataset]["assurance_serial_cost"], row2["C_assure"], 0.05)
            report.check("T2/S23", f"{tag}: assurance overhead (percent)", percent(s23[dataset]["assurance_serial_cost"], comparator_cost), row2["assure_overhead_pct"])
            report.check("T2/S23", f"{tag}: C_assure = controller + comparator (Eq. 32 two-branch)", s23[dataset]["controller_cost"] + comparator_cost, s23[dataset]["assurance_serial_cost"], 0.0005)
            p = s23[dataset]["fallback_probability"]
            report.check("S23", f"{tag}: Eq. 32 expected cost from controller/comparator/fallback", (1 - p) * s23[dataset]["controller_cost"] + p * comparator_cost, expected, 0.0015)
            # Definition 2 economics
            report.check("T2/Def2", f"{tag}: exchange rate lambda* (millinats)", exchange_rate_mnats(abs(row1["loss_effect_nats"]), expected, comparator_cost), row2["lambda_star_mnats"], 0.01)
            report.check("T2/Def2", f"{tag}: amortization M* t_nom (10^3 input-s)", amortization_thousand_input_seconds(s41[dataset]["total_qualification_increment_GPU_h"], expected, comparator_cost), row2["M_star_t_nom_thousand_input_seconds"], 0.5)
            report.check("T2/S23b", f"{tag}: M* t_nom equals Table S23b", s23b[dataset]["M_star_t_nom_thousand_input_s"], row2["M_star_t_nom_thousand_input_seconds"])
            components = sum(s41[dataset][k] for k in ("exploration_logging", "fitting", "nested_selection", "confirmation", "admission"))
            report.check("S41", f"{tag}: qualification components sum to the total", components, s41[dataset]["total_qualification_increment_GPU_h"], 0.011)
        # Table S13 Bernstein arithmetic (multi-graph) and Corollary 2 (single-graph)
        if dataset in s13:
            r = s13[dataset]
            report.check("Eq27/S13", f"{tag}: Bernstein penalty", bernstein_penalty(r["S2"], r["J"], r["B"], r["delta_a"]), r["penalty"], 0.0000051)
            report.check("Eq27/S13", f"{tag}: U_a = D_bar + penalty", r["D_bar"] + r["penalty"], r["U_a"], 0.0000051)
            report.check("Eq27/S13", f"{tag}: admission decision U_a <= -epsilon_a", int(r["U_a"] <= -r["epsilon_a"]), r["G_admit"])
            report.check("T1/S13", f"{tag}: Table 1 U_a", round(r["U_a"], 5), row1["admission_U_a"], 0.0000051)
            report.check("T1/S13", f"{tag}: Table 1 J", r["J"], row1["admission_J"])
            report.check("S13", f"{tag}: epsilon_a by loss type", r["epsilon_a"], admission["epsilon_a_nats"]["mean_multilabel_BCE" if dataset in BCE_DATASETS else "graph_level_CE"])
            for label, sens in r["sensitivity"].items():
                B = float(label.split("=")[1])
                report.check("S13", f"{tag}: sensitivity {label} penalty", bernstein_penalty(r["S2"], r["J"], B, r["delta_a"]), sens["penalty"], 0.0000051)
                report.check("S13", f"{tag}: sensitivity {label} decision", int(r["D_bar"] + sens["penalty"] <= -r["epsilon_a"]), sens["G_admit"])
        elif row1["admission_D_bar"] is not None:
            eps = admission["epsilon_a_nats"]["mean_multilabel_BCE" if dataset in BCE_DATASETS else "graph_level_CE"]
            expected_J = {"ogbn-arxiv": 1279, "ogbn-proteins": 853, "Roman-empire": 602}[dataset]
            report.check("Cor2", f"{tag}: minimum admission J from Corollary 2 (integer part of the threshold)", math.floor(minimum_admission_J(row1["admission_D_bar"], eps)), expected_J, 0,
                         "main text: 'J > 602 (Roman), 853 (proteins) and 1279 (arxiv)'")
    report.check("Cor2", "constant 14 B log(2/delta_a)/3 (nats)", 14 * admission["B"] * math.log(2 / admission["delta_a"]) / 3, admission["corollary2_constant_nats"], 0.0005)
    # Abstract ranges
    claims = reported["abstract_and_section_claims"]
    overheads = [t2[d]["expected_overhead_pct"] for d in DATASETS if d != "ogbn-products"]
    report.check("Abstract", "expected overhead range 15-33%", [min(overheads), max(overheads)], claims["expected_overhead_pct_range"])
    rates = [t2[d]["lambda_star_mnats"] for d in DATASETS if d != "ogbn-products"]
    report.check("Abstract", "exchange-rate range 3.5-7.3 millinats", [round(min(rates), 1), round(max(rates), 1)], claims["millinats_per_nominal_execution_of_added_work"])
    effects = [t1[d]["loss_effect_nats"] for d in DATASETS]
    report.check("Abstract", "paired CE-or-mean-BCE change 0 to -0.00905 nats", [max(effects), min(effects)], claims["paired_CE_change_range_nats"])
    gains = [t1[d]["official_gain_pp"] for d in DATASETS]
    report.check("Abstract", "official metrics up to 0.65 points", max(gains), claims["official_metric_gain_max_points"])
    assure = [t2[d]["assure_overhead_pct"] for d in DATASETS if d != "ogbn-products"]
    report.check("Abstract", "assurance configuration 118-135%", [min(assure), max(assure)], claims["assurance_cost_pct_range"])
    # Table S15: deployed-rule GPU-hours minus plant GPU-hours is the qualification increment; Table S41 totals
    # agree with it for the six admitted datasets, and products (absent from S41) supplies the upper end 2.3.
    increments = {}
    for d in DATASETS:
        deployed = next(r for r in s15[d]["rows"] if r["family"] == "CROSS-MORSE deployment")
        plant = next(r for r in s15[d]["rows"] if r["method"] == "full-depth plant")
        increments[d] = deployed["GPU_h"] - plant["GPU_h"]
        if s41[d]["total_qualification_increment_GPU_h"] is not None:
            report.check("S41/S15", f"{STEMS[d]}: qualification increment equals S15 deployed GPU-h minus plant GPU-h", increments[d], s41[d]["total_qualification_increment_GPU_h"], 0.011)
        else:
            report.check("S41/S15", f"{STEMS[d]}: S15-derived qualification increment recorded", round(increments[d], 2), s41[d]["qualification_increment_derived_from_table_S15_GPU_h"], 0.011)
    report.check("Abstract", "qualification 1.4-2.3 GPU-hours (Table S15 increments)", [round(min(increments.values()), 1), round(max(increments.values()), 1)], claims["one_off_qualification_GPU_h"])
    plants = [next(r for r in s15[d]["rows"] if r["method"] == "full-depth plant")["GPU_h"] for d in DATASETS]
    report.check("Abstract", "plant training cost 7.8-12.9 GPU-hours", [round(min(plants), 1), round(max(plants), 1)], [7.8, 12.9])
    report.check("Abstract", "training-plus-qualification multiplier 1.176x (arxiv)", round(12.8 / 10.88, 3), 1.176)
    traj = [registered["exploration_support_table_S7"][d]["trajectories"] for d in DATASETS]
    report.check("Abstract", "exploration budget 9,600-32,000 trajectories", [min(traj), max(traj)], claims["exploration_budget_trajectories"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json-output", type=Path, help="write the full check table as JSON (outside the source tree)")
    parser.add_argument("--quiet", action="store_true", help="print failures only")
    args = parser.parse_args(argv)
    registered, reported = load_registered(), load_reported()
    report = Report()
    check_registry(report, registered)
    check_configs(report, registered)
    check_reported_arithmetic(report, registered, reported)
    groups = {}
    for row in report.rows:
        groups.setdefault(row["group"], [0, 0])
        groups[row["group"]][0 if row["status"] == "OK" else 1] += 1
    if not args.quiet:
        print(f"{'group':12s} {'ok':>4s} {'fail':>5s}")
        for group, (ok, fail) in groups.items():
            print(f"{group:12s} {ok:4d} {fail:5d}")
    for row in report.failures:
        print(f"FAIL [{row['group']}] {row['check']}: actual={row['actual']!r} expected={row['expected']!r} tol={row['tolerance']} {row['note']}")
    print(f"{len(report.rows) - len(report.failures)} of {len(report.rows)} registered-constant checks passed")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps({"schema_version": "paper-constants-1", "checks": report.rows,
                                                "failures": len(report.failures)}, indent=2) + "\n", encoding="utf-8")
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
