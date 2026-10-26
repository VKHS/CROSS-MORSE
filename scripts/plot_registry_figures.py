#!/usr/bin/env python3
"""Regenerate the manuscript's registry-derived figure panels from paper/ and the registry formulas.

Every panel here is a function of the registered constants, the reported ledgers and the closed-form
expressions of the manuscript (Eqs. 9-10, 27, 30, Corollary 2, Proposition 7(ii)); none needs a run.
Panels that need a run (held-out contrasts, decision origins, selection geometry of a regenerated run)
are produced by ``scripts/plot_run_figures.py``.

Outputs (PNG, 200 dpi) into ``--output-dir``:

  registry_spectral_fingerprint.png   Supplementary Figure S15a-c: spectral response, band sensitivity, executed range
  registry_deployment_economics.png   Main Figure 5b-d: exchange rate, net benefit, amortization horizon
  registry_gain_and_cost.png          Main Figure 4c, 4d, 4f: serial compute, gain against overhead, wall-time overhead
  registry_admission_arithmetic.png   Supplementary Figures S17a, S8a, S17g: Bernstein penalty, clipping bound, regret radii
  registry_selection_geometry.png     Main Figure 2c: Table S12 intervals against the registered thresholds
  registry_mechanism_contrasts.png    Supplementary Figure S9: Table S14 contrasts and the S14b sensitivity

Requires numpy and matplotlib. Values are the manuscript's (transcribed); the figures say so in their titles.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._paper_common import (DATASETS, STEMS, band_sensitivity, bernstein_penalty, executed_hops,
                                   executed_identity_coefficient, load_registered, load_reported, parse_actions_from_core,
                                   spectral_response)

# Fixed categorical order (one hue per dataset, never cycled); every mark is also labelled by name.
COLORS = {"ogbn-arxiv": "#2a78d6", "ogbn-proteins": "#eb6834", "ogbn-products": "#1baf7a", "Roman-empire": "#eda100",
          "PascalVOC-SP": "#e87ba4", "COCO-SP": "#008300", "Peptides-func": "#6250d6"}
SHORT = {"ogbn-arxiv": "arxiv", "ogbn-proteins": "proteins", "ogbn-products": "products", "Roman-empire": "Roman",
         "PascalVOC-SP": "Pascal", "COCO-SP": "COCO", "Peptides-func": "Peptides"}
ACTION_COLORS = ["#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
NOTE = "values transcribed from the manuscript; registry formulas evaluated by this script"


def _style(plt):
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "grid.linewidth": 0.5, "lines.linewidth": 1.8, "figure.dpi": 100})


def spectral_fingerprint(plt, registered, out: Path):
    actions = {a["id"]: a for a in parse_actions_from_core()}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    lam = np.linspace(0, 2, 401)
    ax = axes[0]
    ends = []
    for identifier in range(8):
        a = actions[identifier]
        response = [spectral_response(a["rho"], a["nu"], a["eta"], x, a["order"]) for x in lam]
        ax.plot(lam, response, color=ACTION_COLORS[identifier], lw=2.2 if identifier == 0 else 1.6, label=f"A{identifier}")
        ends.append((response[-1], identifier))
    # direct labels at the right end, pushed apart so they never overlap
    placed = []
    for value, identifier in sorted(ends):
        y = value if not placed else max(value, placed[-1] + 0.075)
        placed.append(y)
        ax.text(2.03, y, f"A{identifier}", color=ACTION_COLORS[identifier], va="center", fontsize=7.5)
        if abs(y - value) > 1e-9:
            ax.plot([2.0, 2.025], [value, y], color=ACTION_COLORS[identifier], lw=0.6)
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xlabel("graph frequency λ of $L_{sym}$"); ax.set_ylabel("response $g_a(λ)$"); ax.set_xlim(0, 2.25)
    ax.set_title("a  Spectral response of the registered actions (T = 8)")
    ax = axes[1]
    matrix = np.array([band_sensitivity(actions[i], actions[0]) for i in range(1, 8)])
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=0.65, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="#0b0b0b" if matrix[i, j] < 0.4 else "#f0efec")
    ax.set_yticks(range(7), [f"A{i}" for i in range(1, 8)]); ax.set_xticks(range(6), [f"[{2*b/6:.2f},{2*(b+1)/6:.2f}]" for b in range(6)], rotation=30, ha="right")
    ax.set_xlabel("band $I_b$ (uniform partition of [0, 2])"); ax.grid(False)
    ax.set_title("b  Band sensitivity $s_{ab} = \\max_{λ∈I_b}|g_a(λ) − g_{A0}(λ)|$")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    ax = axes[2]
    hops = [executed_hops(actions[i]["rho"], actions[i]["nu"], actions[i]["eta"], actions[i]["order"]) for i in range(8)]
    coefficients = [executed_identity_coefficient(actions[i]["rho"], actions[i]["nu"], actions[i]["eta"], actions[i]["order"]) for i in range(8)]
    bars = ax.bar(range(8), hops, color=ACTION_COLORS, width=0.7)
    for i, (bar, h, c) in enumerate(zip(bars, hops, coefficients)):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", fontsize=7.5)
        ax.text(bar.get_x() + bar.get_width() / 2, 0.03, f"id {c:.2f}", ha="center", fontsize=6.5, color="#f0efec", rotation=90)
    ax.axhline(hops[0], color="#52514e", ls=":", lw=1)
    ax.set_xticks(range(8), [f"A{i}" for i in range(8)]); ax.set_ylabel("executed range (hops per step)"); ax.set_ylim(0, 1.15)
    ax.set_title("c  Executed range (Eq. 10) and identity coefficient (Table S6b)")
    fig.suptitle(f"Registry fingerprint — Supplementary Figure S15a–c ({NOTE})", fontsize=9, y=1.02)
    fig.tight_layout(); fig.savefig(out / "registry_spectral_fingerprint.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def deployment_economics(plt, reported, out: Path):
    t1 = reported["main_table_1_policy_and_deployment_ledger"]
    t2 = reported["main_table_2_gain_and_cost_ledger"]
    s23 = reported["supplementary_table_S23_assurance_and_expected_compute"]
    s23b = reported["supplementary_table_S23b_nominal_execution_time_and_amortization"]
    s41 = reported["supplementary_table_S41_qualification_cost_components"]
    admitted = [d for d in DATASETS if d != "ogbn-products"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axes[0]
    for i, d in enumerate(admitted):
        r = t2[d]
        lo, hi = r["lambda_star_range"]
        ax.barh(i, r["lambda_star_mnats"], color=COLORS[d], height=0.6)
        ax.plot([lo, hi], [i, i], color="#0b0b0b", lw=1.2); ax.plot([lo, hi], [i, i], "|", color="#0b0b0b", ms=8)
        ax.text(hi + 0.3, i, f"{r['lambda_star_mnats']:.2f} mnats; {r['lambda_star_metric_rate']} pt per unit", va="center", fontsize=7.5)
    ax.set_yticks(range(len(admitted)), [SHORT[d] for d in admitted]); ax.invert_yaxis(); ax.set_xlim(0, 20)
    ax.set_xlabel("millinats of risk removed per nominal execution of added work (bars: range from the 95% interval of Δ)"); ax.set_title("b  Exchange rate λ* = |Δ| / ΔE C (Definition 2)")
    ax = axes[1]
    grid = np.linspace(0, 10, 101)
    for d in admitted:
        delta = abs(t1[d]["loss_effect_nats"]) * 1000
        extra = s23[d]["expected_serial_cost"] - s23[d]["comparator_cost"]
        ax.plot(grid, delta - grid * extra, color=COLORS[d], label=SHORT[d])
        if d in ("ogbn-arxiv", "PascalVOC-SP"):
            extra_assure = s23[d]["assurance_serial_cost"] - s23[d]["comparator_cost"]
            ax.plot(grid, delta - grid * extra_assure, color=COLORS[d], ls="--", lw=1.1)
    ax.axhline(0, color="#0b0b0b", lw=0.9); ax.set_ylim(-9, 11); ax.legend(ncol=3, loc="upper right", frameon=False)
    ax.set_xlabel("λ (millinats a deployer pays per nominal execution)"); ax.set_ylabel("net benefit (millinats)")
    ax.set_title("c  Net benefit U(q) = |Δ| − λ ΔE C (dashed: at assurance cost)")
    ax = axes[2]
    t_nom = np.logspace(-4, 2, 200)
    offsets = {"ogbn-arxiv": (1.4, 2.2), "ogbn-proteins": (1.4, 0.45), "Roman-empire": (1.4, 1.0), "PascalVOC-SP": (1.5, 2.0), "COCO-SP": (1.5, 0.5), "Peptides-func": (1.5, 1.0)}
    for d in admitted:
        extra = s23[d]["expected_serial_cost"] - s23[d]["comparator_cost"]
        m_star = 3600 * s41[d]["total_qualification_increment_GPU_h"] / (0.1 * extra * t_nom)
        ax.plot(t_nom, m_star, color=COLORS[d], label=SHORT[d])
        point = s23b[d].get("t_nom_measured_s", s23b[d].get("t_nom_measured_ms", 0) / 1000.0)
        ax.plot(point, s23b[d]["M_star_inputs"], "o", mfc="none", mec=COLORS[d], ms=7, mew=1.5)
        fx, fy = offsets[d]
        ax.text(point * fx, s23b[d]["M_star_inputs"] * fy, f"{SHORT[d]} {s23b[d]['M_star_inputs']:.1e}", fontsize=7, color=COLORS[d], va="center")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("$t_{nom}$: wall time per nominal execution (s); hollow markers: reported operating points (Table S23b)"); ax.set_ylabel("$M^*$: inputs served before $C_{qual}/M ≤ θ ΔE C$")
    ax.set_title("d  Amortization horizon M* (θ = 0.1)")
    fig.suptitle(f"Deployment economics — main Figure 5b–d ({NOTE})", fontsize=9, y=1.02)
    fig.tight_layout(); fig.savefig(out / "registry_deployment_economics.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def gain_and_cost(plt, reported, out: Path):
    t1 = reported["main_table_1_policy_and_deployment_ledger"]
    t2 = reported["main_table_2_gain_and_cost_ledger"]
    s23 = reported["supplementary_table_S23_assurance_and_expected_compute"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axes[0]
    x = np.arange(len(DATASETS)); w = 0.27
    ax.bar(x - w, [t2[d]["C_qC"] for d in DATASETS], w, color="#52514e", label="$q_C$ alone")
    ax.bar(x, [t2[d]["C_expected"] for d in DATASETS], w, color="#2a78d6", label="CROSS-MORSE, expected (Eq. 32)")
    ax.bar(x + w, [t2[d]["C_assure"] for d in DATASETS], w, color="#9ec5f4", hatch="//", edgecolor="#2a78d6", label="two-branch assurance $C_{assure}$")
    for i, d in enumerate(DATASETS):
        if t2[d]["expected_overhead_pct"]:
            ax.text(i, t2[d]["C_expected"] + 0.3, f"+{t2[d]['expected_overhead_pct']}%", ha="center", fontsize=7, color="#2a78d6")
            ax.text(i + w, t2[d]["C_assure"] + 0.3, f"+{t2[d]['assure_overhead_pct']}%", ha="center", fontsize=7, color="#184f95")
        else:
            ax.text(i, t2[d]["C_expected"] + 0.3, "returns $q_C$", ha="center", fontsize=7, color="#52514e")
    ax.set_xticks(x, [SHORT[d] for d in DATASETS]); ax.set_ylabel("serial compute / one nominal execution"); ax.legend(frameon=False, loc="upper left")
    ax.set_title("c  Serial compute per dataset (Table 2)")
    ax = axes[1]
    for d in DATASETS:
        r1, r2 = t1[d], t2[d]
        if r1["loss_effect_interval_nats"] is None:
            ax.plot(0, 0, "o", color=COLORS[d]); ax.text(0.5, 0.0003, f"{SHORT[d]} (returns $q_C$)", fontsize=7.5); continue
        lo, hi = r1["loss_effect_interval_nats"]
        ax.errorbar(r2["expected_overhead_pct"], r1["loss_effect_nats"], yerr=[[r1["loss_effect_nats"] - lo], [hi - r1["loss_effect_nats"]]],
                    fmt="o", color=COLORS[d], capsize=3, ms=6)
        dy = {"ogbn-arxiv": -0.0006, "ogbn-proteins": 0.0005, "Peptides-func": 0.0006, "Roman-empire": 0.0004, "PascalVOC-SP": -0.0005, "COCO-SP": 0.0005}[d]
        ax.text(r2["expected_overhead_pct"] + 1.0, r1["loss_effect_nats"] + dy, f"{SHORT[d]}, {r1['official_gain_pp']:+.2f} {r1['official_metric']} pp" if d != "Peptides-func" else f"{SHORT[d]}, +0.0058 AP", fontsize=7, va="center")
    ax.axhline(0, color="#0b0b0b", lw=0.8); ax.set_xlim(-3, 42); ax.set_xlabel("expected overhead over $q_C$ (%; Eq. 32)"); ax.set_ylabel("paired loss effect vs $q_C$ (nats)")
    ax.set_title("d  Gain and expected overhead, jointly (Tables 1–2)")
    ax = axes[2]
    for i, d in enumerate(DATASETS):
        ax.bar(i - 0.2, t2[d]["expected_overhead_pct"], 0.4, color="#c3c2b7", label="expected serial overhead (Table 2)" if i == 0 else None)
        ax.bar(i + 0.2, s23[d]["wall_time_overhead_pct"], 0.4, color=COLORS[d], label="wall-time overhead (Table S23)" if i == 0 else None)
        ax.text(i + 0.2, s23[d]["wall_time_overhead_pct"] + 0.8, f"{s23[d]['wall_time_overhead_pct']}%", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(DATASETS)), [SHORT[d] for d in DATASETS]); ax.set_ylabel("overhead over $q_C$ (%)"); ax.legend(frameon=False, loc="upper left")
    ax.set_title("f  Wall-time overhead per dataset (timing protocol E17)")
    fig.suptitle(f"Cost and gain — main Figure 4c, d, f ({NOTE})", fontsize=9, y=1.02)
    fig.tight_layout(); fig.savefig(out / "registry_gain_and_cost.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def admission_arithmetic(plt, registered, reported, out: Path):
    adm = registered["admission_eq27_theorem2_table_S13"]
    s13 = reported["supplementary_table_S13_independent_graph_admission"]
    t1 = reported["main_table_1_policy_and_deployment_ledger"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axes[0]
    J = np.unique(np.logspace(1, 4.2, 300).astype(int))
    for d, r in s13.items():
        penalty = [bernstein_penalty(r["S2"], j, r["B"], r["delta_a"]) for j in J if j > 1]
        ax.plot([j for j in J if j > 1], penalty, color=COLORS[d], label=f"{SHORT[d]} (S² = {r['S2']:.0e})")
        ax.plot(r["J"], r["penalty"], "s", color=COLORS[d], ms=6)
        ax.axhline(abs(r["D_bar"]) - r["epsilon_a"], color=COLORS[d], ls=":", lw=0.9)
    floor = [14 * adm["B"] * math.log(2 / adm["delta_a"]) / (3 * (j - 1)) for j in J if j > 1]
    ax.plot([j for j in J if j > 1], floor, color="#0b0b0b", ls="--", lw=1, label="variance-free floor 14B log(2/δ)/(3(J−1))")
    for d, J_min in (("ogbn-arxiv", 1279), ("ogbn-proteins", 853), ("Roman-empire", 602)):
        ax.axvline(J_min, color=COLORS[d], lw=0.8, alpha=0.7); ax.text(J_min * 1.06, 1.2e-4, f"{SHORT[d]}: J > {J_min}", rotation=90, fontsize=6.5, color=COLORS[d], va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-4, 1e-1); ax.set_ylabel("penalty $U_a − \\bar D$ (nats)")
    ax.set_xlabel("admission units J (dotted: admissible |D̄| − ε_a; verticals: Corollary 2 minimum J of the single-graph effects)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right"); ax.set_title("a  Empirical-Bernstein penalty (Eq. 27)")
    ax = axes[1]
    names = list(s13)
    bounds = [100 * ((r["J"] - 1) * r["S2"] / r["J"] + r["D_bar"] ** 2) / r["B"] ** 2 for r in s13.values()]
    bars = ax.bar(range(len(names)), bounds, color=[COLORS[d] for d in names], width=0.6)
    for bar, b in zip(bars, bounds):
        ax.text(bar.get_x() + bar.get_width() / 2, b + 0.08, f"{b:.2f}%", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)), [SHORT[d] for d in names]); ax.set_ylabel("maximum clipped fraction (%)"); ax.set_ylim(0, 4.2)
    ax.set_title("b  Clipping-moment bound (Note S24)"); ax.set_xlabel("f ≤ [(J−1)S²/J + D̄²]/B²")
    ax = axes[2]
    n = np.logspace(math.log10(64), 4, 200)
    A, delta_sup, B = 10, 0.05, 0.10
    ax.plot(n, 2 * (2 * B) * np.sqrt(np.log(2 * A / delta_sup) / (2 * n)), color="#0b0b0b", label="Hoeffding 2e(n)")
    for sigma, color in ((0.005, "#2a78d6"), (0.007, "#eb6834"), (0.010, "#1baf7a")):
        bernstein = 2 * (np.sqrt(2 * sigma ** 2 * np.log(4 * A / delta_sup) / n) + 7 * (2 * B) * np.log(4 * A / delta_sup) / (3 * (n - 1)))
        ax.plot(n, bernstein, color=color, label=f"Bernstein, σ₁ = {sigma}")
    for d in ("ogbn-arxiv", "PascalVOC-SP", "COCO-SP"):
        ax.axhline(abs(t1[d]["loss_effect_nats"]), color=COLORS[d], ls=":", lw=0.9); ax.text(70, abs(t1[d]["loss_effect_nats"]) * 1.08, f"{SHORT[d]} effect", fontsize=6.5, color=COLORS[d])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("independent logged units per (stratum, action)"); ax.set_ylabel("one-step regret radius 2e(n) (nats)")
    ax.legend(frameon=False, fontsize=6.5); ax.set_title("c  One-step regret radius (Proposition 7)"); ax.set_xlabel("independent logged units per (stratum, action); |A| = 10, δ_sup = 0.05, B = 0.10")
    fig.suptitle(f"Admission and regret arithmetic — Supplementary Figures S17a, S8a, S17g ({NOTE})", fontsize=9, y=1.02)
    fig.tight_layout(); fig.savefig(out / "registry_admission_arithmetic.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def selection_geometry(plt, registered, reported, out: Path):
    s12 = reported["supplementary_table_S12_selection_ledger"]
    s12c = reported["supplementary_table_S12c_selection_rule_audit"]
    margins = registered["selection_eq19_algorithm_S4"]
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(10, 12), sharex=True)
    for ax, d in zip(axes, DATASETS):
        rows = [r for r in s12[d] if r["route"] != "qC"]
        eps_r = margins["epsilon_R_nats"]["mean_multilabel_BCE" if d in ("ogbn-proteins", "Peptides-func") else "graph_level_CE"]
        u_min = s12c[d]["U_min"]
        if u_min is not None:
            ax.axvspan(u_min, u_min + margins["epsilon_eq_nats"], color="#eda100", alpha=0.25, lw=0)
        ax.axvline(-eps_r, color="#e34948", lw=1)
        ax.axvline(0, color="#0b0b0b", lw=0.8)
        for i, r in enumerate(rows):
            feedback = r["route"] in ("one-step", "radius", "MPC-RH")
            color = COLORS[d] if feedback else "#9a9994"
            ax.plot([r["L_s"], r["U_s"]], [i, i], color=color, lw=2 if feedback else 1.2)
            ax.plot(r["Delta_s_hat"], i, "o" if feedback else "s", color=color, ms=5 if feedback else 4)
            if r["selected"]:
                ax.plot(r["Delta_s_hat"], i, "*", color="#eda100", ms=12, mec="#0b0b0b", mew=0.5)
            ax.text(1.01, i, f"{r['route']} (E={r['E']}, {r['profile']})", fontsize=7, va="center", transform=ax.get_yaxis_transform())
        ax.set_yticks([]); ax.set_ylim(-0.7, len(rows) - 0.3); ax.set_ylabel(SHORT[d], rotation=0, ha="right", va="center", fontsize=8.5, color=COLORS[d])
        ax.text(0.01, 0.04, f"selected: {s12c[d]['selected']}; registered reading (ε_eq on U_s): {s12c[d]['registered_reading_eps_eq_0.002_on_U_s']}", fontsize=6.8, va="bottom", transform=ax.transAxes, color="#52514e")
    axes[-1].set_xlabel("simultaneous selection interval [L_s, U_s] of each route relative to the nested $q_C$ (nats); red: −ε_R; band: [U_min, U_min + ε_eq]; star: selected")
    axes[0].set_title(f"Selection geometry — main Figure 2c from Supplementary Tables S12 and S12c ({NOTE})", fontsize=9)
    axes[-1].set_xlim(-0.0125, 0.0055)
    fig.tight_layout(); fig.savefig(out / "registry_selection_geometry.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def mechanism_contrasts(plt, reported, out: Path):
    s14 = reported["supplementary_table_S14_mechanism_contrasts_submitted_protocol"]
    s14b = {r["contrast"]: r for r in reported["supplementary_table_S14b_separated_data_sensitivity"]}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, r in enumerate(s14):
        lo, hi = r["interval"]
        ax.plot([lo, hi], [i, i], color="#0b0b0b", lw=1.6); ax.plot(r["effect"], i, "o", color="#0b0b0b", ms=5)
        ax.plot([r["margin"] * (1 if r["effect"] > 0 else -1)], [i], "|", color="#008300", ms=12, mew=1.5)
        label = f"n = {r['n']}; raw P {r['raw_P']}; Holm {r['holm_P']}"
        if r["contrast"] in s14b:
            s = s14b[r["contrast"]]
            lo2, hi2 = s["interval"]
            ax.plot([lo2, hi2], [i + 0.3, i + 0.3], color="#2a78d6", lw=1.6); ax.plot(s["effect"], i + 0.3, "o", mfc="none", mec="#2a78d6", ms=5)
            label += f" | SE sensitivity: J {s['J_old']}→{s['J_new']}, Holm {s['holm_P']}"
        ax.text(0.0115, i + 0.1, label, fontsize=6.6, va="center", color="#52514e")
    ax.axvline(0, color="#0b0b0b", lw=0.8)
    ax.set_yticks(range(len(s14)), [r["contrast"] for r in s14]); ax.invert_yaxis(); ax.set_xlim(-0.016, 0.03)
    ax.set_xlabel("contrast (nats); green tick: practical margin (S11); black: reported protocol (S14)\n"
                  "blue: analytical sample-size sensitivity, fixed effect and rescaled SE (S14b)")
    ax.set_title(f"Held-out mechanism contrasts — Supplementary Figure S9 ({NOTE})", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "registry_mechanism_contrasts.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for generated PNG files; existing figure files are replaced")
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    registered, reported = load_registered(), load_reported()
    spectral_fingerprint(plt, registered, out)
    deployment_economics(plt, reported, out)
    gain_and_cost(plt, reported, out)
    admission_arithmetic(plt, registered, reported, out)
    selection_geometry(plt, registered, reported, out)
    mechanism_contrasts(plt, reported, out)
    for name in sorted(p.name for p in out.glob("registry_*.png")):
        print(out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
