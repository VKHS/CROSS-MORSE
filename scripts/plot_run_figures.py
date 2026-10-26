#!/usr/bin/env python3
"""Draw verified decision panels of completed runs.

For every run directory given, one PNG with four panels is written:

  a  selection geometry      every registered candidate's simultaneous interval [L_s, U_s] relative to the nested
                             comparator (select/decision.json), the registered threshold −ε_R and the near-best band
  b  confirmation contrasts  held-out mechanism contrasts with intervals and Holm-adjusted P (confirm/decision.json)
  c  admission               the admission statistic against its threshold (admit/decision.json)
  d  test stage              paired effect of q_D over q_C with its interval, and the official scores of q0, qN, qC, qD
                             (final/results_summary.json), with the manuscript's reported value of the same dataset
                             optionally drawn for official runs with --compare-manuscript

The panels are read from the sealed decision files only; nothing is recomputed. ``scripts/reproduce_results.py``
is the tool that recomputes the decisions from the observations. Requires numpy and matplotlib.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._paper_common import BCE_DATASETS, MULTI_GRAPH, ROUTE_NAMES, load_reported, read_json
from scripts.compare_with_manuscript import RunReader
from cross_morse.review import EvidenceError, NumericEvidence

COLORS = {"ogbn-arxiv": "#2a78d6", "ogbn-proteins": "#eb6834", "ogbn-products": "#1baf7a", "Roman-empire": "#eda100",
          "PascalVOC-SP": "#e87ba4", "COCO-SP": "#008300", "Peptides-func": "#6250d6"}


def draw(run: RunReader, reported: dict, out: Path, plt) -> Path:
    dataset = run.dataset
    color = COLORS.get(dataset, "#2a78d6")
    t1 = reported["main_table_1_policy_and_deployment_ledger"].get(dataset, {})
    s15 = reported["supplementary_table_S15_method_comparison"].get(dataset, {"rows": [], "score_unit": ""})
    candidates = run.optional("select/candidates.json") or []
    selection = run.optional("select/decision.json") or {}
    confirmation = run.optional("confirm/decision.json") or {}
    admission = run.optional("admit/decision.json") or {}
    summary = run.optional("final/results_summary.json") or {}
    config = run.config
    eps_r = config.get("selection", {}).get("epsilon_r", 0.003)
    eps_eq = config.get("selection", {}).get("epsilon_eq", 0.002)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    # a) selection geometry
    ax = axes[0, 0]
    lower, upper, mean = selection.get("lower") or [], selection.get("upper") or [], selection.get("mean") or []
    eligible = set(selection.get("eligible_indices", []))
    rows = [j for j in range(len(candidates)) if j < len(upper) and upper[j] is not None]
    rows.sort(key=lambda j: (candidates[j].get("is_open_loop", True), upper[j]))
    for k, j in enumerate(rows):
        c = candidates[j]
        feedback = not c.get("is_open_loop", True)
        col = color if feedback else "#9a9994"
        lo = lower[j] if j < len(lower) and lower[j] is not None else upper[j]
        mid = mean[j] if j < len(mean) and mean[j] is not None else upper[j]
        ax.plot([lo, upper[j]], [k, k], color=col, lw=1.6 if feedback else 1.0)
        ax.plot(mid, k, "o" if feedback else "s", color=col, ms=4)
        if j in eligible:
            ax.plot(mid, k, "*", color="#eda100", ms=10, mec="#0b0b0b", mew=0.4)
    u_min = selection.get("minimum_upper")
    if isinstance(u_min, (int, float)) and np.isfinite(u_min):
        ax.axvspan(u_min, u_min + eps_eq, color="#eda100", alpha=0.25, lw=0)
    ax.axvline(-eps_r, color="#e34948", lw=1); ax.axvline(0, color="#0b0b0b", lw=0.8)
    ax.set_yticks(range(len(rows)), [f"{ROUTE_NAMES.get(candidates[j]['route'], candidates[j]['route'])} E={candidates[j]['ensemble']} {candidates[j].get('profile', '')}" for j in rows], fontsize=5.5)
    ax.set_xlabel("simultaneous interval relative to the nested q_C (nats); red: −ε_R; band: [U_min, U_min + ε_eq]; star: eligible (Q_s)")
    ax.set_title(f"a  Selection geometry ({len(rows)} candidates; |Q_s| = {len(eligible)}; r_max = {selection.get('max_radius', float('nan')):.4f})", fontsize=9)
    # b) confirmation contrasts
    ax = axes[0, 1]
    contrasts = confirmation.get("contrasts", [])
    for k, row in enumerate(contrasts):
        lo, hi, m = row.get("lower"), row.get("upper"), row.get("mean")
        if m is None:
            ax.text(0, k, f"{row.get('kind')}: unavailable ({row.get('unavailable_reason', '')})", fontsize=7, va="center"); continue
        ax.plot([lo if lo is not None else m, hi if hi is not None else m], [k, k], color="#0b0b0b", lw=1.6); ax.plot(m, k, "o", color="#0b0b0b", ms=5)
        margin = row.get("practical_margin", 0.0) * row.get("beneficial_direction", 1)
        ax.plot(margin, k, "|", color="#008300", ms=12, mew=1.5)
        ax.text(0.02, 0.9 - 0.12 * k, f"{row['kind']}: mean {m:+.5f}, Holm P {row.get('holm_p_value_zero', float('nan')):.3g}, passes = {row.get('passes')}", transform=ax.transAxes, fontsize=7, color="#52514e")
    ax.axvline(0, color="#0b0b0b", lw=0.8)
    ax.set_yticks(range(len(contrasts)), [f"{r['kind']} (cand {r['candidate_index']})" for r in contrasts], fontsize=7)
    ax.set_xlabel("held-out contrast (nats); green tick: practical margin of Table S11")
    ax.set_title(f"b  Confirmation: q_M = candidate {confirmation.get('confirmed_index')} (downgrades: {len(confirmation.get('downgrades', []))})", fontsize=9)
    # c) admission
    ax = axes[1, 0]
    eps_a = config.get("admission", {}).get("epsilon_a", 0.003)
    if dataset in MULTI_GRAPH and admission.get("empirical_bernstein"):
        eb = admission["empirical_bernstein"]
        ax.bar([0, 1, 2], [eb.get("mean", 0), eb.get("penalty", 0), eb.get("upper", 0)], color=[color, "#c3c2b7", "#0b0b0b"], width=0.6)
        ax.set_xticks([0, 1, 2], ["clipped mean D̄", "Bernstein penalty", "U_a"])
        ax.set_title(f"c  Graph-unit admission (conditional bound): J = {eb.get('units')}, U_a = {eb.get('upper', float('nan')):+.5f}; G_admit = {admission.get('Gadmit')}", fontsize=9)
        if t1.get("admission_U_a") is not None:
            ax.plot(2, t1["admission_U_a"], "o", mfc="none", mec="#e34948", ms=9, mew=1.5); ax.text(2.35, t1["admission_U_a"], f"reported descriptive U_a {t1['admission_U_a']}\n(reused validation)", fontsize=7, color="#e34948", va="center")
    else:
        loss = admission.get("loss", {})
        ax.bar([0, 1], [loss.get("mean", 0), loss.get("upper", 0)], color=[color, "#0b0b0b"], width=0.6)
        ax.set_xticks([0, 1], ["clipped mean D̄", "descriptive upper U_desc"])
        ax.set_title(f"c  Descriptive admission on {loss.get('units')} dependent units; G_admit = {admission.get('Gadmit')}", fontsize=9)
        if t1.get("admission_D_bar") is not None:
            ax.plot(0, t1["admission_D_bar"], "o", mfc="none", mec="#e34948", ms=9, mew=1.5); ax.text(0.35, t1["admission_D_bar"], f"reported D̄ {t1['admission_D_bar']}", fontsize=7, color="#e34948", va="center")
    ax.axhline(-eps_a, color="#e34948", lw=1, ls=":"); ax.axhline(0, color="#0b0b0b", lw=0.8); ax.set_ylabel("nats")
    # d) test stage
    ax = axes[1, 1]
    effect = summary.get("paired_qD_minus_qC_loss", {})
    if effect:
        lo, hi, m = effect.get("lower"), effect.get("upper"), effect.get("mean")
        ax.plot([lo if lo is not None else m, hi if hi is not None else m], [0, 0], color=color, lw=2); ax.plot(m, 0, "o", color=color, ms=7)
        ax.text(m, 0.08, f"run: {m:+.5f}", ha="center", fontsize=7.5, color=color)
    if t1.get("loss_effect_nats") is not None:
        interval = t1.get("loss_effect_interval_nats") or [t1["loss_effect_nats"], t1["loss_effect_nats"]]
        ax.plot(interval, [-0.3, -0.3], color="#e34948", lw=1.2); ax.plot(t1["loss_effect_nats"], -0.3, "o", mfc="none", mec="#e34948", ms=8, mew=1.5)
        ax.text(t1["loss_effect_nats"], -0.42, f"reported: {t1['loss_effect_nats']:+.5f}", ha="center", fontsize=7.5, color="#e34948")
    synthetic = config.get("dataset") == "synthetic" or config.get("scientific_run") is not True
    split_label = "synthetic test allocation" if synthetic else "official test split"
    ax.axvline(0, color="#0b0b0b", lw=0.8); ax.set_ylim(-0.7, 0.4); ax.set_yticks([]); ax.set_xlabel(f"paired {t1.get('loss_name', 'loss')} effect q_D − q_C on the {split_label} (nats)")
    models = summary.get("models", {})
    scale = 1.0 if dataset == "Peptides-func" else 100.0
    metric_label = ("AP (fraction)" if dataset == "Peptides-func" else "ROC-AUC (%)" if dataset == "ogbn-proteins"
                    else "macro F1 (%)" if dataset in ("PascalVOC-SP", "COCO-SP") else "accuracy (%)")
    scores = ", ".join(f"{name} {models[name]['metric'] * scale:.3f}" for name in ("q0", "qN", "qC", "qD") if name in models and models[name].get("metric") is not None)
    reported_rows = {r["method"]: r["official_score"] for r in s15.get("rows", [])}
    deployed_reported = next((v for k, v in reported_rows.items() if "deployed" in k or "returned" in k), None)
    comparison_text = f"; reported deployed score {deployed_reported}" if t1 else ""
    ax.set_title(f"d  Test {metric_label}: {scores}; admitted = {summary.get('admitted')}{comparison_text}", fontsize=8.5)
    scope = "synthetic software example; not benchmark performance" if synthetic else "official-data run; sealed numerical records verified"
    overlay = "; hollow red markers: manuscript values" if t1 else ""
    fig.suptitle(f"{dataset} — {scope}{overlay}", fontsize=10, y=0.995)
    fig.tight_layout()
    path = out / f"run_{run.root.name}_{dataset}.png"
    # Exclusive creation also prevents following a file symlink after preflight.
    with path.open("xb") as handle:
        fig.savefig(handle, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_roots", type=Path, nargs="+", help="completed run directories")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for the PNG files (outside the runs)")
    parser.add_argument("--compare-manuscript", action="store_true", help="overlay reported values for verified official-data runs only")
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    raw = args.output_dir.absolute()
    if any(p.is_symlink() for p in (raw, *raw.parents)):
        raise EvidenceError("Output directory cannot contain symlinks")
    out = raw.resolve()
    source = Path(__file__).resolve().parents[1]
    if out.is_relative_to(source) or source.is_relative_to(out):
        raise EvidenceError("Output must be outside the source tree and its ancestors")
    reported = load_reported() if args.compare_manuscript else {"main_table_1_policy_and_deployment_ledger": {}, "supplementary_table_S15_method_comparison": {}}
    runs = []
    destinations = set()
    for root in args.run_roots:
        run = RunReader(root)
        if out.is_relative_to(run.root) or run.root.is_relative_to(out):
            raise SystemExit("--output-dir must be outside the run directory")
        if run.dataset not in COLORS:
            raise EvidenceError(f"Unrecognized task configuration: {run.dataset}")
        target = out / f"run_{run.root.name}_{run.dataset}.png"
        if target in destinations:
            raise EvidenceError("Runs produce the same figure name; use unique run directory basenames")
        if target.exists() or target.is_symlink():
            raise EvidenceError("Choose a fresh figure destination; existing files are preserved")
        destinations.add(target)
        if args.compare_manuscript:
            run.verify()
        else:
            evidence = NumericEvidence(run.root)
            if evidence.json("config.json") != run.config:
                raise EvidenceError("Configuration changed while opening the run")
            for relative in ("select/candidates.json", "select/decision.json", "confirm/decision.json",
                             "admit/decision.json", "policy_lock.json", "final/results_summary.json"):
                evidence.json(relative)
            run.evidence = evidence
        runs.append(run)
    out.mkdir(parents=True, exist_ok=True)
    for run in runs:
        print(draw(run, reported, out, plt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
