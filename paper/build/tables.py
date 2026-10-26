"""Build the paper's tables from evidence records.

Each table is written three times: ``<name>.tex`` (a plain ``tabular`` that compiles without
extra packages), ``<name>.md`` and ``<name>.csv``.  A header line in every file names the
runs the values come from.  A cell that the run contract does not provide is printed as
``--`` (LaTeX ``--``, Markdown/CSV ``—``); no value is ever estimated to fill a cell.

Table identifiers follow the revised manuscript:

T1   policy and deployment ledger          (main Table 1)
T2   gain and cost ledger                  (main Table 2)
S4   target-regime qualification           (Supplementary Table S4)
S12  route-level candidate ledger          (Supplementary Table S12)
S12c selection rule under three readings   (Supplementary Table S12c)
S13  admission arithmetic                  (Supplementary Table S13)
S14  held-out mechanism contrasts          (Supplementary Table S14)
S15  one-row-per-method comparison         (Supplementary Table S15, internal routes)
S23  compute accounting                    (Supplementary Table S23)
S41  qualification cost components         (Supplementary Table S41)
M1   split, seed and environment manifest  (new; answers the reviewers' manifest request)
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .evidence import DATASET_ORDER, ROUTE_LABELS, decision_roles, family_representatives
from .selection import READINGS, selection_readings

MISSING = None
TABLE_IDS = ("T1", "T2", "S4", "S12", "S12c", "S13", "S14", "S15", "S23", "S41", "M1")


class Cell:
    """A formatted value with its three renderings."""

    def __init__(self, text: str | None):
        self.text = text

    def tex(self) -> str:
        if self.text is None:
            return "--"
        return (self.text.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
                .replace("_", r"\_").replace("#", r"\#"))

    def md(self) -> str:
        return "—" if self.text is None else self.text.replace("|", r"\|")

    def csv(self) -> str:
        return "" if self.text is None else self.text


def fmt(value, digits=4, suffix="") -> Cell:
    if value is None:
        return Cell(None)
    if isinstance(value, bool):
        return Cell("yes" if value else "no")
    if isinstance(value, int):
        return Cell(f"{value}{suffix}")
    if isinstance(value, float):
        return Cell(f"{value:.{digits}f}{suffix}")
    return Cell(str(value))


def num(value, digits=4, signed=False, suffix="") -> Cell:
    if value is None:
        return Cell(None)
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return Cell(text + suffix)


def interval(block, digits=4, signed=True, scale=1.0, suffix="") -> Cell:
    if not block or block.get("mean") is None:
        return Cell(None)
    mean = block["mean"] * scale
    text = f"{mean:+.{digits}f}" if signed else f"{mean:.{digits}f}"
    if block.get("lower") is not None and block.get("upper") is not None:
        lo, hi = block["lower"] * scale, block["upper"] * scale
        text += f" [{lo:+.{digits}f}, {hi:+.{digits}f}]" if signed else f" [{lo:.{digits}f}, {hi:.{digits}f}]"
    return Cell(text + suffix)


def route_text(candidate) -> Cell:
    if not candidate:
        return Cell(None)
    label = candidate.get("route_label") or candidate.get("route")
    parts = [label]
    if candidate.get("route") in {"mpc_ol", "mpc_rh"} and candidate.get("depth") is not None:
        parts.append(f"H = {candidate['depth']}")
    if candidate.get("ensemble") is not None:
        parts.append(f"E = {candidate['ensemble']}")
    if candidate.get("profile") not in (None, "none"):
        parts.append(str(candidate["profile"]))
    return Cell(", ".join(parts))


def loss_label(record) -> str:
    name = (record.get("task") or {}).get("loss_name") or ""
    return {"cross_entropy": "CE", "bce": "BCE", "binary_cross_entropy": "BCE"}.get(name, name or "loss")


def metric_label(record) -> str:
    name = (record.get("task") or {}).get("metric") or ""
    return {"accuracy": "accuracy", "rocauc": "ROC-AUC", "macro_f1": "macro F1", "ap": "AP"}.get(name, name)


def admission_text(record) -> Cell:
    adm = record["admission"]
    if adm.get("Gadmit") is None:
        return Cell(None)
    if adm.get("candidate_already_comparator"):
        return Cell("G_admit = 0; comparator returned (no feedback candidate survived selection and confirmation)")
    eb = adm.get("empirical_bernstein") or {}
    loss = adm.get("loss") or {}
    if adm.get("independent_graph_guarantee") and eb.get("upper") is not None:
        text = (f"G_admit = {adm['Gadmit']}; Theorem 2: U_a = {eb['upper']:+.5f} on J = {eb.get('units')} "
                f"independent graphs, epsilon_a = {adm.get('epsilon_a')}")
    else:
        text = f"G_admit = {adm['Gadmit']}; descriptive, {loss.get('units')} dependent units: D-bar = "
        text += f"{loss['mean']:+.5f}" if loss.get("mean") is not None else "--"
        if loss.get("upper") is not None:
            text += f", U_desc = {loss['upper']:+.5f}"
        if eb.get("upper") is not None:
            text += f" (empirical-Bernstein value {eb['upper']:+.5f}, no coverage statement)"
    return Cell(text)


def confirmation_text(record) -> Cell:
    conf = record["confirmation"]
    rows = conf.get("contrasts") or []
    if not rows:
        prov = record["selection"].get("provisional")
        if prov and conf.get("confirmed_index") == conf.get("comparator_index") and prov.get("index") == conf.get("comparator_index"):
            return Cell("no feedback candidate in Q_s; q_C retained")
        return Cell("no contrasts recorded")
    parts = []
    for row in rows:
        kind = row.get("kind")
        if row.get("mean") is None:
            parts.append(f"{kind}: unavailable")
            continue
        parts.append(f"{kind} {row['mean']:+.5f} (Holm P = {row.get('holm_p_value_zero'):.3g}; "
                     f"{'pass' if row.get('passes') else 'fail'})")
    if conf.get("downgrades"):
        parts.append("downgraded: " + "; ".join(d.get("reason", "") for d in conf["downgrades"]))
    return Cell("; ".join(parts))


# ----------------------------------------------------------------------------- tables

def table_T1(records):
    header = ["Dataset", "exact q_C identity; provisional", "confirmation output", "global admission (criterion)",
              "input fallback", "final effect over q_C; scope"]
    rows = []
    for r in records:
        sel = r["selection"]
        fin = r["final"]
        effect = interval(fin.get("paired_metric_pp"), 2, True, 1.0, f" {metric_label(r)} pp")
        loss = interval(fin.get("paired_loss"), 5, True, 1.0, f" {loss_label(r)}")
        final = Cell(None) if effect.text is None and loss.text is None else Cell(
            "; ".join(c.text for c in (effect, loss) if c.text) + f"; {fin.get('statistical_scope') or ''}".rstrip("; "))
        rows.append([Cell(r["dataset"]),
                     Cell("; ".join(c.text for c in (route_text(sel.get("qC")), route_text(sel.get("provisional"))) if c.text) or None),
                     confirmation_text(r), admission_text(r),
                     Cell(None),  # serving-time input gate is not part of the released runner
                     final])
    note = ("input fallback: the serving-time compatibility gate g_input is not implemented in this release, "
            "so the fraction is not measured.")
    return header, rows, note


def table_T2(records):
    header = ["Dataset", "q_C identity, E", "deployed rule, E", "C_qC", "C_expected (overhead)",
              "C_assure (overhead)", "lambda* (mnats per nominal unit)", "qualification (GPU-h or CPU-h)",
              "test-stage effect over q_C"]
    rows = []
    for r in records:
        c = r["cost"]
        over = f" ({c['overhead_percent']:+.0f}%)" if c.get("overhead_percent") is not None else ""
        assure_over = None
        if c.get("C_assure") is not None and c.get("C_qC"):
            assure_over = f" ({100.0 * (c['C_assure'] - c['C_qC']) / c['C_qC']:+.0f}%)"
        hours = None if c.get("qualification_seconds_total") is None else c["qualification_seconds_total"] / 3600.0
        fin = r["final"]
        eff = interval(fin.get("paired_metric_pp"), 2, True, 1.0, " pp")
        los = interval(fin.get("paired_loss"), 5, True, 1.0, f" {loss_label(r)}")
        rows.append([Cell(r["dataset"]), route_text(r["selection"].get("qC")), route_text(r["admission"].get("deployed")),
                     num(c.get("C_qC"), 3), Cell(None if c.get("C_expected") is None else f"{c['C_expected']:.3f}{over}"),
                     Cell(None if c.get("C_assure") is None else f"{c['C_assure']:.3f}{assure_over or ''}"),
                     num(c.get("exchange_rate_mnats_per_nominal_unit"), 2),
                     Cell(None if hours is None else f"{hours:.4f} h ({r['environment'].get('device') or 'device?'})"),
                     Cell("; ".join(x.text for x in (eff, los) if x.text) or None)])
    note = ("Costs are the deployed sparse/dense multiply-add count normalised by one nominal E = 1 execution, as "
            "recorded in select/candidates.json. C_expected equals C_qD because the released runner has no "
            "serving-time input gate (fallback probability not measured). C_assure = C_qD + C_qC when a feedback "
            "rule is deployed, otherwise C_qC. Qualification hours sum the explore, fit, select, confirm and admit "
            "stage durations recorded in stages/*.json on the device named in the cell.")
    return header, rows, note


def table_S4(records):
    header = ["Dataset", "identity test", "depth gate (PDD, DIL)", "DIL gap", "registry support fraction",
              "headroom U(Delta_oracle)", "initial entry pass", "complete entry pass", "enforced"]
    rows = []
    for r in records:
        e = r["entry"]
        head = e.get("headroom") or {}
        rows.append([Cell(r["dataset"]), fmt(e.get("identity_pass")), fmt(e.get("depth_gate_pass")),
                     num(e.get("dil_gap"), 4), num(e.get("registry_support_fraction"), 3),
                     num(head.get("upper"), 5, True), fmt(e.get("initial_entry_pass")), fmt(e.get("complete_pass")),
                     fmt(e.get("enforced"))])
    note = ("Entry statistics are evaluated on the training-only unit (internal D_fit validation and D_explore), "
            "as the contract of main Eq. (5) prescribes. The headroom column is the upper confidence limit of the "
            "finite evaluated-family oracle recorded in select/entry_complete.json.")
    return header, rows, note


def role_text(record, candidate) -> Cell:
    roles = decision_roles(record).get(candidate["index"], [])
    if "selected" in roles and "deployed" not in roles and record["admission"].get("Gadmit") == 1:
        roles = roles + ["not deployed"]
    elif "selected" in roles and "deployed" not in roles:
        roles = roles + ["not admitted"]
    return Cell("; ".join(roles) or None)


def table_S12(records):
    header = ["Dataset", "route", "E", "profile", "Delta_s (select loss minus q_C family)", "L_s", "U_s",
              "feasible", "eligible (U_s < -eps_R)", "role"]
    rows = []
    for r in records:
        qc_cand = r["selection"].get("qC") or {}
        for c in family_representatives(r, "select_upper"):
            delta = None
            if c.get("select_loss") is not None and qc_cand.get("select_loss") is not None:
                delta = c["select_loss"] - qc_cand["select_loss"]
            rows.append([Cell(r["dataset"]), Cell(c["route_label"]), fmt(c.get("ensemble")), Cell(str(c.get("profile"))),
                         num(delta, 5, True), num(c.get("select_lower"), 5, True), num(c.get("select_upper"), 5, True),
                         fmt(c.get("feasible")), fmt(c.get("eligible")), role_text(r, c)])
    note = ("Representative configurations per route family: the comparator q_C, the selected rule and the deployed "
            "rule whenever they belong to the family, otherwise the configuration with the smallest simultaneous "
            "upper limit U_s. The complete candidate ledger of every (route, E, profile) configuration is "
            "select/candidate_ledger.csv in each run. Intervals are the nested paired max-t bootstrap of "
            "Supplementary Algorithm S4 relative to the reconstructed comparator.")
    return header, rows, note


def table_S12c(records):
    header = ["Dataset", "U_min (eligible routes)"] + [
        {"registered": "eps_eq = 0.002 on U_s (registered): near-best -> choice",
         "point-0.002": "eps_eq = 0.002 on Delta_s: near-best -> choice",
         "point-0.001": "eps_eq = 0.001 on Delta_s: near-best -> choice"}[name] for name, *_ in READINGS] + [
        "selected", "registered reading", "registered rule on all configurations"]
    rows = []
    for r in records:
        audit = selection_readings(r)
        if not audit.get("available"):
            rows.append([Cell(r["dataset"])] + [Cell(None)] * (len(header) - 1))
            continue
        cells = [Cell(r["dataset"]), num(audit["u_min"], 5, True)]
        for name, *_ in READINGS:
            reading = audit["readings"][name]
            if not reading.get("available", True):
                cells.append(Cell(None))
                continue
            near = ", ".join(ROUTE_LABELS.get(x, x) for x in reading["near_best_routes"]) or "—"
            choice = ROUTE_LABELS.get(reading["choice_route"], reading["choice_route"]) if reading["choice_route"] else "q_C"
            cells.append(Cell(f"{near} -> {choice}"))
        selected = ROUTE_LABELS.get(audit["selected_route"], audit["selected_route"]) if audit["selected_route"] else "q_C"
        verdict = "consistent" if audit["registered_consistent"] else "inconsistent at the registered rule"
        per_split = (r.get("campaign") or {}).get("selection_readings")
        if per_split:
            agree = sum(1 for a in per_split if a.get("available") and a["registered_consistent"])
            verdict += f" (consistent in {agree} of {len(per_split)} splits)"
        whole = ("reproduces the sealed selection" if audit["registered_all_configurations_reproduces_selection"]
                 else f"differs: index {audit['registered_all_configurations_index']} vs sealed {audit['selected_index']}")
        cells += [Cell(selected), Cell(verdict), Cell(whole)]
        rows.append(cells)
    note = ("Representative row of each feedback route family as in Table S12 (smallest U_s); eligible routes have "
            "U_s < -eps_R and pass the metric gate; the comparator's family is not a candidate. Near-best sets and the "
            "lexicographic choice are computed under the three equivalence readings of Supplementary Table S12c from "
            "the sealed select/decision.json. The last column applies main Eq. (19) to every (route, E, profile) "
            "configuration, as the runner does, and checks it against the run's own provisional selection.")
    return header, rows, note


def table_S13(records):
    header = ["Dataset", "J (units)", "B", "epsilon_a", "D-bar", "S^2", "penalty", "U_a", "clipped fraction",
              "independent graphs", "G_admit"]
    rows = []
    for r in records:
        adm = r["admission"]
        eb = adm.get("empirical_bernstein") or {}
        rows.append([Cell(r["dataset"]), fmt(eb.get("units")), num(adm.get("clip"), 2), num(adm.get("epsilon_a"), 4),
                     num(eb.get("mean"), 5, True), num(eb.get("sample_variance"), 6), num(eb.get("penalty"), 5),
                     num(eb.get("upper"), 5, True), num(eb.get("clipping_fraction"), 3),
                     fmt(adm.get("independent_graph_guarantee")), fmt(adm.get("Gadmit"))])
    note = ("Empirical-Bernstein arithmetic of main Eq. (27) as recorded in admit/decision.json. The bound is a "
            "coverage statement only where the admission units are independent graphs that no decision fixing "
            "q_M has used (independent graphs = yes); on dependent units it is descriptive.")
    return header, rows, note


def table_S14(records):
    header = ["Dataset", "contrast", "candidate -> predecessor", "effect", "95% interval", "units",
              "raw P (zero)", "Holm P", "margin", "P vs margin", "passes"]
    rows = []
    for r in records:
        for row in r["confirmation"].get("contrasts") or []:
            iv = Cell(None) if row.get("lower") is None else Cell(f"[{row['lower']:+.5f}, {row['upper']:+.5f}]")
            rows.append([Cell(r["dataset"]), Cell(row.get("kind")),
                         Cell(f"{row.get('candidate_index')} -> {row.get('predecessor_index')}"),
                         num(row.get("mean"), 5, True), iv, fmt(row.get("units")),
                         num(row.get("p_value_zero"), 4), num(row.get("holm_p_value_zero"), 4),
                         num(row.get("practical_margin"), 4), num(row.get("p_value_margin"), 4), fmt(row.get("passes"))])
    if not rows:
        rows.append([Cell("(no confirmation contrasts were executed: no feedback candidate entered Q_s in these runs)")]
                    + [Cell(None)] * (len(header) - 1))
    note = ("One Holm family per run, formed from the predeclared contrasts of the frozen confirmation map "
            "(Supplementary Table S11); margins are applied to the held-out point estimate.")
    return header, rows, note


def table_S15(records):
    header = ["Dataset", "method", "family", "E", "serial cost", "wall time (s)", "selection loss", "selection metric",
              "test score (%)", "test loss (nats)"]
    rows = []
    for r in records:
        qc = r["selection"].get("qC_index")
        chosen = r["admission"].get("deployed_index")
        for c in family_representatives(r, "select_loss"):
            role = role_text(r, c).text
            tag = f" ({role})" if role else ""
            test_score = test_loss = None
            if c["index"] == chosen:
                test_score = r["final"]["models"]["qD"].get("metric_percent")
                test_loss = r["final"]["models"]["qD"].get("loss_nats")
            elif c["index"] == qc:
                test_score = r["final"]["models"]["qC"].get("metric_percent")
                test_loss = r["final"]["models"]["qC"].get("loss_nats")
            rows.append([Cell(r["dataset"]), Cell(c["route_label"] + tag),
                         Cell("feedback" if c["feedback"] else "open loop"), fmt(c.get("ensemble")),
                         num(c.get("cost"), 3), num(c.get("serial_wall_seconds"), 4), num(c.get("select_loss"), 5),
                         num(c.get("select_metric"), 4), num(test_score, 2), num(test_loss, 5)])
        for name in ("q0", "qN"):
            m = r["final"]["models"][name]
            rows.append([Cell(r["dataset"]), Cell({"q0": "feature-only predictor q_0", "qN": "full-depth plant q_N"}[name]),
                         Cell("reference"), fmt(m.get("ensemble")), Cell(None), Cell(None), Cell(None), Cell(None),
                         num(m.get("metric_percent"), 2), num(m.get("loss_nats"), 5)])
    note = ("Internal routes of the released runner, one row per route family (the comparator q_C, the selected rule "
            "and the deployed rule whenever they belong to the family, otherwise the configuration with the lowest "
            "selection-stage loss). Test scores are recorded only for the comparator and the deployed rule, which "
            "are the only candidates evaluated on D_test. Retrained external architectures (AMP, ADMP-GNN, GRAMA, "
            "Graph Mamba, Early-Exit GNN) are outside the released runner and have no rows here.")
    return header, rows, note


def table_S23(records):
    header = ["Dataset", "route/E", "controller cost C_qD", "comparator cost C_qC", "fallback probability",
              "assurance cost", "expected cost", "wall time q_D (s)", "wall time q_C (s)", "qualification (h)", "device"]
    rows = []
    for r in records:
        c = r["cost"]
        hours = None if c.get("qualification_seconds_total") is None else c["qualification_seconds_total"] / 3600.0
        rows.append([Cell(r["dataset"]), route_text(r["admission"].get("deployed")), num(c.get("C_qD"), 3),
                     num(c.get("C_qC"), 3), Cell(None), num(c.get("C_assure"), 3), num(c.get("C_expected"), 3),
                     num((c.get("wall_seconds") or {}).get("qD"), 4), num((c.get("wall_seconds") or {}).get("qC"), 4),
                     num(hours, 4), Cell(r["environment"].get("gpu") or r["environment"].get("device"))])
    note = ("Serial costs in nominal E = 1 executions from the candidate records; wall times are the measured serial "
            "execution of the candidate on the run's device after warm-up. The fallback probability column is not "
            "measured by the released runner (no serving-time input gate), so expected cost equals C_qD.")
    return header, rows, note


def table_S41(records):
    header = ["Dataset", "training (h)", "exploration logging (h)", "observer/critic/sequence fitting (h)",
              "nested selection (h)", "confirmation (h)", "admission (h)", "total qualification (h)", "device"]
    rows = []
    for r in records:
        c = r["cost"]
        q = c.get("qualification_seconds") or {}

        def h(x):
            return None if x is None else x / 3600.0
        rows.append([Cell(r["dataset"]), num(h(c.get("training_seconds")), 4), num(h(q.get("explore")), 4),
                     num(h(q.get("fit")), 4), num(h(q.get("select")), 4), num(h(q.get("confirm")), 4),
                     num(h(q.get("admit")), 4), num(h(c.get("qualification_seconds_total")), 4),
                     Cell(r["environment"].get("gpu") or r["environment"].get("device"))])
    note = ("Measured stage durations (stages/<stage>.json, seconds_this_attempt) converted to hours on the device "
            "recorded in environment.json. These are wall-clock hours of one run, not multiplied by any "
            "allocation factor.")
    return header, rows, note


def table_M1(records):
    header = ["Dataset", "profile", "data source", "split unit", "split sizes (fit/explore/select/confirm/admit/test)",
              "partition seed", "training seeds", "ensemble seeds q_C / q_D", "bootstrap seeds (select/admit/final)",
              "Python / Torch / CUDA", "device", "source identity", "config identity"]
    rows = []
    for r in records:
        d = r["data"]
        s = r["seeds"]
        sizes = d.get("split_sizes") or {}
        size_text = None
        if sizes:
            size_text = "/".join(str(sizes.get(k, "?")) for k in ("fit", "explore", "select", "confirm", "admit", "test"))
        env = r["environment"]
        ens = s.get("ensemble_seeds") or {}
        bs = s.get("bootstrap_seeds") or {}
        rows.append([Cell(r["dataset"]), Cell(r.get("profile")), Cell(d.get("source")), Cell(d.get("split_unit")),
                     Cell(size_text), fmt(s.get("partition_seed")), Cell(str(s.get("training_seeds"))),
                     Cell(f"{ens.get('qC')} / {ens.get('qD')}"),
                     Cell(f"{bs.get('selection')} / {bs.get('admission')} / {bs.get('final')}"),
                     Cell(f"{env.get('python')} / {env.get('torch')} / {env.get('cuda_runtime')}"),
                     Cell(env.get("gpu") or env.get("device")),
                     Cell((r["identities"].get("source_sha256") or "")[:16]),
                     Cell((r["identities"].get("config_sha256") or "")[:16])])
    note = ("Every value is read from the sealed run: prepared/metadata.json (split rule, sizes and role hashes), "
            "config.json (seeds), final/results_summary.json (ensemble members), the decision records (bootstrap "
            "seeds), environment.json and run_state.json. The full role membership arrays are prepared/splits.npz; "
            "manifests/ holds the extracted per-run manifests.")
    return header, rows, note


BUILDERS = {"T1": table_T1, "T2": table_T2, "S4": table_S4, "S12": table_S12, "S12c": table_S12c, "S13": table_S13,
            "S14": table_S14,
            "S15": table_S15, "S23": table_S23, "S41": table_S41, "M1": table_M1}
TITLES = {"T1": "Policy and deployment ledger (main Table 1)", "T2": "Gain and cost ledger (main Table 2)",
          "S4": "Target-regime qualification (Supplementary Table S4)",
          "S12": "Route-level candidate-selection ledger (Supplementary Table S12)",
          "S12c": "Selection rule of main Eq. (19) under three equivalence readings (Supplementary Table S12c)",
          "S13": "Independent-graph admission arithmetic (Supplementary Table S13)",
          "S14": "Held-out mechanism contrasts (Supplementary Table S14)",
          "S15": "One-row-per-method comparison, internal routes (Supplementary Table S15)",
          "S23": "Compute accounting (Supplementary Table S23)",
          "S41": "Qualification cost components (Supplementary Table S41)",
          "M1": "Split, seed and environment manifest"}


def provenance_line(records, label) -> str:
    runs = ", ".join(f"{r['dataset']}:{r['run_name']}" for r in records)
    sources = sorted({(r["identities"].get("source_sha256") or "?")[:12] for r in records})
    return (f"generated from {len(records)} run(s) [{label}]; runtime source {', '.join(sources)}; "
            f"runs: {runs}")


def render_tex(name, header, rows, note, prov, synthetic: bool) -> str:
    cols = "l" * len(header)
    lines = [f"% {TITLES[name]}", f"% {prov}", "% Do not edit: regenerate with `make tables`."]
    if synthetic:
        lines.append("% SYNTHETIC SOFTWARE EXAMPLES: these values are not benchmark results.")
    lines += [r"\begin{table}[t]", r"\centering", r"\small",
              r"\caption{" + TITLES[name] + (" (synthetic example runs)" if synthetic else "") + ". " + note + "}",
              f"\\label{{tab:{name}}}", r"\begin{tabular}{" + cols + "}", r"\hline",
              " & ".join(Cell(h).tex() for h in header) + r" \\", r"\hline"]
    for row in rows:
        lines.append(" & ".join(c.tex() for c in row) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def render_md(name, header, rows, note, prov, synthetic: bool) -> str:
    lines = [f"# {TITLES[name]}", "", f"_{prov}_", ""]
    if synthetic:
        lines += ["**Synthetic software examples: these values are not benchmark results.**", ""]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(c.md() for c in row) + " |")
    lines += ["", note, ""]
    return "\n".join(lines)


def render_csv(name, header, rows, note, prov, synthetic: bool) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    banner = " -- SYNTHETIC SOFTWARE EXAMPLES: these values are not benchmark results" if synthetic else ""
    writer.writerow([f"# {TITLES[name]}{banner}"])
    writer.writerow([f"# {prov}"])
    writer.writerow(header)
    for row in rows:
        writer.writerow([c.csv() for c in row])
    return out.getvalue()


def build_tables(records: list[dict], output_dir: Path, label: str, tables=TABLE_IDS) -> dict:
    """Write every table and return {table: {format: path}}."""
    records = sorted(records, key=lambda r: DATASET_ORDER.index(r["dataset"]) if r["dataset"] in DATASET_ORDER else 99)
    synthetic = any(not r.get("scientific_run") for r in records)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prov = provenance_line(records, label)
    written = {}
    for name in tables:
        header, rows, note = BUILDERS[name](records)
        paths = {}
        for suffix, text in (("tex", render_tex(name, header, rows, note, prov, synthetic)),
                             ("md", render_md(name, header, rows, note, prov, synthetic)),
                             ("csv", render_csv(name, header, rows, note, prov, synthetic))):
            path = output_dir / f"{name}.{suffix}"
            path.write_text(text, encoding="utf-8")
            paths[suffix] = path.name
        written[name] = paths
    index = {"label": label, "synthetic_examples": synthetic, "tables": written,
             "datasets": [r["dataset"] for r in records],
             "runs": {r["dataset"]: {"run_name": r["run_name"], "source_sha256": r["identities"].get("source_sha256"),
                                     "config_sha256": r["identities"].get("config_sha256"),
                                     "scientific_run": r.get("scientific_run"), "profile": r.get("profile")}
                      for r in records}}
    (output_dir / "INDEX.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return written
