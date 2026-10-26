"""Draw the paper's numerical figures from evidence records (Matplotlib, Agg backend).

F1  paired test effect against expected serial overhead         (main Fig. 4d)
F2  method map: official score against serial cost per dataset  (main Fig. 5a)
F3  selection geometry: simultaneous intervals per route family, −ε_R, the near-best band
    and the roles of q_C and the selected rule                   (main Fig. 2c)
F4  depth dynamics of the frozen plant at entry                  (Supplementary Fig. S14b)
F5  qualification time by stage                                  (Supplementary Table S41 as bars)
F6  held-out mechanism contrasts                                 (Supplementary Fig. S9)

Figures are written as PNG and SVG without embedded dates so that reruns on the same
platform give identical files; byte identity across platforms is not expected.
"""
from __future__ import annotations

import json
from pathlib import Path

from .evidence import DATASET_ORDER, decision_roles, family_representatives

FIGURE_IDS = ("F1", "F2", "F3", "F4", "F5", "F6")
TITLES = {"F1": "Paired test effect against expected serial overhead",
          "F2": "Method map: official score against serial cost",
          "F3": "Selection geometry: simultaneous intervals per route family",
          "F4": "Depth dynamics of the frozen plant at entry",
          "F5": "Qualification time by stage",
          "F6": "Held-out mechanism contrasts"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                         "figure.dpi": 110, "savefig.dpi": 200, "svg.hashsalt": "cross-morse"})
    return plt


def _save(fig, output_dir: Path, name: str) -> dict:
    paths = {}
    for suffix in ("png", "svg"):
        path = output_dir / f"{name}.{suffix}"
        meta = {"Date": None} if suffix == "svg" else {"Software": None}
        fig.savefig(path, bbox_inches="tight", metadata=meta)
        paths[suffix] = path.name
    return paths


def _banner(fig, records):
    if any(not r.get("scientific_run") for r in records):
        fig.text(0.5, 0.995, "SYNTHETIC SOFTWARE EXAMPLES — not benchmark results", ha="center", va="top",
                 fontsize=8, color="#b00020")


def fig_F1(plt, records, out):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for r in records:
        c = r["cost"]
        pl = r["final"].get("paired_loss") or {}
        if c.get("overhead_percent") is None or pl.get("mean") is None:
            continue
        err = None
        if pl.get("lower") is not None and pl.get("upper") is not None:
            err = [[pl["mean"] - pl["lower"]], [pl["upper"] - pl["mean"]]]
        ax.errorbar([c["overhead_percent"]], [pl["mean"]], yerr=err, fmt="o", capsize=2, label=r["dataset"])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("expected serial overhead over q_C (%)")
    ax.set_ylabel("paired test loss effect q_D − q_C (nats)")
    ax.set_title(TITLES["F1"])
    ax.legend(ncol=2, frameon=False)
    _banner(fig, records)
    return _save(fig, out, "F1")


def fig_F2(plt, records, out):
    n = len(records)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(1.9 * max(n, 1) + 0.6, 2.6), squeeze=False)
    for ax, r in zip(axes[0], records):
        qc, qd = r["selection"].get("qC") or {}, r["admission"].get("deployed") or {}
        nominal = next((c for c in r["candidates"] if c["route"] == "nominal" and c["ensemble"] == 1), None)
        points = [("q_N", nominal.get("cost") if nominal else None, r["final"]["models"]["qN"].get("metric_percent"), "s"),
                  ("q_C", qc.get("cost"), r["final"]["models"]["qC"].get("metric_percent"), "D"),
                  ("q_D", qd.get("cost"), r["final"]["models"]["qD"].get("metric_percent"), "o")]
        for label, x, y, marker in points:
            if x is None or y is None:
                continue
            ax.plot([x], [y], marker, ms=5, label=label)
        ax.set_title(r["dataset"], fontsize=8)
        ax.set_xlabel("serial cost (nominal = 1)")
    axes[0][0].set_ylabel("official test score (%)")
    axes[0][0].legend(frameon=False)
    fig.suptitle(TITLES["F2"], y=1.04)
    _banner(fig, records)
    fig.tight_layout()
    return _save(fig, out, "F2")


def fig_F3(plt, records, out):
    n = len(records)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(1.9 * max(n, 1) + 0.6, 2.8), squeeze=False)
    for ax, r in zip(axes[0], records):
        roles = decision_roles(r)
        labels, lows, highs, marked = [], [], [], []
        for c in family_representatives(r, "select_upper"):
            if c.get("select_lower") is None or c.get("select_upper") is None:
                continue
            role = roles.get(c["index"], [])
            labels.append(c["route_label"] + (f" ({', '.join(role)})" if role else ""))
            lows.append(c["select_lower"])
            highs.append(c["select_upper"])
            marked.append(bool(set(role) & {"selected", "deployed"}))
        y = list(range(len(labels)))
        for yi, lo, hi, sel in zip(y, lows, highs, marked):
            ax.plot([lo, hi], [yi, yi], "-", color="#d62728" if sel else "#1f77b4", lw=2)
        eps = r["selection"].get("epsilon_r")
        eps_eq = r["selection"].get("epsilon_eq")
        if eps is not None:
            ax.axvline(-eps, color="#d62728", lw=0.8, ls="--", label="−ε_R")
            # near-best band [U_min, U_min + ε_eq] of the registered reading, over the eligible configurations
            eligible = [c["select_upper"] for c in r["candidates"]
                        if c.get("select_upper") is not None and c["select_upper"] < -eps]
            if eligible and eps_eq is not None:
                u_min = min(eligible)
                ax.axvspan(u_min, u_min + eps_eq, color="#ff7f0e", alpha=0.25, lw=0, label="near-best band")
        ax.axvline(0, color="k", lw=0.5)
        ax.locator_params(axis="x", nbins=3)
        ax.tick_params(axis="x", labelsize=6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_title(r["dataset"], fontsize=8)
    axes[0][0].set_xlabel("[L_s, U_s] relative to nested q_C (nats)")
    axes[0][0].legend(frameon=False)
    fig.suptitle(TITLES["F3"], y=1.04)
    _banner(fig, records)
    fig.tight_layout()
    return _save(fig, out, "F3")


def fig_F4(plt, records, out):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for r in records:
        e = r["entry"]
        grid, loss = e.get("depth_grid"), e.get("depth_loss")
        if not grid or not loss:
            continue
        ax.plot(grid, loss, "o-", ms=3, label=r["dataset"])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("propagation depth")
    ax.set_ylabel("internal-validation loss (nats)")
    ax.set_title(TITLES["F4"])
    ax.legend(ncol=2, frameon=False)
    _banner(fig, records)
    return _save(fig, out, "F4")


def fig_F5(plt, records, out):
    stages = ("train", "explore", "fit", "select", "confirm", "admit")
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    bottoms = [0.0] * len(records)
    names = [r["dataset"] for r in records]
    for stage in stages:
        values = []
        for r in records:
            if stage == "train":
                v = r["cost"].get("training_seconds")
            else:
                v = (r["cost"].get("qualification_seconds") or {}).get(stage)
            values.append(0.0 if v is None else v / 3600.0)
        ax.bar(names, values, bottom=bottoms, label=stage)
        bottoms = [b + v for b, v in zip(bottoms, values)]
    ax.set_ylabel("wall-clock hours on the run's device")
    ax.set_title(TITLES["F5"])
    ax.tick_params(axis="x", rotation=30)
    ax.legend(ncol=3, frameon=False)
    _banner(fig, records)
    return _save(fig, out, "F5")


def fig_F6(plt, records, out):
    rows = []
    for r in records:
        for row in r["confirmation"].get("contrasts") or []:
            if row.get("mean") is not None:
                rows.append((f"{r['dataset']} / {row.get('kind')}", row["mean"], row.get("lower"), row.get("upper"),
                             row.get("practical_margin"), row.get("passes")))
    fig, ax = plt.subplots(figsize=(4.6, max(1.8, 0.35 * len(rows) + 1.2)))
    if not rows:
        ax.text(0.5, 0.5, "no confirmation contrasts were executed in these runs\n"
                          "(no feedback candidate entered Q_s)", ha="center", va="center", fontsize=8)
        ax.set_axis_off()
    else:
        for i, (label, mean, lo, hi, margin, passes) in enumerate(rows):
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [i, i], "-", color="#1f77b4", lw=2)
            ax.plot([mean], [i], "o", color="#1f77b4" if passes else "#7f7f7f")
            if margin is not None:
                ax.plot([-margin, margin], [i - 0.3, i - 0.3], "|", color="#2ca02c")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([row[0] for row in rows])
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("held-out contrast (nats); green ticks: practical margin")
    ax.set_title(TITLES["F6"])
    _banner(fig, records)
    return _save(fig, out, "F6")


BUILDERS = {"F1": fig_F1, "F2": fig_F2, "F3": fig_F3, "F4": fig_F4, "F5": fig_F5, "F6": fig_F6}


def build_figures(records: list[dict], output_dir: Path, label: str, figures=FIGURE_IDS) -> dict:
    plt = _plt()
    records = sorted(records, key=lambda r: DATASET_ORDER.index(r["dataset"]) if r["dataset"] in DATASET_ORDER else 99)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in figures:
        written[name] = BUILDERS[name](plt, records, output_dir)
        plt.close("all")
    index = {"label": label, "figures": written, "titles": {k: TITLES[k] for k in figures},
             "synthetic_examples": any(not r.get("scientific_run") for r in records),
             "datasets": [r["dataset"] for r in records]}
    (output_dir / "INDEX.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return written
