#!/usr/bin/env python3
"""Replay seven complete synthetic runs and regenerate their tables and figures.

Numerical observations, rather than the paper registry, are the inputs. No
training, network access, checkpoint deserialization or PyTorch import occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

SOURCE = Path(__file__).resolve().parents[1]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from cross_morse.review import EvidenceError, NumericEvidence, replay_results
from cross_morse.provenance import current_source_sha256

TASKS = (("arxiv", "ogbn-arxiv"), ("proteins", "ogbn-proteins"),
         ("products", "ogbn-products"), ("roman_empire", "Roman-empire"),
         ("pascalvoc_sp", "PascalVOC-SP"), ("coco_sp", "COCO-SP"),
         ("peptides_func", "Peptides-func"))
PREDICTORS = ("q0", "qN", "qC", "qD")
SCOPE = "Synthetic software examples; not official benchmark scores or manuscript reproduction."
MARKER = "reproduction_manifest.json"
SCHEMA = "cross-morse-example-reproduction-1"


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def collect(runs_root):
    """Verify every shipped artifact, then recompute the complete decisions."""
    root = Path(runs_root).resolve()
    expected_source = current_source_sha256()
    entries = []
    for stem, dataset in TASKS:
        run = root / stem
        if run.is_symlink():
            raise EvidenceError(f"Example directories cannot be symlinks: {stem}")
        e = NumericEvidence(run)
        c = e.json("config.json")
        if (c.get("dataset") != "synthetic" or c.get("scientific_run") is not False
                or c.get("profile") != "smoke" or c.get("benchmark_dataset") != dataset
                or c.get("data", {}).get("download") is not False):
            raise EvidenceError(f"Expected the synthetic smoke fixture for {dataset}")
        if e.identities["source_sha256"] != expected_source:
            raise EvidenceError(f"Runtime source differs from the example: {stem}")
        # Includes checkpoint bytes, optimizer states, trajectories and splits.
        # Logical tensor state is checked only by evaluate_checkpoints.py.
        for relative in e.sealed:
            e.file(relative)
        result = replay_results(run)
        if result["status"] != "PASS" or result["scientific_run"]:
            raise EvidenceError(f"Unexpected replay result for {dataset}")
        meta = e.json("prepared/metadata.json")
        final = e.json("final/results_summary.json")
        entries.append({"stem": stem, "benchmark_dataset": dataset,
            "identities": dict(e.identities), "dataset": "synthetic", "scientific_run": False,
            "training_seeds": c["training"]["seeds"], "partition_seed": c["seed"],
            "num_nodes": meta["num_nodes"], "num_graphs": meta["num_graphs"],
            "prediction_level": meta["prediction_level"], "output_dimensions": meta["num_outputs"],
            "comparator_route": final["comparator"]["route"], "deployed_route": final["deployed"]["route"],
            "sealed_files_verified": len(e.sealed), "sealed_bytes_verified": sum(i["bytes"] for i in e.sealed.values()),
            "replay": result})
    return {"schema_version": SCHEMA, "scope": SCOPE,
            "runtime_source_sha256": expected_source, "runs": entries}


def table_rows(report):
    rows = []
    for run in report["runs"]:
        r = run["replay"]
        effect = r["paired_qD_minus_qC_loss"]
        metric_effect = r["paired_qD_minus_qC_metric_pp"]
        for name in PREDICTORS:
            model = r["models"][name]
            rows.append({"task_configuration": run["benchmark_dataset"], "data_kind": "synthetic",
                "predictor": name, "prediction_level": run["prediction_level"],
                "metric_name": r["task"]["metric"], "metric_fraction": model["metric"],
                "loss_name": model["loss_name"], "loss_nats": model["loss_nats"],
                "test_samples": model["test_samples"], "test_units": model["test_units"],
                "admitted": r["admitted"], "deployed_route": run["deployed_route"],
                "paired_qD_minus_qC_loss_mean": effect["mean"],
                "paired_qD_minus_qC_loss_lower": effect["lower"],
                "paired_qD_minus_qC_loss_upper": effect["upper"],
                "paired_qD_minus_qC_metric_pp": metric_effect["mean"],
                "config_sha256": run["identities"]["config_sha256"],
                "unit_sha256": run["identities"]["unit_sha256"],
                "source_sha256": run["identities"]["source_sha256"]})
    return rows


def rendered_tables(report):
    rows = table_rows(report)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    md = ["# Recomputed example results", "", SCOPE, "",
          "Every value below is recomputed from the sealed predictions and labels. "
          "Selection, confirmation and admission are also replayed. Metric scores "
          "are fractions in [0, 1], including AP; loss is in nats.", "",
          "| Task configuration | Metric | q0 | qN | qC | qD | qD − qC loss | Admission |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    for run in report["runs"]:
        r = run["replay"]
        values = " | ".join(f"{r['models'][name]['metric']:.6f}" for name in PREDICTORS)
        md.append(f"| {run['benchmark_dataset']} | {r['task']['metric']} | {values} | "
                  f"{r['paired_qD_minus_qC_loss']['mean']:+.8f} | {'admitted' if r['admitted'] else 'comparator retained'} |")
    dimensions = {run["output_dimensions"] for run in report["runs"]}
    dimension_text = (f"Generated fixtures have {next(iter(dimensions))} classes or labels and reduced training budgets. "
                      if len(dimensions) == 1 else "Generated fixtures have the output dimensions recorded per run and reduced training budgets. ")
    md += ["", "## How to interpret the examples", "",
           dimension_text +
           "They exercise the seven task paths, split isolation, numerical review and "
           "checkpoint replay. They do not estimate performance on the named official datasets.", "",
           "The paper's measurements are rendered separately by `make paper`. "
           "This report does not read those measurements or compare synthetic scores with them.", "",
           "`metrics.csv` contains all 28 predictor rows, losses, paired effects and source/configuration/unit identities. "
           "`replay.json` retains every recomputed decision and verified numeric input. "
           "The original probability arrays, splits, checkpoint files and trajectories remain in the input run directories.", ""]
    return {"metrics.csv": stream.getvalue(), "metrics.md": "\n".join(md), "replay.json": json_text(report)}


def draw_figures(report, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "svg.hashsalt": SCHEMA, "axes.spines.top": False, "axes.spines.right": False})
    colors = ("#a6b3c3", "#607990", "#19334e", "#14848b")
    fig, axes = plt.subplots(2, 4, figsize=(14.4, 7.4), sharey=True)
    for ax, run in zip(axes.flat, report["runs"]):
        r = run["replay"]
        values = [r["models"][name]["metric"] for name in PREDICTORS]
        bars = ax.bar(PREDICTORS, values, color=colors, width=0.65, zorder=3)
        ax.set_title(run["benchmark_dataset"], loc="left", fontsize=11, weight="bold", color="#19334e")
        ax.set_ylim(0, 1.10)
        ax.set_yticks((0, .25, .5, .75, 1.0))
        ax.grid(axis="y", alpha=.18, zorder=0)
        ax.set_xlabel(r["task"]["metric"] + " · synthetic test data", fontsize=9)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + .025, f"{value:.3f}", ha="center", fontsize=8)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, .86, "Reading this figure", fontsize=12, weight="bold", color="#19334e")
    axes.flat[-1].text(0, .71, "q0  Propagation-free predictor\nqN  Full-depth nominal predictor\nqC  Selected open-loop comparator\nqD  Locked deployed rule", va="top", fontsize=10, linespacing=1.7)
    dimensions = {run["output_dimensions"] for run in report["runs"]}
    dimension_text = (f"{next(iter(dimensions))} generated classes or labels." if len(dimensions) == 1
                      else "Output dimensions vary by task.")
    axes.flat[-1].text(0, .25, dimension_text + "\nScores demonstrate executable paths;\nthey are not benchmark estimates.", va="top", fontsize=9, color="#506176", linespacing=1.5)
    fig.suptitle("Recomputed scores from the included software examples", x=.055, ha="left", fontsize=19, weight="bold", color="#19334e", y=.98)
    fig.text(.055, .925, "Seven task configurations · saved predictions and labels · complete numerical replay", color="#506176", fontsize=11)
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_ylabel("Task metric (fraction)")
    fig.subplots_adjust(left=.06, right=.98, top=.84, bottom=.11, hspace=.48, wspace=.28)
    fig.text(.055, .025, "Source: sealed synthetic run artifacts. Official benchmark measurements are presented separately in the paper registry.", fontsize=9, color="#506176")
    paths = []
    for extension in ("svg", "png"):
        path = Path(directory) / f"example_scores.{extension}"
        metadata = {"Date": None} if extension == "svg" else {"Software": "CROSS-MORSE reproduce_examples.py"}
        fig.savefig(path, dpi=170, metadata=metadata)
        paths.append(path.name)
    plt.close(fig)
    return paths


def safe_destination(output, runs_root, *, readonly=False):
    raw = Path(output).absolute()
    if any(p.is_symlink() for p in (raw, *raw.parents)):
        raise EvidenceError("Report destination cannot contain symlinks")
    out = raw.resolve()
    runs = Path(runs_root).resolve()
    if (not readonly and out.is_relative_to(SOURCE)) or out.is_relative_to(runs) or runs.is_relative_to(out) or SOURCE.is_relative_to(out):
        raise EvidenceError("Report destination must be separate from the source and input run trees")
    if out.exists() and not out.is_dir():
        raise EvidenceError("Report destination must be a directory")
    return out


def write_reports(report, out, *, figures=False, check=False):
    tables = rendered_tables(report)
    if check:
        bad = [name for name, text in tables.items() if not (out/name).is_file() or (out/name).read_text() != text]
        if bad:
            raise EvidenceError("Generated numerical views differ or are missing: " + ", ".join(bad))
        return sorted(tables)
    # Repeated make invocations may refresh this tool's own output only.
    reserved = {*tables, MARKER, "example_scores.svg", "example_scores.png"}
    collisions = [name for name in reserved if (out/name).exists() or (out/name).is_symlink()]
    if collisions:
        if (out/MARKER).is_symlink():
            raise EvidenceError("The existing report manifest cannot be a symlink")
        try:
            previous = json.loads((out/MARKER).read_text())
        except (OSError, ValueError) as exc:
            raise EvidenceError("Existing report files have no generator manifest; choose a new output directory") from exc
        if previous.get("schema_version") != SCHEMA or not isinstance(previous.get("files"), list):
            raise EvidenceError("The existing report manifest belongs to another format")
        known = {item["path"]: item for item in previous["files"]}
        for name in collisions:
            p = out/name
            if name == MARKER:
                continue
            if p.is_symlink() or not p.is_file() or name not in known or sha(p) != known[name]["sha256"]:
                raise EvidenceError(f"An existing report was changed outside this generator: {name}")
        if not figures and any((out/name).exists() for name in ("example_scores.svg", "example_scores.png")):
            if previous.get("runtime_source_sha256") != report["runtime_source_sha256"] or (out/"replay.json").read_text() != tables["replay.json"]:
                raise EvidenceError("The inputs changed; use --figures to refresh the existing figures too")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cross-morse-reproduce-", dir=out.parent) as temp:
        staging = Path(temp)
        for name, text in tables.items():
            (staging/name).write_text(text)
        names = list(tables)
        if figures:
            names += draw_figures(report, staging)
        elif collisions:
            # A refresh without plotting preserves valid generated figures.
            for name in ("example_scores.svg", "example_scores.png"):
                if (out/name).is_file():
                    (staging/name).write_bytes((out/name).read_bytes())
                    names.append(name)
        manifest = {"schema_version": SCHEMA, "scope": SCOPE,
            "runtime_source_sha256": report["runtime_source_sha256"],
            "files": [{"path": name, "bytes": (staging/name).stat().st_size, "sha256": sha(staging/name)} for name in sorted(names)]}
        (staging/MARKER).write_text(json_text(manifest))
        out.mkdir(parents=True, exist_ok=True)
        for name in [*names, MARKER]:
            os.replace(staging/name, out/name)
    return sorted(names)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=SOURCE/"examples/runs")
    parser.add_argument("--output-dir", type=Path, required=True, help="generated reports outside source and runs")
    parser.add_argument("--figures", action="store_true", help="also draw SVG/PNG with Matplotlib")
    parser.add_argument("--check", action="store_true", help="replay and compare existing numerical tables without writing")
    args = parser.parse_args(argv)
    try:
        out = safe_destination(args.output_dir, args.runs_root, readonly=args.check)
        result = collect(args.runs_root)
        files = write_reports(result, out, figures=args.figures, check=args.check)
        print(json_text({"status": "PASS", "scope": SCOPE, "complete_runs": len(result["runs"]),
                         "predictors": 4 * len(result["runs"]), "output_directory": str(out), "files": files}))
        return 0
    except (OSError, ValueError, KeyError, ImportError, TypeError) as exc:
        print(json_text({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
