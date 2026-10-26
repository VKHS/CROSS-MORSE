#!/usr/bin/env python3
"""Validate the manuscript source index and render the paper-results overview.

The canonical numbers are in paper/reported_results.json. The manuscript and supplement
are not distributed with the repository; paper/sources.json identifies them by SHA-256,
byte size and page count, and maps every transcribed table group to its document, pages
and evidence type. This standard-library tool checks that index, verifies the PDF bytes
when an author supplies local copies (``--manuscript-pdf``/``--supplementary-pdf``), and
renders docs/PAPER_RESULTS.md and results/paper_summary.csv. It verifies transcription
provenance and deterministic rendering, not empirical replication or raw records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._paper_common import DATASETS, ROOT

KINDS = {"author_reported_measurements", "derived_arithmetic", "analytical_sensitivity", "mixed"}


def read_json(path: Path) -> dict:
    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def _verify_pdf_bytes(path: Path, source: dict, label: str) -> str:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        if not handle.read(5).startswith(b"%PDF-"):
            raise ValueError(f"Not a PDF: {label}")
        handle.seek(0)
        for block in iter(lambda: handle.read(2**20), b""):
            size += len(block)
            digest.update(block)
    if digest.hexdigest() != source["sha256"] or size != source["byte_size"]:
        raise ValueError(f"PDF content identity changed: {label}")
    return digest.hexdigest()


def validate_sources(root: Path, reported: dict, index: dict, local_pdfs: dict | None = None) -> dict:
    """Verify the source index and every table-group/page/evidence declaration.

    Each document is identified by SHA-256, byte size and page count. A document with a
    ``bundled_path`` inside the repository is verified from its bytes (paths are resolved
    before containment checks; symlinks escaping the repository are rejected). The
    manuscript is not distributed here, so its identity is checked as recorded unless an
    author passes a local copy in ``local_pdfs``, which is then verified byte for byte.
    A successful check does not verify the underlying experiments.
    """
    root = root.resolve()
    local_pdfs = local_pdfs or {}
    if index.get("schema_version") != "paper-sources-1":
        raise ValueError("Unsupported paper source index")
    documents = index.get("documents", {})
    if set(documents) != {"main", "supplementary"}:
        raise ValueError("Both manuscript PDF sources must be identified")
    verified = {}
    bytes_verified = {}
    for name, source in documents.items():
        if not re.fullmatch(r"[0-9a-f]{64}", source.get("sha256", "")):
            raise ValueError(f"Invalid SHA-256 for {name}")
        if not isinstance(source.get("pages"), int) or source["pages"] < 1:
            raise ValueError(f"Invalid page count for {name}")
        if not isinstance(source.get("byte_size"), int) or source["byte_size"] < 1:
            raise ValueError(f"Invalid byte size for {name}")
        if name in local_pdfs:
            verified[name] = _verify_pdf_bytes(Path(local_pdfs[name]), source, str(local_pdfs[name]))
            bytes_verified[name] = True
            continue
        if source.get("bundled_path") is None:
            verified[name] = source["sha256"]
            bytes_verified[name] = False
            continue
        relative = Path(source["bundled_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid relative PDF path: {relative}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing PDF or path outside repository: {relative}")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            if not handle.read(5).startswith(b"%PDF-"):
                raise ValueError(f"Not a PDF: {relative}")
            handle.seek(0)
            for block in iter(lambda: handle.read(2**20), b""):
                size += len(block)
                digest.update(block)
        if digest.hexdigest() != source["sha256"] or size != source["byte_size"]:
            raise ValueError(f"PDF content identity changed: {relative}")
        verified[name] = digest.hexdigest()
        bytes_verified[name] = True
    table_keys = {key for key in reported if not key.startswith("_")}
    groups = index.get("table_groups", {})
    if set(groups) != table_keys:
        raise ValueError(f"Table provenance coverage differs: {sorted(set(groups) ^ table_keys)}")
    for key, group in groups.items():
        if group.get("source") not in documents or group.get("evidence_kind") not in KINDS:
            raise ValueError(f"Invalid source/evidence kind: {key}")
        pages = group.get("pages", [])
        if not pages or any(type(p) is not int or p < 1 or p > documents[group["source"]]["pages"] for p in pages):
            raise ValueError(f"Invalid source page: {key}")
        if not group.get("label") or not group.get("note"):
            raise ValueError(f"Missing provenance interpretation: {key}")
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite number in reported registry")
        if isinstance(value, dict):
            for v in value.values():
                finite(v)
        elif isinstance(value, list):
            for v in value:
                finite(v)
    finite(reported)
    return {"pdf_sha256": verified, "pdf_bytes_verified": bytes_verified, "table_groups_checked": len(groups),
            "scope": "document identity, page provenance and rendering; not empirical replication"}


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |",
                       "|" + "|".join("---" for _ in header) + "|"] +
                      ["| " + " | ".join(("—" if c is None else str(c)).replace("|", "\\|") for c in row) + " |" for row in rows]) + "\n"


def score_rows(reported):
    comparison = reported["supplementary_table_S15_method_comparison"]
    for dataset in DATASETS:
        block = comparison[dataset]
        native = next(r for r in block["rows"] if r["method"] == "full-depth plant")
        comparator = next(r for r in block["rows"] if "(qC" in r["method"])
        deployed = next(r for r in block["rows"] if r["family"] == "CROSS-MORSE deployment")
        yield dataset, block, native, comparator, deployed


def signed(value, digits=5):
    return f"{value:+.{digits}f}".replace("-", "−")


def interval(value, digits=5):
    return "[" + ", ".join(signed(x, digits) for x in value) + "]"


def render_overview(reported: dict, index: dict) -> str:
    t1 = reported["main_table_1_policy_and_deployment_ledger"]
    t2 = reported["main_table_2_gain_and_cost_ledger"]
    claims = reported["abstract_and_section_claims"]
    max_gain = claims["official_metric_gain_max_points"]
    max_loss = abs(min(claims["paired_CE_change_range_nats"]))
    overhead_lo, overhead_hi = claims["expected_overhead_pct_range"]
    lines = [
        "# Paper results", "",
        "<!-- Generated by scripts/render_paper_results.py from paper/reported_results.json and paper/sources.json. -->", "",
        "CROSS-MORSE reports a different useful control regime across seven benchmarks: radius control on ogbn-arxiv and ogbn-proteins, "
        "receding-horizon control on Roman-empire, PascalVOC-SP and COCO-SP, a boundary radius result on Peptides-func, and return of the "
        f"open-loop comparator on ogbn-products. The reported paired loss improvement reaches **{max_loss:.5f} nats**, official-metric gains reach "
        f"**{max_gain:.2f} percentage points**, and the six feedback deployments have **{overhead_lo}–{overhead_hi}% expected serial-work overhead** over their selected comparators.", "",
        "These are **our reported measurements, transcribed from the submitted manuscript for comparison only**. The repository preserves their printed values and makes the numerical "
        "ledgers inspectable; the paper's run-derived tables and figures are built from completed run directories by `make paper` ([paper/README.md](../paper/README.md)). "
        "The separately bundled CPU lifecycle checks establish software behavior; they are not reruns of these benchmark experiments. "
        "Main abstract, p1; Table 1, p11; "
        "Table 2, p14.", "",
        "The reported admission decisions are **descriptive on all seven benchmarks**. The three independent-graph benchmarks reused validation "
        "for confirmation and admission, so their reported bounds do not instantiate the independent-graph theorem. The single-graph benchmarks "
        "use dependent cores or official splits. This scope accompanies the measurements throughout this page. "
        "Main pp7,11; Supplementary Note S26, p31.", "",
        "## 1. Paired test effects against the selected comparator", "",
        "The estimand is `loss(qD) − loss(qC)`; negative loss differences favor deployment. `qC` is the selected open-loop comparator, "
        "and `E` is the retained ensemble size. Brackets below are the paper's **95% paired test intervals**, not ranges over configurations. "
        "CE and mean multilabel BCE are reported in nats. Official gains are percentage points, including the explicit conversion from native AP for Peptides.", "",
    ]
    rows = []
    for d in DATASETS:
        r = t1[d]
        loss = "0 (identity)" if r["loss_effect_interval_nats"] is None else signed(r["loss_effect_nats"]) + " " + interval(r["loss_effect_interval_nats"])
        gain = "0 (identity)" if r["official_gain_interval_pp"] is None else signed(r["official_gain_pp"], 2) + " " + interval(r["official_gain_interval_pp"], 2)
        qc = r["qC_identity"] if d != "ogbn-proteins" else "learned stop / fixed sequence†"
        rows.append([d, qc, r["deployed_rule"], r["deployed_E"], r["loss_name"], loss, gain])
    lines += [table(["Dataset", "qC", "Reported qD", "E", "Loss", "Paired loss difference [95% CI]", "Metric gain [95% CI], pp"], rows),
        "Source: main Table 1, p11. Peptides' native result is **+0.0058 AP [0.0009, 0.0107]**, "
        "equivalent to +0.58 [0.09, 1.07] AP percentage points. †The paper names proteins' comparator in two ways; the unavailable full "
        "selection record would identify the single executed rule (Table S35, R18, p23).", "",
        "**Products is an informative comparator-return result.** Its supported improvement set is empty, so admission is not evaluated and "
        "`qD = qC` exactly. The zero effect is an execution identity, not an estimated improvement with a zero-width confidence interval. "
        "The state-destruction test has P = 0.42 against zero; equivalence within ±0.002 nats was not tested. "
        "Main Table 1; Table S14, p10.", "",
        "## 2. Absolute official scores and what is not tabulated", "",
        "`q0` denotes the upstream predictor, `qN` the nominal full-depth frozen plant, `qC` the selected comparator, and `qD` the deployment. "
        "Table S15 supplies the latter three official scores. Its full-depth plant row is **qN**; it is not a q0 measurement.", "",
    ]
    rows = []
    for d, block, native, comparator, deployed in score_rows(reported):
        digits = 3 if d == "Peptides-func" else 2
        rows.append([d, block["score_unit"], "not tabulated", f"{native['official_score']:.{digits}f}",
                     f"{comparator['official_score']:.{digits}f}", f"{deployed['official_score']:.{digits}f}"])
    lines += [table(["Dataset", "Official-score unit", "q0", "qN", "qC", "qD"], rows),
        "Source: Table S15, pp11–12. **Absolute test losses for q0/qN/qC/qD and q0's absolute "
        "official scores are not tabulated in these PDFs.** The paired effects above remain available. The reference plant-loss levels in Table S26 "
        "and depth-integrated losses in Table S16 are different quantities and are not substituted for missing absolute test losses.", "",
        "Peptides has an explicitly retained cross-table discrepancy: Table S26 prints 0.7200 − 0.7140 = 0.0060 AP, whereas main Table 1 reports "
        "+0.0058 AP. The 0.0002 difference exceeds the 0.0001 maximum rounding discrepancy for two four-decimal scores. We retain the main "
        "paired estimate and the separately reported absolute scores; unrounded per-seed observations are not provided. "
        "Table S26 caption, p17.", "",
        "## 3. Compute: expected execution, assurance, and tail model", "",
        "All costs below are in units of one nominal full-depth execution. Expected cost follows the reported controller/comparator branch ledger. "
        "Assurance cost is the separate configuration `Ccontroller + CqC`; it is not the attained maximum of the deployed rule. "
        "The 99th-percentile column is a resolvent-frequency **scenario model**, not an observed maximum or a hard cap.", "",
    ]
    rows = []
    for d in DATASETS:
        r = t2[d]
        tail = "—" if r["C_099"] is None else f"{r['C_099']:.1f} [{r['C_099_range'][0]:.1f}, {r['C_099_range'][1]:.1f}]"
        rate = "—" if r["lambda_star_mnats"] is None else f"{r['lambda_star_mnats']:.2f}"
        rows.append([d, f"{r['C_qC']:.1f}", f"{r['C_expected']:.2f}", f"{r['expected_overhead_pct']}%", f"{r['C_assure']:.1f}", tail, rate])
    lines += [table(["Dataset", "qC cost", "Expected cost", "Expected overhead", "Assurance cost", "Modelled C0.99 [scenarios]", "Derived λ*, millinats / added unit"], rows),
        "Sources: main Table 2, p14; Table S23, p16; "
        "Table S44, p28. Percentages retain the paper's rounding from its more precise ledger. "
        "The exchange rate is derived from paired loss improvement divided by added expected work; its uncertainty comes from the paired effect interval.", "",
        "The paper reports qualification increments of 1.4–2.3 GPU-hours. Table S41 separates component costs, while Table S35 retains an "
        "accounting-unit mismatch between the per-checkpoint, ensemble and search totals. GH200 timings are author measurements; current source "
        "ledgers and timings are measured afresh in their execution environment. Full-state sensing cannot explain the reported proteins/Pascal "
        "multipliers at every site: executed sketch width, sensing period and resolvent frequency are needed, and are not tabulated. "
        "Tables S6c, S35 R15/R30, S41; [compute guide](COMPUTE_BUDGET.md).", "",
        "## 4. What confirmation and admission establish", "",
        "The paper's confirmation map retains a route when its beneficial point estimate exceeds the practical margin and its zero-effect test "
        "passes Holm adjustment. Evidence against zero, evidence beyond the margin, and equivalence are distinct. Equality at a margin chooses "
        "the simpler route; a predecessor must remain in the selection-eligible set, otherwise the output is qC. "
        "Table S11, p8.", "",
    ]
    rows = []
    for r in reported["supplementary_table_S14_mechanism_contrasts_submitted_protocol"]:
        rows.append([r["contrast"], signed(r["effect"]), interval(r["interval"]), f"{r['n']} {r['statistical_unit']}", r["holm_P"], r["P_vs_margin"]])
    lines += [table(["Contrast", "Effect, nats", "95% interval", "Reported unit", "Holm P vs zero", "P vs practical margin"], rows),
        "Source: Table S14, p10, one reported Holm family of nine contrasts. State destruction changes "
        "controller observations, preserving the plant state. Arxiv/proteins radius contrasts pass the paper's point-estimate rule, while their "
        "one-sided tests do not establish improvement beyond 0.0015 nats. Peptides is similarly a boundary result. Its five-fold label count is "
        "not the fraction of test graphs receiving feedback.", "",
        "For the three independent-graph datasets, the reported admission arithmetic is:", "",
    ]
    rows = []
    for d, r in reported["supplementary_table_S13_independent_graph_admission"].items():
        rows.append([d, r["J"], signed(r["D_bar"]), f"{r['S2']:.6f}", f"{r['penalty']:.6f}", signed(r["U_a"], 6), r["epsilon_a"], "descriptive pass"])
    lines += [table(["Dataset", "Graphs J", "Clipped mean", "Variance", "Bound penalty", "Ua", "Required margin", "Reported status"], rows),
        "The clip is B = 0.10 and δa = 0.025; all three rows report a passing official-metric gate. The theorem's target is expected clipped "
        "paired loss, not unrestricted unclipped harm. These are moments for a fixed deployed ensemble. The main test intervals also resample "
        "optimization seeds, so their larger widths describe a different uncertainty population. Fixed-moment sensitivity at B = 0.20 makes "
        "Peptides fail the admission inequality; this calculation does not re-clip raw observations. "
        "Table S13, pp9–10; Note S24, p31.", "",
        "**Separated-data sensitivity is not a rerun.** Table S14b rescales standard errors at fixed effects to the registered confirmation sizes "
        "850/11,329/1,087. Its model gives label-retention probabilities 0.85, >0.99 and 0.36 for Pascal, COCO and Peptides. Under the displayed "
        "Peptides sensitivity outcome the radius label fails and one-step is outside Qs, so the map returns qC with exactly zero effect and no "
        "admission step. These hypothetical statistics neither replace the reported radius measurement nor repair earlier validation reuse. "
        "Table S14b, pp10–11; Note S26, p31.", "",
        "## 5. Counterfactuals and stronger comparators", "",
        "The paper reports initial-input schedule prediction (E3), distillation (E4), replanning without new observations (E5), and budget-aware "
        "execution (E6). The central arm outcomes below use **millinats relative to qC**. Their configuration ranges remain in the full registry; "
        "they are not confidence intervals.", "",
    ]
    rows = []
    for d, r in reported["supplementary_table_S46_budget_matched_frontier"].items():
        rows.append([d, f"{r['feedback_effect_mnats']:.1f}", f"{r['E3_predictor']:.1f}", f"{r['E4_distilled']:.1f}",
                     f"{r['E5_no_observation_replanning']:.1f}", f"{r['E6_qB_effect']:.1f}", f"{r['E6_overhead_pct']}%"])
    lines += [table(["Dataset", "Reported feedback", "E3 predictor", "E4 distilled", "E5 no observation", "E6 budget aware", "E6 overhead"], rows),
        "Source: Tables S46–S46b, p29. The common-ceiling protocol is specified, but arm-level paired "
        "intervals, adjusted P values, actual-work observations and budget mismatches are not tabulated. The point outcomes can be reported without "
        "inventing those missing statistics. Tables S42/S47 are analytical power and decision-boundary calculations, not observed rejection frequencies. "
        "These auxiliary arms and external architecture reruns are reported evidence; they are outside the packaged nine-route runner. "
        "[Claims ledger](../paper/claims_ledger.md); [implementation differences](../paper/KNOWN_DIFFERENCES.md).", "",
        "Additional reported comparisons include native-site versus inserted-site feedback (Table S45), matched reruns of external architectures "
        "(Tables S15/S15b), and the extended comparator ensemble grid E = 1,2,4,8,12,16,20 (Tables S26/S26b). "
        "The extra checkpoints in the extended grid are separately costed diagnostics. Noninteger cost-matched ensemble sizes and ensemble-spend "
        "crossovers are analytical sensitivity models, not executable ensembles or properties established for a learned predictor.", "",
        "## 6. Decision rules and qualifications retained in this registry", "",
        "| Issue | How the public record handles it | Source |", "|---|---|---|",
        "| Registered selector versus reported selections | Keep the reported MPC-RH deployments. Eq.19 with εeq=0.002 on upper limits selects radius on three representative rows; it matches four of seven. The full executed trace is needed to identify the actual rule. | S12c p9, main p11 |",
        "| Support statistics | Preserve current reported counts, propensities and qualified fractions; corrected per-cell ESS/max-weight summaries are explicitly missing. Withdrawn inconsistent dataset-level pairs are not recycled as evidence. | S7 p5, main pp17–18 |",
        "| Headroom | A label-informed oracle measures available opportunity. Its confidence limit is not an attained task gain or an executable label-free policy. | Main Proposition4 p7, S30 p20 |",
        "| Factorial evidence | Arxiv cell010 changes predictions at zero activity and is excluded from an additivity interpretation. Pascal's eight decision-capable cells use a different factor definition. Paired interactions need joint covariance. | S16–S16b pp12–13, S17d p38 |",
        "| Conditional guarantees | Max-t coverage and selective-sensing invariance are assumptions/audit targets. Checking formulas or software does not establish those events for all future inputs. | Main pp6–9, [method scope](METHOD_OVERVIEW.md) |", "",
        "## 7. Trace a value or attach the underlying run", "",
        "The single numeric registry is [paper/reported_results.json](../paper/reported_results.json). "
        "[paper/sources.json](../paper/sources.json) maps every numerical group to a PDF, exact SHA-256, page and evidence type. "
        "The broader [table rendering](../paper/REPORTED_RESULTS.md) and [CSV summary](../results/paper_summary.csv) are generated views. "
        "A blank CSV value means not tabulated, never an inferred zero.", "",
        "```bash", "python -B scripts/render_paper_results.py --check", "python -B scripts/render_reported_results.py --check",
        "python -B scripts/check_paper_constants.py", "```", "",
        "These commands verify the recorded source identities, provenance coverage, generated views and registered arithmetic. They do not rerun a benchmark. "
        "To validate a completed run, replay its sealed statistics, re-execute checkpoints, or attach a manuscript-run binding, follow "
        "[Results and replay](RESULTS.md) and the [reviewer guide](REVIEWER_GUIDE.md). Our paper-run prediction/checkpoint archives enter the repository "
        "through the run-archive contract ([RUN_ARCHIVES.md](RUN_ARCHIVES.md)); the binding templates do not assert identities for archives that are not yet ingested.", "",
        "### Manuscript sources", "",
        "The manuscript and the supplementary information are not distributed with the repository. They are "
        "identified below; `python -B scripts/render_paper_results.py --check --manuscript-pdf <file> "
        "--supplementary-pdf <file>` verifies a copy byte for byte.", "",
    ]
    for key, source in index["documents"].items():
        name = "Main manuscript" if key == "main" else "Supplementary information"
        lines += [f"- {name} — {source['pages']} pages, {source['byte_size']:,} bytes; SHA-256 `{source['sha256']}`."]
    return "\n".join(lines) + "\n"


def render_csv(reported: dict, index: dict) -> str:
    out = io.StringIO(newline="")
    columns = ["dataset", "score_unit", "q0_score", "qN_score", "qC_score", "qD_score", "loss_name",
               "paired_qD_minus_qC_loss_nats", "loss_CI95_lower", "loss_CI95_upper", "metric_gain_pp",
               "metric_CI95_lower_pp", "metric_CI95_upper_pp", "qC_identity", "qD_route", "ensemble_E",
               "expected_cost", "expected_overhead_percent", "assurance_cost", "modelled_C099",
               "admission_scope", "evidence_kind", "effect_source", "score_source", "main_pdf_sha256", "supplementary_pdf_sha256"]
    writer = csv.DictWriter(out, columns, lineterminator="\n")
    writer.writeheader()
    for d, block, native, comparator, deployed in score_rows(reported):
        r = reported["main_table_1_policy_and_deployment_ledger"][d]
        c = reported["main_table_2_gain_and_cost_ledger"][d]
        loss_ci = r["loss_effect_interval_nats"] or [None, None]
        metric_ci = r["official_gain_interval_pp"] or [None, None]
        writer.writerow(dict(dataset=d, score_unit=block["score_unit"], q0_score=None,
            qN_score=native["official_score"], qC_score=comparator["official_score"], qD_score=deployed["official_score"],
            loss_name=r["loss_name"], paired_qD_minus_qC_loss_nats=r["loss_effect_nats"],
            loss_CI95_lower=loss_ci[0], loss_CI95_upper=loss_ci[1], metric_gain_pp=r["official_gain_pp"],
            metric_CI95_lower_pp=metric_ci[0], metric_CI95_upper_pp=metric_ci[1], qC_identity=r["qC_identity"],
            qD_route=r["deployed_rule"], ensemble_E=r["deployed_E"], expected_cost=c["C_expected"],
            expected_overhead_percent=c["expected_overhead_pct"], assurance_cost=c["C_assure"], modelled_C099=c["C_099"],
            admission_scope="not evaluated; comparator return" if d == "ogbn-products" else "descriptive",
            evidence_kind="our reported measurements, transcribed for comparison only; cost model separately labelled",
            effect_source="main Table 1 p11; Table 2 p14", score_source="supplementary Table S15 pp11-12",
            main_pdf_sha256=index["documents"]["main"]["sha256"], supplementary_pdf_sha256=index["documents"]["supplementary"]["sha256"]))
    return out.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify source identities and current generated views without writing")
    parser.add_argument("--manuscript-pdf", type=Path, help="local copy of the main manuscript to verify byte for byte")
    parser.add_argument("--supplementary-pdf", type=Path, help="local copy of the supplement to verify byte for byte")
    args = parser.parse_args(argv)
    try:
        reported = read_json(ROOT / "paper/reported_results.json")
        index = read_json(ROOT / "paper/sources.json")
        local = {name: path for name, path in (("main", args.manuscript_pdf), ("supplementary", args.supplementary_pdf)) if path}
        result = validate_sources(ROOT, reported, index, local)
        views = {ROOT / "docs/PAPER_RESULTS.md": render_overview(reported, index),
                 ROOT / "results/paper_summary.csv": render_csv(reported, index)}
        for path, content in views.items():
            if args.check:
                if not path.is_file() or path.read_text(encoding="utf-8") != content:
                    raise ValueError(f"Stale generated view: {path.relative_to(ROOT)}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
        checked = sum(result["pdf_bytes_verified"].values())
        identity = (f"{checked} manuscript PDF(s) verified byte for byte" if checked
                    else "manuscript identities checked as recorded (PDFs not distributed)")
        print(f"{identity}; {result['table_groups_checked']} table groups; "
              f"generated views {'current' if args.check else 'written'}. No benchmark execution performed.")
        return 0
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
