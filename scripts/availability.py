#!/usr/bin/env python3
"""Render and check the Data and Code Availability statement from the repository's actual state.

    python -B scripts/availability.py            # rewrite AVAILABILITY.md and the README block
    python -B scripts/availability.py --check    # fail if either is stale or claims more than exists

The statement is derived from these sources of fact, never typed by hand:

* ``release/release_scope.json`` — repository URL and code-archive DOI;
* the runtime source fingerprint of the tree (``make fingerprint``), which identifies the code without a
  release version: every run seals the same identity into its ``run_state.json``;
* ``runs/index.json`` — which official-data run archives have been verified and ingested, with
  their deposit DOIs (written only by ``scripts/ingest_run.py``);
* ``records/index.json`` — which measurement-record sets outside the eight-stage run are deposited and
  traced to their raw logs (written only by ``scripts/records.py index``);
* the tree itself — the seven synthetic example runs, the manifests and the generated tables.

Both reviewers found that an earlier availability statement promised artifacts the repository
did not hold.  This script makes that impossible to repeat: the manuscript statement is copied
from ``AVAILABILITY.md``, and CI fails when the file no longer matches the tree.  The statement
names exactly the tables and figures that ``paper/build`` produces (``BUILT_DISPLAYS``); the
others depend on experiments outside the release and are listed in ``paper/KNOWN_DIFFERENCES.md``.

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))
from scripts._paper_common import runtime_source_fingerprint  # noqa: E402
from scripts.records import summary_state as records_state  # noqa: E402
DATASETS = ["ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func"]
STEMS = {"ogbn-arxiv": "arxiv", "ogbn-products": "products", "ogbn-proteins": "proteins", "Roman-empire": "roman_empire",
         "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp", "Peptides-func": "peptides_func"}
START, END = "<!-- availability:start -->", "<!-- availability:end -->"
# The manuscript displays that paper/build reproduces from run directories (tables T1, T2, S4, S12, S12c,
# S13-S15, S23, S41; figures F1 = Fig. 4d, F2 = Fig. 5a, F3 = Fig. 2c, F4 = Fig. S14b, F6 = Fig. S9).
# tests/test_availability.py keeps this text in step with paper/build.
BUILT_DISPLAYS = ("the run-derived columns of main Tables 1–2 and Supplementary Tables S4, S12, S12c, S13–S15, S23 and S41, "
                  "main Figs. 2c, 4d and 5a, and Supplementary Figs. S9 and S14b")
NOT_BUILT = ("The tables of the experiments outside the eight-stage run are printed from their deposited measurement "
             "records (data statement); their figure panels and the analytical tail-cost model of Supplementary Table "
             "S44 have no builder here, and `paper/KNOWN_DIFFERENCES.md` lists every difference between this code and "
             "the manuscript.")


def _json(path: Path, default=None):
    if not path.is_file():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def state(root: Path = ROOT) -> dict:
    scope = _json(root / "release" / "release_scope.json")
    index = _json(root / "runs" / "index.json", {"schema": "cross-morse-runs-index-1", "runs": []})
    official = {}
    for entry in index.get("runs", []):
        if entry.get("verified") and entry.get("scientific_run") and entry.get("profile") == "full":
            official[entry["dataset"]] = entry
    examples = [d for d in DATASETS if (root / "examples" / "runs" / STEMS[d] / "final" / "results_summary.json").is_file()]
    registered = sorted(p.name for p in (root / "manifests" / "registered").glob("*_full.json"))
    registered_pilot = sorted(p.name for p in (root / "manifests" / "registered").glob("*_pilot.json"))
    records = records_state(root / "records") if (root / "records" / "specs.json").is_file() else None
    tables_examples = (root / "paper" / "generated" / "examples" / "tables" / "T1.tex").is_file()
    tables_official = (root / "paper" / "generated" / "official" / "tables" / "T1.tex").is_file()
    return {"scope": scope, "official": official, "examples": examples, "registered_manifests": registered,
            "source_sha256": runtime_source_fingerprint(root),
            "registered_pilot_manifests": registered_pilot, "records": records,
            "tables_examples": tables_examples, "tables_official": tables_official,
            "missing_official": [d for d in DATASETS if d not in official]}


def _table_list(labels: list[str]) -> str:
    """'Supplementary Table S15', ... -> 'Supplementary Tables S6c, S15, ... and S46b' in numeric order."""
    import re as _re
    names = sorted({label.split()[-1] for label in labels},
                   key=lambda s: (int(_re.search(r"\d+", s).group()), s))
    return "Supplementary Tables " + (", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0])


def records_sentence(st: dict) -> str:
    records = st.get("records")
    if not records:
        return ""
    labels = [label for tables in records["tables"].values() for label in tables]
    k, n = len(records["verified"]), len(records["defined"])
    which = "all record sets" if k == n else f"{k} of {n} record sets ({', '.join(records['verified']) or 'none'})"
    return (f" Measurement records of the experiments outside the eight-stage run ({_table_list(labels)}) are deposited "
            f"under `records/` together with the raw logs every value is read from, for {which}; `make records-tables` "
            "prints those tables from them.")


def render(st: dict) -> str:
    scope = st["scope"]
    url = scope.get("repository_url") or "https://github.com/VKHS/CROSS-MORSE"
    code_doi = scope.get("code_archive_doi")
    where = (f"the CROSS-MORSE GitHub repository ({url}; runtime source identity {st['source_sha256'][:12]}…, "
             "printed by `make fingerprint` and sealed into every run record")
    where += f"; code archived at https://doi.org/{code_doi})" if code_doi else ")"
    n = len(st["official"])
    dois = sorted({e.get("deposit_doi") for e in st["official"].values() if e.get("deposit_doi")})
    lines = ["# Data and code availability", "",
             "This file is generated by `scripts/availability.py` from `release/release_scope.json`, "
             "`runs/index.json` and the repository tree. Copy the two statements below into the manuscript; "
             "`make availability-check` fails when they no longer describe the repository.", "",
             "## Statement for the manuscript", "",
             "**Code availability.** The complete implementation of CROSS-MORSE (the `cross_morse` package: frozen "
             "plants, transport adapter, action registry, sensing, observer, consequence models, the nine policy "
             "families, selection, confirmation and admission statistics, and the sealed eight-stage lifecycle), the "
             "dataset adapters for the seven benchmarks, one entry point and three configurations per dataset, the "
             "Slurm launchers used on the Arrhenius cluster together with its site profile, the environment "
             "specifications, the test suite, the scripts that build, from completed run directories, "
             f"{BUILT_DISPLAYS} (`make tables`, `make figures`), and the scripts that verify and tabulate deposited "
             f"measurement records (`make records-check`, `make records-tables`) are available in {where}. {NOT_BUILT}", ""]
    data = ("**Data availability.** All seven datasets are official public releases (OGB, the heterophily benchmark "
            "collection and LRGB) obtained from their providers through the repository's adapters; no dataset is "
            "redistributed. The registered split rule, seeds, fold structure and margins of every configuration are "
            "in `manifests/registered/`, and the role membership hashes, seeds and recorded environment of every "
            "completed run are extracted into `manifests/<label>/` by `make manifests`. ")
    if n == len(DATASETS):
        data += ("The run archives behind the reported results — prepared data and split indices, frozen "
                 "checkpoints, propensity-logged exploration trajectories, candidate and final predictions, selection, "
                 "confirmation and admission records, stage manifests and environment records — are deposited at "
                 + ", ".join(f"https://doi.org/{d}" for d in dois) +
                 " and registered in `runs/index.json`; `make tables figures` rebuilds the tables and figures "
                 "named in the code statement from them." + records_sentence(st))
    else:
        done = ", ".join(sorted(st["official"])) or "none"
        data += (f"Verified official-data run archives are registered for {n} of {len(DATASETS)} datasets ({done}); "
                 f"the archives for {', '.join(st['missing_official'])} are not yet deposited. Until they are, the "
                 "numerical results in the manuscript are the authors' reported measurements, transcribed with page "
                 "provenance in `paper/reported_results.json` for comparison only. The repository contains seven "
                 "complete synthetic end-to-end example runs with the same artifact types (checkpoints, predictions, "
                 "trajectories, statistical outputs and stage manifests) under `examples/runs/`, from which "
                 "`make reproduce` recomputes every decision and `make tables figures` builds the tables and figures "
                 "named in the code statement; the same commands build them from the official archives once those "
                 "are ingested with `scripts/ingest_run.py`." + records_sentence(st))
    lines += [data, "", "## What the repository holds today", "",
              "| Item the reviewers asked for | Present | Where |", "|---|---|---|",
              "| Experiment code | yes | `cross_morse/`, `run_cross_morse*.py`, `configs/` |",
              "| Data | adapters and provider sources; no redistribution | `cross_morse/datasets.py`, `data/sources.json` |",
              f"| Split/seed manifests | registered: {len(st['registered_manifests'])} full-profile and "
              f"{len(st.get('registered_pilot_manifests', []))} pilot-profile files; per run: `manifests/examples/` "
              f"({len(st['examples'])} runs){', `manifests/official/`' if n else ''} | `manifests/` |",
              "| Environment specification | CPU lock of the validated environment; per-run `environment.json`; Arrhenius GPU profile and capture script | `environment/` |",
              f"| Checkpoints, predictions, trajectories, statistical outputs (synthetic examples) | {len(st['examples'])} of 7 runs | `examples/runs/<stem>/` |",
              f"| Checkpoints, predictions, trajectories, statistical outputs (official data, paper results) | {n} of 7 datasets | `runs/`, `runs/index.json` |",
              *([f"| Measurement records outside the eight-stage run ({_table_list([l for t in st['records']['tables'].values() for l in t])}) "
                 f"| {len(st['records']['verified'])} of {len(st['records']['defined'])} record sets | `records/`, `records/index.json` |"]
                if st.get("records") else []),
              f"| Table and figure generators | yes; example build committed{'; official build committed' if st['tables_official'] else ''} | `paper/build/`, `scripts/build_paper.py`, `paper/generated/` |",
              "| Makefile that regenerates tables and figures | yes: `make tables`, `make figures`, `make paper-from-runs` | `Makefile` |",
              "| Slurm launchers with the Arrhenius site profile | yes | `run_cross_morse*.sh`, `configs/arrhenius-site.env.example`, `docs/ARRHENIUS.md` |",
              "", "Generated from the repository state; do not edit by hand.", ""]
    return "\n".join(lines)


def readme_block(st: dict) -> str:
    text = render(st)
    start = text.index("**Code availability.**")
    end = text.index("## What the repository holds today")
    return START + "\n" + text[start:end].rstrip() + "\n" + END


def update_readme(root: Path, st: dict) -> bool:
    path = root / "README.md"
    content = path.read_text(encoding="utf-8")
    if START not in content or END not in content:
        return False
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    new = pattern.sub(lambda _m: readme_block(st), content)
    if new != content:
        path.write_text(new, encoding="utf-8")
    return True


def check(root: Path, st: dict) -> int:
    problems = []
    expected = render(st)
    actual = (root / "AVAILABILITY.md").read_text(encoding="utf-8") if (root / "AVAILABILITY.md").is_file() else ""
    if actual != expected:
        problems.append("AVAILABILITY.md is stale; run `make availability`")
    readme = (root / "README.md").read_text(encoding="utf-8")
    if START in readme and END in readme:
        block = readme[readme.index(START): readme.index(END) + len(END)]
        if block != readme_block(st):
            problems.append("README.md availability block is stale; run `make availability`")
    else:
        problems.append("README.md lacks the availability markers")
    scope = st["scope"]
    all_official = len(st["official"]) == len(DATASETS)
    if bool(scope.get("original_result_archives_included")) != all_official:
        problems.append("release_scope.json original_result_archives_included disagrees with runs/index.json")
    if bool(scope.get("one_command_original_tables_reproduction_claimed")) != all_official:
        problems.append("release_scope.json one_command_original_tables_reproduction_claimed disagrees with runs/index.json")
    if scope.get("full_official_dataset_runs_verified") != sorted(st["official"]):
        problems.append("release_scope.json full_official_dataset_runs_verified disagrees with runs/index.json")
    if scope.get("validated_runtime_source_sha256") not in (None, st["source_sha256"]):
        problems.append("release_scope.json validated_runtime_source_sha256 is not the fingerprint of this tree")
    for p in problems:
        print("CHECK FAILED:", p, file=sys.stderr)
    if problems:
        return 1
    print(f"availability statement matches the tree ({len(st['official'])}/7 official run archives registered)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        st = state(args.root)
        if args.check:
            return check(args.root, st)
        (args.root / "AVAILABILITY.md").write_text(render(st), encoding="utf-8")
        if not update_readme(args.root, st):
            print("warning: README.md has no availability markers; only AVAILABILITY.md was written", file=sys.stderr)
        print(f"wrote AVAILABILITY.md ({len(st['official'])}/7 official run archives registered)")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
