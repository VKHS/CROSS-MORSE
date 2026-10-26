"""The availability statement is derived from the tree and can never claim more than exists."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.availability import (BUILT_DISPLAYS, DATASETS, END, START, check, render, state,  # noqa: E402
                                  update_readme)


def minimal_tree(root: Path, *, official: list[str]) -> None:
    (root / "release").mkdir(parents=True)
    scope = {"repository_url": "https://example.org/repo", "code_archive_doi": None,
             "original_result_archives_included": len(official) == len(DATASETS),
             "one_command_original_tables_reproduction_claimed": len(official) == len(DATASETS),
             "full_official_dataset_runs_verified": sorted(official)}
    (root / "release" / "release_scope.json").write_text(json.dumps(scope))
    (root / "manifests" / "registered").mkdir(parents=True)
    (root / "manifests" / "registered" / "arxiv_full.json").write_text("{}")
    stems = {"ogbn-arxiv": "arxiv", "ogbn-products": "products", "ogbn-proteins": "proteins", "Roman-empire": "roman_empire",
             "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp", "Peptides-func": "peptides_func"}
    for d in DATASETS:
        target = root / "examples" / "runs" / stems[d] / "final"
        target.mkdir(parents=True)
        (target / "results_summary.json").write_text("{}")
    runs = {"schema": "cross-morse-runs-index-1",
            "runs": [{"stem": stems[d], "dataset": d, "path": stems[d], "kind": "run", "profile": "full",
                      "scientific_run": True, "verified": True, "deposit_doi": "10.5281/zenodo.1"} for d in official]}
    (root / "runs").mkdir()
    (root / "runs" / "index.json").write_text(json.dumps(runs))
    (root / "README.md").write_text(f"# fixture\n\n{START}\nstale\n{END}\n\ntail\n")
    # a minimal runtime so that the fixture tree has a source fingerprint of its own
    (root / "cross_morse").mkdir()
    (root / "cross_morse" / "__init__.py").write_text("__version__ = 'fixture'\n")
    (root / "run_cross_morse.py").write_text("print('fixture')\n")


class AvailabilityTests(unittest.TestCase):
    def test_statement_reports_missing_archives_when_none_are_registered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            minimal_tree(root, official=[])
            st = state(root)
            text = render(st)
            self.assertIn("0 of 7", text)
            self.assertIn("not yet deposited", text)
            self.assertNotIn("every table and figure", text)
            self.assertIn("runtime source identity " + st["source_sha256"][:12], text)
            self.assertNotIn("http://", text)

    def test_statement_claims_the_archives_only_when_all_seven_are_registered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            minimal_tree(root, official=list(DATASETS))
            text = render(state(root))
            self.assertIn("deposited at https://doi.org/10.5281/zenodo.1", text)
            self.assertIn("7 of 7", text)
            # even with every archive present, only the displays that paper/build produces are claimed
            self.assertNotIn("every table and figure", text)
            self.assertIn(BUILT_DISPLAYS, text)
            self.assertIn("KNOWN_DIFFERENCES.md", text)

    def test_named_displays_are_exactly_those_paper_build_produces(self):
        def expand(items: str) -> set[str]:
            out = set()
            for part in re.split(r",\s*|\s+and\s+", items):
                part = part.strip()
                m = re.fullmatch(r"(S?)(\d+)[–-]S?(\d+)", part)  # ranges never carry a letter suffix
                if m:
                    out.update(f"{m.group(1)}{n}" for n in range(int(m.group(2)), int(m.group(3)) + 1))
                elif part:
                    out.add(part)
            return out

        main_tables = expand(re.search(r"main Tables ([^ ]+) and Supplementary", BUILT_DISPLAYS).group(1))
        supp_tables = expand(re.search(r"Supplementary Tables (.+?), main Figs", BUILT_DISPLAYS).group(1))
        main_figs = expand(re.search(r"main Figs\. (.+?), and Supplementary Figs", BUILT_DISPLAYS).group(1))
        supp_figs = expand(re.search(r"Supplementary Figs\. (.+)$", BUILT_DISPLAYS).group(1))
        tables_src = (PROJECT / "paper" / "build" / "tables.py").read_text(encoding="utf-8")
        built_tables = set(re.findall(r'"\w+": "[^"]*\((?:main|Supplementary) Table (S?\d+[a-z]?)\)', tables_src))
        figures_doc = (PROJECT / "paper" / "build" / "figures.py").read_text(encoding="utf-8").split('"""')[1]
        built_figs = set(re.findall(r"\((?:main|Supplementary) Fig\. (S?\d+[a-z]?)\)", figures_doc))
        self.assertEqual(main_tables | supp_tables, built_tables)
        self.assertEqual(main_figs | supp_figs, built_figs)

    def test_check_fails_when_scope_flags_overclaim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            minimal_tree(root, official=["ogbn-arxiv"])
            scope = json.loads((root / "release" / "release_scope.json").read_text())
            scope["original_result_archives_included"] = True
            (root / "release" / "release_scope.json").write_text(json.dumps(scope))
            st = state(root)
            (root / "AVAILABILITY.md").write_text(render(st))
            update_readme(root, st)
            self.assertEqual(check(root, st), 1)

    def test_check_passes_after_regeneration_and_fails_when_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            minimal_tree(root, official=["ogbn-arxiv", "COCO-SP"])
            st = state(root)
            (root / "AVAILABILITY.md").write_text(render(st))
            self.assertTrue(update_readme(root, st))
            self.assertEqual(check(root, st), 0)
            readme = (root / "README.md").read_text()
            self.assertIn("2 of 7", readme)
            self.assertTrue(readme.endswith("tail\n"))
            (root / "AVAILABILITY.md").write_text("edited\n")
            self.assertEqual(check(root, st), 1)

    def test_repository_statement_is_current(self):
        st = state(PROJECT)
        self.assertEqual(check(PROJECT, st), 0)


if __name__ == "__main__":
    unittest.main()
