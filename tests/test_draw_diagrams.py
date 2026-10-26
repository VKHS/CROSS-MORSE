"""Readable, reproducible diagrams retain the method's decision boundaries."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import re
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zlib

from scripts import draw_diagrams as draw

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
SCRIPT = ROOT / "scripts" / "draw_diagrams.py"
NS = {"svg": "http://www.w3.org/2000/svg"}


def parsed(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def visible_text(svg: str) -> list[str]:
    return ["".join(element.itertext()) for element in parsed(svg).findall(".//svg:text", NS)]


def contract(svg: str) -> dict:
    return json.loads(parsed(svg).find("svg:metadata", NS).text)


def edge_set(svg: str) -> set[tuple[str, str, str]]:
    return {(element.attrib["data-source"], element.attrib["data-target"], element.attrib["data-role"])
            for element in parsed(svg).findall(".//svg:path", NS) if "data-source" in element.attrib}


def fixture_png(width: int, height: int) -> bytes:
    """A tiny real PNG fixture keeps PNG checks independent of installed renderers."""
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    pixels = (b"\x00" + b"\xff\xff\xff" * width) * height
    return (draw.PNG_SIGNATURE + draw.png_chunk(b"IHDR", header)
            + draw.png_chunk(b"IDAT", zlib.compress(pixels)) + draw.png_chunk(b"IEND", b""))


class DrawDiagramsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rendered = draw.build(sorted(draw.FIGURES))
        cls.constants, cls.results = draw.load_registry()

    def test_figures_are_accessible_self_contained_svg(self):
        for name, svg in self.rendered.items():
            with self.subTest(figure=name):
                root = parsed(svg)
                self.assertEqual(root.attrib["role"], "img")
                self.assertEqual(root.attrib["aria-labelledby"], "title desc")
                self.assertTrue(root.find("svg:title", NS).text)
                self.assertTrue(root.find("svg:desc", NS).text)
                self.assertTrue(contract(svg))
                self.assertNotIn("<image", svg)
                self.assertNotIn("<script", svg)
                self.assertNotIn("https://", svg)
                self.assertNotIn("http://", svg.replace("http://www.w3.org/2000/svg", ""))
                ids = [node.attrib["id"] for node in root.iter() if "id" in node.attrib]
                self.assertEqual(len(ids), len(set(ids)))

    def test_committed_svg_and_png_sources_are_current(self):
        for name, svg in self.rendered.items():
            with self.subTest(figure=name):
                self.assertEqual((ASSETS / f"{name}.svg").read_text(encoding="utf-8"), svg,
                                 "Run scripts/draw_diagrams.py --png to regenerate the figures")
                self.assertTrue(draw.png_matches_svg(ASSETS / f"{name}.png", svg),
                                f"{name}.png is missing, malformed or from a different SVG source")

    def test_layout_has_readable_body_text_and_no_measured_overflow(self):
        for name, drawer in draw.FIGURES.items():
            with self.subTest(figure=name):
                canvas = drawer(self.constants, self.results)
                self.assertEqual(canvas.overflows, [])
                self.assertEqual(canvas.w, 1080)
                # Mathematical sub/superscripts may be smaller; body text is not.
                sizes = [float(node.attrib["font-size"])
                         for node in parsed(canvas.render()).findall(".//svg:text", NS)]
                self.assertGreaterEqual(min(sizes), 15)
                for x, y, width, height in canvas.nodes.values():
                    self.assertGreaterEqual(x, 0)
                    self.assertGreaterEqual(y, 0)
                    self.assertLessEqual(x + width, canvas.w)
                    self.assertLessEqual(y + height, canvas.h)

    def test_mathematical_runs_use_renderer_portable_sizes_and_offsets(self):
        markup = draw.render_markup("H_{0} + H^{+} and q_{C}", 20)
        root = ET.fromstring(f"<text>{markup}</text>")
        self.assertEqual("".join(root.itertext()), "H0 + H+ and qC")
        spans = list(root)
        by_text = {span.text: span for span in spans}
        self.assertGreater(float(by_text["0"].attrib["dy"]), 0)
        self.assertLess(float(by_text["+"].attrib["dy"]), 0)
        self.assertTrue(all("%" not in span.attrib["font-size"] for span in spans))
        self.assertTrue(all("baseline-shift" not in span.attrib for span in spans))
        # A plain centered pill must remain one uninterrupted text run.
        self.assertIn("POLICY LOCK · FINAL RULE FIXED BEFORE TEST", visible_text(self.rendered["lifecycle"]))

    def test_feedback_loop_separates_parameters_actions_and_global_admission(self):
        svg = self.rendered["architecture"]
        specification = contract(svg)
        self.assertFalse(specification["labels_enter_runtime"])
        self.assertFalse(specification["parameter_updates_online"])
        self.assertFalse(specification["global_admission_in_loop"])
        self.assertEqual(specification["ordinary_actions"], list(range(8)))
        self.assertEqual(specification["reset_actions"], [8, 9])
        edges = edge_set(svg)
        required = {("state", "measure", "measurement"), ("predictions", "measure", "measurement"),
                    ("measure", "observer", "flow"), ("observer", "policy", "flow"),
                    ("policy", "ordinary", "ordinary_action"), ("policy", "reset", "reset_action"),
                    ("ordinary", "state", "next_state"), ("reset", "state", "next_state"),
                    ("offline_models", "observer", "frozen_parameters")}
        self.assertTrue(required <= edges)
        offline = [node.attrib["id"] for node in parsed(svg).findall(".//svg:g", NS)
                   if node.attrib.get("data-phase") == "offline"]
        self.assertEqual(offline, ["offline_models"])

    def test_lifecycle_locks_precede_selection_admission_and_test(self):
        svg = self.rendered["lifecycle"]
        specification = contract(svg)
        stages = ["prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate"]
        self.assertEqual(specification["stages"], stages)
        self.assertEqual(specification["registry_lock_before"], "select")
        self.assertEqual(specification["candidate_fixed_before"], "admit")
        self.assertEqual(specification["policy_lock_before"], "evaluate")
        self.assertEqual(specification["training_allocation"],
                         self.constants["data_roles_table_S2"]["official_training_allocation"])
        edges = edge_set(svg)
        self.assertIn(("admit", "retain_candidate", "admission_pass"), edges)
        self.assertIn(("admit", "return_comparator", "admission_fail"), edges)
        self.assertIn(("retain_candidate", "evaluate", "locked_policy"), edges)
        self.assertIn(("return_comparator", "evaluate", "locked_policy"), edges)
        # Neither admission branch can pass its test observations back into selection.
        self.assertFalse(any(source == "evaluate" for source, _, _ in edges))
        self.assertIn("separated", specification["sequence_scope"])
        visible = " ".join(visible_text(svg))
        self.assertIn("reused-validation admission statistics remain descriptive", visible)
        for share in specification["training_allocation"].values():
            self.assertIn(draw.pct(share), visible)

    def test_action_rows_bind_each_setting_to_the_correct_action(self):
        root = parsed(self.rendered["control_depth"])
        registry = self.constants["action_registry_table_S6"]
        for action in registry["actions"]:
            with self.subTest(action=action["id"]):
                group = root.find(f".//svg:g[@id='action-row-A{action['id']}']", NS)
                self.assertIsNotNone(group)
                cells = ["".join(node.itertext()) for node in group.findall("svg:text", NS)]
                self.assertEqual(cells[:2], [f"A{action['id']}", action["role"]])
                self.assertEqual(cells[-1], draw.fmt(action["exploration_probability"], 2))
                derived = registry["derived_table_S6b"].get(f"A{action['id']}")
                if derived is not None:
                    expected = [draw.fmt(action[key], 2) for key in ("nu", "rho", "alpha")]
                    expected.append("—" if action["eta"] is None else draw.fmt(action["eta"], 2))
                    expected += [draw.fmt(derived[key], 3) for key in
                                 ("executed_range_hops_per_step", "executed_identity_coefficient")]
                    self.assertEqual(cells[2:-1], expected)
                else:
                    self.assertEqual(cells[2], "assign H0" if action["id"] == 8 else "assign saved Hsafe")
                    self.assertEqual(cells[3:-1], ["0", "—"])

    def test_transport_identity_depends_on_the_registered_site(self):
        specification = contract(self.rendered["control_depth"])
        self.assertEqual(specification["native_A0"], "P")
        self.assertEqual(specification["inserted_A0"], "I")
        self.assertEqual(specification["source_reference"], "H0")
        self.assertTrue(specification["direct_resets_bypass_update"])
        for key, dataset in (("gps_sites", "PascalVOC-SP"), ("peptide_sites", "Peptides-func")):
            registry = self.constants["backbones_table_S5"][dataset]["identity_test_E1"]
            self.assertEqual(specification[key], {"native": registry["native_sites"],
                                                  "inserted": registry["inserted_sites"]})
        self.assertEqual(specification["table_scope"], "common transport before source retention")

    def test_nine_routes_preserve_open_loop_information_contract(self):
        specification = contract(self.rendered["policy_families"])
        self.assertEqual(len(specification["open_loop"]), 6)
        self.assertEqual(len(specification["feedback"]), 3)
        self.assertFalse(specification["MPC_OL"]["replans"])
        self.assertFalse(specification["MPC_OL"]["intermediate_observations_change_plan"])
        self.assertFalse(specification["MPC_OL"]["A9_allowed"])
        self.assertEqual(specification["MPC_OL"]["tail"], "nominal")
        self.assertEqual(specification["MPC_RH"], {"replans": True, "executes": "first_action"})
        visible = " ".join(visible_text(self.rendered["policy_families"]))
        observer = self.constants["observer_eq12_table_S9"]
        self.assertIn(f"beam {observer['beam_width']}", visible)
        self.assertIn(", ".join(str(h) for h in observer["horizons"]), visible)

    def test_artifact_review_scopes_do_not_overstate_replay(self):
        specification = contract(self.rendered["review_workflow"])
        self.assertFalse(specification["numeric_replay_runs_predictor"])
        self.assertEqual(specification["checkpoint_replay_requires"], ["prepared_graph", "trusted_checkpoints"])
        self.assertFalse(specification["portable_bundle_includes_checkpoints"])
        self.assertFalse(specification["original_chronology_authenticated"])
        visible = " ".join(visible_text(self.rendered["review_workflow"]))
        for name in ("reproduce_results.py", "evaluate_checkpoints.py", "build_review_bundle.py", "verify_review_bundle.py"):
            self.assertIn(name, visible)
            self.assertTrue((ROOT / "scripts" / name).is_file())

    def test_evidence_chain_names_only_make_targets_scripts_and_files_that_exist(self):
        svg = self.rendered["evidence_chain"]
        specification = contract(svg)
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        targets = set(re.findall(r"^([A-Za-z0-9_.-]+)\s*:(?!=)", makefile, flags=re.M))
        for line in re.findall(r"^\.PHONY\s*:(.*)$", makefile, flags=re.M):
            targets.update(line.split())
        for target in specification["verify_targets"] + specification["record_steps"] + ["archive-run", "ingest-run", "paper",
                                                                                          "compare-runs", "verify"]:
            self.assertIn(target, targets, target)
        verify_line = next(line for line in makefile.splitlines() if line.startswith("verify:"))
        for target in specification["verify_targets"]:
            self.assertIn(target, verify_line)
        visible = " ".join(visible_text(svg))
        for name in ("make fingerprint", "make archive-run", "make ingest-run", "make compare-runs", "make verify",
                     "run_cross_morse_<dataset>.sh", "configs/arrhenius-site.env.example", "runs/index.json",
                     "AVAILABILITY.md", "ARCHIVE_MANIFEST.json", "make records-index", "records/index.json"):
            self.assertIn(name, visible, name)
        for path in ("configs/arrhenius-site.env.example", "docs/ARRHENIUS.md", "docs/RUN_ARCHIVES.md",
                     "scripts/archive_run.py", "scripts/ingest_run.py", "scripts/build_paper.py", "paper/build",
                     "records/README.md", "records/specs.json", "scripts/records.py"):
            self.assertTrue((ROOT / path).exists(), path)
        from paper.build.figures import FIGURE_IDS
        from paper.build.tables import TABLE_IDS
        self.assertEqual(specification["paper_outputs"]["tables"], list(TABLE_IDS))
        self.assertEqual(specification["paper_outputs"]["figures"], list(FIGURE_IDS))
        self.assertFalse(specification["chronology_authenticated"])
        edges = edge_set(svg)
        self.assertIn(("configs", "launcher", "handoff"), edges)
        self.assertIn(("run", "archive", "handoff"), edges)
        self.assertIn(("ingest", "evidence", "handoff"), edges)

    def test_decision_chain_names_sealed_run_files_and_built_table_cells(self):
        svg = self.rendered["decision_chain"]
        specification = contract(svg)
        example = ROOT / "examples" / "runs" / "arxiv"
        if (example / "final" / "results_summary.json").is_file():
            for step, files in specification["files"].items():
                for relative in files:
                    self.assertTrue((example / relative).is_file(), f"{step}: {relative}")
        from paper.build.figures import FIGURE_IDS
        from paper.build.tables import TABLE_IDS
        self.assertTrue(set(specification["tables"]) <= set(TABLE_IDS))
        self.assertTrue(set(specification["figures"]) <= set(FIGURE_IDS))
        self.assertEqual(specification["steps"], ["candidates", "select", "confirm", "admit", "lock", "evaluate", "cost"])
        self.assertFalse(specification["fallback_probability_measured"])
        visible = " ".join(visible_text(svg))
        self.assertIn("POLICY LOCK · NO TEST LABEL IS READ BEFORE THIS LINE", visible)
        self.assertIn("not measured by this release", visible)
        edges = edge_set(svg)
        for a, b in zip(specification["steps"], specification["steps"][1:]):
            self.assertIn((a, b, "flow"), edges)
        for step in specification["steps"]:
            self.assertIn((step, step + "_record", "written_to"), edges)

    def test_explanatory_diagrams_do_not_depend_on_performance_results(self):
        for name, drawer in draw.FIGURES.items():
            with self.subTest(figure=name):
                self.assertEqual(drawer(self.constants, {}).render(), self.rendered[name])

    def test_check_mode_reports_stale_and_missing_files_without_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            for name, svg in self.rendered.items():
                (out / f"{name}.svg").write_text(svg, encoding="utf-8")
            command = [sys.executable, "-B", str(SCRIPT), "--output-dir", str(out), "--check"]
            passing = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(passing.returncode, 0, passing.stderr)
            (out / "lifecycle.svg").write_bytes(b"\xffmalformed SVG")
            (out / "architecture.svg").unlink()
            failing = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(failing.returncode, 1)
            self.assertIn("lifecycle.svg", failing.stdout)
            self.assertIn("architecture.svg", failing.stdout)
            self.assertNotIn("Traceback", failing.stderr)

    def test_png_fingerprint_detects_old_source_wrong_size_and_corruption(self):
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="2" height="3"></svg>'
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.png"
            path.write_bytes(fixture_png(4, 6))
            self.assertFalse(draw.png_matches_svg(path, svg))
            draw.stamp_png_source(path, svg)
            self.assertTrue(draw.png_matches_svg(path, svg))
            draw.stamp_png_source(path, svg)  # Retagging must replace, not duplicate, the fingerprint.
            self.assertTrue(draw.png_matches_svg(path, svg))
            self.assertFalse(draw.png_matches_svg(path, svg + "\n"))
            path.write_bytes(fixture_png(5, 6))
            draw.stamp_png_source(path, svg)
            self.assertFalse(draw.png_matches_svg(path, svg))
            path.write_bytes(path.read_bytes()[:-1])
            self.assertFalse(draw.png_matches_svg(path, svg))

    def test_explicit_png_failure_is_visible_and_preserves_prior_png(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            prior = out / "architecture.png"
            prior.write_bytes(b"previous image")
            with patch.object(draw, "_rasterize", return_value=False), redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
                result = draw.main(["--output-dir", str(out), "--figure", "architecture", "--png"])
            self.assertEqual(result, 1)
            self.assertIn("PNG rendering failed", errors.getvalue())
            self.assertEqual(prior.read_bytes(), b"previous image")
            self.assertEqual((out / "architecture.svg").read_text(encoding="utf-8"), self.rendered["architecture"])


if __name__ == "__main__":
    unittest.main()
