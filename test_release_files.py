from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseFileTests(unittest.TestCase):
    def test_contract_has_six_roles(self) -> None:
        data = json.loads((ROOT / "release" / "scientific_contract.json").read_text(encoding="utf-8"))
        self.assertEqual(
            [x["name"] for x in data["data_roles"]],
            ["D_fit", "D_explore", "D_select", "D_confirm", "D_admit", "D_test"],
        )

    def test_scope_is_code_only(self) -> None:
        data = json.loads((ROOT / "release" / "release_scope.json").read_text(encoding="utf-8"))
        self.assertEqual(data["release_type"], "source-code-only")
        self.assertFalse(data["manuscript_tex_in_repository"])
        self.assertFalse(data["article_files_in_repository"])

    def test_implementation_map_covers_all_datasets(self) -> None:
        data = json.loads((ROOT / "release" / "implementation_map.json").read_text(encoding="utf-8"))
        expected = {"ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func"}
        self.assertEqual(set(data["datasets"]), expected)
        self.assertTrue(all(data["datasets"][name] for name in expected))

    def test_publication_metadata_is_final(self) -> None:
        data = json.loads((ROOT / "release" / "publication.json").read_text(encoding="utf-8"))
        self.assertTrue(data["article_url"].startswith("https://"))
        self.assertTrue(data["supplement_url"].startswith("https://"))
        self.assertTrue(data["doi"].startswith("10."))

    def test_run_manifest_schema_is_stage_aware(self) -> None:
        data = json.loads((ROOT / "release" / "schemas" / "run_manifest.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(data["properties"]["schema_version"]["const"], "2.0")
        self.assertIn("fit", data["properties"]["stage"]["enum"])

    def test_experiment_registry_has_expected_ids(self) -> None:
        data = json.loads((ROOT / "release" / "experiment_registry.json").read_text(encoding="utf-8"))
        ids = {x["id"] for x in data["entries"]}
        self.assertTrue({f"EXP-{i:02d}" for i in range(1, 17)} <= ids)
        self.assertIn("PROV-01", ids)


if __name__ == "__main__":
    unittest.main()
