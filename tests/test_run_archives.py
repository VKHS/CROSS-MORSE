"""Archive packing, verification and ingestion of run directories (standard library only)."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.archive_run import ArchiveError, pack, verify_archive, verify_stage_chain  # noqa: E402
from scripts.ingest_run import ingest, load_index  # noqa: E402

EXAMPLE = PROJECT / "examples" / "runs" / "arxiv"


@unittest.skipUnless((EXAMPLE / "final" / "results_summary.json").is_file(), "bundled examples absent")
class ArchiveTests(unittest.TestCase):
    def test_stage_chain_of_a_bundled_example_verifies(self):
        chain = verify_stage_chain(EXAMPLE)
        self.assertEqual(chain["stages"][-1], "evaluate")
        self.assertGreater(chain["sealed_artifacts"], 100)

    def test_pack_verify_ingest_roundtrip_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = tmp / "deposit" / "arxiv.tar.gz"
            manifest = pack(EXAMPLE, archive, campaign=False)
            self.assertEqual(manifest["dataset"], "ogbn-arxiv")
            self.assertFalse(manifest["scientific_run"])
            verified = verify_archive(archive)
            self.assertEqual(verified["archive_sha256"], manifest["archive_sha256"])
            runs_root = tmp / "runs"
            index_path = runs_root / "index.json"
            entry = ingest(archive, doi="10.5281/zenodo.0", runs_root=runs_root, index_path=index_path, replace=False)
            self.assertTrue(entry["verified"])
            self.assertEqual(entry["deposit_doi"], "10.5281/zenodo.0")
            self.assertEqual(entry["run_source_sha256"], [manifest["source_sha256"]])
            self.assertTrue((runs_root / "arxiv" / "final" / "results_summary.json").is_file())
            index = load_index(index_path)
            self.assertEqual([e["stem"] for e in index["runs"]], ["arxiv"])
            with self.assertRaisesRegex(ArchiveError, "already exists"):
                ingest(archive, doi=None, runs_root=runs_root, index_path=index_path, replace=False)
            # the index-driven table build reads the ingested run
            out = tmp / "out"
            result = subprocess.run([sys.executable, "-B", str(PROJECT / "scripts/build_paper.py"), "tables", "--index",
                                     str(index_path), "--label", "t", "--output-dir", str(out)],
                                    cwd=PROJECT, capture_output=True, text=True, timeout=600)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((out / "tables" / "T1.tex").is_file())

    def test_tampered_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = tmp / "arxiv.tar.gz"
            pack(EXAMPLE, archive, campaign=False)
            extracted = tmp / "x"
            with tarfile.open(archive) as tar:
                tar.extractall(extracted, filter="data")
            (extracted / "arxiv" / "final" / "results.csv").write_bytes(b"model,ensemble\nqD,2\n")
            bad = tmp / "bad.tar.gz"
            with tarfile.open(bad, "w:gz") as tar:
                tar.add(extracted / "arxiv", arcname="arxiv")
            with self.assertRaisesRegex(ArchiveError, "differs from its manifest entry"):
                verify_archive(bad)

    def test_sealed_output_change_breaks_the_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "arxiv"
            shutil.copytree(EXAMPLE, run)
            (run / "final" / "qD_probabilities.npy").write_bytes(b"not the sealed predictions")
            with self.assertRaisesRegex(ArchiveError, "sealed output changed"):
                verify_stage_chain(run)

    def test_campaign_pack_requires_split_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ArchiveError, "no split_"):
                pack(Path(tmp), Path(tmp) / "c.tar.gz", campaign=True)
            campaign = Path(tmp) / "campaign"
            for i in range(2):
                shutil.copytree(EXAMPLE, campaign / f"split_{i:02d}")
            manifest = pack(campaign, Path(tmp) / "c.tar.gz", campaign=True)
            self.assertEqual(manifest["kind"], "campaign")
            self.assertEqual(sorted(manifest["splits"]), ["split_00", "split_01"])
            self.assertEqual(verify_archive(Path(tmp) / "c.tar.gz")["kind"], "campaign")


if __name__ == "__main__":
    unittest.main()
