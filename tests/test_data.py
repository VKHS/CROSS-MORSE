"""Split leakage, graph fidelity, safe caches and official-reader integration."""
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from scipy import sparse
import torch

from cross_morse.data import (
    LEAF_ROLES, _buffers, _download_archive, _load_ogb, _ogb_noninteractive, _partition,
    _simple_undirected, load_data, prepare_data,
)


def config():
    return {"dataset": "synthetic", "seed": 17, "data": {
        "synthetic_nodes": 192, "synthetic_features": 8, "synthetic_classes": 4,
        "num_clusters": 8, "partitioner": "balanced", "buffer_hops": 2,
        "internal_val_fraction": 0.1, "save_buffers": True,
    }}


def raw_fixture(path, x=None):
    n = 80
    if x is None:
        x = np.arange(n*4, dtype=np.float32).reshape(n, 4)
        x[:, 3] = 7.0
    edges = np.stack((np.arange(n-2), np.arange(1, n-1))).astype(np.int64)
    edges = np.concatenate([edges, edges[:, :3], np.array([[0, 10], [0, 10]])], axis=1)
    y = (np.arange(n) % 4).astype(np.int64)
    np.savez(path, x=x, edge_index=edges, y=y,
             train_idx=np.arange(48), valid_idx=np.arange(48, 64), test_idx=np.arange(64, 80))
    return x, edges, y


class PreparedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def prepare(self, cfg=None, name="result"):
        return prepare_data(self.root / "raw", self.root / name, cfg or config())

    def test_separated_protocol_exact_coverage_and_role_access(self):
        data = self.prepare()
        self.assertEqual(data.x.dtype, torch.float32)
        self.assertEqual(data.edge_index.dtype, torch.int64)
        self.assertEqual(data.y.dtype, torch.int64)
        self.assertEqual(data.x.device.type, "cpu")
        all_nodes = np.concatenate([data.splits[name] for name in LEAF_ROLES])
        np.testing.assert_array_equal(np.sort(all_nodes), np.arange(192))
        self.assertEqual(len(data.splits["fit"]), int(int(.6*192)*.6))
        fit_union = np.sort(np.r_[data.splits["fit_train"], data.splits["fit_val"]])
        np.testing.assert_array_equal(fit_union, np.sort(data.splits["fit"]))
        labels = data.labels("admit")
        np.testing.assert_array_equal(labels, data.y.numpy()[data.splits["admit"]])
        labels[:] = -99
        self.assertTrue(np.all(data.labels("admit") >= 0))
        with self.assertRaises(KeyError):
            data.labels("all")
        self.assertFalse(data.metadata["buffer_audit"]["independent_graph_guarantee"])

    def test_roundtrip_and_fingerprint_are_reproducible(self):
        a = self.prepare(name="a")
        b = self.prepare(name="b")
        c = load_data(self.root / "a")
        d = self.prepare(name="a")
        self.assertEqual(a.metadata["fingerprint"], b.metadata["fingerprint"])
        self.assertEqual(a.metadata["fingerprint"], c.metadata["fingerprint"])
        torch.testing.assert_close(a.x, d.x, rtol=0, atol=0)
        torch.testing.assert_close(a.edge_index, c.edge_index, rtol=0, atol=0)
        for role in a.splits:
            np.testing.assert_array_equal(a.splits[role], c.splits[role])

    def test_standardization_uses_only_fit_train_features(self):
        path = self.root / "input.npz"
        original, _, _ = raw_fixture(path)
        cfg = config()
        cfg["data"].update(graph_npz=str(path), feature_standardization=True)
        data = self.prepare(cfg)
        train = data.splits["fit_train"]
        with np.load(self.root / "result/prepared/normalization.npz", allow_pickle=False) as archive:
            mean, scale = archive["mean"], archive["scale"]
        np.testing.assert_allclose(mean, original[train].astype(np.float64).mean(0))
        self.assertEqual(scale[3], 1.0)
        np.testing.assert_allclose(data.x.numpy()[train].mean(0), np.zeros(4), atol=1e-6)
        modified = original.copy()
        modified[np.setdiff1d(np.arange(80), train), :3] += 100000
        second_path = self.root / "changed_nonfit.npz"
        raw_fixture(second_path, modified)
        second_cfg = copy.deepcopy(cfg)
        second_cfg["data"]["graph_npz"] = str(second_path)
        second = self.prepare(second_cfg, "changed")
        torch.testing.assert_close(data.x[train], second.x[train], rtol=0, atol=0)
        np.testing.assert_array_equal(data.cluster_ids, second.cluster_ids)

    def test_full_graph_is_preserved_and_duplicates_loops_removed(self):
        path = self.root / "graph.npz"
        _, original, _ = raw_fixture(path)
        cfg = config()
        cfg["data"]["graph_npz"] = str(path)
        data = self.prepare(cfg)
        expected = {(int(a), int(b)) for a, b in original.T if a != b}
        expected |= {(b, a) for a, b in expected}
        actual = {tuple(pair) for pair in data.edge_index.numpy().T.tolist()}
        self.assertEqual(actual, expected)
        self.assertEqual(data.num_nodes, 80)  # Node 79 remains as an isolate.
        self.assertEqual(data.metadata["isolated_nodes"], 1)
        self.assertEqual(data.metadata["buffer_audit"]["propagation_context"], "full_graph")

    def test_cache_rejects_changed_configuration_and_corruption(self):
        self.prepare()
        cfg = config()
        cfg["seed"] += 1
        with self.assertRaisesRegex(ValueError, "different preprocessing"):
            self.prepare(cfg)
        path = self.root / "result/prepared/clusters.npz"
        with path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "integrity check"):
            load_data(self.root / "result")

    def test_source_cache_drift_is_rejected(self):
        path = self.root / "raw.npz"
        raw_fixture(path)
        cfg = config()
        cfg["data"]["graph_npz"] = str(path)
        self.prepare(cfg)
        raw_fixture(path, np.ones((80, 4), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "source graph NPZ changed"):
            self.prepare(cfg)

    def test_invalid_split_and_object_array_are_rejected(self):
        path = self.root / "bad.npz"
        x, edges, y = raw_fixture(path)
        np.savez(path, x=x, edge_index=edges, y=y,
                 train_idx=np.arange(48), valid_idx=np.arange(47, 64), test_idx=np.arange(64, 80))
        cfg = config()
        cfg["data"]["graph_npz"] = str(path)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.prepare(cfg)
        np.savez(path, x=np.array([[object()]], dtype=object), edge_index=edges, y=y,
                 train_idx=np.arange(48), valid_idx=np.arange(48, 64), test_idx=np.arange(64, 80))
        with self.assertRaisesRegex(ValueError, "Object arrays"):
            self.prepare(cfg)

    def test_non_finite_features_and_fractional_edges_are_rejected(self):
        path = self.root / "bad.npz"
        x, edges, y = raw_fixture(path)
        x[0, 0] = np.nan
        np.savez(path, x=x, edge_index=edges, y=y, train_idx=np.arange(48), valid_idx=np.arange(48, 64), test_idx=np.arange(64, 80))
        cfg = config()
        cfg["data"]["graph_npz"] = str(path)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.prepare(cfg)
        x, edges, y = raw_fixture(path)
        np.savez(path, x=x, edge_index=edges.astype(float) + 0.25, y=y,
                 train_idx=np.arange(48), valid_idx=np.arange(48, 64), test_idx=np.arange(64, 80))
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            self.prepare(cfg)


class ContextAndPartitionTests(unittest.TestCase):
    def test_halos_are_exact_on_path_graph(self):
        edges = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
        _, adjacency = _simple_undirected(edges, 5)
        clusters = np.array([0, 0, 1, 1, 1])
        audit, packed = _buffers(adjacency, clusters, 2, True)
        np.testing.assert_array_equal(packed["context_nodes"][:packed["context_indptr"][1]], [0, 1, 2, 3])
        self.assertEqual(audit["halo_sizes"], [2, 2])
        self.assertEqual(adjacency.nnz, 8)  # The full graph is not modified.

    def test_real_graph_cannot_silently_use_balanced_fallback(self):
        adjacency = sparse.eye(5, format="csr")
        spec = {"dataset": "ogbn-arxiv", "seed": 1, "data": {"num_clusters": 2, "partitioner": "balanced"}}
        with self.assertRaisesRegex(ValueError, "only for explicitly synthetic"):
            _partition(adjacency, spec)

    @unittest.skipUnless(importlib.util.find_spec("pymetis"), "pymetis is required for the real benchmark")
    def test_actual_metis_partition_is_reproducible(self):
        edges = np.array([np.arange(24), (np.arange(24)+1) % 24], dtype=np.int64)
        _, adjacency = _simple_undirected(edges, 24)
        spec = {"dataset": "ogbn-arxiv", "seed": 1, "data": {"num_clusters": 4, "partitioner": "metis"}}
        a, metadata = _partition(adjacency, spec)
        b, _ = _partition(adjacency, spec)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(np.unique(a), np.arange(4))
        self.assertEqual(metadata["partitioner"], "metis")

    def test_download_authorization_and_module_state_restored(self):
        original = lambda url: False
        module = SimpleNamespace(decide_download=original)
        with _ogb_noninteractive(module, False):
            with self.assertRaises(FileNotFoundError):
                module.decide_download("https://official.example/data.zip")
            with self.assertRaises(RuntimeError):
                module.input("update?")
        self.assertIs(module.decide_download, original)
        self.assertFalse(hasattr(module, "input"))
        self.assertFalse(hasattr(module, "download_url"))
        with redirect_stdout(io.StringIO()), _ogb_noninteractive(module, True):
            self.assertTrue(module.decide_download("https://official.example/data.zip"))
        self.assertIs(module.decide_download, original)

    def test_incomplete_download_is_never_committed_and_valid_zip_reuses(self):
        body = io.BytesIO()
        with zipfile.ZipFile(body, "w") as archive:
            archive.writestr("arxiv/raw/example.txt", "complete official-style fixture")
        payload = body.getvalue()

        class Response(io.BytesIO):
            def __init__(self, data, expected):
                super().__init__(data)
                self.headers = {"Content-Length": str(expected)}

        with tempfile.TemporaryDirectory() as temp:
            url = "https://official.example/arxiv.zip"
            with patch("urllib.request.urlopen", return_value=Response(payload[:20], len(payload))):
                with self.assertRaisesRegex(RuntimeError, "Incomplete official OGB download"):
                    _download_archive(url, temp, False)
            self.assertFalse((Path(temp) / "arxiv.zip").exists())
            self.assertEqual(list(Path(temp).glob("*.part")), [])
            with patch("urllib.request.urlopen", return_value=Response(payload, len(payload))):
                path = _download_archive(url, temp, False)
            with patch("urllib.request.urlopen", side_effect=AssertionError("Verified archive should be reused")):
                self.assertEqual(path, _download_archive(url, temp, False))


@unittest.skipUnless(importlib.util.find_spec("ogb"), "ogb is required for the real benchmark")
class OfficialReaderTests(unittest.TestCase):
    def test_raw_official_reader_ignores_processed_and_split_pickles(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "ogbn_arxiv"
            raw, split, processed = dataset / "raw", dataset / "split/time", dataset / "processed"
            for folder in (raw, split, processed):
                folder.mkdir(parents=True)
            (dataset / "RELEASE_v1.txt").write_text("test fixture, not benchmark evidence\n")
            values = {
                "edge": np.array([[0, 1], [1, 2]]), "num-node-list": [[3]], "num-edge-list": [[2]],
                "node-feat": np.arange(12).reshape(3, 4) / 3, "node-label": [[0], [1], [2]], "node_year": [[2010], [2018], [2020]],
            }
            for name, array in values.items():
                pd.DataFrame(array).to_csv(raw / f"{name}.csv.gz", index=False, header=False, compression="gzip")
            for role, node in (("train", 0), ("valid", 1), ("test", 2)):
                pd.DataFrame([node]).to_csv(split / f"{role}.csv.gz", index=False, header=False, compression="gzip")
            (processed / "data_processed").write_bytes(b"invalid pickle must never be read")
            (split / "split_dict.pt").write_bytes(b"invalid pickle must never be read")
            with patch("torch.load", side_effect=AssertionError("No pickle loads allowed")), redirect_stdout(io.StringIO()):
                x, edges, y, official, metadata = _load_ogb(root, False)
            self.assertEqual(x.shape, (3, 4))
            self.assertEqual(edges.shape, (2, 2))
            np.testing.assert_array_equal(official["test"], [2])
            self.assertFalse(metadata["legacy_pickle_cache_loaded"])


if __name__ == "__main__":
    unittest.main()
