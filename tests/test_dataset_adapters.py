"""Offline adapter regressions; tiny format fixtures, no benchmark downloads."""
import gzip
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from cross_morse.data import (LEAF_ROLES, ROLE_NAMES, _effective_spec, _partition,
                              _validate_arrays, load_data, prepare_data)
from cross_morse.datasets import (ATOM_FEATURE_DIMS, BENCHMARKS, COCO_LABEL_IDS,
                                  INDEPENDENT_DATASETS, atom_one_hot,
                                  canonical_edges, incident_edge_mean,
                                  load_benchmark, load_lrgb, load_ogb, load_roman,
                                  pack_lrgb, read_lrgb_tuples, remap_coco_labels,
                                  validate_graph_pointers)


def config(name, **changes):
    data = {"partitioner": "graphs" if name in INDEPENDENT_DATASETS else "balanced",
            "num_clusters": 8, "buffer_hops": 1, "synthetic_nodes": 160,
            "synthetic_graphs": 80, "synthetic_nodes_per_graph": 4,
            "synthetic_features": 6, "synthetic_classes": 4}
    data.update(changes)
    return {"dataset": "synthetic", "benchmark_dataset": name, "seed": 17, "data": data}


def sp_graph(labels=(0, 1, 2)):
    n = len(labels)
    return (torch.arange(n*14).reshape(n, 14).float(),
            torch.tensor([[1., 2.], [3., 4.], [1., 2.], [3., 4.]]),
            torch.tensor([[0, 1, 1, 2], [1, 2, 0, 1]], dtype=torch.long),
            torch.tensor(labels))


def peptide_graph(offset=0):
    atoms = torch.zeros((3, 9), dtype=torch.long)
    atoms[:, 0] = torch.tensor([5, 6, 7])
    atoms[:, 1] = 4  # OGB miscellaneous chirality category is valid.
    bonds = torch.tensor([[0, 0, 0], [2, 1, 1], [0, 0, 0], [2, 1, 1]])
    edges = torch.tensor([[0, 1, 1, 2], [1, 2, 0, 1]], dtype=torch.long)
    labels = ((torch.arange(10)+offset) % 2).float()[None]
    return atoms, bonds, edges, labels


class PreparedTaskTests(unittest.TestCase):
    def test_all_seven_fixture_tasks_roundtrip_with_explicit_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in BENCHMARKS:
                with self.subTest(name=name):
                    root = tmp / name
                    data = prepare_data(tmp / "cache", root, config(name))
                    restored = load_data(root)
                    self.assertEqual(data.metadata["fingerprint"], restored.metadata["fingerprint"])
                    self.assertFalse(data.metadata["is_benchmark_data"])
                    self.assertEqual(data.metadata["benchmark_dataset"], name)
                    self.assertEqual(data.graph_ptr is not None, name in INDEPENDENT_DATASETS)
                    self.assertEqual(len(data.global_sample_ids), data.num_samples)
                    if name == "ogbn-proteins":
                        self.assertEqual(data.y.shape, (160, 4))
                        self.assertEqual(data.y.dtype, torch.float32)
                        self.assertEqual(data.metadata["metric"], "rocauc")
                        self.assertIsNotNone(data.edge_weight)
                    elif name == "Peptides-func":
                        self.assertEqual(data.y.shape, (80, 4))
                        self.assertEqual(data.metadata["label_axis"], "graph")
                        self.assertEqual(data.cluster_ids.shape, (80,))
                        self.assertEqual(data.metadata["metric"], "ap")
                    else:
                        self.assertEqual(data.y.ndim, 1)
                        self.assertEqual(data.y.dtype, torch.int64)
                        self.assertEqual(data.num_classes, 4)
                    if name in INDEPENDENT_DATASETS:
                        self.assertEqual(data.num_graphs, 80)
                        self.assertEqual(set(data.graph_splits), set(ROLE_NAMES))
                        validate_graph_pointers(data.graph_ptr, data.edge_ptr, data.num_nodes, data.edge_index.numpy())

    def test_complete_graph_roles_and_episode_labels_share_parent_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in ("PascalVOC-SP", "Peptides-func"):
                with self.subTest(name=name):
                    data = prepare_data(tmp / "cache", tmp / name, config(name))
                    raw_y = data.y.clone()
                    g = int(data.graph_splits["fit_train"][0])
                    episode = data.graph(g)  # sealing happens before any scope
                    self.assertEqual(episode.num_graphs, 1)
                    self.assertEqual(episode.metadata["graph_id"], g)
                    self.assertEqual(episode.num_nodes, 4)
                    self.assertTrue(np.all(episode.cluster_ids == g))
                    self.assertEqual(episode.edge_index.max().item(), 3)
                    expected_ids = np.array([g]) if name == "Peptides-func" else np.arange(g*4, g*4+4)
                    np.testing.assert_array_equal(episode.global_sample_ids, expected_ids)
                    with self.assertRaises(PermissionError):
                        episode.labels("fit_train")
                    with data._label_scope("train"):
                        expected = raw_y[expected_ids].numpy()
                        np.testing.assert_array_equal(episode.labels("fit_train"), expected)
                        self.assertEqual(len(episode.splits["test"]), 0)
                        with self.assertRaises(PermissionError):
                            episode.labels("test")
                        with self.assertRaises(RuntimeError):
                            with episode._label_scope("evaluate"):
                                pass
                    with self.assertRaises(PermissionError):
                        episode.labels("fit_train")
                    for graph_id in range(data.num_graphs):
                        memberships = [role for role in LEAF_ROLES if graph_id in data.graph_splits[role]]
                        self.assertEqual(len(memberships), 1)
                        row_ids = np.flatnonzero(data.cluster_ids == graph_id)
                        self.assertTrue(np.isin(row_ids, data.splits[memberships[0]]).all())

    def test_multilabel_missing_targets_remain_missing_and_invalid_values_fail(self):
        x, edges = np.ones((3, 2)), np.empty((2, 0), dtype=np.int64)
        y = np.array([[0, 1], [np.nan, -1], [1, 0]], dtype=np.float32)
        _, _, result, outputs = _validate_arrays(x, edges, y, 2, task_type="multilabel")
        self.assertEqual(outputs, 2)
        self.assertTrue(np.isnan(result[1]).all())
        with self.assertRaises(ValueError):
            _validate_arrays(x, edges, [[0, 1], [2, 0], [1, 0]], 2, task_type="multilabel")
        with self.assertRaises(ValueError):
            _validate_arrays(x, edges, y, 112, task_type="multilabel")

    def test_degree_stratified_cores_are_reproducible_and_explicit(self):
        from scipy import sparse
        n = 160
        src, dst = np.triu_indices(n, k=1)
        keep = (src+dst) % 9 == 0
        adjacency = sparse.coo_matrix((np.ones(keep.sum()), (src[keep], dst[keep])), shape=(n, n)).tocsr()
        adjacency = adjacency + adjacency.T
        cfg = config("ogbn-proteins", partitioner="degree_stratified", degree_strata=8)
        spec = _effective_spec(cfg)
        first, info = _partition(adjacency, spec)
        second, _ = _partition(adjacency, spec)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(info["partitioner"], "degree_stratified")
        self.assertIn("implementation choice", info["partition_scope"])
        self.assertLessEqual(np.ptp(np.bincount(first)), 1)

    def test_real_preparation_rejects_a_truncated_independent_corpus(self):
        source = pack_lrgb("COCO-SP", {role: [sp_graph((0, 13, 90))] for role in ("train", "valid", "test")})
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("cross_morse.data._load_source", return_value=source):
                with self.assertRaisesRegex(ValueError, "retain all 123286 official graphs"):
                    prepare_data(Path(tmp)/"cache", Path(tmp)/"output", {"dataset": "COCO-SP", "data": {"partitioner": "graphs"}})


class RawFormatTests(unittest.TestCase):
    def test_edge_features_and_weights_coalesce_in_the_same_order(self):
        edges = np.array([[0, 1, 0, 2, 1], [1, 0, 1, 2, 2]], dtype=np.int64)
        weights = np.array([1., 3., 5., 100., 7.])
        attr = np.stack((weights, weights*2), axis=1)
        canonical, adjacency, result, scalar = canonical_edges(edges, 4, attr, weights)
        expected = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
        np.testing.assert_array_equal(canonical, expected)
        np.testing.assert_allclose(scalar, [3, 3, 7, 7])
        np.testing.assert_allclose(result[:, 0], scalar)
        np.testing.assert_allclose(result[:, 1], scalar*2)
        self.assertEqual(adjacency.getrow(3).nnz, 0)
        with self.assertRaises(ValueError):
            canonical_edges(edges, 4, attr, -weights)

    def test_incident_means_include_both_endpoints_and_isolates(self):
        edges = np.array([[0, 0, 1, 2], [1, 2, 0, 0]], dtype=np.int64)
        attr = np.array([[2., 4.], [6., 8.], [2., 4.], [6., 8.]])
        np.testing.assert_allclose(incident_edge_mean(edges, attr, 4), [[4, 6], [2, 4], [6, 8], [0, 0]])

    def test_roman_uses_requested_official_split_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "roman_empire" / "raw"
            directory.mkdir(parents=True)
            n = 80
            masks = {key: np.zeros((10, n), dtype=bool) for key in ("train_masks", "val_masks", "test_masks")}
            for split in range(10):
                order = np.roll(np.arange(n), split)
                masks["train_masks"][split, order[:48]] = True
                masks["val_masks"][split, order[48:64]] = True
                masks["test_masks"][split, order[64:]] = True
            path = directory / "roman_empire.npz"
            np.savez(path, node_features=np.ones((n, 300), dtype=np.float32),
                     node_labels=np.arange(n) % 18, edges=np.array([[0, 1], [1, 2]]), **masks)
            with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
                source = load_roman(tmp, False, 7)
            np.testing.assert_array_equal(source.official_splits["train"], np.flatnonzero(masks["train_masks"][7]))
            np.testing.assert_array_equal(source.edge_index, [[0, 1], [1, 2]])
            self.assertEqual(source.metadata["num_outputs"], 18)
            self.assertEqual(source.metadata["split_index"], 7)
            with self.assertRaises(ValueError):
                load_roman(tmp, False, 10)

    def test_coco_remapping_preserves_all81_official_categories(self):
        np.testing.assert_array_equal(remap_coco_labels(np.array(COCO_LABEL_IDS)), np.arange(81))
        with self.assertRaises(ValueError):
            remap_coco_labels(np.array([12]))
        source = pack_lrgb("COCO-SP", {role: [sp_graph((0, 13, 90))] for role in ("train", "valid", "test")})
        self.assertEqual(source.metadata["num_outputs"], 81)
        np.testing.assert_array_equal(source.y, [0, 12, 80]*3)
        np.testing.assert_array_equal(source.graph_ptr, [0, 3, 6, 9])
        validate_graph_pointers(source.graph_ptr, source.edge_ptr, len(source.x), source.edge_index)

    def test_peptide_graph_targets_one_hot_atoms_and_nonnegative_weights(self):
        source = pack_lrgb("Peptides-func", {role: [peptide_graph(i)] for i, role in enumerate(("train", "valid", "test"))})
        self.assertEqual(source.y.shape, (3, 10))
        self.assertEqual(source.x.shape, (9, sum(ATOM_FEATURE_DIMS)))
        np.testing.assert_allclose(source.x.sum(1), 9)
        self.assertEqual(source.metadata["label_axis"], "graph")
        self.assertEqual(source.metadata["num_outputs"], 10)
        self.assertTrue(np.all(source.edge_weight >= 1))
        np.testing.assert_allclose(source.edge_weight, 1 + source.edge_attr.mean(1))
        bad = np.zeros((2, 9), dtype=np.int64)
        bad[0, 0] = 119
        with self.assertRaises(ValueError):
            atom_one_hot(bad)

    def test_cross_graph_edges_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Cross-graph"):
            validate_graph_pointers([0, 2, 4], [0, 1, 2], 4, np.array([[0, 2], [2, 3]]))

    def test_restricted_raw_tuple_reader_and_official_cache_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name, item, suffix in (("PascalVOC-SP", sp_graph(), ".pickle"),
                                       ("Peptides-func", peptide_graph(), ".pt")):
                directory = tmp / name.lower() / "raw"
                directory.mkdir(parents=True)
                for role in ("train", "val", "test"):
                    path = directory / (role+suffix)
                    if suffix == ".pt":
                        torch.save([item], path)
                    else:
                        with path.open("wb") as stream:
                            pickle.dump([item], stream)
                    self.assertEqual(len(read_lrgb_tuples(path)), 1)
                with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
                    source = load_lrgb(tmp, False, name)
                self.assertEqual(len(source.graph_ptr), 4)
                self.assertEqual(source.metadata["num_outputs"], 10 if name == "Peptides-func" else 21)

    def test_unsafe_pickle_is_rejected_without_executing_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, marker = Path(tmp)/"bad.pickle", Path(tmp)/"executed"
            class Payload:
                def __reduce__(self):
                    return os.system, ("touch " + str(marker),)
            path.write_bytes(pickle.dumps([Payload()]))
            with self.assertRaises(pickle.UnpicklingError):
                read_lrgb_tuples(path)
            self.assertFalse(marker.exists())

    def test_missing_raw_data_cannot_download_without_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
                for name in ("Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func"):
                    with self.subTest(name=name), self.assertRaises(FileNotFoundError):
                        load_benchmark(tmp, name, False)


class OGBRawAdapters(unittest.TestCase):
    @staticmethod
    def write_csv(path, values):
        with gzip.open(path, "wt") as stream:
            np.savetxt(stream, values, delimiter=",", fmt="%.8g")

    def test_products_and_proteins_use_official_raw_splits_and_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, split in (("ogbn-products", "sales_ranking"), ("ogbn-proteins", "species")):
                with self.subTest(name=name):
                    directory = Path(tmp) / name.replace("-", "_")
                    raw, official = directory/"raw", directory/"split"/split
                    raw.mkdir(parents=True)
                    official.mkdir(parents=True)
                    (directory/"RELEASE_v1.txt").write_text("fixture")
                    edges = np.array([[0, 1], [0, 2]])
                    self.write_csv(raw/"edge.csv.gz", edges)
                    self.write_csv(raw/"num-node-list.csv.gz", [[4]])
                    self.write_csv(raw/"num-edge-list.csv.gz", [[2]])
                    if name == "ogbn-products":
                        self.write_csv(raw/"node-feat.csv.gz", np.ones((4, 100)))
                        self.write_csv(raw/"node-label.csv.gz", [[0], [1], [2], [3]])
                    else:
                        attr = np.array([[.2]*8, [.6]*8])
                        self.write_csv(raw/"edge-feat.csv.gz", attr)
                        self.write_csv(raw/"node_species.csv.gz", [[1], [1], [2], [3]])
                        self.write_csv(raw/"node-label.csv.gz", np.tile(np.arange(112) % 2, (4, 1)))
                    for role, indices in (("train", [0, 1]), ("valid", [2]), ("test", [3])):
                        self.write_csv(official/f"{role}.csv.gz", np.array(indices)[:, None])
                    (directory/"processed").mkdir()
                    (directory/"processed"/"data_processed").write_bytes(b"must not deserialize")
                    (official/"split_dict.pt").write_bytes(b"must not deserialize")
                    with mock.patch("torch.load", side_effect=AssertionError("No pickle cache may load")):
                        source = load_ogb(tmp, False, name)
                    self.assertEqual(source.metadata["official_split"], split)
                    np.testing.assert_array_equal(source.official_splits["test"], [3])
                    if name == "ogbn-proteins":
                        self.assertEqual(source.y.shape, (4, 112))
                        self.assertEqual(source.metadata["num_outputs"], 112)
                        np.testing.assert_allclose(source.x, np.array([[.4]*8, [.2]*8, [.6]*8, [0.]*8]), rtol=1e-6)
                        np.testing.assert_allclose(source.edge_weight, source.edge_attr.mean(1))
                    else:
                        self.assertEqual(source.metadata["num_outputs"], 47)


if __name__ == "__main__":
    unittest.main()
