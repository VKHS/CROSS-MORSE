"""Official raw-data adapters and numeric graph packing for the seven benchmarks.

The public formats are documented by OGB and the benchmark authors/PyG:
https://ogb.stanford.edu/docs/nodeprop/
https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/lrgb.html
https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/heterophilous_graph_dataset.html

Downloads require an explicit flag. Processed Python-object caches are never
loaded. The LRGB authors distribute tensor tuples: a restricted raw unpickler
accepts only tensor/NumPy reconstruction, and tensor archives use weights_only.
Prepared artifacts contain numeric arrays and JSON only.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import fcntl
import importlib
import io
import os
from pathlib import Path
import pickle
import stat
import tempfile
import urllib.parse
import urllib.request
import zipfile

import numpy as np
from scipy import sparse
import torch

from .tasks import task_spec


BENCHMARKS = ("ogbn-arxiv", "ogbn-products", "ogbn-proteins", "Roman-empire",
              "PascalVOC-SP", "COCO-SP", "Peptides-func")
INDEPENDENT_DATASETS = frozenset({"PascalVOC-SP", "COCO-SP", "Peptides-func"})
ROMAN_URL = "https://github.com/yandex-research/heterophilous-graphs/raw/main/data/roman_empire.npz"
LRGB_SOURCES = {
    "PascalVOC-SP": ("https://www.dropbox.com/s/8x722ai272wqwl4/pascalvocsp.zip?dl=1", "voc_superpixels_edge_wt_region_boundary", "pickle"),
    "COCO-SP": ("https://www.dropbox.com/s/r6ihg1f4pmyjjy0/cocosp.zip?dl=1", "coco_superpixels_edge_wt_region_boundary", "pickle"),
    "Peptides-func": ("https://www.dropbox.com/s/ycsq37q8sxs1ou8/peptidesfunc.zip?dl=1", "peptidesfunc", "pt"),
}
# Fixed OGB 1.3.6 vocabulary, including its miscellaneous categories. Using
# one-hot columns avoids imposing a numeric distance on categorical atom codes.
ATOM_FEATURE_DIMS = (119, 5, 12, 12, 10, 6, 6, 2, 2)
BOND_FEATURE_DIMS = (5, 6, 2)
COCO_LABEL_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16,
                  17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
                  34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48,
                  49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                  63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                  81, 82, 84, 85, 86, 87, 88, 89, 90)


@dataclass
class RawDataset:
    x: np.ndarray
    edge_index: np.ndarray
    y: np.ndarray
    official_splits: dict[str, np.ndarray]
    metadata: dict
    graph_ptr: np.ndarray | None = None
    edge_ptr: np.ndarray | None = None
    edge_attr: np.ndarray | None = None
    edge_weight: np.ndarray | None = None


def task_metadata(name, num_outputs=None):
    task = task_spec(name)
    count = task.num_outputs if num_outputs is None else int(num_outputs)
    return {"benchmark_dataset": task.dataset, "task_type": task.task_type,
            "prediction_level": task.prediction_level, "label_axis": task.label_axis,
            "metric": task.metric, "num_outputs": count, "num_classes": count,
            "num_tasks": count if task.task_type == "multilabel" else 1,
            "independent_graphs": task.dataset in INDEPENDENT_DATASETS}


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def validate_graph_pointers(graph_ptr, edge_ptr, n, edges):
    """Validate a complete disjoint union, including each edge's graph boundary."""
    ptr, eptr = np.asarray(graph_ptr), np.asarray(edge_ptr)
    if ptr.ndim != 1 or eptr.ndim != 1 or ptr.dtype.kind not in "iu" or eptr.dtype.kind not in "iu":
        raise ValueError("graph_ptr and edge_ptr must be one-dimensional integer arrays")
    ptr, eptr = ptr.astype(np.int64), eptr.astype(np.int64)
    if len(ptr) < 2 or len(ptr) != len(eptr) or ptr[0] != 0 or eptr[0] != 0:
        raise ValueError("Graph pointers must start at zero and contain one boundary per graph")
    if ptr[-1] != n or eptr[-1] != edges.shape[1] or np.any(np.diff(ptr) <= 0) or np.any(np.diff(eptr) < 0):
        raise ValueError("Graph pointers do not describe all nonempty graphs and edge entries")
    for g in range(len(ptr) - 1):
        local_edges = edges[:, eptr[g]:eptr[g+1]]
        if local_edges.size and (local_edges.min() < ptr[g] or local_edges.max() >= ptr[g+1]):
            raise ValueError(f"Cross-graph or misordered edge in graph {g}")
    return np.ascontiguousarray(ptr), np.ascontiguousarray(eptr)


def canonical_edges(edges, n, edge_attr=None, edge_weight=None):
    """Symmetrize, remove loops, and average duplicate/reciprocal edge attributes.

    Every output channel shares the same row-sorted COO indexing. Averages treat
    raw edge entries as observations; already reciprocal OGB entries therefore
    preserve their weights. Sparse matrices are processed one channel at a time
    rather than allocating a doubled E-by-feature tensor.
    """
    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu":
        raise ValueError("edge_index must be integer [2,E]")
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("Edge endpoint out of range")
    attr = None if edge_attr is None else np.asarray(edge_attr)
    if attr is not None:
        if attr.ndim == 1:
            attr = attr[:, None]
        if attr.ndim != 2 or attr.shape[0] != edges.shape[1] or not np.isfinite(attr).all():
            raise ValueError("edge_attr must be a finite [E,D] matrix aligned with edge_index")
    weight = None if edge_weight is None else np.asarray(edge_weight, dtype=np.float32)
    if weight is not None and (weight.shape != (edges.shape[1],) or not np.isfinite(weight).all() or np.any(weight < 0)):
        raise ValueError("edge_weight must be finite, nonnegative and aligned with edge_index")
    count = sparse.coo_matrix((np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])), shape=(n, n)).tocsr()
    count = (count + count.T).tocsr()
    count.setdiag(0)
    count.eliminate_zeros()
    count.sum_duplicates()
    count.sort_indices()
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(count.indptr))
    cols = count.indices.astype(np.int64)
    out_edges = np.stack((rows, cols))

    def average(values):
        channel = sparse.coo_matrix((np.asarray(values, dtype=np.float32), (edges[0], edges[1])), shape=(n, n)).tocsr()
        channel = (channel + channel.T).tocsr()
        summed = np.asarray(channel[rows, cols]).reshape(-1)
        return np.asarray(summed / count.data, dtype=np.float32)

    out_attr = None
    if attr is not None:
        out_attr = np.empty((len(rows), attr.shape[1]), dtype=np.float32)
        for column in range(attr.shape[1]):
            out_attr[:, column] = average(attr[:, column])
    out_weight = None if weight is None else average(weight)
    adjacency = count.astype(bool)
    return np.ascontiguousarray(out_edges), adjacency, out_attr, out_weight


def incident_edge_mean(edges, edge_attr, n):
    """Mean edge features at both endpoints; isolated nodes receive zeros."""
    edges, attr = np.asarray(edges), np.asarray(edge_attr, dtype=np.float32)
    if attr.ndim != 2 or attr.shape[0] != edges.shape[1] or not np.isfinite(attr).all():
        raise ValueError("Incident-edge features must be finite and edge aligned")
    degree = np.bincount(edges[0], minlength=n) + np.bincount(edges[1], minlength=n)
    result = np.zeros((n, attr.shape[1]), dtype=np.float32)
    for c in range(attr.shape[1]):
        sums = np.bincount(edges[0], weights=attr[:, c], minlength=n)
        sums += np.bincount(edges[1], weights=attr[:, c], minlength=n)
        result[:, c] = sums / np.maximum(degree, 1)
    return result


def atom_one_hot(x):
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != len(ATOM_FEATURE_DIMS) or x.dtype.kind not in "iu":
        raise ValueError("Peptides atoms require nine integer OGB categorical fields")
    encoded = np.zeros((len(x), sum(ATOM_FEATURE_DIMS)), dtype=np.float32)
    offset = 0
    for column, size in enumerate(ATOM_FEATURE_DIMS):
        values = x[:, column]
        if np.any(values < 0) or np.any(values >= size):
            raise ValueError(f"Peptides atom column {column} is outside its OGB vocabulary")
        encoded[np.arange(len(x)), offset + values] = 1
        offset += size
    return encoded


def bond_scalar_weights(edge_attr):
    """Explicit positive scalar convention where the manuscript leaves it open.

    The original categorical bond attributes are also retained. This fixed
    mapping is a reproducible modelling choice, not a reported recovered map.
    """
    values = np.asarray(edge_attr)
    if values.ndim != 2 or values.shape[1] != 3 or values.dtype.kind not in "iu":
        raise ValueError("Peptides bonds require three integer OGB categorical fields")
    for c, size in enumerate(BOND_FEATURE_DIMS):
        if np.any(values[:, c] < 0) or np.any(values[:, c] >= size):
            raise ValueError(f"Peptides bond column {c} is outside its OGB vocabulary")
    return 1.0 + values.astype(np.float32).mean(axis=1)


def remap_coco_labels(values):
    y = np.asarray(values)
    if y.dtype.kind not in "iu" or np.any(y < 0) or np.any(y > max(COCO_LABEL_IDS)):
        raise ValueError("Invalid raw COCO category IDs")
    lookup = np.full(max(COCO_LABEL_IDS) + 1, -1, dtype=np.int64)
    lookup[np.asarray(COCO_LABEL_IDS)] = np.arange(len(COCO_LABEL_IDS))
    mapped = lookup[y]
    if np.any(mapped < 0):
        raise ValueError("Raw COCO labels include an ID outside the official 81 categories")
    return mapped


def _safe_storage_from_bytes(value):
    return torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)


class _TensorTupleUnpickler(pickle.Unpickler):
    """Minimal raw-tuple compatibility, with no arbitrary class imports."""
    def find_class(self, module, name):
        try:
            multiarray = importlib.import_module("numpy._core.multiarray")
        except ModuleNotFoundError:  # NumPy 1.26 compatibility
            multiarray = importlib.import_module("numpy.core.multiarray")
        allowed = {
            ("collections", "OrderedDict"): OrderedDict,
            ("builtins", "set"): set, ("builtins", "slice"): slice,
            ("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype,
            ("torch", "Size"): torch.Size,
            ("torch._utils", "_rebuild_tensor"): torch._utils._rebuild_tensor,
            ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
            ("torch.storage", "_load_from_bytes"): _safe_storage_from_bytes,
        }
        for namespace in ("numpy.core.multiarray", "numpy._core.multiarray"):
            allowed[(namespace, "_reconstruct")] = multiarray._reconstruct
            allowed[(namespace, "scalar")] = multiarray.scalar
        if (module, name) in allowed:
            return allowed[(module, name)]
        raise pickle.UnpicklingError(f"Unsupported object in raw dataset: {module}.{name}")


def read_lrgb_tuples(path):
    path = Path(path)
    if path.suffix == ".pt":
        result = torch.load(path, map_location="cpu", weights_only=True)
    else:
        with path.open("rb") as stream:
            result = _TensorTupleUnpickler(stream).load()
    if not isinstance(result, (list, tuple)) or not result:
        raise ValueError("An LRGB split must contain a nonempty sequence of raw tensor tuples")
    return result


def _download_file(url, target, *, validate=None):
    """Atomic download; callers check authorization before reaching this helper."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".part", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream, urllib.request.urlopen(url, timeout=60) as response:
            length = response.headers.get("Content-Length")
            expected = None if length is None else int(length)
            total = 0
            while block := response.read(1024 * 1024):
                stream.write(block)
                total += len(block)
            stream.flush()
            os.fsync(stream.fileno())
        if expected is not None and total != expected:
            raise RuntimeError(f"Incomplete official dataset download: received {total} of {expected} bytes")
        if validate is not None:
            validate(Path(temporary))
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_roman_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"node_features", "node_labels", "edges", "train_masks", "val_masks", "test_masks"}
        if not required.issubset(archive.files):
            raise ValueError("Roman-empire archive is missing required numeric arrays")
        for name in required:
            if archive[name].dtype.hasobject:
                raise ValueError("Object arrays are not accepted in Roman-empire data")


def load_roman(data_root, allow_download, split_index=0):
    split_index = int(split_index)
    if split_index < 0 or split_index >= 10:
        raise ValueError("Roman-empire split_index must be in 0..9")
    directory = Path(data_root) / "roman_empire" / "raw"
    path = directory / "roman_empire.npz"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".download.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not path.is_file():
            if not allow_download:
                raise FileNotFoundError(f"Roman-empire raw data missing: {path}; set data.download=true for the official download")
            _download_file(ROMAN_URL, path, validate=_validate_roman_archive)
    _validate_roman_archive(path)
    with np.load(path, allow_pickle=False) as archive:
        x, y, edges = archive["node_features"], archive["node_labels"], archive["edges"]
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("Roman-empire raw edges must have shape [E,2]")
        official = {}
        for role, key in (("train", "train_masks"), ("valid", "val_masks"), ("test", "test_masks")):
            masks = archive[key]
            if masks.shape != (10, len(x)) or masks.dtype.kind not in "biu" or np.any((masks != 0) & (masks != 1)):
                raise ValueError("Roman-empire requires ten official boolean split masks [10,N]")
            official[role] = np.flatnonzero(masks[split_index]).astype(np.int64)
    from .data import _file_sha256
    metadata = {**task_metadata("Roman-empire"), "source": "official_heterophilous_graphs_npz",
                "source_file": str(path.resolve()), "source_file_sha256": _file_sha256(path),
                "source_url": ROMAN_URL, "official_split": "official_ten_splits",
                "split_index": split_index, "official_split_count": 10, "is_benchmark_data": True}
    return RawDataset(x, edges.T.copy(), y, official, metadata)


def load_ogb(data_root, allow_download, name="ogbn-arxiv"):
    from .data import _ogb_noninteractive
    try:
        from ogb.nodeproppred import NodePropPredDataset
        from ogb.io.read_graph_raw import read_csv_graph_raw
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Official OGB loading requires ogb and pandas from requirements.txt") from exc
    name = task_spec(name).dataset
    expected_split = {"ogbn-arxiv": "time", "ogbn-products": "sales_ranking", "ogbn-proteins": "species"}[name]
    module = importlib.import_module(NodePropPredDataset.__module__)

    class RawOnlyDataset(NodePropPredDataset):
        def pre_process(self):
            raw = Path(self.root) / "raw"
            if not (raw / "edge.csv.gz").is_file():
                if (Path(self.root) / "processed" / "data_processed").exists():
                    raise FileNotFoundError("OGB processed pickle exists without raw CSV; restore official raw data. Unsafe cache loading is disabled")
                return super().pre_process()
            def extra(key):
                value = str(self.meta_info[key])
                return [] if value == "None" else value.split(",")
            self.graph = read_csv_graph_raw(str(raw), add_inverse_edge=self.meta_info["add_inverse_edge"] == "True",
                                            additional_node_files=extra("additional node files"),
                                            additional_edge_files=extra("additional edge files"))[0]
            self.labels = pd.read_csv(raw / "node-label.csv.gz", compression="gzip", header=None).to_numpy()

        def get_idx_split(self, split_type=None):
            kind = self.meta_info["split"] if split_type is None else split_type
            if kind != expected_split:
                raise ValueError(f"Expected official {name} {expected_split} split, received {kind!r}")
            directory = Path(self.root) / "split" / kind
            return {role: pd.read_csv(directory / f"{role}.csv.gz", compression="gzip", header=None).to_numpy().reshape(-1).astype(np.int64)
                    for role in ("train", "valid", "test")}

    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    with (data_root / ".cross_morse_ogb.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with _ogb_noninteractive(module, allow_download):
            dataset = RawOnlyDataset(name=name, root=str(data_root))
            graph, y = dataset[0]
            official = dataset.get_idx_split()
    edges, n = graph["edge_index"], int(graph["num_nodes"])
    edge_attr = graph.get("edge_feat")
    edge_weight = None
    if name == "ogbn-proteins":
        if edge_attr is None or np.asarray(edge_attr).shape != (edges.shape[1], 8) or np.any(np.asarray(edge_attr) < 0):
            raise ValueError("ogbn-proteins requires eight nonnegative edge-feature channels")
        x = incident_edge_mean(edges, edge_attr, n)
        edge_weight = np.asarray(edge_attr, dtype=np.float32).mean(axis=1)
    else:
        x = graph["node_feat"]
    if x is None or len(x) != n:
        raise ValueError("OGB feature matrix does not match num_nodes")
    metadata = {**task_metadata(name), "source": "official_ogb_raw",
                "ogb_dataset_version": str(dataset.meta_info["version"]), "official_split": expected_split,
                "is_benchmark_data": True, "legacy_pickle_cache_loaded": False,
                "source_url": "https://ogb.stanford.edu/docs/nodeprop/"}
    if name == "ogbn-proteins":
        metadata.update({"node_feature_rule": "arithmetic mean of incident edge features; isolated nodes zero",
                         "edge_weight_rule": "arithmetic mean of eight nonnegative edge-feature channels",
                         "edge_weight_mapping_scope": "explicit implementation convention; scalar reduction not specified in manuscript"})
    return RawDataset(x, edges, y, official, metadata, edge_attr=edge_attr, edge_weight=edge_weight)


def _lrgb_raw_directory(data_root, name, allow_download):
    from .data import _verify_zip
    url, extracted_folder, suffix = LRGB_SOURCES[name]
    base = Path(data_root) / name.lower()
    expected = [f"{split}.{suffix}" for split in ("train", "val", "test")]
    # Support both this adapter's layout and the authors' extracted directory.
    possibilities = (base / "raw", Path(data_root) / extracted_folder, base / extracted_folder)
    for candidate in possibilities:
        if all((candidate / file).is_file() for file in expected):
            return candidate
    if not allow_download:
        raise FileNotFoundError(f"Official {name} raw files missing under {base / 'raw'}; set data.download=true to download")
    base.mkdir(parents=True, exist_ok=True)
    with (base / ".download.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        destination = base / "raw"
        if all((destination / file).is_file() for file in expected):
            return destination
        if destination.exists():
            raise FileExistsError(f"Incomplete raw directory exists; preserve or move it before downloading: {destination}")
        archive_path = base / Path(urllib.parse.urlsplit(url).path).name
        if not archive_path.exists():
            _download_file(url, archive_path, validate=_verify_zip)
        _verify_zip(archive_path)
        with tempfile.TemporaryDirectory(prefix="extract-", dir=base) as temporary:
            temporary = Path(temporary)
            with zipfile.ZipFile(archive_path) as archive:
                for item in archive.infolist():
                    if stat.S_ISLNK(item.external_attr >> 16):
                        raise ValueError("Dataset archive contains a symbolic link")
                archive.extractall(temporary)
            extracted = temporary / extracted_folder
            if not all((extracted / file).is_file() for file in expected):
                raise ValueError("Official LRGB archive does not contain the expected split files")
            os.replace(extracted, destination)
        return destination


def pack_lrgb(name, split_graphs, metadata=None):
    """Pack official raw tuples on CPU; graph boundaries remain explicit."""
    name = task_spec(name).dataset
    if name not in INDEPENDENT_DATASETS:
        raise ValueError("LRGB packing is only defined for the three independent-graph benchmarks")
    xs, ys, edge_chunks, attrs, weights = [], [], [], [], []
    node_ptr, edge_ptr = [0], [0]
    official, graph_count = {}, 0
    for role in ("train", "valid", "test"):
        start = graph_count
        for item in split_graphs[role]:
            if not isinstance(item, (tuple, list)) or len(item) != 4:
                raise ValueError("Each raw LRGB graph must be (x, edge_attr, edge_index, y)")
            x, attr, edges, y = map(_numpy, item)
            if name == "Peptides-func":
                weight = bond_scalar_weights(attr)
                x = atom_one_hot(x)
                y = np.asarray(y, dtype=np.float32).reshape(1, -1)
                if y.shape[1] != 10:
                    raise ValueError("Peptides-func requires ten binary graph targets")
            else:
                if x.ndim != 2 or x.shape[1] != 14:
                    raise ValueError("Superpixel node features must contain fourteen channels")
                if attr.ndim != 2 or attr.shape[1] != 2:
                    raise ValueError("Superpixel edge attributes must contain two channels")
                y = np.asarray(y).reshape(-1)
                if len(y) != len(x):
                    raise ValueError("Superpixel labels must align with graph nodes")
                if name == "COCO-SP":
                    y = remap_coco_labels(y)
                weight = None
            if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
                raise ValueError("LRGB graphs must have nonempty finite node features")
            edges, _, attr, weight = canonical_edges(edges, len(x), attr, weight)
            xs.append(np.asarray(x, dtype=np.float32))
            ys.append(y)
            edge_chunks.append(edges + node_ptr[-1])
            attrs.append(attr)
            if weight is not None:
                weights.append(weight)
            node_ptr.append(node_ptr[-1] + len(x))
            edge_ptr.append(edge_ptr[-1] + edges.shape[1])
            graph_count += 1
        if graph_count == start:
            raise ValueError(f"Empty official {role} split")
        official[role] = np.arange(start, graph_count, dtype=np.int64)
    info = {**task_metadata(name), **(metadata or {}), "num_graphs": graph_count,
            "graph_storage": "CPU disjoint union; execute and sense one complete graph at a time",
            "official_split_axis": "graph", "canonical_edges": True,
            "edge_coalescing": "undirected; remove self-loops; arithmetic mean of duplicate and reciprocal attributes"}
    if name == "Peptides-func":
        info.update({"node_feature_rule": "concatenated one-hot encoding of nine OGB atom categorical fields",
                     "atom_feature_dims": list(ATOM_FEATURE_DIMS), "raw_node_feature_count": 9,
                     "edge_weight_rule": "1 + arithmetic mean of three OGB bond categorical codes",
                     "edge_weight_mapping_scope": "explicit implementation convention; scalar reduction not specified in manuscript"})
    elif name == "COCO-SP":
        info["label_mapping"] = {str(value): index for index, value in enumerate(COCO_LABEL_IDS)}
    return RawDataset(np.concatenate(xs), np.concatenate(edge_chunks, axis=1), np.concatenate(ys), official, info,
                      np.asarray(node_ptr, dtype=np.int64), np.asarray(edge_ptr, dtype=np.int64),
                      np.concatenate(attrs), None if not weights else np.concatenate(weights))


def load_lrgb(data_root, allow_download, name):
    name = task_spec(name).dataset
    directory = _lrgb_raw_directory(data_root, name, allow_download)
    suffix = LRGB_SOURCES[name][2]
    files = {role: directory / f"{raw}.{suffix}" for role, raw in (("train", "train"), ("valid", "val"), ("test", "test"))}
    # One split file is decoded at a time. The raw iterator releases it before
    # loading the next split; packed numeric arrays remain on the CPU.
    class SplitIterator:
        def __getitem__(self, role):
            return iter(read_lrgb_tuples(files[role]))
    from .data import _file_sha256
    info = {"source": "official_lrgb_raw_tensor_tuples", "source_url": LRGB_SOURCES[name][0],
            "source_files_sha256": {str(file.resolve()): _file_sha256(file) for file in files.values()},
            "official_split": "official_train_val_test_graphs", "is_benchmark_data": True,
            "legacy_pickle_cache_loaded": False, "raw_deserialization": "restricted tensor/NumPy tuples; weights_only tensor archives"}
    return pack_lrgb(name, SplitIterator(), info)


def load_benchmark(data_root, name, allow_download=False, split_index=0):
    name = task_spec(name).dataset
    if name.startswith("ogbn-"):
        return load_ogb(data_root, allow_download, name)
    if name == "Roman-empire":
        return load_roman(data_root, allow_download, split_index)
    if name in INDEPENDENT_DATASETS:
        return load_lrgb(data_root, allow_download, name)
    raise ValueError(f"No real adapter for {name}")
