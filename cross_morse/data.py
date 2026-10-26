"""Stage-separated preparation for seven node and graph benchmarks.

The split follows Supplementary Table S2: 60/20/10/10 percent of official
training units become fit/explore/select/confirm; official validation is
admission-only and official test is evaluation-only. An internal subset of
fit selects the frozen checkpoint. Units are nodes in a transductive graph,
or complete independent graphs for the three LRGB datasets. An independent
corpus is stored on the CPU as a disjoint union and exposed through graph().

The paper does not specify exactly how buffers enter propagation. This
reconstruction records k-hop halos around METIS cores as context diagnostics,
but never substitutes an induced core/halo graph for the full graph. Cores are
dependent resampling units, so their bootstrap has no independent-graph claim.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Iterator
import urllib.request
import zipfile

import numpy as np
from scipy import sparse
import torch

from .tasks import task_spec
from .datasets import (RawDataset, BENCHMARKS, INDEPENDENT_DATASETS, canonical_edges,
                       load_benchmark, task_metadata, validate_graph_pointers)


SCHEMA_VERSION = 2
ROLE_NAMES = ("fit_train", "fit_val", "fit", "explore", "select", "confirm", "admit", "test")
LEAF_ROLES = ("fit_train", "fit_val", "explore", "select", "confirm", "admit", "test")
STAGE_LABEL_ROLES = {
    "prepare": frozenset(),
    "train": frozenset({"fit_train", "fit_val", "fit"}),
    "explore": frozenset({"explore"}),
    "fit": frozenset(),  # consumes previously recorded D_explore targets
    "select": frozenset({"explore", "select"}),  # training-only entry diagnostic
    "confirm": frozenset({"confirm"}),
    "admit": frozenset({"admit"}),
    "evaluate": frozenset({"test"}),
}
_OGB_LOCK = threading.RLock()
_DATA_DEFAULTS = {
    "num_clusters": 32,
    "buffer_hops": 2,
    "partitioner": "metis",
    "internal_val_fraction": 0.1,
    "feature_standardization": False,
    "synthetic_nodes": 192,
    "synthetic_features": 16,
    "synthetic_classes": 4,
    "synthetic_graphs": 80,
    "synthetic_nodes_per_graph": 8,
    "split_index": 0,
    "degree_strata": 8,
    "download": False,
    "save_buffers": False,
}


class StageLabelStore:
    """Application-level label boundary; not an OS security sandbox.

    Prepared NPZ files still contain the complete labelled dataset. The runtime
    deliberately exposes only copied role-specific labels, while online policy
    functions receive features/edges alone. Direct tensor indexing is rejected.
    """

    def __init__(self, labels, splits, *, root=None, global_indices=None):
        self._values = labels
        self._splits = splits
        self._stage = None
        self._root = self if root is None else root._root
        self._global_indices = global_indices
        self.accesses = [] if root is None else self._root.accesses

    def __getitem__(self, index):
        raise PermissionError("Unscoped label indexing is disabled; use labels(role) inside its lifecycle stage")

    def numpy(self):
        raise PermissionError("The lifecycle never exposes the complete label array")

    def read(self, role):
        stage = self._root._stage
        if role not in STAGE_LABEL_ROLES.get(stage, ()):
            raise PermissionError(f"Label role {role!r} is unavailable during stage {stage!r}")
        ids = self._splits[role]
        global_ids = ids if self._global_indices is None else self._global_indices[ids]
        self.accesses.append({"stage": stage, "role": role, "nodes": len(ids)})
        return self._root._values[global_ids].detach().cpu().numpy().copy()

    def view(self, global_indices, local_splits):
        if self is not self._root:
            global_indices = self._global_indices[global_indices]
        return StageLabelStore(None, local_splits, root=self._root,
                               global_indices=np.asarray(global_indices, dtype=np.int64))

    @contextmanager
    def _scope(self, stage):
        if stage not in STAGE_LABEL_ROLES:
            raise ValueError(f"Unknown lifecycle stage: {stage}")
        root = self._root
        if root._stage is not None:
            raise RuntimeError("Overlapping label scopes are not allowed")
        root._stage = stage
        try:
            yield self.accesses
        finally:
            root._stage = None


@dataclass
class GraphData:
    """Offline data only. Online code must receive x/edges, not this object."""

    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor | StageLabelStore
    splits: dict[str, np.ndarray]
    cluster_ids: np.ndarray
    metadata: dict[str, Any]
    graph_ptr: np.ndarray | None = None
    edge_ptr: np.ndarray | None = None
    graph_splits: dict[str, np.ndarray] | None = None
    edge_attr: torch.Tensor | None = None
    edge_weight: torch.Tensor | None = None
    global_sample_ids: np.ndarray | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.x.shape[1])

    @property
    def num_classes(self) -> int:
        return int(self.metadata["num_classes"])

    @property
    def num_samples(self) -> int:
        return int(self.metadata.get("num_samples", self.num_nodes))

    @property
    def num_graphs(self) -> int:
        return 1 if self.graph_ptr is None else len(self.graph_ptr) - 1

    @property
    def node_to_graph(self) -> np.ndarray:
        if self.graph_ptr is None:
            return np.zeros(self.num_nodes, dtype=np.int64)
        return np.repeat(np.arange(self.num_graphs, dtype=np.int64), np.diff(self.graph_ptr))

    def graph(self, graph_id: int) -> "GraphData":
        """A complete local graph sharing its corpus's stage authorization.

        No labels are read when constructing the view. ``global_sample_ids``
        maps the local head's rows to the corpus labels/predictions, including
        the single graph-level row for Peptides-func.
        """
        if self.graph_ptr is None or self.edge_ptr is None or self.graph_splits is None:
            raise ValueError("graph() is only defined for an independent-graph corpus")
        if int(graph_id) != graph_id or not 0 <= int(graph_id) < self.num_graphs:
            raise IndexError("Graph ID outside this corpus")
        graph_id = int(graph_id)
        start, stop = (int(v) for v in self.graph_ptr[graph_id:graph_id+2])
        first, last = (int(v) for v in self.edge_ptr[graph_id:graph_id+2])
        graph_level = self.metadata.get("label_axis") == "graph"
        parent_ids = np.array([graph_id], dtype=np.int64) if graph_level else np.arange(start, stop, dtype=np.int64)
        local_ids = np.arange(len(parent_ids), dtype=np.int64)
        membership = {role: bool(np.any(self.graph_splits[role] == graph_id)) for role in ROLE_NAMES}
        splits = {role: local_ids.copy() if membership[role] else np.empty(0, dtype=np.int64) for role in ROLE_NAMES}
        # Protect the parent before making a view, including views requested
        # before a lifecycle scope begins. Child scope and parent scope coincide.
        self.protect_labels()
        labels = self.y.view(parent_ids, splits)
        global_ids = parent_ids if self.global_sample_ids is None else self.global_sample_ids[parent_ids]
        global_graph_id = int(self.metadata.get("graph_id", graph_id)) if self.num_graphs == 1 else graph_id
        metadata = {**self.metadata, "num_nodes": stop-start, "num_samples": len(parent_ids),
                    "num_graphs": 1, "edge_entries": last-first, "independent_graphs": True,
                    "graph_id": global_graph_id, "parent_fingerprint": self.metadata.get("fingerprint"),
                    "split_sizes": {role: len(ids) for role, ids in splits.items()}}
        if self.metadata.get("fingerprint"):
            metadata["fingerprint"] = hashlib.sha256(f"{self.metadata['fingerprint']}:graph:{graph_id}".encode()).hexdigest()
        return GraphData(self.x[start:stop], self.edge_index[:, first:last] - start, labels,
                         splits, np.full(len(parent_ids), global_graph_id, dtype=np.int64), metadata,
                         graph_ptr=np.array([0, stop-start], dtype=np.int64),
                         edge_ptr=np.array([0, last-first], dtype=np.int64),
                         graph_splits={role: np.array([0], dtype=np.int64) if membership[role] else np.empty(0, dtype=np.int64) for role in ROLE_NAMES},
                         edge_attr=None if self.edge_attr is None else self.edge_attr[first:last],
                         edge_weight=None if self.edge_weight is None else self.edge_weight[first:last],
                         global_sample_ids=global_ids.copy())

    def labels(self, role: str) -> np.ndarray:
        """Return a copy containing only the labels assigned to this role."""
        if role not in self.splits:
            raise KeyError(f"Unknown label role {role!r}; expected one of {ROLE_NAMES}.")
        if isinstance(self.y, StageLabelStore):
            return self.y.read(role)
        return self.y[self.splits[role]].detach().cpu().numpy().copy()

    def protect_labels(self):
        """Seal a prepared offline fixture when it enters the runtime pipeline."""
        if not isinstance(self.y, StageLabelStore):
            self.y = StageLabelStore(self.y, self.splits)
        return self

    @contextmanager
    def _label_scope(self, stage):
        self.protect_labels()
        with self.y._scope(stage) as accesses:
            yield accesses


def role_labels(data, role):
    """One accessor for production GraphData and narrow unit-test fixtures."""
    if hasattr(data, "labels"):
        return data.labels(role)
    # A small fixture may implement only indexed labels. Production GraphData
    # always takes the guarded branch above; this does not unseal its store.
    return data.y[data.splits[role]].detach().cpu().numpy().copy()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _array_sha256(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(_json_bytes({"shape": list(a.shape), "dtype": a.dtype.str}))
    if a.size:
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Uncompressed ZIP entries avoid large compression CPU/memory overhead."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _effective_spec(config: dict[str, Any]) -> dict[str, Any]:
    data = dict(_DATA_DEFAULTS)
    data.update(config.get("data", {}))
    if data.get("graph_npz"):
        data["graph_npz"] = str(Path(data["graph_npz"]).expanduser().resolve())
    # Download authorization affects access, not the statistical preparation.
    data.pop("download", None)
    dataset = task_spec(config.get("dataset", "ogbn-arxiv")).dataset
    benchmark = task_spec(config).dataset
    spec = {"dataset": dataset, "benchmark_dataset": benchmark, "seed": int(config.get("seed", 1)), "data": data}
    if dataset not in {*BENCHMARKS, "synthetic"}:
        raise ValueError("Dataset must be a registered benchmark or an explicit synthetic fixture")
    if dataset != "synthetic" and benchmark != dataset:
        raise ValueError("benchmark_dataset can override task semantics only for synthetic fixtures")
    if not 0 < float(data["internal_val_fraction"]) < 1:
        raise ValueError("data.internal_val_fraction must lie strictly between zero and one.")
    for key in ("num_clusters", "buffer_hops", "synthetic_nodes", "synthetic_features", "synthetic_classes",
                "synthetic_graphs", "synthetic_nodes_per_graph", "degree_strata"):
        if int(data[key]) != data[key] or int(data[key]) < (0 if key == "buffer_hops" else 1):
            raise ValueError(f"data.{key} must be a valid nonnegative/positive integer.")
    if int(data["split_index"]) != data["split_index"] or not 0 <= int(data["split_index"]) < 10:
        raise ValueError("data.split_index must be an integer in 0..9")
    if benchmark != "Roman-empire" and int(data["split_index"]) != 0:
        raise ValueError("data.split_index is only meaningful for Roman-empire")
    return spec


def _indices(values: Any, name: str, n: int) -> np.ndarray:
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array.")
    a = np.asarray(a, dtype=np.int64)
    if not len(a) or a.min() < 0 or a.max() >= n:
        raise ValueError(f"{name} is empty or contains out-of-range node indices.")
    if len(np.unique(a)) != len(a):
        raise ValueError(f"{name} contains duplicate node indices.")
    return np.ascontiguousarray(a)


def _validate_official_splits(splits: dict[str, Any], n: int) -> dict[str, np.ndarray]:
    out = {name: _indices(splits[name], f"official_{name}", n) for name in ("train", "valid", "test")}
    all_nodes = np.concatenate([out[name] for name in ("train", "valid", "test")])
    if len(all_nodes) != n or not np.array_equal(np.sort(all_nodes), np.arange(n, dtype=np.int64)):
        raise ValueError("Official train/validation/test splits must be disjoint and cover every node exactly once.")
    return out


def _validate_roles(splits: dict[str, Any], n: int) -> dict[str, np.ndarray]:
    if set(splits) != set(ROLE_NAMES):
        raise ValueError(f"Prepared split roles must be exactly {ROLE_NAMES}.")
    out = {name: _indices(splits[name], name, n) for name in ROLE_NAMES}
    all_nodes = np.concatenate([out[name] for name in LEAF_ROLES])
    if len(all_nodes) != n or not np.array_equal(np.sort(all_nodes), np.arange(n, dtype=np.int64)):
        raise ValueError("Prepared leaf roles overlap or do not cover every node.")
    if not np.array_equal(np.sort(out["fit"]), np.sort(np.concatenate([out["fit_train"], out["fit_val"]]))):
        raise ValueError("fit must be the disjoint union of fit_train and fit_val.")
    return out


def _make_roles(official: dict[str, np.ndarray], n: int, seed: int, fraction: float) -> dict[str, np.ndarray]:
    train = np.random.default_rng(seed).permutation(np.sort(official["train"]))
    cuts = [int(len(train) * f) for f in (0.6, 0.8, 0.9)]
    fit, explore, select, confirm = np.split(train, cuts)
    if min(map(len, (fit, explore, select, confirm))) == 0 or len(fit) < 2:
        raise ValueError("Too few training nodes for the separated 60/20/10/10 protocol.")
    fit_shuffled = np.random.default_rng(np.random.SeedSequence([seed, 137])).permutation(fit)
    val_count = max(1, int(len(fit) * fraction))
    if val_count >= len(fit):
        raise ValueError("Internal fit validation leaves no fit_train nodes.")
    roles = {
        "fit_train": fit_shuffled[val_count:], "fit_val": fit_shuffled[:val_count], "fit": fit,
        "explore": explore, "select": select, "confirm": confirm,
        "admit": official["valid"].copy(), "test": official["test"].copy(),
    }
    return _validate_roles(roles, n)


def _validate_arrays(x: Any, edges: Any, labels: Any, num_classes: int | None = None,
                     *, task_type="multiclass", label_count=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    features = np.asarray(x, dtype=np.float32)
    if features.ndim != 2 or min(features.shape) <= 0 or not np.isfinite(features).all():
        raise ValueError("x must be a nonempty finite two-dimensional float feature matrix.")
    n = len(features)
    edge_index = np.asarray(edges)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype.kind not in "iu":
        raise ValueError("edge_index must have shape [2, E] and an integer dtype.")
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= n):
        raise ValueError("edge_index contains out-of-range node IDs.")
    y = np.asarray(labels)
    samples = n if label_count is None else int(label_count)
    if task_type == "multilabel":
        if y.ndim != 2 or y.shape[0] != samples or not y.shape[1] or y.dtype.kind not in "biuf":
            raise ValueError("Multilabel y must have shape [prediction rows, tasks]")
        if np.isinf(y).any() or np.any(np.isfinite(y) & (y >= 0) & (y != 0) & (y != 1)):
            raise ValueError("Multilabel targets must be binary or missing (NaN/negative sentinel)")
        y = np.asarray(y, dtype=np.float32)
        y[y < 0] = np.nan
        classes = int(y.shape[1]) if num_classes is None else int(num_classes)
        if classes != y.shape[1]:
            raise ValueError("Multilabel task count differs from the declared output dimension")
        return np.ascontiguousarray(features), np.ascontiguousarray(edge_index), np.ascontiguousarray(y), classes
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1 or len(y) != samples or not np.isfinite(y).all() or not np.equal(y, np.floor(y)).all():
        raise ValueError("y must contain one finite integer class label for every prediction row.")
    y = np.asarray(y, dtype=np.int64)
    classes = int(y.max()) + 1 if num_classes is None else int(num_classes)
    if classes < 2 or y.min() < 0 or y.max() >= classes:
        raise ValueError("Labels are outside the declared multiclass classification range.")
    return np.ascontiguousarray(features), np.ascontiguousarray(edge_index), np.ascontiguousarray(y), classes


def _simple_undirected(edges: np.ndarray, n: int) -> tuple[np.ndarray, sparse.csr_matrix]:
    # Boolean COO avoids integer overflow when many duplicate edges occur.
    adjacency = sparse.coo_matrix((np.ones(edges.shape[1], dtype=bool), (edges[0], edges[1])), shape=(n, n)).tocsr()
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(False)
    adjacency.eliminate_zeros()
    adjacency.sum_duplicates()
    adjacency.sort_indices()
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(adjacency.indptr))
    canonical = np.stack((rows, adjacency.indices.astype(np.int64)), axis=0)
    return np.ascontiguousarray(canonical), adjacency


def _synthetic(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    d, seed = spec["data"], spec["seed"]
    n, features, classes = (int(d[k]) for k in ("synthetic_nodes", "synthetic_features", "synthetic_classes"))
    if n < max(32, classes * 2):
        raise ValueError("Synthetic fixtures require at least max(32, 2*num_classes) nodes.")
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.arange(n) % classes).astype(np.int64)
    prototypes = rng.normal(size=(classes, features))
    x = (prototypes[y] + rng.normal(scale=0.8, size=(n, features))).astype(np.float32)
    # Ring plus local-class/random neighbors, with no artificial data leakage
    # beyond the declared synthetic graph generation process.
    src = np.repeat(np.arange(n, dtype=np.int64), 4)
    dst = np.empty((n, 4), dtype=np.int64)
    dst[:, 0] = (np.arange(n) + 1) % n
    dst[:, 1] = rng.integers(0, n, size=n)
    for label in range(classes):
        members = np.flatnonzero(y == label)
        dst[members, 2:] = rng.choice(members, size=(len(members), 2))
    edges = np.stack((src, dst.reshape(-1)))
    indices = rng.permutation(n).astype(np.int64)
    a, b = int(0.6*n), int(0.8*n)
    official = {"train": indices[:a], "valid": indices[a:b], "test": indices[b:]}
    return x, edges, y, official, {"source": "synthetic_fixture", "num_classes": classes, "is_benchmark_data": False}


def _synthetic_source(spec):
    """Small deterministic fixtures exercising each benchmark's real task axis."""
    benchmark, d = spec["benchmark_dataset"], spec["data"]
    task = task_spec(benchmark)
    rng = np.random.default_rng(np.random.SeedSequence([spec["seed"], 811]))
    classes = int(d["synthetic_classes"])
    metadata = {**task_metadata(benchmark, classes), "source": "synthetic_fixture",
                "is_benchmark_data": False, "scientific_run": False}
    if benchmark not in INDEPENDENT_DATASETS:
        # A binary-task count is not a latent multiclass vocabulary: retain a
        # small prototype fixture even when exercising all 112 protein outputs.
        latent_spec = spec if task.task_type != "multilabel" else {**spec, "data": {**d, "synthetic_classes": min(4, classes)}}
        x, edges, y, official, _ = _synthetic(latent_spec)
        attr = weight = None
        if task.task_type == "multilabel":
            y = (rng.normal(size=(len(x), classes)) + x[:, :1] * 0.25 > 0).astype(np.float32)
            attr = rng.uniform(0.05, 1, size=(edges.shape[1], 8)).astype(np.float32)
            weight = attr.mean(1)
            metadata["edge_weight_rule"] = "synthetic mean of eight nonnegative edge channels"
        return RawDataset(x, edges, y, official, metadata, edge_attr=attr, edge_weight=weight)
    graphs, nodes, features = (int(d[k]) for k in ("synthetic_graphs", "synthetic_nodes_per_graph", "synthetic_features"))
    if graphs < 32 or nodes < 2:
        raise ValueError("Independent-graph fixtures require at least 32 graphs and 2 nodes per graph")
    x = rng.normal(size=(graphs * nodes, features)).astype(np.float32)
    ptr = np.arange(graphs + 1, dtype=np.int64) * nodes
    edges = []
    for g in range(graphs):
        ids = np.arange(nodes, dtype=np.int64)
        edges.append(np.stack((ids, (ids+1) % nodes)) + ptr[g])
    edges = np.concatenate(edges, axis=1)
    eptr = np.arange(graphs + 1, dtype=np.int64) * nodes
    if task.label_axis == "graph":
        y = (rng.normal(size=(graphs, classes)) > 0).astype(np.float32)
        attr = np.stack((rng.integers(0, 5, edges.shape[1]), rng.integers(0, 6, edges.shape[1]), rng.integers(0, 2, edges.shape[1])), axis=1)
        weight = 1 + attr.astype(np.float32).mean(1)
    else:
        y = rng.integers(0, classes, graphs * nodes, dtype=np.int64)
        attr = rng.uniform(size=(edges.shape[1], 2)).astype(np.float32)
        weight = None
    order = rng.permutation(graphs).astype(np.int64)
    a, b = int(0.6 * graphs), int(0.8 * graphs)
    official = {"train": order[:a], "valid": order[a:b], "test": order[b:]}
    metadata.update({"num_graphs": graphs, "official_split_axis": "graph",
                     "graph_storage": "CPU disjoint union; execute and sense one complete graph at a time"})
    return RawDataset(x, edges, y, official, metadata, ptr, eptr, attr, weight)


def _verify_zip(path: Path) -> None:
    """Reject incomplete archives before OGB extracts them into its cache."""
    try:
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                name = Path(item.filename.replace("\\", "/"))
                if name.is_absolute() or ".." in name.parts:
                    raise ValueError("Dataset ZIP contains an unsafe member path.")
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"Dataset ZIP CRC check failed for {bad}.")
    except zipfile.BadZipFile as exc:
        raise ValueError("Dataset ZIP is incomplete or malformed.") from exc


def _download_archive(url: str, folder: str, log: bool = True) -> str:
    """Stream to EOF, check Content-Length and ZIP CRC, then atomically commit.

    OGB 1.3.6's fixed read-loop count can leave a partial nonzero cache file
    reusable after an incomplete transfer. Only verified archives are reused.
    """
    target = Path(folder) / url.rsplit("/", 1)[-1]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            _verify_zip(target)
            return str(target)
        except ValueError:
            quarantined = target.with_name(target.name + f".invalid.{time.time_ns()}")
            target.rename(quarantined)
            print(f"[DATA] Preserved invalid cached archive as {quarantined}", flush=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".part", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream, urllib.request.urlopen(url, timeout=60) as response:
            expected_text = response.headers.get("Content-Length")
            expected = int(expected_text) if expected_text is not None else None
            received = 0
            if log:
                print(f"[DATA] Downloading {url}; expected_bytes={expected}", flush=True)
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                stream.write(block)
                received += len(block)
            stream.flush()
            os.fsync(stream.fileno())
        if expected is not None and received != expected:
            raise RuntimeError(f"Incomplete official OGB download: received {received:,} of {expected:,} bytes from {url}. No dataset was committed; retry on a network that can deliver the complete archive.")
        _verify_zip(Path(temporary))
        os.replace(temporary, target)
        return str(target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _ogb_noninteractive(module: Any, allow_download: bool) -> Iterator[None]:
    """Patch only OGB's module globals, never builtins.input or torch.load."""
    with _OGB_LOCK:
        original_decision = module.decide_download
        missing = object()
        original_input = getattr(module, "input", missing)
        original_download = getattr(module, "download_url", missing)

        def download_decision(url: str) -> bool:
            if not allow_download:
                raise FileNotFoundError("OGB raw data are missing. Set data.download=true to authorize a noninteractive official download, or supply data.graph_npz.")
            print(f"[DATA] Authorized official OGB download: {url}", flush=True)
            return True

        def reject_update(prompt: str = "") -> str:
            raise RuntimeError("OGB reports an outdated or unversioned dataset directory. Automatic deletion/update is disabled. Preserve or move that directory explicitly, then download a current official copy. " + prompt)

        def validated_download(url: str, folder: str, log: bool = True) -> str:
            if not allow_download:
                raise PermissionError("Official dataset download requires data.download=true.")
            return _download_archive(url, folder, log)

        module.decide_download = download_decision
        module.input = reject_update
        module.download_url = validated_download
        try:
            yield
        finally:
            module.decide_download = original_decision
            if original_input is missing:
                delattr(module, "input")
            else:
                module.input = original_input
            if original_download is missing:
                delattr(module, "download_url")
            else:
                module.download_url = original_download


def _load_ogb(data_root: Path, allow_download: bool):
    """Compatibility entry point for the arxiv raw reader and existing callers."""
    from .datasets import load_ogb as load_raw_ogb
    source = load_raw_ogb(data_root, allow_download, "ogbn-arxiv")
    return source.x, source.edge_index, source.y, source.official_splits, source.metadata


def _load_source(data_root: Path, spec: dict[str, Any], allow_download: bool):
    path_text = spec["data"].get("graph_npz")
    if path_text:
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"Trusted raw graph NPZ not found: {path}")
        required = {"x", "edge_index", "y", "train_idx", "valid_idx", "test_idx"}
        with np.load(path, allow_pickle=False) as archive:
            if not required.issubset(archive.files):
                raise ValueError(f"Raw graph NPZ must contain {sorted(required)}; missing {sorted(required-set(archive.files))}.")
            x, edges, y = (archive[name] for name in ("x", "edge_index", "y"))
            official = {name: archive[f"{name}_idx"] for name in ("train", "valid", "test")}
            extras = {key: archive[key] if key in archive.files else None for key in ("graph_ptr", "edge_ptr", "edge_attr", "edge_weight")}
        task = task_spec(spec["benchmark_dataset"])
        outputs = task.num_outputs
        if spec["dataset"] == "synthetic" and task.task_type == "multiclass":
            outputs = int(np.asarray(y).max()) + 1
        metadata = {**task_metadata(task.dataset, outputs), "source": "trusted_raw_npz", "source_file": str(path),
                    "source_file_sha256": _file_sha256(path), "is_benchmark_data": spec["dataset"] != "synthetic",
                    "official_split": "provided train/valid/test indices; official provenance asserted by exporter"}
        if (extras["graph_ptr"] is not None) != (task.dataset in INDEPENDENT_DATASETS):
            raise ValueError("graph_npz must include graph_ptr/edge_ptr exactly for independent-graph benchmarks")
        return RawDataset(x, edges, y, official, metadata, **extras)
    if spec["dataset"] == "synthetic":
        return _synthetic_source(spec)
    return load_benchmark(data_root, spec["dataset"], allow_download, int(spec["data"]["split_index"]))


def _partition(adjacency: sparse.csr_matrix, spec: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    d, n = spec["data"], adjacency.shape[0]
    parts = int(d["num_clusters"])
    if parts > n:
        raise ValueError("num_clusters cannot exceed the number of nodes.")
    kind = d["partitioner"]
    if kind == "balanced":
        if spec["dataset"] != "synthetic":
            raise ValueError("Balanced random cores are allowed only for explicitly synthetic fixtures; use METIS for ogbn-arxiv.")
        order = np.random.default_rng(np.random.SeedSequence([spec["seed"], 919])).permutation(n)
        clusters = np.empty(n, dtype=np.int64)
        clusters[order] = np.arange(n, dtype=np.int64) % parts
        info = {"partitioner": "balanced_synthetic_only", "partition_seed": spec["seed"]}
    elif kind == "degree_stratified":
        # Table S1 specifies degree-stratified proteins cores but not a unique
        # partition algorithm. Balance rank-based degree strata across cores;
        # retain this exact rule in provenance rather than claiming METIS.
        degree = np.diff(adjacency.indptr)
        ranked = np.lexsort((np.arange(n), degree))
        strata = min(int(d["degree_strata"]), n)
        rng = np.random.default_rng(np.random.SeedSequence([spec["seed"], 933]))
        clusters = np.empty(n, dtype=np.int64)
        cursor = 0
        for group in np.array_split(ranked, strata):
            group = rng.permutation(group)
            clusters[group] = (np.arange(len(group)) + cursor) % parts
            cursor = (cursor + len(group)) % parts
        info = {"partitioner": "degree_stratified", "partition_seed": spec["seed"],
                "degree_strata": strata, "degree_rule": "unweighted canonical neighbor count",
                "partition_algorithm": "degree-rank quantiles; seeded permutation within each stratum; balanced round-robin core assignment",
                "partition_scope": "explicit implementation choice; manuscript does not specify degree-stratification algorithm",
                "connected_cores_required": False}
    elif kind == "metis":
        try:
            import pymetis
        except ImportError as exc:
            raise ImportError("METIS partitioning requires pymetis. No random partition fallback is allowed for ogbn-arxiv.") from exc
        options = pymetis.Options(seed=int(spec["seed"]) % 2147483647)
        if hasattr(pymetis, "CSRAdjacency"):
            dtype = pymetis.zero_copy_dtype()
            csr = pymetis.CSRAdjacency(np.asarray(adjacency.indptr, dtype=dtype), np.asarray(adjacency.indices, dtype=dtype))
            result = pymetis.part_graph(parts, adjacency=csr, options=options)
        else:
            result = pymetis.part_graph(parts, xadj=adjacency.indptr, adjncy=adjacency.indices, options=options)
        cuts, membership = result
        clusters = np.asarray(membership, dtype=np.int64)
        info = {"partitioner": "metis", "partition_seed": spec["seed"], "undirected_cut_edges": int(cuts)}
    else:
        raise ValueError("Transductive data.partitioner must be 'metis', 'degree_stratified', or 'balanced' (synthetic only).")
    if clusters.shape != (n,) or not np.array_equal(np.unique(clusters), np.arange(parts)):
        raise ValueError("Partitioner returned missing/invalid cores; every registered core must be nonempty.")
    return np.ascontiguousarray(clusters), info


def _buffers(adjacency: sparse.csr_matrix, clusters: np.ndarray, hops: int, save: bool) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    core_sizes, halo_sizes, context_sizes = [], [], []
    core_chunks, context_chunks = [], []
    for cluster in range(int(clusters.max()) + 1):
        core = np.flatnonzero(clusters == cluster).astype(np.int64)
        visited = np.zeros(len(clusters), dtype=bool)
        visited[core] = True
        frontier = core
        for _ in range(hops):
            if not len(frontier):
                break
            neighbors = np.unique(adjacency[frontier].indices)
            frontier = neighbors[~visited[neighbors]]
            visited[frontier] = True
        context = np.flatnonzero(visited).astype(np.int64)
        core_sizes.append(len(core))
        context_sizes.append(len(context))
        halo_sizes.append(len(context) - len(core))
        if save:
            core_chunks.append(core)
            context_chunks.append(context)
    info = {"buffer_hops": hops, "core_sizes": core_sizes, "halo_sizes": halo_sizes,
            "core_plus_halo_sizes": context_sizes, "buffer_membership_saved": save,
            "propagation_context": "full_graph", "buffer_role": "context_diagnostic_only",
            "statistical_unit": "METIS core" if len(clusters) else "core",
            "independent_graph_guarantee": False,
            "interpretation": "One transductive graph; cores are dependent descriptive resampling units. Halos do not induce or truncate the propagation graph."}
    packed = {}
    if save:
        packed = {"core_nodes": np.concatenate(core_chunks), "core_indptr": np.r_[0, np.cumsum(core_sizes)].astype(np.int64),
                  "context_nodes": np.concatenate(context_chunks), "context_indptr": np.r_[0, np.cumsum(context_sizes)].astype(np.int64)}
    return info, packed


def _fingerprint(metadata: dict[str, Any]) -> str:
    value = {key: metadata.get(key) for key in (
        "schema_version", "preparation_config", "array_hashes", "official_split_hashes",
        "role_hashes", "graph_role_hashes", "cluster_sha256", "task_type", "label_axis",
        "metric", "num_outputs", "independent_graphs", "edge_weight_rule", "node_feature_rule",
        "partitioner", "partition_algorithm", "split_index")}
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _graph_roles_to_rows(graph_roles, ptr, graph_level):
    if graph_level:
        return {role: ids.copy() for role, ids in graph_roles.items()}
    return {role: np.concatenate([np.arange(ptr[g], ptr[g+1], dtype=np.int64) for g in ids])
            for role, ids in graph_roles.items()}


def _canonical_source(source):
    n, edges = len(source.x), source.edge_index
    if source.graph_ptr is None:
        if source.edge_attr is None and source.edge_weight is None:
            edges, adjacency = _simple_undirected(edges, n)
            return edges, adjacency, None, None, None, None
        edges, adjacency, attr, weight = canonical_edges(edges, n, source.edge_attr, source.edge_weight)
        return edges, adjacency, attr, weight, None, None
    ptr, old_eptr = validate_graph_pointers(source.graph_ptr, source.edge_ptr, n, edges)
    if source.metadata.get("canonical_edges"):
        return edges, None, source.edge_attr, source.edge_weight, ptr, old_eptr
    chunks, attrs, weights, eptr = [], [], [], [0]
    for g in range(len(ptr)-1):
        begin, end = old_eptr[g:g+2]
        attr = None if source.edge_attr is None else source.edge_attr[begin:end]
        weight = None if source.edge_weight is None else source.edge_weight[begin:end]
        local, _, attr, weight = canonical_edges(edges[:, begin:end]-ptr[g], int(ptr[g+1]-ptr[g]), attr, weight)
        chunks.append(local + ptr[g])
        if attr is not None:
            attrs.append(attr)
        if weight is not None:
            weights.append(weight)
        eptr.append(eptr[-1]+local.shape[1])
    return (np.concatenate(chunks, axis=1), None, None if not attrs else np.concatenate(attrs),
            None if not weights else np.concatenate(weights), ptr, np.asarray(eptr, dtype=np.int64))


def _graph_data(arrays, roles, clusters, metadata, graph_roles=None):
    tensor = lambda name: None if name not in arrays else torch.from_numpy(arrays[name])
    return GraphData(tensor("x"), tensor("edge_index"), tensor("y"), roles, clusters, metadata,
                     graph_ptr=arrays.get("graph_ptr"), edge_ptr=arrays.get("edge_ptr"),
                     graph_splits=graph_roles, edge_attr=tensor("edge_attr"), edge_weight=tensor("edge_weight"),
                     global_sample_ids=np.arange(len(arrays["y"]), dtype=np.int64))


def prepare_data(data_root: Path, output_dir: Path, config: dict[str, Any]) -> GraphData:
    """Prepare once, seal labels/graph boundaries in provenance, and return CPU data."""
    data_root, output_dir = Path(data_root), Path(output_dir)
    prepared = output_dir / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    spec = _effective_spec(config)
    with (prepared / ".prepare.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        metadata_path = prepared / "metadata.json"
        if metadata_path.is_file():
            old = json.loads(metadata_path.read_text())
            if old.get("preparation_config") != spec:
                raise ValueError("Prepared data were created with a different preprocessing/split configuration. Use a new output directory.")
            source_file = old.get("source_file")
            if source_file and Path(source_file).exists() and _file_sha256(Path(source_file)) != old.get("source_file_sha256"):
                raise ValueError("The source graph NPZ changed since preparation; use a new output directory.")
            return load_data(output_dir)
        source = _load_source(data_root, spec, bool(config.get("data", {}).get("download", False)))
        independent = source.graph_ptr is not None
        if independent != bool(source.metadata.get("independent_graphs")):
            raise ValueError("Dataset task and independent-graph pointers disagree")
        task = task_spec(spec["benchmark_dataset"], source.metadata)
        graph_count = len(source.graph_ptr)-1 if independent else 1
        samples = graph_count if task.label_axis == "graph" else len(source.x)
        x, raw_edges, y, classes = _validate_arrays(source.x, source.edge_index, source.y,
                                                   source.metadata.get("num_classes"),
                                                   task_type=task.task_type, label_count=samples)
        source.x, source.edge_index, source.y = x, raw_edges, y
        n, features = x.shape
        expected = {"ogbn-arxiv": (169343, 128, 40), "ogbn-products": (2449029, 100, 47),
                    "ogbn-proteins": (132534, 8, 112), "Roman-empire": (22662, 300, 18)}
        if spec["dataset"] in expected and (n, features, classes) != expected[spec["dataset"]]:
            raise ValueError(f"{spec['dataset']} must have nodes/features/outputs {expected[spec['dataset']]}; received {(n, features, classes)}. Use dataset='synthetic' for fixtures.")
        graph_counts = {"PascalVOC-SP": 11355, "COCO-SP": 123286, "Peptides-func": 15535}
        if spec["dataset"] in graph_counts and graph_count != graph_counts[spec["dataset"]]:
            raise ValueError(f"{spec['dataset']} must retain all {graph_counts[spec['dataset']]} official graphs; received {graph_count}. Use dataset='synthetic' for fixtures, not a partial benchmark cache.")
        official = _validate_official_splits(source.official_splits, graph_count if independent else n)
        unit_roles = _make_roles(official, graph_count if independent else n, spec["seed"], float(spec["data"]["internal_val_fraction"]))
        graph_roles = unit_roles if independent else None
        source_arrays = {"x": x, "edge_index": raw_edges, "y": y}
        for key in ("graph_ptr", "edge_ptr", "edge_attr", "edge_weight"):
            if getattr(source, key) is not None:
                source_arrays[key] = np.asarray(getattr(source, key))
        source_hashes = {key: _array_sha256(value) for key, value in source_arrays.items()}
        edge_index, adjacency, edge_attr, edge_weight, ptr, eptr = _canonical_source(source)
        if independent:
            roles = _validate_roles(_graph_roles_to_rows(unit_roles, ptr, task.label_axis == "graph"), samples)
            clusters = np.arange(graph_count, dtype=np.int64) if task.label_axis == "graph" else np.repeat(np.arange(graph_count, dtype=np.int64), np.diff(ptr))
            partition_metadata = {"partitioner": "graphs", "num_statistical_units": graph_count,
                                  "partition_algorithm": "one complete official graph per unit; no cross-graph propagation"}
            buffer_metadata = {"buffer_hops": 0, "buffer_role": "not_applicable", "buffer_membership_saved": False,
                               "propagation_context": "one_complete_graph_per_episode", "statistical_unit": "complete graph",
                               "independent_graph_guarantee": False,
                               "interpretation": "Complete graphs are the benchmark sampling units; statistical independence remains an assumption."}
            packed = {}
            k = graph_count
        else:
            roles = unit_roles
            clusters, partition_metadata = _partition(adjacency, spec)
            buffer_metadata, packed = _buffers(adjacency, clusters, int(spec["data"]["buffer_hops"]), bool(spec["data"]["save_buffers"]))
            names = {"metis": "METIS core", "degree_stratified": "degree-stratified core", "balanced_synthetic_only": "balanced synthetic core"}
            buffer_metadata["statistical_unit"] = names[partition_metadata["partitioner"]]
            k = int(spec["data"]["num_clusters"])
        normalization = {}
        if spec["data"]["feature_standardization"]:
            fit_nodes = roles["fit_train"]
            if independent and task.label_axis == "graph":
                fit_nodes = np.concatenate([np.arange(ptr[g], ptr[g+1], dtype=np.int64) for g in graph_roles["fit_train"]])
            train_x = x[fit_nodes].astype(np.float64)
            mean, scale = train_x.mean(0), train_x.std(0)
            scale[scale < 1e-12] = 1.0
            x = ((x - mean.astype(np.float32)) / scale.astype(np.float32)).astype(np.float32)
            if not np.isfinite(x).all():
                raise ValueError("Feature standardization generated non-finite values.")
            normalization = {"mean": mean, "scale": scale}
        arrays = {"x": x, "edge_index": edge_index, "y": y}
        for key, value in (("graph_ptr", ptr), ("edge_ptr", eptr), ("edge_attr", edge_attr), ("edge_weight", edge_weight)):
            if value is not None:
                arrays[key] = np.ascontiguousarray(value)
        files = {"graph.npz": arrays, "splits.npz": roles, "clusters.npz": {"cluster_ids": clusters},
                 "official_splits.npz": official}
        if independent:
            files["graph_splits.npz"] = graph_roles
        if normalization:
            files["normalization.npz"] = normalization
        if packed:
            files["buffers.npz"] = packed
        for filename, values in files.items():
            _atomic_npz(prepared / filename, **values)
        metadata = {
            **source.metadata, **partition_metadata,
            "schema_version": SCHEMA_VERSION, "dataset": spec["dataset"], "benchmark_dataset": spec["benchmark_dataset"],
            "preparation_config": spec, "num_nodes": int(n), "num_features": int(features), "num_classes": int(classes),
            "num_outputs": int(classes), "num_samples": int(samples), "num_graphs": int(graph_count),
            "task_type": task.task_type, "label_axis": task.label_axis, "prediction_level": task.label_axis,
            "metric": task.metric, "independent_graphs": independent,
            "input_directed_edge_entries": int(raw_edges.shape[1]), "edge_entries": int(edge_index.shape[1]),
            "undirected_edges": int(edge_index.shape[1] // 2),
            "isolated_nodes": int((np.bincount(edge_index[0], minlength=n) == 0).sum()),
            "graph_rule": "simple_undirected_no_self_loops; operator handles isolated nodes as identity rows",
            "edge_attribute_rule": "arithmetic mean of duplicate and reciprocal edge attributes, aligned with canonical edges",
            "split_rule": "seeded permutation of official train; floor cuts at 60%,80%,90%; remaining units in confirm",
            "split_unit": "graph" if independent else "node", "official_split_axis": "graph" if independent else "node",
            "internal_val_rule": "independent seeded permutation within fit; floor(fraction*fit), minimum one complete unit",
            "split_sizes": {key: int(len(value)) for key, value in roles.items()},
            "graph_split_sizes": None if graph_roles is None else {key: int(len(value)) for key, value in graph_roles.items()},
            "role_core_counts": {key: np.bincount(clusters[value], minlength=k).tolist() for key, value in roles.items()},
            "feature_standardization": {"enabled": bool(normalization), "fit_on": "nodes belonging to fit_train units" if normalization else None},
            "source_array_hashes": source_hashes,
            "array_hashes": {key: _array_sha256(value) for key, value in arrays.items()},
            "official_split_hashes": {key: _array_sha256(value) for key, value in official.items()},
            "role_hashes": {key: _array_sha256(value) for key, value in roles.items()},
            "graph_role_hashes": None if graph_roles is None else {key: _array_sha256(value) for key, value in graph_roles.items()},
            "cluster_sha256": _array_sha256(clusters), "buffer_audit": buffer_metadata,
            "artifact_sha256": {name: _file_sha256(prepared / name) for name in files},
        }
        metadata["fingerprint"] = _fingerprint(metadata)
        _atomic_json(metadata_path, metadata)  # commit marker is written last
        return _graph_data(arrays, roles, clusters, metadata, graph_roles)


def load_data(output_dir: Path) -> GraphData:
    """Load numeric artifacts and verify both task semantics and graph boundaries."""
    prepared = Path(output_dir) / "prepared"
    metadata = json.loads((prepared / "metadata.json").read_text())
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported prepared-data schema version.")
    if _fingerprint(metadata) != metadata.get("fingerprint"):
        raise ValueError("Prepared metadata fingerprint mismatch.")
    for name, expected in metadata["artifact_sha256"].items():
        if Path(name).name != name or _file_sha256(prepared / name) != expected:
            raise ValueError(f"Prepared artifact integrity check failed: {name}")
    with np.load(prepared / "graph.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    task = task_spec(metadata["benchmark_dataset"], metadata)
    x, edges, y, _ = _validate_arrays(arrays["x"], arrays["edge_index"], arrays["y"], metadata["num_classes"],
                                     task_type=task.task_type, label_count=metadata["num_samples"])
    arrays.update(x=x, edge_index=edges, y=y)
    with np.load(prepared / "splits.npz", allow_pickle=False) as archive:
        roles = _validate_roles({name: archive[name] for name in archive.files}, len(y))
    with np.load(prepared / "clusters.npz", allow_pickle=False) as archive:
        clusters = archive["cluster_ids"]
    independent = bool(metadata["independent_graphs"])
    graph_roles = None
    if independent:
        ptr, eptr = validate_graph_pointers(arrays["graph_ptr"], arrays["edge_ptr"], len(x), edges)
        if len(ptr)-1 != metadata["num_graphs"]:
            raise ValueError("Prepared graph count disagrees with graph_ptr")
        with np.load(prepared / "graph_splits.npz", allow_pickle=False) as archive:
            graph_roles = _validate_roles({name: archive[name] for name in archive.files}, len(ptr)-1)
        expected_roles = _graph_roles_to_rows(graph_roles, ptr, task.label_axis == "graph")
        if any(not np.array_equal(roles[role], expected_roles[role]) for role in ROLE_NAMES):
            raise ValueError("Prepared label roles split an independent graph")
        k = len(ptr)-1
        expected_clusters = np.arange(k, dtype=np.int64) if task.label_axis == "graph" else np.repeat(np.arange(k, dtype=np.int64), np.diff(ptr))
        if not np.array_equal(clusters, expected_clusters):
            raise ValueError("Prepared independent-graph unit membership is malformed")
    else:
        if "graph_ptr" in arrays or "edge_ptr" in arrays:
            raise ValueError("Transductive data must not contain independent-graph pointers")
        k = int(metadata["preparation_config"]["data"]["num_clusters"])
    if x.shape != (metadata["num_nodes"], metadata["num_features"]) or edges.shape[1] != metadata["edge_entries"]:
        raise ValueError("Prepared graph dimensions disagree with metadata.")
    if clusters.dtype.kind not in "iu" or clusters.shape != (len(y),) or not np.array_equal(np.unique(clusters), np.arange(k)):
        raise ValueError("Prepared cluster membership is malformed.")
    if "edge_attr" in arrays:
        attr = arrays["edge_attr"]
        if attr.ndim != 2 or len(attr) != edges.shape[1] or not np.isfinite(attr).all():
            raise ValueError("Prepared edge attributes are malformed")
    if "edge_weight" in arrays:
        weight = arrays["edge_weight"]
        if weight.shape != (edges.shape[1],) or not np.isfinite(weight).all() or np.any(weight < 0):
            raise ValueError("Prepared edge weights are malformed")
    if {key: _array_sha256(value) for key, value in arrays.items()} != metadata["array_hashes"] or _array_sha256(clusters) != metadata["cluster_sha256"]:
        raise ValueError("Prepared array fingerprint mismatch.")
    if {key: _array_sha256(value) for key, value in roles.items()} != metadata["role_hashes"]:
        raise ValueError("Prepared role fingerprint mismatch.")
    if graph_roles is not None and {key: _array_sha256(value) for key, value in graph_roles.items()} != metadata["graph_role_hashes"]:
        raise ValueError("Prepared graph-role fingerprint mismatch")
    return _graph_data(arrays, roles, clusters.astype(np.int64), metadata, graph_roles)
