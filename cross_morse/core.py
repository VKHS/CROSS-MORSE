"""Frozen graph plant, exact CROSS-MORSE actions and label-free sensors.

The transport registry implements main Eqs. (8)--(11) and supplementary S6.
The PDFs do not supply the original encoder code or spectral coefficients.
Accordingly the MLP/ReLU encoder and equal-width Jackson windows below are
explicit reconstruction choices, not recovered original checkpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import math
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .tasks import TaskSpec, task_spec, probabilities, confidence_statistics


@dataclass(frozen=True)
class Action:
    id: int
    name: str
    rho: float
    nu: float
    alpha: float
    eta: float | None = None
    order: int = 8
    probability: float = 0.0
    mode: str = "transport"


ACTIONS: tuple[Action, ...] = (
    Action(0, "nominal", 0.00, 0.70, 0.25, probability=0.28),
    Action(1, "weak_local", 0.20, 0.55, 0.25, probability=0.09),
    Action(2, "strong_local", 0.20, 0.85, 0.20, probability=0.09),
    Action(3, "short_safe", 0.25, 0.70, 0.25, 0.50, probability=0.09),
    Action(4, "short_recovery", 0.30, 0.65, 0.35, 0.50, probability=0.09),
    Action(5, "medium", 0.30, 0.70, 0.20, 0.20, probability=0.09),
    Action(6, "medium_recovery", 0.35, 0.60, 0.35, 0.20, probability=0.09),
    Action(7, "long_conservative", 0.25, 0.50, 0.40, 0.08, probability=0.08),
    Action(8, "source_reset", 0.0, 0.0, 1.0, probability=0.05, mode="source"),
    Action(9, "last_safe_reset", 0.0, 0.0, 1.0, probability=0.05, mode="last_safe"),
)


def _action(action: Action | int) -> Action:
    if isinstance(action, Action):
        return action
    if not 0 <= int(action) < len(ACTIONS):
        raise ValueError(f"Unknown action id {action}")
    return ACTIONS[int(action)]


def action_operator_response(action: Action | int, eigenvalues: Tensor) -> Tensor:
    """Polynomial response T_a(lambda) for lambda of L_sym, not of P.

    Direct recovery is a hybrid state assignment and has no such polynomial;
    asking for its response raises rather than inventing one.
    """
    a = _action(action)
    if a.mode != "transport":
        raise ValueError("Direct recovery has no transport polynomial")
    p = 1.0 - eigenvalues
    kernel = p
    if a.eta is not None:
        power = torch.ones_like(p)
        resolvent = a.eta * power
        decay = 1.0 - a.eta
        for t in range(1, a.order + 1):
            power = power * p
            resolvent = resolvent + a.eta * decay**t * power
        resolvent = resolvent + decay ** (a.order + 1)
        kernel = (1.0 - a.rho) * p + a.rho * resolvent
    return (1.0 - a.nu) + a.nu * kernel


def nominal_operator_response(eigenvalues: Tensor, baseline: str = "shared") -> Tensor:
    """Nominal transport at one controlled site, before source retention.

    Shared blocks use registry A0, designated native blocks use P, and inserted
    sites bypass transport. This is not the response of the complete nonlinear
    block, nor does it include an action's source-retention contribution.
    """
    if baseline == "shared":
        return action_operator_response(ACTIONS[0], eigenvalues)
    if baseline == "native_mean":
        return 1.0 - eigenvalues
    if baseline == "inserted_identity":
        return torch.ones_like(eigenvalues)
    raise ValueError(f"Unknown nominal transport baseline {baseline!r}")


def executed_hops(action: Action | int) -> float:
    a = _action(action)
    if a.mode != "transport":
        return 0.0
    if a.eta is None:
        return a.nu
    resolvent_hops = sum(t * a.eta * (1 - a.eta)**t for t in range(a.order + 1))
    return a.nu * ((1 - a.rho) + a.rho * resolvent_hops)


def identity_mass(action: Action | int) -> float:
    """Truncation-tail mass in K, NOT the executed operator's diagonal."""
    a = _action(action)
    return (1 - a.eta)**(a.order + 1) if a.eta is not None else 0.0


def executed_identity_coefficient(action: Action | int) -> float:
    """Coefficient of P**0; excludes graph-dependent return paths."""
    a = _action(action)
    if a.mode != "transport":
        raise ValueError("Hybrid resets do not have a transport identity coefficient")
    return (1 - a.nu) + (a.nu * a.rho * (a.eta + identity_mass(a))
                          if a.eta is not None else 0.0)


TRANSPORT_HISTORY_DEFINITION = (
    "kernel transport mass and expected kernel hops before source retention; "
    "inserted A0 bypass=0/0; designated native A0 mean aggregation=1/1; "
    "other transports use registered nu and kernel hop moment; hybrid resets=0/0; "
    "these are transport descriptors, not complete block work or effective output displacement"
)


def site_transport_accounting(action: Action | int, site: int, architecture) -> dict:
    """One history/penalty convention for observed and model-predicted steps.

    ``architecture`` is either a plant or its sealed configuration. The ordinary
    shared-transport case retains its historical nu/hop coordinates. The native
    GPS/GINE A0 call is PH, whereas its inserted A0 is identity (Table S5b/S5c).
    No descriptor counts a reset as graph transport or source retention as hops.
    """
    a = _action(action)
    if int(site) != site or site < 0:
        raise ValueError("Controlled site must be a nonnegative integer")
    if isinstance(architecture, Mapping):
        m = architecture.get("model", architecture)
        backbone = m.get("backbone")
        if backbone is None:
            dataset = architecture.get("benchmark_dataset", architecture.get("dataset"))
            backbone = {"PascalVOC-SP": "gps", "COCO-SP": "gps", "Peptides-func": "linear_gine"}.get(dataset, "shared_transport")
        depth = m.get("depth")
        if depth is not None and site >= depth:
            raise ValueError("Controlled site outside the configured depth")
        native = True
        if backbone in {"gps", "linear_gine"}:
            if depth is None:
                raise ValueError("Designated-site accounting requires the controlled depth")
            layers = int(m.get("native_layers", 6 if backbone == "gps" else 8))
            if layers < 1:
                raise ValueError("Native layer count must be positive")
            if backbone == "gps":
                if depth < layers + 2 or (depth - 2) % layers:
                    raise ValueError("Invalid GPS control-site layout")
                stride = (depth - 2) // layers
                native = 1 <= site < depth - 1 and (site - 1) % stride == 0
            else:
                if depth < layers or depth % layers:
                    raise ValueError("Invalid linear-GINE control-site layout")
                native = site % (depth // layers) == 0
    else:
        backbone = getattr(architecture, "backbone", "shared_transport")
        native = bool(architecture.is_native_site(site))
    bypass = not native and a.id == 0
    if a.mode != "transport" or bypass:
        mass = hops = 0.
    elif native and backbone in {"gps", "linear_gine"} and a.id == 0:
        mass = hops = 1.
    else:
        mass, hops = float(a.nu), float(executed_hops(a))
    return {"site_kind": "native" if native else "inserted", "mass": mass,
            "hops": hops, "inserted_a0_bypass": bypass}


def site_action_summary(actions, architecture) -> dict:
    """Counts of completed actual transitions, separated by native/inserted site."""
    native, inserted = [0] * len(ACTIONS), [0] * len(ACTIONS)
    for site, action in enumerate(actions):
        accounting = site_transport_accounting(action, site, architecture)
        counts = native if accounting["site_kind"] == "native" else inserted
        counts[_action(action).id] += 1
    return {"native_site_action_counts": native, "inserted_site_action_counts": inserted,
            "native_site_count": sum(native), "inserted_site_count": sum(inserted),
            "inserted_a0_bypass_count": inserted[0],
            "transport_history_definition": TRANSPORT_HISTORY_DEFINITION}


class GraphOperator:
    """Sparse random walk P and symmetric normalized adjacency S.

    Input edge_index must describe the already-symmetrized graph. Duplicate
    pairs are de-duplicated (their explicit weights are averaged). Nonnegative
    edge weights produce P=D^-1 W and S=D^-1/2 W D^-1/2. Zero-weight pairs are
    omitted. Only isolated vertices receive an added self-loop;
    no other self-loop convention is silently imposed. COO construction and all
    products remain sparse. Message products use fp32 internally and return the
    caller's state dtype, including under dense bf16 autocast.
    """
    def __init__(self, edge_index: Tensor, num_nodes: int,
                 device: str | torch.device | None = None,
                 edge_weight: Tensor | None = None,
                 graph_ptr: Tensor | None = None):
        if num_nodes < 1:
            raise ValueError("num_nodes must be positive")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        self.device = torch.device(device if device is not None else edge_index.device)
        self.num_nodes = int(num_nodes)
        self.graph_ptr = None
        if graph_ptr is not None:
            ptr = torch.as_tensor(graph_ptr, dtype=torch.long, device=self.device)
            if ptr.ndim != 1 or len(ptr) < 2 or int(ptr[0]) != 0 or int(ptr[-1]) != num_nodes or not torch.all(ptr[1:] > ptr[:-1]):
                raise ValueError("graph_ptr must partition all nodes into nonempty graphs")
            self.graph_ptr = ptr
        self.num_graphs = 1 if self.graph_ptr is None else len(self.graph_ptr) - 1
        edges = edge_index.to(device=self.device, dtype=torch.long)
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= num_nodes):
            raise ValueError("edge index outside graph")
        if self.graph_ptr is not None and edges.numel():
            membership = torch.bucketize(edges, self.graph_ptr[1:-1], right=True)
            if not torch.equal(membership[0], membership[1]):
                raise ValueError("An edge crosses independent graph boundaries")
        self.weighted = edge_weight is not None
        values = (torch.ones(edges.shape[1], device=self.device, dtype=torch.float32)
                  if edge_weight is None else torch.as_tensor(edge_weight, device=self.device, dtype=torch.float32))
        if values.shape != (edges.shape[1],) or not torch.isfinite(values).all() or torch.any(values < 0):
            raise ValueError("edge_weight must contain one finite nonnegative scalar per edge")
        adjacency = torch.sparse_coo_tensor(edges, values, (num_nodes, num_nodes),
                                            device=self.device, check_invariants=True).coalesce()
        if self.weighted:
            multiplicity = torch.sparse_coo_tensor(edges, torch.ones_like(values), (num_nodes, num_nodes),
                                                  device=self.device, check_invariants=True).coalesce().values()
            values = adjacency.values() / multiplicity
        else:
            values = torch.ones_like(adjacency.values())
        edges = adjacency.indices()
        keep = values > 0
        edges, values = edges[:, keep], values[keep]
        degree = torch.zeros(num_nodes, device=self.device).index_add_(0, edges[0], values)
        isolated = torch.where(degree == 0)[0]
        if isolated.numel():
            edges = torch.cat((edges, torch.stack((isolated, isolated))), dim=1)
            values = torch.cat((values, torch.ones(len(isolated), device=self.device)))
            adjacency = torch.sparse_coo_tensor(
                edges, values,
                (num_nodes, num_nodes), device=self.device, check_invariants=True).coalesce()
            edges, values = adjacency.indices(), adjacency.values()
            degree = torch.zeros(num_nodes, device=self.device).index_add_(0, edges[0], values)
        self.edge_index = edges
        self.edge_weight = values
        self.degree = degree
        self.sqrt_degree = degree.sqrt()
        self.num_edges = int(edges.shape[1])
        row, col = edges
        p_values = values * degree[row].reciprocal()
        s_values = values * (degree[row] * degree[col]).rsqrt()
        self.P = torch.sparse_coo_tensor(edges, p_values, (num_nodes, num_nodes),
                                       device=self.device, check_invariants=True).coalesce()
        self.S = torch.sparse_coo_tensor(edges, s_values, (num_nodes, num_nodes),
                                       device=self.device, check_invariants=True).coalesce()
        self.reset_counters()

    def reset_counters(self) -> None:
        self.counters = {"sparse_products": 0, "sparse_multiply_adds": 0,
                         "random_walk_products": 0, "symmetric_products": 0}

    @property
    def sparse_products(self) -> int:
        return self.counters["sparse_products"]

    @property
    def sparse_multiply_adds(self) -> int:
        return self.counters["sparse_multiply_adds"]

    def _multiply(self, matrix: Tensor, h: Tensor, kind: str) -> Tensor:
        if h.ndim != 2 or h.shape[0] != self.num_nodes:
            raise ValueError("state must have shape [num_nodes, width]")
        self.counters["sparse_products"] += 1
        self.counters["sparse_multiply_adds"] += self.num_edges * int(h.shape[1])
        self.counters[kind] += 1
        # Dense matmuls may autocast to bf16; sparse mm is explicitly fp32.
        with torch.autocast(device_type=self.device.type, enabled=False):
            out = torch.sparse.mm(matrix, h.float())
        return out.to(h.dtype)

    def apply(self, h: Tensor) -> Tensor:
        return self._multiply(self.P, h, "random_walk_products")

    def symmetric(self, h: Tensor) -> Tensor:
        return self._multiply(self.S, h, "symmetric_products")

    def dirichlet(self, h: Tensor, eps: float = 1e-12) -> Tensor:
        hf = h.float()
        sh = self.symmetric(hf)
        value = (hf * (hf - sh)).sum() / (hf.square().sum() + eps)
        return value.clamp_min(0.0)


def apply_kernel(graph: GraphOperator, h: Tensor, action: Action | int) -> Tensor:
    """Apply K_a=(1-rho)P+rho*Ktilde, before transport-mass mixing.

    The inserted-site form in the main Definition 1 and Supplementary Table S6
    explicitly uses this kernel, with nu appearing once outside its bracket.
    """
    a = _action(action)
    if a.mode != "transport":
        raise ValueError("A recovery action is not a transport matrix")
    ph = graph.apply(h)
    kernel = ph
    if a.eta is not None:
        decay = 1.0 - a.eta
        # P**0 costs no sparse product; PH above is reused for t=1.
        resolvent = (a.eta + decay ** (a.order + 1)) * h
        resolvent = resolvent + a.eta * decay * ph
        power = ph
        for t in range(2, a.order + 1):
            power = graph.apply(power)
            resolvent = resolvent + a.eta * decay**t * power
        kernel = (1.0 - a.rho) * ph + a.rho * resolvent
    return kernel


def apply_transport(graph: GraphOperator, h: Tensor, action: Action | int) -> Tensor:
    """Apply main Eq. (8)'s T=(1-nu)I+nu*K_a; no trainable update or reset."""
    a = _action(action)
    kernel = apply_kernel(graph, h, a)
    return (1.0 - a.nu) * h + a.nu * kernel


class FrozenPlant(nn.Module):
    """Four-layer encoder, shared native transport update, and fixed head.

    Architecture choice: input linear + ReLU/dropout followed by hidden linear
    blocks, a shared hidden linear/ReLU update, post-reference LayerNorm, and a
    linear classification head. All normal actions retain the source H0. This
    exact choice is disclosed because the papers do not specify the missing
    encoder internals. No parameter count is padded to the paper's reported 3.1M.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 encoder_layers: int = 4, dropout: float = 0.1,
                 activation: str = "relu", task: TaskSpec | dict | None = None,
                 dense_chunk_size: int = 0):
        super().__init__()
        if min(input_dim, hidden_dim, num_classes, encoder_layers) < 1:
            raise ValueError("plant dimensions and layer count must be positive")
        if activation not in ("relu", "silu", "gelu"):
            raise ValueError("activation must be relu, silu, or gelu")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.encoder_layers = encoder_layers
        self.activation_name = activation
        self.task = (TaskSpec(num_outputs=num_classes) if task is None
                     else task_spec(task))
        if self.task.num_outputs is not None and self.task.num_outputs != num_classes:
            raise ValueError("Task outputs and head dimension differ")
        if dense_chunk_size < 0:
            raise ValueError("dense_chunk_size must be nonnegative")
        self.dense_chunk_size = int(dense_chunk_size)
        self.backbone = "shared_transport"
        self.control_depth = None
        self._graph_ptr = None
        self._bound_num_nodes = None
        self.encoder = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(encoder_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.shared = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self._frozen = False
        self.reset_counters()

    def _activation(self, h: Tensor) -> Tensor:
        return {"relu": F.relu, "silu": F.silu, "gelu": F.gelu}[self.activation_name](h)

    def reset_counters(self) -> None:
        self.counters = {"dense_updates": 0, "dense_multiply_adds": 0,
                         "head_calls": 0, "encoder_calls": 0,
                         "normalization_elements": 0, "resets": 0,
                         "attention_calls": 0, "readout_pooling_elements": 0}

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def bind_graph(self, graph: GraphOperator) -> "FrozenPlant":
        """Bind only runtime topology; no graph state enters the checkpoint.

        Graph-level heads pool within graph_ptr. Node-level heads leave rows
        unchanged. Controllers must execute independent graphs one at a time;
        the readout also supports a disjoint union for isolated equivalence tests.
        """
        self._graph_ptr = graph.graph_ptr
        self._bound_num_nodes = graph.num_nodes
        if self.task.prediction_level == "graph" and self._graph_ptr is None:
            raise ValueError("Graph-level prediction requires explicit graph_ptr")
        return self

    def _linear(self, layer: nn.Linear, h: Tensor) -> Tensor:
        self.counters["dense_multiply_adds"] += int(h.shape[0]) * layer.in_features * layer.out_features
        if self.dense_chunk_size and len(h) > self.dense_chunk_size:
            return torch.cat([layer(part) for part in h.split(self.dense_chunk_size)], dim=0)
        return layer(h)

    def is_native_site(self, site: int) -> bool:
        if int(site) != site or site < 0 or (self.control_depth is not None and site >= self.control_depth):
            raise ValueError("Controlled site index outside the registered trajectory")
        return True

    def instrumentation_metadata(self) -> dict:
        return {"backbone": self.backbone, "controlled_depth": self.control_depth,
                "native_site_count": self.control_depth, "inserted_site_count": 0,
                "site_indexing": "zero-based transition d to d+1",
                "task": self.task.to_dict(), "parameter_count": self.parameter_count,
                "dense_chunk_size": self.dense_chunk_size,
                "transport_history_definition": TRANSPORT_HISTORY_DEFINITION}

    def encode(self, x: Tensor) -> Tensor:
        self.counters["encoder_calls"] += 1
        h = x
        for layer in self.encoder:
            h = self.dropout(self._activation(self._linear(layer, h)))
        return h

    def native_step(self, h: Tensor, h0: Tensor, graph: GraphOperator,
                    site: int | None = None) -> Tensor:
        """Native A0 path, independent from registry/adapter dispatch for E1."""
        ph = graph.apply(h)
        message = (1.0 - 0.70) * h + 0.70 * ph
        propagated = self._activation(self._linear(self.shared, message))
        self.counters["dense_updates"] += 1
        self.counters["normalization_elements"] += h.numel()
        return self.norm((1.0 - 0.25) * propagated + 0.25 * h0)

    def step(self, h: Tensor, h0: Tensor, graph: GraphOperator,
             action_id: int | Action, last_safe: Tensor | None = None,
             site: int | None = None) -> Tensor:
        a = _action(action_id)
        if a.mode == "source":
            self.counters["resets"] += 1
            return h0.clone()
        if a.mode == "last_safe":
            self.counters["resets"] += 1
            # At initialization the source is also the last safe state.
            return (h0 if last_safe is None else last_safe).clone()
        message = apply_transport(graph, h, a)
        propagated = self._activation(self._linear(self.shared, message))
        self.counters["dense_updates"] += 1
        self.counters["normalization_elements"] += h.numel()
        return self.norm((1.0 - a.alpha) * propagated + a.alpha * h0)

    def logits(self, h: Tensor, graph: GraphOperator | None = None) -> Tensor:
        if graph is not None:
            self.bind_graph(graph)
        if self.task.prediction_level == "graph":
            if self._graph_ptr is None or self._bound_num_nodes != len(h):
                raise RuntimeError("Bind the current graph before graph-level readout")
            ptr = self._graph_ptr
            # A separate reduction per graph prevents any cross-graph readout.
            h = torch.stack([h[int(start):int(end)].mean(0) for start, end in zip(ptr[:-1], ptr[1:])])
            self.counters["readout_pooling_elements"] += self._bound_num_nodes * self.hidden_dim
        self.counters["head_calls"] += 1
        return self._linear(self.head, h)

    def freeze(self) -> "FrozenPlant":
        self.eval()
        self.requires_grad_(False)
        self._frozen = True
        return self

    def train(self, mode: bool = True) -> "FrozenPlant":
        if mode and getattr(self, "_frozen", False):
            raise RuntimeError("A frozen plant cannot enter training mode")
        return super().train(mode)

    def forward(self, x: Tensor, graph: GraphOperator, depth: int) -> Tensor:
        self.bind_graph(graph)
        h0 = self.encode(x)
        h = h0
        for site in range(depth):
            h = self.native_step(h, h0, graph, site=site)
        return self.logits(h)


class EdgeWeightedPlant(FrozenPlant):
    """Protein shared transport with explicit edge-feature kernel weights (S5b)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = "edge_weighted_transport"

    @staticmethod
    def _check_weights(graph):
        if not graph.weighted:
            raise ValueError("The protein plant requires explicit nonnegative edge weights")

    def native_step(self, h, h0, graph, site=None):
        self._check_weights(graph)
        return super().native_step(h, h0, graph, site=site)

    def step(self, h, h0, graph, action_id, last_safe=None, site=None):
        self._check_weights(graph)
        return super().step(h, h0, graph, action_id, last_safe, site=site)


class HeterophilyPlant(FrozenPlant):
    """Separate ego and neighbour linear maps for Roman-empire (Table S5b).

    The neighbour branch retains the nonnegative normalized transport; the ego
    branch has its own learned map. Their sum is activated before source mixing.
    This specifies the otherwise unavailable native update, rather than claiming
    the original parameterization was recovered from a checkpoint.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ego = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.backbone = "heterophily"

    def _update(self, h, h0, message, alpha):
        propagated = self._activation(self._linear(self.ego, h) + self._linear(self.shared, message))
        self.counters["dense_updates"] += 1
        self.counters["normalization_elements"] += h.numel()
        return self.norm((1 - alpha) * propagated + alpha * h0)

    def native_step(self, h, h0, graph, site=None):
        message = (1 - .70) * h + .70 * graph.apply(h)
        return self._update(h, h0, message, .25)

    def step(self, h, h0, graph, action_id, last_safe=None, site=None):
        a = _action(action_id)
        if a.mode != "transport":
            self.counters["resets"] += 1
            return (h0 if a.mode == "source" or last_safe is None else last_safe).clone()
        return self._update(h, h0, apply_transport(graph, h, a), a.alpha)


class _GPSBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.local_first = nn.Linear(width, width)
        self.local_second = nn.Linear(width, width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.local_norm = nn.LayerNorm(width)
        self.attention_norm = nn.LayerNorm(width)
        self.ff_first = nn.Linear(width, 2 * width)
        self.ff_second = nn.Linear(2 * width, width)
        self.ff_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)


class _LinearGINEBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.epsilon = nn.Parameter(torch.zeros(()))
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)


class DesignatedSitePlant(FrozenPlant):
    """Table S5b native blocks plus parameter-free inserted control sites.

    Site indices are zero-based. GPS places a native block at the first site of
    each block's interval, with an encoder and decoder boundary. Linear GINE
    uses the first site of each equal-size interval. Exact positions within an
    interval are a declared choice because only counts are specified in S5b.

    At an inserted site A0 returns the input tensor literally. Other transports
    apply V=(1-nu)H+nu*((1-alpha)*K_a H+alpha*H0); no learned layer is added.
    K_a is the non-lazy kernel, so nu is applied exactly once (S6 inserted form).
    A8/A9 remain direct hybrid resets. At native sites only the local aggregation
    is replaced; the GPS attention module and its graph-local computation law
    are unchanged. The native aggregation is the weighted normalized mean.
    """
    def __init__(self, *args, control_depth, native_layers, backbone, attention_heads=4, **kwargs):
        super().__init__(*args, **kwargs)
        if native_layers < 1 or control_depth < 1:
            raise ValueError("Positive native and controlled depths are required")
        self.control_depth, self.native_layers = int(control_depth), int(native_layers)
        self.backbone = backbone
        # These shared-update parameters belong only to transport plants.
        del self.shared
        del self.norm
        if backbone == "gps":
            if (control_depth - 2) % native_layers or control_depth < native_layers + 2:
                raise ValueError("GPS depth must equal native_layers * substeps + two boundary sites")
            if self.hidden_dim % attention_heads:
                raise ValueError("GPS hidden width must be divisible by attention_heads")
            substeps = (control_depth - 2) // native_layers
            self.native_sites = tuple(1 + i * substeps for i in range(native_layers))
            self.blocks = nn.ModuleList([_GPSBlock(self.hidden_dim, attention_heads, self.dropout.p)
                                         for _ in range(native_layers)])
        elif backbone == "linear_gine":
            if control_depth % native_layers:
                raise ValueError("Linear GINE depth must be a multiple of native_layers")
            substeps = control_depth // native_layers
            self.native_sites = tuple(i * substeps for i in range(native_layers))
            self.blocks = nn.ModuleList([_LinearGINEBlock(self.hidden_dim, self.dropout.p)
                                         for _ in range(native_layers)])
        else:
            raise ValueError("Unknown designated-site backbone")
        self._site_to_layer = {site: layer for layer, site in enumerate(self.native_sites)}

    def _site(self, site):
        if site is None:
            raise ValueError("Designated-site backbones require the zero-based controlled site index")
        if int(site) != site or site < 0 or site >= self.control_depth:
            raise ValueError("Controlled site index outside the registered trajectory")
        return int(site)

    def is_native_site(self, site):
        return self._site(site) in self._site_to_layer

    def instrumentation_metadata(self):
        result = super().instrumentation_metadata()
        result.update(native_site_count=self.native_layers,
                      inserted_site_count=self.control_depth - self.native_layers,
                      native_sites=list(self.native_sites),
                      native_aggregation="nonnegative degree-normalized linear mean",
                      native_position_choice="first site in each block interval",
                      inserted_nominal="literal identity",
                      global_attention="unchanged graph-local branch" if self.backbone == "gps" else "none")
        return result

    def _attention(self, block, h, graph):
        ptr = graph.graph_ptr
        if ptr is None:
            raise ValueError("GPS requires explicit independent-graph boundaries")
        outputs = []
        for start, end in zip(ptr[:-1], ptr[1:]):
            local = h[int(start):int(end)].unsqueeze(0)
            value, _ = block.attention(local, local, local, need_weights=False)
            outputs.append(value.squeeze(0))
            n = int(end - start)
            self.counters["attention_calls"] += 1
            # Q/K/V and output projections plus QK^T and attention-value products.
            self.counters["dense_multiply_adds"] += 4 * n * self.hidden_dim**2 + 2 * n*n * self.hidden_dim
        return torch.cat(outputs, dim=0)

    def _native_block(self, layer, h, aggregation, graph):
        block = self.blocks[layer]
        self.counters["dense_updates"] += 1
        if self.backbone == "gps":
            local = self._linear(block.local_second,
                                 self._activation(self._linear(block.local_first, h + aggregation)))
            local = block.local_norm(h + block.dropout(local))
            global_branch = block.attention_norm(h + block.dropout(self._attention(block, h, graph)))
            combined = local + global_branch
            ff = self._linear(block.ff_second,
                              block.dropout(self._activation(self._linear(block.ff_first, combined))))
            self.counters["normalization_elements"] += 3 * h.numel()
            return block.ff_norm(combined + block.dropout(ff))
        combined = (1 + block.epsilon) * h + aggregation
        propagated = self._linear(block.second, self._activation(self._linear(block.first, combined)))
        self.counters["normalization_elements"] += h.numel()
        return block.norm(h + block.dropout(propagated))

    def native_step(self, h, h0, graph, site=None):
        site = self._site(site)
        if site not in self._site_to_layer:
            return h
        return self._native_block(self._site_to_layer[site], h, graph.apply(h), graph)

    def step(self, h, h0, graph, action_id, last_safe=None, site=None):
        site = self._site(site)
        a = _action(action_id)
        if a.mode != "transport":
            self.counters["resets"] += 1
            return (h0 if a.mode == "source" or last_safe is None else last_safe).clone()
        if site not in self._site_to_layer:
            if a.id == 0:
                return h
            transported = apply_kernel(graph, h, a)
            return (1 - a.nu) * h + a.nu * ((1 - a.alpha) * transported + a.alpha * h0)
        # The native A0 law is independent of the inserted-site transport law.
        aggregation = (graph.apply(h) if a.id == 0 else
                       (1 - a.alpha) * apply_transport(graph, h, a) + a.alpha * h0)
        return self._native_block(self._site_to_layer[site], h, aggregation, graph)


def build_plant(input_dim: int, num_outputs: int, model_config: dict,
                task: TaskSpec | None = None) -> FrozenPlant:
    """Construct the declared dataset backbone without inspecting labels."""
    m = model_config
    spec = TaskSpec(num_outputs=num_outputs) if task is None else task_spec(task)
    default_backbone = {"ogbn-proteins": "edge_weighted_transport", "Roman-empire": "heterophily",
                        "PascalVOC-SP": "gps", "COCO-SP": "gps", "Peptides-func": "linear_gine"}.get(spec.dataset, "shared_transport")
    backbone = m.get("backbone", default_backbone)
    args = dict(input_dim=input_dim, hidden_dim=int(m["hidden_dim"]), num_classes=num_outputs,
                encoder_layers=int(m["encoder_layers"]), dropout=float(m.get("dropout", .1)),
                activation=m.get("activation", "relu"), task=spec,
                dense_chunk_size=int(m.get("dense_chunk_size", 0)))
    if backbone in {"gps", "linear_gine"}:
        model = DesignatedSitePlant(**args, control_depth=int(m["depth"]),
                                    native_layers=int(m.get("native_layers", 6 if backbone == "gps" else 8)),
                                    attention_heads=int(m.get("attention_heads", 4)), backbone=backbone)
    else:
        classes = {"shared_transport": FrozenPlant, "edge_weighted_transport": EdgeWeightedPlant,
                   "heterophily": HeterophilyPlant}
        if backbone not in classes:
            raise ValueError(f"Unknown backbone: {backbone}")
        model = classes[backbone](**args)
        model.control_depth = int(m["depth"])
    return model


def chebyshev_window_coefficients(degree: int = 8, windows: int = 6) -> Tensor:
    """Analytic equal-interval indicator coefficients with Jackson damping.

    lambda in [0,2] maps to x=lambda-1. The c0 convention is literal c0,
    not c0/2. Damping uses angle pi/(degree+2), retaining degree+1 terms.
    These fully specified coefficients replace unavailable original windows.
    """
    if degree < 1 or windows < 1:
        raise ValueError("positive polynomial degree and window count required")
    coeff = torch.empty((windows, degree + 1), dtype=torch.float64)
    angle = math.pi / (degree + 2)
    for b in range(windows):
        left, right = 2.0 * b / windows, 2.0 * (b + 1) / windows
        theta_left = math.acos(max(-1.0, min(1.0, left - 1.0)))
        theta_right = math.acos(max(-1.0, min(1.0, right - 1.0)))
        coeff[b, 0] = (theta_left - theta_right) / math.pi
        for k in range(1, degree + 1):
            raw = 2.0 * (math.sin(k * theta_left) - math.sin(k * theta_right)) / (math.pi * k)
            damping = ((degree - k + 2) * math.cos(k * angle)
                       + math.sin(k * angle) / math.tan(angle)) / (degree + 2)
            coeff[b, k] = raw * damping
    return coeff.float()


class MeasurementBank:
    """Deterministic, label-free full-graph sensors with 25 raw coordinates.

    Three observer innovation features from S8 are provided by the observer,
    not computed circularly inside this measurement map. Optional ``mask``
    changes pooling of node-wise quantities AFTER full-graph filtering. The
    Dirichlet coordinate remains the global normalized energy. A mask never
    creates an induced graph or changes the native propagation operator.

    Full uses three registered equal-interval windows from a six-window bank.
    Lean preserves the schema but zeros all 16 spectral/reachability coordinates,
    an explicit reconstructed profile. Robust normalization is fitted offline
    by the caller; this class never accepts labels or fits statistics online.
    """
    def __init__(self, graph: GraphOperator, hidden_dim: int,
                 bands: Iterable[int] = (0, 2, 5), degree: int = 8,
                 sketch_dim: int = 0, seed: int = 0, profile: str = "full",
                 mask: Tensor | None = None, task: TaskSpec | dict | None = None,
                 plant: FrozenPlant | None = None):
        self.graph = graph
        if graph.num_graphs != 1:
            raise ValueError("Independent-graph sensors require one graph per controller episode")
        self.task = TaskSpec() if task is None else task_spec(task)
        if self.task.prediction_level == "graph" and graph.graph_ptr is None:
            raise ValueError("Graph-level sensors require graph_ptr")
        if self.task.prediction_level == "graph" and mask is not None:
            raise ValueError("A node pooling mask cannot index graph-level predictions")
        self.hidden_dim = hidden_dim
        self.bands = tuple(int(b) for b in bands)
        if len(self.bands) != 3 or len(set(self.bands)) != 3 or any(b < 0 or b >= 6 for b in self.bands):
            raise ValueError("Exactly three distinct bands from 0..5 are required")
        if profile.lower() not in ("full", "lean"):
            raise ValueError("profile must be Full or Lean")
        if sketch_dim < 0:
            raise ValueError("sketch_dim must be nonnegative")
        self.profile = profile.lower()
        self.degree = degree
        self.sketch_dim = min(sketch_dim, hidden_dim)
        self.seed = seed
        self.mask = mask.to(graph.device) if mask is not None else None
        if self.mask is not None and self.mask.dtype == torch.bool and self.mask.numel() != graph.num_nodes:
            raise ValueError("Boolean pooling mask has wrong node count")
        self.coefficients = chebyshev_window_coefficients(degree).to(graph.device)
        self._active_coefficients = self.coefficients[list(self.bands)]
        rng = torch.Generator(device="cpu").manual_seed(seed)
        self.projection = None
        if 0 < self.sketch_dim < hidden_dim:
            self.projection = (torch.randn(hidden_dim, self.sketch_dim, generator=rng)
                               / math.sqrt(self.sketch_dim)).to(graph.device)
        self.rank_width = min(32, hidden_dim)
        self.rank_projection = (torch.randn(hidden_dim, self.rank_width, generator=rng)
                                / math.sqrt(self.rank_width)).to(graph.device)
        self.nominal_baselines = None
        if getattr(plant, "backbone", None) in {"gps", "linear_gine"}:
            self.nominal_baselines = tuple(
                "native_mean" if plant.is_native_site(site) else "inserted_identity"
                for site in range(plant.control_depth))
        self.sensitivities_by_baseline = {}
        for baseline in ("shared", "native_mean", "inserted_identity"):
            sensitivities = []
            for action in ACTIONS[1:8]:
                row = []
                for b in self.bands:
                    grid = torch.linspace(2 * b / 6, 2 * (b + 1) / 6, 1025, device=graph.device)
                    diff = action_operator_response(action, grid) - nominal_operator_response(grid, baseline)
                    row.append(diff.abs().max())
                sensitivities.append(torch.stack(row))
            self.sensitivities_by_baseline[baseline] = torch.stack(sensitivities)
        # Retain the shared-block table for callers inspecting registry geometry.
        # Runtime measurements select the actual next site's nominal baseline.
        self.sensitivities = self.sensitivities_by_baseline["shared"]
        self.names = ["entropy", "margin", "source_js", "effective_rank_entropy",
                      "log1p_dirichlet", "relative_depth", "normalized_mass",
                      "normalized_hops", "log_state_energy"]
        self.names += [f"band_power_{b}" for b in self.bands]
        self.names += [f"source_alignment_{b}" for b in self.bands]
        self.names += [f"degree_weighted_power_{b}" for b in self.bands]
        self.names += [f"reachability_A{i}" for i in range(1, 8)]
        self.entropy_index = 0
        self.feature_dim = len(self.names)
        self._prepared = False
        self.reset_counters()

    def nominal_baseline(self, depth: int, max_depth: int) -> str | None:
        """Next controlled site's nominal transport; no next action at terminal."""
        if int(depth) != depth or depth < 0 or depth > max_depth:
            raise ValueError("Measurement depth must lie on the registered trajectory")
        if self.nominal_baselines is not None and len(self.nominal_baselines) != max_depth:
            raise ValueError("Measurement depth disagrees with the designated-site plant")
        if depth == max_depth:
            return None
        return "shared" if self.nominal_baselines is None else self.nominal_baselines[int(depth)]

    def reset_counters(self) -> None:
        self.counters = {"measurements": 0, "preparations": 0,
                         "dense_multiply_adds": 0, "spectral_products": 0,
                         "rank_eigendecompositions": 0}

    def _pool(self, h: Tensor, mask: Tensor | None = None) -> Tensor:
        m = self.mask if mask is None else mask.to(h.device)
        return h if m is None else h[m]

    def _project(self, h: Tensor) -> Tensor:
        hf = h.float()
        if self.projection is None:
            return hf
        self.counters["dense_multiply_adds"] += h.shape[0] * self.hidden_dim * self.sketch_dim
        with torch.autocast(device_type=h.device.type, enabled=False):
            return hf @ self.projection

    def _filter(self, h: Tensor) -> list[Tensor]:
        # Rescaled normalized Laplacian L-I = -S, with eight products reused
        # for all three windows. The weighted-state pass is charged separately.
        t0 = h.float()
        result = [self._active_coefficients[b, 0] * t0 for b in range(3)]
        t1 = -self.graph.symmetric(t0)
        self.counters["spectral_products"] += 1
        for b in range(3):
            result[b] = result[b] + self._active_coefficients[b, 1] * t1
        for k in range(2, self.degree + 1):
            t2 = -2.0 * self.graph.symmetric(t1) - t0
            self.counters["spectral_products"] += 1
            for b in range(3):
                result[b] = result[b] + self._active_coefficients[b, k] * t2
            t0, t1 = t1, t2
        return result

    @torch.no_grad()
    def prepare(self, h0: Tensor, logits0: Tensor) -> None:
        if h0.shape != (self.graph.num_nodes, self.hidden_dim):
            raise ValueError("source state shape mismatch")
        self.counters["preparations"] += 1
        expected_rows = self.graph.num_graphs if self.task.prediction_level == "graph" else self.graph.num_nodes
        if logits0.ndim != 2 or logits0.shape[0] != expected_rows:
            raise ValueError("Source logits do not match the task prediction level")
        self.source_probabilities = probabilities(logits0.detach(), self.task)
        self.source_filtered = (self._filter(self._project(h0.detach()))
                                if self.profile == "full" else None)
        self._prepared = True

    @torch.no_grad()
    def measure(self, h: Tensor, logits: Tensor, depth: int, max_depth: int,
                accumulated_mass: float = 0.0, accumulated_hops: float = 0.0,
                mask: Tensor | None = None) -> Tensor:
        if not self._prepared:
            raise RuntimeError("Call prepare(h0, logits0) before measure")
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        baseline = self.nominal_baseline(depth, max_depth)
        self.counters["measurements"] += 1
        eps = 1e-12
        pooled_h = self._pool(h.float(), mask)
        if self.task.prediction_level == "graph":
            if mask is not None:
                raise ValueError("A node pooling mask cannot index graph-level predictions")
            p, p0 = probabilities(logits, self.task), self.source_probabilities
        else:
            p = self._pool(probabilities(logits, self.task), mask)
            p0 = self._pool(self.source_probabilities, mask)
        if not p.shape[0]:
            raise ValueError("pooling mask must include at least one node")
        entropy_rows, margin_rows, js_rows = confidence_statistics(p, p0, self.task)
        entropy, margin, js = entropy_rows.mean(), margin_rows.mean(), js_rows.mean()
        # A fixed small sketch followed by a <=32x32 covariance eigensolve,
        # never an SVD of a node-by-hidden state or a dense node operator.
        with torch.autocast(device_type=h.device.type, enabled=False):
            small = pooled_h @ self.rank_projection
            gram = small.T @ small
            singular = torch.linalg.eigvalsh(gram).clamp_min(0).sqrt()
        self.counters["dense_multiply_adds"] += pooled_h.shape[0] * (
            self.hidden_dim * self.rank_width + self.rank_width**2)
        self.counters["rank_eigendecompositions"] += 1
        singular_distribution = singular / singular.sum().clamp_min(eps)
        rank_entropy = -(singular_distribution * singular_distribution.clamp_min(eps).log()).sum()
        rank_entropy = rank_entropy / max(math.log(min(pooled_h.shape[0], self.rank_width)), eps)
        rank_entropy = rank_entropy.clamp(0.0, 1.0)
        dirichlet = self.graph.dirichlet(h).log1p()
        energy = pooled_h.square().sum()
        values = [entropy, margin, js.clamp_min(0.0), rank_entropy, dirichlet,
                  p.new_tensor(depth / max_depth), p.new_tensor(accumulated_mass / max_depth),
                  p.new_tensor(accumulated_hops / max_depth), (energy + eps).log()]
        if self.profile == "full":
            projected = self._project(h)
            filtered = self._filter(projected)
            powers = torch.stack([self._pool(g, mask).square().sum() for g in filtered])
            power_fractions = powers / (powers.sum() + eps)
            alignments = []
            for current, source in zip(filtered, self.source_filtered):
                current, source = self._pool(current, mask), self._pool(source, mask)
                alignments.append((current * source).sum() /
                                  (current.norm() * source.norm() + eps))
            weighted = projected * self.graph.sqrt_degree[:, None]
            weighted_filtered = self._filter(weighted)
            weighted_powers = torch.stack([self._pool(g, mask).square().sum() for g in weighted_filtered])
            weighted_fractions = weighted_powers / (weighted_powers.sum() + eps)
            # Degree-weighted powers are the proper row-stochastic bound
            # coordinate. This is a partial statistic, not a certified bound:
            # omitted-band, polynomial/grid error and retention terms are excluded.
            # No action remains at terminal depth, so its seven reach coordinates
            # are zero. They otherwise compare with the actual next site's A0.
            with torch.autocast(device_type=h.device.type, enabled=False):
                reach = (weighted_powers.new_zeros(7) if baseline is None else
                         (self.sensitivities_by_baseline[baseline].square()
                          @ weighted_powers).clamp_min(0).sqrt())
            values.extend(power_fractions.unbind())
            values.extend(alignments)
            values.extend(weighted_fractions.unbind())
            values.extend(reach.unbind())
        else:
            values.extend(p.new_zeros(16).unbind())
        result = torch.stack(values).float()
        if not torch.isfinite(result).all():
            raise FloatingPointError("Non-finite sensor value; controller must recover")
        return result

    def metadata(self) -> dict:
        return {"schema": self.names, "dimension": self.feature_dim,
                "observer_innovation_features_external": 3,
                "profile": self.profile, "bands": list(self.bands),
                "candidate_band_edges": [2 * i / 6 for i in range(7)],
                "degree": self.degree, "jackson_coefficients": self.coefficients.cpu().tolist(),
                "sketch_dim": self.sketch_dim, "rank_sketch_width": self.rank_width,
                "seed": self.seed, "pooling_mask": self.mask is not None,
                "task": self.task.to_dict(),
                "transport_history_definition": TRANSPORT_HISTORY_DEFINITION,
                "confidence_definition": ("mean Bernoulli entropy; mean absolute 2p-1 margin; mean Bernoulli source JS"
                                          if self.task.task_type == "multilabel" else "categorical entropy; top-two margin; categorical source JS"),
                "episode_scope": "one independent graph" if self.graph.graph_ptr is not None else "transductive graph",
                "dirichlet_pooling": "full_graph",
                "reconstruction_choice": "equal intervals; active bands fixed before labels; not recovered original windows",
                "reachability_nominal_responses": {"shared": "0.3 + 0.7*(1-lambda)",
                                                   "native_mean": "1-lambda",
                                                   "inserted_identity": "1"},
                "reachability_baseline_by_site": (list(self.nominal_baselines)
                                                  if self.nominal_baselines is not None else "shared"),
                "reachability_terminal": "zero: no next controlled action",
                "reachability_grid_points_per_band": 1025,
                "reachability_status": "partial degree-weighted transport statistic before source retention; not a certified displacement bound"}
