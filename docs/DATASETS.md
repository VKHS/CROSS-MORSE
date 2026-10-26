# Datasets and task semantics

The same control lifecycle runs on seven benchmarks. Each adapter preserves the
official train, validation and test allocation and makes the prediction axis
explicit. The OGB and Roman-empire datasets are single transductive graphs; the
three LRGB datasets are corpora of complete graphs, executed one graph at a time.

## Benchmark matrix

| Dataset | Prediction rows | Outputs | Loss | Reporting metric | Official split | Input features |
|---|---|---:|---|---|---|---|
| ogbn-arxiv | Nodes in one graph | 40 classes | Cross-entropy | Accuracy | Publication time | 128 node features |
| ogbn-products | Nodes in one graph | 47 classes | Cross-entropy | Accuracy | Sales ranking | 100 node features |
| ogbn-proteins | Nodes in one graph | 112 binary tasks | Mean binary cross-entropy | Mean ROC-AUC over eligible tasks | Species | Mean incident-edge features, 8 channels |
| Roman-empire | Nodes in one graph | 18 classes | Cross-entropy | Accuracy | Ten official train/validation/test masks | 300 node features |
| PascalVOC-SP | Nodes within complete image graphs | 21 classes | Cross-entropy | Macro F1 | Official graph split | 14 node features; 2 edge attributes |
| COCO-SP | Nodes within complete image graphs | 81 classes | Cross-entropy | Macro F1 | Official graph split | 14 node features; 2 edge attributes |
| Peptides-func | One row per complete molecule graph | 10 binary tasks | Mean binary cross-entropy | Mean average precision over eligible tasks | Official graph split | 9 categorical atom fields; 3 categorical bond fields |

The OGB task definitions and official splits are documented by the
[Open Graph Benchmark](https://ogb.stanford.edu/docs/nodeprop/). Roman-empire's
arrays and ten masks come from the
[heterophilous-graphs repository](https://github.com/yandex-research/heterophilous-graphs).
The independent-graph tasks and metrics follow
[LRGB](https://github.com/vijaydwivedi75/lrgb); the distributed tuple formats and
COCO label mapping are also specified in the
[PyG LRGB loader](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/lrgb.html).
The complete LRGB corpora contain 11,355 PascalVOC-SP graphs, 123,286 COCO-SP
graphs and 15,535 Peptides-func graphs. Real preparation rejects truncated
corpora; a pilot reduces the compute configuration while retaining the dataset.

| Dataset | Official size (provider statistics) | Official split | Statistical unit in this study (Supplementary Table S1) |
|---|---|---|---|
| ogbn-arxiv | 169,343 nodes; 1,166,243 directed edges | time (train / validation / test by year) | 32 METIS cores with a 2-hop buffer (dependent) |
| ogbn-products | 2,449,029 nodes; 61,859,140 edges | sales ranking | 64 METIS cores with a 2-hop buffer (dependent) |
| ogbn-proteins | 132,534 nodes; 39,561,252 edges | species | 32 degree-stratified cores with a 2-hop buffer (dependent) |
| Roman-empire | 22,662 nodes; 32,927 undirected edges (65,854 directed entries after canonicalization) | ten official masks | the official split; paired analysis over the ten splits |
| PascalVOC-SP | 11,355 graphs (superpixel nodes with 21 classes) | official graph split | complete graph; 1,428 validation graphs (admission), 850 D_confirm graphs |
| COCO-SP | 123,286 graphs (81 classes) | official graph split | complete graph; 5,000 validation graphs (admission), 11,329 D_confirm graphs |
| Peptides-func | 15,535 graphs (10 binary tasks) | official graph split | complete graph; 2,331 validation graphs (admission), 1,087 D_confirm graphs |

The OGB counts are the providers' published statistics; the Roman-empire counts
are the ones recorded by our preparation check
(`validation_evidence/roman_official_preparation.json`); the D_confirm graph counts
are the manuscript's (Supplementary Table S39). A run records its own counts in
`prepared/*.json`.

[`cross_morse/tasks.py`](../cross_morse/tasks.py) defines losses, probability
transforms and metrics. Multiclass heads use softmax; multilabel heads use
independent sigmoids. Missing multilabel entries are masked in BCE and metrics.
Losses average observed tasks within each prediction row. ROC-AUC and average
precision are computed from pooled targets and predictions for each eligible
task; they are not averages of per-node or single-graph scores. Macro F1 is also
recomputed from pooled class predictions. Stored metric values are fractions,
not percentages.

## Preparation and label roles

The adapter divides **only the official training allocation** into fit, explore,
select and confirm using a seeded permutation with cumulative floor cuts at
60%, 80% and 90%. A separate permutation divides fit into `fit_train` and
`fit_val`; the latter selects the frozen checkpoint. Official validation is
`admit`, and official test remains `test`.

For transductive benchmarks, this partition assigns nodes to label roles on a
shared graph. For LRGB, it assigns **whole graphs** to roles. A PascalVOC-SP or
COCO-SP image therefore cannot contribute some nodes to fitting and other nodes
to admission. A Peptides-func graph supplies exactly one prediction row.

| Runtime stage | Labels available through the stage-scoped accessor |
|---|---|
| `prepare` | None through the runtime accessor; offline preparation validates and stores the official arrays |
| `train` | `fit_train`, `fit_val`, `fit` |
| `explore` | `explore` |
| `fit` | Previously logged exploration targets; no direct dataset-label access |
| `select` | `select`; `explore` for the training-only entry diagnostic |
| `confirm` | `confirm` |
| `admit` | `admit` |
| `evaluate` | `test` |

`GraphData.graph(g)` creates a graph view without reading its labels. It shares
the parent dataset's active label scope, including when the view is constructed
before that scope begins. Online policy execution receives graph inputs and
plant predictions rather than a label accessor. These are application-level
boundaries; the offline prepared files contain the complete labelled dataset.

The clean allocation above follows Supplementary Table S2. The supplementary
discussion distinguishes this separated sequence from the submitted LRGB
protocol that reused validation for confirmation and admission. This code's
separated allocation does not make the submitted descriptive statistics into
formal guarantees. Statistical assumptions and decision gates are described in
[the method overview](METHOD_OVERVIEW.md) and
[implementation contract](IMPLEMENTATION_CONTRACT.md).

## Graph representation and weighted transport

All adapters produce undirected, row-sorted COO edges without self-loops.
Duplicate and reciprocal edge attributes are combined by arithmetic mean and
remain aligned with the canonical edge list. The graph operator supplies
identity rows for isolated nodes. Edge weights, when present, are finite and
nonnegative; they enter the degree-normalized random-walk and symmetric kernels.

**Proteins.** The node input is the arithmetic mean of the eight edge-feature
channels over incident edges at both endpoints; an isolated node receives
zeros. The scalar transport weight is the mean of those eight channels. The
species split is read from official CSV indices. The manuscript specifies
edge-weighted transport but leaves the scalar reduction unspecified, so the
mean reduction is recorded as an explicit implementation convention.

**Peptides.** Each atom field is one-hot encoded using the fixed OGB 1.3.6
vocabulary sizes `(119, 5, 12, 12, 10, 6, 6, 2, 2)`, producing 174 input columns.
The original three bond attributes are retained. The scalar transport weight is
`1 + mean(bond categorical codes)`, a positive, deterministic convention recorded
in the dataset metadata. This reduction is an implementation choice because
the manuscript does not specify the mapping from bond categories to scalar
weights. The categorical vocabulary is defined in
[OGB's feature utilities](https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py).
The graph-level head pools hidden states within the current molecule only.

**COCO.** Raw category IDs are sparse within the range 0–90. The reader applies
the official mapping to consecutive labels 0–80 and rejects IDs outside the
81-category vocabulary.

For transductive statistical units, the full profiles register 32 METIS cores
for arxiv, 32 degree-stratified cores for proteins and 64 METIS cores for products,
with two-hop buffer metadata. The proteins implementation partitions nodes into
degree-rank quantiles, shuffles each stratum with the registered seed and
assigns balanced cores by round-robin allocation. This makes the unspecified
degree-stratification algorithm reproducible; it does not require connected
cores. METIS and degree-stratified cores remain dependent units within one
graph. For Roman-empire, a run identifies one official split; the ten-split
campaign and paired split summary retain that distinction. Within-run core
diagnostics do not replace the repeated-split analysis.

## Official raw files and downloads

Downloads occur only when `data.download=true` and the required raw files are
absent. The real `pilot` and `full` profiles set this flag; `smoke` does not need
network access. `--data-root` selects the shared cache. The cluster's existing
`CM_DATA_ROOT` path can hold all seven datasets, even if its directory name is
`ogb_cache`.

| Dataset | Raw files beneath the data root |
|---|---|
| ogbn-arxiv | `ogbn_arxiv/raw/` CSV files and `ogbn_arxiv/split/time/{train,valid,test}.csv.gz` |
| ogbn-products | `ogbn_products/raw/` CSV files and `ogbn_products/split/sales_ranking/{train,valid,test}.csv.gz` |
| ogbn-proteins | `ogbn_proteins/raw/` CSV files and `ogbn_proteins/split/species/{train,valid,test}.csv.gz` |
| Roman-empire | `roman_empire/raw/roman_empire.npz` |
| PascalVOC-SP | `pascalvoc-sp/raw/{train,val,test}.pickle` |
| COCO-SP | `coco-sp/raw/{train,val,test}.pickle` |
| Peptides-func | `peptides-func/raw/{train,val,test}.pt` |

The Roman archive contains `node_features`, `node_labels`, `edges` and three
mask arrays of shape `[10, num_nodes]`. `--split-index 0` through
`--split-index 9` selects a mask before the training-role subdivision. The
[official heterophilous dataset loader](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/heterophilous_graph_dataset.html)
documents this format.

LRGB raw files contain lists of `(x, edge_attr, edge_index, y)` tuples. The
adapter accepts the official extracted superpixel directories as well as the
cache layout above. The published raw sources are:

- [Roman-empire NPZ](https://github.com/yandex-research/heterophilous-graphs/raw/main/data/roman_empire.npz).
- [PascalVOC-SP archive](https://www.dropbox.com/s/8x722ai272wqwl4/pascalvocsp.zip?dl=1).
- [COCO-SP archive](https://www.dropbox.com/s/r6ihg1f4pmyjjy0/cocosp.zip?dl=1).
- [Peptides-func archive](https://www.dropbox.com/s/ycsq37q8sxs1ou8/peptidesfunc.zip?dl=1).

The adapter validates completed downloads before committing them. It reads OGB
raw CSV files and ignores legacy processed and split pickle caches. Superpixel
raw files use a restricted Tensor/NumPy unpickler; peptide tensor archives use
`torch.load(..., weights_only=True)`. Unsupported Python objects fail explicitly.
There is no unsafe-deserialization override. PyG and RDKit are not required to
read the distributed graphs; PyTorch and the dependencies in
[`requirements.txt`](../requirements.txt) suffice.

An existing incomplete raw directory is preserved rather than overwritten.
Move or repair that directory before retrying. Cached prepared data are checked
against their source/configuration fingerprint; a different split or
preprocessing rule requires a separate output directory.

Example: prepare the official Roman-empire data for split 0 without training:

```bash
python run_cross_morse.py --dataset Roman-empire --profile pilot \
  --split-index 0 --stage prepare --device cpu \
  --data-root /path/to/dataset-cache --output-dir /path/to/roman-split-0
```

The dedicated Python entry points and cluster launchers use the same adapters.
See [getting started](GETTING_STARTED.md) and the
[Arrhenius guide](HPC.md) for execution commands.

## Prepared artifacts and resource scope

[`cross_morse/data.py`](../cross_morse/data.py) writes numeric NPZ files and JSON
metadata. `graph.npz` holds features, canonical edges and labels, plus optional
`edge_attr`, `edge_weight`, `graph_ptr` and `edge_ptr`. `splits.npz` indexes the
prediction rows; `graph_splits.npz` additionally indexes whole graphs for LRGB.
`clusters.npz` identifies the resampling unit of each prediction row. Array,
split and file hashes bind these values to the saved preparation configuration.

LRGB corpora reside on the CPU as disjoint unions. Graph and edge pointers
define exact episode boundaries; cross-graph edges are rejected. The training
and execution paths move one complete graph at a time to the selected device.
`global_sample_ids` maps its local predictions back to the corpus, preserving
the node axis for superpixels and graph axis for peptides. The sensor never
pools multiple independent graphs into one controller observation. Training
can accumulate gradients over a registered number of complete episodes.

Transductive execution retains the complete graph and hidden-state arrays.
Dense projection chunking, where configured, limits a projection's temporary
working size; it does not make the sparse graph or all hidden states local.
METIS buffers are recorded context diagnostics, not an implementation of
partition-local halo minibatching. This distinction matters particularly for
ogbn-products and for the manuscript's partition-aware plant description.
Full graph state, weighted sparse kernels, spectral probes, exploration
trajectories and saved predictions can require substantial host memory, GPU
memory and storage. CPU packing of COCO-SP is also substantial even though GPU
execution uses separate episodes. No full-profile throughput or memory-budget
claim follows from the small adapter tests.

`smoke` profiles generate synthetic inputs with the requested benchmark's task
type and prediction axis. Their `synthetic_classes` field controls the number
of output columns, including multilabel tasks, and they are marked
`scientific_run=false` and `is_benchmark_data=false`. They exercise the lifecycle
without downloading a benchmark. Actual raw readers retain the official output
counts in the table above; fixture checks separately verify the 112 protein
tasks and 10 peptide tasks. Adapter validation and synthetic lifecycle execution
do not establish reproduced paper metrics or completed full benchmark runs.
