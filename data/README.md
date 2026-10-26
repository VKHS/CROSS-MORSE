# Data sources and split identities

[`sources.json`](sources.json) is a machine-readable catalogue of the seven
official data sources and this implementation's split rules. Raw provider data
are downloaded to a separate cache; this directory contains only documentation
and the source catalogue. Complete prepared **synthetic** datasets, including
their exact split arrays, are included in [`examples/runs/`](../examples/runs/); the
registered split rule and seeds of every official-data configuration are in
[`manifests/registered/`](../manifests/registered/), and the membership hashes a
completed run actually used are extracted into `manifests/<label>/` by
`make manifests`.

## Official datasets

| Dataset | Prediction task | Official split used | Statistical unit in this implementation |
|---|---|---|---|
| ogbn-arxiv | Node classification; accuracy | OGB time split | METIS core; full graph remains available to propagation |
| ogbn-products | Node classification; accuracy | OGB sales-ranking split | METIS core; full graph remains available to propagation |
| ogbn-proteins | 112 binary node targets; macro ROC-AUC | OGB species split | Degree-stratified core |
| Roman-empire | Node classification; accuracy | One of the ten official mask selections, indexed `0`–`9` | METIS core |
| PascalVOC-SP | Node classification; macro F1 | Official train/validation/test graph files | Complete graph |
| COCO-SP | Node classification; macro F1 | Official train/validation/test graph files | Complete graph |
| Peptides-func | Ten binary graph targets; macro average precision | Official train/validation/test graph files | Complete graph |

The primary source references are the [OGB node-property documentation](https://ogb.stanford.edu/docs/nodeprop/),
the [Roman-empire authors' repository](https://github.com/yandex-research/heterophilous-graphs),
and the [Long Range Graph Benchmark repository](https://github.com/vijaydwivedi75/lrgb).
The [PyG LRGB loader](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/lrgb.html)
documents the raw split archives used by the adapter. Exact download URLs and
expected cache paths are in `sources.json` and
[`cross_morse/datasets.py`](../cross_morse/datasets.py). OGB resolves its archive
through the installed OGB package's dataset/version metadata.

## Executable role allocation

Only the official **training** units are permuted and divided into `fit`,
`explore`, `select`, and `confirm`. The implementation sorts the official IDs,
uses `numpy.random.default_rng(config.seed)`, and cuts the permutation at
`floor(0.6N)`, `floor(0.8N)`, and `floor(0.9N)`. The remaining training units
form `confirm`. A separate generator, seeded by `SeedSequence([seed, 137])`,
divides `fit` into `fit_train` and `fit_val`. The latter controls checkpoint
selection. Official validation becomes `admit`; official test becomes `test`.
The exact implementation is in [`cross_morse/data.py`](../cross_morse/data.py).

For LRGB, these operations allocate complete graphs. PascalVOC-SP and COCO-SP
then map each allocated graph to all its node-label rows; Peptides-func uses one
label row per graph. `graph_splits.npz` retains graph identities, and
`splits.npz` retains prediction-row identities. Model execution uses individual
graph episodes. Transductive cores are dependent statistical units and do not
make the underlying graph into independent observations.

This is the executable separated-role protocol. The manuscript's reported
measurements describe reused validation data; the generated split rule is not
presented as a recovered list of those historical memberships. See
[`docs/THEORY_AND_SCOPE.md`](../docs/THEORY_AND_SCOPE.md) and
[`docs/PAPER_RESULTS.md`](../docs/PAPER_RESULTS.md).

## What identifies a prepared run

| Run artifact | Recorded information |
|---|---|
| `prepared/graph.npz` | Numeric features, canonical edges and labels; graph/edge pointers and edge channels where applicable |
| `prepared/official_splits.npz` | Actual provider train/validation/test IDs, or the fixture's generated counterparts |
| `prepared/splits.npz` | Exact prediction-row IDs for all eight roles, including the internal fit split |
| `prepared/graph_splits.npz` | Exact graph-role IDs for independent-graph datasets |
| `prepared/clusters.npz` | Statistical-unit ID for every prediction row |
| `prepared/metadata.json` | Source and transformation metadata, task dimensions, split sizes, array/file hashes and preparation fingerprint |
| `provenance/stages/00_prepare.json` | File seals for preparation, effective configuration, environment and evaluator snapshot |

The run metadata hashes source arrays and prepared files. Roman-empire and LRGB
also record hashes of their raw input files. OGB records the dataset version,
official split and array identities; the implementation does not invent a
historical provider ZIP checksum. The provider data remain subject to their
original terms.

## Prepare, verify, and preserve

From the repository root, a configured official pilot can be prepared with:

```bash
python -B run_cross_morse.py --dataset ogbn-arxiv --profile pilot \
  --data-root /path/to/dataset-cache --output-dir /path/to/arxiv_pilot \
  --device cpu --stage prepare
```

The committed pilot/full configurations set `data.download=true`. This permits
the adapter to fetch missing official files; it does not replace an existing
run or silently delete an older dataset cache. An existing compatible run can
be continued with `--resume`. Prepared data, splits and seeds are immutable
within that run's provenance chain.

Check the catalogue and all configured experiment profiles without downloading
data or importing NumPy/PyTorch:

```bash
python -B scripts/artifact_inventory.py --check-static
```

After an intentional adapter/configuration change, review it and regenerate the
two derived static manifests with `--write-static`. Actual benchmark data,
historical author split files and paper-result checkpoints are not created by
that command. See [`docs/ARTIFACTS.md`](../docs/ARTIFACTS.md) for complete-run
inventory and replay commands.
