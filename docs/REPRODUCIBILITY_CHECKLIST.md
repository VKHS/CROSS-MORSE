# Reproducibility checklist

[Repository home](../README.md) · [Reviewer guide](REVIEWER_GUIDE.md) · [Validation](VALIDATION.md)

The table describes the evidence we ship with this source. It distinguishes a
provided implementation or record from an experiment that has actually been
executed. Exact local validation outcomes and environment versions are recorded
in [VALIDATION.json](VALIDATION.json).

## Research specification

| Item | Evidence shipped | Interpretation |
|---|---|---|
| Method and assumptions | Our manuscript and supplement (identified by SHA-256 in [`paper/sources.json`](../paper/sources.json)); [theory guide](THEORY_AND_SCOPE.md) | Exact identities, deterministic bounds and conditional statements are distinguished |
| Dataset coverage | Seven task adapters, separate launchers, 21 profiles; [dataset guide](DATASETS.md) | Pilot/full profiles use official data; smoke profiles use task-specific fixtures |
| Policy and action definitions | Ten actions, nine policy families; [method map](METHOD_MAPPING.md) | Constants checked against the paper's registered values |
| Architectural choices | [Implementation contract](IMPLEMENTATION_CONTRACT.md), [differences ledger](../paper/KNOWN_DIFFERENCES.md) | Model counts, sites, windows and execution choices are explicit |
| Manuscript observations | [Reported-results JSON](../paper/reported_results.json), [paper results guide](PAPER_RESULTS.md) | Our reported measurements transcribed with original units and source locations, for comparison only |
| Source-document identity | [Paper provenance](../paper/sources.json) | Exact PDF files and SHA-256 values accompany the records |

## Execution and information boundaries

| Item | Evidence shipped | Interpretation |
|---|---|---|
| Environment | [Observed CPU environment and exact package lock](../environment/README.md), `requirements.txt`, `pyproject.toml` | CPU examples have a recorded environment; cluster execution selects a host-appropriate Torch build |
| Included execution archives | [Seven complete synthetic examples](../examples/README.md), [artifact inventories](../examples/inventory) | Actual data, split/seed records, checkpoints, trajectories, predictions and statistical outputs are included |
| Static experiment manifests | [Data sources](../data/sources.json), [profile and seed manifest](../experiments/profiles.json) | Generated from the implemented adapters and 21 configurations; not invented historical run identities |
| Data preparation | Official splits, feature processing, metadata and prepared-data identities | All ten Roman masks are supported |
| Label roles | Separate fit/explore/select/confirm/admit/test access | The reported validation reuse is not silently reproduced as independent admission |
| Frozen predictor | Model tensor inventories and nominal-identity checks | Predictor weights and head remain fixed during controller development |
| Graph isolation | Per-graph episodes, graph-grouped holdouts and whole-unit resampling | Graph collections retain their statistical unit |
| Comparator selection | Nested reconstruction in max-t resampling | No fixed winner is substituted for the nested comparator |
| Admission and fallback | Locked candidate and exact comparator return | Coverage still requires the theorem's actual assumptions |
| Source and output integrity | Eight linked stage manifests and complete source inventory | Byte identities detect modification; they do not prove chronology |
| Resume | Stage and output verification before reuse | Incompatible source/configuration changes require a separate run |

## Verification and measurements

| Item | Status location | Interpretation |
|---|---|---|
| Unit and integration tests | [Validation record](VALIDATION.md) and raw logs | Actual recorded local checks, including their skips |
| Seven task lifecycles | Per-task numerical and checkpoint replay reports | Software behavior on small synthetic fixtures |
| Benchmark measurements | Main and supplementary tables and their registry; run archives under `runs/` once ingested ([run archives](RUN_ARCHIVES.md)) | Our reported outcomes, transcribed; not synthetic scores. The tables are rebuilt from the archives by `make paper` |
| Run-to-paper comparison | `scripts/compare_with_manuscript.py` | Sealed official-run observations compared descriptively |
| Roman aggregate | `scripts/summarize_roman_splits.py`, `--roman-campaign` comparison | Ten paired splits on one graph, not independent graphs |
| Figure regeneration | Registry and run plotting scripts, `scripts/draw_diagrams.py` for the block diagrams; [assets guide](assets/README.md) | Analytical, reported-measurement and computed-run plots keep separate inputs; the block diagrams read the registry and are checked against it |
| Makefile reproduction | `make reproduce`, `make example-paper`, `make paper`, `make registry-views`, `make verify` | Complete example observations are replayed; the paper's run-derived tables and figures are built from run directories (examples now, official archives once ingested); registry views use the transcribed measurements for comparison only; checks do not silently train or download |
| Original result artifacts | [Run archives](RUN_ARCHIVES.md), [status](STATUS.md), [results guide](RESULTS.md) and binding tools | Our official-data run archives are deposited and ingested through the run-archive contract; none is registered yet, and the generated [availability statement](../AVAILABILITY.md) says so |
| Timing and resources | [Compute guide](COMPUTE_BUDGET.md) | Reported GH200 measurements are distinct from checkout requirements and local timings |
| Hosted continuous integration | `.github/workflows/quality.yml` | Workflow provided; hosted execution is recorded by GitHub when it runs |

## Reporting a repeat experiment

Retain the resolved configuration, dataset/preprocessing identity, statistical
units, full candidate ledger, frozen model inventory, stage manifests and final
observations. Record the hardware, software environment, wall times and actual
execution profiles. Use numerical replay and checkpoint reexecution before
claiming computational agreement. Preserve policy differences and comparator
fallbacks in the report, including when a result differs from the paper.

The strongest scientific comparison also requires the original model and
execution settings, the appropriate statistical unit, and a justified association
between the reported table and the run. Where the source uses an explicitly
different architecture or protocol, report that condition with the comparison.
The [claims ledger](../paper/claims_ledger.md) identifies which additional
experiments, comparator arms and hardware measurements remain separate from the
core implementation.
