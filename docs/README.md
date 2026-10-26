# CROSS-MORSE documentation

[Repository home](../README.md) · [Availability](../AVAILABILITY.md) · [Status](STATUS.md) · [Paper build and registry](../paper/README.md)

## Understand the method

| Read | What it answers |
|---|---|
| [Method overview](METHOD_OVERVIEW.md) | What is controlled, what stays frozen, and how open-loop and feedback policies are compared |
| [Theory and scope](THEORY_AND_SCOPE.md) | What the theoretical statements assume, what they establish, and which conditions the reported experiments satisfy |
| [Manuscript-to-code map](METHOD_MAPPING.md) | Which source object implements every equation, proposition, algorithm, table and experiment of the manuscript |
| [Implementation contract](IMPLEMENTATION_CONTRACT.md) | The exact architecture, estimators, information boundaries and statistical interpretation the implementation uses |
| [Known differences](../paper/KNOWN_DIFFERENCES.md) | Where the released source departs from the manuscript's description, with status |

## Run and inspect an experiment

| Read | What it answers |
|---|---|
| [Getting started](GETTING_STARTED.md) | Installing, executing a smoke, pilot or full run, validating and resuming |
| [Included executable examples](../examples/README.md) | Seven complete synthetic runs with data, split indices, checkpoints, predictions and trajectories |
| [Observed environment](../environment/README.md) | The exact CPU package lock, installer and compatibility checker |
| [Data acquisition](../data/README.md) | Official provider sources, download/preparation commands and cache boundaries |
| [Experiment profiles](../experiments/README.md) | Configuration identities, partition seeds, training seeds and split choices for all 21 profiles |
| [Datasets](DATASETS.md) | Tasks, official splits, raw files, sizes and model configurations for the seven datasets |
| [Compute budget](COMPUTE_BUDGET.md) | What a full reproduction costs as the manuscript measured it, and what each profile requests |
| [HPC and Slurm](HPC.md) | The generic launcher contract: cluster profiles, environment selection, submission, campaigns and resume |
| [Arrhenius](ARRHENIUS.md) | How we run the seven datasets on the Arrhenius GH200 cluster: site profile, job shape, storage layout, environment capture, archiving |
| [Registered and per-run manifests](../manifests/) | The registered split rule, seeds, folds and margins of every configuration, and the role membership hashes, seeds and environment each completed run used |
| [Run archives](RUN_ARCHIVES.md) | What a completed run contains, and how it is packed, verified, deposited, ingested and turned into the paper's tables and figures |
| [Measurement records](../records/README.md) | The record sets of the experiments outside the eight-stage run (S15/S15b, S16/S16b, S19, S23, S23b, S6c, S25, S26b, S45, S46/S46b): layout, value-by-value trace to raw logs, printed tables |

## Review the evidence

| Read | What it answers |
|---|---|
| [Reviewer guide](REVIEWER_GUIDE.md) | The reviewers' list item by item, and an ordered review path from source inspection through numerical evidence to executable checks |
| [Availability statement](../AVAILABILITY.md) | The Data and Code Availability statement generated from the repository state, with the item-by-item table of what is present |
| [Status](STATUS.md) | What is complete, what is pending on Arrhenius and deposit, and the open code–manuscript differences |
| [Artifact contract](ARTIFACTS.md) | What is included, which files establish each claim, and how the complete inventories are verified |
| [Paper build and registry](../paper/README.md) | The two roles of `paper/`: tables and figures built from runs, and the transcribed manuscript registry used for comparison only |
| [Example paper tables](../paper/generated/examples/tables/T1.md) | The paper's tables built from the seven synthetic example runs (`make example-paper`); the official build lands in `paper/generated/official/` |
| [Selection readings (Table S12c)](../paper/generated/examples/tables/S12c.md) | For every run, the three equivalence readings of main Eq. (19), the route selected and the registered rule checked against the sealed selection |
| [Recomputed example results](../examples/generated/metrics.md) | Scores and decisions regenerated from the included predictions and labels |
| [Paper results](PAPER_RESULTS.md) | All seven datasets together as transcribed from the submitted manuscript: paired effects, absolute official scores, compute and interpretation |
| [Results and evidence](RESULTS.md) | Record fields, how a run is compared with the manuscript, how a value is bound to its run |
| [Claims ledger](../paper/claims_ledger.md) | Every quantitative claim of the manuscript, its table, its run artifact and its status in this release |
| [Reported results](../paper/REPORTED_RESULTS.md) | The manuscript's tables, rendered from the machine-readable registry |
| [Reproducibility checklist](REPRODUCIBILITY_CHECKLIST.md) | The standard checklist, item by item, with the file that satisfies each |
| [Source provenance](../paper/sources.json) | PDF byte identities, page references and evidence classifications for 32 numerical groups |
| [Validation record](VALIDATION.md) | Which local software checks were executed, in which environment, with what outcome |

## Contribute

[Contributing](../CONTRIBUTING.md) covers reproducible reports, scientific changes, the registry rules, validation and
source packaging. The issue forms distinguish implementation defects, reproduction questions and research discussions.

## Figures

[System architecture](assets/architecture.svg) ·
[Eight-stage lifecycle](assets/lifecycle.svg) ·
[One control depth and the action registry](assets/control_depth.svg) ·
[Nine policy families](assets/policy_families.svg) ·
[Evidence tiers](assets/review_workflow.svg) ·
[Evidence chain: source → run → archive → tables](assets/evidence_chain.svg) ·
[Decision chain: statistics → run files → table cells](assets/decision_chain.svg) ·
[Figure 1 of the manuscript](assets/figure1_lifecycle.png) (system and policy lifecycle) ·
[Method schematic](assets/method_overview.svg) ·
[Reported performance and inference work](assets/reported_results.svg) ·
[Figure sources and regeneration](assets/README.md)
