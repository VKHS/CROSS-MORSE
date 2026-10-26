# Results and replay

The [paper-results overview](PAPER_RESULTS.md) presents the measurements we report in the submitted manuscript across all seven datasets, with exact PDF/table/page provenance, paired loss intervals, official scores, mechanism contrasts and compute accounting. Its numbers are transcribed from the submitted main manuscript and supplementary information into one canonical [results registry](../paper/reported_results.json), which we use for comparison only. The paper's tables and figures themselves are built from completed run directories by `make paper` ([paper/README.md](../paper/README.md), [RUN_ARCHIVES.md](RUN_ARCHIVES.md)).

A completed executable run produces its own evidence: frozen model identities, predictions, selection and admission decisions, and resource ledgers. The workflow below lets a reader inspect that evidence, recompute the numerical results, re-execute the checkpoints, and compare the resulting quantities with the paper. These are different review tasks, and each report names its scope.

## Choose the evidence you need

| Question | Evidence and command | What it establishes |
|---|---|---|
| What measurements does the paper report? | [Paper results](PAPER_RESULTS.md), [full table rendering](../paper/REPORTED_RESULTS.md), [CSV summary](../results/paper_summary.csv) | Our reported values transcribed at printed precision, with unavailable quantities left missing |
| Are the manuscript identities and generated registry views intact? | `make registry-check` (`scripts/render_paper_results.py --check` and `scripts/check_paper_constants.py`) | recorded SHA-256 identities of the manuscript and supplement (byte-verified when a local copy is passed), complete table/page provenance, current generated Markdown/CSV, registry laws, configured thresholds and selected cross-table identities; the Peptides score discrepancy remains explicit |
| Which tables and figures are built from run directories? | `make example-paper` (seven synthetic examples), `make paper` (official archives in `runs/index.json`) | Every table and figure the run contract supports, under `paper/generated/<label>/`; `make example-paper-check` verifies the committed example build |
| Are a completed run's artifacts and parent chain intact? | `python -B scripts/validate_run_manifest.py "$RUN_ROOT" --require-complete --json` | The eight sealed stages and their actual files match the declared hashes |
| Can its statistics be reconstructed from saved observations? | `python -B scripts/reproduce_results.py "$RUN_ROOT" --output "$REPORT_ROOT/numeric_replay.json"` | Numerical replay from sealed predictions, unit membership and decisions; no training rerun |
| Do saved models reproduce the predictions? | `python -B scripts/evaluate_checkpoints.py ...` below | Checkpoint execution against saved q0/qN/qC/qD predictions, within declared tolerances |
| How do its quantities compare with the paper? | `make compare RUN_ROOT=... REPORT_ROOT=...` (one run) or `make compare-runs` (every ingested archive) | A verified numerical comparison, with protocol differences identified; not a formal replication test |
| Which software checks accompany this package? | [Validation record](VALIDATION.md) and [machine-readable evidence](VALIDATION.json) | The recorded environment, tests, synthetic lifecycle runs and official-data preparation scope |

The bundled local verification uses task-shaped synthetic fixtures; it does not supply a fresh seven-dataset benchmark run. The prediction, checkpoint, seed and timing archives of our official-data runs are deposited separately and ingested into `runs/` ([RUN_ARCHIVES.md](RUN_ARCHIVES.md)); the generated [availability statement](../AVAILABILITY.md) states which of them are registered. Until all seven are, the manuscript's numbers are our reported measurements, transcribed for comparison.

## Read a completed run

Use an existing completed run and a new report directory outside it. Reports are written separately so that the sealed scientific record remains unchanged.

```bash
RUN_ROOT=/absolute/path/to/completed_full_run
REPORT_ROOT=/absolute/path/to/review_reports
mkdir -p "$REPORT_ROOT"

python -B scripts/validate_run_manifest.py "$RUN_ROOT" --require-complete --json
python -B scripts/reproduce_results.py "$RUN_ROOT" \
  --output "$REPORT_ROOT/numeric_replay.json"
```

| Scientific object | Run artifact | Interpretation |
|---|---|---|
| `q0` upstream predictor | `final/q0_probabilities.npy`; `final/results_summary.json` → `models.q0` | Upstream reference kept separately from the controlled plant |
| `qN` nominal full-depth plant | `final/qN_probabilities.npy`; summary → `models.qN` | Frozen nominal execution, evaluated at the comparator's ensemble size |
| `qC` selected open-loop comparator | `final/qC_probabilities.npy`; `select/candidates.json`; `select/decision.json` | Selected rule, ensemble, profile and nested comparator reconstruction |
| `qD` deployment | `final/qD_probabilities.npy`; `policy_lock.json`; `admit/decision.json` | Locked feedback rule or exact global return of `qC` |
| Paired test loss | summary → `paired_qD_minus_qC_loss.{mean,lower,upper}` | CE or mean multilabel BCE in nats; negative favors `qD` |
| Paired official-metric gain | summary → `paired_qD_minus_qC_metric_pp.{mean,lower,upper}` | Percentage points; positive favors `qD` |
| Per-unit observations | `final/paired_test_units.npz` | Unit-level paired losses and sample membership used in uncertainty calculations |
| Eligible candidate family | `select/candidate_ledger.csv`; `select/decision.json` | Feasibility, simultaneous limits, eligible indices, complexity ordering and fold choices |
| Mechanism confirmation | `confirm/paired_contrasts.json`; `confirm/decision.json` | Contrast definitions, raw and adjusted P values, margins and downgrades |
| Admission | `admit/decision.json` | The clipped effect, declared statistical unit, rule and pass/return decision |
| Models and source | `train/model_inventory.json`; `fit/frozen_models.json`; `provenance/stages/*.json` | Actual tensor counts, frozen-model hashes, source/config/unit identities and parent-linked stages |
| Work and timing | `predictions/<id>.json`; `select/candidates.json`; stage markers | Executed operation ledger and environment-specific timings, including separately recorded qualification work |

The manuscript's Table S15 reports absolute official scores for `qN`, `qC` and `qD`; it does **not** supply q0's absolute score. Absolute test losses for the four predictors are also not tabulated. The implementation records these quantities for its runs, but a missing paper value is not filled from another estimand. See [availability and units](PAPER_RESULTS.md#2-absolute-official-scores-and-what-is-not-tabulated).

### Statistical units and uncertainty

For arxiv, proteins and products, buffered graph cores are descriptive statistical units within one dependent graph. Roman's paper result aggregates ten official splits; one split's core-based interval is a diagnostic. For Pascal/COCO, complete graphs are resampling units even though prediction is node-level. Peptides predicts graph-level multilabel probabilities. A node or graph with missing targets contributes only through its valid targets; undefined official metrics are not silently replaced with zero.

Official metrics are accuracy, ROC-AUC, macro F1 or average precision according to the dataset. The run stores native fractions and an explicit percentage-point form. Native Peptides AP `0.0058` corresponds to `0.58` percentage points. Nondecomposable official metrics are recomputed on the sampled observations; they are not replaced by a mean of per-unit accuracies.

Main Table 1's test intervals include optimization-seed and evaluation-unit uncertainty, whereas Table S13 conditions on a fixed ensemble. A single run's unit bootstrap therefore has a different population from a repeated-training interval. Numerical agreement with a printed interval does not establish that the resampling hierarchies agree. Main pp6–7; Supplementary Note S24, p31.

### Returning qC is a complete outcome

A failed selection, confirmation or admission gate returns the locked comparator. The final `qD` and `qC` prediction files are then byte-identical, the paired effects are exactly zero, and the returned route and reasons remain in the record. This is the reported products regime and a valid possible outcome for a fresh run. Global comparator return, per-input fallback, recovery within a trajectory and a budget-aware late restart are separate events; a global return does not have an input-fallback fraction.

## Compare a run with the paper

```bash
make compare RUN_ROOT="$RUN_ROOT" REPORT_ROOT="$REPORT_ROOT"
# equivalently
python -B scripts/compare_with_manuscript.py "$RUN_ROOT" \
  --output "$REPORT_ROOT/paper_comparison.json"
# every archive registered in runs/index.json
make compare-runs REPORT_ROOT="$REPORT_ROOT"
```

The command verifies the run's sealed numerical evidence before producing JSON and a Markdown companion. Synthetic fixtures are not eligible for manuscript comparison. A single Roman split is labelled diagnostic; use the ten-split campaign path for its paper-level aggregate:

```bash
CAMPAIGN_ROOT=/absolute/path/to/roman_campaign
python -B scripts/compare_with_manuscript.py "$CAMPAIGN_ROOT" \
  --roman-campaign --output "$REPORT_ROOT/roman_comparison.json"
```

`exact` means agreement at the printed tolerance of the transcribed value. `interval` means the value lies inside the specified reported interval or standard-deviation band. These categories are descriptive comparisons: neither is a formal replication test, a prediction interval for a new training run, or proof that an archive produced the paper. `outside` and `differs` remain visible; a completed comparison can legitimately report them. Missing or ineligible evidence receives a distinct status. The CLI exits nonzero when evidence is invalid/incomplete or the eligible comparison has mismatches.

Three differences matter when interpreting a fresh run. The packaged source uses separated multi-graph confirmation/admission roles, while the reported measurements reused validation. Supplementary Table S14b is an **analytical sample-size sensitivity**, not a numeric target from a measured separated-data rerun. The registered selection rule differs from three reported choices on representative rows; those rows alone cannot identify the executed full selector. Finally, architecture counts, sensing configuration and products execution have declared implementation differences. [Full difference ledger](../paper/KNOWN_DIFFERENCES.md); [implementation contract](IMPLEMENTATION_CONTRACT.md).

## Re-execute checkpoints

For checkpoints whose local provenance you trust:

```bash
python -B scripts/evaluate_checkpoints.py "$RUN_ROOT" \
  --trust-local-checkpoints --device cuda --threads 2 \
  --models q0 qN qC qD \
  --output "$REPORT_ROOT/checkpoint_replay.json"
```

Use a CUDA-capable environment for a BF16 full-profile archive. For an FP32 run, including the bundled synthetic examples, use `--device cpu`; the runtime rejects BF16 execution on CPU. This uses the sealed configuration, data preparation and locked policies. The report names probability tolerances, deviations and model identities. Repeated timing can be requested with `--warmup 2 --repetitions 5`; those are measurements on the current hardware and execution environment. They do not replace the paper's GH200 observations. Use the [compute guide](COMPUTE_BUDGET.md) for cost definitions and the [getting-started guide](GETTING_STARTED.md) for fresh training.

## Attach our underlying result archives

We attach existing records through a separate declaration rather than rewriting the run or its history. The templates in [`paper/bindings/`](../paper/bindings) contain printed values and null run identities; they are templates, not evidence of historical checkpoint hashes. The archives themselves enter the repository through `make ingest-run` ([RUN_ARCHIVES.md](RUN_ARCHIVES.md)).

When an archive is the actual result-producing run, we declare that relationship explicitly:

```bash
python -B scripts/compare_with_manuscript.py "$RUN_ROOT" \
  --emit-binding "$REPORT_ROOT/manuscript_run_binding.json" \
  --relationship exact_result_producing_run \
  --manuscript-identifier "CROSS-MORSE: submitted 20-page main manuscript"

python -B scripts/check_result_binding.py "$RUN_ROOT" \
  "$REPORT_ROOT/manuscript_run_binding.json"
```

Use `refactor_with_prediction_equivalence` only when that is the documented relationship, or `post_result_revision` for a subsequent implementation/run; these are schema tokens, not conclusions inferred from matching numbers. A binding checks actual source/config/unit identities and result fields against our declaration. It cannot establish chronology, prior registration or unavailable historical provenance. A Roman campaign retains the ten sealed split identities and its aggregate report; the comparison command does not invent one single-run binding for them.

The [reviewer guide](REVIEWER_GUIDE.md) provides shorter routes for source inspection, numerical replay and checkpoint review. The [claims ledger](../paper/claims_ledger.md) distinguishes our reported measurements, implemented instruments, arithmetic checks and software tests.

## Regenerate tables and figures with Make

`make reproduce REPORT_ROOT=../cm_reports/examples` verifies the included seven-run inventories, recomputes decisions and scores from their saved observations, and generates 28 predictor rows plus SVG/PNG score panels. The committed reference views are in [`examples/generated/`](../examples/generated). The example data are synthetic, and their figure never overlays the paper's benchmark scores.

`make paper` builds the paper's tables, figures and evidence records from the official run archives registered in `runs/index.json` (`make tables` and `make figures` for one kind); `make example-paper` does the same from the seven synthetic examples and `make example-paper-check` verifies the committed example build. `make registry-views` regenerates the views of the transcribed registry (the detailed table rendering, the readable results overview, `results/paper_summary.csv`, the registry figure panels and the explanatory SVGs) and `make registry-check` verifies them; those registry targets do not retrain a predictor or independently measure the reported experiments, and they are never a source for a paper table.

For one completed run, `make run-figures RUN_ROOT=... REPORT_ROOT=...` verifies the sealed decision inputs and draws its four decision panels into a fresh destination. An explicit `--compare-manuscript` flag on `scripts/plot_run_figures.py` adds manuscript overlays for official-data runs only. Synthetic runs cannot request that overlay. Existing figures are preserved; choose another output directory to make another report.
