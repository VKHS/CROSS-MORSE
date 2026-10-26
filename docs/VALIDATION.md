# Software validation

Validation date: **25 October 2026**. The complete suite passed **345 of 346 discovered tests**, with **one expected skip**, **zero failures** and **zero errors**. The skipped check requires a private cluster profile, which is intentionally absent from the public package. All seven task-specific synthetic experiments completed the eight-stage lifecycle, manifest verification, numerical replay, checkpoint reexecution and completed-run resume. Actual official Roman-empire data were also prepared and their ten split masks checked. The complete unit-test outcome and execution environment are recorded in [VALIDATION.json](VALIDATION.json) and the [test log](validation_evidence/unit_tests.log).

These checks establish software behavior and evidence consistency. Full official benchmark training, Arrhenius GPU execution, hosted GitHub Actions execution and agreement with the paper's reported numerical results have not been assessed by this validation.

## Executed task workflows

| Task configuration | Metric path | Eight stages and manifests | Numeric replay | Checkpoint replay | Resume |
|---|---|---|---|---|---|
| [ogbn-arxiv](validation_evidence/arxiv/validation.json) | Accuracy | PASS | PASS | PASS | PASS |
| [ogbn-proteins](validation_evidence/proteins/validation.json) | ROC-AUC | PASS | PASS | PASS | PASS |
| [ogbn-products](validation_evidence/products/validation.json) | Accuracy | PASS | PASS | PASS | PASS |
| [Roman-empire](validation_evidence/roman_empire/validation.json) | Accuracy | PASS | PASS | PASS | PASS |
| [PascalVOC-SP](validation_evidence/pascalvoc_sp/validation.json) | Macro F1 | PASS | PASS | PASS | PASS |
| [COCO-SP](validation_evidence/coco_sp/validation.json) | Macro F1 | PASS | PASS | PASS | PASS |
| [Peptides-func](validation_evidence/peptides_func/validation.json) | Average precision | PASS | PASS | PASS | PASS |

The stages are `prepare`, `train`, `explore`, `fit`, `select`, `confirm`, `admit` and `evaluate`. Each single-graph fixture contains 160 nodes. Each independent graph fixture contains 80 graphs with eight nodes per graph. All fixtures use four generated classes or labels, two training seeds, three training epochs, 12 exploration trajectories and 40 registered candidates. Control depth is six for the single-graph fixtures and eight for the graph collections. Pilot and full profiles retain the official task dimensions.

Checkpoint reexecution checked `q0`, `qN`, `qC` and `qD` for every task: **28 predictor comparisons**, with maximum absolute probability difference **0.0**. Resume preserved every sealed output hash. All seven synthetic admission decisions retained their selected open-loop comparator; the deployed and comparator prediction files were byte-identical. This exercises the exact fallback behavior. It does not demonstrate a feedback performance gain.

The 56 lifecycle manifests and the official preparation manifest also passed the bundled JSON schema with date-time format validation: **57 manifests**. The runtime manifest verifier separately checked file identities and stage ancestry. Each task's evidence directory retains the actual commands and exit codes, lifecycle log, manifest check, numerical replay, checkpoint replay and resume record.

A [portable numerical evidence bundle](validation_evidence/portable_review_export.json) was exported from the Peptides-func synthetic run. The exporter verified the copied evidence and replayed it with the same result, without changing the source run or including trained checkpoints. This checks the graph-level review path; the bundle is a software example.

## Included artifacts and Makefile workflows

The seven completed runs are included under [`examples/runs/`](../examples/runs), with **2,170 inventoried files**, **2,051 stage-sealed artifacts** and **190,197,703 bytes**. They contain 350 `.pt` files, including controller/optimizer/resume states; 28 of those are best selected predictor checkpoints. The archives also contain 588 `.npy` files, 189 `.npz` files and 84 exploration trajectories with 576 transition records. The [per-run inventories](../examples/inventory) verify concrete splits, seeds, checkpoint bytes, predictions and stage ancestry without loading checkpoint objects.

Actual `make reproduce` execution verified all seven inventories, recomputed the decisions and 28 final predictor scores, and produced CSV, Markdown, replay JSON, SVG, PNG and a checksum manifest. The [committed numerical views](../examples/generated) pass the same read-only recomputation check. Idempotent regeneration, protection of user-edited reports and preservation of unrelated files are covered by the test suite.

Execution of the registry-view targets (then named `make paper diagrams-check`, now `make registry-views` and `make diagrams-check`) regenerated the two paper table views and summary CSV byte-for-byte, produced eight paper-value figure files and five explanatory SVG diagrams, and verified the five committed PNGs against their SVG source identities. Output paths containing spaces were exercised. All five explanatory diagrams were rendered and visually inspected. These outputs visualize the transcribed registry and the method; they do not create new benchmark measurements.

The actual [Makefile replay log](validation_evidence/make_reproduce.log), [paper-generation log](validation_evidence/make_paper.log), [environment check](validation_evidence/environment_check.log) and [dependency check](validation_evidence/pip_check.log) accompany this record. The portable numerical review export is deliberately smaller: it omits the complete example directories and all checkpoint binaries, and declares that reduced source scope.

## Official-data preparation

The Roman-empire preparation stage read the official raw NPZ and produced a sealed artifact with **22,662 nodes, 300 input features, 18 classes and 65,854 canonical directed edge entries**. Split zero contains 11,331 training, 5,665 validation and 5,666 test nodes. All ten official masks have disjoint and exhaustive roles. This check stopped after preparation; it did not train a predictor or execute the ten-split experiment campaign.

The [preparation record](validation_evidence/roman_official_preparation.json) contains the source URL, raw-file SHA-256, prepared-data identity, partition sizes and command. Its [stage manifest](validation_evidence/roman_official_prepare_manifest.json) records the sealed artifacts. Preparation of the other six complete official datasets was not executed in this environment.

## Environment and commands

The recorded environment is Python `3.12.14`, PyTorch `2.10.0+cpu`, NumPy `2.3.5`, Linux x86_64, CPU and FP32. Synthetic experiments use two PyTorch threads. Unit tests use two OpenMP/OpenBLAS threads. OGB and PyMetis were installed, so their dependency-specific checks ran. The complete observed environment records **53 installed distributions**, including SciPy `1.17.0` and Matplotlib `3.10.8`. Its [exact version lock](../environment/requirements-cpu.lock), [machine-readable capture](../environment/observed_environment.json), installer and version checker accompany the package. The installed versions match the lock, and `pip check` passed. This is a CPU software-validation environment; the lock does not claim GPU-environment or platform-wheel identity.

From the repository root, use a fresh run directory and keep reports outside the source tree:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -B -m unittest discover -s tests -v

python -B run_cross_morse_peptides_func.py --profile smoke --device cpu --threads 2 --data-root ../cm_data --output-dir ../cm_runs/peptides_smoke

python -B scripts/validate_run_manifest.py ../cm_runs/peptides_smoke --require-complete --json

python -B scripts/reproduce_results.py ../cm_runs/peptides_smoke --output ../cm_reports/peptides_numeric_replay.json

python -B scripts/evaluate_checkpoints.py ../cm_runs/peptides_smoke --trust-local-checkpoints --device cpu --threads 2 --output ../cm_reports/peptides_checkpoint_replay.json

python -B run_cross_morse_peptides_func.py --profile smoke --device cpu --threads 2 --data-root ../cm_data --output-dir ../cm_runs/peptides_smoke --resume
```

The runtime source identity shared by all seven runs and the official preparation is:

```text
9e0154c11c0ebc67d533a84b6d6a80c4e77a8d76e6d2cc165bcff5897980b108
```

It covers `run_cross_morse*.py` and `cross_morse/*.py`. Verify the current source against the record using only the standard library:

```bash
python -B scripts/source_fingerprint.py --validation-record docs/VALIDATION.json
```

Documentation, scripts, tests, configuration and public metadata have their bytes and executable flags covered by the release inventory and checksum file.

## Behaviors under test

The suite checks task-specific losses and metrics, official split handling, missing-label masks, graph-level pooling and graph isolation. Whole-graph metric observations are recomputed for macro F1, ROC-AUC and AP during resampling. Tests cover equal graph weighting and column-shaped multiclass labels.

Transport and sensor tests distinguish shared transport, native mean and inserted identity sites. They verify the site-specific nominal response, terminal reachability coordinates, native/inserted action accounting, identity bypass and recovery behavior. Controller tests cover complete-trajectory cross-fitting, support checks, held-out calibration, observer recovery, selection, confirmation and the global admission decision.

Review tests execute a small lifecycle, replay saved numerical observations, reexecute the four frozen predictors, export portable evidence and detect tampering. Manuscript-comparison tests reject incomplete stages, synthetic benchmark claims, malformed records, nonfinite values and unsafe report destinations; Roman-empire comparisons distinguish a single split from the ten-split aggregate. Package tests cover archive integrity and executable flags.

The paper registry independently passes **584 arithmetic and configuration checks**. The byte identities of both submitted PDFs and all **32 numerical source groups** are verified, and the generated result pages and CSV match their canonical registry. These are transcription, provenance and arithmetic checks. Comparison-unit fixtures deliberately use reported values to test the reporting logic; they are not benchmark runs. No reported score replaces a model's computed predictions.

## Interpretation

Small synthetic execution does not establish official benchmark quality, large-graph memory use, full-profile runtime or GPU precision behavior. Products' full-graph hidden states in particular require resource measurement on the intended hardware. The included CI workflow specifies hosted checks; this record describes the local checks actually executed.

Numerical replay verifies submitted observations and arithmetic. Checkpoint reexecution verifies recorded predictor outputs. Associating either with a manuscript table requires the authors' result-producing archives and completed bindings described in [Results and evidence](RESULTS.md). The reported measurements remain available with their provenance in [Paper results](PAPER_RESULTS.md).

## Fresh-clone verification of the rebuilt package (26 October 2026)

After the repository was rebuilt around the run-archive contract, the tree was committed,
cloned fresh and verified in the clone, without PyTorch, on Linux x86_64 with CPython 3.13.17,
NumPy 2.5.3 and Matplotlib 3.11.2:

| Check | Outcome |
|---|---|
| `git ls-files --eol examples/runs/arxiv/final/results.csv` | `i/crlf w/crlf attr/-text`: sealed artifacts are byte-exact after checkout |
| `make verify` (registry arithmetic, fingerprint, seven artifact inventories, example replay check, example paper-build check, manifest checks, measurement-record check, availability check, strict audit) | PASS |
| `make reproduce` | 7 complete runs, 28 predictor rows recomputed, PASS |
| `make example-paper` into a temporary directory | 11 tables (including S12c) and 6 figures built from the seven example runs; every run's registered selection reproduced from its sealed record |
| `scripts/archive_run.py` → `scripts/ingest_run.py` → `scripts/build_paper.py --index` on the Peptides-func example | archive packed (311 files), verified, ingested, tables built from the ingested run |
| `make test-no-torch` (13 modules) | 177 tests, 0 failures, 0 errors, 2 expected skips |
| `tests/test_paper_build.py::AdmittedRunTests`, `SelectionReadingTests` | the deployment branch of evidence, tables and figures runs on an admitted fixture; the three readings of Table S12c follow their registered margins |
| `tests/test_records.py` | measurement-record sets: manifests, value-by-value trace to raw logs at printed precision, printed tables, ensemble-law fit |

The runtime source identity was unchanged by the rebuild (`9e0154c11c0ebc67d533a84b6d6a80c4e77a8d76e6d2cc165bcff5897980b108`),
so the 25 October results above still describe this runtime. The PyTorch-dependent parts of
the suite and the end-to-end synthetic training were not re-executed on 26 October; the hosted
workflow (`.github/workflows/quality.yml`) runs them on every push, and its latest result is on
the repository's Actions tab.
