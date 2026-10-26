# Contributing to CROSS-MORSE

[Repository home](README.md) · [Implementation contract](docs/IMPLEMENTATION_CONTRACT.md)

Contributions are welcome in method documentation, reproducibility, performance
engineering and clearly scoped scientific extensions. A useful contribution makes
the change and its evidence easy to assess.

## Start with the right discussion

| Contribution | Use |
|---|---|
| Reproduction question or unexpected result | **Reproduction question** issue form |
| Disagreement between a manuscript value, its registry transcription or a computed result | **Manuscript correspondence** issue form |
| A reproducible code defect | **Bug report** issue form |
| A methodological extension or documentation improvement | **Research or documentation discussion** issue form |
| A concrete implementation or documentation change | A pull request with the repository's template |

For a result question, include the source identity, configuration, command,
observed behavior and relevant result field. Include the specific manuscript
location when asking about a table or figure. A passing result-binding check is
more useful than a rounded number without its run record.

## Explain scientific changes precisely

Identify changes to loss definitions, actions and reset references, available
information, support, calibration, candidate registries, resampling, gates and
resource-accounting units. Add a regression case that would fail under the old
behavior when correcting a computational defect.

Numerical changes belong to a new experiment. Preserve the source and records
that produced existing results. Use the [reviewer guide](docs/REVIEWER_GUIDE.md)
for paired comparisons between experiments and the [results guide](docs/RESULTS.md)
to link manuscript results to their supporting artifacts.

## Keep the manuscript registry honest

`paper/registered_constants.json` and `paper/reported_results.json` transcribe the
identified manuscript. Cite the exact table, equation or figure when correcting
a transcription. Keep computed values in their run directories and compare them
with `scripts/compare_with_manuscript.py`. Numeric-review tests need NumPy from
`requirements-review.txt`; the constant checker and registry renderer themselves
use the standard library. After a registry edit:

```bash
python -B scripts/render_reported_results.py          # regenerate paper/REPORTED_RESULTS.md
python -B scripts/check_paper_constants.py
python -B -m unittest discover -s tests -p test_paper_registry.py -v
python -B -m unittest discover -s tests -p test_compare_with_manuscript.py -v
```

A change to a registered constant in code or `configs/*_full.json` must match its
cited manuscript definition or be identified as a distinct experiment. Explain
the distinction in the pull request and in `paper/KNOWN_DIFFERENCES.md`.

Runtime files (`run_cross_morse*.py`, `cross_morse/*.py`) determine the source
identity recorded by each run. `make fingerprint` compares those bytes with
`docs/VALIDATION.json`; `make source-id` prints the current identity. When runtime
changes, execute the appropriate checks and generate a validation record from
their actual reports. Do not update a recorded hash or outcome without performing
the corresponding validation. Preserve result-producing archives with their
original source, configuration and evidence.

## Validate according to the change

For computational changes, use an appropriate environment and run:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -B -m unittest discover -s tests -v
make smoke RUN_ROOT=../cm_runs/contribution_smoke
make replay RUN_ROOT=../cm_runs/contribution_smoke REPORT_ROOT=../cm_reports/contribution
make reexecute RUN_ROOT=../cm_runs/contribution_smoke REPORT_ROOT=../cm_reports/contribution
```

Use a new run directory. `make smoke DATASET=Peptides-func` selects another
synthetic task fixture; choose a distinct `RUN_ROOT` for it. The `reexecute` Make
target defaults to CPU and two threads for these FP32 examples. For a trusted
GPU run, select `DEVICE=cuda THREADS=16` and the corresponding environment; BF16
checkpoints require a compatible CUDA device. See the
[reviewer guide](docs/REVIEWER_GUIDE.md) for replay semantics.

For a documentation-only change, check links, command syntax, terminology and
figures, then verify that computational files and configs remain unchanged
(`make fingerprint` confirms the runtime identity).
Repeat runtime checks when the changed instructions affect execution. Record
actual checks and distinguish passes from skips. Preserve validation logs with
their recorded environment, source identity and counts.

Our GitHub workflow runs `make verify` on a fresh clone (release audit, source
fingerprint, registry checks, example inventories and replay, the example paper
build, manifest and availability checks, the NumPy-only tests), then installs CPU
PyTorch for the complete test suite and one synthetic end-to-end run that is
archived, ingested and turned into tables. A workflow definition is not evidence
of a hosted pass; inspect the actual Actions run. It does not train on official
benchmark data or validate Arrhenius GPU execution.

The block diagrams under `docs/assets/` are generated: edit `scripts/draw_diagrams.py` (or the registry files it reads), run `make diagrams`, and commit the regenerated SVG and PNG files together; `make diagrams-check` is part of `make registry-check` and of continuous integration. The optional numerical figure scripts require Matplotlib. After the runtime dependencies
are installed, `python -m pip install --no-deps -e .` installs the core package
for development; `python -m pip install -e '.[plots]'` also installs the plotting
extra. Execute the repository's `run_cross_morse.py` and `scripts/*.py` commands
from the checkout because configs, review scripts and the manuscript registry
are repository resources rather than installed console commands.

## Refresh and export the public source

After intentional repository changes, regenerate the inventory and audit it:

```bash
python -B scripts/create_release_manifest.py \
  --root . --output RELEASE_MANIFEST.json --profile public
python -B scripts/release_audit.py --root . --profile public --strict
python -B scripts/build_release.py \
  --source . --profile public --output ../cross_morse_public.zip
```

Choose an unused output filename, or use the exporter's explicit `--replace`
option after deciding to replace that archive. Keep the output outside the
source tree. The release inventory records file integrity; it is separate from
an experiment's source/configuration/stage identities.

Keep new experiment outputs and official benchmark caches outside the source
tree. The seven explicitly inventoried synthetic archives under `examples/runs/`
are intentional release assets; preserve their sealed bytes. Publish full
benchmark numerical bundles and checkpoint archives as separately identified
experiment assets, following the [artifact contract](docs/ARTIFACTS.md) and
[results guide](docs/RESULTS.md). Preserve
[LICENSE](LICENSE), [NOTICE](NOTICE) and existing contributor attribution.
