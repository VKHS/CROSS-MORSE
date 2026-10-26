# paper/: two things kept apart

[Repository home](../README.md) · [Run archives](../docs/RUN_ARCHIVES.md) · [Reviewer guide](../docs/REVIEWER_GUIDE.md)

## 1. `build/` and `generated/`: tables and figures computed from runs

`paper/build/` is the code that turns completed run directories into the paper's tables and
figures. It reads only sealed run records; it never reads the manuscript registry.

| Module / command | Output |
|---|---|
| `paper/build/evidence.py` | one evidence record per run (or per Roman-empire campaign): identities, data roles and seeds, entry contract, candidate family, selection, confirmation, admission, final test scores, cost ledger, stage durations |
| `paper/build/tables.py` | T1 (main Table 1), T2 (main Table 2), S4, S12, S12c, S13, S14, S15 (internal routes), S23, S41 and M1 (split/seed/environment manifest), each as `.tex`, `.md` and `.csv` |
| `paper/build/selection.py` | the selection rule of main Eq. (19) under the three equivalence readings of Supplementary Table S12c, and the registered rule over all configurations checked against the run's sealed selection |
| `paper/build/figures.py` | F1 effect against overhead (main Fig. 4d), F2 method map (main Fig. 5a), F3 selection geometry (main Fig. 2c), F4 depth dynamics (Supplementary Fig. S14b), F5 qualification time (Supplementary Table S41), F6 mechanism contrasts (Supplementary Fig. S9), each as `.png` and `.svg` |
| `make example-paper` | the build from the seven synthetic example runs, committed under `generated/examples/` and checked in CI (`make example-paper-check`) |
| `make paper` | the build from the official run archives registered in `runs/index.json`, written to `generated/official/` |

Every table file starts with the runs it was built from and the runtime source identity. A
quantity the released runner does not measure is printed as a dash and named in the table
note (today: the serving-time input-fallback probability, so the expected cost equals the
controller cost). Synthetic builds carry a banner in every file.

The experiments that run outside the eight-stage lifecycle (the budget-matched arms E3–E6, the
inserted-site attribution E13, the extended ensemble grid E14, the factorials, the negative
controls, the cross-backbone substitutions, the input gate, the nominal timing, the executed
sensor settings and the retrained external baselines) enter as measurement-record sets under
[`records/`](../records/README.md); `make records-tables` prints their tables into
`generated/official/records/` once the sets are deposited. `paper/KNOWN_DIFFERENCES.md` lists
every difference between the code and the manuscript and `docs/STATUS.md` how each is resolved.

## 2. The manuscript registry: transcribed values, comparison only

`reported_results.json`, `registered_constants.json`, `sources.json`, `REPORTED_RESULTS.md`,
`claims_ledger.md` and `KNOWN_DIFFERENCES.md` transcribe the manuscript. The manuscript and the
supplementary information are not distributed here; `sources.json` identifies them by SHA-256,
byte size and page count, and every page reference in the repository is to those documents.
They exist so that

* `make registry-check` can verify the manuscript's registered constants against the shipped
  configurations and its cross-table arithmetic (584 checks), and
* `make compare-runs` can place every run-derived value next to the manuscript value and
  report agreement at printed precision.

They are never a source for a table or a figure. `scripts/render_reported_results.py` and
`scripts/plot_registry_figures.py` draw views of the transcription (`make registry-views`)
for that comparison; the files they write carry the label "registry".

When the manuscript changes, the transcription is updated with the new document identities and
page references, and the comparison is rerun.
