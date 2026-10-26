# Split, seed and environment manifests

Two kinds of manifest, both written by `scripts/run_manifests.py` (`make manifests`) and
checked in CI (`make manifests-check`):

| Directory | Content | Source |
|---|---|---|
| `registered/<stem>_<profile>.json` | the registered design of every pilot and full configuration: partition seed, training seeds, data-role rule, folds, bootstrap replicates, margins, action registry settings | `configs/*.json` |
| `examples/<stem>.json` | what each synthetic example run actually used: role membership hashes and sizes, official split hashes, seeds, ensemble members, bootstrap seeds of every decision stage, locked policy hashes, recorded environment and package versions | `examples/runs/<stem>/` |
| `official/<stem>.json` (+ `<stem>_splits.npz`, `<stem>_official_splits.npz`, …) | the same for the ingested official-data run archives, with the membership index arrays copied next to the manifest | `runs/<stem>/` after `make ingest-run` |

The data-role rule is the one the implementation applies: a seeded permutation of the
official training units cut at 60 % / 80 % / 90 % into D_fit / D_explore / D_select with the
remainder in D_confirm; the official validation split is D_admit and the official test split
is D_test; graph collections are split by whole graphs; Roman-empire repeats the procedure
per official split. The membership arrays themselves are `prepared/splits.npz` and
`prepared/official_splits.npz` in every run, and their SHA-256 values are the `unit_sha256`
sealed into the run's stage manifests.
