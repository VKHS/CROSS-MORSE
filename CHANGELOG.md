# Changes

The repository carries no release version. A state of the code is identified by its git commit
and by the runtime source fingerprint printed by `make fingerprint` (`9e0154c11c0e…` for the
runtime in this tree), which every run seals into its `run_state.json`. The package metadata
reads its distribution version from the sealed runtime (`cross_morse.__version__`) so that a
checkout and a run record always name the same implementation.

## Changes since the repository the reviewers examined (October 2026)

The runtime source (`cross_morse/*.py`, `run_cross_morse*.py`) is byte-identical to the
validated runtime, so every sealed run, replay and re-execution record remains valid. Everything
around it was rebuilt around what the two review reports asked for.

### Added

- `paper/build/` and `scripts/build_paper.py`: the run-derived columns of main Tables 1–2 and
  Supplementary Tables S4, S12, S12c, S13–S15, S23 and S41, the split/seed/environment manifest M1, and
  main Figs. 2c, 4d and 5a and Supplementary Figs. S9 and S14b, built from completed run
  directories; `make tables`, `make figures`, `make paper`, `make example-paper` and the CI
  check that the committed example build equals a fresh one. Tables S12 and S15 and figure F3
  always show the comparator q_C, the selected rule and the deployed rule within their route
  families; F3 draws the near-best band of the registered selection reading.
- `paper/build/selection.py`: Supplementary Table S12c from every run — the selection rule of main
  Eq. (19) under its three equivalence readings on the representative rows, the route selected,
  and the registered rule applied to every configuration and checked against the run's sealed
  selection.
- `records/` and `scripts/records.py`: the measurement-record contract for the experiments
  outside the eight-stage run (Supplementary Tables S6c, S15/S15b, S16/S16b, S19, S23, S23b, S25,
  S26b, S45, S46/S46b). A record set carries its raw logs, its code and a `summary.csv` whose every
  value is re-read from its raw file; `make records-index`, `make records-check` (part of
  `make verify`) and `make records-tables`; the availability statement reports the deposited sets.
- `scripts/archive_run.py`, `scripts/ingest_run.py`, `runs/`: pack a run or campaign into a
  verifiable archive, verify, deposit, ingest and index it; `make archive-run`,
  `make ingest-run`, `make runs-check`.
- `scripts/run_manifests.py`, `manifests/`: registered split/seed manifests of every
  configuration and per-run manifests (role membership hashes, seeds, bootstrap seeds,
  environment); `make manifests`, `make manifests-check`.
- `scripts/availability.py`, `AVAILABILITY.md`: the Data and Code Availability statement
  generated from the repository state and checked in CI (`make availability-check`). It names
  exactly the displays that `paper/build` produces and points to `paper/KNOWN_DIFFERENCES.md`
  for the rest; a test keeps the named list equal to the builders' outputs.
- `configs/arrhenius-site.env.example`, `docs/ARRHENIUS.md`,
  `environment/arrhenius/capture_environment.sh`: the Arrhenius site profile, walkthrough and
  GPU environment capture.
- Two diagrams in `docs/assets/` drawn by `scripts/draw_diagrams.py` and checked in CI:
  `evidence_chain` (from the source identity through Arrhenius execution, the run-archive
  contract and the paper build to the manuscript) and `decision_chain` (every decision
  statistic, the run file it is read from and the table cell it fills).
- `docs/RUN_ARCHIVES.md`, `docs/STATUS.md`, rewritten `README.md`, `docs/REVIEWER_GUIDE.md`
  and `paper/README.md`.
- Tests: `tests/test_paper_build.py` (including the deployment branch of the builders on an
  admitted fixture, since every bundled example ends with `G_admit = 0`),
  `tests/test_run_archives.py`, `tests/test_availability.py`.

### Fixed

- Fourteen sealed CSV artifacts of the example runs were normalised to LF by
  `.gitattributes`, which broke `make verify`, `make reproduce`, the strict audit and two
  tests on every fresh clone. The artifacts are byte-exact again and `examples/runs/**` and
  `runs/**` are excluded from text normalisation.
- The private-path scan flags project-disk paths only when they contain a concrete
  allocation identifier, so the public site profile can state the storage layout.
- CSV tables of synthetic builds carry the synthetic-examples banner in their first line, as
  every other generated file does.
- `paper/claims_ledger.md` cites `tests/test_policy_guardrails.py` (the file it named before
  did not exist).

### Changed

- The manuscript and the supplementary information are not distributed with the repository.
  `paper/sources.json` identifies them by SHA-256, byte size and page count; page references in
  the documentation are plain text; `scripts/render_paper_results.py --manuscript-pdf
  --supplementary-pdf` verifies a local copy byte for byte.

- `make paper` builds from run archives; the registry views moved to `make registry-views`
  and the registry checks to `make registry-check`.
- `release/release_scope.json` records the repository URL, the code DOI (once archived), the
  validated runtime identity and the run-archive contract; its flags are checked against
  `runs/index.json` and the tree.
- No release version anywhere: `pyproject.toml` reads the distribution version from the
  sealed runtime, `CITATION.cff` and `.zenodo.json` identify the software by repository and
  commit, and the availability statement names the runtime source identity.

## Reference implementation (October 2026)

Implementation, launchers, 21 configurations, seven synthetic example runs, manuscript
registry and validation record.
