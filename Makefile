PYTHON ?= python3
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# ---------------------------------------------------------------- locations
WORK_ROOT ?= ../cm_work
RUN_ROOT ?= $(WORK_ROOT)/runs/sample
REPORT_ROOT ?= $(WORK_ROOT)/reports
DATA_ROOT ?= $(WORK_ROOT)/data
DATASET ?= ogbn-arxiv
PROFILE ?= pilot
DEVICE ?= cpu
THREADS ?= 2
EXAMPLE_RUNS_ROOT ?= examples/runs
EXAMPLE_INVENTORY_ROOT ?= examples/inventory
EXAMPLE_GENERATED_ROOT ?= examples/generated
EXAMPLE_STEMS := arxiv products proteins roman_empire pascalvoc_sp coco_sp peptides_func
RUNS_INDEX ?= runs/index.json
PAPER_EXAMPLES_ROOT ?= paper/generated/examples
PAPER_OFFICIAL_ROOT ?= paper/generated/official
MANIFEST_ROOT ?= manifests
REGISTRY_FIGURE_ROOT ?= docs/assets
DIAGRAM_ROOT ?= docs/assets
VENV_DIR ?= ../cm_cpu_env
PRIVATE_ZIP ?= ../cross_morse_private.zip
PUBLIC_ZIP ?= ../cross_morse_public.zip
VALIDATION_RECORD ?= docs/VALIDATION.json
ARCHIVE ?=
OUTPUT ?=
DOI ?=

.PHONY: help verify check test test-no-torch test-release \
 tables figures paper paper-from-runs example-paper example-paper-check \
 manifests manifests-check availability availability-check \
 records-index records-check records-tables \
 archive-run archive-campaign ingest-run runs-check compare compare-runs \
 reproduce audit-artifacts example-views-check audit fingerprint source-id \
 registry-check registry-views diagrams diagrams-png diagrams-check \
 manifest verify-manifest release-private release-public release \
 prepare-data train-sample train-samples sample samples smoke replay reexecute review-bundle run-figures \
 env-cpu environment-check

help:
	@printf '%s\n' \
	  'Paper tables and figures FROM RUN DIRECTORIES (the reviewers'"'"' request):' \
	  '  make tables               rebuild every table from the official run archives in runs/ (RUNS_INDEX)' \
	  '  make figures              rebuild every numerical figure from the same archives' \
	  '  make paper                tables + figures + per-run evidence + record tables under paper/generated/official' \
	  '  make example-paper        the same build from the seven synthetic example runs (committed)' \
	  '  make compare-runs         compare run-derived values with the manuscript registry' \
	  'Run archives (checkpoints, predictions, trajectories, statistics):' \
	  '  make archive-run RUN_ROOT=/abs/suite OUTPUT=../deposits/x.tar.gz   verify the stage chain and pack' \
	  '  make archive-campaign RUN_ROOT=/abs/campaign OUTPUT=...             same for the Roman-empire campaign' \
	  '  make ingest-run ARCHIVE=../deposits/x.tar.gz DOI=10.5281/zenodo.N  verify, extract into runs/, index' \
	  '  make runs-check           re-verify every registered run archive' \
  'Measurement records outside the eight-stage run (records/, see records/README.md):' \
  '  make records-index        hash every deposited record set and trace every value to its raw file' \
  '  make records-check        re-verify the deposited sets against their manifests and raw files' \
  '  make records-tables       print their supplementary tables under paper/generated/official/records' \
	  'Manifests and the availability statement:' \
	  '  make manifests            registered design manifests (configs) and per-run split/seed/environment manifests' \
	  '  make availability         regenerate AVAILABILITY.md and the README block from the tree' \
	  'Checks (no dataset download, no training, no PyTorch needed):' \
	  '  make verify / check       everything CI runs: audit, fingerprint, registry, examples, example paper build, manifests, availability' \
	  '  make reproduce            replay the bundled example predictions; regenerate CSV/Markdown/JSON/SVG under REPORT_ROOT' \
	  '  make test                 complete unit-test suite (needs PyTorch);  make test-no-torch for the NumPy-only subset' \
	  'Manuscript registry (transcribed values, comparison only):' \
	  '  make registry-check       arithmetic and provenance checks of paper/reported_results.json' \
	  '  make registry-views       redraw the registry figures and explanatory diagrams' \
	  'Execution:' \
	  '  make prepare-data DATASET=... PROFILE=pilot|full   prepare official data (may download, never trains)' \
	  '  make train-sample / train-samples                 synthetic end-to-end runs on CPU (needs PyTorch)' \
	  '  make env-cpu / environment-check                  create or check the recorded CPU environment' \
	  'Packaging: manifest, verify-manifest, release-public / release-private, source-id.' \
	  'Variables: PYTHON, RUNS_INDEX, RUN_ROOT, REPORT_ROOT, DATA_ROOT, DATASET, PROFILE, THREADS; quote paths with spaces.'

# ---------------------------------------------------------------- paper from runs
tables:
	"$(PYTHON)" -B scripts/build_paper.py tables --index "$(RUNS_INDEX)" --label official --output-dir "$(PAPER_OFFICIAL_ROOT)"

figures:
	"$(PYTHON)" -B scripts/build_paper.py figures --index "$(RUNS_INDEX)" --label official --output-dir "$(PAPER_OFFICIAL_ROOT)"

paper-from-runs:
	"$(PYTHON)" -B scripts/build_paper.py all --index "$(RUNS_INDEX)" --label official --output-dir "$(PAPER_OFFICIAL_ROOT)"

paper: paper-from-runs records-tables

example-paper:
	"$(PYTHON)" -B scripts/build_paper.py all --runs-root "$(EXAMPLE_RUNS_ROOT)" --label examples --output-dir "$(PAPER_EXAMPLES_ROOT)"

example-paper-check:
	"$(PYTHON)" -B scripts/build_paper.py all --runs-root "$(EXAMPLE_RUNS_ROOT)" --label examples --output-dir "$(PAPER_EXAMPLES_ROOT)" --check

compare-runs:
	@test -f "$(RUNS_INDEX)" || { printf '%s\n' 'no runs/index.json: ingest run archives first (make ingest-run)' >&2; exit 2; }
	@for stem in $$("$(PYTHON)" -c 'import json,sys; [print(e["stem"]) for e in json.load(open(sys.argv[1]))["runs"] if e.get("verified")]' "$(RUNS_INDEX)"); do \
	  if [ -d "runs/$$stem/split_00" ]; then \
	    "$(PYTHON)" -B scripts/compare_with_manuscript.py "runs/$$stem" --roman-campaign --output "$(REPORT_ROOT)/compare/$$stem.json"; \
	  else \
	    "$(PYTHON)" -B scripts/compare_with_manuscript.py "runs/$$stem" --output "$(REPORT_ROOT)/compare/$$stem.json"; \
	  fi; \
	done

compare:
	"$(PYTHON)" -B scripts/compare_with_manuscript.py "$(RUN_ROOT)" --output "$(REPORT_ROOT)/manuscript_comparison.json"

# ---------------------------------------------------------------- run archives
archive-run:
	@test -n "$(RUN_ROOT)" -a -n "$(OUTPUT)" || { printf '%s\n' 'usage: make archive-run RUN_ROOT=/abs/suite OUTPUT=/abs/archive.tar.gz' >&2; exit 2; }
	"$(PYTHON)" -B scripts/archive_run.py "$(RUN_ROOT)" --output "$(OUTPUT)"

archive-campaign:
	@test -n "$(RUN_ROOT)" -a -n "$(OUTPUT)" || { printf '%s\n' 'usage: make archive-campaign RUN_ROOT=/abs/campaign OUTPUT=/abs/archive.tar.gz' >&2; exit 2; }
	"$(PYTHON)" -B scripts/archive_run.py "$(RUN_ROOT)" --campaign --output "$(OUTPUT)"

ingest-run:
	@test -n "$(ARCHIVE)" || { printf '%s\n' 'usage: make ingest-run ARCHIVE=/abs/archive.tar.gz [DOI=10.5281/zenodo.N]' >&2; exit 2; }
	"$(PYTHON)" -B scripts/ingest_run.py "$(ARCHIVE)" $(if $(DOI),--doi "$(DOI)",)
	"$(PYTHON)" -B scripts/run_manifests.py runs --index "$(RUNS_INDEX)" --label official --with-indices
	"$(PYTHON)" -B scripts/availability.py

records-index:
	"$(PYTHON)" -B scripts/records.py index
	"$(PYTHON)" -B scripts/availability.py

records-check:
	"$(PYTHON)" -B scripts/records.py check

records-tables:
	"$(PYTHON)" -B scripts/records.py tables --output-dir "$(PAPER_OFFICIAL_ROOT)/records"

runs-check:
	"$(PYTHON)" -B scripts/ingest_run.py --check

# ---------------------------------------------------------------- manifests, availability
manifests:
	"$(PYTHON)" -B scripts/run_manifests.py registered
	"$(PYTHON)" -B scripts/run_manifests.py runs --runs-root "$(EXAMPLE_RUNS_ROOT)" --label examples
	@if [ -f "$(RUNS_INDEX)" ]; then "$(PYTHON)" -B scripts/run_manifests.py runs --index "$(RUNS_INDEX)" --label official --with-indices; fi

manifests-check:
	"$(PYTHON)" -B scripts/run_manifests.py registered --check
	"$(PYTHON)" -B scripts/run_manifests.py runs --runs-root "$(EXAMPLE_RUNS_ROOT)" --label examples --check
	@if [ -f "$(RUNS_INDEX)" ]; then "$(PYTHON)" -B scripts/run_manifests.py runs --index "$(RUNS_INDEX)" --label official --check; fi

availability:
	"$(PYTHON)" -B scripts/availability.py

availability-check:
	"$(PYTHON)" -B scripts/availability.py --check

# ---------------------------------------------------------------- bundled examples
audit-artifacts:
	"$(PYTHON)" -B scripts/artifact_inventory.py --check-static
	@for stem in $(EXAMPLE_STEMS); do \
	  "$(PYTHON)" -B scripts/artifact_inventory.py "$(EXAMPLE_RUNS_ROOT)/$$stem" \
	    --output "$(EXAMPLE_INVENTORY_ROOT)/$$stem.json" --check; \
	done

# Recompute decisions from observations. No PyTorch/checkpoint loading or fitting.
reproduce: audit-artifacts
	"$(PYTHON)" -B scripts/reproduce_examples.py --runs-root "$(EXAMPLE_RUNS_ROOT)" \
	  --output-dir "$(REPORT_ROOT)" --figures

example-views-check:
	"$(PYTHON)" -B scripts/reproduce_examples.py --runs-root "$(EXAMPLE_RUNS_ROOT)" \
	  --output-dir "$(EXAMPLE_GENERATED_ROOT)" --check

# ---------------------------------------------------------------- manuscript registry (comparison only)
registry-check:
	"$(PYTHON)" -B scripts/check_paper_constants.py
	"$(PYTHON)" -B scripts/render_reported_results.py --check
	"$(PYTHON)" -B scripts/render_paper_results.py --check
	"$(PYTHON)" -B scripts/draw_diagrams.py --check --png

registry-views:
	"$(PYTHON)" -B scripts/render_reported_results.py
	"$(PYTHON)" -B scripts/render_paper_results.py
	"$(PYTHON)" -B scripts/plot_paper_summary.py --output-dir "$(REGISTRY_FIGURE_ROOT)"
	"$(PYTHON)" -B scripts/plot_registry_figures.py --output-dir "$(REGISTRY_FIGURE_ROOT)/registry"
	"$(PYTHON)" -B scripts/draw_diagrams.py --output-dir "$(DIAGRAM_ROOT)"

diagrams:
	"$(PYTHON)" -B scripts/draw_diagrams.py --output-dir "$(DIAGRAM_ROOT)"

diagrams-png:
	"$(PYTHON)" -B scripts/draw_diagrams.py --output-dir "$(DIAGRAM_ROOT)" --png

diagrams-check:
	"$(PYTHON)" -B scripts/draw_diagrams.py --check --png

# ---------------------------------------------------------------- whole-tree checks
verify: registry-check fingerprint audit-artifacts example-views-check example-paper-check manifests-check records-check availability-check runs-check-if-present audit
check: verify

.PHONY: runs-check-if-present
runs-check-if-present:
	@if [ -f "$(RUNS_INDEX)" ]; then "$(PYTHON)" -B scripts/ingest_run.py --check; else printf '%s\n' 'runs/index.json absent: no official run archives registered yet'; fi

audit:
	"$(PYTHON)" -B scripts/release_audit.py --root . --strict

fingerprint:
	"$(PYTHON)" -B scripts/source_fingerprint.py --validation-record "$(VALIDATION_RECORD)"

source-id:
	"$(PYTHON)" -B scripts/source_fingerprint.py

test:
	"$(PYTHON)" -B -m unittest discover -s tests -v

test-no-torch:
	"$(PYTHON)" -B -m unittest discover -s tests -p 'test_paper_registry.py' -v
	"$(PYTHON)" -B -m unittest discover -s tests -p 'test_compare_with_manuscript.py' -v
	"$(PYTHON)" -B -m unittest discover -s tests -p 'test_source_fingerprint_cli.py' -v
	"$(PYTHON)" -B -m unittest discover -s tests -p 'test_draw_diagrams.py' -v
	"$(PYTHON)" -B -m unittest discover -s tests -p 'test_make_workflows.py' -v
	"$(PYTHON)" -B -m unittest discover -s tests -p 'test_environment_spec.py' -v
	"$(PYTHON)" -B -m unittest discover -s tests -p 'test_artifact_inventory.py' -v
	"$(PYTHON)" -B -m unittest discover -s tests -p 'test_reproduce_examples.py' -v
	"$(PYTHON)" -B -m unittest discover -s tests -p 'test_release_*.py' -v
	"$(PYTHON)" -B -m unittest discover -s tests -p 'test_paper_build.py' -v
	"$(PYTHON)" -B -m unittest discover -s tests -p 'test_run_archives.py' -v
	"$(PYTHON)" -B -m unittest discover -s tests -p 'test_availability.py' -v
	"$(PYTHON)" -B -m unittest discover -s tests -p 'test_records.py' -v

test-release:
	"$(PYTHON)" -B -m unittest discover -s tests -p 'test_release_*.py' -v
	"$(PYTHON)" -B -m unittest discover -s tests -p test_make_workflows.py -v
	"$(PYTHON)" -B -m unittest discover -s tests -p test_environment_spec.py -v

manifest:
	"$(PYTHON)" -B scripts/create_release_manifest.py --root . --output RELEASE_MANIFEST.json

verify-manifest:
	"$(PYTHON)" -B scripts/create_release_manifest.py --root . --verify RELEASE_MANIFEST.json

release-private:
	"$(PYTHON)" -B scripts/build_release.py --source . --profile private --output "$(PRIVATE_ZIP)"

release-public:
	"$(PYTHON)" -B scripts/build_release.py --source . --profile public --output "$(PUBLIC_ZIP)"

release: release-public

# ---------------------------------------------------------------- execution
prepare-data:
	@case "$(PROFILE)" in pilot|full) ;; *) printf '%s\n' 'prepare-data requires PROFILE=pilot or PROFILE=full; synthetic smoke is not official data.' >&2; exit 2 ;; esac
	"$(PYTHON)" -B run_cross_morse.py --dataset "$(DATASET)" --profile "$(PROFILE)" \
	  --device "$(DEVICE)" --threads "$(THREADS)" --stage prepare --data-root "$(DATA_ROOT)" --output-dir "$(RUN_ROOT)"

train-sample:
	"$(PYTHON)" -B run_cross_morse.py --dataset "$(DATASET)" --profile smoke \
	  --device cpu --threads "$(THREADS)" --stage all --data-root "$(DATA_ROOT)" --output-dir "$(RUN_ROOT)"

train-samples:
	@for stem in $(EXAMPLE_STEMS); do \
	  "$(PYTHON)" -B "run_cross_morse_$$stem.py" --profile smoke --device cpu --threads "$(THREADS)" \
	    --stage all --data-root "$(DATA_ROOT)" --output-dir "$(WORK_ROOT)/runs/$$stem"; \
	done

sample smoke: train-sample
samples: train-samples

replay:
	"$(PYTHON)" -B scripts/reproduce_results.py "$(RUN_ROOT)" --output "$(REPORT_ROOT)/replay.json"

reexecute:
	"$(PYTHON)" -B scripts/evaluate_checkpoints.py "$(RUN_ROOT)" --trust-local-checkpoints \
	  --device "$(DEVICE)" --threads "$(THREADS)" --output "$(REPORT_ROOT)/checkpoints.json"

review-bundle:
	"$(PYTHON)" -B scripts/build_review_bundle.py "$(RUN_ROOT)" --output "$(REPORT_ROOT)/review_evidence.zip"

run-figures:
	"$(PYTHON)" -B scripts/plot_run_figures.py "$(RUN_ROOT)" --output-dir "$(REPORT_ROOT)/run-figures"

env-cpu:
	bash environment/create_cpu_env.sh --python "$(PYTHON)" --venv "$(VENV_DIR)"

environment-check:
	"$(PYTHON)" -B environment/check_environment.py
