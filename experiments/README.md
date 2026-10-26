# Experiment profiles and seeds

[`profiles.json`](profiles.json) is generated from the **21 committed JSON
configurations**: seven benchmarks, each with `smoke`, `pilot`, and `full`
profiles. Each entry includes the configuration path, its file and canonical
JSON SHA-256 identities, task dimensions, master/split seed, training seeds,
model/training settings, offline trajectory count, and selection/admission
settings. The complete controller and sensor settings remain in the referenced
configuration file.

## Profiles and evidence

| Profile | Data | Purpose | What completion establishes |
|---|---|---|---|
| `smoke` | Generated fixtures; `dataset=synthetic`, `scientific_run=false` | Exercise the complete lifecycle, task mathematics and artifact paths | Executable software evidence on the saved fixture |
| `pilot` | Official dataset and official split | A smaller compute budget for adapter/runtime investigation | A new official-data run with its own exact provenance |
| `full` | Official dataset and official split | The registered benchmark-oriented configuration | A new benchmark execution; paper agreement requires separate comparison and linkage |

Our bundled complete examples use the seven smoke configurations. Their output
dimension is four, including the multilabel fixtures. The real proteins and
Peptides adapters retain their official 112 and ten targets. Each example
contains two training seeds and twelve exploration trajectories. These small
settings do not replace the full configurations.

The full profiles configure master/split seed `314159` and training seeds
`[1, 2, ..., 12]`. These are the executable configuration choices, extracted per
profile into [`manifests/registered/`](../manifests/registered/) by `make manifests`.
The seeds a completed run actually used are read from its sealed records into
`manifests/<label>/`; no claim is made that the configured list by itself
identifies the seeds of any historical run.
The seed ensemble is a fixed prefix of separately trained members; a bootstrap
over graph/core units does not estimate uncertainty over retraining seeds.

## Inspect before executing

These commands use the standard library and print the effective configuration:

```bash
python -B run_cross_morse.py --dataset ogbn-arxiv --profile full --print-config
python -B run_cross_morse.py --dataset Roman-empire --profile full \
  --split-index 7 --print-config
python -B scripts/artifact_inventory.py --check-static
```

Changing `--split-index` changes the effective configuration and therefore the
run identity. It does not alter the committed default profile. All other
dataset entry points use the same configuration resolver and lifecycle.

## Execute a new software example

Use a new output directory; committed examples are review artifacts:

```bash
python -B run_cross_morse.py --dataset ogbn-arxiv --profile smoke \
  --data-root /tmp/cross_morse_fixture_data --output-dir /tmp/cross_morse_smoke \
  --device cpu --threads 2 --stage all

python -B scripts/artifact_inventory.py /tmp/cross_morse_smoke \
  --output /tmp/cross_morse_smoke_inventory.json
python -B scripts/reproduce_results.py /tmp/cross_morse_smoke \
  --output /tmp/cross_morse_smoke_replay.json
```

The CPU environment specification is in [`environment/`](../environment/).
Official full executions use the same Python entry point with `--profile full`
and the intended device/data cache. Resource requirements depend on the real
graph and model, and smoke timings are not benchmark compute estimates. Use
[`docs/COMPUTE_BUDGET.md`](../docs/COMPUTE_BUDGET.md) and the cluster launchers for
planning.

## Roman-empire: ten independent run directories

Roman-empire provides ten official mask selections. A single `--split-index 0`
run covers one selection, not the ten-split aggregate. The launcher campaign
support records indices `0` through `9`; alternatively, explicit Python runs
can be issued as follows:

```bash
for split in 0 1 2 3 4 5 6 7 8 9; do
  python -B run_cross_morse.py --dataset Roman-empire --profile full \
    --split-index "$split" --data-root /path/to/dataset-cache \
    --output-dir "/path/to/roman_full/split_${split}" --device cuda --stage all
done
```

Those explicit directories are separate runs. The campaign summarizer expects
the launcher's `campaign.json` and `split_00`–`split_09` layout; use the campaign
launcher for that specific aggregation command. Both routes preserve the
selected official mask index in the effective run configuration. Do not label
one mask's score or confidence interval as the ten-mask paper aggregate.

## Attach completed benchmark artifacts

Preserve the entire completed run and its runtime source snapshot. Pack it with
`make archive-run`, deposit it, and ingest it with `make ingest-run`
([`docs/RUN_ARCHIVES.md`](../docs/RUN_ARCHIVES.md)); or generate an inventory
outside the run, replay the statistical decisions, and reexecute the trusted
checkpoint predictions by hand. A result binding can then refer to the existing
source/config/unit/prediction identities without editing the sealed history. See
[`docs/ARTIFACTS.md`](../docs/ARTIFACTS.md) and [`docs/RESULTS.md`](../docs/RESULTS.md).

The static profile manifest describes what is configured. The stage manifests,
saved numerical observations and executable checks establish what a particular
run actually contains.
