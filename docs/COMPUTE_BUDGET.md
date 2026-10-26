# Compute budget

Plan allocations from the resolved configuration and measurements on the target
hardware. The manuscript's timings below are **our reported measurements**, transcribed
from Supplementary Tables S5, S15, S23, S23b and S41 into
[`paper/reported_results.json`](../paper/reported_results.json) for comparison.
Their stated hardware is one NVIDIA GH200 96 GB per run. They are not measurements
of this checkout and do not establish that a full profile fits a particular
allocation. The local checks actually executed are recorded in
[VALIDATION.md](VALIDATION.md).

Sections 1–3 preserve the manuscript's accounting units. Section 4 describes the
executable profiles, and section 5 gives a practical measurement procedure.
[`check_paper_constants.py`](../scripts/check_paper_constants.py) checks registered
constants and ledger arithmetic; it does not time a training or deployment run.

## 1. Reported plant training (Table S5)

| Dataset | Reported backbone | Depth × width | Precision | Optimizer and schedule | Epochs / patience | Reported time per seed | 12 × reported per-seed time |
|---|---|---:|---|---|---:|---:|---:|
| ogbn-arxiv | MLP4-512 shared-transport plant | 256 × 512 | bf16 plant / fp32 statistics | AdamW 3e-4, wd 1e-4, cosine | 500 / 50 | 1.8 h | 21.6 h |
| ogbn-proteins | MLP4-512, edge-weighted aggregation | 128 × 512 | bf16 / fp32 | AdamW 2e-4, wd 1e-5, cosine | 400 / 40 | 2.4 h | 28.8 h |
| ogbn-products | scalable MLP3-512, batch 65,536 | 64 × 512 | bf16 / fp32 evaluator | AdamW 5e-4, wd 1e-5, cosine | 300 / 30 | 4.8 h | 57.6 h |
| Roman-empire | heterophily encoder4-384 | 256 × 384 | fp32 | AdamW 1e-3, wd 5e-4 | 800 / 100 | 0.45 h | 5.4 h (per split) |
| PascalVOC-SP | GraphGPS-style 6 × 128, batch 32 | 128 × 128 | bf16 / fp32 | AdamW 5e-4, wd 1e-5 | 300 / 30 | 1.1 h | 13.2 h |
| COCO-SP | GraphGPS-style 6 × 128, batch 32 | 128 × 128 | bf16 / fp32 | AdamW 5e-4, wd 1e-5 | 300 / 30 | 1.3 h | 15.6 h |
| Peptides-func | GINE-style 8 × 128 (linear aggregation), batch 128 | 64 × 128 | bf16 / fp32 | AdamW 1e-3, wd 1e-5 | 250 / 30 | 0.9 h | 10.8 h |

The last column is arithmetic, not a separately reported ensemble measurement.
It multiplies the per-seed entry by 12 and does not include this implementation's
separately trained `q0`, controller development or final evaluation. Table S15
reports plant GPU-hours of 10.88, 11.39, 12.92, 7.82, 9.01, 10.29 and 9.61 in the
dataset order above under its shared 48-trial HPO ceiling. These are different
accounting units from the Table S5 per-checkpoint times. Table S41 and the
manuscript's accounting note R15 distinguish per-checkpoint, per-ensemble and
search-level quantities; they should not be added as if they were one run ledger.

The products row describes the manuscript's scalable plant and batch setting.
This source uses exact full-graph sparse propagation and a 65,536-row limit on
individual dense/readout calls. That implementation choice is disclosed in the
[implementation contract](IMPLEMENTATION_CONTRACT.md). The source does not execute
the complete external-baseline HPO campaign described by Table S15.

## 2. Reported controller qualification (Table S41)

Table S41 labels these entries as the entire qualification search per dataset.
Values are reported GPU-hours; the component shares describe that ledger.

| Component | Share | arxiv | proteins | Roman | Pascal | COCO | Peptides |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exploration logging | 0.40 | 0.77 | 0.80 | 0.55 | 0.64 | 0.72 | 0.68 |
| Observer, critic and sequence-model fitting | 0.15 | 0.29 | 0.30 | 0.21 | 0.24 | 0.27 | 0.25 |
| Nested comparator and candidate selection | 0.35 | 0.67 | 0.70 | 0.48 | 0.56 | 0.63 | 0.59 |
| Confirmation evaluation | 0.05 | 0.10 | 0.10 | 0.07 | 0.08 | 0.09 | 0.08 |
| Admission evaluation | 0.05 | 0.10 | 0.10 | 0.07 | 0.08 | 0.09 | 0.08 |
| **Total increment (GPU-h)** | | **1.92** | **2.01** | **1.38** | **1.59** | **1.81** | **1.69** |

The reported totals correspond to the Table S15 difference between the deployed
rule's GPU-hours and the plant's; printed component rounding can affect their
sum. Products, which returns `qC`, has no measured component row in Table S41.
Its Table S15 difference is **2.28 GPU-h**, derived as `15.2 - 12.92`. The reported
arxiv ratio `12.8 / 10.88` is approximately **1.176×**. These arithmetic relations
do not predict the duration of the repository's exploration, fitting or candidate
evaluation loops.

## 3. Reported deployment accounting (Tables S23 and S23b)

Serial costs use one nominal full-depth execution as their unit. Memory, latency
and policy choices in this table are manuscript observations; they are not
automatic configuration choices or resource guarantees for a regenerated run.

| Dataset | Deployed rule / E | Controller cost | Comparator cost | Fallback probability | Expected cost (Eq. 32) | Assurance cost | Wall-time overhead | GPU / CPU memory (GiB) | t_nom per input |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| ogbn-arxiv | radius / 4 | 4.700 | 4.000 | 0.118 | 4.617 (+15%) | 8.700 (+118%) | 13% | 15.1 / 21.0 | 9.54 s |
| ogbn-proteins | radius / 4 | 4.800 | 4.000 | 0.147 | 4.682 (+17%) | 8.800 (+120%) | 14% | 37.1 / 44.0 | 9.69 s |
| ogbn-products | `qC` (phase) / 2 | — | 2.100 | — | 2.100 (0%) | 2.100 (0%) | 0% | 64.5 / 118.0 | — |
| Roman-empire | MPC-RH / 4 | 5.400 | 4.000 | 0.129 | 5.219 (+30%) | 9.400 (+135%) | 25% | 12.4 / 18.0 | 1.16 s |
| PascalVOC-SP | MPC-RH / 8 | 10.500 | 8.000 | 0.081 | 10.298 (+29%) | 18.500 (+131%) | 24% | 6.1 / 12.0 | 0.73 ms |
| COCO-SP | MPC-RH / 8 | 10.800 | 8.000 | 0.071 | 10.601 (+33%) | 18.800 (+135%) | 27% | 7.8 / 16.0 | 0.73 ms |
| Peptides-func | radius / 4 | 4.900 | 4.000 | 0.188 | 4.731 (+18%) | 8.900 (+123%) | 15% | 18.6 / 25.0 | 0.56 ms |

The `t_nom` entries follow Table S23b's per-input convention. Its COCO entry
shares the PascalVOC-SP anchor; the registry retains that explanatory note and
the separate training-time reference values. Inspect those accounting fields
when using nominal times in an amortization calculation.

The expected cost is the static-branch case of Eq. (32). The assurance cost is the
separately provisioned two-branch configuration `C_controller + C_qC`; it is not
an observed maximum or a measured 99th percentile. The modelled per-input
`C_0.99` in Table S44 is an analytical tail quantity and is not produced by this
source. Repeated timing of the same prepared graph or corpus is another distinct
quantity. A run's saved policy identities and operation traces determine which
branch and workload were actually executed.

## 4. What the released profiles request

| Profile | Data | Model | Development computation | Purpose |
|---|---|---|---|---|
| `smoke` | generated task-shaped fixture: 160 nodes, or 80 graphs × 8 nodes | depth 6–8, width 16, two seeds, at most three epochs | 12 trajectories, 80 resamples, `n_min = 2`, entry contract not enforced | CPU software check; our CI workflow exercises this profile |
| `pilot` | complete official graph or corpus and official splits | depth 32; width 128 for single-graph tasks, 64 for independent-graph tasks; two seeds, at most 30 epochs | 24 trajectories, 500 resamples, `n_min = 64`, entry contract not enforced | integration and resource measurement; support limitations make comparator return a valid outcome |
| `full` | complete official graph or corpus and official splits | dataset-specific backbone settings, twelve seeds | 9,600–32,000 trajectories, 10,000 resamples, entry contract enforced | the configured scientific protocol; inspect exact definitions and results |

The [full-profile table](GETTING_STARTED.md#full-profile-settings) lists each
dataset's width, depth, epoch ceiling and trajectory budget. The registry's actual
candidate count is written by the run; it is not inferred from a generic profile
label. Every training seed fits the plant and its separate feature-only `q0`.
The full Roman campaign runs ten official masks, each with the configured seed
list, so its jobs must be budgeted together.

### Main resource drivers

| Workload | Resource implication |
|---|---|
| Four single-graph tasks | Full graph operators and `N × hidden_dim` states remain resident during an episode. Reducing a dense chunk size does not partition sparse propagation. |
| Products | The large graph, dense hidden states, sparse kernels and spectral work all contribute. The reported 64.5 GiB figure above is not a capacity estimate for this implementation. |
| Three independent-graph tasks | One complete graph is materialized on the accelerator at a time; training accumulates gradients across the configured graph batch. CPU preparation and storage still contain the whole corpus. |
| GPS superpixel tasks | Attention is local to each graph, so the largest episode contributes to peak memory even when average graphs are small. |
| Exploration and candidate evaluation | Trajectory count, controlled depth, spectral sensing, retained seeds, horizons and policy registry size all multiply work. Saved predictions and traces consume disk space. |
| Selection and replay | Nested comparator reconstruction and resampling can require substantial CPU time and memory. Increasing GPU capacity does not remove that work. |

All our real-data profiles use BF16 except Roman-empire, which uses FP32.
BF16 requires a compatible CUDA device; CPU smoke uses FP32. The Slurm wrapper
requests one GPU and 16 CPU threads by default, but no explicit host-memory or
GPU-model constraint. Its default walltime is a scheduling request, not a finish
estimate. Full runs can span allocations through the snapshot and resume workflow
in [HPC.md](HPC.md).

## 5. Measure a run before scaling

First run the synthetic fixture to verify execution. Then run the real-data
pilot on the intended hardware with fresh output and report paths. For example,
on Linux with GNU `time`:

```bash
mkdir -p ../cm_reports
/usr/bin/time -v -o ../cm_reports/products_pilot_resources.txt \
  python -B run_cross_morse.py --dataset ogbn-products --profile pilot \
  --device cuda --threads 16 --data-root ../cm_data \
  --output-dir ../cm_runs/products_resource_pilot
```

Keep `environment.json`, the resolved `config.json`, prepared-data metadata,
`train/model_inventory.json` and the stage records with that resource report.
`stages/*.json` and training reports expose `seconds_this_attempt`; after a
resumed run, that field is not a complete sum of all prior allocations. Preserve
launcher logs and scheduler accounting as well. GNU `time` reports host-process
resources; it does not measure GPU allocated memory.

To measure fixed predictors after a completed run, use the trusted-checkpoint
evaluator from the corresponding source snapshot:

```bash
python -B scripts/evaluate_checkpoints.py ../cm_runs/products_resource_pilot \
  --trust-local-checkpoints --device cuda --threads 16 --models qC qD \
  --warmup 3 --repetitions 20 \
  --output ../cm_reports/products_pilot_inference.json
```

This records synchronized execution times, CUDA allocated-memory peaks and work
counters for the saved models. It includes the tool's declared sensing, tracing,
copy and ensemble work; models are preloaded. Report training and controller
development separately. Its repeat percentile is conditional on the fixed
prepared inputs, not a latency-tail estimate over unseen graphs. Use the measured
pilot to choose allocation limits, then measure the full configuration itself;
width, depth, seed count and planning work do not imply a universal linear
walltime scaling rule.
