# Executable implementation contract

This document defines the implemented algorithm, dataset-specific execution and
scientific interpretation. It accompanies our submitted main manuscript
and supplement, whose reported measurements have their
own provenance. Material implementation choices and differences are listed in
[`paper/KNOWN_DIFFERENCES.md`](../paper/KNOWN_DIFFERENCES.md);
the registered constants the contract must honour are in
[`paper/registered_constants.json`](../paper/registered_constants.json) and are
checked by `scripts/check_paper_constants.py`. A completed run's resolved configuration, actual tensor
inventory, prepared-data identity and sealed observations specify what ran.
Manuscript agreement requires the corresponding result-producing records. The
[method overview](METHOD_OVERVIEW.md) explains the algorithm, and
[theory and scope](THEORY_AND_SCOPE.md) states which conclusions follow under
which assumptions.

## Frozen predictor and execution unit

The implementation constructs its frozen backbones; `build_plant` chooses the variant
from the dataset configuration. Training fits the predictor and a separate
feature-only `q0` before exploration. Subsequent stages freeze the encoder,
propagation modules, normalization, attention and prediction head. The supported
operator is nonnegative, state-independent and linear in node states, with a
row-normalizable weight matrix. It does not include standard nonlinear GINE edge
messages or a controlled state-dependent attention kernel (main p. 4, Definition
1).

| Dataset | Backbone convention | Full control depth / width | Execution episode |
|---|---|---|---|
| ogbn-arxiv | Four-layer encoder, shared learned propagation update | 256 / 512 | Complete transductive graph |
| ogbn-products | Three-layer encoder, shared update, chunked dense calls | 64 / 512 | Complete transductive graph |
| ogbn-proteins | Four-layer encoder, nonnegative edge-weighted transport | 128 / 512 | Complete transductive graph |
| Roman-empire | Separate ego and neighbor maps, four-layer encoder | 256 / 384 | Complete graph for one official mask |
| PascalVOC-SP | Six local-plus-global GPS-style blocks; graph-local attention | 128 / 128 | One complete independent graph |
| COCO-SP | Six local-plus-global GPS-style blocks; graph-local attention | 128 / 128 | One complete independent graph |
| Peptides-func | Eight linear-aggregation GINE-style blocks; graph mean readout | 64 / 128 | One complete independent graph |

The encoder, local aggregation, normalization and readout details are explicit
implementation choices. In particular, the arxiv plant with 128 input features,
width 512 and 40 classes has **1,138,216 parameters**, while Supplementary Table S5
lists 3.1M (supplement p. 3). This source therefore
does not establish architectural identity with the reported plant. No parameter
count is forced to match a table by adding unused tensors.
Each run writes `train/model_inventory.json` from actual checkpoint tensors and
records instrumentation metadata. Architecture descriptions in a result should
be checked against that inventory.

### Native and inserted sites

The full GPS configurations expose six native local-aggregation sites at
zero-based transitions `[1, 22, 43, 64, 85, 106]` within 128 control transitions.
The remaining 122 transitions include encoder/decoder boundaries and inserted
transport sites. The full GINE configuration exposes eight native sites at
`[0, 8, 16, 24, 32, 40, 48, 56]` within 64 transitions. Exact placement uses the
first site of each declared block interval; the manuscript supplies counts but
does not fully specify every index (supplement p. 4, Table S5b).

A0 at an inserted site is a **literal identity**, without learned update,
normalization or dropout. A0 at a native site executes the corresponding native
block. Nonzero native-site transport uses message-space source retention,
`(1-alpha) * [(1-nu)H + nu K_a H] + alpha H0`, passed to the native block.
The papers leave this backbone-specific retention convention unspecified.
Inserted sites use the non-lazy kernel once inside
`(1-nu)H + nu[(1-alpha)K_a H + alpha H0]`. GPS global attention retains its
graph-local computation law and parameters; its activations can change when the
adapter changes the block input. A8/A9 are direct assignments
at both kinds of site. Instrumentation metadata and identity regression tests
make these boundaries inspectable. The nominal-identity audit compares the
independent `native_step` and A0-dispatch paths of this plant, at each registered
site and its final head. This establishes their equality within the declared
precision tolerance; it does not compare against an unavailable original
checkpoint. Generic hooks for arbitrary third-party models are outside this
repository's adapter interface.

### Graph boundaries and memory

For the three independent graph datasets, the CPU-side prepared corpus stores
explicit graph/edge pointers and labels on the appropriate prediction axis.
Training and prediction materialize one complete graph on the accelerator.
Training accumulates gradients across the configured number of graphs. Every
control episode starts with fresh source state, sensors, observer, plan and safe
anchor. Attention and graph pooling never span different examples.

For OGB and Roman-empire, control is global to the complete graph. Cores summarize
losses and define resampling groups; they do not execute independent policies.
Buffer halos are recorded context metadata. The products dense chunk size limits
individual dense calls but retains the full `N × hidden_dim` hidden states and
full sparse propagation. **This is not a partition-local products implementation.**
Equivalence to the manuscript's partition-local boundary treatment is not
established by full-graph execution. Memory/runtime must be measured on the
target hardware before a full run; a pilot uses the official dataset with
reduced compute and can still be substantial.

## Data roles, outputs and losses

[DATASETS.md](DATASETS.md) specifies official files, features and masks. The official
training portion is divided 60/20/10/10 into `Dfit`, `Dexplore`, `Dselect` and
`Dconfirm`. `Dfit` has its own internal checkpoint-validation subset. Official
validation is `Dadmit`; official test is opened only after the policy lock.
Independent graph roles contain complete graphs. These are the separated roles
of supplement p. 26, Table S39, not the
reused-validation sequence behind the reported multi-graph results. All ten
Roman masks remain available through `--split-index 0` through `9` and the campaign launcher.

Multiclass tasks use softmax and cross-entropy. Multilabel tasks use independent
sigmoids and masked binary cross-entropy, averaging observed tasks within a
sample. Fully unlabelled samples do not become zero-loss observations. Peptides
predictions pool within each graph and use graph labels; superpixel tasks retain
node labels. Task metadata determines head width without reading held-out labels.

The checkpoint rule is internal validation loss for the OGB, Roman and superpixel
tasks. Peptides selects maximum internal validation AP, breaking ties by BCE.
Graph loss is the mean of the complete-graph losses, with equal graph weight.
For a single transductive graph, core means are weighted by their scored-node
counts. These weights are preserved in saved decision records.

## Actions and reference states

A0–A9 are registered in `core.py`, with constants from Supplementary Tables S6/S6b, pp. 4–5.
Resolvents use mass-corrected truncation and reuse the first propagation product.
A8 assigns the encoded source; A9 assigns the last recorded safe state. Direct
resets do not run the learned update or LayerNorm. Open-loop planning excludes a
dynamic A9 reference; exact diagnostic replay can supply its reference index.
Phase/fixed-sequence construction canonicalizes proposed A9 actions to A8 and
records that provenance. Fixed-sequence proposal ranking uses the original
exploration loss as a heuristic; selection evaluates the actual registered
sequence. MPC-OL excludes A9 from every consequence-model query, including
simulated recovery, and uses the modeled source reset A8 when needed.

The shared anchor predicate is `recovery.label_free_safe`: a nonempty finite
measurement vector with normalized entropy at most 0.99. Exploration and feedback
execution use the same rule. Invalid observer state can trigger internal recovery
but does not change this anchor-update rule. This is an execution convention,
not a certificate of predictive safety. The compressed consequence state does not
contain the full saved anchor tensor. MPC-RH's predicted A9 consequence is an
approximation conditioned on that compressed state; execution restores the
actual saved tensor, and traces bind its depth separately.

## Measurements and observer

Configurations retain bands `[0, 2, 5]` from six degree-eight Chebyshev windows.
The windows have explicitly defined equal-interval coefficients with Jackson
damping. This is fixed band selection, whereas main p. 5, Eq. 11
specifies selection inside training folds followed by freezing. Degree-weighted
sensing is computed and charged separately. The unsketched Full measurement uses 17 sparse products:
eight unweighted, eight degree-weighted and one Dirichlet product. Source
preparation uses eight. Dense/readout/rank work is additional.

The effective-rank coordinate uses a fixed projection of width at most 32 and a
small covariance eigensolve. Optional Gaussian spectral projection reduces filter
width; its band-power variance bound does not bound downstream action changes or
accuracy. Our full profiles use full-state spectral filtering.
`controller.observer_period=2` sets the observer correction cadence; sensing still
occurs at each feedback depth. The manuscript's measured sensing multipliers on
proteins and Pascal exclude full-state degree-eight acquisition at every site,
but the executed sketch width, sensor period and resolvent fraction are not
tabulated (supplement p. 5, Table S6c;
p. 21, Note S19). This implementation's
operation ledger must be used when interpreting a new run's cost.

Reachability uses 1,025 grid points per retained interval. Its nominal polynomial
is site-dependent: `0.3+0.7*(1-lambda)` for shared blocks, `1-lambda` for designated
native aggregation, and `1` for inserted identity. The seven terminal coordinates
are zero because no next control action remains. Metadata records these baselines.
The descriptors compare transport **before source retention** and omit discarded
bands, window-approximation corrections, a certified continuous supremum,
degree/norm prefactors, differing retention/reference terms and native-update
constants. They are partial features, not the full bound of Proposition 6.

Lean zeroes the spectral/reachability coordinates and fits a separate controller.
Multilabel confidence uses mean binary entropy, threshold margins and binary
source divergence rather than categorical softmax uncertainty. Graph predictions
supply graph-level confidence while hidden-state geometry uses that graph's
nodes. Measured operation ledgers govern resource claims.

The observer uses fitting-group robust normalization/PCA, conditioned ridge
dynamics and low-rank-plus-diagonal covariance estimates. Its consistency
statistics differ from critic uncertainty. The critic's target compares an action
followed by nominal continuation with nominal continuation from the same state
(main Eq. 13). It is not a comparator-relative complete-policy effect.

The explicit observer convention is median/MAD scaling clipped to `[-5,5]`,
PCA dimension 16, ridge dynamics conditioned on depth bin/action, covariance
rank four plus a positive diagonal, and measurement covariance formed from
`0.25 * Q + 1e-3 * I` with the same covariance projection. The NIS threshold is an
empirical internal-validation quantile. These identify a concrete state estimator;
they are not recovered noise-model parameters from the manuscript.

## Calibration and assessment

`Dexplore` records are divided into model fitting, radius fitting and assessment,
approximately 60/20/20 subject to rounding and minimum group counts. For a single
graph, the group is `(seed_index, trajectory)`. For independent graphs, every
trajectory and seed of one graph stays in the **same group**. Sequential observer
filtering still resets on each trajectory, while fitting and calibration holdouts
respect complete-graph membership.

Observer, critic and sequence models fit only the model-fitting groups. Internal
cross-fitting respects the same holdout groups. The final model freezes before
radius fitting and is not subsequently refitted on those held-out observations.
The action critic has hidden widths 128/128/64, SiLU and dropout 0.1, with
mean/log-scale outputs. Transition, stage and terminal heads are separate
two-layer width-128 residual models. Full profiles use five cross-fit folds,
AdamW at learning rate 0.001 and weight decay 0.0001, batch size 512, at most 200
epochs and patience 20; early stopping uses only internal model-fitting groups
(main p. 18, Methods;
supplement p. 6, Table S9).
Group-wise residual maxima provide empirical cell/action radii; cells with too
few groups use action-pooled radii, and unobserved actions have infinite radius.
The nominal-relative A0 target and radius are exactly zero.

The configured residual quantile is 0.95 and is reported as empirical calibration.
Assessment uses different groups and the same arithmetic as execution:

`mean + kappa * sigma + residual_radius + lambda_rho * rho`.

Here `rho` is the kernel-mixture coefficient; it differs from `nu`, the lazy
transport mass. The separate operational radius `r_op` in main Eq. 16 is zero.
MPC uses action-critic scores for shortlisting, then discounted stage predictions
plus their scale and declared `rho`/executed-hop costs, with a separately learned
nominal-suffix terminal value.

Logged-history assessment includes matching behavior actions, sample/group
counts and an inverse-propensity diagnostic. It does not measure fully deployed
closed-loop state coverage or MPC sequence coverage. Those fields remain
`NOT_MEASURED`; additional trajectories on one graph do not create independent
graphs or establish a finite-sample conformal guarantee.

## Policy family and target qualification

The nine core families are nominal, best depth, learned stop, phase schedule,
fixed sequence, MPC-OL, one-step, radius and MPC-RH. Learned stopping is an
initial-measurement nearest-template predictor of development-best depth. On a
single graph, its own checkpoint's initial template is available; that setting
measures checkpoint-specific depth choice. On graph datasets, development
examples provide templates applied to separate graph episodes.

Realized candidate counts are emitted by the registry. They depend on distinct
sequence proposals and configured profiles/horizons/ensembles; the implementation
does not force a manuscript candidate count. The headroom gate chooses one
complete evaluated candidate by weighted unit loss. A per-unit optimistic oracle
is retained only as a diagnostic and cannot qualify the gate. This finite-family
diagnostic is not the paper's per-input supported-sequence beam oracle
(main p. 7, Proposition 4;
supplement p. 20, Table S30).

The support mask uses depth/entropy cells and one mean inverse-propensity weight
per complete group in each action/cell. A full profile requires propensity at
least 0.05, ESS at least 64 and maximum normalized weight at most 0.20. The same
weight vector defines both statistics. Repeated steps do not inflate the group
count. These are overlap diagnostics, not a per-action improvement guarantee.
The paper withdraws its incompatible dataset-level ESS and maximum-weight
summary; a new run's `support.cells` record is the relevant check
(supplement p. 5, Table S7).

## Selection, confirmation and deployment

Selection uses the registered upper-limit rule and `epsilon_eq=0.002`. Nested
comparator reconstruction is repeated in every bootstrap draw. The representative
rows of Supplementary Table S12c reproduce only four of the seven reported route
choices under that rule; this runtime does not retune the criterion to force the
reported choices. Failed candidates remain ineligible with unavailable measurements; they receive no fabricated
comparator scores. Confirmation can retain an eligible policy or descend to an
eligible predecessor or `qC`. Missing required contrasts remain unavailable and
cannot pass. Holm adjustment uses the declared per-dataset/per-policy contrast
family; it does not reproduce the paper's nine cross-dataset rows as one family.
Matched state destruction changes controller observations while
preserving plant states and intersects donor/recipient support; its matching keys
are depth, previous action and initial entropy bin. This matching resolution is
a declared choice, not exact conditioning on complete histories.

Accuracy is additive over observed labels. **Macro F1, ROC-AUC and AP are not.**
For those metrics, selection retains candidate probability observations and
reconstructs the out-of-fold comparator predictions. Each resample recomputes the
benchmark metric on weighted observations. Fixed-policy admission and test
contrasts also resample complete units and recompute the metric. Rank-based
metrics can be expensive at the full 10,000-resample budget; full-data selection
throughput has not been measured by the local fixture checks. Multilabel AUC/AP
follow the benchmark convention of macro averaging eligible observed tasks.

On the four single-graph datasets, admission and intervals are descriptive under
core dependence. For the three independent graph datasets, loss admission uses
the clipped empirical-Bernstein upper bound with equal graph weighting, together
with the configured improvement and metric gates. Its interpretation requires
independent admission graphs from the target distribution and a fixed policy and
comparator selected without admission labels. Numerical checks do not verify
those population assumptions. The paper's reported multi-graph admission numbers
reuse the validation observations that fixed confirmation, so Theorem 2 does not
apply to those reported numbers (main pp. 6–7;
supplement p. 31, Note S26). A new separated
run still requires the stated independence and no-adaptation assumptions before
its computed bound has that interpretation.

A rejected global admission copies `qC`'s prediction bytes exactly to `qD` on the
prepared benchmark. Internal recovery is a separate event. A general serving-time
input gate for novel graphs is not implemented. Test intervals condition on the
saved prefix ensemble; they do not include repeated-retraining uncertainty.

Numerical policy failures have a typed diagnostic trace. Before a comparator is
locked they make a candidate infeasible with unavailable scores; a failed
confirmation intervention cannot support a mechanism claim. Where a comparator
is locked, the pipeline may return its saved predictions and records that event.
This failure path is not the optional budget-aware `qB` policy or a proof of a
hard per-input work cap.

Roman's campaign summary uses ten paired split effects and a Student-t descriptive
interval. Its overlapping masks are not ten independent graphs, and the summary
does not change the ten sealed per-run decisions.

## Scope and limitations

All seven dataset adapters, task metrics and the core lifecycle are implemented.
Full official benchmark runs and the original weights are not included in the
source package. Local validation scope and evidence are in [VALIDATION.md](VALIDATION.md).
E3 initial-input sequence prediction, E4 distillation, E5 no-observation replanning,
optional `qB`, external-baseline HPO, generic backbone hooks and the complete
E1–E20 campaign are not represented as completed experiments here.

Sparse and dense multiply-add counters cover named operations; they do not count
all elementwise work, memory traffic or host overhead. Total wall times include
CUDA synchronization, while per-component host timing is not a kernel-level
profiler. Neither those counters nor the model-call planning budget implement
Algorithm S3's complete per-operation upper bounds and reserved comparator
restart. Expected cost, assurance cost, a measured maximum and a hard cap retain
separate meanings (supplement pp. 30–31, Notes S22–S23).

Integrity checks establish consistency of the shipped files. They do not independently
establish experiment chronology, untouched labels or manuscript agreement. Report
scientific values with their actual run artifacts, source, settings and units.
