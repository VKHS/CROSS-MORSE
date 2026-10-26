# Manuscript-to-code map

This page maps the principal method objects and the E1–E20 experiment programme
of **CROSS-MORSE: Cross-Spectral Control of Message Passing in Frozen Graph Neural
Networks** to executable functions, records and stated limits. The source PDFs
are our submitted main manuscript and supplement.
Page references below use their printed page numbers.

The [method overview](METHOD_OVERVIEW.md) explains the algorithm, the
[implementation contract](IMPLEMENTATION_CONTRACT.md) specifies concrete choices,
and [theory and scope](THEORY_AND_SCOPE.md) explains the assumptions behind each
type of conclusion. [Known differences](../paper/KNOWN_DIFFERENCES.md) and
[results](RESULTS.md) separate reported measurements from a new run's observations.

**Status vocabulary:** *executed* means a function implements the stated operation;
*fixture-tested* means a test checks it on a specified small input; *arithmetic*
means a calculation can be regenerated from the registered constants; *conditional*
means a mathematical result needs the stated assumptions; *partial* and *outside*
identify the limits of the runtime. None of these labels, alone, says the source
reproduces the original trained model or a reported benchmark measurement.

| Paper pages | Objects |
|---|---|
| Main 2–3 | Feedback value, information restrictions, lifecycle; Eqs. 1–4 |
| Main 4 | Supported class, entry/support, transport and resolvent; Definition 1, Eqs. 5–9 |
| Main 5 | Range, measurements, observer, targets, routes, scores and selection; Eqs. 10–19 |
| Main 6–7 | Confirmation, transport/replay/perturbation results, selection/admission, headroom and costs |
| Main 8–9 | Spectral sensitivity, regret, sensing and budget/comparator identities; Propositions 6–11 |
| Supplement 3–7 | Plants, site layout, actions, sensor schema, controller settings and Algorithms S1–S4 |
| Supplement 14–15, 20–21, 30–31 | Proofs and assumption details |
| Supplement 24–27 | Experiment definitions, rates, observation schema and data-use/information contracts |

## 1. Main text

### Definitions and the plant

| Anchor | Statement | Implementation | Status |
|---|---|---|---|
| Definition 1, p. 4 | Readable states and nonnegative state-independent linear aggregation; native update/head/precision fixed | [`core.py`](../cross_morse/core.py): explicit `FrozenPlant`, `EdgeWeightedPlant`, `HeterophilyPlant`, `DesignatedSitePlant` and `build_plant`; `GraphOperator` builds `P` and symmetric `S` | executed for declared plants; no generic third-party hooks; standard nonlinear GINE and controlled attention kernels excluded |
| Definition 1; E1; Tables S5b/S5c, supplement p. 4 | A0 reproduces native execution; hybrid native and inserted sites distinguished | `training.audit_training`, independent `native_step`, A0 `step`; GPS 6 native/122 inserted, Peptides 8 native/56 inserted | executed and fixture-tested; identity is for the constructed plant, not unavailable original checkpoints |
| Eq. (1), p. 2 | Budget- and family-specific value of feedback, relative to initial-input open-loop policies | `policies.make_registry` fixes the compared families and resource configurations; the infima over unrestricted policy classes are not computed | conceptual definition; finite registered comparison executed |
| Eq. (2) | Head `p_l = h_head(H_l)`, measurement map `y_l`, observer `s_l`, policy `a_l = π_q(s_l)` | `FrozenPlant.logits`; `MeasurementBank.measure`; `learning.Controller.observe`; `policies.execute_policy` | executed |
| Eq. (3) | Policy lifecycle `Q_OL → q_C`, `Q_FB ∪ {q_C} → q̂_s → q_M`, `q_M → q̃_M → q_D` | [`pipeline.py`](../cross_morse/pipeline.py): `_stage_select` (nested `q_C`, provisional `q̂_s`), `_stage_confirm` (`q_M`), `_stage_admit` (`q_D`, `policy_lock.json`), `_stage_evaluate` | executed; `tests/test_lifecycle.py` |
| Eq. (4), p. 3 | Input compatibility/recovery gives executable `q̃M`; global admission chooses it or `qC` | `Pipeline._stage_admit`, `statistics.admission`; rejected global admission copies the comparator prediction bytes; prepared-graph identity check records static compatibility | global gate executed; general novel-input `g_input` outside |
| Eq. (5), p. 4 | Training-only plant/depth/support/headroom entry criteria | `training.audit_training`, `Pipeline.entry_completion`; `entry_initial.json`, `entry_complete.json` | executed with finite-candidate headroom diagnostic; not the paper's per-input beam oracle; reported LRGB Table S4 uses a different post-decision unit |
| Eq. (6) | Log-depth-integrated loss on a geometric depth grid | `audit_training`: depth grid `{1, 2, 4, …, L}`, `depth_<loss>` and `dil_gap` | executed |
| Eq. (7), p. 4 | Propensity ≥0.05, ESS ≥64, maximum normalized weight ≤0.20 on the same cell weights | `learning._support`, `Controller.supported`; one group weight per action/stratum, complete-graph grouping for independent examples | executed; repeated depths do not increase the group count; paper Table S7's withdrawn summary is not reused |
| Eq. (8) | Common adapter action `(ρ, ν, α, m)`: kernel mixture, transport mass, native update, reference retention | `core.apply_kernel` (`K_a = (1−ρ)P + ρ K̃`), `apply_transport` (`T = (1−ν)I + ν K_a`), backbone `step` methods (`φ(T H W + b)`, LayerNorm, `α` retention of `H_0`); inserted-site form `V_a(H, R)` in `DesignatedSitePlant.step` | executed; `tests/test_task_plants.py::test_inserted_transport_formula_and_hybrid_resets` |
| Eq. (9), Lemma 2 | Mass-corrected truncated resolvent `K̃_{η,T}`, `T = 8`, `‖K_η − K̃_{η,T}‖_∞ ≤ 2(1−η)^{T+1}` | `apply_kernel` (eight sparse products, first product reused), `identity_mass`; registry constants `ACTIONS` | executed; `tests/test_core.py::test_transport_agrees_with_independent_dense_resolvent`; Table S6b reproduced by `check_paper_constants.py` |
| Eq. (10), p. 5; Table S6b | Expected transported hops and identity coefficient, distinct from maximum reach and graph-return diagonal mass | `executed_hops`, `executed_identity_coefficient`, `site_transport_accounting` | executed and arithmetic checks; designated native A0 is 1/1 mass/hops, inserted A0 is 0/0 |
| Eq. (11), p. 5; Table S8 | Band power, source alignment, degree-weighted power and seven action-reach coordinates | `MeasurementBank`, shared Chebyshev recurrence, fixed Jackson windows and bands `[0,2,5]`; Full/Lean fitted separately | executed with declared fixed windows; full acquisition costs 16 filter products plus one Dirichlet product; observer cadence does not skip acquisition |
| Eq. (12) | Depth/action-conditioned observer with state dimension 16, covariance rank 4, NIS/bias/whiteness validation | `learning.ConditionedObserver` (`transition`, `observe`, `start`), `_fit_observer`, `ControllerState.valid` | executed; `tests/test_learning.py::test_observer_psd_and_missing_history_recovery` |
| Eq. (13) | One-action target `Δ_{l,h}(a) = E[ℓ_{l+h}(a → π_N) − ℓ_{l+h}(π_N)]`, `h = 1` | [`exploration.py`](../cross_morse/exploration.py): `collect_trajectory` (paired counterfactual branch from the same state; A0 target exactly zero) | executed; `tests/test_exploration_targets.py` |
| Eq. (14) | Selection effect `Δ_sel(q)` against `q_C`; clipped admission mean `μ_adm^clip` with `B = 0.10` | `statistics.nested_selection` (paired to the reconstructed comparator), `statistics.admission` (clip on the paired unit difference) | executed; `tests/test_statistics.py` |
| Eq. (15) | Families `Π_OL = {nominal, depth, stop, phase, seq, MPC-OL}`, `Π_FB = {one-step, radius, MPC-RH}` | [`policies.py`](../cross_morse/policies.py): `make_registry` (route-specific profiles, horizon/κ and ensemble configurations), `execute_policy` | executed; `tests/test_policy_guardrails.py::test_registry_is_fixed_from_exploration_and_configuration_without_selection` |
| Eq. (16), p. 5 | Mean-only and uncertainty-adjusted action scores | `Controller.score`, `scoring.action_scores`: mean + κ scale + empirical radius + `lambda_rho*rho`; `r_op=0` | executed with empirical calibration; no selected-policy coverage theorem |
| Eq. (17), p. 5; Table S9 | Discounted model-based sequence search | `policies.beam_plan`; separate transition, stage and terminal networks; γ=.97, horizons 1/4/8/16/32, beam16/top2; uncertainty and transport terms declared in config | executed approximate beam search; nominal completion beyond MPC-OL horizon; no uniform sequence-error certificate |
| Eq. (18), p. 5 | Feasible candidate set constrained by support, metric and resources | `execute_policy` support/work traces; `_stage_select` eligibility and metric gates; failed executions have unavailable scores | executed on registered evaluated inputs; not a proof for every possible future state or a hard work cap |
| Eq. (19), p. 5; Algorithm S4, supplement p. 7 | Eligibility `U_s < -epsilon_R`, upper-limit near-best band .002, deterministic simplicity; nested comparator reconstruction | `statistics.nested_selection`, `_reconstruct_comparator`, `_max_t_interval`; resampling reconstructs comparator family/configuration choices | executed; coverage remains conditional; Table S12c representative rows match only 4/7 reported choices under the written rule |
| Eq. (20), p. 6; Table S11, supplement p. 8 | Confirmation retains an eligible configuration, eligible simpler predecessor or `qC` | `statistics.confirmation`, `holm_adjust`; required unavailable contrasts cannot pass; `DonorBank` changes controller observations, not plant states | executed per-run declared Holm family; not the paper's nine-row cross-dataset multiplicity object |
| Lemma 1 | Convex-mixture transport is nonnegative and row-stochastic | `GraphOperator` (nonnegative weights, degree normalization), `apply_transport` | fixture-tested (`tests/test_core.py::test_sparse_graph_mass_and_isolate`) |
| Proposition 1, p. 6 | Replay realized actions and recovery anchors deterministically | `execute_policy(replay_actions=..., replay_safe_depths=...)` | fixture-tested; execution identity does not establish the value of feedback |
| Proposition 2, p. 6; Table S22 | Fixed-mode finite-depth perturbation bound | Same actions, recovery modes, anchor indices and reference rules required; full trajectory-tube audit is not emitted | conditional bound; paper's sampled-Lipschitz plug-in tube is empirical, not worst-case certification |
| Proposition 3, p. 6; Table S21 | A sufficient score-gap condition for action stability | Feedback traces record `score_gap`; certified Lipschitz regions and switch-fraction audit are not computed | partial instrumentation; conditional result |
| Theorem 1, p. 6; Table S20 | Improvement and near-best selection on simultaneous family coverage | `nested_selection` computes the family intervals, eligible set and maximum radius | conditional theorem; code executes the estimator, not a proof of coverage |
| Theorem 2, pp. 6–7; Corollary 2; Table S13 | Empirical-Bernstein upper limit for fixed-policy clipped loss on fresh independent graphs | `statistics.empirical_bernstein`, `admission(independent_graphs=True)`; minimum-sample arithmetic in constant checker | formula executed; population assumptions required; reported reused-validation limits are descriptive |
| Corollary 1, pp. 6–7 | Joint per-dataset statement by a union bound | Selection/admission functions compute the associated statistics | conditional on both events' coverage assumptions; numerical execution alone is not a 95% certificate |
| Proposition 4, p. 7; Table S30 | Supported per-input oracle headroom limits attainable gain within its sequence family | `Pipeline.entry_completion` instead evaluates the best single complete candidate at E=1 on `Dexplore`; per-unit optimism is diagnostic only | partial; different oracle family/estimand explicitly declared |
| Proposition 5, Tables S26, S26b | Equal-compute break-even under the ensemble law | ensemble grid E ∈ {1, 2, 4, 8, 12} in the registry; the law fit and E ∈ {16, 20} are outside | partial (D15) |
| Definition 2, p. 7; Eq. (30) | Exchange rate and amortization given measured benefit/work and consistent units | `scripts/check_paper_constants.py` checks the transcribed arithmetic; new runs emit their own work records | arithmetic; source-ledger unit reconciliation and target-hardware measurement are still required |
| Proposition 6, p. 8; Note S17, supplement p. 20 | Ideal-band displacement bound and exact alignment identity | `action_operator_response`, `nominal_operator_response`, `MeasurementBank.sensitivities_by_baseline`; 1,025-point grids against shared/native/inserted A0; terminal reach=0 | partial descriptor; omitted-band/window/degree/retention/native-update terms are not certified; E18 empirical audit outside |
| Proposition 7, p. 8; Tables S28/S28b | One-step regret conditional on simultaneous critic errors; independent-unit concentration examples | `_support` emits overlap statistics, not a concentration guarantee | analytical/conditional; self-normalized ESS cannot substitute for independent sample count |
| Proposition 8, p. 8; Note S20 | Sequence regret ≤2e_H + search error under uniform model error | `beam_plan` implements approximate search; no exact-enumeration E9 audit | conditional; action RMSE does not establish uniform sequence error |
| Proposition 9, pp. 8–9; Algorithm S3 | Selective sensing with invariant support/mode/search and valid state radius; optional restart-reserved work cap | Acquisition certificate and `qB` are outside; runtime planning-call limits are a different mechanism | outside; no hard-cap claim |
| Proposition 10, p. 9; E2–E6 | Budget-matched and priced-work boundaries; ensemble-spend sensitivity | Additional predictor, distillation, no-observation and `qB` arms outside | analytical identities/sensitivity and separately reported experiments |
| Proposition 11, p. 9; Note S21 | Comparator-relative advantage sums to complete-policy value difference | The actual critic estimates Eq. 13's short-horizon nominal-relative target; comparator-advantage training is not implemented | analytical identity; distinct estimand |
| Eqs. (32)–(33), pp. 12–13; Table S23 | Expected and branch-conditional cost accounting | Prediction traces record executed/deployed/qualification work and wall time; repeated checkpoint timing is available | ledger executed; all-operation upper bounds, our original GH200 timing and the observed-max campaign are not part of the release |

### Supplementary algorithms

| Anchor | Implementation | Status |
|---|---|---|
| Algorithm S1 (offline qualification, selection, confirmation, reporting) | `Pipeline.run` over `prepare → train → explore → fit → select → confirm → admit → evaluate`, each stage sealed by a hash-chained manifest (`provenance.py`) | executed; `tests/test_provenance.py` |
| Algorithm S2, supplement pp. 6–7 | `execute_policy`: support, observer validity, recovery priority and label-free anchors; frozen benchmark identity supplies compatibility | core loop executed; general novel-input gate outside |
| Algorithm S3 (budget-aware rule `q_B`) | outside (D15) | outside |
| Algorithm S4 (nested comparator reconstruction inside every max-t resample) | `statistics.nested_selection` | executed |

## 2. Action registry

Source: Supplementary Tables S6/S6b, pp. 4–5.

The executable registry is [`core.ACTIONS`](../cross_morse/core.py); `scripts/check_paper_constants.py` reads it
by AST (no import) and verifies it against [`paper/registered_constants.json`](../paper/registered_constants.json).

| ID | Role | ρ | ν | α | Kernel | η, T | Exploration p | Executed range (hops/step) | Identity coefficient |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| A0 | nominal | 0.00 | 0.70 | 0.25 | one step `P` | — | 0.28 | 0.700 | 0.300 |
| A1 | weak local | 0.20 | 0.55 | 0.25 | one step | — | 0.09 | 0.550 | 0.450 |
| A2 | strong local | 0.20 | 0.85 | 0.20 | one step | — | 0.09 | 0.850 | 0.150 |
| A3 | short safe | 0.25 | 0.70 | 0.25 | resolvent | 0.50, 8 | 0.09 | 0.697 | 0.388 |
| A4 | short recovery | 0.30 | 0.65 | 0.35 | resolvent | 0.50, 8 | 0.09 | 0.646 | 0.448 |
| A5 | medium | 0.30 | 0.70 | 0.20 | resolvent | 0.20, 8 | 0.09 | 0.964 | 0.370 |
| A6 | medium recovery | 0.35 | 0.60 | 0.35 | resolvent | 0.20, 8 | 0.09 | 0.864 | 0.470 |
| A7 | long conservative | 0.25 | 0.50 | 0.40 | resolvent | 0.08, 8 | 0.08 | 0.603 | 0.569 |
| A8 | source reset | — | — | — | assign `H_0` | — | 0.05 | 0 | — |
| A9 | last-safe recovery | — | — | — | assign `H_{τ_safe}` | — | 0.05 | 0 | — |

Within the transport primitive, A0–A2 use the one-step kernel `P`, so `ρ` leaves their operator unchanged. At inserted
GPS/linear-GINE sites A0 is a literal bypass; their native A0 is `P`, with mass/hops 1/1. The table's range and identity columns describe the common transport coordinates before source retention. Open-loop schedules canonicalize A9 to A8 before registration and MPC-OL never
queries A9; feedback routes keep the dynamic last-safe anchor and record its depth for exact replay.

## 3. Statistics and data stages (Supplementary Tables S1, S2, S39)

| Stage | Data role | Statistical unit | Implementation |
|---|---|---|---|
| `train` | D_fit (60% of official training; internal checkpoint subset) | — | `training.train`, `audit_training` |
| `explore` | D_explore (20%) | complete trajectories; complete graphs retained across seeds | `exploration.explore` |
| `fit` | logged D_explore targets | trajectory / graph groups (60/20/20 model fit, radius fit, assessment) | `learning.fit_controller`, `calibration_partition` |
| `select` | D_select (10%); D_explore for the entry diagnostic | 32/64 buffered cores (single graph) or complete graphs; 5 outer × 4 inner folds | `_stage_select`, `nested_selection` |
| `confirm` | D_confirm (10%) | same units; Holm family | `_stage_confirm`, `confirmation` |
| `admit` | official validation (D_admit) | descriptive core bootstrap for all four single-graph runs; independent-graph empirical-Bernstein arithmetic for Pascal/COCO/Peptides; Roman campaign separately summarizes ten split effects | `_stage_admit`, `admission` |
| `evaluate` | official test (D_test) | paired whole-unit bootstrap conditional on the fixed ensemble | `_stage_evaluate`, `review.replay_results` |

This is the separated sequence of Table S39, supplement p. 26.
The reported multi-graph confirmation contrasts used the validation split that
also served admission; their admission statistics are descriptive. Table S14b
is an analytical sample-size sensitivity, not an independently evaluated target
for these stages. See [theory and scope](THEORY_AND_SCOPE.md#5-selection-and-admission-are-conditional-population-statements).

## 4. Experiments E1–E20 (Supplementary Table S36)

| ID | Question | Repository coverage |
|---|---|---|
| E1 | Nominal identity (native vs adapter) | executed: `audit_training`, `tests/test_core.py` |
| E2 | Predictability of the feedback policy's sequences from the initial input (`φ_x^seq`, `φ_x^gain`) | outside |
| E3 | Initial-input schedule predictor at the ceiling | outside |
| E4 | Initial-input distillation | outside |
| E5 | No-observation replanning | outside |
| E6 | Selective sensing / budget-aware rule | outside |
| E7 | Spectral specificity (full / deleted / scrambled bands) | partial: Full versus Lean profiles (`sensor.profiles`); band deletion and scrambling are not automated |
| E8 | Observer specificity (raw / EMA / global / conditioned) | outside (only the conditioned observer is implemented) |
| E9 | Search versus model error (exact enumeration, `H ≤ 4`) | outside |
| E10 | Separated data-use sequence | separated label roles executed; original reported-data reuse is not repaired |
| E11 | Dependence sensitivity (cores, buffers) | runnable by editing `data.num_clusters` and `data.buffer_hops` in a copied config |
| E12 | Retraining-allowed alternatives | outside |
| E13 | Attribution of inserted-site transport (three arms) | outside |
| E14 | Comparator on the extended ensemble grid | partial: E ∈ {1, 2, 4, 8, 12} |
| E15 | Selection trace (candidate ledger, `|Q_s|`, replayed selection) | executed: `select/candidate_ledger.csv`, `select/decision.json`, `scripts/reproduce_results.py` |
| E16 | Uncertainty hierarchy | Partial: replay recomputes the implemented fixed-ensemble whole-unit intervals; original seed-by-graph/retraining resampling is not reconstructed |
| E17 | Timing protocol | partial: CUDA-synchronized wall time per prediction, repeated executions via `scripts/evaluate_checkpoints.py` |
| E18 | Spectral displacement audit | outside |
| E19 | Per-cell ESS and maximum-weight audit | Support statistics emitted from common cell weights; the full original experiment workbook and mask-dependent reruns are not part of the release |
| E20 | Full/Lean profiles and inserted-site parameters | Runtime profiles and confirmation contrast executed; original executed sketch/period/profile records are not recovered |

## 5. Evidence and records (Supplementary Tables S24, S38)

| Required identity (Table S24) | Where a run records it |
|---|---|
| `q_0`, `q_N^ctrl`: architecture, preprocessing, external-data status, split and hashes | `config.json`, `prepared/*.json`, `train/model_inventory.json`, `provenance/frozen_models.json` |
| Observer/risk models: schema, folds, propensities, calibration, hash | `fit/<profile>/audit.json`, `calibration_partition.json`, `calibration_assessment.json`, `controller.pt` hash in `fit/frozen_models.json` |
| `q_C`: family, nested selections, frequencies, resources | `select/decision.json::nested_comparator_fold_choices`, `bootstrap_final_comparator_counts`; `select/candidates.json::cost` |
| `q̂_s`, `q_M`: candidate ledger, confirmation contrast, prediction hash | `select/candidate_ledger.csv`, `confirm/paired_contrasts.json`, `confirm/decision.json`, `predictions/<id>.json::prediction_sha256` |
| `q̃_M`: support strata, recovery priorities, hash | `predictions/<id>.json::decisions[]` (`supported_actions`, `origin`, `support_stratum`, `anchor_depth`) |
| `q_D`: admission arithmetic, branch identity, prediction hash | `admit/decision.json`, `policy_lock.json`, `final/results_summary.json::probability_sha256` |
| Figures/tables: source data, generator, environment, commit | `final/test_predictions.npz`, `final/paired_test_units.npz`, `environment.json`, `run_state.json::source_sha256` (`scripts/source_fingerprint.py`) |

The runtime records the principal identities of Table S38 (dataset, backbone, checkpoint/ensemble, policy, execution profile, data stage,
statistical unit, evaluation protocol, score, paired effect, uncertainty, cost and observation path) through `final/results_summary.json` together with the stage decision files; [`RESULTS.md`](RESULTS.md) lists the
field names. A schema or file hash establishes record consistency; it does not independently establish experimental chronology or supply an absent observation.
