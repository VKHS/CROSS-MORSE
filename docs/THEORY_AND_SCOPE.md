# Theory, assumptions and claim scope

CROSS-MORSE connects transport algebra, state estimation and a staged policy
comparison. These parts support different kinds of conclusions. An exact kernel
identity does not establish predictive improvement, and a computed confidence
limit does not establish the assumptions needed for its coverage.

This guide reads our submitted main manuscript together with its
supplement. The [implementation contract](IMPLEMENTATION_CONTRACT.md)
defines the executed choices, [method map](METHOD_MAPPING.md) identifies the
corresponding functions, and [results guide](RESULTS.md) distinguishes reported
measurements from repository validation and new experiments.

## 1. What kind of statement is being made?

| Statement | What it establishes | Required scope |
|---|---|---|
| Convex transport and mass-corrected resolvent; Lemmas 1–2 | Nonnegative unit row sums and an operator approximation bound | Nonnegative row-stochastic kernels and registered coefficients |
| Deterministic replay; Proposition 1 | The same realized actions and reset anchors reproduce the same trajectory | Same input, frozen plant, execution law and deterministic arithmetic |
| Perturbation and spectral sensitivity; Propositions 2 and 6 | Bounds on state or loss displacement | Specified norms, operator/update constants, references and mode assumptions |
| Action stability and search regret; Propositions 3, 8 and 9 | Conditional action or objective comparisons | Invariant feasible sets/modes and valid score/state/sequence error bounds |
| Finite-family selection; Theorem 1 | Improvement and near-best selection on a simultaneous-coverage event | Coverage over the entire prespecified feasible family |
| Clipped-loss admission; Theorem 2 | A one-sided population statement for a fixed executable policy | Fresh independent admission graphs, bounded paired differences and no prior adaptation to them |
| Ensemble-spend and timing models | Sensitivity to stated cost/ensemble assumptions | The assumptions and units of the particular model |
| Reported task scores, mechanism contrasts and latency | Empirical observations under their recorded protocol | Their actual checkpoint, policy, data role, unit and measurement instrument |

The paper gives this classification in main pp. 6–9
and Supplementary Figure S19, p. 39.

## 2. Exact transport and execution identities

For a nonnegative row-stochastic `P`, the mass-corrected order-`T` resolvent is

$$
\widetilde K_{\eta,T}
=\eta\sum_{t=0}^{T}(1-\eta)^tP^t+(1-\eta)^{T+1}I.
$$

Its coefficients sum to one. Mixing it with `P` and then with `I` preserves
nonnegativity and unit row sums. The omitted infinite tail and its replacement
both have row mass `(1-eta)^(T+1)`, giving

$$
\|K_\eta-\widetilde K_{\eta,T}\|_\infty
\leq 2(1-\eta)^{T+1}.
$$

These are algebraic properties of the transport, implemented in `apply_kernel`
and `apply_transport`. They do not imply that greater mixing reduces the task
loss. Likewise, the polynomial's coefficient of `P^0` is not the complete
graph-dependent diagonal: longer walks can return to their starting node
(main pp. 4–6, Eqs. 9–10 and 21;
supplement pp. 5 and 14).

Nominal identity is a separate interface requirement. The repository compares
the independent native and A0-dispatch paths. At a designated GPS/linear-GINE
native site A0 uses the native mean aggregation; at an inserted site it is a
literal bypass. Replay also preserves the stored depth of every last-safe
reference. Replaying a realized feedback sequence verifies execution; it does
not demonstrate that an initial-input policy could have selected that sequence
without feedback.

## 3. What the spectral descriptor can explain

Let `L_sym` be the symmetric normalized Laplacian and let ideal intervals `I_b`
partition its spectrum. For polynomial transport responses `g_a` and `g_a'`,
define the ideal band power `P_b(H)` and sensitivity `s_aa'b` by

$$
P_b(H)=\|\mathbf 1_{I_b}(L_{\rm sym})H\|_F^2,
\qquad s_{aa'b}=\sup_{\lambda\in I_b}|g_a(\lambda)-g_{a'}(\lambda)|.
$$

Diagonalization gives

$$
\|(g_a-g_{a'})(L_{\rm sym})H\|_F^2
\leq \sum_b s_{aa'b}^2P_b(H).
$$

A Lipschitz native update converts this transport bound into a state-displacement
bound. With a fixed linear head, a further norm factor bounds mean multiclass
cross-entropy displacement. This describes **how much** the next state can move;
the sign of its task-loss change must be learned or measured
(main p. 8, Proposition 6;
supplement p. 20, Note S17).

The complete bound needs more than the retained polynomial-window powers:

- Replacing ideal bands with `G_b` adds window-approximation terms; omitted
  intervals retain their own contribution.
- For `P=D^-1 A`, the band powers must act on `D^(1/2)H`, with the corresponding
  `d_min^(-1/2)` norm factor.
- Different source-retention coefficients or reference states add explicit
  reference terms. Native-update and head constants must match the actual plant.
- A finite grid maximum is a numerical estimate of a continuous supremum unless
  a separate interpolation or extremum argument certifies it.

`MeasurementBank` therefore exposes a **partial transport descriptor**: three
fixed Jackson windows, degree-weighted powers, and grid sensitivities against the
actual next site's nominal transport. Its common, native and inserted baselines
are respectively `0.3I+0.7P`, `P` and `I`. It does not compute all correction terms
above. The descriptor cannot be read as a certified displacement bound, an action
ranking guarantee or a predictive-safety threshold. The paper's E7 and E18
distinguish useful policy information from the existence of the analytical bound.

The source alignment has a more direct identity. With the implemented denominator
regularizer `epsilon`,

$$
\|G_b(H_0-H)\|_F^2
=\|G_bH_0\|_F^2+\|G_bH\|_F^2
-2c_b\big(\|G_bH_0\|_F\|G_bH\|_F+\varepsilon\big).
$$

Keeping the final `epsilon` term makes this equality exact. Alignment quantifies
source-relative displacement in a band; it does not establish that resetting to
the source improves a label-dependent loss.

## 4. Observer validity, support and risk prediction answer different questions

The observer's normalized innovation, bias and whiteness diagnose consistency of
its state model. The critic predicts a one-action **nominal-relative** effect.
Selection compares complete policies with **the selected open-loop comparator**.
Admission uses **clipped unit-wise complete-policy differences**. These are
distinct estimands (main p. 5, Eqs. 12–14).

Support qualification restricts decisions to cells with recorded behavior-policy
overlap. ESS and the maximum normalized weight are calculated from the same
weights. They do not turn repeated graph trajectories into independent graphs,
nor can their ESS simply replace the independent sample count in Proposition 7's
distribution-free regret bound. The latest supplement withdraws its incompatible
dataset-level support summary and requires per-cell recomputation
(supplement p. 5, Table S7;
pp. 20–21, Proposition 7).

The runtime's radii are empirical group-max residual quantiles from held-out
`Dexplore` groups, followed by a separate logged-history assessment. A match
between the behavior action and a selected action provides a diagnostic on those
logged states. It is not a coverage theorem for the deployed closed-loop state
distribution or for all simulated MPC sequences.

Similarly, beam search's theoretical regret bound `2e_H + zeta_H` needs a uniform
error bound over the compared sequences and a bound on search suboptimality.
One-step RMSE and a larger beam do not establish those quantities. The runtime
fits separate transition, stage and terminal models but does not claim the E9
exact-enumeration audit (supplement p. 30, Note S20).

## 5. Selection and admission are conditional population statements

Theorem 1 assumes simultaneous coverage of all feasible complete-policy effects.
On that event, the registered upper-limit selector either returns `qC` or a
qualified improving policy, and confirmation stays within the eligible set plus
`qC`. Reconstructing the comparator inside max-t resampling implements the
specified estimator; it does not prove exact coverage under arbitrary dependence
among graph cores (main p. 6, Theorem 1).

For independent admission graphs, let `D_j` be the clipped loss difference of the
fixed executable candidate and comparator, in `[-B,B]`. Theorem 2 computes

$$
U_a=\overline D+
\sqrt{\frac{2S^2\log(2/\delta_a)}{J}}+
\frac{14B\log(2/\delta_a)}{3(J-1)}.
$$

Under its assumptions, passing `U_a <= -epsilon_a` bounds the **expected clipped
loss difference**. It does not guarantee improvement on every input, an
unclipped-tail bound or noninferiority of every official metric. The metric gate
is a separate comparison. On dependent single-graph cores and overlapping Roman
splits, admission remains descriptive.

The paper explicitly discloses that its reported Pascal, COCO and Peptides
confirmation used the same validation graphs later used for admission. Those
reported admission limits are therefore descriptive. Table S14b preserves point
estimates and rescales standard errors for a different sample size; it is not a
fresh confirmation experiment and does not repair reuse. The repository's
separated roles support a new evaluation, whose formal interpretation still
depends on genuinely fresh admission observations and all earlier policy choices
being fixed (main pp. 6–7;
supplement pp. 26 and 31, Table S39 and Note S26).

The union-bound statement in Corollary 1 inherits these assumptions. Running the
two numerical functions does not by itself establish a 95% lifecycle guarantee.
Mechanism-label Holm tests and final test intervals also have their own units
and error statements; they are not substitutes for the selection/admission
events.

## 6. Resource statements and additional comparators

The optional `qB` rule preserves a comparator-restart reserve before **every**
paid operation through the final prediction. Its hard work bound requires valid
upper bounds for every such operation. The core runtime's model-call budget and
work counters do not implement that certificate. Expected cost, assurance cost
for serially computing both branches, a modelled 99th percentile, an observed
maximum and a hard cap are distinct quantities
(supplement pp. 30–31, Note S22).

The paper's E3 initial-input predictor, E4 distillation, E5 replanning without
observations and E6 selective sensing are additional comparisons beyond the nine
core route families. Their retained-gain and priced-work identities organize
budget-matched comparisons; the ensemble-spend alternative additionally assumes
that spare work can purchase appropriate extra checkpoints. These experiments
and the general serving-time input gate are not automated by this runtime.
Their reported outcomes belong to the paper's evidence record, while newly
executed outcomes require their own predictions, paired uncertainty and cost
measurements (main p. 9, Propositions 10–11;
supplement pp. 26–30, Tables S40 and S46).
