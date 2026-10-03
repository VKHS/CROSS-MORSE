# Scientific contract

The public implementation must match the method described in the article rather than merely use the same project name.

## Frozen predictor

The backbone and task head are trained before CROSS-MORSE. Controller development may fit observers, critics, sequence models, or policy-selection objects on their declared development stages, but it must not update the learned backbone, native update functions, numerical precision, or original prediction head.

A claim-bearing run should record SHA-256 identities for the backbone and head before and after controller fitting. The two hashes must be identical. `release_tools.guardrails.FrozenModelAudit` provides a framework-neutral helper for PyTorch modules.

## Separated data roles

The registered lifecycle is:

| Role | Allocation/use | Permitted decision |
|---|---|---|
| `D_fit` | 60% of official training, with an internal checkpoint subset | fit the plant/head before freezing; audit entry definitions |
| `D_explore` | 20% of official training | collect logged trajectories; fit observer and consequence models |
| `D_select` | 10% of official training | reconstruct the comparator and select complete policy configurations |
| `D_confirm` | 10% of official training, untouched by fitting | apply the frozen confirmation map without refitting |
| `D_admit` | official validation/calibration, disjoint from checkpointing | compare one fixed executable policy with one fixed comparator |
| `D_test` | official test | evaluator-only final reporting; no further decision |

Dataset-specific implementations may represent units differently, but they must preserve the decision separation and record immutable unit manifests.

## Official-test firewall

Before `D_test`:

- test labels and label-derived metrics are inaccessible;
- no summary consumed by fitting, selection, confirmation, admission, fallback construction, or release eligibility contains test-derived quantities;
- the comparator and final executable policy are frozen and hashed;
- the test stage cannot write or replace a policy-selection artifact;
- no post-test downgrade, promotion, threshold change, or configuration choice is allowed.

## Online information

Online policy decisions use graph structure, features, frozen representations/predictions, action history, and calibrated model state. Labels and label-derived tensors are not valid online inputs.

## Fallback

Static incompatibility, unsupported states/actions, recovery failures, or a failed global admission gate must execute the declared comparator branch. Fallback denominators must identify whether they count graphs, episodes, trajectories, or another unit.

The machine-readable contract is `release/scientific_contract.json`.
