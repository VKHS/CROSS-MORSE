# Experiment map

The journal article and Supplementary Information are the authoritative numerical record. This page maps implementation work packages to those locations without copying a second manually maintained results table.

| ID | Implementation work package | Journal location |
|---|---|---|
| `EXP-01` | statistical units and dependence treatment | Supplementary Table S1 |
| `EXP-02` | six-stage split/fold/seed lifecycle | Supplementary Table S2 |
| `EXP-03` | target-regime qualification and failure actions | Supplementary Tables S3-S4; main Table 1 and Figure 2 |
| `EXP-04` | frozen backbone, head, optimizer, checkpoint, and hardware registry | Supplementary Table S5 |
| `EXP-05` | ten-action registry, logged propensities, support, ESS, and weight audit | Supplementary Tables S6-S7 |
| `EXP-06` | online measurements, observer, critic, sequence models, and sensitivity | Supplementary Tables S8-S9 |
| `EXP-07` | implementation-equivalent open-loop and feedback policies | Supplementary Table S10 |
| `EXP-08` | confirmation map and full candidate-selection ledger | Supplementary Tables S11-S12; main Table 2 |
| `EXP-09` | independent-graph clipped-loss admission | Supplementary Table S13; main Table 2 |
| `EXP-10` | held-out state, radius, and replanning mechanism tests | Supplementary Table S14; main Figure 3 |
| `EXP-11` | protocol-matched baselines and resource comparison | Supplementary Table S15; main Table 3 |
| `EXP-12` | equal-budget factorial and within-block ablations | Supplementary Tables S16-S17; main Figure 4 |
| `EXP-13` | support/selection sensitivity and negative controls | Supplementary Tables S18-S19; main Figure 5 |
| `EXP-14` | selection-theorem, action-gap, and trajectory-tube diagnostics | Supplementary Tables S20-S22 |
| `EXP-15` | worst-case and expected compute accounting | Supplementary Table S23; main Figure 4 |
| `EXP-16` | cross-backbone validation of the post-training adapter | Supplementary Table S25 |
| `PROV-01` | claim-to-artifact identity requirements for new reruns | Supplementary Table S24 |

Run `python scripts/list_experiments.py --json` for the machine-readable registry.

A dataset implementation should document which command produces each applicable work package. Where the publication reports a diagnostic rather than a reusable command, the implementation README should describe the generating stage and output schema instead of inventing a one-line reproduction claim.
