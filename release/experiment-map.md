# Experiment map

The executable lifecycle and source-to-paper references are in
[docs/METHOD_MAPPING.md](../docs/METHOD_MAPPING.md). The machine-readable inventory
is [experiment_map.json](experiment_map.json). The eight executable stages and
nine registered policy families support ogbn-arxiv, ogbn-products, ogbn-proteins,
Roman-empire, PascalVOC-SP, COCO-SP and Peptides-func through the shared
[`run_cross_morse.py`](../run_cross_morse.py) entry point.

Each dataset has three configuration profiles: `smoke` uses a generated fixture
for software checks, `pilot` uses official data with reduced computation, and
`full` specifies the registered dataset experiment. Synthetic outputs remain
test-only. The inventory distinguishes task/metric checks, whole-graph isolation,
native and inserted A0 identity, and Roman's ten-mask campaign from completed
benchmark measurements.

A listed implementation check is not a measured benchmark result. An implemented
route is not equivalent to every experimental arm bearing a related name.
In particular learned stopping/MPC-OL do not implement E3 complete-sequence
prediction, E4 distillation or E5 no-observation replanning. qB is not the
model-call limit. See the explicit exclusions in the implementation contract.

The manuscript's registered constants and reported results are transcribed in
[`paper/`](../paper/README.md); `scripts/check_paper_constants.py` verifies the
executable registry and the full profiles against them, and
`scripts/compare_with_manuscript.py` compares a completed run with the reported
values. Neither inserts a reported value into a computed result.
