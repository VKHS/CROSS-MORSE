# Recomputed example results

Synthetic software examples; not official benchmark scores or manuscript reproduction.

Every value below is recomputed from the sealed predictions and labels. Selection, confirmation and admission are also replayed. Metric scores are fractions in [0, 1], including AP; loss is in nats.

| Task configuration | Metric | q0 | qN | qC | qD | qD − qC loss | Admission |
|---|---|---:|---:|---:|---:|---:|---|
| ogbn-arxiv | accuracy | 0.156250 | 0.406250 | 0.281250 | 0.281250 | +0.00000000 | comparator retained |
| ogbn-proteins | rocauc | 0.443767 | 0.574073 | 0.466756 | 0.466756 | +0.00000000 | comparator retained |
| ogbn-products | accuracy | 0.156250 | 0.406250 | 0.281250 | 0.281250 | +0.00000000 | comparator retained |
| Roman-empire | accuracy | 0.156250 | 0.156250 | 0.406250 | 0.406250 | +0.00000000 | comparator retained |
| PascalVOC-SP | macro_f1 | 0.220590 | 0.183279 | 0.190955 | 0.190955 | +0.00000000 | comparator retained |
| COCO-SP | macro_f1 | 0.220590 | 0.183279 | 0.190955 | 0.190955 | +0.00000000 | comparator retained |
| Peptides-func | ap | 0.787494 | 0.545922 | 0.561291 | 0.561291 | +0.00000000 | comparator retained |

## How to interpret the examples

Generated fixtures have 4 classes or labels and reduced training budgets. They exercise the seven task paths, split isolation, numerical review and checkpoint replay. They do not estimate performance on the named official datasets.

The paper's measurements are rendered separately by `make paper`. This report does not read those measurements or compare synthetic scores with them.

`metrics.csv` contains all 28 predictor rows, losses, paired effects and source/configuration/unit identities. `replay.json` retains every recomputed decision and verified numeric input. The original probability arrays, splits, checkpoint files and trajectories remain in the input run directories.
