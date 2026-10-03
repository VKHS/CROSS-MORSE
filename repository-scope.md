# Repository scope

This repository is a **source-code release**, not a manuscript mirror and not a precomputed evidence archive.

## Included

- implementation code supplied by the authors;
- public configuration and execution files;
- dataset preparation instructions that rely on official sources;
- tests and release-integrity tools;
- a machine-readable scientific contract;
- an experiment-to-journal map.

## Excluded by design

- `.tex` manuscript source;
- article and supplementary PDFs;
- journal production files;
- private HPC paths and account details;
- datasets that cannot legally be redistributed;
- generated checkpoints, predictions, trajectories, logs, and result archives unless a future tagged artifact release explicitly includes and hashes them.

## Meaning of reproducibility

The source release supports **independent computational rerunning**: readers obtain the official data, provision compatible hardware/software, execute the registered lifecycle, and compare their outputs with the journal tables.

It does not imply that every published table can be regenerated instantly from small files in the Git checkout. Any future artifact release must be named separately and must include its own immutable manifest.
