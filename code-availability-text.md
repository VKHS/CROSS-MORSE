# Journal code-availability wording

The public repository is intentionally a source-code release and does not mirror the article/Supplementary Information PDFs or claim to include precomputed evidence archives.

Use wording equivalent to the following in the final journal proof:

> **Code availability.** The source code, public configurations, dataset adapters, execution scripts, and validation utilities required to implement and rerun CROSS-MORSE are available in the CROSS-MORSE GitHub repository. The peer-reviewed article and Supplementary Information are hosted by the journal and are not duplicated in the repository. Official datasets must be obtained from their original providers. The GitHub source release does not include precomputed checkpoints, predictions, logged trajectories, or complete experiment-output archives; the reported numerical results and experimental settings are given in the article and Supplementary Information.

If the published article instead states that checkpoints, predictions, trajectories, statistical outputs, figure/table generators, and one-command reconstruction are all available, then a source-code-only repository is not sufficient. In that case either:

1. deposit the promised artifacts in a versioned archival release and link them from the repository; or
2. correct the publication's availability statement.

Do not try to resolve that mismatch by adding empty artifact directories or placeholder manifests.
