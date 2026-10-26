# Generated registry summary (comparison only)

[`paper_summary.csv`](paper_summary.csv) is a generated seven-dataset view of the transcribed manuscript registry [`paper/reported_results.json`](../paper/reported_results.json); it exists so that run-derived values can be compared with the submitted manuscript, and it is never a source for a paper table. It contains the qN/qC/qD official scores, paired test effects and intervals, comparator/deployment identities, and separately labelled expected, assurance and modelled tail costs that we report in the manuscript, plus the exact source PDF hashes. The paper's tables and figures built from run directories live in [`paper/generated/`](../paper/generated/) (`make example-paper`, `make paper`).

Blank fields mean **not tabulated** or not applicable. In particular q0's absolute score is unavailable, and the products comparator-return identity has no estimated confidence interval. Losses use nats; metric gains use percentage points; Peptides absolute scores use native AP. Modelled C0.99 is not an observed maximum or hard cap. All reported admission is descriptive under the paper's data-use sequence.

```bash
python -B scripts/render_paper_results.py --check
```

Run the same command without `--check` (or `make registry-views`) to regenerate this CSV and [`docs/PAPER_RESULTS.md`](../docs/PAPER_RESULTS.md). The manuscript identities (SHA-256, size, pages) and the table/page provenance are in [`paper/sources.json`](../paper/sources.json). These files contain paper measurements and derived views; they are not output from the separately bundled synthetic software checks. See [`docs/RESULTS.md`](../docs/RESULTS.md) for run replay and explicit manuscript bindings.
