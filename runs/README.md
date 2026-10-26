# Official-data run archives

This directory receives the verified run archives behind the paper's reported numbers, one
subdirectory per dataset (`arxiv/`, `proteins/`, `products/`, `roman_empire/` with
`split_00` … `split_09`, `pascalvoc_sp/`, `coco_sp/`, `peptides_func/`). The archives are
large and live at their deposit DOI; git tracks only this file and `index.json`.

`index.json` is written exclusively by `scripts/ingest_run.py` (`make ingest-run`), which
verifies the archive's manifest and the sealed stage chain before registering it. The
availability statement and `make paper` read this index; nothing here is edited by hand.

```bash
make ingest-run ARCHIVE=../deposits/arxiv_full.tar.gz DOI=10.5281/zenodo.NNNNNNN
make runs-check          # re-verify every registered archive
make paper               # tables, figures and evidence under paper/generated/official
```

See [docs/RUN_ARCHIVES.md](../docs/RUN_ARCHIVES.md) for the contract.
