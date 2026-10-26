# Generated tables, figures and evidence

Everything under this directory is produced by `scripts/build_paper.py` from run directories
and is overwritten by the next build; nothing here is edited by hand.

| Label | Source runs | Command | Status |
|---|---|---|---|
| `examples/` | the seven synthetic example runs under `examples/runs/` | `make example-paper` | committed; CI checks that it equals a fresh build (`make example-paper-check`) |
| `official/` | the official-data run archives registered in `runs/index.json` | `make paper` | written once the archives are ingested (see `docs/STATUS.md`) |

Each label holds `evidence/<stem>.json` (one record per run), `tables/<id>.{tex,md,csv}` and
`figures/<id>.{png,svg}`, plus `BUILD.json` and per-directory `INDEX.json`. Table and figure
identifiers follow the manuscript (`paper/README.md`). Builds from synthetic runs carry the
banner "SYNTHETIC SOFTWARE EXAMPLES" in every file.
