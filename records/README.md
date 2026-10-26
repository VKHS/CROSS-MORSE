# Measurement records outside the eight-stage run

[Repository home](../README.md) · [Run archives](../docs/RUN_ARCHIVES.md) · [Availability](../AVAILABILITY.md) ·
[Known differences](../paper/KNOWN_DIFFERENCES.md)

The eight-stage runner produces every number of main Tables 1–2 and of Supplementary Tables S4,
S12, S12c, S13–S15 (internal routes), S23 and S41; those enter through run archives
([docs/RUN_ARCHIVES.md](../docs/RUN_ARCHIVES.md)). The supplementary experiments below were run
outside that lifecycle. Their measurements enter the repository here, as record sets that carry
the raw logs each printed value is read from. `scripts/records.py` checks every value against its
raw file and prints the supplementary table from the set.

## Record sets

Defined in [`specs.json`](specs.json); the directory name of a set is its key.

| Set | Manuscript tables | Experiments | Keys | Quantities |
|---|---|---|---|---|
| `baselines` | S15, S15b | E12, E17 | dataset, method, seed | serial cost, controller wall time, GPU-hours, per-seed official score, HPO trials, parameters |
| `factorials` | S16, S16b | — | dataset, cell | log-depth-integrated loss with its 95% interval, official metric, activity, relative time |
| `negative_controls` | S19 | — | control | residual activity, input fallback, worst pre-fallback harm |
| `input_gate` | S23 | — | dataset | input-fallback probability of `g_input`, deployed wall-time overhead, peak GPU and host memory |
| `nominal_time` | S23b | E17 | dataset | per-input nominal execution time `t_nom` |
| `sensor_parameters` | S6c | E17 | dataset | executed sketch width `k`, sensing period `P_s`, residual fraction `f_res` |
| `cross_backbone` | S25 | — | regime, model | paired effect, outer folds selecting the route, outer folds evaluated |
| `ensemble_grid` | S26b | E14 | dataset, ensemble | comparator loss per ensemble size (the table fits `l(E) = l_inf + D_inf / E`) |
| `inserted_site` | S45 | E13 | dataset, arm | paired effect relative to `q_C`, graph-unit standard error |
| `budget_matched` | S46, S46b | E3–E6 | dataset, arm, configuration | paired effect (millinats), expected serial overhead |

## Layout of one set

```text
records/<set>/
├── summary.csv        one measured value per row
├── raw/               the logs the values are read from (training logs, scheduler accounting, timing traces, metric JSON)
├── code/              the scripts, configurations and commit identities that produced them
├── provenance.json    experiment, produced_by, hardware, period, protocol
└── MANIFEST.json      SHA-256 of every file (written by `make records-index`)
```

`summary.csv` has the set's key columns followed by
`quantity,value,unit,source_file,locator`. The shape of three rows of the `baselines` set (angle
brackets mark what the deposited row contains):

```text
dataset,method,seed,quantity,value,unit,source_file,locator
<dataset>,<method>,,gpu_hours,<value>,GPU-h,raw/<scheduler export>.csv,csv:JobName=<job>;GPUHours
<dataset>,<method>,<seed>,score,<value>,%,raw/<metrics>.json,json:/test/score
<dataset>,<method>,,wall_time,<value>,nominal,raw/<timing trace>.log,regex:wall_time_nominal=([0-9.]+)
```

A locator says where in `source_file` the value is:

| Locator | Meaning |
|---|---|
| `json:/a/b/0` | RFC 6901 pointer into a JSON file |
| `csv:column=value;target` | the first row whose `column` equals `value`; the cell in column `target` |
| `regex:pattern` | the first match of a regular expression with one capture group in a text file |

The value in `summary.csv` must equal the raw value at the precision it is printed with: a raw
`4.663` may be printed as `4.66`, not as `4.70` or `4.6600`.

## Depositing a set

```bash
mkdir -p records/baselines/raw records/baselines/code
cp <logs, scheduler exports, metric files> records/baselines/raw/
cp <launch scripts, configurations, commit list> records/baselines/code/
$EDITOR records/baselines/summary.csv records/baselines/provenance.json
make records-index        # hashes the set, traces every value, writes records/index.json and AVAILABILITY.md
make records-check        # what CI runs on every push
make records-tables       # paper/generated/official/records/<set>.{tex,md,csv}
```

Remove allocation identifiers and personal paths from logs before depositing; the release audit
rejects NAISS project identifiers anywhere in the tree.

## What the check establishes

| Check | Establishes | Does not establish |
|---|---|---|
| `MANIFEST.json` | the deposited files are the files that were indexed | when or where they were produced |
| locator trace | every printed value appears in a deposited raw file | that the raw file comes from the stated experiment; `provenance.json` and `code/` document that |
| printed table | the supplementary table is a function of the deposited summary | statistical claims beyond the summarised values |
