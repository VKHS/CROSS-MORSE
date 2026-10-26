# Figures and repository artwork

The method figures are self-contained SVGs with PNG fallbacks. They explain the
implemented control and qualification paths; the numerical results figures have
separate data sources and generators.

## Block diagrams and workflows

[`scripts/draw_diagrams.py`](../../scripts/draw_diagrams.py) generates seven
figures. Blue denotes the frozen predictor, teal denotes label-free estimation,
amber denotes decisions and actions, and gray denotes data or evidence. Color is
supplementary: every block and decision is also labelled.

| Figure | Reading purpose |
|---|---|
| [architecture.svg](architecture.svg) · [PNG](architecture.png) | Follow a feedback episode from hidden state and prediction to measurement, observer and action. Ordinary site execution and direct resets return to the next hidden state. The offline model block supplies fixed parameters. |
| [lifecycle.svg](lifecycle.svg) · [PNG](lifecycle.png) | Follow the eight stages and their separated label roles. The candidate registry is fixed before selection; confirmation fixes the executable candidate; admission selects that candidate or the comparator; the final policy is locked before test evaluation. |
| [control_depth.svg](control_depth.svg) · [PNG](control_depth.png) | Inspect transport, the truncated resolvent, native versus inserted sites, direct resets and all ten registered actions. |
| [policy_families.svg](policy_families.svg) · [PNG](policy_families.png) | Compare the information and timing contracts of six open-loop and three feedback routes. |
| [review_workflow.svg](review_workflow.svg) · [PNG](review_workflow.png) | Distinguish numerical replay, checkpoint reexecution and portable evidence export. |
| [evidence_chain.svg](evidence_chain.svg) · [PNG](evidence_chain.png) | Follow the repository workflow: source identity and registered configurations, one Slurm job on Arrhenius, the sealed run directory, the deposit of run archives (pack, verify, deposit, ingest) and of measurement-record sets (traced value by value to raw logs), the paper build and the checks of `make verify`. |
| [decision_chain.svg](decision_chain.svg) · [PNG](decision_chain.png) | Follow the seven decision statistics of a run (candidate family, nested selection, confirmation map, admission, policy lock, test-stage effect, cost ledger), the sealed file each is read from and the table cell it fills. |

The first two figures provide the overview of the method; the last two provide the
overview of the repository and of the decision statistics. The remaining figures expand one
relationship at a time. Each SVG contains a descriptive title, an accessible
text description and machine-readable metadata for the depicted boundaries.
The [method overview](../METHOD_OVERVIEW.md) supplies the narrative, while the
[implementation contract](../IMPLEMENTATION_CONTRACT.md) and
[manuscript-to-code map](../METHOD_MAPPING.md) specify the full algorithm.

### Sources and scientific scope

The action values, data allocation, observer/MPC settings and native/inserted
site counts come from
[`paper/registered_constants.json`](../../paper/registered_constants.json).
The explanatory diagrams contain no performance measurements and do not depend
on the values in `reported_results.json`. Their execution summaries are grounded
in the following source paths:

| Diagram | Executable source | Paper anchors |
|---|---|---|
| Feedback loop | [`core.py`](../../cross_morse/core.py): plant `step`, `MeasurementBank`; [`learning.py`](../../cross_morse/learning.py): `Controller`; [`policies.py`](../../cross_morse/policies.py): `execute_policy` | Main pp. 4–5, Eqs. 8–17; supplement pp. 6–7, Algorithm S2 |
| Qualification lifecycle | [`pipeline.py`](../../cross_morse/pipeline.py): stage methods; [`statistics.py`](../../cross_morse/statistics.py): selection, confirmation and admission | Main pp. 2–3, Eqs. 3–4; supplement p. 26, Table S39 |
| Transport and actions | [`core.py`](../../cross_morse/core.py): `ACTIONS`, `apply_kernel`, `apply_transport`, `DesignatedSitePlant.step` | Main pp. 4–5, Eqs. 8–10; supplement pp. 4–5, Tables S5b, S6 and S6b |
| Policy families | [`policies.py`](../../cross_morse/policies.py): `make_registry`, `beam_plan`, `execute_policy` | Main p. 5, Eqs. 15–19; supplement pp. 5–8, Tables S9–S11 |
| Evidence flow | [`reproduce_results.py`](../../scripts/reproduce_results.py), [`evaluate_checkpoints.py`](../../scripts/evaluate_checkpoints.py), [`build_review_bundle.py`](../../scripts/build_review_bundle.py), [`verify_review_bundle.py`](../../scripts/verify_review_bundle.py) | Repository evidence tools; see the [reviewer guide](../REVIEWER_GUIDE.md) |
| Evidence chain | [`run_cross_morse.sh`](../../run_cross_morse.sh): submit, snapshot, requeue; [`pipeline.py`](../../cross_morse/pipeline.py): stage manifests; [`archive_run.py`](../../scripts/archive_run.py), [`ingest_run.py`](../../scripts/ingest_run.py), [`build_paper.py`](../../scripts/build_paper.py), [`availability.py`](../../scripts/availability.py), [`records.py`](../../scripts/records.py); `make verify` in the [Makefile](../../Makefile) | [Run archives](../RUN_ARCHIVES.md); [Arrhenius](../ARRHENIUS.md); [paper/README.md](../../paper/README.md) |
| Decision chain | [`statistics.py`](../../cross_morse/statistics.py): `nested_selection`, `confirmation`, `empirical_bernstein`; [`pipeline.py`](../../cross_morse/pipeline.py): policy lock, evaluate; [`paper/build/evidence.py`](../../paper/build/evidence.py) and [`tables.py`](../../paper/build/tables.py): the cells | Main p. 5, Eq. 19; Main p. 6, Eq. 27; Main p. 12, Eq. 32; supplement p. 8, Tables S11–S12; Tables 1–2 |

In the feedback figure, the predictor and fitted controller parameters remain
fixed while hidden states, observer state and recovery anchors evolve. Full
sensing includes cross-spectral coordinates; Lean retains the same measurement
schema with those coordinates masked. Measurements use states and predictions,
without online labels. Open-loop routes keep their registered plan or stopping
rule and do not repeatedly use this feedback loop.

The lifecycle figure depicts the repository's **separated** roles. It does not
assert that the paper's reported experiments followed that sequence: reported
multi-graph confirmation and admission reused validation data, and their
admission bounds are descriptive. Admission is separate from the internal
A8/A9 recovery actions. Its rejection branch returns the locked comparator.
See [Theory and scope](../THEORY_AND_SCOPE.md) for the statistical assumptions.

In the action figure, the table's range and identity columns refer to the common
transport before source retention. GPS and linear-aggregation GINE-style native
sites use `P` for designated A0; their inserted A0 sites are literal identity.
The shared-update plant instead uses the table's nominal transport coordinates.
A8 and A9 bypass the learned update. Open-loop schedules canonicalize A9 to A8,
and MPC-OL excludes dynamic-anchor A9 from consequence-model search.

### Regenerate and check

From the repository root:

```bash
# Standard library: deterministic SVG generation and freshness check.
python -B scripts/draw_diagrams.py --output-dir docs/assets
python -B scripts/draw_diagrams.py --check

# Rasterize PNG fallbacks at twice the SVG dimensions.
python -B scripts/draw_diagrams.py --output-dir docs/assets --png

# Standard library: check SVG bytes and PNG source fingerprints.
python -B scripts/draw_diagrams.py --check --png

# Regenerate just one figure; --figure can be repeated.
python -B scripts/draw_diagrams.py --figure architecture --output-dir docs/assets --png
```

PNG generation uses an installed CairoSVG, Inkscape or Playwright Chromium, in
that order. It does not download a renderer. An explicit `--png` request returns
a nonzero status if rendering fails and preserves the previous PNG. Pillow, if
available, reduces the flat-color PNGs to a compact palette. The optional `plots`
dependencies supply Matplotlib for the numerical figures; CairoSVG is a separate
optional renderer that can be installed with `python -m pip install cairosvg`.

SVG generation uses an embedded glyph-width table and fixed coordinates; the
same source and registry produce identical SVG bytes. The figures use a local
system font stack, so the exact raster appearance can vary with fonts and
renderer. Mathematical subscripts and superscripts have explicit sizes and
baseline offsets. The PNGs carry their generating SVG's SHA-256 fingerprint;
`--check --png` checks that fingerprint, dimensions and PNG structure without
requiring a renderer. This is a source-freshness check, not a claim of identical
pixels across renderers.

The [focused tests](../../tests/test_draw_diagrams.py) check action rows against
the registry, decision boundaries, readable base font sizes, measured text
bounds, SVG accessibility, stale/missing files and PNG export failure handling.
Render and visually inspect the figures after a layout change; geometry checks
cannot substitute for reviewing the finished page.

## Other artwork

These assets are maintained separately from `draw_diagrams.py`:

| Asset | Purpose |
|---|---|
| [hero.svg](hero.svg) · [PNG](hero.png) | Repository heading and conceptual graph motif |
| [method_overview.svg](method_overview.svg) · [PNG](method_overview.png) | Compact feedback route and separate admission decision |
| [reported_results.svg](reported_results.svg) · [PNG](reported_results.png) | Seven-dataset paired loss effects and reported intervals, with cost quantities from the canonical ledger |
| [figure1_lifecycle.png](figure1_lifecycle.png) | Our lifecycle figure reproduced from the submitted manuscript |
| [registry/](registry) | Numerical panels generated by `scripts/plot_registry_figures.py` from the transcribed registry in `paper/` and its stated formulas (comparison views; the paper's figures built from runs are under `paper/generated/`) |

### Regenerate the numerical figures

With the optional plotting dependencies installed:

```bash
python -m pip install '.[plots]'
python -B scripts/plot_paper_summary.py --output-dir docs/assets
python -B scripts/plot_registry_figures.py --output-dir docs/assets/registry
```

The numerical figures read
[reported_results.json](../../paper/reported_results.json) and the registered
constants. Their provenance is distinct from observations produced by a new run.
Our submitted main manuscript,
supplement,
[source locations](../../paper/sources.json) and
[results interpretation](../PAPER_RESULTS.md) accompany them. Some panels are
analytical calculations: in particular, Table S14b is a sample-size sensitivity,
not an executed experiment with separated confirmation and admission data.
See [NOTICE](../../NOTICE) for the distinction between the software license and
our paper and figure copyright.
