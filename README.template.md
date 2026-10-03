# CROSS-MORSE

**Cross-Spectral Control of Message Passing in Frozen Graph Neural Networks**

CROSS-MORSE is a post-training control framework for instrumentable, frozen message-passing graph neural networks. The controller observes label-free intermediate measurements, restricts decisions to a supported action registry, and compares adaptive execution with the strongest cross-fitted open-loop policy around the same frozen model.

## Repository scope

This is a **source-code release for independent implementation and rerunning**.

Included:

- implementation source code and dataset adapters supplied by the authors;
- experiment configurations and execution scripts;
- public environment and HPC templates;
- tests, scientific-contract checks, and release/provenance tools;
- a map from implementation work packages to the article and Supplementary Information.

Not included:

- manuscript `.tex` files;
- article or Supplementary Information PDFs;
- journal-owned figure or table layouts;
- precomputed checkpoints, predictions, raw trajectories, or result archives, unless a later tagged release explicitly declares and hashes them.

The article and Supplementary Information are the authoritative source for the reported numerical results. This repository is intentionally structured so readers can obtain the official datasets, configure their own environment, and rerun the staged experiments rather than treating copied numbers as regenerated evidence.

## Publication

- Article: {{ARTICLE_LINK}}
- Supplementary Information: {{SUPPLEMENT_LINK}}
- DOI: {{DOI_TEXT}}

The journal-hosted PDFs are not mirrored in this repository.

## Scientific contract

The public implementation is expected to satisfy these invariants:

1. the task backbone and prediction head are trained before CROSS-MORSE, then kept frozen;
2. `D_fit`, `D_explore`, `D_select`, `D_confirm`, `D_admit`, and `D_test` have distinct decision roles;
3. official test labels are evaluator-only and cannot influence fitting, selection, confirmation, admission, fallback design, or release eligibility;
4. the final policy is frozen before official-test evaluation;
5. online policy decisions use label-free measurements;
6. unsupported inputs and failed deployment gates return the declared comparator branch.

The exact machine-readable version is in [`release/scientific_contract.json`](release/scientific_contract.json). Runtime helpers for parameter hashing and stage manifests are in [`release_tools/`](release_tools/).

## Implementation coverage

{{IMPLEMENTATION_TABLE}}

The generated [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md) records the declared entrypoints. The canonical mapping is [`release/implementation_map.json`](release/implementation_map.json). Mapping establishes source coverage, not successful numerical reproduction.

## Start here

```bash
python scripts/release_audit.py --root . --strict
python scripts/list_experiments.py
```

Then read:

- [`docs/quickstart.md`](docs/quickstart.md)
- [`docs/scientific-contract.md`](docs/scientific-contract.md)
- [`docs/experiment-map.md`](docs/experiment-map.md)
- [`docs/reproducibility.md`](docs/reproducibility.md)
- [`docs/hpc.md`](docs/hpc.md)

Dataset-specific training and execution commands remain next to the corresponding implementation code. Full campaigns are computationally substantial; the supplementary methods report the registered seeds, folds, action registry, model configurations, and resource accounting.

## Results policy

The repository does not maintain a second hand-edited numerical table. Reported outcomes, uncertainty intervals, mechanism tests, ablations, robustness checks, and resource measurements are in the journal article and Supplementary Information. New reruns should write machine-readable run manifests conforming to [`release/schemas/run_manifest.schema.json`](release/schemas/run_manifest.schema.json) and should be compared with the published tables without overwriting them.

## Validation and provenance

```bash
python scripts/release_audit.py --root . --strict
python scripts/create_release_manifest.py --root . --output RELEASE_MANIFEST.json
python scripts/create_release_manifest.py --root . --verify RELEASE_MANIFEST.json
```

The audit fails on publication files, private paths, broken Python or Bash syntax, missing dataset entrypoints, known test-blindness/frozen-model anti-patterns, and documentation that overclaims the contents of this source-only release.

## Citation

Use [`CITATION.cff`](CITATION.cff). Add the final journal DOI before making the repository public.

## License

See [`LICENSE`](LICENSE). The maintainers must confirm that the selected license is approved by all rights holders and their institution before publication.
