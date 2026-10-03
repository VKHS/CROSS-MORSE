# Quick start

## 1. Obtain the code

Download or clone the repository. The article and Supplementary Information remain on the journal website; no manuscript TeX or PDF is required to run the code.

## 2. Audit the public tree

```bash
python scripts/release_audit.py --root . --strict
```

This check is intentionally lightweight: it validates release structure, source syntax, privacy, scientific-contract anti-patterns, and seven-dataset entrypoint coverage. It does not train a model or certify the published numerical results.

## 3. Prepare an environment

Use the dependency specification supplied by the scientific implementation. Create environments outside the repository, and keep local dataset, cache, checkpoint, log, and output paths in environment variables or an untracked `.env` file.

A portable cluster template is provided at `release/hpc/cluster.env.example`.

## 4. Review the scientific lifecycle

Read `docs/scientific-contract.md` before launching experiments. In particular, the official test labels must remain inaccessible; the executable policy and comparator are frozen before admission, and the deployment decision is frozen before final testing.

## 5. Locate experiments

```bash
python scripts/list_experiments.py
```

The command lists the implementation work packages and their article/Supplementary Information locations. Dataset-specific run commands are kept next to the corresponding source files and configurations. `release/implementation_map.json` is the canonical seven-dataset entrypoint inventory.

## 6. Record every rerun

Each claim-bearing rerun should emit a JSON manifest conforming to `release/schemas/run_manifest.schema.json`. The manifest records data role, unit identity, code/configuration hashes, frozen-model hashes, policy/comparator hashes, label-access status, evaluator identity, and output hashes.
