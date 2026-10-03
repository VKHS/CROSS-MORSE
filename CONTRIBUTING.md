# Contributing

Changes must preserve the scientific identity of the released method.

Before opening a pull request:

```bash
python scripts/release_audit.py --root . --strict
python -m unittest discover -s tests -p 'test_*.py'
```

A pull request that changes a dataset split, test-access rule, frozen-parameter rule, comparator, action registry, selection margin, confirmation map, admission gate, endpoint, or evaluator is a **scientific protocol change**. It must update `release/scientific_contract.json`, document the reason, and be reviewed separately from routine software maintenance.

Do not commit datasets, private paths, credentials, journal PDFs, manuscript source, checkpoints, predictions, trajectories, or generated result archives to the source-code branch.
