# Migration notes

The upgrader copies the scientific source tree without textual rewriting. This avoids the failure mode in which broad regular expressions alter shell quoting, braces, output filenames, CLI flags, or numerical behavior.

The migration removes only publication formats and known stale repository-alignment files, then overlays public documentation and release tooling. Private paths and unsafe scientific patterns are reported as blocking findings; they are not silently transformed.

After migration, inspect the sibling `*_UPGRADE_REPORT.json`, `*_AUDIT_REPORT.json`, and the generated `IMPLEMENTATION_STATUS.md`, repair every blocking finding in the private source tree, and rerun the upgrader. Do not publish a draft output produced with `--draft`.
