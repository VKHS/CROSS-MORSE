# Public release checklist

A release is ready only when all boxes below can be answered yes.

## Contents and scope

- [ ] No manuscript `.tex`, article PDF, supplementary PDF, LaTeX auxiliary, private result archive, or journal production file is tracked.
- [ ] The README states that the journal is the authoritative numerical source.
- [ ] No absent checkpoint, prediction, trajectory, output archive, or one-command paper reconstruction is promised.
- [ ] The final article and supplementary links are set.
- [ ] The license is approved by all rights holders and the institution.

## Implementation

- [ ] Every one of the seven datasets is bound to a real Python entrypoint in `release/implementation_map.json`, with a dataset-specific configuration/README or an explicitly documented generic runner.
- [ ] Python source compiles and public Bash scripts pass syntax checking.
- [ ] Paths, accounts, hosts, allocations, and credentials are environment-driven and untracked.
- [ ] Frozen-backbone and head hashes are checked in claim-bearing runs.
- [ ] Pre-final stages fail if official-test labels or metrics are accessible.

## Protocol

- [ ] `D_fit`, `D_explore`, `D_select`, `D_confirm`, `D_admit`, and `D_test` are represented in code and manifests.
- [ ] Policy selection, confirmation, and admission are complete before the test stage begins.
- [ ] No post-test decision writes a policy/configuration artifact.
- [ ] New run manifests validate against the strict schema.

## Release

- [ ] `python scripts/release_audit.py --root . --strict` passes.
- [ ] Unit tests pass.
- [ ] `RELEASE_MANIFEST.json` is regenerated and verifies.
- [ ] The public ZIP is created only after all blocking checks pass.
