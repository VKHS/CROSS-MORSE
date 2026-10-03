# Reproducibility model

## Level 1: source integrity

The release audit checks file presence, syntax, privacy, documentation scope, and known high-risk anti-patterns. The SHA-256 release manifest establishes the identity of the public source tree.

## Level 2: implementation rerun

A reader obtains the official dataset, recreates the declared environment, runs the dataset-specific lifecycle, and records a `run_manifest.json`. This level requires substantial compute and is the primary purpose of the public repository.

## Level 3: numerical comparison

The reader compares rerun outputs with the peer-reviewed article and Supplementary Information. Differences should be reported with exact code/configuration/data identities and with optimization seeds distinguished from independent statistical units.

## What the source branch does not claim

The source branch does not claim to contain every checkpoint, prediction, logged trajectory, failed run, or paper figure source. Such files are often too large or unsuitable for Git and must not be implied by a green source-code CI badge.

A future archival artifact release may be attached to a DOI-backed data repository. It should have a separate manifest, immutable hashes, a documented relationship to a tagged code commit, and an explicit inventory of what can be regenerated.

## Determinism and provenance

Every claim-bearing run should record:

- Git commit or source-tree manifest hash;
- environment identity;
- official dataset version and preprocessing hash;
- split/unit-manifest hashes;
- optimization and resampling seeds;
- frozen backbone/head hashes before and after controller fitting;
- comparator, selected policy, confirmed policy, and executable-policy hashes;
- evaluator and raw-output hashes;
- official-test firewall status.
