# Scientific and provenance contract

Specification: [Implementation contract](../docs/IMPLEMENTATION_CONTRACT.md).
Machine-readable fields are in [scientific_contract.json](scientific_contract.json);
data and experiment scope are in [coverage.json](coverage.json) and
[experiment_map.json](experiment_map.json).

Training captures a frozen-model baseline before exploration. Each subsequent
stage verifies checkpoint and tensor state, artifact hashes and immediate parent
identity. Final evaluation declares eight output files, including the
test-observation record. Sealed policy and decision records are checked before
later stages run. The full validator checks actual trusted model tensors.

Numerical replay checks the sealed observations and reconstructs decisions
without deserializing checkpoints. Checkpoint replay verifies predictions through
label-free model execution. These checks establish consistency within the recorded
workflow; they do not provide an external authentication service or an
independent-graph statistical guarantee.

A manuscript result is linked explicitly to the source, configuration and records
that produced it. Each separate experiment retains its own source identity and
artifacts.
