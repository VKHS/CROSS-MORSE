from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .hashing import sha256_file, sha256_state_dict


class StageViolation(RuntimeError):
    """Raised when a lifecycle action violates the registered data-use contract."""


@dataclass(frozen=True)
class FrozenModelAudit:
    """Capture the post-D_fit backbone/head identity and verify later stages.

    The backbone and prediction head may be trained during D_fit. Capture this
    audit only after that training is complete and before D_explore begins.
    """

    backbone_reference: str
    head_reference: str

    @classmethod
    def capture(cls, backbone: Any, head: Any) -> "FrozenModelAudit":
        return cls(
            backbone_reference=sha256_state_dict(backbone),
            head_reference=sha256_state_dict(head),
        )

    def verify(self, backbone: Any, head: Any) -> dict[str, str | bool]:
        backbone_after = sha256_state_dict(backbone)
        head_after = sha256_state_dict(head)
        result: dict[str, str | bool] = {
            "backbone_sha256_before": self.backbone_reference,
            "backbone_sha256_after": backbone_after,
            "head_sha256_before": self.head_reference,
            "head_sha256_after": head_after,
            "frozen_during_stage": (
                backbone_after == self.backbone_reference
                and head_after == self.head_reference
            ),
        }
        if not result["frozen_during_stage"]:
            raise StageViolation("backbone or prediction head changed after D_fit")
        return result


class StageLedger:
    """Fail-closed lifecycle state for one dataset execution.

    The confirmed executable policy and comparator must be frozen after
    confirmation and before admission. Admission then freezes the deployment
    decision before the official test stage is entered.
    """

    ORDER = ("fit", "explore", "select", "confirm", "admit", "test")

    def __init__(self) -> None:
        self._completed: list[str] = []
        self._executable_policy_hash: str | None = None
        self._comparator_hash: str | None = None
        self._deployment_decision_hash: str | None = None

    @property
    def completed(self) -> tuple[str, ...]:
        return tuple(self._completed)

    @property
    def frozen_identities(self) -> dict[str, str | None]:
        return {
            "executable_policy_sha256": self._executable_policy_hash,
            "comparator_sha256": self._comparator_hash,
            "deployment_decision_sha256": self._deployment_decision_hash,
        }

    def complete(self, stage: str) -> None:
        if stage not in self.ORDER:
            raise StageViolation(f"unknown lifecycle stage: {stage}")
        expected = self.ORDER[len(self._completed)] if len(self._completed) < len(self.ORDER) else None
        if stage != expected:
            raise StageViolation(f"expected stage {expected!r}, received {stage!r}")
        if stage == "admit" and (self._executable_policy_hash is None or self._comparator_hash is None):
            raise StageViolation("admission requires a policy and comparator frozen after confirmation")
        if stage == "test" and self._deployment_decision_hash is None:
            raise StageViolation("official testing requires a deployment decision frozen after admission")
        self._completed.append(stage)

    def freeze_executable_policy(self, policy_file: str | Path, comparator_file: str | Path) -> None:
        if not self._completed or self._completed[-1] != "confirm":
            raise StageViolation("freeze the executable policy immediately after confirmation and before admission")
        if self._executable_policy_hash is not None or self._comparator_hash is not None:
            raise StageViolation("executable policy and comparator are already frozen")
        self._executable_policy_hash = sha256_file(policy_file)
        self._comparator_hash = sha256_file(comparator_file)

    def freeze_deployment_decision(self, decision_file: str | Path) -> None:
        if not self._completed or self._completed[-1] != "admit":
            raise StageViolation("freeze the deployment decision after admission and before official testing")
        if self._deployment_decision_hash is not None:
            raise StageViolation("deployment decision is already frozen")
        self._deployment_decision_hash = sha256_file(decision_file)

    def assert_pre_final_test_blind(self, accessed_label_roles: Iterable[str]) -> None:
        roles = {str(role).strip().lower() for role in accessed_label_roles}
        if "test" in roles or "d_test" in roles or "dtest" in roles:
            raise StageViolation("official-test labels were accessed before final evaluation")

    def assert_no_post_test_decision(self) -> None:
        if "test" in self._completed:
            raise StageViolation("no policy, threshold, or deployment decision may change after official testing")
