"""Minimal iSFT/RGT loop for deterministic Phase 1 tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Protocol

from opjax.synthetic.click import ClickTask, VerificationResult


@dataclass(frozen=True)
class RepairRequest:
    task: ClickTask
    failed_output: str
    verifier_reason: str
    round_index: int


@dataclass(frozen=True)
class RepairResult:
    repaired_output: str
    rationale: str
    strategy: str


class Repairer(Protocol):
    def repair(self, request: RepairRequest) -> RepairResult:
        """Return a repaired action target and compact strategy rationale."""
        ...


@dataclass(frozen=True)
class AttemptRecord:
    output: str
    verification: dict[str, object]


@dataclass(frozen=True)
class ISFTRecord:
    task: dict[str, object]
    initial_output: str
    refined_output: str
    rationale: str
    strategy: str
    attempts: list[AttemptRecord]

    @property
    def sft_target(self) -> str:
        return self.refined_output

    @property
    def rgt_target(self) -> str:
        return f"[THINK] {self.strategy} [/THINK] {self.refined_output}"

    def to_dataset_row(self) -> dict[str, object]:
        row = asdict(self)
        row["sft_target"] = self.sft_target
        row["rgt_target"] = self.rgt_target
        return row


Verifier = Callable[[ClickTask, str], VerificationResult]


def run_isft_rgt(
    task: ClickTask,
    *,
    initial_output: str,
    verifier: Verifier,
    repairer: Repairer,
    max_rounds: int = 3,
) -> ISFTRecord:
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")

    attempts: list[AttemptRecord] = []
    current_output = initial_output
    latest_repair = RepairResult(repaired_output=initial_output, rationale="", strategy="")

    for round_index in range(max_rounds + 1):
        verification = verifier(task, current_output)
        attempts.append(AttemptRecord(output=current_output, verification=asdict(verification)))
        if verification.success:
            return ISFTRecord(
                task=task.to_dataset_row(),
                initial_output=initial_output,
                refined_output=current_output,
                rationale=latest_repair.rationale,
                strategy=latest_repair.strategy or "Emit the function call that clicks the target center.",
                attempts=attempts,
            )

        if round_index == max_rounds:
            raise RuntimeError(f"iSFT failed to repair task {task.task_id}: {verification.reason}")

        latest_repair = repairer.repair(
            RepairRequest(
                task=task,
                failed_output=current_output,
                verifier_reason=verification.reason,
                round_index=round_index,
            )
        )
        current_output = latest_repair.repaired_output

    raise AssertionError("unreachable")
