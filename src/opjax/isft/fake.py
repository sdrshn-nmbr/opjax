"""Deterministic repairer for tests and dry runs."""

from __future__ import annotations

from opjax.isft.loop import RepairRequest, RepairResult
from opjax.synthetic.click import solve_click_task


class FakeClickRepairer:
    def repair(self, request: RepairRequest) -> RepairResult:
        return RepairResult(
            repaired_output=solve_click_task(request.task),
            rationale=(
                f"The verifier reported {request.verifier_reason}. "
                f"The target center is ({request.task.target_x}, {request.task.target_y})."
            ),
            strategy="Click the known target center exactly and emit only the function call.",
        )
