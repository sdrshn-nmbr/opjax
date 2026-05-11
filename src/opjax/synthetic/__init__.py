"""Synthetic environments and verifiers."""

from opjax.synthetic.click import ClickTask, VerificationResult, generate_click_task, solve_click_task, verify_click_output

__all__ = [
    "ClickTask",
    "VerificationResult",
    "generate_click_task",
    "solve_click_task",
    "verify_click_output",
]
