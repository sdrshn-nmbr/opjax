"""Claude-backed repairer for iSFT/RGT data refinement."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

import anthropic

from opjax.isft.loop import RepairRequest, RepairResult


class ClaudeRepairer:
    def __init__(self, *, model: str = "claude-sonnet-4-6", include_image: bool = False) -> None:
        self.client = anthropic.Anthropic()
        self.model = model
        self.include_image = include_image

    def repair(self, request: RepairRequest) -> RepairResult:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": self._content(request)}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        payload = json.loads(text)
        return RepairResult(
            repaired_output=payload["repaired_output"],
            rationale=payload["rationale"],
            strategy=payload["strategy"],
        )

    def _content(self, request: RepairRequest) -> list[dict[str, object]]:
        task = request.task
        prompt = (
            "Repair a failed GUI action for an iSFT dataset. "
            "Return JSON only with keys repaired_output, rationale, strategy. "
            "The repaired_output must be a FunctionGemma-style function call like "
            "<start_function_call>call:click{x:<escape>123<escape>,y:<escape>456<escape>}<end_function_call>.\n\n"
            f"Task prompt: {task.prompt}\n"
            f"Image path: {task.image_path}\n"
            f"Image size: {task.width}x{task.height}\n"
            f"Target center from oracle metadata: ({task.target_x}, {task.target_y})\n"
            f"Target radius: {task.target_radius}\n"
            f"Failed output: {request.failed_output}\n"
            f"Verifier reason: {request.verifier_reason}\n"
        )
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        if self.include_image:
            image_path = Path(task.image_path)
            media_type = mimetypes.guess_type(image_path)[0] or "image/png"
            image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                }
            )
        return content
