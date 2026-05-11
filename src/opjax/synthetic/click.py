"""Synthetic click-target curriculum for Phase 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from opjax.actions import ActionParseError, ClickAction, action_to_dict, format_function_call, parse_action


ClickTier = Literal["target", "distractors", "button"]


@dataclass(frozen=True)
class ClickTask:
    task_id: str
    tier: ClickTier
    prompt: str
    image_path: str
    width: int
    height: int
    target_x: int
    target_y: int
    target_radius: int
    metadata: dict[str, int | str]

    def to_dataset_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    reward: float
    reason: str
    parsed_action: dict[str, object] | None = None
    distance_px: float | None = None


def generate_click_task(
    output_dir: str | Path,
    *,
    seed: int,
    tier: ClickTier = "target",
    width: int = 640,
    height: int = 480,
) -> ClickTask:
    rng = random.Random(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    margin = 48
    target_radius = rng.randint(16, 26)
    target_x = rng.randint(margin, width - margin)
    target_y = rng.randint(margin, height - margin)

    image = Image.new("RGB", (width, height), color=(245, 247, 250))
    draw = ImageDraw.Draw(image)

    metadata: dict[str, int | str] = {"seed": seed}
    if tier == "target":
        _draw_target(draw, target_x, target_y, target_radius, fill=(230, 40, 50), outline=(120, 0, 0))
        prompt = "Click the center of the red target."
    elif tier == "distractors":
        _draw_distractors(draw, rng, width, height, target_x, target_y, count=6)
        _draw_target(draw, target_x, target_y, target_radius, fill=(230, 40, 50), outline=(120, 0, 0))
        prompt = "Click the center of the red target, ignoring the other shapes."
        metadata["distractors"] = 6
    elif tier == "button":
        button_w = rng.randint(110, 180)
        button_h = rng.randint(42, 64)
        left = max(8, min(width - button_w - 8, target_x - button_w // 2))
        top = max(8, min(height - button_h - 8, target_y - button_h // 2))
        target_x = left + button_w // 2
        target_y = top + button_h // 2
        target_radius = min(button_w, button_h) // 2
        draw.rounded_rectangle(
            [left, top, left + button_w, top + button_h],
            radius=12,
            fill=(33, 118, 255),
            outline=(18, 67, 150),
            width=3,
        )
        font = ImageFont.load_default()
        draw.text((left + 18, top + button_h // 2 - 6), "Submit", fill=(255, 255, 255), font=font)
        prompt = "Click the Submit button."
        metadata.update({"button_w": button_w, "button_h": button_h})
    else:
        raise ValueError(f"Unknown click tier: {tier}")

    task_id = f"click-{tier}-{seed}"
    image_file = output_path / f"{task_id}.png"
    image.save(image_file)

    return ClickTask(
        task_id=task_id,
        tier=tier,
        prompt=prompt,
        image_path=str(image_file),
        width=width,
        height=height,
        target_x=target_x,
        target_y=target_y,
        target_radius=target_radius,
        metadata=metadata,
    )


def solve_click_task(task: ClickTask) -> str:
    return format_function_call("click", {"x": task.target_x, "y": task.target_y})


def verify_click_output(task: ClickTask, output: str) -> VerificationResult:
    try:
        action = parse_action(output)
    except ActionParseError as exc:
        return VerificationResult(success=False, reward=0.0, reason=f"parse_error: {exc}")

    if not isinstance(action, ClickAction):
        return VerificationResult(
            success=False,
            reward=0.0,
            reason=f"wrong_action_type: {type(action).__name__}",
            parsed_action=action_to_dict(action),
        )

    if not (0 <= action.x < task.width and 0 <= action.y < task.height):
        return VerificationResult(
            success=False,
            reward=0.0,
            reason="click_out_of_bounds",
            parsed_action=action_to_dict(action),
        )

    distance = math.dist((action.x, action.y), (task.target_x, task.target_y))
    success = distance <= task.target_radius
    return VerificationResult(
        success=success,
        reward=1.0 if success else 0.0,
        reason="success" if success else "missed_target",
        parsed_action=action_to_dict(action),
        distance_px=distance,
    )


def _draw_target(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    radius: int,
    *,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
) -> None:
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill, outline=outline, width=3)
    inner = max(4, radius // 3)
    draw.ellipse([x - inner, y - inner, x + inner, y + inner], fill=(255, 245, 245))


def _draw_distractors(
    draw: ImageDraw.ImageDraw,
    rng: random.Random,
    width: int,
    height: int,
    target_x: int,
    target_y: int,
    *,
    count: int,
) -> None:
    colors = [(40, 160, 90), (250, 185, 40), (120, 90, 230), (20, 170, 220)]
    for _ in range(count):
        x = width // 2
        y = height // 2
        for _attempt in range(50):
            x = rng.randint(32, width - 32)
            y = rng.randint(32, height - 32)
            if math.dist((x, y), (target_x, target_y)) > 80:
                break
        radius = rng.randint(12, 24)
        color = rng.choice(colors)
        if rng.random() < 0.5:
            draw.rectangle([x - radius, y - radius, x + radius, y + radius], fill=color, outline=(40, 40, 40), width=2)
        else:
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=(40, 40, 40), width=2)
