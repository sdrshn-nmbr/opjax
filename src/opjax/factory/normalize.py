"""Filter / label / normalize trajectories for training fuel."""

from __future__ import annotations

from dataclasses import dataclass

from opjax.factory.schema import Trajectory


@dataclass
class NormalizeConfig:
    project_allowlist: set[str] | None = None
    min_turns: int = 2
    require_assistant: bool = True
    drop_empty_content: bool = True
    # Paths / markers that must never enter train fuel
    forbidden_substrings: tuple[str, ...] = (
        "/splits/sealed/",
        "deepswe-report",
        "sudarshanbench-sealed",
    )


@dataclass
class NormalizeResult:
    kept: list[Trajectory]
    dropped: int
    reasons: dict[str, int]


def _turn_count(t: Trajectory) -> int:
    return sum(1 for m in t.messages if m.role in {"user", "assistant"})


def _has_forbidden(t: Trajectory, cfg: NormalizeConfig) -> bool:
    blob = " ".join(
        [
            t.trajectory_id,
            t.project or "",
            t.source or "",
            str(t.metadata),
        ]
    ).lower()
    return any(s.lower() in blob for s in cfg.forbidden_substrings)


def label_recovery(t: Trajectory) -> Trajectory:
    """If metadata marks an earlier failure, keep outcome=recovery when later success."""
    meta = dict(t.metadata)
    if t.outcome == "success" and meta.get("had_failure"):
        t.outcome = "recovery"
    return t


def normalize_trajectories(
    trajectories: list[Trajectory],
    cfg: NormalizeConfig | None = None,
) -> NormalizeResult:
    cfg = cfg or NormalizeConfig()
    kept: list[Trajectory] = []
    reasons: dict[str, int] = {}
    dropped = 0

    def bump(reason: str) -> None:
        nonlocal dropped
        dropped += 1
        reasons[reason] = reasons.get(reason, 0) + 1

    for t in trajectories:
        if _has_forbidden(t, cfg):
            bump("forbidden_split_marker")
            continue
        if cfg.project_allowlist is not None:
            if not t.project or t.project not in cfg.project_allowlist:
                bump("project_not_allowlisted")
                continue
        if cfg.drop_empty_content:
            t.messages = [m for m in t.messages if str(m.content).strip()]
        if _turn_count(t) < cfg.min_turns:
            bump("min_turns")
            continue
        if cfg.require_assistant and not any(m.role == "assistant" for m in t.messages):
            bump("no_assistant")
            continue
        kept.append(label_recovery(t))

    return NormalizeResult(kept=kept, dropped=dropped, reasons=reasons)
