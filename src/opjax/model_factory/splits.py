"""Sealed evaluation split helpers — prevent train/report contamination."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


ALLOWED_SPLITS = ("train", "dev", "sealed", "time_forward")


@dataclass
class SplitManifest:
    """Maps task IDs to disjoint split families."""

    version: int = 1
    train: list[str] = field(default_factory=list)
    dev: list[str] = field(default_factory=list)
    sealed: list[str] = field(default_factory=list)
    time_forward: list[str] = field(default_factory=list)
    deepswe_report_split: list[str] = field(default_factory=list)
    notes: str = ""

    def all_task_ids(self) -> set[str]:
        return set(
            self.train
            + self.dev
            + self.sealed
            + self.time_forward
            + self.deepswe_report_split
        )

    def overlaps(self) -> dict[str, list[str]]:
        buckets = {
            "train": set(self.train),
            "dev": set(self.dev),
            "sealed": set(self.sealed),
            "time_forward": set(self.time_forward),
            "deepswe_report_split": set(self.deepswe_report_split),
        }
        bad: dict[str, list[str]] = {}
        names = list(buckets)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                inter = sorted(buckets[a] & buckets[b])
                if inter:
                    bad[f"{a}|{b}"] = inter
        return bad

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "train": self.train,
            "dev": self.dev,
            "sealed": self.sealed,
            "time_forward": self.time_forward,
            "deepswe_report_split": self.deepswe_report_split,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SplitManifest:
        return cls(
            version=int(data.get("version", 1)),
            train=list(data.get("train", [])),
            dev=list(data.get("dev", [])),
            sealed=list(data.get("sealed", [])),
            time_forward=list(data.get("time_forward", [])),
            deepswe_report_split=list(data.get("deepswe_report_split", [])),
            notes=str(data.get("notes", "")),
        )


def load_split_manifest(path: Path | str) -> SplitManifest:
    return SplitManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_split_manifest(path: Path | str, manifest: SplitManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")


def validate_no_train_on_sealed(
    training_task_ids: list[str] | set[str],
    manifest: SplitManifest,
) -> list[str]:
    """Return contaminated IDs if training set intersects sealed or DeepSWE report."""
    forbidden = set(manifest.sealed) | set(manifest.deepswe_report_split)
    return sorted(set(training_task_ids) & forbidden)


def assert_manifest_disjoint(manifest: SplitManifest) -> None:
    overlaps = manifest.overlaps()
    if overlaps:
        raise ValueError(f"split overlap detected: {overlaps}")
