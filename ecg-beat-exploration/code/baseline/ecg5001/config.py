"""Frozen configuration for the ECG 5001 pilot."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


CANONICAL_LEADS = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
_CANONICAL_LEAD_SET = frozenset(CANONICAL_LEADS)


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _default_output_root(data_root: Path) -> Path:
    name = data_root.name or "data"
    return (data_root.parent / f"{name}-ecg5001-output").resolve()


def _is_equal_to_or_below(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _positive_integer(value: Any, field_name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class PilotConfig:
    """Validated, immutable configuration for a single pilot run."""

    data_root: Path
    output_root: Path | None = None
    seed: int = 5001
    train_folds: tuple[int, ...] = tuple(range(1, 9))
    validation_fold: int = 9
    test_fold: int = 10
    per_class_pilot: int = 150
    sampling_rate: int = 500
    heartbeat_length: int = 500
    target_leads: tuple[str, ...] = ("V1", "V5", "V6")

    def __post_init__(self) -> None:
        data_root = _resolved(self.data_root)
        output_root = (
            _default_output_root(data_root)
            if self.output_root is None
            else _resolved(self.output_root)
        )
        if _is_equal_to_or_below(output_root, data_root):
            raise ValueError("output_root must be outside data_root")

        try:
            train_folds = tuple(self.train_folds)
        except TypeError as exc:
            raise ValueError("train_folds must be a sequence of folds") from exc
        all_folds = (*train_folds, self.validation_fold, self.test_fold)
        if not train_folds:
            raise ValueError("train_folds must not be empty")
        if any(type(fold) is not int or not 1 <= fold <= 10 for fold in all_folds):
            raise ValueError("fold values must be integers in the range 1 through 10")
        if len(set(all_folds)) != len(all_folds):
            raise ValueError("fold assignments must not overlap or contain duplicates")

        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        for field_name in ("per_class_pilot", "sampling_rate", "heartbeat_length"):
            _positive_integer(getattr(self, field_name), field_name)

        if isinstance(self.target_leads, (str, bytes)):
            raise ValueError("target_leads must be a non-empty sequence of lead names")
        try:
            target_leads = tuple(self.target_leads)
        except TypeError as exc:
            raise ValueError("target_leads must be a non-empty sequence of lead names") from exc
        if not target_leads:
            raise ValueError("target_leads must not be empty")
        if any(type(lead) is not str or lead not in _CANONICAL_LEAD_SET for lead in target_leads):
            raise ValueError(
                "target_leads must contain only canonical PTB-XL lead names"
            )
        if len(set(target_leads)) != len(target_leads):
            raise ValueError("target_leads must not contain duplicates")

        object.__setattr__(self, "data_root", data_root)
        object.__setattr__(self, "output_root", output_root)
        object.__setattr__(self, "train_folds", train_folds)
        object.__setattr__(self, "target_leads", target_leads)

    @classmethod
    def default(cls, data_root: str | Path) -> PilotConfig:
        """Return the frozen pilot defaults with a sibling output directory."""

        return cls(data_root=data_root)

    @classmethod
    def load_json(cls, path: str | Path) -> PilotConfig:
        """Load configuration, resolving relative roots from the JSON directory."""

        json_path = _resolved(path)
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("configuration JSON must contain an object")
        for field_name in ("data_root", "output_root"):
            value = raw.get(field_name)
            if value is not None:
                value_path = Path(value).expanduser()
                if not value_path.is_absolute():
                    value_path = json_path.parent / value_path
                raw[field_name] = value_path
        return cls(**raw)

    def save_json(self, path: str | Path) -> None:
        """Save configuration as deterministic, human-readable JSON."""

        json_path = _resolved(path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "data_root": str(self.data_root),
            "output_root": str(self.output_root),
            "seed": self.seed,
            "train_folds": list(self.train_folds),
            "validation_fold": self.validation_fold,
            "test_fold": self.test_fold,
            "per_class_pilot": self.per_class_pilot,
            "sampling_rate": self.sampling_rate,
            "heartbeat_length": self.heartbeat_length,
            "target_leads": list(self.target_leads),
        }
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
