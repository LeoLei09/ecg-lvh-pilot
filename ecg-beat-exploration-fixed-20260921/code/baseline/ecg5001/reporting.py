"""Schema-versioned, atomic artifact writing for the ECG 5001 pilot."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1


def utc_run_stamp() -> str:
    """Return the stable, filesystem-safe UTC portion of a run directory name."""

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def json_value(value: Any) -> Any:
    """Convert numpy/pandas values to the explicit JSON null representation."""

    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return value


def source_manifest(paths: Iterable[Path], root: Path) -> list[dict[str, Any]]:
    """Fingerprint only existing source files; this function never writes source data."""

    rows: list[dict[str, Any]] = []
    for path in sorted({Path(path) for path in paths if Path(path).is_file()}):
        stat = path.stat()
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
        )
    return rows


def runtime_code_hash(root: str | Path, *, include_paths: Sequence[str | Path] = ()) -> str:
    """Hash the complete runtime source tree using stable relative paths.

    The digest intentionally covers every Python module below ``root`` rather
    than only the caller, so changing beat extraction, alignment, cohort
    selection, or reporting code invalidates a previously published run.
    ``include_paths`` adds non-module runtime inputs such as the launcher and
    project configuration.  Their logical names are derived from the common
    ancestor of all inputs, keeping the digest independent of the absolute
    checkout path.
    """

    base = Path(root).expanduser().resolve()
    files = sorted(path for path in base.rglob("*.py") if path.is_file())
    extras = [Path(path).expanduser().resolve() for path in include_paths]
    missing = [path for path in extras if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    all_paths = [*files, *extras]
    # Preserve the original digest naming when no extras are requested; adding
    # explicit files broadens the logical root only for that opt-in case.
    common_root = Path(os.path.commonpath([str(base), *(str(path.parent) for path in extras)])) if extras else base
    digest = sha256()
    for path in sorted(all_paths, key=lambda value: value.as_posix()):
        try:
            logical = path.relative_to(common_root).as_posix()
        except ValueError:
            logical = f"external/{path.name}"
        relative = logical.encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def data_dictionary() -> dict[str, Any]:
    """Return the minimal machine-readable contract shared by every artifact."""

    return {
        "schema_version": SCHEMA_VERSION,
        "null": "JSON null / empty CSV field means unavailable; it is never imputed as zero",
        "primary_keys": {
            "cohort_index.csv": ["ecg_id"],
            "pilot_manifest.csv": ["ecg_id"],
            "record_metrics.csv": ["ecg_id"],
            "bsw_probe_metrics.csv": ["ecg_id", "lead", "reference_label"],
        },
        "units": {"signal": "mV", "rr": "ms", "power": "mV^2"},
        "failure_codes": [
            # Cohort/index failures.
            "empty_scp_codes", "malformed_scp_codes", "no_diagnostic_codes",
            "missing_filename", "unsafe_path",
            # WFDB reader and fixed PTB-XL shape validation failures.
            "missing_header", "header_read_error", "malformed_header",
            "unsupported_format", "unsupported_format_modifier", "unsupported_unit",
            "unsupported_multifile", "missing_data", "data_read_error",
            "data_size_limit", "data_size_mismatch", "invalid_digital_sentinel",
            "nonfinite_signal", "noncanonical_leads", "unexpected_signal_shape",
            # SQI, peak, beat, and v_h failures/flags.
            "read_failure", "reader_result_invalid", "invalid_signal", "invalid_shape",
            "invalid_sampling_rate", "invalid_lead_name", "duplicate_lead_names",
            "lead_name_mismatch", "required_leads_missing", "nonfinite_values",
            "signal_too_short", "insufficient_peaks", "rr_out_of_range",
            "invalid_peaks", "insufficient_internal_beats", "insufficient_beats",
            "vh_zero_activity", "gross_artifact_rejected", "low_amplitude_range",
            "excessive_flatline", "repeated_or_anticorrelated_leads", "zero_power",
            # BSW/probe and report-level statuses.
            "insufficient_prototypes", "probe_beat_unavailable", "alignment_failed",
            "source_low_energy", "invalid_alignment", "not_evaluated",
        ],
        "artifact_fields": {
            "representative_beats.npz": ["schema_version", "ecg_ids", "lead_names", "beats_mv"],
            "pilot_report.md": ["schema_version", "engineering_gate", "not_evaluated"],
            "casebook.pdf": ["schema_version", "deterministic audit casebook"],
        },
    }


class AtomicRunWriter:
    """Write a complete new run to a sibling temp directory and rename once."""

    def __init__(self, output_root: str | Path, config_hash: str) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.run_dir = self.output_root / f"run-{utc_run_stamp()}-{config_hash[:12]}"
        self.temp_dir = self.output_root / f".{self.run_dir.name}-{uuid4().hex}.tmp"

    def __enter__(self) -> "AtomicRunWriter":
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.run_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing run: {self.run_dir}")
        self.temp_dir.mkdir()
        return self

    def write_json(self, name: str, payload: Any) -> None:
        (self.temp_dir / name).write_text(
            json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def write_csv(self, name: str, frame: pd.DataFrame) -> None:
        clean = frame.copy()
        clean.insert(0, "schema_version", SCHEMA_VERSION)
        for column in clean.columns:
            clean[column] = clean[column].map(json_value)
        clean.to_csv(self.temp_dir / name, index=False)

    def write_npz(self, name: str, **arrays: np.ndarray) -> None:
        np.savez_compressed(self.temp_dir / name, **arrays)

    def write_text(self, name: str, content: str) -> None:
        (self.temp_dir / name).write_text(content, encoding="utf-8")

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if exc_type is not None:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            return False
        if self.run_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            raise FileExistsError(f"refusing to overwrite existing run: {self.run_dir}")
        os.replace(self.temp_dir, self.run_dir)
        return False


__all__ = [
    "AtomicRunWriter",
    "SCHEMA_VERSION",
    "data_dictionary",
    "json_value",
    "runtime_code_hash",
    "source_manifest",
    "utc_run_stamp",
]
