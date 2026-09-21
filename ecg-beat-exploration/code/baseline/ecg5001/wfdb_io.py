"""Small, read-only reader for the PTB-XL WFDB format-16 record subset.

The pilot purposely does not depend on the ``wfdb`` package.  This module is
not a general WFDB implementation: it accepts one interleaved little-endian
format-16 data file, which is the form used by PTB-XL's 500-Hz records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from .config import CANONICAL_LEADS


@dataclass(frozen=True, slots=True)
class WFDBRecord:
    """Outcome of one waveform read; a failure never contains substitute data."""

    signal: np.ndarray | None
    lead_names: tuple[str, ...]
    sampling_rate: float | None
    sample_count: int | None
    ok: bool
    failure_code: str | None = None
    detail: str | None = None
    record_name: str | None = None

    @property
    def signal_mv(self) -> np.ndarray | None:
        """Physical waveform in mV (compatibility-friendly explicit alias)."""

        return self.signal


@dataclass(frozen=True, slots=True)
class WFDBValidation:
    """Result of applying the PTB-XL pilot's fixed shape requirements."""

    ok: bool
    failure_code: str | None = None
    detail: str | None = None


_GAIN = re.compile(
    r"^(?P<gain>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\((?P<baseline>[+-]?\d+)\))?/(?P<unit>[^\s]+)$"
)
_FORMAT = re.compile(r"^(?P<base>\d+)(?P<modifiers>(?:[x:]\d+|[+]\d+)*)$")
_CANONICAL_BY_LOWER = {lead.lower(): lead for lead in CANONICAL_LEADS}
_MAX_DATA_BYTES = 1_000_000_000


class _UnsupportedFormat(ValueError):
    """The base WFDB storage format is outside this reader's subset."""


class _UnsupportedFormatModifier(ValueError):
    """A format-16 modifier would change the sample storage semantics."""


def _failure(
    failure_code: str,
    detail: str,
    *,
    record_name: str | None = None,
    lead_names: tuple[str, ...] = (),
    sampling_rate: float | None = None,
    sample_count: int | None = None,
) -> WFDBRecord:
    return WFDBRecord(
        signal=None,
        lead_names=lead_names,
        sampling_rate=sampling_rate,
        sample_count=sample_count,
        ok=False,
        failure_code=failure_code,
        detail=detail,
        record_name=record_name,
    )


def _safe_resolve(path: Path, root: Path) -> Path | None:
    """Resolve *path* and return it only when it remains inside *root*."""

    candidate = path.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _canonicalize_lead(name: str) -> str:
    return _CANONICAL_BY_LOWER.get(name.strip().lower(), name.strip())


def _parse_first_line(line: str) -> tuple[str, int, float, int]:
    parts = line.split()
    if len(parts) < 4:
        raise ValueError("first header line needs record, lead count, rate, and samples")
    record_name = parts[0]
    if not record_name or any(character in record_name for character in "/\\"):
        raise ValueError("record name is invalid")
    try:
        lead_count = int(parts[1])
        sampling_rate = float(parts[2].split("/", 1)[0])
        sample_count = int(parts[3])
    except ValueError as exc:
        raise ValueError("first header line has invalid numeric fields") from exc
    if lead_count <= 0 or sample_count <= 0 or not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("first header line has non-positive dimensions or rate")
    return record_name, lead_count, sampling_rate, sample_count


def _parse_signal_line(line: str) -> tuple[str, float, int, str, str]:
    parts = line.split()
    # WFDB signal specification has eight mandatory whitespace-delimited
    # fields here (file through lead description); descriptions may add more.
    if len(parts) < 8:
        raise ValueError("signal header line has too few fields")
    filename, format_token, gain_token = parts[:3]
    if not filename or Path(filename).is_absolute() or "\\" in filename:
        raise ValueError("signal filename is invalid")
    format_match = _FORMAT.fullmatch(format_token)
    if format_match is None:
        raise _UnsupportedFormatModifier("format token is invalid or has unsupported modifiers")
    if format_match.group("base") != "16":
        raise _UnsupportedFormat("only WFDB format 16 is supported")
    modifiers = format_match.group("modifiers")
    # ``+byte_offset`` applies before data bytes.  The implementation supports
    # the explicit zero-offset spelling but refuses any offset or xspf/skew
    # modifier instead of silently decoding different storage semantics.
    if modifiers not in ("", "+0"):
        raise _UnsupportedFormatModifier(
            "format-16 modifiers other than explicit +0 are unsupported"
        )
    match = _GAIN.fullmatch(gain_token)
    if match is None:
        raise ValueError("gain/baseline/unit field is invalid")
    gain = float(match.group("gain"))
    baseline = int(match.group("baseline") or 0)
    unit = match.group("unit")
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("gain must be a finite positive number")
    return filename, gain, baseline, unit, _canonicalize_lead(parts[-1])


def _unit_to_mv(unit: str) -> float | None:
    normalized = unit.strip().lower().replace("µ", "u")
    return {"mv": 1.0, "uv": 0.001, "v": 1000.0}.get(normalized)


def load_wfdb_record(
    header_path: str | Path,
    *,
    data_root: str | Path | None = None,
) -> WFDBRecord:
    """Read one safe, little-endian interleaved WFDB format-16 record.

    Errors caused by a malformed record are represented in the returned
    :class:`WFDBRecord`.  Programmer mistakes are intentionally not hidden.
    """

    header = Path(header_path).expanduser()
    root = Path(data_root).expanduser().resolve() if data_root is not None else header.parent.resolve()
    resolved_header = _safe_resolve(header, root)
    if resolved_header is None:
        return _failure("unsafe_path", "header path escapes data_root")
    if not resolved_header.is_file():
        return _failure("missing_header", f"header does not exist: {resolved_header}")

    try:
        lines = resolved_header.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return _failure("header_read_error", str(exc))
    non_comment = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not non_comment:
        return _failure("malformed_header", "header is empty")
    try:
        record_name, lead_count, sampling_rate, sample_count = _parse_first_line(non_comment[0])
    except ValueError as exc:
        return _failure("malformed_header", str(exc))
    if len(non_comment) < lead_count + 1:
        return _failure(
            "malformed_header", "header has fewer signal lines than declared", record_name=record_name,
            sampling_rate=sampling_rate, sample_count=sample_count,
        )

    specs: list[tuple[str, float, int, float, str]] = []
    for line in non_comment[1 : lead_count + 1]:
        try:
            filename, gain, baseline, unit, lead_name = _parse_signal_line(line)
        except _UnsupportedFormat as exc:
            return _failure("unsupported_format", str(exc), record_name=record_name,
                            sampling_rate=sampling_rate, sample_count=sample_count)
        except _UnsupportedFormatModifier as exc:
            return _failure("unsupported_format_modifier", str(exc), record_name=record_name,
                            sampling_rate=sampling_rate, sample_count=sample_count)
        except ValueError as exc:
            return _failure("malformed_header", str(exc), record_name=record_name,
                            sampling_rate=sampling_rate, sample_count=sample_count)
        unit_scale = _unit_to_mv(unit)
        if unit_scale is None:
            return _failure("unsupported_unit", f"cannot convert {unit!r} to mV", record_name=record_name,
                            sampling_rate=sampling_rate, sample_count=sample_count)
        specs.append((filename, gain, baseline, unit_scale, lead_name))

    filenames = {spec[0] for spec in specs}
    if len(filenames) != 1:
        return _failure("unsupported_multifile", "one interleaved data file is required", record_name=record_name,
                        sampling_rate=sampling_rate, sample_count=sample_count)
    data_path = _safe_resolve(resolved_header.parent / next(iter(filenames)), root)
    lead_names = tuple(spec[4] for spec in specs)
    if data_path is None:
        return _failure("unsafe_path", "data path escapes data_root", record_name=record_name, lead_names=lead_names,
                        sampling_rate=sampling_rate, sample_count=sample_count)
    if not data_path.is_file():
        return _failure("missing_data", f"data does not exist: {data_path}", record_name=record_name, lead_names=lead_names,
                        sampling_rate=sampling_rate, sample_count=sample_count)
    # Do this before ``fromfile``: it detects an odd trailing byte and puts a
    # bounded limit on a malicious header's requested allocation.
    expected_values = sample_count * lead_count
    expected_bytes = expected_values * np.dtype("<i2").itemsize
    if expected_bytes > _MAX_DATA_BYTES:
        return _failure("data_size_limit", f"declared data size {expected_bytes} exceeds reader limit", record_name=record_name,
                        lead_names=lead_names, sampling_rate=sampling_rate, sample_count=sample_count)
    try:
        actual_bytes = data_path.stat().st_size
    except OSError as exc:
        return _failure("data_read_error", str(exc), record_name=record_name, lead_names=lead_names,
                        sampling_rate=sampling_rate, sample_count=sample_count)
    if actual_bytes != expected_bytes:
        return _failure("data_size_mismatch", f"expected {expected_bytes} bytes, found {actual_bytes}",
                        record_name=record_name, lead_names=lead_names, sampling_rate=sampling_rate,
                        sample_count=sample_count)
    try:
        digital = np.fromfile(data_path, dtype="<i2")
    except OSError as exc:
        return _failure("data_read_error", str(exc), record_name=record_name, lead_names=lead_names,
                        sampling_rate=sampling_rate, sample_count=sample_count)
    if digital.size != expected_values:
        return _failure("data_size_mismatch", f"expected {expected_values} int16 values, found {digital.size}",
                        record_name=record_name, lead_names=lead_names, sampling_rate=sampling_rate,
                        sample_count=sample_count)
    digital = digital.reshape(sample_count, lead_count)
    if np.any(digital == np.int16(-32768)):
        return _failure("invalid_digital_sentinel", "format-16 invalid-sample sentinel encountered", record_name=record_name,
                        lead_names=lead_names, sampling_rate=sampling_rate, sample_count=sample_count)
    gains = np.asarray([spec[1] for spec in specs], dtype=float)
    baselines = np.asarray([spec[2] for spec in specs], dtype=float)
    unit_scales = np.asarray([spec[3] for spec in specs], dtype=float)
    signal = ((digital.astype(float) - baselines) / gains) * unit_scales
    if not np.isfinite(signal).all():
        return _failure("nonfinite_signal", "physical conversion produced non-finite values", record_name=record_name,
                        lead_names=lead_names, sampling_rate=sampling_rate, sample_count=sample_count)
    if lead_count == len(CANONICAL_LEADS):
        if set(lead_names) != set(CANONICAL_LEADS) or len(set(lead_names)) != len(CANONICAL_LEADS):
            return _failure("noncanonical_leads", "12-lead record must contain each canonical PTB-XL lead exactly once",
                            record_name=record_name, lead_names=lead_names, sampling_rate=sampling_rate,
                            sample_count=sample_count)
        order = np.asarray([lead_names.index(lead) for lead in CANONICAL_LEADS])
        signal = signal[:, order]
        lead_names = CANONICAL_LEADS
    return WFDBRecord(signal, lead_names, sampling_rate, sample_count, True, record_name=record_name)


def validate_ptbxl_500hz_record(record: WFDBRecord) -> WFDBValidation:
    """Validate the fixed 500-Hz PTB-XL waveform shape used by this pilot."""

    if not record.ok:
        return WFDBValidation(False, record.failure_code or "read_failed", record.detail)
    if (
        record.signal is None
        or record.sampling_rate != 500
        or record.sample_count != 5000
        or record.signal.shape != (5000, 12)
        or record.lead_names != CANONICAL_LEADS
    ):
        return WFDBValidation(
            False,
            "unexpected_signal_shape",
            "expected 500 Hz, 5000 samples, and canonical 12-lead PTB-XL shape",
        )
    if not np.isfinite(record.signal).all():
        return WFDBValidation(False, "nonfinite_signal", "signal contains non-finite values")
    return WFDBValidation(True)


__all__ = ["WFDBRecord", "WFDBValidation", "load_wfdb_record", "validate_ptbxl_500hz_record"]
