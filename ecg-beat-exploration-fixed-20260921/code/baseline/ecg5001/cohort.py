"""Metadata-only PTB-XL cohort construction.

The functions in this module intentionally never open a waveform.  They read
the two PTB-XL metadata tables, normalize diagnostic labels, and create an
audit index that can be passed to later sampling and signal-I/O stages.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_STRICT_LABELS = frozenset({"LVH", "NORM"})
QUALITY_NOTE_FIELDS = (
    "baseline_drift",
    "static_noise",
    "burst_noise",
    "electrodes_problems",
    "extra_beats",
    "pacemaker",
)


def _is_missing(value: Any) -> bool:
    """Return whether *value* is a scalar pandas-style missing value."""

    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, bool) else False


def _present_note(value: Any) -> bool:
    if _is_missing(value):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def parse_scp_codes(value: Any) -> set[str]:
    """Parse a PTB-XL ``scp_codes`` value into its code keys.

    PTB-XL stores this field as a Python-literal dictionary.  Strings are
    parsed exclusively with :func:`ast.literal_eval`; arbitrary expressions
    are therefore never executed.  A mapping is accepted as a convenience for
    callers constructing fixture data, but its keys are normalized in exactly
    the same way.  Empty and malformed values raise ``ValueError`` so the
    cohort builder can retain the source row with a failure code.
    """

    if _is_missing(value) or (isinstance(value, str) and not value.strip()):
        raise ValueError("scp_codes is empty")

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError) as exc:
            raise ValueError("scp_codes is malformed") from exc
    elif isinstance(value, Mapping):
        parsed = value
    else:
        raise ValueError("scp_codes is malformed")

    if not isinstance(parsed, Mapping):
        raise ValueError("scp_codes is malformed")
    if not parsed:
        raise ValueError("scp_codes is empty")

    codes: set[str] = set()
    for key in parsed:
        if isinstance(key, str):
            code = key.strip()
            if code:
                codes.add(code)
    if not codes:
        raise ValueError("scp_codes is empty")
    return codes


def _diagnostic_code_set(statements: pd.DataFrame) -> set[str]:
    if not isinstance(statements, pd.DataFrame):
        raise TypeError("scp_statements must be a pandas DataFrame")

    frame = statements
    # The distributed CSV has ``code`` in the first (index) column.  Fixtures
    # may instead provide it as an ordinary column, so support both forms.
    if "code" in frame.columns:
        code_values = frame["code"]
    else:
        code_values = frame.index
    if "diagnostic" not in frame.columns:
        raise ValueError("scp_statements must contain a diagnostic column")

    codes: set[str] = set()
    for code, diagnostic in zip(code_values, frame["diagnostic"], strict=False):
        if _is_diagnostic(diagnostic) and not _is_missing(code):
            text = str(code).strip()
            if text:
                codes.add(text)
    return codes


def _is_diagnostic(value: Any) -> bool:
    if _is_missing(value):
        return False
    if isinstance(value, bool):
        return value
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return str(value).strip() == "1"


def _failure_for_parse(value: Any) -> str:
    if _is_missing(value) or (isinstance(value, str) and not value.strip()):
        return "empty_scp_codes"
    return "malformed_scp_codes"


def _normalize_fold(value: Any) -> int:
    if _is_missing(value):
        raise ValueError("strat_fold must be an integer in the range 1 through 10")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("strat_fold must be an integer in the range 1 through 10") from exc
    if not numeric.is_integer() or not 1 <= int(numeric) <= 10:
        raise ValueError("strat_fold must be an integer in the range 1 through 10")
    return int(numeric)


def _split_for_fold(
    fold: int,
    train_folds: tuple[int, ...],
    validation_fold: int,
    test_fold: int,
) -> str:
    if fold in train_folds:
        return "train"
    if fold == validation_fold:
        return "validation"
    if fold == test_fold:
        return "test"
    raise ValueError(f"fold {fold} is not assigned to train, validation, or test")


def _safe_record_paths(data_root: Path, filename_hr: Any) -> tuple[str | None, str | None, bool, bool, str]:
    """Resolve record sidecars while guaranteeing containment under ``data_root``."""

    if _is_missing(filename_hr) or not str(filename_hr).strip():
        return None, None, False, False, "missing_filename"

    raw = Path(str(filename_hr).strip()).expanduser()
    candidate = raw if raw.is_absolute() else data_root / raw
    # Resolve lexical ``..`` components and existing symlinks before checking
    # containment.  This catches both traversal strings and symlink escapes.
    resolved_base = candidate.resolve(strict=False)
    try:
        resolved_base.relative_to(data_root)
    except ValueError as exc:
        raise ValueError("filename_hr path must remain under data root") from exc

    hea = resolved_base.with_suffix(".hea")
    dat = resolved_base.with_suffix(".dat")
    for sidecar in (hea, dat):
        try:
            sidecar.relative_to(data_root)
        except ValueError as exc:  # defensive for unusual suffix/path semantics
            raise ValueError("resolved waveform path must remain under data root") from exc
    hea_exists = hea.is_file()
    dat_exists = dat.is_file()
    if hea_exists and dat_exists:
        status = "present"
    elif hea_exists:
        status = "missing_dat"
    elif dat_exists:
        status = "missing_hea"
    else:
        status = "missing_both"
    return str(hea), str(dat), hea_exists, dat_exists, status


def _age_fields(value: Any) -> tuple[float | int | None, float | int | None, bool]:
    """Return ``(age_raw, age_years, age_90_plus)`` with PTB-XL's 300 sentinel."""

    age_raw = None if _is_missing(value) else value
    if age_raw is None:
        return None, None, False
    try:
        numeric = float(age_raw)
    except (TypeError, ValueError):
        return age_raw, None, False
    if numeric == 300:
        return age_raw, None, True
    age_years: float | int = int(numeric) if numeric.is_integer() else numeric
    return age_raw, age_years, bool(numeric >= 90)


def _patient_is_missing(value: Any) -> bool:
    return _is_missing(value)


def _validate_frame_columns(db: pd.DataFrame) -> None:
    required = {"ecg_id", "patient_id", "age", "scp_codes", "strat_fold", "filename_hr"}
    missing = sorted(required.difference(db.columns))
    if missing:
        raise ValueError(f"metadata database is missing required columns: {', '.join(missing)}")


def build_cohort_index(
    database: pd.DataFrame,
    statements: pd.DataFrame,
    data_root: str | Path,
    train_folds: Sequence[int] = tuple(range(1, 9)),
    validation_fold: int = 9,
    test_fold: int = 10,
) -> pd.DataFrame:
    """Build the complete metadata audit index.

    Every input metadata row is retained.  Strict labels and sampling flags
    are derived fields; malformed diagnostic strings become row-level metadata
    failures instead of being silently filtered.  The function performs only
    CSV/DataFrame and filesystem metadata operations and never reads a signal
    file's contents.
    """

    if not isinstance(database, pd.DataFrame):
        raise TypeError("ptbxl_database must be a pandas DataFrame")
    _validate_frame_columns(database)
    if database["ecg_id"].isna().any() or database["ecg_id"].duplicated().any():
        raise ValueError("ecg_id must be unique and non-null")

    try:
        normalized_train = tuple(_normalize_fold(fold) for fold in train_folds)
    except TypeError as exc:
        raise ValueError("train_folds must be a sequence of folds") from exc
    if not normalized_train:
        raise ValueError("train_folds must not be empty")
    normalized_validation = _normalize_fold(validation_fold)
    normalized_test = _normalize_fold(test_fold)
    all_folds = (*normalized_train, normalized_validation, normalized_test)
    if len(set(all_folds)) != len(all_folds):
        raise ValueError("fold assignments must not overlap or contain duplicates")

    root = Path(data_root).expanduser().resolve()
    diagnostic_codes = _diagnostic_code_set(statements)

    # Work from records in source order to make the audit index easy to compare
    # with the input CSV and to preserve all source columns untouched.
    rows: list[dict[str, Any]] = []
    for source in database.to_dict(orient="records"):
        row = dict(source)
        fold = _normalize_fold(source.get("strat_fold"))
        row["strat_fold"] = fold
        row["split"] = _split_for_fold(
            fold, normalized_train, normalized_validation, normalized_test
        )

        age_raw, age_years, age_90_plus = _age_fields(source.get("age"))
        row["age_raw"] = age_raw
        row["age_years"] = age_years
        row["age_90_plus"] = age_90_plus

        parsed_codes: set[str] = set()
        failure: str | None = None
        try:
            parsed_codes = parse_scp_codes(source.get("scp_codes"))
        except ValueError:
            failure = _failure_for_parse(source.get("scp_codes"))
        row["diagnostic_codes"] = sorted(parsed_codes.intersection(diagnostic_codes))
        if failure is None and not row["diagnostic_codes"]:
            failure = "no_diagnostic_codes"
        row["metadata_failure_code"] = failure
        code_set = set(row["diagnostic_codes"])
        row["strict_label"] = (
            next(iter(code_set)) if code_set in ({"LVH"}, {"NORM"}) else None
        )

        hea, dat, hea_exists, dat_exists, status = _safe_record_paths(
            root, source.get("filename_hr")
        )
        row["hea_path"] = hea
        row["dat_path"] = dat
        row["hea_exists"] = hea_exists
        row["dat_exists"] = dat_exists
        row["hea_path_exists"] = hea_exists
        row["dat_path_exists"] = dat_exists
        row["safe_hea_path"] = hea
        row["safe_dat_path"] = dat
        row["file_status"] = status

        present_quality = [
            field for field in QUALITY_NOTE_FIELDS if _present_note(source.get(field))
        ]
        row["quality_note_present"] = bool(present_quality)
        row["quality_note_fields"] = present_quality
        row["quality_note_count"] = len(present_quality)
        rows.append(row)

    out = pd.DataFrame(rows)

    # A patient may have many records within one official split, but never
    # across train/validation/test.  Missing patient IDs are intentionally left
    # available for audit and excluded below from sampling eligibility.
    for patient_id, group in out.groupby("patient_id", dropna=True, sort=False):
        if group["split"].nunique(dropna=False) > 1:
            raise ValueError(f"patient {patient_id} crosses official split boundaries")

    labels_by_patient: dict[Any, set[str]] = {}
    for patient_id, group in out.groupby("patient_id", dropna=True, sort=False):
        labels_by_patient[patient_id] = set(group["strict_label"].dropna().tolist())
    conflicts: list[bool] = []
    eligible: list[bool] = []
    for row in out.to_dict(orient="records"):
        patient = row.get("patient_id")
        conflict = bool(
            not _patient_is_missing(patient)
            and len(labels_by_patient.get(patient, set()).intersection(SUPPORTED_STRICT_LABELS)) > 1
        )
        conflicts.append(conflict)
        eligible.append(
            bool(
                row.get("strict_label") in SUPPORTED_STRICT_LABELS
                and row.get("split") == "train"
                and not _patient_is_missing(patient)
                and not conflict
            )
        )
    out["patient_label_conflict"] = conflicts
    out["sampling_eligible"] = eligible

    # Keep common audit columns easy to discover while retaining every source
    # metadata column.  Lists are deliberately kept as Python lists so callers
    # can inspect code membership without reparsing a serialized string.
    preferred = [
        "ecg_id",
        "patient_id",
        "diagnostic_codes",
        "strict_label",
        "strat_fold",
        "split",
        "filename_hr",
        "device",
        "site",
        "sex",
        "age_raw",
        "age_years",
        "age_90_plus",
        *QUALITY_NOTE_FIELDS,
        "quality_note_present",
        "quality_note_fields",
        "quality_note_count",
        "metadata_failure_code",
        "hea_path",
        "dat_path",
        "hea_exists",
        "dat_exists",
        "file_status",
        "patient_label_conflict",
        "sampling_eligible",
    ]
    ordered = [column for column in preferred if column in out.columns]
    ordered.extend(column for column in out.columns if column not in ordered)
    return out.loc[:, ordered]


def _read_statements_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    # If the first column was a genuine ``code`` field, make the index name
    # explicit; build_cohort_index supports either representation.
    if frame.index.name is None:
        frame.index.name = "code"
    return frame


def load_ptbxl_metadata(
    data_root: str | Path,
    *,
    assert_counts: bool = False,
    expected_counts: Mapping[str, int] | None = None,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """Read PTB-XL metadata CSVs and return a complete cohort audit index.

    Frozen real-dataset counts are opt-in because small fixture datasets are a
    first-class use case.  With ``assert_counts=True`` the PTB-XL 1.0.3
    defaults (21,799 rows, 456 LVH, 9,069 NORM) are checked after construction.
    A caller can supply ``expected_counts``/``expected_rows`` for another
    version or a controlled fixture.
    """

    root = Path(data_root).expanduser().resolve()
    database_path = root / "ptbxl_database.csv"
    statements_path = root / "scp_statements.csv"
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    if not statements_path.is_file():
        raise FileNotFoundError(statements_path)

    database = pd.read_csv(database_path)
    statements = _read_statements_csv(statements_path)
    index = build_cohort_index(database, statements, root)

    if expected_rows is not None and len(index) != expected_rows:
        raise AssertionError(f"expected {expected_rows} metadata rows, found {len(index)}")
    if assert_counts:
        if expected_rows is None and len(index) != 21_799:
            raise AssertionError(f"expected 21799 metadata rows, found {len(index)}")
        counts = {"LVH": 456, "NORM": 9_069}
        if expected_counts is not None:
            counts.update({str(key): int(value) for key, value in expected_counts.items()})
    else:
        counts = {} if expected_counts is None else {
            str(key): int(value) for key, value in expected_counts.items()
        }
    for label, expected in counts.items():
        actual = int(index["strict_label"].eq(label).sum())
        if actual != expected:
            raise AssertionError(f"expected {expected} {label} rows, found {actual}")
    return index


__all__ = [
    "QUALITY_NOTE_FIELDS",
    "SUPPORTED_STRICT_LABELS",
    "build_cohort_index",
    "load_ptbxl_metadata",
    "parse_scp_codes",
]
