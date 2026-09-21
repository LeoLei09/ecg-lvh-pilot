"""Transparent, non-destructive signal-quality indicators for raw ECG traces.

All input values are physical millivolts in a ``(samples, leads)`` array.
Power bands use periodogram bins whose *centres* fall within the documented
closed intervals.  In a 10-second, 100-Hz signal the 0--0.5-Hz band has
0.1-Hz bins; it is therefore a baseline-wander proxy, not a 0.05-Hz measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.signal import periodogram

from .config import CANONICAL_LEADS


_CANONICAL_BY_LOWER = {lead.lower(): lead for lead in CANONICAL_LEADS}
_LIMB_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF")
_FLATLINE_THRESHOLD_MV = 0.001
_ZERO_POWER_THRESHOLD_MV2 = 1e-12


@dataclass(frozen=True, slots=True)
class LeadSQI:
    """Continuous SQI values and per-lead non-fatal status codes."""

    lead_name: str
    amplitude_range_mv: float
    flatline_fraction: float
    boundary_plateau_fraction: float
    abrupt_jump_fraction: float
    baseline_wander_power_ratio: float
    power_49_51_ratio: float
    power_40_100_ratio: float
    total_power_0_100_mv2: float
    failure_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SQIResult:
    """Record-level SQI outcome; quality flags never delete a record."""

    ok: bool
    failure_code: str | None
    detail: str | None
    lead_names: tuple[str, ...]
    lead_metrics: tuple[LeadSQI, ...]
    flags: tuple[str, ...]
    high_correlation_pairs: tuple[tuple[str, str], ...]
    einthoven_residual: float | None
    einthoven_residual_rmse: float | None
    einthoven_normalized_rmse: float | None
    augmented_residuals: tuple[float | None, float | None, float | None]
    augmented_residual_rmse: tuple[float | None, float | None, float | None]
    augmented_normalized_rmse: tuple[float | None, float | None, float | None]
    unavailable_metrics: tuple[str, ...] = ()


def _failed(code: str, detail: str) -> SQIResult:
    return SQIResult(
        ok=False,
        failure_code=code,
        detail=detail,
        lead_names=(),
        lead_metrics=(),
        flags=(code,),
        high_correlation_pairs=(),
        einthoven_residual=None,
        einthoven_residual_rmse=None,
        einthoven_normalized_rmse=None,
        augmented_residuals=(None, None, None),
        augmented_residual_rmse=(None, None, None),
        augmented_normalized_rmse=(None, None, None),
    )


def _canonicalize_lead_names(lead_names: Iterable[str] | None, count: int) -> tuple[str, ...] | SQIResult:
    if lead_names is None:
        return tuple(f"lead_{index}" for index in range(count))
    if isinstance(lead_names, (str, bytes)):
        return _failed("lead_name_mismatch", "lead_names must be a sequence of names")
    try:
        supplied = tuple(lead_names)
    except TypeError:
        return _failed("lead_name_mismatch", "lead_names must be a sequence of names")
    if len(supplied) != count:
        return _failed("lead_name_mismatch", "lead_names length does not match signal columns")
    if any(not isinstance(name, str) or not name.strip() for name in supplied):
        return _failed("invalid_lead_name", "each lead name must be a non-empty string")
    normalized = tuple(_CANONICAL_BY_LOWER.get(name.strip().lower(), name.strip()) for name in supplied)
    if len(set(normalized)) != len(normalized):
        return _failed("duplicate_lead_names", "lead names must be unique after canonicalization")
    return normalized


def _boundary_plateau_fraction(values: np.ndarray) -> float:
    """Fraction in exact min/max runs at least five samples long."""

    if values.size == 0:
        return 0.0
    included = np.zeros(values.size, dtype=bool)
    for boundary_value in (np.min(values), np.max(values)):
        boundary = values == boundary_value
        index = 0
        while index < values.size:
            if not boundary[index]:
                index += 1
                continue
            end = index + 1
            while end < values.size and boundary[end]:
                end += 1
            if end - index >= 5:
                included[index:end] = True
            index = end
    return float(np.mean(included))


def _ratio_for_band(frequencies: np.ndarray, power: np.ndarray, low: float, high: float, total: float) -> float:
    selected = power[(frequencies >= low) & (frequencies <= high)]
    return float(np.sum(selected) / total) if total > 0 else 0.0


def _lead_sqi(values: np.ndarray, sampling_rate: float, lead_name: str) -> LeadSQI:
    amplitude_range = float(np.percentile(values, 99) - np.percentile(values, 1))
    differences = np.abs(np.diff(values))
    if differences.size:
        flatline_fraction = float(np.mean(differences < _FLATLINE_THRESHOLD_MV))
        median_difference = float(np.median(differences))
        mad_difference = float(np.median(np.abs(differences - median_difference)))
        abrupt_jump_fraction = float(np.mean(differences > median_difference + 10 * mad_difference))
    else:
        flatline_fraction = 0.0
        abrupt_jump_fraction = 0.0

    frequencies, density = periodogram(values - np.mean(values), fs=sampling_rate, scaling="density")
    usable = frequencies <= 100.0
    frequencies, density = frequencies[usable], density[usable]
    # Density integrates to power.  Frequency spacing cancels in every ratio,
    # but retaining it makes this stored total explicitly mV^2.
    frequency_step = sampling_rate / values.size
    total_power = float(np.sum(density) * frequency_step)
    failures: tuple[str, ...] = ()
    if total_power <= _ZERO_POWER_THRESHOLD_MV2:
        failures = ("zero_power",)
        baseline_ratio = power_49_51_ratio = power_40_100_ratio = 0.0
    else:
        baseline_ratio = _ratio_for_band(frequencies, density, 0.0, 0.5, float(np.sum(density)))
        power_49_51_ratio = _ratio_for_band(frequencies, density, 49.0, 51.0, float(np.sum(density)))
        power_40_100_ratio = _ratio_for_band(frequencies, density, 40.0, 100.0, float(np.sum(density)))
    return LeadSQI(
        lead_name=lead_name,
        amplitude_range_mv=amplitude_range,
        flatline_fraction=flatline_fraction,
        boundary_plateau_fraction=_boundary_plateau_fraction(values),
        abrupt_jump_fraction=abrupt_jump_fraction,
        baseline_wander_power_ratio=baseline_ratio,
        power_49_51_ratio=power_49_51_ratio,
        power_40_100_ratio=power_40_100_ratio,
        total_power_0_100_mv2=total_power,
        failure_codes=failures,
    )


def _residual_metrics(residual: np.ndarray, scale: float | None) -> tuple[float, float, float | None]:
    mean = float(np.mean(residual))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    return mean, rmse, None if scale is None else rmse / scale


def _limb_residuals(
    signal: np.ndarray, lead_names: tuple[str, ...], lead_metrics: tuple[LeadSQI, ...]
) -> tuple[
    float | None, float | None, float | None,
    tuple[float | None, float | None, float | None],
    tuple[float | None, float | None, float | None],
    tuple[float | None, float | None, float | None],
    tuple[str, ...],
]:
    positions = {name: index for index, name in enumerate(lead_names)}
    ranges = [lead_metrics[positions[name]].amplitude_range_mv for name in _LIMB_LEADS if name in positions]
    scale = max(float(np.median(ranges)), 1e-6) if len(ranges) == len(_LIMB_LEADS) else None
    unavailable: list[str] = []
    if not all(name in positions for name in ("I", "II", "III")):
        einthoven = (None, None, None)
        unavailable.append("einthoven_leads_missing")
    else:
        i, ii, iii = (signal[:, positions[name]] for name in ("I", "II", "III"))
        einthoven = _residual_metrics(ii - i - iii, scale)
    augmented_specs = (
        ("aVR", ("I", "II"), lambda i, ii: -(i + ii) / 2),
        ("aVL", ("I", "II"), lambda i, ii: i - ii / 2),
        ("aVF", ("I", "II"), lambda i, ii: ii - i / 2),
    )
    means: list[float | None] = []
    rmses: list[float | None] = []
    normalized: list[float | None] = []
    for derived, dependencies, expected in augmented_specs:
        required = (derived, *dependencies)
        if not all(name in positions for name in required):
            means.append(None)
            rmses.append(None)
            normalized.append(None)
            unavailable.append(f"{derived}_leads_missing")
            continue
        i, ii = (signal[:, positions[name]] for name in dependencies)
        means.append(_residual_metrics(signal[:, positions[derived]] - expected(i, ii), scale)[0])
        rmses.append(_residual_metrics(signal[:, positions[derived]] - expected(i, ii), scale)[1])
        normalized.append(_residual_metrics(signal[:, positions[derived]] - expected(i, ii), scale)[2])
    return (*einthoven, tuple(means), tuple(rmses), tuple(normalized), tuple(unavailable))


def _correlated_pairs(signal: np.ndarray, lead_names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for left in range(signal.shape[1]):
        for right in range(left + 1, signal.shape[1]):
            if np.std(signal[:, left]) == 0 or np.std(signal[:, right]) == 0:
                continue
            correlation = float(np.corrcoef(signal[:, left], signal[:, right])[0, 1])
            if np.isfinite(correlation) and abs(correlation) > 0.995:
                pairs.append((lead_names[left], lead_names[right]))
    return tuple(pairs)


def compute_sqi(
    signal_mv: np.ndarray,
    sampling_rate: float,
    lead_names: Iterable[str] | None = None,
) -> SQIResult:
    """Calculate transparent SQI on raw physical ECG samples in millivolts.

    Invalid shape/rate/name inputs and non-finite samples return structured
    failures.  Quality findings such as flatline or duplicated leads are kept
    as flags on an otherwise usable result, so callers can review rather than
    silently discard records.
    """

    try:
        signal = np.asarray(signal_mv, dtype=float)
    except (TypeError, ValueError):
        return _failed("invalid_signal", "signal cannot be converted to a numeric array")
    if signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] == 0:
        return _failed("invalid_shape", "signal must have non-zero (samples, leads) shape")
    if (
        isinstance(sampling_rate, (bool, np.bool_))
        or not isinstance(sampling_rate, (int, float, np.integer, np.floating))
        or not np.isfinite(sampling_rate)
        or sampling_rate <= 0
    ):
        return _failed("invalid_sampling_rate", "sampling_rate must be finite and positive")
    names_or_failure = _canonicalize_lead_names(lead_names, signal.shape[1])
    if isinstance(names_or_failure, SQIResult):
        return names_or_failure
    names = names_or_failure
    if not np.isfinite(signal).all():
        return _failed("nonfinite_values", "signal contains NaN or infinity")

    metrics = tuple(_lead_sqi(signal[:, index], float(sampling_rate), name) for index, name in enumerate(names))
    correlations = _correlated_pairs(signal, names)
    (
        einthoven_mean, einthoven_rmse, einthoven_normalized,
        augmented_means, augmented_rmses, augmented_normalized, unavailable,
    ) = _limb_residuals(signal, names, metrics)
    flags: list[str] = []
    if any(metric.amplitude_range_mv < 0.02 for metric in metrics):
        flags.append("low_amplitude_range")
    if any(metric.flatline_fraction > 0.20 for metric in metrics):
        flags.append("excessive_flatline")
    if correlations:
        flags.append("repeated_or_anticorrelated_leads")
    return SQIResult(
        ok=True,
        failure_code=None,
        detail=None,
        lead_names=names,
        lead_metrics=metrics,
        flags=tuple(flags),
        high_correlation_pairs=correlations,
        einthoven_residual=einthoven_mean,
        einthoven_residual_rmse=einthoven_rmse,
        einthoven_normalized_rmse=einthoven_normalized,
        augmented_residuals=augmented_means,
        augmented_residual_rmse=augmented_rmses,
        augmented_normalized_rmse=augmented_normalized,
        unavailable_metrics=unavailable,
    )


def _read_failure_result(record: object) -> SQIResult:
    """Convert an upstream waveform-read outcome into an audit-only SQI result.

    The adapter intentionally uses a structural protocol instead of importing
    :mod:`wfdb_io`, which keeps the two modules acyclic and allows callers to
    pass an equivalent reader result.  A read failure is retained as a row
    with no fabricated signal metrics; it is never converted to a zero trace.
    """

    failure_code = getattr(record, "failure_code", None) or "read_failure"
    failure_code = str(failure_code)
    flags = ("read_failure",) if failure_code == "read_failure" else ("read_failure", failure_code)
    detail = getattr(record, "detail", None)
    if detail is not None:
        detail = str(detail)
    return SQIResult(
        ok=False,
        failure_code=failure_code,
        detail=detail,
        lead_names=tuple(getattr(record, "lead_names", ()) or ()),
        lead_metrics=(),
        flags=flags,
        high_correlation_pairs=(),
        einthoven_residual=None,
        einthoven_residual_rmse=None,
        einthoven_normalized_rmse=None,
        augmented_residuals=(None, None, None),
        augmented_residual_rmse=(None, None, None),
        augmented_normalized_rmse=(None, None, None),
        unavailable_metrics=(failure_code,),
    )


def sqi_from_wfdb(record: object, lead_names: Iterable[str] | None = None) -> SQIResult:
    """Compute SQI from a WFDB-like reader result without hiding read errors.

    ``record`` follows the small structural interface exposed by
    :class:`ecg5001.wfdb_io.WFDBRecord`: ``ok``, ``signal``,
    ``sampling_rate``, ``lead_names``, ``failure_code`` and ``detail``.  A
    failed read returns an audit result carrying the upstream failure code;
    successful reads delegate to :func:`compute_sqi`.
    """

    if not bool(getattr(record, "ok", False)):
        return _read_failure_result(record)
    signal = getattr(record, "signal", None)
    sampling_rate = getattr(record, "sampling_rate", None)
    names = lead_names if lead_names is not None else getattr(record, "lead_names", None)
    if signal is None or sampling_rate is None:
        malformed = _read_failure_result(record)
        return SQIResult(
            ok=False,
            failure_code="read_failure",
            detail="successful reader result lacks signal or sampling_rate",
            lead_names=malformed.lead_names,
            lead_metrics=(),
            flags=("read_failure", "reader_result_invalid"),
            high_correlation_pairs=(),
            einthoven_residual=None,
            einthoven_residual_rmse=None,
            einthoven_normalized_rmse=None,
            augmented_residuals=(None, None, None),
            augmented_residual_rmse=(None, None, None),
            augmented_normalized_rmse=(None, None, None),
            unavailable_metrics=("reader_result_invalid",),
        )
    return compute_sqi(signal, sampling_rate, names)


__all__ = ["LeadSQI", "SQIResult", "compute_sqi", "sqi_from_wfdb"]
