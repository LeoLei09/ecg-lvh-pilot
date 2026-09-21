"""Auditable R-peak, heartbeat-window and ``v_h`` pilot features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from .config import CANONICAL_LEADS


_REQUIRED = ("I", "II", "V4", "V5", "V6", "aVR")
# Engineering-only reject rule, deliberately not a clinical artifact label:
# a heartbeat window is rejected if its absolute high-pass amplitude exceeds
# 20 mV.  This catches acquisition-scale impulses while keeping the decision
# fully reproducible.  The threshold is frozen for the pilot audit trail.
_GROSS_ARTIFACT_MAX_ABS_MV = 20.0


@dataclass(frozen=True, slots=True)
class PeakResult:
    ok: bool
    failure_code: str | None
    detail: str | None
    confidence: float
    peaks: np.ndarray
    filtered: np.ndarray
    composite: np.ndarray


@dataclass(frozen=True, slots=True)
class RRSummary:
    count: int
    median_ms: float | None
    cv: float | None
    rmssd_ms: float | None
    mad_ms: float | None
    iqr_ms: float | None
    variability_available: bool


@dataclass(frozen=True, slots=True)
class VHResult:
    ok: bool
    failure_code: str | None
    detail: str | None
    per_lead_vh: np.ndarray
    v: np.ndarray
    a: np.ndarray
    all_leads_low_vh: bool


@dataclass(frozen=True, slots=True)
class BeatResult:
    ok: bool
    failure_code: str | None
    detail: str | None
    confidence: float
    beats: np.ndarray
    representative: np.ndarray
    valid_count: int
    rr_count: int
    rr: RRSummary
    vh: VHResult | None
    vh_failure_code: str | None
    flags: tuple[str, ...]
    artifact_rejected_count: int
    artifact_rule: str


def _failure_peak(code: str, detail: str) -> PeakResult:
    return PeakResult(False, code, detail, 0.0, np.empty(0, dtype=int), np.empty((0, 0)), np.empty(0))


def _names(lead_names: Iterable[str] | None, count: int) -> tuple[str, ...] | tuple[str, str]:
    if lead_names is None:
        if count == len(CANONICAL_LEADS):
            return CANONICAL_LEADS
        return tuple(f"lead_{i}" for i in range(count))
    if isinstance(lead_names, (str, bytes)):
        return ("lead_name_mismatch", "lead_names must be a sequence")
    try:
        values = tuple(lead_names)
    except TypeError:
        return ("lead_name_mismatch", "lead_names must be a sequence")
    if len(values) != count:
        return ("lead_name_mismatch", "lead_names length does not match signal columns")
    if any(not isinstance(x, str) or not x.strip() for x in values):
        return ("invalid_lead_name", "lead names must be non-empty strings")
    canon = {x.lower(): x for x in _REQUIRED}
    out = tuple(canon.get(x.strip().lower(), x.strip()) for x in values)
    # Unknown repetitions do not obstruct the required-lead diagnostic.  For
    # known leads, duplicates would make the composite ambiguous.
    known = tuple(name for name in out if name in _REQUIRED)
    if len(set(known)) != len(known):
        return ("duplicate_lead_names", "required lead names must be unique")
    return out


def _validate_signal(signal: object, sampling_rate: object, lead_names: Iterable[str] | None):
    try:
        x = np.asarray(signal, dtype=float)
    except (TypeError, ValueError):
        return None, None, _failure_peak("invalid_signal", "signal is not numeric")
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        return None, None, _failure_peak("invalid_shape", "signal must be (samples, leads)")
    if isinstance(sampling_rate, (bool, np.bool_)) or not isinstance(sampling_rate, (int, float, np.integer, np.floating)) or not np.isfinite(sampling_rate) or sampling_rate <= 0:
        return None, None, _failure_peak("invalid_sampling_rate", "sampling_rate must be positive and finite")
    names = _names(lead_names, x.shape[1])
    if isinstance(names, tuple) and len(names) == 2 and names[0] in {"lead_name_mismatch", "invalid_lead_name", "duplicate_lead_names"}:
        return None, None, _failure_peak(names[0], names[1])
    if not all(name in names for name in _REQUIRED):
        return None, None, _failure_peak("required_leads_missing", "I, II, V4, V5, V6 and aVR are required")
    if not np.isfinite(x).all():
        return None, None, _failure_peak("nonfinite_values", "signal contains NaN or infinity")
    return x, tuple(names), None


def _highpass(signal: np.ndarray, fs: float) -> np.ndarray:
    sos = butter(4, 0.5, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal, axis=0)


def detect_r_peaks(signal: np.ndarray, sampling_rate: float, lead_names: Iterable[str] | None = None) -> PeakResult:
    """Detect composite R anchors with fixed, auditable filtering constants."""
    x, names, failure = _validate_signal(signal, sampling_rate, lead_names)
    if failure is not None:
        return failure
    fs = float(sampling_rate)
    # filtfilt needs padding; surface a stable code rather than leaking scipy errors.
    if x.shape[0] < max(64, int(np.ceil(0.5 * fs))):
        return _failure_peak("signal_too_short", "signal is too short for zero-phase filtering")
    try:
        filtered = _highpass(x, fs)
        sos = butter(4, (5.0, 20.0), btype="bandpass", fs=fs, output="sos")
        band = sosfiltfilt(sos, filtered, axis=0)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _failure_peak("signal_too_short", f"filter padding unavailable: {exc}")
    pos = {name: i for i, name in enumerate(names)}
    composite = sum(band[:, pos[name]] for name in ("I", "II", "V4", "V5", "V6")) - band[:, pos["aVR"]]
    derivative = np.diff(composite, prepend=composite[0])
    kernel = max(1, int(round(0.120 * fs)))
    energy = np.convolve(derivative * derivative, np.ones(kernel) / kernel, mode="same")
    med = float(np.median(energy)); mad = float(np.median(np.abs(energy - med)))
    threshold = med + 3 * mad if mad > 0 else med + 0.1 * (float(np.max(energy)) - med)
    candidates, _ = find_peaks(energy, height=threshold, distance=max(1, int(round(0.250 * fs))))
    refined: list[int] = []
    radius = int(round(0.100 * fs))
    for c in candidates:
        lo, hi = max(0, int(c) - radius), min(x.shape[0], int(c) + radius + 1)
        local = lo + int(np.argmax(np.abs(composite[lo:hi])))
        if not refined or local - refined[-1] >= max(1, int(round(0.250 * fs))):
            refined.append(local)
        elif abs(composite[local]) > abs(composite[refined[-1]]):
            refined[-1] = local
    peaks = np.asarray(refined, dtype=int)
    if peaks.size < 5:
        return PeakResult(False, "insufficient_peaks", "fewer than five R peaks", 0.0, peaks, filtered, composite)
    rr = np.diff(peaks) / fs * 1000.0
    valid = (rr >= 300.0) & (rr <= 2000.0)
    confidence = float(np.mean(valid)) if rr.size else 0.0
    if confidence < 0.75:
        return PeakResult(False, "rr_out_of_range", "less than 75% RR intervals in 300–2000 ms", confidence, peaks, filtered, composite)
    return PeakResult(True, None, None, confidence, peaks, filtered, composite)


def summarize_rr(interval_samples: np.ndarray, sampling_rate: float) -> RRSummary:
    values = np.asarray(interval_samples, dtype=float) / float(sampling_rate) * 1000.0
    values = values[np.isfinite(values)]
    count = int(values.size)
    if count == 0:
        return RRSummary(0, None, None, None, None, None, False)
    median = float(np.median(values)); mad = float(np.median(np.abs(values - median)))
    iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
    if count < 2:
        return RRSummary(count, median, None, None, mad, iqr, False)
    sd = float(np.std(values, ddof=1)); cv = sd / median if median != 0 else None
    rmssd = float(np.sqrt(np.mean(np.diff(values) ** 2)))
    return RRSummary(count, median, cv, rmssd, mad, iqr, True)


def ventricular_heterogeneity(beats: np.ndarray) -> VHResult:
    h = np.asarray(beats, dtype=float)
    if h.ndim != 3 or h.shape[0] < 3 or h.shape[2] < 2:
        return VHResult(False, "insufficient_beats", "v_h requires at least three beats and T>=2", np.empty(0), np.empty(0), np.empty(0), False)
    if not np.isfinite(h).all():
        return VHResult(False, "vh_zero_activity", "beats contain NaN or infinity", np.full(h.shape[1], np.nan), np.empty(0), np.empty(0), False)
    n, leads, t = h.shape
    v = np.mean(np.std(h, axis=0, ddof=1), axis=1)
    a = np.mean(np.std(h, axis=2, ddof=1), axis=0)
    vh = np.divide(v, a, out=np.full(leads, np.nan), where=a > 1e-8)
    if np.any(~np.isfinite(vh)):
        return VHResult(False, "vh_zero_activity", "one or more leads have near-zero activity", vh, v, a, False)
    return VHResult(True, None, None, vh, v, a, bool(leads == 12 and np.all(vh < 0.3)))


def extract_beats(signal: np.ndarray, peaks: np.ndarray, sampling_rate: float, lead_names: Iterable[str] | None = None) -> BeatResult:
    x, names, failure = _validate_signal(signal, sampling_rate, lead_names)
    if failure is not None:
        return BeatResult(False, failure.failure_code, failure.detail, 0.0, np.empty((0, 0, 0)), np.empty((0, 0)), 0, 0, summarize_rr(np.array([]), sampling_rate), None, None, (), 0, _artifact_rule())
    if x.shape[0] < 64:
        return BeatResult(False, "signal_too_short", "signal is too short for zero-phase filtering", 0.0, np.empty((0, 0, 0)), np.empty((0, 0)), 0, 0, summarize_rr(np.array([]), sampling_rate), None, None, (), 0, _artifact_rule())
    if isinstance(peaks, (str, bytes)):
        return BeatResult(False, "invalid_peaks", "peaks must be a one-dimensional numeric sequence", 0.0, np.empty((0, 0, 0)), np.empty((0, 0)), 0, 0, summarize_rr(np.array([]), sampling_rate), None, None, (), 0, _artifact_rule())
    try:
        supplied_peaks = np.asarray(peaks, dtype=float)
    except (TypeError, ValueError):
        return BeatResult(False, "invalid_peaks", "peaks cannot be converted to numeric indices", 0.0, np.empty((0, 0, 0)), np.empty((0, 0)), 0, 0, summarize_rr(np.array([]), sampling_rate), None, None, (), 0, _artifact_rule())
    if supplied_peaks.ndim != 1 or not np.isfinite(supplied_peaks).all():
        return BeatResult(False, "invalid_peaks", "peaks must be a finite one-dimensional numeric sequence", np.nan, np.empty((0, 0, 0)), np.empty((0, 0)), 0, 0, summarize_rr(np.array([]), sampling_rate), None, None, (), 0, _artifact_rule())
    if not np.all(supplied_peaks == np.floor(supplied_peaks)):
        return BeatResult(False, "invalid_peaks", "peak indices must be integers", 0.0, np.empty((0, 0, 0)), np.empty((0, 0)), 0, 0, summarize_rr(np.array([]), sampling_rate), None, None, (), 0, _artifact_rule())
    p = supplied_peaks.astype(int)
    if np.any(p < 0) or np.any(p >= x.shape[0]):
        return BeatResult(False, "invalid_peaks", "peak indices are outside signal bounds", 0.0, np.empty((0, 0, 0)), np.empty((0, 0)), 0, 0, summarize_rr(np.array([]), sampling_rate), None, None, (), 0, _artifact_rule())
    if p.ndim != 1 or np.any(np.diff(p) <= 0):
        return BeatResult(False, "invalid_peaks", "peaks must be strictly increasing", 0.0, np.empty((0, 0, 0)), np.empty((0, 0)), 0, 0, summarize_rr(np.array([]), sampling_rate), None, None, (), 0, _artifact_rule())
    if p.size < 3:
        return BeatResult(False, "insufficient_internal_beats", "fewer than three peaks cannot form internal beats", 0.0, np.empty((0, x.shape[1], 500)), np.empty((0, 0)), 0, max(0, p.size - 1), summarize_rr(np.diff(p), sampling_rate), None, None, (), 0, _artifact_rule())
    try:
        filtered = _highpass(x, float(sampling_rate))
    except (ValueError, np.linalg.LinAlgError) as exc:
        return BeatResult(False, "signal_too_short", str(exc), 0.0, np.empty((0, 0, 0)), np.empty((0, 0)), 0, 0, summarize_rr(np.array([]), sampling_rate), None, None, (), 0, _artifact_rule())
    intervals = np.diff(p)
    rr = summarize_rr(intervals, float(sampling_rate))
    in_range = (intervals / float(sampling_rate) * 1000.0 >= 300.0) & (intervals / float(sampling_rate) * 1000.0 <= 2000.0)
    confidence = float(np.mean(in_range)) if in_range.size else 0.0
    if confidence < 0.75:
        return BeatResult(False, "rr_out_of_range", "less than 75% RR intervals in 300–2000 ms", confidence, np.empty((0, x.shape[1], 500)), np.empty((0, 0)), 0, rr.count, rr, None, None, (), 0, _artifact_rule())
    windows = []; artifact_rejected_count = 0
    for i in range(1, p.size - 1):
        if not (in_range[i - 1] and in_range[i]):
            continue
        left = (p[i - 1] + p[i]) / 2.0; right = (p[i] + p[i + 1]) / 2.0
        if left < 0 or right > filtered.shape[0] - 1 or right - left < 2:
            continue
        coords = np.concatenate((np.linspace(left, p[i], 250, endpoint=False), np.linspace(p[i], right, 250, endpoint=False)))
        beat = np.vstack([np.interp(coords, np.arange(filtered.shape[0]), filtered[:, j]) for j in range(filtered.shape[1])])
        if not np.isfinite(beat).all():
            continue
        if np.max(np.abs(beat)) > _GROSS_ARTIFACT_MAX_ABS_MV:
            artifact_rejected_count += 1
            continue
        windows.append(beat)
    beats = np.asarray(windows, dtype=float)
    if beats.shape[0] < 3:
        return BeatResult(False, "insufficient_internal_beats", "fewer than three valid internal beats", confidence, beats.reshape((0, x.shape[1], 500)) if beats.size == 0 else beats, np.empty((0, 0)), int(beats.shape[0]), rr.count, rr, None, None, ("gross_artifact_rejected",) if artifact_rejected_count else (), artifact_rejected_count, _artifact_rule())
    representative = np.mean(beats, axis=0); vh = ventricular_heterogeneity(beats)
    flags = (("gross_artifact_rejected",) if artifact_rejected_count else ()) + (("vh_zero_activity",) if not vh.ok else ())
    return BeatResult(True, None, None, confidence, beats, representative, int(beats.shape[0]), rr.count, rr, vh, vh.failure_code, flags, artifact_rejected_count, _artifact_rule())


def _artifact_rule() -> str:
    return f"max_abs_mv>{_GROSS_ARTIFACT_MAX_ABS_MV:g} on high-pass beat window"


__all__ = ["PeakResult", "RRSummary", "VHResult", "BeatResult", "detect_r_peaks", "extract_beats", "summarize_rr", "ventricular_heterogeneity"]
