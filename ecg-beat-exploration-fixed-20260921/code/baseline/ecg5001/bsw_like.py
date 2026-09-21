"""Deterministic, directional BSW-like beat alignment for the pilot.

This is a transparent dynamic-programming baseline, not the paper's
undisclosed continuous optimiser or Blossom hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly


_BEAT_SAMPLES = 500
_ALIGNMENT_SAMPLES = 100
_BAND_RADIUS = 20
_MAX_SHIFT = 100.0


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """One directional alignment satisfying ``amplitude * source ~= warped``."""

    ok: bool
    failure_code: str | None
    detail: str | None
    amplitude: float
    shift: np.ndarray
    warped_signal: np.ndarray
    residual: np.ndarray
    reconstruction_rmse: float
    distance: float
    monotone: bool
    in_domain: bool


def _failure(code: str, detail: str) -> AlignmentResult:
    empty = np.empty(0, dtype=float)
    return AlignmentResult(False, code, detail, np.nan, empty, empty, empty, np.nan, np.nan, False, False)


def _as_beat(value: object, name: str) -> tuple[np.ndarray | None, AlignmentResult | None]:
    try:
        beat = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None, _failure("invalid_shape", f"{name} is not numeric")
    if beat.ndim != 1 or beat.size != _BEAT_SAMPLES:
        return None, _failure("invalid_shape", f"{name} must contain exactly 500 samples")
    if not np.isfinite(beat).all():
        return None, _failure("nonfinite_signal", f"{name} contains NaN or infinity")
    return beat, None


def _standardize(beat: np.ndarray) -> np.ndarray:
    median = float(np.median(beat))
    iqr = float(np.percentile(beat, 75) - np.percentile(beat, 25))
    return (beat - median) / max(iqr, 1e-8)


def _dynamic_path(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a deterministic DTW path, preferring diagonal/vertical/horizontal."""

    count = source.size
    cost = np.full((count, count), np.inf)
    previous = np.full((count, count), -1, dtype=np.int8)
    cost[0, 0] = (source[0] - target[0]) ** 2
    # Codes preserve the prescribed priority: diagonal, vertical, horizontal.
    moves = ((-1, -1, 0), (-1, 0, 1), (0, -1, 2))
    for i in range(count):
        lo, hi = max(0, i - _BAND_RADIUS), min(count, i + _BAND_RADIUS + 1)
        for j in range(lo, hi):
            if i == 0 and j == 0:
                continue
            best_cost = np.inf
            best_move = -1
            for di, dj, code in moves:
                parent_i, parent_j = i + di, j + dj
                if parent_i >= 0 and parent_j >= 0:
                    candidate = cost[parent_i, parent_j]
                    if candidate < best_cost:
                        best_cost, best_move = candidate, code
            if best_move >= 0:
                cost[i, j] = best_cost + (source[i] - target[j]) ** 2
                previous[i, j] = best_move
    if not np.isfinite(cost[-1, -1]):  # defensive; the diagonal is always in band
        raise ValueError("no alignment path")
    reverse: list[tuple[int, int]] = []
    i = j = count - 1
    while True:
        reverse.append((i, j))
        if i == 0 and j == 0:
            break
        code = int(previous[i, j])
        if code == 0:
            i, j = i - 1, j - 1
        elif code == 1:
            i -= 1
        elif code == 2:
            j -= 1
        else:
            raise ValueError("broken alignment path")
    return np.asarray(reverse[::-1], dtype=int)


def _target_map(path: np.ndarray) -> np.ndarray:
    values = np.full(_ALIGNMENT_SAMPLES, np.nan)
    for source_index in range(_ALIGNMENT_SAMPLES):
        targets = path[path[:, 0] == source_index, 1]
        if targets.size:
            values[source_index] = float(np.mean(targets))
    observed = np.flatnonzero(np.isfinite(values))
    return np.interp(np.arange(_ALIGNMENT_SAMPLES), observed, values[observed])


def diagnostic_distance(amplitude: np.ndarray, shift: np.ndarray) -> float:
    """Return the frozen directional distance using sample standard deviations."""

    r = np.asarray(amplitude, dtype=float)
    s = np.asarray(shift, dtype=float)
    if r.ndim != 1 or s.ndim != 1 or not r.size or not s.size or not np.isfinite(r).all() or not np.isfinite(s).all():
        return np.nan
    r_sd = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
    s_sd = float(np.std(s, ddof=1)) if s.size > 1 else 0.0
    return float(10.0 * (np.max(np.abs(r - 1.0)) + r_sd) + (np.max(np.abs(s)) + s_sd) / 500.0)


def align_pair(source: object, target: object) -> AlignmentResult:
    """Align a 500-sample ``source`` beat directionally to a ``target`` beat."""

    f, source_failure = _as_beat(source, "source")
    if source_failure is not None:
        return source_failure
    g, target_failure = _as_beat(target, "target")
    if target_failure is not None:
        return target_failure
    assert f is not None and g is not None
    centered_source = f - np.mean(f)
    source_energy = float(np.dot(centered_source, centered_source))
    if source_energy < 1e-12:
        return _failure("source_low_energy", "demeaned source energy is below 1e-12 mV^2")

    source_small = resample_poly(f, 1, 5)
    target_small = resample_poly(g, 1, 5)
    try:
        mapping_small = _target_map(_dynamic_path(_standardize(source_small), _standardize(target_small)))
    except ValueError as exc:
        return _failure("alignment_failed", str(exc))
    coordinate = np.arange(_BEAT_SAMPLES, dtype=float)
    # The resampled nodes span both original endpoints: map 0..99 exactly
    # onto 0..499 rather than multiplying by five (which shifts identities).
    mapping = np.interp(
        coordinate,
        np.linspace(0.0, 499.0, _ALIGNMENT_SAMPLES),
        mapping_small * ((_BEAT_SAMPLES - 1) / (_ALIGNMENT_SAMPLES - 1)),
    )
    shift = np.clip(mapping - coordinate, -_MAX_SHIFT, _MAX_SHIFT)
    sample_positions = coordinate + shift
    warped = np.interp(sample_positions, coordinate, g, left=g[0], right=g[-1])
    centered_warped = warped - np.mean(warped)
    regularization = 1e-6 * max(source_energy, 1.0)
    amplitude = float(np.clip(np.dot(centered_source, centered_warped) / (source_energy + regularization), 0.0, 5.0))
    residual = warped - amplitude * f
    reconstruction_rmse = float(np.sqrt(np.mean(residual * residual)) / max(float(np.sqrt(np.mean(f * f))), 1e-8))
    amplitude_vector = np.full(_BEAT_SAMPLES, amplitude)
    monotone = bool(np.all(np.diff(mapping) >= -1e-9))
    in_domain = bool(np.all((mapping >= 0.0) & (mapping <= _BEAT_SAMPLES - 1)))
    finite = bool(np.isfinite(warped).all() and np.isfinite(residual).all() and np.isfinite(reconstruction_rmse))
    if not (monotone and in_domain and finite):
        return AlignmentResult(False, "invalid_alignment", "alignment diagnostics are not finite, monotone, and in-domain", amplitude, shift, warped, residual, reconstruction_rmse, np.nan, monotone, in_domain)
    return AlignmentResult(True, None, None, amplitude, shift, warped, residual, reconstruction_rmse, diagnostic_distance(amplitude_vector, shift), monotone, in_domain)


__all__ = ["AlignmentResult", "align_pair", "diagnostic_distance"]
