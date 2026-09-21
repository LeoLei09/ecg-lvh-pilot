"""Deterministic, patient-disjoint selection for the ECG 5001 pilot.

This module deliberately operates on cohort metadata only.  It neither opens
records nor considers any waveform-derived quality or outcome field.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import math
from typing import Any
import unicodedata

import numpy as np
import pandas as pd


_LABELS = ("LVH", "NORM")
_STRATUM_COLUMNS = ("device", "site", "quality_note_present")
_REQUIRED_COLUMNS = {
    "ecg_id", "patient_id", "strict_label", "strat_fold",
    "patient_label_conflict", "sampling_eligible", *_STRATUM_COLUMNS,
}


def _hash(*parts: object) -> str:
    return sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _normal_stratum_value(value: object) -> str:
    """Canonical metadata value suitable for an orderable stratum key."""

    if _missing(value):
        return "<MISSING>"
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFC", value).strip()
        return normalized if normalized else "<MISSING>"
    return str(value)


def _patient_text(value: object) -> str:
    return unicodedata.normalize("NFC", str(value)).strip()


def _eligible_records(cohort: pd.DataFrame, seed: int) -> pd.DataFrame:
    missing = sorted(_REQUIRED_COLUMNS.difference(cohort.columns))
    if missing:
        raise ValueError("cohort is missing required columns: " + ", ".join(missing))
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    filtered = cohort.loc[
        cohort["strict_label"].isin(_LABELS)
        & cohort["strat_fold"].isin(range(1, 9))
        & cohort["patient_id"].notna()
        & ~cohort["patient_label_conflict"].fillna(True).astype(bool)
        & cohort["sampling_eligible"].fillna(False).astype(bool)
    ].copy()
    filtered["_patient_key"] = filtered["patient_id"].map(_patient_text)
    filtered = filtered.loc[filtered["_patient_key"] != ""].copy()
    filtered["record_hash"] = [
        _hash("record", seed, patient, ecg_id)
        for patient, ecg_id in zip(filtered["_patient_key"], filtered["ecg_id"], strict=True)
    ]
    # The full key makes duplicate/mixed-type source ordering irrelevant.
    filtered["_ecg_key"] = filtered["ecg_id"].map(str)
    filtered = filtered.sort_values(
        ["strict_label", "_patient_key", "record_hash", "_ecg_key"], kind="mergesort"
    )
    return filtered.drop_duplicates(["strict_label", "_patient_key"], keep="first")


def _quotas(groups: dict[tuple[str, str, str], list[dict[str, Any]]], count: int) -> dict[tuple[str, str, str], tuple[int, float]]:
    total = sum(len(rows) for rows in groups.values())
    quotas: dict[tuple[str, str, str], int] = {}
    remainders: dict[tuple[str, str, str], float] = {}
    for stratum, rows in groups.items():
        ideal = count * len(rows) / total
        quotas[stratum] = min(len(rows), math.floor(ideal))
        remainders[stratum] = ideal - math.floor(ideal)

    remaining = count - sum(quotas.values())
    # Standard largest remainder grants at most one residual seat to each
    # stratum, ordered by its remainder and then its lexical tuple.  A further
    # round is only a deterministic capacity redistribution safeguard.
    ordered = sorted(groups, key=lambda key: (-remainders[key], key))
    for key in ordered:
        if not remaining:
            break
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            remaining -= 1
    while remaining:
        available = [key for key in ordered if quotas[key] < len(groups[key])]
        if not available:
            raise AssertionError("stratum capacities cannot satisfy pilot count")
        quotas[available[0]] += 1
        remaining -= 1
    return {key: (quotas[key], remainders[key]) for key in groups}


def select_pilot(cohort: pd.DataFrame, *, per_class: int = 150, seed: int = 5001) -> pd.DataFrame:
    """Select exactly ``per_class`` patients for each strict label.

    The returned record for every patient is fixed by a record-domain hash;
    sample and role domains are independent so changing one decision does not
    silently alter another.  For 150 patients roles are always 100/50; for
    smaller pilots the prototype pool is ``ceil(2*n/3)``.
    """

    if type(per_class) is not int or per_class <= 0:
        raise ValueError("per_class must be a positive integer")
    records = _eligible_records(cohort, seed)
    selected: list[dict[str, Any]] = []

    for label in _LABELS:
        candidates = records.loc[records["strict_label"] == label].copy()
        if len(candidates) < per_class:
            raise ValueError(
                f"insufficient eligible patients for {label}: {len(candidates)} available, {per_class} required"
            )
        candidates["stratum_device"] = candidates["device"].map(_normal_stratum_value)
        candidates["stratum_site"] = candidates["site"].map(_normal_stratum_value)
        candidates["stratum_quality_note_present"] = candidates["quality_note_present"].map(_normal_stratum_value)
        candidates["_stratum_tuple"] = list(zip(
            candidates["stratum_device"], candidates["stratum_site"], candidates["stratum_quality_note_present"]
        ))
        candidates["stratum"] = candidates["_stratum_tuple"].map(
            lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        )
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in candidates.to_dict("records"):
            groups[row["_stratum_tuple"]].append(row)
        quotas = _quotas(groups, per_class)
        for stratum in sorted(groups):
            quota, remainder = quotas[stratum]
            rows = groups[stratum]
            rows.sort(key=lambda row: (_hash("sample", seed, label, row["_patient_key"]), row["_patient_key"], row["record_hash"], row["_ecg_key"]))
            for sample_rank, row in enumerate(rows[:quota], start=1):
                row["sample_hash"] = _hash("sample", seed, label, row["_patient_key"])
                row["role_hash"] = _hash("role", seed, label, row["_patient_key"])
                row["stratum_quota"] = quota
                row["stratum_capacity"] = len(rows)
                row["stratum_remainder"] = remainder
                row["selection_reason"] = "largest_remainder_stratum_sample"
                row["sample_rank"] = sample_rank
                selected.append(row)

    out = pd.DataFrame(selected)
    prototype_count = 100 if per_class == 150 else math.ceil(2 * per_class / 3)
    for label in _LABELS:
        mask = out["strict_label"] == label
        ranked = out.loc[mask].sort_values(
            ["role_hash", "_patient_key", "record_hash", "_ecg_key"], kind="mergesort"
        ).index.tolist()
        out.loc[ranked[:prototype_count], "role"] = "prototype_pool"
        out.loc[ranked[prototype_count:], "role"] = "exploratory_probe"
        out.loc[mask, "role_quota"] = prototype_count
        out.loc[ranked, "role_rank"] = range(1, len(ranked) + 1)

    if out["_patient_key"].duplicated().any():
        raise AssertionError("pilot roles/classes must be patient-disjoint")
    if out.groupby("strict_label")["_patient_key"].nunique().to_dict() != {label: per_class for label in _LABELS}:
        raise AssertionError("pilot class counts are not exact")
    out = out.sort_values(
        ["strict_label", "role", "sample_hash", "role_hash", "_patient_key", "record_hash", "_ecg_key"],
        kind="mergesort",
    ).reset_index(drop=True)
    return out.drop(columns=["_patient_key", "_ecg_key", "_stratum_tuple"], errors="ignore")


__all__ = ["select_pilot"]
