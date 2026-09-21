"""The deliberately small, injectable orchestration entry point for the pilot."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Any
from dataclasses import replace
from time import perf_counter
import platform

import numpy as np
import pandas as pd
import scipy

from .beats import detect_r_peaks, extract_beats
from .bsw_like import align_pair
from .cohort import load_ptbxl_metadata
from .pilot import select_pilot
from .reporting import AtomicRunWriter, SCHEMA_VERSION, data_dictionary, runtime_code_hash, source_manifest
from .sqi import sqi_from_wfdb
from .wfdb_io import WFDBRecord, load_wfdb_record, validate_ptbxl_500hz_record


Reader = Callable[[str | Path], WFDBRecord]
_TARGET_LEADS = ("V1", "V5", "V6")
_LABELS = ("LVH", "NORM")


def _default_reader(data_root: Path) -> Reader:
    return lambda path: load_wfdb_record(path, data_root=data_root)


def _as_row(result: object, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_ok": getattr(result, "ok", None),
        f"{prefix}_failure_code": getattr(result, "failure_code", None),
        f"{prefix}_detail": getattr(result, "detail", None),
    }


def _prototype_is_eligible(beat: object) -> bool:
    """Apply the paper's low-variability prototype gate.

    A prototype candidate is retained only when beat extraction and the
    twelve-lead ``v_h`` calculation succeed and *all* leads are below 0.3.
    Records failing this gate remain in the audit table but do not contribute
    to class references.
    """

    vh = getattr(beat, "vh", None)
    return bool(
        getattr(beat, "ok", False)
        and vh is not None
        and getattr(vh, "ok", False)
        and getattr(vh, "all_leads_low_vh", False)
    )


def _first_failure_code(frame: pd.DataFrame) -> pd.Series:
    """Return the first available processing failure in audit priority order."""

    result = pd.Series(pd.NA, index=frame.index, dtype="object")
    for column in (
        "read_failure_code",
        "peak_failure_code",
        "beat_failure_code",
        "vh_failure_code",
        "sqi_failure_code",
    ):
        if column not in frame:
            continue
        values = frame[column]
        missing = values.isna() | values.astype("string").str.strip().eq("")
        result = result.mask(result.isna() & ~missing, values)
    return result


def _record_row(manifest_row: dict[str, Any], record: WFDBRecord) -> tuple[dict[str, Any], np.ndarray | None, bool]:
    row = {"ecg_id": manifest_row["ecg_id"], "patient_id": manifest_row["patient_id"], "strict_label": manifest_row["strict_label"], "role": manifest_row["role"], "strat_fold": manifest_row["strat_fold"], "sample_hash": manifest_row["sample_hash"]}
    row.update(_as_row(record, "read"))
    sqi = sqi_from_wfdb(record)
    row.update(_as_row(sqi, "sqi"))
    row["sqi_flags"] = list(sqi.flags)
    row["sqi_low_amplitude"] = "low_amplitude_range" in sqi.flags
    row["sqi_flatline"] = "excessive_flatline" in sqi.flags
    row["sqi_repeated_or_anticorrelated"] = "repeated_or_anticorrelated_leads" in sqi.flags
    row["sqi_per_lead"] = [
        {"lead": metric.lead_name, "amplitude_range_mv": metric.amplitude_range_mv,
         "flatline_fraction": metric.flatline_fraction,
         "boundary_plateau_fraction": metric.boundary_plateau_fraction,
         "abrupt_jump_fraction": metric.abrupt_jump_fraction,
         "baseline_wander_power_ratio": metric.baseline_wander_power_ratio,
         "power_49_51_ratio": metric.power_49_51_ratio,
         "power_40_100_ratio": metric.power_40_100_ratio,
         "total_power_0_100_mv2": metric.total_power_0_100_mv2,
         "failure_codes": list(metric.failure_codes)}
        for metric in sqi.lead_metrics
    ]
    if sqi.lead_metrics:
        row.update({
            "sqi_amplitude_range_mv": float(np.median([m.amplitude_range_mv for m in sqi.lead_metrics]),
            ), "sqi_flatline_fraction": float(np.max([m.flatline_fraction for m in sqi.lead_metrics])),
            "sqi_boundary_plateau_fraction": float(np.max([m.boundary_plateau_fraction for m in sqi.lead_metrics])),
            "sqi_abrupt_jump_fraction": float(np.max([m.abrupt_jump_fraction for m in sqi.lead_metrics])),
            "sqi_baseline_wander_power_ratio": float(np.median([m.baseline_wander_power_ratio for m in sqi.lead_metrics])),
            "sqi_power_49_51_ratio": float(np.median([m.power_49_51_ratio for m in sqi.lead_metrics])),
            "sqi_power_40_100_ratio": float(np.median([m.power_40_100_ratio for m in sqi.lead_metrics])),
        })
    if not record.ok or record.signal is None or record.sampling_rate is None:
        row.update({"peak_ok": False, "peak_failure_code": "read_failure", "beat_ok": False, "beat_failure_code": "read_failure", "rr_median_ms": None, "rr_cv": None, "rr_rmssd_ms": None, "rr_mad_ms": None, "rr_iqr_ms": None, "vh_failure_code": "read_failure", "vh_values": None, "artifact_rejected_count": 0})
        return row, None, False
    peak = detect_r_peaks(record.signal, record.sampling_rate, record.lead_names)
    row.update(_as_row(peak, "peak")); row["peak_confidence"] = peak.confidence
    if not peak.ok:
        row.update({"beat_ok": False, "beat_failure_code": peak.failure_code, "rr_median_ms": None, "rr_cv": None, "rr_rmssd_ms": None, "rr_mad_ms": None, "rr_iqr_ms": None, "vh_failure_code": peak.failure_code, "vh_values": None, "artifact_rejected_count": 0})
        return row, None, False
    beat = extract_beats(record.signal, peak.peaks, record.sampling_rate, record.lead_names)
    row.update(_as_row(beat, "beat")); row["beat_confidence"] = beat.confidence
    row["rr_median_ms"] = beat.rr.median_ms
    row["rr_cv"] = beat.rr.cv
    row["rr_rmssd_ms"] = beat.rr.rmssd_ms
    row["rr_mad_ms"] = beat.rr.mad_ms
    row["rr_iqr_ms"] = beat.rr.iqr_ms
    row["vh_failure_code"] = beat.vh_failure_code
    row["vh_values"] = None if beat.vh is None else beat.vh.per_lead_vh.tolist()
    row["vh_all_leads_lt_0_3"] = bool(beat.vh and beat.vh.all_leads_low_vh)
    row["valid_beat_count"] = beat.valid_count
    row["artifact_rejected_count"] = beat.artifact_rejected_count
    row["artifact_rule"] = beat.artifact_rule
    prototype_eligible = _prototype_is_eligible(beat)
    return row, beat.representative if beat.ok else None, prototype_eligible


def _prototype_references(manifest: pd.DataFrame, representatives: dict[int, np.ndarray], prototype_eligible: set[int]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Return eligible class means and their successful, patient-level denominators."""

    references: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for label in _LABELS:
        ids = manifest.loc[
            manifest["role"].eq("prototype_pool") & manifest["strict_label"].eq(label), "ecg_id"
        ].astype(int)
        successful = [representatives[ecg_id] for ecg_id in ids if ecg_id in representatives and ecg_id in prototype_eligible]
        counts[label] = len(successful)
        if len(successful) >= 20:
            references[label] = np.mean(np.stack(successful), axis=0)
    return references, counts


def _probe_rows(manifest: pd.DataFrame, representatives: dict[int, np.ndarray], prototype_eligible: set[int]) -> list[dict[str, Any]]:
    """Calculate the required bidirectional mean distance for each attempted probe."""

    references, prototype_counts = _prototype_references(manifest, representatives, prototype_eligible)
    positions = {name: index for index, name in enumerate(("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"))}
    rows: list[dict[str, Any]] = []
    for record in manifest.loc[manifest["role"].eq("exploratory_probe")].to_dict("records"):
        ecg_id = int(record["ecg_id"])
        for lead in _TARGET_LEADS:
            for reference_label in _LABELS:
                base = {"ecg_id": ecg_id, "lead": lead, "reference_label": reference_label, "prototype_success_count": prototype_counts[reference_label]}
                if reference_label not in references:
                    rows.append({**base, "ok": False, "failure_code": "insufficient_prototypes", "distance": None})
                    continue
                if ecg_id not in representatives:
                    rows.append({**base, "ok": False, "failure_code": "probe_beat_unavailable", "distance": None})
                    continue
                source = representatives[ecg_id][positions[lead]]
                target = references[reference_label][positions[lead]]
                forward, reverse = align_pair(source, target), align_pair(target, source)
                if not (forward.ok and reverse.ok and np.isfinite(forward.distance) and np.isfinite(reverse.distance)):
                    codes = [result.failure_code for result in (forward, reverse) if result.failure_code]
                    rows.append({**base, "ok": False, "failure_code": codes[0] if codes else "alignment_failed", "distance": None})
                    continue
                rows.append({**base, "ok": True, "failure_code": None, "distance": float((forward.distance + reverse.distance) / 2)})
    return rows


def _casebook_pdf(casebook: pd.DataFrame) -> bytes:
    """Build a valid fixed-layout PDF containing every audit-case row."""

    lines = ["schema_version=1", "ECG 5001 deterministic audit casebook"]
    lines.extend(f"{row.ecg_id} | {row.inclusion_reason}" for row in casebook.itertuples())

    # Keep each page within the visible media box, while retaining every row
    # instead of silently dropping records when a casebook grows.
    page_lines = [lines[index : index + 35] for index in range(0, len(lines), 35)] or [[]]
    page_ids = [4 + 2 * index for index in range(len(page_lines))]
    content_ids = [page_id + 1 for page_id in page_ids]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{' '.join(f'{page_id} 0 R' for page_id in page_ids)}] /Count {len(page_ids)} >>".encode("ascii"),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for page_id, content_id, chunk in zip(page_ids, content_ids, page_lines, strict=True):
        escaped = [
            str(line).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            for line in chunk
        ]
        stream = "BT /F1 10 Tf 50 760 Td " + " ".join(f"({line}) Tj 0 -20 Td" for line in escaped) + " ET"
        stream_bytes = stream.encode("latin-1", errors="replace")
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("ascii")
        objects[content_id] = (
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"\nendstream"
        )

    body = [b"%PDF-1.4\n"]
    offsets = [0]
    for number in range(1, max(objects) + 1):
        content = objects[number]
        offsets.append(sum(len(part) for part in body))
        body.append(f"{number} 0 obj\n".encode("ascii") + content + b"\nendobj\n")
    xref = sum(len(part) for part in body)
    size = max(objects) + 1
    body.append(
        f"xref\n0 {size}\n0000000000 65535 f \n".encode("ascii")
        + b"".join(f"{offset:010d} 00000 n \n".encode("ascii") for offset in offsets[1:])
        + f"trailer << /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    return b"".join(body)


def _pilot_report(summary: dict[str, Any]) -> str:
    """Render the machine summary as a compact, independently readable report."""

    def encoded(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    gate = summary["engineering_gate"]
    lines = [
        "# ECG 5001 pilot",
        "",
        f"schema_version: {summary['schema_version']}",
        f"config: {encoded(summary['config'])}",
        f"config_hash: {summary['config_hash']}",
        f"code_hash: {summary['code_hash']}",
        f"source_manifest_hash: {summary['source_manifest_hash']}",
        f"unconditional_denominator: {summary['unconditional_denominator']}",
        f"conditional_denominators: {encoded(summary['conditional_denominators'])}",
        f"prototype_denominator_per_class: {summary['prototype_denominator_per_class']}",
        f"probe_denominator_per_class: {summary['probe_denominator_per_class']}",
        f"probe_denominator_total: {summary['probe_denominator_total']}",
        f"per_class_rates: {encoded(summary['per_class_rates'])}",
        f"per_lead_rates: {encoded(summary['per_lead_rates'])}",
        f"elapsed_seconds: {summary['elapsed_seconds']}",
        f"Engineering gate: {gate['status']} (ok={gate['ok']})",
        "",
        "Limitations: engineering feasibility only; no clinical-performance, generalization, or exact-paper-reproduction claim.",
        "Long-horizon outcomes: `not_evaluated` (GARCH/EGARCH and 3-minute records are outside this pilot).",
        "",
    ]
    return "\n".join(lines)


def run_pilot(
    data_root: str | Path,
    output_root: str | Path,
    *,
    per_class: int = 150,
    seed: int = 5001,
    reader: Reader | None = None,
) -> dict[str, Any]:
    """Run the train-only pilot and atomically publish its auditable artifacts.

    The reader is injected for tests and is called only for rows selected by
    ``select_pilot``; because that sampler has a folds 1--8 predicate, folds
    9/10 remain metadata-only by construction.
    """

    started = perf_counter()
    root = Path(data_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    if destination == root or root in destination.parents:
        raise ValueError("output_root must be outside data_root")
    index = load_ptbxl_metadata(root, assert_counts=(per_class == 150))
    manifest = select_pilot(index, per_class=per_class, seed=seed)
    if not manifest["strat_fold"].isin(range(1, 9)).all():
        raise AssertionError("waveform manifest contains a held-out fold")
    active_reader = reader or _default_reader(root)
    metric_rows: list[dict[str, Any]] = []
    representatives: dict[int, np.ndarray] = {}
    prototype_eligible: set[int] = set()
    selected_paths = [Path(path) for path in manifest["hea_path"]]
    selected_paths.extend(Path(path) for path in manifest["dat_path"])
    source_before = source_manifest([root / "ptbxl_database.csv", root / "scp_statements.csv", *selected_paths], root)
    for selected in manifest.to_dict("records"):
        record = active_reader(selected["hea_path"])
        if record.ok:
            validation = validate_ptbxl_500hz_record(record)
            if not validation.ok:
                record = replace(record, signal=None, ok=False, failure_code=validation.failure_code, detail=validation.detail)
        metric, representative, is_prototype_eligible = _record_row(selected, record)
        metric_rows.append(metric)
        if representative is not None:
            representatives[int(selected["ecg_id"])] = representative
        if is_prototype_eligible and selected["role"] == "prototype_pool":
            prototype_eligible.add(int(selected["ecg_id"]))
    metric_frame = pd.DataFrame(metric_rows)
    # Fixed schema remains present even when every selected reader result fails
    # before continuous SQI calculation (a common audit scenario).
    for column in (
        "sqi_amplitude_range_mv", "sqi_flatline_fraction", "sqi_boundary_plateau_fraction",
        "sqi_abrupt_jump_fraction", "sqi_baseline_wander_power_ratio", "sqi_power_49_51_ratio",
        "sqi_power_40_100_ratio", "sqi_per_lead", "sqi_rank",
    ):
        if column not in metric_frame:
            metric_frame[column] = None
    probes = pd.DataFrame(_probe_rows(manifest, representatives, prototype_eligible))
    beats = np.stack(list(representatives.values())) if representatives else np.empty((0, 12, 500), dtype=float)
    ecg_ids = np.asarray(list(representatives), dtype=int)
    config_hash = sha256(json.dumps({"per_class": per_class, "seed": seed}, sort_keys=True).encode()).hexdigest()
    source_after = source_manifest([root / "ptbxl_database.csv", root / "scp_statements.csv", *selected_paths], root)
    if source_before != source_after:
        raise RuntimeError("source manifest changed during read-only pilot")
    prototype_counts = _prototype_references(manifest, representatives, prototype_eligible)[1]
    read_success = int(metric_frame["read_ok"].fillna(False).sum())
    beat_success = int(metric_frame["beat_ok"].fillna(False).sum())
    probe_by_patient = probes.groupby("ecg_id")["ok"].all() if not probes.empty else pd.Series(dtype=bool)
    case_rows = metric_frame.copy()
    case_rows["failure_code"] = _first_failure_code(case_rows)
    case_rows["review_flag"] = case_rows["sqi_flags"].astype(str).ne("[]")
    case_rows["sqi_rank"] = case_rows["sqi_amplitude_range_mv"].rank(method="dense", ascending=True, na_option="bottom")
    case_rows["inclusion_reason"] = np.where(case_rows["failure_code"].notna(), case_rows["failure_code"], np.where(case_rows["review_flag"], "sqi_review", "class_audit"))
    casebook = case_rows.sort_values(["failure_code", "review_flag", "sqi_rank", "sample_hash"], na_position="last").groupby(["strict_label", "inclusion_reason"], dropna=False).head(5)
    prototype_role_count = int(manifest["role"].eq("prototype_pool").sum())
    probe_role_count = int(manifest["role"].eq("exploratory_probe").sum())
    gate_applies = per_class == 150
    summary = {
        "schema_version": SCHEMA_VERSION, "unconditional_denominator": int(len(manifest)),
        "prototype_denominator_per_class": int(prototype_role_count // 2),
        "probe_denominator_per_class": int(probe_role_count // 2), "probe_denominator_total": probe_role_count, "config_hash": config_hash,
        "config": {"data_root": str(root), "output_root": str(destination), "per_class": per_class, "seed": seed},
        "default_parameters": {"per_class": 150, "seed": 5001, "sampling_rate": 500, "samples": 5000, "leads": 12},
        "input_manifest": source_before, "source_manifest_hash": sha256(json.dumps(source_before, sort_keys=True).encode()).hexdigest(), "source_manifest_before": source_before,
        "source_manifest_after": source_after,
        "non_audit_artifact_rows": [str(value) for value in manifest["filename_hr"]],
        "strict_counts": {label: int(index.strict_label.eq(label).sum()) for label in _LABELS},
        "patient_split_assertion": True,
        "code_hash": runtime_code_hash(
            Path(__file__).resolve().parent,
            include_paths=(
                Path(__file__).resolve().parents[2] / "scripts" / "run_pilot.py",
                Path(__file__).resolve().parents[2] / "pyproject.toml",
            ),
        ),
        "package_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "elapsed_seconds": perf_counter() - started, "conditional_denominators": {"read": read_success, "beat": beat_success, "probe_patients": int(len(probe_by_patient))},
        "per_class_rates": metric_frame.groupby("strict_label")["beat_ok"].mean().fillna(0).to_dict(),
        "per_lead_rates": {lead: float(probes.loc[probes["lead"].eq(lead), "ok"].mean()) if not probes.empty else 0.0 for lead in _TARGET_LEADS},
        "engineering_gate": {"applies": gate_applies, "status": "not_applicable" if not gate_applies else ("pass" if read_success >= 297 and beat_success >= 270 and min(prototype_counts.values(), default=0) >= 20 and int(probe_by_patient.sum()) >= 80 else "fail"), "ok": None if not gate_applies else bool(read_success >= 297 and beat_success >= 270 and min(prototype_counts.values(), default=0) >= 20 and int(probe_by_patient.sum()) >= 80), "read_success": read_success, "read_threshold": 297, "beat_success": beat_success, "beat_threshold": 270, "prototype_success_per_class": prototype_counts, "prototype_min_per_class": 20, "probe_all_six_success": int(probe_by_patient.sum()), "probe_threshold": 80},
        "long_horizon": "not_evaluated",
    }
    with AtomicRunWriter(output_root, config_hash) as writer:
        writer.write_csv("cohort_index.csv", index)
        writer.write_json("cohort_summary.json", {"schema_version": SCHEMA_VERSION, "row_count": len(index), "fold_counts": index["strat_fold"].value_counts().sort_index().to_dict(), "strict_counts": summary["strict_counts"], "patient_split_assertion": True, "input_manifest": source_before})
        writer.write_csv("pilot_manifest.csv", manifest)
        writer.write_csv("record_metrics.csv", metric_frame)
        writer.write_npz("representative_beats.npz", schema_version=np.asarray(SCHEMA_VERSION), ecg_ids=ecg_ids, lead_names=np.asarray(("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")), beats_mv=beats)
        writer.write_csv("bsw_probe_metrics.csv", probes)
        writer.write_csv("casebook_manifest.csv", casebook)
        (writer.temp_dir / "casebook.pdf").write_bytes(_casebook_pdf(casebook))
        writer.write_json("data_dictionary.json", data_dictionary())
        writer.write_text("pilot_report.md", _pilot_report(summary))
        writer.write_json("summary.json", summary)
    summary["run_dir"] = str(writer.run_dir)
    return summary


__all__ = ["run_pilot"]
