from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def write_summary_rows(
    rows: list[dict[str, Any]],
    *,
    output_json_path: str | Path,
    output_csv_path: str | Path,
    root_key: str,
) -> tuple[Path, Path]:
    json_path = Path(output_json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({root_key: rows}, indent=2), encoding="utf-8")
    csv_path = Path(output_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return json_path, csv_path


def write_single_row_summary(
    row: dict[str, Any],
    *,
    output_json_path: str | Path,
    output_csv_path: str | Path,
    root_key: str = "summary",
) -> tuple[Path, Path]:
    return write_summary_rows(
        [row],
        output_json_path=output_json_path,
        output_csv_path=output_csv_path,
        root_key=root_key,
    )


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None and not math.isnan(float(row[key]))]
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def build_final_project_summary(
    *,
    motif_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    motif_ok = [row for row in motif_rows if row.get("status") == "ok"]

    def _best_row(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
        scored_rows = [
            row
            for row in rows
            if row.get(key) is not None and not math.isnan(float(row[key]))
        ]
        if not scored_rows:
            return None
        return max(scored_rows, key=lambda row: float(row[key]))

    best_motif = _best_row(motif_ok, "held_out_similarity_pr_auc")

    notes: list[str] = []
    if best_motif is not None:
        notes.append(
            "best held-out motif PR-AUC came from "
            f"{best_motif['arm_name']} on {best_motif['task_name']} "
            f"({best_motif.get('variant_name', 'refined_core')})"
        )

    return {
        "refinement_row_count": len(motif_rows),
        "motif_completed_row_count": len(motif_ok),
        "motif_mean_held_out_similarity_pr_auc": _mean_metric(motif_ok, "held_out_similarity_pr_auc"),
        "variant_names": sorted(
            {
                str(row.get("variant_name", "refined_core"))
                for row in motif_rows
                if row.get("variant_name") is not None
            }
        ),
        "best_motif_row": best_motif,
        "notes": notes,
    }
