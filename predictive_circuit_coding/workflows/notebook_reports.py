from __future__ import annotations

from pathlib import Path
from typing import Any


def build_notebook_preflight_rows(
    *,
    path_status: dict[str, bool],
) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "item": key,
            "status": ("OK" if bool(value) else "Missing"),
            "icon": ("check" if bool(value) else "x"),
        }
        for key, value in path_status.items()
    )


def load_pipeline_display_tables(
    *,
    refinement_summary_csv_path: str | Path,
    final_summary_csv_path: str | Path,
    training_history_csv_path: str | Path = "",
) -> dict[str, Any]:
    try:
        import pandas as pd
    except Exception:
        return {
            "refinement": [],
            "final": [],
            "training_history": [],
        }

    def _load(path: str | Path):
        candidate = Path(path)
        if not candidate.is_file():
            return pd.DataFrame()
        return pd.read_csv(candidate)

    refinement = _load(refinement_summary_csv_path)
    final = _load(final_summary_csv_path)
    training_history = _load(training_history_csv_path)

    arm_order = {
        "encoder_raw": 0,
        "encoder_token_normalized": 1,
        "encoder_probe_weighted": 2,
        "encoder_aligned_oracle": 3,
    }

    def _sort_for_comparison(df):
        if "arm_name" not in df.columns:
            return df
        sorted_df = df.copy()
        sorted_df["_arm_order"] = sorted_df["arm_name"].map(arm_order).fillna(99)
        sort_columns = ["_arm_order"]
        if "task_name" in sorted_df.columns:
            sort_columns = ["task_name", "_arm_order"]
        return sorted_df.sort_values(sort_columns, na_position="last").drop(columns=["_arm_order"])

    if not refinement.empty:
        refinement = _sort_for_comparison(refinement)
        preferred = [
            "task_name",
            "arm_name",
            "variant_name",
            "status",
            "failure_reason",
            "feature_family",
            "encoder_training_status",
            "geometry_mode",
            "candidate_geometry_mode",
            "claim_safe",
            "candidate_count",
            "cluster_count",
            "held_out_test_probe_pr_auc",
            "held_out_test_probe_roc_auc",
            "held_out_similarity_pr_auc",
            "held_out_similarity_roc_auc",
            "cluster_persistence_mean",
            "silhouette_score",
        ]
        refinement = refinement[[column for column in preferred if column in refinement.columns]]

    if not training_history.empty:
        preferred = [
            "epoch",
            "global_step",
            "variant_name",
            "reconstruction_target_mode",
            "count_normalization_mode",
            "learning_rate",
            "evaluated",
            "became_best",
            "best_epoch_so_far",
            "best_predictive_improvement_so_far",
            "train_total_loss",
            "train_predictive_improvement",
            "train_predictive_loss",
            "valid_predictive_improvement",
            "valid_predictive_loss",
            "valid_predictive_baseline_mse",
            "valid_predictive_raw_mse",
        ]
        training_history = training_history[[column for column in preferred if column in training_history.columns]]

    return {
        "refinement": refinement,
        "final": final,
        "training_history": training_history,
    }


def build_pipeline_summary_figure(
    *,
    refinement_df,
    final_df,
    alignment_df=None,
    title: str = "Predictive Circuit Coding Summary",
):
    import matplotlib.pyplot as plt
    import numpy as np

    def _truncate(label: str, n: int = 30) -> str:
        return label if len(label) <= n else label[:n - 1] + "\u2026"

    def _empty_axis(axis, heading: str, message: str) -> None:
        axis.set_title(heading, fontsize=12, fontweight="bold")
        axis.axis("off")
        axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)

    def _series(values_df, column: str):
        if values_df is None or getattr(values_df, "empty", True) or column not in values_df.columns:
            return None
        numeric = values_df[column]
        try:
            return numeric.astype(float)
        except Exception:
            return numeric

    n_refinement = len(refinement_df) if refinement_df is not None and not getattr(refinement_df, "empty", True) else 0
    fig_height = max(11, max(n_refinement, 4) * 1.2)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes = plt.subplots(2, 2, figsize=(16, fig_height), constrained_layout=True)
        figure.suptitle(title, fontsize=16, fontweight="bold", y=1.01)

        probe_axis = axes[0, 0]
        if refinement_df is None or getattr(refinement_df, "empty", True):
            _empty_axis(probe_axis, "Probe Signal", "No refinement rows available.")
        else:
            probe_plot_df = refinement_df.copy()
            probe_plot_df["label"] = (
                probe_plot_df["task_name"].astype(str) + " | " + probe_plot_df["arm_name"].astype(str)
            ).apply(_truncate)
            probe_plot_df = probe_plot_df.head(8)
            x = np.arange(len(probe_plot_df))
            width = 0.35
            probe_pr = _series(probe_plot_df, "held_out_test_probe_pr_auc")
            probe_roc = _series(probe_plot_df, "held_out_test_probe_roc_auc")
            probe_axis.bar(
                x - width / 2,
                probe_pr if probe_pr is not None else 0.0,
                width=width,
                color="#3b7ddd",
                alpha=0.85,
                label="Probe PR-AUC",
            )
            if probe_roc is not None:
                probe_axis.bar(
                    x + width / 2,
                    probe_roc,
                    width=width,
                    color="#577590",
                    alpha=0.85,
                    label="Probe ROC-AUC",
                )
            probe_axis.set_xticks(x)
            probe_axis.set_xticklabels(probe_plot_df["label"].tolist(), rotation=40, ha="right", fontsize=8)
            probe_axis.set_ylim(0.0, 1.05)
            probe_axis.set_ylabel("Score", fontsize=10)
            probe_axis.set_title("Probe Signal", fontsize=12, fontweight="bold")
            probe_axis.grid(axis="y", alpha=0.4)
            probe_axis.legend(loc="upper right", frameon=False, fontsize=9)

        motif_axis = axes[0, 1]
        if refinement_df is None or getattr(refinement_df, "empty", True):
            _empty_axis(motif_axis, "Motif Transfer", "No refinement rows available.")
        else:
            motif_plot_df = refinement_df.copy()
            motif_plot_df["label"] = (
                motif_plot_df["task_name"].astype(str) + " | " + motif_plot_df["arm_name"].astype(str)
            ).apply(_truncate)
            motif_plot_df = motif_plot_df.head(8)
            x = np.arange(len(motif_plot_df))
            width = 0.5
            motif_scores = _series(motif_plot_df, "held_out_similarity_pr_auc")
            cluster_counts = _series(motif_plot_df, "cluster_count")
            candidate_counts = _series(motif_plot_df, "candidate_count")
            scores = motif_scores if motif_scores is not None else np.zeros(len(motif_plot_df))
            motif_axis.bar(x, scores, width=width, color="#2a9d8f", alpha=0.85)
            if cluster_counts is not None:
                for index, value in enumerate(cluster_counts.tolist()):
                    motif_axis.text(
                        x[index],
                        float(scores.iloc[index]) * 0.95,
                        f"k={int(value)}",
                        va="top",
                        ha="center",
                        fontsize=8,
                        color="white",
                        fontweight="bold",
                    )
            if candidate_counts is not None:
                for index, value in enumerate(candidate_counts.tolist()):
                    motif_axis.text(
                        x[index],
                        float(scores.iloc[index]) * 0.05,
                        f"n={int(value)}",
                        va="bottom",
                        ha="center",
                        fontsize=8,
                        color="white",
                        fontweight="bold",
                    )
            motif_axis.set_xticks(x)
            motif_axis.set_xticklabels(motif_plot_df["label"].tolist(), rotation=40, ha="right", fontsize=8)
            motif_axis.set_ylim(0.0, 1.05)
            motif_axis.set_ylabel("Held-out Motif PR-AUC", fontsize=10)
            motif_axis.set_title("Motif Transfer", fontsize=12, fontweight="bold")
            motif_axis.grid(axis="y", alpha=0.4)

        safety_axis = axes[1, 0]
        if refinement_df is None or getattr(refinement_df, "empty", True):
            _empty_axis(safety_axis, "Claim Safety", "No refinement rows available.")
        else:
            safety_df = refinement_df.copy()
            if "claim_safe" in safety_df.columns:
                safety_counts = safety_df["claim_safe"].astype(bool).value_counts()
                labels = ["claim-safe", "diagnostic"]
                values = [
                    int(safety_counts.get(True, 0)),
                    int(safety_counts.get(False, 0)),
                ]
            else:
                labels = ["unknown"]
                values = [len(safety_df)]
            safety_axis.bar(labels, values, color=["#2a9d8f", "#e07a1f"][: len(values)], alpha=0.85)
            safety_axis.set_ylabel("Rows", fontsize=10)
            safety_axis.set_title("Claim Safety", fontsize=12, fontweight="bold")
            safety_axis.grid(axis="y", alpha=0.4)

        summary_axis = axes[1, 1]
        summary_axis.axis("off")
        summary_axis.set_title("Run Summary", fontsize=12, fontweight="bold")
        summary_lines: list[str] = []
        if final_df is not None and not getattr(final_df, "empty", True):
            row = final_df.iloc[0]
            summary_lines.append(f"Refinement rows: {row.get('motif_row_count', row.get('refinement_row_count', 'n/a'))}")
            summary_lines.append(
                f"Mean motif held-out PR-AUC: {row.get('motif_mean_held_out_similarity_pr_auc', 'n/a')}"
            )
            notes = row.get("notes")
            if isinstance(notes, str):
                try:
                    import ast

                    parsed_notes = ast.literal_eval(notes)
                    if isinstance(parsed_notes, list):
                        notes = parsed_notes
                except Exception:
                    notes = [notes]
            if isinstance(notes, list):
                summary_lines.append("")
                summary_lines.append("Notes:")
                summary_lines.extend(f"  \u2022 {note}" for note in notes[:4])
        else:
            summary_lines.append("No final summary rows available.")

        if alignment_df is not None and not getattr(alignment_df, "empty", True):
            summary_lines.append("")
            summary_lines.append("Alignment diagnostic present.")

        summary_axis.text(
            0.05,
            0.95,
            "\n".join(summary_lines),
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
            linespacing=1.6,
        )

    return figure


def build_synthetic_pipeline_summary_tables() -> dict[str, Any]:
    import pandas as pd

    refinement_df = pd.DataFrame(
        [
            {
                "task_name": "stimulus_change",
                "arm_name": "encoder_raw",
                "status": "ok",
                "feature_family": "encoder",
                "geometry_mode": "raw",
                "candidate_geometry_mode": "embedding",
                "claim_safe": True,
                "candidate_count": 32,
                "cluster_count": 4,
                "held_out_test_probe_pr_auc": 0.66,
                "held_out_test_probe_roc_auc": 0.68,
                "held_out_similarity_pr_auc": 0.61,
                "held_out_similarity_roc_auc": 0.63,
                "cluster_persistence_mean": 0.58,
                "silhouette_score": 0.42,
            },
            {
                "task_name": "trials_go",
                "arm_name": "encoder_token_normalized",
                "status": "ok",
                "feature_family": "encoder",
                "geometry_mode": "token_normalized",
                "candidate_geometry_mode": "embedding",
                "claim_safe": True,
                "candidate_count": 24,
                "cluster_count": 2,
                "held_out_test_probe_pr_auc": 0.58,
                "held_out_test_probe_roc_auc": 0.6,
                "held_out_similarity_pr_auc": 0.54,
                "held_out_similarity_roc_auc": 0.56,
                "cluster_persistence_mean": 0.41,
                "silhouette_score": 0.38,
            },
            {
                "task_name": "stimulus_change",
                "arm_name": "encoder_probe_weighted",
                "status": "ok",
                "feature_family": "encoder",
                "geometry_mode": "raw",
                "candidate_geometry_mode": "probe_weighted",
                "claim_safe": True,
                "candidate_count": 18,
                "cluster_count": 3,
                "held_out_test_probe_pr_auc": 0.52,
                "held_out_test_probe_roc_auc": 0.57,
                "held_out_similarity_pr_auc": 0.49,
                "held_out_similarity_roc_auc": 0.52,
                "cluster_persistence_mean": 0.36,
                "silhouette_score": 0.31,
            },
            {
                "task_name": "trials_go",
                "arm_name": "encoder_aligned_oracle",
                "status": "ok",
                "feature_family": "encoder",
                "geometry_mode": "aligned_oracle",
                "candidate_geometry_mode": "embedding",
                "claim_safe": False,
                "candidate_count": 20,
                "cluster_count": 2,
                "held_out_test_probe_pr_auc": 0.72,
                "held_out_test_probe_roc_auc": 0.75,
                "held_out_similarity_pr_auc": 0.64,
                "held_out_similarity_roc_auc": 0.67,
                "cluster_persistence_mean": 0.5,
                "silhouette_score": 0.4,
            },
        ]
    )
    final_df = pd.DataFrame(
        [
            {
                "motif_row_count": len(refinement_df),
                "motif_mean_held_out_similarity_pr_auc": float(refinement_df["held_out_similarity_pr_auc"].mean()),
                "claim_safe_row_count": int(refinement_df["claim_safe"].sum()),
                "diagnostic_row_count": int((~refinement_df["claim_safe"]).sum()),
                "notes": [
                    "refinement rows compare trained-encoder discovery transforms",
                    "oracle alignment is diagnostic and not claim-safe",
                ],
            }
        ]
    )
    return {
        "refinement": refinement_df,
        "final": final_df,
    }


def write_synthetic_pipeline_summary_preview(
    output_path: str | Path,
    *,
    dpi: int = 160,
) -> Path:
    import matplotlib.pyplot as plt

    tables = build_synthetic_pipeline_summary_tables()
    figure = build_pipeline_summary_figure(
        refinement_df=tables["refinement"],
        final_df=tables["final"],
        title="Predictive Circuit Coding Summary Preview",
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return target
