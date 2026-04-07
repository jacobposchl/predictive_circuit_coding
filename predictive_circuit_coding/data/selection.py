from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

from predictive_circuit_coding.data.catalog import (
    SessionCatalog,
    SessionCatalogRecord,
    build_session_catalog_from_manifest,
    load_session_catalog,
    project_catalog_to_session_manifest,
    write_session_catalog,
    write_session_catalog_csv,
)
from predictive_circuit_coding.data.config import DataPreparationConfig, SplitPlanningConfig, load_preparation_config
from predictive_circuit_coding.data.layout import PreparationWorkspace, build_workspace
from predictive_circuit_coding.data.manifest import load_session_manifest
from predictive_circuit_coding.data.splits import SplitManifest, build_split_manifest, load_split_manifest, write_split_manifest
from predictive_circuit_coding.training.config import DatasetSelectionConfig, ExperimentConfig
from predictive_circuit_coding.windowing import build_torch_brain_config


@dataclass(frozen=True)
class DatasetSelectionSummary:
    output_name: str
    session_count: int
    subject_count: int
    total_units: int
    total_trials: int
    split_counts: dict[str, int]


@dataclass(frozen=True)
class ResolvedDatasetView:
    prep_config: DataPreparationConfig
    workspace: PreparationWorkspace
    session_catalog: SessionCatalog
    split_manifest: SplitManifest
    split_manifest_path: Path
    config_dir: Path
    config_name_prefix: str
    dataset_split: str | None
    selection_active: bool
    selection_summary: DatasetSelectionSummary | None
    session_catalog_path: Path


def _sanitize_output_name(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return text or "runtime_selection"


def _load_id_file(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _normalized_strings(values: tuple[str, ...]) -> set[str]:
    return {str(value).strip() for value in values if str(value).strip()}


def _default_split_config(prep_config: DataPreparationConfig) -> SplitPlanningConfig:
    return prep_config.splits


def _selection_split_config(
    *,
    selection: DatasetSelectionConfig,
    prep_config: DataPreparationConfig,
) -> SplitPlanningConfig:
    base = _default_split_config(prep_config)
    return SplitPlanningConfig(
        seed=selection.split_seed if selection.split_seed is not None else base.seed,
        primary_axis=selection.split_primary_axis or base.primary_axis,
        train_fraction=selection.train_fraction if selection.train_fraction is not None else base.train_fraction,
        valid_fraction=selection.valid_fraction if selection.valid_fraction is not None else base.valid_fraction,
        discovery_fraction=(
            selection.discovery_fraction if selection.discovery_fraction is not None else base.discovery_fraction
        ),
        test_fraction=selection.test_fraction if selection.test_fraction is not None else base.test_fraction,
    )


def filter_session_catalog(
    catalog: SessionCatalog,
    *,
    selection: DatasetSelectionConfig,
) -> SessionCatalog:
    include_session_ids = _normalized_strings(selection.session_ids) | _load_id_file(selection.session_ids_file)
    include_subject_ids = _normalized_strings(selection.subject_ids) | _load_id_file(selection.subject_ids_file)
    exclude_session_ids = _normalized_strings(selection.exclude_session_ids) | _load_id_file(selection.exclude_session_ids_file)
    exclude_subject_ids = _normalized_strings(selection.exclude_subject_ids) | _load_id_file(selection.exclude_subject_ids_file)
    allowed_experience_levels = _normalized_strings(selection.experience_levels)
    allowed_session_types = _normalized_strings(selection.session_types)
    allowed_image_sets = _normalized_strings(selection.image_sets)
    allowed_project_codes = _normalized_strings(selection.project_codes)
    allowed_brain_regions = _normalized_strings(selection.brain_regions_any)
    allowed_session_numbers = set(selection.session_numbers)

    selected: list[SessionCatalogRecord] = []
    for record in catalog.records:
        if include_session_ids and record.session_id not in include_session_ids:
            continue
        if include_subject_ids and record.subject_id not in include_subject_ids:
            continue

        if allowed_experience_levels and (record.experience_level or "") not in allowed_experience_levels:
            continue
        if allowed_session_types and (record.session_type or "") not in allowed_session_types:
            continue
        if allowed_image_sets and (record.image_set or "") not in allowed_image_sets:
            continue
        if allowed_project_codes and (record.project_code or "") not in allowed_project_codes:
            continue
        if allowed_session_numbers and record.session_number not in allowed_session_numbers:
            continue
        if allowed_brain_regions and not any(region in allowed_brain_regions for region in record.brain_regions):
            continue

        if selection.min_n_units is not None and record.n_units < selection.min_n_units:
            continue
        if selection.max_n_units is not None and record.n_units > selection.max_n_units:
            continue
        if selection.min_trial_count is not None and record.trial_count < selection.min_trial_count:
            continue
        if selection.max_trial_count is not None and record.trial_count > selection.max_trial_count:
            continue
        if selection.min_duration_s is not None and record.duration_s < selection.min_duration_s:
            continue
        if selection.max_duration_s is not None and record.duration_s > selection.max_duration_s:
            continue

        if record.session_id in exclude_session_ids:
            continue
        if record.subject_id in exclude_subject_ids:
            continue
        selected.append(record)

    return SessionCatalog(
        dataset_id=catalog.dataset_id,
        source_name=catalog.source_name,
        records=tuple(selected),
    )


def _selection_counts(split_manifest: SplitManifest) -> dict[str, int]:
    counts: dict[str, int] = {}
    for assignment in split_manifest.assignments:
        counts[assignment.split] = counts.get(assignment.split, 0) + 1
    return counts


def _selection_root_dirs(workspace: PreparationWorkspace, output_name: str) -> tuple[Path, Path]:
    return (
        workspace.manifests / "selections" / output_name,
        workspace.splits / "selections" / output_name,
    )


def _resolved_runtime_subset_view(
    *,
    experiment_config: ExperimentConfig,
    prep_config: DataPreparationConfig,
    workspace: PreparationWorkspace,
    runtime_catalog: SessionCatalog,
) -> ResolvedDatasetView:
    if experiment_config.runtime_subset is None:
        raise ValueError("runtime_subset is not configured for this experiment.")
    split_manifest = load_split_manifest(experiment_config.runtime_subset.split_manifest_path)
    return ResolvedDatasetView(
        prep_config=prep_config,
        workspace=workspace,
        session_catalog=runtime_catalog,
        split_manifest=split_manifest,
        split_manifest_path=experiment_config.runtime_subset.split_manifest_path,
        config_dir=experiment_config.runtime_subset.config_dir,
        config_name_prefix=experiment_config.runtime_subset.config_name_prefix,
        dataset_split=None,
        selection_active=False,
        selection_summary=None,
        session_catalog_path=experiment_config.runtime_subset.session_catalog_path,
    )


def materialize_runtime_selection(
    *,
    experiment_config: ExperimentConfig,
    prep_config: DataPreparationConfig,
    workspace: PreparationWorkspace,
    catalog: SessionCatalog,
) -> ResolvedDatasetView:
    if not experiment_config.dataset_selection.is_active:
        raise ValueError("Runtime dataset selection is not active for this experiment config.")

    output_name = _sanitize_output_name(experiment_config.dataset_selection.output_name)
    selected_catalog = filter_session_catalog(catalog, selection=experiment_config.dataset_selection)
    if not selected_catalog.records:
        raise ValueError(
            "Dataset selection matched zero sessions. Relax the dataset_selection filters or check the catalog metadata."
        )
    split_config = _selection_split_config(
        selection=experiment_config.dataset_selection,
        prep_config=prep_config,
    )
    selected_manifest = project_catalog_to_session_manifest(selected_catalog)
    selected_split_manifest = build_split_manifest(selected_manifest, config=split_config)

    selection_manifest_dir, selection_split_dir = _selection_root_dirs(workspace, output_name)
    selection_manifest_dir.mkdir(parents=True, exist_ok=True)
    selection_split_dir.mkdir(parents=True, exist_ok=True)

    selected_catalog_json = selection_manifest_dir / "selected_session_catalog.json"
    selected_catalog_csv = selection_manifest_dir / "selected_session_catalog.csv"
    selected_split_manifest_path = selection_split_dir / "selected_split_manifest.json"
    write_session_catalog(selected_catalog, selected_catalog_json)
    write_session_catalog_csv(selected_catalog, selected_catalog_csv)
    write_split_manifest(selected_split_manifest, selected_split_manifest_path)
    for split_name in ("train", "valid", "discovery", "test"):
        build_torch_brain_config(
            workspace=workspace,
            dataset_id=selected_split_manifest.dataset_id,
            session_ids=[
                assignment.recording_id.split("/", 1)[1]
                for assignment in selected_split_manifest.assignments
                if assignment.split == split_name
            ],
            split=split_name,
            output_dir=selection_split_dir,
            filename_prefix="torch_brain_selected",
        )

    subject_count = len({record.subject_id for record in selected_catalog.records})
    return ResolvedDatasetView(
        prep_config=prep_config,
        workspace=workspace,
        session_catalog=selected_catalog,
        split_manifest=selected_split_manifest,
        split_manifest_path=selected_split_manifest_path,
        config_dir=selection_split_dir,
        config_name_prefix="torch_brain_selected",
        dataset_split=None,
        selection_active=True,
        selection_summary=DatasetSelectionSummary(
            output_name=output_name,
            session_count=len(selected_catalog.records),
            subject_count=subject_count,
            total_units=sum(record.n_units for record in selected_catalog.records),
            total_trials=sum(record.trial_count for record in selected_catalog.records),
            split_counts=_selection_counts(selected_split_manifest),
        ),
        session_catalog_path=selected_catalog_json,
    )


def resolve_runtime_dataset_view(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
) -> ResolvedDatasetView:
    prep_config = load_preparation_config(data_config_path)
    workspace = build_workspace(prep_config)
    if experiment_config.runtime_subset is not None:
        runtime_catalog = load_session_catalog(experiment_config.runtime_subset.session_catalog_path)
        return _resolved_runtime_subset_view(
            experiment_config=experiment_config,
            prep_config=prep_config,
            workspace=workspace,
            runtime_catalog=runtime_catalog,
        )
    if workspace.session_catalog_path.is_file():
        catalog = load_session_catalog(workspace.session_catalog_path)
    elif workspace.session_manifest_path.is_file():
        manifest = load_session_manifest(workspace.session_manifest_path)
        catalog = build_session_catalog_from_manifest(manifest)
        write_session_catalog(catalog, workspace.session_catalog_path)
        write_session_catalog_csv(catalog, workspace.session_catalog_csv_path)
    else:
        raise FileNotFoundError(
            "Session catalog not found. Run `pcc-prepare-data build-session-catalog` or the full "
            f"prepare command so {workspace.session_catalog_path} exists."
        )
    if experiment_config.dataset_selection.is_active:
        return materialize_runtime_selection(
            experiment_config=experiment_config,
            prep_config=prep_config,
            workspace=workspace,
            catalog=catalog,
        )
    if not workspace.split_manifest_path.is_file():
        raise FileNotFoundError(
            "Canonical split manifest not found. Run `pcc-prepare-data build-session-catalog` or the full "
            f"prepare command so {workspace.split_manifest_path} exists."
        )
    split_manifest = load_split_manifest(workspace.split_manifest_path)
    return ResolvedDatasetView(
        prep_config=prep_config,
        workspace=workspace,
        session_catalog=catalog,
        split_manifest=split_manifest,
        split_manifest_path=workspace.split_manifest_path,
        config_dir=workspace.splits,
        config_name_prefix="torch_brain",
        dataset_split=None,
        selection_active=False,
        selection_summary=None,
        session_catalog_path=workspace.session_catalog_path,
    )
