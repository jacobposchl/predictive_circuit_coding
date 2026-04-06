from __future__ import annotations

import numpy as np
import torch

from predictive_circuit_coding.tokenization import build_population_window_collator
from predictive_circuit_coding.training import DataRuntimeConfig


def _build_window_sample(*, session_id: str, subject_id: str, window_start_s: float, unit_ids: list[str], regions: list[str], depths: list[float], spikes: dict[int, list[float]]):
    from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

    domain = Interval(
        start=np.asarray([window_start_s], dtype=np.float64),
        end=np.asarray([window_start_s + 10.0], dtype=np.float64),
    )
    timestamps: list[float] = []
    unit_index: list[int] = []
    for index, values in spikes.items():
        timestamps.extend(values)
        unit_index.extend([index] * len(values))
    order = np.argsort(np.asarray(timestamps, dtype=np.float64))
    timestamps_np = np.asarray(timestamps, dtype=np.float64)[order]
    unit_index_np = np.asarray(unit_index, dtype=np.int64)[order]
    return Data(
        brainset=Data(id="allen_visual_behavior_neuropixels"),
        session=Data(id=session_id),
        subject=Data(id=subject_id),
        units=ArrayDict(
            id=np.asarray(unit_ids, dtype=object),
            brain_region=np.asarray(regions, dtype=object),
            probe_depth_um=np.asarray(depths, dtype=np.float32),
        ),
        spikes=IrregularTimeSeries(
            timestamps=timestamps_np,
            unit_index=unit_index_np,
            domain=domain,
        ),
        trials=Interval(
            start=np.asarray([window_start_s + 0.5], dtype=np.float64),
            end=np.asarray([window_start_s + 1.0], dtype=np.float64),
            go=np.asarray([True], dtype=bool),
            hit=np.asarray([True], dtype=bool),
            is_change=np.asarray([True], dtype=bool),
            catch=np.asarray([False], dtype=bool),
            miss=np.asarray([False], dtype=bool),
            false_alarm=np.asarray([False], dtype=bool),
            correct_reject=np.asarray([False], dtype=bool),
            aborted=np.asarray([False], dtype=bool),
            auto_rewarded=np.asarray([False], dtype=bool),
            response_time=np.asarray([0.25], dtype=np.float64),
            change_frame=np.asarray([12], dtype=np.int32),
        ),
        stimulus_presentations=Interval(
            start=np.asarray([window_start_s + 0.0], dtype=np.float64),
            end=np.asarray([window_start_s + 0.25], dtype=np.float64),
            stimulus_name=np.asarray(["images"], dtype=object),
            image_name=np.asarray(["im1"], dtype=object),
            is_change=np.asarray([False], dtype=bool),
            active=np.asarray([True], dtype=bool),
            rewarded=np.asarray([False], dtype=bool),
            omitted=np.asarray([False], dtype=bool),
            flashes_since_change=np.asarray([3], dtype=np.int32),
        ),
        optotagging=Interval(
            start=np.asarray([window_start_s + 0.75], dtype=np.float64),
            end=np.asarray([window_start_s + 0.76], dtype=np.float64),
            condition=np.asarray(["pulse"], dtype=object),
            level=np.asarray([0.8], dtype=np.float64),
            duration=np.asarray([0.01], dtype=np.float64),
            stimulus_name=np.asarray(["opto"], dtype=object),
        ),
        domain=domain,
    )


def _build_real_allen_named_window_sample(*, session_id: str, subject_id: str, window_start_s: float):
    from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

    domain = Interval(
        start=np.asarray([window_start_s], dtype=np.float64),
        end=np.asarray([window_start_s + 10.0], dtype=np.float64),
    )
    return Data(
        brainset=Data(id="allen_visual_behavior_neuropixels"),
        session=Data(id=session_id),
        subject=Data(id=subject_id),
        units=ArrayDict(
            id=np.asarray(["u0", "u1"], dtype=object),
            structure_acronym=np.asarray(["VISp", "LP"], dtype=object),
            probe_vertical_position=np.asarray([100.0, 200.0], dtype=np.float32),
        ),
        spikes=IrregularTimeSeries(
            timestamps=np.asarray([window_start_s + 0.1, window_start_s + 0.3], dtype=np.float64),
            unit_index=np.asarray([0, 1], dtype=np.int64),
            domain=domain,
        ),
        trials=Interval(
            start=np.asarray([window_start_s + 0.5], dtype=np.float64),
            end=np.asarray([window_start_s + 1.0], dtype=np.float64),
            go=np.asarray([True], dtype=bool),
        ),
        domain=domain,
    )


def test_population_window_collator_bins_counts_and_preserves_provenance():
    config = DataRuntimeConfig(
        bin_width_ms=20.0,
        context_bins=500,
        patch_bins=50,
        min_unit_spikes=0,
        max_units=None,
        padding_strategy="mask",
        include_trials=True,
        include_stimulus_presentations=True,
        include_optotagging=True,
    )
    collator = build_population_window_collator(config)
    sample_a = _build_window_sample(
        session_id="session_a",
        subject_id="mouse_a",
        window_start_s=5.0,
        unit_ids=["u0", "u1"],
        regions=["VISp", "LP"],
        depths=[100.0, 200.0],
        spikes={
            0: [5.005, 5.025, 6.005],
            1: [5.505],
        },
    )
    sample_b = _build_window_sample(
        session_id="session_b",
        subject_id="mouse_b",
        window_start_s=20.0,
        unit_ids=["u2", "u3", "u4"],
        regions=["VISl", "VISpm", "LP"],
        depths=[150.0, 250.0, 350.0],
        spikes={
            0: [20.005],
            1: [20.405],
            2: [21.205, 21.225],
        },
    )

    batch = collator([sample_a, sample_b])

    assert batch.counts.shape == (2, 3, 500)
    assert batch.patch_counts.shape == (2, 3, 10, 50)
    assert torch.equal(batch.unit_mask[0], torch.tensor([True, True, False]))
    assert batch.counts[0, 0, 0].item() == 1.0
    assert batch.counts[0, 0, 1].item() == 1.0
    assert batch.counts[0, 0, 50].item() == 1.0
    assert batch.counts[0, 1, 25].item() == 1.0
    assert batch.provenance.recording_ids[0] == "allen_visual_behavior_neuropixels/session_a"
    assert batch.provenance.unit_ids[0] == ("u0", "u1")
    assert batch.provenance.unit_regions[1] == ("VISl", "VISpm", "LP")
    assert batch.provenance.patch_start_s[0, 0].item() == 5.0
    assert batch.provenance.patch_end_s[0, 0].item() == 6.0
    assert "trials" in batch.provenance.event_annotations[0]
    assert "stimulus_presentations" in batch.provenance.event_annotations[0]
    assert batch.provenance.event_annotations[0]["trials"]["is_change"] == (True,)
    assert batch.provenance.event_annotations[0]["trials"]["change_frame"] == (12,)
    assert batch.provenance.event_annotations[0]["trials"]["response_time"] == (0.25,)
    assert batch.provenance.event_annotations[0]["stimulus_presentations"]["active"] == (True,)
    assert batch.provenance.event_annotations[0]["stimulus_presentations"]["flashes_since_change"] == (3,)
    assert batch.provenance.event_annotations[0]["optotagging"]["condition"] == ("pulse",)
    assert batch.provenance.event_annotations[0]["optotagging"]["level"] == (0.8,)


def test_population_window_collator_supports_real_allen_unit_field_names():
    config = DataRuntimeConfig(
        bin_width_ms=20.0,
        context_bins=500,
        patch_bins=50,
        min_unit_spikes=0,
        max_units=None,
        padding_strategy="mask",
        include_trials=True,
        include_stimulus_presentations=True,
        include_optotagging=True,
    )
    collator = build_population_window_collator(config)
    batch = collator([
        _build_real_allen_named_window_sample(
            session_id="session_real_names",
            subject_id="mouse_real_names",
            window_start_s=0.0,
        )
    ])

    assert batch.provenance.unit_ids[0] == ("u0", "u1")
    assert batch.provenance.unit_regions[0] == ("VISp", "LP")
    assert torch.equal(batch.provenance.unit_depth_um[0], torch.tensor([100.0, 200.0], dtype=torch.float32))
