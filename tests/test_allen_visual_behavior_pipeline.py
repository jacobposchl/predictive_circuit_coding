from __future__ import annotations

from argparse import Namespace
import importlib.util
from pathlib import Path

import pandas as pd

from predictive_circuit_coding.data import load_temporaldata_session


def _load_pipeline_module():
    path = Path("brainsets_local_pipelines/allen_visual_behavior_neuropixels/pipeline.py").resolve()
    spec = importlib.util.spec_from_file_location("allen_visual_behavior_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_visual_behavior_manifest_uses_session_table_and_subset_filters(tmp_path, monkeypatch):
    module = _load_pipeline_module()

    class FakeCache:
        def get_ecephys_session_table(self):
            frame = pd.DataFrame(
                [
                    {"ecephys_session_id": 101, "mouse_id": "m1"},
                    {"ecephys_session_id": 102, "mouse_id": "m2"},
                    {"ecephys_session_id": 103, "mouse_id": "m3"},
                ]
            )
            return frame.set_index("ecephys_session_id")

    session_ids_file = tmp_path / "session_ids.txt"
    session_ids_file.write_text("102\n103\n", encoding="utf-8")
    monkeypatch.setattr(module, "_build_cache", lambda raw_dir: FakeCache())

    manifest = module.Pipeline.get_manifest(
        raw_dir=tmp_path / "raw",
        args=Namespace(session_ids_file=session_ids_file, max_sessions=1),
    )

    assert list(manifest["session_id"]) == ["102"]
    assert list(manifest.index) == ["102"]


def test_visual_behavior_process_writes_expected_temporaldata_payload(tmp_path):
    module = _load_pipeline_module()

    class FakeSession:
        metadata = {
            "mouse_id": "544456",
            "sex": "M",
            "age_in_days": 120,
            "date_of_acquisition": "2020-11-19 23:18:01+00:00",
            "equipment_name": "NP.1",
            "full_genotype": "Sst-IRES-Cre/wt;Ai32/wt",
        }

        get_units_calls = []

        def get_units(self, **kwargs):
            self.get_units_calls.append(kwargs)
            frame = pd.DataFrame(
                {
                    "peak_channel_id": [11, 12, 13],
                    "snr": [2.5, 1.8, 1.0],
                    "firing_rate": [5.0, 3.0, 1.0],
                    "isi_violations": [0.1, 0.2, 0.8],
                    "presence_ratio": [0.99, 0.97, 0.50],
                    "amplitude_cutoff": [0.05, 0.08, 0.20],
                    "quality": ["good", "good", "noise"],
                    "probe_id": [1, 1, 1],
                },
                index=pd.Index([1001, 1002, 1003], name="unit_id"),
            )
            if kwargs.get("filter_by_validity", False):
                frame = frame.loc[frame["quality"] == "good"]
            amplitude_cutoff_maximum = kwargs.get("amplitude_cutoff_maximum")
            if amplitude_cutoff_maximum is not None:
                frame = frame.loc[frame["amplitude_cutoff"] <= amplitude_cutoff_maximum]
            presence_ratio_minimum = kwargs.get("presence_ratio_minimum")
            if presence_ratio_minimum is not None:
                frame = frame.loc[frame["presence_ratio"] >= presence_ratio_minimum]
            isi_violations_maximum = kwargs.get("isi_violations_maximum")
            if isi_violations_maximum is not None:
                frame = frame.loc[frame["isi_violations"] <= isi_violations_maximum]
            return frame

        def get_channels(self, **kwargs):
            return pd.DataFrame(
                {
                    "structure_acronym": ["VISp", "LP", "root"],
                    "probe_vertical_position": [1200.0, 900.0, 800.0],
                    "probe_horizontal_position": [10.0, 20.0, 30.0],
                },
                index=pd.Index([11, 12, 13], name="channel_id"),
            )

        spike_times = {
            1001: [0.1, 0.4, 0.9],
            1002: [0.2, 0.5, 1.0],
            1003: [0.3, 0.6],
        }

        stimulus_presentations = pd.DataFrame(
            {
                "start_time": [0.0, 0.5],
                "stop_time": [0.25, 0.75],
                "active": [True, True],
                "is_change": [False, True],
                "stimulus_name": ["images", "images"],
                "image_name": ["im1", "im2"],
            }
        )

        trials = pd.DataFrame(
            {
                "start_time": [0.0],
                "stop_time": [1.1],
                "go": [True],
                "hit": [True],
            }
        )

        optotagging_table = pd.DataFrame(
            {
                "start_time": [0.8],
                "stop_time": [0.81],
                "condition": ["pulse"],
                "level": [0.8],
                "duration": [0.01],
            }
        )

    pipeline = module.Pipeline(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        args=Namespace(
            reprocess=False,
            filter_by_validity=True,
            filter_out_of_brain_units=True,
            amplitude_cutoff_maximum=0.1,
            presence_ratio_minimum=0.95,
            isi_violations_maximum=0.5,
        ),
    )
    fake_session = FakeSession()
    pipeline.process((fake_session, {"session_id": "1064644573"}))

    output_path = tmp_path / "processed" / "1064644573.h5"
    assert output_path.is_file()

    session = load_temporaldata_session(output_path, lazy=False)
    assert fake_session.get_units_calls[0]["filter_by_validity"] is True
    assert fake_session.get_units_calls[0]["filter_out_of_brain_units"] is True
    assert fake_session.get_units_calls[0]["amplitude_cutoff_maximum"] == 0.1
    assert fake_session.get_units_calls[0]["presence_ratio_minimum"] == 0.95
    assert fake_session.get_units_calls[0]["isi_violations_maximum"] == 0.5
    assert session.session.id == "1064644573"
    assert session.subject.id == "544456"
    assert len(session.units.id) == 2
    assert list(session.units.structure_acronym.astype(str)) == ["VISp", "LP"]
    assert list(session.units.presence_ratio.astype(float)) == [0.99, 0.97]
    assert list(session.units.amplitude_cutoff.astype(float)) == [0.05, 0.08]
    assert list(session.units.isi_violations.astype(float)) == [0.1, 0.2]
    assert len(session.stimulus_presentations.start) == 2
    assert len(session.trials.start) == 1
    assert len(session.optotagging.start) == 1
