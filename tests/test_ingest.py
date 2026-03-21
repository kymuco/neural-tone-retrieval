from __future__ import annotations

import json
from pathlib import Path
import wave

from neural_tone_retrieval import ingest_dataset_directory, inspect_wav_file, load_dataset_manifest
from neural_tone_retrieval.cli import main


def test_inspect_wav_file_reads_basic_audio_metadata(tmp_path: Path) -> None:
    wav_path = tmp_path / "chugs.wav"
    write_test_wav(wav_path, sample_rate_hz=48_000, channels=2, duration_sec=0.5)

    info = inspect_wav_file(wav_path)

    assert info.sample_rate_hz == 48_000
    assert info.channels == 2
    assert info.frame_count == 24_000
    assert info.duration_sec == 0.5


def test_ingest_dataset_directory_builds_manifest_from_wavs_and_sidecars(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    wav_path = raw_dir / "tight_rhythm.wav"
    write_test_wav(wav_path, sample_rate_hz=44_100, channels=1, duration_sec=1.0)
    wav_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "content_group_id": "riff_001",
                "session_id": "session_a",
                "guitar_id": "guitar_rg",
                "pickup_position": "bridge",
                "tuning": "Drop_C",
                "technique_tags": ["palm_mute", "power_chords"],
                "bpm": 150.0,
                "license_ref": "project-owned",
                "split": "test",
            }
        ),
        encoding="utf-8",
    )

    manifest = ingest_dataset_directory(
        raw_dir,
        dataset_name="ntr-ingest",
        dataset_version="0.2.0",
    )

    assert manifest.dataset_name == "ntr-ingest"
    assert manifest.summary()["source_clips"] == 1
    assert manifest.summary()["split_assignments"] == 1
    source_clip = manifest.source_clips[0]
    assert source_clip.content_group_id == "riff_001"
    assert source_clip.tuning == "Drop_C"
    assert source_clip.technique_tags == ("palm_mute", "power_chords")
    assert source_clip.duration_sec == 1.0
    assert manifest.split_assignments[0].split.value == "test"


def test_cli_dataset_ingest_writes_manifest_file(tmp_path: Path, capsys) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_test_wav(raw_dir / "lead_take.wav", sample_rate_hz=44_100, channels=1, duration_sec=0.25)
    manifest_path = tmp_path / "manifest.json"

    assert (
        main(
            [
                "dataset",
                "ingest",
                str(raw_dir),
                str(manifest_path),
                "--dataset-name",
                "ntr-cli",
                "--dataset-version",
                "0.3.0",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Ingest OK" in output
    assert "source_clips=1" in output

    manifest = load_dataset_manifest(manifest_path)
    assert manifest.dataset_name == "ntr-cli"
    assert manifest.dataset_version == "0.3.0"


def write_test_wav(
    path: Path,
    *,
    sample_rate_hz: int,
    channels: int,
    duration_sec: float,
) -> None:
    frame_count = int(sample_rate_hz * duration_sec)
    silence_frame = (b"\x00\x00" * channels)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate_hz)
        stream.writeframes(silence_frame * frame_count)
