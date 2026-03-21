from __future__ import annotations

import json
from pathlib import Path
import wave

from neural_tone_retrieval import (
    build_feature_manifest,
    extract_baseline_wav_features,
    ingest_dataset_directory,
    load_dataset_manifest,
)
from neural_tone_retrieval.cli import main


def test_extract_baseline_wav_features_returns_expected_keys(tmp_path: Path) -> None:
    wav_path = tmp_path / "dry.wav"
    write_test_wav(wav_path, sample_rate_hz=44_100, channels=1, duration_sec=0.5)

    features = extract_baseline_wav_features(wav_path)

    assert features["sample_rate_hz"] == 44_100
    assert features["frame_count"] == 22_050
    assert features["duration_sec"] == 0.5
    assert "spectral_centroid_hz" in features
    assert "spectral_rolloff_hz" in features
    assert "spectral_flatness" in features


def test_build_feature_manifest_writes_feature_artifacts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    wav_path = raw_dir / "riff.wav"
    write_test_wav(wav_path, sample_rate_hz=44_100, channels=1, duration_sec=0.5)
    input_manifest = ingest_dataset_directory(raw_dir, dataset_name="ntr-features", dataset_version="0.4.0")
    output_manifest_path = tmp_path / "manifests" / "features_manifest.json"

    feature_manifest = build_feature_manifest(
        input_manifest,
        audio_root=raw_dir,
        output_manifest_path=output_manifest_path,
    )

    assert feature_manifest.summary()["features"] == 1
    assert output_manifest_path.exists()
    feature_record = feature_manifest.features[0]
    feature_artifact = next(
        artifact for artifact in feature_manifest.artifacts if artifact.artifact_id == feature_record.artifact_id
    )
    feature_path = output_manifest_path.parent / feature_artifact.uri
    payload = json.loads(feature_path.read_text(encoding="utf-8"))
    assert payload["extractor_id"] == "baseline-handcrafted-v1"
    assert payload["source_clip_id"] == input_manifest.source_clips[0].source_clip_id
    assert "spectral_centroid_hz" in payload["features"]


def test_cli_features_extract_writes_feature_manifest(tmp_path: Path, capsys) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_test_wav(raw_dir / "clip.wav", sample_rate_hz=44_100, channels=1, duration_sec=0.25)
    source_manifest_path = tmp_path / "source_manifest.json"
    feature_manifest_path = tmp_path / "feature_manifest.json"

    ingest_dataset_directory(
        raw_dir,
        output_manifest_path=source_manifest_path,
        dataset_name="ntr-cli-features",
        dataset_version="0.5.0",
    )

    assert (
        main(
            [
                "features",
                "extract",
                str(source_manifest_path),
                str(raw_dir),
                str(feature_manifest_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Features OK" in output
    assert "features=1" in output

    manifest = load_dataset_manifest(feature_manifest_path)
    assert manifest.summary()["features"] == 1


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
