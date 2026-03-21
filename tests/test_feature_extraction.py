from __future__ import annotations

import json
from pathlib import Path
import struct
import wave

from neural_tone_retrieval import (
    build_feature_manifest,
    build_render_manifest,
    create_chain_spec,
    create_chain_stage,
    create_dataset_manifest,
    extract_baseline_wav_features,
    FeatureSubjectType,
    ingest_dataset_directory,
    load_dataset_manifest,
    save_dataset_manifest,
    StageType,
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


def test_build_feature_manifest_writes_render_feature_artifacts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    wav_path = raw_dir / "riff.wav"
    write_square_wav(wav_path, sample_rate_hz=44_100, duration_sec=0.25, switch_every=12)
    wav_path.with_suffix(".json").write_text(
        json.dumps({"split": "train", "content_group_id": "riff_rendered_001"}),
        encoding="utf-8",
    )

    source_manifest = ingest_dataset_directory(
        raw_dir,
        dataset_name="ntr-rendered-features",
        dataset_version="1.1.0",
    )
    chain = create_test_chain()
    source_with_chains = create_dataset_manifest(
        dataset_name=source_manifest.dataset_name,
        dataset_version=source_manifest.dataset_version,
        artifacts=source_manifest.artifacts,
        source_clips=source_manifest.source_clips,
        chain_specs=(chain,),
        split_assignments=source_manifest.split_assignments,
        runs=source_manifest.runs,
    )
    render_manifest_path = tmp_path / "manifests" / "render_manifest.json"
    render_manifest = build_render_manifest(
        source_with_chains,
        audio_root=raw_dir,
        output_manifest_path=render_manifest_path,
    )
    feature_manifest_path = tmp_path / "manifests" / "render_features_manifest.json"

    feature_manifest = build_feature_manifest(
        render_manifest,
        audio_root=render_manifest_path.parent,
        output_manifest_path=feature_manifest_path,
        subject_type=FeatureSubjectType.RENDERED_CLIP,
    )

    assert feature_manifest.summary()["features"] == 1
    feature_record = feature_manifest.features[0]
    assert feature_record.subject_type == FeatureSubjectType.RENDERED_CLIP
    assert feature_record.split is not None and feature_record.split.value == "train"
    feature_artifact = next(
        artifact for artifact in feature_manifest.artifacts if artifact.artifact_id == feature_record.artifact_id
    )
    payload = json.loads((feature_manifest_path.parent / feature_artifact.uri).read_text(encoding="utf-8"))
    assert payload["render_id"] == render_manifest.renders[0].render_id
    assert payload["chain_id"] == chain.chain_id
    assert payload["subject_type"] == FeatureSubjectType.RENDERED_CLIP.value


def test_cli_features_extract_writes_render_feature_manifest(tmp_path: Path, capsys) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    wav_path = raw_dir / "clip.wav"
    write_square_wav(wav_path, sample_rate_hz=44_100, duration_sec=0.2, switch_every=10)

    source_manifest = ingest_dataset_directory(
        raw_dir,
        dataset_name="ntr-cli-rendered-features",
        dataset_version="1.2.0",
    )
    source_with_chains = create_dataset_manifest(
        dataset_name=source_manifest.dataset_name,
        dataset_version=source_manifest.dataset_version,
        artifacts=source_manifest.artifacts,
        source_clips=source_manifest.source_clips,
        chain_specs=(create_test_chain(),),
        split_assignments=source_manifest.split_assignments,
        runs=source_manifest.runs,
    )
    source_with_chains_path = tmp_path / "source_with_chains.json"
    save_dataset_manifest(source_with_chains, source_with_chains_path)
    render_manifest_path = tmp_path / "render_manifest.json"
    feature_manifest_path = tmp_path / "render_features_manifest.json"

    assert (
        main(
            [
                "render",
                "build",
                str(source_with_chains_path),
                str(raw_dir),
                str(render_manifest_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "features",
                "extract",
                str(render_manifest_path),
                str(render_manifest_path.parent),
                str(feature_manifest_path),
                "--subject-type",
                FeatureSubjectType.RENDERED_CLIP.value,
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Features OK" in output
    assert "features=1" in output

    manifest = load_dataset_manifest(feature_manifest_path)
    assert manifest.features[0].subject_type == FeatureSubjectType.RENDERED_CLIP


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


def write_square_wav(
    path: Path,
    *,
    sample_rate_hz: int,
    duration_sec: float,
    switch_every: int,
    amplitude: int = 10_000,
) -> None:
    frame_count = int(sample_rate_hz * duration_sec)
    samples = bytearray()
    for index in range(frame_count):
        sample = amplitude if ((index // switch_every) % 2 == 0) else -amplitude
        samples.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate_hz)
        stream.writeframes(bytes(samples))


def create_test_chain():
    return create_chain_spec(
        chain_name="tight-rhythm",
        chain_family="modern_metal",
        stages=[
            create_chain_stage(
                stage_type=StageType.GATE,
                processor_id="gate",
                params={"threshold_db": -42.0},
            ),
            create_chain_stage(
                stage_type=StageType.OD,
                processor_id="ts9",
                params={"drive": 0.3, "tone": 0.65, "level": 0.85},
            ),
            create_chain_stage(
                stage_type=StageType.AMP,
                processor_id="5150",
                params={"gain": 0.7, "bass": 0.45, "mid": 0.4, "treble": 0.62},
            ),
            create_chain_stage(
                stage_type=StageType.CAB,
                processor_id="ir_loader",
                params={"ir_id": "mesa_v30_sm57"},
            ),
            create_chain_stage(
                stage_type=StageType.EQ,
                processor_id="post_eq",
                params={"low_cut_hz": 90.0, "high_cut_hz": 9500.0},
            ),
        ],
    )
