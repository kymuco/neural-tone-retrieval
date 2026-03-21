from __future__ import annotations

from pathlib import Path
import struct
import wave

from neural_tone_retrieval import (
    build_baseline_index,
    build_feature_manifest,
    ingest_dataset_directory,
    load_dataset_manifest,
    search_baseline_index,
)
from neural_tone_retrieval.cli import main


def test_build_baseline_index_writes_index_artifact(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    clip_a = raw_dir / "clip_a.wav"
    clip_b = raw_dir / "clip_b.wav"
    write_square_wav(clip_a, sample_rate_hz=44_100, duration_sec=0.5, switch_every=10)
    write_square_wav(clip_b, sample_rate_hz=44_100, duration_sec=0.5, switch_every=120)

    source_manifest = ingest_dataset_directory(raw_dir, dataset_name="ntr-index", dataset_version="0.6.0")
    features_manifest_path = tmp_path / "manifests" / "features_manifest.json"
    feature_manifest = build_feature_manifest(
        source_manifest,
        audio_root=raw_dir,
        output_manifest_path=features_manifest_path,
    )
    index_manifest_path = tmp_path / "manifests" / "index_manifest.json"

    index_manifest = build_baseline_index(
        feature_manifest,
        manifest_root=features_manifest_path.parent,
        output_manifest_path=index_manifest_path,
    )

    index_artifacts = [artifact for artifact in index_manifest.artifacts if artifact.artifact_type == "index"]
    assert len(index_artifacts) == 1
    assert index_manifest_path.exists()


def test_search_baseline_index_returns_nearest_source_clip(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    clip_a = raw_dir / "clip_a.wav"
    clip_b = raw_dir / "clip_b.wav"
    write_square_wav(clip_a, sample_rate_hz=44_100, duration_sec=0.5, switch_every=8)
    write_square_wav(clip_b, sample_rate_hz=44_100, duration_sec=0.5, switch_every=160)

    source_manifest = ingest_dataset_directory(raw_dir, dataset_name="ntr-search", dataset_version="0.7.0")
    features_manifest_path = tmp_path / "manifests" / "features_manifest.json"
    feature_manifest = build_feature_manifest(
        source_manifest,
        audio_root=raw_dir,
        output_manifest_path=features_manifest_path,
    )
    index_manifest_path = tmp_path / "manifests" / "index_manifest.json"
    index_manifest = build_baseline_index(
        feature_manifest,
        manifest_root=features_manifest_path.parent,
        output_manifest_path=index_manifest_path,
    )

    query, hits = search_baseline_index(
        index_manifest,
        manifest_root=index_manifest_path.parent,
        query_audio_path=clip_a,
        top_k=2,
    )

    assert query.query_type.value == "external_audio"
    assert len(hits) == 2
    assert hits[0].source_clip_id == source_manifest.source_clips[0].source_clip_id
    assert hits[0].distance <= hits[1].distance


def test_cli_index_build_and_search_query(tmp_path: Path, capsys) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    clip_a = raw_dir / "clip_a.wav"
    clip_b = raw_dir / "clip_b.wav"
    write_square_wav(clip_a, sample_rate_hz=44_100, duration_sec=0.25, switch_every=6)
    write_square_wav(clip_b, sample_rate_hz=44_100, duration_sec=0.25, switch_every=90)

    source_manifest_path = tmp_path / "source_manifest.json"
    feature_manifest_path = tmp_path / "features_manifest.json"
    index_manifest_path = tmp_path / "index_manifest.json"

    ingest_dataset_directory(
        raw_dir,
        output_manifest_path=source_manifest_path,
        dataset_name="ntr-cli-index",
        dataset_version="0.8.0",
    )
    source_manifest = load_dataset_manifest(source_manifest_path)
    build_feature_manifest(
        source_manifest,
        audio_root=raw_dir,
        output_manifest_path=feature_manifest_path,
    )

    assert (
        main(
            [
                "index",
                "build",
                str(feature_manifest_path),
                str(index_manifest_path),
            ]
        )
        == 0
    )
    build_output = capsys.readouterr().out
    assert "Index OK" in build_output

    assert (
        main(
            [
                "search",
                "query",
                str(index_manifest_path),
                str(clip_a),
                "--top-k",
                "2",
            ]
        )
        == 0
    )
    search_output = capsys.readouterr().out
    assert "Search OK" in search_output
    assert "rank=1" in search_output
    assert "source_clip_id=" in search_output


def write_square_wav(
    path: Path,
    *,
    sample_rate_hz: int,
    duration_sec: float,
    switch_every: int,
    amplitude: int = 12_000,
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
