from __future__ import annotations

from pathlib import Path
import struct
import wave

from neural_tone_retrieval import (
    build_baseline_index,
    build_feature_manifest,
    build_render_manifest,
    create_chain_spec,
    create_chain_stage,
    create_dataset_manifest,
    FeatureSubjectType,
    ingest_dataset_directory,
    load_dataset_manifest,
    save_dataset_manifest,
    search_baseline_index,
    StageType,
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


def test_search_baseline_index_returns_render_match_for_rendered_index(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    wav_path = raw_dir / "clip.wav"
    write_square_wav(wav_path, sample_rate_hz=44_100, duration_sec=0.25, switch_every=8)
    wav_path.with_suffix(".json").write_text(
        '{"split": "train", "content_group_id": "riff_rendered_search"}',
        encoding="utf-8",
    )

    source_manifest = ingest_dataset_directory(
        raw_dir,
        dataset_name="ntr-rendered-search",
        dataset_version="1.3.0",
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
    render_features_manifest_path = tmp_path / "manifests" / "render_features_manifest.json"
    render_feature_manifest = build_feature_manifest(
        render_manifest,
        audio_root=render_manifest_path.parent,
        output_manifest_path=render_features_manifest_path,
        subject_type=FeatureSubjectType.RENDERED_CLIP,
    )
    render_index_manifest_path = tmp_path / "manifests" / "render_index_manifest.json"
    render_index_manifest = build_baseline_index(
        render_feature_manifest,
        manifest_root=render_features_manifest_path.parent,
        output_manifest_path=render_index_manifest_path,
        subject_type=FeatureSubjectType.RENDERED_CLIP,
    )

    render_record = render_manifest.renders[0]
    render_artifact = next(
        artifact for artifact in render_manifest.artifacts if artifact.artifact_id == render_record.artifact_id
    )
    render_audio_path = render_manifest_path.parent / render_artifact.uri
    query, hits = search_baseline_index(
        render_index_manifest,
        manifest_root=render_index_manifest_path.parent,
        query_audio_path=render_audio_path,
        top_k=1,
    )

    assert query.query_type.value == "external_audio"
    assert len(hits) == 1
    assert hits[0].candidate_render_id == render_record.render_id
    assert hits[0].chain_id == chain.chain_id
    assert hits[0].source_clip_id == render_record.source_clip_id
    assert hits[0].preview_uri == render_artifact.uri


def test_cli_rendered_index_build_and_search_query(tmp_path: Path, capsys) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    wav_path = raw_dir / "clip.wav"
    write_square_wav(wav_path, sample_rate_hz=44_100, duration_sec=0.25, switch_every=7)

    source_manifest = ingest_dataset_directory(
        raw_dir,
        dataset_name="ntr-cli-rendered-index",
        dataset_version="1.4.0",
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
    source_with_chains_path = tmp_path / "source_with_chains.json"
    save_dataset_manifest(source_with_chains, source_with_chains_path)
    render_manifest_path = tmp_path / "render_manifest.json"
    render_features_manifest_path = tmp_path / "render_features_manifest.json"
    render_index_manifest_path = tmp_path / "render_index_manifest.json"

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
                str(render_features_manifest_path),
                "--subject-type",
                FeatureSubjectType.RENDERED_CLIP.value,
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "index",
                "build",
                str(render_features_manifest_path),
                str(render_index_manifest_path),
                "--subject-type",
                FeatureSubjectType.RENDERED_CLIP.value,
            ]
        )
        == 0
    )
    build_output = capsys.readouterr().out
    assert "Index OK" in build_output

    render_manifest = load_dataset_manifest(render_manifest_path)
    render_record = render_manifest.renders[0]
    render_artifact = next(
        artifact for artifact in render_manifest.artifacts if artifact.artifact_id == render_record.artifact_id
    )
    render_audio_path = render_manifest_path.parent / render_artifact.uri

    assert (
        main(
            [
                "search",
                "query",
                str(render_index_manifest_path),
                str(render_audio_path),
                "--top-k",
                "1",
            ]
        )
        == 0
    )
    search_output = capsys.readouterr().out
    assert "Search OK" in search_output
    assert f"render_id={render_record.render_id}" in search_output
    assert f"chain_id={chain.chain_id}" in search_output


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
