from __future__ import annotations

import json
from pathlib import Path
import struct
import wave

from neural_tone_retrieval import (
    build_render_manifest,
    create_chain_spec,
    create_chain_stage,
    create_dataset_manifest,
    ingest_dataset_directory,
    load_dataset_manifest,
    save_dataset_manifest,
    StageType,
)
from neural_tone_retrieval.cli import main
from neural_tone_retrieval.feature_extraction import load_wav_as_mono_floats


def test_build_render_manifest_writes_rendered_audio(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    wav_path = raw_dir / "riff.wav"
    write_square_wav(wav_path, sample_rate_hz=44_100, duration_sec=0.25, switch_every=12)
    wav_path.with_suffix(".json").write_text(
        json.dumps({"split": "train", "content_group_id": "riff_001"}),
        encoding="utf-8",
    )

    source_manifest = ingest_dataset_directory(raw_dir, dataset_name="ntr-render", dataset_version="0.9.0")
    chain = create_test_chain()
    manifest = create_dataset_manifest(
        dataset_name=source_manifest.dataset_name,
        dataset_version=source_manifest.dataset_version,
        artifacts=source_manifest.artifacts,
        source_clips=source_manifest.source_clips,
        chain_specs=(chain,),
        split_assignments=source_manifest.split_assignments,
        runs=source_manifest.runs,
    )
    output_manifest_path = tmp_path / "manifests" / "render_manifest.json"

    render_manifest = build_render_manifest(
        manifest,
        audio_root=raw_dir,
        output_manifest_path=output_manifest_path,
    )

    assert render_manifest.summary()["renders"] == 1
    render_record = render_manifest.renders[0]
    assert render_record.split is not None and render_record.split.value == "train"
    render_artifact = next(
        artifact for artifact in render_manifest.artifacts if artifact.artifact_id == render_record.artifact_id
    )
    render_path = output_manifest_path.parent / render_artifact.uri
    assert render_path.exists()

    source_samples, _ = load_wav_as_mono_floats(wav_path)
    render_samples, _ = load_wav_as_mono_floats(render_path)
    assert render_samples != source_samples


def test_cli_render_build_writes_manifest(tmp_path: Path, capsys) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    wav_path = raw_dir / "clip.wav"
    write_square_wav(wav_path, sample_rate_hz=44_100, duration_sec=0.2, switch_every=10)

    source_manifest = ingest_dataset_directory(raw_dir, dataset_name="ntr-render-cli", dataset_version="1.0.0")
    chain = create_test_chain()
    manifest = create_dataset_manifest(
        dataset_name=source_manifest.dataset_name,
        dataset_version=source_manifest.dataset_version,
        artifacts=source_manifest.artifacts,
        source_clips=source_manifest.source_clips,
        chain_specs=(chain,),
        split_assignments=source_manifest.split_assignments,
        runs=source_manifest.runs,
    )
    source_manifest_path = tmp_path / "source_with_chains.json"
    save_dataset_manifest(manifest, source_manifest_path)
    render_manifest_path = tmp_path / "render_manifest.json"

    assert (
        main(
            [
                "render",
                "build",
                str(source_manifest_path),
                str(raw_dir),
                str(render_manifest_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Render OK" in output
    assert "renders=1" in output

    render_manifest = load_dataset_manifest(render_manifest_path)
    assert render_manifest.summary()["renders"] == 1


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
