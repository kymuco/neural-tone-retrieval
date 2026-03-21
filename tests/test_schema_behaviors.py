from __future__ import annotations

import pytest

from neural_tone_retrieval import (
    ArtifactFormat,
    ArtifactRecord,
    ArtifactType,
    RenderRecord,
    SourceClipRecord,
    SplitName,
    StageType,
    build_content_split_assignments,
    create_chain_spec,
    create_chain_stage,
    create_registry,
    resolve_render_split,
)


def test_chain_spec_ids_are_deterministic() -> None:
    stages = [
        create_chain_stage(
            stage_type=StageType.OD,
            processor_id="ts9",
            params={"drive": 0.25, "tone": 0.55},
        ),
        create_chain_stage(
            stage_type=StageType.AMP,
            processor_id="5150",
            params={"gain": 0.72, "presence": 0.44},
        ),
    ]

    chain_a = create_chain_spec(
        chain_name="rhythm-tight",
        chain_family="modern_metal",
        stages=stages,
        amp_family="5150",
        gain_bucket="high",
    )
    chain_b = create_chain_spec(
        chain_name="rhythm-tight-duplicate-name-does-not-matter",
        chain_family="modern_metal",
        stages=stages,
        amp_family="5150",
        gain_bucket="high",
    )

    assert chain_a.chain_id == chain_b.chain_id


def test_content_group_split_propagates_to_render() -> None:
    source_clip = SourceClipRecord(
        artifact_id="artifact_source",
        content_group_id="riff_001",
    )
    render = RenderRecord(
        artifact_id="artifact_render",
        source_clip_id=source_clip.source_clip_id,
        chain_id="chain_001",
    )
    assignments = build_content_split_assignments({"riff_001": SplitName.TEST})

    split = resolve_render_split(render, [source_clip], assignments)

    assert split is SplitName.TEST


def test_registry_rejects_missing_render_dependencies() -> None:
    registry = create_registry(dataset_name="ntr-dev", dataset_version="0.1.0")
    render = RenderRecord(
        artifact_id="artifact_render",
        source_clip_id="clip_missing",
        chain_id="chain_missing",
    )

    with pytest.raises(KeyError):
        registry.add_render(render)


def test_registry_accepts_wired_records() -> None:
    registry = create_registry(dataset_name="ntr-dev", dataset_version="0.1.0")
    source_artifact = ArtifactRecord(
        artifact_type=ArtifactType.SOURCE_CLIP,
        uri="data/raw/source.wav",
        format=ArtifactFormat.WAV,
        artifact_id="artifact_source",
    )
    render_artifact = ArtifactRecord(
        artifact_type=ArtifactType.RENDERED_CLIP,
        uri="data/interim/render.wav",
        format=ArtifactFormat.WAV,
        artifact_id="artifact_render",
    )
    source_clip = SourceClipRecord(
        artifact_id=source_artifact.artifact_id,
        content_group_id="riff_001",
        source_clip_id="clip_001",
    )
    chain = create_chain_spec(
        chain_name="tight-rhythm",
        chain_family="modern_metal",
        stages=[
            create_chain_stage(stage_type=StageType.GATE, processor_id="gate"),
            create_chain_stage(stage_type=StageType.AMP, processor_id="5150"),
        ],
    )
    render = RenderRecord(
        artifact_id=render_artifact.artifact_id,
        source_clip_id=source_clip.source_clip_id,
        chain_id=chain.chain_id,
    )

    registry.add_artifact(source_artifact)
    registry.add_artifact(render_artifact)
    registry.add_source_clip(source_clip)
    registry.add_chain_spec(chain)
    registry.add_render(render)

    manifest = registry.to_manifest()

    assert manifest.summary()["renders"] == 1
