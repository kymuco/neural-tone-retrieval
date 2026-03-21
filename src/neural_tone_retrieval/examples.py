"""Example manifests and configs shipped with the project source tree."""

from __future__ import annotations

from pathlib import Path

from neural_tone_retrieval.api import (
    ArtifactFormat,
    ArtifactRecord,
    ArtifactType,
    ControlledReampConfig,
    DatasetSection,
    FeatureSet,
    RenderSection,
    RetrievalSection,
    RunRecord,
    RunStatus,
    RunType,
    SplitName,
    StageType,
    build_content_split_assignments,
    create_chain_spec,
    create_chain_stage,
    create_dataset_manifest,
)
from neural_tone_retrieval.serde import render_controlled_reamp_config_toml, save_dataset_manifest


def build_example_dataset_manifest() -> object:
    source_a_artifact = ArtifactRecord(
        artifact_type=ArtifactType.SOURCE_CLIP,
        uri="data/raw/di/chugs_e_standard_01.wav",
        format=ArtifactFormat.WAV,
        artifact_id="artifact_di_chugs_e_standard_01",
    )
    source_b_artifact = ArtifactRecord(
        artifact_type=ArtifactType.SOURCE_CLIP,
        uri="data/raw/di/lead_drop_d_01.wav",
        format=ArtifactFormat.WAV,
        artifact_id="artifact_di_lead_drop_d_01",
    )
    source_a = create_source_clip(
        artifact_id=source_a_artifact.artifact_id,
        content_group_id="riff_chugs_e_standard_01",
        source_clip_id="clip_chugs_e_standard_01",
        session_id="session_2026_03_21_a",
        guitar_id="guitar_ibanez_rg",
        pickup_position="bridge",
        tuning="E_standard",
        technique_tags=("palm_mute", "power_chords"),
        bpm=140.0,
        duration_sec=4.2,
        sample_rate_hz=44_100,
        channels=1,
        license_ref="project-owned",
    )
    source_b = create_source_clip(
        artifact_id=source_b_artifact.artifact_id,
        content_group_id="riff_lead_drop_d_01",
        source_clip_id="clip_lead_drop_d_01",
        session_id="session_2026_03_21_b",
        guitar_id="guitar_solar_a1",
        pickup_position="bridge",
        tuning="Drop_D",
        technique_tags=("lead", "alternate_picking"),
        bpm=110.0,
        duration_sec=5.1,
        sample_rate_hz=44_100,
        channels=1,
        license_ref="project-owned",
    )
    chain_rhythm = create_chain_spec(
        chain_name="tight-rhythm",
        chain_family="modern_metal",
        amp_family="5150",
        cab_family="mesa_v30",
        ir_id="ir_mesa_v30_sm57",
        gain_bucket="high",
        brightness_bucket="balanced",
        fx_tags=("gate", "od", "eq"),
        stages=[
            create_chain_stage(
                stage_type=StageType.GATE,
                processor_id="fortin_gate",
                params={"threshold_db": -48},
            ),
            create_chain_stage(
                stage_type=StageType.OD,
                processor_id="ts9",
                params={"drive": 0.12, "tone": 0.58, "level": 0.83},
            ),
            create_chain_stage(
                stage_type=StageType.AMP,
                processor_id="amp_5150_block_letter",
                params={"gain": 0.68, "bass": 0.46, "mid": 0.39, "treble": 0.62},
            ),
            create_chain_stage(
                stage_type=StageType.CAB,
                processor_id="ir_loader",
                params={"ir_id": "ir_mesa_v30_sm57", "mix": 1.0},
            ),
            create_chain_stage(
                stage_type=StageType.EQ,
                processor_id="post_eq",
                params={"low_cut_hz": 80, "high_cut_hz": 10_000},
            ),
        ],
        notes="Reference rhythm chain for tight modern metal tones.",
    )
    chain_lead = create_chain_spec(
        chain_name="saturated-lead",
        chain_family="modern_lead",
        amp_family="5150",
        cab_family="mesa_v30",
        ir_id="ir_mesa_v30_r121",
        gain_bucket="high",
        brightness_bucket="bright",
        fx_tags=("gate", "od", "reverb"),
        stages=[
            create_chain_stage(
                stage_type=StageType.GATE,
                processor_id="fortin_gate",
                params={"threshold_db": -44},
            ),
            create_chain_stage(
                stage_type=StageType.OD,
                processor_id="ts9",
                params={"drive": 0.18, "tone": 0.63, "level": 0.79},
            ),
            create_chain_stage(
                stage_type=StageType.AMP,
                processor_id="amp_5150_block_letter",
                params={"gain": 0.74, "bass": 0.43, "mid": 0.48, "treble": 0.65},
            ),
            create_chain_stage(
                stage_type=StageType.CAB,
                processor_id="ir_loader",
                params={"ir_id": "ir_mesa_v30_r121", "mix": 1.0},
            ),
            create_chain_stage(
                stage_type=StageType.REVERB,
                processor_id="plate_reverb",
                params={"mix": 0.12, "decay_sec": 1.8},
            ),
        ],
        notes="Lead chain with slightly brighter voicing and room for sustain.",
    )
    assignments = build_content_split_assignments(
        {
            source_a.content_group_id: SplitName.TRAIN,
            source_b.content_group_id: SplitName.TEST,
        }
    )
    run = RunRecord(
        run_type=RunType.INGEST,
        run_id="run_example_ingest_v1",
        status=RunStatus.COMPLETED,
        inputs_json={"source": "repo_example_bundle"},
        outputs_json={"registered_source_clips": 2, "registered_chain_specs": 2},
        metrics_json={"duration_sec": 0.5},
    )
    return create_dataset_manifest(
        dataset_name="neural-tone-retrieval-dev",
        dataset_version="0.1.0",
        artifacts=(source_a_artifact, source_b_artifact),
        source_clips=(source_a, source_b),
        chain_specs=(chain_rhythm, chain_lead),
        split_assignments=assignments,
        runs=(run,),
    )


def build_example_controlled_reamp_config(
    *,
    manifest_uri: str = "./example_dataset_manifest.json",
    raw_di_root: str = "../../data/raw",
    output_root: str = "../../artifacts/dev",
) -> ControlledReampConfig:
    return ControlledReampConfig(
        dataset=DatasetSection(
            name="neural-tone-retrieval-dev",
            version="0.1.0",
            manifest_uri=manifest_uri,
            raw_di_root=raw_di_root,
            output_root=output_root,
            selected_splits=(SplitName.TRAIN, SplitName.VAL, SplitName.TEST),
        ),
        render=RenderSection(
            target_sample_rate_hz=44_100,
            normalize_input=True,
            peak_target_dbfs=-1.0,
            tail_sec=0.25,
            include_chain_ids=(),
        ),
        retrieval=RetrievalSection(
            feature_sets=(FeatureSet.MEL, FeatureSet.MFCC, FeatureSet.SPECTRAL),
            embedding_model_id="baseline-handcrafted-v1",
            top_k=5,
        ),
        notes="Example controlled re-amp and retrieval bootstrap config.",
    )


def write_example_bundle(destination: str | Path, *, force: bool = False) -> dict[str, Path]:
    destination_path = Path(destination)
    if destination_path.exists() and any(destination_path.iterdir()) and not force:
        raise FileExistsError(
            f"Destination already exists and is not empty: {destination_path}"
        )
    destination_path.mkdir(parents=True, exist_ok=True)

    manifest_path = destination_path / "example_dataset_manifest.json"
    config_path = destination_path / "controlled_reamp.toml"

    manifest = build_example_dataset_manifest()
    config = build_example_controlled_reamp_config()

    save_dataset_manifest(manifest, manifest_path)
    config_path.write_text(
        render_controlled_reamp_config_toml(config),
        encoding="utf-8",
    )

    return {
        "manifest_path": manifest_path,
        "config_path": config_path,
    }


def create_source_clip(**kwargs: object):
    from neural_tone_retrieval.api import SourceClipRecord

    return SourceClipRecord(**kwargs)
