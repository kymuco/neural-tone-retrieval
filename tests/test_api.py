from __future__ import annotations

from neural_tone_retrieval import (
    ArtifactFormat,
    ArtifactRecord,
    ArtifactType,
    CatalogRegistry,
    DatasetManifest,
    SplitName,
    create_dataset_manifest,
    create_registry,
)


def test_public_exports_are_available() -> None:
    artifact = ArtifactRecord(
        artifact_type=ArtifactType.SOURCE_CLIP,
        uri="data/example.wav",
        format=ArtifactFormat.WAV,
    )
    manifest = create_dataset_manifest(
        dataset_name="ntr-dev",
        dataset_version="0.1.0",
        artifacts=[artifact],
    )

    assert isinstance(manifest, DatasetManifest)
    assert manifest.summary()["artifacts"] == 1
    assert SplitName.TEST == "test"


def test_create_registry_uses_manifest_metadata() -> None:
    manifest = create_dataset_manifest(
        dataset_name="ntr-dev",
        dataset_version="0.1.0",
    )

    registry = create_registry(manifest=manifest)

    assert isinstance(registry, CatalogRegistry)
    assert registry.dataset_name == "ntr-dev"
