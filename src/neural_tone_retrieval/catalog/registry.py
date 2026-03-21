"""In-memory registry that validates and groups project records."""

from __future__ import annotations

from typing import TypeVar

from neural_tone_retrieval.catalog.manifests import DatasetManifest
from neural_tone_retrieval.schemas.artifacts import ArtifactRecord
from neural_tone_retrieval.schemas.chains import ChainSpec
from neural_tone_retrieval.schemas.dataset import RenderRecord, SourceClipRecord, SplitAssignment
from neural_tone_retrieval.schemas.features import FeatureRecord
from neural_tone_retrieval.schemas.retrieval import EmbeddingRecord
from neural_tone_retrieval.schemas.runs import RunRecord
from neural_tone_retrieval.settings import DEFAULT_DATASET_NAME, DEFAULT_DATASET_VERSION

T = TypeVar("T")


class DuplicateRecordError(ValueError):
    """Raised when a registry entry reuses an existing identity."""


class CatalogRegistry:
    """Mutable registry over an immutable manifest snapshot."""

    def __init__(
        self,
        manifest: DatasetManifest | None = None,
        *,
        dataset_name: str = DEFAULT_DATASET_NAME,
        dataset_version: str = DEFAULT_DATASET_VERSION,
    ) -> None:
        if manifest is None:
            manifest = DatasetManifest(dataset_name=dataset_name, dataset_version=dataset_version)
        self.dataset_name = manifest.dataset_name
        self.dataset_version = manifest.dataset_version

        self._artifacts = list(manifest.artifacts)
        self._source_clips = list(manifest.source_clips)
        self._chain_specs = list(manifest.chain_specs)
        self._renders = list(manifest.renders)
        self._features = list(manifest.features)
        self._embeddings = list(manifest.embeddings)
        self._split_assignments = list(manifest.split_assignments)
        self._runs = list(manifest.runs)

        self._artifact_index = {record.artifact_id: record for record in self._artifacts}
        self._source_clip_index = {record.source_clip_id: record for record in self._source_clips}
        self._chain_index = {record.chain_id: record for record in self._chain_specs}
        self._render_index = {record.render_id: record for record in self._renders}
        self._feature_index = {record.feature_id: record for record in self._features}
        self._embedding_index = {record.embedding_id: record for record in self._embeddings}
        self._split_index = {record.key: record for record in self._split_assignments}
        self._run_index = {record.run_id: record for record in self._runs}

        self._validate_existing_references()

    def to_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            dataset_name=self.dataset_name,
            dataset_version=self.dataset_version,
            artifacts=tuple(self._artifacts),
            source_clips=tuple(self._source_clips),
            chain_specs=tuple(self._chain_specs),
            renders=tuple(self._renders),
            features=tuple(self._features),
            embeddings=tuple(self._embeddings),
            split_assignments=tuple(self._split_assignments),
            runs=tuple(self._runs),
        )

    def add_artifact(self, record: ArtifactRecord) -> ArtifactRecord:
        self._add_unique(record.artifact_id, record, self._artifact_index, self._artifacts)
        return record

    def add_source_clip(self, record: SourceClipRecord) -> SourceClipRecord:
        self._require_known(record.artifact_id, self._artifact_index, "artifact_id")
        self._add_unique(record.source_clip_id, record, self._source_clip_index, self._source_clips)
        return record

    def add_chain_spec(self, record: ChainSpec) -> ChainSpec:
        self._add_unique(record.chain_id, record, self._chain_index, self._chain_specs)
        return record

    def add_render(self, record: RenderRecord) -> RenderRecord:
        self._require_known(record.artifact_id, self._artifact_index, "artifact_id")
        self._require_known(record.source_clip_id, self._source_clip_index, "source_clip_id")
        self._require_known(record.chain_id, self._chain_index, "chain_id")
        self._add_unique(record.render_id, record, self._render_index, self._renders)
        return record

    def add_feature(self, record: FeatureRecord) -> FeatureRecord:
        self._require_known(record.artifact_id, self._artifact_index, "artifact_id")
        self._require_known(record.subject_artifact_id, self._artifact_index, "subject_artifact_id")
        self._add_unique(record.feature_id, record, self._feature_index, self._features)
        return record

    def add_embedding(self, record: EmbeddingRecord) -> EmbeddingRecord:
        self._require_known(record.artifact_id, self._artifact_index, "artifact_id")
        self._require_known(record.subject_artifact_id, self._artifact_index, "subject_artifact_id")
        self._add_unique(record.embedding_id, record, self._embedding_index, self._embeddings)
        return record

    def add_split_assignment(self, record: SplitAssignment) -> SplitAssignment:
        key = record.key
        existing = self._split_index.get(key)
        if existing is not None and existing != record:
            raise DuplicateRecordError(f"Split assignment already exists for key {key!r}")
        if existing is None:
            self._split_assignments.append(record)
            self._split_index[key] = record
        return record

    def add_run(self, record: RunRecord) -> RunRecord:
        if record.config_artifact_id is not None:
            self._require_known(record.config_artifact_id, self._artifact_index, "config_artifact_id")
        self._add_unique(record.run_id, record, self._run_index, self._runs)
        return record

    def get_source_clip(self, source_clip_id: str) -> SourceClipRecord:
        return self._require_known(source_clip_id, self._source_clip_index, "source_clip_id")

    def get_chain_spec(self, chain_id: str) -> ChainSpec:
        return self._require_known(chain_id, self._chain_index, "chain_id")

    def get_render(self, render_id: str) -> RenderRecord:
        return self._require_known(render_id, self._render_index, "render_id")

    def _validate_existing_references(self) -> None:
        for record in self._source_clips:
            self._require_known(record.artifact_id, self._artifact_index, "artifact_id")
        for record in self._renders:
            self._require_known(record.artifact_id, self._artifact_index, "artifact_id")
            self._require_known(record.source_clip_id, self._source_clip_index, "source_clip_id")
            self._require_known(record.chain_id, self._chain_index, "chain_id")
        for record in self._features:
            self._require_known(record.artifact_id, self._artifact_index, "artifact_id")
            self._require_known(record.subject_artifact_id, self._artifact_index, "subject_artifact_id")
        for record in self._embeddings:
            self._require_known(record.artifact_id, self._artifact_index, "artifact_id")
            self._require_known(record.subject_artifact_id, self._artifact_index, "subject_artifact_id")
        for record in self._runs:
            if record.config_artifact_id is not None:
                self._require_known(record.config_artifact_id, self._artifact_index, "config_artifact_id")

    @staticmethod
    def _add_unique(record_id: str, record: object, index: dict[str, object], sink: list[object]) -> None:
        existing = index.get(record_id)
        if existing is not None and existing != record:
            raise DuplicateRecordError(f"Record id already exists: {record_id}")
        if existing is None:
            index[record_id] = record
            sink.append(record)

    @staticmethod
    def _require_known(record_id: str, index: dict[str, T], field_name: str) -> T:
        record = index.get(record_id)
        if record is None:
            raise KeyError(f"Unknown {field_name}: {record_id}")
        return record
