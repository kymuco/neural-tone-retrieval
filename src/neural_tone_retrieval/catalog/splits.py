"""Helpers for content-disjoint split assignment and lookup."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from neural_tone_retrieval.schemas.dataset import (
    RenderRecord,
    SourceClipRecord,
    SplitAssignment,
    SplitGroupType,
    SplitName,
)
from neural_tone_retrieval.settings import DEFAULT_SPLIT_PROTOCOL_ID


class SplitConflictError(ValueError):
    """Raised when split assignments disagree for the same group key."""


def build_content_split_assignments(
    group_to_split: Mapping[str, SplitName],
    *,
    split_protocol_id: str = DEFAULT_SPLIT_PROTOCOL_ID,
) -> tuple[SplitAssignment, ...]:
    return tuple(
        SplitAssignment(
            split_protocol_id=split_protocol_id,
            split_protocol_name=split_protocol_id,
            group_type=SplitGroupType.CONTENT_GROUP,
            group_id=group_id,
            split=split,
        )
        for group_id, split in sorted(group_to_split.items())
    )


def resolve_source_clip_split(
    source_clip: SourceClipRecord,
    split_assignments: Iterable[SplitAssignment],
    *,
    split_protocol_id: str = DEFAULT_SPLIT_PROTOCOL_ID,
) -> SplitName | None:
    split_index = _index_assignments(split_assignments, split_protocol_id=split_protocol_id)
    return split_index.get((SplitGroupType.CONTENT_GROUP, source_clip.content_group_id))


def resolve_render_split(
    render: RenderRecord,
    source_clips: Iterable[SourceClipRecord],
    split_assignments: Iterable[SplitAssignment],
    *,
    split_protocol_id: str = DEFAULT_SPLIT_PROTOCOL_ID,
) -> SplitName | None:
    source_clip_index = {clip.source_clip_id: clip for clip in source_clips}
    source_clip = source_clip_index.get(render.source_clip_id)
    if source_clip is None:
        raise KeyError(f"Unknown source_clip_id for render {render.render_id}: {render.source_clip_id}")
    return resolve_source_clip_split(
        source_clip,
        split_assignments,
        split_protocol_id=split_protocol_id,
    )


def _index_assignments(
    split_assignments: Iterable[SplitAssignment],
    *,
    split_protocol_id: str,
) -> dict[tuple[SplitGroupType, str], SplitName]:
    indexed: dict[tuple[SplitGroupType, str], SplitName] = {}
    for assignment in split_assignments:
        if assignment.split_protocol_id != split_protocol_id:
            continue
        key = (assignment.group_type, assignment.group_id)
        previous = indexed.get(key)
        if previous is not None and previous != assignment.split:
            raise SplitConflictError(f"Conflicting split assignment for key {key!r}")
        indexed[key] = assignment.split
    return indexed
