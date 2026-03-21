# Neural Tone Retrieval

> Learn guitar tone embeddings from controlled re-amped DI data for search, comparison, and recommendation of signal chains.

Neural Tone Retrieval is an applied ML project focused on guitar tone search. The core idea is to build a controlled dataset from dry DI guitar phrases, re-amp them through known signal chains, and train retrieval models that can map a query audio clip to the closest tones, presets, or chain families.

## Why This Exists

Most guitar audio found online has weak or missing ground truth. The amp, cab, IR, EQ, post-processing, and recording chain are often unknown, which makes supervised learning noisy and hard to trust. This project takes the opposite approach:

- build a controlled re-amped dataset
- preserve exact chain provenance and metadata
- train retrieval models that focus on tone rather than riff identity
- evaluate results with reproducible metrics and splits

## Project Scope

The initial scope is intentionally narrow and engineering-heavy:

- offline dataset generation from dry DI guitar clips
- declarative signal-chain specifications with full metadata
- retrieval baselines using audio features and nearest-neighbor search
- learned audio embeddings for tone similarity
- reproducible evaluation and experiment tracking

Out of scope for v1:

- real-time plugin hosting
- exact knob-by-knob preset reconstruction
- redistribution of commercial plugins, IR libraries, or restricted audio packs

## Repository Layout

```text
.
|-- README.md
|-- LICENSE
|-- DATA_LICENSE.md
|-- THIRD_PARTY.md
|-- docs/
|-- scripts/
|-- src/
|   `-- neural_tone_retrieval/
|       |-- api.py
|       |-- settings.py
|       |-- utils.py
|       |-- catalog/
|       `-- schemas/
|-- tests/
`-- data/
```

Some directories such as `data/`, `artifacts/`, `runs/`, and `models/` are expected to stay untracked by default unless there is a clear reason to version small example assets.

## Implemented V0 Core

The repository now includes the first schema and registry layer:

- `schemas/` for artifacts, dataset records, chain specs, retrieval records, and run provenance
- `catalog/` for manifests, split propagation, and in-memory validation/indexing
- `api.py` as the stable public Python surface for future CLI and service layers

## Controlled Re-Amp Bootstrap

The repository now includes:

- tracked example manifests and configs for a controlled re-amp workflow
- JSON serialization for dataset manifests
- TOML configuration loading for controlled re-amp and retrieval runs
- a thin `ntr` CLI for example generation and validation

Example commands:

```bash
python -m neural_tone_retrieval init-example examples/controlled_reamp
python -m neural_tone_retrieval manifest show examples/controlled_reamp/example_dataset_manifest.json
python -m neural_tone_retrieval config show examples/controlled_reamp/controlled_reamp.toml
python -m neural_tone_retrieval dataset ingest data/raw manifests/source_manifest.json
python -m neural_tone_retrieval render build manifests/source_with_chains.json data/raw manifests/render_manifest.json
python -m neural_tone_retrieval features extract manifests/source_manifest.json data/raw manifests/features_manifest.json
python -m neural_tone_retrieval index build manifests/features_manifest.json manifests/index_manifest.json
python -m neural_tone_retrieval search query manifests/index_manifest.json query.wav --top-k 5
```

Optional sidecar metadata can live next to a WAV file as `clip.json` and may include fields such as:

```json
{
  "content_group_id": "riff_001",
  "session_id": "session_a",
  "guitar_id": "guitar_rg",
  "pickup_position": "bridge",
  "tuning": "Drop_C",
  "technique_tags": ["palm_mute", "power_chords"],
  "bpm": 150.0,
  "license_ref": "project-owned",
  "split": "test"
}
```

## Licensing

- Source code in this repository is licensed under Apache-2.0. See `LICENSE`.
- Project-owned dataset metadata and future original data releases are governed separately. See `DATA_LICENSE.md`.
- Third-party audio, IRs, presets, model weights, and plugin-related assets are tracked separately. See `THIRD_PARTY.md`.

## Current Status

The repository has a bootstrapped package layout and schema layer. The next milestones are:

1. implement the offline re-amp generation pipeline
2. ship a retrieval baseline with feature extraction and nearest-neighbor search
3. add embedding training and evaluation
4. layer a thin CLI on top of the public API

## Suggested GitHub Description

Applied ML system for guitar tone retrieval from controlled re-amped DI data.
