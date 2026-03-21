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

## Planned Repository Layout

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
|-- tests/
`-- data/
```

Some directories such as `data/`, `artifacts/`, `runs/`, and `models/` are expected to stay untracked by default unless there is a clear reason to version small example assets.

## Licensing

- Source code in this repository is licensed under Apache-2.0. See `LICENSE`.
- Project-owned dataset metadata and future original data releases are governed separately. See `DATA_LICENSE.md`.
- Third-party audio, IRs, presets, model weights, and plugin-related assets are tracked separately. See `THIRD_PARTY.md`.

## Current Status

The repository is in bootstrap mode. The next milestones are:

1. define the dataset schema and artifact model
2. implement the offline re-amp generation pipeline
3. ship a retrieval baseline with feature extraction and nearest-neighbor search
4. add embedding training and evaluation

## Suggested GitHub Description

Applied ML system for guitar tone retrieval from controlled re-amped DI data.
