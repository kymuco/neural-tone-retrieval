# Controlled Re-Amp Example

This directory contains a minimal tracked example bundle for the first Neural Tone Retrieval workflow.

Files:

- `example_dataset_manifest.json`: source clips, chain specs, split assignments, and an example ingest run
- `controlled_reamp.toml`: pipeline config that points at the example manifest

Useful commands:

```bash
python -m neural_tone_retrieval manifest show examples/controlled_reamp/example_dataset_manifest.json
python -m neural_tone_retrieval config show examples/controlled_reamp/controlled_reamp.toml
```
