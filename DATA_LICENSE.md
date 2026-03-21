# Data Licensing Policy

This repository separates source-code licensing from data, audio, presets, IRs, model artifacts, and other non-code assets.

## Code vs. Data

The Apache-2.0 license in `LICENSE` applies to source code and documentation unless stated otherwise. It does not automatically apply to audio recordings, rendered tones, impulse responses, plugin presets, model weights, or dataset archives.

## Project-Owned Data

Unless a file or directory says otherwise, any future project-owned dataset metadata created in this repository may be released under the terms explicitly attached to that asset at the time of publication.

Examples of project-owned metadata may include:

- manifest files
- split definitions
- evaluation annotations
- schema files
- derived labels created inside the project

If original audio recordings created specifically for this project are published later, they should be marked with their own license at the directory or file level before redistribution.

## Third-Party and Restricted Assets

Third-party assets keep their original license terms and are not relicensed by this repository.

This includes, but is not limited to:

- DI guitar packs
- IR libraries
- plugin presets exported from commercial software
- rendered audio derived from restricted source material
- pretrained checkpoints obtained from external sources

Do not assume that a file is redistributable just because it appears in a local experiment directory.

## Current Repository Policy

- No bundled dataset is currently declared as generally redistributable.
- Any future public data release should be documented before publication.
- Third-party asset usage must be recorded in `THIRD_PARTY.md`.

## Contributor Rule

If you add data or artifacts to this repository, document:

- where they came from
- who owns them
- whether redistribution is allowed
- which license or usage restriction applies
