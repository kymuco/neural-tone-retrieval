# Third-Party Asset Policy

This file tracks non-code assets and dependencies that require separate licensing review before they are committed, published, or redistributed.

## Policy

Do not commit or publish third-party assets unless their license explicitly allows it.

Common high-risk asset classes include:

- commercial DI packs
- commercial or proprietary IR libraries
- plugin presets exported from paid software
- rendered audio derived from restricted inputs
- pretrained model weights with non-permissive terms

## Asset Register

Use the table below to document every third-party asset family that enters the project.

| Asset | Source | License | Redistributable | Notes |
| --- | --- | --- | --- | --- |
| None recorded yet | - | - | - | Repository bootstrap state |

## Contributor Checklist

Before adding an external asset, verify all of the following:

- the source is known
- the license is known
- redistribution status is known
- derivative outputs are allowed if relevant
- the asset is recorded in this file

If any of the above is unclear, keep the asset out of version control and treat it as restricted until clarified.
