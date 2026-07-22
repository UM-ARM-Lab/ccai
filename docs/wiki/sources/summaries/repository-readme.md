# CCAI Repository README

The root README describes the repository's legacy task-data, recovery-data, model-training, simulation-evaluation, and hardware-evaluation workflow.

Last updated: 2026-07-22

Related: [Overview](../../overview.md) · [Task and recovery loop](../../concepts/task-and-recovery-loop.md)

## Source

- [README.md](../../../../README.md)

## Extracted claims

- Nominal task data is generated without perturbation, while recovery data is generated with perturbations such as external wrenches.
- The README distinguishes task and recovery diffusion-model training from recovery simulation and hardware evaluation.
- It records operator-facing hardware prerequisites, including motion capture and a hand controller, for the older hardware flow.

## Current relevance

This is a useful high-level historical map, but current Proto5 Isaac Sim behavior must be checked against the active entrypoint and YAML profile because the README predates parts of that integration.
