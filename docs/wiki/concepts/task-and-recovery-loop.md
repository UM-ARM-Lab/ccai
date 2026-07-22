# Task And Recovery Loop

CCAI separates nominal task execution from perturbation-aware recovery so that data and models can address normal manipulation and failure handling differently.

Last updated: 2026-07-22

Related: [Repository README](../sources/summaries/repository-readme.md) · [Proto5 recovery workflow](../workflows/proto5-isaacsim-screwdriver-recovery.md)

## Current model

The README describes nominal task data generation without perturbation and recovery data generation with perturbations. Recovery evaluation uses a controller and task/recovery model context to continue or restore manipulation after off-nominal states.

The current Isaac Sim entrypoint preserves this division through the legacy recovery loop while making the concrete simulator and policy runtime depend on YAML configuration.

## Historical note

The root README is a high-level source, not a guarantee that every current Proto5/Isaac Sim default matches the older Allegro/Touchlegro examples. Preserve both descriptions and resolve differences from active code and YAML.
