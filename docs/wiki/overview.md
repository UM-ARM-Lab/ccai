# CCAI Overview

CCAI develops task execution and recovery workflows for dexterous manipulation, with current repository evidence centered on screwdriver data generation, diffusion models, and recovery evaluation.

Last updated: 2026-07-22

Related: [Wiki index](index.md) · [Task and recovery loop](concepts/task-and-recovery-loop.md) · [Proto5 recovery workflow](workflows/proto5-isaacsim-screwdriver-recovery.md)

## Current synthesis

The root [README](../../README.md) describes a pipeline that separates nominal task data/model work from perturbation-driven recovery data/model work, then evaluates recovery in simulation or on hardware. The current [Isaac Sim recovery entrypoint](sources/summaries/isaacsim-recovery-entrypoint.md) keeps the legacy recovery loop while routing it through IsaacLab-backed screwdriver environments.

The Proto5 recovery path crosses repository boundaries: CCAI owns execution and recovery orchestration, while model_mismatch supplies DiffPF-related policies and Isaac Sim hand environments supply the simulator assets and environments. Treat concrete configuration and source paths as the current authority when these layers differ from older README instructions.

## Open questions

- Which checked Proto5 YAML profiles are the preferred operational profiles for a given hardware or simulation campaign?
- Which historical recovery claims remain representative after the Isaac Sim and DiffPF integration work?

## Evidence

- [Repository README](../../README.md)
- [Isaac Sim recovery entrypoint](../../examples/screwdriver_isaacsim_recovery.py)
- [Proto5 configuration directory](../../examples/config/proto5/)
