# Proto5 Screwdriver Recovery System

The current Proto5 recovery system joins CCAI's recovery loop with Isaac Sim hand environments and optional DiffPF policy execution from model_mismatch.

Last updated: 2026-07-22

Related: [Isaac Sim recovery entrypoint](../sources/summaries/isaacsim-recovery-entrypoint.md) · [Proto5 recovery workflow](../workflows/proto5-isaacsim-screwdriver-recovery.md)

## Components

- CCAI's recovery entrypoint owns configuration loading, experiment directories, the legacy trial loop, and recovery controller setup.
- The Isaac Sim recovery environment supplies simulator-backed state and action execution for the screwdriver task.
- model_mismatch is located by project-path helpers and may supply DiffPF normal or recovery action policies when the selected configuration enables them.
- A YAML profile selects the hand, mode, controller, artifact paths, and run parameters.

## Evidence

- [Recovery entrypoint](../../../examples/screwdriver_isaacsim_recovery.py)
- [Proto5 configs](../../../examples/config/proto5/)

## Open questions

- Which interfaces are stable enough to document as a cross-repository compatibility contract rather than a current implementation detail?
