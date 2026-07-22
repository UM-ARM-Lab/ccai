# Isaac Sim Screwdriver Recovery Entrypoint

`examples/screwdriver_isaacsim_recovery.py` is the current CCAI entrypoint for Isaac Sim screwdriver recovery datasets and execution.

Last updated: 2026-07-22

Related: [Proto5 recovery workflow](../../workflows/proto5-isaacsim-screwdriver-recovery.md) · [Proto5 recovery system](../../entities/proto5-screwdriver-recovery-system.md)

## Source

- [examples/screwdriver_isaacsim_recovery.py](../../../../examples/screwdriver_isaacsim_recovery.py)
- [Proto5 YAML profiles](../../../../examples/config/proto5/)

## Extracted claims

- The entrypoint states that it keeps the legacy recovery loop and pickle artifacts while using `IsaacSimScrewdriverRecoveryEnv` through IsaacLab environments.
- It loads YAML configuration, resolves the simulator mode, and supports both simulation and hardware environment construction.
- It resolves model_mismatch and isaacsim-hand-envs paths through CCAI project-path helpers, making those repositories operational dependencies of this flow.

## Current relevance

Use this source and the selected YAML profile as the contract for a recovery run. Do not infer runtime behavior solely from similarly named legacy scripts or from a profile intended for another mode.
