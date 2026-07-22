# Proto5 Isaac Sim Screwdriver Recovery

The Proto5 recovery workflow runs CCAI's legacy recovery logic against Isaac Sim screwdriver environments with YAML-selected simulation or hardware behavior.

Last updated: 2026-07-22

Related: [Recovery entrypoint source](../sources/summaries/isaacsim-recovery-entrypoint.md) · [Proto5 recovery system](../entities/proto5-screwdriver-recovery-system.md)

## Workflow

1. Select a YAML profile from [examples/config/proto5](../../../examples/config/proto5/) and invoke [examples/screwdriver_isaacsim_recovery.py](../../../examples/screwdriver_isaacsim_recovery.py).
2. The entrypoint loads the configuration, resolves project dependencies, and creates either the Isaac Sim recovery environment or the hardware environment according to `mode`.
3. It loads task/recovery samplers and optional DiffPF action policies when configuration enables them, then executes the legacy trial loop.
4. It writes the resolved run configuration into the experiment directory and records trial-specific artifacts under the selected experiment path.

## Configuration ownership

The active YAML profile is the primary run contract. Validate the selected profile's hand, mode, controller, device, dataset/model paths, and output fields before treating a command as reproducible.

## Evidence

- [Recovery entrypoint](../../../examples/screwdriver_isaacsim_recovery.py)
- [Proto5 configuration directory](../../../examples/config/proto5/)
