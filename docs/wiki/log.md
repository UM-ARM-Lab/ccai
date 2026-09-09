# CCAI Wiki Log

The log is append-only and records durable wiki maintenance.

Last updated: 2026-07-22

Related: [Wiki index](index.md) · [Overview](overview.md)

## [2026-07-22] ingest | initial repository baseline

- Added the index, overview, source summaries, a system page, a task/recovery concept page, and a Proto5 Isaac Sim workflow page.
- Grounded the baseline in the root README, `examples/screwdriver_isaacsim_recovery.py`, and the checked Proto5 configuration directory.
- Marked the relationship between older README guidance and the current Isaac Sim path as a source-validation boundary rather than assuming they are identical.

## [2026-09-09] implementation | Valve timed completion client

- Added the Valve-only actionlib client, imported selected armtree timing,
  canonical joint-order and mocap corrections, and preserved legacy modes.
- Added mocked completion, timeout, preemption and stale-observation tests.
- Updated the index and Valve workflow; implementation uses an isolated worktree
  to preserve the pre-existing local feature branch and unrelated YAML.
