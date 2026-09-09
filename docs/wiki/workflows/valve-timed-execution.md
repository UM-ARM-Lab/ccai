# Valve timed execution

The Valve ROS adapter owns completion and fresh-joint-state waiting for timed repeats.

Last updated: 2026-09-09

Related: [Index](../index.md)

`hardware/allegro_ros.py` supports the opt-in `valve_timed_repeat` command mode.
`hardware/valve_timed_client.py` submits one canonical 16-joint action to
`/allegroHand/execute_valve_action`, then requires successful completion and a
subsequently received joint sample timestamped after completion. Defaults in
generic CCAI remain the legacy repeat mode. Missing completion or stale data
raises rather than silently switching command modes.

The matched Valve profile uses 40 repeats in 1/3 s. Model_mismatch's hardware
adapter additionally waits for post-completion mocap data. Its YAML launch and
ROS-overlay instructions are in `docs/valve_matched_execution.md` in that repo.
`initialize_control=False` suppresses the initial gravcomp publication for dry
runs; existing generic callers retain their previous initialization behavior.

Selected armtree uncommitted changes supply joint-order conversion, timed
command messages and the Valve yaw convention `-Euler_X`. They do not supply
gains, driver or tactile-sensor configuration. Tests use mocked ROS clients;
no hardware actuation is part of this validation.
