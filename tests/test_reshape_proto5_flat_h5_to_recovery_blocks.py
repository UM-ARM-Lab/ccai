import h5py
import numpy as np

from scripts import reshape_proto5_flat_h5_to_recovery_blocks as reshaper


def _write_flat_h5(path):
    row_count = 8
    trial_index = np.asarray([10, 10, 10, 10, 10, 10, 11, 11], dtype=np.int64)
    recover = np.asarray([False, True, True, False, True, True, True, False], dtype=np.bool_)
    likelihood = np.asarray([-100.0, -90.0, -80.0, -50.0, -95.0, -85.0, -88.0, -40.0], dtype=np.float32)
    q = np.zeros((row_count, 2, 3, 4), dtype=np.float32)
    q_wrist = np.zeros((row_count, 2, 2), dtype=np.float32)
    observation = np.zeros((row_count, 2, 3), dtype=np.float32)
    robot_joint_pos_full = np.zeros((row_count, 2, 18), dtype=np.float32)
    contact_state = np.zeros((row_count, 2, 3), dtype=np.float32)
    contact_wrenches = np.zeros((row_count, 2, 3, 6), dtype=np.float32)
    contact_forces = np.zeros((row_count, 2, 3, 3), dtype=np.float32)
    contact_points = np.zeros((row_count, 2, 3, 3), dtype=np.float32)
    for row in range(row_count):
        q[row, 0, :, :] = float(row * 10)
        q[row, 1, :, :] = float(row * 10 + 1)
        q_wrist[row, 0, :] = float(row * 10)
        q_wrist[row, 1, :] = float(row * 10 + 1)
        observation[row, 0, :] = float(row * 10)
        observation[row, 1, :] = float(row * 10 + 1)
        robot_joint_pos_full[row, 0, :] = float(row * 10)
        robot_joint_pos_full[row, 1, :] = float(row * 10 + 1)
        contact_state[row, 0, :] = float(row * 10)
        contact_state[row, 1, :] = float(row * 10 + 1)
        contact_wrenches[row, 0, :, :] = float(row * 10)
        contact_wrenches[row, 1, :, :] = float(row * 10 + 1)
        contact_forces[row, 0, :, :] = float(row * 10)
        contact_forces[row, 1, :, :] = float(row * 10 + 1)
        contact_points[row, 0, :, :] = float(row * 10)
        contact_points[row, 1, :, :] = float(row * 10 + 1)

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        h5.create_dataset("q", data=q)
        h5.create_dataset("q_wrist", data=q_wrist)
        h5.create_dataset("observation", data=observation)
        h5.create_dataset("robot_joint_pos_full", data=robot_joint_pos_full)
        h5.create_dataset("action", data=np.arange(row_count * 12, dtype=np.float32).reshape(row_count, 1, 3, 4))
        h5.create_dataset("contact_plan", data=np.ones((row_count, 1, 3), dtype=np.float32))
        h5.create_dataset("contact_state", data=contact_state)
        h5.create_dataset("contact_wrenches", data=contact_wrenches)
        h5.create_dataset("contact_forces", data=contact_forces)
        h5.create_dataset("contact_points", data=contact_points)
        h5.create_dataset("contact_mode", data=np.asarray(["turn", "thumb", "index", "turn", "middle", "thumb", "index", "turn"], dtype=string_dtype))
        h5.create_dataset("episode_num_steps", data=np.arange(row_count, dtype=np.int64))
        h5.create_dataset("stage_index", data=np.asarray([1, 3, 5, 6, 8, 10, 12, 13], dtype=np.int64))
        h5.create_dataset("trial_index", data=trial_index)
        h5.create_dataset("recover", data=recover)
        h5.create_dataset("likelihood", data=likelihood)
        h5.create_dataset("screwdriver_friction", data=np.full((row_count,), 2.5, dtype=np.float32))
        h5.create_dataset("yaw_joint_friction", data=np.full((row_count,), 0.03, dtype=np.float32))
        h5.create_dataset("robot_joint_names", data=np.asarray([f"joint_{idx}" for idx in range(18)], dtype=string_dtype))


def test_reshape_flat_h5_groups_recovery_runs_until_id_or_episode_end(tmp_path):
    source = tmp_path / "flat.h5"
    output = tmp_path / "blocks.h5"
    _write_flat_h5(source)

    summary = reshaper.write_recovery_blocks_h5(source, output)

    assert summary["source_rows"] == 8
    assert summary["total_recovery_blocks"] == 3
    assert summary["terminal_reason_counts"] == {"episode_end": 1, "return_to_id": 2}

    with h5py.File(output, "r") as h5:
        assert h5.attrs["schema"] == "proto5_screwdriver_recovery_blocks_from_flat_v1"
        np.testing.assert_array_equal(h5["action_lengths"][:], [2, 2, 1])
        np.testing.assert_array_equal(h5["trajectory_lengths"][:], [3, 3, 2])
        np.testing.assert_array_equal(h5["valid_action_mask"][:], [[True, True], [True, True], [True, False]])
        np.testing.assert_array_equal(h5["valid_state_mask"][:], [[True, True, True], [True, True, True], [True, True, False]])
        assert [value.decode() for value in h5["terminal_reason"][:]] == [
            "return_to_id",
            "episode_end",
            "return_to_id",
        ]
        np.testing.assert_array_equal(h5["terminal_row_index"][:], [3, -1, 7])
        np.testing.assert_allclose(h5["likelihood"][0, :3], [-100.0, -90.0, -50.0])
        np.testing.assert_allclose(h5["likelihood"][1, :3], [-50.0, -95.0, -85.0])
        np.testing.assert_allclose(h5["likelihood"][2, :2], [-88.0, -40.0])
        assert h5["final_likelihood"][0] == np.float32(-50.0)
        assert h5["final_likelihood"][1] == np.float32(-85.0)
        assert h5["final_likelihood"][2] == np.float32(-40.0)
        assert h5["q"][0, 0, 0, 0] == np.float32(10.0)
        assert h5["q"][0, 1, 0, 0] == np.float32(11.0)
        assert h5["q"][0, 2, 0, 0] == np.float32(30.0)
        assert h5["q"][1, 2, 0, 0] == np.float32(51.0)
        assert h5["q"][2, 1, 0, 0] == np.float32(70.0)
        assert [value.decode() for value in h5["contact_mode"][0, :2]] == ["thumb", "index"]
