import pickle
import sys
from pathlib import Path

import numpy as np
import torch

CCAI_ROOT = Path(__file__).resolve().parents[1]
if str(CCAI_ROOT) not in sys.path:
    sys.path.insert(0, str(CCAI_ROOT))

from ccai.dataset import AllegroScrewDriverDataset


def _make_row(state_dim, action_dim, *, yaw=0.0, wrist_value=0.0):
    row = np.zeros(state_dim + action_dim + 9, dtype=np.float32)
    if state_dim == 17:
        row[12:14] = wrist_value
    row[state_dim - 3] = 0.0
    row[state_dim - 2] = 0.0
    row[state_dim - 1] = yaw
    row[state_dim:state_dim + action_dim] = np.arange(action_dim, dtype=np.float32) + 10.0
    row[state_dim + action_dim:] = np.arange(9, dtype=np.float32) + 20.0
    return row


def _write_screwdriver_trial(root, *, state_dim, action_dim):
    trial_dir = root / "screwdriver_csvgd_csvto_closed_loop" / "trial_000000"
    trial_dir.mkdir(parents=True)
    row_width = state_dim + action_dim + 9
    data = {
        "pre_action_likelihoods": [],
        "final_likelihoods": [],
        "executed_contacts": [],
        "dropped": False,
        "dropped_recovery": False,
    }
    for horizon in (1, 2, 3):
        start = _make_row(state_dim, action_dim, yaw=0.0, wrist_value=2.0).reshape(1, 1, row_width)
        plan = np.stack(
            [
                _make_row(state_dim, action_dim, yaw=-1.0, wrist_value=2.0)
                for _ in range(horizon)
            ],
            axis=0,
        ).reshape(1, 1, horizon, row_width)
        data[horizon] = {
            "starts": start,
            "plans": plan,
            "contact_state": np.ones((1, 3), dtype=np.float32),
        }
    with open(trial_dir / "traj_data.p", "wb") as handle:
        pickle.dump(data, handle)
    with open(trial_dir / "trajectory.pkl", "wb") as handle:
        pickle.dump([np.zeros(state_dim, dtype=np.float32)], handle)
    return trial_dir.parent


def _write_recovery_trial(root):
    state_dim = 15
    action_dim = 12
    row_width = state_dim + action_dim + 9
    trial_dir = root / "screwdriver_csvgd_csvto_closed_loop" / "trial_000000"
    trial_dir.mkdir(parents=True)

    starts = np.stack(
        [
            _make_row(state_dim, action_dim, yaw=0.0).reshape(1, row_width),
            _make_row(state_dim, action_dim, yaw=-0.1).reshape(1, row_width),
        ],
        axis=0,
    )
    plans = np.stack(
        [
            _make_row(state_dim, action_dim, yaw=-1.0).reshape(1, 1, row_width),
            _make_row(state_dim, action_dim, yaw=-1.1).reshape(1, 1, row_width),
        ],
        axis=0,
    )
    data = {
        "pre_action_likelihoods": [[0.0], [1.0], [0.5]],
        "final_likelihoods": [[0.0], [1.0], [0.5]],
        "executed_contacts": ["turn", "index", "middle"],
        "dropped": False,
        "dropped_recovery": False,
        1: {
            "starts": starts,
            "plans": plans,
            "contact_state": np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
        },
    }
    with open(trial_dir / "traj_data.p", "wb") as handle:
        pickle.dump(data, handle)
    with open(trial_dir / "trajectory.pkl", "wb") as handle:
        pickle.dump([np.zeros(state_dim, dtype=np.float32)], handle)
    return trial_dir.parent


def _load_dataset(tmp_path, *, state_dim, action_dim, cosine_sine=False):
    dataset_root = _write_screwdriver_trial(tmp_path / f"proto5_screwdriver_{state_dim}_{action_dim}", state_dim=state_dim, action_dim=action_dim)
    return AllegroScrewDriverDataset(
        [dataset_root],
        max_T=3,
        dx=state_dim,
        cosine_sine=cosine_sine,
        states_only=False,
        best_traj_only=True,
        recovery=False,
    )


def test_proto5_tracked_wrist_state_loads_38_wide_rows(tmp_path):
    dataset = _load_dataset(tmp_path, state_dim=17, action_dim=12)

    assert len(dataset) > 0
    assert dataset.trajectories.shape[-1] == 38
    assert dataset.du == 21
    dataset.compute_norm_constants()
    assert dataset.mean.shape[0] == 38
    assert dataset.std.shape[0] == 38


def test_proto5_tracked_wrist_with_wrist_control_loads_40_wide_rows(tmp_path):
    dataset = _load_dataset(tmp_path, state_dim=17, action_dim=14)

    assert len(dataset) > 0
    assert dataset.trajectories.shape[-1] == 40
    assert dataset.du == 23
    dataset.compute_norm_constants()
    assert dataset.mean.shape[0] == 40
    assert dataset.std.shape[0] == 40


def test_cosine_sine_adds_one_yaw_dimension_independent_of_action_width(tmp_path):
    dataset_38 = _load_dataset(tmp_path, state_dim=17, action_dim=12, cosine_sine=True)
    dataset_40 = _load_dataset(tmp_path, state_dim=17, action_dim=14, cosine_sine=True)

    traj_38, _, mask_38 = dataset_38[0]
    traj_40, _, mask_40 = dataset_40[0]
    assert traj_38.shape[-1] == 39
    assert traj_40.shape[-1] == 41
    assert mask_38.shape[-1] == 39
    assert mask_40.shape[-1] == 41
    dataset_38.compute_norm_constants()
    dataset_40.compute_norm_constants()
    assert dataset_38.mean.shape[0] == 39
    assert dataset_40.mean.shape[0] == 41


def test_roll_pitch_yaw_filter_uses_last_three_state_dims(tmp_path):
    dataset = _load_dataset(tmp_path, state_dim=17, action_dim=12)

    assert len(dataset) > 0
    wrist_columns = dataset.trajectories[:, :, 12:14]
    assert torch.all(wrist_columns == 2.0)
    final_roll_pitch = dataset.trajectories[:, -1, 14:16]
    assert torch.all(final_roll_pitch.abs() <= 0.25)


def test_recovery_dataset_filters_to_likelihood_improving_trajectories_by_default(tmp_path):
    dataset_root = _write_recovery_trial(tmp_path / "recovery_default_filter")

    dataset = AllegroScrewDriverDataset(
        [dataset_root],
        max_T=1,
        dx=15,
        best_traj_only=True,
        recovery=True,
    )

    assert len(dataset) == 1
    assert dataset.trajectory_type.tolist() == [[1.0, -1.0, -1.0]]


def test_recovery_dataset_can_keep_all_recovery_trajectories(tmp_path):
    dataset_root = _write_recovery_trial(tmp_path / "recovery_all")

    dataset = AllegroScrewDriverDataset(
        [dataset_root],
        max_T=1,
        dx=15,
        best_traj_only=True,
        recovery=True,
        filter_recovery_trajectories=False,
    )

    assert len(dataset) == 2
    assert dataset.trajectory_type.tolist() == [
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
    ]
