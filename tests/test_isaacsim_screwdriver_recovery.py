import importlib.util
import os
import pathlib
import sys
import types

import numpy as np
import pytest
import torch

from ccai.utils import isaacsim_screwdriver_recovery as isaacsim_recovery_utils
from ccai.utils.isaacsim_screwdriver_recovery import (
    ALLEGRO_ACTIVE_JOINT_NAMES,
    ALLEGRO_DEFAULT_FULL_JOINT_POS,
    ALLEGRO_RING_JOINT_NAMES,
    OBJ_ORIENTATION_JOINT_NAMES,
    PROTO5_ACTIVE_JOINT_NAMES,
    PROTO5_ALL_JOINT_NAMES,
    PROTO5_WRIST_JOINT_NAMES,
    HardwareScrewdriverRecoveryEnv,
    HardwareVisualizationShim,
    IsaacSimScrewdriverRecoveryEnv,
    active12_to_env_action,
    local_force_at_position_to_world,
    pack_ccai_state,
    sample_screwdriver_body_poke,
    tee_stdout_to_file,
)
from ccai.utils.recovery_utils import build_pregrasp_reference_target_kwargs, create_allegro_screwdriver_problem


def test_proto5_recovery_hand_spec_uses_shared_isaacsim_defaults():
    spec = importlib.util.spec_from_file_location(
        "proto5_defaults_for_test",
        isaacsim_recovery_utils.PROTO5_DEFAULTS_PATH,
    )
    defaults = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(defaults)

    hand_spec = isaacsim_recovery_utils.get_hand_spec("proto5")

    assert hand_spec.robot_root_pos == tuple(defaults.PROTO5_SCREWDRIVER_ROOT_POS)
    assert hand_spec.robot_root_rot_wxyz == tuple(defaults.PROTO5_SCREWDRIVER_ROOT_ROT)
    assert hand_spec.all_joint_names == tuple(defaults.ALL_JOINT_NAMES)
    assert hand_spec.active_joint_names == tuple(defaults.ACTIVE_FINGER_JOINT_NAMES)
    assert hand_spec.default_full_joint_pos == tuple(defaults.DEFAULT_FULL_JOINT_POS)


_ENTRYPOINT_PATH = pathlib.Path(__file__).resolve().parents[1] / "examples" / "screwdriver_isaacsim_recovery.py"
_ENTRYPOINT_SPEC = importlib.util.spec_from_file_location("screwdriver_isaacsim_recovery_entrypoint", _ENTRYPOINT_PATH)
screwdriver_isaacsim_recovery = importlib.util.module_from_spec(_ENTRYPOINT_SPEC)
_ENTRYPOINT_SPEC.loader.exec_module(screwdriver_isaacsim_recovery)


def test_tee_stdout_to_file_copies_prints_to_trial_log(tmp_path, capsys):
    log_path = tmp_path / "trial_1" / "stdout.log"

    with tee_stdout_to_file(log_path):
        print("trial log line")

    assert "trial log line" in capsys.readouterr().out
    assert log_path.read_text(encoding="utf-8") == "trial log line\n"


def _entrypoint_args(config_path, **overrides):
    values = {
        "config": config_path,
        "hand": "proto5",
        "headless": None,
        "no_video": False,
        "num_envs": 1,
        "sim_device": "cuda:0",
        "proto5_control_wrist": None,
        "steps_per_action": None,
        "action_repeat": None,
        "save_recovery_frames": None,
        "start_ind": None,
        "end_ind": None,
        "skip_pregrasp": None,
        "pregrasp_only": None,
        "visualize_executed_rollout": None,
        "experiment_name": None,
        "debug_progress": False,
        "planner_yaw_joint_friction_override": None,
        "planner_use_env_yaw_joint_friction": None,
        "planner_yaw_friction_model_path": None,
        "disable_planner_yaw_friction_model": None,
        "planner_yaw_inertia_model_path": None,
        "planner_use_yaw_inertia_model": None,
        "use_pregrasp_reference_targets": None,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_isaacsim_recovery_defaults_match_csvto_cadence(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("controllers:\n  csvgd: {}\nexternal_wrench_perturb: false\nrand_pct: 0.333\nrandom_force_magnitude: 1.0\n")

    config = screwdriver_isaacsim_recovery.load_config(_entrypoint_args(config_path))

    assert config["steps_per_action"] == 40
    assert config["action_repeat"] == 3
    assert config["save_recovery_frames"] is True
    assert config["visualize_executed_rollout"] is True


def test_isaacsim_recovery_can_disable_executed_rollout_visualization(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "controllers:\n"
        "  csvgd: {}\n"
        "external_wrench_perturb: false\n"
        "rand_pct: 0.333\n"
        "random_force_magnitude: 1.0\n"
        "visualize_executed_rollout: false\n"
    )

    config = screwdriver_isaacsim_recovery.load_config(_entrypoint_args(config_path))

    assert config["visualize_executed_rollout"] is False


def test_isaacsim_recovery_run_dir_uses_collision_safe_top_level_stamp(tmp_path, monkeypatch):
    monkeypatch.setattr(screwdriver_isaacsim_recovery.time, "strftime", lambda fmt: "20260709_1530")

    controller_dir = tmp_path / "csvgd"

    first = screwdriver_isaacsim_recovery._experiment_log_run_dir(controller_dir, temperature=0.5)
    first.mkdir(parents=True)
    second = screwdriver_isaacsim_recovery._experiment_log_run_dir(controller_dir, temperature=0.5)

    assert first == controller_dir / "20260709_1530_temp0p5"
    assert second == controller_dir / "20260709_1530_temp0p5_run02"
    assert second / "trial_2" == controller_dir / "20260709_1530_temp0p5_run02" / "trial_2"


def test_isaacsim_recovery_visualize_false_defaults_to_headless(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("controllers:\n  csvgd: {}\nvisualize: false\n")

    config = screwdriver_isaacsim_recovery.load_config(_entrypoint_args(config_path))

    assert config["visualize"] is False
    assert config["headless"] is True


def test_isaacsim_recovery_visualize_true_defaults_to_visible(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("controllers:\n  csvgd: {}\nvisualize: true\n")

    config = screwdriver_isaacsim_recovery.load_config(_entrypoint_args(config_path))

    assert config["visualize"] is True
    assert config["headless"] is False


def test_isaacsim_recovery_loads_per_finger_min_force(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            (
                "controllers:",
                "  csvgd: {}",
                "external_wrench_perturb: false",
                "rand_pct: 0.333",
                "random_force_magnitude: 1.0",
                "min_force:",
                "  index: 0.25",
                "  middle: 0.75",
                "  thumb: 1.25",
                "",
            )
        )
    )

    config = screwdriver_isaacsim_recovery.load_config(_entrypoint_args(config_path))

    assert config["min_force_dict"] == {
        "index": pytest.approx(0.25),
        "middle": pytest.approx(0.75),
        "thumb": pytest.approx(1.25),
    }


def test_hardware_recovery_load_config_preserves_hardware_mode_and_disables_sim_noise(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            (
                "controllers:",
                "  csvgd: {}",
                "mode: hardware",
                "hand: proto5",
                "external_wrench_perturb: true",
                "randomize_obj_start: true",
                "save_recovery_frames: true",
                "hardware_execute: true",
                "",
            )
        )
    )

    config = screwdriver_isaacsim_recovery.load_config(_entrypoint_args(config_path))

    assert config["mode"] == "hardware"
    assert config["simulator"] == "hardware"
    assert config["external_wrench_perturb"] is False
    assert config["randomize_obj_start"] is False
    assert config["save_recovery_frames"] is False
    assert config["hardware_execute"] is True
    assert config["hardware_use_live_screwdriver_position"] is True
    assert config["hardware_ros_config"] == str(screwdriver_isaacsim_recovery.DEFAULT_HARDWARE_ROS_CONFIG)


def _write_hardware_initialization_h5(path, **datasets):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as h5_file:
        for name, value in datasets.items():
            h5_file.create_dataset(name, data=np.asarray(value, dtype=np.float32))


def _proto5_hardware_init_config(dataset_path, **overrides):
    config = {
        "mode": "hardware",
        "hand": "proto5",
        "dataset_path": str(dataset_path),
        "seed": 2,
        "proto5_control_wrist": False,
    }
    config.update(overrides)
    return config


def test_proto5_hardware_initialization_uses_seed_selected_validation_row_and_joint_targets(monkeypatch, tmp_path):
    dataset_path = tmp_path / "initialization.h5"
    initial_joint_targets = np.arange(36, dtype=np.float32).reshape(3, 12)
    _write_hardware_initialization_h5(dataset_path, initial_joint_targets=initial_joint_targets)
    monkeypatch.setattr(
        screwdriver_isaacsim_recovery,
        "_resolve_proto5_validation_dataset_row",
        lambda seed: seed - 1,
    )

    initialization = screwdriver_isaacsim_recovery._load_proto5_hardware_initialization(
        _proto5_hardware_init_config(dataset_path)
    )

    assert initialization["validation_ordinal"] == 2
    assert initialization["trajectory_row"] == 1
    assert initialization["target_source"] == "initial_joint_targets"
    np.testing.assert_allclose(initialization["initial_target"], initial_joint_targets[1])


def test_proto5_hardware_initialization_falls_back_to_initial_state(monkeypatch, tmp_path):
    dataset_path = tmp_path / "initialization.h5"
    initial_state = np.arange(45, dtype=np.float32).reshape(3, 15)
    _write_hardware_initialization_h5(dataset_path, initial_state=initial_state)
    monkeypatch.setattr(
        screwdriver_isaacsim_recovery,
        "_resolve_proto5_validation_dataset_row",
        lambda seed: 2,
    )

    initialization = screwdriver_isaacsim_recovery._load_proto5_hardware_initialization(
        _proto5_hardware_init_config(dataset_path, seed=0)
    )

    assert initialization["trajectory_row"] == 2
    assert initialization["target_source"] == "initial_state[:12]"
    np.testing.assert_allclose(initialization["initial_target"], initial_state[2, :12])
    np.testing.assert_allclose(initialization["initial_orientation"], initial_state[2, 12:15])


def test_proto5_hardware_initialization_preserves_q_initial_orientation(monkeypatch, tmp_path):
    dataset_path = tmp_path / "initialization.h5"
    q = np.zeros((3, 2, 16), dtype=np.float32)
    q[1, 0, :12] = np.arange(12, dtype=np.float32)
    q[1, 0, 12:15] = np.array([0.11, -0.22, 0.33], dtype=np.float32)
    _write_hardware_initialization_h5(dataset_path, q=q)
    monkeypatch.setattr(
        screwdriver_isaacsim_recovery,
        "_resolve_proto5_validation_dataset_row",
        lambda seed: 1,
    )

    initialization = screwdriver_isaacsim_recovery._load_proto5_hardware_initialization(
        _proto5_hardware_init_config(dataset_path, seed=0)
    )

    assert initialization["target_source"] == "q[0, :12]"
    np.testing.assert_allclose(initialization["initial_target"], np.arange(12, dtype=np.float32))
    np.testing.assert_allclose(initialization["initial_orientation"], np.array([0.11, -0.22, 0.33], dtype=np.float32))


def test_proto5_hardware_initialization_requires_dataset_path():
    with pytest.raises(ValueError, match="requires dataset_path"):
        screwdriver_isaacsim_recovery._load_proto5_hardware_initialization(
            {
                "mode": "hardware",
                "hand": "proto5",
                "seed": 0,
            }
        )


def test_proto5_hardware_initialization_rejects_wrong_target_dim(monkeypatch, tmp_path):
    dataset_path = tmp_path / "initialization.h5"
    _write_hardware_initialization_h5(
        dataset_path,
        initial_joint_targets=np.zeros((1, 13), dtype=np.float32),
    )
    monkeypatch.setattr(
        screwdriver_isaacsim_recovery,
        "_resolve_proto5_validation_dataset_row",
        lambda seed: 0,
    )

    with pytest.raises(ValueError, match="incompatible dimension"):
        screwdriver_isaacsim_recovery._load_proto5_hardware_initialization(
            _proto5_hardware_init_config(dataset_path)
        )


def test_load_single_screwdriver_shape_from_h5_dataset(tmp_path):
    dataset_path = tmp_path / "shape.h5"
    _write_hardware_initialization_h5(
        dataset_path,
        q=np.zeros((3, 2, 15), dtype=np.float32),
        screwdriver_shape_id=np.zeros(3, dtype=np.int64),
        screwdriver_body_height=np.full(3, 0.12, dtype=np.float32),
        screwdriver_body_diameter=np.full(3, 0.035, dtype=np.float32),
    )

    shape = screwdriver_isaacsim_recovery.load_single_screwdriver_shape_from_dataset(dataset_path)

    assert shape["screwdriver_shape_id"] == 0
    assert shape["screwdriver_body_height"] == pytest.approx(0.12)
    assert shape["screwdriver_body_diameter"] == pytest.approx(0.035)


def test_load_single_screwdriver_shape_rejects_mixed_h5_shapes(tmp_path):
    dataset_path = tmp_path / "mixed_shape.h5"
    _write_hardware_initialization_h5(
        dataset_path,
        q=np.zeros((3, 2, 15), dtype=np.float32),
        screwdriver_shape_id=np.zeros(3, dtype=np.int64),
        screwdriver_body_height=np.asarray([0.12, 0.13, 0.12], dtype=np.float32),
        screwdriver_body_diameter=np.full(3, 0.035, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="requires a single screwdriver body shape"):
        screwdriver_isaacsim_recovery.load_single_screwdriver_shape_from_dataset(dataset_path)


def test_load_config_can_take_screwdriver_shape_from_dataset(monkeypatch, tmp_path):
    dataset_path = tmp_path / "shape.h5"
    planner_urdf_path = tmp_path / "generated_screwdriver_3d.urdf"
    planner_urdf_path.write_text("<robot name='screwdriver'/>", encoding="utf-8")
    monkeypatch.setattr(
        screwdriver_isaacsim_recovery,
        "resolve_screwdriver_planner_urdf_path",
        lambda body_height, body_diameter, **kwargs: str(planner_urdf_path),
    )
    _write_hardware_initialization_h5(
        dataset_path,
        q=np.zeros((2, 2, 15), dtype=np.float32),
        screwdriver_shape_id=np.zeros(2, dtype=np.int64),
        screwdriver_body_height=np.full(2, 0.12, dtype=np.float32),
        screwdriver_body_diameter=np.full(2, 0.035, dtype=np.float32),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            (
                "controllers:",
                "  csvgd: {}",
                f"dataset_path: '{dataset_path}'",
                "screwdriver_shape_from_dataset: true",
                "",
            )
        )
    )

    config = screwdriver_isaacsim_recovery.load_config(_entrypoint_args(config_path))

    assert config["screwdriver_shape_dataset_id"] == 0
    assert config["screwdriver_body_height"] == pytest.approx(0.12)
    assert config["screwdriver_body_diameter"] == pytest.approx(0.035)
    assert config["planner_screwdriver_urdf_path"] == str(planner_urdf_path)
    assert "screwdriver_shape_id" not in config


def test_make_isaacsim_env_forwards_dataset_shape_to_proto5_cfg(monkeypatch):
    captured = {}

    def fake_cfg_factory(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            sim=types.SimpleNamespace(create_stage_in_memory=False),
            terminations=types.SimpleNamespace(success=object(), screwdriver_dropped=object()),
        )

    gym_module = types.ModuleType("gymnasium")
    gym_module.make = lambda gym_id, cfg: types.SimpleNamespace(gym_id=gym_id, cfg=cfg)
    proto5_module = types.ModuleType("isaacsim_hand_envs.proto5_screwdriver_turning")
    proto5_module.get_proto5_screwdriver_turning_rl_env_cfg = fake_cfg_factory
    monkeypatch.setitem(sys.modules, "gymnasium", gym_module)
    monkeypatch.setitem(sys.modules, "isaacsim_hand_envs", types.ModuleType("isaacsim_hand_envs"))
    monkeypatch.setitem(sys.modules, "isaacsim_hand_envs.proto5_screwdriver_turning", proto5_module)
    monkeypatch.setattr(
        isaacsim_recovery_utils,
        "get_hand_spec",
        lambda hand: types.SimpleNamespace(gym_id="Proto5ScrewdriverTurning-v0"),
    )
    monkeypatch.setattr(
        isaacsim_recovery_utils,
        "IsaacSimScrewdriverRecoveryEnv",
        lambda env, **kwargs: types.SimpleNamespace(env=env, kwargs=kwargs),
    )

    screwdriver_isaacsim_recovery.make_isaacsim_env(
        {
            "hand": "proto5",
            "num_envs": 1,
            "steps_per_action": 40,
            "episode_length_s": 1000.0,
            "sim_device": "cuda:0",
            "no_video": True,
            "save_recovery_frames": False,
            "friction_coefficient": 0.9,
            "screwdriver_friction": 2.5,
            "yaw_joint_friction": 0.03,
            "proto5_control_wrist": False,
            "external_wrench_perturb": False,
            "rand_pct": 0.333,
            "random_force_magnitude": 1.0,
            "action_repeat": 3,
            "screwdriver_body_height": 0.12,
            "screwdriver_body_diameter": 0.035,
        }
    )

    assert captured["screwdriver_body_height"] == pytest.approx(0.12)
    assert captured["screwdriver_body_diameter"] == pytest.approx(0.035)
    assert captured["screwdriver_shape_id"] is None


def test_simulation_initial_grasp_from_dataset_uses_local_trial_row(tmp_path):
    dataset_path = tmp_path / "initial_grasp.h5"
    q = np.zeros((2, 3, 16), dtype=np.float32)
    q[1, 0, :15] = np.arange(15, dtype=np.float32) + 100.0
    screwdriver_pos_robot = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        dtype=np.float32,
    )
    _write_hardware_initialization_h5(dataset_path, q=q, screwdriver_pos_robot=screwdriver_pos_robot)

    initialization = screwdriver_isaacsim_recovery.load_simulation_dataset_initial_grasp(
        {
            "initial_grasp_from_dataset": True,
            "dataset_path": str(dataset_path),
            "proto5_control_wrist": False,
        },
        trial_index=11,
        start_ind=10,
    )

    assert initialization["trajectory_row"] == 1
    assert initialization["target_source"] == "q[0, :12]"
    np.testing.assert_allclose(initialization["initial_target"], q[1, 0, :12])
    np.testing.assert_allclose(initialization["initial_orientation"], q[1, 0, 12:15])
    np.testing.assert_allclose(initialization["initial_state"], q[1, 0])
    np.testing.assert_allclose(initialization["screwdriver_pos_robot"], screwdriver_pos_robot[1])


def test_initial_grasp_dataset_row_can_cycle_for_long_parallel_runs(monkeypatch):
    monkeypatch.setattr(
        screwdriver_isaacsim_recovery,
        "_trajectory_count_for_dataset_path",
        lambda dataset_path: 307,
    )

    assert screwdriver_isaacsim_recovery.select_initial_grasp_dataset_row(
        "unused.h5",
        trial_index=0,
        start_ind=0,
        cycle=True,
    ) == 0
    assert screwdriver_isaacsim_recovery.select_initial_grasp_dataset_row(
        "unused.h5",
        trial_index=306,
        start_ind=0,
        cycle=True,
    ) == 306
    assert screwdriver_isaacsim_recovery.select_initial_grasp_dataset_row(
        "unused.h5",
        trial_index=307,
        start_ind=0,
        cycle=True,
    ) == 0
    assert screwdriver_isaacsim_recovery.select_initial_grasp_dataset_row(
        "unused.h5",
        trial_index=9999,
        start_ind=0,
        cycle=True,
    ) == 175


def test_load_config_applies_parallel_launcher_overrides(tmp_path):
    config_path = tmp_path / "config.yaml"
    experiment_dir = tmp_path / "parallel_run"
    config_path.write_text("controllers:\n  csvgd:\n    device: cuda:7\n")

    config = screwdriver_isaacsim_recovery.load_config(
        _entrypoint_args(
            config_path,
            experiment_dir=experiment_dir,
            write_pregrasp_states=False,
            cycle_initial_grasp_dataset=True,
            controller_device="cuda:0",
        )
    )

    assert config["experiment_dir"] == str(experiment_dir)
    assert config["write_pregrasp_states"] is False
    assert config["cycle_initial_grasp_dataset"] is True
    assert config["controllers"]["csvgd"]["device"] == "cuda:0"


def test_simulation_initial_grasp_prefers_full_joint_state_h5_schema(tmp_path):
    h5py = pytest.importorskip("h5py")
    dataset_path = tmp_path / "full_joint_initial_grasp.h5"
    full_joint_names = np.asarray(
        [
            "RHand_WRZ_joint",
            "RHand_WRY_joint",
            *PROTO5_ACTIVE_JOINT_NAMES,
        ],
        dtype=object,
    )
    full = np.zeros((1, 2, len(full_joint_names)), dtype=np.float32)
    active_values = np.arange(12, dtype=np.float32) + 10.0
    full[0, 0, 2:] = active_values
    observation = np.zeros((1, 2, 3), dtype=np.float32)
    observation[0, 0] = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    q = np.full((1, 2, 3, 4), -99.0, dtype=np.float32)

    with h5py.File(dataset_path, "w") as h5_file:
        h5_file.create_dataset("robot_joint_pos_full", data=full)
        h5_file.create_dataset("observation", data=observation)
        h5_file.create_dataset("q", data=q)
        h5_file.attrs["robot_full_joint_names"] = full_joint_names

    initialization = screwdriver_isaacsim_recovery.load_simulation_dataset_initial_grasp(
        {
            "initial_grasp_from_dataset": True,
            "dataset_path": str(dataset_path),
            "hand": "proto5",
            "proto5_control_wrist": False,
        },
        trial_index=0,
        start_ind=0,
    )

    assert initialization["target_source"] == "robot_joint_pos_full[0, active_joints]"
    np.testing.assert_allclose(initialization["initial_target"], active_values)
    np.testing.assert_allclose(initialization["initial_orientation"], observation[0, 0])
    np.testing.assert_allclose(initialization["initial_state"], np.concatenate((active_values, observation[0, 0])))


def test_apply_simulation_dataset_initial_grasp_sets_dataset_state():
    calls = []

    class Env:
        device = torch.device("cpu")

        def __init__(self):
            self.state = torch.zeros(16, dtype=torch.float32)
            self.screwdriver_pos_robot = None

        def reset(self):
            calls.append("reset")

        def set_pose(self, state, screwdriver_pos_robot=None):
            calls.append("set_pose")
            self.state = torch.as_tensor(state, dtype=torch.float32).reshape(-1)
            self.screwdriver_pos_robot = (
                None
                if screwdriver_pos_robot is None
                else torch.as_tensor(screwdriver_pos_robot, dtype=torch.float32).reshape(-1)
            )

        def step(self, action):
            calls.append("step")

    env = Env()
    initial_state = np.arange(16, dtype=np.float32)

    screwdriver_isaacsim_recovery.apply_simulation_dataset_initial_grasp(
        env,
        {
            "trajectory_row": 3,
            "target_source": "q[0, :12]",
            "initial_target": initial_state[:12],
            "initial_orientation": initial_state[12:15],
            "initial_state": initial_state,
            "screwdriver_pos_robot": np.array([0.4, 0.5, 0.6], dtype=np.float32),
        },
        device="cpu",
    )

    assert calls == ["reset", "set_pose"]
    torch.testing.assert_close(env.state, torch.arange(16, dtype=torch.float32))
    torch.testing.assert_close(env.screwdriver_pos_robot, torch.tensor([0.4, 0.5, 0.6]))


def test_send_proto5_hardware_initial_pose_steps_before_confirmation(monkeypatch):
    calls = []

    class Env:
        def step(self, action):
            calls.append(("step", torch.as_tensor(action).detach().cpu().clone()))

        def capture_observed_object_orientation(self):
            calls.append(("capture_orientation", None))
            return torch.tensor([[0.4, 0.5, 0.6]], dtype=torch.float32)

    def fake_input(prompt):
        calls.append(("input", prompt))
        return ""

    monkeypatch.setattr("builtins.input", fake_input)

    screwdriver_isaacsim_recovery.send_proto5_hardware_initial_pose_and_wait(
        Env(),
        {
            "validation_ordinal": 2,
            "trajectory_row": 7,
            "target_source": "initial_joint_targets",
            "initial_target": np.arange(12, dtype=np.float32),
            "initial_orientation": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        },
        device="cpu",
    )

    assert calls[0][0] == "step"
    torch.testing.assert_close(calls[0][1], torch.arange(12, dtype=torch.float32).reshape(1, 12))
    assert calls[1][0] == "input"
    assert calls[2][0] == "capture_orientation"


def test_isaacsim_recovery_rejects_frame_saving_without_cameras(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("controllers:\n  csvgd: {}\n")

    with pytest.raises(ValueError, match="save_recovery_frames=True requires cameras"):
        screwdriver_isaacsim_recovery.load_config(_entrypoint_args(config_path, no_video=True))


def test_screwdriver_body_poke_sampler_bounds_magnitude_and_seed():
    generator_a = torch.Generator(device="cpu").manual_seed(7)
    point_a, force_a = sample_screwdriver_body_poke(
        8,
        random_force_magnitude=2.0,
        generator=generator_a,
    )
    generator_b = torch.Generator(device="cpu").manual_seed(7)
    point_b, force_b = sample_screwdriver_body_poke(
        8,
        random_force_magnitude=2.0,
        generator=generator_b,
    )

    torch.testing.assert_close(point_a, point_b)
    torch.testing.assert_close(force_a, force_b)
    torch.testing.assert_close(torch.linalg.norm(point_a[:, :2], dim=-1), torch.full((8,), 0.02))
    assert torch.all(point_a[:, 2] >= 0.0)
    assert torch.all(point_a[:, 2] <= 0.1)
    force_norm = torch.linalg.norm(force_a, dim=-1)
    assert torch.all(force_norm >= 1.0)
    assert torch.all(force_norm <= 2.0)


def test_local_force_at_position_to_world_identity_and_rotated():
    point = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    force = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    pos = torch.zeros(1, 3, dtype=torch.float64)

    force_w, torque_w = local_force_at_position_to_world(
        point,
        force,
        pos,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
    )
    assert force_w.dtype == torch.float64
    assert torque_w.dtype == torch.float64
    torch.testing.assert_close(force_w, force)
    torch.testing.assert_close(torque_w, torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64))

    sqrt_half = 2.0 ** -0.5
    force_w, torque_w = local_force_at_position_to_world(
        point,
        force,
        pos,
        torch.tensor([[sqrt_half, 0.0, 0.0, sqrt_half]], dtype=torch.float64),
    )
    torch.testing.assert_close(force_w, torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float64), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torque_w, torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64), atol=1e-6, rtol=1e-6)


def test_pack_ccai_state_uses_16_slot_layout_with_yaw_cap():
    active = torch.arange(12, dtype=torch.float32)
    obj = torch.tensor([0.1, 0.2, 0.3])

    q = pack_ccai_state(active, obj)

    assert q.shape == (1, 16)
    torch.testing.assert_close(q[0, :12], active)
    torch.testing.assert_close(q[0, 12:16], torch.tensor([0.1, 0.2, 0.3, 0.3]))


def test_active12_to_env_action_hand_layouts():
    active = torch.arange(12, dtype=torch.float32).reshape(1, 12)
    default = torch.arange(16, dtype=torch.float32)

    allegro_action = active12_to_env_action(active, hand="allegro", default_dof_pos=default)
    torch.testing.assert_close(allegro_action[:, :12], active)
    torch.testing.assert_close(allegro_action[:, 12:16], default[8:12].reshape(1, 4))

    allegro_active_only_action = active12_to_env_action(
        active,
        hand="allegro",
        default_dof_pos=default,
        allegro_action_dim=12,
    )
    torch.testing.assert_close(allegro_active_only_action, active)

    proto5_action = active12_to_env_action(active, hand="proto5")
    torch.testing.assert_close(proto5_action, active)

    proto5_wrist_action = active12_to_env_action(
        active,
        hand="proto5",
        proto5_control_wrist=True,
        current_wrist_joints=torch.tensor([[0.25, -0.05]]),
    )
    assert proto5_wrist_action.shape == (1, 14)
    torch.testing.assert_close(proto5_wrist_action[:, :12], active)
    torch.testing.assert_close(proto5_wrist_action[:, 12:14], torch.tensor([[0.25, -0.05]]))


class _FakeHardwareRuntime:
    def __init__(self):
        self.state15 = torch.arange(15, dtype=torch.float32).reshape(1, 15)
        self.state15[:, 12:15] = torch.tensor([[0.1, 0.2, 0.3]])
        self.command_targets = []
        self.reset_calls = 0
        self.close_calls = 0
        self.position = torch.tensor([[0.7, 0.8, 0.9]], dtype=torch.float32)
        self.wrenches = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)
        self.points = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3) * 0.01

    def get_current_state(self, device):
        return self.state15.to(device=device)

    def get_screwdriver_position_robot(self, device, dtype=torch.float32):
        return self.position.to(device=device, dtype=dtype)

    def step(self, action_target):
        self.command_targets.append(torch.as_tensor(action_target, dtype=torch.float32).detach().clone())
        self.state15[:, :12] = self.command_targets[-1].reshape(1, -1)[:, :12]
        return self.state15

    def reset(self):
        self.reset_calls += 1
        return self.state15, {}

    def close(self):
        self.close_calls += 1

    def read_wrenches_robot(self, device, dtype=torch.float32):
        return self.wrenches.to(device=device, dtype=dtype)

    def read_contact_forces_robot(self, device, dtype=torch.float32):
        return self.wrenches[..., :3].to(device=device, dtype=dtype)

    def get_contact_points_robot(self, device, dtype=torch.float32):
        return self.points.to(device=device, dtype=dtype)


class _FakeNamedHardwareRuntime(_FakeHardwareRuntime):
    def __init__(self, names, positions):
        super().__init__()
        self.measured_names = tuple(names)
        self.measured_positions = torch.as_tensor(positions, dtype=torch.float32).reshape(1, -1)

    def get_measured_joint_state(self, device, dtype=torch.float32):
        return self.measured_names, self.measured_positions.to(device=device, dtype=dtype)


def test_hardware_recovery_env_packs_12_joint_plus_observed_pose_state():
    runtime = _FakeHardwareRuntime()
    env = HardwareScrewdriverRecoveryEnv(
        {
            "hand": "proto5",
            "sim_device": "cpu",
            "screwdriver_friction": 2.5,
            "yaw_joint_friction": 0.03,
        },
        runtime=runtime,
        device="cpu",
    )

    state = env.get_state()

    assert state["q"].shape == (1, 16)
    torch.testing.assert_close(state["q"][0, :12], torch.arange(12, dtype=torch.float32))
    torch.testing.assert_close(state["q"][0, 12:16], torch.tensor([0.1, 0.2, 0.3, 0.3]))
    torch.testing.assert_close(env.table_pose, torch.tensor([0.7, 0.8, 0.9]))
    assert env.get_environment_parameters() == {
        "screwdriver_friction": pytest.approx(2.5),
        "yaw_joint_friction": pytest.approx(0.03),
    }


def test_hardware_recovery_env_can_keep_hardcoded_position_with_live_orientation():
    runtime = _FakeHardwareRuntime()
    runtime.position = torch.tensor([[0.7, 0.8, 0.9]], dtype=torch.float32)
    env = HardwareScrewdriverRecoveryEnv(
        {
            "hand": "proto5",
            "sim_device": "cpu",
            "hardware_use_live_screwdriver_position": False,
        },
        runtime=runtime,
        device="cpu",
    )

    state = env.get_state()

    torch.testing.assert_close(
        env.table_pose,
        torch.tensor(isaacsim_recovery_utils.DEFAULT_SCREWDRIVER_TABLE_POSE, dtype=torch.float32),
    )
    torch.testing.assert_close(state["q"][0, 12:16], torch.tensor([0.1, 0.2, 0.3, 0.3]))


def test_proto5_hardware_full_dof_reference_maps_named_live_wrist_and_fingers():
    names = (
        PROTO5_ACTIVE_JOINT_NAMES[:4]
        + PROTO5_WRIST_JOINT_NAMES
        + PROTO5_ACTIVE_JOINT_NAMES[4:]
    )
    positions_by_name = {name: 10.0 + idx for idx, name in enumerate(names)}
    runtime = _FakeNamedHardwareRuntime(names, [positions_by_name[name] for name in names])
    env = HardwareScrewdriverRecoveryEnv(
        {"hand": "proto5", "sim_device": "cpu", "hardware_track_wrist_state": True},
        runtime=runtime,
        device="cpu",
    )

    full = env.get_full_dof_reference()

    assert full.shape == (18,)
    for joint_name in PROTO5_WRIST_JOINT_NAMES + PROTO5_ACTIVE_JOINT_NAMES:
        expected = torch.tensor(positions_by_name[joint_name], dtype=torch.float32)
        torch.testing.assert_close(full[PROTO5_ALL_JOINT_NAMES.index(joint_name)], expected)
    state = env.get_state()
    assert state["q"].shape == (1, 16)


def test_proto5_hardware_full_dof_reference_requires_wrist_when_tracking_enabled():
    runtime = _FakeNamedHardwareRuntime(PROTO5_ACTIVE_JOINT_NAMES, torch.arange(12, dtype=torch.float32))
    env = HardwareScrewdriverRecoveryEnv(
        {"hand": "proto5", "sim_device": "cpu", "hardware_track_wrist_state": True},
        runtime=runtime,
        device="cpu",
    )

    with pytest.raises(RuntimeError, match="wrist joints are unavailable"):
        env.get_full_dof_reference()


def test_hardware_recovery_env_uses_dataset_orientation_when_runtime_reports_zero():
    runtime = _FakeHardwareRuntime()
    runtime.state15[:, 12:15] = 0.0
    env = HardwareScrewdriverRecoveryEnv({"hand": "proto5", "sim_device": "cpu"}, runtime=runtime, device="cpu")

    env.set_observed_object_orientation(torch.tensor([0.4, -0.2, 0.7]))

    state = env.get_state()

    torch.testing.assert_close(state["q"][0, 12:16], torch.tensor([0.4, -0.2, 0.7, 0.7]))


def test_hardware_recovery_env_prefers_nonzero_runtime_orientation_over_dataset_fallback():
    runtime = _FakeHardwareRuntime()
    env = HardwareScrewdriverRecoveryEnv({"hand": "proto5", "sim_device": "cpu"}, runtime=runtime, device="cpu")

    env.set_observed_object_orientation(torch.tensor([0.4, -0.2, 0.7]))

    state = env.get_state()

    torch.testing.assert_close(state["q"][0, 12:16], torch.tensor([0.1, 0.2, 0.3, 0.3]))


def test_hardware_recovery_env_can_freeze_dataset_orientation_over_live_runtime():
    runtime = _FakeHardwareRuntime()
    env = HardwareScrewdriverRecoveryEnv(
        {
            "hand": "proto5",
            "sim_device": "cpu",
            "hardware_use_live_screwdriver_orientation": False,
        },
        runtime=runtime,
        device="cpu",
    )

    env.set_observed_object_orientation(torch.tensor([0.4, -0.2, 0.7]))

    state = env.get_state()

    torch.testing.assert_close(state["q"][0, 12:16], torch.tensor([0.4, -0.2, 0.7, 0.7]))


def test_hardware_recovery_env_can_capture_current_orientation_as_observed_fallback():
    runtime = _FakeHardwareRuntime()
    env = HardwareScrewdriverRecoveryEnv(
        {
            "hand": "proto5",
            "sim_device": "cpu",
            "hardware_use_live_screwdriver_orientation": False,
        },
        runtime=runtime,
        device="cpu",
    )
    env.set_observed_object_orientation(torch.tensor([0.4, -0.2, 0.7]))
    runtime.state15[:, 12:15] = torch.tensor([[0.8, 0.9, 1.0]])

    captured = env.capture_observed_object_orientation()
    state = env.get_state()

    torch.testing.assert_close(captured, torch.tensor([[0.8, 0.9, 1.0]]))
    torch.testing.assert_close(state["q"][0, 12:16], torch.tensor([0.8, 0.9, 1.0, 1.0]))


def test_hardware_recovery_env_step_and_set_pose_delegate_active_12d_targets():
    runtime = _FakeHardwareRuntime()
    env = HardwareScrewdriverRecoveryEnv({"hand": "proto5", "sim_device": "cpu"}, runtime=runtime, device="cpu")

    env.step(torch.arange(16, dtype=torch.float32).reshape(1, 16))
    env.set_pose(torch.arange(20, dtype=torch.float32))

    assert len(runtime.command_targets) == 2
    torch.testing.assert_close(runtime.command_targets[0], torch.arange(12, dtype=torch.float32).reshape(1, 12))
    torch.testing.assert_close(runtime.command_targets[1], torch.arange(12, dtype=torch.float32).reshape(1, 12))


def test_hardware_recovery_env_tactile_methods_return_runtime_signals_without_noise():
    runtime = _FakeHardwareRuntime()
    env = HardwareScrewdriverRecoveryEnv({"hand": "proto5", "sim_device": "cpu"}, runtime=runtime, device="cpu")

    tactile = env.get_tactile_observation()

    torch.testing.assert_close(tactile["contact_wrenches"], runtime.wrenches)
    torch.testing.assert_close(tactile["contact_forces"], runtime.wrenches[..., :3])
    torch.testing.assert_close(tactile["contact_points"], runtime.points)
    torch.testing.assert_close(env.get_force_sensor_data(), runtime.wrenches[..., :3].reshape(1, 9))


def test_hardware_visualization_shim_does_not_publish_commands():
    runtime = _FakeHardwareRuntime()
    env = HardwareScrewdriverRecoveryEnv({"hand": "proto5", "sim_device": "cpu"}, runtime=runtime, device="cpu")
    shim = HardwareVisualizationShim(env)

    shim.set_pose(torch.ones(16))
    shim.zero_obj_velocity()
    shim.write_image()

    assert runtime.command_targets == []
    torch.testing.assert_close(shim.get_state()["q"], env.get_state()["q"])


class _FakeData:
    def __init__(self, joint_pos, default_joint_pos=None):
        self.joint_pos = joint_pos
        if default_joint_pos is not None:
            self.default_joint_pos = default_joint_pos


class _FakeAsset:
    def __init__(self, joint_names, joint_pos, root_physx_view=None, default_joint_pos=None):
        self._joint_name_to_id = {name: idx for idx, name in enumerate(joint_names)}
        self.data = _FakeData(joint_pos, default_joint_pos=default_joint_pos)
        if root_physx_view is not None:
            self.root_physx_view = root_physx_view

    def find_joints(self, joint_names, preserve_order=True):
        if isinstance(joint_names, str):
            joint_names = (joint_names,)
        return [self._joint_name_to_id[name] for name in joint_names], list(joint_names)


class _FakeScene(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.write_count = 0
        self.update_count = 0

    def write_data_to_sim(self):
        self.write_count += 1

    def update(self, dt):
        self.update_count += 1


class _FakeSim:
    def __init__(self, *, has_gui=False, has_rtx=False):
        self._has_gui = bool(has_gui)
        self._has_rtx = bool(has_rtx)
        self.forward_count = 0
        self.render_count = 0

    def has_gui(self):
        return self._has_gui

    def has_rtx_sensors(self):
        return self._has_rtx

    def forward(self):
        self.forward_count += 1

    def render(self):
        self.render_count += 1

    def get_physics_dt(self):
        return 1.0 / 120.0


class _FakeUnwrapped:
    def __init__(self, robot, obj, *, scene_extra=None, sim=None):
        self.device = torch.device("cpu")
        self.num_envs = 1
        scene_items = {"robot": robot, "obj": obj}
        if scene_extra:
            scene_items.update(scene_extra)
        self.scene = _FakeScene(scene_items)
        self.sim = sim if sim is not None else _FakeSim()


class _FakeActionSpace:
    def __init__(self, shape):
        self.shape = shape


class _FakeEnv:
    def __init__(self, robot, obj, action_shape=(16,), scene_extra=None, sim=None):
        self.unwrapped = _FakeUnwrapped(robot, obj, scene_extra=scene_extra, sim=sim)
        self.action_space = _FakeActionSpace(action_shape)
        self.last_action = None
        self.actions = []

    def step(self, action):
        self.last_action = action
        self.actions.append(action.detach().clone())
        return {"step": len(self.actions), "action": action}

    def reset(self):
        return None


def _wrapper_from_scene(scene, *, hand="proto5"):
    wrapper = object.__new__(IsaacSimScrewdriverRecoveryEnv)
    wrapper._unwrapped = types.SimpleNamespace(scene=scene)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 1
    wrapper.hand = hand
    return wrapper


def test_isaacsim_wrapper_reports_screwdriver_position_in_robot_frame():
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        )
    )
    obj = types.SimpleNamespace(
        data=types.SimpleNamespace(root_pos_w=torch.tensor([[1.4, 2.5, 3.6]], dtype=torch.float32))
    )
    wrapper = _wrapper_from_scene({"robot": robot, "obj": obj})

    position = wrapper.get_screwdriver_position_robot("cpu")

    torch.testing.assert_close(position, torch.tensor([[0.4, 0.5, 0.6]], dtype=torch.float32))


def test_isaacsim_wrapper_contact_wrenches_use_model_mismatch_proto5_6af_helper():
    from model_mismatch.utils.proto5_wrenches import extract_proto5_6af_wrenches_robot_frame

    body_names = [
        "RHand_I3Y_LINK",
        "RHand_I6AF_LINK",
        "RHand_M3Y_LINK",
        "RHand_M6AF_LINK",
        "RHand_T3Y_LINK",
        "RHand_T6AF_LINK",
    ]
    raw_wrenches = torch.zeros((1, len(body_names), 6), dtype=torch.float32)
    for finger_idx, body_name in enumerate(("RHand_I6AF_LINK", "RHand_M6AF_LINK", "RHand_T6AF_LINK")):
        raw_wrenches[0, body_names.index(body_name)] = torch.arange(6, dtype=torch.float32) + 10.0 * finger_idx
    robot = types.SimpleNamespace(
        body_names=body_names,
        data=types.SimpleNamespace(
            body_incoming_joint_wrench_b=raw_wrenches,
            body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * len(body_names), dtype=torch.float32).transpose(0, 1),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        ),
    )
    wrapper = _wrapper_from_scene({"robot": robot, "obj": types.SimpleNamespace()})

    expected = extract_proto5_6af_wrenches_robot_frame(robot, env_ids=[0], device="cpu")

    torch.testing.assert_close(wrapper.get_contact_wrenches(strict=True), expected)


class _FakeCamera:
    def __init__(self):
        rgb = torch.tensor(
            [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]],
            dtype=torch.float32,
        )
        self.data = types.SimpleNamespace(output={"rgb": rgb})


class _FakePhysxView:
    def __init__(self, friction_properties):
        self._friction_properties = friction_properties

    def get_dof_friction_properties(self):
        return self._friction_properties


class _FakeRandomizationObj:
    def __init__(self):
        self.data = types.SimpleNamespace(
            root_pos_w=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            root_lin_vel_w=torch.ones(1, 3),
            root_ang_vel_w=torch.ones(1, 3),
            joint_pos=torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32),
            joint_vel=torch.ones(1, 4),
        )
        self.last_root_pose = None
        self.last_root_velocity = None
        self.last_joint_pos = None
        self.last_joint_vel = None

    def write_root_pose_to_sim(self, root_pose, env_ids):
        self.last_root_pose = root_pose.clone()

    def write_root_velocity_to_sim(self, root_velocity, env_ids):
        self.last_root_velocity = root_velocity.clone()

    def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids):
        self.last_joint_pos = joint_pos.clone()
        self.last_joint_vel = joint_vel.clone()


class _FakeRandomizationEnv:
    def __init__(self):
        self.device = torch.device("cpu")
        self.num_envs = 1
        self.scene = {"obj": _FakeRandomizationObj()}
        self.table_pose = torch.zeros(3)
        self.obj_pose = torch.zeros(3)
        self.synced = False

    def _obj_orientation_joint_ids(self):
        return [0, 1, 2]

    def _sync_scene(self):
        self.synced = True


def test_wrapper_get_state_packs_allegro_ccai_order():
    robot_joint_names = ALLEGRO_ACTIVE_JOINT_NAMES[:8] + ALLEGRO_RING_JOINT_NAMES + ALLEGRO_ACTIVE_JOINT_NAMES[8:]
    robot = _FakeAsset(robot_joint_names, torch.arange(16, dtype=torch.float32).reshape(1, 16))
    obj = _FakeAsset(OBJ_ORIENTATION_JOINT_NAMES, torch.tensor([[0.1, 0.2, 0.3]]))
    wrapper = IsaacSimScrewdriverRecoveryEnv(_FakeEnv(robot, obj), hand="allegro")

    q = wrapper.get_state()["q"]

    assert q.shape == (1, 16)
    torch.testing.assert_close(q[0, :12], torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15], dtype=torch.float32))
    torch.testing.assert_close(q[0, 12:16], torch.tensor([0.1, 0.2, 0.3, 0.3]))


def test_wrapper_get_state_packs_proto5_active_joints():
    robot = _FakeAsset(PROTO5_ALL_JOINT_NAMES, torch.arange(18, dtype=torch.float32).reshape(1, 18))
    obj = _FakeAsset(OBJ_ORIENTATION_JOINT_NAMES, torch.tensor([[0.4, 0.5, 0.6]]))
    wrapper = IsaacSimScrewdriverRecoveryEnv(_FakeEnv(robot, obj), hand="proto5")

    q = wrapper.get_state()["q"]
    expected_active = torch.tensor(
        [PROTO5_ALL_JOINT_NAMES.index(name) for name in PROTO5_ACTIVE_JOINT_NAMES],
        dtype=torch.float32,
    )

    assert q.shape == (1, 16)
    torch.testing.assert_close(q[0, :12], expected_active)
    torch.testing.assert_close(q[0, 12:16], torch.tensor([0.4, 0.5, 0.6, 0.6]))


def test_allegro_fallback_default_dof_pos_has_zero_ring_joints():
    fallback = torch.tensor(ALLEGRO_DEFAULT_FULL_JOINT_POS, dtype=torch.float32)

    torch.testing.assert_close(fallback[8:12], torch.zeros(4))


@pytest.mark.parametrize(
    ("hand", "joint_names"),
    [
        ("allegro", ALLEGRO_ACTIVE_JOINT_NAMES[:8] + ALLEGRO_RING_JOINT_NAMES + ALLEGRO_ACTIVE_JOINT_NAMES[8:]),
        ("proto5", PROTO5_ALL_JOINT_NAMES),
    ],
)
def test_wrapper_refreshes_ordered_default_dof_pos_from_wrapped_env(hand, joint_names):
    robot_joint_names = tuple(reversed(joint_names))
    default_joint_pos = torch.arange(len(robot_joint_names), dtype=torch.float32).reshape(1, -1)
    robot = _FakeAsset(
        robot_joint_names,
        torch.zeros_like(default_joint_pos),
        default_joint_pos=default_joint_pos,
    )
    obj = _FakeAsset(OBJ_ORIENTATION_JOINT_NAMES, torch.tensor([[0.1, 0.2, 0.3]]))
    wrapper = IsaacSimScrewdriverRecoveryEnv(_FakeEnv(robot, obj), hand=hand)
    expected = torch.tensor([robot_joint_names.index(name) for name in joint_names], dtype=torch.float32)

    torch.testing.assert_close(wrapper.default_dof_pos[0], expected)

    robot.data.default_joint_pos = default_joint_pos + 100.0
    wrapper.reset()

    torch.testing.assert_close(wrapper.default_dof_pos[0], expected + 100.0)


def test_wrapper_step_uses_refreshed_allegro_default_ring_targets():
    default_joint_pos = torch.tensor(
        [[0.1, 0.6, 0.6, 0.6, -0.1, 0.5, 0.9, 0.9, 4.0, 5.0, 6.0, 7.0, 1.2, 0.3, 0.3, 1.2]],
        dtype=torch.float32,
    )
    robot_joint_names = ALLEGRO_ACTIVE_JOINT_NAMES[:8] + ALLEGRO_RING_JOINT_NAMES + ALLEGRO_ACTIVE_JOINT_NAMES[8:]
    robot = _FakeAsset(
        robot_joint_names,
        torch.zeros_like(default_joint_pos),
        default_joint_pos=default_joint_pos,
    )
    obj = _FakeAsset(OBJ_ORIENTATION_JOINT_NAMES, torch.tensor([[0.1, 0.2, 0.3]]))
    env = _FakeEnv(robot, obj, action_shape=(16,))
    wrapper = IsaacSimScrewdriverRecoveryEnv(env, hand="allegro")
    wrapper._clear_external_force_torque = lambda: None
    active = torch.arange(12, dtype=torch.float32)

    wrapper.step(active)

    torch.testing.assert_close(env.last_action[:, :12], active.reshape(1, 12))
    torch.testing.assert_close(env.last_action[:, 12:16], torch.tensor([[4.0, 5.0, 6.0, 7.0]]))


def test_wrapper_physical_readback_returns_sampled_proto5_friction():
    robot = _FakeAsset(PROTO5_ALL_JOINT_NAMES, torch.arange(18, dtype=torch.float32).reshape(1, 18))
    obj = _FakeAsset(OBJ_ORIENTATION_JOINT_NAMES, torch.tensor([[0.4, 0.5, 0.6]]))
    wrapper = IsaacSimScrewdriverRecoveryEnv(_FakeEnv(robot, obj), hand="proto5")
    wrapper.unwrapped._proto5_contact_friction_tensor = torch.tensor([3.25])
    wrapper.unwrapped._screwdriver_joint_friction_tensor = torch.tensor([0.17])

    env_params = wrapper.get_environment_parameters()

    assert env_params == {
        "screwdriver_friction": pytest.approx(3.25),
        "yaw_joint_friction": pytest.approx(0.17),
    }


def test_wrapper_physical_readback_matches_allegro_collector_fallbacks():
    robot_joint_names = ALLEGRO_ACTIVE_JOINT_NAMES[:8] + ALLEGRO_RING_JOINT_NAMES + ALLEGRO_ACTIVE_JOINT_NAMES[8:]
    robot = _FakeAsset(robot_joint_names, torch.arange(16, dtype=torch.float32).reshape(1, 16))
    joint_friction = torch.zeros(1, 3, 3)
    joint_friction[0, 2, 0] = 0.23
    obj = _FakeAsset(
        OBJ_ORIENTATION_JOINT_NAMES,
        torch.tensor([[0.1, 0.2, 0.3]]),
        root_physx_view=_FakePhysxView(joint_friction),
    )
    wrapper = IsaacSimScrewdriverRecoveryEnv(_FakeEnv(robot, obj), hand="allegro")
    wrapper.unwrapped._screwdriver_friction_values = {0: {"static_friction": 2.75}}

    env_params = wrapper.get_environment_parameters()

    assert env_params == {
        "screwdriver_friction": pytest.approx(2.75),
        "yaw_joint_friction": pytest.approx(0.23),
    }


def test_recovery_physical_kwargs_use_env_yaw_friction_by_default():
    class Env:
        def get_environment_parameters(self, env_id=0):
            return {"screwdriver_friction": 3.0, "yaw_joint_friction": 0.21}

    kwargs = screwdriver_isaacsim_recovery.get_recovery_planner_physical_kwargs(
        Env(),
        {
            "friction_coefficient": 0.9,
            "yaw_joint_friction": 0.03,
            "planner_use_env_yaw_joint_friction": True,
            "disable_planner_yaw_friction_model": True,
            "planner_use_yaw_inertia_model": False,
        },
    )

    assert kwargs["friction_coefficient"] == pytest.approx(0.9)
    assert kwargs["yaw_joint_friction"] == pytest.approx(0.03)
    assert kwargs["yaw_friction_model_path"] is None
    assert kwargs["yaw_inertia_model_path"] is None


def test_recovery_physical_kwargs_respect_yaw_override_and_model_toggles(tmp_path):
    friction_model = tmp_path / "yaw_friction_model.json"
    inertia_model = tmp_path / "yaw_inertia_model.json"
    friction_model.write_text("{}", encoding="utf-8")
    inertia_model.write_text("{}", encoding="utf-8")

    class Env:
        def get_environment_parameters(self, env_id=0):
            return {"screwdriver_friction": 1.5, "yaw_joint_friction": 0.31}

    kwargs = screwdriver_isaacsim_recovery.get_recovery_planner_physical_kwargs(
        Env(),
        {
            "planner_use_env_yaw_joint_friction": False,
            "planner_yaw_joint_friction_override": 0.04,
            "disable_planner_yaw_friction_model": False,
            "planner_yaw_friction_model_path": str(friction_model),
            "planner_use_yaw_inertia_model": True,
            "planner_yaw_inertia_model_path": str(inertia_model),
        },
    )

    assert kwargs["yaw_joint_friction"] == pytest.approx(0.04)
    assert kwargs["yaw_friction_model_path"] == str(friction_model)
    assert kwargs["yaw_inertia_model_path"] == str(inertia_model)


def test_recovery_physical_kwargs_pass_planner_screwdriver_urdf(monkeypatch, tmp_path):
    planner_urdf_path = tmp_path / "screwdriver_3d.urdf"
    planner_urdf_path.write_text(
        "<robot name='screwdriver'><link name='base'><inertial><mass value='0.42'/></inertial></link></robot>",
        encoding="utf-8",
    )
    if str(screwdriver_isaacsim_recovery.MODEL_MISMATCH_PATH) not in sys.path:
        sys.path.insert(0, str(screwdriver_isaacsim_recovery.MODEL_MISMATCH_PATH))
    from model_mismatch.utils import screwdriver_csvto_planning

    captured = {}

    def fake_physical_kwargs(env_params, **kwargs):
        captured["env_params"] = env_params
        captured.update(kwargs)
        return {
            "object_asset_path": str(kwargs["screwdriver_urdf_path"]),
            "object_mass": 0.42,
            "friction_coefficient": 1.0,
            "yaw_joint_friction": 0.2,
        }

    monkeypatch.setattr(
        screwdriver_csvto_planning,
        "get_screwdriver_turn_problem_physical_kwargs",
        fake_physical_kwargs,
    )

    class Env:
        def get_environment_parameters(self, env_id=0):
            return {"screwdriver_friction": 2.0, "yaw_joint_friction": 0.2}

    kwargs = screwdriver_isaacsim_recovery.get_recovery_planner_physical_kwargs(
        Env(),
        {
            "planner_screwdriver_urdf_path": str(planner_urdf_path),
            "planner_use_env_yaw_joint_friction": True,
            "disable_planner_yaw_friction_model": True,
            "planner_use_yaw_inertia_model": False,
        },
    )

    assert captured["screwdriver_urdf_path"] == str(planner_urdf_path)
    assert kwargs["object_asset_path"] == str(planner_urdf_path)
    assert kwargs["object_mass"] == pytest.approx(0.42)


def test_randomize_isaacsim_object_start_matches_collector_fields():
    env = _FakeRandomizationEnv()
    config = {
        "randomize_obj_start": True,
        "obj_position_noise_range_x": [-0.01, 0.02],
        "obj_position_noise_range_y": [0.03, 0.04],
        "obj_position_noise_range_z": [-0.05, -0.02],
        "obj_orientation_noise_std": 0.5,
    }
    rng = np.random.default_rng(17)
    expected_rng = np.random.default_rng(17)
    expected_offset = torch.tensor(
        [
            expected_rng.uniform(-0.01, 0.02),
            expected_rng.uniform(0.03, 0.04),
            expected_rng.uniform(-0.05, -0.02),
        ],
        dtype=torch.float32,
    )
    expected_roll_pitch = torch.tensor(
        expected_rng.normal(0.0, 0.5 * 0.2, (2,)),
        dtype=torch.float32,
    )
    expected_yaw = torch.tensor(
        expected_rng.normal(0.0, 0.5),
        dtype=torch.float32,
    )

    result = screwdriver_isaacsim_recovery.randomize_isaacsim_object_start(env, config, rng)

    expected_pos = torch.tensor([1.0, 2.0, 3.0]) + expected_offset
    torch.testing.assert_close(result["screwdriver_pos_offset_world"][0], expected_offset)
    torch.testing.assert_close(result["screwdriver_pos_world"][0], expected_pos)
    torch.testing.assert_close(env.scene["obj"].last_root_pose[0, :3], expected_pos)
    torch.testing.assert_close(env.scene["obj"].last_root_velocity, torch.zeros(1, 6))
    torch.testing.assert_close(env.table_pose, expected_pos)
    torch.testing.assert_close(env.obj_pose, expected_pos)

    expected_joint_pos = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    expected_joint_pos[0, 0:2] += expected_roll_pitch
    expected_joint_pos[0, 2] += expected_yaw
    torch.testing.assert_close(result["obj_joint_pos"], expected_joint_pos)
    torch.testing.assert_close(env.scene["obj"].last_joint_pos, expected_joint_pos)
    torch.testing.assert_close(env.scene["obj"].last_joint_vel[:, :3], torch.zeros(1, 3))
    assert env.synced


def test_wrapper_step_handles_headless_none_frame_id():
    robot_joint_names = ALLEGRO_ACTIVE_JOINT_NAMES[:8] + ALLEGRO_RING_JOINT_NAMES + ALLEGRO_ACTIVE_JOINT_NAMES[8:]
    robot = _FakeAsset(robot_joint_names, torch.arange(16, dtype=torch.float32).reshape(1, 16))
    obj = _FakeAsset(OBJ_ORIENTATION_JOINT_NAMES, torch.tensor([[0.1, 0.2, 0.3]]))
    env = _FakeEnv(robot, obj, action_shape=(12,))
    wrapper = IsaacSimScrewdriverRecoveryEnv(env, hand="allegro")
    wrapper._clear_external_force_torque = lambda: None
    wrapper._step_index = 5
    wrapper.frame_id = None

    wrapper.step(torch.arange(12, dtype=torch.float32))

    assert wrapper.frame_id is None
    assert wrapper._step_index == 1
    torch.testing.assert_close(env.last_action, torch.arange(12, dtype=torch.float32).reshape(1, 12))


def test_wrapper_repeats_recovery_action_three_times():
    robot_joint_names = ALLEGRO_ACTIVE_JOINT_NAMES[:8] + ALLEGRO_RING_JOINT_NAMES + ALLEGRO_ACTIVE_JOINT_NAMES[8:]
    robot = _FakeAsset(robot_joint_names, torch.arange(16, dtype=torch.float32).reshape(1, 16))
    obj = _FakeAsset(OBJ_ORIENTATION_JOINT_NAMES, torch.tensor([[0.1, 0.2, 0.3]]))
    env = _FakeEnv(robot, obj, action_shape=(12,))
    wrapper = IsaacSimScrewdriverRecoveryEnv(env, hand="allegro", action_repeat=3)
    wrapper._clear_external_force_torque = lambda: None

    result = wrapper.step(torch.arange(12, dtype=torch.float32))

    assert result["step"] == 3
    assert len(env.actions) == 3
    assert wrapper._step_index == 3
    for action in env.actions:
        torch.testing.assert_close(action, torch.arange(12, dtype=torch.float32).reshape(1, 12))


def test_wrapper_reset_and_set_pose_force_render():
    robot_joint_names = ALLEGRO_ACTIVE_JOINT_NAMES[:8] + ALLEGRO_RING_JOINT_NAMES + ALLEGRO_ACTIVE_JOINT_NAMES[8:]
    robot = _FakeAsset(robot_joint_names, torch.arange(16, dtype=torch.float32).reshape(1, 16))
    obj = _FakeAsset(OBJ_ORIENTATION_JOINT_NAMES, torch.tensor([[0.1, 0.2, 0.3]]))
    sim = _FakeSim(has_gui=True)
    env = _FakeEnv(robot, obj, action_shape=(12,), sim=sim)
    wrapper = IsaacSimScrewdriverRecoveryEnv(env, hand="allegro")

    wrapper.reset()
    reset_render_count = sim.render_count
    assert reset_render_count >= 1

    wrapper._write_robot_root_default = lambda env_ids: None
    wrapper._write_obj_root_default = lambda env_ids: None
    wrapper._write_robot_joint_state = lambda active_joint_pos, env_ids: None
    wrapper._write_obj_joint_state = lambda obj_orientation, env_ids: None
    wrapper.set_pose(torch.zeros(16, dtype=torch.float32))

    assert sim.render_count > reset_render_count


def test_wrapper_set_pose_updates_object_pose_from_robot_frame_position():
    class PoseData:
        def __init__(self, root_pos):
            root_state = torch.zeros(1, 13, dtype=torch.float32)
            root_state[0, :3] = torch.as_tensor(root_pos, dtype=torch.float32)
            root_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            self.default_root_state = root_state

    class PoseAsset:
        def __init__(self, root_pos):
            self.data = PoseData(root_pos)
            self.last_root_pose = None
            self.last_root_velocity = None

        def write_root_link_pose_to_sim(self, root_pose, env_ids):
            self.last_root_pose = root_pose.detach().clone()

        def write_root_com_velocity_to_sim(self, root_velocity, env_ids):
            self.last_root_velocity = root_velocity.detach().clone()

    class PoseScene(dict):
        env_origins = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32)

    wrapper = object.__new__(IsaacSimScrewdriverRecoveryEnv)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 1
    wrapper._unwrapped = types.SimpleNamespace(
        scene=PoseScene(
            {
                "robot": PoseAsset([1.0, 2.0, 3.0]),
                "obj": PoseAsset([0.0, 0.0, 1.205]),
            }
        )
    )
    wrapper.table_pose = torch.zeros(3)
    wrapper.obj_pose = torch.zeros(3)
    wrapper._write_robot_root_default = lambda env_ids: None
    wrapper._write_robot_joint_state = lambda active_joint_pos, env_ids: None
    wrapper._write_obj_joint_state = lambda obj_orientation, env_ids: None
    wrapper._sync_scene = lambda: None
    wrapper._record_frame = lambda force_render=False, sync_joint_targets=False: None

    wrapper.set_pose(torch.zeros(15), screwdriver_pos_robot=torch.tensor([0.4, 0.5, 0.6]))

    expected_pos = torch.tensor([11.4, 22.5, 33.6], dtype=torch.float32)
    torch.testing.assert_close(wrapper.scene["obj"].last_root_pose[0, :3], expected_pos)
    torch.testing.assert_close(wrapper.table_pose, expected_pos)
    torch.testing.assert_close(wrapper.obj_pose, expected_pos)


def test_wrapper_saves_initial_and_repeated_step_frames(tmp_path):
    robot_joint_names = ALLEGRO_ACTIVE_JOINT_NAMES[:8] + ALLEGRO_RING_JOINT_NAMES + ALLEGRO_ACTIVE_JOINT_NAMES[8:]
    robot = _FakeAsset(robot_joint_names, torch.arange(16, dtype=torch.float32).reshape(1, 16))
    obj = _FakeAsset(OBJ_ORIENTATION_JOINT_NAMES, torch.tensor([[0.1, 0.2, 0.3]]))
    env = _FakeEnv(
        robot,
        obj,
        action_shape=(12,),
        scene_extra={"tiled_camera": _FakeCamera()},
        sim=_FakeSim(has_rtx=True),
    )
    wrapper = IsaacSimScrewdriverRecoveryEnv(env, hand="allegro", action_repeat=3, save_recovery_frames=True)
    wrapper._clear_external_force_torque = lambda: None
    wrapper.frame_fpath = tmp_path

    wrapper.frame_id = 0
    assert wrapper.frame_id == 1
    assert (tmp_path / "frame_000000.png").exists()

    wrapper.step(torch.arange(12, dtype=torch.float32))

    assert wrapper.frame_id == 4
    assert (tmp_path / "frame_000001.png").exists()
    assert (tmp_path / "frame_000002.png").exists()
    assert (tmp_path / "frame_000003.png").exists()


def test_wrapper_frame_saving_requires_tiled_camera(tmp_path):
    robot_joint_names = ALLEGRO_ACTIVE_JOINT_NAMES[:8] + ALLEGRO_RING_JOINT_NAMES + ALLEGRO_ACTIVE_JOINT_NAMES[8:]
    robot = _FakeAsset(robot_joint_names, torch.arange(16, dtype=torch.float32).reshape(1, 16))
    obj = _FakeAsset(OBJ_ORIENTATION_JOINT_NAMES, torch.tensor([[0.1, 0.2, 0.3]]))
    wrapper = IsaacSimScrewdriverRecoveryEnv(
        _FakeEnv(robot, obj, action_shape=(12,), sim=_FakeSim(has_rtx=True)),
        hand="allegro",
        save_recovery_frames=True,
    )
    wrapper.frame_fpath = tmp_path

    with pytest.raises(RuntimeError, match="requires a tiled_camera sensor"):
        wrapper.frame_id = 0


def _load_legacy_recovery_module(monkeypatch, executed_modes, created_problems=None):
    def extract_state_vector(state, num_fingers, device, obj_dof=None, slice_end=None, hardcoded_dim=None):
        q = state["q"].reshape(-1).to(device=device, dtype=torch.float32).clone()
        if slice_end is not None:
            q = q[:slice_end]
        return q

    class FakePlanner:
        def __init__(self, problem):
            self.problem = problem

        def step(self, start):
            return torch.stack((torch.ones(12), torch.ones(12) * 2.0)), None

    class FakeTrajectoryExecutor:
        def __init__(self, params, env, sim_viz_env):
            self.env = env

        def execute_traj(self, **kwargs):
            mode = kwargs["mode"]
            executed_modes.append(mode)
            if mode == "turn":
                self.env.state[-1] = -1.2
            traj = self.env.state[:15].clone()
            return traj, [], [], [], [], [], [], False, kwargs["episode_num_steps"] + 1

    stubs = {
        "isaac_victor_envs": types.ModuleType("isaac_victor_envs"),
        "isaac_victor_envs.utils": types.ModuleType("isaac_victor_envs.utils"),
        "pytorch_kinematics": types.ModuleType("pytorch_kinematics"),
        "matplotlib": types.ModuleType("matplotlib"),
        "matplotlib.pyplot": types.ModuleType("matplotlib.pyplot"),
        "ccai.utils.allegro_utils": types.ModuleType("ccai.utils.allegro_utils"),
        "ccai.utils.recovery_utils": types.ModuleType("ccai.utils.recovery_utils"),
        "ccai.allegro_contact": types.ModuleType("ccai.allegro_contact"),
        "ccai.baselines.allegro_recovery_baselines": types.ModuleType("ccai.baselines.allegro_recovery_baselines"),
        "ccai.planning.contact_planning": types.ModuleType("ccai.planning.contact_planning"),
        "ccai.execution.trial_executor": types.ModuleType("ccai.execution.trial_executor"),
        "ccai.models.management.model_manager": types.ModuleType("ccai.models.management.model_manager"),
    }
    stubs["isaac_victor_envs.utils"].get_assets_dir = lambda: "/tmp"
    stubs["ccai.utils.allegro_utils"].convert_yaw_to_sine_cosine = lambda x: x
    stubs["ccai.utils.allegro_utils"].convert_sine_cosine_to_yaw = lambda x: x
    stubs["ccai.utils.allegro_utils"].visualize_trajectory = lambda *args, **kwargs: None
    stubs["ccai.utils.allegro_utils"].partial_to_full_state = lambda *args, **kwargs: None
    stubs["ccai.utils.allegro_utils"].extract_state_vector = extract_state_vector
    def create_problem(*args, **kwargs):
        if created_problems is not None:
            created_problems.append({"args": args, "kwargs": kwargs})
        return types.SimpleNamespace(
            dx=15,
            contact_scenes_for_viz=None,
            fingers=["index", "middle", "thumb"],
        )

    stubs["ccai.utils.recovery_utils"].create_allegro_screwdriver_problem = create_problem
    stubs["ccai.utils.recovery_utils"].create_planner = lambda problem, mode, params: FakePlanner(problem)
    stubs["ccai.utils.recovery_utils"].add_to_dataset = lambda *args, **kwargs: None
    stubs["ccai.utils.recovery_utils"].partial_to_full_trajectory = lambda *args, **kwargs: None
    stubs["ccai.utils.recovery_utils"].full_to_partial_trajectory = lambda *args, **kwargs: None
    stubs["ccai.utils.recovery_utils"].create_mode_planner_dict = lambda *args, **kwargs: {}
    stubs["ccai.utils.recovery_utils"].build_pregrasp_reference_target_kwargs = lambda *args, **kwargs: {}
    stubs["ccai.utils.recovery_utils"].stack_execution_timeseries_for_save = lambda *args, **kwargs: None
    stubs["ccai.allegro_contact"].AllegroManipulationProblem = type(
        "AllegroManipulationProblem",
        (),
        {"__init__": lambda self, *args, **kwargs: None},
    )
    stubs["ccai.allegro_contact"].PositionControlConstrainedSVGDMPC = object
    stubs["ccai.baselines.allegro_recovery_baselines"].BaselineRecoveryController = object
    stubs["ccai.baselines.allegro_recovery_baselines"].BaselineOODDetector = object
    stubs["ccai.baselines.allegro_recovery_baselines"].get_baseline_contact_sequence = lambda *args, **kwargs: ["turn"]
    stubs["ccai.baselines.allegro_recovery_baselines"].get_num_envs_for_baseline = lambda *args, **kwargs: 1
    stubs["ccai.baselines.allegro_recovery_baselines"].handle_baseline_trajectory_processing = (
        lambda traj, plans, contact, device: (traj, plans)
    )
    stubs["ccai.planning.contact_planning"].ContactPlanner = object
    stubs["ccai.execution.trial_executor"].TrajectoryExecutor = FakeTrajectoryExecutor
    stubs["ccai.models.management.model_manager"].ModelManager = object

    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = pathlib.Path(__file__).resolve().parents[1] / "examples" / "allegro_screwdriver.py"
    spec = importlib.util.spec_from_file_location(f"legacy_recovery_flow_{id(executed_modes)}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTrialEnv:
    def __init__(self):
        self.device = torch.device("cpu")
        self.state = torch.zeros(16, dtype=torch.float32)
        self.table_pose = torch.tensor([0.0, 0.0, 1.205])
        self.obj_pose = self.table_pose
        self.world_trans = object()
        self.default_dof_pos = torch.zeros(1, 16)
        self.external_wrench_perturb = True
        self.wrench_perturb_inds = []
        self.actions = []
        self.reset_count = 0

    def get_state(self):
        return {"q": self.state.reshape(1, -1).clone()}

    def step(self, action):
        self.actions.append(action.detach().clone())
        self.state[:12] = action.reshape(-1)[:12].to(dtype=torch.float32)

    def set_pose(self, state):
        self.state = state.reshape(-1).detach().clone().to(dtype=torch.float32)

    def set_external_wrench_perturb(self, enabled, rand_pct=None):
        self.external_wrench_perturb = bool(enabled)

    def reset(self):
        self.reset_count += 1


def _trial_params(pregrasp_only):
    return {
        "fingers": ["index", "middle", "thumb"],
        "device": "cpu",
        "visualize": False,
        "mode": "simulation",
        "external_wrench_perturb": False,
        "valve_goal": torch.zeros(3),
        "T": 1,
        "live_recovery": False,
        "skip_pregrasp": False,
        "pregrasp_only": pregrasp_only,
    }


def test_do_trial_runs_pregrasp_then_turn_when_not_pregrasp_only(monkeypatch, tmp_path):
    executed_modes = []
    legacy = _load_legacy_recovery_module(monkeypatch, executed_modes)
    legacy.all_pregrasp_states.clear()

    legacy.do_trial(_FakeTrialEnv(), _trial_params(pregrasp_only=False), tmp_path)

    assert executed_modes == ["turn"]
    assert len(legacy.all_pregrasp_states) == 1


def test_do_trial_preserves_pregrasp_only_short_circuit(monkeypatch, tmp_path):
    executed_modes = []
    legacy = _load_legacy_recovery_module(monkeypatch, executed_modes)
    legacy.all_pregrasp_states.clear()

    legacy.do_trial(_FakeTrialEnv(), _trial_params(pregrasp_only=True), tmp_path)

    assert executed_modes == []
    assert len(legacy.all_pregrasp_states) == 1


def test_do_trial_skip_pregrasp_stage_goes_directly_to_turn(monkeypatch, tmp_path):
    executed_modes = []
    legacy = _load_legacy_recovery_module(monkeypatch, executed_modes)
    legacy.all_pregrasp_states.clear()
    params = _trial_params(pregrasp_only=False)
    params["skip_pregrasp_stage"] = True
    env = _FakeTrialEnv()
    env.state[12:15] = torch.tensor([0.1, 0.2, 0.9])

    legacy.do_trial(env, params, tmp_path)

    assert executed_modes == ["turn"]
    assert len(legacy.all_pregrasp_states) == 0


def test_do_trial_uses_configured_min_force_dict(monkeypatch, tmp_path):
    executed_modes = []
    created_problems = []
    legacy = _load_legacy_recovery_module(monkeypatch, executed_modes, created_problems)
    legacy.all_pregrasp_states.clear()
    params = _trial_params(pregrasp_only=False)
    params["min_force_dict"] = {"index": 0.25, "middle": 0.75, "thumb": 1.25}

    legacy.do_trial(_FakeTrialEnv(), params, tmp_path)

    turn_problem = next(problem for problem in created_problems if problem["args"][0] == "turn")
    assert turn_problem["kwargs"]["min_force_dict"] == {
        "index": pytest.approx(0.25),
        "middle": pytest.approx(0.75),
        "thumb": pytest.approx(1.25),
    }


def test_create_problem_passes_proto5_full_dof_metadata():
    captured = {}

    class Proto5Screwdriver:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class Env:
        default_dof_pos = torch.arange(18, dtype=torch.float32).reshape(1, 18)
        table_pose = torch.tensor([0.0, 0.0, 1.205])
        obj_pose = table_pose
        world_trans = object()

    params = {
        "T": 3,
        "T_orig": 4,
        "chain": object(),
        "object_location": torch.tensor([0.0, 0.0, 1.205]),
        "object_type": "screwdriver",
        "optimize_force": True,
        "proto5_control_wrist": True,
        "robot_sdf_path_prefix": "/tmp/proto5",
        "friction_coefficient": 1.85,
        "yaw_joint_friction": 0.07,
        "dt": 0.5,
        "object_asset_path": "/tmp/generated_screwdriver_3d.urdf",
        "object_mass": 0.355,
        "yaw_friction_model_path": "/tmp/yaw_friction.json",
        "yaw_inertia_model_path": "/tmp/yaw_inertia.json",
    }

    create_allegro_screwdriver_problem(
        "turn",
        torch.zeros(15),
        torch.zeros(3),
        params,
        Env(),
        "cpu",
        AllegroScrewdriver=Proto5Screwdriver,
    )

    torch.testing.assert_close(captured["default_dof_pos"], torch.arange(18, dtype=torch.float32))
    torch.testing.assert_close(captured["full_dof_reference"], torch.arange(18, dtype=torch.float32))
    assert captured["control_wrist"] is True
    assert captured["robot_sdf_path_prefix"] == "/tmp/proto5"
    assert captured["friction_coefficient"] == pytest.approx(1.85)
    assert captured["yaw_joint_friction"] == pytest.approx(0.07)
    assert captured["dt"] == pytest.approx(0.5)
    assert captured["object_asset_path"] == "/tmp/generated_screwdriver_3d.urdf"
    assert captured["object_mass"] == pytest.approx(0.355)
    assert captured["yaw_friction_model_path"] == "/tmp/yaw_friction.json"
    assert captured["yaw_inertia_model_path"] == "/tmp/yaw_inertia.json"


def test_proto5_point_cache_setup_requires_curated_6af_caches(monkeypatch, tmp_path):
    ccai_root = tmp_path / "ccai"
    cache_dir = ccai_root / "data" / "cache" / "proto5_points"
    cache_dir.mkdir(parents=True)
    for cache_name in screwdriver_isaacsim_recovery.PROTO5_POINT_CACHE_NAMES:
        (cache_dir / cache_name).write_bytes(b"cache")

    monkeypatch.setattr(screwdriver_isaacsim_recovery, "CCAI_PATH", ccai_root)
    monkeypatch.setenv("PYTORCH_VOLUMETRIC_POINTS_CACHE_DIR", "/tmp/old-cache")

    screwdriver_isaacsim_recovery.ensure_proto5_point_cache({"hand": "proto5"})

    assert os.environ["PYTORCH_VOLUMETRIC_POINTS_CACHE_DIR"] == str(cache_dir)


def test_proto5_point_cache_setup_does_not_accept_broader_link_caches(monkeypatch, tmp_path):
    ccai_root = tmp_path / "ccai"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "RHand_I2Y_LINK_points_cache.pkl").write_bytes(b"broad-cache")
    monkeypatch.setattr(screwdriver_isaacsim_recovery, "CCAI_PATH", ccai_root)
    monkeypatch.setattr(screwdriver_isaacsim_recovery, "PROTO5_POINT_CACHE_SOURCE_DIRS", (source_dir,))

    with pytest.raises(FileNotFoundError, match="RHand_I6AF_LINK_points_cache.pkl"):
        screwdriver_isaacsim_recovery.ensure_proto5_point_cache({"hand": "proto5"})


def test_pregrasp_reference_targets_builds_target_kwargs_from_reference_problem():
    captured = {}

    class Problem:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.contact_points_object = torch.tensor([[1.0, 2.0, 3.0]])
            self.contact_points_rob_link = torch.tensor([[4.0, 5.0, 6.0]])

    class Env:
        default_dof_pos = torch.arange(18, dtype=torch.float32).reshape(1, 18)
        table_pose = torch.tensor([0.0, 0.0, 1.205])
        obj_pose = table_pose
        world_trans = object()

    params = {
        "T": 3,
        "chain": object(),
        "object_location": torch.tensor([0.0, 0.0, 1.205]),
        "object_type": "screwdriver",
        "optimize_force": True,
        "fingers": ["index"],
        "pregrasp_target_contact_patch_cost_weight": 11.0,
        "pregrasp_target_contact_link_cost_weight": 12.0,
        "pregrasp_target_contact_patch_mode": "constraint",
    }

    kwargs = build_pregrasp_reference_target_kwargs(params, Env(), "cpu", Problem)

    assert captured["T"] == 1
    assert captured["regrasp_fingers"] == ["index"]
    assert captured["full_dof_goal"] is True
    assert captured["fingertip_contact_only"] is True
    torch.testing.assert_close(kwargs["target_contact_points_object"], torch.tensor([[1.0, 2.0, 3.0]]))
    torch.testing.assert_close(kwargs["target_contact_points_rob_link"], torch.tensor([[4.0, 5.0, 6.0]]))
    assert kwargs["use_default_ee_locs_cost"] is False
    assert kwargs["target_contact_patch_cost_weight"] == pytest.approx(11.0)
    assert kwargs["target_contact_link_cost_weight"] == pytest.approx(12.0)
    assert kwargs["target_contact_patch_mode"] == "constraint"
