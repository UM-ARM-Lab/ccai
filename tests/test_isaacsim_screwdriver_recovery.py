import importlib.util
import os
import pathlib
import sys
import types

import numpy as np
import pytest
import torch

from ccai.utils.isaacsim_screwdriver_recovery import (
    ALLEGRO_ACTIVE_JOINT_NAMES,
    ALLEGRO_DEFAULT_FULL_JOINT_POS,
    ALLEGRO_RING_JOINT_NAMES,
    OBJ_ORIENTATION_JOINT_NAMES,
    PROTO5_ACTIVE_JOINT_NAMES,
    PROTO5_ALL_JOINT_NAMES,
    IsaacSimScrewdriverRecoveryEnv,
    active12_to_env_action,
    local_force_at_position_to_world,
    pack_ccai_state,
    sample_screwdriver_body_poke,
)
from ccai.utils.recovery_utils import build_pregrasp_reference_target_kwargs, create_allegro_screwdriver_problem


_ENTRYPOINT_PATH = pathlib.Path(__file__).resolve().parents[1] / "examples" / "screwdriver_isaacsim_recovery.py"
_ENTRYPOINT_SPEC = importlib.util.spec_from_file_location("screwdriver_isaacsim_recovery_entrypoint", _ENTRYPOINT_PATH)
screwdriver_isaacsim_recovery = importlib.util.module_from_spec(_ENTRYPOINT_SPEC)
_ENTRYPOINT_SPEC.loader.exec_module(screwdriver_isaacsim_recovery)


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
    pass


class _FakeUnwrapped:
    def __init__(self, robot, obj):
        self.device = torch.device("cpu")
        self.num_envs = 1
        self.scene = _FakeScene(robot=robot, obj=obj)


class _FakeActionSpace:
    def __init__(self, shape):
        self.shape = shape


class _FakeEnv:
    def __init__(self, robot, obj, action_shape=(16,)):
        self.unwrapped = _FakeUnwrapped(robot, obj)
        self.action_space = _FakeActionSpace(action_shape)
        self.last_action = None

    def step(self, action):
        self.last_action = action
        return action

    def reset(self):
        return None


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
            "planner_use_env_yaw_joint_friction": True,
            "disable_planner_yaw_friction_model": True,
            "planner_use_yaw_inertia_model": False,
        },
    )

    assert kwargs["friction_coefficient"] == pytest.approx(2.0)
    assert kwargs["yaw_joint_friction"] == pytest.approx(0.21)
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


def _load_legacy_recovery_module(monkeypatch, executed_modes):
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
    stubs["ccai.utils.recovery_utils"].create_allegro_screwdriver_problem = (
        lambda *args, **kwargs: types.SimpleNamespace(dx=15)
    )
    stubs["ccai.utils.recovery_utils"].create_planner = lambda problem, mode, params: FakePlanner(problem)
    stubs["ccai.utils.recovery_utils"].add_to_dataset = lambda *args, **kwargs: None
    stubs["ccai.utils.recovery_utils"].partial_to_full_trajectory = lambda *args, **kwargs: None
    stubs["ccai.utils.recovery_utils"].full_to_partial_trajectory = lambda *args, **kwargs: None
    stubs["ccai.utils.recovery_utils"].create_mode_planner_dict = lambda *args, **kwargs: {}
    stubs["ccai.utils.recovery_utils"].build_pregrasp_reference_target_kwargs = lambda *args, **kwargs: {}
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
