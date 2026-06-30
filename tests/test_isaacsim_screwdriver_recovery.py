import torch

from ccai.utils.isaacsim_screwdriver_recovery import (
    ALLEGRO_ACTIVE_JOINT_NAMES,
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
from ccai.utils.recovery_utils import create_allegro_screwdriver_problem


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
    def __init__(self, joint_pos):
        self.joint_pos = joint_pos


class _FakeAsset:
    def __init__(self, joint_names, joint_pos):
        self._joint_name_to_id = {name: idx for idx, name in enumerate(joint_names)}
        self.data = _FakeData(joint_pos)

    def find_joints(self, joint_names, preserve_order=True):
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
