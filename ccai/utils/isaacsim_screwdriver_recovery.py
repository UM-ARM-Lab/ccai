"""Isaac Sim compatibility layer for CCAI screwdriver recovery.

The legacy recovery code expects an Isaac Gym-style executor surface.  This
module adapts the IsaacLab screwdriver environments to that surface without
pulling IsaacLab into import-time unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


SCREWDRIVER_HAND_ALLEGRO = "allegro"
SCREWDRIVER_HAND_PROTO5 = "proto5"
SCREWDRIVER_HAND_CHOICES = (SCREWDRIVER_HAND_ALLEGRO, SCREWDRIVER_HAND_PROTO5)

CCAI_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_ROOT = CCAI_ROOT.parent
ISAACSIM_HAND_ENVS_PATH = DOCUMENTS_ROOT / "github" / "isaacsim-hand-envs"

ALLEGRO_ACTIVE_JOINT_NAMES = (
    "allegro_hand_hitosashi_finger_finger_joint_0",
    "allegro_hand_hitosashi_finger_finger_joint_1",
    "allegro_hand_hitosashi_finger_finger_joint_2",
    "allegro_hand_hitosashi_finger_finger_joint_3",
    "allegro_hand_naka_finger_finger_joint_4",
    "allegro_hand_naka_finger_finger_joint_5",
    "allegro_hand_naka_finger_finger_joint_6",
    "allegro_hand_naka_finger_finger_joint_7",
    "allegro_hand_oya_finger_joint_12",
    "allegro_hand_oya_finger_joint_13",
    "allegro_hand_oya_finger_joint_14",
    "allegro_hand_oya_finger_joint_15",
)
ALLEGRO_RING_JOINT_NAMES = (
    "allegro_hand_kusuri_finger_finger_joint_8",
    "allegro_hand_kusuri_finger_finger_joint_9",
    "allegro_hand_kusuri_finger_finger_joint_10",
    "allegro_hand_kusuri_finger_finger_joint_11",
)

PROTO5_WRIST_JOINT_NAMES = (
    "RHand_WRZ_joint",
    "RHand_WRY_joint",
)
PROTO5_INDEX_JOINT_NAMES = (
    "RHand_I1Z_joint",
    "RHand_I1Y_joint",
    "RHand_I2Y_joint",
    "RHand_I3Y_joint",
)
PROTO5_MIDDLE_JOINT_NAMES = (
    "RHand_M1Z_joint",
    "RHand_M1Y_joint",
    "RHand_M2Y_joint",
    "RHand_M3Y_joint",
)
PROTO5_RING_JOINT_NAMES = (
    "RHand_R1Z_joint",
    "RHand_R1Y_joint",
    "RHand_R2Y_joint",
    "RHand_R3Y_joint",
)
PROTO5_THUMB_JOINT_NAMES = (
    "RHand_T1Z_joint",
    "RHand_T1Y_joint",
    "RHand_T2Y_joint",
    "RHand_T3Y_joint",
)
PROTO5_ACTIVE_JOINT_NAMES = PROTO5_INDEX_JOINT_NAMES + PROTO5_MIDDLE_JOINT_NAMES + PROTO5_THUMB_JOINT_NAMES
PROTO5_FROZEN_JOINT_NAMES = PROTO5_WRIST_JOINT_NAMES + PROTO5_RING_JOINT_NAMES
PROTO5_ALL_JOINT_NAMES = (
    PROTO5_WRIST_JOINT_NAMES
    + PROTO5_INDEX_JOINT_NAMES
    + PROTO5_MIDDLE_JOINT_NAMES
    + PROTO5_RING_JOINT_NAMES
    + PROTO5_THUMB_JOINT_NAMES
)

OBJ_ORIENTATION_JOINT_NAMES = (
    "table_screwdriver_joint_1",
    "table_screwdriver_joint_2",
    "table_screwdriver_joint_3",
)
SCREWDRIVER_BODY_NAME = "screwdriver_body"

ALLEGRO_DEFAULT_FULL_JOINT_POS = (
    0.1,
    0.6,
    0.6,
    0.6,
    -0.1,
    0.5,
    0.9,
    0.9,
    0.0,
    0.0,
    0.0,
    0.0,
    1.2,
    0.3,
    0.3,
    1.2,
)
PROTO5_DEFAULT_FULL_JOINT_POS = (
    0.271,
    -0.005,
    0.132,
    0.743,
    0.202,
    0.486,
    -0.33,
    1.03,
    0.371,
    0.203,
    -0.349,
    0.0,
    0.0,
    0.002,
    -0.156,
    0.165,
    0.86,
    0.439,
)

ALLEGRO_ROBOT_ROOT_POS = (0.0, -0.095, 1.33)
ALLEGRO_ROBOT_ROOT_ROT_WXYZ = (0.664463, 0.2418448, 0.2418448, 0.664463)
PROTO5_ROBOT_ROOT_POS = (-0.36898, -0.15366, 1.07923)
PROTO5_ROBOT_ROOT_ROT_WXYZ = (0.71151, 0.62842, -0.20813, 0.23565)
DEFAULT_SCREWDRIVER_TABLE_POSE = (0.0, 0.0, 1.205)


@dataclass(frozen=True)
class ScrewdriverHandSpec:
    name: str
    gym_id: str
    active_joint_names: tuple[str, ...]
    ring_joint_names: tuple[str, ...]
    wrist_joint_names: tuple[str, ...]
    frozen_joint_names: tuple[str, ...]
    all_joint_names: tuple[str, ...]
    default_full_joint_pos: tuple[float, ...]
    robot_root_pos: tuple[float, float, float]
    robot_root_rot_wxyz: tuple[float, float, float, float]
    urdf_path: Path
    planner_robot_sdf_path_prefix: Path


HAND_SPECS = {
    SCREWDRIVER_HAND_ALLEGRO: ScrewdriverHandSpec(
        name=SCREWDRIVER_HAND_ALLEGRO,
        gym_id="AllegroScrewdriverTurning-v0",
        active_joint_names=ALLEGRO_ACTIVE_JOINT_NAMES,
        ring_joint_names=ALLEGRO_RING_JOINT_NAMES,
        wrist_joint_names=(),
        frozen_joint_names=ALLEGRO_RING_JOINT_NAMES,
        all_joint_names=ALLEGRO_ACTIVE_JOINT_NAMES[:8] + ALLEGRO_RING_JOINT_NAMES + ALLEGRO_ACTIVE_JOINT_NAMES[8:],
        default_full_joint_pos=ALLEGRO_DEFAULT_FULL_JOINT_POS,
        robot_root_pos=ALLEGRO_ROBOT_ROOT_POS,
        robot_root_rot_wxyz=ALLEGRO_ROBOT_ROOT_ROT_WXYZ,
        urdf_path=CCAI_ROOT / "allegro_hand_right.urdf",
        planner_robot_sdf_path_prefix=Path.home()
        / "Documents"
        / "github"
        / "isaacgym-arm-envs"
        / "isaac_victor_envs"
        / "assets"
        / "xela_models",
    ),
    SCREWDRIVER_HAND_PROTO5: ScrewdriverHandSpec(
        name=SCREWDRIVER_HAND_PROTO5,
        gym_id="Proto5ScrewdriverTurning-v0",
        active_joint_names=PROTO5_ACTIVE_JOINT_NAMES,
        ring_joint_names=PROTO5_RING_JOINT_NAMES,
        wrist_joint_names=PROTO5_WRIST_JOINT_NAMES,
        frozen_joint_names=PROTO5_FROZEN_JOINT_NAMES,
        all_joint_names=PROTO5_ALL_JOINT_NAMES,
        default_full_joint_pos=PROTO5_DEFAULT_FULL_JOINT_POS,
        robot_root_pos=PROTO5_ROBOT_ROOT_POS,
        robot_root_rot_wxyz=PROTO5_ROBOT_ROOT_ROT_WXYZ,
        urdf_path=ISAACSIM_HAND_ENVS_PATH
        / "isaacsim_hand_envs"
        / "assets"
        / "urdf"
        / "proto5"
        / "hmf_hand_proto5_release_right_csvto.urdf",
        planner_robot_sdf_path_prefix=ISAACSIM_HAND_ENVS_PATH / "isaacsim_hand_envs" / "assets" / "urdf" / "proto5",
    ),
}


def get_hand_spec(hand: str) -> ScrewdriverHandSpec:
    hand = str(hand).lower()
    if hand not in HAND_SPECS:
        raise ValueError(f"Unsupported screwdriver hand {hand!r}; expected one of {SCREWDRIVER_HAND_CHOICES}.")
    return HAND_SPECS[hand]


def create_world_transform(hand: str, device: str | torch.device):
    from pytorch_kinematics import transforms as tf

    spec = get_hand_spec(hand)
    return tf.Transform3d(
        pos=torch.tensor(spec.robot_root_pos, device=device, dtype=torch.float32),
        rot=torch.tensor(spec.robot_root_rot_wxyz, device=device, dtype=torch.float32),
        device=device,
    )


def _as_2d_tensor(value, *, device: str | torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _rand(shape, *, device, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.rand(shape, device=device, generator=generator)


def sample_screwdriver_body_poke(
    batch_size: int = 1,
    random_force_magnitude: float = 1.5,
    *,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the legacy screwdriver-body local force and local application point."""
    pos = _rand((batch_size, 2), device=device, generator=generator)
    angle = torch.pi * (2.0 * pos[:, 0] - 1.0)
    local_point = torch.stack(
        (
            torch.sin(angle) * 0.02,
            torch.cos(angle) * 0.02,
            pos[:, 1] * 0.1,
        ),
        dim=-1,
    )

    force_dir = -torch.stack(
        (
            local_point[:, 0],
            local_point[:, 1],
            torch.zeros(batch_size, device=device, dtype=local_point.dtype),
        ),
        dim=-1,
    )
    force_dir = force_dir / torch.linalg.norm(force_dir, dim=-1, keepdim=True).clamp_min(1.0e-8)

    angle_z = (_rand((batch_size,), device=device, generator=generator) - 0.5) * 0.2
    cos_angle = torch.cos(angle_z)
    sin_angle = torch.sin(angle_z)
    rotated_force_dir = torch.stack(
        (
            force_dir[:, 0] * cos_angle - force_dir[:, 1] * sin_angle,
            force_dir[:, 0] * sin_angle + force_dir[:, 1] * cos_angle,
            force_dir[:, 2],
        ),
        dim=-1,
    )

    magnitude = (0.5 + 0.5 * _rand((batch_size,), device=device, generator=generator)) * float(random_force_magnitude)
    return local_point, rotated_force_dir * magnitude.unsqueeze(-1)


def quat_wxyz_to_matrix(quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat_wxyz = torch.as_tensor(quat_wxyz)
    quat_wxyz = quat_wxyz / torch.linalg.norm(quat_wxyz, dim=-1, keepdim=True).clamp_min(1.0e-8)
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    two = 2.0
    return torch.stack(
        (
            1 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quat_wxyz.shape[:-1] + (3, 3))


def rotate_vectors_by_quat(vectors: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    rot = quat_wxyz_to_matrix(quat_wxyz.to(device=vectors.device, dtype=vectors.dtype))
    return torch.matmul(rot, vectors.unsqueeze(-1)).squeeze(-1)


def local_force_at_position_to_world(
    local_point: torch.Tensor,
    local_force: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a body-local force-at-point to world force and torque about the body origin."""
    local_force_ref = torch.as_tensor(local_force)
    dtype = local_force_ref.dtype if torch.is_floating_point(local_force_ref) else torch.float32
    device = local_force_ref.device
    local_point = torch.as_tensor(local_point, device=device, dtype=dtype)
    local_force = local_force_ref.to(device=device, dtype=dtype)
    body_quat_w = torch.as_tensor(body_quat_w, device=device, dtype=dtype)
    body_pos_w = torch.as_tensor(body_pos_w, device=device, dtype=dtype)
    if local_point.ndim == 1:
        local_point = local_point.unsqueeze(0)
    if local_force.ndim == 1:
        local_force = local_force.unsqueeze(0)
    if body_quat_w.ndim == 1:
        body_quat_w = body_quat_w.unsqueeze(0)
    if body_pos_w.ndim == 1:
        body_pos_w = body_pos_w.unsqueeze(0)

    point_offset_w = rotate_vectors_by_quat(local_point, body_quat_w)
    force_w = rotate_vectors_by_quat(local_force, body_quat_w)
    torque_w = torch.cross(point_offset_w, force_w, dim=-1)
    return force_w, torque_w


def pack_ccai_state(active_joint_pos: torch.Tensor, obj_orientation: torch.Tensor, cap_joint: torch.Tensor | None = None):
    active_joint_pos = torch.as_tensor(active_joint_pos)
    obj_orientation = torch.as_tensor(obj_orientation, device=active_joint_pos.device, dtype=active_joint_pos.dtype)
    if active_joint_pos.ndim == 1:
        active_joint_pos = active_joint_pos.unsqueeze(0)
    if obj_orientation.ndim == 1:
        obj_orientation = obj_orientation.unsqueeze(0)
    if active_joint_pos.shape[-1] != 12:
        raise ValueError(f"Expected 12 active joints, got shape {tuple(active_joint_pos.shape)}.")
    if obj_orientation.shape[-1] != 3:
        raise ValueError(f"Expected 3 object orientation joints, got shape {tuple(obj_orientation.shape)}.")
    if cap_joint is None:
        cap_joint = obj_orientation[..., 2:3]
    else:
        cap_joint = torch.as_tensor(cap_joint, device=active_joint_pos.device, dtype=active_joint_pos.dtype)
        if cap_joint.ndim == 1:
            cap_joint = cap_joint.unsqueeze(-1)
    return torch.cat((active_joint_pos, obj_orientation, cap_joint), dim=-1)


def active12_to_env_action(
    active_joint_targets: torch.Tensor,
    *,
    hand: str,
    default_dof_pos: torch.Tensor | Sequence[float] | None = None,
    proto5_control_wrist: bool = False,
    current_wrist_joints: torch.Tensor | None = None,
    allegro_action_dim: int | None = 16,
) -> torch.Tensor:
    """Map 12D absolute CCAI active-finger targets to the selected IsaacLab action layout."""
    spec = get_hand_spec(hand)
    active_joint_targets = torch.as_tensor(active_joint_targets, dtype=torch.float32)
    if active_joint_targets.ndim == 1:
        active_joint_targets = active_joint_targets.unsqueeze(0)
    if active_joint_targets.shape[-1] != 12:
        raise ValueError(f"Expected active_joint_targets trailing dimension 12, got {active_joint_targets.shape[-1]}.")

    if spec.name == SCREWDRIVER_HAND_PROTO5:
        if not proto5_control_wrist:
            return active_joint_targets
        if current_wrist_joints is None:
            raise ValueError("current_wrist_joints is required when proto5_control_wrist=True.")
        current_wrist_joints = torch.as_tensor(
            current_wrist_joints,
            device=active_joint_targets.device,
            dtype=active_joint_targets.dtype,
        )
        if current_wrist_joints.ndim == 1:
            current_wrist_joints = current_wrist_joints.unsqueeze(0)
        if current_wrist_joints.shape[-1] != 2:
            raise ValueError(f"Expected current_wrist_joints trailing dimension 2, got {current_wrist_joints.shape[-1]}.")
        return torch.cat((active_joint_targets, current_wrist_joints), dim=-1)

    if default_dof_pos is None:
        ring_targets = torch.zeros(
            active_joint_targets.shape[:-1] + (4,),
            device=active_joint_targets.device,
            dtype=active_joint_targets.dtype,
        )
    else:
        default_dof_pos = torch.as_tensor(
            default_dof_pos,
            device=active_joint_targets.device,
            dtype=active_joint_targets.dtype,
        ).reshape(-1)
        if default_dof_pos.numel() < 12:
            raise ValueError(f"Expected at least 12 default DOF values for Allegro, got {default_dof_pos.numel()}.")
        ring_targets = default_dof_pos[8:12].expand(active_joint_targets.shape[:-1] + (4,))
    if allegro_action_dim is None:
        allegro_action_dim = 16
    if allegro_action_dim == 12:
        return active_joint_targets
    if allegro_action_dim != 16:
        raise ValueError(f"Unsupported Allegro screwdriver action dimension {allegro_action_dim}; expected 12 or 16.")
    return torch.cat((active_joint_targets, ring_targets), dim=-1)


def _infer_action_dim(env) -> int | None:
    action_space = getattr(env, "single_action_space", None) or getattr(env, "action_space", None)
    shape = getattr(action_space, "shape", None)
    if not shape:
        return None
    return int(np.prod(shape))


class IsaacSimScrewdriverRecoveryEnv:
    """Expose the CCAI recovery executor contract on top of an IsaacLab env."""

    def __init__(
        self,
        env,
        *,
        hand: str = SCREWDRIVER_HAND_ALLEGRO,
        proto5_control_wrist: bool = False,
        external_wrench_perturb: bool = False,
        rand_pct: float | None = None,
        random_force_magnitude: float = 1.5,
    ):
        self.env = env
        self._unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        self.hand = get_hand_spec(hand).name
        self.hand_spec = get_hand_spec(self.hand)
        self.proto5_control_wrist = bool(proto5_control_wrist)
        self.action_dim = _infer_action_dim(env)
        self.device = torch.device(getattr(self._unwrapped, "device", "cpu"))
        self.num_envs = int(getattr(self._unwrapped, "num_envs", 1))
        self.default_dof_pos = torch.tensor(
            self.hand_spec.default_full_joint_pos,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, -1)
        self.table_pose = torch.tensor(DEFAULT_SCREWDRIVER_TABLE_POSE, device=self.device, dtype=torch.float32)
        self.obj_pose = self.table_pose
        self.world_trans = create_world_transform(self.hand, self.device)
        self.external_wrench_perturb = bool(external_wrench_perturb)
        self.external_wrench_perturb_rand_pct = 1.0 / 3.0 if rand_pct is None else float(rand_pct)
        self.random_force_magnitude = float(random_force_magnitude)
        self.wrench_perturb_inds = []
        self._step_index = 0
        self._frame_id = 0
        self.frame_id = 0
        self.frame_fpath = None
        setattr(self._unwrapped, "_ccai_recovery_hand", self.hand)
        setattr(self._unwrapped, "_ccai_recovery_proto5_control_wrist", self.proto5_control_wrist)
        self.refresh_default_dof_pos()

    @property
    def frame_id(self):
        return self._frame_id

    @frame_id.setter
    def frame_id(self, value):
        self._frame_id = value
        if value is None or value == 0:
            self._step_index = 0

    @property
    def unwrapped(self):
        return self._unwrapped

    @property
    def scene(self):
        return self._unwrapped.scene

    def reset(self):
        self.wrench_perturb_inds = []
        self.frame_id = 0
        ret = self.env.reset()
        self.refresh_default_dof_pos()
        return ret

    def refresh_default_dof_pos(self) -> torch.Tensor:
        """Refresh ordered full-hand defaults from the wrapped robot when available."""
        try:
            robot = self.scene["robot"]
            source = getattr(robot.data, "default_joint_pos", None)
            if source is None:
                source = getattr(robot.data, "joint_pos", None)
            if source is None:
                return self.default_dof_pos
            joint_pos = _as_2d_tensor(source, device=self.device, dtype=torch.float32)
            self.default_dof_pos = joint_pos[:, self._all_joint_ids()].clone()
        except (AttributeError, KeyError, IndexError, RuntimeError, ValueError):
            pass
        return self.default_dof_pos

    def get_environment_parameters(self, env_id: int = 0) -> dict[str, float]:
        """Read physical parameters currently sampled in the IsaacLab env."""
        env_id = int(env_id)
        if self.hand == SCREWDRIVER_HAND_PROTO5:
            screwdriver_friction = 1.0
            yaw_joint_friction = 0.0
            if hasattr(self._unwrapped, "_proto5_contact_friction_tensor"):
                screwdriver_friction = float(self._unwrapped._proto5_contact_friction_tensor[env_id].item())
            if hasattr(self._unwrapped, "_screwdriver_joint_friction_tensor"):
                yaw_joint_friction = float(self._unwrapped._screwdriver_joint_friction_tensor[env_id].item())
            return {
                "screwdriver_friction": screwdriver_friction,
                "yaw_joint_friction": yaw_joint_friction,
            }

        screwdriver_friction = 1.0
        if hasattr(self._unwrapped, "_screwdriver_friction_values"):
            friction_info = self._unwrapped._screwdriver_friction_values.get(env_id, {})
            screwdriver_friction = float(friction_info.get("static_friction", 1.0))

        yaw_joint_friction = 0.0
        try:
            obj = self.scene["obj"]
            yaw_joint_idx = obj.find_joints("table_screwdriver_joint_3")[0][0]
            joint_frictions = obj.root_physx_view.get_dof_friction_properties()
            yaw_joint_friction = float(joint_frictions[env_id, yaw_joint_idx, 0].item())
        except (KeyError, IndexError, AttributeError):
            if hasattr(self._unwrapped, "_yaw_joint_friction_values"):
                friction_info = self._unwrapped._yaw_joint_friction_values.get(env_id, {})
                yaw_joint_friction = float(friction_info.get("friction", 0.0))

        return {
            "screwdriver_friction": screwdriver_friction,
            "yaw_joint_friction": yaw_joint_friction,
        }

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
        return None

    def _env_ids(self) -> torch.Tensor:
        return torch.arange(self.num_envs, device=self.device, dtype=torch.long)

    @staticmethod
    def _find_joints(asset, joint_names: Sequence[str]) -> list[int]:
        joint_ids, _ = asset.find_joints(tuple(joint_names), preserve_order=True)
        return list(joint_ids)

    def _active_joint_ids(self) -> list[int]:
        robot = self.scene["robot"]
        return self._find_joints(robot, self.hand_spec.active_joint_names)

    def _ring_joint_ids(self) -> list[int]:
        if not self.hand_spec.ring_joint_names:
            return []
        return self._find_joints(self.scene["robot"], self.hand_spec.ring_joint_names)

    def _wrist_joint_ids(self) -> list[int]:
        if not self.hand_spec.wrist_joint_names:
            return []
        return self._find_joints(self.scene["robot"], self.hand_spec.wrist_joint_names)

    def _all_joint_ids(self) -> list[int]:
        return self._find_joints(self.scene["robot"], self.hand_spec.all_joint_names)

    def _obj_orientation_joint_ids(self) -> list[int]:
        obj = self.scene["obj"]
        try:
            return self._find_joints(obj, OBJ_ORIENTATION_JOINT_NAMES)
        except Exception:
            return [0, 1, 2]

    def get_state(self):
        robot = self.scene["robot"]
        obj = self.scene["obj"]
        active_joint_pos = robot.data.joint_pos[:, self._active_joint_ids()].to(dtype=torch.float32)
        obj_orientation = obj.data.joint_pos[:, self._obj_orientation_joint_ids()].to(dtype=torch.float32)
        q = pack_ccai_state(active_joint_pos, obj_orientation)
        return {
            "q": q,
            "screwdriver_ori_euler": obj_orientation,
            "screwdriver_ori": obj_orientation,
            "screwdriver_angle": obj_orientation[:, 2:3],
        }

    def _write_robot_root_default(self, env_ids: torch.Tensor) -> None:
        robot = self.scene["robot"]
        root_state = robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] += self.scene.env_origins[env_ids]
        robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def _write_obj_root_default(self, env_ids: torch.Tensor) -> None:
        obj = self.scene["obj"]
        root_state = obj.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] += self.scene.env_origins[env_ids]
        obj.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        obj.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def _set_proto5_frozen_targets(self, full_joint_pos: torch.Tensor, env_ids: torch.Tensor) -> None:
        if self.hand != SCREWDRIVER_HAND_PROTO5:
            return
        frozen_ids = self._wrist_joint_ids() + self._ring_joint_ids()
        if (
            not hasattr(self._unwrapped, "_proto5_frozen_joint_targets")
            or self._unwrapped._proto5_frozen_joint_targets.shape[-1] != len(frozen_ids)
        ):
            self._unwrapped._proto5_frozen_joint_targets = torch.zeros(
                (self.num_envs, len(frozen_ids)),
                device=self.device,
                dtype=full_joint_pos.dtype,
            )
        self._unwrapped._proto5_frozen_joint_targets[env_ids] = full_joint_pos[:, frozen_ids]

    def _write_robot_joint_state(self, active_joint_pos: torch.Tensor, env_ids: torch.Tensor) -> None:
        robot = self.scene["robot"]
        joint_pos = robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(robot.data.default_joint_vel[env_ids])
        joint_pos[:, self._active_joint_ids()] = active_joint_pos
        self._set_proto5_frozen_targets(joint_pos, env_ids)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        robot.set_joint_position_target(joint_pos, env_ids=env_ids)

    def _write_obj_joint_state(self, obj_orientation: torch.Tensor, env_ids: torch.Tensor) -> None:
        obj = self.scene["obj"]
        joint_pos = obj.data.joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(obj.data.joint_vel[env_ids])
        joint_pos[:, self._obj_orientation_joint_ids()] = obj_orientation
        obj.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _sync_scene(self) -> None:
        self.scene.write_data_to_sim()
        if hasattr(self._unwrapped.sim, "forward"):
            self._unwrapped.sim.forward()
        dt = self._unwrapped.sim.get_physics_dt()
        self.scene.update(dt)

    def set_pose(self, state):
        state = _as_2d_tensor(state, device=self.device, dtype=torch.float32)
        if state.shape[-1] < 15:
            raise ValueError(f"Expected state with at least 15 values, got shape {tuple(state.shape)}.")
        if state.shape[0] == 1 and self.num_envs > 1:
            state = state.repeat(self.num_envs, 1)
        env_ids = self._env_ids()
        self._write_robot_root_default(env_ids)
        self._write_obj_root_default(env_ids)
        self._write_robot_joint_state(state[:, :12], env_ids)
        self._write_obj_joint_state(state[:, 12:15], env_ids)
        self._sync_scene()

    def zero_obj_velocity(self):
        obj = self.scene["obj"]
        env_ids = self._env_ids()
        zero_root_vel = torch.zeros((len(env_ids), 6), device=self.device, dtype=obj.data.default_root_state.dtype)
        obj.write_root_com_velocity_to_sim(zero_root_vel, env_ids=env_ids)
        joint_pos = obj.data.joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(obj.data.joint_vel[env_ids])
        obj.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._sync_scene()

    def set_external_wrench_perturb(self, enabled, rand_pct=None):
        self.external_wrench_perturb = bool(enabled)
        if rand_pct is not None:
            self.external_wrench_perturb_rand_pct = float(rand_pct)

    def _body_pose(self, obj, body_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(obj.data, "body_pose_w"):
            body_pose = obj.data.body_pose_w[:, body_idx, :7]
            return body_pose[:, :3], body_pose[:, 3:7]
        return obj.data.root_link_pos_w, obj.data.root_link_quat_w

    def _set_external_force_torque(self, forces: torch.Tensor, torques: torch.Tensor) -> None:
        obj = self.scene["obj"]
        body_ids, _ = obj.find_bodies(SCREWDRIVER_BODY_NAME)
        obj.set_external_force_and_torque(
            forces.reshape(self.num_envs, 1, 3),
            torques.reshape(self.num_envs, 1, 3),
            env_ids=self._env_ids(),
            body_ids=body_ids,
        )

    def _clear_external_force_torque(self) -> None:
        zeros = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._set_external_force_torque(zeros, zeros)

    def _maybe_apply_external_perturbation(self) -> bool:
        if not self.external_wrench_perturb:
            self._clear_external_force_torque()
            return False
        if np.random.rand() >= self.external_wrench_perturb_rand_pct:
            self._clear_external_force_torque()
            return False

        obj = self.scene["obj"]
        body_ids, _ = obj.find_bodies(SCREWDRIVER_BODY_NAME)
        body_idx = int(body_ids[0])
        body_pos_w, body_quat_w = self._body_pose(obj, body_idx)
        local_point, local_force = sample_screwdriver_body_poke(
            self.num_envs,
            self.random_force_magnitude,
            device=self.device,
        )
        force_w, torque_w = local_force_at_position_to_world(local_point, local_force, body_pos_w, body_quat_w)
        self._set_external_force_torque(force_w, torque_w)
        frame_index = self._step_index if self.frame_id is None else self.frame_id
        self.wrench_perturb_inds.append(frame_index)
        return True

    def _current_wrist_joints(self) -> torch.Tensor | None:
        wrist_ids = self._wrist_joint_ids()
        if not wrist_ids:
            return None
        return self.scene["robot"].data.joint_pos[:, wrist_ids].to(dtype=torch.float32)

    def step(self, action):
        active_targets = _as_2d_tensor(action, device=self.device, dtype=torch.float32)
        if active_targets.shape[-1] < 12:
            raise ValueError(f"Expected action with at least 12 values, got shape {tuple(active_targets.shape)}.")
        active_targets = active_targets[:, :12]
        env_action = active12_to_env_action(
            active_targets,
            hand=self.hand,
            default_dof_pos=self.default_dof_pos[0],
            proto5_control_wrist=self.proto5_control_wrist,
            current_wrist_joints=self._current_wrist_joints() if self.proto5_control_wrist else None,
            allegro_action_dim=self.action_dim if self.hand == SCREWDRIVER_HAND_ALLEGRO else None,
        ).to(device=self.device)
        self._maybe_apply_external_perturbation()
        ret = self.env.step(env_action)
        self._clear_external_force_torque()
        if self.frame_id is not None:
            self.frame_id += 1
        self._step_index += 1
        return ret

    def get_force_sensor_data(self, *args, **kwargs):
        return torch.zeros((self.num_envs, 9), device=self.device, dtype=torch.float32)
