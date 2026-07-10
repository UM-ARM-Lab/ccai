"""Isaac Sim compatibility layer for CCAI screwdriver recovery.

The legacy recovery code expects an Isaac Gym-style executor surface.  This
module adapts the IsaacLab screwdriver environments to that surface without
pulling IsaacLab into import-time unit tests.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence, TextIO

import numpy as np
import torch

from ccai.utils.project_paths import find_model_mismatch_root, resolve_isaacsim_hand_envs_path


SCREWDRIVER_HAND_ALLEGRO = "allegro"
SCREWDRIVER_HAND_PROTO5 = "proto5"
SCREWDRIVER_HAND_CHOICES = (SCREWDRIVER_HAND_ALLEGRO, SCREWDRIVER_HAND_PROTO5)

CCAI_ROOT = Path(__file__).resolve().parents[2]
MODEL_MISMATCH_PATH = find_model_mismatch_root(CCAI_ROOT)
DOCUMENTS_ROOT = MODEL_MISMATCH_PATH.parent
ISAACSIM_HAND_ENVS_PATH = resolve_isaacsim_hand_envs_path(MODEL_MISMATCH_PATH)
PROTO5_DEFAULTS_PATH = ISAACSIM_HAND_ENVS_PATH / "isaacsim_hand_envs" / "assets" / "robot" / "proto5_defaults.py"
SCREWDRIVER_POSITION_DEFAULTS_PATH = (
    MODEL_MISMATCH_PATH / "model_mismatch" / "utils" / "screwdriver_position_defaults.py"
)


class _StdoutTee:
    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return bool(self._streams and self._streams[0].isatty())

    def __getattr__(self, name: str):
        return getattr(self._streams[0], name)

    @property
    def encoding(self) -> str | None:
        return self._streams[0].encoding if self._streams else None

    @property
    def errors(self) -> str | None:
        return self._streams[0].errors if self._streams else None


@contextlib.contextmanager
def tee_stdout_to_file(log_path: str | Path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
        tee_stdout = _StdoutTee(original_stdout, log_file)
        sys.stdout = tee_stdout
        try:
            yield log_path
        finally:
            sys.stdout = original_stdout
            tee_stdout.flush()


def _load_proto5_defaults():
    spec = importlib.util.spec_from_file_location("_isaacsim_hand_envs_proto5_defaults", PROTO5_DEFAULTS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Proto5 defaults from {PROTO5_DEFAULTS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PROTO5_DEFAULTS = _load_proto5_defaults()


def _load_default_screwdriver_position_robot() -> tuple[float, float, float]:
    spec = importlib.util.spec_from_file_location(
        "_model_mismatch_screwdriver_position_defaults",
        SCREWDRIVER_POSITION_DEFAULTS_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            "Could not load screwdriver position defaults from "
            f"{SCREWDRIVER_POSITION_DEFAULTS_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(float(value) for value in module.DEFAULT_SCREWDRIVER_POSITION_ROBOT)


DEFAULT_SCREWDRIVER_POSITION_ROBOT = _load_default_screwdriver_position_robot()

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

PROTO5_WRIST_JOINT_NAMES = tuple(_PROTO5_DEFAULTS.WRIST_JOINT_NAMES)
PROTO5_INDEX_JOINT_NAMES = tuple(_PROTO5_DEFAULTS.INDEX_JOINT_NAMES)
PROTO5_MIDDLE_JOINT_NAMES = tuple(_PROTO5_DEFAULTS.MIDDLE_JOINT_NAMES)
PROTO5_RING_JOINT_NAMES = tuple(_PROTO5_DEFAULTS.RING_JOINT_NAMES)
PROTO5_THUMB_JOINT_NAMES = tuple(_PROTO5_DEFAULTS.THUMB_JOINT_NAMES)
PROTO5_ACTIVE_JOINT_NAMES = tuple(_PROTO5_DEFAULTS.ACTIVE_FINGER_JOINT_NAMES)
PROTO5_FROZEN_JOINT_NAMES = tuple(_PROTO5_DEFAULTS.FROZEN_JOINT_NAMES)
PROTO5_ALL_JOINT_NAMES = tuple(_PROTO5_DEFAULTS.ALL_JOINT_NAMES)
PROTO5_FINGERTIP_LINK_BODY_NAMES = (
    "RHand_ITIP_LINK",
    "RHand_MTIP_LINK",
    "RHand_TTIP_LINK",
)
PROTO5_6AF_BODY_NAMES = (
    "RHand_I6AF_LINK",
    "RHand_M6AF_LINK",
    "RHand_T6AF_LINK",
)
PROTO5_6AF_PARENT_BODY_NAMES = (
    "RHand_I3Y_LINK",
    "RHand_M3Y_LINK",
    "RHand_T3Y_LINK",
)
PROTO5_CONTACT_SENSOR_NAMES = (
    "index_screwdriver_contact",
    "middle_screwdriver_contact",
    "thumb_screwdriver_contact",
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
PROTO5_DEFAULT_FULL_JOINT_POS = tuple(_PROTO5_DEFAULTS.DEFAULT_FULL_JOINT_POS)

ALLEGRO_ROBOT_ROOT_POS = (0.0, -0.095, 1.33)
ALLEGRO_ROBOT_ROOT_ROT_WXYZ = (0.664463, 0.2418448, 0.2418448, 0.664463)
PROTO5_ROBOT_ROOT_POS = tuple(_PROTO5_DEFAULTS.PROTO5_SCREWDRIVER_ROOT_POS)
PROTO5_ROBOT_ROOT_ROT_WXYZ = tuple(_PROTO5_DEFAULTS.PROTO5_SCREWDRIVER_ROOT_ROT)
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


def rotate_vectors_by_inverse_quat(vectors: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat_wxyz = torch.as_tensor(quat_wxyz, device=vectors.device, dtype=vectors.dtype)
    inv_quat = torch.cat((quat_wxyz[..., :1], -quat_wxyz[..., 1:]), dim=-1)
    return rotate_vectors_by_quat(vectors, inv_quat)


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
        action_repeat: int = 1,
        save_recovery_frames: bool = False,
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
        self.action_repeat = max(1, int(action_repeat))
        self.save_recovery_frames = bool(save_recovery_frames)
        self.wrench_perturb_inds = []
        self._step_index = 0
        self._frame_id = 0
        self.frame_fpath = None
        self.frame_id = 0
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
        if value == 0 and self.save_recovery_frames and self.frame_fpath is not None:
            self._record_frame(force_render=True, sync_joint_targets=False)

    @property
    def unwrapped(self):
        return self._unwrapped

    @property
    def scene(self):
        return self._unwrapped.scene

    def reset(self):
        self.wrench_perturb_inds = []
        self._frame_id = 0
        self._step_index = 0
        ret = self.env.reset()
        self.refresh_default_dof_pos()
        self.force_render(sync_joint_targets=True)
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

    def get_full_dof_reference(self, env_id: int = 0) -> torch.Tensor:
        """Return the current full-hand joint state in the planner's semantic order."""
        env_id = int(env_id)
        try:
            robot = self.scene["robot"]
            source = getattr(robot.data, "joint_pos", None)
            if source is None:
                source = getattr(robot.data, "default_joint_pos", None)
            if source is None:
                raise AttributeError("joint_pos")
            joint_pos = _as_2d_tensor(source, device=self.device, dtype=torch.float32)
            return joint_pos[env_id, self._all_joint_ids()].clone()
        except (AttributeError, KeyError, IndexError, RuntimeError, ValueError):
            return self.default_dof_pos[env_id].detach().clone()

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

    @staticmethod
    def _find_bodies(asset, body_names: Sequence[str]) -> list[int]:
        body_ids, _ = asset.find_bodies(tuple(body_names), preserve_order=True)
        return list(body_ids)

    def _robot_body_ids(self, body_names: Sequence[str], cache_attr: str) -> list[int]:
        cached = getattr(self, cache_attr, None)
        if cached is not None:
            return list(cached)
        robot = self.scene["robot"]
        body_ids = self._find_bodies(robot, body_names)
        if len(body_ids) != len(body_names):
            raise ValueError(f"Could not resolve robot bodies {tuple(body_names)}.")
        setattr(self, cache_attr, tuple(int(idx) for idx in body_ids))
        return body_ids

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

    def get_screwdriver_position_robot(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return the object root position in the robot root frame."""
        device = self.device if device is None else torch.device(device)
        env_ids = self._env_ids()
        robot = self.scene["robot"]
        obj = self.scene["obj"]
        root_pos_w = robot.data.root_pos_w.index_select(0, env_ids.to(robot.data.root_pos_w.device)).to(
            device=device,
            dtype=dtype,
        )
        root_quat_w = robot.data.root_quat_w.index_select(0, env_ids.to(robot.data.root_quat_w.device)).to(
            device=device,
            dtype=dtype,
        )
        obj_pos_source = getattr(obj.data, "root_pos_w", None)
        if obj_pos_source is None:
            obj_pos_source = getattr(obj.data, "root_link_pos_w", None)
        if obj_pos_source is None:
            raise RuntimeError("Object asset does not expose root_pos_w or root_link_pos_w.")
        obj_pos_w = obj_pos_source.index_select(0, env_ids.to(obj_pos_source.device)).to(device=device, dtype=dtype)
        return rotate_vectors_by_inverse_quat(obj_pos_w - root_pos_w, root_quat_w).to(dtype=dtype)

    def get_contact_state(self, threshold: float = 1.0e-6) -> torch.Tensor:
        """Return simulator contact flags per fingertip in index/middle/thumb order."""
        flags = []
        for sensor_name in PROTO5_CONTACT_SENSOR_NAMES:
            try:
                sensor = self.scene[sensor_name]
                data = sensor.data
                forces = getattr(data, "net_forces_w_history", None)
                if forces is None:
                    forces = getattr(data, "net_forces_w", None)
                if forces is None:
                    raise AttributeError(sensor_name)
                forces = torch.as_tensor(forces, device=self.device, dtype=torch.float32)
                if forces.ndim == 4:
                    norm = torch.linalg.norm(forces, dim=-1).amax(dim=(1, 2))
                elif forces.ndim == 3:
                    norm = torch.linalg.norm(forces, dim=-1).amax(dim=1)
                elif forces.ndim == 2:
                    norm = torch.linalg.norm(forces, dim=-1)
                else:
                    raise ValueError(f"Unsupported contact force shape {tuple(forces.shape)} for {sensor_name}.")
                flags.append((norm > float(threshold)).to(dtype=torch.float32))
            except (AttributeError, KeyError, RuntimeError, ValueError):
                flags.append(torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32))
        return torch.stack(flags, dim=-1)

    def get_contact_points(self) -> torch.Tensor:
        """Return fingertip/contact-point positions in the robot frame."""
        robot = self.scene["robot"]
        try:
            body_ids = self._robot_body_ids(PROTO5_FINGERTIP_LINK_BODY_NAMES, "_proto5_tip_body_ids")
            body_pos_w = robot.data.body_pos_w[:, body_ids].to(device=self.device, dtype=torch.float32)
            root_pos_w = robot.data.root_pos_w.to(device=self.device, dtype=torch.float32)
            root_quat_w = robot.data.root_quat_w.to(device=self.device, dtype=torch.float32)
            root_quat_per_finger = root_quat_w[:, None, :].expand(-1, len(body_ids), -1)
            return rotate_vectors_by_inverse_quat(
                (body_pos_w - root_pos_w[:, None, :]).reshape(-1, 3),
                root_quat_per_finger.reshape(-1, 4),
            ).reshape(self.num_envs, len(body_ids), 3)
        except (AttributeError, KeyError, RuntimeError, ValueError):
            return torch.zeros((self.num_envs, 3, 3), device=self.device, dtype=torch.float32)

    def get_contact_wrenches(self, *, strict: bool = False) -> torch.Tensor:
        """Return Proto5 6AF incoming joint wrenches in the robot frame."""
        if self.hand != SCREWDRIVER_HAND_PROTO5:
            if strict:
                raise RuntimeError("6D contact wrench telemetry is only implemented for Proto5.")
            return torch.zeros((self.num_envs, 3, 6), device=self.device, dtype=torch.float32)

        robot = self.scene["robot"]
        try:
            if str(MODEL_MISMATCH_PATH) not in sys.path:
                sys.path.insert(0, str(MODEL_MISMATCH_PATH))
            from model_mismatch.utils.proto5_wrenches import extract_proto5_6af_wrenches_robot_frame

            return extract_proto5_6af_wrenches_robot_frame(
                robot,
                env_ids=self._env_ids(),
                device=self.device,
            )
        except (AttributeError, KeyError, RuntimeError, ValueError) as exc:
            if strict:
                raise RuntimeError("Proto5 6D contact wrench telemetry is unavailable.") from exc
            return torch.zeros((self.num_envs, 3, 6), device=self.device, dtype=torch.float32)

    def get_contact_forces(self, *, strict: bool = False) -> torch.Tensor:
        return self.get_contact_wrenches(strict=strict)[..., :3]

    def get_tactile_observation(self, *, strict_wrenches: bool = False) -> dict[str, torch.Tensor]:
        wrenches = self.get_contact_wrenches(strict=strict_wrenches)
        return {
            "contact_state": self.get_contact_state(),
            "contact_points": self.get_contact_points(),
            "contact_wrenches": wrenches,
            "contact_forces": wrenches[..., :3],
        }

    def _write_robot_root_default(self, env_ids: torch.Tensor) -> None:
        robot = self.scene["robot"]
        root_state = robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] += self.scene.env_origins[env_ids]
        robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def _write_obj_root_default(self, env_ids: torch.Tensor, screwdriver_pos_robot=None) -> None:
        obj = self.scene["obj"]
        root_state = obj.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] += self.scene.env_origins[env_ids]
        if screwdriver_pos_robot is not None:
            robot = self.scene["robot"]
            position_robot = torch.as_tensor(
                screwdriver_pos_robot,
                device=self.device,
                dtype=root_state.dtype,
            )
            if position_robot.ndim == 1:
                position_robot = position_robot.reshape(1, 3)
            if position_robot.shape[0] == 1 and len(env_ids) != 1:
                position_robot = position_robot.expand(len(env_ids), -1)
            if position_robot.shape != (len(env_ids), 3):
                raise ValueError(
                    "screwdriver_pos_robot must have shape (3,) or (num_envs, 3), "
                    f"got {tuple(position_robot.shape)} for num_envs={len(env_ids)}."
                )
            robot_root_state = robot.data.default_root_state[env_ids].clone()
            robot_root_state[:, 0:3] += self.scene.env_origins[env_ids]
            root_state[:, 0:3] = robot_root_state[:, 0:3] + rotate_vectors_by_quat(
                position_robot,
                robot_root_state[:, 3:7].to(device=self.device, dtype=position_robot.dtype),
            )
        obj.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        obj.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
        return root_state[:, 0:3].detach().clone()

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
        if self.hand == SCREWDRIVER_HAND_PROTO5 and hasattr(robot.data, "joint_pos"):
            joint_pos = robot.data.joint_pos[env_ids].clone()
        else:
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
        self._render_if_available()

    def _render_if_available(self) -> bool:
        sim = getattr(self._unwrapped, "sim", None)
        if sim is None:
            return False
        has_gui = bool(sim.has_gui()) if hasattr(sim, "has_gui") else False
        has_rtx = bool(sim.has_rtx_sensors()) if hasattr(sim, "has_rtx_sensors") else False
        if not (has_gui or has_rtx):
            return False
        sim.render()
        return True

    def force_render(self, *, sync_joint_targets: bool = False) -> None:
        """Synchronize direct state writes to USD/Fabric and update camera/viewer pixels."""
        sim = getattr(self._unwrapped, "sim", None)
        scene = getattr(self._unwrapped, "scene", None)
        if sim is None or scene is None:
            return
        if sync_joint_targets:
            try:
                robot = self.scene["robot"]
                env_ids = self._env_ids()
                robot.set_joint_position_target(robot.data.joint_pos[env_ids].clone(), env_ids=env_ids)
            except (AttributeError, KeyError, RuntimeError, ValueError):
                pass
        if hasattr(scene, "write_data_to_sim"):
            scene.write_data_to_sim()
        if hasattr(sim, "forward"):
            sim.forward()
        if hasattr(scene, "update") and hasattr(sim, "get_physics_dt"):
            scene.update(sim.get_physics_dt())
        self._render_if_available()

    def _frame_path(self) -> Path | None:
        if self.frame_id is None or self.frame_fpath is None:
            return None
        frame_dir = Path(self.frame_fpath)
        frame_dir.mkdir(parents=True, exist_ok=True)
        return frame_dir / f"frame_{int(self.frame_id):06d}.png"

    def _write_camera_frame(self, save_path: Path) -> None:
        try:
            camera = self.scene["tiled_camera"]
        except KeyError as exc:
            raise RuntimeError(
                "save_recovery_frames=True requires a tiled_camera sensor. "
                "Make sure cameras are enabled for the IsaacLab environment."
            ) from exc
        try:
            rgb_data = camera.data.output["rgb"]
        except (AttributeError, KeyError) as exc:
            raise RuntimeError("tiled_camera does not expose RGB output for recovery frame capture.") from exc
        if rgb_data.shape[0] < 1:
            raise RuntimeError(f"tiled_camera RGB output has no env-0 frame: shape={tuple(rgb_data.shape)}")
        img_np = rgb_data[0].detach().cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        from PIL import Image

        img = Image.fromarray(img_np)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img.save(save_path)

    def _record_frame(self, *, force_render: bool = False, sync_joint_targets: bool = False) -> None:
        if self.frame_id is None:
            return
        if self.save_recovery_frames and self.frame_fpath is not None:
            if force_render:
                self.force_render(sync_joint_targets=sync_joint_targets)
            frame_path = self._frame_path()
            if frame_path is not None:
                self._write_camera_frame(frame_path)
        self._frame_id += 1

    def set_pose(self, state, screwdriver_pos_robot=None):
        state = _as_2d_tensor(state, device=self.device, dtype=torch.float32)
        if state.shape[-1] < 15:
            raise ValueError(f"Expected state with at least 15 values, got shape {tuple(state.shape)}.")
        if state.shape[0] == 1 and self.num_envs > 1:
            state = state.repeat(self.num_envs, 1)
        env_ids = self._env_ids()
        self._write_robot_root_default(env_ids)
        try:
            obj_root_pos = self._write_obj_root_default(env_ids, screwdriver_pos_robot=screwdriver_pos_robot)
        except TypeError:
            if screwdriver_pos_robot is not None:
                raise
            obj_root_pos = self._write_obj_root_default(env_ids)
        if screwdriver_pos_robot is not None:
            self.table_pose = obj_root_pos[0].detach().clone().to(device=self.device, dtype=torch.float32)
            self.obj_pose = self.table_pose
        self._write_robot_joint_state(state[:, :12], env_ids)
        self._write_obj_joint_state(state[:, 12:15], env_ids)
        self._sync_scene()
        self._record_frame(force_render=False)

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
        ret = None
        for _ in range(self.action_repeat):
            # self._maybe_apply_external_perturbation()
            ret = self.env.step(env_action)
            # self._clear_external_force_torque()
            self.force_render(sync_joint_targets=False)
            self._record_frame(force_render=False)
            self._step_index += 1
        return ret

    def get_force_sensor_data(self, *args, **kwargs):
        return self.get_contact_forces().reshape(self.num_envs, 9)


class HardwareVisualizationShim:
    """No-op visualization surface for legacy hardware-mode branches."""

    def __init__(self, env):
        self.env = env
        self.frame_fpath = None
        self.frame_id = None

    def set_pose(self, *args, **kwargs):
        return None

    def zero_obj_velocity(self):
        return None

    def write_image(self):
        return None

    def get_state(self):
        return self.env.get_state()


class HardwareScrewdriverRecoveryEnv:
    """Expose the CCAI recovery contract on top of the model_mismatch hardware runtime."""

    is_hardware = True

    def __init__(self, config: dict, *, runtime=None, device: str | torch.device | None = None):
        self.config = dict(config)
        self.hand = get_hand_spec(str(self.config.get("hand", SCREWDRIVER_HAND_PROTO5))).name
        if self.hand != SCREWDRIVER_HAND_PROTO5:
            raise ValueError("HardwareScrewdriverRecoveryEnv currently supports hand: proto5.")
        self.hand_spec = get_hand_spec(self.hand)
        self.device = torch.device(device or self.config.get("sim_device", self.config.get("device", "cpu")))
        self.num_envs = 1
        self.action_dim = 12
        self.proto5_control_wrist = bool(self.config.get("proto5_control_wrist", False))
        self.hardware_track_wrist_state = bool(self.config.get("hardware_track_wrist_state", True))
        self.hardware_use_live_screwdriver_position = bool(
            self.config.get("hardware_use_live_screwdriver_position", True)
        )
        self.hardware_use_live_screwdriver_orientation = bool(
            self.config.get("hardware_use_live_screwdriver_orientation", True)
        )
        self.hardware_debug_mocap_orientation = bool(
            self.config.get("hardware_debug_mocap_orientation", False)
        )
        self.default_dof_pos = torch.tensor(
            self.hand_spec.default_full_joint_pos,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, -1)
        self.world_trans = create_world_transform(self.hand, self.device)
        csvto_position_robot = torch.as_tensor(
            self.config.get(
                "csvto_screwdriver_position_robot",
                DEFAULT_SCREWDRIVER_POSITION_ROBOT,
            ),
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 3)
        self.csvto_default_object_pose_world = self.world_trans.transform_points(
            csvto_position_robot
        )[0]
        self.table_pose = self.csvto_default_object_pose_world.detach().clone()
        self.obj_pose = self.table_pose
        self.external_wrench_perturb = False
        self.wrench_perturb_inds = []
        self.frame_fpath = None
        self.frame_id = None
        self._observed_object_orientation_fallback = None
        self.runtime = runtime if runtime is not None else self._create_runtime()
        self._update_object_pose()

    @property
    def unwrapped(self):
        return self

    def _create_runtime(self):
        if str(MODEL_MISMATCH_PATH) not in sys.path:
            sys.path.insert(0, str(MODEL_MISMATCH_PATH))
        from model_mismatch.utils.screwdriver_hardware_runtime import create_hardware_runtime

        args = self._runtime_args_from_config()
        return create_hardware_runtime(args, self.device)

    def _runtime_args_from_config(self):
        default_q = list(self.hand_spec.default_full_joint_pos)
        values = {
            "hand": self.hand,
            "proto5_control_wrist": self.proto5_control_wrist,
            "hardware_track_wrist_state": self.hardware_track_wrist_state,
            "hardware_ros_config": self.config.get("hardware_ros_config"),
            "hardware_profile": self.config.get("hardware_profile", self.hand),
            "hardware_execute": bool(self.config.get("hardware_execute", False)),
            "hardware_command_topic": self.config.get("hardware_command_topic"),
            "hardware_joint_state_topic": self.config.get("hardware_joint_state_topic"),
            "hardware_mocap_topic": self.config.get("hardware_mocap_topic"),
            "hardware_proto5_wrench_topic": self.config.get("hardware_proto5_wrench_topic"),
            "hardware_proto5_wrench_fixed_joint_names": self.config.get("hardware_proto5_wrench_fixed_joint_names"),
            "hardware_num_repeat": int(self.config.get("hardware_num_repeat", 10)),
            "hardware_command_mode": self.config.get("hardware_command_mode", "repeat"),
            "hardware_command_duration_s": float(self.config.get("hardware_command_duration_s", 1.0 / 12.0)),
            "hardware_allow_placeholder_wrenches": bool(
                self.config.get("hardware_allow_placeholder_wrenches", False)
            ),
            "hardware_default_q16": self.config.get("hardware_default_q16", default_q),
            "hardware_node_name": self.config.get("hardware_node_name", "screwdriver_recovery_hardware"),
            "hardware_screwdriver_mocap_object": self.config.get(
                "hardware_screwdriver_mocap_object",
                "blue_screwdriver_catching",
            ),
        }
        return SimpleNamespace(**values)

    def _raw_state15_from_runtime(self) -> torch.Tensor:
        if hasattr(self.runtime, "get_current_state"):
            state = self.runtime.get_current_state(self.device)
        elif hasattr(self.runtime, "get_state"):
            state = self.runtime.get_state()
            if isinstance(state, dict):
                state = state["q"]
        else:
            raise AttributeError("Hardware runtime must expose get_current_state(...) or get_state().")
        state = torch.as_tensor(state, device=self.device, dtype=torch.float32).reshape(1, -1)
        if state.shape[-1] < 15:
            raise RuntimeError(f"Expected hardware state with at least 15 values, got {tuple(state.shape)}.")
        return state[:, :15].clone()

    def _apply_observed_object_orientation(self, state15: torch.Tensor) -> tuple[torch.Tensor, bool]:
        state15 = state15.clone()
        use_observed_orientation = (
            self._observed_object_orientation_fallback is not None
            and (
                not self.hardware_use_live_screwdriver_orientation
                or torch.allclose(state15[:, 12:15], torch.zeros_like(state15[:, 12:15]))
            )
        )
        if use_observed_orientation:
            state15[:, 12:15] = self._observed_object_orientation_fallback.to(
                device=state15.device,
                dtype=state15.dtype,
            )
        return state15, bool(use_observed_orientation)

    @staticmethod
    def _diagnostic_value(value):
        if value is None:
            return None
        try:
            return torch.as_tensor(value).detach().cpu().reshape(-1).tolist()
        except Exception:
            return value

    def print_mocap_orientation_diagnostic(self, *, context: str) -> None:
        if not self.hardware_debug_mocap_orientation:
            return
        raw_state15 = self._raw_state15_from_runtime()
        state15, used_observed_orientation = self._apply_observed_object_orientation(raw_state15)
        snapshot = {}
        snapshot_error = None
        if hasattr(self.runtime, "read_hardware_snapshot"):
            try:
                snapshot = self.runtime.read_hardware_snapshot()
            except Exception as exc:
                snapshot_error = repr(exc)
        profile = getattr(self.runtime, "profile", None)
        observed = self._observed_object_orientation_fallback
        print(
            "Hardware mocap orientation diagnostic "
            f"context={context} "
            f"runtime_orientation={self._diagnostic_value(raw_state15[:, 12:15])} "
            f"returned_orientation={self._diagnostic_value(state15[:, 12:15])} "
            f"observed_fallback={self._diagnostic_value(observed)} "
            f"used_observed_fallback={used_observed_orientation} "
            f"live_orientation_enabled={self.hardware_use_live_screwdriver_orientation} "
            f"snapshot_orientation={self._diagnostic_value(snapshot.get('mocap_orientation'))} "
            f"snapshot_orientation_source={snapshot.get('mocap_orientation_source')} "
            f"snapshot_base_configured={snapshot.get('mocap_debug_base_configured')} "
            f"snapshot_base_msg_received={snapshot.get('mocap_debug_base_msg_received')} "
            f"snapshot_object_stamp_s={snapshot.get('mocap_debug_object_stamp_s')} "
            f"snapshot_base_stamp_s={snapshot.get('mocap_debug_base_stamp_s')} "
            f"snapshot_stamp_delta_s={snapshot.get('mocap_debug_stamp_delta_s')} "
            f"snapshot_object_receive_age_s={snapshot.get('mocap_debug_object_receive_age_s')} "
            f"snapshot_base_receive_age_s={snapshot.get('mocap_debug_base_receive_age_s')} "
            f"snapshot_receive_delta_s={snapshot.get('mocap_debug_receive_delta_s')} "
            f"snapshot_object_position={self._diagnostic_value(snapshot.get('mocap_debug_object_position'))} "
            f"snapshot_object_quat_xyzw={self._diagnostic_value(snapshot.get('mocap_debug_object_quat_xyzw'))} "
            f"snapshot_object_euler_rxyz={self._diagnostic_value(snapshot.get('mocap_debug_object_euler_rxyz'))} "
            f"snapshot_base_position={self._diagnostic_value(snapshot.get('mocap_debug_base_position'))} "
            f"snapshot_base_quat_xyzw={self._diagnostic_value(snapshot.get('mocap_debug_base_quat_xyzw'))} "
            f"snapshot_base_euler_rxyz={self._diagnostic_value(snapshot.get('mocap_debug_base_euler_rxyz'))} "
            f"mocap_topic={getattr(profile, 'mocap_topic', None)} "
            f"turning_base_topic={getattr(profile, 'screwdriver_turning_base_topic', None)} "
            f"snapshot_error={snapshot_error}",
            flush=True,
        )

    def _state15_from_runtime(self) -> torch.Tensor:
        raw_state15 = self._raw_state15_from_runtime()
        state15, _ = self._apply_observed_object_orientation(raw_state15)
        return state15

    def get_measured_joint_state(self) -> tuple[tuple[str, ...], torch.Tensor]:
        if hasattr(self.runtime, "get_measured_joint_state"):
            names, positions = self.runtime.get_measured_joint_state(self.device, dtype=torch.float32)
            return tuple(str(name) for name in names), _as_2d_tensor(positions, device=self.device, dtype=torch.float32)
        if hasattr(self.runtime, "get_named_joint_positions"):
            names, positions = self.runtime.get_named_joint_positions(self.device, dtype=torch.float32)
            return tuple(str(name) for name in names), _as_2d_tensor(positions, device=self.device, dtype=torch.float32)
        if hasattr(self.runtime, "get_full_joint_positions"):
            positions = self.runtime.get_full_joint_positions(self.device, dtype=torch.float32)
            names = tuple(getattr(self.runtime, "state_joint_names", ()))
            if not names and hasattr(self.runtime, "profile"):
                profile = self.runtime.profile
                names = tuple(getattr(profile, "state_joint_names", ()) or getattr(profile, "command_joint_names", ()))
            if names:
                return tuple(str(name) for name in names), _as_2d_tensor(positions, device=self.device, dtype=torch.float32)
        raise RuntimeError(
            "hardware_track_wrist_state=True requires the hardware runtime to expose named measured joints."
        )

    def get_full_dof_reference(self, env_id: int = 0) -> torch.Tensor:
        full = self.default_dof_pos[int(env_id)].detach().clone()
        names, positions = self.get_measured_joint_state()
        row = positions[int(env_id) if positions.shape[0] > int(env_id) else 0].reshape(-1)
        if len(names) != int(row.numel()):
            raise RuntimeError(
                f"Hardware measured joint names/positions length mismatch: names={len(names)}, positions={int(row.numel())}."
            )
        by_name = {name: row[idx] for idx, name in enumerate(names)}
        missing_wrist = [name for name in self.hand_spec.wrist_joint_names if name not in by_name]
        if self.hardware_track_wrist_state and missing_wrist:
            raise RuntimeError(
                "hardware_track_wrist_state=True but wrist joints are unavailable from hardware state: "
                f"{missing_wrist}."
            )
        for joint_idx, name in enumerate(self.hand_spec.all_joint_names):
            if name in by_name:
                full[joint_idx] = by_name[name].to(device=full.device, dtype=full.dtype)
        return full

    def get_wrist_joint_state(self, env_id: int = 0) -> torch.Tensor:
        full = self.get_full_dof_reference(env_id=env_id)
        return full[: len(self.hand_spec.wrist_joint_names)].detach().clone()

    def set_observed_object_orientation(self, orientation) -> None:
        orientation = torch.as_tensor(orientation, device=self.device, dtype=torch.float32).reshape(1, -1)
        if orientation.shape[-1] != 3:
            raise ValueError(f"Expected 3 object orientation values, got shape {tuple(orientation.shape)}.")
        self._observed_object_orientation_fallback = orientation

    def capture_observed_object_orientation(self) -> torch.Tensor:
        if hasattr(self.runtime, "get_screwdriver_orientation_euler"):
            orientation = self.runtime.get_screwdriver_orientation_euler(self.device, dtype=torch.float32)
            orientation = torch.as_tensor(orientation, device=self.device, dtype=torch.float32).reshape(1, -1)
            if orientation.shape[-1] != 3:
                raise ValueError(
                    "Expected runtime screwdriver orientation with 3 values, "
                    f"got shape {tuple(orientation.shape)}."
                )
        else:
            orientation = self._raw_state15_from_runtime()[:, 12:15]
        self.set_observed_object_orientation(orientation)
        return self._observed_object_orientation_fallback.detach().clone()

    def _update_object_pose(self) -> None:
        if not self.hardware_use_live_screwdriver_position:
            self.table_pose = self.csvto_default_object_pose_world.detach().clone()
            self.obj_pose = self.table_pose
            return
        if hasattr(self.runtime, "get_screwdriver_position_robot"):
            try:
                position = self.runtime.get_screwdriver_position_robot(self.device, dtype=torch.float32)
                self.table_pose = torch.as_tensor(position, device=self.device, dtype=torch.float32).reshape(1, 3)[0]
                self.obj_pose = self.table_pose
            except Exception:
                pass

    def get_screwdriver_position_robot(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return the live model-frame position used by position-conditioned policies.

        This is intentionally independent of ``table_pose``.  CSVTO consumes
        ``table_pose`` as a world-frame object asset pose, while DiffPF consumes
        this robot/model-frame observation after the hardware profile's mocap,
        flange-frame, and offset transforms have been applied by the runtime.
        """
        device = self.device if device is None else torch.device(device)
        if not hasattr(self.runtime, "get_screwdriver_position_robot"):
            raise AttributeError(
                "Hardware runtime does not expose get_screwdriver_position_robot()."
            )
        position = self.runtime.get_screwdriver_position_robot(device, dtype=dtype)
        return torch.as_tensor(position, device=device, dtype=dtype).reshape(1, 3)

    def reset(self):
        self.wrench_perturb_inds = []
        if hasattr(self.runtime, "reset"):
            ret = self.runtime.reset()
        else:
            ret = self._state15_from_runtime(), {}
        self._update_object_pose()
        return ret

    def close(self):
        if hasattr(self.runtime, "close"):
            return self.runtime.close()
        return None

    def get_state(self):
        state15 = self._state15_from_runtime()
        self._update_object_pose()
        orientation = state15[:, 12:15]
        q = pack_ccai_state(state15[:, :12], orientation)
        return {
            "q": q,
            "screwdriver_ori_euler": orientation,
            "screwdriver_ori": orientation,
            "screwdriver_angle": orientation[:, 2:3],
        }

    def set_pose(self, state):
        state = _as_2d_tensor(state, device=self.device, dtype=torch.float32)
        if state.shape[-1] < 12:
            raise ValueError(f"Expected state with at least 12 active joints, got shape {tuple(state.shape)}.")
        return self.step(state[:, :12])

    def step(self, action):
        active_targets = _as_2d_tensor(action, device=self.device, dtype=torch.float32)
        if active_targets.shape[-1] < 12:
            raise ValueError(f"Expected action with at least 12 values, got shape {tuple(active_targets.shape)}.")
        ret = self.runtime.step(active_targets[:, :12])
        self._update_object_pose()
        return ret

    def zero_obj_velocity(self):
        return None

    def force_render(self, *args, **kwargs):
        return None

    def set_external_wrench_perturb(self, enabled, rand_pct=None):
        del enabled, rand_pct
        self.external_wrench_perturb = False

    def get_contact_wrenches(self, *, strict: bool = False) -> torch.Tensor:
        try:
            return self.runtime.read_wrenches_robot(self.device, dtype=torch.float32)
        except Exception:
            if strict:
                raise
            return torch.zeros((1, 3, 6), device=self.device, dtype=torch.float32)

    def get_contact_forces(self, *, strict: bool = False) -> torch.Tensor:
        if hasattr(self.runtime, "read_contact_forces_robot"):
            try:
                return self.runtime.read_contact_forces_robot(self.device, dtype=torch.float32)
            except Exception:
                if strict:
                    raise
        return self.get_contact_wrenches(strict=strict)[..., :3]

    def get_contact_points(self) -> torch.Tensor:
        if hasattr(self.runtime, "get_contact_points_robot"):
            try:
                return self.runtime.get_contact_points_robot(self.device, dtype=torch.float32)
            except Exception:
                pass
        return torch.zeros((1, 3, 3), device=self.device, dtype=torch.float32)

    def get_contact_state(self, threshold: float = 1.0e-6) -> torch.Tensor:
        forces = self.get_contact_forces()
        return (torch.linalg.norm(forces, dim=-1) > float(threshold)).to(dtype=torch.float32)

    def get_tactile_observation(self, *, strict_wrenches: bool = False) -> dict[str, torch.Tensor]:
        wrenches = self.get_contact_wrenches(strict=strict_wrenches)
        return {
            "contact_state": (torch.linalg.norm(wrenches[..., :3], dim=-1) > 1.0e-6).to(dtype=torch.float32),
            "contact_points": self.get_contact_points(),
            "contact_wrenches": wrenches,
            "contact_forces": wrenches[..., :3],
        }

    def get_force_sensor_data(self, *args, **kwargs):
        return self.get_contact_forces().reshape(self.num_envs, 9)

    def get_environment_parameters(self, env_id: int = 0) -> dict[str, float]:
        del env_id
        return {
            "screwdriver_friction": float(
                self.config.get("screwdriver_friction", self.config.get("friction_coefficient", 1.0))
            ),
            "yaw_joint_friction": float(self.config.get("yaw_joint_friction", 0.0)),
        }
