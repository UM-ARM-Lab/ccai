import torch
from torch.func import vmap

from ccai.allegro_contact import AllegroObjectProblem
from ccai.allegro_screwdriver_problem import (
    AllegroScrewdriver,
    PROTO5_ACTIVE_JOINT_MAX,
    PROTO5_ACTIVE_JOINT_MIN,
    PROTO5_FULL_JOINT_INDEX,
    Proto5Screwdriver,
)
from ccai.utils import allegro_utils


def test_proto5_screwdriver_default_dof_pos_is_full_reference(monkeypatch):
    captured_kwargs = {}

    def fake_allegro_init(self, *args, **kwargs):
        captured_kwargs.update(kwargs)
        self.dx = 0
        self.du = 0
        self.dg = 0
        self.dh = 0
        self.dz = 0
        self.x_min = None
        self.x_max = None
        self.squared_slack = True

    monkeypatch.setattr(AllegroScrewdriver, "__init__", fake_allegro_init)
    full_reference = torch.arange(18, dtype=torch.float32)

    Proto5Screwdriver(
        full_dof_reference=full_reference,
        device="cpu",
    )

    default_dof_pos = captured_kwargs["default_dof_pos"]
    assert default_dof_pos.shape == (18,)
    torch.testing.assert_close(default_dof_pos, full_reference)
    assert captured_kwargs["full_robot_dof"] == 18
    assert captured_kwargs["contact_patch_link_frame_z_max"] == -0.003
    assert captured_kwargs["fingertip_contact_only"] is True
    assert captured_kwargs["filter_self_collision_query_points"] is False


def test_proto5_screwdriver_allows_contact_patch_z_max_override(monkeypatch):
    captured_kwargs = {}

    def fake_allegro_init(self, *args, **kwargs):
        captured_kwargs.update(kwargs)
        self.dx = 0
        self.du = 0
        self.dg = 0
        self.dh = 0
        self.dz = 0
        self.x_min = None
        self.x_max = None
        self.squared_slack = True

    monkeypatch.setattr(AllegroScrewdriver, "__init__", fake_allegro_init)

    Proto5Screwdriver(
        full_dof_reference=torch.zeros(18, dtype=torch.float32),
        contact_patch_link_frame_z_max=-0.004,
        device="cpu",
    )

    assert captured_kwargs["contact_patch_link_frame_z_max"] == -0.004


def test_proto5_screwdriver_wrist_control_passes_14d_controlled_mapping(monkeypatch):
    captured_kwargs = {}

    def fake_allegro_init(self, *args, **kwargs):
        captured_kwargs.update(kwargs)
        self.dx = 0
        self.du = 0
        self.dg = 0
        self.dh = 0
        self.dz = 0
        self.x_min = None
        self.x_max = None
        self.squared_slack = True

    monkeypatch.setattr(AllegroScrewdriver, "__init__", fake_allegro_init)
    full_reference = torch.arange(18, dtype=torch.float32)

    Proto5Screwdriver(
        full_dof_reference=full_reference,
        control_wrist=True,
        device="cpu",
    )

    expected_index = (
        PROTO5_FULL_JOINT_INDEX["index"]
        + PROTO5_FULL_JOINT_INDEX["middle"]
        + PROTO5_FULL_JOINT_INDEX["thumb"]
        + PROTO5_FULL_JOINT_INDEX["wrist"]
    )
    expected_min = torch.cat(
        [
            PROTO5_ACTIVE_JOINT_MIN["index"],
            PROTO5_ACTIVE_JOINT_MIN["middle"],
            PROTO5_ACTIVE_JOINT_MIN["thumb"],
            PROTO5_ACTIVE_JOINT_MIN["wrist"],
        ],
        dim=0,
    )
    expected_max = torch.cat(
        [
            PROTO5_ACTIVE_JOINT_MAX["index"],
            PROTO5_ACTIVE_JOINT_MAX["middle"],
            PROTO5_ACTIVE_JOINT_MAX["thumb"],
            PROTO5_ACTIVE_JOINT_MAX["wrist"],
        ],
        dim=0,
    )

    assert captured_kwargs["robot_dof"] == 14
    assert captured_kwargs["controlled_joint_index"] == expected_index
    torch.testing.assert_close(captured_kwargs["controlled_joint_min"], expected_min)
    torch.testing.assert_close(captured_kwargs["controlled_joint_max"], expected_max)


def test_partial_to_full_dof_pos_preserves_proto5_frozen_joints():
    class ConcreteAllegroObjectProblem(AllegroObjectProblem):
        def _con_eq(self, *args, **kwargs):
            raise NotImplementedError

        def _con_ineq(self, *args, **kwargs):
            raise NotImplementedError

    problem = object.__new__(ConcreteAllegroObjectProblem)
    problem.device = "cpu"
    problem.fingers = ["index", "middle", "thumb"]
    problem.num_fingers = 3
    problem.full_robot_dof = 18
    problem.joint_index = PROTO5_FULL_JOINT_INDEX
    problem.full_dof_reference = torch.arange(18, dtype=torch.float32)

    partial = torch.tensor(
        [
            100.0,
            101.0,
            102.0,
            103.0,
            200.0,
            201.0,
            202.0,
            203.0,
            300.0,
            301.0,
            302.0,
            303.0,
        ],
        dtype=torch.float32,
    )

    full = problem._partial_to_full_dof_pos(partial, reference=problem.full_dof_reference)
    expected = torch.arange(18, dtype=torch.float32)
    expected[2:6] = partial[0:4]
    expected[6:10] = partial[4:8]
    expected[14:18] = partial[8:12]

    assert full.shape == (18,)
    torch.testing.assert_close(full, expected)
    torch.testing.assert_close(problem._full_to_partial_dof_pos(full), partial)


def test_partial_to_full_dof_pos_maps_proto5_wrist_when_controlled():
    class ConcreteAllegroObjectProblem(AllegroObjectProblem):
        def _con_eq(self, *args, **kwargs):
            raise NotImplementedError

        def _con_ineq(self, *args, **kwargs):
            raise NotImplementedError

    problem = object.__new__(ConcreteAllegroObjectProblem)
    problem.device = "cpu"
    problem.fingers = ["index", "middle", "thumb"]
    problem.num_fingers = 3
    problem.full_robot_dof = 18
    problem.robot_dof = 14
    problem.joint_index = PROTO5_FULL_JOINT_INDEX
    problem.controlled_joint_index = (
        PROTO5_FULL_JOINT_INDEX["index"]
        + PROTO5_FULL_JOINT_INDEX["middle"]
        + PROTO5_FULL_JOINT_INDEX["thumb"]
        + PROTO5_FULL_JOINT_INDEX["wrist"]
    )
    problem.full_dof_reference = torch.arange(18, dtype=torch.float32)

    partial = torch.arange(14, dtype=torch.float32) + 100.0
    full = problem._partial_to_full_dof_pos(partial, reference=problem.full_dof_reference)
    expected = torch.arange(18, dtype=torch.float32)
    expected[2:6] = partial[0:4]
    expected[6:10] = partial[4:8]
    expected[14:18] = partial[8:12]
    expected[0:2] = partial[12:14]

    torch.testing.assert_close(full, expected)
    torch.testing.assert_close(problem._full_to_partial_dof_pos(full), partial)


def test_partial_to_full_state_maps_proto5_active_joints_and_supports_vmap():
    class ConcreteAllegroObjectProblem(AllegroObjectProblem):
        def _con_eq(self, *args, **kwargs):
            raise NotImplementedError

        def _con_ineq(self, *args, **kwargs):
            raise NotImplementedError

    problem = object.__new__(ConcreteAllegroObjectProblem)
    problem.device = "cpu"
    problem.fingers = ["index", "middle", "thumb"]
    problem.num_fingers = 3
    problem.full_robot_dof = 18
    problem.joint_index = PROTO5_FULL_JOINT_INDEX
    problem.full_dof_reference = torch.arange(18, dtype=torch.float32)

    partial = torch.tensor(
        [
            [100.0, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 203.0, 300.0, 301.0, 302.0, 303.0],
            [110.0, 111.0, 112.0, 113.0, 210.0, 211.0, 212.0, 213.0, 310.0, 311.0, 312.0, 313.0],
        ],
        dtype=torch.float32,
    )
    expected = torch.arange(18, dtype=torch.float32).repeat(2, 1)
    expected[:, 2:6] = partial[:, 0:4]
    expected[:, 6:10] = partial[:, 4:8]
    expected[:, 14:18] = partial[:, 8:12]

    torch.testing.assert_close(problem._partial_to_full_state(partial), expected)
    torch.testing.assert_close(vmap(problem._partial_to_full_state)(partial), expected)


def test_partial_to_full_dof_pos_preserves_allegro_ring_insert_behavior():
    class ConcreteAllegroObjectProblem(AllegroObjectProblem):
        def _con_eq(self, *args, **kwargs):
            raise NotImplementedError

        def _con_ineq(self, *args, **kwargs):
            raise NotImplementedError

    problem = object.__new__(ConcreteAllegroObjectProblem)
    problem.device = "cpu"
    problem.fingers = ["index", "middle", "thumb"]
    problem.num_fingers = 3
    problem.full_robot_dof = 16
    problem.joint_index = {
        "index": [0, 1, 2, 3],
        "middle": [4, 5, 6, 7],
        "ring": [8, 9, 10, 11],
        "thumb": [12, 13, 14, 15],
    }
    problem.full_dof_reference = torch.arange(16, dtype=torch.float32)
    partial = torch.arange(12, dtype=torch.float32) + 100.0

    full = problem._partial_to_full_dof_pos(partial, reference=problem.full_dof_reference)
    legacy_full = torch.cat((partial[:8], problem.full_dof_reference[8:12], partial[8:]))

    torch.testing.assert_close(full, legacy_full)
    torch.testing.assert_close(problem._full_to_partial_dof_pos(full), partial)


def test_visualize_trajectory_expands_proto5_partial_q_to_full_dof(monkeypatch, tmp_path):
    captured = {}

    class FakeScene:
        device = "cpu"

        def get_visualization_meshes(self, q, theta, pcd=None):
            captured["q"] = q.clone()
            captured["theta"] = theta.clone()
            assert q.shape[-1] == 18
            return [], []

    def fake_visualize_window(
        trajectory,
        scene,
        scene_path,
        fingers,
        obj_dof,
        headless=False,
        task="screwdriver",
        pcd=None,
        full_dof_reference=None,
            joint_index=None,
            camera_mode="preset",
            camera_parameters_path=None,
            save_camera_parameters_path=None,
            camera_setup_only=False,
        ):
        allegro_utils._collect_visualization_geometry(
            trajectory[0],
            scene,
            fingers,
            obj_dof,
            pcd=pcd,
            full_dof_reference=full_dof_reference,
            joint_index=joint_index,
        )

    import subprocess

    monkeypatch.setattr(allegro_utils, "_visualize_trajectory_window", fake_visualize_window)
    monkeypatch.setattr(subprocess, "call", lambda *args, **kwargs: 0)

    partial_q = torch.arange(12, dtype=torch.float32) + 100.0
    theta = torch.tensor([0.1, 0.2, 0.3, 0.0], dtype=torch.float32)
    full_reference = torch.arange(18, dtype=torch.float32)
    expected_q = full_reference.reshape(1, 18).clone()
    expected_q[:, 2:6] = partial_q[0:4]
    expected_q[:, 6:10] = partial_q[4:8]
    expected_q[:, 14:18] = partial_q[8:12]

    allegro_utils.visualize_trajectory(
        torch.cat((partial_q, theta)).reshape(1, -1),
        FakeScene(),
        tmp_path,
        fingers=["index", "middle", "thumb"],
        obj_dof=4,
        render_backend="window",
        full_dof_reference=full_reference,
        joint_index=PROTO5_FULL_JOINT_INDEX,
    )

    torch.testing.assert_close(captured["q"], expected_q)
    torch.testing.assert_close(captured["theta"], theta.reshape(1, 4))


def test_auto_camera_uses_combined_geometry_bounds():
    class FakeBBox:
        def __init__(self, min_bound, max_bound):
            self._min_bound = min_bound
            self._max_bound = max_bound

        def get_min_bound(self):
            return self._min_bound

        def get_max_bound(self):
            return self._max_bound

    class FakeGeometry:
        def __init__(self, min_bound, max_bound):
            self._bbox = FakeBBox(min_bound, max_bound)

        def get_axis_aligned_bounding_box(self):
            return self._bbox

    geometries = [
        FakeGeometry([0.0, -1.0, 2.0], [1.0, 1.0, 3.0]),
        FakeGeometry([-2.0, 3.0, -1.0], [2.0, 5.0, 1.0]),
    ]

    camera = allegro_utils._compute_auto_camera_from_geometries(geometries)

    assert camera is not None
    torch.testing.assert_close(
        torch.as_tensor(camera["center"], dtype=torch.float32),
        torch.tensor([0.0, 2.0, 1.0], dtype=torch.float32),
    )
