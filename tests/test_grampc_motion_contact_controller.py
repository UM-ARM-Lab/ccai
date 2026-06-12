import importlib.util
import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from ccai.controller import grampc_motion_contact_controller as grampc_mod
from ccai.controller.grampc_motion_contact_controller import (
    GRAMPCMotionContactTracker,
    PYGRAMPC_INSTALL_INSTRUCTION,
)


@dataclass
class DummyConfig:
    K_e: float = 100.0
    K_P: float = 3.0
    K_D: float = 1.0
    force_threshold: float = 0.2
    dt: float = 0.03
    horizon_length: int = 2
    w_f: float = 5.0
    w_q: float = 7.0
    w_p: float = 11.0
    w_u: float = 0.1
    w_ori: float = 0.3
    dq: int = 12
    df: int = 9


class FakeTransform:
    def __init__(self, matrix):
        self._matrix = matrix

    def get_matrix(self):
        return torch.tensor(self._matrix, dtype=torch.float32).unsqueeze(0)


class FakeChain:
    def __init__(self):
        self.frame_to_idx = {"index_tip": 0, "middle_tip": 1, "thumb_tip": 3}

    def forward_kinematics(self, q_full):
        q = q_full.detach().cpu().numpy()[0]
        out = {}
        finger_slices = {
            "index_tip": slice(0, 4),
            "middle_tip": slice(4, 8),
            "thumb_tip": slice(12, 16),
        }
        offsets = {
            "index_tip": np.array([0.0, 0.0, 0.0]),
            "middle_tip": np.array([0.05, 0.0, 0.0]),
            "thumb_tip": np.array([0.0, 0.05, 0.0]),
        }
        for name, slc in finger_slices.items():
            mat = np.eye(4)
            joints = q[slc]
            mat[:3, 3] = offsets[name] + np.array([joints.sum(), joints[0], joints[-1]])
            out[name] = FakeTransform(mat)
        return out


class FakeRobotSDF:
    def __init__(self):
        self.chain = FakeChain()


class FakeContactScenes:
    def __init__(self):
        self.robot_sdf = FakeRobotSDF()


class FakeProblem:
    def __init__(self):
        self.fingers = ["index", "middle", "thumb"]
        self.contact_fingers = ["index", "middle", "thumb"]
        self.num_fingers = 3
        self.device = "cpu"
        self.dx = 13
        self.skip_csvto = False
        self.ee_names = {
            "index": "index_tip",
            "middle": "middle_tip",
            "thumb": "thumb_tip",
        }
        self.joint_index = {
            "index": [0, 1, 2, 3],
            "middle": [4, 5, 6, 7],
            "ring": [8, 9, 10, 11],
            "thumb": [12, 13, 14, 15],
        }
        self.contact_scenes = FakeContactScenes()


def make_tracker(params=None):
    return GRAMPCMotionContactTracker(FakeProblem(), DummyConfig(), params=params or {})


def make_reference_state(q, qd, f):
    ref = np.zeros(13 + 12 + 9)
    ref[:12] = q
    ref[13:25] = qd
    ref[25:34] = f
    return ref


def test_dimensions_for_three_finger_allegro():
    tracker = make_tracker()
    assert tracker.nq == 12
    assert tracker.df == 9
    assert tracker.Nx == 33
    assert tracker.Nu == 12
    assert tracker.problem.Nx == 33
    assert tracker.problem.Nu == 12


def test_reference_conversion_uses_problem_finger_order():
    tracker = make_tracker(params={"force_upper_bound": 2.0})
    q = np.arange(12, dtype=float)
    qd = q + 10.0
    force = np.array([3.0, 0.0, 0.0, 0.0, -4.0, 0.0, 0.0, 0.0, 0.5])
    ref = make_reference_state(q, qd, force)
    tracker.set_reference_trajectory(lambda _t: ref, normals=np.tile([1.0, 0.0, 0.0], (3, 1)))

    knots = tracker._reference_knots()

    np.testing.assert_allclose(knots[0, :12], q)
    np.testing.assert_allclose(knots[0, 12:24], qd)
    np.testing.assert_allclose(knots[0, 24:33], [2.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.5])


def test_missing_pygrampc_error_is_actionable(monkeypatch):
    monkeypatch.setattr(grampc_mod, "_PYGRAMPC_IMPORT_ERROR", ImportError("missing"))
    with pytest.raises(ImportError, match="git\\+https://github.com/grampc/pygrampc"):
        grampc_mod.ensure_pygrampc_available()
    monkeypatch.setattr(grampc_mod, "_PYGRAMPC_IMPORT_ERROR", None)
    assert PYGRAMPC_INSTALL_INSTRUCTION in grampc_mod.PYGRAMPC_INSTALL_INSTRUCTION


def test_hfmc_weights_force_normal_and_position_tangent():
    tracker = make_tracker()
    normals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tracker.set_reference_trajectory(lambda _t: np.zeros(34), normals=normals)
    tracker.calc_HFMC_matrices(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]))

    np.testing.assert_allclose(tracker.Qf[:3, :3], tracker.Qf_scalar * np.diag([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(tracker.Qp[0][:3, :3], tracker.Qp_scalar * np.diag([0.0, 1.0, 1.0]))
    np.testing.assert_allclose(tracker.Qf[3:6, 3:6], np.zeros((3, 3)))
    np.testing.assert_allclose(tracker.Qp[1][:3, :3], tracker.Qp_scalar * np.eye(3))
    np.testing.assert_allclose(tracker.Qf[6:9, 6:9], tracker.Qf_scalar * np.diag([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(tracker.Qp[2][:3, :3], tracker.Qp_scalar * np.diag([1.0, 1.0, 0.0]))


def test_grampc_solve_returns_finite_delta():
    tracker = make_tracker(params={"force_upper_bound": 2.0, "dq_scale": 0.5})
    q = np.zeros(12)
    qd = np.ones(12) * 0.02
    f = np.array([0.5, 0.0, 0.0] * 3)
    ref = make_reference_state(q, qd, f)
    tracker.set_reference_trajectory(
        lambda _t: ref,
        normals=np.tile([1.0, 0.0, 0.0], (3, 1)),
        contact_points=np.zeros((3, 3)),
    )

    delta_q = tracker.solve(0.0, q, qd, f)

    assert delta_q.shape == (12,)
    assert np.all(np.isfinite(delta_q))


def test_recovery_utils_qp_backend_selects_legacy_controller(monkeypatch):
    fake_allegro_contact = types.ModuleType("ccai.allegro_contact")

    class FakeBase:
        def __init__(self, problem, params):
            self.problem = problem
            self.params = params
            self.online_iters = params.get("online_iters", 1)
            self.warmup_iters = params.get("warmup_iters", 1)

    fake_allegro_contact.AllegroManipulationProblem = object
    fake_allegro_contact.PositionControlConstrainedSVGDMPC = FakeBase

    fake_allegro_utils = types.ModuleType("ccai.utils.allegro_utils")
    fake_allegro_utils.visualize_trajectory = lambda *args, **kwargs: None

    fake_tactile = types.ModuleType("ccai.controller.tactile_feedback_controller")
    fake_tactile.ControllerConfig = DummyConfig

    class FakeQPController:
        def __init__(self, problem, config):
            self.problem = problem
            self.config = config

    fake_tactile.TactileFeedbackQPController = FakeQPController

    monkeypatch.setitem(sys.modules, "ccai.allegro_contact", fake_allegro_contact)
    monkeypatch.setitem(sys.modules, "ccai.utils.allegro_utils", fake_allegro_utils)
    monkeypatch.setitem(sys.modules, "ccai.controller.tactile_feedback_controller", fake_tactile)

    spec = importlib.util.spec_from_file_location(
        "recovery_utils_under_test", "ccai/utils/recovery_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fake_problem = FakeProblem()
    planner = module.ConstraintScheduledSVGDMPC(
        fake_problem,
        {"tactile_controller": True, "tactile_controller_backend": "qp"},
        mode="simulation",
    )

    assert isinstance(planner.tactile_controller, FakeQPController)
    assert planner.tactile_controller_backend == "qp"
