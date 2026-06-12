"""
ROS-free GRAMPC motion-contact tracker for Allegro simulation.

This ports the low-level SoftContactV3 controller structure from the
Director-of-G/in_hand_manipulation_2 reference while adapting kinematics and
reference conversion to the local AllegroManipulationProblem interfaces.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from scipy.interpolate import make_interp_spline
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation

FULL_FINGER_LIST = ["index", "middle", "ring", "thumb"]

try:
    from pygrampc import Grampc, ProblemDescription
except ImportError as exc:  # pragma: no cover - exercised via helper tests.
    Grampc = None
    ProblemDescription = object
    _PYGRAMPC_IMPORT_ERROR = exc
else:
    _PYGRAMPC_IMPORT_ERROR = None


PYGRAMPC_INSTALL_INSTRUCTION = (
    "pygrampc is required for tactile_controller_backend='grampc'. "
    "Install it with: python -m pip install pybind11 && "
    "python -m pip install 'git+https://github.com/grampc/pygrampc'"
)


@dataclass
class Pose6D:
    translation: np.ndarray
    rotation: np.ndarray


def ensure_pygrampc_available():
    if _PYGRAMPC_IMPORT_ERROR is not None:
        raise ImportError(PYGRAMPC_INSTALL_INSTRUCTION) from _PYGRAMPC_IMPORT_ERROR


def compute_pose_error(t1, r1, t2, r2):
    p_e = t2 - t1
    r_e = Rotation.from_matrix(np.dot(r2, r1.T)).as_rotvec()
    return np.concatenate([p_e, r_e])


def cross_product_matrix(v):
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def _as_numpy(value, dtype=float):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


class SoftContactV3(ProblemDescription):
    """
    GRAMPC problem for stacked soft-contact dynamics.

    State: [q, q_d, f_ext], with q in local partial Allegro finger order and
    f_ext stacked by contact_fingers.
    """

    def __init__(
        self,
        Qq,
        Qp,
        Qf,
        R,
        Kd,
        Kp,
        jac_func: Callable,
        k_func: Callable,
        fk_func: Callable,
        nc=1,
        Nhor=10,
        enable_coupling=False,
    ):
        ensure_pygrampc_available()
        super().__init__()
        nq = Kp.shape[0]
        self.nq = nq
        self.nc = nc

        self.Nx = 2 * nq + 3 * nc
        self.Nu = nq
        self.Np = 0
        self.Ng = 0
        self.Nh = 0
        self.NgT = 0
        self.NhT = 0
        self.Nhor = Nhor
        self.enable_coupling = enable_coupling

        self.Qq = Qq.copy()
        self.Qp = [q.copy() for q in Qp]
        self.Qf = Qf.copy()
        self.R = R.copy()
        self.Gmat = np.zeros((6, 3 * nc))

        self.Kdinv = np.linalg.inv(Kd)
        self.Kp = Kp.copy()

        self.jac_func = jac_func
        self.k_func = k_func
        self.fk_func = fk_func

        self.xdes_spline_stored = None
        self.J_stored = {}
        self.K_stored = {}
        self.K_coup_stored = {}
        self.pdes_stored = {}

    def reset(self):
        self.xdes_spline_stored = None
        self.J_stored.clear()
        self.K_stored.clear()
        self.K_coup_stored.clear()
        self.pdes_stored.clear()

    def initialize_xdes_spline(self, t_knots, xdes_knots):
        self.xdes_spline_stored = make_interp_spline(t_knots, xdes_knots, k=1)

    def get_xdes(self, t):
        return self.xdes_spline_stored(t)

    def get_jacobian(self, t, q, trans_only=False):
        t_key = round(float(t), 4)
        if t_key not in self.J_stored:
            self.J_stored[t_key] = self.jac_func(q)
        J = self.J_stored[t_key]
        if trans_only:
            return [jac[:3] for jac in J]
        return J

    def get_stiffness(self, t, J):
        t_key = round(float(t), 4)
        if t_key not in self.K_stored:
            self.K_stored[t_key] = self.k_func(J)
        return self.K_stored[t_key]

    def get_K_coup(self, t, K_bar):
        t_key = round(float(t), 4)
        if t_key in self.K_coup_stored:
            return self.K_coup_stored[t_key]

        GKGT = self.Gmat[[0, 4, 5]] @ K_bar @ self.Gmat[[0, 4, 5]].T
        if np.linalg.matrix_rank(GKGT) < 3 or not self.enable_coupling:
            K_coup = K_bar
        else:
            K_coup = (
                K_bar
                + K_bar
                @ self.Gmat[[0, 4, 5]].T
                @ np.linalg.inv(GKGT)
                @ self.Gmat[[0, 4, 5]]
                @ K_bar
            )
        self.K_coup_stored[t_key] = K_coup
        return K_coup

    def get_pdes(self, t, q):
        t_key = round(float(t), 4)
        if t_key not in self.pdes_stored:
            self.pdes_stored[t_key] = self.fk_func(q)
        return self.pdes_stored[t_key]

    def set_Qf(self, Qf):
        self.Qf[:] = Qf.copy()

    def set_Qq(self, Qq):
        self.Qq[:] = Qq.copy()

    def set_Qp(self, Qp):
        self.Qp = [q.copy() for q in Qp]

    def set_R(self, R):
        self.R[:] = R.copy()

    def set_grasping_matrix(self, Gmat):
        self.Gmat[:] = Gmat.copy()

    def get_Q(self):
        return self.Qq.copy(), [q.copy() for q in self.Qp], self.Qf.copy()

    def get_R(self):
        return self.R.copy()

    def ffct(self, out, t, x, u, p):
        nq = self.nq
        q, qd, fext = x[:nq], x[nq : 2 * nq], x[2 * nq :]

        J = self.get_jacobian(t, q, trans_only=True)
        J_vcat = np.vstack(J)
        K = self.get_stiffness(t, J)
        K_bar = block_diag(*K)
        K_coup = self.get_K_coup(t, K_bar)

        out[:nq] = u + self.Kdinv @ (self.Kp @ (qd - q) - J_vcat.T @ fext)
        out[nq : 2 * nq] = u
        out[2 * nq :] = K_coup @ J_vcat @ u

    def dfdx_vec(self, out, t, x, vec, u, p):
        nq = self.nq
        q = x[:nq]
        J = self.get_jacobian(t, q, trans_only=True)
        J_vcat = np.vstack(J)
        out[:] = np.block(
            [-self.Kdinv @ self.Kp, self.Kdinv @ self.Kp, -self.Kdinv @ J_vcat.T]
        ).T @ vec[:nq]

    def dfdu_vec(self, out, t, x, vec, u, p):
        nq = self.nq
        q = x[:nq]
        J = self.get_jacobian(t, q, trans_only=True)
        J_vcat = np.vstack(J)
        K = self.get_stiffness(t, J)
        K_bar = block_diag(*K)
        K_coup = self.get_K_coup(t, K_bar)
        out[:] = np.block([[np.eye(nq)], [np.eye(nq)], [K_coup @ J_vcat]]).T @ vec

    def lfct(self, out, t, x, u, p, xdes, udes):
        nq = self.nq
        xdes_itp = self.get_xdes(t)
        qdes = xdes_itp[nq : 2 * nq]
        pdes = self.get_pdes(t, qdes)
        fdes = xdes_itp[2 * nq :]

        qd, fext = x[nq : 2 * nq], x[2 * nq :]
        pd = self.fk_func(qd)

        Qq, Qp, Qf = self.get_Q()
        R = self.get_R()
        out[0] = (
            (qd - qdes).T @ Qq @ (qd - qdes)
            + (fext - fdes).T @ Qf @ (fext - fdes)
            + u.T @ R @ u
        )

        for i in range(self.nc):
            pe = compute_pose_error(
                t1=pdes[i].translation,
                r1=pdes[i].rotation,
                t2=pd[i].translation,
                r2=pd[i].rotation,
            )
            out[0] += pe.T @ Qp[i] @ pe

    def dldx(self, out, t, x, u, p, xdes, udes):
        nq = self.nq
        xdes_itp = self.get_xdes(t)
        qdes = xdes_itp[nq : 2 * nq]
        pdes = self.get_pdes(t, qdes)
        fdes = xdes_itp[2 * nq :]

        qd, fext = x[nq : 2 * nq], x[2 * nq :]
        pd = self.fk_func(qd)
        Qq, Qp, Qf = self.get_Q()

        out[:] = 0.0
        out[nq : 2 * nq] = 2 * Qq @ (qd - qdes)
        out[2 * nq :] = 2 * Qf @ (fext - fdes)
        J = self.get_jacobian(t, qd)
        for i in range(self.nc):
            pe = compute_pose_error(
                t1=pdes[i].translation,
                r1=pdes[i].rotation,
                t2=pd[i].translation,
                r2=pd[i].rotation,
            )
            out[nq : 2 * nq] += 2 * (np.dot(Qp[i], pe)).T @ J[i]

    def dldu(self, out, t, x, u, p, xdes, udes):
        out[:] = 2 * self.get_R() @ u


class GRAMPCMotionContactTracker:
    def __init__(self, problem, config, params: Optional[dict] = None):
        ensure_pygrampc_available()
        params = params or {}
        self.problem_owner = problem
        self.config = config
        self.fingers = list(problem.fingers)
        self.contact_fingers = list(getattr(problem, "contact_fingers", self.fingers))
        self.nq = 4 * len(self.fingers)
        self.nc = len(self.contact_fingers)
        self.df = 3 * self.nc
        self.Nx = 2 * self.nq + self.df
        self.Nu = self.nq

        self.dt = float(getattr(config, "dt", params.get("dt", 1.0 / 12.0)))
        self.mpc_horizon = int(getattr(config, "horizon_length", params.get("horizon_length", 2)))
        self.force_lower_bound = float(
            getattr(config, "force_threshold", params.get("force_threshold", 0.2))
        )
        self.force_upper_bound = float(params.get("force_upper_bound", 10.0))
        self.dq_scale = float(params.get("dq_scale", 1.0))
        self.enable_coupling = bool(params.get("enable_coupling", False))

        self.Kp = float(getattr(config, "K_P", params.get("K_P", 3.0))) * np.eye(self.nq)
        self.Kd = float(getattr(config, "K_D", params.get("K_D", 1.0))) * np.eye(self.nq)
        self.Ke_scalar = float(getattr(config, "K_e", params.get("K_e", 200.0)))

        self.Qq = float(getattr(config, "w_q", params.get("w_q", 20.0))) * np.eye(self.nq)
        self.Qf_scalar = float(getattr(config, "w_f", params.get("w_f", 1.0)))
        self.Qp_scalar = float(getattr(config, "w_p", params.get("w_p", 1.0)))
        self.Qp_ori_scalar = float(getattr(config, "w_ori", params.get("w_ori", 0.1)))
        self.R = float(getattr(config, "w_u", params.get("w_u", 1.0))) * np.eye(self.nq)

        self.Qf = np.zeros((self.df, self.df))
        self.Qp = [np.zeros((6, 6)) for _ in range(self.nc)]
        self.Ke = [np.zeros((3, 3)) for _ in range(self.nc)]
        self.object_normals = np.zeros((self.mpc_horizon, self.nc, 3))
        self.object_contact_points = np.zeros((self.nc, 3))
        self.object_position = np.zeros(3)
        self.reference_spline = None

        self._all_joint_index = sum([problem.joint_index[f] for f in self.fingers], [])
        self._finger_partial_indices = {
            finger: list(range(4 * i, 4 * i + 4)) for i, finger in enumerate(self.fingers)
        }
        self._contact_force_rows = self._build_contact_force_rows()
        self._options_path = write_soft_contact_options(self.Nx, self.dt, self.mpc_horizon)

        soft_problem = SoftContactV3(
            self.Qq,
            self.Qp,
            self.Qf,
            self.R,
            self.Kd,
            self.Kp,
            self.compute_jacobian,
            self.compute_stiffness,
            self.compute_fk,
            self.nc,
            self.mpc_horizon,
            self.enable_coupling,
        )
        self.problem = soft_problem
        self.solver = Grampc(soft_problem, self._options_path, plot_prediction=False)
        self.u0 = np.zeros(self.nq)

    def _build_contact_force_rows(self):
        rows = []
        if not self.contact_fingers:
            return rows
        if all(finger in self.fingers for finger in self.contact_fingers):
            for finger in self.contact_fingers:
                idx = self.fingers.index(finger)
                rows.append(slice(3 * idx, 3 * idx + 3))
        else:
            for idx in range(self.nc):
                rows.append(slice(3 * idx, 3 * idx + 3))
        return rows

    def set_reference_trajectory(self, reference_spline_or_knots, normals=None, contact_points=None):
        self.reference_spline = reference_spline_or_knots
        normals_np = _as_numpy(normals)
        if normals_np is not None:
            normals_np = self._select_contact_rows(normals_np, row_width=3)
            normals_np = normals_np.reshape(self.nc, 3)
            norm = np.linalg.norm(normals_np, axis=1, keepdims=True)
            self.object_normals[:] = normals_np[None, :, :] / np.maximum(norm, 1e-9)
        if contact_points is not None:
            points_np = self._select_contact_rows(_as_numpy(contact_points), row_width=3)
            self.object_contact_points[:] = points_np.reshape(self.nc, 3)

    def solve(self, t0, state_q, q_d_init, f_ext_init, *_, **__):
        if self.reference_spline is None:
            raise RuntimeError("GRAMPCMotionContactTracker reference trajectory is not set.")

        q = self._first_n(_as_numpy(state_q), self.nq)
        qd = self._first_n(_as_numpy(q_d_init), self.nq)
        fext = self._reshape_force(f_ext_init)
        x0 = np.concatenate([q, qd, fext])

        xdes = self._reference_knots()
        self.calc_HFMC_matrices(xdes[0, 2 * self.nq :])
        self.problem.set_Qq(self.Qq)
        self.problem.set_Qp(self.Qp)
        self.problem.set_Qf(self.Qf)
        self.problem.set_R(self.R)
        self.problem.set_grasping_matrix(self.compute_grasping_matrix())
        self.problem.reset()
        self.problem.initialize_xdes_spline(self.dt * np.arange(self.mpc_horizon), xdes)

        self.solver.set_param({"x0": x0, "u0": self.u0, "t0": 0.0})
        self.solver.run()
        u0 = np.asarray(self.solver.sol.unext, dtype=float).reshape(self.nq)
        if hasattr(self.solver, "rws") and getattr(self.solver.rws, "u", None) is not None:
            try:
                self.u0 = np.asarray(self.solver.rws.u[:, 1], dtype=float).reshape(self.nq)
            except Exception:
                self.u0 = u0.copy()
        else:
            self.u0 = u0.copy()

        delta_q = self.dq_scale * u0 * self.dt
        if not np.all(np.isfinite(delta_q)):
            raise RuntimeError(f"GRAMPC returned non-finite delta_q: {delta_q}")
        return delta_q

    def _reference_knots(self):
        knots = np.zeros((self.mpc_horizon, self.Nx))
        for i in range(self.mpc_horizon):
            ref_state = _as_numpy(self.reference_spline(i * self.dt)).reshape(-1)
            q_ref = self._first_n(ref_state, self.nq)
            qd_offset = getattr(self.problem_owner, "dx", self.nq) + 0
            qd_ref = self._slice_or_default(ref_state, qd_offset, self.nq, q_ref)
            f_offset = qd_offset + self.nq
            f_ref = self._slice_or_default(ref_state, f_offset, self.df, np.zeros(self.df))
            f_ref = self._threshold_force(f_ref.reshape(self.nc, 3), self.force_upper_bound).reshape(-1)
            knots[i] = np.concatenate([q_ref, qd_ref, f_ref])
        return knots

    def _slice_or_default(self, values, start, length, default):
        if values.shape[0] >= start + length:
            return values[start : start + length]
        return np.asarray(default, dtype=float).reshape(length)

    def _first_n(self, values, n):
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape[0] < n:
            raise ValueError(f"Expected at least {n} values, got {values.shape[0]}.")
        return values[:n]

    def _select_contact_rows(self, values, row_width):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, row_width)
        if values.shape[0] == self.nc:
            return values
        if values.shape[0] == len(self.fingers):
            return np.stack([values[self.fingers.index(finger)] for finger in self.contact_fingers])
        if values.shape[0] == len(FULL_FINGER_LIST):
            return np.stack([values[FULL_FINGER_LIST.index(finger)] for finger in self.contact_fingers])
        return values[: self.nc]

    def _reshape_force(self, force):
        force_np = _as_numpy(force)
        if force_np is None:
            return np.zeros(self.df)
        selected = self._select_contact_rows(force_np, row_width=3)
        return selected.reshape(self.df)

    def _threshold_force(self, force, threshold):
        magnitude = np.linalg.norm(force, axis=1, keepdims=True)
        direction = np.zeros_like(force)
        active = magnitude[:, 0] > 1e-5
        direction[active] = force[active] / magnitude[active]
        return np.minimum(magnitude, threshold) * direction

    def calc_HFMC_matrices(self, desired_force):
        desired_force = desired_force.reshape(self.nc, 3)
        in_contact = np.linalg.norm(desired_force, axis=1) > self.force_lower_bound
        avg_normal = np.mean(self.object_normals, axis=0)
        self.Qf.fill(0.0)
        for i in range(self.nc):
            Qp_i = np.zeros((6, 6))
            if not in_contact[i] or np.linalg.norm(avg_normal[i]) < 1e-5:
                Qf_i = np.zeros((3, 3))
                Qp_trans = np.eye(3)
            else:
                n_i = avg_normal[i] / np.linalg.norm(avg_normal[i])
                Qf_i = np.outer(n_i, n_i)
                Qp_trans = np.eye(3) - Qf_i
            Qp_i[:3, :3] = self.Qp_scalar * Qp_trans
            Qp_i[3:, 3:] = self.Qp_ori_scalar * np.eye(3)
            self.Qp[i] = Qp_i
            self.Qf[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)] = self.Qf_scalar * Qf_i
            self.Ke[i] = self.Ke_scalar * Qf_i

    def compute_jacobian(self, q=None):
        q = np.zeros(self.nq) if q is None else np.asarray(q, dtype=float).reshape(self.nq)
        chain = self.problem_owner.contact_scenes.robot_sdf.chain
        ee_names = [self.problem_owner.ee_names[finger] for finger in self.contact_fingers]
        if hasattr(chain, "jacobian"):
            q_full = self._q_full_tensor(q).repeat(len(ee_names), 1)
            frame_indices = [
                self.problem_owner.contact_scenes.robot_sdf.chain.frame_to_idx[ee_name]
                for ee_name in ee_names
            ]
            link_indices = torch.tensor(frame_indices, device=self.problem_owner.device).long()
            J_full = chain.jacobian(q_full, link_indices=link_indices)
            J_np = _as_numpy(J_full)
            return [J_np[i, :, self._all_joint_index] for i in range(len(ee_names))]
        return self._finite_difference_jacobian(q)

    def _finite_difference_jacobian(self, q, eps=1e-5):
        base = self.compute_fk(q)
        out = []
        for i in range(self.nc):
            J_i = np.zeros((6, self.nq))
            for j in range(self.nq):
                q_eps = q.copy()
                q_eps[j] += eps
                pose_eps = self.compute_fk(q_eps)[i]
                pe = compute_pose_error(
                    base[i].translation,
                    base[i].rotation,
                    pose_eps.translation,
                    pose_eps.rotation,
                )
                J_i[:, j] = pe / eps
            out.append(J_i)
        return out

    def compute_stiffness(self, J_all):
        Kp_inv = np.linalg.inv(self.Kp)
        result = []
        for Ke, J in zip(self.Ke, J_all):
            K_cart_inv = J @ Kp_inv @ J.T
            result.append(np.linalg.inv(np.eye(3) + Ke @ K_cart_inv) @ Ke)
        return result

    def compute_fk(self, q=None):
        q = np.zeros(self.nq) if q is None else np.asarray(q, dtype=float).reshape(self.nq)
        q_full = self._q_full_tensor(q)
        chain = self.problem_owner.contact_scenes.robot_sdf.chain
        fk_result = chain.forward_kinematics(q_full)
        poses = []
        for finger in self.contact_fingers:
            matrix = fk_result[self.problem_owner.ee_names[finger]].get_matrix()
            matrix_np = _as_numpy(matrix)[0]
            poses.append(Pose6D(matrix_np[:3, 3].copy(), matrix_np[:3, :3].copy()))
        return poses

    def _q_full_tensor(self, q):
        q_tensor = torch.tensor(q.reshape(1, -1), device=self.problem_owner.device).float()
        partial_fingers = torch.chunk(q_tensor, chunks=len(self.fingers), dim=-1)
        partial_dict = dict(zip(self.fingers, partial_fingers))
        full = []
        for finger in FULL_FINGER_LIST:
            if finger in partial_dict:
                full.append(partial_dict[finger])
            else:
                full.append(torch.zeros_like(partial_fingers[0]))
        return torch.cat(full, dim=-1)

    def compute_grasping_matrix(self):
        Gmat = np.zeros((6, self.df))
        for i in range(self.nc):
            Gmat[0:3, 3 * i : 3 * (i + 1)] = np.eye(3)
            rel_contact_p = self.object_contact_points[i] - self.object_position
            Gmat[3:, 3 * i : 3 * (i + 1)] = cross_product_matrix(rel_contact_p)
        return Gmat


def write_soft_contact_options(nx, dt, horizon_length, path=None):
    options = {
        "Parameters": {
            "x0": [0.0] * nx,
            "xdes": [0.0] * nx,
            "Thor": float(dt) * float(horizon_length),
            "dt": float(dt),
            "t0": 0.0,
        },
        "Options": {
            "Nhor": int(horizon_length),
            "TerminalCost": "off",
        },
    }
    if path is None:
        path = os.path.join(
            tempfile.gettempdir(), f"ccai_soft_contact_grampc_Nx{nx}_Nhor{horizon_length}.json"
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(options, f, indent=2)
    return path
