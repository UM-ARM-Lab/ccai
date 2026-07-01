import sys
import types

import torch

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
allegro_utils = types.ModuleType("ccai.utils.allegro_utils")
allegro_utils.convert_yaw_to_sine_cosine = lambda x: x
allegro_utils.convert_sine_cosine_to_yaw = lambda x: x
sys.modules["ccai.utils.allegro_utils"] = allegro_utils

recovery_utils = types.ModuleType("ccai.utils.recovery_utils")
recovery_utils.create_experiment_paths = lambda fpath, fname, mode=None, create_goal_subdir=True: {
    "mode_fpath": fpath,
    "goal_fpath": fpath,
}
recovery_utils.save_goal_info = lambda *args, **kwargs: None
recovery_utils.save_projection_results = lambda *args, **kwargs: None
recovery_utils.partial_to_full_trajectory = lambda *args, **kwargs: args[0]
recovery_utils.full_to_partial_trajectory = lambda *args, **kwargs: args[0]
recovery_utils.setup_and_visualize_trajectory = lambda *args, **kwargs: None
recovery_utils.get_screwdriver_plan_camera_path = lambda *args, **kwargs: None
sys.modules["ccai.utils.recovery_utils"] = recovery_utils

trajectory_shortcut = types.ModuleType("ccai.trajectory_shortcut")
trajectory_shortcut.shortcut_trajectory = lambda x, *args, **kwargs: x
sys.modules["ccai.trajectory_shortcut"] = trajectory_shortcut

baselines = types.ModuleType("ccai.baselines.allegro_recovery_baselines")
baselines.should_skip_diff_init = lambda *args, **kwargs: False
sys.modules["ccai.baselines.allegro_recovery_baselines"] = baselines

from ccai.execution.trial_executor import TrajectoryExecutor


class _FakeEnv:
    device = torch.device("cpu")
    external_wrench_perturb = False

    def __init__(self):
        self.q = torch.zeros(1, 16)
        self.step_calls = []

    def get_state(self):
        return {"q": self.q.clone()}

    def step(self, action):
        action = torch.as_tensor(action, dtype=torch.float32).reshape(1, -1)
        self.step_calls.append(action.clone())
        self.q[:, :12] = action[:, :12]
        self.q[:, 14] -= 0.1

    def zero_obj_velocity(self):
        pass

    def get_tactile_observation(self):
        value = float(len(self.step_calls))
        return {
            "contact_state": torch.tensor([[value > 0, False, True]], dtype=torch.float32),
            "contact_wrenches": torch.ones(1, 3, 6) * value,
            "contact_forces": torch.ones(1, 3, 3) * value,
            "contact_points": torch.ones(1, 3, 3) * (value + 0.5),
        }


class _FakePolicy:
    def __init__(self):
        self.plan_calls = 0
        self.observe_calls = 0
        self.reset_after_recovery_calls = 0

    def plan_next(self, env, step_idx):
        self.plan_calls += 1
        delta = torch.ones(12) * 0.01 * (step_idx + 1)
        state = env.get_state()["q"][0, :15]
        return {
            "delta_action": delta,
            "absolute_action_target": state[:12] + delta,
            "selected_plan_rows": torch.cat((state, delta, torch.zeros(9))),
            "likelihood_stats": {"ok": True},
        }

    def observe_transition(self, **kwargs):
        self.observe_calls += 1

    def reset_after_recovery(self, env):
        self.reset_after_recovery_calls += 1


class _FakeSampler:
    def check_id(self, *args, **kwargs):
        return True, torch.tensor(0.0)


class _FakePlanner:
    def __init__(self):
        self.problem = types.SimpleNamespace(
            T=0,
            obj_dof=3,
            obj_joint_dim=1,
            dx=15,
            goal=torch.zeros(3),
            data={},
        )
        self.warmed_up = True
        self.x = torch.zeros(1, 16)


def _data():
    return {
        "pre_action_likelihoods": [],
        "final_likelihoods": [],
        "csvto_times": [],
    }


def test_proto5_normal_policy_branch_logs_contact_timeseries_and_rows():
    env = _FakeEnv()
    policy = _FakePolicy()
    data = _data()
    params = {
        "device": "cpu",
        "mode": "simulation",
        "live_recovery": True,
        "OOD_metric": "likelihood",
        "likelihood_num_samples": 1,
        "likelihood_threshold": -15,
        "diffpf_execution_horizon": 2,
        "controller": "csvgd",
        "recovery_controller": "csvgd",
    }

    actual, planned, *_rest = TrajectoryExecutor(params, env).execute_traj(
        planner=None,
        mode="turn",
        env=env,
        data=data,
        trajectory_sampler_orig=_FakeSampler(),
        num_fingers=3,
        obj_dof=3,
        episode_num_steps=0,
        max_episode_num_steps=10,
        normal_action_policy=policy,
    )

    assert policy.plan_calls == 2
    assert policy.observe_calls == 2
    assert policy.reset_after_recovery_calls == 0
    assert len(env.step_calls) == 2
    assert actual.shape == (2, 27)
    assert [tuple(plan.shape) for plan in planned] == [(1, 1, 36), (1, 1, 36)]
    assert len(data["contact_state"]) == 3
    assert len(data["contact_wrenches"]) == 3
    assert len(data["contact_forces"]) == 3
    assert len(data["contact_points"]) == 3
    assert len(data["contact_plan"]) == 2
    torch.testing.assert_close(data["contact_plan"][0], torch.ones(3))
    assert len(data["hri_diffpf_records"]) == 2
    first_record = data["hri_diffpf_records"][0]
    assert first_record["actions"].shape == (1, 12)
    assert first_record["contact_plan"].shape == (1, 3)
    torch.testing.assert_close(torch.as_tensor(first_record["actions"][0]), torch.ones(12) * 0.01)
    torch.testing.assert_close(torch.as_tensor(first_record["contact_plan"][0]), torch.ones(3))
    torch.testing.assert_close(torch.as_tensor(first_record["states"][0, :12]), torch.zeros(12))
    torch.testing.assert_close(torch.as_tensor(first_record["states"][1, :12]), torch.ones(12) * 0.01)


def test_recovery_branch_resets_normal_policy_without_observing_recovery_transition():
    env = _FakeEnv()
    policy = _FakePolicy()
    data = _data()
    params = {
        "device": "cpu",
        "mode": "simulation",
        "live_recovery": True,
        "OOD_metric": "likelihood",
        "likelihood_num_samples": 1,
        "likelihood_threshold": -15,
        "controller": "csvgd",
        "recovery_controller": "csvgd",
        "visualize_plan": False,
        "visualize_recovery_plan": False,
        "T": 0,
        "T_orig": 0,
    }

    actual, planned, *_rest = TrajectoryExecutor(params, env).execute_traj(
        planner=_FakePlanner(),
        mode="index",
        env=env,
        data=data,
        trajectory_sampler_orig=None,
        num_fingers=3,
        obj_dof=3,
        episode_num_steps=0,
        max_episode_num_steps=10,
        normal_action_policy=policy,
        recover=True,
    )

    assert actual == []
    assert planned == []
    assert policy.plan_calls == 0
    assert policy.observe_calls == 0
    assert policy.reset_after_recovery_calls == 1


def test_turn_recovery_segment_does_not_reset_normal_policy_belief():
    env = _FakeEnv()
    policy = _FakePolicy()
    data = _data()
    params = {
        "device": "cpu",
        "mode": "simulation",
        "live_recovery": True,
        "OOD_metric": "likelihood",
        "likelihood_num_samples": 1,
        "likelihood_threshold": -15,
        "controller": "csvgd",
        "recovery_controller": "csvgd",
        "visualize_plan": False,
        "visualize_recovery_plan": False,
        "T": 0,
        "T_orig": 0,
    }

    actual, planned, *_rest = TrajectoryExecutor(params, env).execute_traj(
        planner=_FakePlanner(),
        mode="turn",
        env=env,
        data=data,
        trajectory_sampler_orig=None,
        num_fingers=3,
        obj_dof=3,
        episode_num_steps=0,
        max_episode_num_steps=10,
        normal_action_policy=policy,
        recover=True,
    )

    assert actual == []
    assert planned == []
    assert policy.plan_calls == 0
    assert policy.observe_calls == 0
    assert policy.reset_after_recovery_calls == 0
