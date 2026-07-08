import sys
import types

import torch

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
allegro_utils = types.ModuleType("ccai.utils.allegro_utils")
allegro_utils.convert_yaw_to_sine_cosine = lambda x: x
allegro_utils.convert_sine_cosine_to_yaw = lambda x: x
allegro_utils.visualize_trajectory = lambda *args, **kwargs: None
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
from ccai.utils.screwdriver_yaw_wrap import (
    reset_screwdriver_yaw_wrap,
    unwrap_screwdriver_task_state_yaw,
    update_screwdriver_yaw_wrap_after_recovery,
    wrap_screwdriver_task_state_yaw,
)


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
        self.reset_belief_args = []
        self.plan_step_indices = []
        self.observe_step_indices = []

    def plan_next(self, env, step_idx):
        self.plan_calls += 1
        self.plan_step_indices.append(int(step_idx))
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
        self.observe_step_indices.append(int(kwargs["step_idx"]))

    def reset_after_recovery(self, env, *, reset_belief=True):
        self.reset_belief_args.append(reset_belief)
        self.reset_after_recovery_calls += 1


class _FakeSampler:
    def check_id(self, *args, **kwargs):
        return True, torch.tensor(0.0)


class _RecordingSampler:
    def __init__(self):
        self.states = []

    def check_id(self, state, *args, **kwargs):
        del args, kwargs
        self.states.append(torch.as_tensor(state).detach().clone())
        return True, torch.tensor(0.0)


class _SequenceSampler:
    def __init__(self, id_sequence):
        self.id_sequence = list(id_sequence)
        self.states = []
        self.calls = 0

    def check_id(self, state, *args, **kwargs):
        del args, kwargs
        self.states.append(torch.as_tensor(state).detach().clone())
        idx = min(self.calls, len(self.id_sequence) - 1)
        id_check = bool(self.id_sequence[idx])
        likelihood = torch.tensor(0.0 if id_check else -100.0)
        self.calls += 1
        return id_check, likelihood


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


class _ResetRecordingPlanner:
    def __init__(self, particle_count=4):
        self.N = particle_count
        self.problem = types.SimpleNamespace(
            T=0,
            obj_dof=3,
            obj_joint_dim=1,
            dx=15,
            du=12,
            goal=torch.zeros(3),
            data={},
        )
        self.warmed_up = True
        self.x = torch.zeros(particle_count, 1, 27)
        self.reset_calls = []

    def reset(self, start, initial_x=None, **kwargs):
        self.reset_calls.append(
            {
                "start": start.detach().clone(),
                "initial_x": None if initial_x is None else initial_x.detach().clone(),
                "kwargs": kwargs,
            }
        )
        self.warmed_up = False
        if initial_x is not None:
            self.x = initial_x.detach().clone()


class _FakeTrajectorySampler:
    T = 3

    def __init__(self, yaw):
        self.yaw = float(yaw)
        self.starts = []

    def sample(self, *, N, start, H, constraints):
        del H, constraints
        self.starts.append(torch.as_tensor(start).detach().clone())
        samples = torch.zeros(N, 3, 27)
        samples[..., 14] = self.yaw
        return samples, None, torch.arange(N, dtype=torch.float32)


def _data():
    return {
        "pre_action_likelihoods": [],
        "final_likelihoods": [],
        "csvto_times": [],
    }


def test_screwdriver_yaw_wrap_advances_only_after_recovery_update():
    params = {}
    initial = torch.zeros(15)
    initial[-1] = 1.0
    reset_screwdriver_yaw_wrap(params, initial)

    crossed = initial.clone()
    crossed[-1] = 1.0 - torch.pi / 2.0 - 0.2
    before_update = wrap_screwdriver_task_state_yaw(params, crossed)
    torch.testing.assert_close(before_update[-1], crossed[-1])

    update_screwdriver_yaw_wrap_after_recovery(params, crossed)
    after_update = wrap_screwdriver_task_state_yaw(params, crossed)
    torch.testing.assert_close(after_update[-1], torch.tensor(0.8))
    unwrapped = unwrap_screwdriver_task_state_yaw(params, after_update)
    torch.testing.assert_close(unwrapped[-1], crossed[-1])
    torch.testing.assert_close(crossed[-1], torch.tensor(1.0 - torch.pi / 2.0 - 0.2))


def test_likelihood_check_uses_wrapped_yaw_without_mutating_raw_state():
    env = _FakeEnv()
    sampler = _RecordingSampler()
    data = _data()
    data["pre_action_likelihoods"].append([])
    params = {
        "device": "cpu",
        "mode": "simulation",
        "live_recovery": True,
        "OOD_metric": "likelihood",
        "likelihood_num_samples": 1,
        "likelihood_threshold": -15,
    }
    initial = torch.zeros(15)
    initial[-1] = 1.0
    reset_screwdriver_yaw_wrap(params, initial)
    raw_state = initial.clone()
    raw_state[-1] = 1.0 - torch.pi / 2.0 - 0.2
    update_screwdriver_yaw_wrap_after_recovery(params, raw_state)

    TrajectoryExecutor(params, env)._check_exit_conditions(
        1,
        raw_state,
        None,
        sampler,
        None,
        False,
        data,
        [],
        [],
        None,
    )

    assert len(sampler.states) == 1
    torch.testing.assert_close(sampler.states[0][-1], torch.tensor(0.8))
    torch.testing.assert_close(raw_state[-1], torch.tensor(1.0 - torch.pi / 2.0 - 0.2))


def test_diffusion_initial_samples_unwrap_yaw_before_planner_use(tmp_path, monkeypatch):
    import ccai.execution.trial_executor as trial_executor_module

    monkeypatch.setattr(trial_executor_module, "convert_yaw_to_sine_cosine", lambda x: x)
    monkeypatch.setattr(trial_executor_module, "convert_sine_cosine_to_yaw", lambda x: x)
    params = {
        "device": "cpu",
        "diff_init": True,
        "sine_cosine": True,
        "N_contact_plan": 2,
        "N": 1,
        "T": 2,
        "T_orig": 2,
    }
    initial = torch.zeros(15)
    initial[-1] = 1.0
    reset_screwdriver_yaw_wrap(params, initial)
    raw_state = initial.clone()
    raw_state[-1] = 1.0 - torch.pi / 2.0 - 0.2
    update_screwdriver_yaw_wrap_after_recovery(params, raw_state)
    sampler = _FakeTrajectorySampler(yaw=0.8)

    initial_samples, _new_T, _sim_rollouts = TrajectoryExecutor(params, _FakeEnv())._handle_initial_sampling(
        mode="turn",
        trajectory_sampler=None,
        trajectory_sampler_orig=sampler,
        recover=False,
        skip_diff_init=False,
        initial_samples=None,
        state=raw_state,
        contact=torch.ones(2, 3),
        num_fingers=3,
        obj_dof=3,
        mode_fpath=tmp_path,
        planner=types.SimpleNamespace(problem=types.SimpleNamespace(dx=15, du=12)),
    )

    torch.testing.assert_close(sampler.starts[0].reshape(-1)[-1], torch.tensor(0.8))
    torch.testing.assert_close(initial_samples[0, 0, 14], torch.tensor(1.0 - torch.pi / 2.0 - 0.2))


def test_reused_recovery_planner_resets_with_diffusion_initial_samples(tmp_path):
    env = _FakeEnv()
    planner = _ResetRecordingPlanner(particle_count=4)
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
        "diff_init": True,
        "task_model_path": "task.pt",
        "N": 1,
        "recovery_N": 4,
        "N_contact_plan": 4,
        "T": 1,
        "T_orig": 1,
    }
    full_samples = torch.zeros(2, 2, 27)
    full_samples[0, 1, 0] = 10.0
    full_samples[1, 1, 0] = 20.0

    TrajectoryExecutor(params, env).execute_traj(
        planner=planner,
        mode="thumb_middle",
        env=env,
        goal=torch.zeros(3),
        fname="thumb_middle_regrasp",
        initial_samples=full_samples,
        recover=True,
        fpath=tmp_path,
        data=data,
        trajectory_sampler_orig=None,
        num_fingers=3,
        obj_dof=3,
        episode_num_steps=0,
        max_episode_num_steps=10,
    )

    assert len(planner.reset_calls) == 1
    initial_x = planner.reset_calls[0]["initial_x"]
    assert initial_x.shape == (4, 1, 27)
    torch.testing.assert_close(initial_x[:, 0, 0], torch.tensor([10.0, 20.0, 10.0, 20.0]))


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
    assert first_record["likelihood"] == 0.0
    torch.testing.assert_close(torch.as_tensor(first_record["actions"][0]), torch.ones(12) * 0.01)
    torch.testing.assert_close(torch.as_tensor(first_record["contact_plan"][0]), torch.ones(3))
    torch.testing.assert_close(torch.as_tensor(first_record["states"][0, :12]), torch.zeros(12))
    torch.testing.assert_close(torch.as_tensor(first_record["states"][1, :12]), torch.ones(12) * 0.01)


def test_normal_diffpf_policy_uses_global_episode_step_indices(tmp_path):
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

    *_unused, episode_steps = TrajectoryExecutor(params, env).execute_traj(
        planner=None,
        mode="turn",
        env=env,
        data=data,
        trajectory_sampler_orig=_FakeSampler(),
        num_fingers=3,
        obj_dof=3,
        episode_num_steps=5,
        max_episode_num_steps=10,
        normal_action_policy=policy,
        fpath=tmp_path,
    )

    assert policy.plan_step_indices == [5, 6]
    assert policy.observe_step_indices == [5, 6]
    assert episode_steps == 7


def test_diffpf_recovery_resets_once_and_exits_when_likelihood_returns_id():
    env = _FakeEnv()
    recovery_policy = _FakePolicy()
    sampler = _SequenceSampler([False, True])
    data = _data()
    params = {
        "device": "cpu",
        "mode": "simulation",
        "live_recovery": True,
        "OOD_metric": "likelihood",
        "likelihood_num_samples": 1,
        "likelihood_threshold": -15,
        "diffpf_execution_horizon": 10,
        "recovery_diffpf_execution_horizon": 4,
        "controller": "csvgd",
        "recovery_controller": "diffpf",
    }

    actual, planned, _initial, _sim_rollouts, *_middle, recover, episode_steps = TrajectoryExecutor(
        params,
        env,
    ).execute_traj(
        planner=None,
        mode="diffpf_recovery",
        env=env,
        data=data,
        trajectory_sampler_orig=sampler,
        num_fingers=3,
        obj_dof=3,
        episode_num_steps=0,
        max_episode_num_steps=10,
        recover=True,
        recovery_action_policy=recovery_policy,
    )

    assert recover is False
    assert episode_steps == 2
    assert recovery_policy.reset_after_recovery_calls == 1
    assert recovery_policy.reset_belief_args == [True]
    assert recovery_policy.plan_calls == 2
    assert recovery_policy.observe_calls == 2
    assert sampler.calls == 2
    assert len(env.step_calls) == 2
    assert actual.shape == (2, 27)
    assert [tuple(plan.shape) for plan in planned] == [(1, 1, 36), (1, 1, 36)]
    assert len(data["pre_action_likelihoods"][-1]) == 2
    assert len(data["recovery_policy_times"]) == 2
    assert len(data["contact_plan"]) == 2
    torch.testing.assert_close(data["contact_plan"][0], torch.ones(3))
    assert len(data["hri_diffpf_records"]) == 2
    assert data["hri_diffpf_records"][0]["recover"] is True


def test_diffpf_recovery_uses_recovery_execution_horizon_without_per_action_reset():
    env = _FakeEnv()
    recovery_policy = _FakePolicy()
    sampler = _SequenceSampler([False, False, False])
    data = _data()
    params = {
        "device": "cpu",
        "mode": "simulation",
        "live_recovery": True,
        "OOD_metric": "likelihood",
        "likelihood_num_samples": 1,
        "likelihood_threshold": -15,
        "diffpf_execution_horizon": 10,
        "recovery_diffpf_execution_horizon": 2,
        "controller": "csvgd",
        "recovery_controller": "diffpf",
    }

    *_prefix, recover, episode_steps = TrajectoryExecutor(params, env).execute_traj(
        planner=None,
        mode="diffpf_recovery",
        env=env,
        data=data,
        trajectory_sampler_orig=sampler,
        num_fingers=3,
        obj_dof=3,
        episode_num_steps=0,
        max_episode_num_steps=10,
        recover=True,
        recovery_action_policy=recovery_policy,
    )

    assert recover is True
    assert episode_steps == 2
    assert recovery_policy.plan_calls == 2
    assert recovery_policy.observe_calls == 2
    assert recovery_policy.reset_after_recovery_calls == 1
    assert recovery_policy.reset_belief_args == [True]
    assert sampler.calls == 2


def test_continued_diffpf_recovery_uses_monotonic_policy_step_indices():
    env = _FakeEnv()
    recovery_policy = _FakePolicy()
    sampler = _SequenceSampler([False, False, True])
    data = _data()
    params = {
        "device": "cpu",
        "mode": "simulation",
        "live_recovery": True,
        "OOD_metric": "likelihood",
        "likelihood_num_samples": 1,
        "likelihood_threshold": -15,
        "diffpf_execution_horizon": 10,
        "recovery_diffpf_execution_horizon": 2,
        "controller": "csvgd",
        "recovery_controller": "diffpf",
    }
    executor = TrajectoryExecutor(params, env)

    *_prefix, recover, episode_steps = executor.execute_traj(
        planner=None,
        mode="diffpf_recovery",
        env=env,
        data=data,
        trajectory_sampler_orig=sampler,
        num_fingers=3,
        obj_dof=3,
        episode_num_steps=0,
        max_episode_num_steps=10,
        recover=True,
        recovery_action_policy=recovery_policy,
    )
    assert recover is True
    assert episode_steps == 2

    *_prefix, recover, episode_steps = executor.execute_traj(
        planner=None,
        mode="diffpf_recovery",
        env=env,
        data=data,
        trajectory_sampler_orig=sampler,
        num_fingers=3,
        obj_dof=3,
        episode_num_steps=episode_steps,
        max_episode_num_steps=10,
        recover=True,
        recovery_action_policy=recovery_policy,
        reset_recovery_policy=False,
    )

    assert recover is False
    assert episode_steps == 3
    assert recovery_policy.reset_after_recovery_calls == 1
    assert recovery_policy.plan_step_indices == [0, 1, 2]
    assert recovery_policy.observe_step_indices == [0, 1, 2]


def test_hri_diffpf_record_saves_optional_full_joint_and_wrist_fields():
    env = _FakeEnv()
    env.hand_spec = types.SimpleNamespace(
        all_joint_names=tuple(f"joint_{idx}" for idx in range(18)),
        wrist_joint_names=("joint_0", "joint_1"),
    )
    executor = TrajectoryExecutor({"device": "cpu", "trial_index": 2, "current_stage": 3}, env)
    tactile = {
        "contact_state": torch.zeros(3),
        "contact_wrenches": torch.zeros(3, 6),
        "contact_forces": torch.zeros(3, 3),
        "contact_points": torch.zeros(3, 3),
    }
    data = {}

    executor._append_hri_diffpf_record(
        data,
        pre_state15=torch.zeros(15),
        post_state15=torch.ones(15),
        delta12=torch.ones(12) * 0.1,
        contact_plan=torch.ones(3),
        pre_tactile=tactile,
        post_tactile=tactile,
        pre_full_dof_reference=torch.arange(18, dtype=torch.float32),
        post_full_dof_reference=torch.arange(18, dtype=torch.float32) + 100.0,
        mode="turn",
        recover=False,
        episode_num_steps=0,
    )

    record = data["hri_diffpf_records"][0]
    assert record["full_joint_pos"].shape == (2, 18)
    assert record["wrist_joint_pos"].shape == (2, 2)
    assert record["full_joint_names"] == env.hand_spec.all_joint_names
    torch.testing.assert_close(torch.as_tensor(record["wrist_joint_pos"][1]), torch.tensor([100.0, 101.0]))


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
    assert policy.reset_belief_args == [True]


def test_recovery_branch_passes_no_reset_config_to_normal_policy():
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
        "diffpf_reset_belief_after_recovery": False,
        "T": 0,
        "T_orig": 0,
    }

    TrajectoryExecutor(params, env).execute_traj(
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

    assert policy.reset_after_recovery_calls == 1
    assert policy.reset_belief_args == [False]


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
