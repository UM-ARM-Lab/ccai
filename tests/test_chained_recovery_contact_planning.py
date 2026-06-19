import sys
import types

import pytest
import torch

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
pk_module = types.ModuleType("pytorch_kinematics")
pk_module.transforms = types.ModuleType("pytorch_kinematics.transforms")
sys.modules.setdefault("pytorch_kinematics", pk_module)
sys.modules.setdefault("pytorch_kinematics.transforms", pk_module.transforms)
recovery_utils_module = types.ModuleType("ccai.utils.recovery_utils")


def _get_contact_state_mappings():
    contact_label_to_vec = {
        "pregrasp": 0,
        "thumb_middle": 1,
        "index": 2,
        "turn": 3,
        "thumb": 4,
        "middle": 5,
    }
    contact_vec_to_label = dict((v, k) for k, v in contact_label_to_vec.items())
    contact_state_dict = {
        "all": torch.tensor([0.0, 0.0, 0.0]),
        "index": torch.tensor([0.0, 1.0, 1.0]),
        "thumb_middle": torch.tensor([1.0, 0.0, 0.0]),
        "turn": torch.tensor([1.0, 1.0, 1.0]),
        "thumb": torch.tensor([1.0, 1.0, 0.0]),
        "middle": torch.tensor([1.0, 0.0, 1.0]),
    }
    contact_state_dict_flip = dict([(tuple(v.numpy()), k) for k, v in contact_state_dict.items()])
    return contact_label_to_vec, contact_vec_to_label, contact_state_dict, contact_state_dict_flip


recovery_utils_module.get_contact_state_mappings = _get_contact_state_mappings
recovery_utils_module.create_visualization_paths = lambda *args, **kwargs: None
recovery_utils_module.save_goal_info = lambda *args, **kwargs: None
recovery_utils_module.save_recovery_info = lambda *args, **kwargs: None
sys.modules["ccai.utils.recovery_utils"] = recovery_utils_module

from ccai.planning.contact_planning import ChainedRecoveryNode, ContactPlanner


class DummyTurnProblem:
    obj_dof = 3


class FakeJointRecoverySampler:
    T = 3

    def __init__(self, expansions):
        self.expansions = list(expansions)
        self.calls = []

    def sample(self, N, start, H, constraints=None, project=False):
        self.calls.append(
            {
                "N": N,
                "start_shape": tuple(start.shape),
                "H": H,
                "constraints": constraints,
                "project": project,
            }
        )
        trajectories, modes, likelihoods = self.expansions.pop(0)
        assert trajectories.shape[0] == N
        assert modes.shape[0] == N
        assert likelihoods.shape[0] == N
        return trajectories.clone(), modes.clone(), likelihoods.clone()


class FakeTaskSampler:
    def __init__(self):
        self.calls = []

    def check_id(self, state, N, threshold=None, likelihood_only=False):
        self.calls.append(
            {
                "state": state.clone(),
                "N": N,
                "threshold": threshold,
                "likelihood_only": likelihood_only,
            }
        )
        return state[0].item()


class FakeBatchedTaskSampler:
    T = 3

    def __init__(self):
        self.calls = []

    def sample(self, N, H, start, constraints=None):
        self.calls.append(
            {
                "N": N,
                "H": H,
                "start": start.detach().clone(),
                "constraints": constraints.detach().clone(),
            }
        )
        return torch.zeros(N, H, start.shape[-1]), None, start[:, 0].clone()


def _params(**overrides):
    params = {
        "chained_recovery_contact_search": True,
        "sine_cosine": False,
        "N_contact_plan": 4,
        "likelihood_num_samples": 2,
        "likelihood_threshold": 0.0,
        "max_recovery_stages": 2,
        "fingers": ["index", "middle", "thumb"],
    }
    params.update(overrides)
    return params


def _raw_mode(name):
    vectors = {
        "index": torch.tensor([0.0, 1.0, 1.0]),
        "thumb_middle": torch.tensor([1.0, 0.0, 0.0]),
    }
    return 2 * vectors[name] - 1


def _raw_contact_vector(vector):
    return 2 * torch.tensor(vector) - 1


def _trajectories(terminal_scores):
    traj = torch.zeros(len(terminal_scores), 3, 36)
    for i, score in enumerate(terminal_scores):
        traj[i, -1, 0] = float(score)
    return traj


def _planner(expansions, params=None):
    task_sampler = FakeTaskSampler()
    planner = ContactPlanner(
        params or _params(),
        env=None,
        trajectory_sampler=FakeJointRecoverySampler(expansions),
        trajectory_sampler_orig=task_sampler,
        turn_problem=DummyTurnProblem(),
    )
    return planner, task_sampler


def test_chained_expansion_resamples_by_recovery_likelihood_before_grouping(monkeypatch):
    recovery_likelihoods = torch.log(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    modes = torch.stack(
        [
            _raw_mode("index"),
            _raw_mode("index"),
            _raw_mode("thumb_middle"),
            _raw_mode("thumb_middle"),
        ]
    )
    planner, _ = _planner([(_trajectories([10, 20, 30, 40]), modes, recovery_likelihoods)])

    captured = {}

    def fake_multinomial(weights, num_samples, replacement):
        captured["weights"] = weights.detach().clone()
        assert num_samples == 4
        assert replacement is True
        return torch.tensor([2, 3, 3, 1], device=weights.device)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)

    root = ChainedRecoveryNode(contact_sequence=[], terminal_states=torch.zeros(1, 15))
    children = planner._expand_chained_recovery_node(root)

    expected = torch.exp(recovery_likelihoods) / torch.exp(recovery_likelihoods).sum()
    assert torch.allclose(captured["weights"], expected)
    assert planner.trajectory_sampler.calls[0]["constraints"] is None

    child_counts = {tuple(child.contact_sequence): child.trajectories.shape[0] for child in children}
    assert child_counts[("thumb_middle",)] == 3
    assert child_counts[("index",)] == 1
    for child in children:
        assert child.csvto_seed_trajectories.shape == (4, 3, 36)
        assert child.csvto_seed_trajectories[:, -1, 0].tolist() == pytest.approx([10, 20, 30, 40])

    child_scores = {tuple(child.contact_sequence): child.score for child in children}
    expected_index_score = torch.logsumexp(
        recovery_likelihoods[:2] + torch.tensor([10.0, 20.0]),
        dim=0,
    ) - torch.logsumexp(recovery_likelihoods[:2], dim=0)
    expected_thumb_middle_score = torch.logsumexp(
        recovery_likelihoods[2:] + torch.tensor([30.0, 40.0]),
        dim=0,
    ) - torch.logsumexp(recovery_likelihoods[2:], dim=0)
    assert child_scores[("index",)] == pytest.approx(expected_index_score.item())
    assert child_scores[("thumb_middle",)] == pytest.approx(expected_thumb_middle_score.item())


def test_chained_search_returns_full_sequence_when_terminal_child_reaches_threshold(monkeypatch):
    first_modes = torch.stack(
        [
            _raw_mode("index"),
            _raw_mode("index"),
            _raw_mode("thumb_middle"),
            _raw_mode("thumb_middle"),
        ]
    )
    second_modes = torch.stack([_raw_mode("index")] * 8)
    expansions = [
        (_trajectories([-5, -5, -1, -1]), first_modes, torch.zeros(4)),
        (_trajectories([-5, -5, -5, -5, 1, 2, 1, 2]), second_modes, torch.zeros(8)),
    ]
    planner, _ = _planner(expansions)

    def identity_multinomial(weights, num_samples, replacement):
        return torch.arange(num_samples, device=weights.device)

    monkeypatch.setattr(torch, "multinomial", identity_multinomial)

    contact_sequence, goal_config, initial_samples, likelihoods, _ = planner._plan_chained_joint_recovery_contacts(
        torch.zeros(15)
    )

    assert contact_sequence == ["thumb_middle", "index"]
    assert goal_config[0].item() == pytest.approx(2.0)
    assert initial_samples.shape[0] == 4
    assert likelihoods.tolist() == pytest.approx([1.0, 2.0, 1.0, 2.0])
    assert planner.trajectory_sampler.calls[0]["constraints"] is None
    assert planner.trajectory_sampler.calls[1]["N"] == 8
    assert planner.trajectory_sampler.calls[1]["start_shape"] == (8, 15)


def test_chained_search_uses_chained_likelihood_threshold_for_termination(monkeypatch):
    modes = torch.stack(
        [
            _raw_mode("index"),
            _raw_mode("index"),
            _raw_mode("thumb_middle"),
            _raw_mode("thumb_middle"),
        ]
    )
    planner, task_sampler = _planner(
        [(_trajectories([-2, -2, 1, 1]), modes, torch.zeros(4))],
        params=_params(
            likelihood_threshold=10.0,
            chained_recovery_likelihood_threshold=0.0,
            max_recovery_stages=2,
        ),
    )

    def identity_multinomial(weights, num_samples, replacement):
        return torch.arange(num_samples, device=weights.device)

    monkeypatch.setattr(torch, "multinomial", identity_multinomial)

    contact_sequence, goal_config, _, likelihoods, _ = planner._plan_chained_joint_recovery_contacts(torch.zeros(15))

    assert contact_sequence == ["thumb_middle"]
    assert goal_config[0].item() == pytest.approx(1.0)
    assert likelihoods.tolist() == pytest.approx([1.0, 1.0])
    assert [call["threshold"] for call in task_sampler.calls] == [10.0, 10.0]


def test_chained_search_returns_raw_outputs_as_csvto_seeds_even_if_filtered(monkeypatch):
    recovery_likelihoods = torch.log(torch.tensor([1.0, 100.0, 3.0, 4.0]))
    modes = torch.stack(
        [
            _raw_mode("index"),
            _raw_contact_vector([0.0, 1.0, 0.0]),  # Undecodable mode; filtered for search only.
            _raw_mode("thumb_middle"),
            _raw_mode("thumb_middle"),
        ]
    )
    planner, _ = _planner([(_trajectories([10, 99, 30, 40]), modes, recovery_likelihoods)])

    captured = {}

    def fake_multinomial(weights, num_samples, replacement):
        captured["weights"] = weights.detach().clone()
        assert num_samples == 4
        assert replacement is True
        return torch.tensor([1, 2, 2, 1], device=weights.device)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)

    contact_sequence, goal_config, initial_samples, likelihoods, _ = planner._plan_chained_joint_recovery_contacts(
        torch.zeros(15)
    )

    expected_search_weights = torch.tensor([1.0, 3.0, 4.0]) / 8.0
    assert torch.allclose(captured["weights"], expected_search_weights)
    assert contact_sequence == ["thumb_middle"]
    assert goal_config[0].item() == pytest.approx(40.0)
    assert likelihoods.tolist() == pytest.approx([30.0, 40.0, 40.0, 30.0])
    assert initial_samples.shape == (4, 3, 36)
    assert initial_samples[:, -1, 0].tolist() == pytest.approx([10.0, 99.0, 30.0, 40.0])


def test_chained_frontier_expands_all_nodes_in_one_sampler_call(monkeypatch):
    first_parent = ChainedRecoveryNode(
        contact_sequence=["index"],
        terminal_states=torch.tensor([[1.0] + [0.0] * 14, [2.0] + [0.0] * 14]),
    )
    second_parent = ChainedRecoveryNode(
        contact_sequence=["thumb_middle"],
        terminal_states=torch.tensor([[3.0] + [0.0] * 14]),
    )
    modes = torch.stack(
        [
            _raw_mode("index"),
            _raw_mode("index"),
            _raw_mode("thumb_middle"),
            _raw_mode("thumb_middle"),
            _raw_mode("index"),
            _raw_mode("index"),
            _raw_mode("thumb_middle"),
            _raw_mode("thumb_middle"),
        ]
    )
    planner, _ = _planner([(_trajectories([1, 2, 3, 4, 5, 6, 7, 8]), modes, torch.zeros(8))])

    def identity_multinomial(weights, num_samples, replacement):
        return torch.arange(num_samples, device=weights.device)

    monkeypatch.setattr(torch, "multinomial", identity_multinomial)

    children = planner._expand_chained_recovery_frontier([first_parent, second_parent])

    assert len(planner.trajectory_sampler.calls) == 1
    assert planner.trajectory_sampler.calls[0]["N"] == 8
    assert planner.trajectory_sampler.calls[0]["start_shape"] == (8, 15)
    assert [child.contact_sequence for child in children] == [
        ["index", "index"],
        ["index", "thumb_middle"],
        ["thumb_middle", "index"],
        ["thumb_middle", "thumb_middle"],
    ]


def test_chained_frontier_scores_unique_terminal_states_once_after_grouping(monkeypatch):
    modes = torch.stack(
        [
            _raw_mode("index"),
            _raw_mode("thumb_middle"),
            _raw_mode("index"),
            _raw_mode("thumb_middle"),
        ]
    )
    planner, _ = _planner([(_trajectories([1, 1, 2, 3]), modes, torch.zeros(4))])
    task_sampler = FakeBatchedTaskSampler()
    planner.trajectory_sampler_orig = task_sampler

    def identity_multinomial(weights, num_samples, replacement):
        return torch.arange(num_samples, device=weights.device)

    monkeypatch.setattr(torch, "multinomial", identity_multinomial)

    root = ChainedRecoveryNode(contact_sequence=[], terminal_states=torch.zeros(1, 15))
    children = planner._expand_chained_recovery_node(root)

    assert len(task_sampler.calls) == 1
    assert task_sampler.calls[0]["N"] == 6
    assert task_sampler.calls[0]["start"].shape == (6, 16)
    assert task_sampler.calls[0]["constraints"].shape == (6, 3)
    assert [child.contact_sequence for child in children] == [["index"], ["thumb_middle"]]
    assert children[0].terminal_likelihoods.tolist() == pytest.approx([1.0, 2.0])
    assert children[1].terminal_likelihoods.tolist() == pytest.approx([1.0, 3.0])


def test_chained_search_returns_best_visited_node_at_max_depth(monkeypatch):
    modes = torch.stack(
        [
            _raw_mode("index"),
            _raw_mode("thumb_middle"),
            _raw_mode("thumb_middle"),
            _raw_mode("index"),
        ]
    )
    planner, _ = _planner(
        [(_trajectories([-4, -2, -2, -4]), modes, torch.zeros(4))],
        params=_params(max_recovery_stages=1, likelihood_threshold=10.0),
    )

    def identity_multinomial(weights, num_samples, replacement):
        return torch.arange(num_samples, device=weights.device)

    monkeypatch.setattr(torch, "multinomial", identity_multinomial)

    contact_sequence, goal_config, _, likelihoods, _ = planner._plan_chained_joint_recovery_contacts(torch.zeros(15))

    assert contact_sequence == ["thumb_middle"]
    assert goal_config[0].item() == pytest.approx(-2.0)
    assert likelihoods.tolist() == pytest.approx([-2.0, -2.0])
