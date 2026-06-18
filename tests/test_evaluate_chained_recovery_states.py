import csv
import pickle
from pathlib import Path

import torch

from examples import evaluate_chained_recovery_states as eval_states


class FakeEnv:
    def __init__(self):
        self.device = "cpu"
        self.q = torch.arange(16, dtype=torch.float32)
        self.reset_calls = 0
        self.set_pose_calls = []
        self.zero_velocity_calls = 0

    def reset(self):
        self.reset_calls += 1
        self.q = torch.arange(16, dtype=torch.float32)

    def get_state(self):
        return {"q": self.q.clone()}

    def set_pose(self, q, zero_velocity=False):
        self.set_pose_calls.append((q.clone(), zero_velocity))
        self.q = q.clone()

    def zero_obj_velocity(self):
        self.zero_velocity_calls += 1


class FakeTaskSampler:
    def __init__(self):
        self.calls = []

    def check_id(self, state, n, threshold=None, likelihood_only=False):
        self.calls.append(
            {
                "state": state.detach().clone(),
                "n": n,
                "threshold": threshold,
                "likelihood_only": likelihood_only,
            }
        )
        return state[0].item() + 0.5


class FakeContactPlanner:
    def __init__(self):
        self.calls = []

    def plan_recovery_contacts(self, state, stage, fpath, all_stage, index_regrasp_planner):
        self.calls.append(
            {
                "state": state.detach().clone(),
                "stage": stage,
                "fpath": Path(fpath),
                "all_stage": all_stage,
                "index_regrasp_planner": index_regrasp_planner,
            }
        )
        goal = torch.arange(15, dtype=torch.float32) + 100
        initial_samples = torch.ones(2, 3, 15)
        likelihood = torch.tensor([-4.0, -1.25, -3.0])
        return ["thumb_middle", "index"], goal, initial_samples, likelihood, 1.5


class FakeExecutor:
    def __init__(self, env):
        self.env = env
        self.calls = []

    def execute_traj(self, **kwargs):
        self.calls.append(kwargs)
        q = self.env.q.clone()
        q[:15] = q[:15] + len(self.calls)
        self.env.q = q
        return (
            torch.zeros(1, 15),
            [torch.ones(1, 15)],
            kwargs["initial_samples"],
            torch.full((1, 15), 2.0),
            {"path": torch.ones(1)},
            {"points": torch.ones(1)},
            {"distance": torch.ones(1)},
            False,
            len(self.calls),
        )


def _config(tmp_path):
    config = {
        "experiment_name": "fake_eval",
        "mode": "simulation",
        "visualize": True,
        "task_model_path": "task.pt",
        "model_path": "recovery.pt",
        "generate_context": True,
        "chained_recovery_contact_search": True,
        "likelihood_num_samples": 7,
        "likelihood_threshold": -10.0,
        "T": 3,
        "T_orig": 4,
        "fingers": ["index", "middle", "thumb"],
        "controllers": {"csvgd": {"device": "cpu", "N": 2}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text("\n".join([f"{key}: {value!r}" for key, value in config.items()]))
    return config_path


def test_evaluate_chained_recovery_state_preserves_sim_slot_and_writes_outputs(tmp_path, monkeypatch):
    saved_state = torch.arange(15, dtype=torch.float32) + 10
    states_path = tmp_path / "states.pkl"
    with open(states_path, "wb") as f:
        pickle.dump([saved_state.numpy()], f)

    env = FakeEnv()
    task_sampler = FakeTaskSampler()
    contact_planner = FakeContactPlanner()
    executor = FakeExecutor(env)

    def fake_setup(config):
        params = eval_states.build_params(config)
        return eval_states.EvaluationContext(
            config=config,
            params=params,
            env=env,
            trajectory_sampler=object(),
            trajectory_sampler_orig=task_sampler,
            contact_planner=contact_planner,
            trajectory_executor=executor,
            turn_problem=object(),
            mode_planner_dict={"thumb_middle": "tm_planner", "index": "index_planner"},
            min_force_dict={"thumb": 1.0, "middle": 1.0, "index": 1.0},
            allegro_screwdriver_cls=object,
        )

    monkeypatch.setattr(eval_states, "setup_context", fake_setup)

    output_dir = tmp_path / "out"
    result = eval_states.evaluate_states(
        config_path=_config(tmp_path),
        states_path=states_path,
        output_dir=output_dir,
        start_index=0,
        end_index=1,
        no_viewer=True,
        seed=123,
    )

    assert len(result["summary_rows"]) == 1
    assert env.reset_calls == 1
    assert env.zero_velocity_calls == 1
    set_pose_q, zero_velocity = env.set_pose_calls[0]
    assert zero_velocity is True
    assert torch.allclose(set_pose_q[:15], saved_state)
    assert set_pose_q[15].item() == 15.0

    assert len(contact_planner.calls) == 1
    assert torch.allclose(contact_planner.calls[0]["state"], saved_state)
    assert contact_planner.calls[0]["stage"] == 0
    assert contact_planner.calls[0]["all_stage"] == 0
    assert contact_planner.calls[0]["index_regrasp_planner"] is None

    assert [call["mode"] for call in executor.calls] == ["thumb_middle", "index"]
    assert executor.calls[0]["planner"] == "tm_planner"
    assert executor.calls[1]["planner"] == "index_planner"
    assert executor.calls[0]["recover"] is True
    assert executor.calls[1]["recover"] is True
    assert executor.calls[0]["initial_samples"] is not None
    assert executor.calls[1]["initial_samples"] is None
    assert torch.equal(executor.calls[0]["goal"], torch.arange(15, dtype=torch.float32) + 100)
    assert torch.equal(executor.calls[1]["goal"], torch.arange(15, dtype=torch.float32) + 100)

    summary_path = output_dir / "summary.csv"
    results_path = output_dir / "results.pkl"
    assert summary_path.exists()
    assert results_path.exists()

    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["state_index"] == "0"
    assert rows[0]["contact_sequence"] == '["thumb_middle", "index"]'
    assert float(rows[0]["predicted_likelihood"]) == -1.25
    assert float(rows[0]["initial_likelihood"]) == 10.5
    assert float(rows[0]["actual_likelihood"]) == 13.5
    assert float(rows[0]["likelihood_error"]) == 14.75

    with open(results_path, "rb") as f:
        records = pickle.load(f)
    assert records[0]["executed_contacts"] == ["thumb_middle", "index"]
    assert torch.allclose(records[0]["initial_state"], saved_state)
    assert torch.allclose(records[0]["full_initial_q"][:15], saved_state)
    assert records[0]["full_initial_q"][15].item() == 15.0
    assert len(records[0]["executor_artifacts"]) == 2
