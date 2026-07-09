import json
import sys
import types

import torch


isaac_utils = types.ModuleType("isaac_victor_envs.utils")
isaac_utils.get_assets_dir = lambda: "/tmp"
isaac_tasks = types.ModuleType("isaac_victor_envs.tasks")
isaac_allegro = types.ModuleType("isaac_victor_envs.tasks.allegro")
isaac_allegro.AllegroScrewdriverTurningEnv = object
isaac_ros = types.ModuleType("isaac_victor_envs.tasks.allegro_ros")
isaac_ros.RosAllegroScrewdriverTurningEnv = object
sys.modules.setdefault("isaac_victor_envs", types.ModuleType("isaac_victor_envs"))
sys.modules["isaac_victor_envs.utils"] = isaac_utils
sys.modules["isaac_victor_envs.tasks"] = isaac_tasks
sys.modules["isaac_victor_envs.tasks.allegro"] = isaac_allegro
sys.modules["isaac_victor_envs.tasks.allegro_ros"] = isaac_ros
sys.modules.setdefault("pytorch_kinematics", types.ModuleType("pytorch_kinematics"))
sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))

allegro_utils = types.ModuleType("ccai.utils.allegro_utils")
allegro_utils.convert_yaw_to_sine_cosine = lambda x: x
allegro_utils.convert_sine_cosine_to_yaw = lambda x: x
allegro_utils.visualize_trajectory = lambda *args, **kwargs: None
allegro_utils.partial_to_full_state = lambda x, *args, **kwargs: x
allegro_utils.extract_state_vector = lambda state, *args, **kwargs: state
sys.modules["ccai.utils.allegro_utils"] = allegro_utils

recovery_utils = types.ModuleType("ccai.utils.recovery_utils")
recovery_utils.create_allegro_screwdriver_problem = lambda *args, **kwargs: None
recovery_utils.create_planner = lambda *args, **kwargs: None
recovery_utils.add_to_dataset = lambda *args, **kwargs: None
recovery_utils.partial_to_full_trajectory = lambda x, *args, **kwargs: x
recovery_utils.full_to_partial_trajectory = lambda x, *args, **kwargs: x
recovery_utils.create_mode_planner_dict = lambda *args, **kwargs: {}
sys.modules["ccai.utils.recovery_utils"] = recovery_utils

allegro_contact = types.ModuleType("ccai.allegro_contact")
allegro_contact.AllegroManipulationProblem = object
allegro_contact.PositionControlConstrainedSVGDMPC = object
sys.modules["ccai.allegro_contact"] = allegro_contact

baselines = types.ModuleType("ccai.baselines.allegro_recovery_baselines")
baselines.BaselineRecoveryController = object
baselines.BaselineOODDetector = object
baselines.get_baseline_contact_sequence = lambda *args, **kwargs: []
baselines.get_num_envs_for_baseline = lambda *args, **kwargs: 1
baselines.handle_baseline_trajectory_processing = lambda traj, plans, *args, **kwargs: (traj, plans)
sys.modules["ccai.baselines.allegro_recovery_baselines"] = baselines

contact_planning = types.ModuleType("ccai.planning.contact_planning")
contact_planning.ContactPlanner = object
sys.modules["ccai.planning.contact_planning"] = contact_planning
trial_executor = types.ModuleType("ccai.execution.trial_executor")
trial_executor.TrajectoryExecutor = object
sys.modules["ccai.execution.trial_executor"] = trial_executor
model_manager = types.ModuleType("ccai.models.management.model_manager")
model_manager.ModelManager = object
sys.modules["ccai.models.management.model_manager"] = model_manager

from examples import allegro_screwdriver


class DummyTurnProblem:
    contact_scenes_for_viz = object()
    fingers = ["index", "middle", "thumb"]


def test_save_executed_rollout_visualization_combines_segments_and_metadata(tmp_path, monkeypatch):
    rendered = {}

    def fake_visualize_trajectory(traj, contact_scenes, fpath, fingers, obj_dof):
        rendered["traj"] = traj.clone()
        rendered["fpath"] = fpath
        rendered["fingers"] = fingers
        rendered["obj_dof"] = obj_dof

    monkeypatch.setattr(allegro_screwdriver, "visualize_trajectory", fake_visualize_trajectory)

    log_run_dir = tmp_path / "20260706_1430_temp0p5"
    trial_dir = log_run_dir / "trial_1"
    actual_trajectory = [
        torch.arange(15, dtype=torch.float32),
        torch.full((2, 36), 2.0),
        [],
    ]
    final_state = torch.full((15,), 9.0)
    selected_recovery = {
        "selected_node_id": 3,
        "selected_particle_index": 1,
        "log_run_name": "20260706_1430_temp0p5",
        "log_run_dir": str(log_run_dir),
        "recovery_likelihood_temperature": 0.5,
    }

    viz_path = allegro_screwdriver.save_executed_rollout_visualization(
        trial_dir,
        actual_trajectory,
        final_state,
        ["thumb_middle", "index"],
        DummyTurnProblem(),
        num_fingers=3,
        obj_dof=3,
        selected_recovery=selected_recovery,
        temperature=1.0,
        all_stage=8,
    )

    assert viz_path == trial_dir / "executed_rollout"
    assert (viz_path / "img").exists()
    assert (viz_path / "gif").exists()
    assert rendered["fpath"] == viz_path
    assert rendered["obj_dof"] == 4
    assert rendered["traj"].shape == (4, 16)
    assert torch.allclose(rendered["traj"][0, :15], torch.arange(15, dtype=torch.float32))
    assert torch.allclose(rendered["traj"][1, :15], torch.full((15,), 2.0))
    assert torch.allclose(rendered["traj"][2, :15], torch.full((15,), 2.0))
    assert torch.allclose(rendered["traj"][3, :15], final_state)

    with open(viz_path / "metadata.json") as f:
        metadata = json.load(f)
    assert metadata["executed_contacts"] == ["thumb_middle", "index"]
    assert metadata["num_frames"] == 4
    assert metadata["trial_dir"] == str(trial_dir)
    assert metadata["log_run_name"] == "20260706_1430_temp0p5"
    assert metadata["log_run_dir"] == str(log_run_dir)
    assert metadata["all_stage"] == 8
    assert metadata["recovery_likelihood_temperature"] == 0.5
    assert metadata["selected_recovery"] == selected_recovery


def test_experiment_log_run_dir_uses_collision_safe_top_level_stamp(tmp_path, monkeypatch):
    monkeypatch.setattr(allegro_screwdriver.time, "strftime", lambda fmt: "20260706_1430")

    controller_dir = tmp_path / "csvgd"

    first = allegro_screwdriver._experiment_log_run_dir(controller_dir, temperature=2.0)
    first.mkdir(parents=True)
    second = allegro_screwdriver._experiment_log_run_dir(controller_dir, temperature=2.0)

    assert first == controller_dir / "20260706_1430_temp2p0"
    assert second == controller_dir / "20260706_1430_temp2p0_run02"
