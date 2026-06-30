import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "screwdriver_isaacsim_recovery.py"


def _load_recovery_module():
    spec = importlib.util.spec_from_file_location("screwdriver_isaacsim_recovery", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_diffpf_defaults_match_model_mismatch_collector(monkeypatch, tmp_path):
    module = _load_recovery_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment_name: diffpf_defaults\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "screwdriver_isaacsim_recovery.py",
            "--config",
            str(config_path),
        ],
    )

    config = module.load_config(module.parse_args())

    assert config["diffpf_sample_horizon"] == 8
    assert config["diffpf_execution_horizon"] == 1
    assert config["diffpf_num_trajectories"] == 128
    assert config["diffpf_trajectory_selection_mode"] == "max_reward"
    assert config["diffpf_likelihood_mask"] == "inverse_dynamics"
    assert config["diffpf_likelihood_temperature"] == 10.0
    assert config["diffpf_likelihood_reward_scope"] == "per_step"
