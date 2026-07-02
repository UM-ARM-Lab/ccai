import importlib.util
import sys
from pathlib import Path

import pytest


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
    assert config["diffpf_trajectory_selection_mode"] == "max_reward_times_exp_likelihood"
    assert config["diffpf_likelihood_mask"] == "inverse_dynamics"
    assert config["diffpf_likelihood_temperature"] == 10.0
    assert config["diffpf_likelihood_reward_scope"] == "per_step"
    assert config["diffpf_reset_belief_after_recovery"] is True


def test_diffpf_values_are_loaded_from_yaml(monkeypatch, tmp_path):
    module = _load_recovery_module()
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"placeholder")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: diffpf_yaml",
                f"diffpf_checkpoint: '{checkpoint_path}'",
                "diffpf_ema_decay: 0.995",
                "diffpf_compile_model: true",
                "diffpf_sample_horizon: 6",
                "diffpf_execution_horizon: 3",
                "diffpf_num_trajectories: 64",
                "diffpf_trajectory_selection_mode: 'mean_reward'",
                "diffpf_likelihood_mask: 'all'",
                "diffpf_likelihood_temperature: 2.5",
                "diffpf_likelihood_reward_scope: 'trajectory'",
                "diffpf_reset_belief_after_recovery: false",
            ]
        ),
        encoding="utf-8",
    )
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

    assert config["diffpf_checkpoint"] == str(checkpoint_path)
    assert config["diffpf_ema_decay"] == 0.995
    assert config["diffpf_compile_model"] is True
    assert config["diffpf_sample_horizon"] == 6
    assert config["diffpf_execution_horizon"] == 3
    assert config["diffpf_num_trajectories"] == 64
    assert config["diffpf_trajectory_selection_mode"] == "mean_reward"
    assert config["diffpf_likelihood_mask"] == "all"
    assert config["diffpf_likelihood_temperature"] == 2.5
    assert config["diffpf_likelihood_reward_scope"] == "trajectory"
    assert config["diffpf_reset_belief_after_recovery"] is False


def test_model_compilation_is_yaml_configured(monkeypatch, tmp_path):
    module = _load_recovery_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: no_compile",
                "compile_models: false",
            ]
        ),
        encoding="utf-8",
    )
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

    assert config["compile_models"] is False


def test_diffpf_checkpoint_is_not_a_cli_flag(monkeypatch, tmp_path):
    module = _load_recovery_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment_name: diffpf_cli_rejected\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "screwdriver_isaacsim_recovery.py",
            "--config",
            str(config_path),
            "--diffpf_checkpoint",
            str(tmp_path / "checkpoint.pt"),
        ],
    )

    with pytest.raises(SystemExit):
        module.parse_args()
