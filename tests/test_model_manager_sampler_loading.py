import torch

from ccai.models.management import model_manager as model_manager_module
from ccai.models.management.model_manager import ModelManager


def _state_dict(prefix="model.diffusion_model.model", channels=(128, 256, 512)):
    state = {
        "x_mean": torch.zeros(37),
        "x_std": torch.ones(37),
    }
    in_channels = [37, *channels[:-1]]
    for level, (in_ch, out_ch) in enumerate(zip(in_channels, channels)):
        key = f"{prefix}.downs.{level}.0.blocks.0.block.0.weight"
        state[key] = torch.zeros(out_ch, in_ch, 3)
    return state


def test_infers_plain_task_sampler_dim_mults_from_down_blocks():
    state = _state_dict(channels=(128, 256, 512))

    arch = ModelManager._infer_sampler_architecture(state)

    assert arch == {"hidden_dim": 128, "dim_mults": (1, 2, 4)}


def test_infers_joint_recovery_sampler_dim_mults_from_nested_temporal_unet():
    state = _state_dict(
        prefix="model.diffusion_model.model.temporal_unet",
        channels=(64, 128),
    )

    arch = ModelManager._infer_sampler_architecture(state)

    assert arch == {"hidden_dim": 64, "dim_mults": (1, 2)}


def test_load_sampler_passes_inferred_architecture_to_trajectory_sampler(monkeypatch, tmp_path):
    loaded_state = _state_dict(channels=(128, 256, 512))
    captured = {}

    class FakeDiffusionModel:
        classifier = "set"

        def set_compilation_cache_dir(self, cache_dir):
            captured["cache_dir"] = cache_dir

    class FakeModel:
        diffusion_model = FakeDiffusionModel()

    class FakeTrajectorySampler:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.model = FakeModel()

        def load_state_dict(self, state_dict, strict):
            captured["strict"] = strict
            captured["loaded_keys"] = set(state_dict)

        def to(self, device):
            captured["device"] = device

        def send_norm_constants_to_submodels(self):
            captured["sent_norm_constants"] = True

    monkeypatch.setattr(model_manager_module, "TrajectorySampler", FakeTrajectorySampler)
    monkeypatch.setattr(
        model_manager_module.torch,
        "load",
        lambda path, map_location=None: loaded_state,
    )

    manager = ModelManager(
        config={
            "T": 12,
            "sine_cosine": True,
            "type": "diffusion",
            "experiment_name": "plain_task_model",
            "use_guidance": False,
            "likelihood_threshold": -15,
        },
        params={"device": "cpu"},
        ccai_path=tmp_path,
    )

    manager._load_sampler("checkpoint.pt", dim_mults=(1, 2), T=12, recovery=False)

    assert captured["kwargs"]["hidden_dim"] == 128
    assert captured["kwargs"]["dim_mults"] == (1, 2, 4)
    assert captured["kwargs"]["generate_context"] is False
    assert captured["strict"] is False
    assert "cache_dir" in captured
