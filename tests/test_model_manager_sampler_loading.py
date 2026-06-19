import sys
import types

import torch

torchdiffeq_stub = types.ModuleType("torchdiffeq")
torchdiffeq_stub.odeint_adjoint = None
sys.modules.setdefault("torchdiffeq", torchdiffeq_stub)

from ccai.models.management import model_manager as model_manager_module
from ccai.models.management.model_manager import ModelManager
from ccai.models import temporal as temporal_module


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


def test_load_trajectory_samplers_warms_task_and_recovery_models(monkeypatch, tmp_path):
    loaded_state = _state_dict(channels=(128, 256, 512))
    warmups = []
    created_samplers = []

    class FakeDiffusionModel:
        classifier = "set"

        def set_compilation_cache_dir(self, cache_dir):
            self.cache_dir = cache_dir

    class FakeModel:
        def __init__(self):
            self.diffusion_model = FakeDiffusionModel()

    class FakeTrajectorySampler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = FakeModel()
            created_samplers.append(self)

        def load_state_dict(self, state_dict, strict):
            self.strict = strict

        def to(self, device):
            self.device = device

        def send_norm_constants_to_submodels(self):
            self.sent_norm_constants = True

        def warmup_model(self, warmup_batch_size=16, warmup_horizon=None):
            warmups.append(
                {
                    "generate_context": self.kwargs["generate_context"],
                    "warmup_batch_size": warmup_batch_size,
                    "warmup_horizon": warmup_horizon,
                }
            )

    monkeypatch.setattr(model_manager_module, "TrajectorySampler", FakeTrajectorySampler)
    monkeypatch.setattr(
        model_manager_module.torch,
        "load",
        lambda path, map_location=None: loaded_state,
    )

    manager = ModelManager(
        config={
            "T": 3,
            "T_orig": 12,
            "sine_cosine": True,
            "type": "diffusion",
            "experiment_name": "compiled_sampler_test",
            "use_guidance": False,
            "likelihood_threshold": -15,
            "model_path": "recovery.pt",
            "task_model_path": "task.pt",
            "generate_context": True,
            "compile_warmup_batch_size": 4,
        },
        params={"device": "cpu", "recovery_controller": "csvgd", "task_model_path": "task.pt"},
        ccai_path=tmp_path,
    )

    recovery_sampler, task_sampler, classifier = manager.load_trajectory_samplers()

    assert recovery_sampler is created_samplers[0]
    assert task_sampler is created_samplers[1]
    assert classifier is None
    assert warmups == [
        {"generate_context": True, "warmup_batch_size": 4, "warmup_horizon": None},
        {"generate_context": False, "warmup_batch_size": 4, "warmup_horizon": None},
    ]


def test_load_trajectory_samplers_disables_model_compilation(monkeypatch, tmp_path):
    loaded_state = _state_dict(channels=(128, 256, 512))
    warmups = []
    compilation_enabled = []

    class FakeCompileModel:
        def set_compilation_enabled(self, enabled):
            compilation_enabled.append(enabled)

    class FakeDiffusionModel:
        classifier = "set"

        def __init__(self):
            self.model = FakeCompileModel()

        def set_compilation_cache_dir(self, cache_dir):
            self.cache_dir = cache_dir

    class FakeModel:
        def __init__(self):
            self.diffusion_model = FakeDiffusionModel()

    class FakeTrajectorySampler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = FakeModel()

        def load_state_dict(self, state_dict, strict):
            self.strict = strict

        def to(self, device):
            self.device = device

        def send_norm_constants_to_submodels(self):
            self.sent_norm_constants = True

        def warmup_model(self, warmup_batch_size=16, warmup_horizon=None):
            warmups.append(warmup_batch_size)

    monkeypatch.setattr(model_manager_module, "TrajectorySampler", FakeTrajectorySampler)
    monkeypatch.setattr(
        model_manager_module.torch,
        "load",
        lambda path, map_location=None: loaded_state,
    )

    manager = ModelManager(
        config={
            "T": 3,
            "T_orig": 12,
            "sine_cosine": True,
            "type": "diffusion",
            "experiment_name": "uncompiled_sampler_test",
            "use_guidance": False,
            "likelihood_threshold": -15,
            "model_path": "recovery.pt",
            "task_model_path": "task.pt",
            "generate_context": True,
            "compile_models": False,
        },
        params={"device": "cpu", "recovery_controller": "csvgd", "task_model_path": "task.pt"},
        ccai_path=tmp_path,
    )

    manager.load_trajectory_samplers()

    assert warmups == []
    assert compilation_enabled == [False, False]


def test_compilation_mixin_get_or_compile_method_invokes_torch_compile(monkeypatch):
    calls = []

    def fake_compile(fn, **kwargs):
        calls.append({"fn": fn, "kwargs": kwargs})

        def compiled(*args, **inner_kwargs):
            return fn(*args, **inner_kwargs)

        return compiled

    monkeypatch.setattr(temporal_module.torch, "compile", fake_compile)
    mixin = temporal_module.CompilationMixin()

    def original(value):
        return value + 1

    compiled = mixin._get_or_compile_method("original", original)

    assert compiled(1) == 2
    assert calls == [{"fn": original, "kwargs": {"mode": "max-autotune"}}]
    assert mixin._get_or_compile_method("original", original) is compiled


def test_compilation_mixin_can_return_original_method_without_torch_compile(monkeypatch):
    calls = []

    monkeypatch.setattr(
        temporal_module.torch,
        "compile",
        lambda fn, **kwargs: calls.append({"fn": fn, "kwargs": kwargs}),
    )
    mixin = temporal_module.CompilationMixin()

    def original(value):
        return value + 1

    mixin.set_compilation_enabled(False)
    fn = mixin._get_or_compile_method("original", original)

    assert fn is original
    assert fn(1) == 2
    assert calls == []
