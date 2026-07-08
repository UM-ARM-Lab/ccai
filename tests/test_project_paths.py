import importlib
import importlib.util
from pathlib import Path

from ccai.utils.project_paths import (
    find_model_mismatch_root,
    resolve_isaac_victor_envs_path,
    resolve_isaacsim_hand_envs_path,
    resolve_torch_cg_path,
)


def _write_model_mismatch_markers(root: Path) -> None:
    (root / "model_mismatch").mkdir(parents=True)
    (root / "model_mismatch" / "__init__.py").write_text("", encoding="utf-8")
    (root / "examples" / "evaluation").mkdir(parents=True)


def test_find_model_mismatch_root_from_nested_src_ccai_layout(tmp_path):
    repo_root = tmp_path / "model_mismatch"
    _write_model_mismatch_markers(repo_root)
    ccai_root = repo_root / "src" / "ccai"
    ccai_root.mkdir(parents=True)

    assert find_model_mismatch_root(ccai_root) == repo_root.resolve()


def test_find_model_mismatch_root_from_sibling_layout(tmp_path):
    workspace = tmp_path / "Documents"
    model_mismatch_root = workspace / "model_mismatch"
    ccai_root = workspace / "ccai"
    _write_model_mismatch_markers(model_mismatch_root)
    ccai_root.mkdir(parents=True)

    assert find_model_mismatch_root(ccai_root) == model_mismatch_root.resolve()


def test_dependency_roots_resolve_from_current_checkout():
    repo_root = find_model_mismatch_root(Path(__file__))

    assert (resolve_isaacsim_hand_envs_path(repo_root) / "isaacsim_hand_envs").is_dir()
    assert (resolve_isaac_victor_envs_path(repo_root) / "isaac_victor_envs").is_dir()
    assert (resolve_torch_cg_path(repo_root) / "torch_cg").is_dir()


def test_recovery_entrypoint_paths_resolve_to_checkout_root():
    repo_root = find_model_mismatch_root(Path(__file__))
    script_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "screwdriver_isaacsim_recovery.py"
    )
    spec = importlib.util.spec_from_file_location(
        "screwdriver_isaacsim_recovery_path_test",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.MODEL_MISMATCH_PATH == repo_root
    assert (
        module.MODEL_MISMATCH_PATH
        / "examples"
        / "evaluation"
        / "screwdriver_diffpf_policy.py"
    ).is_file()
    assert (module.ISAACSIM_HAND_ENVS_PATH / "isaacsim_hand_envs").is_dir()
    assert (module.ISAACGYM_ARM_ENVS_PATH / "isaac_victor_envs").is_dir()
    assert (module.TORCH_CG_PATH / "torch_cg").is_dir()


def test_recovery_utility_paths_resolve_to_checkout_root():
    repo_root = find_model_mismatch_root(Path(__file__))
    module = importlib.import_module("ccai.utils.isaacsim_screwdriver_recovery")

    assert module.MODEL_MISMATCH_PATH == repo_root
    assert (module.PROTO5_DEFAULTS_PATH).is_file()


def test_recovery_utils_plan_camera_path_uses_checkout_root():
    repo_root = find_model_mismatch_root(Path(__file__))
    module = importlib.import_module("ccai.utils.recovery_utils")

    assert module.MODEL_MISMATCH_PATH == repo_root
    assert module.DEFAULT_PROTO5_PLAN_CAMERA_PATH == repo_root / "scripts" / "proto5_plan_camera.json"
