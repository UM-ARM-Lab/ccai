import pickle
from types import SimpleNamespace

import numpy as np

from scripts.render_proto5_saved_recovery_visualizations import render_posthoc_visualizations


def test_render_proto5_saved_recovery_visualizations_writes_artifacts_and_manifest(tmp_path):
    experiment = tmp_path / "proto5_screwdriver_hardware_recovery"
    trial_dir = experiment / "csvgd" / "trial_1"
    trial_dir.mkdir(parents=True)
    traj_data = {
        2: {
            "plans": np.zeros((1, 1, 2, 36), dtype=np.float32),
            "inits": np.ones((1, 2, 36), dtype=np.float32),
        },
        "pre_action_likelihoods": [],
    }
    with open(trial_dir / "traj_data.p", "wb") as handle:
        pickle.dump(traj_data, handle)
    with open(trial_dir / "trajectory.pkl", "wb") as handle:
        pickle.dump([np.zeros((1, 15), dtype=np.float32)], handle)

    def fake_context_builder(*, config, full_dof_reference):
        return SimpleNamespace(
            scene=object(),
            fingers=["index", "middle", "thumb"],
            obj_dof=3,
            full_dof_reference=full_dof_reference,
            joint_index={},
            controlled_joint_index=list(range(12)),
            camera_parameters_path=None,
            device="cpu",
        )

    def fake_visualizer(trajectory, scene, scene_fpath, fingers, obj_dof, **kwargs):
        out = __import__("pathlib").Path(scene_fpath)
        with open(out / "traj.pkl", "wb") as handle:
            pickle.dump(np.asarray(trajectory, dtype=np.float32), handle)
        (out / "img").mkdir(parents=True, exist_ok=True)
        (out / "gif").mkdir(parents=True, exist_ok=True)
        (out / "img" / "im_0000.png").write_bytes(b"fake png")
        (out / "gif" / "trajectory.gif").write_bytes(b"fake gif")

    manifest = render_posthoc_visualizations(
        experiment=experiment,
        trial=1,
        config=None,
        include={"plans", "samples"},
        render_backend="offscreen",
        visualizer=fake_visualizer,
        context_builder=fake_context_builder,
    )

    output_root = trial_dir / "posthoc_viz"
    assert (output_root / "plans" / "horizon_002" / "plan_0000" / "traj.pkl").exists()
    assert (output_root / "samples" / "horizon_002" / "sample_0000" / "traj.pkl").exists()
    assert (output_root / "executed" / "traj.pkl").exists()
    assert (output_root / "plans" / "horizon_002" / "plan_0000" / "img" / "im_0000.png").exists()
    manifest_text = (output_root / "manifest.yaml").read_text(encoding="utf-8")
    assert manifest["wrist_source"] == "default_proto5_wrist_values"
    assert "Saved run does not contain live wrist state" in manifest_text
