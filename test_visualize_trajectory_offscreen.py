from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from ccai.utils.allegro_utils import visualize_trajectory


class _FakeScene:
    device = "cpu"

    def get_visualization_meshes(self, q, env_q, pcd=None):
        robot_mesh = o3d.geometry.TriangleMesh.create_box(width=0.08, height=0.04, depth=0.03)
        robot_mesh.compute_vertex_normals()
        robot_mesh.paint_uniform_color([0.2, 0.6, 0.9])
        robot_mesh.translate([-0.04, -0.02, -0.015])

        object_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.035)
        object_mesh.compute_vertex_normals()
        object_mesh.paint_uniform_color([0.8, 0.3, 0.3])
        object_mesh.translate([0.12, 0.0, 0.0])

        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.02, 0.01, 0.015],
                [0.04, -0.01, 0.02],
            ],
            dtype=np.float64,
        )
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.paint_uniform_color([0.0, 0.0, 0.0])

        return [robot_mesh], [point_cloud, object_mesh]


def test_visualize_trajectory_offscreen_writes_nonempty_png(monkeypatch, tmp_path):
    monkeypatch.setattr("subprocess.call", lambda *args, **kwargs: 0)

    scene_path = Path(tmp_path) / "viz"
    (scene_path / "img").mkdir(parents=True, exist_ok=True)
    (scene_path / "gif").mkdir(parents=True, exist_ok=True)

    trajectory = torch.zeros((1, 16), dtype=torch.float32)
    visualize_trajectory(
        trajectory,
        _FakeScene(),
        scene_path,
        fingers=["index", "middle", "thumb"],
        obj_dof=4,
        render_backend="offscreen",
        task="screwdriver",
    )

    image_path = scene_path / "img" / "im_0000.png"
    assert image_path.exists()

    image = np.asarray(o3d.io.read_image(str(image_path)))
    assert image.size > 0
    assert float(image.var()) > 0.0
