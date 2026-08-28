import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from hesai_xt32 import detect_unsupervised as detector


class GroundPlaneSelectionTest(unittest.TestCase):
    def test_height_prior_skips_larger_plane_near_sensor_origin(self):
        rng = np.random.default_rng(11)

        false_xy = rng.uniform(-12.0, 12.0, size=(2000, 2))
        false_z = rng.normal(0.0, 0.008, size=(2000, 1))
        false_plane = np.column_stack((false_xy, false_z))

        ground_xy = rng.uniform(-12.0, 12.0, size=(1200, 2))
        ground_z = (
            -1.20
            + 0.04 * ground_xy[:, 0]
            - 0.02 * ground_xy[:, 1]
            + rng.normal(0.0, 0.008, size=1200)
        )
        ground_plane = np.column_stack((ground_xy, ground_z))
        points = np.vstack((false_plane, ground_plane))

        with redirect_stdout(io.StringIO()):
            plane_model, selected_points = detector.estimate_ground_plane(points)
        metrics = detector.ground_plane_metrics(plane_model)

        self.assertGreater(len(selected_points), 1000)
        self.assertAlmostEqual(metrics["sensor_height_m"], 1.20, delta=0.08)
        self.assertLess(metrics["tilt_degrees"], 5.0)


class FarProposalFilterTest(unittest.TestCase):
    def test_rejects_sparse_far_cluster(self):
        points = np.array(
            [
                [25.0 + index * 0.2, -0.5 + (index % 3) * 0.5, index * 0.1]
                for index in range(9)
            ],
            dtype=np.float64,
        )
        result = detector.fit_filtered_obb(points, 0, (1.0, 0.0, 0.0))
        self.assertIsNone(result)

    def test_rejects_thin_far_cluster(self):
        x = np.linspace(24.0, 28.0, 20)
        y = np.tile([0.0, 0.05], 10)
        z = np.linspace(0.0, 1.2, 20)
        points = np.column_stack((x, y, z))
        result = detector.fit_filtered_obb(points, 0, (1.0, 0.0, 0.0))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
