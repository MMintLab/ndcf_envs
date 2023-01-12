import unittest

import numpy as np
from ncf_envs.press_simulator import get_wrist_wrench


class TestWrenchAdjoint(unittest.TestCase):

    def test_wrench_adjoint_no_torque(self):
        wrist_pose = np.array([0.0, 0.0, 0.1, np.pi, 0.0, 0.0])
        contact_points = np.array([[0.0, 0.0, 0.0]])
        contact_forces = -np.array([[0.0, 0.0, 1.0]])

        gt_wrist_wrench = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
        wrist_wrench = get_wrist_wrench(contact_points, contact_forces, wrist_pose)
        self.assertTrue(np.allclose(gt_wrist_wrench, wrist_wrench))

    def test_wrench_adjoint_cancel_torque(self):
        wrist_pose = np.array([0.0, 0.0, 1.0, np.pi, 0.0, 0.0])
        contact_points = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        contact_forces = -np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        gt_wrist_wrench = np.array([0.0, 0.0, -2.0, 0.0, 0.0, 0.0])
        wrist_wrench = get_wrist_wrench(contact_points, contact_forces, wrist_pose)
        self.assertTrue(np.allclose(gt_wrist_wrench, wrist_wrench))

    def test_wrench_adjoint_single_torque(self):
        wrist_pose = np.array([0.0, 0.0, 1.0, np.pi, 0.0, 0.0])
        contact_points = np.array([[1.0, 0.0, 0.0]])
        contact_forces = -np.array([[0.0, 0.0, 1.0]])

        gt_wrist_wrench = np.array([0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
        wrist_wrench = get_wrist_wrench(contact_points, contact_forces, wrist_pose)
        self.assertTrue(np.allclose(gt_wrist_wrench, wrist_wrench))

    def test_wrench_adjoint_multi_torque(self):
        wrist_pose = np.array([1.0, 0.0, 1.0, np.pi, 0.0, 0.0])
        contact_points = np.array([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])
        contact_forces = -np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])

        gt_wrist_wrench = np.array([0.0, 0.0, -3.0, 2.0, -2.5, 0.0])
        wrist_wrench = get_wrist_wrench(contact_points, contact_forces, wrist_pose)
        self.assertTrue(np.allclose(gt_wrist_wrench, wrist_wrench))


if __name__ == '__main__':
    unittest.main()
