import unittest

import numpy as np
from numpy.testing import assert_allclose

from hippylibX.modeling.expression import AnisTensor2D, Mollifier


class TestAnisTensor2D(unittest.TestCase):
    def test_default_identity(self):
        """Tests that the default tensor is the 2x2 identity."""
        A = AnisTensor2D()
        x_in = np.array([[0.1, 0.5], [0.2, 0.6]])
        values = A(x_in)
        expected_components = np.array([1.0, 0.0, 0.0, 1.0])
        self.assertEqual(values.shape, (4, 2))
        assert_allclose(values[:, 0], expected_components)
        assert_allclose(values[:, 1], expected_components)

    def test_set_and_diagonal_tensor(self):
        """Tests the .set() method for a 90-degree rotation."""
        A = AnisTensor2D()
        A.set(theta0=5.0, theta1=2.0, alpha=np.pi / 2.0)
        x_in = np.array([[0.0], [0.0]])
        values = A(x_in)

        expected_components = np.array([5.0, 0.0, 0.0, 2.0])

        self.assertEqual(values.shape, (4, 1))

        assert_allclose(values[:, 0], expected_components, atol=1e-15)

    def test_45_degree_rotation(self):
        """Tests a non-trivial 45-degree rotation."""
        A = AnisTensor2D()
        A.set(theta0=3.0, theta1=1.0, alpha=np.pi / 4.0)
        x_in = np.array([[0.0], [0.0]])
        values = A(x_in)

        expected_components = np.array([2.0, 1.0, 1.0, 2.0])

        self.assertEqual(values.shape, (4, 1))
        assert_allclose(values[:, 0], expected_components)


class TestMollifier(unittest.TestCase):
    def setUp(self):
        """Create clean helpers for each test."""
        self.A_helper = AnisTensor2D()
        self.m_helper = Mollifier()

    def test_single_isotropic_kernel(self):
        """
        Tests a single kernel at (0,0) with A=I, l=1, o=2.
        This should be a standard Gaussian: exp(-(x^2 + y^2))
        """
        self.A_helper.set(theta0=1.0, theta1=1.0, alpha=0.0)
        self.m_helper.set(self.A_helper, l=1.0, o=2.0)
        self.m_helper.addLocation(0.0, 0.0)

        x_in = np.array([[0.0, 1.0], [0.0, 0.0]])
        values = self.m_helper(x_in)

        expected_values = np.array([[1.0, np.exp(-1.0)]])

        self.assertEqual(values.shape, (1, 2))
        assert_allclose(values, expected_values)

    def test_multiple_isotropic_kernels(self):
        """
        Tests the summation of two kernels at (0,0) and (2,0).
        f(x) = exp(-(x^2 + y^2)) + exp(-((x-2)^2 + y^2))
        """
        self.A_helper.set(theta0=1.0, theta1=1.0, alpha=0.0)
        self.m_helper.set(self.A_helper, l=1.0, o=2.0)
        self.m_helper.addLocation(0.0, 0.0)
        self.m_helper.addLocation(2.0, 0.0)

        x_in = np.array([[0.0, 1.0], [0.0, 0.0]])
        values = self.m_helper(x_in)

        val1 = 1.0 + np.exp(-4.0)
        val2 = 2.0 * np.exp(-1.0)

        expected_values = np.array([[val1, val2]])

        self.assertEqual(values.shape, (1, 2))
        assert_allclose(values, expected_values)

    def test_single_anisotropic_kernel(self):
        """
        Tests an elliptical kernel. A = diag(4, 1), so B = diag(1, 0.25).
        f(x) = exp(-(x^2 + 0.25*y^2))
        """
        # A = diag(4, 1), l=1, o=2
        self.A_helper.set(theta0=4.0, theta1=1.0, alpha=0.0)
        self.m_helper.set(self.A_helper, l=1.0, o=2.0)
        self.m_helper.addLocation(0.0, 0.0)

        # Test points: (0,0), (1,0), (0,1), (2,0)
        x_in = np.array([[0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 0.0]])

        values = self.m_helper(x_in)

        # B_00 = it0*sa^2 + it1*ca^2 = 0.25*0 + 1.0*1 = 1.0
        # B_11 = it0*ca^2 + it1*sa^2 = 0.25*1 + 1.0*0 = 0.25
        # So B = diag(1.0, 0.25). f(x,y) = exp(-(x^2 + 0.25*y^2))

        # At (0,0): e=0. val=exp(0)=1
        # At (1,0): e=1. val=exp(-1)
        # At (0,1): e=0.25. val=exp(-0.25)
        # At (2,0): e=4. val=exp(-4)
        expected_values = np.array([[1.0, np.exp(-1.0), np.exp(-0.25), np.exp(-4.0)]])

        self.assertEqual(values.shape, (1, 4))
        assert_allclose(values, expected_values)


if __name__ == "__main__":
    unittest.main()
