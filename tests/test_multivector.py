# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

# script to perform B-Orthogonalization of a multivector - Omega and return the result
# Omega.dot(Bq) which is expected to be an identity matrix of the order Omega.nvec.

import unittest

import petsc4py
from mpi4py import MPI

import dolfinx as dlx
import dolfinx.fem.petsc
import numpy as np
import ufl

import hippylibX as hpx


class TestMultiVector(unittest.TestCase):
    def setUp(self):
        """
        Set up a small, fast environment for unit tests.
        This runs *before* every single test_... method.
        """
        self.comm = MPI.COMM_WORLD
        self.nx, self.ny = 10, 10
        self.nvec = 3  # Use a small number of vectors for speed

        # 1. Create mesh and function space
        self.msh = dlx.mesh.create_unit_square(self.comm, self.nx, self.ny, dlx.mesh.CellType.quadrilateral)
        self.Vh = dlx.fem.functionspace(self.msh, ("Lagrange", 1))

        # 2. Create a sample matrix (used for MvMult tests)
        trial = ufl.TrialFunction(self.Vh)
        test = ufl.TestFunction(self.Vh)
        varfM = ufl.inner(trial, test) * ufl.Measure("dx", metadata={"quadrature_degree": 4})
        self.M = dlx.fem.petsc.assemble_matrix(dlx.fem.form(varfM))
        self.M.assemble()

        # # 3. Create a sample vector to define parallel layout
        # self.sample_vec = self.M.createVecLeft()
        self.sample_vec = hpx.compat.create_vector(dlx.fem.form(test * ufl.dx))
        # # 4. Create the main MultiVector for testing
        hpx.parRandom.replay()  # Reset random seed for reproducibility
        self.mv = hpx.MultiVector.createFromVec(self.sample_vec, self.nvec)
        # self.sample_vec.destroy()
        hpx.parRandom.normal(1.0, self.mv)  # Fill with random data

    def tearDown(self):
        """
        Clean up PETSc objects after each test.
        """
        # self.comm.Barrier()
        self.M.destroy()
        self.sample_vec.destroy()
        # self.mv is destroyed by its own __del__

    # --- Your Original Test (integrated) ---
    def test_b_orthogonalize(self):
        """
        Tests the B-orthogonalization (your original test).
        Uses larger dimensions for stability.
        """
        hpx.parRandom.replay()
        Bq, _ = self.mv.Borthogonalize(self.M)
        result = self.mv.dot(Bq)  # This should be Q.dot(B*Q) = Q*B*Q
        # result = multi_vector_testing(nx, ny, nvec)
        # check_output(self, result, self.nvec)
        self.assertEqual(result.shape, (self.nvec, self.nvec))
        self.assertTrue(np.allclose(result, np.eye(self.nvec), atol=1e-6))

    # --- New Tests ---

    def test_creation_from_vec(self):
        """
        Tests if the MultiVector is created with the correct dimensions.
        """
        self.assertEqual(self.mv.nvec, self.nvec)
        self.assertEqual(len(self.mv.data), self.nvec)
        self.assertIsInstance(self.mv[0], petsc4py.PETSc.Vec)
        self.assertEqual(self.mv[0].getSize(), self.sample_vec.getSize())

    def test_creation_from_multivec(self):
        """
        Tests that createFromMultiVec performs a deep copy.
        """
        mv_copy = hpx.MultiVector.createFromMultiVec(self.mv)

        # Check that they are different objects
        self.assertIsNot(self.mv, mv_copy)
        self.assertIsNot(self.mv[0], mv_copy[0])

        # Check that values are initially the same
        norm_orig = self.mv.norm(petsc4py.PETSc.NormType.N2)
        norm_copy = mv_copy.norm(petsc4py.PETSc.NormType.N2)
        self.assertTrue(np.allclose(norm_orig, norm_copy))

        # Modify the copy and check that the original is unchanged
        mv_copy.scale(123.45)
        norm_copy_new = mv_copy.norm(petsc4py.PETSc.NormType.N2)

        self.assertFalse(np.allclose(norm_orig, norm_copy_new))

    def test_scale_scalar(self):
        """
        Tests scaling the MultiVector by a single float.
        """
        norm_before = self.mv.norm(petsc4py.PETSc.NormType.N2)
        alpha = 2.5

        self.mv.scale(alpha)

        norm_after = self.mv.norm(petsc4py.PETSc.NormType.N2)
        self.assertTrue(np.allclose(norm_after, alpha * norm_before))

    def test_scale_array(self):
        """
        Tests scaling the MultiVector by a numpy array.
        """
        norm_before = self.mv.norm(petsc4py.PETSc.NormType.N2)
        alpha = np.array([1.0, 2.0, 3.0])

        self.mv.scale(alpha)

        norm_after = self.mv.norm(petsc4py.PETSc.NormType.N2)
        self.assertTrue(np.allclose(norm_after, alpha * norm_before))

    def test_dot_vec(self):
        """
        Tests the dot product of a MultiVector and a single Vec.
        """
        v = self.sample_vec.duplicate()
        v.set(1.0)  # Fill with ones

        result = self.mv.dot(v)  # Should be an array of sums

        self.assertEqual(result.shape, (self.nvec,))

        # Calculate expected result (dot with 1.0 is the sum)
        expected = np.array([self.mv[i].sum() for i in range(self.nvec)])

        self.assertTrue(np.allclose(result, expected))
        v.destroy()

    def test_dot_multivec(self):
        """
        Tests the dot product of two MultiVectors.
        """
        # Create a second, different MultiVector
        mv2 = hpx.MultiVector.createFromVec(self.sample_vec, self.nvec)
        hpx.parRandom.normal(1.0, mv2)  # Fills with *different* random data

        result = self.mv.dot(mv2)  # Should be an (nvec, nvec) matrix

        self.assertEqual(result.shape, (self.nvec, self.nvec))

        # Calculate expected result
        expected = np.zeros((self.nvec, self.nvec))
        for i in range(self.nvec):
            for j in range(self.nvec):
                expected[i, j] = self.mv[i].dot(mv2[j])

        self.assertTrue(np.allclose(result, expected))

    def test_axpy(self):
        """
        Tests the axpy operation: self = self + alpha * Y
        """
        # Create original copies to check against
        mv_orig = hpx.MultiVector.createFromMultiVec(self.mv)
        Y = hpx.MultiVector.createFromMultiVec(self.mv)
        Y.scale(0.5)  # Y = 0.5 * mv

        alpha = 2.0

        # Operation: mv = mv + 2.0 * Y = mv + 2.0 * (0.5 * mv_orig) = 2.0 * mv_orig
        self.mv.axpy(alpha, Y)

        norm_after = self.mv.norm(petsc4py.PETSc.NormType.N2)
        norm_expected = 2.0 * mv_orig.norm(petsc4py.PETSc.NormType.N2)

        self.assertTrue(np.allclose(norm_after, norm_expected))

    def test_reduce(self):
        """
        Tests the reduce operation: y = y + sum(alpha[i] * self[i])
        """
        y = self.sample_vec.duplicate()
        y.set(0.0)  # Start with y = 0

        alpha = np.array([1.0, 2.0, 3.0])

        self.mv.reduce(y, alpha)

        # Calculate expected result
        expected = self.sample_vec.duplicate()
        expected.set(0.0)
        for i in range(self.nvec):
            expected.axpy(alpha[i], self.mv[i])

        # Check that y and expected are the same
        y.axpy(-1.0, expected)  # y = y - expected
        self.assertLess(y.norm(), 1e-12)

        y.destroy()
        expected.destroy()

    def test_mat_mv_mult(self):
        """
        Tests the helper function MatMvMult: Y = A * X
        """
        Y = hpx.MultiVector.createFromVec(self.sample_vec, self.nvec)
        hpx.MatMvMult(self.M, self.mv, Y)

        # Calculate expected result
        for i in range(self.nvec):
            expected_vec = self.sample_vec.duplicate()
            self.M.mult(self.mv[i], expected_vec)

            # Check Y[i] == expected_vec

            # This is the fix:
            # Instead of creating a new 'diff' vector, we modify Y[i]
            # and check if it becomes zero.
            Y[i].axpy(-1.0, expected_vec)  # Y[i] = Y[i] - expected_vec

            self.assertLess(Y[i].norm(), 1e-12)  # Norm of (Y[i] - expected_vec)

            expected_vec.destroy()

    def test_mv_ds_mat_mult(self):
        """
        Tests the helper function MvDSmatMult: Y = X * A (dense matrix)
        """
        X = hpx.MultiVector.createFromMultiVec(self.mv)
        Y = hpx.MultiVector.createFromVec(self.sample_vec, self.nvec)

        # Create a dense (nvec, nvec) numpy matrix
        A = np.arange(1, self.nvec**2 + 1).reshape((self.nvec, self.nvec))

        hpx.MvDSmatMult(X, A, Y)

        # Calculate expected result: Y[j] = sum_i( X[i] * A[i,j] )
        for j in range(self.nvec):
            expected_vec = self.sample_vec.duplicate()
            expected_vec.set(0.0)
            for i in range(self.nvec):
                expected_vec.axpy(A[i, j], X[i])

            # Check Y[j] == expected_vec

            # This is the fix:
            # We modify Y[j] and check if the result is zero.
            Y[j].axpy(-1.0, expected_vec)  # Y[j] = Y[j] - expected_vec

            self.assertLess(Y[j].norm(), 1e-12)  # Norm of (Y[j] - expected_vec)

            expected_vec.destroy()


if __name__ == "__main__":
    unittest.main()
