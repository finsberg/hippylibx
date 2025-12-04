# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

import unittest

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx as dlx
import numpy as np

import hippylibX as hpx


class Testing_Execution(unittest.TestCase):
    def setUp(self):
        """
        This method is run automatically before every single test.
        It sets up a common environment for all tests.
        """
        # --- 1. Setup the Mesh and Function Space ---
        nx, ny = 10, 10
        self.prior_param = {"gamma": 0.03, "delta": 0.3}
        self.comm = MPI.COMM_WORLD

        # We use the triangle mesh as it was the one that worked
        self.msh = dlx.mesh.create_unit_square(self.comm, nx, ny, dlx.mesh.CellType.triangle)

        # The main function space
        self.Vh = dlx.fem.functionspace(self.msh, ("Lagrange", 1))

        # --- 2. Initialize the Prior Object ---
        # This calls the factory function 'BiLaplacianPrior'
        self.prior = hpx.BiLaplacianPrior(self.Vh, self.prior_param["gamma"], self.prior_param["delta"], mean=None)

        # --- 3. Utility: Create a random vector ---
        # Reset the random seed for reproducible tests
        hpx.parRandom.replay()
        self.x_rand = dlx.la.vector(self.Vh.dofmap.index_map, self.Vh.dofmap.index_map_bs)
        hpx.parRandom.normal(1.0, self.x_rand)

        # --- 4. Utility: Create temporary vectors in Vh ---
        self.y1 = dlx.la.vector(self.Vh.dofmap.index_map, self.Vh.dofmap.index_map_bs)
        self.y2 = dlx.la.vector(self.Vh.dofmap.index_map, self.Vh.dofmap.index_map_bs)
        self.tmp1 = dlx.la.vector(self.Vh.dofmap.index_map, self.Vh.dofmap.index_map_bs)
        self.tmp2 = dlx.la.vector(self.Vh.dofmap.index_map, self.Vh.dofmap.index_map_bs)

    # --- Your Original (Fixed) Test ---
    def test_prior_mass_matrix(self):
        """
        Tests the mass matrix factor: M = S * S^T
        where S = prior.sqrtM
        """
        # We need the special intermediate vector 'tmp_q' in the Qh space
        tmp_q = dlx.la.vector(self.prior.Qh.dofmap.index_map, self.prior.Qh.dofmap.index_map_bs)

        # y1 = (S * S^T) * x
        self.prior.sqrtM.multTranspose(self.x_rand.petsc_vec, tmp_q.petsc_vec)  # tmp_q = S^T * x
        self.prior.sqrtM.mult(tmp_q.petsc_vec, self.y1.petsc_vec)  # y1 = S * tmp_q

        # y2 = M * x
        self.prior.M.mult(self.x_rand.petsc_vec, self.y2.petsc_vec)

        # Compare y1 and y2
        self.y2.petsc_vec.axpy(-1.0, self.y1.petsc_vec)  # y2 = y2 - y1
        value = self.y2.petsc_vec.norm(PETSc.NormType.N2)

        self.assertLessEqual(np.abs(value), 1e-6, "prior.sqrtM factor failed (M != S*S^T)")

    # --- New Test 1: Check Precision Matrix Definition (UPDATED) ---
    def test_precision_matrix_definition(self):
        """
        Tests the definition of the precision operator:
        R = A * M^{-1} * A

        We check this by applying it to a random vector x:
        R*x == A * (M_solver.solve(A*x))
        """
        # 1. Compute y1 = R*x using the provided operator
        self.prior.R.mult(self.x_rand.petsc_vec, self.y1.petsc_vec)

        # 2. Compute y2 = A * M^{-1} * (A * x) manually

        # tmp1 = A * x
        self.prior.A.mult(self.x_rand.petsc_vec, self.tmp1.petsc_vec)

        # tmp2 = M^{-1} * tmp1
        self.prior.Msolver.solve(self.tmp1.petsc_vec, self.tmp2.petsc_vec)

        # y2 = A * tmp2
        self.prior.A.mult(self.tmp2.petsc_vec, self.y2.petsc_vec)

        # 3. Compare y1 and y2
        self.y2.petsc_vec.axpy(-1.0, self.y1.petsc_vec)  # y2 = y2 - y1
        value = self.y2.petsc_vec.norm(PETSc.NormType.N2)

        # We may need a slightly looser tolerance due to iterative solves
        self.assertLessEqual(np.abs(value), 1e-5, "Precision matrix definition failed (R != A*M_inv*A)")

    # --- New Test 2: Check Precision/Covariance Inverse (NEW) ---
    def test_precision_solver_inverse(self):
        """
        Tests that Rsolver is the inverse of the R operator.
        Rsolver * (R * x) == x
        """
        # 1. Compute y1 = R * x
        self.prior.R.mult(self.x_rand.petsc_vec, self.y1.petsc_vec)

        # 2. Compute y2 = R^{-1} * y1 (using Rsolver)
        self.prior.Rsolver.solve(self.y1.petsc_vec, self.y2.petsc_vec)

        # 3. Compare y2 and the original x_rand
        self.y2.petsc_vec.axpy(-1.0, self.x_rand.petsc_vec)  # y2 = y2 - x
        value = self.y2.petsc_vec.norm(PETSc.NormType.N2)

        # Tolerance depends on the accuracy of the inner solvers
        self.assertLessEqual(np.abs(value), 1e-5, "Rsolver is not the inverse of R (R_inv * R * x != x)")

    # --- New Test 3: Check Covariance Solver Definition (NEW) ---
    def test_covariance_solver_definition(self):
        """
        Tests the definition of the covariance operator (Rsolver):
        C = A^{-1} * M * A^{-1}

        We check this by applying it to a random vector x:
        Rsolver.solve(x) == A_solver.solve(M * (A_solver.solve(x)))
        """
        # 1. Compute y1 = C*x using Rsolver
        self.prior.Rsolver.solve(self.x_rand.petsc_vec, self.y1.petsc_vec)

        # 2. Compute y2 = A^{-1} * M * (A^{-1} * x) manually

        # tmp1 = A^{-1} * x
        self.prior.Asolver.solve(self.x_rand.petsc_vec, self.tmp1.petsc_vec)

        # tmp2 = M * tmp1
        self.prior.M.mult(self.tmp1.petsc_vec, self.tmp2.petsc_vec)

        # y2 = A^{-1} * tmp2
        self.prior.Asolver.solve(self.tmp2.petsc_vec, self.y2.petsc_vec)

        # 3. Compare y1 and y2
        self.y2.petsc_vec.axpy(-1.0, self.y1.petsc_vec)  # y2 = y2 - y1
        value = self.y2.petsc_vec.norm(PETSc.NormType.N2)

        self.assertLessEqual(np.abs(value), 1e-5, "Covariance solver definition failed (C != A_inv*M*A_inv)")

    # --- New Test 4: Check Matrix Symmetry ---
    def test_matrix_symmetry(self):
        """
        Tests if the M, A, and R operators are symmetric.
        A matrix K is symmetric if <y, K*x> == <x, K*y> for any x, y.
        """
        # Create a second random vector 'y'
        y_rand = dlx.la.vector(self.Vh.dofmap.index_map, self.Vh.dofmap.index_map_bs)
        try:
            hpx.parRandom.normal(1.0, y_rand)  # Fills with new random values
        except AttributeError:
            y_rand.petsc_vec.setRandom()

        # Temporary vectors
        Kx = self.y1
        Ky = self.y2

        matrices_to_check = {
            "Mass (M)": self.prior.M,
            "SqrtPrecision (A)": self.prior.A,
            "Precision (R)": self.prior.R,  # R = A M^{-1} A, which should be symmetric
        }

        for name, K in matrices_to_check.items():
            # Kx = K*x
            K.mult(self.x_rand.petsc_vec, Kx.petsc_vec)
            # Ky = K*y
            K.mult(y_rand.petsc_vec, Ky.petsc_vec)

            # val1 = <y, K*x>
            val1 = y_rand.petsc_vec.dot(Kx.petsc_vec)
            # val2 = <x, K*y>
            val2 = self.x_rand.petsc_vec.dot(Ky.petsc_vec)

            self.assertLessEqual(np.abs(val1 - val2), 1e-6, f"Symmetry test failed for {name} operator")

    # --- New Test 5: Check Laplacian Nullspace (UPDATED) ---
    def test_laplacian_nullspace(self):
        """
        Tests the nullspace of the Laplacian part of the operator.
        We create a new prior with delta=0.0 and robin_bc=False.
        The resulting 'A' matrix should be gamma * L.
        When applied to a constant vector 1, the result should be zero.
        """
        # 1. Create a special prior with delta=0 (so A = gamma*L)
        prior_L_only = hpx.BiLaplacianPrior(self.Vh, gamma=0.03, delta=0.0, mean=None, robin_bc=False)

        # 2. Create a constant vector of all ones
        x_const = dlx.la.vector(self.Vh.dofmap.index_map, self.Vh.dofmap.index_map_bs)
        x_const.petsc_vec.set(1.0)

        # 3. Compute y = (gamma*L)*x_const
        prior_L_only.A.mult(x_const.petsc_vec, self.y1.petsc_vec)

        # 4. Check the norm of the result
        value = self.y1.petsc_vec.norm(PETSc.NormType.N2)

        self.assertLessEqual(np.abs(value), 1e-6, "Laplacian nullspace test failed (L*1 != 0)")


# class Testing_Execution(unittest.TestCase):
#     def test_prior_mass_matrix(self):
#         # --- 1. Setup the Mesh and Function Space ---

#         # Define the mesh resolution (10x10 elements)
#         nx, ny = 10, 10
#         # Define parameters for the BiLaplacian prior (regularization terms)
#         prior_param = {"gamma": 0.03, "delta": 0.3}

#         # Initialize the MPI communicator for parallel execution
#         comm = MPI.COMM_WORLD
#         # Create a 2D unit square mesh with quadrilateral cells
#         # msh = dlx.mesh.create_unit_square(comm, nx, ny, dlx.mesh.CellType.quadrilateral)
#         msh = dlx.mesh.create_unit_square(comm, nx, ny, dlx.mesh.CellType.triangle)
#         # Create a finite element function space (Standard P1 Lagrange elements)
#         # All our fields and vectors will live in this space.
#         Vh = dlx.fem.functionspace(msh, ("Lagrange", 1))
#         # Vh = dlx.fem.functionspace(msh, ("Discontinuous Lagrange", 0))

#         # --- 2. Initialize the Prior Object ---

#         # Instantiate a BiLaplacian prior object. This object will hold
#         # matrices related to the prior, including the mass matrix (M)
#         # and its square-root factor (sqrtM) based on the function space Vh.
#         prior = hpx.BiLaplacianPrior(Vh, prior_param["gamma"], prior_param["delta"], mean=None)
#         # breakpoint()
#         # --- 3. Set up Test Vectors ---

#         # Create a PETSc vector 'x' that is compatible with our function space
#         x = dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)

#         # Reset the parallel random number generator seed for reproducibility
#         hpx.parRandom.replay()
#         # Fill the vector 'x' with random values from a standard normal distribution.
#         # This will be our input test vector.
#         hpx.parRandom.normal(1.0, x)

#         # Allocate memory for output vectors
#         # y1 will store the result of (S * S^T) * x
#         y1 = dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)
#         # y2 will store the result of M * x
#         y2 = dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)
#         # tmp is a temporary vector for the intermediate step
#         tmp = dlx.la.vector(prior.Qh.dofmap.index_map, prior.Qh.dofmap.index_map_bs)

#         # breakpoint()

#         # --- 4. Perform Matrix-Vector Multiplications ---

#         # GOAL: Test if M*x is equal to (S * S^T)*x
#         # where S = prior.sqrtM and M = prior.M

#         # First, calculate y1 = (S * S^T) * x
#         # Step 4a: tmp = S^T * x
#         # (S.multTranspose(x, tmp) -> tmp = S_transpose * x)
#         prior.sqrtM.multTranspose(x.petsc_vec, tmp.petsc_vec)

#         # Step 4b: y1 = S * tmp  (which is S * (S^T * x))
#         # (S.mult(tmp, y1) -> y1 = S * tmp)
#         prior.sqrtM.mult(tmp.petsc_vec, y1.petsc_vec)

#         # Second, calculate y2 = M * x directly
#         # (M.mult(x, y2) -> y2 = M * x)
#         prior.M.mult(x.petsc_vec, y2.petsc_vec)

#         # --- 5. Compare Results and Assert ---

#         # Now, y1 should be equal to y2 if S*S^T = M.
#         # We check this by computing the difference vector: y2 - y1
#         # axpy(a, x, y) computes y = a*x + y
#         # So, y2.axpy(-1.0, y1) computes y2 = -1.0*y1 + y2
#         y2.petsc_vec.axpy(-1.0, y1.petsc_vec)

#         # Calculate the L2 norm (Euclidean length) of the difference vector (y2 - y1)
#         value = y2.petsc_vec.norm(PETSc.NormType.N2)

#         # Assert that the norm of the difference is less than or equal to 1e-6.
#         # This confirms that y1 and y2 are equal, within numerical tolerance.
#         self.assertLessEqual(
#             np.abs(value),  # The computed norm of the difference
#             1e-6,  # The tolerance for floating-point error
#             "prior_sqrtM creation failed",  # Error message if assertion fails
#         )


if __name__ == "__main__":
    unittest.main()
