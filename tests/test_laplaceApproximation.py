import unittest

from mpi4py import MPI

import dolfinx as dlx
import numpy as np

# --- Or just import via hippylib ---
import hippylibX as hpx


class TestPosterior(unittest.TestCase):
    def setUp(self):
        """
        Set up a complete, valid environment for testing the posterior.
        This involves:
        1. Creating a mesh and function space.
        2. Creating a 'prior' object (BiLaplacianPrior).
        3. Creating a 'MultiVector' U and making it R-orthogonal.
        4. Defining eigenvalues 'd'.
        5. Creating a 'mean' (MAP) vector.
        6. Instantiating the 'LaplaceApproximator' to be tested.
        """
        self.comm = MPI.COMM_WORLD
        self.nx, self.ny = 20, 20  # Small mesh for speed
        self.nvec = 3  # Small number of eigenvectors
        hpx.parRandom.replay()

        # 1. Create mesh and function space
        self.msh = dlx.mesh.create_unit_square(self.comm, self.nx, self.ny, dlx.mesh.CellType.triangle)
        self.Vh = dlx.fem.functionspace(self.msh, ("Lagrange", 1))

        # 2. Create the prior
        self.prior = hpx.BiLaplacianPrior(self.Vh, gamma=0.01, delta=0.01, mean=None)

        # 3. Create an R-orthogonal MultiVector U
        # We start with random vectors...
        tmp = self.prior.generate_parameter(0)
        U_temp = hpx.MultiVector.createFromVec(tmp.petsc_vec, self.nvec)
        hpx.parRandom.normal(1.0, U_temp)

        # ...and then use Borthogonalize with B=prior.R to make them R-orthogonal.
        # This modifies U_temp in-place.
        U_temp.Borthogonalize(self.prior.R)
        self.U = U_temp

        # 4. Define eigenvalues
        self.d = np.array([10.0, 5.0, 2.0], dtype=np.float64)

        # 5. Create a mean (MAP) vector (e.g., a vector of all ones)
        self.mean = self.prior.generate_parameter(0)
        self.mean.petsc_vec.set(1.0)

        # 6. Instantiate the main posterior object
        self.posterior = hpx.LaplaceApproximator(self.prior, self.d, self.U, self.mean)

        # 7. Create utility vectors for testing
        self.x_rand = self.prior.generate_parameter(0)
        hpx.parRandom.normal(1.0, self.x_rand)  # A random vector in Vh
        self.y1 = self.prior.generate_parameter(0)
        self.y2 = self.prior.generate_parameter(0)
        self.tmp1 = self.prior.generate_parameter(0)
        self.tmp2 = self.prior.generate_parameter(0)

    def tearDown(self):
        """Clean up PETSc objects."""
        # The __del__ methods in the classes should handle most of this,
        # but it's good practice.
        self.prior.M.destroy()
        self.prior.A.destroy()
        self.x_rand.petsc_vec.destroy()
        self.y1.petsc_vec.destroy()
        self.y2.petsc_vec.destroy()
        self.tmp1.petsc_vec.destroy()
        self.tmp2.petsc_vec.destroy()
        self.mean.petsc_vec.destroy()

    # --- Tests for LowRankHessian ---

    def test_hessian_mult(self):
        """
        Tests the LowRankHessian 'mult' method.
        Verifies: y = (R + R*U*D*U^T*R) * x
        """
        Hlr = self.posterior.Hlr
        R = self.prior.R
        x_vec = self.x_rand.petsc_vec

        # 1. Compute y1 = Hlr.mult(x)
        y1_vec = self.y1.petsc_vec
        Hlr.mult(x_vec, y1_vec)

        # 2. Compute y2 = (R + R*U*D*U^T*R) * x manually
        y2_vec = self.y2.petsc_vec
        R.mult(x_vec, y2_vec)  # y2 = R*x

        lr_part_in = self.tmp1.petsc_vec.copy()
        R.mult(x_vec, lr_part_in)  # lr_part_in = R*x

        lr_part_out = self.tmp2.petsc_vec
        Hlr.LowRankH.mult(lr_part_in, lr_part_out)  # lr_part_out = U*D*U^T*(R*x)

        R_lr_part_out = self.tmp1.petsc_vec
        R.mult(lr_part_out, R_lr_part_out)  # R_lr_part_out = R*(U*D*U^T*R*x)

        y2_vec.axpy(1.0, R_lr_part_out)  # y2 = R*x + R*U*D*U^T*R*x

        # 3. Compare y1 and y2
        y1_vec.axpy(-1.0, y2_vec)
        self.assertLess(y1_vec.norm(), 1e-6)

        lr_part_in.destroy()

    def test_hessian_solve_is_inverse(self):
        """
        Tests if LowRankHessian 'solve' is the inverse of 'mult'.
        Verifies: Hlr.solve(Hlr.mult(x)) == x
        """
        Hlr = self.posterior.Hlr
        x_vec = self.x_rand.petsc_vec

        y_vec = self.y1.petsc_vec  # y = H*x
        z_vec = self.y2.petsc_vec  # z = H_inv*y

        # 1. y = H*x
        Hlr.mult(x_vec, y_vec)

        # 2. z = H_inv*y
        Hlr.solve(y_vec, z_vec)

        # 3. Compare z and x
        z_vec.axpy(-1.0, x_vec)
        self.assertLess(z_vec.norm() / x_vec.norm(), 1e-5)  # Use relative tolerance

    # # --- Tests for LowRankPosteriorSampler ---

    def test_sampler_mult(self):
        """
        Tests the LowRankPosteriorSampler 'mult' method.
        Verifies: s_post = (I - U*S*U^T*R) * s_prior
        """
        sampler = self.posterior.sampler
        R = self.prior.R
        s_prior_vec = self.x_rand.petsc_vec

        # 1. Compute s_post_1 = sampler.mult(s_prior)
        s_post1_vec = self.y1.petsc_vec
        sampler.mult(self.x_rand, self.y1)  # Note: takes dlx.la.Vector

        # 2. Compute s_post_2 = (I - U*S*U^T*R) * s_prior manually
        s_post2_vec = self.y2.petsc_vec

        tmp_R_s_prior = self.tmp1.petsc_vec
        R.mult(s_prior_vec, tmp_R_s_prior)  # tmp_R_s_prior = R * s_prior

        tmp_lr_part = self.tmp2.petsc_vec
        sampler.lrsqrt.mult(tmp_R_s_prior, tmp_lr_part)  # tmp_lr_part = (U*S*U^T) * (R*s_prior)

        s_post2_vec.set(0.0)
        s_post2_vec.axpy(1.0, s_prior_vec)  # s_post2 = s_prior
        s_post2_vec.axpy(-1.0, tmp_lr_part)  # s_post2 = s_prior - (U*S*U^T*R*s_prior)

        # 3. Compare s_post1 and s_post2
        s_post1_vec.axpy(-1.0, s_post2_vec)
        self.assertLess(s_post1_vec.norm(), 1e-6)

    # # --- Tests for LaplaceApproximator ---

    def test_posterior_sample_conventions(self):
        """
        Tests that both 'sample' calling conventions are equivalent.
        1. sample(noise, s_prior, s_post)
        2. sample(prior.sample(noise), s_post)
        """
        # 1. Create white noise
        noise = self.prior.generate_parameter("noise")
        hpx.parRandom.normal(1.0, noise)

        s_prior_1 = self.prior.generate_parameter(0)
        s_post_1 = self.prior.generate_parameter(0)

        s_prior_2 = self.prior.generate_parameter(0)
        s_post_2 = self.prior.generate_parameter(0)

        # 2. Path 1: sample(noise, s_prior, s_post)
        self.posterior.sample(noise, s_prior_1, s_post_1, add_mean=False)

        # 3. Path 2: sample(s_prior, s_post)
        #    First, generate the *same* prior sample from the *same* noise
        self.prior.sample(noise, s_prior_2, add_mean=False)
        #    Then, call the 2-argument sample
        self.posterior.sample(s_prior_2, s_post_2, add_mean=False)

        # 4. Compare results
        # Check priors are identical
        s_prior_1.petsc_vec.axpy(-1.0, s_prior_2.petsc_vec)
        self.assertLess(s_prior_1.petsc_vec.norm(), 1e-12)

        # Check posteriors are identical
        s_post_1.petsc_vec.axpy(-1.0, s_post_2.petsc_vec)
        self.assertLess(s_post_1.petsc_vec.norm(), 1e-12)

        noise.petsc_vec.destroy()

    def test_posterior_sample_add_mean(self):
        """
        Tests that 'add_mean=True' correctly adds the posterior mean.
        """
        s_prior = self.x_rand

        # self.y1 will hold the sample *without* the mean
        self.posterior.sample(s_prior, self.y1, add_mean=False)

        # self.y2 will hold the sample *with* the mean
        self.posterior.sample(s_prior, self.y2, add_mean=True)

        # Now, check if self.y2 = self.y1 + self.mean
        # We do this by checking if (self.y2 - self.y1 - self.mean) is zero.

        # y2 = y2 - y1
        self.y2.petsc_vec.axpy(-1.0, self.y1.petsc_vec)

        # y2 = y2 - mean
        self.y2.petsc_vec.axpy(-1.0, self.mean.petsc_vec)

        self.assertLess(self.y2.petsc_vec.norm(), 1e-12)

    def test_kl_divergence(self):
        """
        Tests the klDistanceFromPrior calculation against a manual NumPy computation.
        """
        kld, logdet, trace, shift = self.posterior.klDistanceFromPrior(sub_comp=True)

        # Manual calculation based on the formulas
        d = self.d
        dplus1 = d + 1.0

        logdet_expected = 0.5 * np.sum(np.log(dplus1))
        trace_expected = -0.5 * np.sum(d / dplus1)
        shift_expected = self.prior.cost(self.mean)
        kld_expected = logdet_expected + trace_expected + shift_expected

        self.assertTrue(np.isclose(logdet, logdet_expected))
        self.assertTrue(np.isclose(trace, trace_expected))
        self.assertTrue(np.isclose(shift, shift_expected))
        self.assertTrue(np.isclose(kld, kld_expected))

    def test_cost_calculation(self):
        """
        Tests the posterior.cost(m) calculation.
        Verifies cost = 0.5 * <m-mean, H*(m-mean)>
        This explicitly validates the fix for the `dm = m - self.mean` bug.
        """
        # 1. Get the cost from the (fixed) method
        # We use self.x_rand as our input vector 'm'
        cost_val = self.posterior.cost(self.x_rand)

        # 2. Calculate the expected value manually using parallel-safe operations

        # Manually compute dm = m - mean
        dm_petsc = self.x_rand.petsc_vec.copy()
        dm_petsc.axpy(-1.0, self.mean.petsc_vec)

        # Manually compute H_dm = H * dm
        H_dm = self.prior.generate_parameter(0)
        H_dm_petsc = H_dm.petsc_vec
        self.posterior.Hlr.mult(dm_petsc, H_dm_petsc)

        # Manually compute 0.5 * <H_dm, dm>
        expected_val = 0.5 * H_dm_petsc.dot(dm_petsc)

        # 3. Compare the method's result to the manual, correct calculation
        self.assertTrue(
            np.isclose(cost_val, expected_val),
            f"Cost value {cost_val} does not match expected {expected_val}",
        )

        # 4. Clean up temporary vectors
        dm_petsc.destroy()
        H_dm_petsc.destroy()


if __name__ == "__main__":
    unittest.main()
