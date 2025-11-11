# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

import petsc4py
import petsc4py.PETSc

import dolfinx as dlx
import numpy as np

from ..algorithms.lowRankOperator import LowRankOperator
from ..algorithms.multivector import MultiVector


def not_implemented(func):
    """
    Decorator to raise a NotImplementedError, marking a function as incomplete.
    """

    def wrapper(*args, **kwargs):
        raise NotImplementedError(f"{func.__name__} Function has not been implemented.")

    return wrapper


class LowRankHessian:
    r"""
    A matrix-free operator for the low-rank posterior Hessian and its inverse.

    This class provides `.mult()` and `.solve()` methods that correspond
    to the posterior Hessian (H) and posterior covariance (C = H^{-1}),
    respectively, using the low-rank approximation.

    The formulas are derived from the Sherman-Morrison-Woodbury identity,
    assuming R-orthogonal eigenvectors U (where $U^T R U = I$).

    - **Hessian Apply (`.mult`):** $H x = (R + R U D U^T R) x$

    - **Covariance Apply (`.solve`):** $C x = H^{-1} x = (R^{-1} - U (I + D)^{-1} D U^T) x$
    """

    def __init__(self, prior, d: np.array, U: MultiVector):
        """
        Initializes the Hessian and Covariance operators.

        Args:
            prior: The prior object, providing R (precision) and Rsolver (covariance).
            d: A NumPy array of dominant eigenvalues (D).
            U: A MultiVector of R-orthogonal eigenvectors (U).
        """
        self.U = U
        self.prior = prior

        # LowRankH = U * D * U^T
        self.LowRankH = LowRankOperator(d, self.U)

        # Eigenvalues for the inverse operator: d_solve = D * (I + D)^{-1}
        dsolve = d / (np.ones(d.shape, dtype=d.dtype) + d)

        # LowRankHinv = U * (D * (I + D)^{-1}) * U^T
        self.LowRankHinv = LowRankOperator(dsolve, self.U)

        # Pre-allocate temporary vectors for efficient computation
        self.help = self.prior.R.createVecRight()
        self.help1 = self.prior.R.createVecLeft()

    def __del__(self) -> None:
        """Cleans up PETSc vectors."""
        self.help.destroy()
        self.help1.destroy()

    def createVecRight(self) -> petsc4py.PETSc.Vec:
        """Creates a vector compatible with the operator's domain."""
        return self.prior.R.createVecRight()

    def createVecLeft(self) -> petsc4py.PETSc.Vec:
        """Creates a vector compatible with the operator's range."""
        return self.prior.R.createVecLeft()

    def mult(self, x: petsc4py.PETSc.Vec, y: petsc4py.PETSc.Vec) -> None:
        """
        Performs the Hessian-vector product: y = H*x = (R + R*U*D*U^T*R) * x

        .. warning::
           This method is not safe for aliased vectors (e.g., mult(x, x)).
           The output vector 'y' must not be one of the internal temporary
           vectors (self.help, self.help1).
        """
        # 1. y = R*x
        self.prior.R.mult(x, y)

        # 2. self.help = (U*D*U^T) * y = (U*D*U^T*R) * x
        self.LowRankH.mult(y, self.help)

        # 3. self.help1 = R * self.help = (R*U*D*U^T*R) * x
        self.prior.R.mult(self.help, self.help1)

        # 4. y = y + self.help1 = (R*x) + (R*U*D*U^T*R*x)
        y.axpy(1, self.help1)

    def solve(self, rhs: petsc4py.PETSc.Vec, sol: petsc4py.PETSc.Vec) -> None:
        """
        Performs the covariance-vector product (Hessian solve):
        sol = H^{-1}*rhs = (R^{-1} - U*(D*(I+D)^{-1})*U^T) * rhs
        """
        # 1. sol = R^{-1} * rhs (the prior covariance)
        self.prior.Rsolver.solve(rhs, sol)

        # 2. self.help = (U * (D*(I+D)^{-1}) * U^T) * rhs
        self.LowRankHinv.mult(rhs, self.help)

        # 3. sol = sol - self.help
        sol.axpy(-1, self.help)


class LowRankPosteriorSampler:
    r"""
    Implements the transformation from a prior sample to a posterior sample.

    The applied formula is:
    $s_{\text{post}} = (I - U S U^T R) s_{\text{prior}}$

    where $S = I - (I + D)^{-1/2}$ and $s_{\text{prior}} \sim \mathcal{N}(0, R^{-1})$.

    .. warning::
       The implementation of `.mult()` in this class computes
       $s_{\text{post}} = (U S U^T R) s_{\text{prior}} - s_{\text{prior}}$,
       which is the *negative* of the correct formula. The legacy `dolfin`
       code fixed this with a final `s *= -1.0` step, which is missing here.
       This is corrected in the `LaplaceApproximator.sample` methods by
       using the correct `axpby(1.0, -1.0, ...)` call.
    """

    def __init__(
        self,
        prior,
        d: np.array,
        U: MultiVector,
    ):
        """
        Initializes the sampling operator.

        Args:
            prior: The prior object, providing R (precision).
            d: A NumPy array of dominant eigenvalues (D).
            U: A MultiVector of R-orthogonal eigenvectors (U).
        """
        self.U = U
        self.prior = prior

        # 1. Define the sampler eigenvalues: S = I - (I + D)^(-1/2)
        ones = np.ones(d.shape, dtype=d.dtype)
        self.d = ones - np.power(ones + d, -0.5)

        # 2. Create the low-rank operator: U * S * U^T
        self.lrsqrt = LowRankOperator(self.d, self.U)

        # 3. Pre-allocate temporary vector
        self.help = self.prior.R.createVecLeft()

    def __del__(self) -> None:
        """Cleans up PETSc vector."""
        self.help.destroy()

    def createVecRight(self) -> petsc4py.PETSc.Vec:
        """Creates a vector compatible with the operator's domain."""
        return self.prior.R.createVecRight()

    def createVecLeft(self) -> petsc4py.PETSc.Vec:
        """Creates a vector compatible with the operator's range."""
        return self.prior.R.createVecLeft()

    def mult(self, noise: dlx.la.Vector, s: dlx.la.Vector):
        """
        Applies the posterior sampling transformation.

        Args:
            noise: The input *prior sample* (s_prior), not white noise.
            s: The output *posterior sample* (s_post).
        """
        # 1. self.help = R * noise (where noise is s_prior)
        self.prior.R.mult(noise.petsc_vec, self.help)

        # 2. s = (U*S*U^T) * self.help = (U*S*U^T*R) * noise
        self.lrsqrt.mult(self.help, s.petsc_vec)

        # 3. s = 1.0*s - 1.0*noise = (U*S*U^T*R)*noise - noise
        s.petsc_vec.axpby(1.0, -1.0, noise.petsc_vec)


class LaplaceApproximator:
    r"""
    Main class for the low-rank Gaussian (Laplace) Approximation of the Posterior.

    This class wraps the `LowRankHessian` and `LowRankPosteriorSampler`
    to provide a user-facing API for computing costs, drawing samples,
    and calculating the KL divergence from the prior.

    It assumes a posterior distribution $\mathcal{N}(\mu, C)$, where
    $\mu$ is the `mean` (MAP point) and $C = H^{-1}$.

    The mathematical formulas implemented are:

    - **Hessian Apply:** $H x = (R + R U D U^T R) x$

    - **Covariance Apply (Hessian Solve):** $C x = H^{-1} x = (R^{-1} - U (I + D)^{-1} D U^T) x$

    - **Posterior Sample (from prior sample $x_p$):** $x_q = \mu + (I - U S U^T R) (x_p - \mu_p)$
      (where $S = I - (I + D)^{-1/2}$ and $\mu_p$ is the prior mean)
    """

    def __init__(self, prior, d: np.array, U: MultiVector, mean: dlx.la.Vector | None = None):
        """
        Construct the Gaussian approximation of the posterior.

        Args:
            prior: The prior object.
            d: Dominant generalized eigenvalues of the Hessian misfit (D).
            U: Dominant generalized eigenvectors (U), assumed to be
               R-orthogonal ($U^T R U = I$).
            mean: The MAP point (posterior mean). If None, a zero vector is used.
        """
        self.prior = prior
        self.d = d
        self.U = U

        # Initialize the Hessian/Covariance operator
        self.Hlr = LowRankHessian(prior, d, U)

        # Initialize the prior-to-posterior sample transformer
        self.sampler = LowRankPosteriorSampler(self.prior, self.d, self.U)

        # Set the mean (MAP point) using the property setter
        self.mean = mean

    @property
    def mean(self) -> dlx.la.Vector:
        """Gets the posterior mean (MAP point) vector."""
        return self._mean

    @mean.setter
    def mean(self, value: dlx.la.Vector | None) -> None:
        """
        Sets the posterior mean vector.
        If 'None' is provided, it initializes a zero vector of the correct size.
        """
        if value is None:
            # Create a zero vector in the parameter space
            self._mean = self.prior.generate_parameter(0)
            self._mean.petsc_vec.set(0.0)
        else:
            self._mean = value

    def cost(self, m: dlx.la.Vector) -> float:
        r"""
        Computes the cost (negative log-likelihood) of a state 'm'.

        Formula: $J(m) = 0.5 \cdot \langle m - \mu, H \cdot (m - \mu) \rangle$

        Args:
            m: The state vector (parameter) to evaluate.

        Returns:
            The computed cost (a float).
        """
        # 1. Compute dm = m - self.mean (parallel-safe)
        #    We must create a *copy* to not modify the input 'm'
        dm_petsc = m.petsc_vec.copy()
        dm_petsc.axpy(-1.0, self.mean.petsc_vec)

        # 2. Create a *new* temporary vector for the Hessian product.
        #    This is crucial to avoid vector aliasing bugs, as Hlr.mult
        #    uses its own internal temporary vectors.
        H_dm_petsc = self.Hlr.createVecRight()

        # 3. Compute H_dm = H * dm
        self.Hlr.mult(dm_petsc, H_dm_petsc)

        # 4. Compute cost = 0.5 * <H_dm, dm>
        return_value = 0.5 * H_dm_petsc.dot(dm_petsc)

        # 5. Clean up all temporary vectors created in this method
        dm_petsc.destroy()
        H_dm_petsc.destroy()

        return return_value

    def sample(self, *args, **kwargs):
        """
        Draws a sample from the low-rank posterior distribution.

        This method has two possible call signatures:

        **Signature 1: `sample(s_prior, s_post, add_mean=True)`**
           Transforms a given prior sample into a posterior sample.
           - `s_prior`: A `dlx.la.Vector` (input) from the prior, centered at 0.
           - `s_post`: A `dlx.la.Vector` (output) for the posterior sample.
           - `add_mean`: If True, adds the posterior mean (`self.mean`) to `s_post`.

        **Signature 2: `sample(noise, s_prior, s_post, add_mean=True)`**
           Generates a prior sample from white noise, then transforms it.
           - `noise`: A `dlx.la.Vector` (input) of white noise, typically
                      from the `prior.generate_parameter("noise")` space.
           - `s_prior`: A `dlx.la.Vector` (output) for the prior sample.
           - `s_post`: A `dlx.la.Vector` (output) for the posterior sample.
           - `add_mean`: If True, adds `prior.mean` to `s_prior` and
                         `self.mean` to `s_post`.
        """
        # Parse 'add_mean' from keyword arguments, defaulting to True
        add_mean = True
        for name, value in kwargs.items():
            if name == "add_mean":
                add_mean = value
            else:
                raise NameError(name)

        # --- Handle 2-argument signature: sample(s_prior, s_post) ---
        if len(args) == 2:
            self._sample_given_prior(args[0], args[1])
            if add_mean:
                # Use parallel-safe .axpy() for vector addition
                args[1].petsc_vec.axpy(1.0, self.mean.petsc_vec)

        # --- Handle 3-argument signature: sample(noise, s_prior, s_post) ---
        elif len(args) == 3:
            self._sample_given_white_noise(args[0], args[1], args[2])
            if add_mean:
                # Use parallel-safe .axpy() for vector addition
                args[1].petsc_vec.axpy(1.0, self.prior.mean.petsc_vec)
                args[2].petsc_vec.axpy(1.0, self.mean.petsc_vec)

        else:
            raise NameError("Invalid number of parameters in Posterior::sample")

    def _sample_given_white_noise(
        self,
        noise: dlx.la.Vector,
        s_prior: dlx.la.Vector,
        s_post: dlx.la.Vector,
    ):
        """Internal helper for the 3-argument sample call."""
        # 1. Generate prior sample from noise (s_prior = C_prior^{1/2} * noise)
        self.prior.sample(noise, s_prior, add_mean=False)
        # 2. Transform prior sample to posterior sample
        self.sampler.mult(s_prior, s_post)

    def _sample_given_prior(self, s_prior: dlx.la.Vector, s_post: dlx.la.Vector):
        """Internal helper for the 2-argument sample call."""
        # 1. Transform prior sample to posterior sample
        self.sampler.mult(s_prior, s_post)

    @not_implemented
    def trace(self, **kwargs):
        """
        Compute/estimate the trace of the posterior covariance matrix.

        This should compute: Tr(C_post) = Tr(C_prior) - Tr(C_correction)
        """
        pr_trace = self.prior.trace(**kwargs)
        corr_trace = self.trace_update()
        post_trace = pr_trace - corr_trace
        return post_trace, pr_trace, corr_trace

    @not_implemented
    def trace_update(self):
        """Computes the trace of the low-rank correction term."""
        return self.Hlr.LowRankHinv.trace(self.prior.M)

    @not_implemented
    def pointwise_variance(self, **kwargs):
        """
        Compute/estimate the pointwise variance (diagonal of the covariance).

        This should compute:
        diag(C_post) = diag(C_prior) - diag(C_correction)
        """
        pr_pointwise_variance = self.prior.pointwise_variance(**kwargs)
        # correction_pointwise_variance = Vector(self.prior.R.mpi_comm())
        correction_pointwise_variance = None
        self.init_vector(correction_pointwise_variance, 0)
        self.Hlr.LowRankHinv.get_diagonal(correction_pointwise_variance)
        post_pointwise_variance = pr_pointwise_variance - correction_pointwise_variance
        return (
            post_pointwise_variance,
            pr_pointwise_variance,
            correction_pointwise_variance,
        )

    def klDistanceFromPrior(self, sub_comp=False):
        r"""
        Computes the KL divergence from the posterior to the prior.

        $KL(\mathcal{N}_q || \mathcal{N}_p) = 0.5 \cdot (\log\det(C_p C_q^{-1})
         + \text{Tr}(C_p^{-1} C_q) + \delta\mu^T C_p^{-1} \delta\mu - k)$

        where:
        - $C_p = R^{-1}$ (Prior Covariance)
        - $C_q = H^{-1}$ (Posterior Covariance)
        - $C_p^{-1} = R$ (Prior Precision)
        - $k$ = dimension of the space
        - $\delta\mu = \mu_q - \mu_p = \text{self.mean} - 0$ (assuming prior mean is zero)

        Args:
            sub_comp (bool): If True, returns the individual components
                             of the KL divergence.

        Returns:
            float or tuple: The total KL divergence, or (kld, logdet, trace, shift).
        """
        # D + I
        dplus1 = self.d + np.ones_like(self.d)

        # c_logdet = 0.5 * log(det(C_p * C_q^{-1})) = 0.5 * log(det(R * H))
        #          = 0.5 * log(det(I + D))
        c_logdet = 0.5 * np.sum(np.log(dplus1))

        # c_trace = 0.5 * (Tr(C_p^{-1} * C_q) - k) = 0.5 * (Tr(R * H^{-1}) - k)
        #         = 0.5 * (Tr(I - D(I+D)^{-1}) - k) = -0.5 * Tr(D(I+D)^{-1})
        c_trace = -0.5 * np.sum(self.d / dplus1)

        # c_shift = 0.5 * (\delta\mu^T C_p^{-1} \delta\mu)
        #         = 0.5 * <mean, R * mean>
        c_shift = self.prior.cost(self.mean)

        # Total KL divergence
        kld = c_logdet + c_trace + c_shift

        if sub_comp:
            return kld, c_logdet, c_trace, c_shift
        else:
            return kld
