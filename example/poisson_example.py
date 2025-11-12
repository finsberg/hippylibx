# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""
End-to-end example of a Bayesian inverse problem for the Poisson equation.

This script demonstrates solving a finite-dimensional Bayesian inverse problem
governed by the Poisson equation with Robin boundary conditions. The goal
is to infer the spatially varying log-diffusion coefficient 'm'.

The workflow is as follows:
1.  Define the forward PDE and misfit (likelihood) UFL forms.
2.  Set up the problem: create a "true" parameter 'm_true' and generate
    synthetic, noisy observation data 'd'.
3.  Define the prior (BiLaplacian) and the full Bayesian 'Model'.
4.  Solve for the Maximum A Posteriori (MAP) point using a Newton-CG solver.
5.  Compute the low-rank Laplace approximation of the posterior by solving
    a generalized eigenproblem for the Hessian of the misfit.
6.  Draw samples from the resulting approximate posterior and prior.
7.  Save results (MAP point, samples, etc.) to VTK/XDMF files.
"""

from pathlib import Path
from typing import Dict

from mpi4py import MPI

import dolfinx as dlx
import dolfinx.fem.petsc
import numpy as np
import ufl
from matplotlib import pyplot as plt

import hippylibX as hpx


class Poisson_Approximation:
    """
    Functor class defining the weak form (residual) of the Poisson PDE.

    This class defines the forward problem:
    -div(exp(m) * grad(u)) = f  in Omega
    exp(m) * du/dn + alpha*u = 0  on dOmega
    """

    def __init__(self, alpha: float, f: float):
        """
        Initializes the PDE parameters.

        Args:
            alpha: The Robin boundary condition coefficient.
            f: The source term.
        """
        self.alpha = alpha
        self.f = f
        self.dx = ufl.Measure("dx", metadata={"quadrature_degree": 4})
        self.ds = ufl.Measure("ds", metadata={"quadrature_degree": 4})

    def __call__(
        self,
        u: dlx.fem.Function,
        m: dlx.fem.Function,
        p: dlx.fem.Function,
    ) -> ufl.form.Form:
        """
        Returns the UFL weak form (residual) of the PDE.

        This is the "F(u, m; p) = 0" form required by hIPPYlibx.

        Args:
            u: The state variable (Function from Vh_phi).
            m: The parameter variable (Function from Vh_m).
            p: The test function (Function from Vh_phi).

        Returns:
            The UFL form for the PDE residual.
        """
        return (
            ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * self.dx  # Diffusion term
            + self.alpha * ufl.inner(u, p) * self.ds  # Robin BC term
            - self.f * p * self.dx  # Source term
        )


class PoissonMisfitForm:
    """
    Functor class defining the weak form of the data misfit (log-likelihood).

    This represents the cost functional:
    J(u) = 1 / (2*sigma^2) * integral( (u - d)^2 ) dx
    """

    def __init__(self, d: float, sigma2: float):
        """
        Initializes the misfit parameters.

        Args:
            d: The observation data.
            sigma2: The noise variance.
        """
        self.d = d
        self.sigma2 = sigma2
        self.dx = ufl.Measure("dx", metadata={"quadrature_degree": 4})

    def __call__(self, u: dlx.fem.Function, m: dlx.fem.Function) -> ufl.form.Form:
        """
        Returns the UFL weak form for the misfit.

        Args:
            u: The state variable (solution of the PDE).
            m: The parameter variable (unused in this form, but required by API).

        Returns:
            The UFL form for the misfit functional.
        """
        return 0.5 / self.sigma2 * ufl.inner(u - self.d, u - self.d) * self.dx


def run_inversion(
    nx: int,
    ny: int,
    noise_variance: float,
    prior_param: Dict[str, float],
    outdir: Path = Path("results-poisson-example"),
) -> Dict[str, Dict[str, float]]:
    """
    Executes the full end-to-end Bayesian inversion workflow.

    Args:
        nx: Number of elements in the x-direction for the mesh.
        ny: Number of elements in the y-direction for the mesh.
        noise_variance: Variance (sigma^2) of the observation noise.
        prior_param: Dictionary of prior parameters (e.g., "gamma", "delta").
        outdir: Output directory for saving results.

    Returns:
        A dictionary containing key results from the inversion,
        including model verification, optimizer status, and eigen-decomposition.
    """
    sep = "\n" + "#" * 80 + "\n"
    comm = MPI.COMM_WORLD
    rank = comm.rank
    nproc = comm.size

    # --- 1. SET UP MPI, MESH, AND FUNCTION SPACES ---
    msh = dlx.mesh.create_unit_square(comm, nx, ny)

    # State and Adjoint space: P2 Lagrange elements
    Vh_phi = dlx.fem.functionspace(msh, ("Lagrange", 2))
    # Parameter space: P1 Lagrange elements
    Vh_m = dlx.fem.functionspace(msh, ("Lagrange", 1))

    # hIPPYlibx requires a list of [State, Parameter, Adjoint] spaces
    Vh = [Vh_phi, Vh_m, Vh_phi]

    ndofs = [
        Vh_phi.dofmap.index_map.size_global * Vh_phi.dofmap.index_map_bs,
        Vh_m.dofmap.index_map.size_global * Vh_m.dofmap.index_map_bs,
    ]
    hpx.master_print(comm, sep, "Set up the mesh and finite element spaces", sep)
    hpx.master_print(comm, "Number of dofs: STATE={0}, PARAMETER={1}".format(*ndofs))

    # --- 2. DEFINE THE FORWARD PDE PROBLEM ---
    alpha = 100.0  # Robin BC coefficient
    f = 1.0  # Source term
    pde_handler = Poisson_Approximation(alpha, f)
    # Encapsulate the UFL form in a hIPPYlibx PDE problem
    pde = hpx.PDEVariationalProblem(Vh, pde_handler, [], [], is_fwd_linear=True)

    # --- 3. CREATE GROUND TRUTH AND SYNTHETIC DATA ---
    m_true = dlx.fem.Function(Vh_m)
    # Define a true parameter field (a circular inclusion)
    m_true.interpolate(
        lambda x: np.log(2 + 7 * (((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) ** 0.5 > 0.2)),
    )
    m_true.x.scatter_forward()  # Synchronize parallel vector
    m_true = m_true.x  # Get the underlying dolfinx.la.Vector

    # Solve the PDE to get the "true" state
    u_true = pde.generate_state()
    x_true = [u_true, m_true, None]  # [state, parameter, adjoint]
    pde.solveFwd(u_true, x_true)

    # --- 4. DEFINE THE LIKELIHOOD (MISFIT) ---
    # Create synthetic data 'd' by copying the true state...
    d = dlx.fem.Function(Vh[hpx.STATE])
    d.x.array[:] = u_true.array[:]
    # ...and adding random noise
    hpx.parRandom.normal_perturb(np.sqrt(noise_variance), d.x)
    d.x.scatter_forward()

    # Create the misfit handler and encapsulate it
    misfit_form = PoissonMisfitForm(d, noise_variance)
    misfit = hpx.NonGaussianContinuousMisfit(Vh, misfit_form)

    # --- 5. DEFINE THE PRIOR AND FULL MODEL ---
    # Set a constant prior mean
    prior_mean = dlx.fem.Function(Vh_m)
    prior_mean.x.array[:] = np.log(2)
    prior_mean = prior_mean.x  # Get the underlying dolfinx.la.Vector

    # Create a BiLaplacian prior
    prior = hpx.BiLaplacianPrior(Vh_m, prior_param["gamma"], prior_param["delta"], mean=prior_mean)

    # Combine PDE, Prior, and Misfit into a single Model object
    model = hpx.Model(pde, prior, misfit)

    # --- 6. (OPTIONAL) VERIFY THE MODEL (e.g., GRADIENT CHECK) ---
    # This is a sanity check to verify the gradient implementation
    noise = prior.generate_parameter("noise")
    m0 = prior.generate_parameter(0)
    hpx.parRandom.normal(1.0, noise)
    prior.sample(noise, m0)  # Draw a random sample from the prior

    # Run hIPPYlibx's built-in verification tool
    data_misfit_True = hpx.modelVerify(
        model,
        m0,
        is_quadratic=False,
        misfit_only=True,  # Check gradient of misfit only
        verbose=(rank == 0),
    )

    data_misfit_False = hpx.modelVerify(
        model,
        m0,
        is_quadratic=False,
        misfit_only=False,  # Check gradient of the full cost (misfit + prior)
        verbose=(rank == 0),
    )

    # # #######################################

    # --- 7. SOLVE FOR THE MAP POINT (INVERSION) ---
    # Set the initial guess for the optimization to the prior mean
    initial_guess_m = prior.generate_parameter(0)
    initial_guess_m.array[:] = prior_mean.array[:]

    # x is the composite vector [state, parameter, adjoint]
    x = [
        model.generate_vector(hpx.STATE),
        initial_guess_m,
        model.generate_vector(hpx.ADJOINT),
    ]
    if rank == 0:
        print(sep, "Find the MAP point", sep)

    # Set parameters for the Newton-CG solver
    parameters = hpx.ReducedSpaceNewtonCG_ParameterList()
    parameters["rel_tolerance"] = 1e-6
    parameters["abs_tolerance"] = 1e-9
    parameters["max_iter"] = 500
    parameters["cg_coarse_tolerance"] = 5e-1
    parameters["globalization"] = "LS"  # Use Line Search
    parameters["GN_iter"] = 20  # Use Gauss-Newton approximation for 20 iters
    if rank != 0:
        parameters["print_level"] = -1  # Suppress output on non-master ranks

    # Create the solver and run it
    solver = hpx.ReducedSpaceNewtonCG(model, parameters)
    x = solver.solve(x)

    # Print optimization results
    if solver.converged:
        hpx.master_print(comm, "\nConverged in ", solver.it, " iterations.")
    else:
        hpx.master_print(comm, "\nNot Converged")

    hpx.master_print(comm, "Termination reason: ", solver.termination_reasons[solver.reason])
    hpx.master_print(comm, "Final gradient norm: ", solver.final_grad_norm)
    hpx.master_print(comm, "Final cost: ", solver.final_cost)

    # --- 8. POST-PROCESS AND SAVE MAP RESULTS ---
    # Convert the resulting vectors back to dolfinx Functions for saving
    m_fun = hpx.vector2Function(x[hpx.PARAMETER], Vh[hpx.PARAMETER], name="m_map")
    m_true_fun = hpx.vector2Function(m_true, Vh[hpx.PARAMETER], name="m_true")

    # Project state (P2) to P1 for lighter visualization files
    V_P1 = dlx.fem.functionspace(msh, ("Lagrange", 1))

    u_true_fun = dlx.fem.Function(V_P1, name="u_true")
    u_true_fun.interpolate(hpx.vector2Function(u_true, Vh[hpx.STATE]))
    u_true_fun.x.scatter_forward()

    u_map_fun = dlx.fem.Function(V_P1, name="u_map")
    u_map_fun.interpolate(hpx.vector2Function(x[hpx.STATE], Vh[hpx.STATE]))
    u_map_fun.x.scatter_forward()

    d_fun = dlx.fem.Function(V_P1, name="data")
    d_fun.interpolate(d)
    d_fun.x.scatter_forward()

    # Save MAP point, true parameter, and data to a .bp file (VTK)
    with dlx.io.VTXWriter(
        msh.comm,
        "poisson_Robin_BiLaplacian_prior_np{0:d}_Prior.bp".format(nproc),
        [m_fun, m_true_fun, u_map_fun, u_true_fun, d_fun],
    ) as vtx:
        vtx.write(0.0)

    # Store optimizer results
    optimizer_results = {}
    if solver.termination_reasons[solver.reason] == "Norm of the gradient less than tolerance":
        optimizer_results["optimizer"] = True
    else:
        optimizer_results["optimizer"] = False

    # --- 9. COMPUTE LOW-RANK HESSIAN APPROXIMATION ---
    # Define the Hessian of the misfit
    Hmisfit = hpx.ReducedHessian(model, misfit_only=True)

    # Set parameters for the randomized eigensolver
    k = 80  # Number of eigenvalues to find
    p = 20  # Oversampling
    if rank == 0:
        print("Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k, p))

    # Create a MultiVector of random vectors for the eigensolver
    Omega = hpx.MultiVector(x[hpx.PARAMETER].petsc_vec, k + p)
    hpx.parRandom.normal(1.0, Omega)

    # Solve the generalized eigenproblem: Hmisfit*U = R*U*D
    # 'd' = eigenvalues, 'U' = eigenvectors
    d, U = hpx.doublePassG(Hmisfit.mat, prior.R, prior.Rsolver, Omega, k, s=1)

    # --- 10. PERFORM LAPLACE APPROXIMATION & SAMPLING ---
    # Create the Laplace approximation object using the computed eigenpairs
    lap_aprx = hpx.LaplaceApproximator(prior, d, U)

    # Set the mean of the posterior approximation to the computed MAP point
    lap_aprx.mean = prior.generate_parameter(0)
    lap_aprx.mean.array[:] = x[hpx.PARAMETER].array[:]

    # Prepare for sampling
    noise = prior.generate_parameter("noise")
    num_samples_generate = 5
    use_vtx = False  # Use XDMF instead of VTX

    # Create dolfinx Functions to store the samples
    prior_sample = dlx.fem.Function(Vh[hpx.PARAMETER], name="prior_sample")
    posterior_sample = dlx.fem.Function(Vh[hpx.PARAMETER], name="posterior_sample")

    if use_vtx:
        # VTX output (binary)
        with dlx.io.VTXWriter(
            msh.comm,
            outdir / "poisson_Robin_prior_Bilaplacian_samples_np{0:d}.bp".format(nproc),
            [prior_sample, posterior_sample],
        ) as vtx:
            for i in range(num_samples_generate):
                hpx.parRandom.normal(1.0, noise)
                # Draw a sample from the prior and posterior
                lap_aprx.sample(noise, prior_sample.x, posterior_sample.x)
                prior_sample.x.scatter_forward()
                posterior_sample.x.scatter_forward()
                vtx.write(float(i))
    else:
        # XDMF output (more portable)
        with dlx.io.XDMFFile(
            msh.comm,
            outdir / "poisson_Robin_prior_Bilaplacian_samples_np{0:d}.xdmf".format(nproc),
            "w",
        ) as file:
            file.write_mesh(msh)  # Write the mesh once
            for i in range(num_samples_generate):
                hpx.parRandom.normal(1.0, noise)  # Generate new white noise
                # Draw a sample from the prior and posterior
                lap_aprx.sample(noise, prior_sample.x, posterior_sample.x)
                prior_sample.x.scatter_forward()
                posterior_sample.x.scatter_forward()
                # Write the samples as new timesteps
                file.write_function(prior_sample, float(i))
                file.write_function(posterior_sample, float(i))

    # --- 11. COLLECT AND RETURN RESULTS ---
    eigen_decomposition_results = {"A": Hmisfit, "B": prior, "k": k, "d": d, "U": U}

    final_results = {
        "data_misfit_True": data_misfit_True,
        "data_misfit_False": data_misfit_False,
        "optimizer_results": optimizer_results,
        "eigen_decomposition_results": eigen_decomposition_results,
    }

    return final_results


if __name__ == "__main__":
    # --- SCRIPT DRIVER ---

    # Set mesh resolution
    nx = 64
    ny = 64

    # Set noise variance (sigma^2)
    noise_variance = 1e-6

    # Set prior parameters (gamma and delta for BiLaplacian)
    prior_param = {"gamma": 0.07, "delta": 0.7}

    # Run the full inversion
    final_results = run_inversion(nx, ny, noise_variance, prior_param)

    # --- PLOT EIGENVALUES (on rank 0 only) ---
    k, d = (
        final_results["eigen_decomposition_results"]["k"],
        final_results["eigen_decomposition_results"]["d"],
    )
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        # Plot the model verification figure
        plt.savefig("poisson_result_FD_Gradient_Hessian_Check")

        # Plot the computed eigenvalues
        plt.figure()
        plt.plot(range(0, k), d, "b*", range(0, k), np.ones(k), "-r")
        plt.yscale("log")
        plt.title("Generalized Eigenvalues (d)")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend(["Computed Eigenvalues", "Threshold = 1.0"])
        plt.savefig("poisson_Eigen_Decomposition_results.png")
