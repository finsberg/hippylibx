# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

"""
End-to-end example of a Bayesian inverse problem for Quantitative
Photoacoustic Tomography (QPACT), modeled as a diffusion approximation
to radiative transfer.

This script demonstrates:
1.  Solving a non-linear inverse problem (PACT misfit).
2.  Using a **VariationalRegularization (H1 Tikhonov) Prior**.
3.  Reading a mesh from an external XDMF file.
4.  Computing the MAP point and a low-rank (Laplace) approximation
    of the posterior Hessian.
5.  This script stops after the eigensolver and does not draw samples.
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


class DiffusionApproximation:
    r"""
    Functor class defining the weak form (residual) of the diffusion PDE.

    This PDE models the light fluence 'u' inside the tissue, given
    the absorption coefficient $\mu_a = \exp(m)$.

    The strong form of the PDE is:
    $-\nabla \cdot (D \nabla u) + \exp(m) u = 0$  in $\Omega$
    $D \frac{\partial u}{\partial n} + 0.5(u - u_0) = 0$  on $\partial\Omega$

    where:
    - u: Light fluence (state)
    - m: Log-absorption coefficient (parameter)
    - D: Diffusion coefficient
    - u0: Incident light fluence (Robin boundary data)
    """

    def __init__(self, D: float, u0: float):
        """
        Initializes the PDE parameters.

        Args:
            D: The diffusion coefficient.
            u0: The incident fluence (Robin boundary condition).
        """
        self.D = D
        self.u0 = u0
        self.dx = ufl.Measure("dx", metadata={"quadregree": 4})
        self.ds = ufl.Measure("ds", metadata={"quadregree": 4})

    def __call__(
        self,
        u: dlx.fem.Function,
        m: dlx.fem.Function,
        p: dlx.fem.Function,
    ) -> ufl.form.Form:
        """
        Returns the UFL weak form (residual) of the PDE.

        Args:
            u: The state variable (fluence).
            m: The parameter variable (log-absorption).
            p: The test function.

        Returns:
            The UFL form for the PDE residual.
        """
        return (
            # Diffusion term: (D*grad(u), grad(p))_Omega
            ufl.inner(self.D * ufl.grad(u), ufl.grad(p)) * ufl.dx(metadata={"quadrature_degree": 4})
            # Absorption term: (exp(m)*u, p)_Omega
            + ufl.exp(m) * ufl.inner(u, p) * self.dx
            # Robin boundary term: 0.5 * (u - u0, p)_dOmega
            + 0.5 * ufl.inner(u - self.u0, p) * self.ds
        )


class PACTMisfitForm:
    r"""
    Functor class defining the misfit (log-likelihood) for PACT.

    In PACT, the observed data 'd' is the *absorbed energy*, which is
    the product of the absorption coefficient $\mu_a = \exp(m)$ and the
    fluence $u$.

    The cost functional is:
    $J(u, m) = \frac{1}{2\sigma^2} \int_{\Omega} (u \cdot \exp(m) - d)^2 \, dx$

    This is a **non-linear** misfit functional as it depends on a product
    of the state 'u' and the parameter 'm'.
    """

    def __init__(self, d: float, sigma2: float):
        """
        Initializes the misfit parameters.

        Args:
            d: The (noisy) observation data of absorbed energy.
            sigma2: The noise variance.
        """
        self.sigma2 = sigma2
        self.d = d
        self.dx = ufl.Measure("dx", metadata={"quadrature_degree": 4})

    def __call__(self, u: dlx.fem.Function, m: dlx.fem.Function) -> ufl.form.Form:
        """
        Returns the UFL weak form for the misfit.

        Args:
            u: The state variable (fluence).
            m: The parameter variable (log-absorption).

        Returns:
            The UFL form for the misfit functional.
        """
        # The observed quantity: H = u * exp(m)
        observed_H = u * ufl.exp(m)

        # 0.5/sigma^2 * (H - d, H - d)_Omega
        return 0.5 / self.sigma2 * ufl.inner(observed_H - self.d, observed_H - self.d) * self.dx


def run_inversion(
    mesh_filename: Path,
    noise_variance: float,
    prior_param: Dict[str, float],
    outdir: Path = Path("results-sfsi-toy-gaussian-reg"),
) -> Dict[str, Dict[str, float]]:
    """
    Executes the full end-to-end Bayesian inversion workflow for the QPACT problem.
    (H1 Tikhonov Prior + Non-linear Misfit)

    Args:
        mesh_filename: Path to the XDMF mesh file.
        noise_variance: Variance (sigma^2) of the observation noise.
        prior_param: Dictionary of prior parameters (e.g., "gamma", "delta").
        outdir: Output directory for saving results.

    Returns:
        A dictionary containing key results from the inversion.
    """
    sep = "\n" + "#" * 80 + "\n"
    comm = MPI.COMM_WORLD
    rank = comm.rank
    nproc = comm.size

    # --- 1. SET UP MPI, MESH, AND FUNCTION SPACES ---
    # Read the mesh from the provided XDMF file
    fname = mesh_filename
    fid = dlx.io.XDMFFile(comm, fname, "r")
    msh = fid.read_mesh(name="mesh")

    # State and Adjoint space: P2 Lagrange elements
    Vh_phi = dlx.fem.functionspace(msh, ("Lagrange", 2))
    # Parameter space: P1 Lagrange elements
    Vh_m = dlx.fem.functionspace(msh, ("Lagrange", 1))
    # hIPPYlibx composite space list: [State, Parameter, Adjoint]
    Vh = [Vh_phi, Vh_m, Vh_phi]

    ndofs = [
        Vh_phi.dofmap.index_map.size_global * Vh_phi.dofmap.index_map_bs,
        Vh_m.dofmap.index_map.size_global * Vh_m.dofmap.index_map_bs,
    ]
    hpx.master_print(comm, sep, "Set up the mesh and finite element spaces", sep)
    hpx.master_print(comm, "Number of dofs: STATE={0}, PARAMETER={1}".format(*ndofs))

    # --- 2. DEFINE THE FORWARD PDE PROBLEM ---
    u0 = 1.0  # Incident fluence
    D = 1.0 / 24.0  # Diffusion coefficient
    pde_handler = DiffusionApproximation(D, u0)

    # Encapsulate the UFL form in a hIPPYlibx PDE problem
    pde = hpx.PDEVariationalProblem(Vh, pde_handler, [], [], is_fwd_linear=True)

    # --- 3. CREATE GROUND TRUTH ---
    m_true = dlx.fem.Function(Vh_m)
    # Define a true parameter field (a circular inclusion)
    m_true.interpolate(
        lambda x: np.log(0.01) + 3.0 * (((x[0] - 2.0) * (x[0] - 2.0) + (x[1] - 2.0) * (x[1] - 2.0)) < 1.0),
    )
    m_true.x.scatter_forward()  # Synchronize parallel vector
    m_true = m_true.x  # Get the underlying dolfinx.la.Vector

    # Solve the PDE to get the "true" state 'u_true'
    u_true = pde.generate_state()
    x_true = [u_true, m_true, None]  # [state, parameter, adjoint]
    pde.solveFwd(u_true, x_true)

    # Create dolfinx Function wrappers for helper operations
    xfun = [dlx.fem.Function(Vhi) for Vhi in Vh]

    # --- 4. CREATE SYNTHETIC DATA (LIKELIHOOD) ---
    # We must generate data 'd' that matches the PACTMisfitForm

    # Get Function representations of the true state and parameter
    hpx.updateFromVector(xfun[hpx.STATE], u_true)
    u_fun_true = xfun[hpx.STATE]
    hpx.updateFromVector(xfun[hpx.PARAMETER], m_true)
    m_fun_true = xfun[hpx.PARAMETER]

    # 1. Compute the *true absorbed energy*, H_true = u_true * exp(m_true)
    d = dlx.fem.Function(Vh[hpx.STATE])  # Data 'd' lives in the state space
    expr = u_fun_true * ufl.exp(m_fun_true)

    # 2. Project this UFL expression into the data vector 'd'
    hpx.projection(expr, d)

    # 3. Add Gaussian noise to the data vector
    hpx.parRandom.normal_perturb(np.sqrt(noise_variance), d.x)
    d.x.scatter_forward()

    # 4. Initialize the misfit functional with the noisy data 'd'
    misfit_form = PACTMisfitForm(d, noise_variance)
    misfit = hpx.NonGaussianContinuousMisfit(Vh, misfit_form)

    # --- 5. DEFINE THE PRIOR (H1 Tikhonov Regularization) ---
    # Set a constant prior mean
    prior_mean = dlx.fem.Function(Vh_m)
    prior_mean.x.array[:] = np.log(0.01)

    prior_gamma = prior_param["gamma"]
    prior_delta = prior_param["delta"]

    # This handler defines the UFL form for the prior cost:
    # R(m) = 0.5 * (gamma * ||grad(m - m_mean)||^2 + delta * ||m - m_mean||^2)
    prior_handler = hpx.H1TikhonvFunctional(prior_gamma, prior_delta, prior_mean)

    # This class wraps the UFL form into a hIPPYlibX-compatible prior object
    # It will automatically assemble the prior Hessian (R) and its solver (Rsolver).
    prior = hpx.VariationalRegularization(Vh_m, prior_handler)

    # --- 6. DEFINE THE FULL BAYESIAN MODEL ---
    # Combine PDE, Prior, and Misfit into a single Model object
    model = hpx.Model(pde, prior, misfit)

    # --- 7. (OPTIONAL) VERIFY THE MODEL (GRADIENT CHECK) ---
    # Define a test point 'm0' for the gradient check
    m0 = dlx.fem.Function(Vh_m)
    m0.interpolate(
        lambda x: (2 * np.log(0.01) + 3) / 2 + 3 / 2 * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]),
    )
    m0.x.scatter_forward()
    m0 = m0.x

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

    # # # # #######################################

    # --- 8. SOLVE FOR THE MAP POINT (INVERSION) ---
    # Set the initial guess for the optimization to the prior mean
    initial_guess_m = pde.generate_parameter()
    initial_guess_m.array[:] = prior_mean.x.array[:]

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

    # --- 9. POST-PROCESS AND SAVE MAP RESULTS ---
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
        outdir / "qpact_Variational_Regularization_prior_np{0:d}_Prior.bp".format(nproc),
        [m_fun, m_true_fun, u_map_fun, u_true_fun, d_fun],
    ) as vtx:
        vtx.write(0.0)

    # Store optimizer results
    optimizer_results = {}
    if solver.termination_reasons[solver.reason] == "Norm of the gradient less than tolerance":
        optimizer_results["optimizer"] = True
    else:
        optimizer_results["optimizer"] = False

    # --- 10. COMPUTE LOW-RANK HESSIAN APPROXIMATION ---
    # Define the Hessian of the misfit (using Gauss-Newton approximation)
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
    d, U = hpx.doublePassG(Hmisfit.mat, prior.R, prior.Rsolver, Omega, k, s=1)

    # --- 11. COLLECT AND RETURN RESULTS ---
    # (Note: This script stops after finding the eigenpairs and does not
    # proceed to create a LaplaceApproximator or draw samples.)
    eigen_decomposition_results = {"A": Hmisfit, "B": prior, "k": k, "d": d, "U": U}

    final_results = {
        "data_misfit_True": data_misfit_True,
        "data_misfit_False": data_misfit_False,
        "optimizer_results": optimizer_results,
        "eigen_decomposition_results": eigen_decomposition_results,
    }

    return final_results

    #######################################


if __name__ == "__main__":
    # --- SCRIPT DRIVER ---
    comm = MPI.COMM_WORLD

    # Set problem parameters
    noise_variance = 1e-6
    # Set prior parameters (gamma for H1, delta for L2)
    prior_param = {"gamma": 0.15, "delta": 3.0}

    # Define the path to the mesh file
    mesh_filename = Path("./meshes/circle.xdmf")

    # Check if the mesh file exists. If not, try to create it.
    if not mesh_filename.is_file():
        try:
            import create_circle_mesh
        except ImportError:
            if comm.rank == 0:
                print("Error: Mesh file 'meshes/circle.xdmf' not found.")
                print("Please run 'create_circle_mesh.py' to generate it.")
            exit(1)

        # Wait for all processes before creating the mesh
        comm.Barrier()

        # Run the mesh creation script
        if comm.rank == 0:
            print("Mesh file not found. Running create_circle_mesh.py...")
        create_circle_mesh.main(comm, mesh_filename)
        if comm.rank == 0:
            print("...Mesh created.")

    # Run the full inversion
    final_results = run_inversion(mesh_filename, noise_variance, prior_param)

    # --- PLOT EIGENVALUES (on rank 0 only) ---
    k, d = (
        final_results["eigen_decomposition_results"]["k"],
        final_results["eigen_decomposition_results"]["d"],
    )
    if comm.rank == 0:
        # Plot the model verification figure
        plt.savefig("qpact_result_FD_Gradient_Hessian_Check")

        # Plot the computed eigenvalues
        plt.figure()
        plt.plot(range(0, k), d, "b*", range(0, k), np.ones(k), "-r")
        plt.yscale("log")
        plt.title("Generalized Eigenvalues (d)")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend(["Computed Eigenvalues", "Threshold = 1.0"])
        plt.savefig("qpact_Eigen_Decomposition_results.png")
