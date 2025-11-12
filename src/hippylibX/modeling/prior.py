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
from mpi4py import MPI

import basix.ufl
import dolfinx as dlx
import dolfinx.fem.petsc
import numpy as np
import ufl


# decorator for functions in classes that are not used -> may not be needed in the final
# version of X
def unused_function(func):
    """
    A decorator to mark functions as unused.
    This can be useful for silencing linters or indicating planned features.
    """
    return None


class _BilaplacianR:
    """
    A matrix-free operator for the precision matrix R = A * M^{-1} * A.

    This class creates a PETSc 'Mat' shell that behaves like a matrix
    but computes its action (matrix-vector product) on-the-fly using
    the provided 'A' matrix and 'Msolver'.
    """

    def __init__(self, A: petsc4py.PETSc.Mat, Msolver: petsc4py.PETSc.KSP):
        """
        Initializes the matrix-free operator.

        Args:
            A: The PETSc matrix 'A' (from the sqrt_precision_varf_handler).
            Msolver: A configured PETSc KSP solver for the mass matrix 'M'.
        """
        self.A = A
        self.Msolver = Msolver

        # Pre-allocate temporary vectors for intermediate steps in mult()
        # This avoids re-creating vectors on every matrix-vector product.
        self.help1 = self.A.createVecLeft()
        self.help2 = self.A.createVecRight()

        # Create the PETSc "matrix shell" (a Mat of type 'python')
        # This shell has the same dimensions and communicator as 'A'
        self.petsc_wrapper = petsc4py.PETSc.Mat().createPython(
            self.A.getSizes(),
            comm=self.A.getComm(),
        )
        # Set the Python context to this instance, so PETSc can call its methods
        self.petsc_wrapper.setPythonContext(self)
        # Finalize the setup of the matrix shell
        self.petsc_wrapper.setUp()

    def __del__(self):
        """Ensures the PETSc matrix wrapper is destroyed."""
        self.petsc_wrapper.destroy()

    @property
    def mat(self) -> petsc4py.PETSc.Mat:
        """Public accessor for the PETSc-compatible matrix shell."""
        return self.petsc_wrapper

    def mpi_comm(self) -> MPI.Intracomm:
        """Returns the MPI communicator."""
        return self.A.comm

    def mult(self, mat, x: petsc4py.PETSc.Vec, y: petsc4py.PETSc.Vec) -> None:
        """
        Defines the matrix-vector product y = R*x.

        This method is called by PETSc when self.petsc_wrapper.mult(x, y) is invoked.
        It computes y = (A * M^{-1} * A) * x.

        Args:
            mat: The PETSc Mat shell (self.petsc_wrapper).
            x: The input vector.
            y: The output vector.
        """
        # Step 1: help1 = A * x
        self.A.mult(x, self.help1)
        # Step 2: help2 = M^{-1} * help1
        self.Msolver.solve(self.help1, self.help2)
        # Step 3: y = A * help2
        self.A.mult(self.help2, y)


class _BilaplacianRsolver:
    """
    A matrix-free operator for the *inverse* of the precision matrix (the covariance matrix).
    This implements C = R^{-1} = A^{-1} * M * A^{-1}.

    This class provides a '.solve()' method compatible with PETSc KSP,
    acting as a preconditioner or direct solver.
    """

    def __init__(self, Asolver: petsc4py.PETSc.KSP, M: petsc4py.PETSc.Mat):
        """
        Initializes the matrix-free solver.

        Args:
            Asolver: A configured PETSc KSP solver for the 'A' matrix.
            M: The PETSc mass matrix 'M'.
        """
        self.Asolver = Asolver
        self.M = M
        # Pre-allocate temporary vectors for intermediate steps
        self.help1, self.help2 = self.M.createVecLeft(), self.M.createVecLeft()

    def solve(self, b: petsc4py.PETSc.Vec, x: petsc4py.PETSc.Vec) -> int:
        """
        Computes the action of the inverse precision (covariance) x = C*b.
        It solves the system R*x = b by computing x = (A^{-1} * M * A^{-1}) * b.

        Args:
            b: The right-hand side vector.
            x: The solution vector.

        Returns:
            The total number of iterations from the 'A' solver.
        """
        # Step 1: help1 = A^{-1} * b
        self.Asolver.solve(b, self.help1)
        nit = self.Asolver.its
        # Step 2: help2 = M * help1
        self.M.mult(self.help1, self.help2)
        # Step 3: x = A^{-1} * help2
        self.Asolver.solve(self.help2, x)
        nit += self.Asolver.its
        return nit

    def generate_vector(self) -> petsc4py.PETSc.Vec:
        """Generates a vector compatible with the operator's domain/range."""
        return self.M.createVecLeft()


class SqrtPrecisionPDE_Prior:
    """
    Implements a prior model with covariance matrix C = A^{-1} * M * A^{-1}.

    'A' is the finite element matrix from the 'sqrt_precision_varf_handler'.
    This class assembles the component matrices (A, M) and solvers,
    and provides the matrix-free precision operator (R) and covariance
    solver (Rsolver).
    """

    def __init__(
        self,
        Vh: dlx.fem.FunctionSpace,
        sqrt_precision_varf_handler,
        mean: dlx.la.Vector | None = None,
    ):
        """
        Construct the prior model.

        Args:
            Vh: The dolfinx FunctionSpace for the parameter.
            sqrt_precision_varf_handler: A function that takes (trial, test)
                functions and returns the UFL form for the 'A' matrix.
            mean: The prior mean vector (a dolfinx.la.Vector). If None,
                a zero vector is assumed.
        """

        self.dx = ufl.Measure("dx", metadata={"quadrilature_degree": 4})
        self.ds = ufl.Measure("ds", metadata={"quadrilature_degree": 4})

        self.Vh = Vh
        self.sqrt_precision_varf_handler = sqrt_precision_varf_handler

        # Define default PETSc solver options for the M (mass) matrix
        # Uses Conjugate Gradient (cg) with a simple Jacobi preconditioner.
        self.petsc_options_M = {
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": "1e-12",
            "ksp_max_it": "1000",
            "ksp_error_if_not_converged": "true",
            "ksp_initial_guess_nonzero": "false",
        }
        # Define default PETSc solver options for the A ("stiffness") matrix
        # Uses Conjugate Gradient (cg) with HYPRE's BoomerAMG preconditioner.
        self.petsc_options_A = {
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": "1e-12",
            "ksp_max_it": "1000",
            "ksp_error_if_not_converged": "true",
            "ksp_initial_guess_nonzero": "false",
        }

        # --- 1. Assemble Mass Matrix (M) and Solver ---
        trial = ufl.TrialFunction(Vh)
        test = ufl.TestFunction(Vh)

        # Define the standard mass matrix UFL form: (u, v) dx
        varfM = ufl.inner(trial, test) * self.dx

        # Assemble the PETSc matrix
        self.M = dlx.fem.petsc.assemble_matrix(dlx.fem.form(varfM))
        self.M.assemble()

        # Create and configure the KSP solver for M
        self.Msolver = self._createsolver(self.petsc_options_M)
        self.Msolver.setOperators(self.M)

        # --- 2. Assemble "Stiffness" Matrix (A) and Solver ---

        # Assemble the PETSc matrix 'A' using the user-provided UFL form
        self.A = dlx.fem.petsc.assemble_matrix(
            dlx.fem.form(sqrt_precision_varf_handler(trial, test)),
        )
        self.A.assemble()

        # Create and configure the KSP solver for A
        self.Asolver = self._createsolver(self.petsc_options_A)

        # Special configuration if using HYPRE AMG
        if self.petsc_options_A["pc_type"] == "hypre":
            pc = self.Asolver.getPC()
            pc.setHYPREType("boomeramg")

        self.Asolver.setOperators(self.A)

        # --- 3. Assemble the sqrtM Operator ---
        # This is a complex part. It builds an operator 'sqrtM' used for
        # sampling, such that s = A^{-1} * sqrtM * noise.
        # It is *not* a literal matrix square root of self.M.
        # It's a projection from a discontinuous quadrature space (Qh) to Vh.

        qdegree = 2 * Vh._ufl_element.degree
        metadata = {"quadrature_degree": qdegree}

        num_sub_spaces = Vh.num_sub_spaces

        # Define the quadrature space Qh
        if num_sub_spaces <= 1:  # SCALAR PARAMETER
            # Use a Quadrature element, which has DoFs at quadrature points
            element = basix.ufl.quadrature_element(Vh.mesh.topology.cell_name(), degree=qdegree)

        else:  # Vector FIELD PARAMETER
            # Use a high-order Lagrange element for the vector field
            element = basix.ufl.element(
                "Lagrange",
                Vh.mesh.topology.cell_name(),
                degree=qdegree,
                shape=(num_sub_spaces,),
            )

        # Create the function space Qh
        self.Qh = dlx.fem.functionspace(Vh.mesh, element)

        # Trial and Test functions in Qh
        ph = ufl.TrialFunction(self.Qh)
        qh = ufl.TestFunction(self.Qh)

        # Assemble the mass matrix Mqh in the Qh space
        Mqh = dlx.fem.petsc.assemble_matrix(
            dlx.fem.form(ufl.inner(ph, qh) * ufl.dx(metadata=metadata)),
        )
        Mqh.assemble()

        # --- Create a diagonal matrix from Mqh ---
        # This is a form of "mass lumping" combined with a scaling.
        ones = Mqh.createVecRight()
        ones.set(1.0)
        dMqh = Mqh.createVecLeft()

        # Get the row-sums of Mqh (which equals the diagonal for this element)
        Mqh.mult(ones, dMqh)

        # Get the local numpy array and compute 1.0 / sqrt(diag_entries)
        # This is the operation that gave the .pointwiseSqrt() error before
        dMqh.setArray(ones.getArray() / np.sqrt(dMqh.getArray()))

        # Overwrite Mqh to be a diagonal matrix with these 1/sqrt(diag) entries
        Mqh.setDiagonal(dMqh)

        # Assemble the "mixed mass matrix" projecting from Qh to Vh
        MixedM = dlx.fem.petsc.assemble_matrix(
            dlx.fem.form(ufl.inner(ph, test) * ufl.dx(metadata=metadata)),
        )
        MixedM.assemble()

        # Define sqrtM = MixedM * Mqh (where Mqh is now diagonal)
        self.sqrtM = MixedM.matMult(Mqh)

        # --- 4. Setup Final Operators ---

        # Create the matrix-free precision operator R = A * M^{-1} * A
        self._R = _BilaplacianR(self.A, self.Msolver)

        # Create the matrix-free covariance solver C = A^{-1} * M * A^{-1}
        self.Rsolver = _BilaplacianRsolver(self.Asolver, self.M)

        # Set the mean using the property setter
        self.mean = mean

    @property
    def mean(self) -> dlx.la.Vector:
        """The mean vector of the prior."""
        return self._mean

    @mean.setter
    def mean(self, value: dlx.la.Vector | None) -> None:
        """
        Sets the mean vector.
        If value is None, a zero vector of the correct size is created.
        """
        if value is None:
            # Create a new zero vector in the Vh space
            self._mean = self.generate_parameter(0)
        else:
            self._mean = value

    @property
    def R(self) -> petsc4py.PETSc.Mat:
        """Public accessor for the PETSc-compatible precision operator R."""
        return self._R.mat

    def generate_parameter(self, dim: int | str) -> dlx.la.Vector:
        """
        Initialize a vector 'x' compatible with the prior's spaces.

        Args:
            dim: If "noise", create a vector in the 'Qh' (noise) space.
                 Otherwise, create a vector in the 'Vh' (parameter) space.
        """
        if dim == "noise":
            # Vector compatible with the noise space Qh
            return dlx.la.vector(self.Qh.dofmap.index_map)
        else:
            # Vector compatible with the parameter space Vh
            return dlx.la.vector(self.Vh.dofmap.index_map)

    def sample(self, noise: dlx.la.Vector, s: dlx.la.Vector, add_mean=True) -> None:
        """
        Given 'noise' ~ N(0, I), compute a sample 's' from the prior.

        The formula is s = mean + A^{-1} * sqrtM * noise.

        Args:
            noise: A random vector (in the Qh space) of standard normals.
            s: The output sample vector (in the Vh space).
            add_mean: If True, add the prior mean to the sample.
        """
        # Create a temporary vector in Vh space
        rhs = self.sqrtM.createVecLeft()

        # Step 1: rhs = sqrtM * noise (project from Qh to Vh)
        self.sqrtM.mult(noise.petsc_vec, rhs)

        # Step 2: s = A^{-1} * rhs (solve the "stiffness" system)
        self.Asolver.solve(rhs, s.petsc_vec)

        # Step 3: Add the mean if requested
        if add_mean:
            s.petsc_vec.axpy(1.0, self.mean.petsc_vec)

        # Clean up temporary vector
        rhs.destroy()

    def _createsolver(self, petsc_options: dict) -> petsc4py.PETSc.KSP:
        """
        Internal helper function to create a PETSc KSP solver
        and configure it from a dictionary of options.
        """
        ksp = petsc4py.PETSc.KSP().create(self.Vh.mesh.comm)
        # Use a unique prefix to avoid PETSc options collisions
        problem_prefix = f"dolfinx_solve_{id(self)}"
        ksp.setOptionsPrefix(problem_prefix)

        # Set PETSc options from the dictionary
        opts = petsc4py.PETSc.Options()
        opts.prefixPush(problem_prefix)
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v
        opts.prefixPop()
        ksp.setFromOptions()

        return ksp

    def cost(self, m: dlx.la.Vector) -> float:
        """
        Computes the negative log-likelihood (cost) of a parameter 'm'.
        cost = 0.5 * <m - mean, R * (m - mean)>

        Args:
            m: The parameter vector (in Vh space).

        Returns:
            The computed cost as a float.
        """
        # d = m - mean (but we compute mean - m and rely on symmetry of R)
        d = self.mean.petsc_vec.copy()
        d.axpy(-1.0, m.petsc_vec)

        # Rd = R * d
        Rd = self.generate_parameter(0)
        self.R.mult(d, Rd.petsc_vec)

        # return_value = 0.5 * <Rd, d>
        return_value = 0.5 * Rd.petsc_vec.dot(d)

        d.destroy()
        return return_value

    def grad(self, m: dlx.la.Vector, out: dlx.la.Vector) -> None:
        """
        Computes the gradient of the cost function.
        grad = R * (m - mean)

        Args:
            m: The parameter vector (in Vh space).
            out: The output gradient vector (in Vh space).
        """
        # d = m - mean
        d = m.petsc_vec.copy()
        d.axpy(-1.0, self.mean.petsc_vec)

        # out = R * d
        self.R.mult(d, out.petsc_vec)

        d.destroy()

    def setLinearizationPoint(self, m: dlx.la.Vector, gauss_newton_approx=False) -> None:
        """
        Placeholder method, possibly for API compatibility. Does nothing.
        """
        return

    def __del__(self):
        """
        Custom destructor to explicitly destroy all created PETSc objects
        and avoid memory leaks.
        """
        self.Msolver.destroy()
        self.Asolver.destroy()
        self.M.destroy()
        self.A.destroy()
        self.sqrtM.destroy()


def BiLaplacianPrior(
    Vh: dlx.fem.FunctionSpace,
    gamma: float,
    delta: float,
    Theta=None,
    mean: dlx.la.Vector | None = None,
    robin_bc: bool = False,
) -> SqrtPrecisionPDE_Prior:
    """
    This is a factory function that constructs a SqrtPrecisionPDE_Prior
    with a specific 'A' matrix: A = gamma*L + delta*M,
    where L is the Laplacian and M is the mass matrix.

    This corresponds to a Bi-Laplacian prior, as the covariance C is
    C = R^{-1} = (A M^{-1} A)^{-1} = A^{-1} M A^{-1}.
    If A = gamma*L + delta*M, this is a general Bi-Laplacian type covariance.

    The docstring's original formula C = (delta*I + gamma*div(Theta*grad))^{-2}
    seems to describe a different model (C = A^{-2}), not C = A^{-1} M A^{-1}.
    This function implements the C = A^{-1} M A^{-1} model.

    Args:
        Vh: The finite element space for the parameter.
        gamma: Coefficient for the Laplacian (stiffness) term.
        delta: Coefficient for the mass matrix (identity) term.
        Theta: Optional SPD tensor for anisotropic diffusion in the Laplacian.
        mean: The prior mean vector.
        robin_bc: If True, adds a Robin boundary term to the 'A' matrix
                  to mitigate boundary artifacts.

    Returns:
        An instance of SqrtPrecisionPDE_Prior.
    """

    def sqrt_precision_varf_handler(
        trial: ufl.TrialFunction,
        test: ufl.TestFunction,
    ) -> ufl.form.Form:
        """
        This inner function defines the UFL form for the 'A' matrix,
        which is A = gamma*L + delta*M + [Robin term].
        """
        dx = ufl.Measure("dx", metadata={"quadrature_degree": 4})
        ds = ufl.ds(metadata={"quadrature_degree": 4})

        # Define the Laplacian (stiffness) term varfL
        if Theta is None:
            # Standard isotropic Laplacian: (grad(u), grad(v)) dx
            varfL = ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx
        else:
            # Anisotropic Laplacian: (Theta*grad(u), grad(v)) dx
            varfL = ufl.inner(Theta * ufl.grad(trial), ufl.grad(test)) * dx

        # Define the mass term varfM: (u, v) dx
        varfM = ufl.inner(trial, test) * dx

        # Define the Robin boundary term varf_robin: (u, v) ds
        varf_robin = ufl.inner(trial, test) * ds

        if robin_bc:
            # A specific heuristic for the Robin coefficient
            robin_coeff = gamma * ufl.sqrt(delta / gamma) / 1.42
        else:
            robin_coeff = 0.0

        # Return the complete UFL form for 'A'
        return gamma * varfL + delta * varfM + robin_coeff * varf_robin

    # Create and return the prior object, passing in the UFL handler
    return SqrtPrecisionPDE_Prior(Vh, sqrt_precision_varf_handler, mean)
