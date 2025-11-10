# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-
import math
import numbers
from abc import ABC, abstractmethod
from functools import partial

import petsc4py
import petsc4py.PETSc
from mpi4py import MPI

import basix.ufl
import dolfinx as dlx
import dolfinx.fem.petsc
import numpy as np
import ufl

from ..algorithms.linalg import Operator2Solver, Solver2Operator
from ..algorithms.multivector import MultiVector
from ..algorithms.randomizedEigensolver import doublePassG
from ..utils.random import parRandom


# decorator for functions in classes that are not used -> may not be needed in the final
# version of X
def unused_function(func):
    return None


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NEW HELPER CLASS PORTED FROM OLD CODE
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class _RinvM:
    """
    Operator that models the action of R^{-1}M.
    Used in trace estimation.
    """

    def __init__(self, Rsolver: petsc4py.PETSc.KSP, M: petsc4py.PETSc.Mat):
        self.Rsolver = Rsolver
        self.M = M
        self.comm = M.getComm()

        self.temp_vec = M.createVecLeft()

        self.petsc_wrapper = petsc4py.PETSc.Mat().createPython(
            self.M.getSizes(),
            comm=self.comm,
        )
        self.petsc_wrapper.setPythonContext(self)
        self.petsc_wrapper.setUp()

    def __del__(self):
        self.petsc_wrapper.destroy()
        self.temp_vec.destroy()

    @property
    def mat(self) -> petsc4py.PETSc.Mat:
        return self.petsc_wrapper

    def mpi_comm(self) -> MPI.Intracomm:
        return self.comm

    def mult(self, mat, x: petsc4py.PETSc.Vec, y: petsc4py.PETSc.Vec) -> None:
        self.M.mult(x, self.temp_vec)
        self.Rsolver.solve(self.temp_vec, y)

    def generate_vector(self, dim: int) -> petsc4py.PETSc.Vec:
        if dim == 0:
            return self.M.createVecRight()
        else:
            return self.M.createVecLeft()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# EXISTING HELPER CLASS (NO CHANGES)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class _BilaplacianR:
    """
    Operator that represent the action of the regularization/precision matrix
    for the Bilaplacian prior.
    """

    def __init__(self, A: petsc4py.PETSc.Mat, Msolver: petsc4py.PETSc.KSP):
        self.A = A  # should be petsc4py.PETSc.Mat
        self.Msolver = Msolver

        self.help1 = self.A.createVecLeft()
        self.help2 = self.A.createVecRight()

        self.petsc_wrapper = petsc4py.PETSc.Mat().createPython(
            self.A.getSizes(),
            comm=self.A.getComm(),
        )
        self.petsc_wrapper.setPythonContext(self)
        self.petsc_wrapper.setUp()

    def __del__(self):
        if hasattr(self, "petsc_wrapper") and self.petsc_wrapper:
            self.petsc_wrapper.destroy()
        if hasattr(self, "help1") and self.help1:
            self.help1.destroy()
        if hasattr(self, "help2") and self.help2:
            self.help2.destroy()

    @property
    def mat(self) -> petsc4py.PETSc.Mat:
        return self.petsc_wrapper

    def mpi_comm(self) -> MPI.Intracomm:
        return self.A.getComm()

    def mult(self, mat, x: petsc4py.PETSc.Vec, y: petsc4py.PETSc.Vec) -> None:
        self.A.mult(x, self.help1)
        self.Msolver.solve(self.help1, self.help2)
        self.A.mult(self.help2, y)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# EXISTING HELPER CLASS (NO CHANGES)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class _BilaplacianRsolver:
    """
    Operator that represent the action of the inverse the regularization/precision matrix
    for the Bilaplacian prior.
    """

    def __init__(self, Asolver: petsc4py.PETSc.KSP, M: petsc4py.PETSc.Mat):
        self.Asolver = Asolver
        self.M = M
        self.help1, self.help2 = self.M.createVecLeft(), self.M.createVecLeft()

    def __del__(self):
        if hasattr(self, "help1") and self.help1:
            self.help1.destroy()
        if hasattr(self, "help2") and self.help2:
            self.help2.destroy()

    def solve(self, b: petsc4py.PETSc.Vec, x: petsc4py.PETSc.Vec) -> int:
        self.Asolver.solve(b, self.help1)
        nit = self.Asolver.getIterationNumber()
        self.M.mult(self.help1, self.help2)
        self.Asolver.solve(self.help2, x)
        nit += self.Asolver.getIterationNumber()
        return nit

    def generate_vector(self) -> petsc4py.PETSc.Vec:
        return self.M.createVecLeft()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NEW ABSTRACT BASE CLASS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class _Prior(ABC):
    """
    Abstract base class to describe the prior model.
    """

    def __init__(self):
        self._mean = None
        self.R = None
        self.Rsolver = None
        self.M = None
        self.Vh = None

    @property
    def mean(self) -> dlx.la.Vector:
        return self._mean

    @mean.setter
    def mean(self, value: dlx.la.Vector | None) -> None:
        if value is None:
            self._mean = self.generate_parameter(0)
        else:
            self._mean = value

    @abstractmethod
    def generate_parameter(self, dim: int | str) -> dlx.la.Vector:
        """
        Initialize a vector :code:`x` to be compatible with the range/domain of :math:`R`.

        If :code:`dim == "noise"` initialize :code:`x` to be compatible with the size of
        white noise used for sampling.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, noise: dlx.la.Vector, s: dlx.la.Vector, add_mean=True) -> None:
        """
        Given :code:`noise` :math:`\\sim \\mathcal{N}(0, I)` compute a sample :code:`s` from the prior.

        If :code:`add_mean == True` add the prior mean value to :code:`s`.
        """
        raise NotImplementedError

    def cost(self, m: dlx.la.Vector) -> float:
        """Compute the prior cost: 0.5 * (m-mean)^T * R * (m-mean)"""
        d = self.mean.petsc_vec.copy()
        d.axpy(-1.0, m.petsc_vec)
        Rd = self.generate_parameter(0)
        self.R.mult(d, Rd.petsc_vec)
        return_value = 0.5 * Rd.petsc_vec.dot(d)
        d.destroy()
        return return_value

    def grad(self, m: dlx.la.Vector, out: dlx.la.Vector) -> None:
        """Compute the gradient of the prior cost: R * (m-mean)"""
        d = m.petsc_vec.copy()
        d.axpy(-1.0, self.mean.petsc_vec)
        self.R.mult(d, out.petsc_vec)
        d.destroy()

    def getHessianPreconditioner(self) -> petsc4py.PETSc.KSP:
        """Return the preconditioner for Newton-CG (the Rsolver)"""
        return self.Rsolver

    def setLinearizationPoint(self, m: dlx.la.Vector, gauss_newton_approx=False) -> None:
        """Placeholder function, can be overloaded by child classes if needed."""
        return

    def trace(
        self,
        method: str = "Exact",
        tol: float = 1e-1,
        min_iter: int = 20,
        max_iter: int = 100,
        r: int = 200,
    ) -> float:
        """
        Compute/estimate the trace of the prior covariance operator (C = R^{-1}M).

        - If :code:`method=="Exact"` we compute the trace exactly by summing the
          diagonal entries of :math:`R^{-1}M`.
          (Requires `hippylibx.algorithms.linalg.get_diagonal`)

        - If :code:`method=="Estimator"` use the trace estimator algorithms.
          (Requires `hippylibx.algorithms.traceEstimator.TraceEstimator`)

        - If :code:`method=="Randomized"` use the randomized eigensolver.
          (Requires `hippylibx`'s `randomizedEigensolver`, `MultiVector`,
          `parRandom`, and linalg wrappers)
        """
        # op is a PETSc.Mat (python type) that applies R^{-1} * M
        op = _RinvM(self.Rsolver, self.M).mat

        if method == "Exact":
            # Create a dolfinx vector to store the diagonal
            marginal_variance = self.generate_parameter(0)

            # Call the assumed function `get_diagonal`
            # It operates on the PETSc matrix `op` and writes to the PETSc vector
            get_diagonal(op, marginal_variance.petsc_vec)

            # .sum() is a PETSc vector method
            trace_val = marginal_variance.petsc_vec.sum()
            marginal_variance.petsc_vec.destroy()
            op.destroy()
            return trace_val

        elif method == "Estimator":
            # Call the assumed TraceEstimator
            tr_estimator = TraceEstimator(op, False, tol)
            tr_exp, tr_var = tr_estimator(min_iter, max_iter)
            op.destroy()
            return tr_exp

        elif method == "Randomized":
            # This method computes Tr(R^{-1}M) by computing Tr(M R^{-1}),
            # which has the same eigenvalues.
            # It solves the generalized eigenproblem M R^{-1} v = lambda v,
            # which is equivalent to R^{-1} v = lambda M^{-1} v.
            if self.Msolver is None:
                op.destroy()
                raise AttributeError("Msolver is not initialized in this Prior object. Cannot use 'Randomized' trace method.")

            # Create a template vector for MultiVector
            dummy_vec = self.generate_parameter(0)

            # Create the block of random vectors
            Omega = MultiVector(dummy_vec.petsc_vec, r)
            parRandom.normal(1.0, Omega)  # Assumed function
            dummy_vec.petsc_vec.destroy()

            # Create operator wrappers (Assumed classes)
            # This replicates the old code's logic:
            # A_op = R^{-1}, B_op = M^{-1}, B_solver = M
            # Solves A v = \lambda B v  =>  R^{-1} v = \lambda M^{-1} v
            # This is equivalent to M R^{-1} v = \lambda v
            # The trace Tr(M R^{-1}) == Tr(R^{-1} M)
            Rsolver_op = Solver2Operator(self.Rsolver)
            Msolver_op = Solver2Operator(self.Msolver)
            M_solver = Operator2Solver(self.M)

            # Call the assumed randomized eigensolver
            d, _ = doublePassG(
                Rsolver_op,
                Msolver_op,
                M_solver,
                Omega,
                r,
                s=1,
                # check=False,
            )
            # op.destroy()
            return d.sum()  # d is assumed to be a numpy array of eigenvalues

        else:
            op.destroy()
            raise NameError(f"Unknown trace method: {method}")

    def pointwise_variance(self, method, k=1000000, r=200):
        """
        Compute/estimate the prior pointwise variance.
        """

        pw_var = self.generate_parameter(0)

        if method == "Exact":
            get_diagonal(Solver2Operator(self.Rsolver), pw_var)
        elif method == "Estimator":
            estimate_diagonal_inv2(self.Rsolver, k, pw_var)
        elif method == "Randomized":
            Omega = MultiVector(pw_var.petsc_vec, r)
            parRandom.normal(1.0, Omega)

            Rsolver_op = Solver2Operator(self.Rsolver)
            Msolver_op = Solver2Operator(self.Msolver)
            M_solver = Operator2Solver(self.M)

            # Call the assumed randomized eigensolver
            d, U = doublePassG(
                Rsolver_op,
                Msolver_op,
                M_solver,
                Omega,
                r,
                s=1,
                # check=False,
            )
            # d, U = doublePass(Solver2Operator(self.Rsolver), Omega, r, s=1, check=False)

            for i in np.arange(U.nvec):
                pw_var.petsc_vec.axpy(d[i], U[i] * U[i])
        else:
            raise NameError("Unknown method")

        return pw_var
        # This method depends on external hippylib.algorithms modules
        # (linalg.get_diagonal, linalg.estimate_diagonal_inv2,
        # randomizedEigensolver.doublePass)
        # raise NotImplementedError(
        #     "pointwise_variance() requires algorithms from hippylib (linalg, "
        #     "randomizedEigensolver) which are not available in this snippet."
        # )
        # pw_var = self.generate_parameter(0)
        # if method == "Exact":
        #     # Needs hippylib.algorithms.linalg.get_diagonal
        #     pass
        # elif method == "Estimator":
        #     # Needs hippylib.algorithms.linalg.estimate_diagonal_inv2
        #     pass
        # elif method == "Randomized":
        #     # Needs hippylib.algorithms.multivector.MultiVector
        #     # Needs hippylib.utils.random.parRandom
        #     # Needs hippylib.algorithms.randomizedEigensolver.doublePass
        #     pass
        # return pw_var

    def _createsolver(self, petsc_options: dict) -> petsc4py.PETSc.KSP:
        """Helper to create a PETSc KSP solver."""
        ksp = petsc4py.PETSc.KSP().create(self.Vh.mesh.comm)
        problem_prefix = f"dolfinx_solve_{id(self)}_{np.random.randint(1_000_000)}"
        ksp.setOptionsPrefix(problem_prefix)

        # Set PETSc options
        opts = petsc4py.PETSc.Options()
        opts.prefixPush(problem_prefix)

        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v
        opts.prefixPop()
        ksp.setFromOptions()

        # Set HYPRE AMG type if specified
        if petsc_options.get("pc_type") == "hypre":
            pc = ksp.getPC()
            pc.setHYPREType("boomeramg")

        return ksp


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# EXISTING CLASS - MODIFIED TO INHERIT FROM _Prior
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class SqrtPrecisionPDE_Prior(_Prior):
    """
    This class implement a prior model with covariance matrix
    :math:`C = A^{-1} M A^-1`,
    where A is the finite element matrix arising from discretization of sqrt_precision_varf_handler
    """

    def __init__(
        self,
        Vh: dlx.fem.FunctionSpace,
        sqrt_precision_varf_handler,
        mean: dlx.la.Vector | None = None,
        petsc_options_A: dict | None = None,
        petsc_options_M: dict | None = None,
    ):
        """
        Construct the prior model.
        Input:

        - :code:`Vh`:              the finite element space for the parameter
        - :code:sqrt_precision_varf_handler: the PDE representation of the  sqrt of the covariance operator
        - :code:`mean`:            the prior mean
        - :code:`petsc_options_A`: PETSc options for the Asolver
        - :code:`petsc_options_M`: PETSc options for the Msolver
        """
        super().__init__()
        qdegree = 2 * Vh.ufl_element().degree
        metadata = {"quadrature_degree": qdegree}
        self.dx = ufl.Measure("dx", metadata=metadata)
        self.ds = ufl.Measure("ds", metadata=metadata)

        self.Vh = Vh
        self.sqrt_precision_varf_handler = partial(sqrt_precision_varf_handler, dx=self.dx, ds=self.ds)

        self.petsc_options_M = petsc_options_M or {
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": "1e-12",
            "ksp_max_it": "1000",
            "ksp_error_if_not_converged": "true",
            "ksp_initial_guess_nonzero": "false",
        }
        self.petsc_options_A = petsc_options_A or {
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": "1e-12",
            "ksp_max_it": "1000",
            "ksp_error_if_not_converged": "true",
            "ksp_initial_guess_nonzero": "false",
        }

        trial = ufl.TrialFunction(Vh)
        test = ufl.TestFunction(Vh)

        varfM = ufl.inner(trial, test) * self.dx

        self.M = dlx.fem.petsc.assemble_matrix(dlx.fem.form(varfM))
        self.M.assemble()

        self.Msolver = self._createsolver(self.petsc_options_M)
        self.Msolver.setOperators(self.M)

        self.A = dlx.fem.petsc.assemble_matrix(
            dlx.fem.form(self.sqrt_precision_varf_handler(trial, test)),
        )
        self.A.assemble()

        self.Asolver = self._createsolver(self.petsc_options_A)
        self.Asolver.setOperators(self.A)
        breakpoint()

        qdegree = 2 * Vh.ufl_element().degree

        metadata = {"quadrature_degree": qdegree}

        num_sub_spaces = Vh.num_sub_spaces

        if num_sub_spaces <= 1:  # SCALAR PARAMETER
            element = basix.ufl.quadrature_element(Vh.mesh.topology.cell_name(), degree=qdegree)
        else:  # Vector FIELD PARAMETER
            # Use VectorElement with Quadrature family
            element = basix.ufl.element(
                "Quadrature",
                Vh.mesh.topology.cell_name(),
                degree=qdegree,
                shape=(num_sub_spaces,),
            )

        self.Qh = dlx.fem.functionspace(Vh.mesh, element)

        ph = ufl.TrialFunction(self.Qh)
        qh = ufl.TestFunction(self.Qh)

        Mqh_form = dlx.fem.form(ufl.inner(ph, qh) * self.dx)
        Mqh = dlx.fem.petsc.assemble_matrix(Mqh_form)
        Mqh.assemble()

        ones = Mqh.createVecRight()
        ones.set(1.0)
        dMqh = Mqh.createVecLeft()
        Mqh.mult(ones, dMqh)
        # Avoid division by zero if dMqh has zero entries
        with dMqh.localForm() as loc:
            arr = loc.array
            arr_sqrt = np.sqrt(arr, where=arr > 0, out=np.full_like(arr, 1e-100))
            arr_inv_sqrt = np.divide(1.0, arr_sqrt, where=arr_sqrt > 0, out=np.zeros_like(arr))
            dMqh.setArray(ones.getArray() * arr_inv_sqrt)

        Mqh.setDiagonal(dMqh)
        ones.destroy()
        dMqh.destroy()

        MixedM_form = dlx.fem.form(ufl.inner(ph, test) * self.dx)
        MixedM = dlx.fem.petsc.assemble_matrix(MixedM_form)
        MixedM.assemble()

        self.sqrtM = MixedM.matMult(Mqh)
        Mqh.destroy()
        MixedM.destroy()

        self._R_obj = _BilaplacianR(self.A, self.Msolver)
        self.R = self._R_obj.mat  # Expose the petsc_wrapper

        self.Rsolver = _BilaplacianRsolver(self.Asolver, self.M)
        self.mean = mean  # Use the property setter

    def generate_parameter(self, dim: int | str) -> dlx.la.Vector:
        """
        Initialize a vector :code:`x` to be compatible with the range/domain of :math:`R`.

        If :code:`dim == "noise"` initialize :code:`x` to be compatible with the size of
        white noise used for sampling.
        """
        if dim == "noise":
            # Vector for noise space Qh
            return dlx.la.vector(self.Qh.dofmap.index_map, self.Qh.dofmap.index_map_bs)
        else:
            # Vector for parameter space Vh
            return dlx.la.vector(self.Vh.dofmap.index_map, self.Vh.dofmap.index_map_bs)

    def sample(self, noise: dlx.la.Vector, s: dlx.la.Vector, add_mean=True) -> None:
        """
        Given :code:`noise` :math:`\\sim \\mathcal{N}(0, I)` compute a sample :code:`s` from the prior.

        If :code:`add_mean == True` add the prior mean value to :code:`s`.
        """
        rhs = self.sqrtM.createVecLeft()
        self.sqrtM.mult(noise.petsc_vec, rhs)
        self.Asolver.solve(rhs, s.petsc_vec)

        if add_mean:
            s.petsc_vec.axpy(1.0, self.mean.petsc_vec)

        rhs.destroy()

    def __del__(self):
        # Clean up PETSc objects
        if hasattr(self, "Msolver") and self.Msolver:
            self.Msolver.destroy()
        if hasattr(self, "Asolver") and self.Asolver:
            self.Asolver.destroy()
        if hasattr(self, "M") and self.M:
            self.M.destroy()
        if hasattr(self, "A") and self.A:
            self.A.destroy()
        if hasattr(self, "sqrtM") and self.sqrtM:
            self.sqrtM.destroy()
        if hasattr(self, "_R_obj") and self._R_obj:
            del self._R_obj
        if hasattr(self, "Rsolver") and self.Rsolver:
            del self.Rsolver


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NEW CLASS PORTED FROM OLD CODE
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class LaplacianPrior(_Prior):
    """
    This class implements a prior model with covariance matrix
    :math:`C = (\\delta I - \\gamma \\Delta) ^ {-1}`.
    """

    def __init__(
        self,
        Vh: dlx.fem.FunctionSpace,
        gamma: float,
        delta: float,
        mean: dlx.la.Vector | None = None,
        petsc_options: dict | None = None,
    ):
        """
        Construct the prior model.
        Input:

        - :code:`Vh`:              the finite element space for the parameter
        - :code:`gamma` and :code:`delta`: the coefficient in the PDE
        - :code:`mean`:            the prior mean
        - :code:`petsc_options`:   PETSc options for Rsolver and Msolver
        """
        super().__init__()
        if delta == 0.0:
            raise ValueError("Intrinsic Gaussian Priors (delta=0.0) are not supported")

        self.Vh = Vh
        ndim = Vh.mesh.geometry.dim
        qdegree = 2 * Vh.ufl_element().degree
        metadata = {"quadrature_degree": qdegree}
        self.dx = ufl.Measure("dx", metadata=metadata)

        # Default PETSc options
        petsc_options_R_default = {
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": "1e-12",
            "ksp_max_it": "100",
            "ksp_error_if_not_converged": "true",
            "ksp_initial_guess_nonzero": "false",
        }
        petsc_options_M_default = {
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": "1e-12",
            "ksp_max_it": "100",
            "ksp_error_if_not_converged": "true",
            "ksp_initial_guess_nonzero": "false",
        }
        # Overwrite defaults with user-provided options if any
        if petsc_options:
            petsc_options_R_default.update(petsc_options)
            petsc_options_M_default.update(petsc_options)

        trial = ufl.TrialFunction(Vh)
        test = ufl.TestFunction(Vh)

        varfL = ufl.inner(ufl.grad(trial), ufl.grad(test)) * self.dx
        varfM = ufl.inner(trial, test) * self.dx

        self.M = dlx.fem.petsc.assemble_matrix(dlx.fem.form(varfM))
        self.M.assemble()

        self.R = dlx.fem.petsc.assemble_matrix(dlx.fem.form(gamma * varfL + delta * varfM))
        self.R.assemble()

        self.Rsolver = self._createsolver(petsc_options_R_default)
        self.Rsolver.setOperators(self.R)

        self.Msolver = self._createsolver(petsc_options_M_default)
        self.Msolver.setOperators(self.M)

        # --- Porting the stochastic square root self.sqrtR ---

        # Vector Quadrature element, dim = (ndim + 1)
        # We need (v_0, v_1, ..., v_ndim)
        element = basix.ufl.element(
            "Quadrature",
            Vh.mesh.topology.cell_name(),
            degree=qdegree,
            shape=(ndim + 1,),
        )
        self.Qh = dlx.fem.functionspace(Vh.mesh, element)

        ph = ufl.TrialFunction(self.Qh)
        qh = ufl.TestFunction(self.Qh)
        pph = ufl.split(ph)

        # Assemble Mqh (diagonal matrix)
        Mqh_form = dlx.fem.form(ufl.inner(ph, qh) * self.dx)
        Mqh = dlx.fem.petsc.assemble_matrix(Mqh_form)
        Mqh.assemble()

        ones = Mqh.createVecRight()
        ones.set(1.0)
        dMqh = Mqh.createVecLeft()
        Mqh.mult(ones, dMqh)
        # Avoid division by zero
        with dMqh.localForm() as loc:
            arr = loc.array
            arr_sqrt = np.sqrt(arr, where=arr > 0, out=np.full_like(arr, 1e-100))
            arr_inv_sqrt = np.divide(1.0, arr_sqrt, where=arr_sqrt > 0, out=np.zeros_like(arr))
            dMqh.setArray(ones.getArray() * arr_inv_sqrt)

        Mqh.setDiagonal(dMqh)
        ones.destroy()
        dMqh.destroy()

        # Assemble GG matrix (Mixed element form)
        sqrtdelta = math.sqrt(delta)
        sqrtgamma = math.sqrt(gamma)
        varfGG = sqrtdelta * pph[0] * test * self.dx
        for i in range(ndim):
            varfGG += sqrtgamma * pph[i + 1] * test.dx(i) * self.dx

        GG_form = dlx.fem.form(varfGG)
        GG = dlx.fem.petsc.assemble_matrix(GG_form)
        GG.assemble()

        # self.sqrtR = GG * Mqh
        self.sqrtR = GG.matMult(Mqh)
        GG.destroy()
        Mqh.destroy()

        self.mean = mean

    def generate_parameter(self, dim: int | str) -> dlx.la.Vector:
        """
        Initialize a vector :code:`x` to be compatible with the range/domain of :math:`R`.

        If :code:`dim == "noise"` initialize :code:`x` to be compatible with the size of
        white noise used for sampling.
        """
        if dim == "noise":
            # Vector for noise space Qh
            return dlx.la.vector(self.Qh.dofmap.index_map, self.Qh.dofmap.index_map_bs)
        else:
            # Vector for parameter space Vh
            return dlx.la.vector(self.Vh.dofmap.index_map, self.Vh.dofmap.index_map_bs)

    def sample(self, noise: dlx.la.Vector, s: dlx.la.Vector, add_mean=True) -> None:
        """
        Given :code:`noise` :math:`\\sim \\mathcal{N}(0, I)` compute a sample :code:`s` from the prior.

        If :code:`add_mean == True` add the prior mean value to :code:`s`.
        """
        rhs = self.sqrtR.createVecLeft()
        self.sqrtR.mult(noise.petsc_vec, rhs)
        self.Rsolver.solve(rhs, s.petsc_vec)

        if add_mean:
            s.petsc_vec.axpy(1.0, self.mean.petsc_vec)

        rhs.destroy()

    def __del__(self):
        # Clean up PETSc objects
        if hasattr(self, "Msolver") and self.Msolver:
            self.Msolver.destroy()
        if hasattr(self, "Rsolver") and self.Rsolver:
            self.Rsolver.destroy()
        if hasattr(self, "M") and self.M:
            self.M.destroy()
        if hasattr(self, "R") and self.R:
            self.R.destroy()
        if hasattr(self, "sqrtR") and self.sqrtR:
            self.sqrtR.destroy()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NEW FUNCTION PORTED FROM OLD CODE
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def BiLaplacianComputeCoefficients(sigma2: float, rho: float, ndim: int) -> tuple[float, float]:
    """
    This class is responsible to compute the parameters gamma and delta
    for the BiLaplacianPrior given the marginal variance sigma2 and
    correlation length rho. ndim is the dimension of the domain 2D or 3D
    """

    nu = 2.0 - 0.5 * ndim
    kappa = np.sqrt(8 * nu) / rho

    s = np.sqrt(sigma2) * np.power(kappa, nu) * np.sqrt(np.power(4.0 * np.pi, 0.5 * ndim) / math.gamma(nu))

    gamma = 1.0 / s
    delta = np.power(kappa, 2) / s

    return gamma, delta


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# EXISTING FUNCTION (NO CHANGES)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def BiLaplacianPrior(
    Vh: dlx.fem.FunctionSpace,
    gamma: float,
    delta: float,
    Theta=None,
    mean: dlx.la.Vector | None = None,
    robin_bc: bool = False,
    **kwargs,  # Allow absorbing old solver options
) -> SqrtPrecisionPDE_Prior:
    """
    This function construct an instance of :code"`SqrtPrecisionPDE_Prior`  with covariance matrix
    :math:`C = (\\delta I + \\gamma \\mbox{div } \\Theta \\nabla) ^ {-2}`.

    Input:

    - :code:`Vh`:              the finite element space for the parameter
    - :code:`gamma` and :code:`delta`: the coefficient in the PDE (floats, dl.Constant, dl.Expression, or dl.Function)
    - :code:`Theta`:           the SPD tensor for anisotropic diffusion of the PDE
    - :code:`mean`:            the prior mean
    - :code:`robin_bc`:        whether to use Robin boundary condition to remove boundary artifacts
    """
    # Convert floats to dolfinx.fem.Constant
    if isinstance(gamma, numbers.Number):
        gamma = dlx.fem.Constant(Vh.mesh, petsc4py.PETSc.ScalarType(gamma))
    if isinstance(delta, numbers.Number):
        delta = dlx.fem.Constant(Vh.mesh, petsc4py.PETSc.ScalarType(delta))

    def sqrt_precision_varf_handler(
        trial: ufl.TrialFunction,
        test: ufl.TestFunction,
        dx: ufl.Measure,
        ds: ufl.Measure,
    ) -> ufl.form.Form:
        if Theta is None:
            varfL = ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx
        else:
            varfL = ufl.inner(Theta * ufl.grad(trial), ufl.grad(test)) * dx

        varfM = ufl.inner(trial, test) * dx

        varf_robin = ufl.inner(trial, test) * ds

        if robin_bc:
            # Ensure constants are used in UFL form
            robin_coeff_val = ufl.sqrt(delta / gamma) / 1.42
            robin_coeff = gamma * robin_coeff_val
        else:
            robin_coeff = dlx.fem.Constant(Vh.mesh, petsc4py.PETSc.ScalarType(0.0))

        return gamma * varfL + delta * varfM + robin_coeff * varf_robin

    # Pass petsc_options if they are provided via kwargs
    petsc_options_A = kwargs.get("petsc_options_A")
    petsc_options_M = kwargs.get("petsc_options_M")

    return SqrtPrecisionPDE_Prior(
        Vh,
        sqrt_precision_varf_handler,
        mean,
        petsc_options_A=petsc_options_A,
        petsc_options_M=petsc_options_M,
    )


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NEW FUNCTION PORTED FROM OLD CODE
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def VectorBiLaplacianPrior(
    Vh: dlx.fem.FunctionSpace,
    gamma: list[float | dlx.fem.Constant],
    delta: list[float | dlx.fem.Constant],
    mean: dlx.la.Vector | None = None,
    robin_bc: bool = False,
    **kwargs,  # Allow absorbing old solver options
) -> SqrtPrecisionPDE_Prior:
    """
    This function construct an instance of :code"`SqrtPrecisionPDE_Prior` for a vector valued
    prior distribution with uncoupled components.
    """
    mesh = Vh.mesh
    # Convert float coefficients to dolfinx.fem.Constant
    gamma_const = [dlx.fem.Constant(mesh, petsc4py.PETSc.ScalarType(g)) if isinstance(g, numbers.Number) else g for g in gamma]
    delta_const = [dlx.fem.Constant(mesh, petsc4py.PETSc.ScalarType(d)) if isinstance(d, numbers.Number) else d for d in delta]

    dx = ufl.Measure("dx", metadata={"quadrature_degree": 4})
    ds = ufl.Measure("ds", metadata={"quadrature_degree": 4})

    def comp_varf(trial_i, test_i, gamma_i, delta_i):
        varfL = gamma_i * ufl.inner(ufl.grad(trial_i), ufl.grad(test_i)) * dx
        varfM = delta_i * ufl.inner(trial_i, test_i) * dx
        if robin_bc:
            robin_coeff_val = ufl.sqrt(delta_i / gamma_i) / 1.42
            robin_coeff = gamma_i * robin_coeff_val
        else:
            robin_coeff = dlx.fem.Constant(mesh, petsc4py.PETSc.ScalarType(0.0))

        varf_robin = robin_coeff * ufl.inner(trial_i, test_i) * ds
        return varfL + varfM + varf_robin

    def sqrt_precision_varf_handler(
        trial: ufl.TrialFunction,
        test: ufl.TestFunction,
    ) -> ufl.form.Form:
        varf = dlx.fem.Constant(mesh, 0.0) * ufl.inner(trial, test) * dx
        for trial_i, test_i, gamma_i, delta_i in zip(ufl.split(trial), ufl.split(test), gamma_const, delta_const):
            varf += comp_varf(trial_i, test_i, gamma_i, delta_i)
        return varf

    # Pass petsc_options if they are provided via kwargs
    petsc_options_A = kwargs.get("petsc_options_A")
    petsc_options_M = kwargs.get("petsc_options_M")

    return SqrtPrecisionPDE_Prior(
        Vh,
        sqrt_precision_varf_handler,
        mean,
        petsc_options_A=petsc_options_A,
        petsc_options_M=petsc_options_M,
    )


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# STUB FOR MOLLIFIEDBILAPLACIANPRIOR
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def MollifiedBiLaplacianPrior(*args, **kwargs):
    """
    This function is not implemented.
    Porting requires:
    1. A dolfinx equivalent for `dolfin.CompiledExpression`. This would
       likely involve using `dolfinx.fem.Expression` or interpolating a
       Python function onto a `dolfinx.fem.Function`.
    2. The source code for `ExpressionModule.Mollifier()` to replicate its logic.
    3. A dolfinx equivalent for `hippylib.utils.vector2function.vector2Function`
       to convert the `m_true` vector into a `dolfinx.fem.Function` for
       assembling the right-hand side.
    """
    raise NotImplementedError("MollifiedBiLaplacianPrior is not implemented. See docstring for details on porting.")


# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # STUB FOR GAUSSIANREALPRIOR
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# class GaussianRealPrior(_Prior):
#     """
#     This class is not implemented.
#     Porting from the old 'dolfin' implementation requires:
#     1. A dolfinx equivalent for the `dolfin.FunctionSpace(mesh, "R", 0)`.
#        This is non-trivial. Options include using a `("DG", 0)` space
#        and managing the global DoF mapping, or creating a custom
#        implementation that works directly with PETSc vectors of a
#        specified global size, independent of a mesh.
#     2. The old code used `ufl.as_matrix` to create dolfin.Matrix objects
#        from numpy arrays. The dolfinx equivalent would be to create
#        `petsc4py.PETSc.Mat` objects (e.g., `createAIJ`) and populate
#        them manually from the numpy arrays.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         raise NotImplementedError("GaussianRealPrior is not implemented. See docstring for details on porting.")

#     def generate_parameter(self, dim: int | str) -> dlx.la.Vector:
#         raise NotImplementedError

#     def sample(self, noise: dlx.la.Vector, s: dlx.la.Vector, add_mean=True) -> None:
#         raise NotImplementedError
