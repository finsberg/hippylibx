# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

import abc
import typing

import petsc4py
from mpi4py import MPI

import dolfinx as dlx
import ufl

import hippylibX as hpx


class Misfit(abc.ABC):
    """
    Abstract class to model the misfit component of the cost functional.
    In the following :code:`x` will denote the variable :code:`[u, m, p]`, denoting respectively
    the state :code:`u`, the parameter :code:`m`, and the adjoint variable :code:`p`.

    The methods in the class misfit will usually access the state u and possibly the
    parameter :code:`m`. The adjoint variables will never be accessed.
    """

    @abc.abstractmethod
    def cost(self, x):
        """
        Given x evaluate the cost functional.
        Only the state u and (possibly) the parameter m are accessed."""

    @abc.abstractmethod
    def grad(self, i, x, out):
        """
        Given the state and the paramter in :code:`x`, compute the partial gradient of the misfit
        functional in with respect to the state (:code:`i == STATE`) or with respect to the parameter (:code:`i == PARAMETER`).
        """

    @abc.abstractmethod
    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        """
        Set the point for linearization.

        Inputs:

            :code:`x=[u, m, p]` - linearization point

            :code:`gauss_newton_approx (bool)` - whether to use Gauss Newton approximation
        """

    @abc.abstractmethod
    def apply_ij(self, i, j, dir, out):
        r"""
        Apply the second variation :math:`\delta_{ij}` (:code:`i,j = STATE,PARAMETER`) of the cost in direction :code:`dir`.
        """


class NonGaussianContinuousMisfit(Misfit):
    """
    Abstract class to model the misfit component of the cost functional.
    In the following :code:`x` will denote the variable :code:`[u, m, p]`, denoting respectively
    the state :code:`u`, the parameter :code:`m`, and the adjoint variable :code:`p`.

    The methods in the class misfit will usually access the state u and possibly the
    parameter :code:`m`. The adjoint variables will never be accessed.
    """

    def __init__(
        self,
        Vh: list[dlx.fem.FunctionSpace],
        form: typing.Callable[[dlx.fem.Function, dlx.fem.Function], ufl.form.Form],
        bc0: list[dlx.fem.DirichletBC] | None = None,
    ):
        """_summary_

        Parameters
        ----------
        Vh : list
            The space (STATE, PARAMETER, ADJOINT)
        form : typing.Callable[[dlx.fem.Function, dlx.fem.Function], ufl.form.Form]
            The misfit form depending on state and parameter
        bc0 : list[dlx.fem.DirichletBC] | None, optional
            The Dirichlet boundary conditions for the parameter, by default None
        """
        self.Vh = Vh
        self.form = form
        self.bc0 = bc0 or []

        self.x_test = [
            ufl.TestFunction(Vh[hpx.STATE]),
            ufl.TestFunction(Vh[hpx.PARAMETER]),
        ]
        self.gauss_newton_approx = False

        self.xfun = [dlx.fem.Function(Vhi) for Vhi in Vh]

    def cost(self, x: list[dlx.fem.Function | None]) -> float:
        """
        Given x evaluate the cost functional.
        Only the state u and (possibly) the parameter m are accessed.
        """
        x_state = x[hpx.STATE]
        if x_state is not None:
            hpx.updateFromVector(self.xfun[hpx.STATE], x_state)
        u_fun = self.xfun[hpx.STATE]

        x_param = x[hpx.PARAMETER]
        if x_param is not None:
            hpx.updateFromVector(self.xfun[hpx.PARAMETER], x_param)
        m_fun = self.xfun[hpx.PARAMETER]

        loc_cost = self.form(u_fun, m_fun)
        glb_cost_proc = dlx.fem.assemble_scalar(dlx.fem.form(loc_cost))
        return self.Vh[hpx.STATE].mesh.comm.allreduce(glb_cost_proc, op=MPI.SUM)

    def grad(self, i: int, x: list, out: dlx.la.Vector) -> None:
        """
        Given the state and the paramter in :code:`x`, compute the partial gradient of the misfit
        functional in with respect to the state (:code:`i == STATE`) or with respect to the parameter (:code:`i == PARAMETER`).
        """
        hpx.updateFromVector(self.xfun[hpx.STATE], x[hpx.STATE])
        u_fun = self.xfun[hpx.STATE]

        hpx.updateFromVector(self.xfun[hpx.PARAMETER], x[hpx.PARAMETER])
        m_fun = self.xfun[hpx.PARAMETER]

        x_fun = [u_fun, m_fun]

        out.array[:] = 0.0

        dlx.fem.petsc.assemble_vector(
            out.petsc_vec,
            dlx.fem.form(ufl.derivative(self.form(*x_fun), x_fun[i], self.x_test[i])),
        )
        out.petsc_vec.ghostUpdate(
            petsc4py.PETSc.InsertMode.ADD_VALUES,
            petsc4py.PETSc.ScatterMode.REVERSE,
        )
        dlx.fem.petsc.set_bc(out.petsc_vec, self.bc0)

    def setLinearizationPoint(self, x: list, gauss_newton_approx=False) -> None:
        hpx.updateFromVector(self.xfun[hpx.STATE], x[hpx.STATE])
        u_fun = self.xfun[hpx.STATE]
        hpx.updateFromVector(self.xfun[hpx.PARAMETER], x[hpx.PARAMETER])
        m_fun = self.xfun[hpx.PARAMETER]
        self.x_lin_fun = [u_fun, m_fun]
        self.gauss_newton_approx = gauss_newton_approx

    def apply_ij(self, i: int, j: int, dir: dlx.la.Vector, out: dlx.la.Vector) -> None:
        r"""
        Apply the second variation :math:`\delta_{ij}` (:code:`i,j = STATE,PARAMETER`) of the cost in direction :code:`dir`.
        """
        form = self.form(*self.x_lin_fun)
        if j == hpx.STATE:
            dlx.fem.set_bc(dir.array, self.bc0)

        dir_fun = hpx.vector2Function(dir, self.Vh[j])
        action = dlx.fem.form(
            ufl.derivative(
                ufl.derivative(form, self.x_lin_fun[i], self.x_test[i]),
                self.x_lin_fun[j],
                dir_fun,
            ),
        )
        out.array[:] = 0.0
        dlx.fem.petsc.assemble_vector(out.petsc_vec, action)
        out.petsc_vec.ghostUpdate(
            petsc4py.PETSc.InsertMode.ADD_VALUES,
            petsc4py.PETSc.ScatterMode.REVERSE,
        )
        if i == hpx.STATE:
            dlx.fem.petsc.set_bc(out.petsc_vec, self.bc0)


class ContinuousStateObservation(Misfit):
    """
    This class implements continuous state observations in a
    subdomain :math:`X \subset \Omega` or :math:`X \subset \partial \Omega`.
    """

    def __init__(self, Vh, dX, bcs=None, data=None, noise_variance=None, form=None):
        """
        Constructor:

            :code:`Vh`: the finite element space for the state variable.

            :code:`dX`: the integrator on subdomain `X` where observation are presents. \
            E.g. :code:`dX = ufl.dx` means observation on all :math:`\Omega` and :code:`dX = ufl.ds` means observations on all :math:`\partial \Omega`.

            :code:`bcs`: If the forward problem imposes Dirichlet boundary conditions :math:`u = u_D \mbox{ on } \Gamma_D`;  \
            :code:`bcs` is a list of :code:`dolfin.DirichletBC` object that prescribes homogeneuos Dirichlet conditions :math:`u = 0 \mbox{ on } \Gamma_D`.

            :code:`data` is the data

            :code:`noise_variance` is the variance of the noise

            :code:`form`: if :code:`form = None` we compute the :math:`L^2(X)` misfit: :math:`\int_X (u - u_d)^2 dX,` \
            otherwise the integrand specified in the given form will be used.
        """
        if form is None:
            u, v = dl.TrialFunction(Vh), dl.TestFunction(Vh)
            self.W = dl.assemble(ufl.inner(u, v) * dX)
        else:
            self.W = dl.assemble(form)

        if bcs is None:
            bcs = []
        if isinstance(bcs, dl.DirichletBC):
            bcs = [bcs]

        if len(bcs):
            Wt = Transpose(self.W)
            [bc.zero(Wt) for bc in bcs]
            self.W = Transpose(Wt)
            [bc.zero(self.W) for bc in bcs]

        if data is None:
            self.d = dl.Vector(self.W.mpi_comm())
            self.W.init_vector(self.d, 1)
        else:
            self.d = data

        self.noise_variance = noise_variance

    def cost(self, x):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        r = self.d.copy()
        r.axpy(-1.0, x[STATE])
        Wr = dl.Vector(self.W.mpi_comm())
        self.W.init_vector(Wr, 0)
        self.W.mult(r, Wr)
        return r.inner(Wr) / (2.0 * self.noise_variance)

    def grad(self, i, x, out):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        if i == STATE:
            self.W.mult(x[STATE] - self.d, out)
            out *= 1.0 / self.noise_variance
        elif i == PARAMETER:
            out.zero()
        else:
            raise IndexError()

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        # The cost functional is already quadratic. Nothing to be done here
        return

    def apply_ij(self, i, j, dir, out):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        if i == STATE and j == STATE:
            self.W.mult(dir, out)
            out *= 1.0 / self.noise_variance
        else:
            out.zero()
