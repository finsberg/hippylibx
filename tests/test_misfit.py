from mpi4py import MPI

import dolfinx as dlx
import dolfinx.fem.petsc
import numpy as np
import ufl

import hippylibX as hpx

# Import the real hippylibX

# Import the class to be tested

#


def test_cost():
    """
    Tests the cost() method with a known analytical solution.
    This ensures the basic assembly and hpx.updateFromVector work.
    """

    comm = MPI.COMM_WORLD
    mesh = dlx.mesh.create_unit_square(comm, 5, 5)

    # 1. Function spaces
    V_element = ("Lagrange", 1)
    Vh_state = dlx.fem.functionspace(mesh, V_element)
    Vh_param = dlx.fem.functionspace(mesh, V_element)
    # Vh = [STATE, PARAMETER, ADJOINT]
    Vh = [Vh_state, Vh_param, Vh_state]
    dx = ufl.dx(metadata={"quadrature_degree": 4})

    # 2. Define a Misfit
    # We use a simple analytical form for the 'cost' test,
    # but 'modelVerify' will test any form.
    # J(u, m) = 0.5 * integral(u^2 + m^2) dx
    def misfit_form_func(u_fun, m_fun):
        return 0.5 * (u_fun**2 + m_fun**2) * dx

    misfit = hpx.modeling.misfit.NonGaussianContinuousMisfit(Vh, misfit_form_func)

    # 1. Define functions u = 1.0, m = 2.0
    u = dlx.fem.Function(Vh_state)
    u.interpolate(lambda x: np.full_like(x[0], 1.0))

    m = dlx.fem.Function(Vh_param)
    m.interpolate(lambda x: np.full_like(x[0], 2.0))

    # Dummy adjoint vector (not used by cost)
    p_vec = dlx.fem.Function(Vh_state).x
    x = [u.x, m.x, p_vec]

    # 2. Calculate expected cost
    # J = 0.5 * integral(1^2 + 2^2) dx = 0.5 * integral(5) dx
    # J = 2.5 * Area(Omega) = 2.5 * 1.0 = 2.5
    expected_cost = 2.5

    # 3. Calculate cost using the class
    computed_cost = misfit.cost(x)

    # 4. Assert
    assert np.isclose(computed_cost, expected_cost)
