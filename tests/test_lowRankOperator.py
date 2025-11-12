import petsc4py
from mpi4py import MPI

import dolfinx as dlx
import dolfinx.fem.petsc
import numpy as np
import pytest
from numpy.testing import assert_allclose

from hippylibX.algorithms.lowRankOperator import LowRankOperator
from hippylibX.algorithms.multivector import MultiVector

# --- Test Data ---
RANK = 3
N_DOFS = 5  # N > k
D_VAL = np.array([5.0, 3.0, 1.0])
RNG = np.random.default_rng(seed=1234)


@pytest.fixture(scope="module")
def scipy_ground_truth():
    """
    Creates the numpy ground truth using a random orthonormal U.
    """
    # Create a random (N, k) matrix and get its orthonormal basis (Q)
    U_np, _ = np.linalg.qr(RNG.random((N_DOFS, RANK)))

    # Create test vector x
    X_NP = RNG.random(N_DOFS)

    # --- Calculate ground truth A = U D U^T ---
    D_diag_np = np.diag(D_VAL)
    A_full = U_np @ D_diag_np @ U_np.T

    # --- Calculate ground truth A_pinv = U D_inv U^T ---
    D_inv_diag_np = np.diag(1.0 / D_VAL)
    A_pinv_full = U_np @ D_inv_diag_np @ U_np.T

    # --- Calculate expected results ---
    EXPECTED_MULT = A_full @ X_NP
    EXPECTED_SOLVE = A_pinv_full @ X_NP
    EXPECTED_DIAG = np.diag(A_full)

    # Trace property: tr(UDU^T) = tr(U^T U D) = tr(I*D) = sum(d)
    EXPECTED_TRACE = np.sum(D_VAL)

    # Verify numpy trace calculation
    assert_allclose(np.trace(A_full), EXPECTED_TRACE)

    return (U_np, X_NP, EXPECTED_MULT, EXPECTED_SOLVE, EXPECTED_DIAG, EXPECTED_TRACE)


@pytest.fixture(scope="module")
def setup_old_scipy(scipy_ground_truth):
    (U_np, X_NP, _, _, _, _) = scipy_ground_truth

    # Create dolfin vector template
    mesh = dlx.mesh.create_unit_interval(MPI.COMM_WORLD, N_DOFS - 1)
    V = dlx.fem.functionspace(mesh, ("Lagrange", 1))

    template = dolfinx.fem.petsc.create_vector(V)

    # Create mocks and operator
    U = MultiVector(template, U_np.shape[1])
    # breakpoint()
    # U = MultiVector(V, U_np.shape[1])
    for i in range(U_np.shape[1]):
        num_local_values = U[i].getLocalSize()
        r = U[i].getOwnershipRange()
        with U[i].localForm() as v_array:
            v_array[0:num_local_values] += U_np[r[0] : r[1], i]

        U[i].ghostUpdate(
            addv=petsc4py.PETSc.InsertMode.INSERT,  # type: ignore
            mode=petsc4py.PETSc.ScatterMode.FORWARD,  # type: ignore
        )

        U[i].duplicate()
    A = LowRankOperator(D_VAL, U)
    # Create test vectors
    x_vec = template.copy()

    with x_vec.localForm() as x_array:
        num_local_values = x_vec.getLocalSize()
        r = x_vec.getOwnershipRange()
        x_array[0:num_local_values] += X_NP[r[0] : r[1]]

    y_vec = template.copy()
    A.get_diagonal(y_vec)
    return A, x_vec, y_vec, scipy_ground_truth


def test_old_mult_scipy(setup_old_scipy):
    A, x_vec, y_vec, ground_truth = setup_old_scipy
    EXPECTED_MULT = ground_truth[2]

    r = y_vec.getOwnershipRange()
    # Slice the global expected array to get the local part
    local_expected = EXPECTED_MULT[r[0] : r[1]]

    A.mult(x_vec, y_vec)
    assert_allclose(y_vec.array, local_expected, rtol=1e-14)


def test_old_solve_scipy(setup_old_scipy):
    A, x_vec, y_vec, ground_truth = setup_old_scipy
    EXPECTED_SOLVE = ground_truth[3]

    r = y_vec.getOwnershipRange()
    # Slice the global expected array to get the local part
    local_expected = EXPECTED_SOLVE[r[0] : r[1]]

    A.solve(y_vec, x_vec)  # y_vec is sol, x_vec is rhs
    assert_allclose(y_vec.array, local_expected, rtol=14)


def test_old_get_diagonal_scipy(setup_old_scipy):
    A, _, y_vec, ground_truth = setup_old_scipy
    EXPECTED_DIAG = ground_truth[4]

    r = y_vec.getOwnershipRange()
    # Slice the global expected array to get the local part
    local_expected = EXPECTED_DIAG[r[0] : r[1]]

    A.get_diagonal(y_vec)
    assert_allclose(y_vec.array, local_expected, rtol=1e-14)


def test_old_trace_scipy(setup_old_scipy):
    A, _, _, ground_truth = setup_old_scipy
    EXPECTED_TRACE = ground_truth[5]

    tr = A.trace()
    assert_allclose(tr, EXPECTED_TRACE, rtol=1e-14)
