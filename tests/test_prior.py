from mpi4py import MPI

import dolfinx
import pytest

import hippylibX as hpx


@pytest.mark.parametrize("family, degree", [("Lagrange", 1), ("Lagrange", 2), ("DG", 0), ("DG", 1)])
@pytest.mark.parametrize("shape", [(), (2,)])
def test_prior_sample(family, degree, shape):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)

    V = dolfinx.fem.functionspace(mesh, (family, degree, shape))

    gamma = 1.0
    delta = 0.5
    prior = hpx.BiLaplacianPrior(V, gamma, delta, robin_bc=True)

    noise = prior.generate_parameter("noise")
    hpx.parRandom.normal(1.0, noise)

    sample = dolfinx.fem.Function(prior.Vh, name="prior_sample")
    prior.sample(noise, sample.x)

    # Check that the sample has the correct shape
    if shape == ():
        assert sample.x.array.shape == (V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts,)
    else:
        assert sample.x.array.shape == ((V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts) * V.dofmap.bs,)
