import dolfinx as dlx
from packaging.version import Version

_dolfinx_version = Version(dlx.__version__)


def create_vector(form):
    """Create a PETSc vector compatible with the given form."""
    if _dolfinx_version >= Version("0.10"):
        return dlx.fem.petsc.create_vector(dlx.fem.extract_function_spaces(form))
    else:
        return dlx.fem.petsc.create_vector(form)
