import petsc4py


class UnsetMatrix(petsc4py.PETSc.Mat):
    def __init__(self):
        pass

    def destroy(self):
        pass

    def zeroEntries(self):
        raise RuntimeError("UnsetMatrix: zeroEntries called on unset matrix.")
