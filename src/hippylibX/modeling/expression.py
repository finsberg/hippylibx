import numpy as np

# --- Helper class 1: Anisotropic Tensor ---
# This class holds the parameters for the tensor and can
# also be used to interpolate the tensor field itself.


class AnisTensor2D:
    """
    A helper class to compute a 2D anisotropic tensor.

    This replicates the math of the C++ AnisTensor2D,
    A = R * D * R^T
    where D = diag(theta0, theta1) and R is a rotation.
    """

    def __init__(self, theta0: float = 1.0, theta1: float = 1.0, alpha: float = 0.0):
        self.theta0 = theta0
        self.theta1 = theta1
        self.alpha = alpha

    def set(self, theta0: float, theta1: float, alpha: float):
        """Set the tensor parameters."""
        self.theta0 = theta0
        self.theta1 = theta1
        self.alpha = alpha

    def __call__(self, x):
        """
        Evaluate the tensor.

        Args:
            x: An array of coordinates from dolfinx, shape (gdim, n_points).
               For a 2D mesh, gdim is 2.

        Returns:
            An array of shape (value_shape, n_points), which for a
            2x2 tensor is (2, 2, n_points).
        """
        # breakpoint()
        # This expression is constant w.r.t. x, so we just compute it once.
        sa = np.sin(self.alpha)
        ca = np.cos(self.alpha)
        c00 = self.theta0 * sa * sa + self.theta1 * ca * ca
        c01 = (self.theta0 - self.theta1) * sa * ca
        c11 = self.theta0 * ca * ca + self.theta1 * sa * sa

        # Get the number of points to evaluate

        # Create a (4, 1) column vector of the tensor components [00, 01, 10, 11]
        # and broadcast it to (4, n_points)
        return np.broadcast_to(np.array([c00, c01, c01, c11]).reshape(4, 1), (4, x.shape[1]))


# --- Helper class 2: Mollifier ---
# This class implements the complex, coordinate-dependent scalar field


class Mollifier:
    """
    A helper class to compute an anisotropic mollifier field.

    f(x) = sum( exp( -( ||x - x_i||_B / l )^o ) )

    where B = A_inv and A is the AnisTensor2D.
    """

    def __init__(self):
        # A list of (x, y) tuples
        self.locations = []
        self.nlocations = 0
        self.l = 1.0
        self.o = 2.0

        # Components of the *inverse* tensor B = A^{-1}
        self.b00 = 1.0
        self.b01 = 0.0
        self.b11 = 1.0

    def addLocation(self, x: float, y: float):
        """Adds a new kernel center point."""
        self.locations.append((x, y))
        self.nlocations += 1

    def set(self, A: AnisTensor2D, l: float, o: float):
        """
        Set the mollifier parameters from an AnisTensor2D helper.
        This calculates the components of the inverse tensor B = A^{-1}.
        """
        self.l = l
        self.o = o

        # Get A's parameters
        t0 = A.theta0
        t1 = A.theta1
        alpha = A.alpha

        # Calculate B's parameters (using 1/t0, 1/t1)
        # This defines the components of B = R * D_inv * R^T
        it0 = 1.0 / t0 if t0 != 0 else 0.0
        it1 = 1.0 / t1 if t1 != 0 else 0.0

        sa = np.sin(alpha)
        ca = np.cos(alpha)

        self.b00 = it0 * sa * sa + it1 * ca * ca
        self.b01 = (it0 - it1) * sa * ca
        self.b11 = it0 * ca * ca + it1 * sa * sa

    def __call__(self, x):
        """
        Evaluate the mollifier field at the given coordinates.

        Args:
            x: An array of coordinates from dolfinx, shape (gdim, n_points).

        Returns:
            An array of shape (value_shape, n_points), which for a
            scalar field is (1, n_points).
        """
        # x is (gdim, n_points), e.g., (2, 1000)
        n_points = x.shape[1]

        # Output array, shape (n_points,)
        values = np.zeros(n_points)

        # --- Performance Note ---
        # This is a direct translation of the C++ logic.
        # It loops over all evaluation points, which is slow in Python.
        # A faster, vectorized version is possible but more complex.

        # Loop over all points to evaluate
        for i in range(n_points):
            # Get the 2D coordinate of the i-th point
            x_point = x[:, i]
            val = 0.0

            # Loop over all kernel locations
            for loc in self.locations:
                dx0 = x_point[0] - loc[0]
                dx1 = x_point[1] - loc[1]

                # e = dx.T @ B @ dx
                e = dx0 * dx0 * self.b00 + dx1 * dx1 * self.b11 + 2 * dx0 * dx1 * self.b01

                # val += exp( -( (||x - x_i||_B / l)^o ) )
                val += np.exp(-np.power(e / (self.l**2), 0.5 * self.o))

            values[i] = val

        # .interpolate() expects shape (value_shape, n_points)
        # For a scalar function, value_shape is (1,)
        return values.reshape(1, n_points)
