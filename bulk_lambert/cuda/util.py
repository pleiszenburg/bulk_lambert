
from numba import cuda

from numpy import sin, cos


@cuda.jit(device=True)
def matmul_3x3(A, B, _out):
    """
    very naive matrix multiplication
    A, B, _out: 3x3
    """

    _out[:, :] = 0.0

    for i in range(3):
        for j in range(3):
            for k in range(3):
                _out[i, j] += A[i, k] * B[k, j]


@cuda.jit(device=True)
def rotation_matrix(angle, axis, _matrix): # matrix must be allocated elsewhere

    c = cos(angle)
    s = sin(angle)

    if axis == 0:
        _matrix[0, :] = 1.0, 0.0, 0.0
        _matrix[1, :] = 0.0, c, -s
        _matrix[2, :] = 0.0, s, c
    elif axis == 1:
        _matrix[0, :] = c, 0.0, s
        _matrix[1, :] = 0.0, 1.0, 0.0
        _matrix[2, :] = s, 0.0, c
    elif axis == 2:
        _matrix[0, :] = c, -s, 0.0
        _matrix[1, :] = s, c, 0.0
        _matrix[2, :] = 0.0, 0.0, 1.0
    else:
        raise ValueError("Invalid axis: must be one of 'x', 'y' or 'z'")
