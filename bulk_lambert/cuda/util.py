
from numba import cuda

from numpy import sin, cos

@cuda.jit(device=True)
def rotation_matrix(angle, axis, matrix): # matrix must be allocated elsewhere

    c = cos(angle)
    s = sin(angle)

    if axis == 0:
        matrix[0, :] = 1.0, 0.0, 0.0
        matrix[1, :] = 0.0, c, -s
        matrix[2, :] = 0.0, s, c
    elif axis == 1:
        matrix[0, :] = c, 0.0, s
        matrix[1, :] = 0.0, 1.0, 0.0
        matrix[2, :] = s, 0.0, c
    elif axis == 2:
        matrix[0, :] = c, -s, 0.0
        matrix[1, :] = s, c, 0.0
        matrix[2, :] = 0.0, 0.0, 1.0
    else:
        raise ValueError("Invalid axis: must be one of 'x', 'y' or 'z'")
