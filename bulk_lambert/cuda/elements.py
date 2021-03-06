
from numba import cuda

from math import sin, cos, sqrt

from .util import rotation_matrix, matmul_3x3, matmul_1x3T


@cuda.jit(device=True)
def rv_pqw(
    k, p, ecc, nu,
    _pqw, # 2x3
):
    r"""Returns r and v vectors in perifocal frame.
    Parameters
    ----------
    k : float
        Standard gravitational parameter (km^3 / s^2).
    p : float
        Semi-latus rectum or parameter (km).
    ecc : float
        Eccentricity.
    nu: float
        True anomaly (rad).
    Returns
    -------
    r: ndarray
        Position. Dimension 3 vector
    v: ndarray
        Velocity. Dimension 3 vector
    Notes
    -----
    These formulas can be checked at Curtis 3rd. Edition, page 110. Also the
    example proposed is 2.11 of Curtis 3rd Edition book.
    .. math::
        \vec{r} = \frac{h^2}{\mu}\frac{1}{1 + e\cos(\theta)}\begin{bmatrix}
        \cos(\theta)\\
        \sin(\theta)\\
        0
        \end{bmatrix} \\\\\\
        \vec{v} = \frac{h^2}{\mu}\begin{bmatrix}
        -\sin(\theta)\\
        e+\cos(\theta)\\
        0
        \end{bmatrix}
    Examples
    --------
    >>> from poliastro.constants import GM_earth
    >>> k = GM_earth.value  # Earth gravitational parameter
    >>> ecc = 0.3  # Eccentricity
    >>> h = 60000e6  # Angular momentum of the orbit (m**2 / s)
    >>> nu = np.deg2rad(120)  # True Anomaly (rad)
    >>> p = h**2 / k  # Parameter of the orbit
    >>> r, v = rv_pqw(k, p, ecc, nu)
    >>> # Printing the results
    r = [-5312706.25105345  9201877.15251336    0] [m]
    v = [-5753.30180931 -1328.66813933  0] [m]/[s]
    """

    a = p / (1 + ecc * cos(nu))
    b = sqrt(k / p)

    _pqw[0, 0] = cos(nu) * a
    _pqw[0, 1] = sin(nu) * a
    _pqw[0, 2] = 0.0

    _pqw[1, 0] = -sin(nu) * b
    _pqw[1, 1] = (ecc + cos(nu)) * b
    _pqw[1, 2] = 0.0


@cuda.jit(device=True)
def coe_rotation_matrix(
    inc, raan, argp,
    _r, _r_buffer1, _r_buffer2, # 3x3
):
    """Create a rotation matrix for coe transformation"""

    # r = rotation_matrix(raan, 2)
    rotation_matrix(raan, 2, _r) # set _r

    # r = r @ rotation_matrix(inc, 0)
    rotation_matrix(inc, 0, _r_buffer1) # set _r_buffer1
    matmul_3x3(_r, _r_buffer1, _r_buffer2) # set _r_buffer2 -> _r & _r_buffer1 are free

    # r = r @ rotation_matrix(argp, 2)
    rotation_matrix(argp, 2, _r_buffer1) # set _r_buffer1
    matmul_3x3(_r_buffer2, _r_buffer1, _r) # set _r -> _r_buffer1 & _r_buffer2 are free


@cuda.jit(device=True)
def coe2rv(
    k, p, ecc, inc, raan, argp, nu,
    _r, _r_buffer1, _r_buffer2, # 3x3
    _pqw, _ijk, # 2x3
    ):
    r"""Converts from classical orbital to state vectors.
    Classical orbital elements are converted into position and velocity
    vectors by `rv_pqw` algorithm. A rotation matrix is applied to position
    and velocity vectors to get them expressed in terms of an IJK basis.
    Parameters
    ----------
    k : float
        Standard gravitational parameter (km^3 / s^2).
    p : float
        Semi-latus rectum or parameter (km).
    ecc : float
        Eccentricity.
    inc : float
        Inclination (rad).
    omega : float
        Longitude of ascending node (rad).
    argp : float
        Argument of perigee (rad).
    nu : float
        True anomaly (rad).
    Returns
    -------
    r_ijk: np.array
        Position vector in basis ijk.
    v_ijk: np.array
        Velocity vector in basis ijk.
    Notes
    -----
    .. math::
        \begin{align}
            \vec{r}_{IJK} &= [ROT3(-\Omega)][ROT1(-i)][ROT3(-\omega)]\vec{r}_{PQW}
                               = \left [ \frac{IJK}{PQW} \right ]\vec{r}_{PQW}\\
            \vec{v}_{IJK} &= [ROT3(-\Omega)][ROT1(-i)][ROT3(-\omega)]\vec{v}_{PQW}
                               = \left [ \frac{IJK}{PQW} \right ]\vec{v}_{PQW}\\
        \end{align}
    Previous rotations (3-1-3) can be expressed in terms of a single rotation matrix:
    .. math::
        \left [ \frac{IJK}{PQW} \right ]
    .. math::
        \begin{bmatrix}
        \cos(\Omega)\cos(\omega) - \sin(\Omega)\sin(\omega)\cos(i) & -\cos(\Omega)\sin(\omega) - \sin(\Omega)\cos(\omega)\cos(i) & \sin(\Omega)\sin(i)\\
        \sin(\Omega)\cos(\omega) + \cos(\Omega)\sin(\omega)\cos(i) & -\sin(\Omega)\sin(\omega) + \cos(\Omega)\cos(\omega)\cos(i) & -\cos(\Omega)\sin(i)\\
        \sin(\omega)\sin(i) & \cos(\omega)\sin(i) & \cos(i)
        \end{bmatrix}
    """
    rv_pqw(
        k, p, ecc, nu,
        _pqw, # 2x3
    )
    coe_rotation_matrix(
        inc, raan, argp,
        _r, _r_buffer1, _r_buffer2, # 3x3
    )
    matmul_1x3T(_pqw, _r, _ijk)
