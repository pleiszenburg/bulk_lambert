
import numpy as np
from math import sqrt, pi

from numba import cuda

from .angles import (
    # D_to_M,
    D_to_nu,
    E_to_M,
    E_to_nu,
    F_to_M,
    F_to_nu,
    M_to_D,
    M_to_E,
    M_to_F,
    # nu_to_D,
    # nu_to_E,
    # nu_to_F,
)


@cuda.jit(device=True)
def D_to_M_near_parabolic(D, ecc):
    x = (ecc - 1.0) / (ecc + 1.0) * (D ** 2)
    assert abs(x) < 1
    S = S_x(ecc, x)
    return (
        np.sqrt(2.0 / (1.0 + ecc)) * D + np.sqrt(2.0 / (1.0 + ecc) ** 3) * (D ** 3) * S
    )


@cuda.jit(device=True)
def _kepler_equation_near_parabolic(D, M, ecc):
    return D_to_M_near_parabolic(D, ecc) - M


@cuda.jit(device=True)
def _kepler_equation_prime_near_parabolic(D, M, ecc):
    x = (ecc - 1.0) / (ecc + 1.0) * (D ** 2)
    assert abs(x) < 1
    S = dS_x_alt(ecc, x)
    return np.sqrt(2.0 / (1.0 + ecc)) + np.sqrt(2.0 / (1.0 + ecc) ** 3) * (D ** 2) * S


@cuda.jit(device=True)
def S_x(ecc, x, atol=1e-12):
    assert abs(x) < 1
    S = 0
    k = 0
    while True:
        S_old = S
        S += (ecc - 1 / (2 * k + 3)) * x ** k
        k += 1
        if abs(S - S_old) < atol:
            return S


@cuda.jit(device=True)
def dS_x_alt(ecc, x, atol=1e-12):
    # Notice that this is not exactly
    # the partial derivative of S with respect to D,
    # but the result of arranging the terms
    # in section 4.2 of Farnocchia et al. 2013
    assert abs(x) < 1
    S = 0
    k = 0
    while True:
        S_old = S
        S += (ecc - 1 / (2 * k + 3)) * (2 * k + 3) * x ** k
        k += 1
        if abs(S - S_old) < atol:
            return S


@cuda.jit(device=True)
def M_to_D_near_parabolic(M, ecc, tol=1.48e-08, maxiter=50):
    """Parabolic eccentric anomaly from mean anomaly, near parabolic case.
    Parameters
    ----------
    M : float
        Mean anomaly in radians.
    ecc : float
        Eccentricity (~1).
    Returns
    -------
    D : float
        Parabolic eccentric anomaly.
    """
    D0 = M_to_D(M)

    for _ in range(maxiter):
        fval = _kepler_equation_near_parabolic(D0, M, ecc)
        fder = _kepler_equation_prime_near_parabolic(D0, M, ecc)

        newton_step = fval / fder
        D = D0 - newton_step
        if abs(D - D0) < tol:
            return D

        D0 = D

    return np.nan

@cuda.jit(device=True)
def nu_from_delta_t(delta_t, ecc, k=1.0, q=1.0, delta=1e-2):
    """True anomaly for given elapsed time since periapsis.
    Parameters
    ----------
    delta_t : float
        Time elapsed since periapsis.
    ecc : float
        Eccentricity.
    k : float
        Gravitational parameter.
    q : float
        Periapsis distance.
    delta : float
        Parameter that controls the size of the near parabolic region.
    Returns
    -------
    nu : float
        True anomaly.
    """
    if ecc < 1 - delta:
        # Strong elliptic
        n = sqrt(k * (1 - ecc) ** 3 / q ** 3)
        M = n * delta_t
        # This might represent several revolutions,
        # so we wrap the true anomaly
        E = M_to_E((M + pi) % (2 * pi) - pi, ecc)
        nu = E_to_nu(E, ecc)
    elif 1 - delta <= ecc < 1:
        E_delta = np.arccos((1 - delta) / ecc)
        # We compute M assuming we are in the strong elliptic case
        # and verify later
        n = sqrt(k * (1 - ecc) ** 3 / q ** 3)
        M = n * delta_t
        # We check against abs(M) because E_delta could also be negative
        if E_to_M(E_delta, ecc) <= abs(M):
            # Strong elliptic, proceed
            # This might represent several revolutions,
            # so we wrap the true anomaly
            E = M_to_E((M + pi) % (2 * pi) - pi, ecc)
            nu = E_to_nu(E, ecc)
        else:
            # Near parabolic, recompute M
            n = sqrt(k / (2 * q ** 3))
            M = n * delta_t
            D = M_to_D_near_parabolic(M, ecc)
            nu = D_to_nu(D)
    elif ecc == 1:
        # Parabolic
        n = sqrt(k / (2 * q ** 3))
        M = n * delta_t
        D = M_to_D(M)
        nu = D_to_nu(D)
    elif 1 < ecc <= 1 + delta:
        F_delta = np.arccosh((1 + delta) / ecc)
        # We compute M assuming we are in the strong hyperbolic case
        # and verify later
        n = sqrt(k * (ecc - 1) ** 3 / q ** 3)
        M = n * delta_t
        # We check against abs(M) because F_delta could also be negative
        if F_to_M(F_delta, ecc) <= abs(M):
            # Strong hyperbolic, proceed
            F = M_to_F(M, ecc)
            nu = F_to_nu(F, ecc)
        else:
            # Near parabolic, recompute M
            n = sqrt(k / (2 * q ** 3))
            M = n * delta_t
            D = M_to_D_near_parabolic(M, ecc)
            nu = D_to_nu(D)
    # elif 1 + delta < ecc:
    else:
        # Strong hyperbolic
        n = sqrt(k * (ecc - 1) ** 3 / q ** 3)
        M = n * delta_t
        F = M_to_F(M, ecc)
        nu = F_to_nu(F, ecc)

    return nu
